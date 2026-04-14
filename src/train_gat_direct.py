"""
Phase 2: Query-Conditioned GAT + DirectClassifierHead 학습 스크립트.

DualTowerProjector 기반 train_gat.py와 달리, query를 loss 계산에서 제거한다:
  - GAT 내부에서 query-aware attention이 이미 수행된다고 가정
  - Node embedding을 MLP classifier에 직접 넣어 binary logit 예측
  - BCE loss only (InfoNCE 제거: 조인트 임베딩 공간 없음)

이로써 query가 pipeline에서 3회 중복되던 문제를 제거한다:
  (기존) GAT forward + Projector 재입력 + raw cosine baseline
  (직접) GAT forward 1회만
"""
import os
import yaml
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
import wandb
import argparse

from data.bird_dataset import BIRDGraphDataset, BIRDSuperNodeDataset
from modules.builders.graph_builder import HeteroGraphBuilder, EnrichedHeteroGraphBuilder
from modules.encoders.local_encoder import LocalPLMEncoder
from models.gat_network import SchemaHeteroGAT
from models.direct_classifier import DirectClassifierHead
from utils.logger import setup_logger, get_logger


PATHS = {
    "checkpoint_dir": "./outputs/checkpoints",
    "cache_dir": "./data/processed",
}


def calculate_recall_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int = 15) -> float:
    if labels.sum() == 0:
        return 0.0
    k_actual = min(k, logits.size(0))
    _, top_k_indices = torch.topk(logits, k_actual)
    hits = labels[top_k_indices].sum().item()
    return hits / labels.sum().item()


def validate(gat_model, classifier_heads, loader, device, k=15,
             query_conditioned=False, query_supernode=False):
    gat_model.eval()
    for head in classifier_heads.values():
        head.eval()
    total_recall = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            q_emb = batch['query']

            if query_conditioned:
                if q_emb.dim() == 3:
                    q_pooled = q_emb.mean(dim=1)
                elif q_emb.dim() == 2:
                    q_pooled = q_emb
                else:
                    q_pooled = q_emb.unsqueeze(0)

                augmented_x = {}
                for n_type, x in batch.x_dict.items():
                    node_batch_idx = batch[n_type].batch
                    q_per_node_val = q_pooled[node_batch_idx]
                    augmented_x[n_type] = torch.cat([x, q_per_node_val], dim=-1)
                node_embs_dict = gat_model(augmented_x, batch.edge_index_dict)
            else:
                node_embs_dict = gat_model(batch.x_dict, batch.edge_index_dict)

            for i in range(batch.num_graphs):
                logits_list, labels_list = [], []
                for n_type in ['table', 'column']:
                    if n_type not in classifier_heads:
                        continue
                    mask = (batch[n_type].batch == i)
                    if not mask.any():
                        continue
                    node_emb = node_embs_dict[n_type][mask]
                    score = classifier_heads[n_type](node_emb)
                    logits_list.append(score)
                    labels_list.append(batch[n_type].y[mask])

                if logits_list:
                    all_logits = torch.cat(logits_list)
                    all_labels = torch.cat(labels_list)
                    total_recall += calculate_recall_at_k(all_logits, all_labels, k=k)
                    count += 1
    return total_recall / count if count > 0 else 0.0


def run_train(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    wandb.init(
        project=cfg['project_name'],
        name=cfg['experiment_name'],
        config=cfg,
    )

    setup_logger(log_dir="./logs/", exp_name=cfg['experiment_name'], sub_dir="train")
    logger = get_logger(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(PATHS["checkpoint_dir"], exist_ok=True)
    os.makedirs(PATHS["cache_dir"], exist_ok=True)

    # Builder / Encoder
    builder_type = cfg.get('builder', {}).get('type', 'HeteroGraphBuilder')
    if builder_type == 'EnrichedHeteroGraphBuilder':
        tables_json = cfg['builder'].get('tables_json_path', '')
        builder = EnrichedHeteroGraphBuilder(tables_json_path=tables_json)
        logger.info(f"Using EnrichedHeteroGraphBuilder (tables_json={tables_json})")
    else:
        builder = HeteroGraphBuilder()
    encoder = LocalPLMEncoder()

    logger.info("Loading Training Dataset from NAS...")
    full_train_dataset = BIRDGraphDataset(
        json_path=cfg['paths']["train_json"],
        db_dir=cfg['paths']["train_db_dir"],
        builder=builder,
        encoder=encoder,
    )

    query_conditioned = cfg['model'].get('query_conditioned', False)
    query_supernode = cfg['model'].get('query_supernode', False)

    if query_supernode:
        logger.info("Query Super Node mode: injecting query nodes into graphs...")
        full_train_dataset = BIRDSuperNodeDataset(full_train_dataset)

    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_ds, val_ds = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg['training'].get('batch_size', 8), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['training'].get('batch_size', 8), shuffle=False)

    # Models
    gat_model = SchemaHeteroGAT(
        in_channels=cfg['model']['in_channels'],
        hidden_channels=cfg['model']['hidden_channels'],
        out_channels=cfg['model']['out_channels'],
        num_layers=cfg['model']['num_layers'],
        heads=cfg['model']['heads'],
        query_conditioned=query_conditioned,
        query_supernode=query_supernode,
    ).to(device)

    if query_conditioned:
        logger.info("Query-Conditioned GAT enabled (Concatenation mode)")
    if query_supernode:
        logger.info("Query-Conditioned GAT enabled (Super Node mode)")

    # DirectClassifierHead: node type별 독립 MLP (논문에서 table/column/fk_node 역할이 다르므로)
    classifier_types = ['table', 'column', 'fk_node']
    classifier_heads = nn.ModuleDict({
        nt: DirectClassifierHead(
            in_dim=cfg['model']['out_channels'],
            hidden_dim=cfg['model'].get('classifier_hidden', 256),
            dropout=cfg['model'].get('dropout', 0.1),
        ).to(device)
        for nt in classifier_types
    })
    logger.info(f"DirectClassifierHead initialized for: {list(classifier_heads.keys())}")

    # Lazy-init GATv2Conv weights with a dummy batch
    logger.info("Initializing model parameters with a dummy batch...")
    gat_model.train()
    dummy_batch = full_train_dataset[0].clone().to(device)
    with torch.no_grad():
        if query_conditioned:
            dummy_q = dummy_batch['query']
            if dummy_q.dim() >= 2:
                dummy_q = dummy_q.mean(dim=0, keepdim=True)
            else:
                dummy_q = dummy_q.unsqueeze(0)
            augmented_x = {}
            for nt, x in dummy_batch.x_dict.items():
                q_exp = dummy_q.expand(x.size(0), -1)
                augmented_x[nt] = torch.cat([x, q_exp], dim=-1)
            _ = gat_model(augmented_x, dummy_batch.edge_index_dict)
        else:
            _ = gat_model(dummy_batch.x_dict, dummy_batch.edge_index_dict)

    wandb.watch(gat_model, log="all")
    logger.info("Parameters initialized and WandB watch started.")

    optimizer = torch.optim.AdamW(
        list(gat_model.parameters()) + list(classifier_heads.parameters()),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay'],
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([cfg['training']['pos_weight']]).to(device)
    )

    best_recall = 0.0
    epochs = cfg['training']['epochs']

    for epoch in range(epochs):
        gat_model.train()
        for head in classifier_heads.values():
            head.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            batch = batch.to(device)
            optimizer.zero_grad()

            q_emb = batch['query']
            if query_conditioned:
                if q_emb.dim() == 3:
                    q_emb_pooled = q_emb.mean(dim=1)
                elif q_emb.dim() == 2:
                    q_emb_pooled = q_emb
                else:
                    q_emb_pooled = q_emb.unsqueeze(0)

                augmented_x = {}
                for n_type, x in batch.x_dict.items():
                    node_batch_idx = batch[n_type].batch
                    q_per_node = q_emb_pooled[node_batch_idx]
                    augmented_x[n_type] = torch.cat([x, q_per_node], dim=-1)
                node_embs = gat_model(augmented_x, batch.edge_index_dict)
            else:
                node_embs = gat_model(batch.x_dict, batch.edge_index_dict)

            step_loss = 0.0
            for n_type in classifier_types:
                if n_type not in node_embs or not hasattr(batch[n_type], 'y'):
                    continue
                if batch[n_type].num_nodes == 0:
                    continue
                logits = classifier_heads[n_type](node_embs[n_type])
                bce_loss = criterion(logits, batch[n_type].y)
                step_loss = step_loss + bce_loss

            if not torch.is_tensor(step_loss):
                continue

            step_loss.backward()
            optimizer.step()

            epoch_loss += step_loss.item()

            if step % 10 == 0:
                wandb.log({
                    "train/loss_bce": step_loss.item(),
                    "epoch": epoch + 1,
                })
            pbar.set_postfix({"loss": f"{step_loss.item():.4f}"})

        val_recall = validate(
            gat_model, classifier_heads, val_loader, device, k=15,
            query_conditioned=query_conditioned,
            query_supernode=query_supernode,
        )

        wandb.log({
            "train/epoch_loss": epoch_loss / max(len(train_loader), 1),
            "val/recall_at_15": val_recall,
            "epoch": epoch + 1,
        })

        logger.info(
            f"Epoch {epoch+1} | Loss: {epoch_loss/max(len(train_loader),1):.4f} "
            f"| Val Recall@15: {val_recall:.4f}"
        )

        if val_recall > best_recall:
            best_recall = val_recall
            ckpt_name = cfg.get('checkpoint_name', 'best_gat_direct.pt')
            save_path = os.path.join(PATHS["checkpoint_dir"], ckpt_name)
            torch.save({
                'epoch': epoch + 1,
                'gat_state_dict': gat_model.state_dict(),
                'classifier_state_dict': classifier_heads.state_dict(),
                'recall': val_recall,
            }, save_path)
            wandb.run.summary["best_val_recall"] = best_recall
            logger.info(f"New Best Model Saved! Recall: {best_recall:.4f}")

    logger.info(f"Training Completed. Best Recall: {best_recall:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run_train(args.config)
