"""
s06 ablation — GAT Bottleneck Fix 전용 학습 스크립트.

기존 train_gat_direct.py 에서 확장된 요소:
  - SchemaHeteroGATv2 사용 (PairNorm / Initial residual / JK / Dual-stream)
  - ListNet loss 옵션
  - AntiCollapseRegularizer 옵션 (column 임베딩에 대해 intra-table cosine 제약)

config yaml 의 s06 전용 키 (model):
  pairnorm_mode: 'none' | 'pairnorm'
  pairnorm_scale: float (default 1.0)
  initial_residual_alpha: float (default 0.0)
  jumping_knowledge: 'none' | 'concat' | 'max'
  dual_stream: bool

config yaml 의 s06 전용 키 (training):
  loss_type: 'bce' | 'listnet' | 'bce_listnet'
  anti_collapse_weight: float (default 0.0)
  anti_collapse_tau_max: float (default 0.85)
"""
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass
os.environ.setdefault("WANDB_DIR", str(Path(__file__).resolve().parents[1]))
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
from models.gat_network_v2 import SchemaHeteroGATv2
from models.direct_classifier import DirectClassifierHead
from models.losses import ListNetLoss, AntiCollapseRegularizer, combined_loss
from utils.logger import setup_logger, get_logger


PATHS = {
    "checkpoint_dir": "./outputs/checkpoints",
    "cache_dir": "./data/processed",
}

COL_TO_TAB_EDGE = ("column", "belongs_to", "table")


def calculate_recall_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int = 15) -> float:
    if labels.sum() == 0:
        return 0.0
    k_actual = min(k, logits.size(0))
    _, top_k_indices = torch.topk(logits, k_actual)
    hits = labels[top_k_indices].sum().item()
    return hits / labels.sum().item()


def validate(gat_model, classifier_heads, loader, device, k=15,
             query_conditioned=False, query_supernode=False, dual_stream=False):
    gat_model.eval()
    for head in classifier_heads.values():
        head.eval()
    total_recall = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            q_emb = batch["query"]

            # query pooling (3D → 2D)
            if q_emb.dim() == 3:
                q_pooled = q_emb.mean(dim=1)
            elif q_emb.dim() == 2:
                q_pooled = q_emb
            else:
                q_pooled = q_emb.unsqueeze(0)

            # forward
            node_batch_dict = {
                nt: batch[nt].batch
                for nt in batch.node_types
                if hasattr(batch[nt], "batch")
            }
            if query_conditioned and not dual_stream:
                augmented_x = {}
                for n_type, x in batch.x_dict.items():
                    node_batch_idx = batch[n_type].batch
                    q_per_node = q_pooled[node_batch_idx]
                    augmented_x[n_type] = torch.cat([x, q_per_node], dim=-1)
                node_embs_dict = gat_model(augmented_x, batch.edge_index_dict)
            elif dual_stream:
                # dual_stream: query_emb [B, d_q] 를 node_batch_dict 로 per-graph fusion
                node_embs_dict = gat_model(
                    batch.x_dict, batch.edge_index_dict,
                    query_emb=q_pooled,
                    node_batch_dict=node_batch_dict,
                )
            else:
                node_embs_dict = gat_model(batch.x_dict, batch.edge_index_dict)

            for i in range(batch.num_graphs):
                logits_list, labels_list = [], []
                for n_type in ["table", "column"]:
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
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg_paths = cfg.get("paths", {})
    if "checkpoint_dir" in cfg_paths:
        PATHS["checkpoint_dir"] = cfg_paths["checkpoint_dir"]
    if "cache_dir" in cfg_paths:
        PATHS["cache_dir"] = cfg_paths["cache_dir"]

    wandb.init(
        project=cfg.get("project_name", os.getenv("WANDB_PROJECT", "Text-to-SQL-Alignment")),
        name=cfg["experiment_name"],
        config=cfg,
    )

    setup_logger(log_dir="./logs/", exp_name=cfg["experiment_name"], sub_dir="train")
    logger = get_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(PATHS["checkpoint_dir"], exist_ok=True)
    os.makedirs(PATHS["cache_dir"], exist_ok=True)
    logger.info(f"[s06] checkpoint_dir={PATHS['checkpoint_dir']}")

    # Builder / Encoder
    builder_type = cfg.get("builder", {}).get("type", "HeteroGraphBuilder")
    if builder_type == "EnrichedHeteroGraphBuilder":
        tables_json = cfg["builder"].get("tables_json_path", "")
        builder = EnrichedHeteroGraphBuilder(tables_json_path=tables_json)
        logger.info(f"Using EnrichedHeteroGraphBuilder (tables_json={tables_json})")
    else:
        builder = HeteroGraphBuilder()
    encoder = LocalPLMEncoder()

    logger.info("Loading Training Dataset...")
    full_train_dataset = BIRDGraphDataset(
        json_path=cfg["paths"]["train_json"],
        db_dir=cfg["paths"]["train_db_dir"],
        builder=builder,
        encoder=encoder,
    )

    query_conditioned = cfg["model"].get("query_conditioned", False)
    query_supernode = cfg["model"].get("query_supernode", False)
    dual_stream = cfg["model"].get("dual_stream", False)

    if query_supernode:
        logger.info("Wrapping with BIRDSuperNodeDataset (query_supernode=True)")
        full_train_dataset = BIRDSuperNodeDataset(full_train_dataset)

    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_ds, val_ds = random_split(full_train_dataset, [train_size, val_size])

    # dual_stream 도 node_batch_dict 를 통해 batched forward 지원 (config 에서 지정)
    batch_size = cfg["training"].get("batch_size", 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    # Proposal C H2 (V-1 per-DB dynamic num_layers) config forward.
    # 기본값 'fixed' 에서는 v1 SchemaHeteroGAT 과 bit-wise 동일 경로로 학습. 'D_max' /
    # 'D_max_plus1' 은 diameter_dict 를 model 에 전달하여 runtime resolve. 학습 시에는 일반적으로
    # 'fixed' (= global num_layers) 로 두고, inference 단계에서 selector/config 가 per-DB depth 를
    # 해제하는 경로 (EnsembleSelector.num_layers_mode) 를 쓰는 것을 권장. 필요 시 학습 시에도
    # per-DB depth curriculum 을 돌릴 수 있도록 forward 만 해 둔다.
    diameter_path_cfg = cfg["model"].get("diameter_path")
    diameter_dict_cfg = cfg["model"].get("diameter_dict")
    gat_model = SchemaHeteroGATv2(
        in_channels=cfg["model"]["in_channels"],
        hidden_channels=cfg["model"]["hidden_channels"],
        out_channels=cfg["model"]["out_channels"],
        num_layers=cfg["model"]["num_layers"],
        heads=cfg["model"]["heads"],
        query_conditioned=query_conditioned,
        query_supernode=query_supernode,
        pairnorm_mode=cfg["model"].get("pairnorm_mode", "none"),
        pairnorm_scale=cfg["model"].get("pairnorm_scale", 1.0),
        initial_residual_alpha=cfg["model"].get("initial_residual_alpha", 0.0),
        jumping_knowledge=cfg["model"].get("jumping_knowledge", "none"),
        dual_stream=dual_stream,
        num_layers_mode=cfg["model"].get("num_layers_mode", "fixed"),
        num_layers_fallback=cfg["model"].get(
            "num_layers_fallback", cfg["model"]["num_layers"]
        ),
        diameter_path=diameter_path_cfg,
        diameter_dict=diameter_dict_cfg,
    ).to(device)

    logger.info(
        f"[s06] model: QC={query_conditioned}, SN={query_supernode}, DS={dual_stream}, "
        f"PN={cfg['model'].get('pairnorm_mode', 'none')}, "
        f"α={cfg['model'].get('initial_residual_alpha', 0.0)}, "
        f"JK={cfg['model'].get('jumping_knowledge', 'none')}, "
        f"nl_mode={cfg['model'].get('num_layers_mode', 'fixed')}, "
        f"|diameter_dict|={len(gat_model.diameter_dict)}"
    )

    classifier_types = ["table", "column", "fk_node"]
    classifier_heads = nn.ModuleDict(
        {
            nt: DirectClassifierHead(
                in_dim=cfg["model"]["out_channels"],
                hidden_dim=cfg["model"].get("classifier_hidden", 256),
                dropout=cfg["model"].get("dropout", 0.1),
            ).to(device)
            for nt in classifier_types
        }
    )
    logger.info(f"DirectClassifierHead initialized for: {list(classifier_heads.keys())}")

    # Lazy-init
    logger.info("Initializing model parameters with a dummy batch...")
    gat_model.train()
    dummy_batch = full_train_dataset[0].clone().to(device)
    with torch.no_grad():
        if query_conditioned and not dual_stream:
            dummy_q = dummy_batch["query"]
            if dummy_q.dim() >= 2:
                dummy_q = dummy_q.mean(dim=0, keepdim=True)
            else:
                dummy_q = dummy_q.unsqueeze(0)
            augmented_x = {}
            for nt, x in dummy_batch.x_dict.items():
                q_exp = dummy_q.expand(x.size(0), -1)
                augmented_x[nt] = torch.cat([x, q_exp], dim=-1)
            _ = gat_model(augmented_x, dummy_batch.edge_index_dict)
        elif dual_stream:
            dummy_q = dummy_batch["query"]
            if dummy_q.dim() >= 2:
                dummy_q = dummy_q.mean(dim=0)
            _ = gat_model(dummy_batch.x_dict, dummy_batch.edge_index_dict, query_emb=dummy_q)
        else:
            _ = gat_model(dummy_batch.x_dict, dummy_batch.edge_index_dict)

    wandb.watch(gat_model, log="all")

    optimizer = torch.optim.AdamW(
        list(gat_model.parameters()) + list(classifier_heads.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Loss 설정
    loss_type = cfg["training"].get("loss_type", "bce")
    pos_weight = torch.tensor([cfg["training"]["pos_weight"]]).to(device)
    listnet_fn = ListNetLoss().to(device)
    anti_collapse_weight = float(cfg["training"].get("anti_collapse_weight", 0.0))
    anti_collapse_fn = AntiCollapseRegularizer(
        tau_max=float(cfg["training"].get("anti_collapse_tau_max", 0.85))
    ).to(device)

    logger.info(
        f"[s06] loss: type={loss_type}, pos_weight={pos_weight.item()}, "
        f"anti_collapse_weight={anti_collapse_weight}"
    )

    best_recall = 0.0
    epochs = cfg["training"]["epochs"]

    for epoch in range(epochs):
        gat_model.train()
        for head in classifier_heads.values():
            head.train()
        epoch_loss = 0.0
        epoch_loss_main = 0.0
        epoch_loss_ac = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            batch = batch.to(device)
            optimizer.zero_grad()

            q_emb = batch["query"]
            if q_emb.dim() == 3:
                q_pooled = q_emb.mean(dim=1)
            elif q_emb.dim() == 2:
                q_pooled = q_emb
            else:
                q_pooled = q_emb.unsqueeze(0)

            node_batch_dict = {
                nt: batch[nt].batch
                for nt in batch.node_types
                if hasattr(batch[nt], "batch")
            }
            if query_conditioned and not dual_stream:
                augmented_x = {}
                for n_type, x in batch.x_dict.items():
                    node_batch_idx = batch[n_type].batch
                    q_per_node = q_pooled[node_batch_idx]
                    augmented_x[n_type] = torch.cat([x, q_per_node], dim=-1)
                node_embs = gat_model(augmented_x, batch.edge_index_dict)
            elif dual_stream:
                # batched dual_stream: q_pooled [B, d_q] + node_batch_dict 로 per-graph fusion
                node_embs = gat_model(
                    batch.x_dict, batch.edge_index_dict,
                    query_emb=q_pooled,
                    node_batch_dict=node_batch_dict,
                )
            else:
                node_embs = gat_model(batch.x_dict, batch.edge_index_dict)

            # Main loss (per-query listwise or BCE)
            step_loss_main = 0.0
            n_terms = 0
            if loss_type in ("listnet", "bce_listnet"):
                # per-query 로 쪼개서 계산
                for i in range(batch.num_graphs):
                    logits_q, labels_q = [], []
                    for n_type in classifier_types:
                        if n_type not in node_embs or not hasattr(batch[n_type], "y"):
                            continue
                        mask = (batch[n_type].batch == i)
                        if not mask.any():
                            continue
                        logits = classifier_heads[n_type](node_embs[n_type][mask])
                        logits_q.append(logits)
                        labels_q.append(batch[n_type].y[mask])
                    if logits_q:
                        all_logits = torch.cat(logits_q)
                        all_labels = torch.cat(labels_q)
                        step_loss_main = step_loss_main + combined_loss(
                            all_logits, all_labels, loss_type=loss_type,
                            pos_weight=pos_weight, listnet_fn=listnet_fn,
                        )
                        n_terms += 1
                if n_terms > 0:
                    step_loss_main = step_loss_main / n_terms
                else:
                    continue
            else:  # 'bce' — per-node BCE, batch-wide
                for n_type in classifier_types:
                    if n_type not in node_embs or not hasattr(batch[n_type], "y"):
                        continue
                    if batch[n_type].num_nodes == 0:
                        continue
                    logits = classifier_heads[n_type](node_embs[n_type])
                    step_loss_main = step_loss_main + combined_loss(
                        logits, batch[n_type].y, loss_type="bce",
                        pos_weight=pos_weight, listnet_fn=listnet_fn,
                    )
                    n_terms += 1
                if not torch.is_tensor(step_loss_main):
                    continue

            # Anti-collapse regularization — column embedding 기준
            step_loss_ac = torch.tensor(0.0, device=device)
            if anti_collapse_weight > 0.0 and "column" in node_embs:
                if COL_TO_TAB_EDGE in batch.edge_index_dict:
                    col_embs = node_embs["column"]
                    cb_edge = batch.edge_index_dict[COL_TO_TAB_EDGE]
                    step_loss_ac = anti_collapse_fn(col_embs, cb_edge)

            step_loss = step_loss_main + anti_collapse_weight * step_loss_ac

            step_loss.backward()
            optimizer.step()

            epoch_loss += step_loss.item()
            epoch_loss_main += float(step_loss_main.item() if torch.is_tensor(step_loss_main) else step_loss_main)
            epoch_loss_ac += float(step_loss_ac.item() if torch.is_tensor(step_loss_ac) else step_loss_ac)

            if step % 10 == 0:
                wandb.log({
                    "train/loss_total": step_loss.item(),
                    "train/loss_main": float(step_loss_main.item() if torch.is_tensor(step_loss_main) else step_loss_main),
                    "train/loss_anti_collapse": float(step_loss_ac.item() if torch.is_tensor(step_loss_ac) else step_loss_ac),
                    "epoch": epoch + 1,
                })
            pbar.set_postfix({"loss": f"{step_loss.item():.4f}"})

        val_recall = validate(
            gat_model, classifier_heads, val_loader, device, k=15,
            query_conditioned=query_conditioned,
            query_supernode=query_supernode,
            dual_stream=dual_stream,
        )

        n_batches = max(len(train_loader), 1)
        wandb.log({
            "train/epoch_loss_total": epoch_loss / n_batches,
            "train/epoch_loss_main": epoch_loss_main / n_batches,
            "train/epoch_loss_anti_collapse": epoch_loss_ac / n_batches,
            "val/recall_at_15": val_recall,
            "epoch": epoch + 1,
        })

        logger.info(
            f"Epoch {epoch+1} | Loss: {epoch_loss/n_batches:.4f} "
            f"| Main: {epoch_loss_main/n_batches:.4f} "
            f"| AC: {epoch_loss_ac/n_batches:.4f} "
            f"| Val Recall@15: {val_recall:.4f}"
        )

        if val_recall > best_recall:
            best_recall = val_recall
            ckpt_name = cfg.get("checkpoint_name", "best_gat_s06.pt")
            save_path = os.path.join(PATHS["checkpoint_dir"], ckpt_name)
            torch.save({
                "epoch": epoch + 1,
                "gat_state_dict": gat_model.state_dict(),
                "classifier_state_dict": classifier_heads.state_dict(),
                "recall": val_recall,
                "config": cfg,
            }, save_path)
            wandb.run.summary["best_val_recall"] = best_recall
            logger.info(f"New Best Model Saved! Recall: {best_recall:.4f}")

    logger.info(f"Training Completed. Best Recall: {best_recall:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run_train(args.config)
