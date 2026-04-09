import os
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import wandb
import argparse

# 프로젝트 내부 모듈 임포트
from data.bird_dataset import BIRDGraphDataset  # 아래에 Dataset 코드도 포함해 두었습니다.
from modules.builders.graph_builder import HeteroGraphBuilder
from modules.encoders.token_encoder import TokenEncoder
from models.gat_network import SchemaHeteroGAT
from modules.projectors.dual_tower import DualTowerProjector
from utils.logger import setup_logger, get_logger

# ----------------------------------------------------------------
# 1. 경로 설정
# ----------------------------------------------------------------
PATHS = {
    "train_json": "/SSL_NAS/peoples/khj/thesis/train/train.json",
    "train_db_dir": "/SSL_NAS/peoples/khj/thesis/train/train_databases",
    "test_json": "/home/hyeonjin/thesis_refactored/data/raw/BIRD_dev/dev.json",
    "test_db_dir": "/home/hyeonjin/thesis_refactored/data/raw/BIRD_dev/dev_databases",
    "checkpoint_dir": "./outputs/checkpoints",
    "cache_dir": "./data/processed" # NAS 병목 방지용 로컬 캐시
}

# ----------------------------------------------------------------
# 2. InfoNCE Loss with Hard Negative Mining
# ----------------------------------------------------------------
def compute_batched_infonce_loss(z_q: torch.Tensor, z_n: torch.Tensor, labels: torch.Tensor, 
                                 batch_idx: torch.Tensor, temperature: float = 0.07, num_hard_negatives: int = 15) -> torch.Tensor:
    """
    미니 배치 내의 각 그래프별로 정답(Positive)과 헷갈리는 오답(Hard Negative)을 추출하여 InfoNCE를 계산합니다.
    """
    # 1. 투영된 조인트 임베딩 공간에서의 코사인 유사도 계산
    sim = F.cosine_similarity(z_q, z_n) # [N_nodes]
    
    total_loss = 0.0
    num_graphs = batch_idx.max().item() + 1
    valid_graphs = 0

    for i in range(num_graphs):
        mask = (batch_idx == i)
        if not mask.any(): continue

        g_sim = sim[mask]
        g_labels = labels[mask]

        pos_mask = (g_labels == 1)
        neg_mask = (g_labels == 0)

        # 정답이나 오답 노드가 아예 없는 그래프는 건너뜀 (Zero Division 방지)
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        pos_sim = g_sim[pos_mask]
        neg_sim = g_sim[neg_mask]

        # [Hard Negative Mining] 유사도가 높은 오답 노드들만 K개 추출
        if neg_sim.size(0) > num_hard_negatives:
            hard_neg_sim, _ = torch.topk(neg_sim, num_hard_negatives)
        else:
            hard_neg_sim = neg_sim

        # Log-Sum-Exp를 통한 InfoNCE Loss 계산
        pos_sim_exp = torch.exp(pos_sim / temperature)
        neg_sim_exp_sum = torch.exp(hard_neg_sim / temperature).sum()

        loss = -torch.log(pos_sim_exp / (pos_sim_exp + neg_sim_exp_sum))
        total_loss += loss.mean()
        valid_graphs += 1

    if valid_graphs == 0:
        return torch.tensor(0.0, device=z_q.device, requires_grad=True)
        
    return total_loss / valid_graphs

# ----------------------------------------------------------------
# 2. Validation Recall 계산 함수
# ----------------------------------------------------------------
def calculate_recall_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int = 15) -> float:
    if labels.sum() == 0: return 0.0
    k_actual = min(k, logits.size(0))
    _, top_k_indices = torch.topk(logits, k_actual)
    hits = labels[top_k_indices].sum().item()
    return hits / labels.sum().item()

def validate(gat_model, projector, loader, device, k=15):
    gat_model.eval()
    projector.eval()
    total_recall = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            node_embs_dict = gat_model(batch.x_dict, batch.edge_index_dict)
            q_emb = batch['query']
            
            # 그래프 별로 순회하며 Recall 계산
            for i in range(batch.num_graphs):
                logits_list, labels_list = [], []
                for n_type in ['table', 'column']: # FK 노드는 Recall 평가에서 제외 (선택 사항)
                    mask = (batch[n_type].batch == i)
                    if not mask.any(): continue
                    
                    z_q, z_n = projector(q_emb[i].unsqueeze(0), node_embs_dict[n_type][mask])
                    score = projector.compute_similarity(z_q, z_n)
                    logits_list.append(score)
                    labels_list.append(batch[n_type].y[mask])
                
                if logits_list:
                    all_logits = torch.cat(logits_list)
                    all_labels = torch.cat(labels_list)
                    total_recall += calculate_recall_at_k(all_logits, all_labels, k=k)
                    count += 1
    return total_recall / count if count > 0 else 0.0

# ----------------------------------------------------------------
# 3. 메인 학습 루프
# ----------------------------------------------------------------
def run_train(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    wandb.init(
        project=cfg['project_name'],
        name=cfg['experiment_name'],
        config=cfg
    )

    setup_logger(log_dir="./logs/", exp_name=cfg['experiment_name'], sub_dir="train")
    logger = get_logger(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(PATHS["checkpoint_dir"], exist_ok=True)
    os.makedirs(PATHS["cache_dir"], exist_ok=True)

    # 컴포넌트 초기화
    builder = HeteroGraphBuilder()
    encoder = TokenEncoder() 

    # 데이터셋 로드 (학습용)
    logger.info("🚀 Loading Training Dataset from NAS...")
    full_train_dataset = BIRDGraphDataset(
        json_path=cfg['paths']["train_json"], 
        db_dir=cfg['paths']["train_db_dir"], 
        builder=builder, 
        encoder=encoder
    )
    
    # 9:1 분할 (내부 Validation 생성)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_ds, val_ds = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # 모델 준비
    gat_model = SchemaHeteroGAT(
        in_channels=cfg['model']['in_channels'], 
        hidden_channels=cfg['model']['hidden_channels'], 
        out_channels=cfg['model']['out_channels'],
        num_layers=cfg['model']['num_layers'],
        heads=cfg['model']['heads']
        ).to(device)
    
    projector = DualTowerProjector(
        text_dim=cfg['model']['in_channels'], 
        graph_dim=cfg['model']['hidden_channels'], 
        joint_dim=cfg['model']['hidden_channels']
        ).to(device)

    logger.info("Initializing model parameters with a dummy batch...")
    gat_model.train()
    
    # 데이터셋에서 첫 번째 샘플 하나를 가져옵니다.
    dummy_batch = full_train_dataset[0].clone().to(device)
    
    # 모델에 한 번 통과시켜 가중치를 생성합니다.
    with torch.no_grad():
        _ = gat_model(dummy_batch.x_dict, dummy_batch.edge_index_dict)
    
    # 이제 가중치가 생성되었으므로 wandb.watch가 작동합니다.
    wandb.watch(gat_model, log="all")
    logger.info("✅ Parameters initialized and WandB watch started.")
    
    optimizer = torch.optim.AdamW(
        list(gat_model.parameters()) + list(projector.parameters()), 
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay'])
    
    # 정답 노드(Gold)가 적으므로 가중치 부여 (BCEWithLogitsLoss)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([cfg['training']['pos_weight']]).to(device)
    )

    infonce_lambda = cfg['training'].get('infonce_lambda', 0.5)
    temperature = cfg['training'].get('temperature', 0.07)
    num_hard_negatives = cfg['training'].get('num_hard_negatives', 15)

    best_recall = 0.0
    epochs = cfg['training']['epochs']

    for epoch in range(epochs):
        gat_model.train()
        projector.train()
        epoch_loss = 0
        epoch_bce_loss = 0
        epoch_infonce_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            batch = batch.to(device)
            optimizer.zero_grad()

            # GAT -> Node Embeddings
            node_embs = gat_model(batch.x_dict, batch.edge_index_dict)
            q_emb = batch['query']

            step_bce_loss = 0
            step_infonce_loss = 0
            
            for n_type in ['table', 'column', 'fk_node']:
                if n_type not in node_embs or not hasattr(batch[n_type], 'y'): continue
                if batch[n_type].num_nodes == 0: continue
                
                # 1. 조인트 임베딩 투영
                z_q, z_n = projector(q_emb, node_embs[n_type], batch_index=batch[n_type].batch)
                
                # 2. BCE Loss 계산 (절대적 존재 여부)
                logits = projector.compute_similarity(z_q, z_n)
                bce_loss = criterion(logits, batch[n_type].y)
                step_bce_loss += bce_loss

                # 3. [신규] InfoNCE Loss 계산 (상대적 유사도 최적화 및 하드 네거티브 마이닝)
                infonce_loss = compute_batched_infonce_loss(
                    z_q, z_n, batch[n_type].y, batch[n_type].batch,
                    temperature=temperature, num_hard_negatives=num_hard_negatives
                )
                step_infonce_loss += infonce_loss

            # 4. Joint Loss 백프로파게이션
            total_loss = step_bce_loss + (infonce_lambda * step_infonce_loss)
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_bce_loss += step_bce_loss.item()
            epoch_infonce_loss += step_infonce_loss.item()

            if step % 10 == 0:
                wandb.log({
                    "train/loss_total": total_loss.item(),
                    "train/loss_bce": step_bce_loss.item(),
                    "train/loss_infonce": step_infonce_loss.item(),
                    "epoch": epoch + 1
                })
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "infoNCE": f"{step_infonce_loss.item():.4f}"})

        # 검증 (Recall@15)
        val_recall = validate(gat_model, projector, val_loader, device, k=15)
        
        wandb.log({
            "train/epoch_loss": epoch_loss / len(train_loader),
            "val/recall_at_15": val_recall,
            "epoch": epoch + 1
        })

        logger.info(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | BCE: {epoch_bce_loss/len(train_loader):.4f} | InfoNCE: {epoch_infonce_loss/len(train_loader):.4f} | Val Recall@15: {val_recall:.4f}")

        if val_recall > best_recall:
            best_recall = val_recall
            save_path = os.path.join(PATHS["checkpoint_dir"], "best_gat_model.pt")
            torch.save({
                'epoch': epoch + 1,
                'gat_state_dict': gat_model.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'recall': val_recall
            }, save_path)
            wandb.run.summary["best_val_recall"] = best_recall
            logger.info(f"✨ New Best Model Saved! Recall: {best_recall:.4f}")

    logger.info(f"✅ Training Completed. Best Recall: {best_recall:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training/train_gat_config.yaml")
    args = parser.parse_args()
    
    run_train(args.config)