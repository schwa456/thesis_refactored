import os
import yaml
import torch
import wandb
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader 

from models.node_classifier import SimpleNodeLinker
from data.bird_dataset import BIRDGraphDataset
from modules.builders.graph_builder import HeteroGraphBuilder
from modules.encoders.token_encoder import TokenEncoder
from utils.evaluator import parse_sql_elements
from utils.logger import setup_logger, get_logger

def create_labels(metadata, gold_tables, gold_cols):
    """Gold Schema를 그래프 노드의 이진 레이블(0 or 1) 텐서로 변환합니다."""
    # 1. Table Labels 생성
    num_tables = len(metadata['table_to_id'])
    table_labels = torch.zeros(num_tables, dtype=torch.float)
    for t_name, t_id in metadata['table_to_id'].items():
        if t_name.lower() in gold_tables:
            table_labels[t_id] = 1.0
            
    # 2. Column Labels 생성
    num_cols = len(metadata['col_to_id'])
    column_labels = torch.zeros(num_cols, dtype=torch.float)
    for c_name, c_id in metadata['col_to_id'].items():
        # c_name 형식이 'table.column'인 경우 분리하여 체크
        if '.' in c_name:
            t, c = c_name.lower().split('.', 1)
            if t in gold_tables and c in gold_cols:
                column_labels[c_id] = 1.0
                
    return table_labels, column_labels

def calculate_metrics(logits, labels, threshold=0.5):
    """이진 분류 결과에 대해 Precision, Recall, F1을 계산합니다."""
    preds = (torch.sigmoid(logits) > threshold).float()
    
    tp = (preds * labels).sum().item()
    fp = (preds * (1 - labels)).sum().item()
    fn = ((1 - preds) * labels).sum().item()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return precision, recall, f1

@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    val_loss = 0.0
    metrics = {
        't_rec': 0, 'c_rec': 0, # Recall 집중 관리
        't_pre': 0, 'c_pre': 0
    }
    
    for batch in loader:
        batch = batch.to(device)

        q_embs = batch['query']
        label_t = batch['table'].y.float()
        label_c = batch['column'].y
        print(f"[DEBUG] Column Positive Labels: {label_c.sum().item()} / Total: {label_c.numel()}")
        
        q_embs = batch['query']

        out = model(batch, q_embs)
        column_logits = out['column']
        print(f"[DEBUG] Column Logits Mean: {column_logits.mean().item():.4f}")
        
        preds_c = (torch.sigmoid(column_logits) > 0.5).float()
        print(f"[DEBUG] Column Predicted Positives: {preds_c.sum().item()}")
        batch = batch.to(device)
        
        loss = 0
        if 'table' in out:
            loss += criterion(out['table'], label_t)
            p, r, _ = calculate_metrics(out['table'], label_t)
            metrics['t_pre'] += p; metrics['t_rec'] += r
            
        if 'column' in out:
            loss += criterion(out['column'], label_c)
            p, r, _ = calculate_metrics(out['column'], label_c)
            metrics['c_pre'] += p; metrics['c_rec'] += r
            
        val_loss += loss.item()
        
    n = len(loader)
    # 전체 평균 Recall 산출 (Table Recall과 Column Recall의 산술 평균)
    avg_t_rec = metrics['t_rec'] / n
    avg_c_rec = metrics['c_rec'] / n
    overall_recall = (avg_t_rec + avg_c_rec) / 2
    
    return {
        'loss': val_loss / n,
        't_rec': avg_t_rec,
        'c_rec': avg_c_rec,
        'overall_recall': overall_recall,
        't_pre': metrics['t_pre'] / n,
        'c_pre': metrics['c_pre'] / n
    }

def main(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Config 구조에 맞게 매핑 (기존 train.py 참고)
    exp_name = cfg.get('experiment_name', 'mlp_classifier_train')
    
    wandb.init(project=cfg.get('project_name', "Text-to-SQL-Alignment"), name=exp_name, config=cfg)
    setup_logger(log_dir="./logs/", exp_name=exp_name, sub_dir="train")
    logger = get_logger(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==========================================
    # 1. Dataset & DataLoader (PyG 기반)
    # ==========================================
    builder = HeteroGraphBuilder()
    encoder = TokenEncoder() 
    
    logger.info("🚀 Loading Training Dataset...")
    full_train_dataset = BIRDGraphDataset(
        json_path=cfg['paths']["train_json"], 
        db_dir=cfg['paths']["train_db_dir"], 
        builder=builder, 
        encoder=encoder
    )
    
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_ds, val_ds = random_split(full_train_dataset, [train_size, val_size])

    # PyG DataLoader 사용 (그래프 배치 처리 완벽 지원)
    train_loader = DataLoader(train_ds, batch_size=cfg['training'].get('batch_size', 1), shuffle=True, exclude_keys=['ATT_CLASSES'])
    val_loader = DataLoader(val_ds, batch_size=cfg['training'].get('batch_size', 1), shuffle=False, exclude_keys=['ATT_CLASSES'])

    # ==========================================
    # 2. Model Initialization & Freezing
    # ==========================================
    model = SimpleNodeLinker(
        hidden_dim=cfg['model']['hidden_channels'],
        plm_name=cfg['model']['plm_name'], 
        in_channels=cfg['model']['in_channels']
    ).to(device)

    # [핵심] 1. 사전에 학습된 GAT 가중치 로드
    pretrained_gat_path = os.path.join(cfg['paths'].get('checkpoint_dir', './outputs/checkpoints'), "best_gat_model.pt")
    if os.path.exists(pretrained_gat_path):
        checkpoint = torch.load(pretrained_gat_path, map_location=device)
        # 딕셔너리에서 gat_state_dict만 추출하여 로드
        model.gat.load_state_dict(checkpoint['gat_state_dict'])
        logger.info(f"✅ Loaded pre-trained GAT weights from {pretrained_gat_path}")
    else:
        logger.warning("🚨 Pre-trained GAT weights NOT FOUND. GAT will be randomly initialized!")

    # [핵심] 2. 파라미터 동결 (PLM과 GAT는 얼리고 MLP만 학습)
    for param in model.plm_encoder.parameters():
        param.requires_grad = False
    for param in model.gat.parameters():
        param.requires_grad = True

    wandb.watch(model, log="all")
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(
        trainable_params,
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([cfg['training']['pos_weight']]).to(device)
    )

    # ==========================================
    # 3. Training Loop
    # ==========================================
    best_recall = 0.0
    epochs = cfg['training']['epochs']

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            q_embs = batch['query']
            out = model(batch, q_embs)
            
            loss = 0
            loss += criterion(out['table'], batch['table'].y.float())
            loss += criterion(out['column'], batch['column'].y.float())
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            wandb.log({"train/loss_step": loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        wandb.log({"train/loss_epoch": avg_loss, "epoch": epoch + 1})

        # 에포크 종료 후 검증 수행
        val_res = validate(model, val_loader, device, criterion)
        
        logger.info(f"Epoch {epoch+1} | Val Loss: {val_res['loss']:.4f} | "
                    f"Overall Recall: {val_res['overall_recall']:.4f} "
                    f"(Table: {val_res['t_rec']:.4f}, Col: {val_res['c_rec']:.4f})")
        
        # WandB 로깅
        wandb.log({
            "val/loss": val_res['loss'],
            "val/overall_recall": val_res['overall_recall'],
            "val/table_recall": val_res['t_rec'],
            "val/column_recall": val_res['c_rec'],
            "epoch": epoch + 1
        })
        
        # [수정] Best Recall 기준으로 모델 저장
        if val_res['overall_recall'] > best_recall:
            best_recall = val_res['overall_recall']
            save_path = os.path.join(cfg['paths']['checkpoint_dir'], f"{exp_name}_best_recall.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"✨ New Best Model Saved (Overall Recall: {best_recall:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training/train_classifier_with_gat_config.yaml")
    args = parser.parse_args()
    main(args.config)