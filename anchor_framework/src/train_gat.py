import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass
os.environ.setdefault("WANDB_DIR", str(Path(__file__).resolve().parents[1]))
import json
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
import wandb
import argparse


def _grad_norm(params) -> float:
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += float(p.grad.detach().data.norm(2).item()) ** 2
    return total_sq ** 0.5


def _summarize_layer_stats(layer_stats: Dict[str, Any]) -> Dict[str, float]:
    """Flatten gat.last_layer_stats into a small set of scalar summaries for logging."""
    if not layer_stats:
        return {}
    flat: Dict[str, float] = {}
    flat["gat/forward_time_s"] = float(layer_stats.get("forward_time_s", 0.0))
    for nt, ratio in (layer_stats.get("skip_ratio") or {}).items():
        flat[f"gat/skip_ratio/{nt}"] = float(ratio)
    out = layer_stats.get("output") or {}
    for nt, stat in out.items():
        if not stat:
            continue
        flat[f"gat/out_norm_mean/{nt}"] = float(stat.get("norm_mean", 0.0))
        flat[f"gat/out_dead_ratio/{nt}"] = float(stat.get("dead_ratio", 0.0))
    layers = layer_stats.get("layers") or []
    if layers:
        last = layers[-1]
        for nt, stat in last.items():
            if not stat:
                continue
            flat[f"gat/last_layer_norm_mean/{nt}"] = float(stat.get("norm_mean", 0.0))
            if "delta_norm_mean" in stat:
                flat[f"gat/last_layer_delta/{nt}"] = float(stat["delta_norm_mean"])
    return flat

# 프로젝트 내부 모듈 임포트
from data.bird_dataset import BIRDGraphDataset, BIRDSuperNodeDataset
from modules.builders.graph_builder import HeteroGraphBuilder, EnrichedHeteroGraphBuilder
from modules.encoders.local_encoder import LocalPLMEncoder
from models.gat_network import SchemaHeteroGAT
from modules.projectors.dual_tower import DualTowerProjector
from utils.logger import setup_logger, get_logger

# ----------------------------------------------------------------
# 1. 경로 설정
# ----------------------------------------------------------------
# 기본 경로. 실제 데이터/체크포인트 경로는 config(cfg['paths'])가 우선한다 (run_train 에서 override).
PATHS = {
    "train_json": "data/raw/BIRD_train/train.json",
    "train_db_dir": "data/raw/BIRD_train/train_databases",
    "test_json": "data/raw/BIRD_dev/dev.json",
    "test_db_dir": "data/raw/BIRD_dev/dev_databases",
    "checkpoint_dir": "checkpoints",
    "cache_dir": "data/processed",
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

def validate(gat_model, projector, loader, device, k=15, query_conditioned=False):
    gat_model.eval()
    projector.eval()
    total_recall = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            q_emb = batch['query']
            if query_conditioned:
                # query pooling (token-level → sentence-level)
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
        project=cfg.get('project_name', os.getenv("WANDB_PROJECT", "Text-to-SQL-Alignment")),
        name=cfg['experiment_name'],
        config=cfg
    )

    setup_logger(log_dir="./logs/", exp_name=cfg['experiment_name'], sub_dir="train")
    logger = get_logger(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 체크포인트/캐시 경로도 config 우선 (anchor.yaml weight_path 와 정합 — checkpoints/ 에 저장)
    _cfgp = cfg.get('paths', {}) or {}
    PATHS["checkpoint_dir"] = _cfgp.get('checkpoint_dir', PATHS["checkpoint_dir"])
    PATHS["cache_dir"] = _cfgp.get('cache_dir', PATHS["cache_dir"])
    os.makedirs(PATHS["checkpoint_dir"], exist_ok=True)
    os.makedirs(PATHS["cache_dir"], exist_ok=True)

    # Wave 16 (2026-05-21): encoder model_name 을 cfg 위 받기 (PLM swap 위 정합).
    # Builder 내부 위 node text encoder 도 동일 PLM 위 정합 필수 — 미정합 시 query 1024 + node 384 = 1408 dim mismatch.
    encoder_cfg = cfg.get('nlq_encoder', {}).get('params', {})
    encoder_model_name = encoder_cfg.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')

    # 컴포넌트 초기화
    builder_type = cfg.get('builder', {}).get('type', 'HeteroGraphBuilder')
    if builder_type == 'EnrichedHeteroGraphBuilder':
        tables_json = cfg['builder'].get('tables_json_path', '')
        builder = EnrichedHeteroGraphBuilder(tables_json_path=tables_json, plm_model_name=encoder_model_name)
        logger.info(f"Using EnrichedHeteroGraphBuilder (tables_json={tables_json}, plm={encoder_model_name})")
    else:
        builder = HeteroGraphBuilder(plm_model_name=encoder_model_name)
    encoder = LocalPLMEncoder(model_name=encoder_model_name)
    logger.info(f"Encoder PLM = {encoder_model_name}")

    # 모델 준비 — supernode wrap 은 split 이전에 수행되어야 train/val loader 가 래핑된 데이터셋을 참조.
    query_conditioned = cfg['model'].get('query_conditioned', False)
    query_supernode = cfg['model'].get('query_supernode', False)

    # V-2/V-3 SuperNode ablation (2026-04-21)
    supernode_edge_direction = cfg['model'].get('supernode_edge_direction', 'bidirectional')
    supernode_topk = cfg['model'].get('supernode_topk', None)
    supernode_topk_criterion = cfg['model'].get('supernode_topk_criterion', 'raw')
    # V-3-ext (Directed Top-K SuperNode, 학위 논문 Part III, 2026-05-05)
    # threshold_mode='top_k' (기존, supernode_topk 사용) | 'percentile' (P80 등) | 'abs_tau' (norm 후 절대 cutoff)
    supernode_threshold_mode = cfg['model'].get('supernode_threshold_mode', 'top_k')
    supernode_threshold_value = cfg['model'].get('supernode_threshold_value', None)
    supernode_score_normalization = cfg['model'].get('supernode_score_normalization', 'minmax')

    # 데이터셋 로드 (학습용)
    logger.info("🚀 Loading Training Dataset from NAS...")
    full_train_dataset = BIRDGraphDataset(
        json_path=cfg['paths']["train_json"],
        db_dir=cfg['paths']["train_db_dir"],
        builder=builder,
        encoder=encoder
    )

    # Super Node 모드: split 이전에 래핑해야 random_split 이 래핑된 데이터셋에 대한 Subset 을 반환한다.
    if query_supernode:
        logger.info("Query Super Node mode: injecting query nodes into graphs...")
        full_train_dataset = BIRDSuperNodeDataset(full_train_dataset)

    # 9:1 분할 (내부 Validation 생성)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_ds, val_ds = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    train_log_cfg = cfg.get('training', {}).get('logging', {})
    enable_layer_stats = bool(train_log_cfg.get('enable_layer_stats', True))
    step_log_every = int(train_log_cfg.get('step_log_every', 10))
    layer_stats_every = int(train_log_cfg.get('layer_stats_every', 50))
    jsonl_log_enabled = bool(train_log_cfg.get('jsonl', True))

    gat_model = SchemaHeteroGAT(
        in_channels=cfg['model']['in_channels'],
        hidden_channels=cfg['model']['hidden_channels'],
        out_channels=cfg['model']['out_channels'],
        num_layers=cfg['model']['num_layers'],
        heads=cfg['model']['heads'],
        query_conditioned=query_conditioned,
        query_supernode=query_supernode,
        enable_stats=enable_layer_stats,
        supernode_edge_direction=supernode_edge_direction,
        supernode_topk=supernode_topk,
        supernode_topk_criterion=supernode_topk_criterion,
        supernode_threshold_mode=supernode_threshold_mode,
        supernode_threshold_value=supernode_threshold_value,
        supernode_score_normalization=supernode_score_normalization,
        ).to(device)
    if query_conditioned:
        logger.info("Query-Conditioned GAT enabled (Concatenation mode)")
    if query_supernode:
        logger.info("Query-Conditioned GAT enabled (Super Node mode)")
    
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
        if query_conditioned:
            # query가 token-level [seq_len, dim]일 수 있으므로 mean pooling → [1, dim]
            dummy_q = dummy_batch['query']
            if dummy_q.dim() >= 2:
                dummy_q = dummy_q.mean(dim=0, keepdim=True)  # [1, 384]
            else:
                dummy_q = dummy_q.unsqueeze(0)  # [384] → [1, 384]
            _ = gat_model(dummy_batch.x_dict, dummy_batch.edge_index_dict, query_emb=dummy_q)
        elif query_supernode:
            # Super Node 모드: graph에 이미 query_node가 포함됨
            _ = gat_model(dummy_batch.x_dict, dummy_batch.edge_index_dict)
        else:
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

    # Per-step JSONL log path
    step_log_path: Optional[Path] = None
    if jsonl_log_enabled:
        step_log_dir = Path("./logs") / cfg['experiment_name'] / "train"
        step_log_dir.mkdir(parents=True, exist_ok=True)
        step_log_path = step_log_dir / "train_step.jsonl"
        logger.info(f"Per-step train log → {step_log_path}")

    global_step = 0

    for epoch in range(epochs):
        gat_model.train()
        projector.train()
        epoch_loss = 0
        epoch_bce_loss = 0
        epoch_infonce_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            t_step = time.perf_counter()
            batch = batch.to(device)
            optimizer.zero_grad()

            # Toggle layer_stats computation only on sampled steps (cost)
            if enable_layer_stats:
                gat_model.enable_stats = (global_step % layer_stats_every == 0)

            # GAT -> Node Embeddings
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

            step_bce_loss = 0
            step_infonce_loss = 0
            per_type_bce: Dict[str, float] = {}
            per_type_infonce: Dict[str, float] = {}
            per_type_nodes: Dict[str, int] = {}
            per_type_pos: Dict[str, int] = {}

            for n_type in ['table', 'column', 'fk_node']:
                if n_type not in node_embs or not hasattr(batch[n_type], 'y'): continue
                if batch[n_type].num_nodes == 0: continue

                z_q, z_n = projector(q_emb, node_embs[n_type], batch_index=batch[n_type].batch)

                logits = projector.compute_similarity(z_q, z_n)
                bce_loss = criterion(logits, batch[n_type].y)
                step_bce_loss += bce_loss

                infonce_loss = compute_batched_infonce_loss(
                    z_q, z_n, batch[n_type].y, batch[n_type].batch,
                    temperature=temperature, num_hard_negatives=num_hard_negatives
                )
                step_infonce_loss += infonce_loss

                per_type_bce[n_type] = float(bce_loss.detach().item())
                per_type_infonce[n_type] = float(infonce_loss.detach().item()) if torch.is_tensor(infonce_loss) else float(infonce_loss)
                per_type_nodes[n_type] = int(batch[n_type].num_nodes)
                per_type_pos[n_type] = int(batch[n_type].y.sum().item())

            total_loss = step_bce_loss + (infonce_lambda * step_infonce_loss)
            total_loss.backward()

            # Gradient diagnostics BEFORE optimizer.step()
            gat_grad_norm = _grad_norm(gat_model.parameters())
            proj_grad_norm = _grad_norm(projector.parameters())
            total_grad_norm = (gat_grad_norm ** 2 + proj_grad_norm ** 2) ** 0.5

            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_bce_loss += step_bce_loss.item()
            epoch_infonce_loss += step_infonce_loss.item()
            step_time = time.perf_counter() - t_step
            lr = optimizer.param_groups[0]['lr']
            mem_alloc_gb = float(torch.cuda.memory_allocated(device) / (1024 ** 3)) if torch.cuda.is_available() else 0.0
            mem_peak_gb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if torch.cuda.is_available() else 0.0

            layer_summary = _summarize_layer_stats(gat_model.last_layer_stats) if gat_model.enable_stats else {}

            if step % step_log_every == 0:
                payload = {
                    "train/loss_total": float(total_loss.item()),
                    "train/loss_bce": float(step_bce_loss.item()),
                    "train/loss_infonce": float(step_infonce_loss.item()),
                    "train/grad_norm_total": total_grad_norm,
                    "train/grad_norm_gat": gat_grad_norm,
                    "train/grad_norm_projector": proj_grad_norm,
                    "train/lr": lr,
                    "train/step_time_s": step_time,
                    "train/mem_alloc_gb": mem_alloc_gb,
                    "train/mem_peak_gb": mem_peak_gb,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                }
                for nt, v in per_type_bce.items():
                    payload[f"train/loss_bce/{nt}"] = v
                for nt, v in per_type_infonce.items():
                    payload[f"train/loss_infonce/{nt}"] = v
                for nt, v in per_type_nodes.items():
                    payload[f"train/num_nodes/{nt}"] = v
                for nt, v in per_type_pos.items():
                    payload[f"train/num_pos/{nt}"] = v
                payload.update(layer_summary)
                wandb.log(payload)

                if step_log_path is not None:
                    with open(step_log_path, "a") as fp:
                        row = {"epoch": epoch + 1, "step": step, "global_step": global_step, **payload}
                        fp.write(json.dumps(row) + "\n")

            pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "infoNCE": f"{step_infonce_loss.item():.4f}", "gnorm": f"{total_grad_norm:.2f}"})
            global_step += 1

        # 검증 (Recall@15)
        val_recall = validate(gat_model, projector, val_loader, device, k=15,
                              query_conditioned=query_conditioned)

        # Per-epoch Dirichlet energy + MAD (oversmoothing/energy, oversmoothing/mad)
        os_log = {}
        os_mean_e = 0.0
        os_mean_m = 0.0
        try:
            from utils.oversmoothing_metrics import compute_metrics_v1_gat, metrics_to_wandb_log
            gat_model.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader)).to(device)
                q_emb_s = sample_batch.get('query', None) if hasattr(sample_batch, 'get') else None
                if q_emb_s is None:
                    try:
                        q_emb_s = sample_batch['query']
                    except Exception:
                        q_emb_s = None
                aug_x = sample_batch.x_dict
                if query_conditioned and q_emb_s is not None:
                    if q_emb_s.dim() == 3:
                        q_pool = q_emb_s.mean(dim=1)
                    elif q_emb_s.dim() == 2:
                        q_pool = q_emb_s
                    else:
                        q_pool = q_emb_s.unsqueeze(0)
                    aug_x = {}
                    for nt, x in sample_batch.x_dict.items():
                        nb = sample_batch[nt].batch
                        q_per_node = q_pool[nb]
                        aug_x[nt] = torch.cat([x, q_per_node], dim=-1)
                os_metrics = compute_metrics_v1_gat(
                    gat_model, aug_x, sample_batch.edge_index_dict,
                    query_emb=None, device=str(device))
                os_log = metrics_to_wandb_log(os_metrics)
                os_mean_e = float(os_metrics['energy'].get('mean', 0.0) or 0.0)
                os_mean_m = float(os_metrics['mad'].get('mean', 0.0) or 0.0)
        except Exception as _e:
            logger.warning(f"Oversmoothing metric capture failed: {_e}")

        wandb.log({
            "train/epoch_loss": epoch_loss / len(train_loader),
            "val/recall_at_15": val_recall,
            "epoch": epoch + 1,
            **os_log,
        })

        logger.info(
            f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} "
            f"| BCE: {epoch_bce_loss/len(train_loader):.4f} "
            f"| InfoNCE: {epoch_infonce_loss/len(train_loader):.4f} "
            f"| Val Recall@15: {val_recall:.4f} "
            f"| oversmoothing/energy: {os_mean_e:.4f} "
            f"| oversmoothing/mad: {os_mean_m:.4f}"
        )

        if val_recall > best_recall:
            best_recall = val_recall
            ckpt_name = cfg.get('checkpoint_name', 'best_gat_model.pt')
            save_path = os.path.join(PATHS["checkpoint_dir"], ckpt_name)
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