"""
GAT Bottleneck Analysis — 3축 진단 (Step-wise, CPU-only 가능)

Step 1-a: 목적 함수 불일치 (BCE Loss vs Val Recall@15 trajectory, 기존 로그 파싱)
Step 2  : Over-smoothing — intra-table cosine similarity 레이어별(L0~L3) + t-SNE
Step 3  : Attention weights (per-edge-type entropy) + hop-wise gradient flow

사용:
  # Step 1-a
  python src/analysis/gat_bottleneck_analysis.py step1 \
    --logs logs/train/foo.log logs/train/bar.log \
    --output_dir outputs/analysis/gat_bottleneck/

  # Step 2
  python src/analysis/gat_bottleneck_analysis.py step2 \
    --config configs/training/train_gat_query_supernode.yaml \
    --checkpoint outputs/checkpoints/best_gat_query_supernode.pt \
    --output_dir outputs/analysis/gat_bottleneck/

  # Step 3
  python src/analysis/gat_bottleneck_analysis.py step3 \
    --config configs/training/train_gat_query_supernode.yaml \
    --checkpoint outputs/checkpoints/best_gat_query_supernode.pt \
    --output_dir outputs/analysis/gat_bottleneck/
"""
import os
import re
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

from data.bird_dataset import BIRDGraphDataset, BIRDSuperNodeDataset
from modules.builders.graph_builder import HeteroGraphBuilder, EnrichedHeteroGraphBuilder
from modules.encoders.local_encoder import LocalPLMEncoder
from models.gat_network import SchemaHeteroGAT
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# Step 1-a: BCE vs Recall@15 trajectory (기존 로그 파싱)
# ============================================================

EPOCH_PATTERN = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s*\|\s*"
    r"Loss:\s*(?P<loss>[\d.]+)\s*\|\s*"
    r"(?:BCE:\s*(?P<bce>[\d.]+)\s*\|\s*)?"
    r"(?:InfoNCE:\s*(?P<infonce>[\d.]+)\s*\|\s*)?"
    r"Val Recall@15:\s*(?P<recall>[\d.]+)"
)


def parse_training_log(log_path: Path) -> dict:
    records = {"epoch": [], "loss": [], "bce": [], "infonce": [], "recall": []}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = EPOCH_PATTERN.search(line)
            if not m:
                continue
            records["epoch"].append(int(m.group("epoch")))
            records["loss"].append(float(m.group("loss")))
            records["bce"].append(float(m.group("bce")) if m.group("bce") else np.nan)
            records["infonce"].append(float(m.group("infonce")) if m.group("infonce") else np.nan)
            records["recall"].append(float(m.group("recall")))
    return {k: np.array(v) for k, v in records.items()}


def detect_divergence(bce: np.ndarray, recall: np.ndarray, window: int = 10) -> Optional[int]:
    if len(bce) < 2 * window:
        return None

    def slope(arr, i, w):
        lo, hi = max(0, i - w), min(len(arr), i + w)
        xs = np.arange(lo, hi)
        ys = arr[lo:hi]
        mask = ~np.isnan(ys)
        if mask.sum() < 2:
            return np.nan
        return np.polyfit(xs[mask], ys[mask], 1)[0]

    if np.isnan(bce).all():
        return None
    for i in range(window, len(bce) - window):
        sb = slope(bce, i, window)
        sr = slope(recall, i, window)
        if np.isnan(sb) or np.isnan(sr):
            continue
        if sb < -1e-4 and sr <= 1e-4:
            return int(i)
    return None


def plot_objective_mismatch(records: dict, model_name: str, output_dir: Path, divergence_epoch: Optional[int]) -> Path:
    epochs = records["epoch"]
    bce = records["bce"]
    recall = records["recall"]
    has_bce = not np.isnan(bce).all()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    y_primary = bce if has_bce else records["loss"]
    ax1.plot(epochs, y_primary, color="tab:red", marker="o", markersize=3,
             label="BCE Loss" if has_bce else "Total Loss")
    ax1.set_ylabel("BCE Loss" if has_bce else "Total Loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_xlabel("Epoch")

    ax2 = ax1.twinx()
    ax2.plot(epochs, recall, color="tab:blue", marker="s", markersize=3, label="Val Recall@15")
    ax2.set_ylabel("Val Recall@15", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    if divergence_epoch is not None:
        ax1.axvline(divergence_epoch, color="gray", linestyle="--", alpha=0.6)
        ax1.text(divergence_epoch, ax1.get_ylim()[1] * 0.95,
                 f" divergence @ ep{divergence_epoch}", color="gray", fontsize=9)

    plt.title(f"Objective Mismatch — {model_name}")
    fig.tight_layout()
    plot_path = output_dir / f"objective_mismatch_{model_name}.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def plot_multi_model_overlay(all_records: dict, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, rec in all_records.items():
        ax.plot(rec["epoch"], rec["recall"], marker="o", markersize=2, label=name, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Recall@15")
    ax.set_title("Val Recall@15 Trajectory Across Models")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "recall_trajectory_overlay.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def summarize_step1(records: dict, divergence_epoch: Optional[int]) -> dict:
    has_bce = not np.isnan(records["bce"]).all()
    primary = records["bce"] if has_bce else records["loss"]
    recall = records["recall"]
    best_idx = int(np.argmax(recall))
    return {
        "num_epochs": int(len(records["epoch"])),
        "final_bce_or_loss": float(primary[-1]) if len(primary) else float("nan"),
        "min_bce_or_loss": float(np.nanmin(primary)),
        "min_bce_epoch": int(np.nanargmin(primary)) + 1,
        "best_recall": float(recall[best_idx]),
        "best_recall_epoch": int(records["epoch"][best_idx]),
        "final_recall": float(recall[-1]),
        "divergence_epoch": divergence_epoch,
        "has_bce_in_log": has_bce,
    }


def run_step1a(log_paths: List[Path], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = {}
    logger.info("=" * 60)
    logger.info("Step 1-a: Objective Mismatch (BCE vs Recall@15 trajectory)")
    logger.info("=" * 60)

    for log_path in log_paths:
        if not log_path.exists():
            logger.warning(f"Log not found: {log_path}")
            continue
        model_name = log_path.stem
        rec = parse_training_log(log_path)
        if len(rec["epoch"]) == 0:
            logger.warning(f"No matching epoch lines in {log_path}")
            continue
        primary = rec["bce"] if not np.isnan(rec["bce"]).all() else rec["loss"]
        divergence = detect_divergence(primary, rec["recall"])
        plot_path = plot_objective_mismatch(rec, model_name, output_dir, divergence)
        summary = summarize_step1(rec, divergence)
        all_records[model_name] = rec

        logger.info(f"[{model_name}]")
        for k, v in summary.items():
            logger.info(f"  {k}: {v}")
        logger.info(f"  plot → {plot_path}")

    if len(all_records) >= 2:
        overlay_path = plot_multi_model_overlay(all_records, output_dir)
        logger.info(f"Overlay plot → {overlay_path}")

    logger.info("Step 1-a complete.")


# ============================================================
# Step 2: Over-smoothing (intra-table cosine similarity)
# ============================================================

COL_TO_TAB_EDGE = ("column", "belongs_to", "table")


def _build_dataset(cfg: dict) -> BIRDGraphDataset:
    """Dev set용 BIRDGraphDataset 생성. query_supernode=True면 SuperNode 래핑."""
    builder_type = cfg.get("builder", {}).get("type", "HeteroGraphBuilder")
    if builder_type == "EnrichedHeteroGraphBuilder":
        tables_json = cfg["builder"].get("tables_json_path",
                                          "data/raw/BIRD_dev/dev_tables.json")
        # Dev set용 tables_json 사용 (train set 경로는 학습용)
        dev_tables = "data/raw/BIRD_dev/dev_tables.json"
        builder = EnrichedHeteroGraphBuilder(tables_json_path=dev_tables)
        logger.info(f"Builder: EnrichedHeteroGraphBuilder (dev_tables={dev_tables})")
    else:
        builder = HeteroGraphBuilder()
        logger.info("Builder: HeteroGraphBuilder")

    encoder = LocalPLMEncoder()
    dev_json = cfg["paths"].get("test_json", "data/raw/BIRD_dev/dev.json")
    dev_db_dir = cfg["paths"].get("test_db_dir", "data/raw/BIRD_dev/dev_databases")

    dataset = BIRDGraphDataset(
        json_path=dev_json,
        db_dir=dev_db_dir,
        builder=builder,
        encoder=encoder,
    )
    if cfg["model"].get("query_supernode", False):
        logger.info("Wrapping with BIRDSuperNodeDataset (query_supernode=True)")
        dataset = BIRDSuperNodeDataset(dataset)
    return dataset


def _build_model(cfg: dict, checkpoint_path: Path, device: torch.device) -> SchemaHeteroGAT:
    model = SchemaHeteroGAT(
        in_channels=cfg["model"]["in_channels"],
        hidden_channels=cfg["model"]["hidden_channels"],
        out_channels=cfg["model"]["out_channels"],
        num_layers=cfg["model"]["num_layers"],
        heads=cfg["model"]["heads"],
        query_conditioned=cfg["model"].get("query_conditioned", False),
        query_supernode=cfg["model"].get("query_supernode", False),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("gat_state_dict", "model_state_dict", "state_dict"):
            if key in ckpt:
                state = ckpt[key]
                break
        else:
            state = ckpt
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    return model


def extract_layerwise_embeddings(model: SchemaHeteroGAT, batch, query_emb: Optional[torch.Tensor]):
    """
    SchemaHeteroGAT.forward를 재현하되 레이어별 hidden을 반환.
    Returns: list of dicts [L0_raw, L1, L2, L3, L_out], each {node_type: Tensor [N, D]}
    """
    x_dict = {nt: x for nt, x in batch.x_dict.items()}
    edge_index_dict = batch.edge_index_dict

    embeddings = [{nt: x.detach().clone() for nt, x in x_dict.items()}]  # L0: raw PLM

    if model.query_conditioned and query_emb is not None:
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        augmented = {}
        for nt, x in x_dict.items():
            q_exp = query_emb.expand(x.size(0), -1)
            augmented[nt] = torch.cat([x, q_exp], dim=-1)
        x_dict = augmented

    out_dict = {nt: F.leaky_relu(model.lin_dict[nt](x)) for nt, x in x_dict.items()}

    for i in range(model.num_layers):
        out_dict = model.convs[i](out_dict, edge_index_dict)
        out_dict = {nt: F.elu(x) for nt, x in out_dict.items()}
        embeddings.append({nt: x.detach().clone() for nt, x in out_dict.items()})

    final = {}
    for nt, x in out_dict.items():
        final[nt] = model.out_lin_dict[nt](x) + model.skip_dict[nt](x_dict[nt])
    embeddings.append({nt: x.detach().clone() for nt, x in final.items()})

    return embeddings


def intra_table_sims(col_embs: torch.Tensor, cb_edge_index: torch.Tensor) -> List[float]:
    """
    col_embs: [N_cols, D] — 해당 그래프 column 노드 임베딩
    cb_edge_index: ('column', 'belongs_to', 'table') edge_index [2, E]
    반환: 각 테이블(컬럼 ≥ 2개)의 pairwise cosine sim 평균 리스트
    """
    if cb_edge_index.numel() == 0:
        return []
    col_ids = cb_edge_index[0].tolist()
    tab_ids = cb_edge_index[1].tolist()

    table_to_cols = defaultdict(set)
    for c, t in zip(col_ids, tab_ids):
        table_to_cols[t].add(c)

    sims = []
    normed = F.normalize(col_embs, dim=-1)
    for tab_id, cols in table_to_cols.items():
        cols = sorted(cols)
        if len(cols) < 2:
            continue
        embs = normed[cols]
        sim_matrix = embs @ embs.T
        k = len(cols)
        iu = torch.triu_indices(k, k, offset=1)
        sims.append(sim_matrix[iu[0], iu[1]].mean().item())
    return sims


def plot_oversmoothing_trajectory(all_sims_per_layer: List[List[float]], layer_names: List[str], output_dir: Path) -> Path:
    """
    all_sims_per_layer[l] = 모든 query의 모든 테이블의 intra-table sim (flat list) at layer l
    """
    means = [np.mean(s) if s else float("nan") for s in all_sims_per_layer]
    stds = [np.std(s) if s else 0.0 for s in all_sims_per_layer]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.errorbar(range(len(layer_names)), means, yerr=stds, marker="o", capsize=4, color="tab:purple")
    ax1.set_xticks(range(len(layer_names)))
    ax1.set_xticklabels(layer_names)
    ax1.set_ylabel("Intra-Table Cosine Similarity")
    ax1.set_title("Over-smoothing Trajectory (mean ± std)")
    ax1.axhline(0.85, color="red", linestyle="--", alpha=0.5, label="critical (0.85)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.boxplot(all_sims_per_layer, labels=layer_names, showfliers=False)
    ax2.set_ylabel("Intra-Table Cosine Similarity")
    ax2.set_title("Distribution per Layer")
    ax2.axhline(0.85, color="red", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "oversmoothing_trajectory.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_tsne_per_layer(sample_embeddings: List[torch.Tensor], col_to_table: Dict[int, int],
                        layer_names: List[str], output_dir: Path, query_idx: int) -> Path:
    """샘플 그래프의 column embeddings을 레이어별로 t-SNE, color=table."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("sklearn not available, skipping t-SNE")
        return None

    n_layers = len(sample_embeddings)
    labels = np.array([col_to_table[c] for c in sorted(col_to_table.keys())])
    col_order = sorted(col_to_table.keys())

    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for i, (embs, name, ax) in enumerate(zip(sample_embeddings, layer_names, axes)):
        embs_np = embs[col_order].cpu().numpy()
        if embs_np.shape[0] < 4:
            ax.set_title(f"{name} (n<4)")
            continue
        perplexity = min(30, max(2, embs_np.shape[0] // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
        coords = tsne.fit_transform(embs_np)
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=15, alpha=0.8)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Column Embedding t-SNE per Layer (query_idx={query_idx}, colored by table)")
    fig.tight_layout()
    path = output_dir / f"tsne_query{query_idx}.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_step2(config_path: Path, checkpoint_path: Path, output_dir: Path,
              max_queries: Optional[int], tsne_sample_idx: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    logger.info("=" * 60)
    logger.info("Step 2: Over-smoothing (intra-table cosine similarity)")
    logger.info("=" * 60)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = _build_dataset(cfg)
    model = _build_model(cfg, checkpoint_path, device)

    n_queries = len(dataset) if max_queries is None else min(max_queries, len(dataset))
    logger.info(f"Analyzing {n_queries}/{len(dataset)} queries on CPU...")

    layer_names = ["L0_PLM", "L1_GAT", "L2_GAT", "L3_GAT", "L_out"]
    all_sims_per_layer = [[] for _ in layer_names]

    sample_embeddings = None
    sample_col_to_table = None

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= n_queries:
                break
            batch = batch.to(device)
            q_emb = batch["query"] if "query" in batch else None

            layer_embs = extract_layerwise_embeddings(model, batch, q_emb)

            if COL_TO_TAB_EDGE in batch.edge_index_dict:
                cb_edge = batch.edge_index_dict[COL_TO_TAB_EDGE]
            else:
                continue

            for l, emb_dict in enumerate(layer_embs):
                if "column" not in emb_dict:
                    continue
                sims = intra_table_sims(emb_dict["column"], cb_edge)
                all_sims_per_layer[l].extend(sims)

            if idx == tsne_sample_idx:
                sample_embeddings = [d["column"] for d in layer_embs if "column" in d]
                col_ids = cb_edge[0].tolist()
                tab_ids = cb_edge[1].tolist()
                sample_col_to_table = {c: t for c, t in zip(col_ids, tab_ids)}

            if (idx + 1) % 100 == 0:
                logger.info(f"  processed {idx + 1}/{n_queries}")

    logger.info("Computing summary statistics...")
    for name, sims in zip(layer_names, all_sims_per_layer):
        if sims:
            logger.info(f"  {name}: n={len(sims)}, mean={np.mean(sims):.4f}, std={np.std(sims):.4f}, "
                        f"median={np.median(sims):.4f}")
        else:
            logger.info(f"  {name}: no data")

    traj_path = plot_oversmoothing_trajectory(all_sims_per_layer, layer_names, output_dir)
    logger.info(f"Trajectory plot → {traj_path}")

    if sample_embeddings is not None and sample_col_to_table is not None:
        tsne_path = plot_tsne_per_layer(sample_embeddings, sample_col_to_table,
                                        layer_names, output_dir, tsne_sample_idx)
        if tsne_path:
            logger.info(f"t-SNE plot → {tsne_path}")

    l0_mean = np.mean(all_sims_per_layer[0]) if all_sims_per_layer[0] else float("nan")
    l3_mean = np.mean(all_sims_per_layer[3]) if all_sims_per_layer[3] else float("nan")
    if not np.isnan(l0_mean) and not np.isnan(l3_mean):
        delta = l3_mean - l0_mean
        if l3_mean > 0.85 and delta > 0.2:
            logger.info(f"⚠️  Over-smoothing detected: L0={l0_mean:.4f} → L3={l3_mean:.4f} (Δ={delta:+.4f})")
        else:
            logger.info(f"✅ Over-smoothing not severe: L0={l0_mean:.4f} → L3={l3_mean:.4f} (Δ={delta:+.4f})")

    logger.info("Step 2 complete.")


# ============================================================
# Step 3: Attention entropy (per-edge-type, per-layer) + gradient flow
# ============================================================

def extract_layerwise_attention(model: SchemaHeteroGAT, batch, query_emb) -> List[Dict[tuple, float]]:
    """
    Layer별로 각 edge-type의 attention distribution entropy(평균)를 반환.
    HeteroConv 내부 GATv2Conv를 직접 호출해 return_attention_weights=True로 α를 획득.
    Returns: [{edge_type: mean_entropy}, ...]  (length = num_layers)
    """
    x_dict = {nt: x for nt, x in batch.x_dict.items()}
    edge_index_dict = batch.edge_index_dict

    if model.query_conditioned and query_emb is not None:
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        x_dict = {nt: torch.cat([x, query_emb.expand(x.size(0), -1)], dim=-1) for nt, x in x_dict.items()}

    out_dict = {nt: F.leaky_relu(model.lin_dict[nt](x)) for nt, x in x_dict.items()}

    per_layer_entropy: List[Dict[tuple, float]] = []
    for i in range(model.num_layers):
        hetero_conv = model.convs[i]
        inner_convs = hetero_conv.convs  # ModuleDict keyed by stringified edge_type or tuple
        layer_entropy: Dict[tuple, float] = {}

        next_parts = defaultdict(list)
        for edge_type in edge_index_dict.keys():
            src, _, dst = edge_type
            if src not in out_dict or dst not in out_dict:
                continue
            if edge_type in inner_convs:
                conv = inner_convs[edge_type]
            else:
                continue

            ei = edge_index_dict[edge_type]
            if ei.numel() == 0:
                continue

            out, (att_ei, alpha) = conv(
                (out_dict[src], out_dict[dst]), ei, return_attention_weights=True
            )
            next_parts[dst].append(out)

            # alpha: [num_edges, heads] — per dst-node 정규화 entropy 계산
            alpha_mean = alpha.mean(dim=-1) if alpha.dim() > 1 else alpha  # [E]
            dst_idx = att_ei[1]
            ent_sum = torch.zeros(out_dict[dst].size(0))
            for e in range(alpha_mean.size(0)):
                p = alpha_mean[e].item()
                if p > 1e-12:
                    ent_sum[dst_idx[e]] -= p * np.log(p + 1e-12)
            # 유효 노드(유입 엣지 존재)만 평균
            has_in = torch.zeros(out_dict[dst].size(0), dtype=torch.bool)
            has_in[dst_idx] = True
            if has_in.any():
                layer_entropy[edge_type] = ent_sum[has_in].mean().item()

        # aggregate mean (HeteroConv 기본 aggr='mean') 후 ELU
        merged: Dict[str, torch.Tensor] = {}
        for dst, parts in next_parts.items():
            merged[dst] = torch.stack(parts, dim=0).mean(dim=0)
        for nt in out_dict.keys():
            if nt in merged:
                out_dict[nt] = F.elu(merged[nt])

        per_layer_entropy.append(layer_entropy)

    return per_layer_entropy


def compute_gradient_flow(model: SchemaHeteroGAT, batch, query_emb) -> Dict[str, float]:
    """
    Query-similarity BCE를 proxy loss로 backward → 파라미터 그룹별 gradient norm.
    Returns: {"lin_dict": ..., "conv_L1": ..., ..., "out_lin_dict": ..., "skip_dict": ...}
    """
    model.train()
    model.zero_grad(set_to_none=True)

    out_dict = model(batch.x_dict, batch.edge_index_dict, query_emb)

    if query_emb is None:
        return {}
    q = query_emb if query_emb.dim() == 2 else query_emb.unsqueeze(0)

    total_loss = torch.tensor(0.0)
    n_terms = 0
    for nt in ("table", "column"):
        if nt not in out_dict:
            continue
        node_emb = out_dict[nt]
        if node_emb.size(0) == 0:
            continue
        if not hasattr(batch[nt], "y") or batch[nt].y is None:
            continue
        y = batch[nt].y.float()
        if y.size(0) != node_emb.size(0):
            continue
        q_proj = q[:, : node_emb.size(1)] if q.size(1) >= node_emb.size(1) else \
                 F.pad(q, (0, node_emb.size(1) - q.size(1)))
        logits = F.normalize(node_emb, dim=-1) @ F.normalize(q_proj, dim=-1).T
        logits = logits.squeeze(-1) * 10.0
        total_loss = total_loss + F.binary_cross_entropy_with_logits(logits, y)
        n_terms += 1

    if n_terms == 0:
        return {}
    total_loss = total_loss / n_terms
    total_loss.backward()

    groups: Dict[str, float] = {}

    def _group_norm(named_params) -> float:
        sq = 0.0
        for _, p in named_params:
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum().item())
        return sq ** 0.5

    groups["lin_dict"] = _group_norm(model.lin_dict.named_parameters())
    for i in range(model.num_layers):
        groups[f"conv_L{i+1}"] = _group_norm(model.convs[i].named_parameters())
    groups["out_lin_dict"] = _group_norm(model.out_lin_dict.named_parameters())
    groups["skip_dict"] = _group_norm(model.skip_dict.named_parameters())

    model.eval()
    return groups


def plot_attention_entropy(entropy_per_layer: List[Dict[tuple, List[float]]], output_dir: Path) -> Path:
    """entropy_per_layer[l][edge_type] = list of per-query mean entropy."""
    edge_types = sorted({et for layer in entropy_per_layer for et in layer.keys()}, key=str)
    n_layers = len(entropy_per_layer)

    fig, ax = plt.subplots(figsize=(max(8, len(edge_types) * 1.2), 5))
    width = 0.8 / n_layers
    x = np.arange(len(edge_types))
    for i, layer_dict in enumerate(entropy_per_layer):
        vals = [np.mean(layer_dict.get(et, [float("nan")])) for et in edge_types]
        ax.bar(x + i * width, vals, width, label=f"L{i+1}")
    ax.set_xticks(x + width * (n_layers - 1) / 2)
    ax.set_xticklabels(["→".join(et) if isinstance(et, tuple) else str(et) for et in edge_types],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Attention Entropy (per dst node)")
    ax.set_title("Attention Entropy per Edge Type × Layer (lower = sharper)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "attention_entropy.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_gradient_flow(grad_per_query: Dict[str, List[float]], output_dir: Path) -> Path:
    order = ["lin_dict"] + sorted([k for k in grad_per_query if k.startswith("conv_L")]) + \
            ["out_lin_dict", "skip_dict"]
    order = [k for k in order if k in grad_per_query and grad_per_query[k]]
    means = [np.mean(grad_per_query[k]) for k in order]
    stds = [np.std(grad_per_query[k]) for k in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(order))
    ax.bar(x, means, yerr=stds, capsize=4, color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Gradient L2 Norm")
    ax.set_yscale("log")
    ax.set_title("Hop-wise Gradient Flow (mean ± std, log scale)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = output_dir / "gradient_flow.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_step3(config_path: Path, checkpoint_path: Path, output_dir: Path,
              max_queries: Optional[int]):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    logger.info("=" * 60)
    logger.info("Step 3: Attention entropy + gradient flow")
    logger.info("=" * 60)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = _build_dataset(cfg)
    model = _build_model(cfg, checkpoint_path, device)

    n_queries = len(dataset) if max_queries is None else min(max_queries, len(dataset))
    logger.info(f"Analyzing {n_queries}/{len(dataset)} queries on CPU...")

    num_layers = model.num_layers
    entropy_accum: List[Dict[tuple, List[float]]] = [defaultdict(list) for _ in range(num_layers)]
    grad_accum: Dict[str, List[float]] = defaultdict(list)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for idx, batch in enumerate(loader):
        if idx >= n_queries:
            break
        batch = batch.to(device)
        q_emb = batch["query"] if "query" in batch else None

        try:
            with torch.no_grad():
                per_layer = extract_layerwise_attention(model, batch, q_emb)
            for l, layer_dict in enumerate(per_layer):
                for et, ent in layer_dict.items():
                    entropy_accum[l][et].append(ent)
        except Exception as e:
            logger.warning(f"  [att] query {idx} skipped: {e}")

        try:
            grads = compute_gradient_flow(model, batch, q_emb)
            for k, v in grads.items():
                grad_accum[k].append(v)
        except Exception as e:
            logger.warning(f"  [grad] query {idx} skipped: {e}")

        if (idx + 1) % 50 == 0:
            logger.info(f"  processed {idx + 1}/{n_queries}")

    ent_path = plot_attention_entropy(entropy_accum, output_dir)
    logger.info(f"Attention entropy plot → {ent_path}")

    if grad_accum:
        grad_path = plot_gradient_flow(dict(grad_accum), output_dir)
        logger.info(f"Gradient flow plot → {grad_path}")

        conv_keys = sorted([k for k in grad_accum if k.startswith("conv_L")])
        if len(conv_keys) >= 2:
            first = np.mean(grad_accum[conv_keys[0]])
            last = np.mean(grad_accum[conv_keys[-1]])
            ratio = last / first if first > 0 else float("nan")
            logger.info(f"  grad ratio (last/first conv): {ratio:.4f} "
                        f"({conv_keys[0]}={first:.3e}, {conv_keys[-1]}={last:.3e})")
            if ratio < 0.1:
                logger.info("⚠️  Gradient vanishing in deeper layers")
            elif ratio > 10:
                logger.info("⚠️  Gradient explosion in deeper layers")
            else:
                logger.info("✅ Gradient flow reasonably balanced")

    logger.info("Step 3 complete.")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GAT Bottleneck Analysis")
    subparsers = parser.add_subparsers(dest="step", required=True)

    p1 = subparsers.add_parser("step1", help="Objective mismatch from training logs")
    p1.add_argument("--logs", type=str, nargs="+", required=True)
    p1.add_argument("--output_dir", type=str, default="outputs/analysis/gat_bottleneck/")

    p2 = subparsers.add_parser("step2", help="Over-smoothing via intra-table cosine sim")
    p2.add_argument("--config", type=str, required=True, help="Training config YAML")
    p2.add_argument("--checkpoint", type=str, required=True)
    p2.add_argument("--output_dir", type=str, default="outputs/analysis/gat_bottleneck/")
    p2.add_argument("--max_queries", type=int, default=None,
                    help="Limit number of queries (default: all)")
    p2.add_argument("--tsne_sample_idx", type=int, default=0,
                    help="Query index for t-SNE visualization")

    p3 = subparsers.add_parser("step3", help="Attention entropy + gradient flow")
    p3.add_argument("--config", type=str, required=True)
    p3.add_argument("--checkpoint", type=str, required=True)
    p3.add_argument("--output_dir", type=str, default="outputs/analysis/gat_bottleneck/")
    p3.add_argument("--max_queries", type=int, default=None)

    args = parser.parse_args()

    if args.step == "step1":
        run_step1a([Path(p) for p in args.logs], Path(args.output_dir))
    elif args.step == "step2":
        run_step2(Path(args.config), Path(args.checkpoint), Path(args.output_dir),
                  args.max_queries, args.tsne_sample_idx)
    elif args.step == "step3":
        run_step3(Path(args.config), Path(args.checkpoint), Path(args.output_dir),
                  args.max_queries)


if __name__ == "__main__":
    main()
