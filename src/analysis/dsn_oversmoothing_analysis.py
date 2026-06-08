"""
DSN Over-smoothing Analysis — V-3-ext 3 신규 ckpt + qcond_nl3 baseline 진단.

근거: planning/DECISIONS.md 2026-05-06 (DSN Phase 1 시나리오 A 확정 + saturation 결정적)
의도: V-3-ext 3 ckpt 의 over-smoothing 여부 + GAT 학습이 raw R 한계 회복 못 하는 mechanism 진단

진단 가설:
  H1 (over-smoothing): GAT layer 진행에 따라 schema 노드 임베딩이 indistinguishable 해짐 → score 변별력 손실
  H2 (구조적 한계): directed_from_sn edge + 3-layer 가 schema 정보 전달 부족 (information bottleneck)
  H3 (학습 saturation): 학습이 raw selector signal 능가 못 함 (mechanism 한계, over-smoothing 반증)

H1 ↔ H3 구분: layer-wise intra-table cosine sim trajectory.
  - L_out cosine → 1.0 인 경우 H1 지지
  - L_out cosine 안정 mid-range (0.3~0.5) 인 경우 H3 지지

3-step protocol (s06_bottleneck_comparison 와 동일 형식):
  Step 1: training log parse → epoch / loss / val recall trajectory
  Step 2: 4 ckpt × ~50 queries forward pass → intra-table cosine sim per layer
  Step 3: attention entropy + gradient flow per layer

산출물:
  outputs/analysis/dsn_oversmoothing/{p80, topk20, abstau07, qcond_nl3}/{plots}
  outputs/analysis/dsn_oversmoothing/batch_summary.json
  notebooks/analysis_results/dsn_oversmoothing_analysis.md
"""

import os
import sys
import json
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# v1 분석 함수 재활용 (SchemaHeteroGAT 호환)
from analysis.gat_bottleneck_analysis import (
    parse_training_log,
    plot_objective_mismatch,
    plot_multi_model_overlay,
    summarize_step1,
    detect_divergence,
    _build_dataset,
    extract_layerwise_embeddings,
    intra_table_sims,
    plot_oversmoothing_trajectory,
    plot_tsne_per_layer,
    extract_layerwise_attention,
    compute_gradient_flow,
    plot_attention_entropy,
    plot_gradient_flow,
    COL_TO_TAB_EDGE,
)

# V-3-ext directed_from_sn 호환 attention extract (forward hook 기반, 2026-05-06).
# Phase 1 단계 4 후속 — DECISIONS 2026-05-06 §1(B).
from analysis.extract_layerwise_attention_v2 import (
    extract_layerwise_attention_v2,
    aggregate_attention_metrics,
    plot_attention_entropy_heatmap,
    plot_topk_concentration_heatmap,
)

from torch_geometric.loader import DataLoader
from models.gat_network import SchemaHeteroGAT
from models.gat_network_v2 import SchemaHeteroGATv2
from utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_query_emb(batch) -> Optional[torch.Tensor]:
    """SuperNode dataset 은 'query_node'.x 로 저장, plain dataset 은 'query' 키.
    둘 다 없으면 None.
    """
    try:
        if "query_node" in batch.node_types and batch["query_node"].x is not None:
            return batch["query_node"].x
    except Exception:
        pass
    if "query" in batch:
        q = batch["query"]
        return q if q is not None else None
    return None


def extract_layerwise_dsn(model: SchemaHeteroGAT, batch,
                          query_emb: Optional[torch.Tensor]
                          ) -> List[Dict[str, torch.Tensor]]:
    """Forward hook 기반 layer-wise embedding 추출 (V-2/V-3-ext 호환).

    model.forward 흐름:
      L0_PLM = batch.x_dict (원본)
      lin_dict 후 input_proj
      convs[0..N-1] 통과 → 각 layer hidden 캡처
      L_out = out_lin_dict + skip_dict
    """
    embeddings: List[Dict[str, torch.Tensor]] = []
    # L0_PLM
    embeddings.append({nt: x.detach().clone() for nt, x in batch.x_dict.items()})

    captured: List[Dict[str, torch.Tensor]] = []

    def _hook(module, inputs, output):
        if isinstance(output, dict):
            captured.append({nt: x.detach().clone() for nt, x in output.items()})

    handles = [model.convs[i].register_forward_hook(_hook) for i in range(model.num_layers)]
    try:
        with torch.no_grad():
            final = model(batch.x_dict, batch.edge_index_dict, query_emb)
    finally:
        for h in handles:
            h.remove()

    # captured 는 ELU 적용 전 raw output (model.convs[i]).
    # forward 안의 ELU 는 후처리 이므로, F.elu 적용해서 일관 유지.
    for layer_out in captured:
        embeddings.append({nt: torch.nn.functional.elu(x).detach().clone()
                           for nt, x in layer_out.items()})

    embeddings.append({nt: x.detach().clone() for nt, x in final.items()})
    return embeddings


_V2_TRIGGER_KEYS = (
    "gat_layer_type", "aggregation_type", "drop_message_p", "use_layernorm_pre_softmax",
    "softplus_symmetric_norm", "pairnorm_mode", "initial_residual_alpha",
    "jumping_knowledge", "dual_stream",
)


def _needs_v2(cfg_model: dict) -> bool:
    """v2-specific (mitigation v2/v3/V4 + B5 mitigation set) 옵션 사용 여부 판정.
    어느 하나라도 default 아닌 값이 설정되어 있으면 SchemaHeteroGATv2 빌더 필요.
    """
    defaults = {
        "gat_layer_type": "standard",
        "aggregation_type": "mean",
        "drop_message_p": 0.0,
        "use_layernorm_pre_softmax": False,
        "softplus_symmetric_norm": True,  # only meaningful when gat_layer_type=='softplus'
        "pairnorm_mode": "none",
        "initial_residual_alpha": 0.0,
        "jumping_knowledge": "none",
        "dual_stream": False,
    }
    for k in _V2_TRIGGER_KEYS:
        if k in cfg_model and cfg_model[k] != defaults[k]:
            return True
    return False


def _build_model_dsn(cfg: dict, checkpoint_path: Path, device: torch.device):
    """V-3-ext 옵션 (supernode_edge_direction, threshold_mode/value, score_normalization, topk*) 를
    forward 하여 SchemaHeteroGAT 빌드. v1 _build_model 이 V-3-ext 옵션 forward 안 해서 분리.

    Mitigation V4 등 v2-specific 옵션 사용 시 SchemaHeteroGATv2 빌더로 분기 (DECISIONS 2026-05-11).
    """
    m = cfg["model"]
    kwargs = dict(
        in_channels=m["in_channels"],
        hidden_channels=m["hidden_channels"],
        out_channels=m["out_channels"],
        num_layers=m["num_layers"],
        heads=m["heads"],
        query_conditioned=m.get("query_conditioned", False),
        query_supernode=m.get("query_supernode", False),
    )
    # V-2 / V-3-ext 옵션 forward (default 가 model 측에서 처리됨)
    for k in ("supernode_edge_direction", "supernode_threshold_mode",
              "supernode_threshold_value", "supernode_topk",
              "supernode_topk_criterion", "supernode_score_normalization"):
        if k in m:
            kwargs[k] = m[k]

    if _needs_v2(m):
        # v2 추가 옵션 forward (B5 mitigation + mitigation v2/v3/V4)
        for k in _V2_TRIGGER_KEYS + ("pairnorm_scale",):
            if k in m:
                kwargs[k] = m[k]
        model = SchemaHeteroGATv2(**kwargs).to(device)
        builder_tag = "v2"
    else:
        model = SchemaHeteroGAT(**kwargs).to(device)
        builder_tag = "v1"

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
    logger.info(f"Loaded ckpt[{builder_tag}]: {checkpoint_path.name} "
                f"(L={m['num_layers']}, qcond={kwargs.get('query_conditioned')}, "
                f"qsn={kwargs.get('query_supernode')}, "
                f"dir={kwargs.get('supernode_edge_direction','bidirectional')}, "
                f"thr_mode={kwargs.get('supernode_threshold_mode','top_k')}, "
                f"gat_layer={m.get('gat_layer_type','standard')})")
    return model


# ──────────────────────────────────────────────────────────────
# 4 ckpt 정의
# ──────────────────────────────────────────────────────────────

CKPTS = [
    {
        "name": "p80",
        "label": "DSN p80 (P80)",
        "config": ROOT / "configs/training/train_gat_directed_supernode_p80.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80.pt",
        "log": ROOT / "logs/train/gat_directed_supernode_p80_20260506_001843.log",
        "best_val_recall": 0.6097,
        "raw_R": 0.6133,
    },
    {
        "name": "topk20",
        "label": "DSN topk20",
        "config": ROOT / "configs/training/train_gat_directed_supernode_topk20.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_topk20.pt",
        "log": ROOT / "logs/train/gat_directed_supernode_topk20_20260506_001843.log",
        "best_val_recall": 0.5839,
        "raw_R": 0.6865,
    },
    {
        "name": "abstau07",
        "label": "DSN abstau07 (τ=0.7)",
        "config": ROOT / "configs/training/train_gat_directed_supernode_abstau07.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_abstau07.pt",
        "log": ROOT / "logs/train/gat_directed_supernode_abstau07_20260506_025250.log",
        "best_val_recall": 0.5805,
        "raw_R": 0.4857,
    },
    {
        "name": "qcond_nl3",
        "label": "qcond_nl3 baseline",
        "config": ROOT / "configs/training/diameter_layers/train_qcond_nl3.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_qcond_nl3.pt",
        "log": ROOT / "logs/train/gat_query_conditioned_nl3_20260422_221736.log",
        "best_val_recall": None,  # parse 결과로 채워질 예정
        "raw_R": None,
    },
    # Mitigation V4 — Architectural Intervention (DECISIONS 2026-05-11).
    # 학습 종료 후 자동으로 본 entry 들이 측정됨 (scripts/run_v4_mitigation_sweep.sh stage 3).
    {
        "name": "v4a_lngin_combo",
        "label": "V4-A LN+GIN Combo",
        "config": ROOT / "configs/training/dsn/train_dsn_p80_v4a_lngin_combo.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_v4a_lngin_combo.pt",
        "log": None,  # 학습 후 logger glob 으로 resolve
        "best_val_recall": None,
        "raw_R": None,
    },
    {
        "name": "v4b_aero",
        "label": "V4-B AERO (Softplus+SymNorm)",
        "config": ROOT / "configs/training/dsn/train_dsn_p80_v4b_aero.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_v4b_aero.pt",
        "log": None,
        "best_val_recall": None,
        "raw_R": None,
    },
    # Mitigation V5 — Architectural Intervention 2nd wave (DECISIONS 2026-05-12 V5 plan).
    # ckpt: best_gat_directed_supernode_p80_v5*_with_proj.pt (selector+projector joint train).
    {
        "name": "v5a_gate",
        "label": "V5-A GATE (Conservation Law)",
        "config": ROOT / "configs/training/dsn/train_dsn_p80_v5a_gate.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_v5a_gate_with_proj.pt",
        "log": None,
        "best_val_recall": 0.5571,
        "raw_R": None,
    },
    {
        "name": "v5b_gcnii_L2",
        "label": "V5-B GCNII L=2 (Initial Residual+Identity)",
        "config": ROOT / "configs/training/dsn/train_dsn_p80_v5b_gcnii_L2.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_v5b_gcnii_L2_with_proj.pt",
        "log": None,
        "best_val_recall": 0.6072,
        "raw_R": None,
    },
    {
        "name": "v5b_gcnii_L4",
        "label": "V5-B GCNII L=4",
        "config": ROOT / "configs/training/dsn/train_dsn_p80_v5b_gcnii_L4.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_v5b_gcnii_L4_with_proj.pt",
        "log": None,
        "best_val_recall": 0.5969,
        "raw_R": None,
    },
    {
        "name": "v5b_gcnii_L6",
        "label": "V5-B GCNII L=6",
        "config": ROOT / "configs/training/dsn/train_dsn_p80_v5b_gcnii_L6.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_v5b_gcnii_L6_with_proj.pt",
        "log": None,
        "best_val_recall": 0.5845,
        "raw_R": None,
    },
    {
        "name": "v5c_cum_only",
        "label": "V5-C AERO Cumulative-only",
        "config": ROOT / "configs/training/dsn/train_dsn_p80_v5c_cum_only.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_v5c_cum_only_with_proj.pt",
        "log": None,
        "best_val_recall": 0.5993,
        "raw_R": None,
    },
    {
        "name": "v5c_full",
        "label": "V5-C AERO Full (Cum+Hop+Softplus+SymNorm)",
        "config": ROOT / "configs/training/dsn/train_dsn_p80_v5c_full.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_v5c_full_with_proj.pt",
        "log": None,
        "best_val_recall": 0.5887,
        "raw_R": None,
    },
    {
        "name": "v5c_hop_only",
        "label": "V5-C AERO Hop-attention-only",
        "config": ROOT / "configs/training/dsn/train_dsn_p80_v5c_hop_only.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_v5c_hop_only_with_proj.pt",
        "log": None,
        "best_val_recall": 0.6076,
        "raw_R": None,
    },
]

OUT_DIR = ROOT / "outputs/analysis/dsn_oversmoothing"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _layer_names(num_layers: int) -> List[str]:
    names = ["L0_PLM"]
    for i in range(num_layers):
        names.append(f"L{i+1}_GAT")
    names.append("L_out")
    return names


# ──────────────────────────────────────────────────────────────
# Step 1
# ──────────────────────────────────────────────────────────────

def run_step1() -> Dict[str, dict]:
    logger.info("=" * 60)
    logger.info("Step 1: Training log parse + trajectory plots")
    logger.info("=" * 60)

    all_records = {}
    summaries = {}
    for c in CKPTS:
        log_path = c["log"]
        if not log_path.exists():
            logger.warning(f"  [{c['name']}] log not found: {log_path}")
            continue
        rec = parse_training_log(log_path)
        if len(rec["epoch"]) == 0:
            logger.warning(f"  [{c['name']}] no epoch lines in log")
            continue
        div = detect_divergence(rec.get("bce", np.array([])), rec["recall"])
        plot_objective_mismatch(rec, c["name"], OUT_DIR, div)
        s = summarize_step1(rec, div)
        all_records[c["name"]] = rec
        summaries[c["name"]] = s
        logger.info(f"  [{c['name']}] best Recall@15={s['best_recall']:.4f} @ep{s['best_recall_epoch']}, "
                    f"epochs={s['num_epochs']}, final_bce/loss={s['final_bce_or_loss']:.4f}")

    if len(all_records) >= 2:
        overlay_path = plot_multi_model_overlay(all_records, OUT_DIR)
        logger.info(f"  overlay → {overlay_path}")

    return summaries


# ──────────────────────────────────────────────────────────────
# Step 2: 4 ckpt × ~50 queries forward
# ──────────────────────────────────────────────────────────────

def run_step2_one(c: dict, max_queries: int = 50, tsne_idx: int = 0) -> dict:
    logger.info("-" * 60)
    logger.info(f"Step 2 [{c['name']}]")
    logger.info("-" * 60)

    out_sub = OUT_DIR / c["name"]
    out_sub.mkdir(parents=True, exist_ok=True)

    with open(c["config"], "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    dataset = _build_dataset(cfg)
    model = _build_model_dsn(cfg, c["ckpt"], device)
    n = min(max_queries, len(dataset))

    num_layers = model.num_layers
    layer_names = _layer_names(num_layers)
    all_sims = [[] for _ in layer_names]

    sample_embeddings = None
    sample_col_to_table = None

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= n:
                break
            batch = batch.to(device)
            q_emb = _resolve_query_emb(batch)
            try:
                layer_embs = extract_layerwise_dsn(model, batch, q_emb)
            except Exception as e:
                logger.warning(f"  [{c['name']}] q{idx} skip: {e}")
                continue
            if COL_TO_TAB_EDGE not in batch.edge_index_dict:
                continue
            cb_edge = batch.edge_index_dict[COL_TO_TAB_EDGE]
            for l, ed in enumerate(layer_embs):
                if "column" in ed:
                    all_sims[l].extend(intra_table_sims(ed["column"], cb_edge))
            if idx == tsne_idx:
                sample_embeddings = [d["column"] for d in layer_embs if "column" in d]
                col_ids = cb_edge[0].tolist()
                tab_ids = cb_edge[1].tolist()
                sample_col_to_table = {ci: ti for ci, ti in zip(col_ids, tab_ids)}
            if (idx + 1) % 10 == 0:
                logger.info(f"  [{c['name']}] {idx+1}/{n}")

    plot_oversmoothing_trajectory(all_sims, layer_names, out_sub)
    if sample_embeddings and sample_col_to_table:
        try:
            plot_tsne_per_layer(sample_embeddings, sample_col_to_table, layer_names,
                                out_sub, tsne_idx)
        except Exception as e:
            logger.warning(f"  [{c['name']}] t-SNE fail: {e}")

    stats = {layer_names[l]: {"mean": float(np.mean(s)) if s else float("nan"),
                              "std": float(np.std(s)) if s else float("nan"),
                              "median": float(np.median(s)) if s else float("nan"),
                              "n": len(s)}
             for l, s in enumerate(all_sims)}
    logger.info(f"  [{c['name']}] L0={stats['L0_PLM']['mean']:.4f} → "
                f"L_out={stats['L_out']['mean']:.4f}")
    return {"stats": stats, "layer_names": layer_names}


def plot_cross_model_oversmoothing(step2_all: Dict[str, dict], out: Path) -> Path:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(step2_all.keys())
    x = np.arange(len(names))
    l0 = [step2_all[n]["stats"]["L0_PLM"]["mean"] for n in names]
    deepest = []
    lout = []
    for n in names:
        ln = step2_all[n]["layer_names"]
        gat_layers = [k for k in ln if k.startswith("L") and "GAT" in k]
        deepest.append(step2_all[n]["stats"][gat_layers[-1]]["mean"])
        lout.append(step2_all[n]["stats"]["L_out"]["mean"])
    width = 0.27
    ax.bar(x - width, l0, width, label="L0_PLM", color="tab:gray")
    ax.bar(x, deepest, width, label="deepest_GAT", color="tab:purple")
    ax.bar(x + width, lout, width, label="L_out", color="tab:green")
    ax.axhline(0.85, color="red", linestyle="--", alpha=0.5, label="critical 0.85")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Intra-Table Cosine (mean)")
    ax.set_title("Over-smoothing at Key Depths — DSN 4 ckpt")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    p = out / "cross_model_oversmoothing.png"
    plt.savefig(p, dpi=120)
    plt.close(fig)
    return p


def run_step2(max_queries: int = 50) -> Dict[str, dict]:
    step2_all = {}
    for c in CKPTS:
        if not c["ckpt"].exists():
            logger.warning(f"  [{c['name']}] ckpt missing")
            continue
        step2_all[c["name"]] = run_step2_one(c, max_queries=max_queries)
    if step2_all:
        plot_cross_model_oversmoothing(step2_all, OUT_DIR)
    return step2_all


# ──────────────────────────────────────────────────────────────
# Step 3
# ──────────────────────────────────────────────────────────────

def run_step3_one(c: dict, max_queries: int = 50) -> dict:
    logger.info("-" * 60)
    logger.info(f"Step 3 [{c['name']}]")
    logger.info("-" * 60)
    out_sub = OUT_DIR / c["name"]
    out_sub.mkdir(parents=True, exist_ok=True)

    with open(c["config"], "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cpu")
    dataset = _build_dataset(cfg)
    model = _build_model_dsn(cfg, c["ckpt"], device)

    n = min(max_queries, len(dataset))
    num_layers = model.num_layers
    grad_accum: Dict[str, List[float]] = defaultdict(list)

    # v2 attention extract — V-3-ext directed_from_sn 호환 (forward hook + monkey-patch).
    per_query_attn: List[Dict] = []  # per-query result; aggregate 시 사용

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for idx, batch in enumerate(loader):
        if idx >= n:
            break
        batch = batch.to(device)
        q_emb = _resolve_query_emb(batch)
        try:
            attn = extract_layerwise_attention_v2(model, batch, query_emb=q_emb, topk=5)
            per_query_attn.append(attn)
        except Exception as e:
            if idx == 0:
                logger.warning(f"  [att-v2:{c['name']}] q{idx} skip: {e}")
        try:
            grads = compute_gradient_flow(model, batch, q_emb)
            for k, v in grads.items():
                grad_accum[k].append(v)
        except Exception as e:
            if idx == 0:
                logger.warning(f"  [grad:{c['name']}] q{idx} skip: {e}")
        if (idx + 1) % 10 == 0:
            logger.info(f"  [{c['name']}] {idx+1}/{n}")

    # Aggregate + dump JSON + heatmap → outputs/analysis/dsn_attention/<ckpt>/
    attn_out_dir = ROOT / "outputs/analysis/dsn_attention" / c["name"]
    attn_out_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate_attention_metrics(per_query_attn)
    with open(attn_out_dir / "attention_metrics.json", "w") as fp:
        json.dump(agg, fp, indent=2)
    if agg.get("num_queries", 0) > 0:
        plot_attention_entropy_heatmap(
            agg, str(attn_out_dir / "attention_entropy_layerwise.png"),
            title=f"{c['name']} — Attention entropy (n={agg['num_queries']})")
        plot_topk_concentration_heatmap(
            agg, str(attn_out_dir / "attention_topk5_concentration.png"),
            title=f"{c['name']} — Top-5 attention concentration (n={agg['num_queries']})")
        logger.info(
            f"  [{c['name']}] attn-v2 → {attn_out_dir.relative_to(ROOT)}/"
            f" (n={agg['num_queries']}, layers={agg['num_layers']})"
        )
    else:
        logger.warning(f"  [{c['name']}] attn-v2 produced 0 valid queries")

    if grad_accum:
        plot_gradient_flow(dict(grad_accum), out_sub)

    grad_summary = {k: float(np.mean(v)) for k, v in grad_accum.items() if v}
    conv_keys = sorted([k for k in grad_summary if k.startswith("conv_L")])
    grad_ratio = None
    if len(conv_keys) >= 2:
        first = grad_summary[conv_keys[0]]
        last = grad_summary[conv_keys[-1]]
        grad_ratio = last / first if first > 0 else float("nan")
        logger.info(f"  [{c['name']}] grad ratio = {grad_ratio:.4f}")

    return {"grad_mean": grad_summary, "grad_ratio": grad_ratio,
            "entropy": agg.get("entropy_mean", {}),
            "topk_conc": agg.get("topk_conc_mean", {}),
            "num_layers": num_layers,
            "num_queries_attn": agg.get("num_queries", 0)}


def plot_cross_model_grad_ratio(step3_all: Dict[str, dict], out: Path) -> Path:
    import matplotlib.pyplot as plt
    names = list(step3_all.keys())
    ratios = [step3_all[n]["grad_ratio"] if step3_all[n]["grad_ratio"] is not None
              else float("nan") for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(names, ratios, color="tab:orange", alpha=0.85)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="balanced")
    ax.axhline(0.1, color="red", linestyle="--", alpha=0.5, label="vanish threshold")
    ax.set_ylabel("grad(last conv) / grad(first conv)")
    ax.set_title("Gradient Flow Ratio — DSN 4 ckpt")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    p = out / "cross_model_grad_ratio.png"
    plt.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_cross_model_attention(step3_all: Dict[str, dict], out: Path) -> Path:
    """4 ckpt 의 layer-wise mean attention entropy + top-K concentration 을 비교.

    한 figure 에 2 panel: (a) entropy per layer, (b) top-5 concentration per layer.
    각 ckpt 별로 layer-wise edge_type 평균 → layer 별 단일 값으로 환원.
    """
    import matplotlib.pyplot as plt
    names = list(step3_all.keys())
    if not names:
        return out / "comparison_4ckpt.png"

    def _layer_means(metric_dict: Dict[str, Dict[str, float]]) -> Tuple[List[str], List[float]]:
        if not metric_dict:
            return [], []
        layer_keys = sorted(metric_dict.keys(), key=lambda k: int(k.lstrip("L")))
        means: List[float] = []
        for lk in layer_keys:
            vals = [v for v in metric_dict.get(lk, {}).values()
                    if isinstance(v, float) and v == v]  # filter NaN
            means.append(float(np.mean(vals)) if vals else float("nan"))
        return layer_keys, means

    fig, (ax_e, ax_t) = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(names):
        data = step3_all[name]
        ent = data.get("entropy", {})
        top = data.get("topk_conc", {})
        n_q = data.get("num_queries_attn", 0)
        ent_layers, ent_means = _layer_means(ent)
        top_layers, top_means = _layer_means(top)
        if ent_layers:
            ax_e.plot(ent_layers, ent_means, marker="o", color=cmap(i),
                      label=f"{name} (n={n_q})")
        if top_layers:
            ax_t.plot(top_layers, top_means, marker="o", color=cmap(i),
                      label=f"{name} (n={n_q})")

    ax_e.set_title("Cross-model Attention Entropy (mean over edge types & dst nodes)")
    ax_e.set_xlabel("GAT layer"); ax_e.set_ylabel("mean entropy")
    ax_e.grid(True, alpha=0.3); ax_e.legend(fontsize=8)

    ax_t.set_title("Cross-model Top-5 Attention Concentration")
    ax_t.set_xlabel("GAT layer"); ax_t.set_ylabel("top-5 alpha sum / total")
    ax_t.set_ylim(0, 1.05)
    ax_t.grid(True, alpha=0.3); ax_t.legend(fontsize=8)

    fig.tight_layout()
    p = out / "comparison_4ckpt.png"
    plt.savefig(p, dpi=120)
    plt.close(fig)
    return p


def run_step3(max_queries: int = 50) -> Dict[str, dict]:
    step3_all = {}
    for c in CKPTS:
        if not c["ckpt"].exists():
            continue
        step3_all[c["name"]] = run_step3_one(c, max_queries=max_queries)
    if step3_all:
        plot_cross_model_grad_ratio(step3_all, OUT_DIR)
        # Cross-model attention comparison → outputs/analysis/dsn_attention/comparison_4ckpt.png
        attn_root = ROOT / "outputs/analysis/dsn_attention"
        attn_root.mkdir(parents=True, exist_ok=True)
        comp_path = plot_cross_model_attention(step3_all, attn_root)
        logger.info(f"  cross-model attention plot → {comp_path.relative_to(ROOT)}")
    return step3_all


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_queries", type=int, default=50)
    parser.add_argument("--skip_step1", action="store_true")
    parser.add_argument("--skip_step2", action="store_true")
    parser.add_argument("--skip_step3", action="store_true")
    args = parser.parse_args()

    results: dict = {}

    if not args.skip_step1:
        results["step1"] = run_step1()

    if not args.skip_step2:
        step2_all = run_step2(max_queries=args.max_queries)
        results["step2"] = step2_all

    if not args.skip_step3:
        step3_all = run_step3(max_queries=args.max_queries)
        results["step3"] = step3_all

    summary_path = OUT_DIR / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Batch summary → {summary_path}")


if __name__ == "__main__":
    main()
