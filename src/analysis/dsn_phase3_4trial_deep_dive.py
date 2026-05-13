"""DSN Phase 2 + Phase 3 4-trial Mechanism Deep Dive.

근거:
  - planning/DECISIONS.md 2026-05-06 (Phase 3 #3+#4 결정 + 단계 4-bis 4 mechanism 후보)
  - EXPERIMENT_HISTORY.md "DSN Phase 2 + Phase 3 4-trial Mitigation Sweep" (2026-05-06 → 05-07)
  - 선행 dsn_phase2_mitigation_null_mechanism.md (4 mechanism 정의 + dominance scoring)

분석 대상 (4 ckpt):
  1. Phase 1 P80 (no mit) — best val R 0.6097 (DualTowerProjector + V-1 base)
  2. Phase 2 b8 — best val R 0.6018 (B5 mit + AC fusion)
  3. Phase 3 #3 (Direct AC) — best val R 0.5927 (B5 mit + AC gat_out_L_last)
  4. Phase 3 #4 (Layer-wise LR) — best val R 0.5935 (B5 mit + GAT LR x5)

5 Step:
  Step 1: layer-wise over-smoothing trajectory
  Step 2: attention pattern (top-5 conc + entropy)
  Step 3: gradient flow main vs skip path
  Step 4: raw PLM L0 intra-table cosine
  Step 5: Phase 3 #3 AC=0.62 epoch trajectory + Phase 2/4 AC decay 비교

산출물:
  outputs/analysis/dsn_phase3_4trial_deep_dive/<ckpt>/{summary.json, ac_trajectory.png}
  notebooks/analysis_results/dsn_phase3_mitigation_results.md
"""
from __future__ import annotations

import os
import sys
import json
import re
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_geometric.loader import DataLoader
from data.bird_dataset import BIRDGraphDataset, BIRDSuperNodeDataset
from modules.builders.graph_builder import HeteroGraphBuilder, EnrichedHeteroGraphBuilder
from modules.encoders.local_encoder import LocalPLMEncoder
from models.gat_network import SchemaHeteroGAT
from models.gat_network_v2 import SchemaHeteroGATv2
from analysis.extract_layerwise_attention_v2 import extract_layerwise_attention_v2
from analysis.gat_bottleneck_analysis import (
    intra_table_sims, COL_TO_TAB_EDGE,
)
from utils.logger import get_logger

logger = get_logger(__name__)

OUT_DIR = ROOT / "outputs/analysis/dsn_phase3_4trial_deep_dive"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEV_JSON = ROOT / "data/raw/BIRD_dev/dev.json"


# ──────────────────────────────────────────────────────────────
# 4 ckpt 정의 (Phase 1 / Phase 2 b8 / Phase 3 #3 / Phase 3 #4)
# ──────────────────────────────────────────────────────────────

CKPTS = [
    {
        "name": "phase1_p80",
        "label": "Phase 1 P80 (no mit, baseline)",
        "config": ROOT / "configs/training/train_gat_directed_supernode_p80.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80.pt",
        "log": ROOT / "logs/train/gat_directed_supernode_p80_20260506_001843.log",
        "model_class": "v1",
        "best_val_recall": 0.6097,
        "best_epoch": 91,
    },
    {
        "name": "phase2_b8",
        "label": "Phase 2 b8 (B5 mit, AC fusion)",
        "config": ROOT / "configs/training/train_gat_directed_supernode_p80_b5_mitigation.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_b5_mitigation.pt",
        "log": ROOT / "logs/train/gat_directed_supernode_p80_b5_mitigation_20260506_174522.log",
        "model_class": "v2",
        "best_val_recall": 0.6018,
        "best_epoch": 157,
    },
    {
        "name": "phase3_directAC",
        "label": "Phase 3 #3 (Direct AC on gat_out_L_last)",
        "config": ROOT / "configs/training/train_gat_directed_supernode_p80_b5_phase3_directAC.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_b5_phase3_directAC.pt",
        "log": ROOT / "logs/train/gat_directed_supernode_p80_b5_phase3_directAC_20260507_001849.log",
        "model_class": "v2",
        "best_val_recall": 0.5927,
        "best_epoch": 51,
    },
    {
        "name": "phase3_layerwiseLR",
        "label": "Phase 3 #4 (Layer-wise LR x5)",
        "config": ROOT / "configs/training/train_gat_directed_supernode_p80_b5_phase3_layerwiseLR.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_b5_phase3_layerwiseLR.pt",
        "log": ROOT / "logs/train/gat_directed_supernode_p80_b5_phase3_layerwiseLR_20260507_043405.log",
        "model_class": "v2",
        "best_val_recall": 0.5935,
        "best_epoch": 172,
    },
]


# ──────────────────────────────────────────────────────────────
# Builder helpers (직전 스크립트 동일)
# ──────────────────────────────────────────────────────────────

def load_qid_db() -> Dict[int, str]:
    with open(DEV_JSON, "r") as f:
        dev = json.load(f)
    return {int(d["question_id"]): d.get("db_id", "unknown") for d in dev}


def _resolve_query_emb(batch) -> Optional[torch.Tensor]:
    try:
        if "query_node" in batch.node_types and batch["query_node"].x is not None:
            return batch["query_node"].x
    except Exception:
        pass
    if "query" in batch:
        q = batch["query"]
        return q if q is not None else None
    return None


def build_dataset(cfg: dict):
    builder_type = cfg.get("builder", {}).get("type", "HeteroGraphBuilder")
    if builder_type == "EnrichedHeteroGraphBuilder":
        builder = EnrichedHeteroGraphBuilder(tables_json_path="data/raw/BIRD_dev/dev_tables.json")
    else:
        builder = HeteroGraphBuilder()
    encoder = LocalPLMEncoder()
    dev_json = cfg["paths"].get("test_json", "data/raw/BIRD_dev/dev.json")
    dev_db_dir = cfg["paths"].get("test_db_dir", "data/raw/BIRD_dev/dev_databases")
    dataset = BIRDGraphDataset(json_path=dev_json, db_dir=dev_db_dir,
                                builder=builder, encoder=encoder)
    if cfg["model"].get("query_supernode", False):
        dataset = BIRDSuperNodeDataset(dataset)
    return dataset


def build_model(cfg: dict, ckpt_path: Path, model_class: str, device: torch.device):
    m = cfg["model"]
    if model_class == "v2":
        v2_kwargs = dict(
            in_channels=m["in_channels"],
            hidden_channels=m["hidden_channels"],
            out_channels=m["out_channels"],
            num_layers=m["num_layers"],
            heads=m["heads"],
            query_conditioned=m.get("query_conditioned", False),
            query_supernode=m.get("query_supernode", False),
            pairnorm_mode=m.get("pairnorm_mode", "none"),
            pairnorm_scale=m.get("pairnorm_scale", 1.0),
            initial_residual_alpha=m.get("initial_residual_alpha", 0.0),
            jumping_knowledge=m.get("jumping_knowledge", "none"),
            dual_stream=m.get("dual_stream", False),
        )
        for k in ("supernode_edge_direction", "supernode_threshold_mode",
                  "supernode_threshold_value", "supernode_topk",
                  "supernode_topk_criterion", "supernode_score_normalization"):
            if k in m:
                v2_kwargs[k] = m[k]
        model = SchemaHeteroGATv2(**v2_kwargs).to(device)
    else:
        kwargs = dict(
            in_channels=m["in_channels"],
            hidden_channels=m["hidden_channels"],
            out_channels=m["out_channels"],
            num_layers=m["num_layers"],
            heads=m["heads"],
            query_conditioned=m.get("query_conditioned", False),
            query_supernode=m.get("query_supernode", False),
        )
        for k in ("supernode_edge_direction", "supernode_threshold_mode",
                  "supernode_threshold_value", "supernode_topk",
                  "supernode_topk_criterion", "supernode_score_normalization"):
            if k in m:
                kwargs[k] = m[k]
        model = SchemaHeteroGAT(**kwargs).to(device)

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw, dict):
        for key in ("gat_state_dict", "model_state_dict", "state_dict"):
            if key in raw:
                state = raw[key]
                break
        else:
            state = raw
    else:
        state = raw
    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info(f"  loaded {ckpt_path.name} (class={model_class}, "
                f"L={m['num_layers']}, qcond={m.get('query_conditioned', False)})")
    return model


# ──────────────────────────────────────────────────────────────
# v1/v2 호환 gradient flow (직전 스크립트 동일)
# ──────────────────────────────────────────────────────────────

def compute_gradient_flow_compat(model, batch, query_emb) -> Dict[str, float]:
    model.train()
    model.zero_grad(set_to_none=True)
    out_dict = model(batch.x_dict, batch.edge_index_dict, query_emb=query_emb)
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

    def _gnorm(named_params) -> float:
        sq = 0.0
        for _, p in named_params:
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum().item())
        return sq ** 0.5

    groups: Dict[str, float] = {}
    for name in ("lin_dict", "out_lin_dict", "skip_dict", "jk_lin",
                 "res_proj", "query_encoder", "fusion_head"):
        m = getattr(model, name, None)
        if m is None:
            continue
        try:
            groups[name] = _gnorm(m.named_parameters())
        except Exception:
            pass
    for i in range(model.num_layers):
        try:
            groups[f"conv_L{i+1}"] = _gnorm(model.convs[i].named_parameters())
        except Exception:
            pass
    model.eval()
    return groups


# ──────────────────────────────────────────────────────────────
# Forward hook 기반 layer-wise embeddings (Step 1)
# ──────────────────────────────────────────────────────────────

def extract_layerwise_via_hook(model, batch, query_emb) -> List[Dict[str, torch.Tensor]]:
    """Forward hook 으로 layer 별 hidden capture. v1/v2 모두 호환."""
    embeddings: List[Dict[str, torch.Tensor]] = []
    embeddings.append({nt: x.detach().clone() for nt, x in batch.x_dict.items()})

    captured: List[Dict[str, torch.Tensor]] = []

    def _hook(module, inputs, output):
        if isinstance(output, dict):
            captured.append({nt: x.detach().clone() for nt, x in output.items()})

    handles = [model.convs[i].register_forward_hook(_hook) for i in range(model.num_layers)]
    try:
        with torch.no_grad():
            final = model(batch.x_dict, batch.edge_index_dict, query_emb=query_emb)
    finally:
        for h in handles:
            h.remove()

    for layer_out in captured:
        embeddings.append({nt: F.elu(x).detach().clone() for nt, x in layer_out.items()})

    embeddings.append({nt: x.detach().clone() for nt, x in final.items()})
    return embeddings


# ──────────────────────────────────────────────────────────────
# Step 5: AC loss + main loss epoch trajectory parse
# ──────────────────────────────────────────────────────────────

# Phase 2/3: train_gat_s06.py logs — "Epoch X | Total: Y | Main: M | AC: A | Val Recall@15: R"
PHASE23_PAT = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*"
    r"(?:Total:\s*([\d.]+)\s*\|\s*)?"
    r"(?:Loss:\s*([\d.]+)\s*\|\s*)?"
    r"(?:BCE:\s*[\d.]+\s*\|\s*)?"
    r"(?:InfoNCE:\s*[\d.]+\s*\|\s*)?"
    r"(?:Main:\s*([\d.]+)\s*\|\s*)?"
    r"(?:AC:\s*([\d.]+)\s*\|\s*)?"
    r"Val Recall@15:\s*([\d.]+)"
)


def parse_train_log_v2(log_path: Path) -> Dict[str, np.ndarray]:
    rec = {"epoch": [], "total_loss": [], "main": [], "ac": [], "recall": []}
    if not log_path or not log_path.exists():
        return {k: np.array(v) for k, v in rec.items()}
    with open(log_path, "r") as f:
        for line in f:
            m = PHASE23_PAT.search(line)
            if not m:
                continue
            rec["epoch"].append(int(m.group(1)))
            total = m.group(2) or m.group(3)
            rec["total_loss"].append(float(total) if total else np.nan)
            rec["main"].append(float(m.group(4)) if m.group(4) else np.nan)
            rec["ac"].append(float(m.group(5)) if m.group(5) else np.nan)
            rec["recall"].append(float(m.group(6)))
    return {k: np.array(v) for k, v in rec.items()}


def plot_ac_trajectory(records_by_ckpt: Dict[str, Dict[str, np.ndarray]],
                        out: Path) -> Path:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_map = {"phase2_b8": "tab:orange", "phase3_directAC": "tab:red",
                 "phase3_layerwiseLR": "tab:purple"}
    for name, rec in records_by_ckpt.items():
        if "ac" not in rec or rec["ac"].size == 0:
            continue
        eps = rec["epoch"]
        ac = rec["ac"]
        valid = ~np.isnan(ac)
        if valid.sum() == 0:
            continue
        c = color_map.get(name, "tab:blue")
        ax1.plot(eps[valid], ac[valid], label=f"{name} AC", color=c, linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Anti-Collapse loss", color="tab:red")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.set_title("AC Loss Trajectory (Phase 2 fusion vs Phase 3 #3 gat_out_L_last vs Phase 3 #4 fusion+LRx5)")
    fig.tight_layout()
    p = out / "ac_loss_trajectory.png"
    plt.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_recall_trajectory(records_by_ckpt: Dict[str, Dict[str, np.ndarray]],
                           out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    color_map = {
        "phase1_p80": "tab:blue",
        "phase2_b8": "tab:orange",
        "phase3_directAC": "tab:red",
        "phase3_layerwiseLR": "tab:purple",
    }
    for name, rec in records_by_ckpt.items():
        eps = rec["epoch"]
        rs = rec["recall"]
        if eps.size == 0:
            continue
        c = color_map.get(name, "tab:gray")
        ax.plot(eps, rs, label=name, color=c, linewidth=1.6)
    ax.axhline(0.6097, color="tab:blue", linestyle="--", alpha=0.5, label="Phase 1 ceiling 0.6097")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Recall@15")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("Val R@15 Trajectory — 4 ckpt mitigation null effect")
    fig.tight_layout()
    p = out / "recall_trajectory_overlay.png"
    plt.savefig(p, dpi=120)
    plt.close(fig)
    return p


# ──────────────────────────────────────────────────────────────
# Step 1+2+3+4 — single forward pass per query
# ──────────────────────────────────────────────────────────────

def analyze_top5_raw_cosine(layer_attentions: List[Dict],
                             col_x_norm: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """Mechanism (i): top-5 attention column 의 raw PLM cosine sim."""
    results: Dict[str, Dict[str, float]] = {}
    for layer_idx, layer_dict in enumerate(layer_attentions):
        layer_key = f"L{layer_idx + 1}"
        results[layer_key] = {}
        for et, (att_ei, alpha) in layer_dict.items():
            et_str = "→".join(et)
            if et_str != "column→belongs_to→table":
                continue
            if alpha.numel() == 0:
                continue
            alpha_flat = alpha.mean(dim=-1) if alpha.dim() > 1 else alpha
            src_idx = att_ei[0].tolist()
            dst_idx = att_ei[1].tolist()

            grouped: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
            for e in range(len(src_idx)):
                grouped[dst_idx[e]].append((src_idx[e], float(alpha_flat[e].item())))

            top5_sims = []
            for d, es in grouped.items():
                if len(es) < 2:
                    continue
                top = sorted(es, key=lambda x: -x[1])[:5]
                top_src = [s for s, _ in top]
                if len(top_src) < 2:
                    continue
                vecs = col_x_norm[top_src]
                sim = vecs @ vecs.T
                k = sim.size(0)
                if k < 2:
                    continue
                mask = ~torch.eye(k, dtype=torch.bool)
                top5_sims.append(float(sim[mask].mean().item()))
            if top5_sims:
                results[layer_key][et_str] = float(np.mean(top5_sims))
    return results


def analyze_one_ckpt(c: dict, qid_db: Dict[int, str],
                     max_queries: int = 50) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info(f"Analyzing [{c['name']}] ({c['label']})")
    logger.info("=" * 60)
    if not c["ckpt"].exists():
        logger.warning(f"  ckpt missing: {c['ckpt']}")
        return {}

    with open(c["config"], "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    dataset = build_dataset(cfg)
    model = build_model(cfg, c["ckpt"], c["model_class"], device)

    n = min(max_queries, len(dataset))
    layer_names = ["L0_PLM"] + [f"L{i+1}_GAT" for i in range(model.num_layers)] + ["L_out"]

    # Aggregators
    sims_by_layer = [[] for _ in layer_names]  # Step 1: intra-table cosine
    top5_raw_by_layer: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    entropy_by_layer: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    topk_conc_by_layer: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    grad_norms: Dict[str, List[float]] = defaultdict(list)
    l0_intra_table_sims: List[float] = []

    successful_attn = 0
    successful_step1 = 0
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for idx, batch in enumerate(loader):
        if idx >= n:
            break
        batch = batch.to(device)
        q_emb = _resolve_query_emb(batch)

        cb_edge = batch.edge_index_dict.get(COL_TO_TAB_EDGE)

        # Step 4 — L0 PLM intra-table cosine (mechanism iv)
        if cb_edge is not None and "column" in batch.x_dict:
            try:
                sims = intra_table_sims(batch.x_dict["column"], cb_edge)
                l0_intra_table_sims.extend(sims)
            except Exception:
                pass

        # Step 1 — layer-wise over-smoothing trajectory via hook
        try:
            layer_embs = extract_layerwise_via_hook(model, batch, q_emb)
            if cb_edge is not None:
                for l, ed in enumerate(layer_embs):
                    if "column" in ed:
                        sims_by_layer[l].extend(intra_table_sims(ed["column"], cb_edge))
            successful_step1 += 1
        except Exception as e:
            if idx < 3:
                logger.warning(f"  [layer:{c['name']}] q{idx}: {e}")

        # Step 2 — attention extraction (mechanism i + ii)
        try:
            attn_res = extract_layerwise_attention_v2(model, batch, query_emb=q_emb,
                                                     topk=5, return_raw=True)
            for layer_key, et_map in attn_res["entropy"].items():
                for et_str, v in et_map.items():
                    if not (np.isnan(v) or np.isinf(v)):
                        entropy_by_layer[layer_key][et_str].append(float(v))
            for layer_key, et_map in attn_res["topk_conc"].items():
                for et_str, v in et_map.items():
                    if not (np.isnan(v) or np.isinf(v)):
                        topk_conc_by_layer[layer_key][et_str].append(float(v))

            # Mechanism (i): top-5 attention column 의 raw PLM cosine
            col_x = batch.x_dict.get("column")
            if col_x is not None and col_x.size(0) > 0:
                col_norm = F.normalize(col_x, dim=-1)
                top5 = analyze_top5_raw_cosine(attn_res["raw"], col_norm)
                for layer_key, et_map in top5.items():
                    for et_str, v in et_map.items():
                        top5_raw_by_layer[layer_key][et_str].append(float(v))
            successful_attn += 1
        except Exception as e:
            if idx < 3:
                logger.warning(f"  [attn:{c['name']}] q{idx}: {e}")

        # Step 3 — gradient flow (mechanism iii)
        try:
            grads = compute_gradient_flow_compat(model, batch, q_emb)
            for k, v in grads.items():
                grad_norms[k].append(v)
        except Exception as e:
            if idx < 3:
                logger.warning(f"  [grad:{c['name']}] q{idx}: {e}")

        if (idx + 1) % 10 == 0:
            logger.info(f"  [{c['name']}] {idx+1}/{n} (step1 ok={successful_step1}, attn ok={successful_attn})")

    # Summarize Step 1
    step1 = {layer_names[l]: {
        "mean": float(np.mean(s)) if s else float("nan"),
        "std": float(np.std(s)) if s else float("nan"),
        "n": len(s),
    } for l, s in enumerate(sims_by_layer)}

    # Step 4
    step4 = {
        "mean": float(np.mean(l0_intra_table_sims)) if l0_intra_table_sims else float("nan"),
        "std": float(np.std(l0_intra_table_sims)) if l0_intra_table_sims else float("nan"),
        "n": len(l0_intra_table_sims),
    }

    def _summarize(d):
        out = {}
        for layer_key, et_map in d.items():
            out[layer_key] = {et: float(np.mean(vs)) for et, vs in et_map.items() if vs}
        return out

    grad_summary = {k: float(np.mean(v)) for k, v in grad_norms.items() if v}
    conv_keys = sorted([k for k in grad_summary if k.startswith("conv_L")])
    skip_dep_ratio = float("nan")
    if conv_keys:
        max_conv = max(grad_summary[k] for k in conv_keys)
        skip_norm = grad_summary.get("skip_dict", float("nan"))
        if max_conv > 0 and not np.isnan(skip_norm):
            skip_dep_ratio = float(skip_norm / max_conv)

    summary = {
        "name": c["name"],
        "label": c["label"],
        "model_class": c["model_class"],
        "best_val_recall": c["best_val_recall"],
        "best_epoch": c["best_epoch"],
        "n_queries_step1": successful_step1,
        "n_queries_attn": successful_attn,
        "step1_layer_sims": step1,                             # over-smoothing trajectory
        "mechanism_i_top5_raw_cosine": _summarize(top5_raw_by_layer),
        "mechanism_ii_entropy": _summarize(entropy_by_layer),
        "mechanism_ii_topk5_conc": _summarize(topk_conc_by_layer),
        "mechanism_iii_grad_norm": grad_summary,
        "mechanism_iii_skip_dep_ratio": skip_dep_ratio,
        "mechanism_iv_l0_overall": step4,
    }

    out_sub = OUT_DIR / c["name"]
    out_sub.mkdir(parents=True, exist_ok=True)
    with open(out_sub / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  → {out_sub / 'summary.json'}")
    return summary


# ──────────────────────────────────────────────────────────────
# Step 5 main + plots
# ──────────────────────────────────────────────────────────────

def run_step5_log_trajectory() -> Dict[str, Dict[str, np.ndarray]]:
    """Phase 2/3 의 epoch × Main / AC trajectory parse + plot."""
    logger.info("=" * 60)
    logger.info("Step 5: AC + recall trajectory parse")
    logger.info("=" * 60)
    records: Dict[str, Dict[str, np.ndarray]] = {}
    for c in CKPTS:
        if c.get("log") is None or not c["log"].exists():
            logger.warning(f"  [{c['name']}] log not found")
            continue
        rec = parse_train_log_v2(c["log"])
        if rec["epoch"].size == 0:
            logger.warning(f"  [{c['name']}] empty parse from {c['log']}")
            continue
        records[c["name"]] = rec
        valid_ac = ~np.isnan(rec["ac"])
        ac_summary = ""
        if valid_ac.sum() > 0:
            ac_first = float(rec["ac"][valid_ac][0])
            ac_last = float(rec["ac"][valid_ac][-1])
            ac_summary = f"AC ep{rec['epoch'][valid_ac][0]}={ac_first:.4f} → ep{rec['epoch'][valid_ac][-1]}={ac_last:.4f}"
        logger.info(f"  [{c['name']}] {len(rec['epoch'])} epochs parsed; {ac_summary}")

    # Plots
    if records:
        plot_recall_trajectory(records, OUT_DIR)
        # AC plot only for Phase 2/3 (AC field present)
        ac_records = {k: v for k, v in records.items()
                      if k != "phase1_p80" and not np.isnan(v["ac"]).all()}
        if ac_records:
            plot_ac_trajectory(ac_records, OUT_DIR)
    return records


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_queries", type=int, default=50)
    parser.add_argument("--ckpts", nargs="+", default=None)
    parser.add_argument("--skip_forward", action="store_true",
                        help="Skip 4-ckpt forward (Step 1-4) — only do Step 5 trajectory parse")
    args = parser.parse_args()

    qid_db = load_qid_db()
    selected = CKPTS if not args.ckpts else [c for c in CKPTS if c["name"] in args.ckpts]

    # Step 5: trajectory parse (always)
    log_records = run_step5_log_trajectory()

    if args.skip_forward:
        with open(OUT_DIR / "step5_only_summary.json", "w") as f:
            json.dump({n: {k: v.tolist() for k, v in r.items()}
                       for n, r in log_records.items()}, f, indent=2)
        logger.info("Step 5 only — done.")
        return

    # Step 1-4: forward analysis
    all_summaries: Dict[str, Dict[str, Any]] = {}
    for c in selected:
        s = analyze_one_ckpt(c, qid_db, max_queries=args.max_queries)
        if s:
            all_summaries[c["name"]] = s

    # Save trajectory data into batch summary
    trajectory_export = {}
    for n, r in log_records.items():
        trajectory_export[n] = {k: v.tolist() for k, v in r.items()}

    with open(OUT_DIR / "batch_summary.json", "w") as f:
        json.dump({"step1_to_4": all_summaries, "step5_trajectory": trajectory_export},
                  f, indent=2, default=str)
    logger.info(f"\nBatch summary → {OUT_DIR / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
