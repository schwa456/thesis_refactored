"""DSN 7-trial Mitigation Mechanism Deep Dive (Phase 1 + Phase 2 b8 + Phase 3 #3 + #4 +
Mitigation v2 #1 drop_message + v2 #3 layernorm + v2 #2 sum_aggr).

근거:
  - planning/DECISIONS.md 2026-05-07 (Mitigation v2 #1+#3+#2 구현 완료)
  - 직전 dsn_phase3_4trial_deep_dive.py (4 ckpt × 5 step protocol — 본 스크립트 base)
  - notebooks/analysis_results/dsn_phase3_mitigation_results.md (4-trial dominance scoring,
    mech(ii) GATv2Conv normalization dominant 5/5)

Mitigation v2 신규 ckpt 3 (현재 학습 saturation 확정):
  - v2 #1 drop_message: best R@15 0.5970 @ ep78 — DropMessage GATv2Conv (p=0.2)
  - v2 #3 layernorm:    best R@15 0.6007 @ ep82 — LayerNorm pre-softmax (★ mitigation 최고)
  - v2 #2 sum_aggr:     best R@15 0.5735 @ ep29 — HeteroConv sum aggregation (▼ 최저)

7-trial 통합 분석 — 4 mechanism dominance scoring 의 v2 evidence 보강.

산출물:
  outputs/analysis/dsn_mitigation_v2_7trial/<ckpt>/summary.json
  outputs/analysis/dsn_mitigation_v2_7trial/{recall_trajectory, ac_loss, oversmoothing_heatmap, attention_heatmap}.png
  notebooks/analysis_results/dsn_mitigation_v2_results.md
"""
from __future__ import annotations

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

OUT_DIR = ROOT / "outputs/analysis/dsn_mitigation_v2_7trial"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEV_JSON = ROOT / "data/raw/BIRD_dev/dev.json"


# ──────────────────────────────────────────────────────────────
# 7 ckpt 정의
# ──────────────────────────────────────────────────────────────

CKPTS = [
    {
        "name": "phase1_p80",
        "label": "Phase 1 P80 (no mit, baseline)",
        "config": ROOT / "configs/training/train_gat_directed_supernode_p80.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80.pt",
        "log": ROOT / "logs/train/gat_directed_supernode_p80_20260506_001843.log",
        "model_class": "v1",
        "best_val_recall": 0.6097, "best_epoch": 91,
        "category": "baseline",
    },
    {
        "name": "phase2_b8",
        "label": "Phase 2 b8 (B5 mit, AC fusion)",
        "config": ROOT / "configs/training/train_gat_directed_supernode_p80_b5_mitigation.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_b5_mitigation.pt",
        "log": ROOT / "logs/train/gat_directed_supernode_p80_b5_mitigation_20260506_174522.log",
        "model_class": "v2",
        "best_val_recall": 0.6018, "best_epoch": 157,
        "category": "phase2_3",
    },
    {
        "name": "phase3_directAC",
        "label": "Phase 3 #3 (Direct AC on gat_out_L_last)",
        "config": ROOT / "configs/training/train_gat_directed_supernode_p80_b5_phase3_directAC.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_b5_phase3_directAC.pt",
        "log": ROOT / "logs/train/gat_directed_supernode_p80_b5_phase3_directAC_20260507_001849.log",
        "model_class": "v2",
        "best_val_recall": 0.5927, "best_epoch": 51,
        "category": "phase2_3",
    },
    {
        "name": "phase3_layerwiseLR",
        "label": "Phase 3 #4 (Layer-wise LR x5)",
        "config": ROOT / "configs/training/train_gat_directed_supernode_p80_b5_phase3_layerwiseLR.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_b5_phase3_layerwiseLR.pt",
        "log": ROOT / "logs/train/gat_directed_supernode_p80_b5_phase3_layerwiseLR_20260507_043405.log",
        "model_class": "v2",
        "best_val_recall": 0.5935, "best_epoch": 172,
        "category": "phase2_3",
    },
    {
        "name": "v2_drop_message",
        "label": "v2 #1 DropMessage (p=0.2)",
        "config": ROOT / "configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.pt",
        "log": ROOT / "logs/train/gat_directed_supernode_p80_b5_mitigation_v2_drop_message_20260507_163541.log",
        "model_class": "v2",
        "best_val_recall": 0.5970, "best_epoch": 78,
        "category": "v2",
    },
    {
        "name": "v2_layernorm",
        "label": "v2 #3 LayerNorm pre-softmax ★ best mitigation",
        "config": ROOT / "configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_layernorm.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_b5_mitigation_v2_layernorm.pt",
        # 두 log 중 latest (resume 또는 final)
        "log": ROOT / "logs/train/gat_directed_supernode_p80_b5_mitigation_v2_layernorm_20260507_173237.log",
        "model_class": "v2",
        "best_val_recall": 0.6007, "best_epoch": 82,
        "category": "v2",
    },
    {
        "name": "v2_sum_aggr",
        "label": "v2 #2 Sum aggregation ▼ worst variant",
        "config": ROOT / "configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_sum_aggr.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_b5_mitigation_v2_sum_aggr.pt",
        "log": ROOT / "logs/train/gat_directed_supernode_p80_b5_mitigation_v2_sum_aggr_20260507_175952.log",
        "model_class": "v2",
        "best_val_recall": 0.5735, "best_epoch": 29,
        "category": "v2",
    },
]


# ──────────────────────────────────────────────────────────────
# Builder + Model (v1/v2 자동 분기, v2 mitigation 옵션 forward)
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
        # V-3-ext + Mitigation v2 옵션
        for k in ("supernode_edge_direction", "supernode_threshold_mode",
                  "supernode_threshold_value", "supernode_topk",
                  "supernode_topk_criterion", "supernode_score_normalization",
                  "drop_message_p", "use_layernorm_pre_softmax", "aggregation_type"):
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
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    logger.info(f"  loaded {ckpt_path.name} (class={model_class}, "
                f"L={m['num_layers']}, mit_v2 opts: drop_msg_p={m.get('drop_message_p', 0)}, "
                f"LN={m.get('use_layernorm_pre_softmax', False)}, aggr={m.get('aggregation_type', 'mean')}, "
                f"missing={len(missing)}, unexpected={len(unexpected)})")
    return model


# ──────────────────────────────────────────────────────────────
# Gradient flow (v1/v2 호환)
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
# Forward hook layer-wise embeddings
# ──────────────────────────────────────────────────────────────

def extract_layerwise_via_hook(model, batch, query_emb) -> List[Dict[str, torch.Tensor]]:
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


def analyze_top5_raw_cosine(layer_attentions: List[Dict],
                             col_x_norm: torch.Tensor) -> Dict[str, Dict[str, float]]:
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


# ──────────────────────────────────────────────────────────────
# Step 5 — Log trajectory parse
# ──────────────────────────────────────────────────────────────

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


def parse_train_log(log_path: Path) -> Dict[str, np.ndarray]:
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


# ──────────────────────────────────────────────────────────────
# Per-ckpt analyzer
# ──────────────────────────────────────────────────────────────

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

    sims_by_layer = [[] for _ in layer_names]
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

        # L0 PLM
        if cb_edge is not None and "column" in batch.x_dict:
            try:
                sims = intra_table_sims(batch.x_dict["column"], cb_edge)
                l0_intra_table_sims.extend(sims)
            except Exception:
                pass

        # Layer-wise
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

        # Attention
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

        # Gradient
        try:
            grads = compute_gradient_flow_compat(model, batch, q_emb)
            for k, v in grads.items():
                grad_norms[k].append(v)
        except Exception as e:
            if idx < 3:
                logger.warning(f"  [grad:{c['name']}] q{idx}: {e}")

        if (idx + 1) % 10 == 0:
            logger.info(f"  [{c['name']}] {idx+1}/{n} (step1={successful_step1}, attn={successful_attn})")

    step1 = {layer_names[l]: {
        "mean": float(np.mean(s)) if s else float("nan"),
        "std": float(np.std(s)) if s else float("nan"),
        "n": len(s),
    } for l, s in enumerate(sims_by_layer)}

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
    skip_dep = float("nan")
    if conv_keys:
        max_conv = max(grad_summary[k] for k in conv_keys)
        skip_norm = grad_summary.get("skip_dict", float("nan"))
        if max_conv > 0 and not np.isnan(skip_norm):
            skip_dep = float(skip_norm / max_conv)

    summary = {
        "name": c["name"],
        "label": c["label"],
        "category": c["category"],
        "model_class": c["model_class"],
        "best_val_recall": c["best_val_recall"],
        "best_epoch": c["best_epoch"],
        "n_queries_step1": successful_step1,
        "n_queries_attn": successful_attn,
        "step1_layer_sims": step1,
        "mechanism_i_top5_raw_cosine": _summarize(top5_raw_by_layer),
        "mechanism_ii_entropy": _summarize(entropy_by_layer),
        "mechanism_ii_topk5_conc": _summarize(topk_conc_by_layer),
        "mechanism_iii_grad_norm": grad_summary,
        "mechanism_iii_skip_dep_ratio": skip_dep,
        "mechanism_iv_l0_overall": step4,
    }

    out_sub = OUT_DIR / c["name"]
    out_sub.mkdir(parents=True, exist_ok=True)
    with open(out_sub / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  → {out_sub / 'summary.json'}")
    return summary


# ──────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────

CKPT_COLOR = {
    "phase1_p80":         "tab:blue",
    "phase2_b8":          "tab:orange",
    "phase3_directAC":    "tab:red",
    "phase3_layerwiseLR": "tab:purple",
    "v2_drop_message":    "tab:green",
    "v2_layernorm":       "tab:olive",
    "v2_sum_aggr":        "tab:brown",
}


def plot_recall_trajectory(records: Dict[str, Dict[str, np.ndarray]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, rec in records.items():
        eps = rec["epoch"]
        rs = rec["recall"]
        if eps.size == 0:
            continue
        c = CKPT_COLOR.get(name, "tab:gray")
        ax.plot(eps, rs, label=name, color=c, linewidth=1.4)
    ax.axhline(0.6097, color="tab:blue", linestyle="--", alpha=0.5,
               label="Phase 1 ceiling 0.6097")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Recall@15")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Val R@15 Trajectory — 7-trial mitigation null effect (saturation 확정)")
    fig.tight_layout()
    p = out / "recall_trajectory_overlay_7ckpt.png"
    plt.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_ac_trajectory(records: Dict[str, Dict[str, np.ndarray]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, rec in records.items():
        if name == "phase1_p80":
            continue
        ac = rec.get("ac", np.array([]))
        if ac.size == 0 or np.isnan(ac).all():
            continue
        eps = rec["epoch"]
        valid = ~np.isnan(ac)
        c = CKPT_COLOR.get(name, "tab:gray")
        ax.plot(eps[valid], ac[valid], label=name, color=c, linewidth=1.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Anti-Collapse loss (log)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("AC Loss Trajectory — 6 mitigation variants")
    fig.tight_layout()
    p = out / "ac_loss_trajectory_7ckpt.png"
    plt.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_oversmoothing_heatmap(summaries: Dict[str, Dict], out: Path) -> Path:
    """7 ckpt × 5 layer (L0/L1/L2/L3/L_out) heatmap."""
    layer_keys = ["L0_PLM", "L1_GAT", "L2_GAT", "L3_GAT", "L_out"]
    names = [c["name"] for c in CKPTS if c["name"] in summaries]
    matrix = np.full((len(names), len(layer_keys)), np.nan)
    for i, n in enumerate(names):
        s = summaries[n]
        sims = s.get("step1_layer_sims", {})
        for j, ln in enumerate(layer_keys):
            v = sims.get(ln, {}).get("mean")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(layer_keys)))
    ax.set_xticklabels(layer_keys, rotation=15)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(layer_keys)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v > 0.6 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label="intra-table cosine")
    ax.set_title("Step 2: Over-smoothing trajectory — 7 ckpt × layer (column intra-table cosine)")
    fig.tight_layout()
    p = out / "oversmoothing_heatmap_7ckpt.png"
    plt.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_attention_heatmap(summaries: Dict[str, Dict], out: Path,
                           metric: str = "topk5_conc") -> Path:
    """7 ckpt × layer (col→tab top5_conc 또는 entropy)."""
    names = [c["name"] for c in CKPTS if c["name"] in summaries]
    layer_keys: List[str] = []
    for n in names:
        if metric == "topk5_conc":
            d = summaries[n].get("mechanism_ii_topk5_conc", {})
        else:
            d = summaries[n].get("mechanism_ii_entropy", {})
        layer_keys = sorted(set(layer_keys) | set(d.keys()), key=lambda x: int(x.lstrip("L")))
    if not layer_keys:
        return out / f"attention_heatmap_{metric}.png"

    matrix = np.full((len(names), len(layer_keys)), np.nan)
    for i, n in enumerate(names):
        if metric == "topk5_conc":
            d = summaries[n].get("mechanism_ii_topk5_conc", {})
        else:
            d = summaries[n].get("mechanism_ii_entropy", {})
        for j, lk in enumerate(layer_keys):
            v = d.get(lk, {}).get("column→belongs_to→table")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = "RdYlGn_r" if metric == "topk5_conc" else "RdYlGn"
    vmin, vmax = (0.2, 0.9) if metric == "topk5_conc" else (1.5, 3.5)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(layer_keys)))
    ax.set_xticklabels(layer_keys)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(layer_keys)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v > (vmin+vmax)/2 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label=metric)
    ax.set_title(f"Step 3: col→tab edge softmax {metric} — 7 ckpt × layer")
    fig.tight_layout()
    p = out / f"attention_heatmap_{metric}_7ckpt.png"
    plt.savefig(p, dpi=120)
    plt.close(fig)
    return p


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_queries", type=int, default=50)
    parser.add_argument("--ckpts", nargs="+", default=None)
    parser.add_argument("--skip_forward", action="store_true")
    args = parser.parse_args()

    qid_db = load_qid_db()
    selected = CKPTS if not args.ckpts else [c for c in CKPTS if c["name"] in args.ckpts]

    # Step 5 — log trajectory parse (always)
    log_records: Dict[str, Dict[str, np.ndarray]] = {}
    for c in CKPTS:
        if c.get("log") is None or not c["log"].exists():
            logger.warning(f"  [{c['name']}] log not found")
            continue
        rec = parse_train_log(c["log"])
        if rec["epoch"].size == 0:
            continue
        log_records[c["name"]] = rec
        valid_ac = ~np.isnan(rec["ac"])
        ac_str = ""
        if valid_ac.sum() > 0:
            ac_str = f" AC ep{rec['epoch'][valid_ac][0]}={rec['ac'][valid_ac][0]:.4f}→ep{rec['epoch'][valid_ac][-1]}={rec['ac'][valid_ac][-1]:.4f}"
        logger.info(f"  [{c['name']}] {len(rec['epoch'])} epochs;{ac_str} R best={rec['recall'].max():.4f}")

    if log_records:
        plot_recall_trajectory(log_records, OUT_DIR)
        plot_ac_trajectory(log_records, OUT_DIR)

    if args.skip_forward:
        logger.info("Step 5 only — done.")
        return

    # Step 1-4 forward
    summaries: Dict[str, Dict[str, Any]] = {}
    for c in selected:
        s = analyze_one_ckpt(c, qid_db, max_queries=args.max_queries)
        if s:
            summaries[c["name"]] = s

    # Heatmaps
    if summaries:
        plot_oversmoothing_heatmap(summaries, OUT_DIR)
        plot_attention_heatmap(summaries, OUT_DIR, metric="topk5_conc")
        plot_attention_heatmap(summaries, OUT_DIR, metric="entropy")

    trajectory_export = {n: {k: v.tolist() for k, v in r.items()}
                         for n, r in log_records.items()}
    with open(OUT_DIR / "batch_summary.json", "w") as f:
        json.dump({"step1_to_4": summaries, "step5_trajectory": trajectory_export},
                  f, indent=2, default=str)
    logger.info(f"\nBatch summary → {OUT_DIR / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
