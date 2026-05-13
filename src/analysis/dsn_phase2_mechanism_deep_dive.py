"""DSN Phase 2 Mitigation Null Effect — 4 mechanism deep dive.

근거:
  - planning/DECISIONS.md 2026-05-06 (Phase 2 b8 mitigation null effect 결정적, 4 mechanism 위임)
  - notebooks/analysis_results/dsn_oversmoothing_analysis.md (Phase 1 baseline)
  - src/analysis/extract_layerwise_attention_v2.py (V-3-ext 호환 attention)

4 mechanism 후보 (DECISIONS 단계 4-bis §(d)):
  (i)   Aggregation collapse — top-5 attention 노드의 sibling 유사성
  (ii)  GATv2Conv normalization — edge softmax entropy + topk concentration
  (iii) Skip dependency pathology — gradient norm main vs skip
  (iv)  Schema sibling 유사성 — raw PLM L0 intra-table cosine + per-DB

4 ckpt × 50 queries × column 노드. v1 (SchemaHeteroGAT) / v2 (SchemaHeteroGATv2) 분기 자동.

산출물:
  outputs/analysis/dsn_phase2_mechanism_deep_dive/<ckpt>/{aggregation, normalization, skip, sibling}.json
  notebooks/analysis_results/dsn_phase2_mitigation_null_mechanism.md
"""
from __future__ import annotations

import os
import sys
import json
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")

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


def compute_gradient_flow_compat(model, batch, query_emb) -> Dict[str, float]:
    """v1/v2 호환 gradient flow. None 인 module 은 skip, 있는 것만 측정."""
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
from utils.logger import get_logger

logger = get_logger(__name__)

OUT_DIR = ROOT / "outputs/analysis/dsn_phase2_mechanism_deep_dive"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEV_JSON = ROOT / "data/raw/BIRD_dev/dev.json"


# ──────────────────────────────────────────────────────────────
# 4 ckpt 정의
# ──────────────────────────────────────────────────────────────

CKPTS = [
    {
        "name": "p80_phase1",
        "label": "DSN p80 (Phase 1, no mitigation)",
        "config": ROOT / "configs/training/train_gat_directed_supernode_p80.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80.pt",
        "model_class": "v1",
        "best_val_recall": 0.6097,
    },
    {
        "name": "p80_b5_mitigation",
        "label": "DSN p80 + B5 mitigation (Phase 2, ep126/300 in progress)",
        "config": ROOT / "configs/training/train_gat_directed_supernode_p80_b5_mitigation.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_b5_mitigation.pt",
        "model_class": "v2",
        "best_val_recall": 0.6012,
    },
    {
        "name": "s06_b5",
        "label": "s06 B5 (Dual-Stream reference)",
        "config": ROOT / "configs/experiments/s06_gat_bottleneck_fix/a01_additive_ablation/s06_a01_06_b5_dual_stream.yaml",
        "ckpt": Path("/SSL_NAS/peoples/khj/thesis/checkpoints/s06_gat_bottleneck_fix/best_gat_s06_a01_06_b5.pt"),
        "model_class": "v2",
        "best_val_recall": 0.6073,
    },
    {
        "name": "qcond_nl3",
        "label": "qcond_nl3 baseline (no mitigation)",
        "config": ROOT / "configs/training/diameter_layers/train_qcond_nl3.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_qcond_nl3.pt",
        "model_class": "v1",
        "best_val_recall": 0.6061,
    },
]


# ──────────────────────────────────────────────────────────────
# 데이터 로딩 (qid → db_id 매핑)
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
        dev_tables = "data/raw/BIRD_dev/dev_tables.json"
        builder = EnrichedHeteroGraphBuilder(tables_json_path=dev_tables)
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
        model = SchemaHeteroGATv2(
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
            **{k: m[k] for k in ("supernode_edge_direction", "supernode_threshold_mode",
                                 "supernode_threshold_value", "supernode_topk",
                                 "supernode_topk_criterion", "supernode_score_normalization")
               if k in m},
        ).to(device)
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
    model.load_state_dict(state, strict=False)  # AC head 등 분리 ckpt 허용
    model.eval()
    logger.info(f"  loaded {ckpt_path.name} (class={model_class}, "
                f"L={m['num_layers']}, qcond={m.get('query_conditioned', False)}, "
                f"qsn={m.get('query_supernode', False)})")
    return model


# ──────────────────────────────────────────────────────────────
# Mechanism (i): Aggregation collapse — top-5 attention 노드의 sibling 비율
# ──────────────────────────────────────────────────────────────

def analyze_aggregation_collapse(model, batch, layer_attentions: List[Dict],
                                  col_to_tab: Dict[int, int]) -> Dict[str, Any]:
    """Mechanism (i) Aggregation collapse — top-5 attention 흡수 column 들의
    raw PLM cosine sim 측정.

    column→belongs_to→table edge 의 dst (table) 별:
      - 가장 attention 높은 top-5 src column 추출
      - 그 5 column 의 raw PLM (batch.x_dict['column']) embedding 의 pairwise cosine sim mean
      - 값이 높을수록 (≈ 0.85+) → top-5 가 sibling 비슷 → aggregation 결과 자연 collapse
    """
    layer_results: Dict[str, Dict[str, float]] = {}
    col_x_raw = batch.x_dict.get("column")
    if col_x_raw is None or col_x_raw.size(0) == 0:
        return layer_results

    # Pre-normalize for cosine
    col_norm = F.normalize(col_x_raw, dim=-1)

    for layer_idx, layer_dict in enumerate(layer_attentions):
        layer_key = f"L{layer_idx + 1}"
        layer_results[layer_key] = {}

        for et, (att_ei, alpha) in layer_dict.items():
            et_str = "→".join(et)
            # column→table 만 의미 (column 이 src 이므로 raw PLM cosine 가능)
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

            top5_raw_sims = []
            for d, es in grouped.items():
                if len(es) < 2:
                    continue
                top = sorted(es, key=lambda x: -x[1])[:5]
                top_src = [s for s, _ in top]
                if len(top_src) < 2:
                    continue
                # Pairwise raw PLM cosine of top-5 column
                vecs = col_norm[top_src]  # [k, D]
                sim_matrix = vecs @ vecs.T  # [k, k]
                # Off-diagonal mean
                k = sim_matrix.size(0)
                if k < 2:
                    continue
                mask = ~torch.eye(k, dtype=torch.bool)
                top5_raw_sims.append(float(sim_matrix[mask].mean().item()))

            if top5_raw_sims:
                layer_results[layer_key][et_str] = float(np.mean(top5_raw_sims))
    return layer_results


# ──────────────────────────────────────────────────────────────
# Mechanism (iv): Schema sibling — L0 PLM cosine, per-DB
# ──────────────────────────────────────────────────────────────

def analyze_l0_sibling(batch, db_id: str, sims_by_db: Dict[str, List[float]]):
    cb_edge = batch.edge_index_dict.get(COL_TO_TAB_EDGE)
    if cb_edge is None:
        return
    col_x = batch.x_dict.get("column")
    if col_x is None:
        return
    sims = intra_table_sims(col_x, cb_edge)
    sims_by_db[db_id].extend(sims)


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
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Aggregators
    sibling_ratios_by_layer: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    entropy_by_layer: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    topk_conc_by_layer: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    grad_norms: Dict[str, List[float]] = defaultdict(list)
    l0_sims_by_db: Dict[str, List[float]] = defaultdict(list)

    successful = 0
    for idx, batch in enumerate(loader):
        if idx >= n:
            break
        batch = batch.to(device)
        q_emb = _resolve_query_emb(batch)
        db_id = qid_db.get(idx, "unknown")

        # ── Mechanism (iv): L0 PLM intra-table cosine, per-DB ──
        try:
            analyze_l0_sibling(batch, db_id, l0_sims_by_db)
        except Exception as e:
            if idx == 0:
                logger.warning(f"  [l0] q{idx}: {e}")

        # ── Mechanism (ii) + (i): attention extract ──
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

            # Mechanism (i): sibling ratio of top-5 attention
            cb_edge = batch.edge_index_dict.get(COL_TO_TAB_EDGE)
            col_to_tab: Dict[int, int] = {}
            if cb_edge is not None:
                col_to_tab = {int(s): int(t) for s, t in zip(cb_edge[0].tolist(),
                                                              cb_edge[1].tolist())}
            sib = analyze_aggregation_collapse(model, batch, attn_res["raw"], col_to_tab)
            for layer_key, et_map in sib.items():
                for et_str, v in et_map.items():
                    sibling_ratios_by_layer[layer_key][et_str].append(float(v))
            successful += 1
        except Exception as e:
            if idx < 3:
                logger.warning(f"  [attn] q{idx}: {e}")

        # ── Mechanism (iii): gradient flow ──
        try:
            grads = compute_gradient_flow_compat(model, batch, q_emb)
            for k, v in grads.items():
                grad_norms[k].append(v)
        except Exception as e:
            if idx < 3:
                logger.warning(f"  [grad] q{idx}: {e}")

        if (idx + 1) % 10 == 0:
            logger.info(f"  [{c['name']}] {idx+1}/{n} ({successful} attn ok)")

    # Summarize
    def _summarize_layer_et(d: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
        out = {}
        for layer_key, et_map in d.items():
            out[layer_key] = {et: float(np.mean(vs)) for et, vs in et_map.items() if vs}
        return out

    summary = {
        "name": c["name"],
        "label": c["label"],
        "model_class": c["model_class"],
        "best_val_recall": c["best_val_recall"],
        "n_queries": successful,
        "mechanism_i_top5_raw_cosine": _summarize_layer_et(sibling_ratios_by_layer),
        "mechanism_ii_entropy": _summarize_layer_et(entropy_by_layer),
        "mechanism_ii_topk5_conc": _summarize_layer_et(topk_conc_by_layer),
        "mechanism_iii_grad_norm": {k: float(np.mean(v)) for k, v in grad_norms.items() if v},
        "mechanism_iv_l0_intra_table_cosine": {
            db: {
                "mean": float(np.mean(vs)) if vs else float("nan"),
                "std": float(np.std(vs)) if vs else float("nan"),
                "n": len(vs),
            } for db, vs in l0_sims_by_db.items()
        },
    }

    # 전체 평균 (mechanism iv)
    all_l0_sims = [v for vs in l0_sims_by_db.values() for v in vs]
    summary["mechanism_iv_l0_overall"] = {
        "mean": float(np.mean(all_l0_sims)) if all_l0_sims else float("nan"),
        "std": float(np.std(all_l0_sims)) if all_l0_sims else float("nan"),
        "n": len(all_l0_sims),
    }

    # mechanism iii summary: skip vs main GAT path ratio
    grad = summary["mechanism_iii_grad_norm"]
    conv_keys = sorted([k for k in grad if k.startswith("conv_L")])
    if conv_keys:
        max_conv = max(grad[k] for k in conv_keys)
        skip_norm = grad.get("skip_dict", float("nan"))
        summary["mechanism_iii_skip_dependence_ratio"] = (
            float(skip_norm / max_conv) if max_conv > 0 and not np.isnan(skip_norm) else float("nan")
        )
    else:
        summary["mechanism_iii_skip_dependence_ratio"] = float("nan")

    # Save per-ckpt
    out_sub = OUT_DIR / c["name"]
    out_sub.mkdir(parents=True, exist_ok=True)
    with open(out_sub / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  → {out_sub / 'summary.json'}")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_queries", type=int, default=50)
    parser.add_argument("--ckpts", nargs="+", default=None,
                        help="Subset of names. Default: all.")
    args = parser.parse_args()

    qid_db = load_qid_db()
    selected = CKPTS if not args.ckpts else [c for c in CKPTS if c["name"] in args.ckpts]

    all_summaries: Dict[str, Dict[str, Any]] = {}
    for c in selected:
        s = analyze_one_ckpt(c, qid_db, max_queries=args.max_queries)
        if s:
            all_summaries[c["name"]] = s

    with open(OUT_DIR / "batch_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    logger.info(f"\nBatch summary → {OUT_DIR / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
