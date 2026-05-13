"""DSN 8-trial Mitigation Mechanism Deep Dive (7 ckpt + Mitigation v3 #1 GIN).

근거:
  - planning/DECISIONS.md 2026-05-08 §1(b) — 사용자 결정 A+B 통합 (multi-DB stratified)
  - 선행: dsn_mitigation_v2_final_7trial.md (7-trial Final dominance scoring)
  - 재현 base: dsn_mitigation_v2_7trial.py (5 step protocol) + dsn_phase1_deep_dive.py (multi-DB stratified)

8 ckpt:
  1. phase1_p80
  2. phase2_b8
  3. phase3_directAC
  4. phase3_layerwiseLR
  5. v2_drop_message
  6. v2_layernorm
  7. v2_sum_aggr
  8. mitigation_v3_gin (best val R@15 0.5954, ep246)

5-Step protocol:
  Step 1: training log trajectory parse
  Step 2: layer-wise over-smoothing trajectory (forward hook v1/v2 + GIN 호환)
  Step 3: attention pattern (extract_layerwise_attention_v2) — GIN 은 attention 부재
          → mech(ii-a) 측정 X, message magnitude 대체 측정
  Step 4: gradient flow main GAT vs skip
  Step 5: AC loss trajectory parse — GIN 의 AC fusion decay

multi-DB stratified: 5 queries × 11 DBs = 55 queries (seed=42)

산출물:
  outputs/analysis/dsn_v3_8trial/<ckpt>/summary.json
  outputs/analysis/dsn_v3_8trial/{recall_overlay, ac_loss, oversmoothing_heatmap, attn_heatmap}.png
  notebooks/analysis_results/dsn_mitigation_v3_8trial.md
"""
from __future__ import annotations

import sys
import json
import yaml
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

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

# Reuse
from analysis.dsn_mitigation_v2_7trial import (
    CKPTS as CKPTS_7,
    build_dataset, build_model, _resolve_query_emb,
    compute_gradient_flow_compat, extract_layerwise_via_hook,
    parse_train_log, plot_recall_trajectory, plot_ac_trajectory,
    plot_oversmoothing_heatmap, plot_attention_heatmap,
)
from analysis.dsn_phase1_deep_dive import build_stratified_qids, load_dev

logger = get_logger(__name__)

OUT_DIR = ROOT / "outputs/analysis/dsn_v3_8trial"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 8 ckpt = 7 + GIN
# ──────────────────────────────────────────────────────────────

CKPT_GIN = {
    "name": "v3_gin",
    "label": "v3 #1 GIN-style aggregation",
    "config": ROOT / "configs/training/train_gat_directed_supernode_p80_b5_mitigation_v3_gin.yaml",
    "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80_b5_mitigation_v3_gin.pt",
    "log": ROOT / "logs/train/gat_directed_supernode_p80_b5_mitigation_v3_gin_20260508_171219.log",
    "model_class": "v2",
    "best_val_recall": 0.5954,
    "best_epoch": 246,
    "category": "v3",
}

CKPTS = list(CKPTS_7) + [CKPT_GIN]


# ──────────────────────────────────────────────────────────────
# GIN-aware analyze (single ckpt) — multi-DB stratified
# ──────────────────────────────────────────────────────────────

def analyze_one_ckpt_stratified(c: dict, qids: List[int],
                                 qid_to_db: Dict[int, str]) -> Dict[str, Any]:
    """Multi-DB stratified analysis on single ckpt."""
    logger.info("=" * 60)
    logger.info(f"Analyzing [{c['name']}] ({c['label']}) — multi-DB stratified")
    logger.info("=" * 60)
    if not c["ckpt"].exists():
        logger.warning(f"  ckpt missing: {c['ckpt']}")
        return {}

    with open(c["config"]) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    dataset = build_dataset(cfg)
    model = build_model(cfg, c["ckpt"], c["model_class"], device)

    layer_names = ["L0_PLM"] + [f"L{i+1}_GAT" for i in range(model.num_layers)] + ["L_out"]
    qid_set = set(qids)

    # Aggregators (per-DB)
    sims_by_layer_db: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    attn_top5_by_db: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    attn_ent_by_db: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    grad_norms: Dict[str, List[float]] = defaultdict(list)
    l0_intra_table_sims_by_db: Dict[str, List[float]] = defaultdict(list)

    # Step 3 alternative for GIN: message magnitude / variance per layer
    # Hook capture conv output magnitude per type
    msg_magnitude_by_layer: Dict[str, List[float]] = defaultdict(list)

    target_et_str = "→".join(("column", "belongs_to", "table"))
    successful_attn = 0
    successful_step1 = 0

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for idx, batch in enumerate(loader):
        if idx not in qid_set:
            continue
        db = qid_to_db[idx]
        batch = batch.to(device)
        q_emb = _resolve_query_emb(batch)
        cb_edge = batch.edge_index_dict.get(COL_TO_TAB_EDGE)

        # L0
        if cb_edge is not None and "column" in batch.x_dict:
            try:
                l0_intra_table_sims_by_db[db].extend(intra_table_sims(batch.x_dict["column"], cb_edge))
            except Exception:
                pass

        # Step 2: layer-wise via hook
        try:
            layer_embs = extract_layerwise_via_hook(model, batch, q_emb)
            if cb_edge is not None:
                for li, ed in enumerate(layer_embs):
                    if "column" in ed:
                        ln_name = layer_names[li] if li < len(layer_names) else f"L{li}"
                        sims_by_layer_db[db][ln_name].extend(intra_table_sims(ed["column"], cb_edge))
                # Message magnitude (post-conv ELU output norm) per layer
                for li, ed in enumerate(layer_embs[1:-1]):  # GAT layers
                    if "column" in ed:
                        col_emb = ed["column"]
                        magnitude = float(col_emb.norm(dim=-1).mean().item())
                        msg_magnitude_by_layer[f"L{li+1}_msg_norm_mean"].append(magnitude)
            successful_step1 += 1
        except Exception as e:
            if successful_step1 < 2:
                logger.warning(f"  [layer:{c['name']}] q{idx}: {e}")

        # Step 3: attention (silent skip for GIN — return_attention_weights 미지원)
        try:
            attn = extract_layerwise_attention_v2(model, batch, query_emb=q_emb, topk=5)
            for lk, et_map in attn.get("topk_conc", {}).items():
                v = et_map.get(target_et_str)
                e = attn.get("entropy", {}).get(lk, {}).get(target_et_str)
                if v is not None and not np.isnan(v):
                    attn_top5_by_db[db][lk].append(float(v))
                if e is not None and not np.isnan(e):
                    attn_ent_by_db[db][lk].append(float(e))
            successful_attn += 1
        except Exception as e:
            if successful_attn < 2:
                logger.warning(f"  [attn:{c['name']}] q{idx}: {e}")

        # Step 4: gradient
        try:
            grads = compute_gradient_flow_compat(model, batch, q_emb)
            for k, v in grads.items():
                grad_norms[k].append(v)
        except Exception as e:
            if len(grad_norms) == 0 and idx == qids[0]:
                logger.warning(f"  [grad:{c['name']}] q{idx}: {e}")

    # Aggregate per-DB + overall
    all_dbs = sorted(set(qid_to_db.values()))
    per_db: Dict[str, Dict[str, Any]] = {}
    for db in all_dbs:
        n_q = sum(1 for qid in qids if qid_to_db[qid] == db)
        entry = {"n_queries": n_q}
        entry["layer_sims_mean"] = {
            ln: float(np.mean(sims_by_layer_db[db][ln])) if sims_by_layer_db[db].get(ln) else float("nan")
            for ln in layer_names
        }
        entry["col_to_tab_top5"] = {
            lk: float(np.mean(attn_top5_by_db[db][lk])) if attn_top5_by_db[db].get(lk) else float("nan")
            for lk in attn_top5_by_db[db]
        }
        entry["col_to_tab_entropy"] = {
            lk: float(np.mean(attn_ent_by_db[db][lk])) if attn_ent_by_db[db].get(lk) else float("nan")
            for lk in attn_ent_by_db[db]
        }
        entry["l0_intra_table_mean"] = (
            float(np.mean(l0_intra_table_sims_by_db[db])) if l0_intra_table_sims_by_db[db] else float("nan")
        )
        per_db[db] = entry

    # Overall (모든 55 queries)
    overall_sims: Dict[str, List[float]] = defaultdict(list)
    overall_top5: Dict[str, List[float]] = defaultdict(list)
    overall_ent: Dict[str, List[float]] = defaultdict(list)
    overall_l0: List[float] = []
    for db in all_dbs:
        for ln, vs in sims_by_layer_db[db].items():
            overall_sims[ln].extend(vs)
        for lk, vs in attn_top5_by_db[db].items():
            overall_top5[lk].extend(vs)
        for lk, vs in attn_ent_by_db[db].items():
            overall_ent[lk].extend(vs)
        overall_l0.extend(l0_intra_table_sims_by_db[db])

    overall = {
        "n_queries": len(qids),
        "layer_sims_mean": {ln: float(np.mean(vs)) if vs else float("nan")
                            for ln, vs in overall_sims.items()},
        "col_to_tab_top5": {lk: float(np.mean(vs)) if vs else float("nan")
                            for lk, vs in overall_top5.items()},
        "col_to_tab_entropy": {lk: float(np.mean(vs)) if vs else float("nan")
                                for lk, vs in overall_ent.items()},
        "l0_intra_table_mean": float(np.mean(overall_l0)) if overall_l0 else float("nan"),
    }

    grad_summary = {k: float(np.mean(v)) for k, v in grad_norms.items() if v}
    conv_keys = sorted([k for k in grad_summary if k.startswith("conv_L")])
    skip_dep = float("nan")
    if conv_keys:
        max_conv = max(grad_summary[k] for k in conv_keys)
        skip_norm = grad_summary.get("skip_dict", float("nan"))
        if max_conv > 0 and not np.isnan(skip_norm):
            skip_dep = float(skip_norm / max_conv)

    msg_mag_summary = {k: float(np.mean(v)) if v else float("nan")
                       for k, v in msg_magnitude_by_layer.items()}

    summary = {
        "name": c["name"],
        "label": c["label"],
        "category": c.get("category", "?"),
        "model_class": c["model_class"],
        "best_val_recall": c["best_val_recall"],
        "best_epoch": c["best_epoch"],
        "n_queries_step1": successful_step1,
        "n_queries_attn": successful_attn,
        "overall": overall,
        "per_db": per_db,
        "mechanism_iii_grad_norm": grad_summary,
        "mechanism_iii_skip_dep_ratio": skip_dep,
        "msg_magnitude_per_layer": msg_mag_summary,
    }

    out_sub = OUT_DIR / c["name"]
    out_sub.mkdir(parents=True, exist_ok=True)
    with open(out_sub / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  → {out_sub / 'summary.json'} (n_attn={successful_attn}, n_step1={successful_step1})")
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_db", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpts", nargs="+", default=None)
    parser.add_argument("--skip_forward", action="store_true")
    args = parser.parse_args()

    dev = load_dev()
    qids, qid_to_db = build_stratified_qids(dev, per_db=args.per_db, seed=args.seed)
    logger.info(f"Stratified qids: n={len(qids)} ({args.per_db}/DB × 11 DBs, seed={args.seed})")

    # Step 5: log parse (always)
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

    # Step 1-4 forward (multi-DB stratified)
    selected = CKPTS if not args.ckpts else [c for c in CKPTS if c["name"] in args.ckpts]
    summaries: Dict[str, Dict[str, Any]] = {}
    for c in selected:
        s = analyze_one_ckpt_stratified(c, qids, qid_to_db)
        if s:
            summaries[c["name"]] = s

    # Heatmaps (using overall)
    if summaries:
        # Convert to format compatible with plot_oversmoothing_heatmap (expects step1_layer_sims)
        for n, s in summaries.items():
            s["step1_layer_sims"] = {
                ln: {"mean": v, "std": float("nan"), "n": 0}
                for ln, v in s["overall"]["layer_sims_mean"].items()
            }
            s["mechanism_ii_topk5_conc"] = {
                lk: {"column→belongs_to→table": s["overall"]["col_to_tab_top5"].get(lk, float("nan"))}
                for lk in s["overall"]["col_to_tab_top5"]
            }
            s["mechanism_ii_entropy"] = {
                lk: {"column→belongs_to→table": s["overall"]["col_to_tab_entropy"].get(lk, float("nan"))}
                for lk in s["overall"]["col_to_tab_entropy"]
            }
        # Plot heatmaps
        # Need to match expected key names — re-import + monkey patch
        try:
            from analysis.dsn_mitigation_v2_7trial import (
                plot_oversmoothing_heatmap as _plot_os,
                plot_attention_heatmap as _plot_attn,
                CKPTS as CKPTS_PLOT_ORDER,
            )
            # Patch CKPTS for ordering (include GIN at end)
            import analysis.dsn_mitigation_v2_7trial as _module
            _module.CKPTS = CKPTS  # 8-trial 순서
            _plot_os(summaries, OUT_DIR)
            _plot_attn(summaries, OUT_DIR, metric="topk5_conc")
            _plot_attn(summaries, OUT_DIR, metric="entropy")
            _module.CKPTS = CKPTS_PLOT_ORDER  # restore
        except Exception as e:
            logger.warning(f"plot heatmap fail: {e}")

    trajectory_export = {n: {k: v.tolist() for k, v in r.items()}
                         for n, r in log_records.items()}
    with open(OUT_DIR / "batch_summary.json", "w") as f:
        json.dump({"step1_to_4": summaries, "step5_trajectory": trajectory_export},
                  f, indent=2, default=str)
    logger.info(f"\nBatch summary → {OUT_DIR / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
