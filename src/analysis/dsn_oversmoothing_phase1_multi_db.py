"""DSN Phase 1 Over-smoothing — Multi-DB Stratified 재측정 (n=55).

근거: planning/DECISIONS.md 2026-05-11 (Phase 1 attention metric Multi-DB Stratified 재측정 결정)

목적:
  1. `outputs/analysis/dsn_attention/{p80, topk20, abstau07, qcond_nl3}/attention_metrics.json`
     의 num_queries=2 (single-DB california_schools 한정) caveat 해소.
  2. dsn_oversmoothing_analysis.run_step3_one 의 try/except 가 idx==0 만 warning 하는
     silent skip 의 fail 사유 분포를 모든 query 에 대해 dump.

Protocol: A3 stratified — `dsn_phase1_deep_dive.build_stratified_qids(per_db=5, seed=42)`
  → 11 BIRD-Dev DBs × 5 queries = 55 queries. Stage 5/7 의 다른 ckpt 와 동일 protocol.

대상 4 ckpt: p80, topk20, abstau07, qcond_nl3 (Phase 1 진단 4 ckpt 전부, model_class='v1').

출력:
  outputs/analysis/dsn_oversmoothing_multi_db/<ckpt>/
    attention_metrics.json   — global aggregation (n=55, mean/std)
    per_db_breakdown.json    — 11 DB × layer × edge_type
    fail_log.json            — qid → fail_stage + exception message
    attention_entropy_layerwise.png + attention_topk5_concentration.png
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_geometric.loader import DataLoader

# 재사용
from analysis.dsn_oversmoothing_analysis import CKPTS as DSN_CKPTS, _build_model_dsn
from analysis.dsn_phase1_deep_dive import build_stratified_qids, load_dev
from analysis.dsn_mitigation_v2_7trial import build_dataset, _resolve_query_emb
from analysis.extract_layerwise_attention_v2 import (
    extract_layerwise_attention_v2,
    aggregate_attention_metrics,
    plot_attention_entropy_heatmap,
    plot_topk_concentration_heatmap,
)
from utils.logger import get_logger
import logging as _logging

logger = get_logger(__name__)
# Console output 활성화 (setup_logger 미호출 환경 대비)
_root = _logging.getLogger("ThesisRefactored")
if not _root.hasHandlers():
    _h = _logging.StreamHandler(sys.stdout)
    _h.setLevel(_logging.INFO)
    _h.setFormatter(_logging.Formatter(
        "[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    _root.addHandler(_h)
    _root.setLevel(_logging.INFO)

OUT_DIR = ROOT / "outputs/analysis/dsn_oversmoothing_multi_db"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COL_TO_TAB_EDGE = ("column", "belongs_to", "table")


# ──────────────────────────────────────────────────────────────────────
# Fail diagnostic helper — every query 의 fail 사유를 stage 별로 classify
# ──────────────────────────────────────────────────────────────────────

def _classify_fail(batch, exc: Exception) -> Dict[str, Any]:
    """Step3 에서 silent skip 의 사유를 stage 별로 분류.

    Stages:
      - empty_col_to_tab_edge: `(column, belongs_to, table)` edge 없음
      - empty_supernode_edge:  `(query_node, attends_to_*, *)` edge 가 모두 비어있음
      - missing_query_emb:     `query_node` 도 없고 `query` 도 없음
      - return_attention_unsupported: GATv2Conv 가 return_attention_weights=True 거부
      - forward_dim_mismatch: state_dict load 후 forward dim mismatch
      - other:                 위에 해당 안 됨 (exc str 포함)
    """
    diag: Dict[str, Any] = {"exc_class": type(exc).__name__, "exc_msg": str(exc)[:300]}

    # Edge 진단
    try:
        ei_dict = batch.edge_index_dict
    except Exception:
        ei_dict = {}

    cb = ei_dict.get(COL_TO_TAB_EDGE)
    diag["col_to_tab_edge_count"] = int(cb.size(1)) if cb is not None else 0

    # SuperNode edges
    sn_edge_counts: Dict[str, int] = {}
    for et, ei in ei_dict.items():
        if isinstance(et, tuple) and len(et) == 3 and et[0] == "query_node":
            sn_edge_counts[f"{et[0]}→{et[1]}→{et[2]}"] = int(ei.size(1)) if ei is not None else 0
    diag["supernode_edge_counts"] = sn_edge_counts
    total_sn = sum(sn_edge_counts.values())
    diag["supernode_total_edges"] = total_sn

    # Query emb 존재 여부
    try:
        has_qnode = ("query_node" in batch.node_types
                     and batch["query_node"].x is not None
                     and batch["query_node"].x.numel() > 0)
    except Exception:
        has_qnode = False
    has_qkey = ("query" in batch) and (batch["query"] is not None)
    diag["has_query_emb"] = has_qnode or has_qkey

    # Fail stage 판정 (heuristic — exc msg + edge 상태)
    msg_low = diag["exc_msg"].lower()
    if "return_attention_weights" in msg_low or "unexpected keyword argument" in msg_low:
        diag["fail_stage"] = "return_attention_unsupported"
    elif "size mismatch" in msg_low or "shape" in msg_low and "mismatch" in msg_low:
        diag["fail_stage"] = "forward_dim_mismatch"
    elif diag["col_to_tab_edge_count"] == 0:
        diag["fail_stage"] = "empty_col_to_tab_edge"
    elif sn_edge_counts and total_sn == 0:
        diag["fail_stage"] = "empty_supernode_edge"
    elif not diag["has_query_emb"]:
        diag["fail_stage"] = "missing_query_emb"
    else:
        diag["fail_stage"] = "other"

    return diag


# ──────────────────────────────────────────────────────────────────────
# Single-ckpt analysis (stratified)
# ──────────────────────────────────────────────────────────────────────

def analyze_one_ckpt_stratified(c: dict, qids: List[int],
                                qid_to_db: Dict[int, str]) -> Dict[str, Any]:
    """Multi-DB stratified attention extraction on a single ckpt.

    Returns:
      {
        "global": aggregate_attention_metrics 결과 (n=55, mean/std),
        "per_db": {db: aggregate_attention_metrics(per_db_results)},
        "fail_log": List[{qid, db, fail_stage, exc_class, exc_msg, ...}],
        "summary": {n_success, n_fail, fail_by_stage, ...}
      }
    """
    out_sub = OUT_DIR / c["name"]
    out_sub.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"[{c['name']}] multi-DB stratified — {len(qids)} qids")
    logger.info("=" * 60)

    if not c["ckpt"].exists():
        logger.warning(f"  [{c['name']}] ckpt missing: {c['ckpt']}")
        return {"global": None, "per_db": {}, "fail_log": [], "summary": {"missing_ckpt": True}}

    with open(c["config"], "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    dataset = build_dataset(cfg)
    model = _build_model_dsn(cfg, c["ckpt"], device)
    qid_set = set(qids)

    per_query_all: List[Dict[str, Any]] = []
    per_query_by_db: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    fail_log: List[Dict[str, Any]] = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    success = 0
    fail = 0
    for idx, batch in enumerate(loader):
        if idx not in qid_set:
            continue
        db = qid_to_db.get(idx, "unknown")
        batch = batch.to(device)
        q_emb = _resolve_query_emb(batch)
        try:
            attn = extract_layerwise_attention_v2(model, batch, query_emb=q_emb, topk=5)
            # attn 의 모든 edge_type entropy/topk_conc 가 NaN 이면 fail 로 처리
            ent = attn.get("entropy", {})
            top = attn.get("topk_conc", {})
            all_nan = True
            for lk in ent:
                for v in ent[lk].values():
                    if isinstance(v, float) and v == v:  # not NaN
                        all_nan = False
                        break
                if not all_nan:
                    break
            if all_nan:
                raise RuntimeError("all_nan_attention_metric")
            per_query_all.append(attn)
            per_query_by_db[db].append(attn)
            success += 1
        except Exception as e:
            fail += 1
            diag = _classify_fail(batch, e)
            diag["qid"] = idx
            diag["db"] = db
            fail_log.append(diag)
            # 모든 query 에 대해 warning (silent skip 진단 — 의도된 동작)
            logger.warning(
                f"  [{c['name']}] q{idx} (db={db}) skip "
                f"[stage={diag['fail_stage']}] {diag['exc_class']}: {diag['exc_msg'][:150]}"
            )

    # Global aggregation
    global_agg = aggregate_attention_metrics(per_query_all)

    # Per-DB aggregation
    per_db_agg: Dict[str, Dict[str, Any]] = {}
    for db, results in per_query_by_db.items():
        per_db_agg[db] = aggregate_attention_metrics(results)

    # Summary
    fail_by_stage: Dict[str, int] = defaultdict(int)
    for entry in fail_log:
        fail_by_stage[entry["fail_stage"]] += 1
    summary = {
        "n_qids": len(qids),
        "n_success": success,
        "n_fail": fail,
        "success_rate": success / max(len(qids), 1),
        "fail_by_stage": dict(fail_by_stage),
        "per_db_success": {db: len(per_query_by_db[db]) for db in sorted(qid_to_db.values())},
    }

    # Dump
    with open(out_sub / "attention_metrics.json", "w") as fp:
        json.dump(global_agg, fp, indent=2)
    with open(out_sub / "per_db_breakdown.json", "w") as fp:
        json.dump(per_db_agg, fp, indent=2)
    with open(out_sub / "fail_log.json", "w") as fp:
        json.dump({"fail_log": fail_log, "summary": summary}, fp, indent=2)

    if global_agg.get("num_queries", 0) > 0:
        plot_attention_entropy_heatmap(
            global_agg, str(out_sub / "attention_entropy_layerwise.png"),
            title=f"{c['name']} — Attention entropy (multi-DB, n={global_agg['num_queries']})")
        plot_topk_concentration_heatmap(
            global_agg, str(out_sub / "attention_topk5_concentration.png"),
            title=f"{c['name']} — Top-5 concentration (multi-DB, n={global_agg['num_queries']})")

    logger.info(
        f"  [{c['name']}] DONE — n_success={success}/{len(qids)} "
        f"(success_rate={summary['success_rate']:.2%}), "
        f"fail_by_stage={dict(fail_by_stage)}"
    )
    return {"global": global_agg, "per_db": per_db_agg,
            "fail_log": fail_log, "summary": summary}


# ──────────────────────────────────────────────────────────────────────
# Cross-ckpt comparison plot
# ──────────────────────────────────────────────────────────────────────

def plot_cross_ckpt_attention(results: Dict[str, Dict[str, Any]], out: Path,
                               target_et: str = "column→belongs_to→table") -> Path:
    """4 ckpt 비교 — col→tab edge 의 layer-wise top5_conc / entropy line plot."""
    import matplotlib.pyplot as plt

    fig, (ax_e, ax_t) = plt.subplots(1, 2, figsize=(13, 4.8))
    ckpt_names = [n for n in results if results[n].get("global") is not None]
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]

    for ci, name in enumerate(ckpt_names):
        agg = results[name]["global"]
        if agg is None:
            continue
        ent_mean = agg.get("entropy_mean", {})
        top_mean = agg.get("topk_conc_mean", {})
        layer_keys = sorted(ent_mean.keys(), key=lambda k: int(k.lstrip("L")))
        ent_vals = [ent_mean.get(lk, {}).get(target_et, float("nan")) for lk in layer_keys]
        top_vals = [top_mean.get(lk, {}).get(target_et, float("nan")) for lk in layer_keys]
        c = colors[ci % len(colors)]
        ax_e.plot(layer_keys, ent_vals, marker="o", label=name, color=c)
        ax_t.plot(layer_keys, top_vals, marker="o", label=name, color=c)

    ax_e.set_title(f"Attention entropy per layer — {target_et}")
    ax_e.set_ylabel("Entropy (mean over dst, mean over queries)")
    ax_e.set_xlabel("Layer")
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(fontsize=8)

    ax_t.set_title(f"Top-5 concentration per layer — {target_et}")
    ax_t.set_ylabel("top5_conc")
    ax_t.set_xlabel("Layer")
    ax_t.set_ylim(0, 1)
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=8)

    fig.tight_layout()
    p = out / "comparison_4ckpt_multi_db.png"
    plt.savefig(p, dpi=120)
    plt.close(fig)
    return p


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_db", type=int, default=5,
                        help="Per-DB stratified sample size (default 5 → 11 DBs × 5 = 55 qids)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpts", type=str, default="p80,topk20,abstau07,qcond_nl3",
                        help="Comma-separated ckpt names to analyze.")
    args = parser.parse_args()

    # Stratified qids
    dev = load_dev()
    qids, qid_to_db = build_stratified_qids(dev, per_db=args.per_db, seed=args.seed)
    logger.info(f"Stratified qids: n={len(qids)} ({args.per_db}/DB × {len(set(qid_to_db.values()))} DBs, seed={args.seed})")

    # Save qids manifest
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "stratified_qids.json", "w") as fp:
        json.dump({"qids": qids, "qid_to_db": qid_to_db,
                   "per_db": args.per_db, "seed": args.seed}, fp, indent=2)

    # CKPTS filter
    requested = set(args.ckpts.split(","))
    ckpts = [c for c in DSN_CKPTS if c["name"] in requested]
    if not ckpts:
        logger.error(f"No matching ckpts for {requested}. Available: {[c['name'] for c in DSN_CKPTS]}")
        sys.exit(1)

    results: Dict[str, Dict[str, Any]] = {}
    for c in ckpts:
        try:
            results[c["name"]] = analyze_one_ckpt_stratified(c, qids, qid_to_db)
        except Exception as e:
            logger.error(f"  [{c['name']}] FATAL: {e}")
            logger.error(traceback.format_exc())
            results[c["name"]] = {"global": None, "per_db": {}, "fail_log": [],
                                  "summary": {"fatal_error": str(e)}}

    # Cross-ckpt comparison plot
    try:
        cmp_path = plot_cross_ckpt_attention(results, OUT_DIR)
        logger.info(f"Cross-ckpt comparison plot → {cmp_path.relative_to(ROOT)}")
    except Exception as e:
        logger.warning(f"Cross-ckpt plot failed: {e}")

    # Cross-ckpt summary JSON (single file for analyzer 보고서 인용)
    cross_summary = {
        "qids_total": len(qids),
        "per_db_total": args.per_db,
        "seed": args.seed,
        "ckpts": {},
    }
    for name, r in results.items():
        glob = r.get("global") or {}
        summary = r.get("summary") or {}
        cross_summary["ckpts"][name] = {
            "n_success": summary.get("n_success", 0),
            "n_fail": summary.get("n_fail", 0),
            "success_rate": summary.get("success_rate", 0.0),
            "fail_by_stage": summary.get("fail_by_stage", {}),
            "entropy_mean_col_to_tab": {
                lk: glob.get("entropy_mean", {}).get(lk, {})
                       .get("column→belongs_to→table", None)
                for lk in glob.get("entropy_mean", {})
            },
            "topk_conc_mean_col_to_tab": {
                lk: glob.get("topk_conc_mean", {}).get(lk, {})
                       .get("column→belongs_to→table", None)
                for lk in glob.get("topk_conc_mean", {})
            },
        }
    with open(OUT_DIR / "cross_ckpt_summary.json", "w") as fp:
        json.dump(cross_summary, fp, indent=2)
    logger.info(f"Cross-ckpt summary → {(OUT_DIR / 'cross_ckpt_summary.json').relative_to(ROOT)}")

    logger.info("=" * 60)
    logger.info("Multi-DB stratified analysis complete.")
    for name, r in results.items():
        s = r.get("summary", {})
        logger.info(f"  {name}: success={s.get('n_success', 0)}/{len(qids)}, "
                    f"fail_by_stage={s.get('fail_by_stage', {})}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
