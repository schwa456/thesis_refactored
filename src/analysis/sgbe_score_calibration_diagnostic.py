"""SGBE Score Calibration Diagnostic (DECISIONS 2026-05-12 §SGBE Phase 2.1).

목적: GAT (또는 EnsembleSelector) score 의 column-level calibration 진단.
SGBE (Score-Gated Batch Extractive Filter, planning/filtering_suggestion_by_scholar_agent_2026-05-12.md)
의 θ_keep / θ_drop gating 이 valid 한지 정량 검증 — TP / Filter✗ / TN 3 group 의 score 분포가
실질적으로 분리되는지 확인.

학술 Agent §"세 그룹의 score 분포" 기준값 (anchor stack F1=0.8673 기준):
  - TP        (gold + kept):    mean 0.7108
  - Filter✗   (gold + dropped): mean 0.6394
  - TN        (non-gold + dropped): mean ~0.40

V4 era ckpt (V4-A LN+GIN combo, V4-B AERO) 의 score 분포가 collapse 되었는지 측정.
학술 Agent §"한계와 주의사항": **score 가 collapse 된 경우 (over-smoothing 이 심한 V4-era 결과처럼
score 분포가 균일해지는 경우)** θ_keep / θ_drop 이 무의미해짐 → SGBE 무력.

입력:
  - score_analysis_*.jsonl (Selector 의 column-level score; row format: query_id, node_name, score, is_gold)
  - output_*.jsonl (pipeline 의 query-level pred/gold; row format: question_id, db_id, gold_cols, pred_cols, ...)

출력:
  - <out_dir>/score_distribution.json — 4 group (TP, Filter✗, TN, FP) × {mean, std, n, percentiles} + per-DB
  - <out_dir>/histogram.png — 3 group overlay histogram (TP / Filter✗ / TN)
  - <out_dir>/per_db.csv — 11 BIRD-dev DB × {TP_mean, Filter_X_mean, TN_mean, ...}

Run (anchor stack, BIRD-dev 1534 query):
    PYTHONPATH=src conda run -n base python src/analysis/sgbe_score_calibration_diagnostic.py \\
        --score-analysis outputs/experiments/s04_ablation/pipeline/enriched_qcond_a05_mst_kruskal_glm/score_analysis_s04_pipeline_enriched_qcond_a05_mst_kruskal_glm.jsonl \\
        --pipeline-output outputs/experiments/s04_ablation/pipeline/enriched_qcond_a05_mst_kruskal_glm/output_s04_pipeline_enriched_qcond_a05_mst_kruskal_glm.jsonl \\
        --out-dir outputs/analysis/sgbe_score_calibration/anchor \\
        --tag anchor
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────
# node_name parsing
# ──────────────────────────────────────────────────────────────

def parse_node_kind(node_name: str) -> Tuple[str, str, str]:
    """Return (kind, table, col). kind ∈ {'table', 'column', 'fk', 'other'}.

    Conventions (확인된 anchor stack score_analysis 포맷):
      - 'frpm'                      → table
      - 'frpm.County Name'          → column (table.col_name, no '->')
      - 'frpm.CDS->schools.CDS'     → fk (has '->')
    """
    if "->" in node_name:
        return "fk", "", node_name
    if "." in node_name:
        table, col = node_name.split(".", 1)
        return "column", table, col
    return "table", node_name, ""


def normalize_col(name: str) -> str:
    return name.strip().lower()


# ──────────────────────────────────────────────────────────────
# data loading
# ──────────────────────────────────────────────────────────────

def load_pipeline_output(path: Path) -> Dict[str, Dict]:
    """qid → {db_id, gold_cols_set, pred_cols_set, gold_tables_set, pred_tables_set}"""
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            qid = str(d.get("question_id", d.get("query_id")))
            out[qid] = {
                "db_id": d.get("db_id"),
                "gold_cols": {normalize_col(c) for c in (d.get("gold_cols") or [])},
                "pred_cols": {normalize_col(c) for c in (d.get("pred_cols") or [])},
                "gold_tables": {normalize_col(t) for t in (d.get("gold_tables") or [])},
                "pred_tables": {normalize_col(t) for t in (d.get("pred_tables") or [])},
            }
    return out


def load_score_analysis(path: Path) -> Dict[str, List[Dict]]:
    """qid → list of {node_name, score, is_gold} rows."""
    out: Dict[str, List[Dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            qid = str(d["query_id"])
            out[qid].append({
                "node_name": d["node_name"],
                "score": float(d["score"]),
                "is_gold": str(d.get("is_gold", "")).lower() == "true",
            })
    return out


# ──────────────────────────────────────────────────────────────
# classification + summarization
# ──────────────────────────────────────────────────────────────

def classify_rows(scores: Dict[str, List[Dict]], outputs: Dict[str, Dict],
                  node_kind: str = "column") -> List[Dict]:
    """Cross-join score_analysis × pipeline output. Return per-node rows with
    {qid, db_id, table, col, score, is_gold_query, kept_query, group}.

    group ∈ {'TP', 'FilterX', 'TN', 'FP'}.

    node_kind='column' → only column nodes are kept (학술 Agent 의 SGBE 가 column-level gate).
    """
    rows = []
    qid_intersect = set(scores.keys()) & set(outputs.keys())
    for qid in sorted(qid_intersect, key=lambda x: int(x) if x.isdigit() else x):
        meta = outputs[qid]
        for r in scores[qid]:
            kind, table, col = parse_node_kind(r["node_name"])
            if node_kind != "all" and kind != node_kind:
                continue
            if kind == "column":
                ncol = normalize_col(col)
                is_gold = ncol in meta["gold_cols"]
                kept = ncol in meta["pred_cols"]
            elif kind == "table":
                ntab = normalize_col(table)
                is_gold = ntab in meta["gold_tables"]
                kept = ntab in meta["pred_tables"]
            else:
                continue
            if is_gold and kept:
                grp = "TP"
            elif is_gold and not kept:
                grp = "FilterX"
            elif (not is_gold) and (not kept):
                grp = "TN"
            else:
                grp = "FP"
            rows.append({
                "qid": qid, "db_id": meta["db_id"], "table": table, "col": col,
                "score": r["score"], "is_gold": is_gold, "kept": kept, "group": grp,
            })
    return rows


def summarize_groups(rows: List[Dict]) -> Dict:
    by_grp: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_grp[r["group"]].append(r["score"])
    out = {}
    percentiles = [10, 25, 50, 75, 90]
    for grp, vals in by_grp.items():
        arr = np.array(vals, dtype=np.float64)
        if arr.size == 0:
            continue
        out[grp] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "percentiles": {f"p{p}": float(np.percentile(arr, p)) for p in percentiles},
        }
    return out


def summarize_per_db(rows: List[Dict]) -> Dict[str, Dict]:
    by_db: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        if r["db_id"]:
            by_db[r["db_id"]].append(r)
    out = {}
    for db, db_rows in sorted(by_db.items()):
        out[db] = summarize_groups(db_rows)
        out[db]["n_total"] = len(db_rows)
    return out


def detect_score_collapse(rows: List[Dict], collapse_std_threshold: float = 0.05) -> Dict:
    """학술 Agent 한계 진단 — score 분포가 균일해지는지 (V4-era over-smoothing collapse).

    score_std (전체) < collapse_std_threshold 또는 TP/TN inter-mean spread < 0.05 →
    'collapse' 판정. θ_keep / θ_drop gating 이 무의미해짐.
    """
    if not rows:
        return {"collapsed": False, "reason": "empty rows"}
    arr = np.array([r["score"] for r in rows], dtype=np.float64)
    overall_std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    grp_stats = summarize_groups(rows)
    tp_mean = grp_stats.get("TP", {}).get("mean")
    tn_mean = grp_stats.get("TN", {}).get("mean")
    spread = (tp_mean - tn_mean) if (tp_mean is not None and tn_mean is not None) else None
    collapsed = bool(
        overall_std < collapse_std_threshold or (spread is not None and abs(spread) < 0.05)
    )
    return {
        "collapsed": collapsed,
        "overall_std": overall_std,
        "TP_mean": tp_mean,
        "TN_mean": tn_mean,
        "TP_TN_spread": spread,
        "collapse_threshold_std": collapse_std_threshold,
        "interpretation": (
            "score collapse — θ_keep/θ_drop gating 무력 (학술 Agent §한계와 주의사항)"
            if collapsed
            else "score 분포 분리 유지 — SGBE gating 적용 valid"
        ),
    }


# ──────────────────────────────────────────────────────────────
# plotting
# ──────────────────────────────────────────────────────────────

def plot_histogram(rows: List[Dict], out_path: Path, tag: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[WARN] matplotlib unavailable, skipping histogram", file=sys.stderr)
        return
    by_grp: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_grp[r["group"]].append(r["score"])
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"TP": "tab:green", "FilterX": "tab:orange", "TN": "tab:gray", "FP": "tab:red"}
    bins = np.linspace(0.0, 1.0, 41)
    for grp in ("TN", "FilterX", "TP"):
        if grp in by_grp and len(by_grp[grp]) > 0:
            ax.hist(by_grp[grp], bins=bins, alpha=0.55, label=f"{grp} (n={len(by_grp[grp])})",
                    color=palette.get(grp, None), edgecolor="black", linewidth=0.3)
    ax.axvline(0.65, color="green", linestyle="--", alpha=0.6, label="θ_keep=0.65")
    ax.axvline(0.40, color="red",   linestyle="--", alpha=0.6, label="θ_drop=0.40")
    ax.set_xlabel("Selector score")
    ax.set_ylabel("Count (columns)")
    ax.set_title(f"SGBE score calibration ({tag}) — TP / Filter✗ / TN distribution")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def write_per_db_csv(per_db: Dict[str, Dict], out_path: Path) -> None:
    cols = ["db_id", "n_total", "TP_n", "TP_mean", "FilterX_n", "FilterX_mean",
            "TN_n", "TN_mean", "FP_n", "FP_mean", "TP_TN_spread"]
    with open(out_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for db, stats in sorted(per_db.items()):
            row = [
                db, str(stats.get("n_total", 0)),
                str(stats.get("TP", {}).get("n", 0)),       f"{stats.get('TP', {}).get('mean', 0.0):.4f}",
                str(stats.get("FilterX", {}).get("n", 0)),  f"{stats.get('FilterX', {}).get('mean', 0.0):.4f}",
                str(stats.get("TN", {}).get("n", 0)),       f"{stats.get('TN', {}).get('mean', 0.0):.4f}",
                str(stats.get("FP", {}).get("n", 0)),       f"{stats.get('FP', {}).get('mean', 0.0):.4f}",
                f"{(stats.get('TP', {}).get('mean', 0.0) - stats.get('TN', {}).get('mean', 0.0)):.4f}",
            ]
            f.write(",".join(row) + "\n")


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--score-analysis", required=True, type=Path,
                    help="Selector score_analysis_*.jsonl (query_id/node_name/score/is_gold)")
    ap.add_argument("--pipeline-output", required=True, type=Path,
                    help="Pipeline output_*.jsonl (question_id/db_id/gold_cols/pred_cols/...)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="output dir (outputs/analysis/sgbe_score_calibration/<tag>/)")
    ap.add_argument("--tag", default="anchor", help="anchor / v4a / v4b 등 라벨 (plot title)")
    ap.add_argument("--node-kind", default="column", choices=["column", "table", "all"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SGBE Calib] tag={args.tag}")
    print(f"  score_analysis: {args.score_analysis}")
    print(f"  pipeline_output: {args.pipeline_output}")
    print(f"  out_dir: {args.out_dir}")

    outputs = load_pipeline_output(args.pipeline_output)
    scores = load_score_analysis(args.score_analysis)
    print(f"  loaded {len(scores)} query score sets × {len(outputs)} pipeline outputs")

    rows = classify_rows(scores, outputs, node_kind=args.node_kind)
    print(f"  classified {len(rows)} {args.node_kind} rows")

    grp_stats = summarize_groups(rows)
    per_db = summarize_per_db(rows)
    collapse = detect_score_collapse(rows)

    summary = {
        "tag": args.tag,
        "node_kind": args.node_kind,
        "n_queries": len(set(r["qid"] for r in rows)),
        "n_rows": len(rows),
        "groups": grp_stats,
        "per_db": per_db,
        "collapse_check": collapse,
        "sgbe_thresholds": {"theta_keep": 0.65, "theta_drop": 0.40},
        "reference_means_paper": {"TP": 0.7108, "FilterX": 0.6394, "TN": 0.40},
    }

    json_path = args.out_dir / "score_distribution.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  wrote {json_path}")

    csv_path = args.out_dir / "per_db.csv"
    write_per_db_csv(per_db, csv_path)
    print(f"  wrote {csv_path}")

    plot_path = args.out_dir / "histogram.png"
    plot_histogram(rows, plot_path, tag=args.tag)
    print(f"  wrote {plot_path}")

    # console summary
    print()
    print(f"  ─── group means ({args.tag}) ───")
    for grp in ("TP", "FilterX", "TN", "FP"):
        if grp in grp_stats:
            s = grp_stats[grp]
            print(f"    {grp:<10s}  n={s['n']:6d}  mean={s['mean']:.4f}  std={s['std']:.4f}  "
                  f"p10={s['percentiles']['p10']:.4f}  p90={s['percentiles']['p90']:.4f}")
    print(f"  ─── collapse check ───")
    print(f"    collapsed={collapse['collapsed']}  overall_std={collapse['overall_std']:.4f}  "
          f"TP-TN spread={collapse['TP_TN_spread']}")
    print(f"    → {collapse['interpretation']}")


if __name__ == "__main__":
    main()
