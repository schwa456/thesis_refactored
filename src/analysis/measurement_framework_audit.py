"""Wave 10 Measurement Framework Audit — main.py col-only spec vs phase1 capacity index spec.

For a list of cells, compute R/P/F1 under BOTH measurement specs from the same predictions/output files
to quantify the framework gap.

Spec A — main.py col-only (src/main.py:97-198 + src/utils/evaluator.py:45):
    gold_cols     = {col.lower() for Column nodes in gold SQL (sqlglot)}    # just col name, no table prefix
    pred_cols     = {col.lower() for tbl.col in final_nodes (FK-arrow excluded)}  # just col name
    R             = |pred_cols ∩ gold_cols| / |gold_cols|
    P             = |pred_cols ∩ gold_cols| / |pred_cols|
    Range         = Filter stage (final_nodes 기반) — extractor stage 의 R 측정은 main.py 직접 logging 없음.
                    이전 audit (m2_r_inconsistency_audit_2026-05-18) 의 직접 계산은 extractor_selected_nodes
                    (Wave 7 wave7_relog cell 만 logging) 의 col 만 추출 + FK arrow 제외.

Spec B — phase1 capacity index (score_analysis_*.jsonl 의 is_gold tag + score threshold):
    gold_set      = {tables ∪ columns} as tagged by src/main.py:178-187:
                    is_gold = True if (table.col 형태일 때 tbl in gold_tables AND col in gold_cols)
                             OR (table-only 형태일 때 name in gold_tables)
                    → gold = (#gold_tables) + (#gold_table_x_gold_col matches)
    threshold_set = {node | score >= θ}   (per stage 의미 — Extractor θ 적용 시)
    R_ext         = |gold_set ∩ threshold_set| / |gold_set|

Output: per-cell + overall (R, P, F1) for both specs + ΔR/ΔP/ΔF1 (Spec A - Spec B).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/hyeonjin/thesis_refactored")
OUT_DIR = ROOT / "outputs" / "analysis"


def _read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def spec_a_metrics(output_jsonl: Path) -> dict:
    """Spec A (main.py col-only) — reads output_*.jsonl which already stores per-query recall/precision."""
    rows = _read_jsonl(output_jsonl)
    if not rows:
        return {"R": None, "P": None, "F1": None, "n": 0}
    n = len(rows)
    Rs = [r.get("recall", 0.0) for r in rows]
    Ps = [r.get("precision", 0.0) for r in rows]
    R = sum(Rs) / n
    P = sum(Ps) / n
    F1 = 2 * R * P / (R + P) if (R + P) > 0 else 0.0
    return {"R": R, "P": P, "F1": F1, "n": n}


def spec_b_threshold_metrics(score_jsonl: Path, theta: float = 0.1, col_only: bool = False) -> dict:
    """Spec B (phase1 capacity index) — R/P at extractor stage using score_analysis_*.jsonl with is_gold tags.

    Range: this is "threshold pass" R, i.e. the recall of nodes that survive score >= theta.
    Matches `R_ext` in phase1_sensitivity csv (e.g., R_ext=0.9710 for c01_01 at theta=0.1).

    Args:
        col_only: if True, restrict gold + candidate set to column nodes only (drop table-level nodes).
            Lets us isolate the "tables added to gold" effect of Spec B.
    """
    rows = _read_jsonl(score_jsonl)
    if not rows:
        return {"R": None, "P": None, "F1": None, "n_q": 0}

    by_q = {}
    for r in rows:
        by_q.setdefault(r["query_id"], []).append(r)

    Rs, Ps = [], []
    for qid, recs in by_q.items():
        if col_only:
            recs = [r for r in recs if "." in r["node_name"]]
        gold = sum(1 for r in recs if r["is_gold"])
        if gold == 0:
            continue
        thr_pass = [r for r in recs if r["score"] >= theta]
        gold_thr = sum(1 for r in thr_pass if r["is_gold"])
        Rs.append(gold_thr / gold)
        Ps.append((gold_thr / len(thr_pass)) if thr_pass else 0.0)
    if not Rs:
        return {"R": None, "P": None, "F1": None, "n_q": 0}
    R = sum(Rs) / len(Rs)
    P = sum(Ps) / len(Ps)
    F1 = 2 * R * P / (R + P) if (R + P) > 0 else 0.0
    return {"R": R, "P": P, "F1": F1, "n_q": len(Rs)}


def spec_a_extractor_only(predictions_jsonl: Path, output_jsonl: Path) -> dict:
    """Spec A (main.py col-only) applied to extractor_selected_nodes (FK-arrow excluded) — same evaluator as Wave 7 audit."""
    out_rows = _read_jsonl(output_jsonl)
    gold_per_q = {r["question_id"]: set(c.lower() for c in r.get("gold_cols", [])) for r in out_rows}

    pred_rows = _read_jsonl(predictions_jsonl)
    Rs, Ps = [], []
    has_ext_node_field = 0
    for r in pred_rows:
        qid = r.get("question_id")
        ext_info = r.get("extractor_info") or {}
        ext_dict = ext_info.get("extractor_selected_nodes")
        if ext_dict is None:
            continue
        has_ext_node_field += 1
        ext_cols = {
            c.lower()
            for _, cols in ext_dict.items()
            for c in cols
            if "->" not in c
        }
        gold = gold_per_q.get(qid, set())
        if not gold:
            continue
        inter = gold & ext_cols
        Rs.append(len(inter) / len(gold))
        Ps.append(len(inter) / len(ext_cols) if ext_cols else 0.0)
    if not Rs:
        return {"R": None, "P": None, "F1": None, "n_q": 0, "has_ext_nodes": has_ext_node_field}
    R = sum(Rs) / len(Rs)
    P = sum(Ps) / len(Ps)
    F1 = 2 * R * P / (R + P) if (R + P) > 0 else 0.0
    return {"R": R, "P": P, "F1": F1, "n_q": len(Rs), "has_ext_nodes": has_ext_node_field}


def spec_b_extractor_full(score_jsonl: Path, output_jsonl: Path, theta: float = 0.1) -> dict:
    """Spec B applied to extractor stage — gold_set includes both tables AND columns (as is_gold tagged)."""
    return spec_b_threshold_metrics(score_jsonl, theta=theta)


def audit_cell(cell_root: Path, cell_name: str, theta: float = 0.1) -> dict:
    """Run all measurement variants on a single cell directory."""
    pred = cell_root / "predictions.jsonl"
    output = next(cell_root.glob("output_*.jsonl"), None) or (cell_root / "output.jsonl")
    score = next(cell_root.glob("score_analysis_*.jsonl"), None)
    if score is None:
        score = cell_root / "score_analysis.jsonl"

    res = {"cell": cell_name, "theta": theta}

    # Spec A — main.py col-only filter-stage R/P/F1 (from output_*.jsonl)
    res["spec_a_filter"] = spec_a_metrics(output) if output and output.exists() else None
    # Spec A — main.py col-only extractor-stage R/P/F1 (requires extractor_selected_nodes field)
    if pred.exists() and output and output.exists():
        res["spec_a_extractor"] = spec_a_extractor_only(pred, output)
    else:
        res["spec_a_extractor"] = None
    # Spec B — phase1 capacity-index extractor threshold R/P/F1 (from score_analysis)
    if score and score.exists():
        res["spec_b_extractor"] = spec_b_extractor_full(score, output, theta=theta)
        res["spec_b_col_only_extractor"] = spec_b_threshold_metrics(score, theta=theta, col_only=True)
    else:
        res["spec_b_extractor"] = None
        res["spec_b_col_only_extractor"] = None

    return res


def main():
    cells = [
        # (cell_root, cell_name, theta)
        (ROOT / "outputs/experiments/abl/c01_threshold_sweep/c01_01_theta_0.1", "c01_01 (Wave 5 anchor, θ=0.1)", 0.1),
        (ROOT / "outputs/experiments/abl/c01_threshold_sweep/c01_02_theta_0.2", "c01_02 (θ=0.2)", 0.2),
        (ROOT / "outputs/experiments/abl/c01_threshold_sweep/c01_06_theta_0.6", "c01_06 (θ=0.6)", 0.6),
        (ROOT / "outputs/experiments/abl/c01_threshold_sweep/c01_01_wave7_relog", "c01_01 (Wave 7 relog)", 0.1),
        (ROOT / "outputs/experiments/abl/wave6_recall_biased/wave6_p1_recall_biased_mild", "M1-A mild (Wave 6)", 0.1),
        (ROOT / "outputs/experiments/abl/wave6_recall_biased/wave6_p1_recall_biased_strong", "M1-B strong (Wave 6)", 0.1),
        (ROOT / "outputs/experiments/abl/wave6_recall_biased/wave6_p1_recall_biased_exclusion_rule", "M1-C exclusion (Wave 6)", 0.1),
        (ROOT / "outputs/experiments/abl/wave6_recall_biased/w6_p2a_m2cot_strong", "M2 CoT+Gated (Wave 6)", 0.1),
        (ROOT / "outputs/experiments/abl/wave6_recall_biased/w6_p2_m3_voting", "M3 voting (Wave 6)", 0.1),
        (ROOT / "outputs/experiments/abl/wave6_recall_biased/w6_p2_m4_bidirectional", "M4 Bidirectional (Wave 6)", 0.1),
        (ROOT / "outputs/experiments/abl/wave6_recall_biased/w6_p2_m5_two_stage", "M5 Two-stage (Wave 6)", 0.1),
        (ROOT / "outputs/experiments/abl/wave6_recall_biased/w6_p4_c1_m4_strong", "C1 (M4+strong, Wave 6)", 0.1),
        (ROOT / "outputs/experiments/abl/wave6_recall_biased/w6_p5_c2_m4_majority", "C2 (M4+majority, Wave 6)", 0.1),
        (ROOT / "outputs/baselines/baseline_g_retriever", "B1 G-Retriever (baseline)", 0.1),
        (ROOT / "outputs/baselines/baseline_linkalign", "B2 LinkAlign (baseline)", 0.1),
        (ROOT / "outputs/baselines/baseline_xiyansql", "B3 XiYan-SQL (baseline)", 0.1),
    ]

    summary_rows = []
    for cell_root, name, theta in cells:
        if not cell_root.exists():
            print(f"[SKIP] {name} — not found ({cell_root})", file=sys.stderr)
            continue
        res = audit_cell(cell_root, name, theta=theta)
        summary_rows.append(res)
        a_fil = res.get("spec_a_filter") or {}
        a_ext = res.get("spec_a_extractor") or {}
        b_ext = res.get("spec_b_extractor") or {}
        print(f"\n=== {name} (θ={theta}) ===")
        if a_fil.get("R") is not None:
            print(
                f"  Spec A (main.py col-only) Filter stage:    R={a_fil['R']:.4f} P={a_fil['P']:.4f} F1={a_fil['F1']:.4f} (n={a_fil['n']})"
            )
        if a_ext.get("R") is not None:
            print(
                f"  Spec A (main.py col-only) Extractor stage: R={a_ext['R']:.4f} P={a_ext['P']:.4f} F1={a_ext['F1']:.4f} (n_q={a_ext['n_q']}, ext_node_field_seen={a_ext['has_ext_nodes']})"
            )
        elif a_ext is not None:
            print(
                f"  Spec A Extractor stage: extractor_selected_nodes not logged in predictions.jsonl (has_ext_nodes={a_ext.get('has_ext_nodes')})"
            )
        if b_ext.get("R") is not None:
            print(
                f"  Spec B (phase1 capacity) Extractor θ={theta}: R={b_ext['R']:.4f} P={b_ext['P']:.4f} F1={b_ext['F1']:.4f} (n_q={b_ext['n_q']})"
            )
        b_col = res.get("spec_b_col_only_extractor") or {}
        if b_col.get("R") is not None:
            print(
                f"  Spec B' (col-only, isolate 'table-in-gold' effect): R={b_col['R']:.4f} P={b_col['P']:.4f} F1={b_col['F1']:.4f}"
            )
        if a_ext and a_ext.get("R") is not None and b_ext.get("R") is not None:
            dR = a_ext["R"] - b_ext["R"]
            dP = a_ext["P"] - b_ext["P"]
            dF = a_ext["F1"] - b_ext["F1"]
            print(f"  Δ (Spec A - Spec B) Extractor stage:       ΔR={dR:+.4f} ΔP={dP:+.4f} ΔF1={dF:+.4f}")

    # Save summary
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "measurement_framework_audit_2026-05-18.json"
    with out_json.open("w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\n→ saved: {out_json}")


if __name__ == "__main__":
    main()
