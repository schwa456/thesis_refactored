"""Wave 14 — Wave 9 Baseline Relog Per-Difficulty R/P/F1 Post-hoc 측정.

DECISIONS 2026-05-20 (Wave 14) §2+§3 정합. 3 baseline relog cells × 1534 queries ×
per-difficulty (simple n=925 / moderate n=464 / challenging n=145).

Spec (Wave 13 patch f67fa65 post + Wave 10 Phase B Spec A 통일):
  - Gold:   parse_sql_elements(gold_sql) from dev.json's SQL field (alias-aware)
  - Pred:   main.py:101-125 path on predictions.jsonl's final_nodes (FK arrow excluded, col-only lowercase)
  - R/P:    calculate_schema_metrics (src/utils/evaluator.py:45-58, reused)
  - F1:     two versions —
              F1_perq_mean: mean(per-query F1 = 2·R·P/(R+P))
              F1_harm:      harmonic mean of overall R, P (HISTORY/paper convention)
  - Aggregate: mean over query, per-difficulty 분해 from predictions.jsonl's difficulty field

산출:
  - outputs/baselines/wave9_relog/{cell}/output_rpf1.jsonl (per-query R/P/F1)
  - outputs/analysis/wave9_per_difficulty_rpf1_2026-05-20.csv (12 rows × R/P/F1/EX + n)
  - outputs/analysis/wave9_per_difficulty_rpf1_2026-05-20.json (full summary)
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path("/home/hyeonjin/thesis_refactored")
sys.path.insert(0, str(ROOT / "src"))

# Wave 13 patched evaluator (alias-aware)
from utils.evaluator import parse_sql_elements, calculate_schema_metrics  # noqa: E402

CELLS = [
    ("G_Retriever_relog", ROOT / "outputs/baselines/wave9_relog/g_retriever_relog"),
    ("LinkAlign_relog",   ROOT / "outputs/baselines/wave9_relog/linkalign_relog"),
    ("XiYan_SQL_relog",   ROOT / "outputs/baselines/wave9_relog/xiyansql_relog"),
]

# Sanity check 기준값 (Wave 13 + Wave 9 sources)
SANITY = {
    "G_Retriever_relog": {
        "R_wave13": 0.9176,
        "EX_simple": 0.5114, "EX_moderate": 0.3125, "EX_challenging": 0.2690,
        "F1_harm_wave13": 0.2858,
    },
    "LinkAlign_relog": {
        "R_wave13": 0.7689,
        "EX_simple": 0.4314, "EX_moderate": 0.2112, "EX_challenging": 0.1586,
        "F1_harm_wave13": 0.3618,
    },
    "XiYan_SQL_relog": {
        "R_wave13": 0.5987,
        "EX_simple": 0.3092, "EX_moderate": 0.1358, "EX_challenging": 0.1379,
        "F1_harm_wave13": 0.6730,
    },
}


def _read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def load_gold_sql_lookup() -> Dict[int, str]:
    """question_id → gold SQL string (BIRD-Dev dev.json's SQL field)."""
    with (ROOT / "data/raw/BIRD_dev/dev.json").open() as f:
        items = json.load(f)
    out = {}
    for it in items:
        out[int(it["question_id"])] = it.get("SQL", it.get("query", ""))
    return out


def pred_cols_from_final_nodes(final_nodes: List[str], gold_cols_lower: Set[str]) -> Set[str]:
    """main.py:101-125 정합 — final_nodes 의 col only (FK arrow 제외, lowercase)."""
    pred = set()
    for node in final_nodes or []:
        if not isinstance(node, str):
            continue
        if "->" in node:
            continue
        if "." in node:
            tbl, col = node.split(".", 1)
            col_lower = col.lower()
            tbl_col_lower = f"{tbl.lower()}.{col_lower}"
            if tbl_col_lower in gold_cols_lower:
                pred.add(tbl_col_lower)
            else:
                pred.add(col_lower)
    return pred


def _harm(R: float, P: float) -> float:
    return 2 * R * P / (R + P) if (R + P) > 0 else 0.0


def _agg(records: List[Tuple[float, float, float, int]]) -> Dict:
    if not records:
        return {"n": 0}
    Rs = [r[0] for r in records]
    Ps = [r[1] for r in records]
    F1s = [r[2] for r in records]
    EXs = [r[3] for r in records]
    R_mean = sum(Rs) / len(Rs)
    P_mean = sum(Ps) / len(Ps)
    return {
        "n": len(records),
        "R": R_mean,
        "P": P_mean,
        "F1_perq_mean": sum(F1s) / len(F1s),
        "F1_harm": _harm(R_mean, P_mean),
        "EX": sum(EXs) / len(EXs),
    }


def measure_cell(cell_tag: str, cell_dir: Path, gold_lookup: Dict[int, str]) -> Dict:
    pred_path = cell_dir / "predictions.jsonl"
    rows = _read_jsonl(pred_path)
    out_path = cell_dir / "output_rpf1.jsonl"

    per_query: List[Dict] = []
    by_difficulty: Dict[str, List] = defaultdict(list)
    overall: List = []

    n_missing_gold = 0
    n_missing_difficulty = 0

    for r in rows:
        qid = r.get("question_id")
        diff = r.get("difficulty")
        if qid is None:
            continue
        if not diff:
            n_missing_difficulty += 1
            continue
        gold_sql = gold_lookup.get(int(qid)) or ""
        if not gold_sql:
            n_missing_gold += 1
            continue

        ex = int(r.get("ex_score", 0))
        final_nodes = r.get("final_nodes", []) or []
        _gold_tables, gold_cols = parse_sql_elements(gold_sql)
        gold_cols = {c.lower() for c in gold_cols}
        pred_cols = pred_cols_from_final_nodes(final_nodes, gold_cols)

        R, P, missing, extra = calculate_schema_metrics(pred_cols, gold_cols)
        F1 = _harm(R, P)

        per_query.append({
            "qid": int(qid),
            "db_id": r.get("db_id"),
            "difficulty": diff,
            "recall": round(R, 4),
            "precision": round(P, 4),
            "f1": round(F1, 4),
            "ex": ex,
            "gold_cols_count": len(gold_cols),
            "pred_cols_count": len(pred_cols),
            "intersection_count": len(pred_cols & gold_cols),
        })
        rec = (R, P, F1, ex)
        overall.append(rec)
        by_difficulty[diff].append(rec)

    # Save per-query jsonl
    with out_path.open("w") as f:
        for record in per_query:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "cell_tag": cell_tag,
        "n_missing_gold": n_missing_gold,
        "n_missing_difficulty": n_missing_difficulty,
        "overall": _agg(overall),
        "per_difficulty": {d: _agg(v) for d, v in by_difficulty.items()},
        "output_rpf1_path": str(out_path),
    }
    return summary


def run_sanity_checks(summaries: Dict[str, Dict]) -> Dict[str, Dict]:
    """4 sanity checks: (a) overall R, (b) mass conservation, (c) per-difficulty EX, (d) F1_harm."""
    results = {}
    for cell, s in summaries.items():
        ref = SANITY[cell]
        ov = s["overall"]
        pd = s["per_difficulty"]

        # (a) Overall R exact match
        a_target = ref["R_wave13"]
        a_actual = ov["R"]
        a_pass = abs(a_actual - a_target) < 5e-5

        # (b) Mass conservation: weighted-mean R = overall R
        weighted_R = 0
        total_n = 0
        for d in ("simple", "moderate", "challenging"):
            v = pd.get(d) or {"n": 0, "R": 0}
            weighted_R += v["n"] * v["R"]
            total_n += v["n"]
        weighted_R_mean = weighted_R / total_n if total_n > 0 else 0
        b_pass = abs(weighted_R_mean - ov["R"]) < 5e-5

        # (c) Per-difficulty EX exact match
        c_pass = True
        c_details = {}
        for d in ("simple", "moderate", "challenging"):
            ref_ex = ref[f"EX_{d}"]
            actual_ex = (pd.get(d) or {}).get("EX")
            if actual_ex is None:
                c_pass = False
                c_details[d] = f"missing (target {ref_ex})"
            else:
                ok = abs(actual_ex - ref_ex) < 5e-5
                c_details[d] = f"{actual_ex:.4f} vs {ref_ex:.4f} {'✓' if ok else '✗'}"
                if not ok:
                    c_pass = False

        # (d) F1_harm exact match
        d_target = ref["F1_harm_wave13"]
        d_actual = ov["F1_harm"]
        d_pass = abs(d_actual - d_target) < 5e-5

        results[cell] = {
            "a_overall_R": {"target": a_target, "actual": a_actual, "delta": a_actual - a_target, "pass": a_pass},
            "b_mass_conservation": {"weighted_mean_R": weighted_R_mean, "overall_R": ov["R"], "delta": weighted_R_mean - ov["R"], "pass": b_pass},
            "c_per_difficulty_EX": {"details": c_details, "pass": c_pass},
            "d_F1_harm": {"target": d_target, "actual": d_actual, "delta": d_actual - d_target, "pass": d_pass},
        }
    return results


def main():
    gold_lookup = load_gold_sql_lookup()
    print(f"Loaded {len(gold_lookup)} gold SQL records from dev.json")
    print()

    summaries = {}
    for cell_tag, cell_dir in CELLS:
        print(f"=== {cell_tag} ===")
        summary = measure_cell(cell_tag, cell_dir, gold_lookup)
        summaries[cell_tag] = summary
        ov = summary["overall"]
        print(
            f"  Overall  n={ov['n']}: R={ov['R']:.4f}  P={ov['P']:.4f}  "
            f"F1_perq={ov['F1_perq_mean']:.4f}  F1_harm={ov['F1_harm']:.4f}  EX={ov['EX']:.4f}"
        )
        for d in ("simple", "moderate", "challenging"):
            v = summary["per_difficulty"].get(d)
            if v:
                print(
                    f"    {d:12s} n={v['n']:4d}  R={v['R']:.4f}  P={v['P']:.4f}  "
                    f"F1_perq={v['F1_perq_mean']:.4f}  F1_harm={v['F1_harm']:.4f}  EX={v['EX']:.4f}"
                )
        if summary["n_missing_gold"]:
            print(f"  ⚠ missing gold_sql: {summary['n_missing_gold']}")
        if summary["n_missing_difficulty"]:
            print(f"  ⚠ missing difficulty: {summary['n_missing_difficulty']}")
        print(f"  per-query output → {summary['output_rpf1_path']}")
        print()

    # Sanity checks
    print("=" * 100)
    print("Sanity Check 4종 검증")
    print("=" * 100)
    sanity_results = run_sanity_checks(summaries)
    all_pass = True
    for cell, checks in sanity_results.items():
        print(f"\n--- {cell} ---")
        a = checks["a_overall_R"]
        print(f"  (a) Overall R retain (Wave 13 §2.2 base): actual {a['actual']:.4f} vs target {a['target']:.4f}  Δ={a['delta']:+.6f}  {'✅' if a['pass'] else '❌'}")
        b = checks["b_mass_conservation"]
        print(f"  (b) Mass conservation:  weighted_mean_R {b['weighted_mean_R']:.4f} vs overall_R {b['overall_R']:.4f}  Δ={b['delta']:+.6f}  {'✅' if b['pass'] else '❌'}")
        c = checks["c_per_difficulty_EX"]
        print(f"  (c) Per-difficulty EX retain (Wave 9 §1.1 base):")
        for d, txt in c["details"].items():
            print(f"        {d:12s}: {txt}")
        print(f"      Overall: {'✅' if c['pass'] else '❌'}")
        d = checks["d_F1_harm"]
        print(f"  (d) F1_harm retain (Wave 13 §5.2 base): actual {d['actual']:.4f} vs target {d['target']:.4f}  Δ={d['delta']:+.6f}  {'✅' if d['pass'] else '❌'}")
        if not all(x["pass"] for x in [a, b, d] + [c]):
            all_pass = False
    print()
    print(f"Overall Sanity Check Status: {'✅ ALL PASS' if all_pass else '❌ SOME FAILED'}")

    # Save CSV
    csv_path = ROOT / "outputs/analysis/wave9_per_difficulty_rpf1_2026-05-20.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_tag", "scope", "n", "R", "P", "F1_perq_mean", "F1_harm", "EX"])
        for cell_tag, s in summaries.items():
            ov = s["overall"]
            w.writerow([
                cell_tag, "overall", ov["n"],
                round(ov["R"], 4), round(ov["P"], 4),
                round(ov["F1_perq_mean"], 4), round(ov["F1_harm"], 4),
                round(ov["EX"], 4),
            ])
            for d in ("simple", "moderate", "challenging"):
                v = s["per_difficulty"].get(d)
                if v:
                    w.writerow([
                        cell_tag, d, v["n"],
                        round(v["R"], 4), round(v["P"], 4),
                        round(v["F1_perq_mean"], 4), round(v["F1_harm"], 4),
                        round(v["EX"], 4),
                    ])
    print(f"\n→ csv:  {csv_path}")

    # Save full JSON
    json_path = ROOT / "outputs/analysis/wave9_per_difficulty_rpf1_2026-05-20.json"
    with json_path.open("w") as f:
        json.dump({"summaries": summaries, "sanity": sanity_results}, f, indent=2, default=str)
    print(f"→ json: {json_path}")


if __name__ == "__main__":
    main()
