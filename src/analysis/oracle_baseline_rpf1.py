"""Wave 12 Oracle Baseline R/P/F1 Post-hoc 측정 (LLM-free).

DECISIONS 2026-05-20 (Wave 12) §2 + planning/metric_spec_2026-05-20.md §1.1-1.4 정합.

3 cells × 1534 queries × R/P/F1 (col-level, Spec A 통일):
  - B1 Full Schema       : pred = 모든 DB col (dev_tables.json 의 db_id 위의 모든 table 의 모든 col)
  - B2 Gold Table        : pred = gold_tables 의 모든 col (dev_tables.json 위)
  - B3 Gold Column       : pred = gold_cols 자체 (sanity check: R=P=F1=1.0 expected)

Gold (모든 cell 공통): parse_sql_elements(gold_sql) → gold_cols (lowercase)
계산: calculate_schema_metrics(pred_cols, gold_cols) — src/utils/evaluator.py 재사용
F1: per-query R, P 위에서 2·R·P / (R+P) post-hoc 계산
Aggregate: mean over 1534 query, R/P/F1 4-decimal.

산출:
  - outputs/analysis/glm_baseline/{b1_full,b2_gold_table,b3_gold_column}/output_rpf1.jsonl
    (각 record: {qid, db_id, recall, precision, f1, gold_cols_count, pred_cols_count, intersection_count})
  - outputs/analysis/oracle_baseline_rpf1_2026-05-20.csv (3 rows × R/P/F1/EX overall + per-difficulty)
  - outputs/analysis/oracle_baseline_rpf1_2026-05-20.json (full summary)
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

from utils.evaluator import parse_sql_elements, calculate_schema_metrics  # noqa: E402

BASELINE_DIR = ROOT / "outputs/analysis/glm_baseline"
DEV_TABLES = ROOT / "data/raw/BIRD_dev/dev_tables.json"
DEV_JSON = ROOT / "data/raw/BIRD_dev/dev.json"

CELLS = [
    ("B1_full",        BASELINE_DIR / "b1_full"),
    ("B2_gold_table",  BASELINE_DIR / "b2_gold_table"),
    ("B3_gold_column", BASELINE_DIR / "b3_gold_column"),
]


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


def load_db_schema_lookup() -> Dict[str, Dict[str, Set[str]]]:
    """db_id → {'tables': set, 'cols': set, 'cols_by_table': {tbl_lower: set(col_lower)}}.

    All names lowercased (Spec A 정합).
    Sentinel column `[-1, '*']` 는 제외.
    """
    with DEV_TABLES.open() as f:
        items = json.load(f)
    out: Dict[str, Dict] = {}
    for db in items:
        db_id = db["db_id"]
        table_names = [t.lower() for t in db.get("table_names_original", [])]
        col_pairs = db.get("column_names_original", [])
        all_cols: Set[str] = set()
        cols_by_table: Dict[str, Set[str]] = {t: set() for t in table_names}
        for table_idx, col_name in col_pairs:
            if table_idx == -1:
                continue  # sentinel '*'
            col_lower = col_name.lower()
            all_cols.add(col_lower)
            if 0 <= table_idx < len(table_names):
                cols_by_table[table_names[table_idx]].add(col_lower)
        out[db_id] = {
            "tables": set(table_names),
            "cols": all_cols,
            "cols_by_table": cols_by_table,
        }
    return out


def load_difficulty_lookup() -> Dict[str, str]:
    """question_id → difficulty (BIRD-Dev metadata). Keys stringified to match jsonl qid."""
    if not DEV_JSON.exists():
        return {}
    with DEV_JSON.open() as f:
        items = json.load(f)
    return {str(it["question_id"]): it.get("difficulty", "unknown") for it in items}


def compute_pred_cols(cell_tag: str, gold_sql: str, db_id: str, db_lookup: Dict) -> Set[str]:
    """Per-cell pred_cols definition (Spec A col-level, lowercase)."""
    db_info = db_lookup.get(db_id) or {"tables": set(), "cols": set(), "cols_by_table": {}}
    if cell_tag == "B1_full":
        return set(db_info["cols"])
    if cell_tag == "B2_gold_table":
        gold_tables, _gold_cols = parse_sql_elements(gold_sql)
        gold_tables = {t.lower() for t in gold_tables}
        pred = set()
        for t in gold_tables:
            pred |= db_info["cols_by_table"].get(t, set())
        return pred
    if cell_tag == "B3_gold_column":
        _gt, gold_cols = parse_sql_elements(gold_sql)
        return {c.lower() for c in gold_cols}
    raise ValueError(f"Unknown cell_tag: {cell_tag}")


def measure_cell(
    cell_tag: str,
    cell_dir: Path,
    db_lookup: Dict,
    diff_lookup: Dict[str, str],
) -> Dict:
    pred_path = cell_dir / "predictions.jsonl"
    rows = _read_jsonl(pred_path)
    out_path = cell_dir / "output_rpf1.jsonl"

    per_query: List[Dict] = []
    by_difficulty: Dict[str, List[Tuple[float, float, float, int]]] = defaultdict(list)
    by_db: Dict[str, List[Tuple[float, float, float, int]]] = defaultdict(list)
    overall: List[Tuple[float, float, float, int]] = []

    parse_fail_qids: List[str] = []   # gold_sql parsing failed (empty gold_cols)
    r_below_one_qids: List[str] = []  # R < 1.0 (only meaningful for B1/B2/B3 sanity)

    for r in rows:
        qid = str(r.get("qid"))
        db_id = r.get("db_id")
        gold_sql = r.get("gold_sql") or ""
        ex = 1 if str(r.get("is_correct")).lower() == "true" else 0

        # Gold via main.py-style parse
        _gt, gold_cols_raw = parse_sql_elements(gold_sql)
        gold_cols = {c.lower() for c in gold_cols_raw}
        if not gold_cols:
            parse_fail_qids.append(qid)

        pred_cols = compute_pred_cols(cell_tag, gold_sql, db_id, db_lookup)

        recall, precision, _missing, _extra = calculate_schema_metrics(pred_cols, gold_cols)
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
        if recall < 1.0:
            r_below_one_qids.append(qid)

        per_query.append({
            "qid": qid,
            "db_id": db_id,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "gold_cols_count": len(gold_cols),
            "pred_cols_count": len(pred_cols),
            "intersection_count": len(pred_cols & gold_cols),
            "ex": ex,
        })
        rec = (recall, precision, f1, ex)
        overall.append(rec)
        diff = diff_lookup.get(qid, "unknown")
        by_difficulty[diff].append(rec)
        by_db[db_id].append(rec)

    # Save per-query jsonl
    with out_path.open("w") as f:
        for record in per_query:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _agg(records: List[Tuple[float, float, float, int]]) -> Dict:
        if not records:
            return {"n": 0}
        Rs = [r[0] for r in records]
        Ps = [r[1] for r in records]
        F1s = [r[2] for r in records]
        EXs = [r[3] for r in records]
        return {
            "n": len(records),
            "R": sum(Rs) / len(Rs),
            "P": sum(Ps) / len(Ps),
            "F1": sum(F1s) / len(F1s),
            "EX": sum(EXs) / len(EXs),
        }

    summary = {
        "cell_tag": cell_tag,
        "overall": _agg(overall),
        "per_difficulty": {d: _agg(v) for d, v in by_difficulty.items()},
        "per_db": {d: _agg(v) for d, v in by_db.items()},
        "parse_fail_count": len(parse_fail_qids),
        "parse_fail_qids_preview": parse_fail_qids[:10],
        "r_below_one_count": len(r_below_one_qids),
        "r_below_one_qids_preview": r_below_one_qids[:10],
        "output_rpf1_path": str(out_path),
    }
    return summary


def main():
    db_lookup = load_db_schema_lookup()
    diff_lookup = load_difficulty_lookup()
    print(f"Loaded {len(db_lookup)} DBs from dev_tables.json")
    print(f"Loaded {len(diff_lookup)} difficulty labels from dev.json")
    print()

    summaries = {}
    for cell_tag, cell_dir in CELLS:
        print(f"=== {cell_tag} ({cell_dir.name}) ===")
        summary = measure_cell(cell_tag, cell_dir, db_lookup, diff_lookup)
        summaries[cell_tag] = summary
        ov = summary["overall"]
        print(
            f"  Overall (n={ov['n']}):  R={ov['R']:.4f}  P={ov['P']:.4f}  "
            f"F1={ov['F1']:.4f}  EX={ov['EX']:.4f}"
        )
        for d in ("simple", "moderate", "challenging"):
            v = summary["per_difficulty"].get(d)
            if v:
                print(
                    f"    {d:12s} n={v['n']:4d}  R={v['R']:.4f}  P={v['P']:.4f}  "
                    f"F1={v['F1']:.4f}  EX={v['EX']:.4f}"
                )
        print(
            f"  Gold parsing fail (empty gold_cols): {summary['parse_fail_count']} "
            f"({summary['parse_fail_qids_preview']})"
        )
        print(
            f"  R < 1.0 queries: {summary['r_below_one_count']} "
            f"({summary['r_below_one_qids_preview']})"
        )
        print(f"  per-query output → {summary['output_rpf1_path']}")
        print()

    # Per-DB matrix print (for report inclusion)
    print("=" * 80)
    print("Per-DB R/P/F1 matrix (B1 Full Schema)")
    print("=" * 80)
    b1 = summaries["B1_full"]
    for db_id, v in sorted(b1["per_db"].items(), key=lambda x: -x[1]["n"]):
        print(
            f"  {db_id:30s} n={v['n']:4d}  R={v['R']:.4f}  P={v['P']:.4f}  "
            f"F1={v['F1']:.4f}  EX={v['EX']:.4f}"
        )
    print()
    print("Per-DB R/P/F1 matrix (B2 Gold Table)")
    print("-" * 80)
    b2 = summaries["B2_gold_table"]
    for db_id, v in sorted(b2["per_db"].items(), key=lambda x: -x[1]["n"]):
        print(
            f"  {db_id:30s} n={v['n']:4d}  R={v['R']:.4f}  P={v['P']:.4f}  "
            f"F1={v['F1']:.4f}  EX={v['EX']:.4f}"
        )

    # Summary CSV
    csv_path = ROOT / "outputs/analysis/oracle_baseline_rpf1_2026-05-20.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "cell_tag", "scope",
            "n", "R", "P", "F1", "EX",
        ])
        for tag, summary in summaries.items():
            ov = summary["overall"]
            w.writerow([tag, "overall", ov["n"],
                        round(ov["R"], 4), round(ov["P"], 4),
                        round(ov["F1"], 4), round(ov["EX"], 4)])
            for d in ("simple", "moderate", "challenging"):
                v = summary["per_difficulty"].get(d)
                if v:
                    w.writerow([tag, d, v["n"],
                                round(v["R"], 4), round(v["P"], 4),
                                round(v["F1"], 4), round(v["EX"], 4)])
    print(f"\n→ summary csv: {csv_path}")

    # Full JSON dump
    json_path = ROOT / "outputs/analysis/oracle_baseline_rpf1_2026-05-20.json"
    with json_path.open("w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"→ full json:    {json_path}")


if __name__ == "__main__":
    main()
