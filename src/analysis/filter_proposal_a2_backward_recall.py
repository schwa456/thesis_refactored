"""Filter Proposal Phase 1 — A-2: Backward Recall.

근거:
  - planning/DECISIONS.md 2026-05-13 (학술 Agent Phase 1 GO)
  - planning/filter_proposal_data_spec_2026-05-13.md §2 A-2

목적:
  A-1 의 prelim_sql_full 에서 column 추출 → L_bwd.
  L_fwd = anchor 의 final_nodes (S_fwd) 와 비교.
  recall_fwd / recall_bwd / recall_union 측정 — Direction A backward path 유효성 정량.

Decision Rule (학술 Agent):
  mean(recall_union) - mean(recall_fwd) ≥ 0.05 → backward path 유효

A-1 결과 의존. LLM call 0 (sqlglot parsing only).

Output: outputs/analysis/filter_proposal/A2_backward_recall.jsonl (1534 records)
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict

import sqlglot
from sqlglot.expressions import Column, Table
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_DIR = ROOT / "outputs/analysis/filter_proposal"


def extract_columns_from_sql(sql: Optional[str]) -> Set[str]:
    """sqlglot 으로 column 추출 — col-only (lowercase, table prefix 제거).

    🔧 2026-05-13 bug fix: 직전 버전이 'table.col' (sqlglot 의 SQL alias prefix 포함, e.g. 't1.atom_id')
    형식으로 추출 → gold_cols 와 numerator 정규화가 비대칭 (alias-distinct vs col-only-distinct) 으로
    recall > 1.0 발생 (max 2.0). 본 fix 는 filter_sweep / B1' baseline 의 gold_cols 와 정합 (col-only).

    파싱 실패 또는 None 입력 시 빈 set 반환."""
    if not sql or not isinstance(sql, str):
        return set()
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
        cols = set()
        for node in parsed.find_all(Column):
            name = node.name
            if not name:
                continue
            # col-only — SQL alias 영향 회피 (table prefix 제거)
            cols.add(name.lower())
        return cols
    except Exception:
        return set()


def extract_columns_normalized(cols_raw: List[str]) -> Set[str]:
    """anchor 의 final_nodes (예: 'frpm.County Name' or 'frpm') → col-only normalized set.

    🔧 2026-05-13 bug fix: 직전 'table.col' + 'col' 양쪽 추가 방식 → intersection 이 double-count.
    본 fix 는 col-only 만 추출 (anchor final_nodes 의 'table.col' 에서 col 만 추출).
    'table' 단독 entry 는 제외 (column-level 비교 목적).
    """
    out = set()
    for n in cols_raw or []:
        if not isinstance(n, str):
            continue
        # 'table.col' 형식만 column 으로 간주 (col-only 추출)
        if "." in n:
            col = n.split(".", 1)[1].lower()
            out.add(col)
        # else: table 단독 entry — column 추출에서 제외
    return out


def column_set_normalize_for_compare(cols: Set[str]) -> Set[str]:
    """Column set 의 비교용 정규화 — col-only (lowercase, strip table prefix).

    🔧 2026-05-13 bug fix: 직전 'table.col' + 'col' 둘 다 추가 방식 → set size 가 2× 부풀어
    intersection 이 double-count. 본 fix 는 col-only 로 통일.
    """
    out = set()
    for c in cols:
        c = c.lower().strip()
        if "." in c:
            col = c.split(".", 1)[1]
            out.add(col)
        else:
            out.add(c)
    return out


def _intersect_size(a: Set[str], b: Set[str]) -> int:
    """Symmetric column match — col-only set intersection size."""
    a_norm = column_set_normalize_for_compare(a)
    b_norm = column_set_normalize_for_compare(b)
    return len(a_norm & b_norm)


def run_a2(a1_path: Path, output_path: Path) -> Dict[str, Any]:
    """A-1 결과 → A-2 backward recall 계산."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sum_recall_fwd, sum_recall_bwd, sum_recall_union = 0.0, 0.0, 0.0
    n_valid = 0
    n_truncated = 0
    sum_L_bwd_size, sum_L_fwd_size, sum_restore_size = 0, 0, 0

    n_total = 0
    with open(a1_path) as f_in, open(output_path, "w") as f_out:
        for line in tqdm(f_in, desc="A-2 backward recall"):
            r = json.loads(line)
            qid = r["query_id"]
            db_id = r["db_id"]
            truncated = r.get("truncated_full", False)
            n_total += 1

            if truncated or r.get("prelim_sql_full") is None:
                L_bwd: Set[str] = set()
                n_truncated += 1
            else:
                # col-only extraction (bug fix 2026-05-13)
                L_bwd = extract_columns_from_sql(r["prelim_sql_full"])

            L_fwd_raw = r.get("S_fwd_from_anchor", [])
            L_fwd = extract_columns_normalized(L_fwd_raw)  # col-only normalized

            gold_cols_raw = extract_columns_from_sql(r["gold_sql"])  # col-only set
            gold_cols = column_set_normalize_for_compare(gold_cols_raw)  # idempotent (이미 col-only)

            # Sizes (col-only 기준 — bug fix 후 consistent)
            L_bwd_size = len(L_bwd)
            L_fwd_size = len(L_fwd_raw)  # anchor final_nodes 의 raw size (보고 용도, recall 계산엔 미사용)
            intersection_size = _intersect_size(L_bwd, L_fwd)  # col-only
            restore_set = L_bwd - L_fwd  # col-only set diff
            restore_size = len(restore_set)

            # Recall (col-only gold 와의 intersection, both 분모/numerator 일치)
            if gold_cols:
                gold_size = len(gold_cols)  # col-only-distinct count
                tp_fwd = len(L_fwd & gold_cols)
                tp_bwd = len(L_bwd & gold_cols)
                union_set = L_fwd | L_bwd
                tp_union = len(union_set & gold_cols)
                recall_fwd = tp_fwd / gold_size
                recall_bwd = tp_bwd / gold_size
                recall_union = tp_union / gold_size
                n_valid += 1
                sum_recall_fwd += recall_fwd
                sum_recall_bwd += recall_bwd
                sum_recall_union += recall_union
            else:
                gold_size = 0
                recall_fwd = recall_bwd = recall_union = 0.0

            sum_L_bwd_size += L_bwd_size
            sum_L_fwd_size += L_fwd_size
            sum_restore_size += restore_size

            f_out.write(json.dumps({
                "query_id": qid,
                "db_id": db_id,
                "L_bwd": sorted(L_bwd),  # col-only (bug fix 2026-05-13)
                "L_fwd": L_fwd_raw,  # raw anchor final_nodes (table.col 또는 table) 보존
                "L_bwd_size": L_bwd_size,
                "L_fwd_size": L_fwd_size,
                "intersection_size": intersection_size,
                "restore_size": restore_size,
                "gold_cols": sorted(gold_cols_raw),  # col-only
                "gold_size": gold_size,
                "recall_fwd": round(recall_fwd, 4),
                "recall_bwd": round(recall_bwd, 4),
                "recall_union": round(recall_union, 4),
                "truncated_full": truncated,
            }, ensure_ascii=False) + "\n")

    mean_fwd = sum_recall_fwd / n_valid if n_valid else 0.0
    mean_bwd = sum_recall_bwd / n_valid if n_valid else 0.0
    mean_union = sum_recall_union / n_valid if n_valid else 0.0
    delta_union = mean_union - mean_fwd

    summary = {
        "n_total": n_total,
        "n_valid_gold": n_valid,
        "n_truncated_full": n_truncated,
        "mean_recall_fwd": round(mean_fwd, 4),
        "mean_recall_bwd": round(mean_bwd, 4),
        "mean_recall_union": round(mean_union, 4),
        "mean_delta_union_vs_fwd": round(delta_union, 4),
        "mean_L_bwd_size": round(sum_L_bwd_size / n_total, 2) if n_total else 0,
        "mean_L_fwd_size": round(sum_L_fwd_size / n_total, 2) if n_total else 0,
        "mean_restore_size": round(sum_restore_size / n_total, 2) if n_total else 0,
        # Decision Rule (학술 Agent 5/13)
        "decision_backward_valid": delta_union >= 0.05,  # ≥ 0.05 → backward path 유효
    }
    logger.info(f"A-2 summary: {json.dumps(summary, indent=2)}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1_path", default=None,
                        help="A-1 결과 jsonl path (default: <output_dir>/A1_preliminary_sql_quality.jsonl)")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    a1_path = Path(args.a1_path) if args.a1_path else output_dir / "A1_preliminary_sql_quality.jsonl"
    if not a1_path.exists():
        raise FileNotFoundError(f"A-1 result not found: {a1_path}")

    output_path = output_dir / "A2_backward_recall.jsonl"
    summary = run_a2(a1_path, output_path)

    summary_path = output_dir / "A2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"A-2 summary written: {summary_path}")

    print("\n" + "=" * 60)
    print("A-2 Backward Recall (Direction A backward path 유효성)")
    print("=" * 60)
    print(f"Total: {summary['n_total']} (valid gold: {summary['n_valid_gold']}, truncated: {summary['n_truncated_full']})")
    print(f"mean(recall_fwd):   {summary['mean_recall_fwd']:.4f}  (anchor S_fwd 만)")
    print(f"mean(recall_bwd):   {summary['mean_recall_bwd']:.4f}  (preliminary SQL backward)")
    print(f"mean(recall_union): {summary['mean_recall_union']:.4f}  (forward ∪ backward)")
    print(f"mean Δ (union - fwd): {summary['mean_delta_union_vs_fwd']:+.4f}  (학술 Agent threshold ≥ 0.05)")
    print(f"mean |L_bwd|: {summary['mean_L_bwd_size']:.2f}, mean |L_fwd|: {summary['mean_L_fwd_size']:.2f}, mean |restore|: {summary['mean_restore_size']:.2f}")
    print(f"Decision: backward path 유효? {summary['decision_backward_valid']}")


if __name__ == "__main__":
    main()
