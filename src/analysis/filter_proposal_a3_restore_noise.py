"""Filter Proposal Phase 1 — A-3: Restore Candidate Noise Rate.

근거:
  - planning/DECISIONS.md 2026-05-13 (학술 Agent Phase 1 GO)
  - planning/filter_proposal_data_spec_2026-05-13.md §2 A-3

목적:
  S_restore = L_bwd \\ S_fwd 의 column 들의 precision (gold 비율) + recall_gained 측정.
  Direction A 의 핵심 정량 — SGBE 의 S_keep_hard noise 81.22% 와 직접 비교.

Decision Rule (학술 Agent):
  mean(S_restore_precision) ≥ 0.6 → Direction A 우선 배포 권장
  mean(S_restore_precision) < 0.4 → noise dominant, Direction B/C 우선 검토
  mean(recall_gained_by_restore) ≥ 0.05 → backward 의 net recall lift 의미 있음

A-2 결과 의존. LLM call 0.

🔧 Conditional denominator 정식화 (학술 Agent Phase 4 §2 응답):
  - n_S_restore_nonzero = 488 = mean(S_restore_precision) 의 분모 (S_restore_size > 0 인 query)
  - n_missed_nonzero = 537 = mean(recall_gained_by_restore) 의 분모 (forward 가 ≥1 gold 누락)
  - 두 conditional set 은 다름 (cross-tab: 둘 다 nonzero 370, S only 118, missed only 167, 둘 다 0 879)
  - Phase 1 buggy 의 n_restore_nonzero = 698 → Phase 2 fixed = 488 의 변화 원인:
    bug fix 가 (a) recall 계산 + (b) extract_columns_from_sql + (c) column_set_normalize_for_compare
    의 col-only 통일 까지 포함 → L_bwd 집합 자체가 alias-distinct → col-only-distinct 로 collapse
    (예: 't1.molecule_id' 과 'molecule.molecule_id' 가 같은 col 으로 합쳐짐).
    상세: notebooks/analysis_results/recall_gained_denominator_verification.md §1.4.

Output: outputs/analysis/filter_proposal/A3_restore_noise.jsonl (1534 records)
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Set, Optional, Any
from collections import defaultdict
import statistics as st

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.logger import get_logger
from analysis.filter_proposal_a2_backward_recall import (
    extract_columns_normalized,
    column_set_normalize_for_compare,
    _intersect_size,
)

logger = get_logger(__name__)

DEFAULT_OUTPUT_DIR = ROOT / "outputs/analysis/filter_proposal"


def run_a3(a2_path: Path, output_path: Path) -> Dict[str, Any]:
    """A-2 결과 → A-3 S_restore precision + recall_gained 계산."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    precision_vals = []  # only when S_restore_size > 0
    recall_gained_vals = []  # only when missed_by_fwd > 0
    n_total = 0
    n_truncated = 0
    n_restore_zero = 0
    n_restore_nonzero = 0
    sum_restore_size = 0
    sum_restore_gold_count = 0
    db_breakdown = defaultdict(lambda: {"n": 0, "precision_sum": 0.0, "precision_n": 0,
                                          "recall_gained_sum": 0.0, "recall_gained_n": 0})

    with open(a2_path) as f_in, open(output_path, "w") as f_out:
        for line in tqdm(f_in, desc="A-3 restore noise"):
            r = json.loads(line)
            qid = r["query_id"]
            db_id = r["db_id"]
            truncated = r.get("truncated_full", False)
            if truncated:
                n_truncated += 1
            n_total += 1

            L_bwd = set(r.get("L_bwd", []))
            L_fwd_raw = r.get("L_fwd", [])
            gold_cols_raw = r.get("gold_cols", [])

            # Normalize column sets
            L_fwd_norm = extract_columns_normalized(L_fwd_raw)  # table.col + col-only
            L_fwd_compare = column_set_normalize_for_compare(L_fwd_norm)
            gold_compare = column_set_normalize_for_compare(set(gold_cols_raw))

            # S_restore = L_bwd \ S_fwd (column-level set diff)
            S_restore = set(c for c in L_bwd if c.lower() not in L_fwd_compare)
            S_restore_size = len(S_restore)

            # S_restore ∩ gold (precision numerator)
            S_restore_gold_count = _intersect_size(S_restore, set(gold_cols_raw))

            S_restore_precision = (
                S_restore_gold_count / S_restore_size if S_restore_size > 0 else None
            )

            # recall_gained_by_restore: forward 에서 누락된 gold 중 restore 가 복원한 비율
            missed_by_fwd = set(c for c in gold_cols_raw
                                 if c.lower() not in L_fwd_compare)
            missed_size = len(missed_by_fwd)
            if missed_size > 0:
                gained = _intersect_size(S_restore, missed_by_fwd)
                recall_gained = gained / missed_size
            else:
                # forward 가 모든 gold 포함 → restore 의 net gain 없음
                recall_gained = 0.0

            f_out.write(json.dumps({
                "query_id": qid,
                "db_id": db_id,
                "S_restore": sorted(S_restore),
                "S_restore_size": S_restore_size,
                "S_restore_gold_count": S_restore_gold_count,
                "S_restore_precision": (
                    round(S_restore_precision, 4) if S_restore_precision is not None else None
                ),
                "missed_by_fwd_size": missed_size,
                "recall_gained_by_restore": round(recall_gained, 4),
                "truncated_full": truncated,
            }, ensure_ascii=False) + "\n")

            sum_restore_size += S_restore_size
            sum_restore_gold_count += S_restore_gold_count
            if S_restore_size > 0:
                precision_vals.append(S_restore_precision)
                n_restore_nonzero += 1
            else:
                n_restore_zero += 1
            if missed_size > 0:
                recall_gained_vals.append(recall_gained)

            db = db_breakdown[db_id]
            db["n"] += 1
            if S_restore_precision is not None:
                db["precision_sum"] += S_restore_precision
                db["precision_n"] += 1
            if missed_size > 0:
                db["recall_gained_sum"] += recall_gained
                db["recall_gained_n"] += 1

    mean_precision = st.mean(precision_vals) if precision_vals else 0.0
    median_precision = st.median(precision_vals) if precision_vals else 0.0
    mean_recall_gained = st.mean(recall_gained_vals) if recall_gained_vals else 0.0
    median_recall_gained = st.median(recall_gained_vals) if recall_gained_vals else 0.0
    pooled_precision = (sum_restore_gold_count / sum_restore_size) if sum_restore_size > 0 else 0.0

    per_db = {
        db: {
            "n": v["n"],
            "mean_precision": round(v["precision_sum"] / v["precision_n"], 4) if v["precision_n"] else None,
            "n_with_restore": v["precision_n"],
            "mean_recall_gained": round(v["recall_gained_sum"] / v["recall_gained_n"], 4) if v["recall_gained_n"] else None,
        } for db, v in db_breakdown.items()
    }

    summary = {
        "n_total": n_total,
        "n_truncated_full": n_truncated,
        "n_restore_nonzero": n_restore_nonzero,
        "n_restore_zero": n_restore_zero,
        "mean_S_restore_precision": round(mean_precision, 4),  # per-query mean (S_restore>0 only)
        "median_S_restore_precision": round(median_precision, 4),
        "pooled_S_restore_precision": round(pooled_precision, 4),  # sum 분모로 micro-avg
        "mean_recall_gained_by_restore": round(mean_recall_gained, 4),  # missed_by_fwd>0 only
        "median_recall_gained_by_restore": round(median_recall_gained, 4),
        "mean_restore_size_per_query": round(sum_restore_size / n_total, 2) if n_total else 0,
        # Decision Rules (학술 Agent 5/13)
        "decision_direction_a_priority": mean_precision >= 0.6,  # ≥ 0.6 → Direction A 우선 배포
        "decision_direction_a_post_paper": mean_precision < 0.4,  # < 0.4 → Direction B/C 우선
        "decision_backward_lift_meaningful": mean_recall_gained >= 0.05,  # ≥ 0.05 → net lift 의미
        "per_db_breakdown": per_db,
    }
    logger.info(f"A-3 summary: mean(precision)={mean_precision:.4f}, mean(recall_gained)={mean_recall_gained:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2_path", default=None)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    a2_path = Path(args.a2_path) if args.a2_path else output_dir / "A2_backward_recall.jsonl"
    if not a2_path.exists():
        raise FileNotFoundError(f"A-2 result not found: {a2_path}")

    output_path = output_dir / "A3_restore_noise.jsonl"
    summary = run_a3(a2_path, output_path)

    summary_path = output_dir / "A3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"A-3 summary written: {summary_path}")

    print("\n" + "=" * 60)
    print("A-3 Restore Candidate Noise Rate (Direction A 우선 배포 결정)")
    print("=" * 60)
    print(f"Total: {summary['n_total']} (truncated: {summary['n_truncated_full']})")
    print(f"Queries with non-zero S_restore: {summary['n_restore_nonzero']}")
    print(f"mean(S_restore_precision): {summary['mean_S_restore_precision']:.4f}  (학술 Agent threshold ≥ 0.6)")
    print(f"median(S_restore_precision): {summary['median_S_restore_precision']:.4f}")
    print(f"pooled S_restore_precision: {summary['pooled_S_restore_precision']:.4f}  (micro-avg)")
    print(f"mean(recall_gained_by_restore): {summary['mean_recall_gained_by_restore']:.4f}  (threshold ≥ 0.05)")
    print(f"mean |S_restore| per query: {summary['mean_restore_size_per_query']:.2f}")
    print()
    print(f"Decision — Direction A 우선 배포? {summary['decision_direction_a_priority']}")
    print(f"Decision — Direction A post-paper? {summary['decision_direction_a_post_paper']}")
    print(f"Decision — backward lift 의미 있음? {summary['decision_backward_lift_meaningful']}")


if __name__ == "__main__":
    main()
