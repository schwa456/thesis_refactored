"""Filter Proposal Phase 2 — C-2: Structural Miss Rate per Query.

근거:
  - planning/DECISIONS.md 2026-05-13 (Phase 1 PASS + Phase 2 GO)
  - planning/filter_proposal_data_spec_2026-05-13.md §4 C-2
  - planning/filter_proposal_scholar_agent_response_phase2_2026-05-13.md

목적:
  Anchor (XiYan filter) 의 final_nodes 가 gold SQL 의 JOIN 절 column 을 보존하는지 측정.
  Direction C (Steiner tree based recovery) 의 expected gain 정량 — join col 누락이 클수록 Steiner gain 큼.

Decision Rule (학술 Agent):
  mean(is_join_complete) < 0.80 → Direction C 우선순위 상향 (Steiner tree expected gain 큼)
  mean(is_join_complete) ≥ 0.95 → Direction C post-paper (anchor 가 이미 join col 잘 보존)

LLM 무관, GPU 무관.

Output: outputs/analysis/filter_proposal/C2_structural_miss.jsonl (1534 records)
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.logger import get_logger
from analysis.filter_proposal_c1_fd_graph import (
    extract_required_fks_from_sql,
    _build_col_index,
)

logger = get_logger(__name__)

DEFAULT_OUTPUT_DIR = ROOT / "outputs/analysis/filter_proposal"
DEV_TABLES = ROOT / "data/raw/BIRD_dev/dev_tables.json"
DEV_QUERIES = ROOT / "data/raw/BIRD_dev/dev.json"

# Anchor S_fwd 출처 — Filter Sweep v2 C0 (evidence-aware) 권장
DEFAULT_ANCHOR_PRED = ROOT / "outputs/experiments/s04_ablation/pipeline/filter_sweep/c0_xiyan_glm_sql/predictions.jsonl"


def _idx_to_table_col(db_info: Dict[str, Any], col_idx: int) -> Tuple[str, str]:
    """col_idx → (table_name, col_name)."""
    cols = db_info["column_names_original"]
    tables = db_info["table_names_original"]
    table_idx, col_name = cols[col_idx]
    return tables[table_idx], col_name


def _normalize_final_nodes(final_nodes: List[str]) -> Set[str]:
    """Anchor final_nodes ('table.col' or 'table') → col-only set (lowercase).
    table 단독 entry 는 제외 (column-level 비교 목적)."""
    out = set()
    for n in final_nodes or []:
        if not isinstance(n, str):
            continue
        if "." in n:
            col = n.split(".", 1)[1].lower()
            out.add(col)
    return out


def _full_table_col_set(final_nodes: List[str]) -> Set[Tuple[str, str]]:
    """Anchor final_nodes → set of (table_lower, col_lower) pairs.
    더 엄격한 매칭 (table 까지 같아야 매칭 인정)."""
    out = set()
    for n in final_nodes or []:
        if not isinstance(n, str):
            continue
        if "." in n:
            tbl, col = n.split(".", 1)
            out.add((tbl.lower(), col.lower()))
    return out


def run_c2(dev_tables_path: Path, dev_queries_path: Path,
            anchor_pred_path: Path, output_path: Path) -> Dict[str, Any]:
    """Per-query JOIN col 보존 여부 측정."""
    # DB schema
    with open(dev_tables_path) as f:
        db_infos = {d["db_id"]: d for d in json.load(f)}
    with open(dev_queries_path) as f:
        dev_queries = json.load(f)

    # Anchor predictions
    anchor_pred: Dict[int, List[str]] = {}
    with open(anchor_pred_path) as f:
        for line in f:
            d = json.loads(line)
            qid = d.get("question_id", d.get("qid"))
            anchor_pred[qid] = d.get("final_nodes", []) or []
    logger.info(f"Anchor predictions: {len(anchor_pred)}, dev queries: {len(dev_queries)}")

    # Per-query analysis
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = n_complete = n_required_zero = 0
    sum_missing = 0
    sum_required = 0
    per_db = defaultdict(lambda: {"n": 0, "complete": 0, "req_zero": 0, "sum_missing": 0})
    with open(output_path, "w") as f_out:
        for qid, q in enumerate(tqdm(dev_queries, desc="C-2 structural miss")):
            db_id = q["db_id"]
            db_info = db_infos.get(db_id)
            if db_info is None:
                continue
            gold_sql = q.get("SQL", "")
            required_fk_pairs = extract_required_fks_from_sql(gold_sql, db_info)

            # required join cols = required FK pair 의 양쪽 column index
            required_join_cols: Set[Tuple[str, str]] = set()  # (table_lower, col_lower)
            for li, ri in required_fk_pairs:
                t1, c1 = _idx_to_table_col(db_info, li)
                t2, c2 = _idx_to_table_col(db_info, ri)
                required_join_cols.add((t1.lower(), c1.lower()))
                required_join_cols.add((t2.lower(), c2.lower()))

            # Anchor final_nodes — 두 가지 매칭 (full table.col 또는 col-only)
            anchor_nodes = anchor_pred.get(qid, [])
            anchor_full = _full_table_col_set(anchor_nodes)
            anchor_col_only = _normalize_final_nodes(anchor_nodes)

            # Missing: full table.col 매칭 우선, 그게 없으면 col-only fallback
            missing = []
            for (tbl, col) in required_join_cols:
                if (tbl, col) in anchor_full:
                    continue
                if col in anchor_col_only:
                    continue  # col-only fallback (alias 호환)
                missing.append(f"{tbl}.{col}")
            is_complete = len(missing) == 0

            f_out.write(json.dumps({
                "query_id": qid,
                "db_id": db_id,
                "required_join_cols": sorted(f"{t}.{c}" for t, c in required_join_cols),
                "required_join_col_count": len(required_join_cols),
                "missing_join_cols": sorted(missing),
                "missing_join_col_count": len(missing),
                "is_join_complete": is_complete,
            }, ensure_ascii=False) + "\n")

            n_total += 1
            if not required_join_cols:
                n_required_zero += 1
            else:
                sum_required += len(required_join_cols)
                sum_missing += len(missing)
                if is_complete:
                    n_complete += 1
            d_agg = per_db[db_id]
            d_agg["n"] += 1
            if not required_join_cols:
                d_agg["req_zero"] += 1
            elif is_complete:
                d_agg["complete"] += 1
            d_agg["sum_missing"] += len(missing)

    # required join cols 가 0 인 query (single-table) 는 trivially complete
    # 학술 Agent decision rule 은 'is_join_complete' (전체 mean) 기준
    n_with_join = n_total - n_required_zero
    rate_complete_all = (n_complete + n_required_zero) / n_total if n_total else 0.0
    rate_complete_with_join = n_complete / n_with_join if n_with_join else 0.0

    per_db_summary = {
        db: {
            "n": v["n"],
            "n_with_join": v["n"] - v["req_zero"],
            "rate_complete_with_join": (v["complete"] / (v["n"] - v["req_zero"]))
                if (v["n"] - v["req_zero"]) > 0 else None,
            "mean_missing": (v["sum_missing"] / max(1, (v["n"] - v["req_zero"]))),
        }
        for db, v in per_db.items()
    }

    summary = {
        "n_total": n_total,
        "n_required_zero": n_required_zero,  # single-table query 또는 join 없음
        "n_with_join": n_with_join,
        "n_complete": n_complete,
        "mean_is_join_complete_all": round(rate_complete_all, 4),
        "mean_is_join_complete_with_join": round(rate_complete_with_join, 4),
        "mean_missing_join_cols_per_query": round(sum_missing / max(1, n_with_join), 4),
        "mean_required_join_cols_per_query": round(sum_required / max(1, n_with_join), 4),
        # Decision Rules (학술 Agent) — multi-table queries 기준
        "decision_direction_c_priority_up": rate_complete_with_join < 0.80,
        "decision_direction_c_post_paper": rate_complete_with_join >= 0.95,
        "per_db_breakdown": per_db_summary,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor_pred", default=str(DEFAULT_ANCHOR_PRED))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_path = output_dir / "C2_structural_miss.jsonl"

    summary = run_c2(
        dev_tables_path=DEV_TABLES,
        dev_queries_path=DEV_QUERIES,
        anchor_pred_path=Path(args.anchor_pred),
        output_path=output_path,
    )

    summary_path = output_dir / "C2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"C-2 summary: {json.dumps(summary, indent=2)}")

    print("\n" + "=" * 60)
    print("C-2 Structural Miss Rate per Query")
    print("=" * 60)
    print(f"Total queries: {summary['n_total']}")
    print(f"Queries with required join (multi-table): {summary['n_with_join']}")
    print(f"Queries with no join (single-table): {summary['n_required_zero']}")
    print()
    print(f"mean(is_join_complete) [all queries]: {summary['mean_is_join_complete_all']:.4f}")
    print(f"mean(is_join_complete) [multi-table only]: {summary['mean_is_join_complete_with_join']:.4f}  "
          f"(학술 Agent threshold)")
    print(f"mean(missing_join_cols/query): {summary['mean_missing_join_cols_per_query']:.4f}")
    print()
    print(f"Decision — Direction C priority up? (< 0.80) {summary['decision_direction_c_priority_up']}")
    print(f"Decision — Direction C post-paper? (≥ 0.95) {summary['decision_direction_c_post_paper']}")


if __name__ == "__main__":
    main()
