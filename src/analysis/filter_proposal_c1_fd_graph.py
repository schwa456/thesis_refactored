"""Filter Proposal Phase 2 — C-1: FD Graph Completeness per DB.

근거:
  - planning/DECISIONS.md 2026-05-13 (Phase 1 PASS + Phase 2 GO + A-2 xlsx bug fix)
  - planning/filter_proposal_data_spec_2026-05-13.md §4 C-1
  - planning/filter_proposal_scholar_agent_response_phase2_2026-05-13.md

목적:
  BIRD-Dev + BIRD-Train 의 모든 DB 의 FK/PK 선언 현황 + gold SQL 의 join 에 필요한 FK coverage.
  Direction C (GRAST-SQL FD) 의 feasibility check — FK graph 가 sparse 하면 GRAST 효과 제한.

Decision Rule (학술 Agent):
  mean(fk_coverage_rate) ≥ 0.50 → Direction C feasible
  mean(fk_coverage_rate) < 0.30 → Direction C post-paper

LLM 무관, GPU 무관. sqlglot parsing only.

Output: outputs/analysis/filter_proposal/C1_fd_graph_completeness.csv (~100 rows)
"""
from __future__ import annotations

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict

import sqlglot
from sqlglot.expressions import Column, Table, Join, Condition, EQ
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_DIR = ROOT / "outputs/analysis/filter_proposal"
DEV_TABLES = ROOT / "data/raw/BIRD_dev/dev_tables.json"
DEV_QUERIES = ROOT / "data/raw/BIRD_dev/dev.json"
TRAIN_TABLES = Path("/SSL_NAS/peoples/khj/thesis/train/train_tables.json")
TRAIN_QUERIES = Path("/SSL_NAS/peoples/khj/thesis/train/train.json")


def _build_col_index(db_info: Dict[str, Any]) -> Dict[Tuple[str, str], int]:
    """(table_name_lower, col_name_lower) → col_idx (BIRD tables.json index)."""
    tables = db_info["table_names_original"]
    cols = db_info["column_names_original"]
    idx_map: Dict[Tuple[str, str], int] = {}
    for col_idx, (table_idx, col_name) in enumerate(cols):
        if table_idx < 0:
            continue
        tbl = tables[table_idx]
        idx_map[(tbl.lower(), col_name.lower())] = col_idx
    return idx_map


def _build_alias_resolver(parsed) -> Dict[str, str]:
    """SQL alias → 실제 table name (lowercase)."""
    alias_to_table: Dict[str, str] = {}
    for t in parsed.find_all(Table):
        tname = t.name
        if not tname:
            continue
        alias = t.alias_or_name
        alias_to_table[alias.lower()] = tname.lower()
        # 또한 실제 table name 자체도 매핑
        alias_to_table[tname.lower()] = tname.lower()
    return alias_to_table


def extract_required_fks_from_sql(sql: str, db_info: Dict[str, Any]) -> Set[Tuple[int, int]]:
    """Gold SQL 의 JOIN 절에서 사용된 FK pairs 를 (col_idx_1, col_idx_2) 형식으로 추출.

    매칭 방식: ON 절의 EQ 표현식 의 양쪽 Column → table alias resolve → col_idx 찾기 → 정렬된 tuple.
    """
    if not sql or not isinstance(sql, str):
        return set()
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return set()
    if parsed is None:
        return set()

    col_idx = _build_col_index(db_info)
    alias_to_table = _build_alias_resolver(parsed)

    fk_pairs: Set[Tuple[int, int]] = set()

    # 각 JOIN 의 ON 절에서 EQ 표현식 찾기
    for join in parsed.find_all(Join):
        on_expr = join.args.get("on")
        if on_expr is None:
            continue
        # ON 절 내부의 모든 EQ
        for eq in on_expr.find_all(EQ):
            left = eq.this
            right = eq.expression
            if not isinstance(left, Column) or not isinstance(right, Column):
                continue
            ltbl_alias = (left.table or "").lower()
            rtbl_alias = (right.table or "").lower()
            ltbl = alias_to_table.get(ltbl_alias, ltbl_alias)
            rtbl = alias_to_table.get(rtbl_alias, rtbl_alias)
            lcol = (left.name or "").lower()
            rcol = (right.name or "").lower()
            li = col_idx.get((ltbl, lcol))
            ri = col_idx.get((rtbl, rcol))
            if li is None or ri is None:
                continue
            # Normalize as sorted tuple (FK direction-agnostic)
            pair = tuple(sorted([li, ri]))
            fk_pairs.add(pair)

    # WHERE 절의 implicit join (FROM t1, t2 WHERE t1.x = t2.y) 도 chase
    where = parsed.args.get("where")
    if where is not None:
        for eq in where.find_all(EQ):
            left = eq.this
            right = eq.expression
            if not isinstance(left, Column) or not isinstance(right, Column):
                continue
            ltbl_alias = (left.table or "").lower()
            rtbl_alias = (right.table or "").lower()
            ltbl = alias_to_table.get(ltbl_alias, ltbl_alias)
            rtbl = alias_to_table.get(rtbl_alias, rtbl_alias)
            # WHERE 절에서는 다른 table 끼리만 join 으로 간주
            if ltbl == rtbl or not ltbl or not rtbl:
                continue
            lcol = (left.name or "").lower()
            rcol = (right.name or "").lower()
            li = col_idx.get((ltbl, lcol))
            ri = col_idx.get((rtbl, rcol))
            if li is None or ri is None:
                continue
            pair = tuple(sorted([li, ri]))
            fk_pairs.add(pair)

    return fk_pairs


def normalize_declared_fk(foreign_keys: List[List[int]]) -> Set[Tuple[int, int]]:
    """BIRD tables.json 의 foreign_keys = [[src_col_idx, dst_col_idx], ...] → sorted tuple set."""
    return set(tuple(sorted(p)) for p in foreign_keys)


def run_c1(dev_tables_path: Path, train_tables_path: Path,
           dev_queries_path: Path, train_queries_path: Path,
           output_path: Path,
           include_train: bool = True) -> Dict[str, Any]:
    """BIRD 모든 DB 에 대해 FK coverage 측정.

    coverage = |required FK ∩ declared FK| / |required FK| (분모 0 시 1.0).
    """
    # DB schema 로드
    db_infos: Dict[str, Dict[str, Any]] = {}
    with open(dev_tables_path) as f:
        for d in json.load(f):
            db_infos[d["db_id"]] = d
    if include_train and train_tables_path.exists():
        with open(train_tables_path) as f:
            for d in json.load(f):
                db_infos.setdefault(d["db_id"], d)  # dev 우선
    logger.info(f"Loaded {len(db_infos)} DBs (dev + train)")

    # Queries 로드
    db_queries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(dev_queries_path) as f:
        for q in json.load(f):
            db_queries[q["db_id"]].append(q)
    if include_train and train_queries_path.exists():
        with open(train_queries_path) as f:
            for q in json.load(f):
                db_queries[q["db_id"]].append(q)
    logger.info(f"Loaded queries for {len(db_queries)} DBs")

    # CSV writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fk_cov_rates = []
    rows = []
    for db_id, db_info in tqdm(db_infos.items(), desc="C-1 FD graph"):
        tables = db_info["table_names_original"]
        cols = db_info["column_names_original"]
        foreign_keys = db_info.get("foreign_keys", []) or []
        primary_keys = db_info.get("primary_keys", []) or []

        num_tables = len(tables)
        num_columns = len([c for c in cols if c[0] >= 0])
        num_declared_fk = len(foreign_keys)
        num_declared_pk = len(primary_keys)

        declared_fk_set = normalize_declared_fk(foreign_keys)

        # required FK = gold SQL 의 join 들의 union
        required_fk_set: Set[Tuple[int, int]] = set()
        for q in db_queries.get(db_id, []):
            try:
                req = extract_required_fks_from_sql(q.get("SQL", ""), db_info)
                required_fk_set |= req
            except Exception as e:
                logger.debug(f"  qid={q.get('question_id')} skip: {e}")
                continue

        covered_fk_set = required_fk_set & declared_fk_set
        if required_fk_set:
            coverage = len(covered_fk_set) / len(required_fk_set)
        else:
            coverage = 1.0  # No required FK → trivially 100% covered
        fk_cov_rates.append(coverage)
        rows.append({
            "db_id": db_id,
            "num_tables": num_tables,
            "num_columns": num_columns,
            "num_declared_fk": num_declared_fk,
            "num_declared_pk": num_declared_pk,
            "required_fk_count": len(required_fk_set),
            "covered_fk_count": len(covered_fk_set),
            "fk_coverage_rate": round(coverage, 4),
            "inferred_fk_count": 0,  # post-paper GPT 예측 시 update
        })

    # Sort by coverage asc (worst first)
    rows.sort(key=lambda r: r["fk_coverage_rate"])
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    import statistics as st
    mean_cov = st.mean(fk_cov_rates) if fk_cov_rates else 0.0
    median_cov = st.median(fk_cov_rates) if fk_cov_rates else 0.0
    n_below_50 = sum(1 for c in fk_cov_rates if c < 0.50)
    n_below_30 = sum(1 for c in fk_cov_rates if c < 0.30)

    summary = {
        "n_dbs": len(rows),
        "n_dev_dbs": len(db_infos),  # all (dev + train if include_train)
        "mean_fk_coverage_rate": round(mean_cov, 4),
        "median_fk_coverage_rate": round(median_cov, 4),
        "n_dbs_below_0.50": n_below_50,
        "n_dbs_below_0.30": n_below_30,
        "n_dbs_at_1.00": sum(1 for c in fk_cov_rates if c >= 0.9999),
        # Decision Rules (학술 Agent)
        "decision_direction_c_feasible": mean_cov >= 0.50,
        "decision_direction_c_post_paper": mean_cov < 0.30,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dev_only", action="store_true",
                        help="train_tables.json 미사용 (BIRD-Dev 만)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_path = output_dir / "C1_fd_graph_completeness.csv"

    summary = run_c1(
        dev_tables_path=DEV_TABLES,
        train_tables_path=TRAIN_TABLES,
        dev_queries_path=DEV_QUERIES,
        train_queries_path=TRAIN_QUERIES,
        output_path=output_path,
        include_train=not args.dev_only,
    )

    summary_path = output_dir / "C1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"C-1 summary: {json.dumps(summary, indent=2)}")

    print("\n" + "=" * 60)
    print("C-1 FD Graph Completeness")
    print("=" * 60)
    print(f"DBs analyzed: {summary['n_dbs']}")
    print(f"mean(fk_coverage_rate): {summary['mean_fk_coverage_rate']:.4f}  (학술 Agent threshold ≥ 0.50)")
    print(f"median: {summary['median_fk_coverage_rate']:.4f}")
    print(f"DBs with coverage < 0.50: {summary['n_dbs_below_0.50']}")
    print(f"DBs with coverage < 0.30: {summary['n_dbs_below_0.30']}")
    print(f"DBs with coverage = 1.00: {summary['n_dbs_at_1.00']}")
    print()
    print(f"Decision — Direction C feasible? {summary['decision_direction_c_feasible']}")
    print(f"Decision — Direction C post-paper? {summary['decision_direction_c_post_paper']}")


if __name__ == "__main__":
    main()
