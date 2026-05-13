"""Smoke test for filter_proposal_phase2 (A-2 bug fix + C-1 + C-2).

근거: planning/DECISIONS.md 2026-05-13 (Phase 1 PASS + Phase 2 GO + A-2 xlsx bug fix).
재현: PYTHONPATH=src conda run -n base python -m pytest src/analysis/tests/test_filter_proposal_phase2.py -v
"""
from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.filter_proposal_a2_backward_recall import (
    extract_columns_from_sql,
    extract_columns_normalized,
    column_set_normalize_for_compare,
    _intersect_size,
    run_a2,
)
from analysis.filter_proposal_a3_restore_noise import run_a3
from analysis.filter_proposal_c1_fd_graph import (
    extract_required_fks_from_sql,
    normalize_declared_fk,
    run_c1,
)
from analysis.filter_proposal_c2_structural_miss import (
    _normalize_final_nodes,
    _full_table_col_set,
    run_c2,
)


SAMPLE_DB = {
    "db_id": "sample_db",
    "table_names_original": ["customer", "orders"],
    "column_names_original": [
        [-1, "*"],
        [0, "id"], [0, "name"], [0, "email"],         # idx 1-3
        [1, "order_id"], [1, "customer_id"], [1, "total"],  # idx 4-6
    ],
    "primary_keys": [1, 4],
    "foreign_keys": [[5, 1]],  # orders.customer_id → customer.id
}


# ──────────────────────────────────────────────────────────────
# A-2 bug fix verification — recall ≤ 1.0 with SQL alias
# ──────────────────────────────────────────────────────────────

class TestA2BugFix(unittest.TestCase):
    def test_col_only_extraction(self):
        # Aliased SQL — sqlglot extracts 't1.atom_id' style
        sql = "SELECT t1.atom_id, t1.molecule_id, t2.label FROM atom t1 JOIN molecule t2 ON t1.molecule_id = t2.molecule_id"
        cols = extract_columns_from_sql(sql)
        # 🔧 fix: col-only only (no table prefix)
        self.assertEqual(cols, {"atom_id", "molecule_id", "label"})
        # 이전 bug: would have included 't1.atom_id', 't1.molecule_id', 't2.label', 't2.molecule_id' (alias-distinct)

    def test_extract_columns_normalized_table_only_excluded(self):
        # table 단독 entry 는 제외
        nodes = ["customer.id", "customer.name", "customer"]
        norm = extract_columns_normalized(nodes)
        self.assertEqual(norm, {"id", "name"})

    def test_recall_calc_with_alias_no_overflow(self):
        """Aliased gold + L_bwd with col-only → recall ≤ 1.0 (bug fix verification)."""
        gold = extract_columns_from_sql(
            "SELECT t1.atom_id, t1.molecule_id, t2.label FROM atom t1 JOIN molecule t2 ON t1.molecule_id = t2.molecule_id"
        )
        L_bwd = extract_columns_from_sql(
            "SELECT t1.atom_id, t1.molecule_id, t2.label FROM atom t1 JOIN molecule t2 ON t1.molecule_id = t2.molecule_id"
        )
        # 두 set 모두 {"atom_id", "molecule_id", "label"} → intersection 3, gold_size 3 → recall = 1.0
        self.assertEqual(gold, L_bwd)
        self.assertEqual(len(gold), 3)
        recall_bwd = len(L_bwd & gold) / len(gold)
        self.assertEqual(recall_bwd, 1.0)  # 이전 bug 에서는 1.75

    def test_a2_run_no_recall_overflow(self):
        """E2E: run_a2 의 output 의 recall 모두 ≤ 1.0 보장."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            a1 = tmpdir / "a1.jsonl"
            with open(a1, "w") as f:
                # qid=242 시뮬레이션 — aliased SQL 모두 일치
                f.write(json.dumps({
                    "query_id": 242,
                    "db_id": "toxicology",
                    "gold_sql": "SELECT t1.atom_id, t1.molecule_id, t2.label FROM atom t1 JOIN molecule t2 ON t1.molecule_id = t2.molecule_id",
                    "prelim_sql_full": "SELECT t1.atom_id, t1.molecule_id, t2.label FROM atom t1 JOIN molecule t2 ON t1.molecule_id = t2.molecule_id",
                    "S_fwd_from_anchor": ["molecule.molecule_id"],
                    "truncated_full": False,
                }) + "\n")
            out = tmpdir / "a2.jsonl"
            summary = run_a2(a1, out)
            rec = json.loads(open(out).readline())
            # 본 bug fix: recall ≤ 1.0
            self.assertLessEqual(rec["recall_fwd"], 1.0)
            self.assertLessEqual(rec["recall_bwd"], 1.0)
            self.assertLessEqual(rec["recall_union"], 1.0)
            # gold (col-only) = {atom_id, molecule_id, label}, gold_size = 3
            self.assertEqual(rec["gold_size"], 3)
            # L_bwd (col-only) = {atom_id, molecule_id, label}
            self.assertEqual(set(rec["L_bwd"]), {"atom_id", "molecule_id", "label"})
            # L_fwd col-only = {molecule_id}
            # → recall_fwd = 1/3 ≈ 0.3333, recall_bwd = 3/3 = 1.0, recall_union = 3/3 = 1.0
            self.assertAlmostEqual(rec["recall_fwd"], 0.3333, places=4)
            self.assertAlmostEqual(rec["recall_bwd"], 1.0)
            self.assertAlmostEqual(rec["recall_union"], 1.0)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_a3_precision_bounded(self):
        """A-3 의 S_restore_precision ≤ 1.0 보장 (bug fix downstream)."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            a2 = tmpdir / "a2.jsonl"
            with open(a2, "w") as f:
                f.write(json.dumps({
                    "query_id": 0,
                    "db_id": "sample_db",
                    "L_bwd": ["id", "name", "email"],
                    "L_fwd": ["customer.id"],
                    "gold_cols": ["id", "name"],
                    "truncated_full": False,
                }) + "\n")
            out = tmpdir / "a3.jsonl"
            run_a3(a2, out)
            rec = json.loads(open(out).readline())
            # S_restore = L_bwd - L_fwd_normalized = {id, name, email} - {id} = {name, email}
            # gold ∩ S_restore = {name} → precision = 1/2 = 0.5
            self.assertLessEqual(rec["S_restore_precision"], 1.0)
            self.assertAlmostEqual(rec["S_restore_precision"], 0.5)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ──────────────────────────────────────────────────────────────
# C-1 FD graph
# ──────────────────────────────────────────────────────────────

class TestC1FdGraph(unittest.TestCase):
    def test_extract_required_fks_basic_join(self):
        sql = "SELECT c.name FROM customer c JOIN orders o ON c.id = o.customer_id"
        fks = extract_required_fks_from_sql(sql, SAMPLE_DB)
        # customer.id = idx 1, orders.customer_id = idx 5 → sorted (1, 5)
        self.assertEqual(fks, {(1, 5)})

    def test_extract_required_fks_implicit_join(self):
        # WHERE clause implicit join
        sql = "SELECT c.name FROM customer c, orders o WHERE c.id = o.customer_id"
        fks = extract_required_fks_from_sql(sql, SAMPLE_DB)
        self.assertEqual(fks, {(1, 5)})

    def test_extract_required_fks_no_join(self):
        sql = "SELECT name FROM customer"
        fks = extract_required_fks_from_sql(sql, SAMPLE_DB)
        self.assertEqual(fks, set())

    def test_declared_fk_normalization(self):
        decl = [[5, 1], [3, 4]]  # mixed order
        norm = normalize_declared_fk(decl)
        self.assertEqual(norm, {(1, 5), (3, 4)})

    def test_c1_e2e_small(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            tables_path = tmpdir / "tables.json"
            with open(tables_path, "w") as f:
                json.dump([SAMPLE_DB], f)
            queries_path = tmpdir / "queries.json"
            with open(queries_path, "w") as f:
                json.dump([
                    {"db_id": "sample_db", "SQL": "SELECT c.name FROM customer c JOIN orders o ON c.id = o.customer_id"},
                    {"db_id": "sample_db", "SQL": "SELECT name FROM customer"},
                ], f)
            out = tmpdir / "c1.csv"
            summary = run_c1(tables_path, Path("/no/such/path"), queries_path, Path("/no/such/path"),
                              out, include_train=False)
            self.assertEqual(summary["n_dbs"], 1)
            # required_fk = {(1,5)}, declared_fk = {(1,5)} → coverage = 1.0
            self.assertAlmostEqual(summary["mean_fk_coverage_rate"], 1.0)
            self.assertTrue(summary["decision_direction_c_feasible"])
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ──────────────────────────────────────────────────────────────
# C-2 Structural miss
# ──────────────────────────────────────────────────────────────

class TestC2StructuralMiss(unittest.TestCase):
    def test_normalize_final_nodes_col_only(self):
        nodes = ["customer.id", "customer.name", "orders"]  # last is table-only
        norm = _normalize_final_nodes(nodes)
        self.assertEqual(norm, {"id", "name"})

    def test_full_table_col_set(self):
        nodes = ["customer.id", "orders.customer_id"]
        full = _full_table_col_set(nodes)
        self.assertEqual(full, {("customer", "id"), ("orders", "customer_id")})

    def test_c2_e2e_complete_join(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            tables_path = tmpdir / "tables.json"
            with open(tables_path, "w") as f:
                json.dump([SAMPLE_DB], f)
            queries_path = tmpdir / "queries.json"
            with open(queries_path, "w") as f:
                json.dump([
                    {"db_id": "sample_db", "SQL": "SELECT c.name FROM customer c JOIN orders o ON c.id = o.customer_id"},
                    {"db_id": "sample_db", "SQL": "SELECT name FROM customer"},
                ], f)
            anchor_path = tmpdir / "anchor.jsonl"
            with open(anchor_path, "w") as f:
                # Q0: anchor 가 join col 보존 (customer.id + orders.customer_id)
                f.write(json.dumps({"question_id": 0, "final_nodes": ["customer.id", "customer.name", "orders.customer_id"]}) + "\n")
                # Q1: single-table query
                f.write(json.dumps({"question_id": 1, "final_nodes": ["customer.name"]}) + "\n")
            out = tmpdir / "c2.jsonl"
            summary = run_c2(tables_path, queries_path, anchor_path, out)
            self.assertEqual(summary["n_total"], 2)
            self.assertEqual(summary["n_with_join"], 1)
            self.assertEqual(summary["n_complete"], 1)
            self.assertAlmostEqual(summary["mean_is_join_complete_with_join"], 1.0)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_c2_missing_join(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            tables_path = tmpdir / "tables.json"
            with open(tables_path, "w") as f:
                json.dump([SAMPLE_DB], f)
            queries_path = tmpdir / "queries.json"
            with open(queries_path, "w") as f:
                json.dump([
                    {"db_id": "sample_db", "SQL": "SELECT c.name FROM customer c JOIN orders o ON c.id = o.customer_id"},
                ], f)
            anchor_path = tmpdir / "anchor.jsonl"
            with open(anchor_path, "w") as f:
                # anchor 가 join col 모두 누락
                f.write(json.dumps({"question_id": 0, "final_nodes": ["customer.name"]}) + "\n")
            out = tmpdir / "c2.jsonl"
            summary = run_c2(tables_path, queries_path, anchor_path, out)
            self.assertEqual(summary["n_with_join"], 1)
            self.assertEqual(summary["n_complete"], 0)
            self.assertAlmostEqual(summary["mean_is_join_complete_with_join"], 0.0)
            # → Direction C priority up (< 0.80)
            self.assertTrue(summary["decision_direction_c_priority_up"])
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
