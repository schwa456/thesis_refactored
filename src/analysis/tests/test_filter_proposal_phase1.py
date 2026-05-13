"""Smoke test for filter_proposal_phase1 (A-1/A-2/A-3).

목적:
  - A-1/A-2/A-3 의 helper 함수 unit test (sqlglot parsing, column normalization,
    set difference, recall calculation, truncation guard)
  - LLM API call 은 mock — 실제 GLM API 호출 없음
  - End-to-end small batch (3 query) chain forward 검증 (smoke)

근거: planning/DECISIONS.md 2026-05-13 Phase 1 Step 2 (smoke test before launch).
재현: PYTHONPATH=src conda run -n base python -m pytest src/analysis/tests/test_filter_proposal_phase1.py -v
"""
from __future__ import annotations

import os
import sys
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.filter_proposal_a1_preliminary_sql import (
    count_tokens, build_schema_from_nodes, load_anchor_final_nodes,
    GLM_4_7_TOKEN_LIMIT, _get_tokenizer, run_a1,
)
from analysis.filter_proposal_a2_backward_recall import (
    extract_columns_from_sql, extract_columns_normalized,
    column_set_normalize_for_compare, _intersect_size, run_a2,
)
from analysis.filter_proposal_a3_restore_noise import run_a3


# ──────────────────────────────────────────────────────────────
# Test fixtures
# ──────────────────────────────────────────────────────────────

SAMPLE_DB_SCHEMA_MAP = {
    "sample_db": {
        "db_id": "sample_db",
        "table_names_original": ["customer", "orders"],
        "column_names_original": [
            [-1, "*"],
            [0, "id"], [0, "name"], [0, "email"],
            [1, "order_id"], [1, "customer_id"], [1, "total"],
        ],
    },
}

SAMPLE_DEV = [
    {"db_id": "sample_db", "question": "How many customers?", "evidence": "",
     "SQL": "SELECT COUNT(*) FROM customer"},
    {"db_id": "sample_db", "question": "Customer emails", "evidence": "",
     "SQL": "SELECT email FROM customer"},
    {"db_id": "sample_db", "question": "Total spent", "evidence": "",
     "SQL": "SELECT SUM(total) FROM orders WHERE customer_id = 1"},
]


def _build_temp_sqlite(tmpdir: Path) -> Path:
    db_root = tmpdir / "sample_db"
    db_root.mkdir(parents=True, exist_ok=True)
    db_path = db_root / "sample_db.sqlite"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    c.execute("CREATE TABLE customer (id INTEGER, name TEXT, email TEXT)")
    c.execute("CREATE TABLE orders (order_id INTEGER, customer_id INTEGER, total REAL)")
    c.execute("INSERT INTO customer VALUES (1, 'A', 'a@x.com'), (2, 'B', 'b@x.com')")
    c.execute("INSERT INTO orders VALUES (1, 1, 100.0), (2, 1, 50.0), (3, 2, 75.0)")
    conn.commit()
    conn.close()
    return tmpdir


# ──────────────────────────────────────────────────────────────
# A-1 helpers
# ──────────────────────────────────────────────────────────────

class TestA1Helpers(unittest.TestCase):
    def test_count_tokens_with_tiktoken(self):
        tk = _get_tokenizer()
        msgs = [{"role": "user", "content": "hello world"}]
        n = count_tokens(msgs, tk)
        self.assertGreater(n, 0)
        self.assertLess(n, 10)

    def test_count_tokens_heuristic_fallback(self):
        # Force fallback (no tokenizer)
        msgs = [{"role": "user", "content": "x" * 30}]
        n = count_tokens(msgs, None)
        self.assertEqual(n, 10)  # 30 / 3

    def test_build_schema_from_nodes(self):
        nodes = ["customer.id", "customer.name", "orders.total"]
        s = build_schema_from_nodes(nodes)
        self.assertIn("customer.id", s)
        self.assertEqual(s.count("\n") + 1, 3)

    def test_truncation_threshold(self):
        # Use mixed content (more realistic) — token/char ratio ~1:3-4 for English+special chars
        # GLM token limit 128K → need ~512K chars of mixed text to exceed
        huge_msgs = [{"role": "user",
                       "content": ("schema column with descriptions and metadata " * 25000)}]
        tk = _get_tokenizer()
        n = count_tokens(huge_msgs, tk)
        self.assertGreater(n, GLM_4_7_TOKEN_LIMIT)


# ──────────────────────────────────────────────────────────────
# A-2 helpers (sqlglot parsing)
# ──────────────────────────────────────────────────────────────

class TestA2Helpers(unittest.TestCase):
    def test_extract_columns_basic(self):
        sql = "SELECT id, name FROM customer WHERE email = 'x'"
        cols = extract_columns_from_sql(sql)
        self.assertIn("id", cols)
        self.assertIn("name", cols)
        self.assertIn("email", cols)

    def test_extract_columns_with_table(self):
        sql = "SELECT c.id, c.name FROM customer c JOIN orders o ON c.id = o.customer_id"
        cols = extract_columns_from_sql(sql)
        # alias 'c' may resolve to 'c' or 'customer' depending on sqlglot
        names = {c.split(".")[-1] for c in cols}
        self.assertIn("id", names)
        self.assertIn("name", names)
        self.assertIn("customer_id", names)

    def test_extract_columns_parse_error_returns_empty(self):
        # sqlglot 의 robust parser 는 "THIS IS NOT SQL" 같은 단순 문장도 identifier 로 해석할 수 있음.
        # 진짜 broken syntax (unmatched brackets, invalid tokens 조합) 에서만 빈 set.
        self.assertEqual(extract_columns_from_sql("(((((SELECT FROM"), set())
        self.assertEqual(extract_columns_from_sql(None), set())
        self.assertEqual(extract_columns_from_sql(""), set())

    def test_extract_columns_normalized(self):
        # 🔧 2026-05-13 bug fix: col-only normalization (no 'table.col')
        nodes = ["customer.id", "customer.name", "customer"]  # table-only entry
        norm = extract_columns_normalized(nodes)
        # col-only 만 추출
        self.assertEqual(norm, {"id", "name"})
        # Table-only entry 는 제외
        self.assertNotIn("customer", norm)

    def test_intersect_size_with_col_fallback(self):
        # 'table.col' vs 'col' 매칭 — col-only fallback 인정
        a = {"customer.id", "customer.name"}
        b = {"id"}
        self.assertEqual(_intersect_size(a, b), 1)


# ──────────────────────────────────────────────────────────────
# A-2 E2E (real sqlglot, no LLM)
# ──────────────────────────────────────────────────────────────

class TestA2RunE2E(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_a1_jsonl(self, path: Path, records):
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_a2_basic_recall_calc(self):
        # 3 query × A-1 결과 mock
        a1 = self.tmpdir / "a1.jsonl"
        self._write_a1_jsonl(a1, [
            {
                "query_id": 0, "db_id": "sample_db",
                "gold_sql": "SELECT id, name FROM customer",
                "prelim_sql_full": "SELECT id, name, email FROM customer",
                "S_fwd_from_anchor": ["customer.id", "customer.email"],
                "truncated_full": False,
            },
            {
                "query_id": 1, "db_id": "sample_db",
                "gold_sql": "SELECT email FROM customer",
                "prelim_sql_full": None,  # truncated
                "S_fwd_from_anchor": ["customer.email"],
                "truncated_full": True,
            },
        ])
        out = self.tmpdir / "a2.jsonl"
        summary = run_a2(a1, out)
        self.assertEqual(summary["n_total"], 2)
        self.assertEqual(summary["n_truncated_full"], 1)
        # Q0: gold={id, name}, fwd={id, email}, bwd={id, name, email}
        #   recall_fwd = 1/2 (id matches)
        #   recall_bwd = 2/2 (id, name)
        #   recall_union = 2/2
        records = [json.loads(l) for l in open(out)]
        q0 = next(r for r in records if r["query_id"] == 0)
        self.assertAlmostEqual(q0["recall_fwd"], 0.5)
        self.assertAlmostEqual(q0["recall_bwd"], 1.0)
        self.assertAlmostEqual(q0["recall_union"], 1.0)


# ──────────────────────────────────────────────────────────────
# A-3 E2E
# ──────────────────────────────────────────────────────────────

class TestA3RunE2E(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_a3_precision_calc(self):
        # mock A-2 결과
        a2 = self.tmpdir / "a2.jsonl"
        with open(a2, "w") as f:
            # Q0: L_bwd={id, name, email}, L_fwd=[customer.id, customer.email]
            #     gold=[id, name]
            #     S_restore = {name}  → precision = 1/1 = 1.0 (name is gold)
            #     missed_by_fwd = {name}  → recall_gained = 1/1 = 1.0
            f.write(json.dumps({
                "query_id": 0, "db_id": "sample_db",
                "L_bwd": ["id", "name", "email"],
                "L_fwd": ["customer.id", "customer.email"],
                "gold_cols": ["id", "name"],
                "truncated_full": False,
            }) + "\n")
            # Q1: L_bwd={id, total}, L_fwd=[customer.id]
            #     gold=[id]
            #     S_restore = {total}  → precision = 0 (total not gold)
            #     missed_by_fwd = {}  → recall_gained = 0 (denominator 0)
            f.write(json.dumps({
                "query_id": 1, "db_id": "sample_db",
                "L_bwd": ["id", "total"],
                "L_fwd": ["customer.id"],
                "gold_cols": ["id"],
                "truncated_full": False,
            }) + "\n")
        out = self.tmpdir / "a3.jsonl"
        summary = run_a3(a2, out)
        recs = [json.loads(l) for l in open(out)]
        q0 = next(r for r in recs if r["query_id"] == 0)
        q1 = next(r for r in recs if r["query_id"] == 1)
        self.assertAlmostEqual(q0["S_restore_precision"], 1.0)
        self.assertAlmostEqual(q0["recall_gained_by_restore"], 1.0)
        self.assertAlmostEqual(q1["S_restore_precision"], 0.0)
        self.assertAlmostEqual(q1["recall_gained_by_restore"], 0.0)
        # Mean precision = (1.0 + 0.0) / 2 = 0.5
        self.assertAlmostEqual(summary["mean_S_restore_precision"], 0.5)


# ──────────────────────────────────────────────────────────────
# A-1 small batch E2E (mocked LLM)
# ──────────────────────────────────────────────────────────────

class TestA1RunE2EMockLLM(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.db_dir = _build_temp_sqlite(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_mock_client(self, return_sql="SELECT COUNT(*) FROM customer"):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=return_sql))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5,
                                         prompt_tokens_details=MagicMock(cached_tokens=0))
        mock_client = MagicMock()
        mock_client.client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_a1_small_batch_chain(self):
        mock = self._make_mock_client("SELECT COUNT(*) FROM customer")
        out_dir = self.tmpdir / "filter_proposal"
        out_dir.mkdir(parents=True, exist_ok=True)
        anchor_final_nodes = {0: ["customer.id", "customer.name"],
                                1: ["customer.email"],
                                2: ["orders.total", "orders.customer_id"]}
        summary = run_a1(
            dev_data=SAMPLE_DEV,
            db_schema_map=SAMPLE_DB_SCHEMA_MAP,
            db_dir=str(self.db_dir),
            anchor_final_nodes=anchor_final_nodes,
            api_client=mock,
            model="zai-org/glm-4.7",
            output_path=out_dir / "A1.jsonl",
            max_queries=3,
            resume=False,
        )
        self.assertEqual(summary["n_total"], 3)
        # Verify per-query records
        records = [json.loads(l) for l in open(out_dir / "A1.jsonl")]
        self.assertEqual(len(records), 3)
        for r in records:
            self.assertIn("truncated_full", r)
            self.assertIn("prompt_tokens_full", r)
            self.assertIn("prelim_sql_full", r)
            self.assertIn("prelim_sql_fwd", r)
            self.assertIn("S_fwd_from_anchor", r)

    def test_a1_truncation_yields_null(self):
        """Truncated query 의 결과 = is_executable_full / exec_match_full None."""
        # patch GLM_4_7_TOKEN_LIMIT 를 매우 작은 값으로 — 모든 prompt 가 truncated
        import analysis.filter_proposal_a1_preliminary_sql as a1_mod
        original_limit = a1_mod.GLM_4_7_TOKEN_LIMIT
        a1_mod.GLM_4_7_TOKEN_LIMIT = 10  # 10 토큰 — 모든 schema prompt 가 초과
        try:
            mock = self._make_mock_client()
            out_dir = self.tmpdir / "filter_proposal"
            out_dir.mkdir(parents=True, exist_ok=True)
            anchor_final_nodes = {0: ["customer.id"]}
            summary = run_a1(
                dev_data=SAMPLE_DEV[:1],
                db_schema_map=SAMPLE_DB_SCHEMA_MAP,
                db_dir=str(self.db_dir),
                anchor_final_nodes=anchor_final_nodes,
                api_client=mock,
                model="zai-org/glm-4.7",
                output_path=out_dir / "A1_trunc.jsonl",
                max_queries=1,
                resume=False,
            )
            self.assertEqual(summary["n_truncated_full"], 1)
            self.assertAlmostEqual(summary["truncation_rate"], 1.0)
            rec = json.loads(open(out_dir / "A1_trunc.jsonl").readline())
            self.assertTrue(rec["truncated_full"])
            self.assertIsNone(rec["prelim_sql_full"])
            self.assertIsNone(rec["is_executable_full"])
            self.assertIsNone(rec["exec_match_full"])
            # S_fwd path 는 항상 정상 동작
            self.assertIsNotNone(rec["prelim_sql_fwd"])
        finally:
            a1_mod.GLM_4_7_TOKEN_LIMIT = original_limit


# ──────────────────────────────────────────────────────────────
# Full chain (A1 → A2 → A3) smoke
# ──────────────────────────────────────────────────────────────

class TestFullChain(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.db_dir = _build_temp_sqlite(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_mock_client(self, pred_sql):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=pred_sql))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5,
                                         prompt_tokens_details=MagicMock(cached_tokens=0))
        mock_client = MagicMock()
        mock_client.client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_full_chain_a1_a2_a3(self):
        # Mock LLM: returns a SQL that contains an extra column
        mock = self._make_mock_client("SELECT id, name, email FROM customer")
        out_dir = self.tmpdir / "filter_proposal"
        out_dir.mkdir(parents=True, exist_ok=True)
        # anchor S_fwd = [customer.id] only — backward should ADD name, email
        anchor_final_nodes = {0: ["customer.id"]}
        # Use a gold SQL with multiple cols (id + name)
        dev = [{"db_id": "sample_db",
                "question": "Get customers",
                "evidence": "",
                "SQL": "SELECT id, name FROM customer"}]
        a1_path = out_dir / "A1.jsonl"
        run_a1(dev, SAMPLE_DB_SCHEMA_MAP, str(self.db_dir), anchor_final_nodes,
                mock, "zai-org/glm-4.7", a1_path, max_queries=1, resume=False)
        a2_path = out_dir / "A2.jsonl"
        run_a2(a1_path, a2_path)
        a3_path = out_dir / "A3.jsonl"
        a3_summary = run_a3(a2_path, a3_path)
        # Verify chain consistency
        a2_rec = json.loads(open(a2_path).readline())
        a3_rec = json.loads(open(a3_path).readline())
        self.assertEqual(a2_rec["query_id"], 0)
        self.assertEqual(a3_rec["query_id"], 0)
        # recall_bwd should be 2/2 = 1.0 (id, name in prelim_sql)
        self.assertAlmostEqual(a2_rec["recall_bwd"], 1.0)
        # recall_union should be 2/2 = 1.0
        self.assertAlmostEqual(a2_rec["recall_union"], 1.0)
        # S_restore = {name, email} (L_bwd \ S_fwd 의 col-only)
        # precision = 1/2 (name is gold, email is not)
        self.assertAlmostEqual(a3_rec["S_restore_precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
