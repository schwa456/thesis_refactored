"""Smoke test for glm_baseline_sql_eval.

목적:
  - 3 scenario (full / gold_table / gold_column) 의 helper 함수 (gold extraction,
    schema build, prompt format, SQL exec) 가 정상 동작하는지 mock 환경에서 검증
  - LLM client API call 은 mock 으로 대체 (실제 GLM API 호출 없음)
  - 실패 case (gold SQL 파싱 실패, error pred, query timeout) handle 정상

근거: planning/DECISIONS.md 2026-05-12 (B1'+B2'+B3' GLM 4.7 Baseline 즉시 launch, Step 3)
재현: PYTHONPATH=src conda run -n base python -m pytest src/analysis/tests/test_glm_baseline_sql_eval.py -v
"""

from __future__ import annotations

import os
import sys
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis.glm_baseline_sql_eval import (
    extract_gold_tables, extract_gold_columns,
    build_schema_dot_format, create_messages,
    chat_completion_glm, execute_sql_on_db, eval_ex,
    run_scenario, SCENARIO_DIRS,
)


# ──────────────────────────────────────────────────────────────
# Fixtures
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
    {
        "db_id": "sample_db",
        "question": "How many customers are there?",
        "evidence": "",
        "SQL": "SELECT COUNT(*) FROM customer",
    },
    {
        "db_id": "sample_db",
        "question": "Total orders?",
        "evidence": "",
        "SQL": "SELECT COUNT(order_id) FROM orders",
    },
]


def _build_temp_sqlite(tmpdir: Path) -> Path:
    """간단한 sqlite DB 파일 생성 — execute_sql_on_db 테스트용."""
    db_root = tmpdir / "sample_db"
    db_root.mkdir(parents=True, exist_ok=True)
    db_path = db_root / "sample_db.sqlite"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE customer (id INTEGER, name TEXT, email TEXT)")
    cursor.execute("CREATE TABLE orders (order_id INTEGER, customer_id INTEGER, total REAL)")
    cursor.execute("INSERT INTO customer VALUES (1, 'Alice', 'a@x.com'), (2, 'Bob', 'b@x.com')")
    cursor.execute("INSERT INTO orders VALUES (1, 1, 100.0), (2, 2, 200.0), (3, 1, 300.0)")
    conn.commit()
    conn.close()
    return db_path.parent.parent  # db_dir = tmpdir (sample_db/sample_db.sqlite 의 부모의 부모)


# ──────────────────────────────────────────────────────────────
# Tests — gold extraction
# ──────────────────────────────────────────────────────────────

class TestGoldExtraction(unittest.TestCase):
    def test_extract_tables_basic(self):
        sql = "SELECT a.id FROM customer a JOIN orders b ON a.id=b.customer_id"
        tables = extract_gold_tables(sql)
        self.assertEqual(tables, {"customer", "orders"})

    def test_extract_columns_basic(self):
        sql = "SELECT id, name FROM customer WHERE email='x'"
        cols = extract_gold_columns(sql)
        # sqlglot 추출: id, name, email
        self.assertIn("id", cols)
        self.assertIn("name", cols)
        self.assertIn("email", cols)

    def test_extract_tables_parse_error(self):
        # 깨진 SQL — 빈 set 반환
        sql = "THIS IS NOT SQL ###"
        tables = extract_gold_tables(sql)
        self.assertEqual(tables, set())

    def test_extract_columns_parse_error(self):
        sql = "THIS IS NOT SQL ###"
        cols = extract_gold_columns(sql)
        self.assertEqual(cols, set())


# ──────────────────────────────────────────────────────────────
# Tests — schema build (3 scenario)
# ──────────────────────────────────────────────────────────────

class TestSchemaBuild(unittest.TestCase):
    def test_full_schema(self):
        s = build_schema_dot_format(SAMPLE_DB_SCHEMA_MAP, "sample_db")
        # 모든 column (id/name/email/order_id/customer_id/total)
        self.assertIn("customer.id", s)
        self.assertIn("orders.total", s)
        self.assertEqual(s.count("\n") + 1, 6)  # 6 columns

    def test_gold_table_filter(self):
        s = build_schema_dot_format(SAMPLE_DB_SCHEMA_MAP, "sample_db",
                                     used_tables={"customer"})
        self.assertIn("customer.id", s)
        self.assertNotIn("orders.order_id", s)
        self.assertEqual(s.count("\n") + 1, 3)  # customer 3 columns

    def test_gold_column_filter(self):
        s = build_schema_dot_format(SAMPLE_DB_SCHEMA_MAP, "sample_db",
                                     used_columns={"id", "name"})
        # id 는 customer.id 와 orders 에 없음 → customer.id + customer.name
        self.assertIn("customer.id", s)
        self.assertIn("customer.name", s)
        self.assertNotIn("customer.email", s)
        self.assertNotIn("orders.order_id", s)

    def test_unknown_db(self):
        s = build_schema_dot_format(SAMPLE_DB_SCHEMA_MAP, "no_such_db")
        self.assertEqual(s, "")


# ──────────────────────────────────────────────────────────────
# Tests — prompt
# ──────────────────────────────────────────────────────────────

class TestPrompt(unittest.TestCase):
    def test_messages_structure(self):
        msgs = create_messages("customer.id\ncustomer.name", "How many?", "no evidence")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("backticks", msgs[0]["content"])
        self.assertIn("customer.id", msgs[1]["content"])
        self.assertIn("How many?", msgs[1]["content"])


# ──────────────────────────────────────────────────────────────
# Tests — SQL execution
# ──────────────────────────────────────────────────────────────

class TestSqlExec(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.db_dir = _build_temp_sqlite(self.tmpdir)
        self.db_path = str(self.db_dir / "sample_db" / "sample_db.sqlite")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_valid_sql(self):
        result = execute_sql_on_db(self.db_path, "SELECT COUNT(*) FROM customer")
        self.assertEqual(result, {(2,)})

    def test_syntax_error_returns_str(self):
        result = execute_sql_on_db(self.db_path, "SELEC FROM nowhere")
        self.assertIsInstance(result, str)

    def test_unknown_table_returns_str(self):
        result = execute_sql_on_db(self.db_path, "SELECT * FROM unknown_table")
        self.assertIsInstance(result, str)

    def test_eval_ex_match(self):
        gold = execute_sql_on_db(self.db_path, "SELECT COUNT(*) FROM customer")
        pred = execute_sql_on_db(self.db_path, "SELECT COUNT(id) FROM customer")
        self.assertTrue(eval_ex(gold, pred))

    def test_eval_ex_mismatch(self):
        gold = execute_sql_on_db(self.db_path, "SELECT COUNT(*) FROM customer")
        pred = execute_sql_on_db(self.db_path, "SELECT COUNT(*) FROM orders")
        self.assertFalse(eval_ex(gold, pred))

    def test_eval_ex_gold_error_returns_false(self):
        gold_err = "Error: parse fail"
        pred = {(1,)}
        self.assertFalse(eval_ex(gold_err, pred))


# ──────────────────────────────────────────────────────────────
# Tests — chat_completion error handling (mock)
# ──────────────────────────────────────────────────────────────

class TestChatCompletionMock(unittest.TestCase):
    def test_mock_success(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="SELECT * FROM customer"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5,
                                         prompt_tokens_details=MagicMock(cached_tokens=0))
        mock_client = MagicMock()
        mock_client.client.chat.completions.create.return_value = mock_response

        out = chat_completion_glm(mock_client, "zai-org/glm-4.7",
                                    [{"role": "user", "content": "test"}], max_retries=1)
        self.assertEqual(out, "SELECT * FROM customer")

    def test_mock_failure_returns_error_string(self):
        mock_client = MagicMock()
        mock_client.client.chat.completions.create.side_effect = RuntimeError("network down")
        out = chat_completion_glm(mock_client, "zai-org/glm-4.7",
                                    [{"role": "user", "content": "test"}],
                                    max_retries=1, retry_delay=0.01)
        self.assertTrue(out.startswith("-- Error:"))


# ──────────────────────────────────────────────────────────────
# Tests — run_scenario E2E (mock LLM, real sqlite)
# ──────────────────────────────────────────────────────────────

class TestRunScenarioE2E(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.db_dir = _build_temp_sqlite(self.tmpdir)
        self.output_root = self.tmpdir / "output"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_mock_client(self, pred_sql: str):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=pred_sql))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5,
                                         prompt_tokens_details=MagicMock(cached_tokens=0))
        mock_client = MagicMock()
        mock_client.client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_full_scenario_with_correct_pred(self):
        # Pred = gold → EX 100%
        mock = self._make_mock_client("SELECT COUNT(*) FROM customer")
        out_dir = self.output_root / "b1_full"
        # 첫 query (SELECT COUNT(*) FROM customer) 만 evaluated
        result = run_scenario(
            "full", SAMPLE_DEV[:1], SAMPLE_DB_SCHEMA_MAP, str(self.db_dir),
            mock, "zai-org/glm-4.7", out_dir, max_queries=1, resume=False,
        )
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["correct"], 1)
        self.assertAlmostEqual(result["ex"], 100.0)

    def test_gold_table_scenario(self):
        mock = self._make_mock_client("SELECT COUNT(*) FROM customer")
        out_dir = self.output_root / "b2_gold_table"
        result = run_scenario(
            "gold_table", SAMPLE_DEV[:1], SAMPLE_DB_SCHEMA_MAP, str(self.db_dir),
            mock, "zai-org/glm-4.7", out_dir, max_queries=1, resume=False,
        )
        self.assertEqual(result["total"], 1)
        # predictions.jsonl 의 schema_size 가 gold_table 기준 3 (customer 3 columns)
        with open(out_dir / "predictions.jsonl") as f:
            rec = json.loads(f.readline())
        self.assertEqual(rec["schema_size"], 3)

    def test_gold_column_scenario(self):
        mock = self._make_mock_client("SELECT * FROM customer")
        out_dir = self.output_root / "b3_gold_column"
        result = run_scenario(
            "gold_column", SAMPLE_DEV[:1], SAMPLE_DB_SCHEMA_MAP, str(self.db_dir),
            mock, "zai-org/glm-4.7", out_dir, max_queries=1, resume=False,
        )
        self.assertEqual(result["total"], 1)

    def test_pred_error_handled(self):
        # Pred 가 invalid SQL → execute_sql 이 str 반환, is_correct=False
        mock = self._make_mock_client("THIS IS NOT SQL")
        out_dir = self.output_root / "b1_full_err"
        result = run_scenario(
            "full", SAMPLE_DEV[:1], SAMPLE_DB_SCHEMA_MAP, str(self.db_dir),
            mock, "zai-org/glm-4.7", out_dir, max_queries=1, resume=False,
        )
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["correct"], 0)
        self.assertGreaterEqual(result["pred_error"], 1)

    def test_resume_skips_existing(self):
        mock = self._make_mock_client("SELECT COUNT(*) FROM customer")
        out_dir = self.output_root / "b1_full_resume"
        # 첫 실행
        run_scenario("full", SAMPLE_DEV[:1], SAMPLE_DB_SCHEMA_MAP, str(self.db_dir),
                     mock, "zai-org/glm-4.7", out_dir, max_queries=1, resume=False)
        # Resume 시 LLM call 안 일어나야 함
        mock.client.chat.completions.create.reset_mock()
        # 2 query 로 확장 — 첫 qid=0 skip, qid=1 만 새로 측정
        run_scenario("full", SAMPLE_DEV, SAMPLE_DB_SCHEMA_MAP, str(self.db_dir),
                     mock, "zai-org/glm-4.7", out_dir, max_queries=2, resume=True)
        # qid=1 1 회 call (qid=0 은 resume skip)
        self.assertEqual(mock.client.chat.completions.create.call_count, 1)


class TestScenarioDirs(unittest.TestCase):
    def test_scenario_dirs_mapping(self):
        self.assertEqual(SCENARIO_DIRS["full"], "b1_full")
        self.assertEqual(SCENARIO_DIRS["gold_table"], "b2_gold_table")
        self.assertEqual(SCENARIO_DIRS["gold_column"], "b3_gold_column")


if __name__ == "__main__":
    unittest.main()
