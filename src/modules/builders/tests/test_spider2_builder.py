"""Spider2GraphBuilder unit + 3-backend smoke tests.

DECISIONS 2026-06-09 #3 §G-S2-0 + G-S2-1 spec 정합. test 분류:
- 단위 (DB 무관): backend infer, DDL parser, FK extractor, regex fallback
- 3 backend smoke: bigquery (small + multi-inner) + snowflake + sqlite
- enterprise-scale: max_columns 초과 RuntimeError 검증
- BIRD 인터페이스 호환: required metadata keys 모두 존재

실행:
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src conda run -n base python -m pytest \\
        src/modules/builders/tests/test_spider2_builder.py -v
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from modules.builders.spider2_builder import (  # noqa: E402
    Spider2GraphBuilder,
    infer_backend,
    resolve_spider2_db_paths,
    _parse_ddl_csv,
    _parse_create_table,
    _regex_extract_columns,
    _filter_valid_fks,
    load_spider2_lite_jsonl,
    DEFAULT_SPIDER2_ROOT,
)


SPIDER2_ROOT = PROJECT_ROOT / "data" / "Spider2" / "spider2-lite" / "resource" / "databases"
LITE_JSONL = PROJECT_ROOT / "data" / "Spider2" / "spider2-lite" / "spider2-lite.jsonl"


def _spider2_available() -> bool:
    return SPIDER2_ROOT.exists() and LITE_JSONL.exists()


# 모든 builder 가 노출해야 하는 metadata keys (downstream contract)
REQUIRED_METADATA_KEYS = {
    "table_to_id", "col_to_id", "fk_to_id", "node_metadata",
    "edges", "edge_types", "add_t2t_edges",
    "fk_adjacency", "fk_adjacency_undirected", "fk_reachability",
    "fk_distance", "fk_shortest_paths", "fk_components",
    "fk_num_components", "fk_edge_lookup",
    "schema_diameter", "schema_eccentricity",
    "builder_info",
}
SPIDER2_EXTRA_KEYS = {
    "spider2_backend", "spider2_db", "spider2_inner_datasets",
    "spider2_total_columns", "spider2_parse_errors",
}


# --------------------------------------------------------------------------- #
# Unit tests — DB 무관
# --------------------------------------------------------------------------- #

class TestBackendInfer(unittest.TestCase):

    def test_bq_prefix(self):
        self.assertEqual(infer_backend("bq001"), "bigquery")
        self.assertEqual(infer_backend("bq134"), "bigquery")

    def test_ga_prefix(self):
        self.assertEqual(infer_backend("ga001"), "bigquery")
        self.assertEqual(infer_backend("ga014"), "bigquery")

    def test_sf_prefix(self):
        self.assertEqual(infer_backend("sf001"), "snowflake")

    def test_sf_bq_prefix_takes_precedence_over_sf(self):
        # sf_bq* 가 sf* 보다 긴 prefix → snowflake (둘 다 snowflake 라 동일)
        self.assertEqual(infer_backend("sf_bq001"), "snowflake")

    def test_local_prefix(self):
        self.assertEqual(infer_backend("local001"), "sqlite")
        self.assertEqual(infer_backend("local024"), "sqlite")

    def test_unknown_prefix(self):
        self.assertIsNone(infer_backend("unknown123"))
        self.assertIsNone(infer_backend("xyz001"))


class TestDDLParser(unittest.TestCase):
    """sqlglot + regex fallback parser 단위."""

    def test_sqlite_simple_create(self):
        ddl = """CREATE TABLE pizza_names (
            pizza_id INTEGER,
            pizza_name TEXT
        );"""
        cols, fks, err = _parse_create_table(ddl, "pizza_names", dialect="sqlite")
        self.assertIsNone(err)
        self.assertEqual(len(cols), 2)
        self.assertEqual(cols[0]["name"], "pizza_id")
        self.assertEqual(cols[1]["name"], "pizza_name")
        self.assertEqual(fks, [])

    def test_sqlite_with_foreign_key(self):
        ddl = """CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );"""
        cols, fks, err = _parse_create_table(ddl, "orders", dialect="sqlite")
        self.assertIsNone(err)
        col_names = {c["name"] for c in cols}
        self.assertIn("order_id", col_names)
        self.assertIn("customer_id", col_names)
        # FK 추출
        self.assertEqual(len(fks), 1)
        fk = fks[0]
        self.assertEqual(fk["from_table"], "orders")
        self.assertEqual(fk["from_column"], "customer_id")
        self.assertEqual(fk["to_table"], "customers")
        self.assertEqual(fk["to_column"], "id")

    def test_bigquery_array_struct_types(self):
        """BigQuery 위 ARRAY<...> / STRUCT<...> nested type 위 ColumnDef 인식."""
        ddl = """CREATE TABLE `proj.ds.variants` (
            reference_name STRING,
            alternate_bases ARRAY<STRING>,
            call ARRAY<STRUCT<call_set_id STRING, genotype ARRAY<INT64>>>
        );"""
        cols, fks, err = _parse_create_table(ddl, "variants", dialect="bigquery")
        self.assertIsNone(err, msg=f"sqlglot bigquery dialect failed: {err}")
        col_names = [c["name"] for c in cols]
        # nested STRUCT 내부 column 은 top-level ColumnDef 위 들어가지 않음
        # (BigQuery DDL 위 reference_name / alternate_bases / call 3개만 top-level)
        self.assertIn("reference_name", col_names)
        self.assertIn("alternate_bases", col_names)
        self.assertIn("call", col_names)

    def test_snowflake_create(self):
        ddl = """CREATE TABLE FINANCE.CYBERSYN.AIRPORT_INDEX (
            DOT_CODE NUMBER,
            IATA_CODE TEXT,
            CITY_NAME TEXT
        );"""
        cols, _, err = _parse_create_table(
            ddl, "AIRPORT_INDEX", dialect="snowflake")
        self.assertIsNone(err)
        self.assertEqual(len(cols), 3)
        self.assertEqual({c["name"] for c in cols},
                         {"DOT_CODE", "IATA_CODE", "CITY_NAME"})

    def test_regex_fallback(self):
        """sqlglot 실패 케이스 시뮬레이션 — regex 위 column 명 추출."""
        ddl = """CREATE TABLE weird_table (
            id INTEGER,
            data TEXT NOT NULL,
            score REAL DEFAULT 0.0
        );"""
        cols, fks = _regex_extract_columns(ddl)
        names = {c["name"] for c in cols}
        self.assertEqual(names, {"id", "data", "score"})
        self.assertEqual(fks, [])

    def test_regex_skips_constraint_lines(self):
        ddl = """CREATE TABLE t (
            id INT,
            CONSTRAINT pk_t PRIMARY KEY (id),
            FOREIGN KEY (id) REFERENCES u(uid)
        );"""
        cols, _ = _regex_extract_columns(ddl)
        names = {c["name"] for c in cols}
        self.assertIn("id", names)
        self.assertNotIn("CONSTRAINT", names)
        self.assertNotIn("FOREIGN", names)


class TestFilterValidFKs(unittest.TestCase):

    def test_keep_valid(self):
        cols = {"a": [{"name": "x"}], "b": [{"name": "y"}]}
        fks = [{"from_table": "a", "from_column": "x",
                "to_table": "b", "to_column": "y"}]
        self.assertEqual(_filter_valid_fks(fks, cols), fks)

    def test_drop_unknown_table(self):
        cols = {"a": [{"name": "x"}]}
        fks = [{"from_table": "a", "from_column": "x",
                "to_table": "ghost", "to_column": "y"}]
        self.assertEqual(_filter_valid_fks(fks, cols), [])

    def test_drop_unknown_column(self):
        cols = {"a": [{"name": "x"}], "b": [{"name": "y"}]}
        fks = [{"from_table": "a", "from_column": "missing",
                "to_table": "b", "to_column": "y"}]
        self.assertEqual(_filter_valid_fks(fks, cols), [])


# --------------------------------------------------------------------------- #
# Live BIRD-Dev tests (cond on data availability)
# --------------------------------------------------------------------------- #

@unittest.skipUnless(_spider2_available(), "Spider2 data not available — skip")
class TestSpider2JSONLLoader(unittest.TestCase):

    def test_load_jsonl(self):
        mapping = load_spider2_lite_jsonl(LITE_JSONL)
        self.assertGreater(len(mapping), 200)  # 547 total
        # 일부 instance 위 schema 확인
        for iid, entry in list(mapping.items())[:3]:
            self.assertIn("db", entry)
            self.assertIn("question", entry)


@unittest.skipUnless(_spider2_available(), "Spider2 data not available — skip")
class TestPathResolver(unittest.TestCase):

    def test_sqlite_resolve(self):
        # E_commerce sqlite — DECISIONS 2026-06-10 확인된 path
        backend, paths = resolve_spider2_db_paths(
            "local002", "E_commerce", SPIDER2_ROOT)
        self.assertEqual(backend, "sqlite")
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].name == "DDL.csv")

    def test_bigquery_multi_inner(self):
        # austin 위 5 inner dataset
        backend, paths = resolve_spider2_db_paths(
            "bq999", "austin", SPIDER2_ROOT)
        self.assertEqual(backend, "bigquery")
        self.assertEqual(len(paths), 5)
        for p in paths:
            self.assertEqual(p.name, "DDL.csv")

    def test_unknown_db_raises(self):
        with self.assertRaises(FileNotFoundError):
            resolve_spider2_db_paths(
                "bq999", "__nonexistent_db__", SPIDER2_ROOT)


# --------------------------------------------------------------------------- #
# 3-backend smoke (실제 build)
# --------------------------------------------------------------------------- #

@unittest.skipUnless(_spider2_available(), "Spider2 data not available — skip")
class TestSqliteBackend(unittest.TestCase):
    """sqlite 위 작은 DB (E_commerce) 위 full build smoke."""

    @classmethod
    def setUpClass(cls):
        cls.builder = Spider2GraphBuilder(max_columns=5000)
        cls.data, cls.meta = cls.builder.build(
            db_id="local002", db_dir=str(SPIDER2_ROOT),
            spider2_db_field="E_commerce",
        )

    def test_backend_recorded(self):
        self.assertEqual(self.meta["spider2_backend"], "sqlite")

    def test_required_metadata_keys(self):
        missing = REQUIRED_METADATA_KEYS - set(self.meta.keys())
        self.assertFalse(missing, msg=f"missing required keys: {missing}")
        missing_extra = SPIDER2_EXTRA_KEYS - set(self.meta.keys())
        self.assertFalse(missing_extra, msg=f"missing spider2 keys: {missing_extra}")

    def test_graph_structure(self):
        self.assertIn("table", self.data.node_types)
        self.assertIn("column", self.data.node_types)
        T = len(self.meta["table_to_id"])
        C = len(self.meta["col_to_id"])
        self.assertGreater(T, 0)
        self.assertGreater(C, 0)
        # PLM embedding shape
        self.assertEqual(tuple(self.data["table"].x.shape), (T, 384))
        self.assertEqual(tuple(self.data["column"].x.shape), (C, 384))

    def test_pcst_flat_indexing(self):
        T = len(self.meta["table_to_id"])
        C = len(self.meta["col_to_id"])
        F = len(self.meta["fk_to_id"])
        total = T + C + F
        for (s, d), et in zip(self.meta["edges"], self.meta["edge_types"]):
            self.assertTrue(0 <= s < total)
            self.assertTrue(0 <= d < total)


@unittest.skipUnless(_spider2_available(), "Spider2 data not available — skip")
class TestBigqueryBackend(unittest.TestCase):
    """bigquery 위 small DB (_1000_genomes, 124-line DDL.csv) full build smoke."""

    @classmethod
    def setUpClass(cls):
        cls.builder = Spider2GraphBuilder(max_columns=5000)
        cls.data, cls.meta = cls.builder.build(
            db_id="bq999", db_dir=str(SPIDER2_ROOT),
            spider2_db_field="_1000_genomes",
        )

    def test_backend_recorded(self):
        self.assertEqual(self.meta["spider2_backend"], "bigquery")

    def test_fk_count_zero_for_bigquery(self):
        # bigquery 위 FK 없음 — DECISIONS 2026-06-10 caveat
        self.assertEqual(len(self.meta["fk_to_id"]), 0)

    def test_required_metadata_keys(self):
        missing = REQUIRED_METADATA_KEYS - set(self.meta.keys())
        self.assertFalse(missing, msg=f"missing: {missing}")


@unittest.skipUnless(_spider2_available(), "Spider2 data not available — skip")
class TestBigqueryMultiInner(unittest.TestCase):
    """bigquery 위 multi-inner (austin = 5 datasets) merged graph smoke."""

    @classmethod
    def setUpClass(cls):
        cls.builder = Spider2GraphBuilder(max_columns=5000)
        cls.data, cls.meta = cls.builder.build(
            db_id="bq999", db_dir=str(SPIDER2_ROOT),
            spider2_db_field="austin",
        )

    def test_inner_datasets_5(self):
        self.assertEqual(len(self.meta["spider2_inner_datasets"]), 5)

    def test_multi_inner_merged(self):
        # 5 inner 위 각 ~5~15 table → 합쳐서 multiple tables
        self.assertGreater(len(self.meta["table_to_id"]), 5)


@unittest.skipUnless(_spider2_available(), "Spider2 data not available — skip")
class TestSnowflakeBackend(unittest.TestCase):
    """snowflake 위 first DB full build smoke."""

    @classmethod
    def setUpClass(cls):
        # snowflake 위 한 small DB 선택 (FINANCE__ECONOMICS 같은 큰 DB 는 skip)
        sf_root = SPIDER2_ROOT / "snowflake"
        cls.sf_dbs = sorted(d.name for d in sf_root.iterdir() if d.is_dir())
        # 첫 DB 위 build
        cls.first_db = cls.sf_dbs[0]
        cls.builder = Spider2GraphBuilder(max_columns=5000)
        try:
            cls.data, cls.meta = cls.builder.build(
                db_id="sf999", db_dir=str(SPIDER2_ROOT),
                spider2_db_field=cls.first_db,
            )
            cls.build_ok = True
        except RuntimeError as e:
            # enterprise-scale skip 도 정상 동작
            cls.build_ok = False
            cls.build_err = str(e)

    def test_build_or_skip(self):
        if not self.build_ok:
            # max_columns 초과 위 RuntimeError 인 경우만 허용
            self.assertIn("max_columns", self.build_err.lower())
            return
        self.assertEqual(self.meta["spider2_backend"], "snowflake")
        self.assertGreater(len(self.meta["table_to_id"]), 0)


@unittest.skipUnless(_spider2_available(), "Spider2 data not available — skip")
class TestEnterpriseScaleSkip(unittest.TestCase):
    """max_columns 작은 값 설정 위 enterprise-scale skip RuntimeError 검증."""

    def test_max_columns_skip(self):
        # max_columns=5 위 어떤 DB 도 거의 skip
        builder = Spider2GraphBuilder(max_columns=5)
        # E_commerce 위 column 5개 초과 → RuntimeError
        with self.assertRaises(RuntimeError) as cm:
            builder.build(
                db_id="local002", db_dir=str(SPIDER2_ROOT),
                spider2_db_field="E_commerce",
            )
        self.assertIn("max_columns", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
