"""V6-W3 builder 3 variants 위 unit tests.

5 cases per variant (× 3 variants = 15 tests + 공통 sanity):
1. graph structure 검증 (node type / edge type count)
2. hub identification (variant C 만 — median 기반 hub 인식)
3. virtual node feature init (mean / attention pooling 결과 검증)
4. FK reachability 등 기존 metadata keys 유지 (downstream 호환)
5. PCST flat 인덱싱 + downstream interface 호환 (table_to_id/col_to_id/fk_to_id/
   node_metadata/edges/edge_types 유효성)

실행:
    PYTHONPATH=src conda run -n base python -m pytest \\
        src/modules/builders/tests/test_v6w3_builders.py -v

또는 (단독 script):
    PYTHONPATH=src conda run -n base python \\
        src/modules/builders/tests/test_v6w3_builders.py
"""
from __future__ import annotations

import os
import sys
import unittest
from typing import Dict, Any

import numpy as np
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from modules.builders.v6w3_builders import (
    V6W3VirtualSummaryBuilder,
    V6W3ColumnPoolingBuilder,
    V6W3HubLocalVNBuilder,
    _identify_hub_tables,
    _column_means_per_table,
)


# BIRD-Dev 위 가장 작은 schema (3 tables, ~30 columns) — 빠른 smoke
SMALL_DB = "california_schools"
# Hub natural test bed (1 large table 115-col, RFP H2 complete collapse case)
HUB_DB = "european_football_2"
DB_DIR = os.path.join(PROJECT_ROOT, "data/raw/BIRD_dev/dev_databases")
TABLES_JSON = os.path.join(PROJECT_ROOT, "data/raw/BIRD_dev/dev_tables.json")

# 기존 metadata 위 모든 builder 가 노출해야 하는 keys (downstream contract)
REQUIRED_METADATA_KEYS = {
    'table_to_id', 'col_to_id', 'fk_to_id', 'node_metadata',
    'edges', 'edge_types', 'add_t2t_edges',
    'fk_adjacency', 'fk_adjacency_undirected', 'fk_reachability',
    'fk_distance', 'fk_shortest_paths', 'fk_components',
    'fk_num_components', 'fk_edge_lookup',
    'schema_diameter', 'schema_eccentricity',
    'builder_info',
}


def _check_db_available() -> bool:
    return os.path.exists(os.path.join(DB_DIR, SMALL_DB, f"{SMALL_DB}.sqlite"))


# --------------------------------------------------------------------------- #
# Helper functions (직접 unit 검증 — DB 미접근)
# --------------------------------------------------------------------------- #

class TestHubIdentification(unittest.TestCase):
    """variant C 위 hub identification helper 단독 검증 (DB 무관)."""

    def test_median_basic(self):
        table_to_id = {"small_a": 0, "small_b": 1, "big": 2}
        schema_cols = {
            "small_a": [{"name": "x"}, {"name": "y"}],          # 2 cols
            "small_b": [{"name": "x"}, {"name": "y"}, {"name": "z"}],  # 3
            "big": [{"name": f"c{i}"} for i in range(20)],      # 20
        }
        hubs, thr, counts = _identify_hub_tables(table_to_id, schema_cols, strategy="median")
        # median([2, 3, 20]) = 3.0 → only "big" (20 > 3.0) is hub
        self.assertEqual(hubs, [2])
        self.assertEqual(thr, 3.0)
        self.assertEqual(counts, {0: 2, 1: 3, 2: 20})

    def test_min_columns_filter(self):
        table_to_id = {"a": 0, "b": 1, "c": 2}
        schema_cols = {
            "a": [{"name": "x"}], "b": [{"name": "x"}, {"name": "y"}],
            "c": [{"name": "x"}, {"name": "y"}, {"name": "z"}],
        }
        # median=2, "c" col_count=3 > median but 3 ≤ min_columns(5) → excluded
        hubs, _, _ = _identify_hub_tables(
            table_to_id, schema_cols, strategy="median", min_columns=5)
        self.assertEqual(hubs, [])

    def test_empty_input(self):
        hubs, thr, counts = _identify_hub_tables({}, {}, strategy="median")
        self.assertEqual(hubs, [])
        self.assertEqual(thr, 0.0)
        self.assertEqual(counts, {})


# --------------------------------------------------------------------------- #
# Live BIRD-Dev tests (cond on data availability)
# --------------------------------------------------------------------------- #

@unittest.skipUnless(_check_db_available(), f"BIRD-Dev {SMALL_DB}.sqlite 미존재 — skip")
class TestVariantA_VirtualSummary(unittest.TestCase):
    """Variant A — V6W3VirtualSummaryBuilder smoke + 5 cases."""

    @classmethod
    def setUpClass(cls):
        cls.builder = V6W3VirtualSummaryBuilder(tables_json_path=TABLES_JSON)
        cls.data, cls.meta = cls.builder.build(SMALL_DB, DB_DIR)

    def test_1_graph_structure(self):
        """node type table_summary 존재 + 4 신규 edge type"""
        self.assertIn('table_summary', self.data.node_types)
        edge_type_set = set(self.data.edge_types)
        # 신규 4종 edge type (canonical triples)
        expected_new = {
            ('table', 'has_summary', 'table_summary'),
            ('table_summary', 'summary_of', 'table'),
            ('table_summary', 'summarizes', 'column'),
            ('column', 'aggregated_by', 'table_summary'),
        }
        self.assertTrue(expected_new.issubset(edge_type_set),
                        msg=f"missing edge types: {expected_new - edge_type_set}")
        # 기존 edge types 유지
        existing = {('table', 'has_column', 'column'),
                    ('column', 'belongs_to', 'table')}
        self.assertTrue(existing.issubset(edge_type_set))

    def test_2_summary_count_matches_table_count(self):
        """summary 노드 수 = table 수 (1대1)"""
        T = len(self.meta['table_to_id'])
        self.assertEqual(self.data['table_summary'].x.shape[0], T)
        self.assertEqual(len(self.meta['summary_to_id']), T)

    def test_3_summary_feature_is_column_mean(self):
        """summary feature[i] = 해당 table 의 column embedding mean"""
        schema_info = self.builder._get_schema_info(
            V6W3VirtualSummaryBuilder._resolve_db_path(SMALL_DB, DB_DIR))
        col_to_id = self.meta['col_to_id']
        for table, s_idx in self.meta['summary_to_id'].items():
            cols = schema_info["columns"].get(table, [])
            col_ids = [col_to_id[f"{table}.{c['name']}"]
                       for c in cols if f"{table}.{c['name']}" in col_to_id]
            if not col_ids:
                continue
            expected = self.data['column'].x[col_ids].mean(dim=0)
            actual = self.data['table_summary'].x[s_idx]
            self.assertTrue(torch.allclose(actual, expected, atol=1e-5),
                            msg=f"summary feature mismatch for {table}")

    def test_4_fk_reachability_preserved(self):
        """기존 FK reachability keys 모두 retain"""
        missing = REQUIRED_METADATA_KEYS - set(self.meta.keys())
        self.assertFalse(missing, msg=f"missing required keys: {missing}")
        # 신규 키
        self.assertIn('summary_to_id', self.meta)
        self.assertIn('summary_flat_offset', self.meta)
        self.assertEqual(self.meta['v6w3_variant'], 'A')

    def test_5_pcst_flat_indexing_consistency(self):
        """PCST flat: table → column → fk_node → table_summary 블록 순서"""
        T = len(self.meta['table_to_id'])
        C = len(self.meta['col_to_id'])
        F = len(self.meta['fk_to_id'])
        offset = self.meta['summary_flat_offset']
        self.assertEqual(offset, T + C + F)
        # 모든 edge 위 flat idx 가 valid range [0, T+C+F+T_summary)
        total = T + C + F + T  # summary 수 = T
        for (s, d), et in zip(self.meta['edges'], self.meta['edge_types']):
            self.assertTrue(0 <= s < total, msg=f"src out of range: {s} (type={et})")
            self.assertTrue(0 <= d < total, msg=f"dst out of range: {d} (type={et})")
        # node_metadata 위 summary 등록 확인
        for tbl, s_local in self.meta['summary_to_id'].items():
            flat = offset + s_local
            self.assertIn(flat, self.meta['node_metadata'])
            self.assertIn('__summary__', self.meta['node_metadata'][flat])


@unittest.skipUnless(_check_db_available(), f"BIRD-Dev {SMALL_DB}.sqlite 미존재 — skip")
class TestVariantB_ColumnPooling(unittest.TestCase):
    """Variant B — V6W3ColumnPoolingBuilder smoke + 5 cases."""

    @classmethod
    def setUpClass(cls):
        cls.builder = V6W3ColumnPoolingBuilder(
            tables_json_path=TABLES_JSON, pool_mode="uniform")
        cls.data, cls.meta = cls.builder.build(SMALL_DB, DB_DIR)

    def test_1_no_new_node_type(self):
        """node/edge type 그대로 — table feature 만 override"""
        # 기존 3 node type 만 (fk_node 포함, table_summary 없음)
        self.assertIn('table', self.data.node_types)
        self.assertIn('column', self.data.node_types)
        self.assertNotIn('table_summary', self.data.node_types)
        self.assertNotIn('local_vn', self.data.node_types)

    def test_2_table_feature_is_column_mean(self):
        """uniform mode → table feature[i] = column mean (해당 table)"""
        schema_info = self.builder._get_schema_info(
            V6W3VirtualSummaryBuilder._resolve_db_path(SMALL_DB, DB_DIR))
        col_to_id = self.meta['col_to_id']
        for table, t_idx in self.meta['table_to_id'].items():
            cols = schema_info["columns"].get(table, [])
            col_ids = [col_to_id[f"{table}.{c['name']}"]
                       for c in cols if f"{table}.{c['name']}" in col_to_id]
            if not col_ids:
                continue
            expected = self.data['column'].x[col_ids].mean(dim=0)
            actual = self.data['table'].x[t_idx]
            self.assertTrue(torch.allclose(actual, expected, atol=1e-5),
                            msg=f"pooled table feature mismatch for {table}")

    def test_3_pool_weights_sum_to_one(self):
        """각 table 위 pool_weights sum = 1.0 (non-empty)"""
        for t_idx, w in self.meta['table_pool_weights'].items():
            if w.size == 0:
                continue
            self.assertAlmostEqual(float(w.sum()), 1.0, places=5)

    def test_4_fk_reachability_preserved(self):
        missing = REQUIRED_METADATA_KEYS - set(self.meta.keys())
        self.assertFalse(missing, msg=f"missing required keys: {missing}")
        self.assertIn('table_pool_weights', self.meta)
        self.assertEqual(self.meta['table_pool_mode'], 'uniform')
        self.assertEqual(self.meta['v6w3_variant'], 'B')

    def test_5_pcst_indexing_unchanged(self):
        """B 는 새 node 추가 X — edges/types 갯수 baseline 동일"""
        # baseline EnrichedHeteroGraphBuilder 와 비교
        from modules.builders.graph_builder import EnrichedHeteroGraphBuilder
        baseline = EnrichedHeteroGraphBuilder(tables_json_path=TABLES_JSON)
        _, base_meta = baseline.build(SMALL_DB, DB_DIR)
        self.assertEqual(len(self.meta['edges']), len(base_meta['edges']))
        self.assertEqual(self.meta['edge_types'], base_meta['edge_types'])
        # node_metadata 동일 keys (summary/local_vn 추가 없음)
        self.assertEqual(set(self.meta['node_metadata'].keys()),
                         set(base_meta['node_metadata'].keys()))


@unittest.skipUnless(_check_db_available(), f"BIRD-Dev {SMALL_DB}.sqlite 미존재 — skip")
class TestVariantB_CosineSoftmax(unittest.TestCase):
    """Variant B (cosine_softmax mode) — 별도 smoke"""

    @classmethod
    def setUpClass(cls):
        cls.builder = V6W3ColumnPoolingBuilder(
            tables_json_path=TABLES_JSON, pool_mode="cosine_softmax")
        cls.data, cls.meta = cls.builder.build(SMALL_DB, DB_DIR)

    def test_weights_valid(self):
        for t_idx, w in self.meta['table_pool_weights'].items():
            if w.size == 0:
                continue
            self.assertAlmostEqual(float(w.sum()), 1.0, places=5)
            self.assertTrue(np.all(w >= 0))

    def test_mode_recorded(self):
        self.assertEqual(self.meta['table_pool_mode'], 'cosine_softmax')


@unittest.skipUnless(_check_db_available(), f"BIRD-Dev {SMALL_DB}.sqlite 미존재 — skip")
class TestVariantC_HubLocalVN(unittest.TestCase):
    """Variant C — V6W3HubLocalVNBuilder smoke + 5 cases (small DB)."""

    @classmethod
    def setUpClass(cls):
        cls.builder = V6W3HubLocalVNBuilder(tables_json_path=TABLES_JSON)
        cls.data, cls.meta = cls.builder.build(SMALL_DB, DB_DIR)

    def test_1_graph_structure(self):
        """local_vn node 존재 + 4 신규 edge type (hub > 0 시)"""
        self.assertIn('local_vn', self.data.node_types)
        # hub 0개 케이스 (median 위 동률 tie 등) → edges 없을 수 있음
        if len(self.meta['hub_tables']) > 0:
            edge_type_set = set(self.data.edge_types)
            self.assertIn(('table', 'has_local_vn', 'local_vn'), edge_type_set)
            self.assertIn(('local_vn', 'serves_table', 'table'), edge_type_set)

    def test_2_hub_identification(self):
        """hub_tables 는 median 초과 column count 위 정합"""
        counts = self.meta['table_col_count']
        thr = self.meta['hub_threshold']
        expected_hubs = sorted([t for t, c in counts.items() if c > thr])
        self.assertEqual(sorted(self.meta['hub_tables']), expected_hubs)

    def test_3_local_vn_feature_is_column_mean(self):
        """Local VN feature[i] = hub table 위 column embedding mean"""
        schema_info = self.builder._get_schema_info(
            V6W3VirtualSummaryBuilder._resolve_db_path(SMALL_DB, DB_DIR))
        col_to_id = self.meta['col_to_id']
        id_to_table = {v: k for k, v in self.meta['table_to_id'].items()}
        for hub_tbl_name, vn_idx in self.meta['local_vn_to_id'].items():
            cols = schema_info["columns"].get(hub_tbl_name, [])
            col_ids = [col_to_id[f"{hub_tbl_name}.{c['name']}"]
                       for c in cols if f"{hub_tbl_name}.{c['name']}" in col_to_id]
            if not col_ids:
                continue
            expected = self.data['column'].x[col_ids].mean(dim=0)
            actual = self.data['local_vn'].x[vn_idx]
            self.assertTrue(torch.allclose(actual, expected, atol=1e-5),
                            msg=f"local_vn feature mismatch for {hub_tbl_name}")

    def test_4_fk_reachability_preserved(self):
        missing = REQUIRED_METADATA_KEYS - set(self.meta.keys())
        self.assertFalse(missing, msg=f"missing required keys: {missing}")
        for k in ('hub_tables', 'hub_threshold', 'hub_strategy',
                  'local_vn_to_id', 'local_vn_flat_offset', 'table_col_count'):
            self.assertIn(k, self.meta, msg=f"missing v6w3 key: {k}")
        self.assertEqual(self.meta['v6w3_variant'], 'C')

    def test_5_pcst_flat_indexing_consistency(self):
        T = len(self.meta['table_to_id'])
        C = len(self.meta['col_to_id'])
        F = len(self.meta['fk_to_id'])
        H = len(self.meta['hub_tables'])
        offset = self.meta['local_vn_flat_offset']
        self.assertEqual(offset, T + C + F)
        total = T + C + F + H
        for (s, d), et in zip(self.meta['edges'], self.meta['edge_types']):
            self.assertTrue(0 <= s < total, msg=f"src out of range: {s} (et={et})")
            self.assertTrue(0 <= d < total, msg=f"dst out of range: {d} (et={et})")
        # node_metadata 위 local_vn 등록 (only when H>0)
        for hub_tbl_name, vn_local in self.meta['local_vn_to_id'].items():
            flat = offset + vn_local
            self.assertIn(flat, self.meta['node_metadata'])
            self.assertIn('__local_vn__', self.meta['node_metadata'][flat])


@unittest.skipUnless(
    _check_db_available() and os.path.exists(
        os.path.join(DB_DIR, HUB_DB, f"{HUB_DB}.sqlite")),
    f"BIRD-Dev {HUB_DB}.sqlite 미존재 — skip"
)
class TestVariantC_HubDB(unittest.TestCase):
    """european_football_2 — RFP H2 natural test bed (115-col Player table)."""

    @classmethod
    def setUpClass(cls):
        cls.builder = V6W3HubLocalVNBuilder(tables_json_path=TABLES_JSON)
        cls.data, cls.meta = cls.builder.build(HUB_DB, DB_DIR)

    def test_hub_includes_largest_table(self):
        """gold-deg table (e.g., Player 115-col) 가 hub set 위 포함"""
        counts = self.meta['table_col_count']
        id_to_table = {v: k for k, v in self.meta['table_to_id'].items()}
        # largest table 식별
        max_count = max(counts.values())
        largest_idxs = [i for i, c in counts.items() if c == max_count]
        for idx in largest_idxs:
            self.assertIn(idx, self.meta['hub_tables'],
                          msg=f"largest table {id_to_table[idx]} ({max_count} cols) "
                              f"should be hub")

    def test_hub_count_nonzero(self):
        """multi-table DB 위 hub 1개 이상"""
        if len(self.meta['table_to_id']) >= 3:
            self.assertGreater(len(self.meta['hub_tables']), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
