"""B-II.b smoke test — base heterograph T2T edge toggle.

Verifies that `add_t2t_edges=False` strips the
`(table, table_to_table, table)` edges from the base heterograph and the
PCST flat representation, while leaving FK reachability and the schema
diameter intact (FK reachability uses fk_adjacency only; T2T removal
should not change connectivity).

Usage:
    conda run -n base python scripts/smoke_test_b2b_no_t2t.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from modules.builders.graph_builder import (
    HeteroGraphBuilder,
    EnrichedHeteroGraphBuilder,
)

DB_DIR = "/home/hyeonjin/thesis_refactored/data/raw/BIRD_dev/dev_databases"
TABLES_JSON = "/home/hyeonjin/thesis_refactored/data/raw/BIRD_dev/dev_tables.json"
DB_ID = "california_schools"


def count_t2t(metadata):
    return sum(1 for et in metadata["edge_types"] if et == "table_to_table")


def diff(a, b, label):
    print(f"  {label}: on={a}, off={b}, delta={b-a:+d}")


def run(builder_cls, **base_kwargs):
    print(f"\n=== {builder_cls.__name__} ===")
    on = builder_cls(add_t2t_edges=True, **base_kwargs)
    off = builder_cls(add_t2t_edges=False, **base_kwargs)

    d_on, m_on = on.build(DB_ID, DB_DIR)
    d_off, m_off = off.build(DB_ID, DB_DIR)

    t2t_on = count_t2t(m_on)
    t2t_off = count_t2t(m_off)
    diff(t2t_on, t2t_off, "table_to_table edges (PCST flat)")
    diff(len(m_on["edges"]), len(m_off["edges"]), "total PCST edges")

    on_has_t2t = ('table', 'table_to_table', 'table') in d_on.edge_types
    off_has_t2t = ('table', 'table_to_table', 'table') in d_off.edge_types
    print(f"  HeteroData T2T edge_index present: on={on_has_t2t}, off={off_has_t2t}")

    print(f"  metadata['add_t2t_edges']: on={m_on['add_t2t_edges']}, off={m_off['add_t2t_edges']}")

    reach_on = m_on["fk_reachability"]
    reach_off = m_off["fk_reachability"]
    print(f"  FK reachability identical: {np.array_equal(reach_on, reach_off)}")
    print(f"  FK n_components identical: {m_on['fk_num_components'] == m_off['fk_num_components']}")

    diff(m_on["schema_diameter"], m_off["schema_diameter"], "schema_diameter")
    print(f"  schema_diameter identical: {m_on['schema_diameter'] == m_off['schema_diameter']}")

    assert t2t_off == 0, f"add_t2t_edges=False should yield 0 T2T edges, got {t2t_off}"
    assert not off_has_t2t, "HeteroData should not contain T2T edge type when off"
    assert np.array_equal(reach_on, reach_off), "FK reachability must be invariant under T2T toggle"


if __name__ == "__main__":
    run(HeteroGraphBuilder)
    run(EnrichedHeteroGraphBuilder, tables_json_path=TABLES_JSON)
    print("\nB-II.b smoke OK.")
