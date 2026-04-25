"""Proposal C H2 smoke test — per-DB dynamic num_layers.

Goal: verify that with num_layers_mode='D_max' the EnsembleSelector resolves a
different forward depth per db_id at inference time, and that the resolved depth
matches data/processed/dev_diameter.pt. Also exercise the gat_version='v1' and
gat_version='v2' branches to confirm both load the existing
best_gat_qcond_nl6.pt checkpoint and run forward without retraining.

Run from project root:
    conda run -n base python src/modules/selectors/tests/test_h2_per_db_dynamic.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch
from torch_geometric.data import HeteroData

DIAMETER_PATH = os.path.join(ROOT, "data", "processed", "dev_diameter.pt")
CKPT_PATH = os.path.join(ROOT, "outputs", "checkpoints", "best_gat_qcond_nl6.pt")


def _build_synthetic_graph(num_tables: int = 3, num_cols_per_table: int = 4,
                           num_fk: int = 2, in_dim: int = 384) -> HeteroData:
    """Minimal HeteroData matching the schema the selector expects.

    Node types: table / column / fk_node.
    Edge types: has_column / belongs_to / is_source_of / points_to / table_to_table.
    """
    total_cols = num_tables * num_cols_per_table
    data = HeteroData()
    data['table'].x = torch.randn(num_tables, in_dim)
    data['column'].x = torch.randn(total_cols, in_dim)
    data['fk_node'].x = torch.randn(num_fk, in_dim)

    # table ↔ column (round-robin assignment)
    t_src, c_dst = [], []
    for t in range(num_tables):
        for j in range(num_cols_per_table):
            c = t * num_cols_per_table + j
            t_src.append(t)
            c_dst.append(c)
    tc = torch.tensor([t_src, c_dst], dtype=torch.long)
    data['table', 'has_column', 'column'].edge_index = tc
    data['column', 'belongs_to', 'table'].edge_index = tc.flip(0)

    # column ↔ fk_node (first num_fk columns)
    fk_src = list(range(num_fk))
    fk_dst = list(range(num_fk))
    fke = torch.tensor([fk_src, fk_dst], dtype=torch.long)
    data['column', 'is_source_of', 'fk_node'].edge_index = fke
    data['fk_node', 'points_to', 'column'].edge_index = fke.flip(0)

    # table_to_table: linear chain
    if num_tables > 1:
        t2t_src = list(range(num_tables - 1))
        t2t_dst = list(range(1, num_tables))
        data['table', 'table_to_table', 'table'].edge_index = torch.tensor(
            [t2t_src + t2t_dst, t2t_dst + t2t_src], dtype=torch.long
        )
    else:
        data['table', 'table_to_table', 'table'].edge_index = torch.zeros((2, 0), dtype=torch.long)

    return data


def _node_metadata_for(data: HeteroData) -> dict:
    """Stub node_metadata so candidates length matches the graph."""
    n_t = data['table'].num_nodes
    n_c = data['column'].num_nodes
    n_fk = data['fk_node'].num_nodes
    meta = {}
    idx = 0
    for t in range(n_t):
        meta[idx] = {"type": "table", "name": f"table_{t}"}
        idx += 1
    for c in range(n_c):
        meta[idx] = {"type": "column", "name": f"col_{c}"}
        idx += 1
    for f in range(n_fk):
        meta[idx] = {"type": "fk_node", "name": f"fk_{f}"}
        idx += 1
    return meta


def _make_selector(gat_version: str, num_layers_mode: str = "D_max"):
    from modules.selectors.ensemble_selector import EnsembleSelector
    return EnsembleSelector(
        weight_path=CKPT_PATH,
        alpha=0.0,
        top_k=5,
        in_channels=384,
        hidden_channels=256,
        out_channels=256,
        num_layers=6,
        query_conditioned=True,
        query_supernode=False,
        encoder_type="plm",
        num_layers_mode=num_layers_mode,
        diameter_cache_path=DIAMETER_PATH,
        num_layers_fallback=6,
        gat_version=gat_version,
    )


def test_diameter_dict_loaded():
    assert os.path.exists(DIAMETER_PATH), f"missing {DIAMETER_PATH}"
    d = torch.load(DIAMETER_PATH, map_location="cpu")
    if isinstance(d, dict) and "diameters" in d:
        d = d["diameters"]
    assert isinstance(d, dict) and len(d) >= 11
    vals = sorted(set(d.values()))
    print(f"[test_diameter_dict_loaded] {len(d)} DBs, unique D_max values: {vals}")
    assert len(vals) >= 2, "need at least 2 distinct D_max values to prove per-DB variation"


def test_h2_resolve_and_forward(gat_version: str):
    print(f"\n[test_h2_resolve_and_forward] gat_version={gat_version}")
    sel = _make_selector(gat_version=gat_version, num_layers_mode="D_max")
    dict_diams = sel.diameter_dict
    assert len(dict_diams) >= 11

    graph = _build_synthetic_graph()
    node_md = _node_metadata_for(graph)
    candidates = list(range(len(node_md)))

    # Pick three DBs with distinct D_max to exercise the routing.
    pairs = sorted(dict_diams.items(), key=lambda kv: kv[1])
    small = pairs[0]          # min D_max
    mid = pairs[len(pairs) // 2]
    big = pairs[-1]           # max D_max

    picks = [small, mid, big]
    print(f"  picks (db_id, D_max): {picks}")

    resolved = []
    for db_id, d_max in picks:
        metadata = {"db_id": db_id, "node_metadata": node_md}
        cloned = graph.clone()  # forward edits via .to(device)
        seeds = sel.select(
            scores=[0.0] * len(candidates),
            candidates=candidates,
            question="which students passed their exams?",
            graph_data=cloned,
            metadata=metadata,
        )
        depth = sel.last_resolved_depth
        print(f"    db_id={db_id!r:30s} D_max={d_max} → resolved_depth={depth} "
              f"seeds(head)={seeds[:3]}")
        expected = min(d_max, sel.gat_model.num_layers)
        assert depth == expected, f"expected depth={expected}, got {depth}"
        resolved.append(depth)

    assert len(set(resolved)) >= 2, f"expected varied depths across DBs, got {resolved}"


def test_fixed_mode_unchanged():
    """Baseline: num_layers_mode='fixed' should use full num_layers regardless of db_id."""
    print("\n[test_fixed_mode_unchanged]")
    from modules.selectors.ensemble_selector import EnsembleSelector
    sel = EnsembleSelector(
        weight_path=CKPT_PATH, alpha=0.0, top_k=5,
        num_layers=6, query_conditioned=True, encoder_type="plm",
        num_layers_mode="fixed", gat_version="v1",
    )
    graph = _build_synthetic_graph()
    node_md = _node_metadata_for(graph)
    metadata = {"db_id": "california_schools", "node_metadata": node_md}
    sel.select(
        scores=[0.0] * len(node_md),
        candidates=list(range(len(node_md))),
        question="q",
        graph_data=graph.clone(),
        metadata=metadata,
    )
    # fixed mode → _resolve_active_depth returns None → forward uses full self.num_layers
    assert sel.last_resolved_depth is None, f"fixed mode must not resolve, got {sel.last_resolved_depth}"
    print(f"  OK fixed mode → last_resolved_depth=None (forward uses num_layers={sel.gat_model.num_layers})")


def test_unknown_db_fallback():
    """Unknown db_id → num_layers_fallback."""
    print("\n[test_unknown_db_fallback]")
    sel = _make_selector(gat_version="v1", num_layers_mode="D_max")
    graph = _build_synthetic_graph()
    node_md = _node_metadata_for(graph)
    metadata = {"db_id": "__does_not_exist__", "node_metadata": node_md}
    sel.select(
        scores=[0.0] * len(node_md),
        candidates=list(range(len(node_md))),
        question="q",
        graph_data=graph.clone(),
        metadata=metadata,
    )
    assert sel.last_resolved_depth == sel.num_layers_fallback
    print(f"  OK unknown db_id → depth={sel.last_resolved_depth} (fallback)")


def main():
    test_diameter_dict_loaded()
    test_h2_resolve_and_forward(gat_version="v1")
    test_h2_resolve_and_forward(gat_version="v2")
    test_fixed_mode_unchanged()
    test_unknown_db_fallback()
    print("\nAll H2 smoke tests passed.")


if __name__ == "__main__":
    main()
