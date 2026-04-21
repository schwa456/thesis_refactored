"""B-II smoke test: LineGraphBuilder structural sanity."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from modules.builders.line_graph_builder import LineGraphBuilder, EDGE_TYPE_ORDER

DB_DIR = "data/raw/BIRD_dev/dev_databases"
TABLES_JSON = "data/raw/BIRD_dev/dev_tables.json"


def check(db_id: str, base_name: str, base_params=None):
    builder = LineGraphBuilder(base=base_name, base_params=base_params)
    line_data, line_meta = builder.build(db_id, DB_DIR)

    n_edge_nodes = line_data["edge_node"].x.size(0)
    n_edges_in_orig = len(line_meta["orig_metadata"]["edges"])
    if not builder.skip_macro_edges:
        assert n_edge_nodes == n_edges_in_orig, (
            f"[{db_id}/{base_name}] edge_node count {n_edge_nodes} != "
            f"orig edges {n_edges_in_orig}"
        )
    assert "shares_node" in [r for _, r, _ in line_data.edge_types], (
        f"[{db_id}/{base_name}] missing shares_node edge type"
    )
    line_edge_idx = line_data["edge_node", "shares_node", "edge_node"].edge_index
    assert line_edge_idx.dim() == 2 and line_edge_idx.size(0) == 2
    # FK reachability forwarded
    for k in ("fk_reachability", "fk_components", "edge_node_to_orig",
              "orig_node_to_edges", "edge_type_order"):
        assert k in line_meta, f"[{db_id}/{base_name}] missing line_meta key: {k}"
    print(f"  {base_name:32s}  edge_nodes={n_edge_nodes:4d}  "
          f"line_edges={line_edge_idx.size(1):5d}  "
          f"feat_dim={line_meta['edge_feature_dim']:4d}")


def main():
    print("=" * 72)
    print("B-II smoke test (LineGraphBuilder)")
    print("=" * 72)
    print(f"\nedge_type_order = {EDGE_TYPE_ORDER}\n")

    print("[california_schools]")
    check("california_schools", "HeteroGraphBuilder")
    check("california_schools", "EnrichedHeteroGraphBuilder",
          base_params={"tables_json_path": TABLES_JSON})
    check("california_schools", "TripletGraphBuilder",
          base_params={"tables_json_path": TABLES_JSON})

    print("\n[financial — larger schema]")
    check("financial", "EnrichedHeteroGraphBuilder",
          base_params={"tables_json_path": TABLES_JSON})

    print("\n[skip_macro_edges variant]")
    builder = LineGraphBuilder(
        base="EnrichedHeteroGraphBuilder",
        base_params={"tables_json_path": TABLES_JSON},
        skip_macro_edges=True,
    )
    line_data, line_meta = builder.build("financial", DB_DIR)
    print(f"  edge_nodes (no macro)   = {line_data['edge_node'].x.size(0)}")

    print("\n[OK] smoke test finished")


if __name__ == "__main__":
    main()
