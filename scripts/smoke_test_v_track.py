"""V-Track smoke test (2026-04-21 QCondGAT v2 계열).

Verifies three orthogonal extensions on SchemaHeteroGAT / SchemaHeteroGATv2:

  V-1  num_layers_mode ∈ {fixed, D_max, D_max_plus1}
       — Per-DB dynamic depth via diameter lookup.

  V-2  supernode_edge_direction ∈ {bidirectional, directed_from_sn}
       — directed_from_sn 시 schema→SN edge 삭제, SN 은 self-loop 로 보존.

  V-3  supernode_topk + supernode_topk_criterion='raw'
       — SN → top-k schema (pre-GAT cosine) 만 연결.

그리고 EnsembleSelector.get_raw_scores (V-3 에서 재사용 가능한 utility) 의 shape sanity 확인.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.gat_network import SchemaHeteroGAT  # noqa: E402
from models.gat_network_v2 import SchemaHeteroGATv2  # noqa: E402
from modules.selectors.ensemble_selector import EnsembleSelector  # noqa: E402


torch.manual_seed(0)

# ──────────────────────────────────────────────────────────────
# Synthetic schema hetero graph
# ──────────────────────────────────────────────────────────────
IN = 16
HID = 8
OUT = 8
HEADS = 2
N_T = 4
N_C = 6
N_FK = 2


def build_synth_inputs(query_supernode: bool):
    x_dict = {
        "table": torch.randn(N_T, IN),
        "column": torch.randn(N_C, IN),
        "fk_node": torch.randn(N_FK, IN),
    }
    if query_supernode:
        x_dict["query_node"] = torch.randn(1, IN)

    # base schema edges (arbitrary but valid)
    edge_index_dict = {
        ("table", "has_column", "column"): torch.tensor(
            [[0, 0, 1, 1, 2, 3], [0, 1, 2, 3, 4, 5]], dtype=torch.long
        ),
        ("column", "belongs_to", "table"): torch.tensor(
            [[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 3]], dtype=torch.long
        ),
        ("column", "is_source_of", "fk_node"): torch.tensor(
            [[0, 2], [0, 1]], dtype=torch.long
        ),
        ("fk_node", "points_to", "column"): torch.tensor(
            [[0, 1], [3, 5]], dtype=torch.long
        ),
        ("table", "table_to_table", "table"): torch.tensor(
            [[0, 1], [1, 2]], dtype=torch.long
        ),
    }
    if query_supernode:
        # fully connected SN↔schema initially; model may filter via V-3.
        for nt, n in (("table", N_T), ("column", N_C), ("fk_node", N_FK)):
            src = torch.zeros(n, dtype=torch.long)
            dst = torch.arange(n, dtype=torch.long)
            edge_index_dict[("query_node", f"attends_to_{nt}", nt)] = torch.stack(
                [src, dst], dim=0
            )
            edge_index_dict[(nt, f"attended_by_{nt}", "query_node")] = torch.stack(
                [dst, src], dim=0
            )
    return x_dict, edge_index_dict


def _fresh_edges(query_supernode: bool):
    _, eidx = build_synth_inputs(query_supernode)
    return eidx


# ──────────────────────────────────────────────────────────────
# V-2 checks
# ──────────────────────────────────────────────────────────────
def test_v2_bidirectional_baseline():
    x_dict, eidx = build_synth_inputs(query_supernode=True)
    model = SchemaHeteroGAT(
        in_channels=IN, hidden_channels=HID, out_channels=OUT,
        num_layers=2, heads=HEADS, query_supernode=True,
        supernode_edge_direction="bidirectional",
    )
    q = torch.randn(1, IN)
    out = model(x_dict, eidx, query_emb=q)
    assert "query_node" in out and out["query_node"].shape == (1, OUT)
    assert out["table"].shape == (N_T, OUT)
    print("  [V-2] bidirectional: all node types present ✓")


def test_v2_directed_from_sn():
    x_dict, eidx = build_synth_inputs(query_supernode=True)
    # Strip reverse edges (schema→query) so we can verify model doesn't depend on them.
    for nt in ("table", "column", "fk_node"):
        eidx.pop((nt, f"attended_by_{nt}", "query_node"), None)

    model = SchemaHeteroGAT(
        in_channels=IN, hidden_channels=HID, out_channels=OUT,
        num_layers=2, heads=HEADS, query_supernode=True,
        supernode_edge_direction="directed_from_sn",
    )
    # Internal registration: self-loop conv must exist, reverse convs must not.
    reg_keys = set(model.convs[0].convs.keys())
    assert ("query_node", "self_loop", "query_node") in reg_keys, \
        "directed_from_sn must register SN self-loop conv"
    for nt in ("table", "column", "fk_node"):
        assert (nt, f"attended_by_{nt}", "query_node") not in reg_keys, \
            f"directed_from_sn must NOT register ({nt}, attended_by_{nt}, query_node)"

    q = torch.randn(1, IN)
    out = model(x_dict, eidx, query_emb=q)
    # query_node must survive (self-loop injected in forward).
    assert "query_node" in out and out["query_node"].shape == (1, OUT), \
        f"query_node missing from output (directed_from_sn). keys: {list(out.keys())}"
    print("  [V-2] directed_from_sn: SN preserved via self-loop, no reverse edges ✓")


# ──────────────────────────────────────────────────────────────
# V-3 checks
# ──────────────────────────────────────────────────────────────
def test_v3_topk_edge_filtering():
    x_dict, eidx = build_synth_inputs(query_supernode=True)
    K = 3  # keep 3 schema nodes globally
    model = SchemaHeteroGAT(
        in_channels=IN, hidden_channels=HID, out_channels=OUT,
        num_layers=2, heads=HEADS, query_supernode=True,
        supernode_edge_direction="bidirectional",
        supernode_topk=K, supernode_topk_criterion="raw",
    )
    q = torch.randn(1, IN)
    masks = model._compute_topk_mask(q, x_dict)
    total_kept = sum(int(m.sum()) for m in masks.values())
    assert total_kept == K, f"top-{K} filter kept {total_kept} nodes (expected {K})"

    filtered_eidx = model._apply_topk_to_edges(eidx, masks)
    fwd_total = 0
    rev_total = 0
    for nt in ("table", "column", "fk_node"):
        fwd = filtered_eidx[("query_node", f"attends_to_{nt}", nt)]
        rev = filtered_eidx[(nt, f"attended_by_{nt}", "query_node")]
        fwd_total += int(fwd.size(1))
        rev_total += int(rev.size(1))
    assert fwd_total == K, f"expected {K} forward edges, got {fwd_total}"
    assert rev_total == K, f"expected {K} reverse edges, got {rev_total}"

    # End-to-end forward still runs.
    out = model(x_dict, eidx, query_emb=q)
    assert out["table"].shape == (N_T, OUT)
    print(f"  [V-3] top-{K} edge filter: fwd={fwd_total}, rev={rev_total} ✓")


def test_v3_topk_forbidden_without_supernode():
    try:
        SchemaHeteroGAT(
            in_channels=IN, hidden_channels=HID, out_channels=OUT,
            num_layers=2, heads=HEADS, query_supernode=False,
            supernode_topk=5,
        )
    except ValueError:
        print("  [V-3] topk without query_supernode raises ValueError ✓")
        return
    raise AssertionError("supernode_topk w/o query_supernode must raise ValueError")


# ──────────────────────────────────────────────────────────────
# V-1 checks (gat_network_v2)
# ──────────────────────────────────────────────────────────────
def test_v1_fixed_mode():
    diam = {"db_small": 2, "db_med": 4, "db_big": 9}
    m = SchemaHeteroGATv2(
        in_channels=IN, hidden_channels=HID, out_channels=OUT,
        num_layers=5, heads=HEADS,
        num_layers_mode="fixed", diameter_dict=diam,
    )
    for db in ["db_small", "db_med", "db_big", "missing"]:
        assert m.resolve_num_layers(db_name=db) == 5
    print("  [V-1] fixed: always 5 ✓")


def test_v1_d_max_mode():
    diam = {"db_small": 2, "db_med": 4, "db_big": 9}
    m = SchemaHeteroGATv2(
        in_channels=IN, hidden_channels=HID, out_channels=OUT,
        num_layers=5, heads=HEADS,
        num_layers_mode="D_max", diameter_dict=diam, num_layers_fallback=3,
    )
    assert m.resolve_num_layers(db_name="db_small") == 2
    assert m.resolve_num_layers(db_name="db_med") == 4
    assert m.resolve_num_layers(db_name="db_big") == 5  # clamped to num_layers_max
    assert m.resolve_num_layers(db_name="missing") == 3  # fallback
    print("  [V-1] D_max: clamp to num_layers_max + fallback ✓")


def test_v1_d_max_plus1_mode():
    diam = {"db_small": 2, "db_med": 4, "db_big": 9}
    m = SchemaHeteroGATv2(
        in_channels=IN, hidden_channels=HID, out_channels=OUT,
        num_layers=5, heads=HEADS,
        num_layers_mode="D_max_plus1", diameter_dict=diam,
    )
    assert m.resolve_num_layers(db_name="db_small") == 3  # 2+1
    assert m.resolve_num_layers(db_name="db_med") == 5  # 4+1, clamped
    print("  [V-1] D_max_plus1: D+1 with clamp ✓")


def test_v1_forward_uses_resolved_depth():
    diam = {"db_small": 1, "db_big": 3}
    m = SchemaHeteroGATv2(
        in_channels=IN, hidden_channels=HID, out_channels=OUT,
        num_layers=4, heads=HEADS,
        query_supernode=True, num_layers_mode="D_max", diameter_dict=diam,
    )
    x_dict, eidx = build_synth_inputs(query_supernode=True)
    q = torch.randn(1, IN)
    out_small = m(x_dict, eidx, query_emb=q, db_name="db_small")
    out_big = m(x_dict, eidx, query_emb=q, db_name="db_big")
    assert out_small["table"].shape == (N_T, OUT)
    assert out_big["table"].shape == (N_T, OUT)
    # Also test explicit override
    out_ov = m(x_dict, eidx, query_emb=q, active_num_layers=2)
    assert out_ov["table"].shape == (N_T, OUT)
    print("  [V-1] forward with db_name / active_num_layers override ✓")


def test_v1_diameter_path_load(tmp_dir: str):
    """diameter_path 로부터 torch.load 로 dict 를 읽어오는 경로."""
    diam_file = os.path.join(tmp_dir, "diam.pt")
    torch.save({"db_a": 3, "db_b": 6}, diam_file)
    m = SchemaHeteroGATv2(
        in_channels=IN, hidden_channels=HID, out_channels=OUT,
        num_layers=5, heads=HEADS,
        num_layers_mode="D_max", diameter_path=diam_file,
    )
    assert m.resolve_num_layers(db_name="db_a") == 3
    assert m.resolve_num_layers(db_name="db_b") == 5  # clamped

    # wrapped form: {"diameters": {...}}
    wrapped_file = os.path.join(tmp_dir, "diam_wrap.pt")
    torch.save({"diameters": {"db_a": 2}}, wrapped_file)
    m2 = SchemaHeteroGATv2(
        in_channels=IN, hidden_channels=HID, out_channels=OUT,
        num_layers=5, heads=HEADS,
        num_layers_mode="D_max", diameter_path=wrapped_file,
    )
    assert m2.resolve_num_layers(db_name="db_a") == 2
    print("  [V-1] diameter_path torch.load (raw + wrapped) ✓")


# ──────────────────────────────────────────────────────────────
# v2 integration: V-1 + V-2 + V-3
# ──────────────────────────────────────────────────────────────
def test_v2_integration_all_three():
    m = SchemaHeteroGATv2(
        in_channels=IN, hidden_channels=HID, out_channels=OUT,
        num_layers=4, heads=HEADS,
        query_supernode=True,
        num_layers_mode="D_max_plus1", diameter_dict={"tiny": 1},
        supernode_edge_direction="directed_from_sn",
        supernode_topk=2, supernode_topk_criterion="raw",
        jumping_knowledge="concat",
    )
    x_dict, eidx = build_synth_inputs(query_supernode=True)
    # Strip reverse edges since directed_from_sn.
    for nt in ("table", "column", "fk_node"):
        eidx.pop((nt, f"attended_by_{nt}", "query_node"), None)
    q = torch.randn(1, IN)
    out = m(x_dict, eidx, query_emb=q, db_name="tiny")  # depth = min(1+1, 4)=2
    assert out["query_node"].shape == (1, OUT)
    assert out["table"].shape == (N_T, OUT)
    print("  [V-1+V-2+V-3] combined forward + JK=concat ✓")


# ──────────────────────────────────────────────────────────────
# Ensemble.get_raw_scores utility
# ──────────────────────────────────────────────────────────────
def test_get_raw_scores_shape():
    q = torch.randn(IN)
    nodes = torch.randn(10, IN)
    s = EnsembleSelector.get_raw_scores(q, nodes)
    assert s.shape == (10,) and s.abs().max().item() <= 1.0 + 1e-6
    # Also accepts [1,d] and [K,d] query.
    s2 = EnsembleSelector.get_raw_scores(q.unsqueeze(0), nodes)
    assert s2.shape == (10,)
    s3 = EnsembleSelector.get_raw_scores(torch.randn(3, IN), nodes)
    assert s3.shape == (10,)
    print("  [Ensemble.get_raw_scores] shape [N] for 1d/2d/multi-token query ✓")


# ──────────────────────────────────────────────────────────────
def main():
    import tempfile

    print("=" * 72)
    print("V-Track smoke test (V-1 num_layers_mode, V-2 edge_direction, V-3 topk)")
    print("=" * 72)

    print("\n[gat_network.py — V-2 / V-3]")
    test_v2_bidirectional_baseline()
    test_v2_directed_from_sn()
    test_v3_topk_edge_filtering()
    test_v3_topk_forbidden_without_supernode()

    print("\n[gat_network_v2.py — V-1]")
    test_v1_fixed_mode()
    test_v1_d_max_mode()
    test_v1_d_max_plus1_mode()
    test_v1_forward_uses_resolved_depth()
    with tempfile.TemporaryDirectory() as td:
        test_v1_diameter_path_load(td)

    print("\n[gat_network_v2.py — integration]")
    test_v2_integration_all_three()

    print("\n[EnsembleSelector — V-3 utility]")
    test_get_raw_scores_shape()

    print("\n" + "=" * 72)
    print("[OK] V-Track smoke test passed")
    print("=" * 72)


if __name__ == "__main__":
    main()
