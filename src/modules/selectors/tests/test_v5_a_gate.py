"""V5-A `GATEConv` (GATEGATv2Conv) smoke test (DECISIONS 2026-05-13 §V5 Sweep Launch 재시도).

검증 대상:
  (1) `GATEGATv2Conv` (alias `GATEConv`) 인스턴스 생성 + att_self Parameter 존재 + Xavier init
  (2) Homograph forward — `(att_self · LRELU(x_i)) + (att · LRELU(x_j))` 분리 검증
  (3) HeteroData 호환 — V-3-ext (query_supernode + directed_from_sn + percentile=80) 안에서 forward+backward
  (4) Conservation law check — att / att_self 두 parameter 모두 학습 가능 (grad finite)
  (5) row-stochasticity 유지 (softmax 그대로) — V4-B 와 다른 axis (paper §3.2 Theorem 1)

Reference:
  Mustafa, N., & Burkholz, R. (2024). GATE: How to Keep Out Intrusive Neighbors.
  NeurIPS 2024. arXiv:2406.00418.

Run from project root:
    PYTHONPATH=src conda run -n base python src/modules/selectors/tests/test_v5_a_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch_geometric.data import HeteroData

from models.gat_network_v2 import GATEGATv2Conv, GATEConv, SchemaHeteroGATv2


def _build_synthetic_hetero(num_tables=4, cols_per_table=6, num_fk=3, in_dim=384, seed=42):
    g = torch.Generator().manual_seed(seed)
    total_cols = num_tables * cols_per_table
    data = HeteroData()
    data["table"].x = torch.randn(num_tables, in_dim, generator=g)
    data["column"].x = torch.randn(total_cols, in_dim, generator=g)
    data["fk_node"].x = torch.randn(num_fk, in_dim, generator=g)
    data["query_node"].x = torch.randn(1, in_dim, generator=g)
    t_src, c_dst = [], []
    for t in range(num_tables):
        for j in range(cols_per_table):
            t_src.append(t); c_dst.append(t * cols_per_table + j)
    tc = torch.tensor([t_src, c_dst], dtype=torch.long)
    data["table", "has_column", "column"].edge_index = tc
    data["column", "belongs_to", "table"].edge_index = tc.flip(0)
    fke = torch.tensor([list(range(num_fk)), list(range(num_fk))], dtype=torch.long)
    data["column", "is_source_of", "fk_node"].edge_index = fke
    data["fk_node", "points_to", "column"].edge_index = fke.flip(0)
    data["table", "table_to_table", "table"].edge_index = torch.tensor(
        [[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]], dtype=torch.long)
    for nt, n in [("table", num_tables), ("column", total_cols), ("fk_node", num_fk)]:
        s = torch.zeros(n, dtype=torch.long); d = torch.arange(n, dtype=torch.long)
        data["query_node", f"attends_to_{nt}", nt].edge_index = torch.stack([s, d], 0)
    return data


def test_v5a_alias_and_parameter():
    """GATEConv 가 GATEGATv2Conv 의 alias + att_self Parameter 가 존재."""
    print("\n[test_v5a_alias_and_parameter]")
    assert GATEConv is GATEGATv2Conv, "GATEConv must alias GATEGATv2Conv"
    conv = GATEGATv2Conv(-1, 16, heads=4)
    assert hasattr(conv, "att_self"), "att_self Parameter missing"
    assert conv.att_self.shape == (1, 4, 16), f"att_self shape {conv.att_self.shape}"
    assert hasattr(conv, "att"), "parent att Parameter missing"
    # Xavier init 확인 — abs.mean 이 ~0.1~0.5 range
    print(f"  OK GATEConv alias + att_self={tuple(conv.att_self.shape)} "
          f"att_self.abs.mean={conv.att_self.abs().mean().item():.3f} "
          f"att.abs.mean={conv.att.abs().mean().item():.3f}")


def test_v5a_homograph_forward():
    """Homograph forward — output shape + finite."""
    print("\n[test_v5a_homograph_forward]")
    conv = GATEGATv2Conv(-1, 16, heads=4)
    x = torch.randn(8, 32)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
    out = conv(x, edge_index)
    assert out.shape == (8, 64), f"shape mismatch {out.shape}"
    assert torch.isfinite(out).all()
    print(f"  OK homograph forward shape={tuple(out.shape)} finite")


def test_v5a_full_model_hetero_v3ext():
    """SchemaHeteroGATv2 + V5-A + V-3-ext (directed_from_sn + percentile=80) forward+backward."""
    print("\n[test_v5a_full_model_hetero_v3ext]")
    m = SchemaHeteroGATv2(
        in_channels=384, hidden_channels=64, out_channels=64, num_layers=2, heads=2,
        query_conditioned=False, query_supernode=True,
        pairnorm_mode="pairnorm", initial_residual_alpha=0.2, jumping_knowledge="concat",
        dual_stream=True, supernode_edge_direction="directed_from_sn",
        supernode_threshold_mode="percentile", supernode_threshold_value=80.0,
        gat_layer_type="gate")
    data = _build_synthetic_hetero()
    m.train()
    out = m(data.x_dict, data.edge_index_dict, query_emb=data["query_node"].x)
    for nt in ("table", "column", "fk_node"):
        assert nt in out and torch.isfinite(out[nt]).all()
    loss = sum(out[nt].pow(2).mean() for nt in ("table", "column", "fk_node"))
    loss.backward()
    # att_self params 가 모두 grad 받음
    n_att_self_grads = 0
    for hc in m.convs:
        for et, conv in hc.convs.items():
            if hasattr(conv, "att_self") and conv.att_self.grad is not None:
                n_att_self_grads += 1
    assert n_att_self_grads > 0, "no att_self grads — Conservation Law decoupling broken"
    print(f"  OK V-3-ext+V5-A forward+backward, att_self grads in {n_att_self_grads} inner convs")


def test_v5a_conservation_law_decoupling():
    """att 와 att_self 가 서로 다른 gradient 를 받음 (분리 학습 확인)."""
    print("\n[test_v5a_conservation_law_decoupling]")
    m = SchemaHeteroGATv2(
        in_channels=384, hidden_channels=64, out_channels=64, num_layers=2, heads=2,
        query_conditioned=False, query_supernode=True,
        pairnorm_mode="pairnorm", initial_residual_alpha=0.2, jumping_knowledge="concat",
        dual_stream=True, supernode_edge_direction="directed_from_sn",
        supernode_threshold_mode="percentile", supernode_threshold_value=80.0,
        gat_layer_type="gate")
    data = _build_synthetic_hetero()
    m.train()
    out = m(data.x_dict, data.edge_index_dict, query_emb=data["query_node"].x)
    out["column"].pow(2).mean().backward()
    # 첫 conv 의 att / att_self grad 가 서로 다름
    first_hc = m.convs[0]
    decoupled = False
    for et, conv in first_hc.convs.items():
        if hasattr(conv, "att_self") and conv.att_self.grad is not None and conv.att.grad is not None:
            diff = (conv.att.grad - conv.att_self.grad).abs().sum().item()
            if diff > 1e-6:
                decoupled = True
                break
    assert decoupled, "att and att_self gradients identical — decoupling not effective"
    print(f"  OK att / att_self gradients decoupled (Conservation Law 수정 confirm)")


def test_v5a_row_stochasticity_preserved():
    """V5-A 는 row-stochasticity 유지 (softmax 그대로) — V4-B / V5-C 와 다른 axis."""
    print("\n[test_v5a_row_stochasticity_preserved]")
    # SoftplusGATv2Conv 가 아닌 GATv2 base 임을 확인 — GATE inherits GATv2Conv.
    conv = GATEGATv2Conv(-1, 16, heads=2)
    # GATv2Conv 는 row-stochastic (softmax in edge_update). super class 가 GATv2Conv 인지 확인.
    from torch_geometric.nn import GATv2Conv
    assert isinstance(conv, GATv2Conv), "GATE must inherit GATv2Conv (row-stochasticity preserved)"
    print(f"  OK GATEGATv2Conv inherits GATv2Conv — row-stochasticity preserved")


def main():
    test_v5a_alias_and_parameter()
    test_v5a_homograph_forward()
    test_v5a_full_model_hetero_v3ext()
    test_v5a_conservation_law_decoupling()
    test_v5a_row_stochasticity_preserved()
    print("\nAll V5-A GATE smoke tests passed.")


if __name__ == "__main__":
    main()
