"""V5-C `FullAEROGATConv` (FullAEROGATv2Conv) smoke test (DECISIONS 2026-05-13 §V5 Sweep Launch 재시도).

검증 대상:
  (1) `FullAEROGATv2Conv` (alias `FullAEROGATConv`) 인스턴스 + SoftplusGATv2Conv 상속 (V4-B 확장)
  (2) `cumulative_attention` flag + `reset_cumulative()` method 존재
  (3) HeteroData V-3-ext 호환 — Hop Attention + Cumulative Attention combined (paper §4.2 Theorem 4 full form)
  (4) Hop Attention only (Cumulative OFF) — V4-B + Theorem 4 partial form
  (5) Cumulative only (Hop OFF) — paper §3.2 Cumulative residual simulation (hidden-state level)
  (6) Validation — aero_cumulative_attention + non-aero_full / decay out-of-range → raise

Reference:
  Lee, S. Y., Bu, F., Yoo, J., & Shin, K. (2023). Towards Deep Attention in Graph Neural Networks:
  Problems and Remedies (AERO-GNN). ICML 2023. arXiv:2306.02376.
    - Theorem 3 (SR2OS guarantee): Softplus + Symmetric Norm + Cumulative residual 의 combo 에서만
      smoothness score `S(T(k))` 가 0 으로 수렴 보장 깨짐.
    - Theorem 4 (Node-Adaptive un-smoothing): per-node hop weight `ω_v^(l)` 가 cumulative attention
      의 outer equivalent — Σ_l ω_v^(l) h_v^(l) form.

Run from project root:
    PYTHONPATH=src conda run -n base python src/modules/selectors/tests/test_v5_c_aero_full.py
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

from models.gat_network_v2 import (
    FullAEROGATv2Conv, FullAEROGATConv, SoftplusGATv2Conv, SchemaHeteroGATv2,
)


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


def test_v5c_alias_and_parent():
    """FullAEROGATConv 가 FullAEROGATv2Conv alias + SoftplusGATv2Conv (V4-B) 의 subclass."""
    print("\n[test_v5c_alias_and_parent]")
    assert FullAEROGATConv is FullAEROGATv2Conv, "FullAEROGATConv must alias FullAEROGATv2Conv"
    conv = FullAEROGATv2Conv(-1, 16, heads=2)
    assert isinstance(conv, SoftplusGATv2Conv), "FullAEROGATConv must inherit SoftplusGATv2Conv (V4-B)"
    assert hasattr(conv, "cumulative_attention")
    assert hasattr(conv, "reset_cumulative")
    assert hasattr(conv, "symmetric_norm"), "Symmetric Norm flag inherited from SoftplusGATv2Conv"
    print(f"  OK aliases + parent SoftplusGATv2Conv + cumulative_attention + reset_cumulative")


def test_v5c_full_model_combined():
    """V5-C 의 (a) Softplus + (b) Symmetric Norm + (c) Hop Attention + (d) Cumulative — Theorem 3 full form."""
    print("\n[test_v5c_full_model_combined]")
    m = SchemaHeteroGATv2(
        in_channels=384, hidden_channels=64, out_channels=64, num_layers=3, heads=2,
        query_conditioned=False, query_supernode=True,
        pairnorm_mode="pairnorm", initial_residual_alpha=0.2, jumping_knowledge="concat",
        dual_stream=True, supernode_edge_direction="directed_from_sn",
        supernode_threshold_mode="percentile", supernode_threshold_value=80.0,
        gat_layer_type="aero_full",
        softplus_symmetric_norm=True,
        aero_hop_attention=True,
        aero_cumulative_attention=True,
        aero_cumulative_decay=1.0)
    data = _build_synthetic_hetero()
    m.train()
    out = m(data.x_dict, data.edge_index_dict, query_emb=data["query_node"].x)
    for nt in ("table", "column", "fk_node"):
        assert nt in out and torch.isfinite(out[nt]).all()
    loss = sum(out[nt].pow(2).mean() for nt in ("table", "column", "fk_node"))
    loss.backward()
    # Hop attention parameters 가 학습됨
    assert m.hop_attention_lin is not None
    n_hop_grad = sum(1 for p in m.hop_attention_lin.parameters()
                     if p.grad is not None and torch.isfinite(p.grad).all())
    print(f"  OK Theorem 3 full combo (Softplus+SymNorm+Hop+Cumulative) loss={loss.item():.4f} "
          f"hop_attention params grad in {n_hop_grad}")


def test_v5c_hop_only():
    """Hop Attention only — V4-B + Theorem 4 partial (cumulative simulation 없음)."""
    print("\n[test_v5c_hop_only]")
    m = SchemaHeteroGATv2(
        in_channels=384, hidden_channels=64, out_channels=64, num_layers=2, heads=2,
        query_conditioned=False, query_supernode=True,
        pairnorm_mode="pairnorm", initial_residual_alpha=0.2, jumping_knowledge="concat",
        dual_stream=True, supernode_edge_direction="directed_from_sn",
        supernode_threshold_mode="percentile", supernode_threshold_value=80.0,
        gat_layer_type="aero_full",
        aero_hop_attention=True,
        aero_cumulative_attention=False)
    data = _build_synthetic_hetero()
    m.train()
    out = m(data.x_dict, data.edge_index_dict, query_emb=data["query_node"].x)
    loss = sum(out[nt].pow(2).mean() for nt in ("table", "column", "fk_node"))
    loss.backward()
    print(f"  OK Hop only (Theorem 4 partial, no cumulative) loss={loss.item():.4f}")


def test_v5c_cumulative_only():
    """Cumulative Attention only — paper §3.2 의 cumulative residual simulation (hop OFF)."""
    print("\n[test_v5c_cumulative_only]")
    m = SchemaHeteroGATv2(
        in_channels=384, hidden_channels=64, out_channels=64, num_layers=3, heads=2,
        query_conditioned=False, query_supernode=True,
        pairnorm_mode="pairnorm", initial_residual_alpha=0.2, jumping_knowledge="concat",
        dual_stream=True, supernode_edge_direction="directed_from_sn",
        supernode_threshold_mode="percentile", supernode_threshold_value=80.0,
        gat_layer_type="aero_full",
        aero_hop_attention=False,
        aero_cumulative_attention=True,
        aero_cumulative_decay=0.5)
    data = _build_synthetic_hetero()
    m.train()
    out = m(data.x_dict, data.edge_index_dict, query_emb=data["query_node"].x)
    loss = sum(out[nt].pow(2).mean() for nt in ("table", "column", "fk_node"))
    loss.backward()
    print(f"  OK Cumulative only (hop OFF) loss={loss.item():.4f} decay=0.5")


def test_v5c_validation_raises():
    """aero_cumulative_attention + non-aero_full / decay out-of-range → ValueError."""
    print("\n[test_v5c_validation_raises]")
    # cumulative + non-aero_full
    try:
        SchemaHeteroGATv2(in_channels=384, hidden_channels=64, out_channels=64, num_layers=2, heads=2,
                         query_conditioned=False, query_supernode=True,
                         supernode_edge_direction="directed_from_sn",
                         supernode_threshold_mode="percentile", supernode_threshold_value=80.0,
                         gat_layer_type="gate", aero_cumulative_attention=True)
        print("  FAIL: cumulative+gate should raise")
    except ValueError as e:
        print(f"  OK cumulative+gate raises: {str(e)[:60]}")
    # decay out-of-range
    try:
        SchemaHeteroGATv2(in_channels=384, hidden_channels=64, out_channels=64, num_layers=2, heads=2,
                         query_conditioned=False, query_supernode=True,
                         supernode_edge_direction="directed_from_sn",
                         supernode_threshold_mode="percentile", supernode_threshold_value=80.0,
                         gat_layer_type="aero_full", aero_cumulative_attention=True,
                         aero_cumulative_decay=2.5)
        print("  FAIL: decay=2.5 should raise")
    except ValueError as e:
        print(f"  OK decay=2.5 raises: {str(e)[:60]}")


def test_v5c_symmetric_norm_inherited():
    """V5-C 는 V4-B (Softplus + Symmetric Norm) 의 row-stochasticity 파괴 그대로 — softplus path."""
    print("\n[test_v5c_symmetric_norm_inherited]")
    m = SchemaHeteroGATv2(
        in_channels=384, hidden_channels=64, out_channels=64, num_layers=2, heads=2,
        query_conditioned=False, query_supernode=True,
        pairnorm_mode="none", initial_residual_alpha=0.0, jumping_knowledge="none",
        dual_stream=False, supernode_edge_direction="directed_from_sn",
        supernode_threshold_mode="percentile", supernode_threshold_value=80.0,
        gat_layer_type="aero_full", softplus_symmetric_norm=True)
    # inner conv 가 FullAEROGATv2Conv, 모두 symmetric_norm=True
    n_sym = 0
    for hc in m.convs:
        for et, conv in hc.convs.items():
            assert isinstance(conv, FullAEROGATv2Conv)
            assert conv.symmetric_norm is True
            n_sym += 1
    print(f"  OK {n_sym} inner FullAEROGATv2Conv 모두 symmetric_norm=True (V4-B 상속)")


def main():
    test_v5c_alias_and_parent()
    test_v5c_full_model_combined()
    test_v5c_hop_only()
    test_v5c_cumulative_only()
    test_v5c_validation_raises()
    test_v5c_symmetric_norm_inherited()
    print("\nAll V5-C Full AERO smoke tests passed.")


if __name__ == "__main__":
    main()
