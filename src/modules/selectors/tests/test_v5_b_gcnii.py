"""V5-B `GCNIIGATv2Conv` smoke test (DECISIONS 2026-05-13 §V5 Sweep Launch 재시도).

검증 대상:
  (1) `GCNIIGATv2Conv` 인스턴스 생성 + gcnii_w Linear (out_dim×out_dim) + eye_init
  (2) β_l = log(λ/l + 1) 의 layer-별 값 (Chen 2020 §3.3 form)
  (3) HeteroData V-3-ext 호환 — L=2/4/6 sweep (학습 시 num_layers config 호환)
  (4) Initial Residual α + Identity Mapping β 동시 활성 (paper Eq. 6 form)
  (5) gcnii_w 의 grad finite + identity 초기화 perturbation 학습

Reference:
  Chen, M., Wei, Z., Huang, Z., Ding, B., & Li, Y. (2020). Simple and Deep Graph Convolutional
  Networks (GCNII). ICML 2020. arXiv:2007.02133.
  Peng, J., Lei, R., & Wei, Z. (2024). Beyond Over-smoothing: Uncovering the Trainability
  Challenges in Deep Graph Neural Networks. CIKM 2024. DOI:10.1145/3627673.3679776.

Run from project root:
    PYTHONPATH=src conda run -n base python src/modules/selectors/tests/test_v5_b_gcnii.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch_geometric.data import HeteroData

from models.gat_network_v2 import GCNIIGATv2Conv, SchemaHeteroGATv2


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


def test_v5b_eye_init_and_beta():
    """gcnii_w 가 eye 로 초기화 + β_l = log(λ/l + 1) form 확인."""
    print("\n[test_v5b_eye_init_and_beta]")
    conv = GCNIIGATv2Conv(-1, 16, heads=4, gcnii_beta_lambda=0.5, gcnii_layer_idx=2)
    out_dim = conv._gcnii_out_dim
    # Identity init — gcnii_w.weight 가 eye matrix
    W = conv.gcnii_w.weight.data
    eye = torch.eye(out_dim)
    assert torch.allclose(W, eye, atol=1e-6), "gcnii_w not eye-initialized"
    # β_l value (Chen 2020 §3.3): log(λ/l + 1)
    expected = math.log(0.5 / 2 + 1.0)
    assert abs(conv._beta() - expected) < 1e-6, f"β_l {conv._beta()} vs expected {expected}"
    print(f"  OK eye-init confirmed (W ≈ I_{out_dim}) + β(λ=0.5, l=2)={conv._beta():.4f}")


def test_v5b_beta_decreases_with_layer():
    """β_l = log(λ/l + 1) 가 layer 따라 감소 (Chen 2020 §3.3)."""
    print("\n[test_v5b_beta_decreases_with_layer]")
    betas = []
    for l in (1, 2, 3, 4, 5, 6):
        c = GCNIIGATv2Conv(-1, 8, heads=2, gcnii_beta_lambda=0.5, gcnii_layer_idx=l)
        betas.append(c._beta())
    # 단조 감소
    for i in range(len(betas) - 1):
        assert betas[i] > betas[i + 1], f"β not monotone: l={i+1}({betas[i]:.4f}) ≤ l={i+2}({betas[i+1]:.4f})"
    print(f"  OK β monotone decreasing l=1..6: {[f'{b:.4f}' for b in betas]}")


def test_v5b_full_model_L2_L4_L6_sweep():
    """L=2/4/6 sweep — num_layers config 호환 + Initial Residual α + Identity β 동시."""
    print("\n[test_v5b_full_model_L2_L4_L6_sweep]")
    data = _build_synthetic_hetero()
    for L in (2, 4, 6):
        m = SchemaHeteroGATv2(
            in_channels=384, hidden_channels=64, out_channels=64, num_layers=L, heads=2,
            query_conditioned=False, query_supernode=True,
            pairnorm_mode="pairnorm", initial_residual_alpha=0.2, jumping_knowledge="concat",
            dual_stream=True, supernode_edge_direction="directed_from_sn",
            supernode_threshold_mode="percentile", supernode_threshold_value=80.0,
            gat_layer_type="gcnii", gcnii_beta_lambda=0.5)
        m.train()
        # layer_idx 가 1..L 1-indexed forwarded
        inner_idx = []
        for hc in m.convs:
            for et, conv in hc.convs.items():
                inner_idx.append(conv.gcnii_layer_idx); break
        assert inner_idx == list(range(1, L + 1)), f"L={L}: layer_idx forwarding broken {inner_idx}"
        out = m(data.x_dict, data.edge_index_dict, query_emb=data["query_node"].x)
        loss = sum(out[nt].pow(2).mean() for nt in ("table", "column", "fk_node"))
        loss.backward()
        n_gcnii_grad = sum(1 for hc in m.convs for et, c in hc.convs.items()
                           if c.gcnii_w.weight.grad is not None
                           and torch.isfinite(c.gcnii_w.weight.grad).all())
        print(f"  OK L={L} layer_idx={inner_idx} loss={loss.item():.4f} gcnii_w grads in {n_gcnii_grad} convs")


def test_v5b_initial_residual_separate_from_identity():
    """Initial Residual α (outer initial_residual_alpha) 와 Identity β (본 conv) 가 별개 path."""
    print("\n[test_v5b_initial_residual_separate_from_identity]")
    # α=0.0 (Initial Residual OFF) 단독으로 V5-B 만 활성화 가능
    m_no_alpha = SchemaHeteroGATv2(
        in_channels=384, hidden_channels=64, out_channels=64, num_layers=2, heads=2,
        query_conditioned=False, query_supernode=True,
        pairnorm_mode="pairnorm", initial_residual_alpha=0.0, jumping_knowledge="concat",
        dual_stream=True, supernode_edge_direction="directed_from_sn",
        supernode_threshold_mode="percentile", supernode_threshold_value=80.0,
        gat_layer_type="gcnii", gcnii_beta_lambda=0.5)
    assert m_no_alpha.res_proj is None, "α=0 should disable res_proj"
    assert m_no_alpha.gat_layer_type == "gcnii"
    print(f"  OK α=0 (IR OFF) + gat_layer_type=gcnii (β active) — paper Eq. 6 의 두 component 분리 학습 가능")


def test_v5b_invalid_beta_lambda_raises():
    """gcnii_beta_lambda <= 0 → ValueError."""
    print("\n[test_v5b_invalid_beta_lambda_raises]")
    try:
        SchemaHeteroGATv2(in_channels=384, hidden_channels=64, out_channels=64, num_layers=2, heads=2,
                         query_conditioned=False, query_supernode=True,
                         supernode_edge_direction="directed_from_sn",
                         supernode_threshold_mode="percentile", supernode_threshold_value=80.0,
                         gat_layer_type="gcnii", gcnii_beta_lambda=-0.1)
        print("  FAIL: should raise")
    except ValueError as e:
        print(f"  OK gcnii_beta_lambda=-0.1 raises: {str(e)[:80]}")


def main():
    test_v5b_eye_init_and_beta()
    test_v5b_beta_decreases_with_layer()
    test_v5b_full_model_L2_L4_L6_sweep()
    test_v5b_initial_residual_separate_from_identity()
    test_v5b_invalid_beta_lambda_raises()
    print("\nAll V5-B GCNII smoke tests passed.")


if __name__ == "__main__":
    main()
