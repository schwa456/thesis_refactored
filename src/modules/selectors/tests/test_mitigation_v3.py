"""Smoke test for Mitigation v3 #1 GIN-style aggregation (단계 7).

학위 본 심사 5/22 전 진행 (DECISIONS 2026-05-08 §1(c)). Mitigation v2 7-trial null effect 후속,
aggregation function family 자체 변경으로 mech(ii) softmax 한정인지 검증.

검증 대상:
  (1) `_make_gin_conv` factory — PyG GINConv + LazyLinear MLP 동작.
  (2) SchemaHeteroGATv2 + aggregation_type='gin' forward shape — heterograph 호환.
  (3) HeteroConv 호환성 — inner conv = GINConv 로 등록 확인.
  (4) Backward compat — default ('mean') 시 기존 GATv2Conv 사용 (`#5` regression).
  (5) Config parsing — 신규 v3 config 파싱 + Phase 2 baseline 영향 X.
  (6) GIN 모드 + #1/#3 충돌 검증 (raise expected).

Run from project root:
    PYTHONPATH=src conda run -n base python src/modules/selectors/tests/test_mitigation_v3.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import yaml
from torch_geometric.data import HeteroData
from torch_geometric.nn import GINConv


def _build_synthetic_supernode_graph(num_tables=4, cols_per_table=6, num_fk=3,
                                     in_dim=384, seed=42):
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

    if num_tables > 1:
        s = list(range(num_tables - 1)); d = list(range(1, num_tables))
        data["table", "table_to_table", "table"].edge_index = torch.tensor(
            [s + d, d + s], dtype=torch.long)
    else:
        data["table", "table_to_table", "table"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    for nt in ("table", "column", "fk_node"):
        n = data[nt].num_nodes
        src = torch.zeros(n, dtype=torch.long)
        dst = torch.arange(n, dtype=torch.long)
        data["query_node", f"attends_to_{nt}", nt].edge_index = torch.stack([src, dst], 0)
    return data


def _build_model(in_channels=384, hidden=64, heads=2, num_layers=2, **kwargs):
    from models.gat_network_v2 import SchemaHeteroGATv2
    base = dict(
        in_channels=in_channels,
        hidden_channels=hidden,
        out_channels=hidden,
        num_layers=num_layers,
        heads=heads,
        query_conditioned=False,
        query_supernode=True,
        pairnorm_mode="pairnorm",
        initial_residual_alpha=0.2,
        jumping_knowledge="concat",
        dual_stream=True,
        supernode_edge_direction="directed_from_sn",
        supernode_threshold_mode="percentile",
        supernode_threshold_value=80.0,
    )
    base.update(kwargs)
    return SchemaHeteroGATv2(**base)


def test_gin_factory_and_homograph_forward():
    """_make_gin_conv → PyG GINConv 인스턴스. homograph forward 동작."""
    print("\n[test_gin_factory_and_homograph_forward]")
    from models.gat_network_v2 import _make_gin_conv

    conv = _make_gin_conv(in_channels=-1, hidden_channels=16, heads=4)
    assert isinstance(conv, GINConv), f"factory must return GINConv, got {type(conv)}"
    # forward — homo
    x = torch.randn(8, 32)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [3, 4, 5, 6, 7]], dtype=torch.long)
    out = conv(x, edge_index)
    assert out.shape == (8, 16 * 4), f"unexpected shape {out.shape}"
    print(f"  OK GINConv factory + homo forward shape={tuple(out.shape)}")


def test_gin_factory_bipartite_forward():
    """GINConv bipartite (x_src, x_dst) 호환 — HeteroConv 가 사용하는 호출 패턴."""
    print("\n[test_gin_factory_bipartite_forward]")
    from models.gat_network_v2 import _make_gin_conv

    conv = _make_gin_conv(in_channels=-1, hidden_channels=8, heads=2)
    x_src = torch.randn(5, 16)
    x_dst = torch.randn(7, 16)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [3, 4, 5, 6, 0]], dtype=torch.long)
    out = conv((x_src, x_dst), edge_index)
    assert out.shape == (7, 8 * 2), f"bipartite output shape mismatch: {out.shape}"
    print(f"  OK bipartite forward shape={tuple(out.shape)}")


def test_full_model_gin_forward():
    """SchemaHeteroGATv2 + aggregation_type='gin' forward heterograph."""
    print("\n[test_full_model_gin_forward]")
    m = _build_model(aggregation_type="gin")
    m.eval()
    data = _build_synthetic_supernode_graph()
    out = m(data.x_dict, data.edge_index_dict, query_emb=data["query_node"].x)
    for nt in ("table", "column", "fk_node"):
        assert nt in out, f"{nt} missing in output"
        assert out[nt].size(0) == data[nt].num_nodes
    # 모든 inner conv 가 GINConv
    n_gin, n_gat = 0, 0
    for hc in m.convs:
        for et, conv in hc.convs.items():
            if isinstance(conv, GINConv):
                n_gin += 1
            else:
                n_gat += 1
    assert n_gin > 0 and n_gat == 0, (
        f"all inner convs must be GINConv when aggregation_type='gin', got gin={n_gin}, gat={n_gat}"
    )
    # HeteroConv aggr 는 'mean' (GIN 모드 fix)
    for hc in m.convs:
        assert hc.aggr == "mean", f"GIN mode HeteroConv aggr must be 'mean', got {hc.aggr}"
    print(f"  OK {n_gin} inner GINConvs, HeteroConv aggr='mean'")


def test_backward_compat_default_mean():
    """aggregation_type 미설정 (default='mean') 시 inner conv = GATv2Conv (regression)."""
    print("\n[test_backward_compat_default_mean]")
    from torch_geometric.nn import GATv2Conv
    m = _build_model()  # default
    n_gat, n_gin = 0, 0
    for hc in m.convs:
        for et, conv in hc.convs.items():
            if isinstance(conv, GATv2Conv) and not isinstance(conv, GINConv):
                n_gat += 1
            elif isinstance(conv, GINConv):
                n_gin += 1
    assert n_gat > 0 and n_gin == 0, (
        f"default mode must use GATv2Conv only, got gat={n_gat}, gin={n_gin}"
    )
    print(f"  OK default {n_gat} GATv2Convs (no GINConv)")


def test_gin_incompatible_with_v2_options():
    """aggregation_type='gin' + drop_message_p/use_layernorm_pre_softmax → ValueError."""
    print("\n[test_gin_incompatible_with_v2_options]")
    raised_a = raised_b = False
    try:
        _build_model(aggregation_type="gin", drop_message_p=0.2)
    except ValueError as e:
        raised_a = True
        print(f"  OK gin+drop raise: {str(e)[:80]}")
    try:
        _build_model(aggregation_type="gin", use_layernorm_pre_softmax=True)
    except ValueError as e:
        raised_b = True
        print(f"  OK gin+layernorm raise: {str(e)[:80]}")
    assert raised_a and raised_b, "GIN must be incompatible with v2 #1/#3"


def test_gin_config_parsing():
    """신규 v3 config 파싱 + Phase 2 baseline 영향 X."""
    print("\n[test_gin_config_parsing]")
    cfg_dir = ROOT / "configs/training"
    p_v3 = cfg_dir / "train_gat_directed_supernode_p80_b5_mitigation_v3_gin.yaml"
    p_v2 = cfg_dir / "train_gat_directed_supernode_p80_b5_mitigation.yaml"
    assert p_v3.exists() and p_v2.exists()
    with open(p_v3) as f:
        c3 = yaml.safe_load(f)
    assert c3["model"]["aggregation_type"] == "gin"
    assert c3["model"].get("drop_message_p", 0.0) == 0.0
    assert c3["model"].get("use_layernorm_pre_softmax", False) is False
    with open(p_v2) as f:
        c2 = yaml.safe_load(f)
    assert "aggregation_type" not in c2["model"], (
        "Phase 2 baseline must NOT set aggregation_type (default 'mean' 사용)"
    )
    print(f"  OK v3 cfg: aggregation_type=gin, Phase 2 baseline 보존")


def test_gin_forward_backward_path():
    """GIN forward + backward — gradient 가 GINConv MLP 에 흐름."""
    print("\n[test_gin_forward_backward_path]")
    m = _build_model(aggregation_type="gin")
    m.train()
    data = _build_synthetic_supernode_graph()
    out = m(data.x_dict, data.edge_index_dict, query_emb=data["query_node"].x)
    # column output 으로 dummy loss + backward
    target = torch.zeros_like(out["column"])
    loss = (out["column"] - target).pow(2).mean()
    loss.backward()
    # 어떤 GINConv MLP 에 grad 가 도달
    grad_present = 0
    for hc in m.convs:
        for et, conv in hc.convs.items():
            if isinstance(conv, GINConv):
                for p in conv.parameters():
                    if p.grad is not None and p.grad.abs().sum().item() > 0:
                        grad_present += 1
                        break
    assert grad_present > 0, "no GINConv received gradient"
    print(f"  OK {grad_present} GINConvs received gradient from backward")


def main():
    test_gin_factory_and_homograph_forward()
    test_gin_factory_bipartite_forward()
    test_full_model_gin_forward()
    test_backward_compat_default_mean()
    test_gin_incompatible_with_v2_options()
    test_gin_config_parsing()
    test_gin_forward_backward_path()
    print("\nAll Mitigation v3 (GIN) smoke tests passed.")


if __name__ == "__main__":
    main()
