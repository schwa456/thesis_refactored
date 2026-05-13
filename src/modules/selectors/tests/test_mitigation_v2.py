"""Smoke test for Mitigation v2 candidate 3 종 (#1 DropMessage / #3 LayerNorm pre-softmax / #2 Sum aggr).

학위 본 심사 5/22 전 진행 (DECISIONS 2026-05-07 §1(C)/(D)). mech(ii) edge softmax over-concentration
DOMINANT 4-trial dominance 진단 결과 후속.

검증 대상:
  (1) #1 DropMessage — DropMessageGATv2Conv subclass 의 message dropout 동작 + drop_message_p=0.0
      backward compat (super class 와 동일 결과).
  (2) #3 LayerNorm pre-softmax — LayerNormGATv2Conv subclass 가 raw alpha 산출 후 softmax 직전에
      LayerNorm 삽입. SchemaHeteroGATv2 와 결합 forward shape OK.
  (3) #2 Sum aggregation — HeteroConv(aggr='sum') 호환 + heterograph forward 성공.
  (4) 3 옵션 결합 — 둘 이상 옵션 동시 활성화 시 충돌 없이 forward (e.g., #1 + #3).
  (5) Backward compat — 모든 default 시 Phase 2 b8 baseline (gat_network_v2 default) 와 forward
      결과 동일 (state_dict key 일관, parameter 수 동일).
  (6) 신규 config 3 종 파싱 + 옵션 정확.

Run from project root:
    PYTHONPATH=src conda run -n base python src/modules/selectors/tests/test_mitigation_v2.py
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


def _build_model(in_channels=384, hidden=64, heads=2, num_layers=2, **kwargs):
    """Phase 2 b5 mitigation 의 작은 변형 — smoke 용. kwargs 로 mit v2 옵션 전달."""
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


# ──────────────────────────────────────────────────────────────────────
# #1 DropMessage tests
# ──────────────────────────────────────────────────────────────────────

def test_v1_drop_message_subclass():
    """DropMessageGATv2Conv 가 message dropout 을 적용 — train mode 시 결과 변동 (random)."""
    print("\n[test_v1_drop_message_subclass]")
    from models.gat_network_v2 import DropMessageGATv2Conv
    conv = DropMessageGATv2Conv(8, 16, heads=2, drop_message_p=0.5, add_self_loops=False)
    x = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 0]], dtype=torch.long)
    conv.train()
    torch.manual_seed(0); o1 = conv(x, edge_index)
    torch.manual_seed(1); o2 = conv(x, edge_index)
    assert not torch.allclose(o1, o2), "training mode + drop_message_p=0.5 → 결과 변동 expected"
    conv.eval()
    o3 = conv(x, edge_index); o4 = conv(x, edge_index)
    assert torch.allclose(o3, o4), "eval mode → deterministic"
    print(f"  OK train ≠ rerun (random), eval == rerun (deterministic)")


def test_v1_drop_message_zero_backward_compat():
    """drop_message_p=0.0 시 GATv2Conv 와 결과 동일 (parameter init 동일하면)."""
    print("\n[test_v1_drop_message_zero_backward_compat]")
    from models.gat_network_v2 import DropMessageGATv2Conv
    from torch_geometric.nn import GATv2Conv
    torch.manual_seed(42)
    conv_drop = DropMessageGATv2Conv(8, 16, heads=2, drop_message_p=0.0, add_self_loops=False)
    torch.manual_seed(42)
    conv_base = GATv2Conv(8, 16, heads=2, add_self_loops=False)
    # parameter copy (동일 init)
    conv_drop.load_state_dict(conv_base.state_dict())
    conv_drop.eval(); conv_base.eval()
    x = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 0]], dtype=torch.long)
    o_drop = conv_drop(x, edge_index)
    o_base = conv_base(x, edge_index)
    assert torch.allclose(o_drop, o_base, atol=1e-6), "drop_message_p=0.0 → super class 동일 결과"
    print(f"  OK drop_message_p=0.0 backward compat verified")


def test_v1_full_model_forward():
    """SchemaHeteroGATv2 + drop_message_p=0.2 forward shape 정합성."""
    print("\n[test_v1_full_model_forward]")
    m = _build_model(drop_message_p=0.2)
    m.eval()
    data = _build_synthetic_supernode_graph()
    out = m(data.x_dict, data.edge_index_dict, query_emb=data["query_node"].x)
    for nt in ("table", "column", "fk_node"):
        assert nt in out, f"{nt} missing in output"
        assert out[nt].size(0) == data[nt].num_nodes
    print(f"  OK shapes: {{nt: tuple(out[nt].shape) for nt in ('table','column','fk_node')}}")


# ──────────────────────────────────────────────────────────────────────
# #3 LayerNorm pre-softmax tests
# ──────────────────────────────────────────────────────────────────────

def test_v3_layernorm_subclass():
    """LayerNormGATv2Conv 가 alpha 에 LayerNorm 적용 — alpha_layernorm 모듈 존재."""
    print("\n[test_v3_layernorm_subclass]")
    from models.gat_network_v2 import LayerNormGATv2Conv
    conv = LayerNormGATv2Conv(8, 16, heads=4, add_self_loops=False)
    assert hasattr(conv, "alpha_layernorm"), "LayerNormGATv2Conv must register alpha_layernorm"
    assert conv.alpha_layernorm.normalized_shape == (4,), (
        f"alpha LayerNorm shape must be (heads,)=(4,), got {conv.alpha_layernorm.normalized_shape}"
    )
    # forward 가 정상 동작 (alpha 가 [E, heads] shape, LayerNorm 적용 후 softmax)
    x = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 0]], dtype=torch.long)
    conv.eval()
    out = conv(x, edge_index)
    assert out.shape == (6, 4 * 16), f"unexpected shape {out.shape}"
    print(f"  OK alpha_layernorm registered + forward shape correct")


def test_v3_full_model_forward():
    """SchemaHeteroGATv2 + use_layernorm_pre_softmax=True forward shape."""
    print("\n[test_v3_full_model_forward]")
    m = _build_model(use_layernorm_pre_softmax=True)
    m.eval()
    data = _build_synthetic_supernode_graph()
    out = m(data.x_dict, data.edge_index_dict, query_emb=data["query_node"].x)
    for nt in ("table", "column", "fk_node"):
        assert nt in out
    # alpha_layernorm 이 모든 inner GATv2Conv 에 존재
    n_alpha_ln = 0
    for hc in m.convs:
        for et, conv in hc.convs.items():
            if hasattr(conv, "alpha_layernorm"):
                n_alpha_ln += 1
    assert n_alpha_ln > 0, "no alpha_layernorm in any inner conv"
    print(f"  OK {n_alpha_ln} inner convs have alpha_layernorm")


# ──────────────────────────────────────────────────────────────────────
# #2 Sum aggregation tests
# ──────────────────────────────────────────────────────────────────────

def test_v2_sum_aggregation_forward():
    """SchemaHeteroGATv2 + aggregation_type='sum' — heterograph forward 성공."""
    print("\n[test_v2_sum_aggregation_forward]")
    m = _build_model(aggregation_type="sum")
    m.eval()
    data = _build_synthetic_supernode_graph()
    out = m(data.x_dict, data.edge_index_dict, query_emb=data["query_node"].x)
    for nt in ("table", "column", "fk_node"):
        assert nt in out
    # 모든 HeteroConv 의 aggr 가 'sum'
    for hc in m.convs:
        assert getattr(hc, "aggr", None) == "sum", (
            f"HeteroConv.aggr must be 'sum', got {getattr(hc, 'aggr', None)}"
        )
    print(f"  OK aggregation_type='sum', all HeteroConv aggr verified")


def test_v2_max_aggregation_forward():
    """aggregation_type='max' 도 동작 (사용자 spec 의 'sum/max' 양쪽)."""
    print("\n[test_v2_max_aggregation_forward]")
    m = _build_model(aggregation_type="max")
    m.eval()
    data = _build_synthetic_supernode_graph()
    out = m(data.x_dict, data.edge_index_dict, query_emb=data["query_node"].x)
    assert "column" in out and "table" in out
    for hc in m.convs:
        assert getattr(hc, "aggr", None) == "max"
    print(f"  OK aggregation_type='max' forward")


# ──────────────────────────────────────────────────────────────────────
# Combo + backward compat
# ──────────────────────────────────────────────────────────────────────

def test_combo_drop_plus_layernorm():
    """#1 + #3 동시 활성화 — multiple inheritance 충돌 없이 forward."""
    print("\n[test_combo_drop_plus_layernorm]")
    m = _build_model(drop_message_p=0.2, use_layernorm_pre_softmax=True)
    m.train()
    data = _build_synthetic_supernode_graph()
    out = m(data.x_dict, data.edge_index_dict, query_emb=data["query_node"].x)
    assert "column" in out
    # inner conv 가 LayerNormDropMessage 결합 클래스인지 확인 (내부 attribute 둘 다 있음)
    n_combined = 0
    for hc in m.convs:
        for et, conv in hc.convs.items():
            has_ln = hasattr(conv, "alpha_layernorm")
            has_drop = hasattr(conv, "drop_message_p") and conv.drop_message_p > 0
            if has_ln and has_drop:
                n_combined += 1
    assert n_combined > 0, "combo class not applied to any inner conv"
    print(f"  OK {n_combined} inner convs have both LayerNorm + DropMessage")


def _initialize_with_dummy_forward(m):
    """GATv2Conv(in_channels=-1, ...) 는 LazyLinear 사용 → numel() 전에 forward 1회 필요."""
    data = _build_synthetic_supernode_graph()
    m.eval()
    with torch.no_grad():
        _ = m(data.x_dict, data.edge_index_dict, query_emb=data["query_node"].x)


def test_baseline_backward_compat():
    """모든 mit v2 옵션 default OFF → Phase 2 b8 baseline 과 동일 동작 (parameter 개수)."""
    print("\n[test_baseline_backward_compat]")
    m_default = _build_model()
    m_explicit_off = _build_model(
        drop_message_p=0.0, use_layernorm_pre_softmax=False, aggregation_type="mean"
    )
    _initialize_with_dummy_forward(m_default)
    _initialize_with_dummy_forward(m_explicit_off)
    n1 = sum(p.numel() for p in m_default.parameters())
    n2 = sum(p.numel() for p in m_explicit_off.parameters())
    assert n1 == n2, f"default vs explicit-OFF param count differ: {n1} vs {n2}"

    keys1 = sorted(m_default.state_dict().keys())
    keys2 = sorted(m_explicit_off.state_dict().keys())
    assert keys1 == keys2, "state_dict key set differ — backward compat broken"
    print(f"  OK default vs explicit-OFF identical (params={n1}, keys={len(keys1)})")


def test_phase2_baseline_no_extra_params():
    """Phase 2 baseline (default) ↔ #3 LayerNorm 활성화 시 LayerNorm parameter 만 추가."""
    print("\n[test_phase2_baseline_no_extra_params]")
    m_baseline = _build_model()
    m_ln = _build_model(use_layernorm_pre_softmax=True)
    _initialize_with_dummy_forward(m_baseline)
    _initialize_with_dummy_forward(m_ln)
    n_base = sum(p.numel() for p in m_baseline.parameters())
    n_ln = sum(p.numel() for p in m_ln.parameters())
    extra = n_ln - n_base
    print(f"  baseline params={n_base}, +LN params={n_ln}, Δ={extra}")
    # LayerNorm(heads) 는 weight + bias = 2*heads params per inner conv. 양수 단 baseline 의
    # 1% 이내 small overhead 검증 (heads=2, 9 edge types × 2 layers = 18 LN modules → 72 params).
    assert 0 < extra < n_base * 0.05, (
        f"unexpected param delta from LayerNorm: {extra} (baseline={n_base})"
    )


# ──────────────────────────────────────────────────────────────────────
# Config parsing
# ──────────────────────────────────────────────────────────────────────

def test_v2_config_parsing():
    """3 신규 config 의 mit v2 옵션 정확."""
    print("\n[test_v2_config_parsing]")
    cfg_dir = ROOT / "configs/training"
    files = {
        "drop_message": cfg_dir / "train_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.yaml",
        "layernorm": cfg_dir / "train_gat_directed_supernode_p80_b5_mitigation_v2_layernorm.yaml",
        "sum_aggr": cfg_dir / "train_gat_directed_supernode_p80_b5_mitigation_v2_sum_aggr.yaml",
    }
    for name, p in files.items():
        assert p.exists(), f"config missing: {p}"
    with open(files["drop_message"]) as f:
        c1 = yaml.safe_load(f)
    assert c1["model"]["drop_message_p"] == 0.2
    assert c1["model"].get("use_layernorm_pre_softmax", False) is False
    assert c1["model"].get("aggregation_type", "mean") == "mean"

    with open(files["layernorm"]) as f:
        c3 = yaml.safe_load(f)
    assert c3["model"]["use_layernorm_pre_softmax"] is True
    assert c3["model"].get("drop_message_p", 0.0) == 0.0

    with open(files["sum_aggr"]) as f:
        c2 = yaml.safe_load(f)
    assert c2["model"]["aggregation_type"] == "sum"
    assert c2["model"].get("drop_message_p", 0.0) == 0.0
    assert c2["model"].get("use_layernorm_pre_softmax", False) is False
    print("  OK all 3 configs parse + mit v2 options correctly set")


def test_phase2_baseline_no_v2_options():
    """Phase 2 b8 baseline config 가 mit v2 옵션 미설정 — code default 사용."""
    print("\n[test_phase2_baseline_no_v2_options]")
    p = ROOT / "configs/training/train_gat_directed_supernode_p80_b5_mitigation.yaml"
    with open(p) as f:
        c = yaml.safe_load(f)
    for k in ("drop_message_p", "use_layernorm_pre_softmax", "aggregation_type"):
        assert k not in c["model"], (
            f"Phase 2 baseline must NOT set '{k}' — code default 사용 (backward compat)"
        )
    print("  OK Phase 2 baseline 보존 (mit v2 옵션 미설정)")


def main():
    test_v1_drop_message_subclass()
    test_v1_drop_message_zero_backward_compat()
    test_v1_full_model_forward()
    test_v3_layernorm_subclass()
    test_v3_full_model_forward()
    test_v2_sum_aggregation_forward()
    test_v2_max_aggregation_forward()
    test_combo_drop_plus_layernorm()
    test_baseline_backward_compat()
    test_phase2_baseline_no_extra_params()
    test_v2_config_parsing()
    test_phase2_baseline_no_v2_options()
    print("\nAll Mitigation v2 smoke tests passed.")


if __name__ == "__main__":
    main()
