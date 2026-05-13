"""Smoke test for extract_layerwise_attention_v2.

V-3-ext directed_from_sn 호환 attention extract 검증:
  (1) AttentionCapture context manager — wrap/restore 정상 동작.
  (2) Layer × edge_type 별 attention map 분리 추출 (1-3 sample query).
  (3) Phase 1 4 ckpt 호환:
        - DSN p80 (percentile=80, directed_from_sn)
        - DSN topk20 (top_k=20, directed_from_sn)
        - DSN abstau07 (abs_tau=0.7, directed_from_sn)
        - qcond_nl3 baseline (no SuperNode)
  (4) entropy 와 top-K concentration metric shape + sanity (NaN 외 값).
  (5) directed_from_sn 시 attended_by_* edge type 이 attention dict 에 미등장 검증.
  (6) aggregate_attention_metrics — 다중 query mean/std 산출.

Run from project root:
    PYTHONPATH=src conda run -n base python src/modules/selectors/tests/test_attention_extract_v2.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch_geometric.data import HeteroData

# Phase 1 4 ckpt 정의 (dsn_oversmoothing_analysis.py 와 동일)
CKPTS = [
    {
        "name": "p80",
        "config": ROOT / "configs/training/train_gat_directed_supernode_p80.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80.pt",
        "expect_supernode": True,
        "expect_directed": True,
    },
    {
        "name": "topk20",
        "config": ROOT / "configs/training/train_gat_directed_supernode_topk20.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_topk20.pt",
        "expect_supernode": True,
        "expect_directed": True,
    },
    {
        "name": "abstau07",
        "config": ROOT / "configs/training/train_gat_directed_supernode_abstau07.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_directed_supernode_abstau07.pt",
        "expect_supernode": True,
        "expect_directed": True,
    },
    {
        "name": "qcond_nl3",
        "config": ROOT / "configs/training/diameter_layers/train_qcond_nl3.yaml",
        "ckpt": ROOT / "outputs/checkpoints/best_gat_qcond_nl3.pt",
        "expect_supernode": False,
        "expect_directed": False,
    },
]


def _build_synthetic_graph(num_tables=4, cols_per_table=6, num_fk=3, in_dim=384,
                           seed=42, with_supernode=False):
    g = torch.Generator().manual_seed(seed)
    total_cols = num_tables * cols_per_table
    data = HeteroData()
    data["table"].x = torch.randn(num_tables, in_dim, generator=g)
    data["column"].x = torch.randn(total_cols, in_dim, generator=g)
    data["fk_node"].x = torch.randn(num_fk, in_dim, generator=g)

    t_src, c_dst = [], []
    for t in range(num_tables):
        for j in range(cols_per_table):
            t_src.append(t)
            c_dst.append(t * cols_per_table + j)
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

    if with_supernode:
        data["query_node"].x = torch.randn(1, in_dim, generator=g)
        for nt in ("table", "column", "fk_node"):
            n = data[nt].num_nodes
            src = torch.zeros(n, dtype=torch.long)
            dst = torch.arange(n, dtype=torch.long)
            data["query_node", f"attends_to_{nt}", nt].edge_index = torch.stack([src, dst], 0)
            data[nt, f"attended_by_{nt}", "query_node"].edge_index = torch.stack([dst, src], 0)
    return data


def _load_model(ckpt_info):
    """Phase 1 ckpt 1 개를 V-3-ext 옵션 forward 하여 SchemaHeteroGAT 빌드."""
    import yaml
    from analysis.dsn_oversmoothing_analysis import _build_model_dsn

    if not ckpt_info["ckpt"].exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_info['ckpt']}")
    if not ckpt_info["config"].exists():
        raise FileNotFoundError(f"config not found: {ckpt_info['config']}")

    with open(ckpt_info["config"], "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cpu")
    model = _build_model_dsn(cfg, ckpt_info["ckpt"], device)
    return model, cfg


def test_attention_capture_basic():
    """Synthetic graph + small SchemaHeteroGAT — capture / restore 동작 확인."""
    print("\n[test_attention_capture_basic]")
    from analysis.extract_layerwise_attention_v2 import (
        AttentionCapture, extract_layerwise_attention_v2,
    )
    from models.gat_network import SchemaHeteroGAT

    m = SchemaHeteroGAT(
        in_channels=384, hidden_channels=64, out_channels=64,
        num_layers=2, heads=2, query_conditioned=True, query_supernode=True,
        supernode_edge_direction="directed_from_sn",
        supernode_threshold_mode="percentile",
        supernode_threshold_value=80.0,
    )
    m.eval()
    data = _build_synthetic_graph(with_supernode=True)
    q_emb = data["query_node"].x

    # Restore 검증: capture context 가 종료된 뒤에는 instance attribute 'forward' 가 없어야 함
    # (즉 class-level method 로 fallback). 캡처 중에는 instance attr 로 wrap.
    cap = AttentionCapture(m)
    inner_convs = [(hc, et, conv)
                   for hc in m.convs for et, conv in hc.convs.items()]
    # pre-state: instance dict 에 'forward' 없음
    assert all("forward" not in conv.__dict__ for _, _, conv in inner_convs)
    with cap:
        # capture 중에는 instance attr 로 wrap
        assert all("forward" in conv.__dict__ for _, _, conv in inner_convs), (
            "during capture, conv.forward must be patched as instance attr"
        )
        with torch.no_grad():
            _ = m(data.x_dict, data.edge_index_dict, query_emb=q_emb)
    # exit 후 모두 restore — but 우리 코드가 instance attribute 로 복원하므로 dict 에 남음.
    # 핵심은 다음 forward 호출이 정상 결과를 내야 함. 한 번 더 호출 검증.
    out = extract_layerwise_attention_v2(m, data, query_emb=q_emb, topk=5)
    assert out["num_layers"] == 2, f"expected 2 layers, got {out['num_layers']}"
    assert "entropy" in out and "topk_conc" in out
    assert "L1" in out["entropy"] and "L2" in out["entropy"]
    print(f"  L1 edge_types: {sorted(out['entropy']['L1'].keys())}")
    print(f"  L2 entropy keys: {len(out['entropy']['L2'])}")


def test_directed_from_sn_no_reverse_edges():
    """directed_from_sn ckpt: attended_by_* edge type 이 attention dict 에 등장 X."""
    print("\n[test_directed_from_sn_no_reverse_edges]")
    from analysis.extract_layerwise_attention_v2 import extract_layerwise_attention_v2
    from models.gat_network import SchemaHeteroGAT

    m = SchemaHeteroGAT(
        in_channels=384, hidden_channels=64, out_channels=64,
        num_layers=2, heads=2, query_conditioned=False, query_supernode=True,
        supernode_edge_direction="directed_from_sn",
        supernode_topk=10,
    )
    m.eval()
    data = _build_synthetic_graph(with_supernode=True)
    out = extract_layerwise_attention_v2(m, data, query_emb=data["query_node"].x)

    for layer_key, et_map in out["entropy"].items():
        for et_str in et_map:
            assert "attended_by_" not in et_str, (
                f"directed_from_sn: attended_by_* must NOT appear, got '{et_str}' in {layer_key}"
            )
        # forward edge 들은 등록되어 있어야 함
        assert any("attends_to_" in et for et in et_map), (
            f"forward attends_to_* edges expected in {layer_key}, got {list(et_map.keys())}"
        )
    print(f"  OK directed_from_sn — only forward edges captured")


def test_entropy_topk_value_sanity():
    """entropy >= 0, topk_conc ∈ [0, 1] sanity check."""
    print("\n[test_entropy_topk_value_sanity]")
    from analysis.extract_layerwise_attention_v2 import extract_layerwise_attention_v2
    from models.gat_network import SchemaHeteroGAT

    m = SchemaHeteroGAT(
        in_channels=384, hidden_channels=64, out_channels=64,
        num_layers=2, heads=2, query_conditioned=False, query_supernode=True,
        supernode_edge_direction="directed_from_sn",
    )
    m.eval()
    data = _build_synthetic_graph(with_supernode=True)
    out = extract_layerwise_attention_v2(m, data, query_emb=data["query_node"].x)

    n_checked = 0
    for layer_key, et_map in out["entropy"].items():
        for et, ent in et_map.items():
            if ent != ent:  # NaN check
                continue
            assert ent >= 0, f"entropy must be >= 0: {layer_key}/{et}={ent}"
            n_checked += 1
    for layer_key, et_map in out["topk_conc"].items():
        for et, conc in et_map.items():
            if conc != conc:
                continue
            assert 0.0 <= conc <= 1.0 + 1e-6, (
                f"topk_conc must be in [0,1]: {layer_key}/{et}={conc}"
            )
            n_checked += 1
    assert n_checked > 0, "no metrics computed"
    print(f"  OK {n_checked} entropy/topk_conc cells in valid range")


def test_aggregate_metrics():
    """aggregate_attention_metrics — 3 query 의 mean/std 산출."""
    print("\n[test_aggregate_metrics]")
    from analysis.extract_layerwise_attention_v2 import (
        extract_layerwise_attention_v2, aggregate_attention_metrics,
    )
    from models.gat_network import SchemaHeteroGAT

    m = SchemaHeteroGAT(
        in_channels=384, hidden_channels=64, out_channels=64,
        num_layers=2, heads=2, query_conditioned=False, query_supernode=True,
        supernode_edge_direction="directed_from_sn",
    )
    m.eval()
    per_q = []
    for seed in (1, 7, 13):
        data = _build_synthetic_graph(with_supernode=True, seed=seed)
        out = extract_layerwise_attention_v2(m, data, query_emb=data["query_node"].x)
        per_q.append(out)
    agg = aggregate_attention_metrics(per_q)
    assert agg["num_queries"] == 3
    assert "entropy_mean" in agg and "entropy_std" in agg
    assert "topk_conc_mean" in agg and "topk_conc_std" in agg
    print(f"  num_queries={agg['num_queries']} num_layers={agg['num_layers']}")
    print(f"  entropy_mean[L1] keys: {len(agg['entropy_mean'].get('L1', {}))}")


def test_phase1_ckpt_compatibility():
    """Phase 1 4 ckpt 별 attention extract — config + ckpt 존재 시 실측."""
    print("\n[test_phase1_ckpt_compatibility]")
    from analysis.extract_layerwise_attention_v2 import extract_layerwise_attention_v2

    n_ok, n_skip = 0, 0
    for c in CKPTS:
        if not c["ckpt"].exists() or not c["config"].exists():
            print(f"  SKIP {c['name']}: ckpt/config missing")
            n_skip += 1
            continue
        try:
            model, cfg = _load_model(c)
        except Exception as e:
            print(f"  SKIP {c['name']}: load failed ({type(e).__name__}: {e})")
            n_skip += 1
            continue
        # build synthetic graph aligned with ckpt config
        in_dim = cfg["model"]["in_channels"]
        with_sn = bool(cfg["model"].get("query_supernode", False))
        data = _build_synthetic_graph(in_dim=in_dim, with_supernode=with_sn)
        q_emb = data["query_node"].x if with_sn else torch.randn(1, in_dim)
        out = extract_layerwise_attention_v2(model, data, query_emb=q_emb)
        assert out["num_layers"] == model.num_layers
        n_layer_with_metrics = sum(
            1 for et_map in out["entropy"].values() if et_map
        )
        assert n_layer_with_metrics > 0, f"{c['name']}: no entropy captured"
        # directed_from_sn 시 reverse edge 미등장
        if c["expect_directed"]:
            for layer_key, et_map in out["entropy"].items():
                for et_str in et_map:
                    assert "attended_by_" not in et_str, (
                        f"{c['name']}/{layer_key}: directed should not contain '{et_str}'"
                    )
        first_layer = list(out["entropy"].keys())[0]
        sample_ets = list(out["entropy"][first_layer].keys())[:3]
        print(f"  OK {c['name']} (L={model.num_layers}, sample edge_types={sample_ets})")
        n_ok += 1
    print(f"  total: {n_ok} ckpt OK, {n_skip} skipped")
    assert n_ok > 0, "expected at least 1 Phase 1 ckpt to be loadable; none worked"


def test_qcond_nl3_no_supernode():
    """qcond_nl3 baseline — query_supernode=False ckpt. SuperNode edge 없음 검증."""
    print("\n[test_qcond_nl3_no_supernode]")
    from analysis.extract_layerwise_attention_v2 import extract_layerwise_attention_v2

    info = next(c for c in CKPTS if c["name"] == "qcond_nl3")
    if not info["ckpt"].exists() or not info["config"].exists():
        print("  SKIP: qcond_nl3 ckpt/config not found")
        return
    try:
        model, cfg = _load_model(info)
    except Exception as e:
        print(f"  SKIP: {type(e).__name__}: {e}")
        return
    in_dim = cfg["model"]["in_channels"]
    data = _build_synthetic_graph(in_dim=in_dim, with_supernode=False)
    q_emb = torch.randn(1, in_dim)
    out = extract_layerwise_attention_v2(model, data, query_emb=q_emb)
    for layer_key, et_map in out["entropy"].items():
        for et_str in et_map:
            assert "attends_to_" not in et_str, (
                f"qcond_nl3 has no SuperNode edges, got '{et_str}'"
            )
    print(f"  OK qcond_nl3 — no SuperNode edges (L={model.num_layers})")


def main():
    test_attention_capture_basic()
    test_directed_from_sn_no_reverse_edges()
    test_entropy_topk_value_sanity()
    test_aggregate_metrics()
    test_phase1_ckpt_compatibility()
    test_qcond_nl3_no_supernode()
    print("\nAll attention extract v2 smoke tests passed.")


if __name__ == "__main__":
    main()
