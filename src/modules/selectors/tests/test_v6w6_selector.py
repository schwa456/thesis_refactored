"""Smoke test for V6-W6 (Phase 6) directed SuperNode + V6-W5 self-loop 조합.

검증 항목 (DECISIONS 2026-06-07 #3 + V6 plan §1 Phase 6 정합):
  (1) v6w6_variant 등록 + validation (directed SN 결합 강제: query_supernode=True +
      supernode_edge_direction='directed_from_sn' / standard / edge_type_split·v6w3 비결합).
      a → column self-loop, b → self-loop + per-layer residual (self-contained boolean 결합).
  (2) ckpt save/load round-trip (train_gat_s06 형식) + DirectGATv2Selector auto_config (supernode
      모드 복원) + 5q forward + score range ∈ [0,1].
  (3) ★ mechanism 검증 (결정론적) — baseline (M4 anchor, single-shared-source) L1 hi-deg
      intra-MAD ≈ 0 collapse → W6-a (directed SN + self-loop) > 0 회복. (informative) SN-only
      (self-loop 없음) 도 함께 측정해 self-loop marginal 표시.

학습 ckpt 미존재 단계 — 랜덤 init 모델로 형식 round-trip + 구조적 기제 검증.

Run from project root:
    conda run -n base python src/modules/selectors/tests/test_v6w6_selector.py
"""
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch
from torch_geometric.data import HeteroData

from models.gat_network_v2 import SchemaHeteroGATv2, V6W5_COLUMN_SELF_LOOP_REL
from models.direct_classifier import DirectClassifierHead

IN, HID, OUT = 384, 256, 256
NUM_LAYERS, HEADS = 3, 4
N_COLS, N_FK, N_EXTRA_T = 12, 2, 2


def _base_graph(seed=42, supernode=False):
    """hub table(0) 모든 컬럼 단일 table 소스 (fk edge 0개). supernode=True 면 query_node +
    directed attends_to_* 추가 (directed_from_sn)."""
    g = torch.Generator().manual_seed(seed)
    T = 1 + N_EXTRA_T
    data = HeteroData()
    data["table"].x = torch.randn(T, IN, generator=g)
    data["column"].x = torch.randn(N_COLS, IN, generator=g)
    data["fk_node"].x = torch.randn(N_FK, IN, generator=g)

    tc = torch.tensor([[0] * N_COLS, list(range(N_COLS))], dtype=torch.long)
    data["table", "has_column", "column"].edge_index = tc
    data["column", "belongs_to", "table"].edge_index = tc.flip(0)
    empty = torch.zeros((2, 0), dtype=torch.long)
    data["column", "is_source_of", "fk_node"].edge_index = empty
    data["fk_node", "points_to", "column"].edge_index = empty
    s = list(range(T - 1)); d = list(range(1, T))
    data["table", "table_to_table", "table"].edge_index = torch.tensor(
        [s + d, d + s], dtype=torch.long)

    if supernode:
        data["query_node"].x = torch.randn(1, IN, generator=g)
        # directed_from_sn: query_node → schema (attends_to_*) 만.
        data["query_node", "attends_to_table", "table"].edge_index = torch.tensor(
            [[0] * T, list(range(T))], dtype=torch.long)
        data["query_node", "attends_to_column", "column"].edge_index = torch.tensor(
            [[0] * N_COLS, list(range(N_COLS))], dtype=torch.long)
        data["query_node", "attends_to_fk_node", "fk_node"].edge_index = torch.tensor(
            [[0] * N_FK, list(range(N_FK))], dtype=torch.long)
    return data


def _node_metadata():
    T = 1 + N_EXTRA_T
    md, idx = {}, 0
    for t in range(T):
        md[idx] = {"type": "table", "name": f"t_{t}"}; idx += 1
    for c in range(N_COLS):
        md[idx] = {"type": "column", "name": f"c_{c}"}; idx += 1
    for f in range(N_FK):
        md[idx] = {"type": "fk_node", "name": f"fk_{f}"}; idx += 1
    return md, idx


def _intra_mad(emb):
    n = emb.size(0)
    if n < 2:
        return 0.0
    return (torch.cdist(emb, emb).sum() / (n * (n - 1))).item()


def _l1_col_m4_baseline(device):
    """M4 anchor (query_conditioned, no SN, no self-loop) — single-shared-source collapse."""
    m = SchemaHeteroGATv2(
        in_channels=IN, hidden_channels=HID, out_channels=OUT, num_layers=NUM_LAYERS,
        heads=HEADS, query_conditioned=True, capture_layerwise_outputs=True,
    ).to(device).eval()
    data = _base_graph(supernode=False).to(device)
    q = torch.zeros(1, IN, device=device)
    aug = {nt: torch.cat([x, q.expand(x.size(0), -1)], -1) for nt, x in data.x_dict.items()}
    with torch.no_grad():
        _ = m(aug, data.edge_index_dict)
    return _intra_mad(m.get_captured_layer_outputs()[0]["column"])


def _l1_col_supernode(device, v6w6=None):
    """directed SuperNode 모드 (선택적 v6w6 self-loop) — L1 column embedding."""
    m = SchemaHeteroGATv2(
        in_channels=IN, hidden_channels=HID, out_channels=OUT, num_layers=NUM_LAYERS,
        heads=HEADS, query_supernode=True, supernode_edge_direction="directed_from_sn",
        v6w6_variant=v6w6, capture_layerwise_outputs=True,
    ).to(device).eval()
    data = _base_graph(supernode=True).to(device)
    with torch.no_grad():
        _ = m(data.x_dict, data.edge_index_dict)
    return _intra_mad(m.get_captured_layer_outputs()[0]["column"])


def _make_ckpt(variant, path, device):
    v5 = "a" if variant == "a" else "c"
    model = SchemaHeteroGATv2(
        in_channels=IN, hidden_channels=HID, out_channels=OUT, num_layers=NUM_LAYERS,
        heads=HEADS, query_supernode=True, supernode_edge_direction="directed_from_sn",
        v6w5_variant=v5, v6w6_variant=variant,
    ).to(device).eval()
    heads = torch.nn.ModuleDict({
        nt: DirectClassifierHead(in_dim=OUT, hidden_dim=OUT, dropout=0.0).to(device)
        for nt in ["table", "column", "fk_node"]
    })
    data = _base_graph(supernode=True).to(device)
    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)
    cfg = {
        "experiment_name": f"v6w6_{variant}_s11_smoke",
        "builder": {"type": "EnrichedHeteroGraphBuilder"},
        "model": {
            "in_channels": IN, "hidden_channels": HID, "out_channels": OUT,
            "num_layers": NUM_LAYERS, "heads": HEADS, "classifier_hidden": OUT,
            "query_supernode": True, "supernode_edge_direction": "directed_from_sn",
            "v6w5_variant": v5, "v6w6_variant": variant,
        },
    }
    torch.save({"epoch": 0, "gat_state_dict": model.state_dict(),
                "classifier_state_dict": heads.state_dict(),
                "recall": 0.0, "config": cfg}, path)


def test_model_registers():
    print("\n=== (1) v6w6_variant 등록 + validation ===")
    for variant in ["a", "b"]:
        m = SchemaHeteroGATv2(
            in_channels=IN, hidden_channels=HID, out_channels=OUT, num_layers=NUM_LAYERS,
            heads=HEADS, query_supernode=True, supernode_edge_direction="directed_from_sn",
            v6w6_variant=variant,
        )
        assert V6W5_COLUMN_SELF_LOOP_REL in m.all_edge_types, f"{variant}: self-loop 누락"
        assert m.v6w5_column_self_loop, f"{variant}: column_self_loop 미활성"
        if variant == "b":
            assert m.v6w5_per_layer_residual and m.v6w5_res_proj is not None, "b: residual 누락"
        else:
            assert not m.v6w5_per_layer_residual, "a: residual 불필요"
        print(f"  [OK] {variant}: self_loop={m.v6w5_column_self_loop} "
              f"residual={m.v6w5_per_layer_residual} SN={m.query_supernode}/{m.supernode_edge_direction}")
    # validation: directed SN 결합 강제
    for bad in [dict(), dict(query_supernode=True),
                dict(query_supernode=True, supernode_edge_direction="bidirectional")]:
        try:
            SchemaHeteroGATv2(in_channels=IN, hidden_channels=HID, out_channels=OUT,
                              num_layers=NUM_LAYERS, heads=HEADS, v6w6_variant="a", **bad)
            raise AssertionError(f"validation 미작동: {bad}")
        except ValueError:
            pass
    print("  [OK] validation: query_supernode + directed_from_sn 결합 강제")


def test_mechanism_l1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    print("\n=== (3) mechanism — L1 column intra-MAD ===")
    base = _l1_col_m4_baseline(device)
    sn_only = _l1_col_supernode(device, v6w6=None)
    w6a = _l1_col_supernode(device, v6w6="a")
    w6b = _l1_col_supernode(device, v6w6="b")
    print(f"  baseline (M4, single-source, no self-loop): L1 intra-MAD = {base:.6e}  (collapse ≈0)")
    print(f"  SN-only (directed SN, no self-loop):        L1 intra-MAD = {sn_only:.6e}  (informative)")
    print(f"  W6-a (directed SN + self-loop):             L1 intra-MAD = {w6a:.6e}")
    print(f"  W6-b (+ per-layer residual):                L1 intra-MAD = {w6b:.6e}")
    assert base < 1e-4, f"baseline collapse 전제 위반 (={base:.6e})"
    assert w6a > 1e-3, f"W6-a 회복 실패 (={w6a:.6e})"
    assert w6b > 1e-3, f"W6-b 회복 실패 (={w6b:.6e})"
    print("  → baseline collapse 재현 + W6-a/b L1 분화 회복 (directed SN + self-loop 기제 입증)")


def test_roundtrip_forward():
    from modules.selectors import DirectGATv2Selector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=== (2) ckpt round-trip + forward × 5q (supernode 모드) ===")
    questions = [
        "What is the average salary by department?",
        "List customers with orders in 2023.",
        "How many products are electronics?",
        "Top 5 schools by enrollment.",
        "Patients with more than three visits.",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for variant in ["a", "b"]:
            path = os.path.join(tmp, f"v6w6_{variant}.pt")
            _make_ckpt(variant, path, device)
            sel = DirectGATv2Selector(weight_path=path)
            assert sel.gat_model.v6w6_variant == variant, f"{variant}: auto-config 복원 실패"
            assert sel.query_supernode, f"{variant}: supernode 모드 복원 실패"
            md, num_nodes = _node_metadata()
            cand = list(range(num_nodes))
            for qi, question in enumerate(questions):
                data = _base_graph(seed=100 + qi, supernode=False)  # selector 가 SN 구성
                out = sel.select(scores=None, candidates=cand, question=question,
                                 graph_data=data, metadata={"node_metadata": md})
                s = torch.tensor(sel.latest_scores)
                assert len(sel.latest_scores) == num_nodes, f"{variant} q{qi}: score 개수 불일치"
                assert torch.isfinite(s).all(), f"{variant} q{qi}: NaN/Inf"
                assert (s >= 0).all() and (s <= 1).all(), f"{variant} q{qi}: 범위 위반"
                assert out == cand, f"{variant} q{qi}: candidates 보존 위반"
            print(f"  [OK] variant {variant}: round-trip + 5q "
                  f"(last q score[min={s.min():.4f} max={s.max():.4f} std={s.std():.4f}])")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_model_registers()
    test_mechanism_l1()
    test_roundtrip_forward()
    print("\n[PASS] V6-W6 selector 통합 smoke 전체 통과")
