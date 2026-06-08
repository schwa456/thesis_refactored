"""Smoke test for V6-W5 (Phase 5) GAT layer-level intervention — column self-loop + per-layer residual.

검증 항목 (DECISIONS 2026-06-07 + collapse origin §5.2 정합):
  (1) 3 variants (a self-loop / b residual / c a+b) 모델 등록 — self_loop_column relation (a/c),
      v6w5_res_proj (b/c), validation (variant enum + standard/edge_type_split 격리).
  (2) ckpt save/load round-trip (train_gat_s06 형식) + DirectGATv2Selector auto_config + 5q forward
      + score range ∈ [0,1].
  (3) ★ mechanism 검증 (결정론적, 무학습) — single-shared-source collapse 를 W5-a/b/c 가 깨는가:
      hub 테이블 (모든 컬럼이 단일 table 소스만 수신) 위 L1 column intra-MAD —
        - baseline (v6w5=None): 모든 컬럼 동일 메시지 → L1 intra-MAD ≈ 0 (collapse 재현)
        - W5-a/b/c: 컬럼 자기 L0 가 self-loop / residual 로 유입 → L1 intra-MAD > 0 (collapse 차단)
      analyzer L1 게이트 (hi-deg intra-MAD 0.0136 회복) 측정 가능성 + 기제 직접 입증.

학습 ckpt 미존재 단계 — 랜덤 init 모델로 형식 round-trip + 구조적 기제 검증.

Run from project root:
    conda run -n base python src/modules/selectors/tests/test_v6w5_selector.py
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

from models.gat_network_v2 import (
    SchemaHeteroGATv2, V6W5_COLUMN_SELF_LOOP_REL,
)
from models.direct_classifier import DirectClassifierHead

IN, HID, OUT = 384, 256, 256
NUM_LAYERS, HEADS = 3, 4


def _build_graph(n_cols=12, n_fk=2, n_extra_tables=2, seed=42):
    """hub 테이블 1개 (n_cols 컬럼, 단일 table 소스) + 보조 테이블. M4 base relation."""
    g = torch.Generator().manual_seed(seed)
    T = 1 + n_extra_tables
    data = HeteroData()
    data["table"].x = torch.randn(T, IN, generator=g)
    data["column"].x = torch.randn(n_cols, IN, generator=g)
    data["fk_node"].x = torch.randn(n_fk, IN, generator=g)

    # hub table(0) 의 모든 컬럼 → 단일 소스 (table 0 만 incoming). fk edge 0개 — 모든 컬럼이
    # 순수 단일 공유 소스 (collapse mechanism 정확 재현: incoming = table 1.0/col, 그 외 0).
    t_src = [0] * n_cols
    c_dst = list(range(n_cols))
    tc = torch.tensor([t_src, c_dst], dtype=torch.long)
    data["table", "has_column", "column"].edge_index = tc
    data["column", "belongs_to", "table"].edge_index = tc.flip(0)

    # fk_node 는 노드만 존재 (edge 0개) — 컬럼 incoming 에 fk 기여 없도록.
    empty = torch.zeros((2, 0), dtype=torch.long)
    data["column", "is_source_of", "fk_node"].edge_index = empty
    data["fk_node", "points_to", "column"].edge_index = empty

    s = list(range(T - 1)); d = list(range(1, T))
    data["table", "table_to_table", "table"].edge_index = torch.tensor(
        [s + d, d + s], dtype=torch.long)
    return data, n_cols


def _node_metadata(n_cols, n_fk=2, n_extra_tables=2):
    T = 1 + n_extra_tables
    md, idx = {}, 0
    for t in range(T):
        md[idx] = {"type": "table", "name": f"t_{t}"}; idx += 1
    for c in range(n_cols):
        md[idx] = {"type": "column", "name": f"c_{c}"}; idx += 1
    for f in range(n_fk):
        md[idx] = {"type": "fk_node", "name": f"fk_{f}"}; idx += 1
    return md, idx


def _intra_mad(emb):
    """mean pairwise euclidean distance (intra-table MAD proxy)."""
    n = emb.size(0)
    if n < 2:
        return 0.0
    d = torch.cdist(emb, emb)
    return (d.sum() / (n * (n - 1))).item()


def _forward_capture_l1col(variant, device):
    """variant 모델 (capture on) 을 hub graph 위 forward → L1 column embedding 반환."""
    m = SchemaHeteroGATv2(
        in_channels=IN, hidden_channels=HID, out_channels=OUT,
        num_layers=NUM_LAYERS, heads=HEADS, query_conditioned=True,
        v6w5_variant=variant, capture_layerwise_outputs=True,
    ).to(device)
    m.eval()
    data, n_cols = _build_graph()
    data = data.to(device)
    q = torch.zeros(1, IN, device=device)
    aug = {nt: torch.cat([x, q.expand(x.size(0), -1)], dim=-1) for nt, x in data.x_dict.items()}
    with torch.no_grad():
        _ = m(aug, data.edge_index_dict)
    l1 = m.get_captured_layer_outputs()[0]["column"]  # 첫 GAT layer = L1
    return l1


def _make_ckpt(variant, path, device):
    model = SchemaHeteroGATv2(
        in_channels=IN, hidden_channels=HID, out_channels=OUT,
        num_layers=NUM_LAYERS, heads=HEADS, query_conditioned=True,
        v6w5_variant=variant,
    ).to(device)
    heads = torch.nn.ModuleDict({
        nt: DirectClassifierHead(in_dim=OUT, hidden_dim=OUT, dropout=0.0).to(device)
        for nt in ["table", "column", "fk_node"]
    })
    data, _ = _build_graph()
    data = data.to(device)
    q = torch.zeros(1, IN, device=device)
    aug = {nt: torch.cat([x, q.expand(x.size(0), -1)], dim=-1) for nt, x in data.x_dict.items()}
    model.eval()
    with torch.no_grad():
        _ = model(aug, data.edge_index_dict)
    cfg = {
        "experiment_name": f"v6w5_{variant}_s11_smoke",
        "builder": {"type": "EnrichedHeteroGraphBuilder"},
        "model": {
            "in_channels": IN, "hidden_channels": HID, "out_channels": OUT,
            "num_layers": NUM_LAYERS, "heads": HEADS, "classifier_hidden": OUT,
            "query_conditioned": True, "query_supernode": False,
            "v6w5_variant": variant,
        },
    }
    torch.save({"epoch": 0, "gat_state_dict": model.state_dict(),
                "classifier_state_dict": heads.state_dict(),
                "recall": 0.0, "config": cfg}, path)


def test_model_registers():
    """(1) 3 variants 등록 + validation."""
    print("\n=== (1) v6w5_variant 등록 + validation ===")
    for variant in ["a", "b", "c"]:
        m = SchemaHeteroGATv2(
            in_channels=IN, hidden_channels=HID, out_channels=OUT,
            num_layers=NUM_LAYERS, heads=HEADS, query_conditioned=True,
            v6w5_variant=variant,
        )
        has_sl = V6W5_COLUMN_SELF_LOOP_REL in m.all_edge_types
        has_res = m.v6w5_res_proj is not None
        if variant in ("a", "c"):
            assert has_sl, f"{variant}: self_loop_column relation 누락"
        else:
            assert not has_sl, f"{variant}: self_loop 불필요한데 등록됨"
        if variant in ("b", "c"):
            assert has_res, f"{variant}: v6w5_res_proj 누락"
        else:
            assert not has_res, f"{variant}: res_proj 불필요한데 생성됨"
        print(f"  [OK] {variant}: column_self_loop={m.v6w5_column_self_loop} "
              f"per_layer_residual={m.v6w5_per_layer_residual}")
    # validation: edge_type_split 결합 금지
    for bad in [dict(edge_type_split=True), dict(gat_layer_type="gcnii"), dict(v6w3_variant="A")]:
        try:
            SchemaHeteroGATv2(in_channels=IN, hidden_channels=HID, out_channels=OUT,
                              num_layers=NUM_LAYERS, heads=HEADS, query_conditioned=True,
                              v6w5_variant="a", **bad)
            raise AssertionError(f"validation 미작동: {bad}")
        except ValueError:
            pass
    print("  [OK] validation: edge_type_split / gat_layer_type / v6w3_variant 결합 차단")


def test_mechanism_l1_collapse_break():
    """(3) ★ single-shared-source collapse 를 W5-a/b/c 가 깨는가 (결정론적)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    print("\n=== (3) mechanism — hub L1 column intra-MAD (collapse break) ===")
    base = _intra_mad(_forward_capture_l1col(None, device))
    print(f"  baseline (v6w5=None): L1 intra-MAD = {base:.6e}  (단일 소스 → collapse 예상 ≈0)")
    assert base < 1e-4, f"baseline 이 collapse 되지 않음 (intra-MAD={base:.6e}) — 기제 전제 위반"
    for variant in ["a", "b", "c"]:
        mad = _intra_mad(_forward_capture_l1col(variant, device))
        print(f"  W5-{variant}: L1 intra-MAD = {mad:.6e}  (Δ vs baseline = {mad - base:+.6e})")
        assert mad > 1e-3, f"W5-{variant}: collapse 차단 실패 (intra-MAD={mad:.6e})"
        assert mad > base * 100, f"W5-{variant}: baseline 대비 회복 불충분"
    print("  → baseline collapse 재현 + W5-a/b/c 전부 L1 분화 회복 (기제 직접 입증)")


def test_roundtrip_forward():
    """(2) ckpt round-trip + selector 5q forward + score range."""
    from modules.selectors import DirectGATv2Selector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=== (2) ckpt round-trip + forward × 5q (3 variants) ===")
    questions = [
        "What is the average salary by department?",
        "List customers with orders in 2023.",
        "How many products are electronics?",
        "Top 5 schools by enrollment.",
        "Patients with more than three visits.",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for variant in ["a", "b", "c"]:
            path = os.path.join(tmp, f"v6w5_{variant}.pt")
            _make_ckpt(variant, path, device)
            sel = DirectGATv2Selector(weight_path=path)
            assert sel.gat_model.v6w5_variant == variant, f"{variant}: auto-config 복원 실패"
            md, num_nodes = _node_metadata(12)
            cand = list(range(num_nodes))
            for qi, question in enumerate(questions):
                data, _ = _build_graph(seed=100 + qi)
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
    test_mechanism_l1_collapse_break()
    test_roundtrip_forward()
    print("\n[PASS] V6-W5 selector 통합 smoke 전체 통과")
