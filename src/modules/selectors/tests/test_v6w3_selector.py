"""Smoke test for V6-W3 (hub 차수 축소) selector 통합 — SchemaHeteroGATv2 v6w3_variant.

검증 항목 (DECISIONS 2026-06-06 + v6w3_builders.py 정합):
  (1) 3 variants (A table_summary / B column-pooling / C local_vn) 모델 생성 + 신규 node/edge
      type 등록 (node_types / all_edge_types 확장) 확인.
  (2) ckpt save/load 무오류 — train_gat_s06.py 와 동일 형식 (gat_state_dict + classifier_state_dict
      + epoch + recall + config) round-trip, DirectGATv2Selector auto_config 복원.
  (3) 각 variant × 5q forward pass 무오류 + score range 정상 (sigmoid ∈ [0,1], NaN/Inf 없음).

학습 ckpt 미존재 단계 (root 학습 선행 전) — 랜덤 init 모델로 형식/경로 round-trip 만 검증.

Run from project root:
    conda run -n base python src/modules/selectors/tests/test_v6w3_selector.py
"""
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np
import torch
from torch_geometric.data import HeteroData

from models.gat_network_v2 import SchemaHeteroGATv2, V6W3_NODE_TYPE, V6W3_EDGE_TYPES
from models.direct_classifier import DirectClassifierHead

IN, HID, OUT = 384, 256, 256
NUM_LAYERS, HEADS = 3, 4
# table 별 column 수 — 가변 (variant C hub = col > median 검출용). median([3,8,6,10])=7 → hub={8,10}.
TABLE_COLS = [3, 8, 6, 10]
NUM_FK = 3


def _build_v6w3_graph(variant, seed=42):
    """v6w3_builders.py 의 edge 구성을 그대로 따른 합성 graph (variant A/B/C)."""
    g = torch.Generator().manual_seed(seed)
    T = len(TABLE_COLS)
    C = sum(TABLE_COLS)
    data = HeteroData()
    data["table"].x = torch.randn(T, IN, generator=g)
    data["column"].x = torch.randn(C, IN, generator=g)
    data["fk_node"].x = torch.randn(NUM_FK, IN, generator=g)

    # has_column / belongs_to
    t_src, c_dst, table_col_ids, off = [], [], [], 0
    for t, ncol in enumerate(TABLE_COLS):
        ids = list(range(off, off + ncol))
        table_col_ids.append(ids)
        for c in ids:
            t_src.append(t); c_dst.append(c)
        off += ncol
    tc = torch.tensor([t_src, c_dst], dtype=torch.long)
    data["table", "has_column", "column"].edge_index = tc
    data["column", "belongs_to", "table"].edge_index = tc.flip(0)

    # fk (col i -> fk i, i < NUM_FK)
    fk = list(range(NUM_FK))
    fke = torch.tensor([fk, fk], dtype=torch.long)
    data["column", "is_source_of", "fk_node"].edge_index = fke
    data["fk_node", "points_to", "column"].edge_index = fke.flip(0)

    # t2t
    s = list(range(T - 1)); d = list(range(1, T))
    data["table", "table_to_table", "table"].edge_index = torch.tensor(
        [s + d, d + s], dtype=torch.long)

    if variant == "A":
        # table_summary: 1대1 with table. mean-pool features.
        summ = torch.stack([data["column"].x[ids].mean(0) for ids in table_col_ids])
        data["table_summary"].x = summ
        ts = torch.tensor([list(range(T)), list(range(T))], dtype=torch.long)
        data["table", "has_summary", "table_summary"].edge_index = ts
        data["table_summary", "summary_of", "table"].edge_index = ts.flip(0)
        sc_s, sc_d = [], []
        for t, ids in enumerate(table_col_ids):
            for c in ids:
                sc_s.append(t); sc_d.append(c)
        sc = torch.tensor([sc_s, sc_d], dtype=torch.long)
        data["table_summary", "summarizes", "column"].edge_index = sc
        data["column", "aggregated_by", "table_summary"].edge_index = sc.flip(0)

    elif variant == "C":
        # hub = col count > median
        counts = np.array(TABLE_COLS)
        thr = float(np.median(counts))
        hubs = [t for t, c in enumerate(TABLE_COLS) if c > thr]
        H = len(hubs)
        assert H > 0, "smoke graph 는 hub 가 1개 이상이어야 local_vn conv 가 exercise 됨"
        vn_x = torch.stack([data["column"].x[table_col_ids[h]].mean(0) for h in hubs])
        data["local_vn"].x = vn_x
        ht_s, ht_d, hc_s, hc_d = [], [], [], []
        for vn, h in enumerate(hubs):
            ht_s.append(h); ht_d.append(vn)
            for c in table_col_ids[h]:
                hc_s.append(vn); hc_d.append(c)
        ht = torch.tensor([ht_s, ht_d], dtype=torch.long)
        data["table", "has_local_vn", "local_vn"].edge_index = ht
        data["local_vn", "serves_table", "table"].edge_index = ht.flip(0)
        hc = torch.tensor([hc_s, hc_d], dtype=torch.long)
        data["local_vn", "aggregates", "column"].edge_index = hc
        data["column", "feeds_into", "local_vn"].edge_index = hc.flip(0)

    return data


def _node_metadata():
    """selection target = table/column/fk_node 만 (summary/local_vn 비-target)."""
    T, C = len(TABLE_COLS), sum(TABLE_COLS)
    md, idx = {}, 0
    for t in range(T):
        md[idx] = {"type": "table", "name": f"t_{t}"}; idx += 1
    for c in range(C):
        md[idx] = {"type": "column", "name": f"c_{c}"}; idx += 1
    for f in range(NUM_FK):
        md[idx] = {"type": "fk_node", "name": f"fk_{f}"}; idx += 1
    return md, idx


def _make_ckpt(variant, path, device):
    """랜덤 init 모델을 train_gat_s06.py 형식으로 저장 (lazy param init 포함)."""
    model = SchemaHeteroGATv2(
        in_channels=IN, hidden_channels=HID, out_channels=OUT,
        num_layers=NUM_LAYERS, heads=HEADS,
        query_conditioned=True, query_supernode=False,
        v6w3_variant=variant,
    ).to(device)
    heads = torch.nn.ModuleDict({
        nt: DirectClassifierHead(in_dim=OUT, hidden_dim=OUT, dropout=0.0).to(device)
        for nt in ["table", "column", "fk_node"]
    })
    # lazy param init: variant 별 graph 1회 forward (query concat)
    data = _build_v6w3_graph(variant).to(device)
    q = torch.zeros(1, IN, device=device)
    aug = {nt: torch.cat([x, q.expand(x.size(0), -1)], dim=-1) for nt, x in data.x_dict.items()}
    model.eval()
    with torch.no_grad():
        _ = model(aug, data.edge_index_dict)
    cfg = {
        "experiment_name": f"v6w3_{variant.lower()}_s11_smoke",
        "builder": {"type": {"A": "V6W3VirtualSummaryBuilder",
                              "B": "V6W3ColumnPoolingBuilder",
                              "C": "V6W3HubLocalVNBuilder"}[variant]},
        "model": {
            "in_channels": IN, "hidden_channels": HID, "out_channels": OUT,
            "num_layers": NUM_LAYERS, "heads": HEADS, "classifier_hidden": OUT,
            "query_conditioned": True, "query_supernode": False,
            "v6w3_variant": variant,
        },
    }
    torch.save({
        "epoch": 0, "gat_state_dict": model.state_dict(),
        "classifier_state_dict": heads.state_dict(),
        "recall": 0.0, "config": cfg,
    }, path)


def test_model_registers_new_types():
    """(1) 3 variants 모델이 신규 node/edge type 등록."""
    print("\n=== (1) v6w3_variant node/edge type 등록 ===")
    for variant in ["A", "B", "C"]:
        m = SchemaHeteroGATv2(
            in_channels=IN, hidden_channels=HID, out_channels=OUT,
            num_layers=NUM_LAYERS, heads=HEADS, query_conditioned=True,
            v6w3_variant=variant,
        )
        if variant in V6W3_NODE_TYPE:
            assert V6W3_NODE_TYPE[variant] in m.node_types, f"{variant}: node type 누락"
            for et in V6W3_EDGE_TYPES[variant]:
                assert et in m.all_edge_types, f"{variant}: edge type {et} 누락"
            assert V6W3_NODE_TYPE[variant] in m.lin_dict, f"{variant}: lin_dict 누락"
            assert V6W3_NODE_TYPE[variant] in m.skip_dict, f"{variant}: skip_dict 누락"
            print(f"  [OK] {variant}: +node={V6W3_NODE_TYPE[variant]} "
                  f"+{len(V6W3_EDGE_TYPES[variant])} edge types | node_types={m.node_types}")
        else:  # B
            assert m.node_types == ["table", "column", "fk_node"], "B: 구조 변경 없어야 함"
            assert m.v6w3_variant == "B"
            print(f"  [OK] B: 구조 무변경 (column pooling builder-side) node_types={m.node_types}")


def test_roundtrip_and_forward():
    """(2)+(3) ckpt save/load round-trip + 5q forward + score range."""
    from modules.selectors import DirectGATv2Selector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=== (2)+(3) ckpt round-trip + forward × 5q (3 variants) ===")
    questions = [
        "What is the average salary by department?",
        "List customers with orders in 2023.",
        "How many products are electronics?",
        "Top 5 schools by enrollment.",
        "Patients with more than three visits.",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for variant in ["A", "B", "C"]:
            path = os.path.join(tmp, f"v6w3_{variant}_smoke.pt")
            _make_ckpt(variant, path, device)
            sel = DirectGATv2Selector(weight_path=path)
            assert sel.gat_model.v6w3_variant == variant, f"{variant}: auto-config 복원 실패"
            md, num_nodes = _node_metadata()
            cand = list(range(num_nodes))
            for qi, question in enumerate(questions):
                data = _build_v6w3_graph(variant, seed=100 + qi)
                out = sel.select(scores=None, candidates=cand, question=question,
                                 graph_data=data, metadata={"node_metadata": md})
                s = torch.tensor(sel.latest_scores)
                assert len(sel.latest_scores) == num_nodes, f"{variant} q{qi}: score 개수 불일치"
                assert torch.isfinite(s).all(), f"{variant} q{qi}: NaN/Inf"
                assert (s >= 0).all() and (s <= 1).all(), f"{variant} q{qi}: 범위 위반"
                assert out == cand, f"{variant} q{qi}: candidates 보존 위반"
            print(f"  [OK] variant {variant}: round-trip + 5q forward "
                  f"(last q score[min={s.min():.4f} max={s.max():.4f} std={s.std():.4f}])")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_model_registers_new_types()
    test_roundtrip_and_forward()
    print("\n[PASS] V6-W3 selector 통합 smoke 전체 통과")
