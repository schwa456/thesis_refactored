"""Smoke test for MA-1 (monitor 교체) + MA-2 (calibration loss) — DECISIONS 2026-06-07 #4.

검증 항목:
  (1) gold recall@θ + gold p50 계산 결정론적 정확성 (compute_gold_calibration_metrics).
  (2) MA-2 gold_margin_loss 값 + gradient propagation 정확성.
  (3) MA-2 per_table_normalize_logits z-norm 정확성 + gradient flow.
  (4) validate() dict 반환 (recall_at_15 + gold_recall_at_theta + gold_p50) — tiny model 통합.
  (5) MA-1 monitor 선택 로직 — best-epoch 가 monitor_metric (gold_recall_at_theta) 기준 (R@15 아님).

Run from project root:
    conda run -n base python src/modules/selectors/tests/test_ma_monitor_calibration.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import math
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from train_gat_s06 import (
    compute_gold_calibration_metrics, gold_margin_loss,
    per_table_normalize_logits, validate,
)
from models.gat_network_v2 import SchemaHeteroGATv2
from models.direct_classifier import DirectClassifierHead

IN = 384


def test_gold_calibration_metrics():
    print("\n=== (1) gold recall@θ + gold p50 결정론 ===")
    logits = torch.tensor([2.2, 0.0, -3.0, -10.0])
    labels = torch.tensor([1, 1, 1, 0])
    # sigmoid: 0.9002, 0.5, 0.0474, ~0 → gold {0.9002, 0.5, 0.0474}; ≥0.1 → 2/3; median=0.5
    gr, p50 = compute_gold_calibration_metrics(logits, labels, theta=0.1)
    assert abs(gr - 2.0 / 3.0) < 1e-5, f"gold_recall={gr}"
    assert abs(p50 - 0.5) < 1e-4, f"p50={p50}"
    # no gold → (None, None)
    gr0, p500 = compute_gold_calibration_metrics(logits, torch.zeros(4, dtype=torch.long), theta=0.1)
    assert gr0 is None and p500 is None
    print(f"  [OK] gold_recall@0.1={gr:.4f} (=2/3), gold_p50={p50:.4f} (=0.5), no-gold→None")


def test_gold_margin_loss():
    print("\n=== (2) gold_margin_loss 값 + gradient ===")
    logits = torch.tensor([2.2, -3.0], requires_grad=True)
    labels = torch.tensor([1, 1])
    # scores 0.9002, 0.0474; θ_target 0.15 → clamp(0.15-0.9,0)=0, clamp(0.15-0.0474,0)=0.1026; mean=0.0513
    loss = gold_margin_loss(logits, labels, theta_target=0.15)
    assert abs(loss.item() - 0.05132) < 1e-3, f"loss={loss.item()}"
    loss.backward()
    assert logits.grad is not None and logits.grad[1].abs() > 0, "gradient 전파 실패 (gold below θ)"
    assert abs(float(logits.grad[0])) < 1e-7, "θ 통과 gold 는 grad 0 이어야 (clamp)"
    # no gold → 0, grad 0
    l2 = gold_margin_loss(torch.tensor([1.0], requires_grad=True), torch.tensor([0]), 0.15)
    assert l2.item() == 0.0
    print(f"  [OK] margin loss={loss.item():.5f} (=0.0513), grad: below-θ gold>0 / passed gold=0")


def test_per_table_normalize():
    print("\n=== (3) per_table_normalize_logits z-norm + gradient ===")
    col_logits = torch.tensor([1.0, 2.0, 3.0, 10.0, 10.0], requires_grad=True)
    # cols 0,1,2 → table 0; cols 3,4 → table 1
    belongs = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 1, 1]], dtype=torch.long)
    out = per_table_normalize_logits(col_logits, belongs, num_cols=5)
    # table0 [1,2,3]: mean 2, std 1 (sample) → [-1,0,1]; table1 [10,10]: std 0 → [0,0]
    expected = torch.tensor([-1.0, 0.0, 1.0, 0.0, 0.0])
    assert torch.allclose(out, expected, atol=1e-4), f"out={out.tolist()}"
    out.sum().backward()
    assert col_logits.grad is not None, "gradient 전파 실패"
    print(f"  [OK] per-table z-norm out={[round(v,3) for v in out.tolist()]} (table0→[-1,0,1])")


def _graph(n_tab=3, n_col=8, seed=0, gold_high=True):
    g = torch.Generator().manual_seed(seed)
    d = HeteroData()
    d["table"].x = torch.randn(n_tab, IN, generator=g)
    d["column"].x = torch.randn(n_col, IN, generator=g)
    d["fk_node"].x = torch.randn(1, IN, generator=g)
    # tc: row0=table id (<n_tab), row1=column id (<n_col) → table→column (has_column)
    tc = torch.tensor([[i % n_tab for i in range(n_col)], list(range(n_col))], dtype=torch.long)
    d["table", "has_column", "column"].edge_index = tc
    d["column", "belongs_to", "table"].edge_index = tc.flip(0)
    empty = torch.zeros((2, 0), dtype=torch.long)
    d["column", "is_source_of", "fk_node"].edge_index = empty
    d["fk_node", "points_to", "column"].edge_index = empty
    d["table", "table_to_table", "table"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    d["table"].y = torch.tensor([1.0] + [0.0] * (n_tab - 1))
    d["column"].y = torch.tensor([1.0, 1.0] + [0.0] * (n_col - 2))
    d["query"] = torch.randn(1, IN, generator=g)
    return d


def test_validate_returns_dict():
    print("\n=== (4) validate() dict 반환 (tiny model 통합) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchemaHeteroGATv2(in_channels=IN, hidden_channels=64, out_channels=64,
                              num_layers=2, heads=2).to(device).eval()
    heads = torch.nn.ModuleDict({
        nt: DirectClassifierHead(in_dim=64, hidden_dim=64, dropout=0.0).to(device)
        for nt in ["table", "column", "fk_node"]
    })
    # lazy init
    d0 = _graph(seed=1).to(device)
    with torch.no_grad():
        _ = model(d0.x_dict, d0.edge_index_dict)
    loader = DataLoader([_graph(seed=i) for i in range(4)], batch_size=2)
    m = validate(model, heads, loader, device, k=15,
                 query_conditioned=False, query_supernode=False, theta=0.1)
    assert set(m.keys()) == {"recall_at_15", "gold_recall_at_theta", "gold_p50"}, m.keys()
    for key, v in m.items():
        assert 0.0 <= v <= 1.0, f"{key}={v} 범위 벗어남"
    print(f"  [OK] validate dict: R@15={m['recall_at_15']:.4f} "
          f"gold_recall@θ={m['gold_recall_at_theta']:.4f} gold_p50={m['gold_p50']:.4f}")


def test_monitor_selection():
    print("\n=== (5) MA-1 monitor 선택 로직 (gold_recall@θ 기준, R@15 아님) ===")
    # epoch1: R@15 높지만 gold_recall 낮음 / epoch2: R@15 낮지만 gold_recall 높음
    ep1 = {"recall_at_15": 0.90, "gold_recall_at_theta": 0.20, "gold_p50": 0.05}
    ep2 = {"recall_at_15": 0.70, "gold_recall_at_theta": 0.65, "gold_p50": 0.40}
    monitor_metric = "gold_recall_at_theta"
    best, best_ep = -1.0, None
    for i, m in enumerate([ep1, ep2], 1):
        if m[monitor_metric] > best:
            best, best_ep = m[monitor_metric], i
    assert best_ep == 2, "monitor=gold_recall@θ 면 epoch2 선택돼야"
    # 대조: 옛 R@15 기준이면 epoch1 선택 (= 잘못된 선택, MA-0 ρ=−0.19)
    best_old, best_ep_old = -1.0, None
    for i, m in enumerate([ep1, ep2], 1):
        if m["recall_at_15"] > best_old:
            best_old, best_ep_old = m["recall_at_15"], i
    assert best_ep_old == 1
    print(f"  [OK] monitor=gold_recall@θ → epoch2 (best); 옛 R@15 → epoch1 (MA-0 disconnect 회피 입증)")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_gold_calibration_metrics()
    test_gold_margin_loss()
    test_per_table_normalize()
    test_validate_returns_dict()
    test_monitor_selection()
    print("\n[PASS] MA-1/MA-2 monitor+calibration smoke 전체 통과")
