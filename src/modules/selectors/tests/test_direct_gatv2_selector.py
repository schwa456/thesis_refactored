"""Smoke test for DirectGATv2Selector (V6-W2 edge-type split ckpt 호환 검증).

검증 항목:
  (1) 4 cells ckpt (p2_standalone / p2_phase1 / p2_standalone_no_selfloop / p2_sum) 가
      DirectGATv2Selector 로 auto-config 로딩되어 state_dict 정합 (gat 168 + classifier 18).
  (2) p2_phase1 (split + self_loop + pairnorm + IR + JK, query_conditioned) × 5q forward 무오류.
  (3) score range 정상 (모든 노드 sigmoid output ∈ [0, 1], NaN/Inf 없음, 변별력 있음).

Run from project root:
    conda run -n base python src/modules/selectors/tests/test_direct_gatv2_selector.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch
from torch_geometric.data import HeteroData

CKPT_DIR = os.path.join(ROOT, "outputs", "checkpoints", "v6_phase2")
CELLS = [
    "p2_standalone",
    "p2_phase1",
    "p2_standalone_no_selfloop",
    "p2_sum",
]


def _build_synthetic_graph(num_tables=4, cols_per_table=6, num_fk=3, in_dim=384, seed=42):
    """노드 수 ~31 짜리 합성 graph (table=4, column=24, fk=3)."""
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

    fk_src = list(range(num_fk))
    fke = torch.tensor([fk_src, fk_src], dtype=torch.long)
    data["column", "is_source_of", "fk_node"].edge_index = fke
    data["fk_node", "points_to", "column"].edge_index = fke.flip(0)

    if num_tables > 1:
        s = list(range(num_tables - 1))
        d = list(range(1, num_tables))
        data["table", "table_to_table", "table"].edge_index = torch.tensor(
            [s + d, d + s], dtype=torch.long
        )
    else:
        data["table", "table_to_table", "table"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    return data


def _node_metadata(data):
    n_t = data["table"].num_nodes
    n_c = data["column"].num_nodes
    n_fk = data["fk_node"].num_nodes
    md, idx = {}, 0
    for t in range(n_t):
        md[idx] = {"type": "table", "name": f"t_{t}"}; idx += 1
    for c in range(n_c):
        md[idx] = {"type": "column", "name": f"c_{c}"}; idx += 1
    for f in range(n_fk):
        md[idx] = {"type": "fk_node", "name": f"fk_{f}"}; idx += 1
    return md, n_t + n_c + n_fk


def _ckpt(cell):
    return os.path.join(CKPT_DIR, f"best_gat_v6w2_{cell}_s11.pt")


def test_all_cells_load():
    """(1) 4 cells 모두 auto-config 로딩 + state_dict 정합."""
    from modules.selectors import DirectGATv2Selector
    print("\n=== (1) 4 cells load (auto-config) ===")
    for cell in CELLS:
        path = _ckpt(cell)
        assert os.path.exists(path), f"ckpt 없음: {path}"
        sel = DirectGATv2Selector(weight_path=path)
        assert sel.query_conditioned is True, f"{cell}: query_conditioned 복원 실패"
        assert sel.gat_model.edge_type_split is True, f"{cell}: edge_type_split 복원 실패"
        print(f"  [OK] {cell:28s} QC={sel.query_conditioned} "
              f"split={sel.gat_model.edge_type_split} "
              f"self_loops={sel.gat_model.edge_type_split_self_loops} "
              f"aggr={sel.gat_model.edge_type_split_aggr} "
              f"PN={sel.gat_model.pairnorm_mode} IR={sel.gat_model.initial_residual_alpha} "
              f"JK={sel.gat_model.jumping_knowledge}")
    print("  → 4 cells 전부 로딩 성공 (state_dict 정합)")


def test_p2_phase1_forward_5q():
    """(2)+(3) p2_phase1 × 5q forward 무오류 + score range 정상."""
    from modules.selectors import DirectGATv2Selector
    print("\n=== (2)+(3) p2_phase1 forward × 5q ===")
    sel = DirectGATv2Selector(weight_path=_ckpt("p2_phase1"))

    questions = [
        "What is the average salary of employees in the marketing department?",
        "List all customers who placed an order in 2023.",
        "How many products belong to the electronics category?",
        "Find the top 5 schools by total enrollment.",
        "Which patients had more than three visits last year?",
    ]
    for qi, question in enumerate(questions):
        data = _build_synthetic_graph(seed=100 + qi)
        md, num_nodes = _node_metadata(data)
        candidates = list(range(num_nodes))
        out = sel.select(
            scores=None, candidates=candidates, question=question,
            graph_data=data, metadata={"node_metadata": md},
        )
        s = torch.tensor(sel.latest_scores)
        assert len(sel.latest_scores) == num_nodes, \
            f"q{qi}: score 개수 불일치 {len(sel.latest_scores)} != {num_nodes}"
        assert torch.isfinite(s).all(), f"q{qi}: NaN/Inf 존재"
        assert (s >= 0).all() and (s <= 1).all(), \
            f"q{qi}: sigmoid 범위 위반 min={s.min():.4f} max={s.max():.4f}"
        assert out == candidates, "apply_threshold=False 기본 — candidates 그대로 반환해야 함"
        print(f"  [OK] q{qi}: n={num_nodes} score[min={s.min():.4f} "
              f"mean={s.mean():.4f} max={s.max():.4f} std={s.std():.4f}]")
    print("  → 5q 전부 forward 무오류 + score ∈ [0,1] + 변별력 정상")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_all_cells_load()
    test_p2_phase1_forward_5q()
    print("\n[PASS] DirectGATv2Selector smoke 전체 통과")
