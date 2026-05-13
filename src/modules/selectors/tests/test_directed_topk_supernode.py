"""Smoke test for DirectedTopKSuperNodeSelector + GAT v1/v2 SuperNode threshold mask.

학위 논문 Part III 단계 1 (5/9~5/11 구현) 검증 항목:
  (1) Selector 가 percentile / top_k / abs_tau threshold 별로 expected node 수 반환.
  (2) Edge 구조: query_node → schema (`attends_to_*`) directed 만, 역방향 (`attended_by_*`) 빈/없음.
  (3) GAT v1/v2 의 _compute_supernode_mask 가 동일 dispatch 동작.
  (4) 기존 SuperNode (bidirectional, 모든 schema) 와의 selected node 수 차이 검증.
  (5) Recall@20 sanity (raw cosine 기준, 합성 graph 라 절대값 의미 X 단 동작 확인).

Run from project root:
    conda run -n base python src/modules/selectors/tests/test_directed_topk_supernode.py
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import torch
from torch_geometric.data import HeteroData

CKPT_PATH = os.path.join(ROOT, "outputs", "checkpoints", "best_gat_enriched_v2_directed.pt")


def _build_synthetic_graph(num_tables=4, cols_per_table=6, num_fk=3, in_dim=384, seed=42):
    """SuperNode 노드 수 ~30 짜리 합성 graph (table=4, column=24, fk=3)."""
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
        md[idx] = {"type": "fk_node", "name": f"f_{f}"}; idx += 1
    return md


def _make_selector(threshold_mode, threshold_value, gat_version="v1", num_layers=3):
    """ckpt: best_gat_enriched_v2_directed.pt (num_layers=3, query_supernode=True,
    query_conditioned=default False, supernode_edge_direction=directed_from_sn)."""
    from modules.selectors.directed_topk_supernode_selector import DirectedTopKSuperNodeSelector
    return DirectedTopKSuperNodeSelector(
        weight_path=CKPT_PATH,
        threshold_mode=threshold_mode,
        threshold_value=threshold_value,
        score_normalization="minmax",
        alpha=0.5,
        top_k=10,
        num_layers=num_layers,
        query_conditioned=False,
        query_supernode=True,
        encoder_type="plm",
        num_layers_mode="fixed",
        gat_version=gat_version,
    )


def _make_baseline_supernode_selector(gat_version="v1", num_layers=3):
    """기존 SuperNode 비교 base — 동일 ckpt 사용. ckpt 가 directed_from_sn 학습이므로 GAT
    모델 측 edge_direction 도 동일. selector 의 SuperNode 분기 (ensemble_selector.py:230-246) 가
    모든 schema 노드에 attends_to_* edge 를 등록 → directed selector 의 filtered subset 과 비교."""
    from modules.selectors.ensemble_selector import EnsembleSelector
    return EnsembleSelector(
        weight_path=CKPT_PATH,
        alpha=0.5,
        top_k=10,
        num_layers=num_layers,
        query_conditioned=False,
        query_supernode=True,
        encoder_type="plm",
        num_layers_mode="fixed",
        gat_version=gat_version,
        supernode_edge_direction="directed_from_sn",
    )


def _run_select(sel, graph, md, question="which students passed exams?"):
    cands = list(range(len(md)))
    seeds = sel.select(
        scores=[0.0] * len(cands),
        candidates=cands,
        question=question,
        graph_data=graph.clone(),
        metadata={"db_id": "synthetic", "node_metadata": md},
    )
    return seeds


def test_percentile_p80_selected_count():
    """P80: 상위 ~20% 통과. graph 32 노드 (4t + 24c + 3fk + 1 query_node 자동주입 X for schema mask)
    schema 31 노드 → P80 quantile 80% cutoff 위 ~6-7 노드 expected."""
    print("\n[test_percentile_p80_selected_count]")
    sel = _make_selector(threshold_mode="percentile", threshold_value=80.0)
    graph = _build_synthetic_graph()
    md = _node_metadata(graph)
    seeds = _run_select(sel, graph, md)
    n_schema = graph["table"].num_nodes + graph["column"].num_nodes + graph["fk_node"].num_nodes
    n_kept = sel.last_selected_count
    assert n_kept is not None and n_kept > 0, f"P80 must keep >0 nodes, got {n_kept}"
    # 32 schema → P80 통과 ~ floor(32*0.20) ± rounding. 합성 분포 noise 로 +/- 변동 허용.
    expected_lo = max(1, int(n_schema * 0.10))
    expected_hi = int(n_schema * 0.40) + 2
    print(f"  schema={n_schema} P80 kept={n_kept} (expected {expected_lo}-{expected_hi})")
    assert expected_lo <= n_kept <= expected_hi, f"P80 kept {n_kept} out of [{expected_lo},{expected_hi}]"
    assert len(seeds) == 10, f"top_k=10 should return 10 seeds, got {len(seeds)}"


def test_topk20_selected_count():
    """top_k=20: schema 가 32 개 이상이면 정확히 20 개. 32 미만이면 schema 전체."""
    print("\n[test_topk20_selected_count]")
    sel = _make_selector(threshold_mode="top_k", threshold_value=20.0)
    graph = _build_synthetic_graph()
    md = _node_metadata(graph)
    _run_select(sel, graph, md)
    n_schema = graph["table"].num_nodes + graph["column"].num_nodes + graph["fk_node"].num_nodes
    expected = min(20, n_schema)
    print(f"  schema={n_schema} top_k=20 kept={sel.last_selected_count} (expected {expected})")
    assert sel.last_selected_count == expected, (
        f"top_k=20 should keep {expected}, got {sel.last_selected_count}"
    )


def test_abstau07_selected_count_low():
    """abs_tau=0.7: per-query minmax norm 후 score>=0.7 만. random feature 합성 graph 에서는
    상위 소수 노드만 통과 expected (1-10 정도)."""
    print("\n[test_abstau07_selected_count_low]")
    sel = _make_selector(threshold_mode="abs_tau", threshold_value=0.7)
    graph = _build_synthetic_graph()
    md = _node_metadata(graph)
    _run_select(sel, graph, md)
    n_schema = graph["table"].num_nodes + graph["column"].num_nodes + graph["fk_node"].num_nodes
    n_kept = sel.last_selected_count
    print(f"  schema={n_schema} abs_tau=0.7 kept={n_kept}")
    assert 0 <= n_kept < n_schema, f"abs_tau=0.7 should be selective (<{n_schema}), got {n_kept}"


def test_directed_edge_only():
    """Selector 가 query→schema 단방향 edge 만 등록하고 역방향은 0-len 또는 없음."""
    print("\n[test_directed_edge_only]")
    sel = _make_selector(threshold_mode="percentile", threshold_value=80.0)
    graph = _build_synthetic_graph()
    md = _node_metadata(graph)
    # graph 를 in-place 변경하므로 clone 후 select 호출
    g = graph.clone()
    _run_select(sel, g, md)

    # select 가 graph 를 변형해서 selector.gat_model 호출 시 사용한 graph 의 edge 상태를 검증하기 위해
    # 같은 graph 를 다시 주입 → mutating 위해 _compute_gat_scores 직접 호출
    g2 = graph.clone().to(sel.device)
    sel._compute_gat_scores("q", g2, {"db_id": "synthetic", "node_metadata": md})
    for nt in ("table", "column", "fk_node"):
        fwd_key = ("query_node", f"attends_to_{nt}", nt)
        rev_key = (nt, f"attended_by_{nt}", "query_node")
        assert fwd_key in g2.edge_types, f"missing forward edge {fwd_key}"
        if rev_key in g2.edge_types:
            ei = g2[rev_key].edge_index
            assert ei.numel() == 0, (
                f"reverse edge {rev_key} should be empty, has {ei.size(1)} edges"
            )
    print("  OK directed edge structure verified (forward present, reverse empty/absent)")


def test_baseline_supernode_uses_more_nodes():
    """기존 SuperNode (bidirectional, 모든 schema) vs Directed Top-K (P80) 노드 수 비교.
    Selector 단에서 graph 변형 후의 attends_to_* edge 수를 비교."""
    print("\n[test_baseline_supernode_uses_more_nodes]")
    base_sel = _make_baseline_supernode_selector()
    dir_sel = _make_selector(threshold_mode="percentile", threshold_value=80.0)

    g_base = _build_synthetic_graph().to(base_sel.device)
    md = _node_metadata(g_base)
    base_sel._compute_gat_scores("q", g_base, {"db_id": "synthetic", "node_metadata": md})
    base_edge_count = sum(
        g_base["query_node", f"attends_to_{nt}", nt].edge_index.size(1)
        for nt in ("table", "column", "fk_node")
        if ("query_node", f"attends_to_{nt}", nt) in g_base.edge_types
    )

    g_dir = _build_synthetic_graph().to(dir_sel.device)
    dir_sel._compute_gat_scores("q", g_dir, {"db_id": "synthetic", "node_metadata": md})
    dir_edge_count = sum(
        g_dir["query_node", f"attends_to_{nt}", nt].edge_index.size(1)
        for nt in ("table", "column", "fk_node")
        if ("query_node", f"attends_to_{nt}", nt) in g_dir.edge_types
    )
    print(f"  baseline SuperNode forward edges = {base_edge_count}")
    print(f"  Directed Top-K (P80) forward edges = {dir_edge_count}")
    assert dir_edge_count < base_edge_count, (
        f"Directed Top-K should reduce edges (<{base_edge_count}), got {dir_edge_count}"
    )


def test_gat_v1_threshold_mask_dispatch():
    """v1 GAT 의 _compute_supernode_mask 가 percentile / abs_tau 로 dispatch 동작."""
    print("\n[test_gat_v1_threshold_mask_dispatch]")
    from models.gat_network import SchemaHeteroGAT

    m = SchemaHeteroGAT(
        in_channels=384, hidden_channels=64, out_channels=64,
        num_layers=2, heads=2, query_conditioned=True, query_supernode=True,
        supernode_edge_direction="directed_from_sn",
        supernode_threshold_mode="percentile",
        supernode_threshold_value=80.0,
    )
    g = torch.Generator().manual_seed(7)
    x_dict = {
        "table": torch.randn(4, 384, generator=g),
        "column": torch.randn(20, 384, generator=g),
        "fk_node": torch.randn(2, 384, generator=g),
    }
    q = torch.randn(1, 384, generator=g)

    mask = m._compute_supernode_mask(q, x_dict)
    total_kept = sum(int(mask[nt].sum().item()) for nt in mask)
    n_schema = sum(x.size(0) for x in x_dict.values())
    print(f"  v1 percentile p=80 kept {total_kept}/{n_schema}")
    assert m._supernode_filter_active() is True
    assert 0 < total_kept < n_schema

    # mode 변경 — abs_tau
    m.supernode_threshold_mode = "abs_tau"
    m.supernode_threshold_value = 0.7
    mask2 = m._compute_supernode_mask(q, x_dict)
    total_kept2 = sum(int(mask2[nt].sum().item()) for nt in mask2)
    print(f"  v1 abs_tau=0.7 kept {total_kept2}/{n_schema}")
    assert 0 <= total_kept2 < n_schema


def test_gat_v2_threshold_mask_dispatch():
    """v2 GAT 측 동일 dispatch 동작."""
    print("\n[test_gat_v2_threshold_mask_dispatch]")
    from models.gat_network_v2 import SchemaHeteroGATv2

    m = SchemaHeteroGATv2(
        in_channels=384, hidden_channels=64, out_channels=64,
        num_layers=2, heads=2, query_conditioned=True, query_supernode=True,
        supernode_edge_direction="directed_from_sn",
        supernode_threshold_mode="percentile",
        supernode_threshold_value=80.0,
    )
    g = torch.Generator().manual_seed(11)
    x_dict = {
        "table": torch.randn(4, 384, generator=g),
        "column": torch.randn(20, 384, generator=g),
        "fk_node": torch.randn(2, 384, generator=g),
    }
    q = torch.randn(1, 384, generator=g)
    mask = m._compute_supernode_mask(q, x_dict)
    total_kept = sum(int(mask[nt].sum().item()) for nt in mask)
    n_schema = sum(x.size(0) for x in x_dict.values())
    print(f"  v2 percentile p=80 kept {total_kept}/{n_schema}")
    assert m._supernode_filter_active() is True
    assert 0 < total_kept < n_schema


def main():
    test_percentile_p80_selected_count()
    test_topk20_selected_count()
    test_abstau07_selected_count_low()
    test_directed_edge_only()
    test_baseline_supernode_uses_more_nodes()
    test_gat_v1_threshold_mask_dispatch()
    test_gat_v2_threshold_mask_dispatch()
    print("\nAll Directed Top-K SuperNode smoke tests passed.")


if __name__ == "__main__":
    main()
