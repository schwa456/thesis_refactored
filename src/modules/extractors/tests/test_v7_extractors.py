"""V7-W2 (FKPathfinding) + V7-W3 (SteinerTree) extractor 단위 테스트.

소규모 합성 heterograph (본 framework metadata dict 형식) 위에서 correctness 검증:
  - STE: top-K terminal 사이를 잇는 최소 Steiner point (bridge fk_node) 발견 + connectivity + compactness.
  - FKP: terminal pair 의 FK 최단 경로 union 으로 bridge fk_node 포함 + use_fk_paths=False (FKP-06) 분리.
  - registry 등록 + registry.build() instantiate 검증.

Run from project root:
    PYTHONPATH=src conda run -n base python src/modules/extractors/tests/test_v7_extractors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import networkx as nx


# ---------------------------------------------------------------------------
# 합성 그래프 (본 framework metadata dict 형식)
#
#   tables  : t0(0) t1(1) t2(2)                              num_t=3
#   columns : t0.a(3) t0.b(4) t1.c(5) t1.d(6) t2.e(7) t2.f(8)  num_c=6
#   fk_nodes: "t0.a->t1.c"(9)  "t1.d->t2.e"(10)               num_fk=2
#   total = 11 nodes
#
#   FK join 구조:  t0.a --(fk9)-- t1.c ,  t1.d --(fk10)-- t2.e
#   (column 간 FK join 은 column → fk_node → column 2-hop)
# ---------------------------------------------------------------------------
def _make_synthetic_graph():
    metadata = {
        "table_to_id": {"t0": 0, "t1": 1, "t2": 2},
        "col_to_id": {"t0.a": 0, "t0.b": 1, "t1.c": 2, "t1.d": 3, "t2.e": 4, "t2.f": 5},
        "fk_to_id": {"t0.a->t1.c": 0, "t1.d->t2.e": 1},
        "node_metadata": {
            0: "t0", 1: "t1", 2: "t2",
            3: "t0.a", 4: "t0.b", 5: "t1.c", 6: "t1.d", 7: "t2.e", 8: "t2.f",
            9: "t0.a->t1.c", 10: "t1.d->t2.e",
        },
        "edges": [
            # belongs_to (table, column)
            (0, 3), (0, 4), (1, 5), (1, 6), (2, 7), (2, 8),
            # is_source_of (column, fk_node)
            (3, 9), (6, 10),
            # points_to (fk_node, column)
            (9, 5), (10, 7),
            # table_to_table (table, table) — 양방향
            (0, 1), (1, 0), (1, 2), (2, 1),
        ],
        "edge_types": (
            ["belongs_to"] * 6
            + ["is_source_of"] * 2
            + ["points_to"] * 2
            + ["table_to_table"] * 4
        ),
    }
    # idx: 0    1    2    3    4    5    6    7    8    9    10
    scores = [0.2, 0.2, 0.2, 0.9, 0.5, 0.85, 0.4, 0.8, 0.3, 0.05, 0.05]
    return metadata, scores


def _induced_connected(sel_nodes, sel_edges) -> bool:
    if not sel_nodes:
        return True
    G = nx.Graph()
    G.add_nodes_from(sel_nodes)
    G.add_edges_from(sel_edges)
    return nx.is_connected(G)


# ===========================================================================
# SteinerTreeExtractor
# ===========================================================================
def test_ste_steiner_point_discovery():
    """k=2 terminals {t0.a, t1.c}, cap_to_k=False → bridge fk_node(9) 가 Steiner point 로 포함."""
    print("\n[test_ste_steiner_point_discovery]")
    from modules.extractors.steiner_tree_extractor import SteinerTreeExtractor

    meta, scores = _make_synthetic_graph()
    ext = SteinerTreeExtractor(k=2, terminal_mode="topk",
                               terminal_node_types=["column"], cap_to_k=False,
                               edge_weight_mode="grast")
    nodes, edges = ext.extract(meta, scores)
    nset = set(nodes)
    assert 3 in nset and 5 in nset, f"terminals {{3,5}} must be selected, got {sorted(nset)}"
    assert 9 in nset, f"bridge fk_node 9 (Steiner point) must be discovered, got {sorted(nset)}"
    # 무관 저점수 컬럼 제외 (compact)
    assert not ({4, 6, 7, 8} & nset), f"irrelevant columns must be excluded, got {sorted(nset)}"
    assert _induced_connected(nodes, edges), "induced subgraph must be connected"
    assert len(nset) < len(scores), "must be compact (smaller than full graph)"
    print(f"  OK selected={sorted(nset)} edges={edges} "
          f"(bridge 9 found, {len(nset)}/{len(scores)} nodes)")


def test_ste_cap_to_k_prioritizes_terminals():
    """k=2 cap_to_k=True → 자리 부족으로 Steiner point 탈락, terminal 만 보존 (RFP §1.10 K-under risk)."""
    print("\n[test_ste_cap_to_k_prioritizes_terminals]")
    from modules.extractors.steiner_tree_extractor import SteinerTreeExtractor

    meta, scores = _make_synthetic_graph()
    ext = SteinerTreeExtractor(k=2, terminal_mode="topk",
                               terminal_node_types=["column"], cap_to_k=True)
    nodes, _ = ext.extract(meta, scores)
    nset = set(nodes)
    assert nset == {3, 5}, f"cap_to_k=2 must keep only terminals {{3,5}}, got {sorted(nset)}"
    assert len(nset) <= 2, "output must respect k cap"
    assert ext.last_info["ste_capped"] is True
    assert ext.last_info["ste_pre_cap_node_count"] >= 3, "pre-cap should include Steiner point"
    print(f"  OK capped to {sorted(nset)} (pre_cap={ext.last_info['ste_pre_cap_node_count']})")


def test_ste_three_terminal_connectivity():
    """k=3 terminals {t0.a, t1.c, t2.e}, cap_off → 전체 연결 + bridge 추가 + compact."""
    print("\n[test_ste_three_terminal_connectivity]")
    from modules.extractors.steiner_tree_extractor import SteinerTreeExtractor

    meta, scores = _make_synthetic_graph()
    ext = SteinerTreeExtractor(k=3, terminal_mode="topk",
                               terminal_node_types=["column"], cap_to_k=False)
    nodes, edges = ext.extract(meta, scores)
    nset = set(nodes)
    for t in (3, 5, 7):
        assert t in nset, f"terminal {t} must be selected, got {sorted(nset)}"
    assert _induced_connected(nodes, edges), f"3-terminal tree must be connected: {sorted(nset)} {edges}"
    assert nset - {3, 5, 7}, "must add at least one Steiner point (bridge)"
    assert len(nset) < len(scores), "compact"
    print(f"  OK selected={sorted(nset)} ({len(nset)} nodes, connected, "
          f"steiner_pts={sorted(nset - {3, 5, 7})})")


def test_ste_uniform_weight_mode():
    """edge_weight_mode='uniform' 도 동작 (FK 무비용 우대 없이 hop 수 최소)."""
    print("\n[test_ste_uniform_weight_mode]")
    from modules.extractors.steiner_tree_extractor import SteinerTreeExtractor

    meta, scores = _make_synthetic_graph()
    ext = SteinerTreeExtractor(k=2, terminal_mode="topk",
                               terminal_node_types=["column"], cap_to_k=False,
                               edge_weight_mode="uniform")
    nodes, edges = ext.extract(meta, scores)
    nset = set(nodes)
    assert 3 in nset and 5 in nset
    # uniform 에서도 3-9-5 (2-hop) 가 3-0-1-5 (3-hop) 보다 짧음 → bridge 9 포함
    assert 9 in nset, f"uniform mode: 2-hop FK path bridge expected, got {sorted(nset)}"
    assert _induced_connected(nodes, edges)
    print(f"  OK uniform selected={sorted(nset)}")


# ===========================================================================
# FKPathfindingExtractor
# ===========================================================================
def test_fkp_bridge_union():
    """k=4 use_fk_paths=True → terminal (t0.a,t1.c) FK 최단경로가 bridge fk_node(9) union."""
    print("\n[test_fkp_bridge_union]")
    from modules.extractors.fk_pathfinding_extractor import FKPathfindingExtractor

    meta, scores = _make_synthetic_graph()
    ext = FKPathfindingExtractor(k=4, terminal_mode="topk",
                                 terminal_node_types=["column"], use_fk_paths=True)
    nodes, edges = ext.extract(meta, scores)
    nset = set(nodes)
    # terminals = top-4 columns {3,5,7,4}
    for t in (3, 5, 7, 4):
        assert t in nset, f"terminal {t} must be selected, got {sorted(nset)}"
    assert 9 in nset, f"FK bridge 9 (path t0.a→t1.c) must be unioned, got {sorted(nset)}"
    assert ext.last_info["fkp_paths_found"] == 1, \
        f"exactly 1 FK pair-path expected, got {ext.last_info['fkp_paths_found']}"
    assert _induced_connected_subset(nodes, edges, {3, 5, 9}), \
        "the FK-path component {3,9,5} must be connected"
    print(f"  OK selected={sorted(nset)} paths_found={ext.last_info['fkp_paths_found']} "
          f"fk_union={ext.last_info['fkp_fk_union_count']}")


def test_fkp_no_paths_isolation():
    """use_fk_paths=False (FKP-06) → FK 경로 미수행, bridge fk_node(9) 미포함 (FK 기여 분리)."""
    print("\n[test_fkp_no_paths_isolation]")
    from modules.extractors.fk_pathfinding_extractor import FKPathfindingExtractor

    meta, scores = _make_synthetic_graph()
    ext = FKPathfindingExtractor(k=4, terminal_mode="topk",
                                 terminal_node_types=["column"], use_fk_paths=False)
    nodes, _ = ext.extract(meta, scores)
    nset = set(nodes)
    assert nset == {3, 4, 5, 7}, f"FKP-06 = top-4 columns only, got {sorted(nset)}"
    assert 9 not in nset, "use_fk_paths=False must NOT include FK bridge (contribution isolated)"
    assert ext.last_info["fkp_paths_found"] == 0
    print(f"  OK FKP-06 selected={sorted(nset)} (no bridge — FK contribution isolated)")


def test_fkp_budget_fill():
    """terminal < k → 남은 budget 을 고점수 노드로 채움 (총 k cap)."""
    print("\n[test_fkp_budget_fill]")
    from modules.extractors.fk_pathfinding_extractor import FKPathfindingExtractor

    meta, scores = _make_synthetic_graph()
    # column 6개뿐 → k=8 terminal=6 cols, 남은 2 자리는 고점수 table(0,1) 로 채움
    ext = FKPathfindingExtractor(k=8, terminal_mode="topk",
                                 terminal_node_types=["column"], use_fk_paths=False)
    nodes, _ = ext.extract(meta, scores)
    nset = set(nodes)
    assert len(nset) == 8, f"budget fill must reach k=8, got {len(nset)}"
    assert ext.last_info["fkp_budget_filled"] == 2, \
        f"2 fill nodes expected, got {ext.last_info['fkp_budget_filled']}"
    assert {0, 1} <= nset, f"highest-score non-terminal tables (0,1) should fill, got {sorted(nset)}"
    print(f"  OK filled={ext.last_info['fkp_budget_filled']} selected={sorted(nset)}")


def test_fkp_threshold_mode():
    """terminal_mode='threshold' → score ≥ θ 인 column 만 terminal."""
    print("\n[test_fkp_threshold_mode]")
    from modules.extractors.fk_pathfinding_extractor import FKPathfindingExtractor

    meta, scores = _make_synthetic_graph()
    ext = FKPathfindingExtractor(k=20, terminal_mode="threshold", score_threshold=0.8,
                                 terminal_node_types=["column"], use_fk_paths=True)
    nodes, _ = ext.extract(meta, scores)
    nset = set(nodes)
    # columns with score ≥ 0.8: t0.a(3,0.9), t1.c(5,0.85), t2.e(7,0.8)
    assert {3, 5, 7} <= nset, f"threshold≥0.8 columns must be terminals, got {sorted(nset)}"
    assert ext.last_info["fkp_num_terminals"] == 3
    assert 9 in nset, "FK path (3→5) bridge expected under threshold mode too"
    print(f"  OK threshold terminals={ext.last_info['fkp_num_terminals']} selected={sorted(nset)}")


# ===========================================================================
# Registry
# ===========================================================================
def test_registry_registration_and_build():
    """두 클래스가 REGISTRY['extractor'] 에 등록 + registry.build() 로 instantiate 가능."""
    print("\n[test_registry_registration_and_build]")
    import modules.extractors  # noqa: F401 — triggers @register
    from modules.registry import REGISTRY, build

    assert "SteinerTreeExtractor" in REGISTRY["extractor"], "STE not registered"
    assert "FKPathfindingExtractor" in REGISTRY["extractor"], "FKP not registered"

    ste = build("extractor", {"name": "SteinerTreeExtractor",
                              "params": {"k": 15, "terminal_node_types": ["column", "table"]}})
    fkp = build("extractor", {"name": "FKPathfindingExtractor",
                              "params": {"k": 20, "use_fk_paths": False}})
    assert ste.k == 15 and ste.terminal_node_types == ["column", "table"]
    assert fkp.k == 20 and fkp.use_fk_paths is False
    # smoke extract via built objects
    meta, scores = _make_synthetic_graph()
    n1, _ = ste.extract(meta, scores)
    n2, _ = fkp.extract(meta, scores)
    assert n1 and n2, "built extractors must produce non-empty output"
    print(f"  OK registry build: STE→{len(n1)} nodes, FKP→{len(n2)} nodes")


def test_invalid_params_raise():
    """잘못된 mode 파라미터는 ValueError."""
    print("\n[test_invalid_params_raise]")
    from modules.extractors.steiner_tree_extractor import SteinerTreeExtractor
    from modules.extractors.fk_pathfinding_extractor import FKPathfindingExtractor

    for bad in ("bogus", "TOPK"):
        try:
            SteinerTreeExtractor(terminal_mode=bad)
            assert False, f"expected ValueError for terminal_mode={bad}"
        except ValueError:
            pass
    try:
        SteinerTreeExtractor(edge_weight_mode="bogus")
        assert False, "expected ValueError for edge_weight_mode"
    except ValueError:
        pass
    try:
        FKPathfindingExtractor(terminal_mode="bogus")
        assert False, "expected ValueError for FKP terminal_mode"
    except ValueError:
        pass
    print("  OK invalid params raise ValueError")


def _induced_connected_subset(sel_nodes, sel_edges, subset) -> bool:
    """selected 노드 중 subset 만 추출한 induced subgraph 가 연결인지."""
    s = set(sel_nodes) & set(subset)
    if not s:
        return False
    G = nx.Graph()
    G.add_nodes_from(s)
    for u, v in sel_edges:
        if u in s and v in s:
            G.add_edge(u, v)
    return nx.is_connected(G) if len(s) > 0 else True


if __name__ == "__main__":
    tests = [
        test_ste_steiner_point_discovery,
        test_ste_cap_to_k_prioritizes_terminals,
        test_ste_three_terminal_connectivity,
        test_ste_uniform_weight_mode,
        test_fkp_bridge_union,
        test_fkp_no_paths_isolation,
        test_fkp_budget_fill,
        test_fkp_threshold_mode,
        test_registry_registration_and_build,
        test_invalid_params_raise,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  ✗ FAIL {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ✗ ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{'='*60}")
    if failed == 0:
        print(f"✅ ALL {len(tests)} V7 extractor tests passed.")
    else:
        print(f"❌ {failed}/{len(tests)} tests FAILED.")
        sys.exit(1)
