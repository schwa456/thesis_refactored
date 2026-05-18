"""FK/PK Connectivity Steiner Closure algorithm (학술 agent §2.2/2.3, Wave 8 D2).

DECISIONS 2026-05-18 (Wave 8 M4 Bidirectional 발전 4 Direction) §2 D2 정합.

LLM 호출 0× — DB DDL 기반 FK/PK metadata 와 Extractor 후보 (subgraph) 만 사용.

학술 frame: paper §V.5.x.M.16 candidate (DB-aware Schema Connectivity).
M4 Union 결과의 JOIN path 연결성 부족 (선택된 두 table 사이 FK 연결 column 누락) 을
DB DDL Steiner closure 로 보강 — Extractor 후보 안에서만 검색 (Full Schema 조회 금지).

핵심 제약 (Wave 8 정합):
  - LLM 입력 0× (algorithm only)
  - Full Schema 조회 금지 — extractor_output (Extractor 후보) 안에서만 검색
  - direct FK: M4 Union 의 두 table 사이 직접 FK 가 존재 + 양쪽 FK column 이
    extractor_output 에 있을 때만 추가
  - 1-hop bridge: 두 table 사이 직접 FK 가 없으면 중간 bridge table 1 개를 통해
    간접 연결 — bridge table 의 FK column 들이 extractor_output 에 있을 때만 추가

설계 (학술 agent §2.1 build_local_fk_graph):
  - node id = "{table}.{col}"
  - edge: FK constraint 의 양쪽 노드 모두 extractor_output 에 있을 때만 추가
  - undirected (양방향 JOIN 가능)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


def _normalize_subgraph(s: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    """{table: [col, ...]} or {table: {col: True, ...}} → {table: [col, ...]}."""
    out: Dict[str, List[str]] = {}
    if not isinstance(s, dict):
        return out
    for t, cols in s.items():
        if not isinstance(t, str):
            continue
        if isinstance(cols, dict):
            out[t] = [c for c in cols if isinstance(c, str)]
        elif isinstance(cols, list):
            out[t] = [c for c in cols if isinstance(c, str)]
    return out


def build_local_fk_graph(
    extractor_output: Dict[str, List[str]],
    db_fk_metadata: Dict[str, Any],
) -> nx.Graph:
    """Extractor 후보 columns 사이의 FK 연결 그래프 (학술 agent §2.1).

    Args:
      extractor_output: {table: [col, ...]} — Extractor 후보 schema
      db_fk_metadata: db_fk_extractor.load_db_fk_metadata()[db_id]
        형식: {"tables": {...}, "fk_constraints": [
          {"from_table": tbl, "from_col": col, "to_table": tbl2, "to_col": col2}, ...
        ]}

    Returns:
      undirected nx.Graph with node = "{table}.{col}", edge = FK constraint
      (양쪽 노드 모두 extractor_output 에 있을 때만).
    """
    G = nx.Graph()
    ext = _normalize_subgraph(extractor_output)

    # Node: Extractor 후보 column 만
    for table, cols in ext.items():
        for col in cols:
            node_id = f"{table}.{col}"
            G.add_node(node_id, table=table, col=col)

    # Edge: FK constraints (양쪽 모두 extractor 후보에 있어야 함)
    for fk in (db_fk_metadata or {}).get("fk_constraints", []) or []:
        if not isinstance(fk, dict):
            continue
        from_table = fk.get("from_table")
        from_col = fk.get("from_col")
        to_table = fk.get("to_table")
        to_col = fk.get("to_col")
        if not (from_table and from_col and to_table and to_col):
            continue
        from_node = f"{from_table}.{from_col}"
        to_node = f"{to_table}.{to_col}"
        if G.has_node(from_node) and G.has_node(to_node):
            G.add_edge(from_node, to_node, type="foreign_key")

    return G


def _ensure_added(
    added: Dict[str, List[str]],
    table: str,
    col: str,
    extractor_output: Dict[str, List[str]],
    closed: Dict[str, List[str]],
) -> bool:
    """Helper — add (table, col) to added/closed if it's a valid extractor candidate
    and not already present. Returns True if newly added."""
    if table not in extractor_output:
        return False
    if col not in (extractor_output.get(table) or []):
        return False
    # 이미 closed 에 있으면 추가 안 함
    if col in (closed.get(table) or []):
        return False
    closed.setdefault(table, []).append(col)
    added.setdefault(table, []).append(col)
    return True


def steiner_closure(
    m4_union: Dict[str, List[str]],
    extractor_output: Dict[str, List[str]],
    db_fk_metadata_per_db: Dict[str, Any],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """직접 FK closure (학술 agent §2.2).

    M4 Union 의 테이블 쌍 사이 직접 FK 가 존재하면, 해당 FK 의 양쪽 컬럼이
    extractor_output 에 있을 때만 m4_union 에 추가. Full Schema 조회 없음.

    Args:
      m4_union: {table: [col, ...]} — M4 Forward ∪ Backward 결과
      extractor_output: {table: [col, ...]} — Extractor 후보
      db_fk_metadata_per_db: db_fk_extractor.load_db_fk_metadata()[db_id]
        ({"tables": {...}, "fk_constraints": [...]} per DB)

    Returns:
      (closed_union, added_only)
        - closed_union: m4_union ∪ {추가 FK 컬럼}
        - added_only: {table: [col, ...]} — 새로 추가된 컬럼만 (측정용)
    """
    m4_norm = _normalize_subgraph(m4_union)
    ext_norm = _normalize_subgraph(extractor_output)
    closed: Dict[str, List[str]] = {t: list(cols) for t, cols in m4_norm.items()}
    added: Dict[str, List[str]] = {}

    # m4_union 이 비어 있거나 테이블 1개 이하면 closure 불필요
    selected_tables = list(m4_norm.keys())
    if len(selected_tables) <= 1:
        return closed, added

    fk_constraints = (db_fk_metadata_per_db or {}).get("fk_constraints", []) or []

    # 각 (t1, t2) 쌍 (i < j) 에 대해 직접 FK 검색 — 양 방향 모두 허용
    for i in range(len(selected_tables)):
        for j in range(i + 1, len(selected_tables)):
            t1 = selected_tables[i]
            t2 = selected_tables[j]
            for fk in fk_constraints:
                if not isinstance(fk, dict):
                    continue
                from_tbl = fk.get("from_table")
                from_col = fk.get("from_col")
                to_tbl = fk.get("to_table")
                to_col = fk.get("to_col")
                if not (from_tbl and from_col and to_tbl and to_col):
                    continue
                # (t1, t2) 또는 (t2, t1) 매칭
                if (from_tbl == t1 and to_tbl == t2) or (
                    from_tbl == t2 and to_tbl == t1
                ):
                    # 양쪽 column 모두 extractor 후보에 있을 때만 추가
                    if from_col in (ext_norm.get(from_tbl) or []) and to_col in (
                        ext_norm.get(to_tbl) or []
                    ):
                        _ensure_added(added, from_tbl, from_col, ext_norm, closed)
                        _ensure_added(added, to_tbl, to_col, ext_norm, closed)

    # Sort cols for determinism
    for t in closed:
        closed[t] = sorted(set(closed[t]))
    for t in added:
        added[t] = sorted(set(added[t]))
    return closed, added


def steiner_closure_with_bridge(
    m4_union: Dict[str, List[str]],
    extractor_output: Dict[str, List[str]],
    db_fk_metadata_per_db: Dict[str, Any],
    max_bridge_depth: int = 1,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """1-hop bridge 포함 (학술 agent §2.3).

    선택된 테이블 간 직접 FK 가 없으면 중간 테이블 1 개를 통한 간접 연결 시도.
    bridge 테이블과 그 FK 컬럼들이 extractor_output 에 있을 때만 추가.

    Args:
      m4_union: M4 Forward ∪ Backward
      extractor_output: Extractor 후보
      db_fk_metadata_per_db: per-DB FK metadata
      max_bridge_depth: 허용 중간 테이블 수 (1 = 1-hop bridge)

    Returns:
      (closed_union, added_only) — direct + bridge 모두 포함한 추가 컬럼
    """
    m4_norm = _normalize_subgraph(m4_union)
    ext_norm = _normalize_subgraph(extractor_output)

    # 1) 먼저 direct FK closure
    closed, added = steiner_closure(m4_norm, ext_norm, db_fk_metadata_per_db)
    if len(m4_norm) <= 1:
        return closed, added

    # 2) FK graph 구축 (Extractor 후보 안에서만)
    G = build_local_fk_graph(ext_norm, db_fk_metadata_per_db)
    if G.number_of_nodes() == 0:
        return closed, added

    selected_tables = list(m4_norm.keys())
    # 노드 path 의 최대 노드 수 = 2 + max_bridge_depth
    # (예: t1.colA → bridge.colB → bridge.colC → t2.colD : 4 nodes = 1-hop bridge)
    max_nodes_in_path = 2 + max(0, int(max_bridge_depth)) * 2

    # 3) 각 선택 테이블 쌍에 대해 직접 FK 가 없을 시 bridge 검색
    for i in range(len(selected_tables)):
        for j in range(i + 1, len(selected_tables)):
            t1 = selected_tables[i]
            t2 = selected_tables[j]

            # 이미 direct closure 로 양쪽 column 추가됐는지 확인
            # (added 에 fk_constraint 에 의해 양쪽 column 모두 들어왔으면 skip)
            if _has_direct_fk(t1, t2, db_fk_metadata_per_db, ext_norm):
                continue

            # bridge 탐색 — t1 의 노드 → t2 의 노드 최단 path
            t1_nodes = [
                n for n, data in G.nodes(data=True) if data.get("table") == t1
            ]
            t2_nodes = [
                n for n, data in G.nodes(data=True) if data.get("table") == t2
            ]
            if not t1_nodes or not t2_nodes:
                continue

            best_path: Optional[List[str]] = None
            for n1 in t1_nodes:
                for n2 in t2_nodes:
                    try:
                        if nx.has_path(G, n1, n2):
                            path = nx.shortest_path(G, n1, n2)
                            if len(path) <= max_nodes_in_path:
                                if best_path is None or len(path) < len(best_path):
                                    best_path = path
                    except nx.NetworkXNoPath:
                        continue
            if not best_path:
                continue

            # path 의 모든 intermediate (bridge) 노드 columns 추가
            # path = [t1.colA, bridge_node, ..., t2.colZ]
            # 양 끝점이 t1, t2 이고 intermediate 가 bridge column
            for node_id in best_path:
                node_data = G.nodes[node_id]
                ntable = node_data.get("table")
                ncol = node_data.get("col")
                if not ntable or not ncol:
                    continue
                _ensure_added(added, ntable, ncol, ext_norm, closed)

    # Sort cols for determinism
    for t in closed:
        closed[t] = sorted(set(closed[t]))
    for t in added:
        added[t] = sorted(set(added[t]))
    return closed, added


def _has_direct_fk(
    t1: str,
    t2: str,
    db_fk_metadata: Dict[str, Any],
    extractor_output: Dict[str, List[str]],
) -> bool:
    """t1↔t2 에 직접 FK 가 있고 양쪽 column 이 extractor 후보에 모두 있는지 확인."""
    ext_norm = _normalize_subgraph(extractor_output)
    for fk in (db_fk_metadata or {}).get("fk_constraints", []) or []:
        if not isinstance(fk, dict):
            continue
        ft = fk.get("from_table")
        fc = fk.get("from_col")
        tt = fk.get("to_table")
        tc = fk.get("to_col")
        if not (ft and fc and tt and tc):
            continue
        if (ft == t1 and tt == t2) or (ft == t2 and tt == t1):
            if fc in (ext_norm.get(ft) or []) and tc in (ext_norm.get(tt) or []):
                return True
    return False


def count_added_gold(
    added: Dict[str, List[str]],
    gold: Optional[Dict[str, List[str]]],
) -> Optional[int]:
    """added 중 gold 인 column 수 (gold 가 None 이면 None 반환)."""
    if gold is None:
        return None
    gold_norm = _normalize_subgraph(gold)
    gold_set = {(t, c) for t, cols in gold_norm.items() for c in cols}
    added_set = {(t, c) for t, cols in added.items() for c in cols}
    return len(added_set & gold_set)
