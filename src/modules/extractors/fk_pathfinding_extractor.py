"""V7-W2 — FK Pathfinding Extractor (B1, SchemaGraphSQL 기반).

학술 Agent RFP `scholar_agent_extractor_rfp_2026-06-04.md` §2.8 코드 spec 정합 +
`extractor_redesign_v7_plan_2026-06-04.md` §2 정정 (import path / register / GPU /
graph_data dict interface) 반영.

핵심 mechanism (Safdarian et al., 2025, arXiv:2505.18363):
  고점수 terminal 노드 쌍 사이의 FK 최단 경로를 union 하여 join 경로를 명시적으로 보장.
        C = ∪_{Ts,Td} SP(Ts, Td),  U = ∪_{p∈C} p
  원 논문은 LLM 호출로 source/destination 식별 → 본 framework 는 node_scores 로 대체.
  MSTKruskal 과 달리 FK 경로를 따르는 backbone 구조를 형성.

본 framework 정합 (RFP 객체 method 가정 → 실제 dict/index 접근으로 변환):
  - graph_data 는 dict (metadata). node type 은 flat-index range 로 판정:
        table  : [0, num_t)
        column : [num_t, num_t + num_c)
        fk_node: [num_t + num_c, num_nodes)
  - FK subgraph 는 edge_types ∈ {is_source_of, points_to, table_to_table} 인 edge 만 사용.
    column 간 FK join 은 `column → fk_node → column` 2-hop 으로 표현됨.
  - 남은 budget 은 고점수 노드로 채워 총 k 노드 cap (단, FK 경로 union 이 k 초과 시 trim 안 함 —
    RFP §2.10 'union 결과가 MSTKruskal 보다 커질 수 있음' 정합).
"""
import time
import networkx as nx
from itertools import combinations
from typing import List, Dict, Tuple, Any, Optional

from modules.registry import register
from modules.base import BaseExtractor
from utils.logger import get_logger

logger = get_logger(__name__)


@register("extractor", "FKPathfindingExtractor")
class FKPathfindingExtractor(BaseExtractor):
    """FK-path union extractor.

    Parameters
    ----------
    k : int
        Top-K terminal 개수 (terminal_mode='topk') + budget fill 상한.
    terminal_mode : {"topk", "threshold"}
        terminal 선택 방식.
    score_threshold : float
        terminal_mode='threshold' 시 사용.
    terminal_node_types : List[str] | None
        terminal 후보 노드 타입. 기본 ["column"] (SchemaGraphSQL column-level pathfinding).
    fk_edge_types : List[str] | None
        FK subgraph 구성 edge type. 기본 ["is_source_of", "points_to", "table_to_table"].
    use_fk_paths : bool
        False = FKP-06 mode (FK 경로 union 미수행, 순수 상위 K 노드). FK 경로 순기여 분리용.
    max_terminal_pairs : int
        O(K^2) 경로 조합 폭발 guard (RFP §2.10). 초과 시 truncate + telemetry 기록.
        기본 2000 (topk K≤20 → C(20,2)=190 으로 절대 미발동, 병리적 threshold case 만 차단).
    """

    def __init__(self,
                 k: int = 20,
                 terminal_mode: str = "topk",
                 score_threshold: float = 0.5,
                 terminal_node_types: Optional[List[str]] = None,
                 fk_edge_types: Optional[List[str]] = None,
                 use_fk_paths: bool = True,
                 max_terminal_pairs: int = 2000,
                 **kwargs):
        if terminal_mode not in ("topk", "threshold"):
            raise ValueError(
                f"terminal_mode must be 'topk' or 'threshold', got '{terminal_mode}'"
            )
        self.k = int(k)
        self.terminal_mode = terminal_mode
        self.score_threshold = float(score_threshold)
        self.terminal_node_types = list(terminal_node_types) if terminal_node_types else ["column"]
        self.fk_edge_types = set(fk_edge_types) if fk_edge_types else {
            "is_source_of", "points_to", "table_to_table"
        }
        self.use_fk_paths = bool(use_fk_paths)
        self.max_terminal_pairs = int(max_terminal_pairs)
        self.last_info: Dict[str, Any] = {}
        logger.info(
            f"Initialized FKPathfinding Extractor (k={self.k}, terminal_mode={self.terminal_mode}, "
            f"score_threshold={self.score_threshold}, terminal_node_types={self.terminal_node_types}, "
            f"use_fk_paths={self.use_fk_paths}, fk_edge_types={sorted(self.fk_edge_types)})"
        )

    @staticmethod
    def _node_type(nid: int, num_t: int, num_c: int) -> str:
        if nid < num_t:
            return "table"
        if nid < num_t + num_c:
            return "column"
        return "fk_node"

    def _build_fk_graph(self, graph_data: Dict[str, Any],
                        num_t: int, num_c: int, num_nodes: int) -> nx.Graph:
        """FK-only subgraph (edge_type ∈ fk_edge_types). 모든 노드 포함 (isolated 보존)."""
        G = nx.Graph()
        for nid in range(num_nodes):
            G.add_node(nid, node_type=self._node_type(nid, num_t, num_c))
        edges = graph_data.get('edges', []) or []
        edge_types = graph_data.get('edge_types', []) or []
        for idx, (u, v) in enumerate(edges):
            et = edge_types[idx] if idx < len(edge_types) else 'default'
            if et in self.fk_edge_types:
                G.add_edge(u, v, edge_type=et)
        return G

    def _build_full_graph(self, graph_data: Dict[str, Any], num_nodes: int) -> nx.Graph:
        """전체 edge 그래프 — 최종 selected 노드의 induced edge 산출용."""
        G = nx.Graph()
        for nid in range(num_nodes):
            G.add_node(nid)
        for (u, v) in (graph_data.get('edges', []) or []):
            G.add_edge(u, v)
        return G

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None,
                **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        t_start = time.perf_counter()

        num_t = len(graph_data.get('table_to_id', {}) or {})
        num_c = len(graph_data.get('col_to_id', {}) or {})
        num_nodes = len(node_scores)
        edges = graph_data.get('edges', []) or []

        if num_nodes == 0:
            self.last_info = self._empty_info(num_nodes, len(edges), t_start, num_terminals=0)
            return [], []

        G_fk = self._build_fk_graph(graph_data, num_t, num_c, num_nodes)

        candidates = [
            n for n in range(num_nodes)
            if self._node_type(n, num_t, num_c) in self.terminal_node_types
        ]
        if self.terminal_mode == "topk":
            terminals = sorted(candidates, key=lambda i: node_scores[i], reverse=True)[: self.k]
        else:
            terminals = [i for i in candidates if node_scores[i] >= self.score_threshold]

        if not terminals:
            logger.debug("[FKPath] no terminals selected; returning empty.")
            self.last_info = self._empty_info(num_nodes, len(edges), t_start, num_terminals=0)
            return [], []

        selected = set(int(t) for t in terminals)
        pairs_evaluated = 0
        paths_found = 0
        pairs_truncated = False
        # FK backbone: 모든 terminal pair 의 FK 최단 경로 union
        if self.use_fk_paths and len(terminals) >= 2:
            for u, v in combinations(terminals, 2):
                if pairs_evaluated >= self.max_terminal_pairs:
                    pairs_truncated = True
                    logger.debug(
                        f"[FKPath] max_terminal_pairs={self.max_terminal_pairs} reached "
                        f"(terminals={len(terminals)}); truncating pair enumeration."
                    )
                    break
                pairs_evaluated += 1
                # 연결 불가 terminal 쌍은 안전 skip (RFP §2.10) — 해당 terminal 은 단독 보존
                if u in G_fk and v in G_fk and nx.has_path(G_fk, u, v):
                    try:
                        path = nx.shortest_path(G_fk, u, v)
                        selected.update(int(n) for n in path)
                        paths_found += 1
                    except nx.NetworkXNoPath:
                        pass

        fk_union_count = len(selected)
        # 남은 budget 을 고점수 노드로 채움 (총 k cap). union 이 이미 k 초과 시 trim 안 함.
        filled = 0
        if len(selected) < self.k:
            for nid in sorted(range(num_nodes), key=lambda i: node_scores[i], reverse=True):
                if len(selected) >= self.k:
                    break
                if nid not in selected:
                    selected.add(int(nid))
                    filled += 1

        sel_set = {int(n) for n in selected}
        sel_nodes = sorted(sel_set)
        G_full = self._build_full_graph(graph_data, num_nodes)
        sel_edges = sorted({
            (min(int(u), int(v)), max(int(u), int(v)))
            for u, v in G_full.edges()
            if u in sel_set and v in sel_set
        })

        sel_t = sum(1 for n in sel_nodes if n < num_t)
        sel_c = sum(1 for n in sel_nodes if num_t <= n < num_t + num_c)
        sel_fk = sum(1 for n in sel_nodes if n >= num_t + num_c)

        logger.debug(
            f"[FKPath] k={self.k} mode={self.terminal_mode} use_fk_paths={self.use_fk_paths}: "
            f"{len(terminals)} terminals, {paths_found}/{pairs_evaluated} pair-paths → "
            f"fk_union={fk_union_count}, +{filled} filled → {len(sel_nodes)} nodes "
            f"({sel_t}T/{sel_c}C/{sel_fk}FK), {len(sel_edges)} edges"
        )
        self.last_info = {
            "extractor_type": "FKPathfindingExtractor",
            "extractor_num_input_nodes": int(num_nodes),
            "extractor_num_edges": int(len(edges)),
            "extractor_num_selected_nodes": int(len(sel_nodes)),
            "extractor_num_selected_edges": int(len(sel_edges)),
            "fkp_k": int(self.k),
            "fkp_terminal_mode": self.terminal_mode,
            "fkp_score_threshold": float(self.score_threshold),
            "fkp_terminal_node_types": list(self.terminal_node_types),
            "fkp_use_fk_paths": bool(self.use_fk_paths),
            "fkp_num_terminals": int(len(terminals)),
            "fkp_pairs_evaluated": int(pairs_evaluated),
            "fkp_paths_found": int(paths_found),
            "fkp_pairs_truncated": bool(pairs_truncated),
            "fkp_fk_union_count": int(fk_union_count),
            "fkp_budget_filled": int(filled),
            "extractor_selected_by_type": {
                "table": int(sel_t), "column": int(sel_c), "fk_node": int(sel_fk),
            },
            "extractor_time_s": float(time.perf_counter() - t_start),
        }
        return sel_nodes, sel_edges

    def _empty_info(self, num_nodes: int, num_edges: int, t_start: float,
                    num_terminals: int = 0) -> Dict[str, Any]:
        return {
            "extractor_type": "FKPathfindingExtractor",
            "extractor_num_input_nodes": int(num_nodes),
            "extractor_num_edges": int(num_edges),
            "extractor_num_selected_nodes": 0,
            "extractor_num_selected_edges": 0,
            "fkp_k": int(self.k),
            "fkp_terminal_mode": self.terminal_mode,
            "fkp_use_fk_paths": bool(self.use_fk_paths),
            "fkp_num_terminals": int(num_terminals),
            "no_terminals": True,
            "extractor_time_s": float(time.perf_counter() - t_start),
        }
