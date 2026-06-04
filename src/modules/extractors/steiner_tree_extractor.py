"""V7-W3 — Steiner Tree Extractor (GRAST-SQL 기반).

학술 Agent RFP `scholar_agent_extractor_rfp_2026-06-04.md` §1.8 코드 spec 정합 +
`extractor_redesign_v7_plan_2026-06-04.md` §2 정정 (import path / register / GPU /
graph_data dict interface) 반영.

핵심 mechanism (Hoang et al., 2025, arXiv:2512.16083):
  MSTKruskal : score > θ 인 모든 노드를 terminal 로 → induced spanning tree → over-extract
  SteinerTree: top-K 노드만 terminal → 그 사이를 잇는 최소 중간 노드(Steiner point)만 추가
               → compact extract

본 framework 정합 (RFP 객체 method 가정 → 실제 dict/index 접근으로 변환):
  - graph_data 는 dict (metadata). node type 은 flat-index range 로 판정:
        table  : [0, num_t)
        column : [num_t, num_t + num_c)
        fk_node: [num_t + num_c, num_nodes)
    (num_t=len(table_to_id), num_c=len(col_to_id), num_nodes=len(node_scores))
  - query 노드는 node_scores index 공간에 존재하지 않음 (RFP §1.10 query 제외 권고 자동 충족).
  - edges/edge_types 는 graph_data['edges'] (List[(u,v)] flat idx) + graph_data['edge_types']
    (List[str], 값 ∈ {belongs_to, is_source_of, points_to, table_to_table}) parallel list.
  - GRAST-SQL edge weight: FK-adjacent (is_source_of/points_to) = 0, otherwise = 1.
"""
import time
import networkx as nx
from typing import List, Dict, Tuple, Any, Optional

from modules.registry import register
from modules.base import BaseExtractor
from utils.logger import get_logger

try:
    from networkx.algorithms.approximation import steiner_tree
except ImportError:  # networkx < 3.0
    steiner_tree = None

logger = get_logger(__name__)


@register("extractor", "SteinerTreeExtractor")
class SteinerTreeExtractor(BaseExtractor):
    """Connectivity-preserving extractor via Steiner tree approximation.

    Parameters
    ----------
    k : int
        Top-K terminal 개수 (terminal_mode='topk') / cap_to_k 시 출력 상한.
    terminal_mode : {"topk", "threshold"}
        terminal 선택 방식. topk = node_scores 상위 K, threshold = score ≥ score_threshold.
    score_threshold : float
        terminal_mode='threshold' 시 사용.
    terminal_node_types : List[str] | None
        terminal 후보 노드 타입. 기본 ["column", "table"]. RFP §1.10 권고대로 fk_node/query 제외.
    cap_to_k : bool
        True 시 총 출력 노드를 k 로 제한 (terminal 우선 + Steiner point 점수순). STE-08 은 False.
    edge_weight_mode : {"grast", "uniform"}
        grast = FK-adjacent edge weight 0 / 그 외 1 (GRAST-SQL). uniform = 모든 edge 1.
    fk_edge_types : List[str] | None
        grast 모드에서 weight 0 을 부여할 FK-adjacent edge type. 기본 ["is_source_of", "points_to"].
    """

    def __init__(self,
                 k: int = 20,
                 terminal_mode: str = "topk",
                 score_threshold: float = 0.5,
                 terminal_node_types: Optional[List[str]] = None,
                 cap_to_k: bool = True,
                 edge_weight_mode: str = "grast",
                 fk_edge_types: Optional[List[str]] = None,
                 **kwargs):
        if terminal_mode not in ("topk", "threshold"):
            raise ValueError(
                f"terminal_mode must be 'topk' or 'threshold', got '{terminal_mode}'"
            )
        if edge_weight_mode not in ("grast", "uniform"):
            raise ValueError(
                f"edge_weight_mode must be 'grast' or 'uniform', got '{edge_weight_mode}'"
            )
        self.k = int(k)
        self.terminal_mode = terminal_mode
        self.score_threshold = float(score_threshold)
        self.terminal_node_types = list(terminal_node_types) if terminal_node_types else ["column", "table"]
        self.cap_to_k = bool(cap_to_k)
        self.edge_weight_mode = edge_weight_mode
        # GRAST-SQL: FK-adjacent edge 는 무비용 traversal (RFP §1.8 정합 default set)
        self.fk_edge_types = set(fk_edge_types) if fk_edge_types else {"is_source_of", "points_to"}
        self.last_info: Dict[str, Any] = {}
        logger.info(
            f"Initialized SteinerTree Extractor (k={self.k}, terminal_mode={self.terminal_mode}, "
            f"score_threshold={self.score_threshold}, terminal_node_types={self.terminal_node_types}, "
            f"cap_to_k={self.cap_to_k}, edge_weight_mode={self.edge_weight_mode})"
        )

    @staticmethod
    def _node_type(nid: int, num_t: int, num_c: int) -> str:
        if nid < num_t:
            return "table"
        if nid < num_t + num_c:
            return "column"
        return "fk_node"

    def _build_nx_graph(self, graph_data: Dict[str, Any],
                        num_t: int, num_c: int, num_nodes: int) -> nx.Graph:
        """heterograph metadata → undirected NetworkX graph (GRAST-SQL edge weight)."""
        G = nx.Graph()
        for nid in range(num_nodes):
            G.add_node(nid, node_type=self._node_type(nid, num_t, num_c))
        edges = graph_data.get('edges', []) or []
        edge_types = graph_data.get('edge_types', []) or []
        for idx, (u, v) in enumerate(edges):
            et = edge_types[idx] if idx < len(edge_types) else 'default'
            if self.edge_weight_mode == "grast":
                w = 0.0 if et in self.fk_edge_types else 1.0
            else:
                w = 1.0
            # 병렬 edge (예: 양방향 table_to_table) → 최소 weight 유지 (FK 무비용 경로 우선)
            if G.has_edge(u, v):
                if w < G[u][v].get('weight', 1.0):
                    G[u][v]['weight'] = w
                    G[u][v]['edge_type'] = et
            else:
                G.add_edge(u, v, weight=w, edge_type=et)
        return G

    def _get_terminals(self, candidate_ids: List[int],
                       node_scores: List[float]) -> List[int]:
        if self.terminal_mode == "topk":
            sorted_ids = sorted(candidate_ids, key=lambda i: node_scores[i], reverse=True)
            return sorted_ids[: self.k]
        return [i for i in candidate_ids if node_scores[i] >= self.score_threshold]

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None,
                **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        t_start = time.perf_counter()

        if steiner_tree is None:
            raise ImportError(
                "networkx>=3.0 with networkx.algorithms.approximation.steiner_tree is required "
                "for SteinerTreeExtractor."
            )

        num_t = len(graph_data.get('table_to_id', {}) or {})
        num_c = len(graph_data.get('col_to_id', {}) or {})
        num_nodes = len(node_scores)
        edges = graph_data.get('edges', []) or []

        if num_nodes == 0:
            self.last_info = self._empty_info(num_t, num_c, num_nodes, len(edges), t_start)
            return [], []

        G = self._build_nx_graph(graph_data, num_t, num_c, num_nodes)

        candidate_ids = [
            n for n in range(num_nodes)
            if self._node_type(n, num_t, num_c) in self.terminal_node_types
        ]
        terminals = self._get_terminals(candidate_ids, node_scores)
        if not terminals:
            logger.debug("[SteinerTree] no terminals selected; returning empty.")
            self.last_info = self._empty_info(num_t, num_c, num_nodes, len(edges), t_start,
                                              num_terminals=0)
            return [], []

        selected = set(int(t) for t in terminals)
        steiner_components = 0
        steiner_failures = 0
        # connected component 별 Steiner approximation (terminal 1개 → 단독 보존)
        for comp in nx.connected_components(G):
            t_in_comp = [t for t in terminals if t in comp]
            if len(t_in_comp) < 2:
                continue
            subG = G.subgraph(comp).copy()
            try:
                st = steiner_tree(subG, t_in_comp, weight="weight")
                selected.update(int(n) for n in st.nodes())
                steiner_components += 1
            except Exception as e:  # noqa: BLE001 — robustness over disconnected/edge cases
                logger.debug(f"[SteinerTree] component steiner_tree failed ({e}); "
                             f"keeping {len(t_in_comp)} terminals only.")
                selected.update(int(t) for t in t_in_comp)
                steiner_failures += 1

        pre_cap_count = len(selected)
        capped = False
        # cap_to_k: terminal 우선, 남은 자리는 Steiner point 를 점수 내림차순으로 채움
        if self.cap_to_k and len(selected) > self.k:
            term_set = set(int(t) for t in terminals)
            steiner_pts = sorted(
                [n for n in selected if n not in term_set],
                key=lambda n: node_scores[n] if 0 <= n < num_nodes else 0.0,
                reverse=True,
            )
            keep = max(0, self.k - len(term_set))
            selected = term_set | set(steiner_pts[: keep])
            capped = True

        sel_set = {int(n) for n in selected}
        sel_nodes = sorted(sel_set)
        sel_edges = sorted({
            (min(int(u), int(v)), max(int(u), int(v)))
            for u, v in G.edges()
            if u in sel_set and v in sel_set
        })

        sel_t = sum(1 for n in sel_nodes if n < num_t)
        sel_c = sum(1 for n in sel_nodes if num_t <= n < num_t + num_c)
        sel_fk = sum(1 for n in sel_nodes if n >= num_t + num_c)

        logger.debug(
            f"[SteinerTree] k={self.k} mode={self.terminal_mode}: "
            f"{len(terminals)} terminals → {pre_cap_count} pre-cap → "
            f"{len(sel_nodes)} nodes ({sel_t}T/{sel_c}C/{sel_fk}FK), "
            f"{len(sel_edges)} edges, steiner_comp={steiner_components}, capped={capped}"
        )
        self.last_info = {
            "extractor_type": "SteinerTreeExtractor",
            "extractor_num_input_nodes": int(num_nodes),
            "extractor_num_edges": int(len(edges)),
            "extractor_num_selected_nodes": int(len(sel_nodes)),
            "extractor_num_selected_edges": int(len(sel_edges)),
            "ste_k": int(self.k),
            "ste_terminal_mode": self.terminal_mode,
            "ste_score_threshold": float(self.score_threshold),
            "ste_terminal_node_types": list(self.terminal_node_types),
            "ste_cap_to_k": bool(self.cap_to_k),
            "ste_edge_weight_mode": self.edge_weight_mode,
            "ste_num_terminals": int(len(terminals)),
            "ste_pre_cap_node_count": int(pre_cap_count),
            "ste_capped": bool(capped),
            "ste_steiner_components": int(steiner_components),
            "ste_steiner_failures": int(steiner_failures),
            "extractor_selected_by_type": {
                "table": int(sel_t), "column": int(sel_c), "fk_node": int(sel_fk),
            },
            "extractor_time_s": float(time.perf_counter() - t_start),
        }
        return sel_nodes, sel_edges

    def _empty_info(self, num_t: int, num_c: int, num_nodes: int, num_edges: int,
                    t_start: float, num_terminals: Optional[int] = None) -> Dict[str, Any]:
        return {
            "extractor_type": "SteinerTreeExtractor",
            "extractor_num_input_nodes": int(num_nodes),
            "extractor_num_edges": int(num_edges),
            "extractor_num_selected_nodes": 0,
            "extractor_num_selected_edges": 0,
            "ste_k": int(self.k),
            "ste_terminal_mode": self.terminal_mode,
            "ste_num_terminals": int(num_terminals) if num_terminals is not None else 0,
            "no_terminals": True,
            "extractor_time_s": float(time.perf_counter() - t_start),
        }
