import time
import networkx as nx
from typing import List, Dict, Tuple, Any, Optional

from modules.registry import register
from modules.base import BaseExtractor
from utils.logger import get_logger

logger = get_logger(__name__)


@register("extractor", "MSTKruskalExtractor")
class MSTKruskalExtractor(BaseExtractor):
    """
    "진짜" MST extractor.

    1. `node_scores > score_threshold` 인 노드 → selected_node_ids
    2. graph_data['edges'] 중 양 끝점 모두 selected_node_ids 인 edges → induced subgraph
    3. networkx.minimum_spanning_tree (Kruskal default) 호출 — disconnected 시 minimum
       spanning forest 가 자동 반환됨
    4. selected_nodes = MST(forest) 의 모든 vertex, selected_edges = MST(forest) edges

    Steiner Tree 와의 차이:
      - Steiner: terminal subset 만 connecting + Steiner point 추가 가능
      - MST (Kruskal): induced subgraph 의 모든 vertex spanning, Steiner point 없음
    """
    def __init__(self, score_threshold: float = 0.1, **kwargs):
        self.score_threshold = float(score_threshold)
        self.last_info: Dict[str, Any] = {}
        logger.info(f"Initialized MSTKruskal Extractor (score_threshold={self.score_threshold})")

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None,
                **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        t_start = time.perf_counter()
        edges = graph_data.get('edges', []) or []

        selected_node_ids = {i for i, s in enumerate(node_scores) if s > self.score_threshold}
        induced_node_count = len(selected_node_ids)

        if induced_node_count == 0:
            logger.debug(
                f"[MSTKruskal] no nodes pass threshold {self.score_threshold}; returning empty."
            )
            self.last_info = {
                "extractor_type": "MSTKruskalExtractor",
                "extractor_num_input_nodes": int(len(node_scores)),
                "extractor_num_edges": int(len(edges)),
                "extractor_num_selected_nodes": 0,
                "extractor_num_selected_edges": 0,
                "mst_score_threshold": float(self.score_threshold),
                "mst_induced_node_count": 0,
                "mst_induced_edge_count": 0,
                "mst_node_count": 0,
                "mst_edge_count": 0,
                "mst_components": 0,
                "no_seeds": True,
                "extractor_time_s": float(time.perf_counter() - t_start),
            }
            return [], []

        induced_G = nx.Graph()
        induced_G.add_nodes_from(selected_node_ids)
        induced_edge_count = 0
        for u, v in edges:
            if u in selected_node_ids and v in selected_node_ids:
                induced_G.add_edge(u, v)
                induced_edge_count += 1

        mst = nx.minimum_spanning_tree(induced_G)

        selected_nodes = list(mst.nodes())
        selected_edges = [
            (min(u, v), max(u, v)) for u, v in mst.edges()
        ]

        try:
            num_components = nx.number_connected_components(induced_G)
        except Exception:
            num_components = -1

        logger.debug(
            f"[MSTKruskal] threshold={self.score_threshold}: "
            f"induced {induced_node_count} nodes / {induced_edge_count} edges → "
            f"MST {len(selected_nodes)} nodes / {len(selected_edges)} edges "
            f"({num_components} components)"
        )
        self.last_info = {
            "extractor_type": "MSTKruskalExtractor",
            "extractor_num_input_nodes": int(len(node_scores)),
            "extractor_num_edges": int(len(edges)),
            "extractor_num_selected_nodes": int(len(selected_nodes)),
            "extractor_num_selected_edges": int(len(selected_edges)),
            "mst_score_threshold": float(self.score_threshold),
            "mst_induced_node_count": int(induced_node_count),
            "mst_induced_edge_count": int(induced_edge_count),
            "mst_node_count": int(len(selected_nodes)),
            "mst_edge_count": int(len(selected_edges)),
            "mst_components": int(num_components),
            "extractor_time_s": float(time.perf_counter() - t_start),
        }
        return selected_nodes, selected_edges
