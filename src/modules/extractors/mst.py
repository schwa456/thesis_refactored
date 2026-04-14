import networkx as nx
from typing import List, Dict, Tuple, Any, Optional

from modules.registry import register
from modules.base import BaseExtractor
from utils.logger import get_logger

logger = get_logger(__name__)


def steiner_tree_2approx(G: nx.Graph, terminals: List[int],
                         weight: str = 'weight') -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Kou, Markowsky, Berman (1981) 2-근사 Steiner Tree 알고리즘.

    1. Metric closure: terminal 간 all-pairs shortest paths로 완전 그래프 생성
    2. Metric closure 위에서 MST 구축
    3. MST edge → 원래 graph의 shortest path로 복원
    4. 중복 제거 후 subtree 반환
    """
    # Filter terminals to those actually in the graph
    terminals = [t for t in terminals if t in G]
    if len(terminals) <= 1:
        return terminals, []

    # 1. Metric closure: terminal 간 shortest paths
    metric_closure = nx.Graph()
    paths_cache = {}
    for i, t1 in enumerate(terminals):
        for t2 in terminals[i+1:]:
            try:
                path = nx.shortest_path(G, t1, t2, weight=weight)
                path_weight = sum(
                    G[path[k]][path[k+1]].get(weight, 1.0)
                    for k in range(len(path) - 1)
                )
                metric_closure.add_edge(t1, t2, weight=path_weight)
                paths_cache[(t1, t2)] = path
                paths_cache[(t2, t1)] = list(reversed(path))
            except nx.NetworkXNoPath:
                continue

    if metric_closure.number_of_edges() == 0:
        return terminals, []

    # 2. MST on metric closure
    mst = nx.minimum_spanning_tree(metric_closure, weight=weight)

    # 3. Expand MST edges back to original paths
    steiner_nodes = set()
    steiner_edges = set()
    for u, v in mst.edges():
        path = paths_cache.get((u, v), paths_cache.get((v, u)))
        if path is None:
            continue
        for node in path:
            steiner_nodes.add(node)
        for k in range(len(path) - 1):
            edge = (min(path[k], path[k+1]), max(path[k], path[k+1]))
            steiner_edges.add(edge)

    return list(steiner_nodes), list(steiner_edges)


@register("extractor", "MSTExtractor")
class MSTExtractor(BaseExtractor):
    """
    선택된 시드 노드들을 잇는 Steiner Tree를 추출합니다.
    Kou-Markowsky-Berman 2-근사 알고리즘을 사용하여
    seed 간 metric closure MST를 구한 뒤 원래 경로로 복원합니다.
    """
    def __init__(self, **kwargs):
        logger.info("Initialized MST Extractor (Steiner 2-approx)")

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None,
                **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        if not seed_nodes:
            logger.warning("MST requires seed_nodes. Returning empty.")
            return [], []

        edges = graph_data.get('edges', [])
        G = nx.Graph()
        G.add_edges_from(edges)

        selected_nodes, selected_edges = steiner_tree_2approx(G, seed_nodes)

        logger.debug(f"[MST] Steiner tree: {len(selected_nodes)} nodes from "
                     f"{len(seed_nodes)} seeds ({len(selected_edges)} edges)")
        return selected_nodes, selected_edges
