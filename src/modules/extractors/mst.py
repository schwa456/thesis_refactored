import networkx as nx
from typing import List, Dict, Tuple, Any, Optional

from modules.registry import register
from modules.base import BaseExtractor
from utils.logger import get_logger

logger = get_logger(__name__)

@register("extractor", "MST")
class MSTExtractor(BaseExtractor):
    """ 선택된 시드 노드들을 잇는 최소 신장 트리(Minimum Spanning Tree)를 추출합니다. """
    def __init__(self, **kwargs):
        logger.info("Initialized MST Extractor")

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float], seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        if not seed_nodes:
            logger.warning("MST requires seed_nodes. Returning empty.")
            return [], []

        edges = graph_data.get('edges', [])
        G = nx.Graph()
        G.add_edges_from(edges)

        sub_G = G.subgraph(seed_nodes)
        
        if sub_G.number_of_nodes() == 0:
            return seed_nodes, []

        mst_G = nx.minimum_spanning_tree(sub_G)
        
        selected_nodes = list(mst_G.nodes())
        selected_edges = list(mst_G.edges())
        
        logger.debug(f"[MST] Extracted {len(selected_nodes)} nodes from {len(seed_nodes)} seeds.")
        return selected_nodes, selected_edges