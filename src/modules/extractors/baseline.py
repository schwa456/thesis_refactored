import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from modules.registry import register
from modules.base import BaseExtractor
from utils.logger import get_logger

logger = get_logger(__name__)

@register("extractor", "TopK")
class TopKExtractor(BaseExtractor):
    """ 엣지(연결성) 정보를 무시하고 오직 점수(Prize)가 높은 상위 K개 노드만 추출합니다. """
    def __init__(self, top_k: int = 15, **kwargs):
        self.top_k = top_k
        logger.info(f"Initialized TopK Extractor (k={self.top_k})")

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float], seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        top_k_indices = np.argsort(node_scores)[-self.top_k:][::-1]
        logger.debug(f"[TopK] Extracted exactly {self.top_k} nodes.")
        return top_k_indices.tolist(), []

@register("extractor", "None")
class NoneExtractor(BaseExtractor):
    """ 아무런 추출을 하지 않고 들어온 입력을 그대로 통과시킵니다. (Pass-through) """
    def __init__(self, **kwargs):
        logger.info("Initialized None Extractor (Pass-through)")

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float], seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        nodes = seed_nodes if seed_nodes is not None else list(range(len(node_scores)))
        return nodes, []