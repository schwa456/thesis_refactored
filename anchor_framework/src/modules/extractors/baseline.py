import time
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
        self.last_info: Dict[str, Any] = {}
        logger.info(f"Initialized TopK Extractor (k={self.top_k})")

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float], seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        t_start = time.perf_counter()
        top_k_indices = np.argsort(node_scores)[-self.top_k:][::-1]
        selected = top_k_indices.tolist()
        logger.debug(f"[TopK] Extracted exactly {self.top_k} nodes.")
        scores_arr = np.asarray(node_scores, dtype=np.float64) if len(node_scores) else np.array([])
        self.last_info = {
            "extractor_type": "TopKExtractor",
            "extractor_num_input_nodes": int(len(node_scores)),
            "extractor_num_edges": int(len(graph_data.get('edges', []) or [])),
            "extractor_num_selected_nodes": int(len(selected)),
            "extractor_num_selected_edges": 0,
            "top_k": int(self.top_k),
            "min_selected_score": float(scores_arr[selected].min()) if selected and scores_arr.size else 0.0,
            "max_selected_score": float(scores_arr[selected].max()) if selected and scores_arr.size else 0.0,
            "extractor_time_s": float(time.perf_counter() - t_start),
        }
        return selected, []

@register("extractor", "None")
class NoneExtractor(BaseExtractor):
    """ 아무런 추출을 하지 않고 들어온 입력을 그대로 통과시킵니다. (Pass-through) """
    def __init__(self, **kwargs):
        self.last_info: Dict[str, Any] = {}
        logger.info("Initialized None Extractor (Pass-through)")

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float], seed_nodes: Optional[List[int]] = None, **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        t_start = time.perf_counter()
        nodes = seed_nodes if seed_nodes is not None else list(range(len(node_scores)))
        self.last_info = {
            "extractor_type": "NoneExtractor",
            "extractor_num_input_nodes": int(len(node_scores)),
            "extractor_num_edges": int(len(graph_data.get('edges', []) or [])),
            "extractor_num_selected_nodes": int(len(nodes)),
            "extractor_num_selected_edges": 0,
            "passthrough": True,
            "used_seed_nodes": bool(seed_nodes is not None),
            "extractor_time_s": float(time.perf_counter() - t_start),
        }
        return nodes, []