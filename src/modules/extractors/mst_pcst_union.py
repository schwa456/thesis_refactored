import time
from typing import List, Dict, Tuple, Any, Optional

from modules.registry import register
from modules.base import BaseExtractor
from utils.logger import get_logger

from .mst_kruskal import MSTKruskalExtractor
from .pcst import PCSTExtractor

logger = get_logger(__name__)


@register("extractor", "MSTPCSTUnionExtractor")
class MSTPCSTUnionExtractor(BaseExtractor):
    """
    MST Kruskal ∪ Basic PCST union extractor.

    score_threshold=0.1 통일 양쪽:
      - MSTKruskalExtractor (score_threshold=0.1) → set_A (induced subgraph + Kruskal MST)
      - PCSTExtractor (node_threshold=0.1, default cost params) → set_B (Basic PCST)

    union_nodes = set_A.nodes ∪ set_B.nodes
    union_edges = set_A.edges ∪ set_B.edges (canonical sorted-tuple form)

    의도: MST Kruskal anchor (F1=0.8642) 의 R 상한 검증 + Filter 의 union 처리 능력 정량.

    Edge case:
      - 한쪽이 빈 결과면 union = non-empty side (set union 으로 자연 처리)
      - 양쪽 빈 결과면 ([], []) 반환
    """
    def __init__(self,
                 score_threshold: float = 0.1,
                 base_cost: float = 1.0,
                 belongs_to_cost: float = 0.01,
                 fk_cost: float = 0.05,
                 macro_cost: float = 0.5,   
                 hub_discount: float = 0.2,
                 **kwargs):
        self.score_threshold = float(score_threshold)
        self.base_cost = float(base_cost)
        self.belongs_to_cost = float(belongs_to_cost)
        self.fk_cost = float(fk_cost)
        self.macro_cost = float(macro_cost)
        self.hub_discount = float(hub_discount)

        self.mst = MSTKruskalExtractor(score_threshold=self.score_threshold)
        self.pcst = PCSTExtractor(
            base_cost=self.base_cost,
            belongs_to_cost=self.belongs_to_cost,
            fk_cost=self.fk_cost,
            macro_cost=self.macro_cost,
            hub_discount=self.hub_discount,
            node_threshold=self.score_threshold,
        )
        self.last_info: Dict[str, Any] = {}
        logger.info(
            f"Initialized MSTPCSTUnion Extractor (score_threshold={self.score_threshold}, "
            f"PCST cost: bt={self.belongs_to_cost}, fk={self.fk_cost}, macro={self.macro_cost})"
        )

    @staticmethod
    def _normalize_edges(edges: List[Tuple[int, int]]) -> set:
        """Canonical (min, max) tuple form for set ops, robust to int-like inputs."""
        return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges}

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None,
                **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        t_start = time.perf_counter()

        mst_nodes, mst_edges = self.mst.extract(graph_data, node_scores)
        pcst_nodes, pcst_edges = self.pcst.extract(graph_data, node_scores)

        mst_node_set = {int(n) for n in mst_nodes}
        pcst_node_set = {int(n) for n in pcst_nodes}
        mst_edge_set = self._normalize_edges(mst_edges)
        pcst_edge_set = self._normalize_edges(pcst_edges)

        union_node_set = mst_node_set | pcst_node_set
        union_edge_set = mst_edge_set | pcst_edge_set

        union_nodes = sorted(union_node_set)
        union_edges = sorted(union_edge_set)

        intersection_count = len(mst_node_set & pcst_node_set)
        mst_only_count = len(mst_node_set - pcst_node_set)
        pcst_only_count = len(pcst_node_set - mst_node_set)

        logger.debug(
            f"[MSTPCSTUnion] MST {len(mst_node_set)} ∪ PCST {len(pcst_node_set)} → "
            f"union {len(union_nodes)} (∩={intersection_count}, "
            f"MST-only={mst_only_count}, PCST-only={pcst_only_count})"
        )
        self.last_info = {
            "extractor_type": "MSTPCSTUnionExtractor",
            "extractor_num_input_nodes": int(len(node_scores)),
            "extractor_num_edges": int(len(graph_data.get('edges', []) or [])),
            "extractor_num_selected_nodes": int(len(union_nodes)),
            "extractor_num_selected_edges": int(len(union_edges)),
            "score_threshold": float(self.score_threshold),
            "mst_node_count": int(len(mst_node_set)),
            "mst_edge_count": int(len(mst_edge_set)),
            "pcst_node_count": int(len(pcst_node_set)),
            "pcst_edge_count": int(len(pcst_edge_set)),
            "union_node_count": int(len(union_nodes)),
            "union_edge_count": int(len(union_edges)),
            "intersection_count": int(intersection_count),
            "mst_only_count": int(mst_only_count),
            "pcst_only_count": int(pcst_only_count),
            "extractor_time_s": float(time.perf_counter() - t_start),
        }
        return union_nodes, union_edges
