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

    Phase 4.1 [2026-05-16] — Selector + Extractor 통합 점수 mode:
      - `seed_selection_mode="threshold"` (default, backward compat): 위 기본 동작.
      - `seed_selection_mode="integrated_score"`: 신규 mode.
            s_integrated(v) = α · 𝟙[v ∈ Selector_TopK] + (1-α) · 𝟙[node_scores[v] ≥ score_threshold]
        s_integrated ∈ {0, α, 1-α, 1} 값을 양쪽 sub-extractor 의 새 node_scores 로 전달.
        sub-extractor threshold 는 0.0 으로 override → s_integrated > 0 (= TopK ∪ threshold-pass)
        인 노드만 effective seed 로 진입.
        - α=1.0 → TopK only (Selector top-K 만 활성)
        - α=0.0 → threshold only (현 c01_01 anchor 와 동일)
        - middle α → TopK ∪ threshold-pass union, PCST prize 가 α 에 따라 가중
            (TopK-only 노드 prize=α, threshold-only prize=1-α, intersection prize=1.0).

      Selector top-K 는 `seed_nodes` 인자로 전달됨 (schema_linking.py:197-200).
    """
    def __init__(self,
                 score_threshold: float = 0.1,
                 base_cost: float = 1.0,
                 belongs_to_cost: float = 0.01,
                 fk_cost: float = 0.05,
                 macro_cost: float = 0.5,
                 hub_discount: float = 0.2,
                 seed_selection_mode: str = "threshold",
                 alpha: float = 1.0,
                 **kwargs):
        if seed_selection_mode not in ("threshold", "integrated_score"):
            raise ValueError(
                f"seed_selection_mode must be 'threshold' or 'integrated_score', "
                f"got '{seed_selection_mode}'"
            )
        if not (0.0 <= float(alpha) <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.score_threshold = float(score_threshold)
        self.base_cost = float(base_cost)
        self.belongs_to_cost = float(belongs_to_cost)
        self.fk_cost = float(fk_cost)
        self.macro_cost = float(macro_cost)
        self.hub_discount = float(hub_discount)
        self.seed_selection_mode = seed_selection_mode
        self.alpha = float(alpha)

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
            f"PCST cost: bt={self.belongs_to_cost}, fk={self.fk_cost}, macro={self.macro_cost}, "
            f"seed_selection_mode={self.seed_selection_mode}, alpha={self.alpha})"
        )

    @staticmethod
    def _normalize_edges(edges: List[Tuple[int, int]]) -> set:
        """Canonical (min, max) tuple form for set ops, robust to int-like inputs."""
        return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges}

    def _compute_integrated_scores(self,
                                   node_scores: List[float],
                                   seed_nodes: Optional[List[int]]) -> List[float]:
        """s_integrated(v) = α · 𝟙[v ∈ TopK] + (1-α) · 𝟙[s_v ≥ θ]."""
        topk_set = {int(s) for s in (seed_nodes or [])}
        theta = self.score_threshold
        a = self.alpha
        return [
            a * (1.0 if i in topk_set else 0.0)
            + (1.0 - a) * (1.0 if float(s) >= theta else 0.0)
            for i, s in enumerate(node_scores)
        ]

    def extract(self, graph_data: Dict[str, Any], node_scores: List[float],
                seed_nodes: Optional[List[int]] = None,
                **kwargs) -> Tuple[List[int], List[Tuple[int, int]]]:
        t_start = time.perf_counter()

        topk_set = {int(s) for s in (seed_nodes or [])}
        if self.seed_selection_mode == "integrated_score":
            integrated_scores = self._compute_integrated_scores(node_scores, seed_nodes)
            # Override sub-extractor thresholds to 0.0 so any positive s_integrated qualifies.
            # MSTKruskal: s > 0 strict (excludes s=0). PCST: prize = max(s − 0, 0) = s.
            old_mst_threshold = self.mst.score_threshold
            old_pcst_threshold = self.pcst.node_threshold
            self.mst.score_threshold = 0.0
            self.pcst.node_threshold = 0.0
            try:
                mst_nodes, mst_edges = self.mst.extract(graph_data, integrated_scores)
                pcst_nodes, pcst_edges = self.pcst.extract(graph_data, integrated_scores)
            finally:
                self.mst.score_threshold = old_mst_threshold
                self.pcst.node_threshold = old_pcst_threshold

            # Telemetry split: how many positive s_integrated nodes per source
            integrated_topk_only = sum(
                1 for i, s in enumerate(integrated_scores)
                if i in topk_set and float(node_scores[i]) < self.score_threshold and s > 0
            )
            integrated_threshold_only = sum(
                1 for i, s in enumerate(integrated_scores)
                if i not in topk_set and float(node_scores[i]) >= self.score_threshold and s > 0
            )
            integrated_intersection = sum(
                1 for i, s in enumerate(integrated_scores)
                if i in topk_set and float(node_scores[i]) >= self.score_threshold and s > 0
            )
            integrated_positive_total = sum(1 for s in integrated_scores if s > 0)
        else:
            mst_nodes, mst_edges = self.mst.extract(graph_data, node_scores)
            pcst_nodes, pcst_edges = self.pcst.extract(graph_data, node_scores)
            integrated_topk_only = None
            integrated_threshold_only = None
            integrated_intersection = None
            integrated_positive_total = None

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
            f"[MSTPCSTUnion mode={self.seed_selection_mode}] "
            f"MST {len(mst_node_set)} ∪ PCST {len(pcst_node_set)} → "
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
            "seed_selection_mode": self.seed_selection_mode,
            "alpha": float(self.alpha),
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
        if self.seed_selection_mode == "integrated_score":
            self.last_info.update({
                "integrated_topk_size": int(len(topk_set)),
                "integrated_topk_only": int(integrated_topk_only or 0),
                "integrated_threshold_only": int(integrated_threshold_only or 0),
                "integrated_intersection": int(integrated_intersection or 0),
                "integrated_positive_total": int(integrated_positive_total or 0),
            })
        return union_nodes, union_edges
