"""F4. Uncertainty-Gated Adaptive Depth Filter.

Route each query to a filter of different depth based on selector confidence:
  * High confidence → XiYan (single pass, cheapest).
  * Medium confidence → ReflectionFilter (self-refine).
  * Low confidence → TieredBidirectionalAgentFilter (prune + restore).

Confidence signal: margin between top-k GAT scores and the decision threshold
(default 0.5). Defaults to "high" when no scores available (safe fallback).

Inspired by adaptive computation (Graves 2016), ReFoRCE ambiguous deferral,
and the uncertainty routing already present in AdaptiveMultiAgentFilter.
"""
from typing import Dict, List, Any

from modules.registry import register
from modules.base import BaseFilter
from modules.filters.xiyan_filter import XiYanFilter
from modules.filters.reflection_filter import ReflectionFilter
from modules.filters.bidirectional_agent_filter import (
    TieredBidirectionalAgentFilter,
)
from utils.logger import get_logger

logger = get_logger(__name__)


@register("filter", "AdaptiveDepthFilter")
class AdaptiveDepthFilter(BaseFilter):
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        high_conf_threshold: float = 0.20,
        low_conf_threshold: float = 0.05,
        db_dir: str = "./data/raw/BIRD_dev/dev_databases",
        api_key: str = "vllm",
        base_url: str = "http://localhost:8000/v1",
        reflection_max_iteration: int = 1,
        **kwargs,
    ):
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold

        self._fast = XiYanFilter(
            model_name=model_name,
            max_iteration=1,
            temperature=temperature,
            db_dir=db_dir,
            api_key=api_key,
            base_url=base_url,
        )
        self._medium = ReflectionFilter(
            model_name=model_name,
            max_iteration=reflection_max_iteration,
            temperature=temperature,
            db_dir=db_dir,
            api_key=api_key,
            base_url=base_url,
        )
        self._deep = TieredBidirectionalAgentFilter(
            model_name=model_name,
            temperature=temperature,
            use_graph_context=True,
            db_dir=db_dir,
            api_key=api_key,
            base_url=base_url,
        )
        logger.info(
            f"Initialized AdaptiveDepthFilter "
            f"(high>={high_conf_threshold}, low<={low_conf_threshold})"
        )

    def _estimate_confidence(
        self,
        subgraph: Dict[str, List[str]],
        gat_scores: Dict[str, float],
    ) -> float:
        """Margin of selected nodes above 0.5 decision threshold.

        Returns a scalar in [0, 0.5]. Larger = more confident.
        """
        scores = []
        for tbl, cols in subgraph.items():
            for col in cols:
                node = f"{tbl}.{col}"
                if node in gat_scores:
                    scores.append(gat_scores[node])
        if not scores:
            return self.high_conf_threshold
        scores.sort(reverse=True)
        top_k = scores[: max(1, len(scores) // 3)]
        avg_margin = sum(max(0.0, s - 0.5) for s in top_k) / len(top_k)
        return avg_margin

    def refine(
        self,
        query: str,
        subgraph: Dict[str, List[str]],
        db_id: str = None,
        gat_scores: Dict[str, float] = None,
        tier2_pool: List[str] = None,
        metadata: Dict[str, Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        gat_scores = gat_scores or {}
        conf = self._estimate_confidence(subgraph, gat_scores)

        if conf >= self.high_conf_threshold:
            route = "fast"
            result = self._fast.refine(
                query=query, subgraph=subgraph, db_id=db_id, **kwargs
            )
        elif conf <= self.low_conf_threshold:
            route = "deep"
            result = self._deep.refine(
                query=query,
                subgraph=subgraph,
                db_id=db_id,
                tier2_pool=tier2_pool,
                gat_scores=gat_scores,
                metadata=metadata,
                **kwargs,
            )
        else:
            route = "medium"
            result = self._medium.refine(
                query=query, subgraph=subgraph, db_id=db_id, **kwargs
            )

        result["adaptive_route"] = route
        result["adaptive_confidence"] = conf
        logger.debug(f"[AdaptiveDepth] conf={conf:.3f} route={route}")
        return result
