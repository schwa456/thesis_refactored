"""StackedFilter — chains multiple filters sequentially.

The output `final_nodes` of stage k feeds stage k+1 as its subgraph. Useful
for combinations like Tiered(prune+restore) → Verifier(unit tests).
"""
from typing import Dict, List, Any

from modules.registry import register, build
from modules.base import BaseFilter
from utils.logger import get_logger

logger = get_logger(__name__)


@register("filter", "StackedFilter")
class StackedFilter(BaseFilter):
    def __init__(self, stages: List[Dict[str, Any]], **kwargs):
        self.stages = [build("filter", s) for s in stages]
        logger.info(f"Initialized StackedFilter ({len(self.stages)} stages)")

    @staticmethod
    def _nodes_to_subgraph(nodes: List[str]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for n in nodes:
            if "." not in n:
                out.setdefault(n, [])
                continue
            t, c = n.split(".", 1)
            out.setdefault(t, [])
            if c not in out[t]:
                out[t].append(c)
        return out

    def refine(
        self, query: str, subgraph: Dict[str, List[str]], db_id: str = None, **kwargs
    ) -> Dict[str, Any]:
        current_subgraph = subgraph
        traces = []
        last: Dict[str, Any] = {}
        for i, f in enumerate(self.stages):
            out = f.refine(
                query=query, subgraph=current_subgraph, db_id=db_id, **kwargs
            )
            last = out
            traces.append(f"stage{i+1}={len(out.get('final_nodes', []))}")
            if out.get("status") == "Unanswerable":
                break
            current_subgraph = self._nodes_to_subgraph(out.get("final_nodes", []))
        last["reasoning"] = (
            f"[Stacked {' | '.join(traces)}] " + last.get("reasoning", "")
        )
        return last
