"""StackedFilter — chains multiple filters sequentially.

The output `final_nodes` of stage k feeds stage k+1 as its subgraph. Useful
for combinations like Tiered(prune+restore) → Verifier(unit tests).
"""
import time
from typing import Dict, List, Any

from modules.registry import register, build
from modules.base import BaseFilter
from modules.filters.agents import AgentUtils
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
        t_start = time.perf_counter()
        token_before = AgentUtils.token_snapshot()
        current_subgraph = subgraph
        traces = []
        last: Dict[str, Any] = {}
        stage_infos: List[Dict[str, Any]] = []
        stages_run = 0
        short_circuited_at = None
        stage_time_total = 0.0
        for i, f in enumerate(self.stages):
            stages_run += 1
            out = f.refine(
                query=query, subgraph=current_subgraph, db_id=db_id, **kwargs
            )
            last = out
            stage_info = getattr(f, "last_info", None) or out.get("filter_info") or {}
            stage_infos.append({
                "stage": i + 1,
                "nodes_in": sum(len(v) for v in current_subgraph.values()),
                "nodes_out": len(out.get("final_nodes", []) or []),
                "status": out.get("status"),
                "info": dict(stage_info),
            })
            stage_time_total += float(stage_info.get("filter_time_s", 0.0) or 0.0)
            traces.append(f"stage{i+1}={len(out.get('final_nodes', []))}")
            if out.get("status") == "Unanswerable":
                short_circuited_at = i + 1
                break
            current_subgraph = self._nodes_to_subgraph(out.get("final_nodes", []))
        last["reasoning"] = (
            f"[Stacked {' | '.join(traces)}] " + last.get("reasoning", "")
        )

        token_after = AgentUtils.token_snapshot()
        self.last_info = AgentUtils.build_filter_info(
            filter_type="StackedFilter",
            input_subgraph=subgraph,
            final_nodes=last.get("final_nodes", []) or [],
            status=last.get("status"),
            token_before=token_before,
            token_after=token_after,
            t_start=t_start,
            num_stages=len(self.stages),
            stages_run=stages_run,
            short_circuited_at=short_circuited_at,
            stage_infos=stage_infos,
            stage_time_total_s=float(stage_time_total),
        )
        last["filter_info"] = dict(self.last_info)
        return last
