"""M4 Bidirectional Filter (학술 agent §6, Wave 6 Phase 2 (a+aggressive)).

DECISIONS 2026-05-16 (a+aggressive) launch entry §2 정합:
  - Forward prompt = recall_biased_mild (M1-A 재사용)
  - Backward prompt = bidirectional_backward (SQL Schema Analyst, question 관점
    column 목록 generation)
  - 2 LLM call/query → union (학술 agent §6.2)
  - 측정: backward_added / backward_gold_recovered / backward_precision
    (gold 가 kwargs 로 들어올 때만 — runtime evaluate 시점)

학술 frame: Filter ↔ Selector co-design 의 추가 axis. paper §3 Inter-Module
Co-Design 의 Filter ↔ Selector 새 axis (paper §3.1 갱신 candidate).
"""
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from modules.registry import register
from modules.base import BaseFilter
from modules.filters.agents import AgentUtils
from modules.filters.xiyan_filter import XiYanFilter
from prompts.prompt_manager import PromptManager
from utils.logger import get_logger

logger = get_logger(__name__)


@register("filter", "BidirectionalFilter")
class BidirectionalFilter(BaseFilter):
    """M4 Forward + Backward union (학술 agent §6)."""

    def __init__(
        self,
        model_name: str,
        max_iteration: int = 1,
        temperature: float = 0.0,
        db_dir: str = "./data/raw/BIRD_dev/dev_databases",
        num_examples: int = 3,
        sanitize_output: bool = True,
        forward_section: str = "recall_biased_mild",   # PROMPT_M4_FORWARD = M1-A
        backward_section: str = "bidirectional_backward",
        provider: Optional[str] = "glm",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.max_iteration = max_iteration
        self.temperature = float(temperature)
        self.db_dir = db_dir
        self.num_examples = max(0, int(num_examples))
        self.sanitize_output = bool(sanitize_output)
        self.forward_section = forward_section
        self.backward_section = backward_section
        self.prompt_manager = PromptManager()
        self.client = self._make_llm_client(
            api_key=api_key, base_url=base_url, provider=provider,
        )
        logger.info(
            "Initialized BidirectionalFilter "
            f"(model={model_name}, forward={forward_section}, "
            f"backward={backward_section}, sanitize={self.sanitize_output})"
        )

    # ------------------------------------------------------------------
    # Helpers (XiYan 형식 재사용)
    # ------------------------------------------------------------------
    def _build_schema_str(self, subgraph: Dict[str, List[str]], db_id: Optional[str]) -> str:
        helper = XiYanFilter.__new__(XiYanFilter)
        helper.db_dir = self.db_dir
        helper.num_examples = self.num_examples
        return helper._build_mschema_with_values(subgraph, db_id or "")

    def _call_prompt(
        self,
        section: str,
        query: str,
        schema_str: str,
        example_json_str: Optional[str] = None,
    ) -> str:
        kwargs = {"schema_str": schema_str, "query": query}
        if example_json_str is not None:
            kwargs["example_json_str"] = example_json_str
        prompt = self.prompt_manager.load_prompt(
            file_name='filter', section=section, **kwargs,
        )
        return self.client.generate_text(
            prompt=prompt, model=self.model_name, temperature=self.temperature,
        )

    @staticmethod
    def _parse_json_dict(response: str) -> Dict[str, List[str]]:
        import json
        if not isinstance(response, str) or not response.strip():
            return {}
        cleaned = response.replace("```json", "").replace("```", "").strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return {}
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except Exception:
            return {}
        if not isinstance(parsed, dict):
            return {}
        out: Dict[str, List[str]] = {}
        for t, v in parsed.items():
            if not isinstance(t, str):
                continue
            if isinstance(v, list):
                out[t] = [c for c in v if isinstance(c, str)]
            elif isinstance(v, dict):
                out[t] = [c for c, meta in v.items()
                          if isinstance(c, str) and (
                              meta is True
                              or (isinstance(meta, dict) and meta.get("include"))
                          )]
        return out

    @staticmethod
    def _union_filter_outputs(
        forward: Dict[str, List[str]], backward: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """Forward ∪ Backward — 학술 agent §6.2 정합."""
        all_tables = set(forward) | set(backward)
        merged: Dict[str, List[str]] = {}
        for t in all_tables:
            cols = set(forward.get(t, []) or []) | set(backward.get(t, []) or [])
            if cols:
                merged[t] = sorted(cols)
        return merged

    @staticmethod
    def _flatten(schema: Dict[str, List[str]]) -> Set[str]:
        out: Set[str] = set()
        for t, cols in (schema or {}).items():
            if not cols:
                out.add(t)
                continue
            for c in cols:
                out.add(f"{t}.{c}")
        return out

    @staticmethod
    def analyze_backward_contribution(
        forward_result: Dict[str, List[str]],
        backward_result: Dict[str, List[str]],
        gold: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Backward 가 Forward 에서 놓친 gold 를 회복했는지 정량 (학술 agent §6.2)."""
        fwd_cols = BidirectionalFilter._flatten(forward_result)
        bwd_cols = BidirectionalFilter._flatten(backward_result)
        backward_only = bwd_cols - fwd_cols
        stats: Dict[str, Any] = {
            "backward_added": len(backward_only),
            "forward_count": len(fwd_cols),
            "backward_count": len(bwd_cols),
            "union_count": len(fwd_cols | bwd_cols),
        }
        if gold is not None:
            gold_cols = BidirectionalFilter._flatten(gold)
            recovered = backward_only & gold_cols
            stats["backward_gold_recovered"] = len(recovered)
            stats["backward_precision"] = (
                len(recovered) / len(backward_only) if backward_only else 0.0
            )
        return stats

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def refine(
        self, query: str, subgraph: Dict[str, List[str]], db_id: Optional[str] = None,
        gold: Optional[Dict[str, List[str]]] = None,  # optional kwargs for contribution stats
        **kwargs,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        empty_tok = {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
        if not subgraph:
            self.last_info = AgentUtils.build_filter_info(
                filter_type="BidirectionalFilter",
                input_subgraph={}, final_nodes=[], status="Unanswerable",
                token_before=empty_tok, token_after=empty_tok, t_start=t_start,
                model=self.model_name,
                forward_section=self.forward_section,
                backward_section=self.backward_section,
                forward_count=0, backward_count=0, union_count=0,
                backward_added=0,
            )
            return {
                "status": "Unanswerable", "final_nodes": [],
                "reasoning": "Empty input subgraph",
                "filter_info": dict(self.last_info),
            }
        token_before = AgentUtils.token_snapshot()
        schema_str = self._build_schema_str(subgraph, db_id)
        import json
        example_tables = list(subgraph.keys())[:2]
        example_obj: Dict[str, List[str]] = {}
        for idx, t in enumerate(example_tables):
            example_obj[t] = (subgraph[t] or [])[: (2 if idx == 0 else 1)]
        example_json_str = json.dumps(example_obj)

        # Forward
        try:
            raw_fwd = self._call_prompt(
                self.forward_section, query, schema_str, example_json_str,
            )
        except Exception as e:
            logger.warning(f"[M4 forward] LLM call failed: {e}")
            raw_fwd = ""
        fwd_parsed = self._parse_json_dict(raw_fwd)

        # Backward (note: backward prompt 은 example_json_str 안 받음 — schema_str + query 만)
        try:
            raw_bwd = self._call_prompt(self.backward_section, query, schema_str)
        except Exception as e:
            logger.warning(f"[M4 backward] LLM call failed: {e}")
            raw_bwd = ""
        bwd_parsed = self._parse_json_dict(raw_bwd)

        # Sanitize (학술 agent §2.3) — extractor output 에 없는 entry 제거
        halluc_fwd = 0
        halluc_bwd = 0
        if self.sanitize_output:
            fwd_clean, halluc_fwd = XiYanFilter.sanitize_filter_output(fwd_parsed, subgraph)
            bwd_clean, halluc_bwd = XiYanFilter.sanitize_filter_output(bwd_parsed, subgraph)
        else:
            fwd_clean = fwd_parsed
            bwd_clean = bwd_parsed

        # Union
        merged = self._union_filter_outputs(fwd_clean, bwd_clean)
        final_nodes = sorted(f"{t}.{c}" for t, cols in merged.items() for c in cols)
        status = "Answerable" if final_nodes else "Unanswerable"

        # Backward contribution (gold 없으면 stat 만 partial)
        contrib = self.analyze_backward_contribution(fwd_clean, bwd_clean, gold)

        token_after = AgentUtils.token_snapshot()
        self.last_info = AgentUtils.build_filter_info(
            filter_type="BidirectionalFilter",
            input_subgraph=subgraph, final_nodes=final_nodes, status=status,
            token_before=token_before, token_after=token_after, t_start=t_start,
            model=self.model_name,
            forward_section=self.forward_section,
            backward_section=self.backward_section,
            sanitize_output=self.sanitize_output,
            forward_count=contrib["forward_count"],
            backward_count=contrib["backward_count"],
            union_count=contrib["union_count"],
            backward_added=contrib["backward_added"],
            backward_gold_recovered=contrib.get("backward_gold_recovered"),
            backward_precision=contrib.get("backward_precision"),
            hallucination_removed_forward=halluc_fwd,
            hallucination_removed_backward=halluc_bwd,
        )
        return {
            "status": status, "final_nodes": final_nodes,
            "reasoning": (
                f"[M4 bidirectional] forward={contrib['forward_count']}, "
                f"backward={contrib['backward_count']}, "
                f"backward_added={contrib['backward_added']}, "
                f"union={contrib['union_count']} "
                f"(hallucinated removed: forward={halluc_fwd}, backward={halluc_bwd})"
            ),
            "stats": {
                "forward_count": contrib["forward_count"],
                "backward_count": contrib["backward_count"],
                "backward_added": contrib["backward_added"],
                "union_count": contrib["union_count"],
                "backward_gold_recovered": contrib.get("backward_gold_recovered"),
                "backward_precision": contrib.get("backward_precision"),
                "hallucination_removed_forward": halluc_fwd,
                "hallucination_removed_backward": halluc_bwd,
                "forward_nodes": sorted(
                    f"{t}.{c}" for t, cols in fwd_clean.items() for c in cols
                ),
                "backward_nodes": sorted(
                    f"{t}.{c}" for t, cols in bwd_clean.items() for c in cols
                ),
            },
            "filter_info": dict(self.last_info),
        }
