"""M4 Bidirectional Filter (학술 agent §6, Wave 6 Phase 2 (a+aggressive)).

DECISIONS 2026-05-16 (a+aggressive) launch entry §2 정합:
  - Forward prompt = recall_biased_mild (M1-A 재사용, default)
  - Backward prompt = bidirectional_backward (SQL Schema Analyst, question 관점
    column 목록 generation)
  - 2 LLM call/query → union (학술 agent §6.2)
  - 측정: backward_added / backward_gold_recovered / backward_precision
    (gold 가 kwargs 로 들어올 때만 — runtime evaluate 시점)

Wave 6 Phase 4 (DECISIONS 2026-05-17 §4 Top 2 C1 launch) 갱신:
  - Forward prompt 를 config flag `bidirectional_forward_prompt_mode` 로 교체 가능
    {"recall_biased_mild" | "recall_biased_strong" | "recall_biased_exclusion_rule"}
  - Top 2 C1 (M4 + M1-B strong) — Forward 만 strong 으로 교체, Backward 그대로
  - 기존 `forward_section` param 도 backward compat 으로 retain (priority:
    bidirectional_forward_prompt_mode > forward_section)

학술 frame: Filter ↔ Selector co-design 의 추가 axis. paper §3 Inter-Module
Co-Design 의 Filter ↔ Selector 새 axis (paper §3.1 갱신 bullet 정식 채택 5/17).
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

# Wave 6 Phase 4 (Top 2 C1 launch, 2026-05-17): Forward prompt mode → section 매핑.
# 1-to-1 매핑 (prompts/filter.md 의 section name 과 동일) 단 explicit param 으로
# validation 강화 + 의미론적 명명.
# Wave 6 Phase 5 (Top 2 C2 launch, 2026-05-17): "voting_multi_prompt" mode 추가 —
# Forward 부분을 M3 multi-prompt voting (3 LLM call + OR/MAJORITY/AND) 으로 대체.
_BIDIRECTIONAL_FORWARD_MODES: Dict[str, str] = {
    "recall_biased_mild": "recall_biased_mild",        # M1-A (default, Phase 2 (a+aggressive))
    "recall_biased_strong": "recall_biased_strong",    # M1-B (Top 2 C1)
    "recall_biased_exclusion_rule": "recall_biased_exclusion_rule",  # M1-C
    "voting_multi_prompt": "__voting__",               # Top 2 C2 (composition with M3)
}

_VALID_BIDIRECTIONAL_VOTING_STRATEGIES: Tuple[str, ...] = ("OR", "MAJORITY", "AND")


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
        forward_section: str = "recall_biased_mild",   # PROMPT_M4_FORWARD = M1-A (backward compat)
        backward_section: str = "bidirectional_backward",
        bidirectional_forward_prompt_mode: Optional[str] = None,  # Wave 6 Phase 4 (5/17)
        bidirectional_forward_voting_strategy: str = "MAJORITY",   # Wave 6 Phase 5 (5/17 C2)
        provider: Optional[str] = "glm",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        # Wave 6 Phase 4 (Top 2 C1, 2026-05-17): bidirectional_forward_prompt_mode
        # 우선 사용 (의미론적 + validation 강화). 미명시 시 기존 forward_section 그대로.
        # Wave 6 Phase 5 (Top 2 C2, 2026-05-17): "voting_multi_prompt" mode 추가 —
        # Forward 부분을 M3 multi-prompt voting (composition) 으로 대체.
        if bidirectional_forward_prompt_mode is not None:
            if bidirectional_forward_prompt_mode not in _BIDIRECTIONAL_FORWARD_MODES:
                raise ValueError(
                    f"bidirectional_forward_prompt_mode='{bidirectional_forward_prompt_mode}' "
                    f"invalid. Expected one of "
                    f"{sorted(_BIDIRECTIONAL_FORWARD_MODES.keys())}."
                )
            resolved_forward_section = _BIDIRECTIONAL_FORWARD_MODES[
                bidirectional_forward_prompt_mode
            ]
        else:
            resolved_forward_section = forward_section

        # Voting strategy validation — voting_multi_prompt mode 에서만 의미. 다른 mode 일
        # 경우 stored 만 (logger 에서 indicate).
        if bidirectional_forward_voting_strategy not in _VALID_BIDIRECTIONAL_VOTING_STRATEGIES:
            raise ValueError(
                f"bidirectional_forward_voting_strategy="
                f"'{bidirectional_forward_voting_strategy}' invalid. "
                f"Expected one of {_VALID_BIDIRECTIONAL_VOTING_STRATEGIES}."
            )

        self.model_name = model_name
        self.max_iteration = max_iteration
        self.temperature = float(temperature)
        self.db_dir = db_dir
        self.num_examples = max(0, int(num_examples))
        self.sanitize_output = bool(sanitize_output)
        self.forward_section = resolved_forward_section
        self.backward_section = backward_section
        self.bidirectional_forward_prompt_mode = bidirectional_forward_prompt_mode
        self.bidirectional_forward_voting_strategy = bidirectional_forward_voting_strategy
        self._forward_is_voting = (
            bidirectional_forward_prompt_mode == "voting_multi_prompt"
        )
        self.prompt_manager = PromptManager()
        self.client = self._make_llm_client(
            api_key=api_key, base_url=base_url, provider=provider,
        )
        voting_note = (
            f", voting_strategy={bidirectional_forward_voting_strategy}"
            if self._forward_is_voting else ""
        )
        logger.info(
            "Initialized BidirectionalFilter "
            f"(model={model_name}, forward={resolved_forward_section}, "
            f"forward_prompt_mode={bidirectional_forward_prompt_mode or 'legacy(forward_section)'}, "
            f"backward={backward_section}, sanitize={self.sanitize_output}"
            f"{voting_note})"
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

    def _call_forward(
        self,
        query: str,
        schema_str: str,
        example_json_str: str,
        subgraph: Dict[str, List[str]],
    ) -> Tuple[Dict[str, List[str]], int, Dict[str, Any]]:
        """Forward LLM call — single mode 또는 M3 voting (composition).

        Returns:
            (forward_clean_dict, total_hallucination_removed, forward_diag)
              - forward_clean_dict: voting 시 voted result, single 시 sanitized JSON
              - total_hallucination_removed: 합계 (voting 시 3-prompt 합)
              - forward_diag: llm_calls, voting mode 시 raw_counts/voted_counts
        """
        diag: Dict[str, Any] = {"llm_calls": 0}

        # Single LLM call path (legacy 또는 explicit 3 mild/strong/exclusion_rule)
        if not self._forward_is_voting:
            try:
                raw_fwd = self._call_prompt(
                    self.forward_section, query, schema_str, example_json_str,
                )
            except Exception as e:
                logger.warning(f"[M4 forward] LLM call failed: {e}")
                raw_fwd = ""
            diag["llm_calls"] = 1
            parsed = self._parse_json_dict(raw_fwd)
            if self.sanitize_output:
                fwd_clean, halluc = XiYanFilter.sanitize_filter_output(parsed, subgraph)
            else:
                fwd_clean = parsed
                halluc = 0
            return fwd_clean, halluc, diag

        # Voting path (Wave 6 Phase 5 Top 2 C2, 5/17) — M3 composition
        # local import 로 circular 회피 + scope 제한
        from modules.filters.multi_prompt_voting_filter import (
            MultiPromptVotingFilter,
            _VOTING_PROMPT_SECTIONS,
        )

        per_prompt_clean: Dict[str, Dict[str, List[str]]] = {}
        per_prompt_halluc: Dict[str, int] = {}
        for letter, section in _VOTING_PROMPT_SECTIONS.items():
            try:
                resp = self._call_prompt(section, query, schema_str, example_json_str)
            except Exception as e:
                logger.warning(f"[M4 forward voting {letter}] LLM call failed: {e}")
                resp = ""
            diag["llm_calls"] += 1
            parsed = self._parse_json_dict(resp)
            if self.sanitize_output:
                clean, removed = XiYanFilter.sanitize_filter_output(parsed, subgraph)
            else:
                clean = parsed
                removed = 0
            per_prompt_clean[letter] = clean
            per_prompt_halluc[letter] = int(removed)

        voted = MultiPromptVotingFilter.multi_prompt_voting(
            per_prompt_clean,
            strategy=self.bidirectional_forward_voting_strategy,
        )
        total_halluc = sum(per_prompt_halluc.values())
        diag["raw_counts"] = {
            k: sum(len(v) for v in (per_prompt_clean[k] or {}).values())
            for k in per_prompt_clean
        }
        diag["voted_counts"] = {
            self.bidirectional_forward_voting_strategy:
                sum(len(v) for v in voted.values())
        }
        diag["per_prompt_hallucination_removed"] = per_prompt_halluc
        return voted, total_halluc, diag

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
                forward_prompt_mode=self.bidirectional_forward_prompt_mode,
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

        # Forward — mode-aware (single LLM or voting 3 LLM)
        fwd_clean, halluc_fwd, fwd_diag = self._call_forward(
            query=query, schema_str=schema_str, example_json_str=example_json_str,
            subgraph=subgraph,
        )

        # Backward (note: backward prompt 은 example_json_str 안 받음 — schema_str + query 만)
        try:
            raw_bwd = self._call_prompt(self.backward_section, query, schema_str)
        except Exception as e:
            logger.warning(f"[M4 backward] LLM call failed: {e}")
            raw_bwd = ""
        bwd_parsed = self._parse_json_dict(raw_bwd)

        # Sanitize backward (학술 agent §2.3)
        halluc_bwd = 0
        if self.sanitize_output:
            bwd_clean, halluc_bwd = XiYanFilter.sanitize_filter_output(bwd_parsed, subgraph)
        else:
            bwd_clean = bwd_parsed

        # Union
        merged = self._union_filter_outputs(fwd_clean, bwd_clean)
        final_nodes = sorted(f"{t}.{c}" for t, cols in merged.items() for c in cols)
        status = "Answerable" if final_nodes else "Unanswerable"

        # Backward contribution (gold 없으면 stat 만 partial)
        contrib = self.analyze_backward_contribution(fwd_clean, bwd_clean, gold)

        token_after = AgentUtils.token_snapshot()
        # Wave 11 (Schema Serialization Direction C, 2026-05-19) — F/B 별도 저장:
        # source_tagged_serializer (C-v1) 가 forward_set / backward_set 의 disjoint
        # mask 를 필요로 함. stats / filter_info 양쪽에 노출.
        forward_set_list = sorted(
            f"{t}.{c}" for t, cols in fwd_clean.items() for c in cols
        )
        backward_set_list = sorted(
            f"{t}.{c}" for t, cols in bwd_clean.items() for c in cols
        )

        self.last_info = AgentUtils.build_filter_info(
            filter_type="BidirectionalFilter",
            input_subgraph=subgraph, final_nodes=final_nodes, status=status,
            token_before=token_before, token_after=token_after, t_start=t_start,
            model=self.model_name,
            forward_section=self.forward_section,
            forward_prompt_mode=self.bidirectional_forward_prompt_mode,
            forward_voting_strategy=(
                self.bidirectional_forward_voting_strategy if self._forward_is_voting else None
            ),
            forward_llm_calls=int(fwd_diag.get("llm_calls", 0)),
            forward_raw_counts=fwd_diag.get("raw_counts"),
            forward_voted_counts=fwd_diag.get("voted_counts"),
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
            # Wave 11 spec naming 정합 (filter_stage_infos source) — Serializer 가 직접 활용
            forward_set=forward_set_list,
            backward_set=backward_set_list,
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
                "forward_nodes": forward_set_list,
                "backward_nodes": backward_set_list,
                # Wave 11 spec naming alias — C-v1 source_tagged_serializer 의 entry
                "forward_set": forward_set_list,
                "backward_set": backward_set_list,
            },
            "filter_info": dict(self.last_info),
        }
