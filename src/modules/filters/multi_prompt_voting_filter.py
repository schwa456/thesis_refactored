"""M3 Multi-Prompt OR Voting Filter (학술 agent §5, Wave 6 Phase 2 (a+aggressive)).

DECISIONS 2026-05-16 (a+aggressive) launch entry §2 정합:
  - 3 prompts: recall_biased_mild (M1-A 재사용) + voting_prompt_b (SQL clause
    decomposition) + voting_prompt_c (conservative 3-rule exclusion)
  - 3 raw LLM call/query → 3 voting strategies (OR / MAJORITY / AND) post-processing
  - Single refine() 에서 모든 voting variant 측정 → filter_info 에 동봉
    (root + analyzer 가 yaml-driven default 외 strategies 도 함께 분석 가능)

학술 frame: Inclusion bias axis spectrum 의 extreme — R-P trade-off endpoint
정량 evidence. paper §V.5.x.M.15 candidate 강화 + Filter Dominance narrative
보강.
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

_VOTING_PROMPT_SECTIONS: Dict[str, str] = {
    "A": "recall_biased_mild",     # PROMPT_M3_A = M1-A 재사용
    "B": "voting_prompt_b",         # SQL clause decomposition
    "C": "voting_prompt_c",         # Conservative 3-rule exclusion
}

_VALID_VOTING_STRATEGIES: Tuple[str, ...] = ("OR", "MAJORITY", "AND")
_VOTE_THRESHOLDS: Dict[str, int] = {"OR": 1, "MAJORITY": 2, "AND": 3}


@register("filter", "MultiPromptVotingFilter")
class MultiPromptVotingFilter(BaseFilter):
    """M3 Multi-Prompt OR Voting — 3 prompts × 3 voting strategies (학술 agent §5)."""

    def __init__(
        self,
        model_name: str,
        max_iteration: int = 1,
        temperature: float = 0.0,
        db_dir: str = "./data/raw/BIRD_dev/dev_databases",
        num_examples: int = 3,
        sanitize_output: bool = True,
        voting_strategies: Optional[List[str]] = None,
        default_voting_strategy: str = "OR",
        provider: Optional[str] = "glm",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        # Validate strategies
        strats = list(voting_strategies or list(_VALID_VOTING_STRATEGIES))
        for s in strats:
            if s not in _VALID_VOTING_STRATEGIES:
                raise ValueError(
                    f"voting_strategy '{s}' invalid. "
                    f"Expected subset of {_VALID_VOTING_STRATEGIES}."
                )
        if default_voting_strategy not in strats:
            raise ValueError(
                f"default_voting_strategy='{default_voting_strategy}' must be in "
                f"voting_strategies={strats}."
            )
        self.model_name = model_name
        self.max_iteration = max_iteration
        self.temperature = float(temperature)
        self.db_dir = db_dir
        self.num_examples = max(0, int(num_examples))
        self.sanitize_output = bool(sanitize_output)
        self.voting_strategies = strats
        self.default_voting_strategy = default_voting_strategy
        self.prompt_manager = PromptManager()
        self.client = self._make_llm_client(
            api_key=api_key, base_url=base_url, provider=provider,
        )
        logger.info(
            "Initialized MultiPromptVotingFilter "
            f"(model={model_name}, voting={strats}, default={default_voting_strategy}, "
            f"sanitize={self.sanitize_output})"
        )

    # ------------------------------------------------------------------
    # Voting helper (학술 agent §5.2)
    # ------------------------------------------------------------------
    @staticmethod
    def multi_prompt_voting(
        results: Dict[str, Dict[str, List[str]]],
        strategy: str = "OR",
    ) -> Dict[str, List[str]]:
        """results = {"A": {table: [col, ...]}, "B": ..., "C": ...} → voted dict.

        strategy ∈ {"OR" (≥1 vote → keep), "MAJORITY" (≥2), "AND" (==3)}.
        """
        threshold = _VOTE_THRESHOLDS.get(strategy)
        if threshold is None:
            raise ValueError(f"unknown strategy '{strategy}'")
        all_keys = list(results.keys())
        # union of tables
        all_tables: Set[str] = set()
        for k in all_keys:
            all_tables.update((results.get(k) or {}).keys())
        voted: Dict[str, List[str]] = {}
        for table in all_tables:
            all_cols: Set[str] = set()
            for k in all_keys:
                all_cols.update((results.get(k) or {}).get(table, []) or [])
            kept: List[str] = []
            for col in all_cols:
                votes = sum(
                    1 for k in all_keys
                    if col in ((results.get(k) or {}).get(table, []) or [])
                )
                if votes >= threshold:
                    kept.append(col)
            if kept:
                voted[table] = kept
        return voted

    # ------------------------------------------------------------------
    # LLM call helpers — XiYan 의 schema-with-values 형식 재사용
    # ------------------------------------------------------------------
    def _build_schema_str(self, subgraph: Dict[str, List[str]], db_id: Optional[str]) -> str:
        """XiYanFilter._build_mschema_with_values 와 동일 패턴 재사용."""
        # 임시 XiYanFilter instance 만들어 helper 만 사용 — 가벼움
        # 더 깔끔하게는 helper 를 BaseFilter 로 promotion 가능. 본 구현은 helper-call.
        helper = XiYanFilter.__new__(XiYanFilter)
        helper.db_dir = self.db_dir
        helper.num_examples = self.num_examples
        return helper._build_mschema_with_values(subgraph, db_id or "")

    def _call_prompt(
        self, section: str, query: str, schema_str: str, example_json_str: str,
    ) -> str:
        prompt = self.prompt_manager.load_prompt(
            file_name='filter', section=section,
            schema_str=schema_str, query=query, example_json_str=example_json_str,
        )
        return self.client.generate_text(
            prompt=prompt, model=self.model_name, temperature=self.temperature,
        )

    @staticmethod
    def _parse_json_dict(response: str) -> Dict[str, List[str]]:
        """XiYan parse + 단순화 — {table: [col, ...]} 형식만 채택. 그 외 무시.

        실패 시 빈 dict (recall-safe path 는 caller 가 결정).
        """
        import json
        import re
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
                # M2 형식 ({col: {"include": bool}}) — include=True 만 채택. 본 클래스
                # 는 plain JSON 위주, 단 robust 위해 처리.
                out[t] = [c for c, meta in v.items()
                          if isinstance(c, str) and (
                              meta is True
                              or (isinstance(meta, dict) and meta.get("include"))
                          )]
        return out

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def refine(
        self, query: str, subgraph: Dict[str, List[str]], db_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        empty_tok = {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
        if not subgraph:
            self.last_info = AgentUtils.build_filter_info(
                filter_type="MultiPromptVotingFilter",
                input_subgraph={}, final_nodes=[], status="Unanswerable",
                token_before=empty_tok, token_after=empty_tok, t_start=t_start,
                model=self.model_name, voting_strategies=self.voting_strategies,
                default_voting_strategy=self.default_voting_strategy,
                raw_counts={"A": 0, "B": 0, "C": 0},
                hallucination_removed={"A": 0, "B": 0, "C": 0},
                voted_counts={s: 0 for s in self.voting_strategies},
                voted_nodes={s: [] for s in self.voting_strategies},
            )
            return {
                "status": "Unanswerable", "final_nodes": [],
                "reasoning": "Empty input subgraph",
                "filter_info": dict(self.last_info),
            }

        token_before = AgentUtils.token_snapshot()

        # 1. schema_str + example_json_str (XiYan 양식 정합)
        schema_str = self._build_schema_str(subgraph, db_id)
        example_tables = list(subgraph.keys())[:2]
        example_obj: Dict[str, List[str]] = {}
        for idx, t in enumerate(example_tables):
            example_obj[t] = (subgraph[t] or [])[: (2 if idx == 0 else 1)]
        import json
        example_json_str = json.dumps(example_obj)

        # 2. 3 prompts 호출
        raw_responses: Dict[str, str] = {}
        parsed: Dict[str, Dict[str, List[str]]] = {}
        sanitized: Dict[str, Dict[str, List[str]]] = {}
        hallucinated_removed_by: Dict[str, int] = {}
        for letter, section in _VOTING_PROMPT_SECTIONS.items():
            try:
                resp = self._call_prompt(section, query, schema_str, example_json_str)
            except Exception as e:
                logger.warning(f"[M3 {letter}] LLM call failed: {e}")
                resp = ""
            raw_responses[letter] = resp
            parsed_dict = self._parse_json_dict(resp)
            parsed[letter] = parsed_dict
            if self.sanitize_output:
                san, removed = XiYanFilter.sanitize_filter_output(parsed_dict, subgraph)
                sanitized[letter] = san
                hallucinated_removed_by[letter] = int(removed)
            else:
                sanitized[letter] = parsed_dict
                hallucinated_removed_by[letter] = 0

        # 3. Voting variants 전체 평가
        voted_by_strategy: Dict[str, Dict[str, List[str]]] = {}
        voted_node_lists: Dict[str, List[str]] = {}
        for strat in self.voting_strategies:
            voted = self.multi_prompt_voting(sanitized, strategy=strat)
            voted_by_strategy[strat] = voted
            voted_node_lists[strat] = [
                f"{t}.{c}" for t, cols in voted.items() for c in cols
            ]

        # 4. Default strategy 의 final 채택
        final_nodes = sorted(voted_node_lists[self.default_voting_strategy])
        status = "Answerable" if final_nodes else "Unanswerable"

        token_after = AgentUtils.token_snapshot()
        raw_counts = {
            k: sum(len(v) for v in (sanitized[k] or {}).values())
            for k in sanitized
        }
        voted_counts = {s: len(voted_node_lists[s]) for s in self.voting_strategies}

        self.last_info = AgentUtils.build_filter_info(
            filter_type="MultiPromptVotingFilter",
            input_subgraph=subgraph, final_nodes=final_nodes, status=status,
            token_before=token_before, token_after=token_after, t_start=t_start,
            model=self.model_name,
            voting_strategies=self.voting_strategies,
            default_voting_strategy=self.default_voting_strategy,
            sanitize_output=self.sanitize_output,
            raw_counts=raw_counts,
            hallucination_removed=hallucinated_removed_by,
            voted_counts=voted_counts,
            voted_nodes=voted_node_lists,
            n_input_columns=sum(len(v) for v in subgraph.values()),
            n_final_nodes=len(final_nodes),
        )
        return {
            "status": status, "final_nodes": final_nodes,
            "reasoning": (
                f"[M3 voting] raw={raw_counts}, voted={voted_counts}, "
                f"default={self.default_voting_strategy} → {len(final_nodes)} nodes "
                f"(hallucinated removed: {hallucinated_removed_by})"
            ),
            "stats": {
                "voting_strategies": self.voting_strategies,
                "default_voting_strategy": self.default_voting_strategy,
                "raw_counts": raw_counts,
                "voted_counts": voted_counts,
                "voted_node_lists": voted_node_lists,
                "hallucination_removed": hallucinated_removed_by,
            },
            "filter_info": dict(self.last_info),
        }
