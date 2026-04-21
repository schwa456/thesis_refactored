import json
import ast
import re
import time
from typing import Dict, List, Any, Optional, Set

from modules.registry import register
from modules.base import BaseFilter
from prompts.prompt_manager import PromptManager
from llm_client.api_handler import APIClient
from utils.logger import get_logger

logger = get_logger(__name__)

# 공통 유틸리티 클래스
class AgentUtils:
    @staticmethod
    def generate_ddl(subgraph: Dict[str, List[str]]) -> str:
        ddl_lines = []
        for table, columns in subgraph.items():
            cols_str = ",\n  ".join([f"{col} TEXT" for col in columns])
            ddl_lines.append(f"CREATE TABLE {table} (\n  {cols_str}\n);")
        return "\n\n".join(ddl_lines)

    @staticmethod
    def extract_json(response_text: str) -> dict:
        match = re.search(r'\{.*\}', response_text.replace('\n', ' '), re.DOTALL)
        json_str = match.group() if match else response_text
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(json_str)
            except Exception:
                logger.warning(f"JSON Parsing failed. Returning default fallback.")
                return {"step_by_step_reasoning": "Parse Error", "selected_nodes": [], "final_decision": "Unanswerable"}

    @staticmethod
    def token_snapshot() -> Dict[str, int]:
        """Capture cumulative LLM usage at this moment.

        Delta between a pre-call and post-call snapshot attributes token cost
        to a specific filter invocation (APIClient.TOKEN_USAGE is process-global).
        """
        return dict(APIClient.TOKEN_USAGE)

    @staticmethod
    def token_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
        return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in set(before) | set(after)}

    @staticmethod
    def flatten_schema(schema: Dict[str, List[str]]) -> Set[str]:
        """Flatten {table: [col, ...]} to {'table.col', ...} (tables-without-cols yield bare 'table')."""
        out: Set[str] = set()
        for t, cols in (schema or {}).items():
            if not cols:
                out.add(t)
                continue
            for c in cols:
                out.add(f"{t}.{c}")
        return out

    @staticmethod
    def build_filter_info(
        *,
        filter_type: str,
        input_subgraph: Dict[str, List[str]] = None,
        final_nodes: List[str] = None,
        status: str = None,
        token_before: Dict[str, int] = None,
        token_after: Dict[str, int] = None,
        t_start: Optional[float] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Assemble a standard filter_info dict for `last_info`.

        Guarantees these keys whenever the inputs allow:
          filter_type, filter_status, filter_time_s,
          filter_nodes_before, filter_nodes_after,
          filter_removed, filter_restored,
          filter_llm_calls, filter_tokens_in, filter_cached_tokens, filter_tokens_out.
        Any extra kwargs become filter_<key> entries so callers can emit
        filter-specific fields (iterations, verdicts, tier counts, ...).
        `t_start` should be the `time.perf_counter()` captured at refine() entry.
        """
        info: Dict[str, Any] = {"filter_type": filter_type}
        if status is not None:
            info["filter_status"] = status
        if t_start is not None:
            info["filter_time_s"] = float(time.perf_counter() - t_start)

        input_set: Set[str] = set()
        if input_subgraph is not None:
            input_set = AgentUtils.flatten_schema(input_subgraph)
            info["filter_nodes_before"] = len(input_set)

        final_set: Set[str] = set()
        if final_nodes is not None:
            final_set = set(final_nodes)
            info["filter_nodes_after"] = len(final_set)

        if input_subgraph is not None and final_nodes is not None:
            info["filter_removed"] = len(input_set - final_set)
            info["filter_restored"] = len(final_set - input_set)

        if token_before is not None and token_after is not None:
            delta = AgentUtils.token_delta(token_before, token_after)
            info["filter_llm_calls"] = int(delta.get("calls", 0))
            info["filter_tokens_in"] = int(delta.get("input_tokens", 0))
            info["filter_cached_tokens"] = int(delta.get("cached_input_tokens", 0))
            info["filter_tokens_out"] = int(delta.get("output_tokens", 0))

        for k, v in extra.items():
            info[f"filter_{k}"] = v
        return info


# ==========================================
# 1. Single Agent Filter (단일 검증기)
# ==========================================
@register("filter", "SingleAgentFilter")
class SingleAgentFilter(BaseFilter):
    def __init__(self, model_name: str, temperature: float, api_key: str = None, base_url: str = None, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_manager = PromptManager()
        self.client = APIClient(api_key=api_key, base_url=base_url)
        logger.info(f"Initialized SingleAgentFilter (Model: {model_name})")

    def refine(self, query: str, subgraph: Dict[str, List[str]], **kwargs) -> Dict[str, Any]:
        t_start = time.perf_counter()
        token_before = AgentUtils.token_snapshot()
        ddl_schema = AgentUtils.generate_ddl(subgraph)

        prompt = self.prompt_manager.load_prompt(
            file_name='filter',
            section='single_agent_filter',
            query=query,
            schema_str=ddl_schema
        )

        response = self.client.generate_text(prompt=prompt, model=self.model_name, temperature=self.temperature)
        parsed = AgentUtils.extract_json(response)

        logger.debug(f"Reasoning: {parsed.get("step_by_step_reasoning", "")}")

        final_nodes = parsed.get("selected_nodes", []) or []
        status = "Answerable" if final_nodes else "Unanswerable"
        token_after = AgentUtils.token_snapshot()
        self.last_info = AgentUtils.build_filter_info(
            filter_type="SingleAgentFilter",
            input_subgraph=subgraph,
            final_nodes=final_nodes,
            status=status,
            token_before=token_before,
            token_after=token_after,
            t_start=t_start,
            model=self.model_name,
            parse_ok=bool(parsed.get("selected_nodes") is not None),
        )
        return {
            "status": status,
            "final_nodes": final_nodes,
            "reasoning": parsed.get("step_by_step_reasoning", ""),
            "filter_info": dict(self.last_info),
        }

# ==========================================
# 2. Adaptive Multi-Agent Filter
# ==========================================
@register("filter", "AdaptiveMultiAgentFilter")
class AdaptiveMultiAgentFilter(BaseFilter):
    def __init__(self, model_name: str, uncertainty_threshold: float = 0.6, api_key: str = None, base_url: str = None, **kwargs):
        self.model_name = model_name
        self.threshold = uncertainty_threshold
        self.prompt_manager = PromptManager()
        self.client = APIClient(api_key=api_key, base_url=base_url)
        logger.info(f"Initialized AdaptiveMultiAgentFilter (Threshold: {self.threshold})")

    def _call_agent(self, prompt: str) -> dict:
        response = self.client.generate_text(prompt=prompt, model=self.model_name, temperature=0.1)
        return AgentUtils.extract_json(response)

    def refine(self, query: str, subgraph: Dict[str, List[str]], **kwargs) -> Dict[str, Any]:
        t_start = time.perf_counter()
        token_before = AgentUtils.token_snapshot()
        ddl_schema = AgentUtils.generate_ddl(subgraph)

        # Phase 1: Semantic & Structural 평가
        semantic_prompt = self.prompt_manager.load_prompt(
            file_name='filter',
            section='semantic_agent',
            query=query,
            schema_str=ddl_schema
        )

        structural_prompt = self.prompt_manager.load_prompt(
            file_name='filter',
            section='structural_agent',
            query=query,
            schema_str=ddl_schema
        )

        semantic_res = self._call_agent(semantic_prompt)
        structural_res = self._call_agent(structural_prompt)

        set_a = set(semantic_res.get("selected_nodes", []))
        set_b = set(structural_res.get("selected_nodes", []))

        # Phase 2: Uncertainty 계산
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        uncertainty = 1.0 - (intersection / union) if union > 0 else 1.0

        logger.debug(f"[MultiAgent] Consensus Uncertainty: {uncertainty:.2f}")

        skeptic_triggered = False
        if uncertainty < self.threshold:
            final_nodes = list(set_a.union(set_b))
            status = "Answerable"
            reasoning = "Agents reached consensus."
        else:
            # Phase 3: Skeptic Agent
            logger.debug(f"[MultiAgent] High Uncertainty (> {self.threshold}). Triggering Skeptic Agent.")
            skeptic_triggered = True
            skeptic_prompt = self.prompt_manager.load_prompt(
                file_name='filter',
                section='skeptic_agent',
                query=query,
                schema_str=ddl_schema,
                agent_a=list(set_a),
                agent_b=list(set_b)
            )

            skeptic_res = self._call_agent(skeptic_prompt)
            decision = skeptic_res.get("final_decision", "Unanswerable")
            status = "Unanswerable" if decision == "Unanswerable" else "Answerable"
            final_nodes = [] if status == "Unanswerable" else decision
            reasoning = skeptic_res.get("step_by_step_reasoning", "")

        token_after = AgentUtils.token_snapshot()
        self.last_info = AgentUtils.build_filter_info(
            filter_type="AdaptiveMultiAgentFilter",
            input_subgraph=subgraph,
            final_nodes=final_nodes,
            status=status,
            token_before=token_before,
            token_after=token_after,
            t_start=t_start,
            model=self.model_name,
            uncertainty=float(uncertainty),
            semantic_agent_n=len(set_a),
            structural_agent_n=len(set_b),
            agent_intersection=intersection,
            agent_union=union,
            skeptic_triggered=skeptic_triggered,
            uncertainty_threshold=float(self.threshold),
        )
        return {
            "status": status,
            "uncertainty": uncertainty,
            "final_nodes": final_nodes,
            "reasoning": reasoning,
            "filter_info": dict(self.last_info),
        }

# ==========================================
# 3. None Filter (Pass-through)
# ==========================================
@register("filter", "None")
class NoneFilter(BaseFilter):
    def __init__(self, **kwargs):
        logger.info("Initialized None Filter (No LLM refinement)")

    def refine(self, query: str, subgraph: Dict[str, List[str]], **kwargs) -> Dict[str, Any]:
        t_start = time.perf_counter()
        # 서브그래프의 모든 밸류(컬럼)를 평탄화(Flatten)하여 반환
        all_nodes = []
        for table, cols in subgraph.items():
            all_nodes.extend([f"{table}.{col}" for col in cols])

        self.last_info = AgentUtils.build_filter_info(
            filter_type="NoneFilter",
            input_subgraph=subgraph,
            final_nodes=all_nodes,
            status="Answerable",
            token_before={"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0},
            token_after={"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0},
            t_start=t_start,
        )
        return {
            "status": "Answerable",
            "final_nodes": all_nodes,
            "reasoning": "Bypassed filtering stage.",
            "filter_info": dict(self.last_info),
        }