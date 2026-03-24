import json
import ast
import re
from typing import Dict, List, Any

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


# ==========================================
# 1. Single Agent Filter (단일 검증기)
# ==========================================
@register("filter", "SingleAgentFilter")
class SingleAgentFilter(BaseFilter):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct", temperature: float = 0.0, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_manager = PromptManager()
        self.client = APIClient(api_key="vllm", base_url="http://localhost:8000/v1")
        logger.info(f"Initialized SingleAgentFilter (Model: {model_name})")

    def refine(self, query: str, subgraph: Dict[str, List[str]], **kwargs) -> Dict[str, Any]:
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
        
        return {
            "status": "Answerable" if parsed.get("selected_nodes") else "Unanswerable",
            "final_nodes": parsed.get("selected_nodes", []),
            "reasoning": parsed.get("step_by_step_reasoning", "")
        }

# ==========================================
# 2. Adaptive Multi-Agent Filter
# ==========================================
@register("filter", "AdaptiveMultiAgentFilter")
class AdaptiveMultiAgentFilter(BaseFilter):
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", uncertainty_threshold: float = 0.6, **kwargs):
        self.model_name = model_name
        self.threshold = uncertainty_threshold
        self.prompt_manager = PromptManager()
        self.client = APIClient(api_key="vllm", base_url="http://localhost:8000/v1")
        logger.info(f"Initialized AdaptiveMultiAgentFilter (Threshold: {self.threshold})")

    def _call_agent(self, prompt: str) -> dict:
        response = self.client.generate_text(prompt=prompt, model=self.model_name, temperature=0.1)
        return AgentUtils.extract_json(response)

    def refine(self, query: str, subgraph: Dict[str, List[str]], **kwargs) -> Dict[str, Any]:
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

        if uncertainty < self.threshold:
            return {
                "status": "Answerable",
                "uncertainty": uncertainty,
                "final_nodes": list(set_a.union(set_b)),
                "reasoning": "Agents reached consensus."
            }

        # Phase 3: Skeptic Agent
        logger.debug(f"[MultiAgent] High Uncertainty (> {self.threshold}). Triggering Skeptic Agent.")
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

        return {
            "status": status,
            "uncertainty": uncertainty,
            "final_nodes": final_nodes,
            "reasoning": skeptic_res.get("step_by_step_reasoning", "")
        }

# ==========================================
# 3. None Filter (Pass-through)
# ==========================================
@register("filter", "None")
class NoneFilter(BaseFilter):
    def __init__(self, **kwargs):
        logger.info("Initialized None Filter (No LLM refinement)")

    def refine(self, query: str, subgraph: Dict[str, List[str]], **kwargs) -> Dict[str, Any]:
        # 서브그래프의 모든 밸류(컬럼)를 평탄화(Flatten)하여 반환
        all_nodes = []
        for table, cols in subgraph.items():
            all_nodes.extend([f"{table}.{col}" for col in cols])
            
        return {
            "status": "Answerable",
            "final_nodes": all_nodes,
            "reasoning": "Bypassed filtering stage."
        }