import json
import ast
import re
from typing import Dict, Any, List

from modules.registry import register
from modules.base import BaseSelector
from prompts.prompt_manager import PromptManager
from llm_client.api_handler import APIClient
from utils.logger import get_logger

logger = get_logger(__name__)

@register("selector", "AgentNodeSelector")
class AgentNodeSelector(BaseSelector):
    """ LLM Agent 기반 Seed 선택기, Uncertainty를 PCST의 Prize로 활용 """
    def __init__(self, model_name: str, temperature: float = 0.0, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.prompt_manager = PromptManager()
        self.client = APIClient() 
        logger.info(f"Initialized AgentNodeSelector with model: [{self.model_name}]")

    def select(self, question: str, candidates: List[str], **kwargs) -> Dict[str, float]:
        if not question or not candidates:
            raise ValueError("AgentNodeSelector requires 'question' and 'candidates'.")
        
        prompt = self._construct_prompt(question, candidates)
        logger.debug(f"[AgentSelector] Prompt Preview: {prompt[:300]}...")

        # 💡 통일된 클라이언트 호출
        response = self.client.generate_text(
            prompt=prompt, 
            model=self.model_name, 
            temperature=self.temperature
        )

        logger.debug(f"\n[DEBUG] >>> Agent Raw Output:\n{response}\n{'-'*50}")

        try:
            parsed_result = self._parse_json_response(response)
        except Exception as e:
            logger.error(f"[ERROR] JSON Parsing Failed: {e}")
            return {}
        
        if not parsed_result.get('is_answerable', True):
            logger.info("[AgentSelector] Agent decided the question is UNANSWERABLE.")
            return {}
        
        weighted_seeds = parsed_result.get('selected_items', parsed_result.get('selected_tables', {}))

        final_seeds = {}
        for k, v in weighted_seeds.items():
            try:
                # 점수 정규화 (0.0 ~ 1.0)
                final_seeds[k] = min(max(float(v), 0.0), 1.0)
            except ValueError:
                continue
        
        if not final_seeds:
            logger.warning("[AgentSelector] Parsed JSON has no valid 'selected_items'.")

        metadata = kwargs.get('metadata', None)
        if metadata and 'node_metadata' in metadata:
            total_nodes = len(metadata['node_metadata'])
            all_scores = [0.0] * total_nodes
            
            # 텍스트 형태의 키(테이블.컬럼)를 인덱스로 변환하여 점수 기록
            for cand_text, score in final_seeds.items():
                for idx, name in metadata['node_metadata'].items():
                    if name == cand_text:
                        all_scores[int(idx)] = score
                        break
            
            self.latest_scores = all_scores
        else:
            self.latest_scores = []

        return final_seeds
    
    def _construct_prompt(self, question: str, candidates: List[str]) -> str:
        return self.prompt_manager.load_prompt(
            file_name='selector',
            section='single_agent_selector',
            question=question,
            candidates=candidates
        )
    
    def _parse_json_response(self, response: str) -> dict:
        """
        LLM 응답에서 JSON을 추출하고 파싱합니다. (기존 로직 완벽 유지)
        """
        text_to_parse = response.strip()

        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            text_to_parse = json_match.group(1)
        else:
            json_match = re.search(r"(\{.*\})", response, re.DOTALL)
            if json_match:
                text_to_parse = json_match.group(1)

        try:
            return json.loads(text_to_parse)
        except json.JSONDecodeError:
            pass 

        py_compatible_text = (
            text_to_parse
            .replace("true", "True")
            .replace("false", "False")
            .replace("null", "None")
        )
        
        try:
            return ast.literal_eval(py_compatible_text)
        except Exception:
            raise ValueError(f"Failed to parse JSON content: {text_to_parse[:50]}...")