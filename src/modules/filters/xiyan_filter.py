import os
import re
import json
import sqlite3
from typing import Dict, List, Any

from modules.registry import register
from modules.base import BaseFilter
from llm_client.api_handler import APIClient
from prompts.prompt_manager import PromptManager
from utils.logger import get_logger

logger = get_logger(__name__)

@register("filter", "XiYanFilter")
class XiYanFilter(BaseFilter):
    """
    XiYanSQL의 Iterative Column Selection을 모사한 Filter 모듈.
    LLM 프롬프트 생성 시, 실제 DB에 쿼리를 날려 컬럼별 최대 3개의 예시 데이터(Example Values)를 삽입합니다.
    """
    def __init__(self, model_name: str, max_iteration: int = 1, temperature: float = 0.0, db_dir: str = "./data/raw/BIRD_dev/dev_databases", api_key: str = "vllm", base_url: str = "http://localhost:8000/v1", **kwargs):
        self.model_name = model_name
        self.max_iteration = max_iteration
        self.temperature = temperature
        self.db_dir = db_dir
        self.prompt_manager = PromptManager()
        self.client = APIClient(api_key=api_key, base_url=base_url)
        logger.info(f"Initialized XiYanFilter with DB Value Example Injection (Iterations: {self.max_iteration})")

    def _build_mschema_with_values(self, schema_dict: Dict[str, List[str]], db_id: str) -> str:
        """DB에 직접 접근하여 컬럼별 Example Value를 포함한 M-Schema 문자열을 생성합니다."""
        schema_lines = []
        db_path = os.path.join(self.db_dir, db_id, f"{db_id}.sqlite") if db_id else None
        
        conn = None
        cursor = None
        if db_path and os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
            except Exception as e:
                logger.warning(f"Failed to connect to DB {db_id} for M-Schema generation: {e}")

        for tbl, cols in schema_dict.items():
            table_lines = [f"# Table: {tbl}"]
            col_details = []
            for col in cols:
                col_str = f"{col}"
                # DB에서 Example Value 추출 시도
                if cursor:
                    try:
                        # SQL Injection 및 구문 오류 방지를 위해 쌍따옴표(")로 식별자 감싸기
                        cursor.execute(f'SELECT DISTINCT "{col}" FROM "{tbl}" WHERE "{col}" IS NOT NULL LIMIT 3')
                        values = [str(row[0]) for row in cursor.fetchall()]
                        if values:
                            col_str += f" (Examples: {', '.join(values)})"
                    except Exception as e:
                        logger.debug(f"Could not fetch examples for {tbl}.{col}: {e}")
                
                col_details.append(col_str)
            table_lines.append("  Columns: " + " | ".join(col_details))
            schema_lines.append("\n".join(table_lines))
        
        if conn:
            conn.close()
            
        return "\n".join(schema_lines)

    def refine(self, query: str, subgraph: Dict[str, List[str]], db_id: str = None, **kwargs) -> Dict[str, Any]:
        if not subgraph:
            return {"status": "Unanswerable", "final_nodes": [], "reasoning": "Empty input subgraph"}

        current_schema = subgraph.copy()
        
        for i in range(self.max_iteration):
            logger.debug(f"[XiYanFilter] Iteration {i+1}/{self.max_iteration}")
            
            # 1. M-Schema (Value 포함) 포맷팅
            schema_str = self._build_mschema_with_values(current_schema, db_id)
            
            # 2. Few-shot JSON 예시 동적 구성
            example_tables = list(current_schema.keys())[:2]
            example_json_obj = {}
            for idx, t in enumerate(example_tables):
                example_json_obj[t] = current_schema[t][: (2 if idx == 0 else 1)]
            example_json_str = json.dumps(example_json_obj)

            # 3. LLM 프롬프트 텍스트 구성
            prompt = self.prompt_manager.load_prompt(
                file_name='filter',
                section='xiyan_filter',
                schema_str=schema_str,
                query=query,
                example_json_str=example_json_str
            )
            
            response = self.client.generate_text(prompt=prompt, model=self.model_name, temperature=self.temperature)
            logger.debug(f"[XiYanFilter] LLM Response: {response}")
            
            # 4. JSON 파싱 및 정제
            try:
                # [단계 A] 마크다운 포맷팅 강제 제거
                # LLM이 습관적으로 붙이는 코드 블록 마커를 텍스트에서 완전히 지웁니다.
                clean_response = response.replace("```json", "").replace("```", "").strip()
                
                # [단계 B] 첫 번째 '{' 와 마지막 '}' 위치 추적
                # 정규식의 탐욕적 매칭 오류를 방지하고 최상위 JSON 객체를 정확히 도출합니다.
                start_idx = clean_response.find('{')
                end_idx = clean_response.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    # 인덱스를 기반으로 JSON 문자열만 슬라이싱
                    json_str = clean_response[start_idx : end_idx + 1]
                    
                    # [단계 C] JSON 로드 및 타입 검증
                    selected_columns = json.loads(json_str)
                    
                    if isinstance(selected_columns, dict):
                        current_schema = selected_columns 
                        logger.debug(f"[XiYanFilter] 파싱 성공. 추출된 테이블 수: {len(current_schema)}")
                    else:
                        logger.warning(f"[XiYanFilter] 파싱된 JSON이 Dict 형식이 아닙니다 (타입: {type(selected_columns)}).")
                        break  # 포맷 위반 시 Iteration 중단 후 이전 스키마(Fallback) 사용
                else:
                    raise ValueError("응답 텍스트 내에 유효한 중괄호 '{ ... }' 쌍이 존재하지 않습니다.")
                    
            except json.JSONDecodeError as e:
                # LLM이 따옴표를 빼먹거나 후행 쉼표(Trailing comma)를 넣는 등의 문법 오류를 낸 경우
                logger.warning(f"[XiYanFilter] JSON 디코딩 에러 발생: {e}. 추출 시도된 텍스트: {json_str[:100]}...")
                break
            except Exception as e:
                logger.warning(f"[XiYanFilter] 파싱 중 예기치 않은 예외 발생: {e}")
                break

        # 5. 파이프라인 규격(Flat list)으로 노드 변환
        final_nodes = []
        for table, cols in current_schema.items():
            for col in cols:
                final_nodes.append(f"{table}.{col}")

        return {
            "status": "Answerable" if final_nodes else "Unanswerable",
            "final_nodes": final_nodes,
            "reasoning": f"Filtered with DB Value Examples (iterations={self.max_iteration})"
        }