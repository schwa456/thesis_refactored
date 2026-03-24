from typing import Dict, List, Any
from modules.registry import register
from modules.base import BaseGenerator
from prompts.prompt_manager import PromptManager
from llm_client.api_handler import APIClient
from utils.logger import get_logger

logger = get_logger(__name__)

@register("generator", "LLMSQLGenerator")
class LLMSQLGenerator(BaseGenerator):
    def __init__(self, llm_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", temperature: float = 0.0, **kwargs):
        self.llm_model = llm_model
        self.temperature = temperature
        self.prompt_manager = PromptManager()
        self.client = APIClient(api_key="vllm", base_url="http://localhost:8000/v1")
        logger.info(f"Initialized LLMSQLGenerator (Model: {llm_model})")

    def generate(self, query: str, subgraph: Dict[str, List[str]], **kwargs) -> str:
        # 필터링이 끝난 아주 깨끗한 스키마만 DDL로 변환 (AgentUtils 재활용 또는 직접 구현)
        ddl_lines = []
        for table, columns in subgraph.items():
            cols_str = ",\n  ".join([f"{col}" for col in columns])
            ddl_lines.append(f"CREATE TABLE {table} (\n  {cols_str}\n);")
        schema_ddl = "\n\n".join(ddl_lines)

        prompt = self.prompt_manager.load_prompt(
            file_name="sql_generator",
            section="sql_generator",
            schema_str=schema_ddl,
            query=query
        )

        logger.debug(f"[Generation Prompt]: \n{prompt}")
        
        response = self.client.generate_text(prompt=prompt, model=self.llm_model, temperature=self.temperature)
        
        # Markdown 포맷 제거 (예방 차원)
        sql = response.replace("```sql", "").replace("```", "").strip()
        logger.debug(f"[Generated SQL] {sql}")
        return sql