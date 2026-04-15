from typing import Dict, List, Any
from modules.registry import register
from modules.base import BaseGenerator
from prompts.prompt_manager import PromptManager
from llm_client.api_handler import APIClient
from modules.filters.agents import AgentUtils
from utils.mschema.schema_engine import MSchema
from utils.logger import get_logger

logger = get_logger(__name__)

@register("generator", "LLMSQLGenerator")
class LLMSQLGenerator(BaseGenerator):
    def __init__(self, llm_model: str, temperature: float, **kwargs):
        self.llm_model = llm_model
        self.temperature = temperature
        self.prompt_manager = PromptManager()
        self.client = APIClient()
        logger.info(f"Initialized LLMSQLGenerator (Model: {llm_model})")

    def generate(self, query: str, subgraph: Dict[str, List[str]], **kwargs) -> str:
        schema_ddl = AgentUtils.generate_ddl(subgraph=subgraph)

        selected_tables = list(subgraph.keys())
        selected_columns = []
        for tbl in selected_tables:
            for col in subgraph[tbl]:
                col_name = '.'.join([tbl, col])
                selected_columns.append(col_name)

        #mschema = MSchema()
        #mschema_str = MSchema.to_mschema(selected_tables=selected_tables, selected_columns=selected_columns, example_num=3, show_type_detail=True)

        prompt = self.prompt_manager.load_prompt(
            file_name="sql_generator",
            section="sql_generator",
            schema_str=schema_ddl,
            # schema_str=mschema_str,
            query=query
        )

        logger.debug(f"[Generation Prompt]: \n{prompt}")
        
        response = self.client.generate_text(prompt=prompt, model=self.llm_model, temperature=self.temperature)
        
        # Markdown 포맷 제거 (예방 차원)
        sql = response.replace("```sql", "").replace("```", "").strip()
        logger.debug(f"[Generated SQL] {sql}")
        return sql