"""F1. Reflection Filter — Self-Refine (propose → critique → revise).

Based on Self-Refine (Madaan et al., NeurIPS 2023) and Reflexion (Shinn et al.,
NeurIPS 2023). Unlike the single-pass XiYanFilter, this filter iteratively
self-critiques its own selection and may re-introduce nodes it dropped.
"""
import json
import sqlite3
import os
from typing import Dict, List, Any, Set, Tuple

from modules.registry import register
from modules.base import BaseFilter
from modules.filters.agents import AgentUtils
from llm_client.api_handler import APIClient
from prompts.prompt_manager import PromptManager
from utils.logger import get_logger

logger = get_logger(__name__)


@register("filter", "ReflectionFilter")
class ReflectionFilter(BaseFilter):
    def __init__(
        self,
        model_name: str,
        max_iteration: int = 1,
        temperature: float = 0.0,
        db_dir: str = "./data/raw/BIRD_dev/dev_databases",
        api_key: str = "vllm",
        base_url: str = "http://localhost:8000/v1",
        **kwargs,
    ):
        self.model_name = model_name
        self.max_iteration = max_iteration
        self.temperature = temperature
        self.db_dir = db_dir
        self.prompt_manager = PromptManager()
        self.client = APIClient(api_key=api_key, base_url=base_url)
        logger.info(
            f"Initialized ReflectionFilter (iterations={max_iteration}, model={model_name})"
        )

    def _schema_with_values(self, schema: Dict[str, List[str]], db_id: str) -> str:
        db_path = (
            os.path.join(self.db_dir, db_id, f"{db_id}.sqlite") if db_id else None
        )
        conn = None
        cursor = None
        if db_path and os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
            except Exception as e:
                logger.warning(f"DB connect failed for {db_id}: {e}")

        lines = []
        for tbl, cols in schema.items():
            col_parts = []
            for col in cols:
                s = col
                if cursor:
                    try:
                        cursor.execute(
                            f'SELECT DISTINCT "{col}" FROM "{tbl}" '
                            f'WHERE "{col}" IS NOT NULL LIMIT 3'
                        )
                        vals = [str(r[0]) for r in cursor.fetchall()]
                        if vals:
                            s += f" (Examples: {', '.join(vals)})"
                    except Exception:
                        pass
                col_parts.append(s)
            lines.append(f"# Table: {tbl}\n  Columns: " + " | ".join(col_parts))
        if conn:
            conn.close()
        return "\n".join(lines)

    def _flatten(self, schema: Dict[str, List[str]]) -> List[str]:
        return [f"{t}.{c}" for t, cols in schema.items() for c in cols]

    def _parse_schema_json(self, response: str) -> Dict[str, List[str]]:
        clean = response.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return {}
        try:
            obj = json.loads(clean[start : end + 1])
            if isinstance(obj, dict):
                return {str(k): list(v) for k, v in obj.items() if isinstance(v, list)}
        except Exception as e:
            logger.warning(f"[ReflectionFilter] JSON parse failed: {e}")
        return {}

    def _propose(
        self, query: str, schema: Dict[str, List[str]], db_id: str
    ) -> Dict[str, List[str]]:
        schema_str = self._schema_with_values(schema, db_id)
        example_tables = list(schema.keys())[:2]
        example_obj = {
            t: schema[t][: (2 if i == 0 else 1)] for i, t in enumerate(example_tables)
        }
        prompt = self.prompt_manager.load_prompt(
            file_name="filter",
            section="xiyan_filter",
            schema_str=schema_str,
            query=query,
            example_json_str=json.dumps(example_obj),
        )
        resp = self.client.generate_text(
            prompt=prompt, model=self.model_name, temperature=self.temperature
        )
        parsed = self._parse_schema_json(resp)
        return parsed if parsed else schema

    def _critique(
        self,
        query: str,
        full_schema: Dict[str, List[str]],
        current: Dict[str, List[str]],
        db_id: str,
    ) -> Dict[str, Any]:
        full_str = self._schema_with_values(full_schema, db_id)
        current_str = "\n".join(
            f"{t}.{c}" for t, cols in current.items() for c in cols
        )
        prompt = self.prompt_manager.load_prompt(
            file_name="filter",
            section="reflection_critique",
            query=query,
            full_schema_str=full_str,
            current_selection=current_str,
        )
        resp = self.client.generate_text(
            prompt=prompt, model=self.model_name, temperature=self.temperature
        )
        return AgentUtils.extract_json(resp)

    def _revise(
        self,
        query: str,
        full_schema: Dict[str, List[str]],
        current: Dict[str, List[str]],
        critique_text: str,
        db_id: str,
    ) -> Dict[str, List[str]]:
        full_str = self._schema_with_values(full_schema, db_id)
        current_str = "\n".join(
            f"{t}.{c}" for t, cols in current.items() for c in cols
        )
        prompt = self.prompt_manager.load_prompt(
            file_name="filter",
            section="reflection_revise",
            query=query,
            full_schema_str=full_str,
            current_selection=current_str,
            critique=critique_text,
        )
        resp = self.client.generate_text(
            prompt=prompt, model=self.model_name, temperature=self.temperature
        )
        parsed = self._parse_schema_json(resp)
        return parsed if parsed else current

    def refine(
        self,
        query: str,
        subgraph: Dict[str, List[str]],
        db_id: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if not subgraph:
            return {
                "status": "Unanswerable",
                "final_nodes": [],
                "reasoning": "Empty input subgraph",
            }

        current = self._propose(query, subgraph, db_id)
        reasoning_trace: List[str] = [f"Propose -> {len(self._flatten(current))} nodes"]

        for it in range(self.max_iteration):
            critique = self._critique(query, subgraph, current, db_id)
            verdict = str(critique.get("verdict", "")).lower()
            critique_text = critique.get("critique", "") or critique.get(
                "step_by_step_reasoning", ""
            )
            reasoning_trace.append(f"Iter{it+1} verdict={verdict}")

            if verdict.startswith("sufficient"):
                break

            revised = self._revise(query, subgraph, current, critique_text, db_id)
            if revised:
                current = revised

        final_nodes = self._flatten(current)
        return {
            "status": "Answerable" if final_nodes else "Unanswerable",
            "final_nodes": final_nodes,
            "reasoning": " | ".join(reasoning_trace),
        }
