"""M5 Two-Stage Filter (학술 agent §7, Wave 6 Phase 2 (a+aggressive)).

DECISIONS 2026-05-16 (a+aggressive) launch entry §2 정합:
  - Stage 1 = two_stage_stage1 (Recall-First Coarse Pre-filter, 4-rule conjunctive
    exclusion = M1-C 변형)
  - Stage 2 = two_stage_stage2 (Precision-Second Fine-filter, Stage 1 output 을
    schema input 으로)
  - 2 sequential LLM call/query
  - 측정: stage1_only / two_stage final + stage2_recall_loss / stage2_precision_gain

학술 frame: Sequential Recall→Precision (§V.5.x.M.3 production deployment 추가
candidate).
"""
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from modules.registry import register
from modules.base import BaseFilter
from modules.filters.agents import AgentUtils
from modules.filters.xiyan_filter import XiYanFilter
from prompts.prompt_manager import PromptManager
from utils.logger import get_logger

logger = get_logger(__name__)


@register("filter", "TwoStageFilter")
class TwoStageFilter(BaseFilter):
    """M5 Sequential Recall-First → Precision-Second (학술 agent §7)."""

    def __init__(
        self,
        model_name: str,
        max_iteration: int = 1,
        temperature: float = 0.0,
        db_dir: str = "./data/raw/BIRD_dev/dev_databases",
        num_examples: int = 3,
        sanitize_output: bool = True,
        stage1_section: str = "two_stage_stage1",
        stage2_section: str = "two_stage_stage2",
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
        self.stage1_section = stage1_section
        self.stage2_section = stage2_section
        self.prompt_manager = PromptManager()
        self.client = self._make_llm_client(
            api_key=api_key, base_url=base_url, provider=provider,
        )
        logger.info(
            "Initialized TwoStageFilter "
            f"(model={model_name}, stage1={stage1_section}, "
            f"stage2={stage2_section}, sanitize={self.sanitize_output})"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_schema_str(self, subgraph: Dict[str, List[str]], db_id: Optional[str]) -> str:
        helper = XiYanFilter.__new__(XiYanFilter)
        helper.db_dir = self.db_dir
        helper.num_examples = self.num_examples
        return helper._build_mschema_with_values(subgraph, db_id or "")

    def _format_stage1_for_stage2(
        self,
        stage1_result: Dict[str, List[str]],
        db_id: Optional[str],
    ) -> str:
        """Stage 1 JSON → Stage 2 schema_str 변환 (학술 agent §7.2).

        Stage 1 output (filtered subgraph) 의 example values 는 db 에서 다시 fetch
        (num_examples 가 > 0 일 때 만).
        """
        lines: List[str] = []
        db_path = (
            os.path.join(self.db_dir, db_id, f"{db_id}.sqlite") if db_id else None
        )
        conn = None
        if db_path and os.path.exists(db_path) and self.num_examples > 0:
            try:
                conn = sqlite3.connect(db_path)
            except Exception:
                conn = None
        for table, cols in stage1_result.items():
            lines.append(f"Table: {table}")
            for col in cols:
                samples_str = ""
                if conn is not None:
                    try:
                        cur = conn.cursor()
                        cur.execute(
                            f'SELECT DISTINCT "{col}" FROM "{table}" '
                            f'WHERE "{col}" IS NOT NULL LIMIT {self.num_examples}'
                        )
                        samples = [str(r[0]) for r in cur.fetchall()]
                        if samples:
                            samples_str = f" | Examples: {', '.join(samples)}"
                    except Exception:
                        pass
                lines.append(f"  Column: {col}{samples_str}")
            lines.append("")
        if conn is not None:
            conn.close()
        return "\n".join(lines)

    def _call_stage(
        self,
        section: str,
        **kwargs_for_template,
    ) -> str:
        prompt = self.prompt_manager.load_prompt(
            file_name='filter', section=section, **kwargs_for_template,
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
                filter_type="TwoStageFilter",
                input_subgraph={}, final_nodes=[], status="Unanswerable",
                token_before=empty_tok, token_after=empty_tok, t_start=t_start,
                model=self.model_name,
                stage1_section=self.stage1_section,
                stage2_section=self.stage2_section,
                stage1_count=0, stage2_count=0,
                stage2_removed_count=0,
                hallucination_removed_stage1=0,
                hallucination_removed_stage2=0,
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

        # Stage 1 — Recall-First Coarse
        try:
            raw_s1 = self._call_stage(
                self.stage1_section,
                schema_str=schema_str, query=query,
                example_json_str=example_json_str,
            )
        except Exception as e:
            logger.warning(f"[M5 stage1] LLM call failed: {e}")
            raw_s1 = ""
        s1_parsed = self._parse_json_dict(raw_s1)
        halluc_s1 = 0
        if self.sanitize_output:
            s1_clean, halluc_s1 = XiYanFilter.sanitize_filter_output(s1_parsed, subgraph)
        else:
            s1_clean = s1_parsed
        stage1_nodes = sorted(
            f"{t}.{c}" for t, cols in s1_clean.items() for c in cols
        )

        # Stage 2 — Precision-Second Fine
        if s1_clean:
            stage1_schema_str = self._format_stage1_for_stage2(s1_clean, db_id)
            try:
                raw_s2 = self._call_stage(
                    self.stage2_section,
                    stage1_schema_str=stage1_schema_str, query=query,
                    example_json_str=example_json_str,
                )
            except Exception as e:
                logger.warning(f"[M5 stage2] LLM call failed: {e}")
                raw_s2 = ""
            s2_parsed = self._parse_json_dict(raw_s2)
            # 학술 agent §7.3: Stage 2 sanitize 는 Stage 1 output 기준 (subgraph 아님)
            halluc_s2 = 0
            if self.sanitize_output:
                s2_clean, halluc_s2 = XiYanFilter.sanitize_filter_output(s2_parsed, s1_clean)
            else:
                s2_clean = s2_parsed
        else:
            # Stage 1 empty → Stage 2 skip, final empty
            s2_clean = {}
            halluc_s2 = 0
            raw_s2 = ""

        stage2_nodes = sorted(
            f"{t}.{c}" for t, cols in s2_clean.items() for c in cols
        )

        final_nodes = stage2_nodes
        status = "Answerable" if final_nodes else "Unanswerable"

        stage1_count = len(stage1_nodes)
        stage2_count = len(stage2_nodes)
        stage2_removed = max(0, stage1_count - stage2_count)
        token_after = AgentUtils.token_snapshot()

        self.last_info = AgentUtils.build_filter_info(
            filter_type="TwoStageFilter",
            input_subgraph=subgraph, final_nodes=final_nodes, status=status,
            token_before=token_before, token_after=token_after, t_start=t_start,
            model=self.model_name,
            stage1_section=self.stage1_section,
            stage2_section=self.stage2_section,
            sanitize_output=self.sanitize_output,
            stage1_count=stage1_count,
            stage2_count=stage2_count,
            stage2_removed_count=stage2_removed,
            hallucination_removed_stage1=halluc_s1,
            hallucination_removed_stage2=halluc_s2,
            n_input_columns=sum(len(v) for v in subgraph.values()),
        )
        return {
            "status": status, "final_nodes": final_nodes,
            "reasoning": (
                f"[M5 two-stage] stage1={stage1_count} → stage2={stage2_count} "
                f"(removed {stage2_removed} in fine-filter; "
                f"hallucinated removed: stage1={halluc_s1}, stage2={halluc_s2})"
            ),
            "stats": {
                "stage1_count": stage1_count,
                "stage2_count": stage2_count,
                "stage2_removed_count": stage2_removed,
                "stage1_nodes": stage1_nodes,
                "stage2_nodes": stage2_nodes,
                "hallucination_removed_stage1": halluc_s1,
                "hallucination_removed_stage2": halluc_s2,
            },
            "filter_info": dict(self.last_info),
        }
