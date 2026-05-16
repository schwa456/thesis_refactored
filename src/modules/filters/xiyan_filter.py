import os
import re
import json
import sqlite3
import time
from typing import Dict, List, Any, Optional, Tuple

from modules.registry import register
from modules.base import BaseFilter
from modules.filters.agents import AgentUtils
from prompts.prompt_manager import PromptManager
from utils.logger import get_logger

logger = get_logger(__name__)

# Wave 6 Phase 1 (학술 agent filter improve plan §3, DECISIONS 2026-05-16):
# M1 Recall-Biased Prompt 3 variants + default. PromptManager section 매핑.
_PROMPT_SECTION_BY_MODE: Dict[str, str] = {
    "default": "xiyan_filter",
    "recall_biased_mild": "recall_biased_mild",
    "recall_biased_strong": "recall_biased_strong",
    "recall_biased_exclusion_rule": "recall_biased_exclusion_rule",
}


@register("filter", "XiYanFilter")
class XiYanFilter(BaseFilter):
    """
    XiYanSQL의 Iterative Column Selection을 모사한 Filter 모듈.
    LLM 프롬프트 생성 시, 실제 DB에 쿼리를 날려 컬럼별 최대 3개의 예시 데이터(Example Values)를 삽입합니다.

    Wave 6 Phase 1 (DECISIONS 2026-05-16 §2 + 학술 agent plan §3):
    - prompt_mode ∈ {default, recall_biased_mild, recall_biased_strong,
                      recall_biased_exclusion_rule}
    - sanitize_output (default True): 학술 agent §2.3 hallucination 방지 후처리 —
      LLM 출력에서 input subgraph (extractor output) 에 없는 table/column 제거
    - 측정 메타: filter_prompt_mode / filter_prune_pct / filter_hallucination_removed_count
    """
    def __init__(self, model_name: str, max_iteration: int = 1, temperature: float = 0.0, db_dir: str = "./data/raw/BIRD_dev/dev_databases", api_key: Optional[str] = None, base_url: Optional[str] = None, provider: Optional[str] = None, num_examples: int = 3, prompt_mode: str = "default", sanitize_output: bool = True, **kwargs):
        if prompt_mode not in _PROMPT_SECTION_BY_MODE:
            raise ValueError(
                f"prompt_mode='{prompt_mode}' invalid. "
                f"Expected one of {sorted(_PROMPT_SECTION_BY_MODE.keys())}."
            )
        self.model_name = model_name
        self.max_iteration = max_iteration
        self.temperature = temperature
        self.db_dir = db_dir
        self.provider = provider
        # num_examples: column 별 example value 개수 (0 = 비활성, default 3)
        self.num_examples = max(0, int(num_examples))
        self.prompt_mode = prompt_mode
        self._prompt_section = _PROMPT_SECTION_BY_MODE[prompt_mode]
        self.sanitize_output = bool(sanitize_output)
        self.prompt_manager = PromptManager()
        self.client = self._make_llm_client(api_key=api_key, base_url=base_url, provider=provider)
        logger.info(
            f"Initialized XiYanFilter (Iterations: {self.max_iteration}, "
            f"num_examples: {self.num_examples}, Provider: {provider or 'auto'}, "
            f"prompt_mode: {self.prompt_mode}, sanitize: {self.sanitize_output})"
        )

    # ------------------------------------------------------------------
    # Wave 6 §2.3 Hallucination 방지 후처리
    # ------------------------------------------------------------------
    @staticmethod
    def sanitize_filter_output(
        raw_output: Dict[str, List[str]],
        extractor_output: Dict[str, List[str]],
    ) -> Tuple[Dict[str, List[str]], int]:
        """LLM 출력에서 extractor_output 에 없는 table/column 제거.

        학술 agent plan §2.3 정합. raw_output 의 각 (table, col) 이
        extractor_output 의 schema 에 있는지 확인하고 없으면 제거.

        Returns:
            (sanitized_dict, removed_count)
              - sanitized_dict: {table: [col, ...]} (제거 후, 빈 table 도 제거)
              - removed_count: raw_output 의 (table, col) 중 제거된 entry 수
                              (table 단위 제거도 그 안의 col 수 만큼 계산)
        """
        if not isinstance(raw_output, dict):
            return {}, 0
        ext_keys: Dict[str, set] = {
            t: set(cols or []) for t, cols in (extractor_output or {}).items()
        }
        sanitized: Dict[str, List[str]] = {}
        removed = 0
        for table, cols in raw_output.items():
            if not isinstance(cols, list):
                # LLM 이 list 가 아닌 형식으로 반환한 경우 (dict 등) — 안전하게 skip
                removed += 1
                continue
            if table not in ext_keys:
                # 통째로 hallucinate 한 table — col 수만큼 removed 가산
                removed += len(cols)
                continue
            kept: List[str] = []
            for col in cols:
                if not isinstance(col, str):
                    removed += 1
                    continue
                if col in ext_keys[table]:
                    if col not in kept:
                        kept.append(col)
                else:
                    removed += 1
            if kept:
                sanitized[table] = kept
        return sanitized, removed

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
                # DB에서 Example Value 추출 시도 (num_examples=0 시 skip)
                if cursor and self.num_examples > 0:
                    try:
                        # SQL Injection 및 구문 오류 방지를 위해 쌍따옴표(")로 식별자 감싸기
                        cursor.execute(f'SELECT DISTINCT "{col}" FROM "{tbl}" WHERE "{col}" IS NOT NULL LIMIT {self.num_examples}')
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
        t_start = time.perf_counter()
        if not subgraph:
            self.last_info = AgentUtils.build_filter_info(
                filter_type="XiYanFilter",
                input_subgraph=subgraph or {},
                final_nodes=[],
                status="Unanswerable",
                token_before={"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0},
                token_after={"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0},
                t_start=t_start,
                model=self.model_name,
                iterations_run=0,
                prompt_mode=self.prompt_mode,
                sanitize_output=self.sanitize_output,
                hallucination_removed_count=0,
                prune_pct=0.0,
            )
            return {"status": "Unanswerable", "final_nodes": [], "reasoning": "Empty input subgraph", "filter_info": dict(self.last_info)}

        token_before = AgentUtils.token_snapshot()
        current_schema = subgraph.copy()
        iterations_run = 0
        parse_errors = 0
        iteration_trace: List[Dict[str, Any]] = []
        hallucination_removed_total = 0
        input_node_count = sum(len(v) for v in subgraph.values())

        for i in range(self.max_iteration):
            iterations_run += 1
            logger.debug(f"[XiYanFilter] Iteration {i+1}/{self.max_iteration}")
            cols_before = sum(len(v) for v in current_schema.values())

            # 1. M-Schema (Value 포함) 포맷팅
            schema_str = self._build_mschema_with_values(current_schema, db_id)

            # 2. Few-shot JSON 예시 동적 구성
            example_tables = list(current_schema.keys())[:2]
            example_json_obj = {}
            for idx, t in enumerate(example_tables):
                example_json_obj[t] = current_schema[t][: (2 if idx == 0 else 1)]
            example_json_str = json.dumps(example_json_obj)

            # 3. LLM 프롬프트 텍스트 구성 — Wave 6 prompt_mode 별 section 분기
            prompt = self.prompt_manager.load_prompt(
                file_name='filter',
                section=self._prompt_section,
                schema_str=schema_str,
                query=query,
                example_json_str=example_json_str
            )

            response = self.client.generate_text(prompt=prompt, model=self.model_name, temperature=self.temperature)
            logger.debug(f"[XiYanFilter] LLM Response: {response}")

            # 4. JSON 파싱 및 정제
            iter_parse_ok = False
            stop_iteration = False
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
                        # Wave 6 §2.3 hallucination 방지 후처리 (default-on, recall_biased
                        # 모드 활성 시에도 자동 적용 — sanitize_output=True default)
                        if self.sanitize_output:
                            sanitized_schema, removed = self.sanitize_filter_output(
                                selected_columns, subgraph,
                            )
                            if removed > 0:
                                logger.debug(
                                    f"[XiYanFilter] sanitize removed {removed} "
                                    f"hallucinated entries (mode={self.prompt_mode})."
                                )
                            hallucination_removed_total += int(removed)
                            current_schema = sanitized_schema
                        else:
                            current_schema = selected_columns
                        logger.debug(f"[XiYanFilter] 파싱 성공. 추출된 테이블 수: {len(current_schema)}")
                        iter_parse_ok = True
                    else:
                        logger.warning(f"[XiYanFilter] 파싱된 JSON이 Dict 형식이 아닙니다 (타입: {type(selected_columns)}).")
                        parse_errors += 1
                        stop_iteration = True
                else:
                    raise ValueError("응답 텍스트 내에 유효한 중괄호 '{ ... }' 쌍이 존재하지 않습니다.")

            except json.JSONDecodeError as e:
                # LLM이 따옴표를 빼먹거나 후행 쉼표(Trailing comma)를 넣는 등의 문법 오류를 낸 경우
                logger.warning(f"[XiYanFilter] JSON 디코딩 에러 발생: {e}. 추출 시도된 텍스트: {json_str[:100]}...")
                parse_errors += 1
                stop_iteration = True
            except Exception as e:
                logger.warning(f"[XiYanFilter] 파싱 중 예기치 않은 예외 발생: {e}")
                parse_errors += 1
                stop_iteration = True

            cols_after = sum(len(v) for v in current_schema.values())
            iteration_trace.append({
                "iteration": i + 1,
                "cols_before": cols_before,
                "cols_after": cols_after,
                "cols_removed": cols_before - cols_after,
                "parse_ok": iter_parse_ok,
            })
            if stop_iteration:
                break

        # 5. 파이프라인 규격(Flat list)으로 노드 변환
        final_nodes = []
        for table, cols in current_schema.items():
            for col in cols:
                final_nodes.append(f"{table}.{col}")

        status = "Answerable" if final_nodes else "Unanswerable"
        token_after = AgentUtils.token_snapshot()
        # Wave 6 §2.1 측정 — Prune% 직접 계산 (R_fil/P_fil/F1_fil/FNR/FPR 는 evaluate
        # step 에서 gold 와 합산 계산 필요 — root 책임)
        output_node_count = len(final_nodes)
        prune_pct = (
            (input_node_count - output_node_count) / input_node_count
            if input_node_count > 0 else 0.0
        )
        self.last_info = AgentUtils.build_filter_info(
            filter_type="XiYanFilter",
            input_subgraph=subgraph,
            final_nodes=final_nodes,
            status=status,
            token_before=token_before,
            token_after=token_after,
            t_start=t_start,
            model=self.model_name,
            max_iteration=self.max_iteration,
            iterations_run=iterations_run,
            parse_errors=parse_errors,
            iteration_trace=iteration_trace,
            temperature=float(self.temperature),
            prompt_mode=self.prompt_mode,
            sanitize_output=self.sanitize_output,
            hallucination_removed_count=int(hallucination_removed_total),
            prune_pct=float(prune_pct),
            input_node_count=int(input_node_count),
            output_node_count=int(output_node_count),
        )
        return {
            "status": status,
            "final_nodes": final_nodes,
            "reasoning": (
                f"Filtered with DB Value Examples "
                f"(iterations={self.max_iteration}, mode={self.prompt_mode}, "
                f"sanitize={self.sanitize_output}, hallucination_removed="
                f"{hallucination_removed_total}, prune%={prune_pct:.3f})"
            ),
            "filter_info": dict(self.last_info),
        }