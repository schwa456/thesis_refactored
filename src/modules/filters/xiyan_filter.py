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
from utils.schema_serializer import (
    VALID_FK_HINT_FORMATS,
    VALID_FK_HINT_POSITIONS,
    serialize_schema_with_fk_hints,
)

logger = get_logger(__name__)

# Wave 6 Phase 1 (학술 agent filter improve plan §3, DECISIONS 2026-05-16):
# M1 Recall-Biased Prompt 3 variants + default. PromptManager section 매핑.
_PROMPT_SECTION_BY_MODE: Dict[str, str] = {
    "default": "xiyan_filter",
    "recall_biased_mild": "recall_biased_mild",
    "recall_biased_strong": "recall_biased_strong",
    "recall_biased_exclusion_rule": "recall_biased_exclusion_rule",
}

# Wave 6 Phase 2 (a) (학술 agent §4, DECISIONS 2026-05-16 §2 Phase 2 (a) Spec):
# cot_reasoning=True 시 prompt_mode → CoT section 매핑. Phase 2 (a) 핵심 spec 인
# recall_biased_strong+CoT 결합 우선 구현. 다른 mode 와의 CoT 결합은 향후 Phase
# 2 (b)/(c) trigger 시 추가.
_COT_PROMPT_SECTION_BY_MODE: Dict[str, str] = {
    "default": "cot_default",
    "recall_biased_strong": "cot_recall_biased_strong",
}

# Confidence categorical levels (학술 agent §4.1 원안). gate_level 의 결정:
# threshold (continuous, DECISIONS spec) → gate_level (categorical) 매핑 정합.
_VALID_GATE_LEVELS: Tuple[str, ...] = ("none", "low", "medium", "high")


def _threshold_to_gate_level(threshold: float) -> str:
    """Confidence threshold ∈ [0, 1] → categorical gate_level.

    - threshold ≥ 0.8 : "low"     (low conf 만 override)
    - 0.3 ≤ thr < 0.8 : "medium"  (low + medium override)  ← DECISIONS spec 0.5 정합
    - 0 < thr < 0.3   : "high"    (low + medium + high 모두 override = 사실상 모두 include)
    - threshold ≤ 0   : "none"    (gating 없음, include 만 채택)
    """
    if threshold is None:
        return "none"
    t = float(threshold)
    if t <= 0.0:
        return "none"
    if t < 0.3:
        return "high"
    if t < 0.8:
        return "medium"
    return "low"


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
    def __init__(self, model_name: str, max_iteration: int = 1, temperature: float = 0.0, db_dir: str = "./data/raw/BIRD_dev/dev_databases", api_key: Optional[str] = None, base_url: Optional[str] = None, provider: Optional[str] = None, num_examples: int = 3, prompt_mode: str = "default", sanitize_output: bool = True, cot_reasoning: bool = False, confidence_gated: bool = False, confidence_threshold: float = 0.5, gate_level: Optional[str] = None, fk_hint_format: str = "none", fk_hint_position: str = "inline", **kwargs):
        if prompt_mode not in _PROMPT_SECTION_BY_MODE:
            raise ValueError(
                f"prompt_mode='{prompt_mode}' invalid. "
                f"Expected one of {sorted(_PROMPT_SECTION_BY_MODE.keys())}."
            )
        if cot_reasoning and prompt_mode not in _COT_PROMPT_SECTION_BY_MODE:
            raise ValueError(
                f"cot_reasoning=True but prompt_mode='{prompt_mode}' has no CoT pairing. "
                f"Supported CoT modes: {sorted(_COT_PROMPT_SECTION_BY_MODE.keys())}."
            )
        if gate_level is not None and gate_level not in _VALID_GATE_LEVELS:
            raise ValueError(
                f"gate_level='{gate_level}' invalid. Expected one of {_VALID_GATE_LEVELS}."
            )
        # V7-W1 (FKH, RFP #3) — FK hint 직렬화 옵션. baseline 호환 default ("none").
        if fk_hint_format not in VALID_FK_HINT_FORMATS:
            raise ValueError(
                f"fk_hint_format='{fk_hint_format}' invalid. "
                f"Expected one of {VALID_FK_HINT_FORMATS}."
            )
        if fk_hint_position not in VALID_FK_HINT_POSITIONS:
            raise ValueError(
                f"fk_hint_position='{fk_hint_position}' invalid. "
                f"Expected one of {VALID_FK_HINT_POSITIONS}."
            )
        self.model_name = model_name
        self.max_iteration = max_iteration
        self.temperature = temperature
        self.db_dir = db_dir
        self.provider = provider
        # num_examples: column 별 example value 개수 (0 = 비활성, default 3)
        self.num_examples = max(0, int(num_examples))
        self.prompt_mode = prompt_mode
        self.cot_reasoning = bool(cot_reasoning)
        self.confidence_gated = bool(confidence_gated)
        self.confidence_threshold = float(confidence_threshold)
        # gate_level 우선순위: explicit > threshold-derived. threshold=0.5 → "medium".
        self.gate_level = (
            gate_level
            if gate_level is not None
            else _threshold_to_gate_level(self.confidence_threshold)
        )
        # prompt section 결정 — CoT 활성 시 _COT_PROMPT_SECTION_BY_MODE, 아니면 기본.
        if self.cot_reasoning:
            self._prompt_section = _COT_PROMPT_SECTION_BY_MODE[prompt_mode]
        else:
            self._prompt_section = _PROMPT_SECTION_BY_MODE[prompt_mode]
        self.sanitize_output = bool(sanitize_output)
        # V7-W1 FKH option storage
        self.fk_hint_format = fk_hint_format
        self.fk_hint_position = fk_hint_position
        self.prompt_manager = PromptManager()
        self.client = self._make_llm_client(api_key=api_key, base_url=base_url, provider=provider)
        logger.info(
            f"Initialized XiYanFilter (Iterations: {self.max_iteration}, "
            f"num_examples: {self.num_examples}, Provider: {provider or 'auto'}, "
            f"prompt_mode: {self.prompt_mode}, sanitize: {self.sanitize_output}, "
            f"cot: {self.cot_reasoning}, confidence_gated: {self.confidence_gated}, "
            f"confidence_threshold: {self.confidence_threshold}, "
            f"gate_level: {self.gate_level}, "
            f"fk_hint_format: {self.fk_hint_format}, "
            f"fk_hint_position: {self.fk_hint_position})"
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

    # ------------------------------------------------------------------
    # Wave 6 Phase 2 (a) — CoT parsing + Confidence-Gated post-processing
    # ------------------------------------------------------------------
    @staticmethod
    def parse_cot_output(raw_text: str) -> Tuple[str, Dict[str, Any]]:
        """CoT 프롬프트 출력을 reasoning + JSON 으로 분리 (학술 agent §4.2 정합).

        Returns:
            (reasoning_text, json_dict)
              - json_dict 형식: {table: {col: {"include": bool, "confidence": str}}}
              - JSON parse 실패 시 빈 dict 반환 (recall-safe path 는 caller 가 결정)
        """
        if not isinstance(raw_text, str) or not raw_text.strip():
            return "", {}
        # ---JSON--- separator 로 split (case-insensitive, surrounding whitespace 무시)
        parts = re.split(r"---\s*JSON\s*---", raw_text, maxsplit=1, flags=re.IGNORECASE)
        reasoning = parts[0].strip()
        json_part = parts[1].strip() if len(parts) > 1 else raw_text.strip()
        # 마크다운 fence 제거
        json_part = re.sub(r"^```(?:json)?\s*", "", json_part, flags=re.IGNORECASE)
        json_part = re.sub(r"\s*```$", "", json_part)
        # 첫 '{' ~ 마지막 '}' 슬라이싱 (탐욕 매칭 회피)
        start = json_part.find("{")
        end = json_part.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return reasoning, {}
        try:
            parsed = json.loads(json_part[start : end + 1])
        except json.JSONDecodeError as e:
            logger.warning(f"[XiYanFilter CoT] JSON parse error: {e}")
            return reasoning, {}
        if not isinstance(parsed, dict):
            return reasoning, {}
        return reasoning, parsed

    @staticmethod
    def apply_confidence_gating(
        cot_output: Dict[str, Any],
        gate_level: str = "low",
    ) -> Tuple[Dict[str, List[str]], int, Dict[str, int]]:
        """학술 agent §4.2 confidence-gated post-processing.

        include=False 이면서 confidence 가 gate_level 이하인 column → 강제 include
        (False Negative 방지). categorical {high, medium, low} 기준.

        gate_level:
            - "none"   : gating 없음 (include=True 만 채택)
            - "low"    : low 만 override
            - "medium" : low + medium 모두 override
            - "high"   : low + medium + high 모두 override (사실상 모두 include)

        Returns:
            (final_dict, gated_override_count, confidence_distribution)
              - final_dict: {table: [col, ...]} (include=True 인 col + override 된 col)
              - gated_override_count: confidence-gated 로 강제 retain 된 column 수
              - confidence_distribution: {"high": n, "medium": n, "low": n} per-call
        """
        gate_overrides = {
            "none": set(),
            "low": {"low"},
            "medium": {"low", "medium"},
            "high": {"low", "medium", "high"},
        }
        overrides = gate_overrides.get(gate_level, set())
        final: Dict[str, List[str]] = {}
        distribution = {"high": 0, "medium": 0, "low": 0}
        override_count = 0
        if not isinstance(cot_output, dict):
            return final, 0, distribution

        for table, cols in cot_output.items():
            if not isinstance(cols, dict):
                continue
            kept: List[str] = []
            for col, meta in cols.items():
                if not isinstance(col, str) or not isinstance(meta, dict):
                    continue
                include = bool(meta.get("include", False))
                conf = str(meta.get("confidence", "high")).lower().strip()
                if conf not in ("high", "medium", "low"):
                    conf = "high"  # malformed → 가장 보수적 분류 (override 대상 아님)
                distribution[conf] += 1
                if not include and conf in overrides:
                    include = True
                    override_count += 1
                if include and col not in kept:
                    kept.append(col)
            if kept:
                final[table] = kept
        return final, override_count, distribution

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

    def _serialize_schema(
        self,
        schema_dict: Dict[str, List[str]],
        db_id: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """schema 직렬화 entry — base mschema 위 V7-W1 FKH 옵션 적용.

        본 함수는 LLM prompt 위 schema_str 자리 substitution 의 single source.
        baseline (fk_hint_format='none') 시 _build_mschema_with_values 결과 그대로.
        """
        base_schema = self._build_mschema_with_values(schema_dict, db_id or "")
        if self.fk_hint_format == "none":
            return base_schema
        return serialize_schema_with_fk_hints(
            base_schema=base_schema,
            extracted_subgraph=schema_dict,
            metadata=metadata,
            fk_hint_format=self.fk_hint_format,
            fk_hint_position=self.fk_hint_position,
        )

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
                cot_reasoning=self.cot_reasoning,
                confidence_gated=self.confidence_gated,
                confidence_threshold=self.confidence_threshold,
                gate_level=self.gate_level,
                confidence_distribution={"high": 0, "medium": 0, "low": 0},
                gated_override_count=0,
                raw_filter_count=0,
                final_filter_count=0,
            )
            return {"status": "Unanswerable", "final_nodes": [], "reasoning": "Empty input subgraph", "filter_info": dict(self.last_info)}

        token_before = AgentUtils.token_snapshot()
        current_schema = subgraph.copy()
        iterations_run = 0
        parse_errors = 0
        iteration_trace: List[Dict[str, Any]] = []
        hallucination_removed_total = 0
        input_node_count = sum(len(v) for v in subgraph.values())
        # Wave 6 Phase 2 (a) CoT 측정 누적 변수 (cot_reasoning=True 일 때만 의미)
        confidence_distribution_total = {"high": 0, "medium": 0, "low": 0}
        gated_override_total = 0
        raw_filter_total = 0
        final_filter_total = 0

        for i in range(self.max_iteration):
            iterations_run += 1
            logger.debug(f"[XiYanFilter] Iteration {i+1}/{self.max_iteration}")
            cols_before = sum(len(v) for v in current_schema.values())

            # 1. M-Schema (Value 포함) 포맷팅 — V7-W1 FKH 분기 적용 (default 'none' = baseline)
            schema_str = self._serialize_schema(
                current_schema, db_id, metadata=kwargs.get("metadata"),
            )

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

            # 4. JSON 파싱 및 정제 — CoT 활성 시 별도 path 분기
            iter_parse_ok = False
            stop_iteration = False
            if self.cot_reasoning:
                # CoT path — Wave 6 Phase 2 (a) (학술 agent §4.2)
                cot_reasoning_text, cot_dict = self.parse_cot_output(response)
                if not cot_dict:
                    parse_errors += 1
                    logger.warning(
                        "[XiYanFilter CoT] parse_cot_output produced empty dict — skipping iter."
                    )
                    iteration_trace.append({
                        "iteration": i + 1,
                        "cols_before": cols_before,
                        "cols_after": cols_before,  # 변경 없음
                        "cols_removed": 0,
                        "parse_ok": False,
                        "cot": True,
                    })
                    break
                # raw_filter_count = cot_dict 의 모든 (table, col) 수 (gate 전 후보)
                raw_count = 0
                for _t, _cols in cot_dict.items():
                    if isinstance(_cols, dict):
                        raw_count += len([c for c in _cols if isinstance(c, str)])
                raw_filter_total += raw_count

                if self.confidence_gated:
                    selected_columns, override_n, distribution = (
                        self.apply_confidence_gating(cot_dict, gate_level=self.gate_level)
                    )
                    gated_override_total += int(override_n)
                else:
                    # gating 없음: include=True 만 채택 + distribution 만 측정
                    selected_columns, _, distribution = self.apply_confidence_gating(
                        cot_dict, gate_level="none",
                    )
                # distribution 누적
                for _k in ("high", "medium", "low"):
                    confidence_distribution_total[_k] += int(distribution.get(_k, 0))

                # sanitize 후처리 (default-on)
                if self.sanitize_output:
                    sanitized_schema, removed = self.sanitize_filter_output(
                        selected_columns, subgraph,
                    )
                    if removed > 0:
                        logger.debug(
                            f"[XiYanFilter CoT] sanitize removed {removed} "
                            f"hallucinated entries (mode={self.prompt_mode})."
                        )
                    hallucination_removed_total += int(removed)
                    current_schema = sanitized_schema
                else:
                    current_schema = selected_columns
                final_count = sum(len(v) for v in current_schema.values())
                final_filter_total += final_count
                logger.debug(
                    f"[XiYanFilter CoT] iter {i+1}: raw={raw_count}, "
                    f"override={gated_override_total}, final={final_count}, "
                    f"distribution={distribution}, reasoning_preview="
                    f"{cot_reasoning_text[:160]!r}"
                )
                iteration_trace.append({
                    "iteration": i + 1,
                    "cols_before": cols_before,
                    "cols_after": final_count,
                    "cols_removed": cols_before - final_count,
                    "parse_ok": True,
                    "cot": True,
                    "raw_count": raw_count,
                    "gated_override": int(gated_override_total),
                    "confidence_distribution": dict(distribution),
                })
                iter_parse_ok = True
                # CoT path 완료 → 다음 iteration 진행 또는 break (max_iteration=1 default)
                continue
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
            cot_reasoning=self.cot_reasoning,
            confidence_gated=self.confidence_gated,
            confidence_threshold=self.confidence_threshold,
            gate_level=self.gate_level,
            confidence_distribution=dict(confidence_distribution_total),
            gated_override_count=int(gated_override_total),
            raw_filter_count=int(raw_filter_total),
            final_filter_count=int(final_filter_total),
        )
        return {
            "status": status,
            "final_nodes": final_nodes,
            "reasoning": (
                f"Filtered with DB Value Examples "
                f"(iterations={self.max_iteration}, mode={self.prompt_mode}, "
                f"sanitize={self.sanitize_output}, hallucination_removed="
                f"{hallucination_removed_total}, prune%={prune_pct:.3f}"
                + (
                    f", cot=True, gated={self.confidence_gated}, "
                    f"gate_level={self.gate_level}, "
                    f"override={gated_override_total}, "
                    f"raw={raw_filter_total}, final={final_filter_total}, "
                    f"distribution={confidence_distribution_total}"
                    if self.cot_reasoning else ""
                )
                + ")"
            ),
            "filter_info": dict(self.last_info),
        }