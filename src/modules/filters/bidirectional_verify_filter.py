"""D3 Self-Verification Loop (학술 agent §3, Wave 8 2026-05-18).

DECISIONS 2026-05-18 §2 D3 spec 정합. M4 BidirectionalFilter 위 wrapper —
Sketch SQL 생성 → DB 실행 → error parse → 누락 column recover → loop.

설계 (학술 agent §3.3):
  Step 0  M4 baseline 호출 (anchor 변경 없음)
  Loop  (max_rounds 회):
    Step 1  Sketch SQL 생성 (d3_sketch_sql) — 1 LLM call
    Step 2  DB 실행 (sqlite3, 5s timeout — hard limit)
    Step 3  성공 시 break; 실패 시 parse_missing_from_error → hints
    Step 4  recover_from_extractor(hints, subgraph, current_schema)
            — Extractor 후보 (subgraph) 안에서만 검색 (학술 agent §3.5 정합)
    Step 5  current_schema ← union (current, recovered)

LLM/q: M4 의 2 + verify rounds (1~2)
DB: SQLite 직접 실행 (sandbox-aware, timeout=5s, 학술 agent §3.5 주의 1).

학술 frame: paper §V.5.x.M.17 candidate — execution feedback loop axis (AutoLink
Wang 2025 정합).

핵심 제약 (Wave 8 정합):
  - LLM 입력에 Full Schema 포함 금지 — current_schema (= subgraph 초기값) 만
  - sanitize_filter_output default-on (XiYanFilter static method 재사용)
  - Sketch SQL 의 column 이 Extractor 후보 (subgraph) 에 없으면 recover 안 함
"""
import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from modules.registry import register
from modules.base import BaseFilter
from modules.filters.agents import AgentUtils
from modules.filters.xiyan_filter import XiYanFilter
from modules.filters.bidirectional_filter import BidirectionalFilter
from prompts.prompt_manager import PromptManager
from utils.logger import get_logger

logger = get_logger(__name__)

# 학술 agent §3.2 — SQLite 오류 패턴
_RE_NO_SUCH_COLUMN = re.compile(r"no such column:?\s*([\w\.]+)", re.IGNORECASE)
_RE_NO_SUCH_TABLE = re.compile(r"no such table:?\s*(\w+)", re.IGNORECASE)
_RE_COLUMN_NOT_EXIST = re.compile(
    r"column\s+(\w+)\s+(?:of\s+table\s+(\w+)\s+)?does not exist",
    re.IGNORECASE,
)


@register("filter", "BidirectionalVerifyLoopFilter")
class BidirectionalVerifyLoopFilter(BaseFilter):
    """D3 — M4 위에 Self-Verification Loop wrapper."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        db_dir: str = "./data/raw/BIRD_dev/dev_databases",
        num_examples: int = 3,
        sanitize_output: bool = True,
        # D3-specific
        d3_max_rounds: int = 2,
        d3_db_timeout_s: float = 5.0,
        sketch_section: str = "d3_sketch_sql",
        # M4 base
        m4_bidirectional_forward_prompt_mode: Optional[str] = None,
        m4_bidirectional_forward_voting_strategy: str = "MAJORITY",
        m4_backward_section: str = "bidirectional_backward",
        provider: Optional[str] = "glm",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        if int(d3_max_rounds) < 1 or int(d3_max_rounds) > 5:
            raise ValueError(
                f"d3_max_rounds out of [1, 5], got {d3_max_rounds}."
            )
        if float(d3_db_timeout_s) <= 0:
            raise ValueError(f"d3_db_timeout_s must be > 0, got {d3_db_timeout_s}.")
        self.model_name = model_name
        self.temperature = float(temperature)
        self.db_dir = db_dir
        self.num_examples = max(0, int(num_examples))
        self.sanitize_output = bool(sanitize_output)
        self.d3_max_rounds = int(d3_max_rounds)
        self.d3_db_timeout_s = float(d3_db_timeout_s)
        self.sketch_section = sketch_section
        self.provider = provider
        self.prompt_manager = PromptManager()
        self.client = self._make_llm_client(
            api_key=api_key, base_url=base_url, provider=provider,
        )
        self.m4 = BidirectionalFilter(
            model_name=model_name, temperature=temperature, db_dir=db_dir,
            num_examples=num_examples, sanitize_output=sanitize_output,
            bidirectional_forward_prompt_mode=m4_bidirectional_forward_prompt_mode,
            bidirectional_forward_voting_strategy=m4_bidirectional_forward_voting_strategy,
            backward_section=m4_backward_section,
            provider=provider, api_key=api_key, base_url=base_url,
        )
        logger.info(
            "Initialized BidirectionalVerifyLoopFilter (D3) "
            f"(model={model_name}, max_rounds={self.d3_max_rounds}, "
            f"timeout={self.d3_db_timeout_s}s, sanitize={self.sanitize_output})"
        )

    # ------------------------------------------------------------------
    # Error parsing + recovery helpers (학술 agent §3.2)
    # ------------------------------------------------------------------
    @staticmethod
    def parse_missing_from_error(error_msg: str) -> List[str]:
        """SQLite 오류 메시지에서 누락 column/table 힌트 추출 (학술 agent §3.2)."""
        if not error_msg:
            return []
        hints: List[str] = []
        hints.extend(_RE_NO_SUCH_COLUMN.findall(error_msg))
        hints.extend(_RE_NO_SUCH_TABLE.findall(error_msg))
        for m in _RE_COLUMN_NOT_EXIST.findall(error_msg):
            joined = ".".join(p for p in m if p)
            if joined:
                hints.append(joined)
        return hints

    @staticmethod
    def recover_from_extractor(
        hints: List[str],
        extractor_output: Dict[str, List[str]],
        current_schema: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """오류 힌트에서 Extractor 후보에 있는 column 만 복구 (학술 agent §3.2).

        Hallucination 방지: extractor_output 에 없는 column 은 추가하지 않음.
        """
        recovered: Dict[str, List[str]] = {}
        cur_set = {
            (t, c) for t, cols in (current_schema or {}).items() for c in (cols or [])
        }
        for hint in hints or []:
            if not isinstance(hint, str):
                continue
            parts = hint.split(".")
            hint_table = parts[0] if len(parts) > 1 else None
            hint_col = parts[-1]
            for table, cols in (extractor_output or {}).items():
                if hint_table is not None and hint_table.lower() != table.lower():
                    continue
                if hint_col in (cols or []) and (table, hint_col) not in cur_set:
                    if hint_col not in recovered.setdefault(table, []):
                        recovered[table].append(hint_col)
            # column-only hint with no exact table → 전체 검색
            if hint_table is None:
                for table, cols in (extractor_output or {}).items():
                    if hint_col in (cols or []) and (table, hint_col) not in cur_set:
                        if hint_col not in recovered.setdefault(table, []):
                            recovered[table].append(hint_col)
        return recovered

    # ------------------------------------------------------------------
    # DB execute (sqlite3 with timeout)
    # ------------------------------------------------------------------
    def _execute_sketch_sql(
        self, sql_text: str, db_id: Optional[str],
    ) -> Tuple[bool, Optional[str]]:
        """Sketch SQL DB 실행. (success, error_msg).

        timeout: SQLite 의 progress_handler 로 hard limit. 5s 후 강제 중단 →
        success=False, error_msg="execution timeout".
        """
        if not sql_text or not isinstance(sql_text, str) or not sql_text.strip():
            return False, "empty sql"
        if not db_id:
            return False, "no db_id provided"
        db_path = os.path.join(self.db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            return False, f"db not found: {db_path}"
        # 마크다운 fence 제거
        sql_clean = re.sub(r"^```(?:sql)?\s*", "", sql_text.strip(), flags=re.IGNORECASE)
        sql_clean = re.sub(r"\s*```$", "", sql_clean)
        if not sql_clean:
            return False, "empty sql after cleanup"
        # 첫 SQL 문장만 (semicolon 까지)
        sql_stmt = sql_clean.split(";", 1)[0]
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            # Hard timeout via progress_handler — 매 N opcode 마다 시계 체크
            t_start = time.perf_counter()
            timeout_s = self.d3_db_timeout_s

            def _abort_if_slow() -> int:
                return 1 if (time.perf_counter() - t_start) > timeout_s else 0
            try:
                conn.set_progress_handler(_abort_if_slow, 10000)
            except Exception:
                pass
            cur = conn.cursor()
            cur.execute(sql_stmt)
            _ = cur.fetchmany(1)  # 1 row만 — full scan 회피
            conn.close()
            return True, None
        except sqlite3.OperationalError as e:
            msg = str(e)
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            # progress_handler abort 는 OperationalError("interrupted") 로 옴
            if "interrupted" in msg.lower():
                return False, f"execution timeout ({timeout_s}s)"
            return False, msg
        except Exception as e:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return False, f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _final_nodes_to_subgraph(nodes: List[str]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for n in nodes or []:
            if not isinstance(n, str):
                continue
            if "." in n:
                t, c = n.split(".", 1)
                if c not in out.setdefault(t, []):
                    out[t].append(c)
            else:
                out.setdefault(n, [])
        return out

    @staticmethod
    def _union_subgraphs(*subgraphs: Dict[str, List[str]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for sg in subgraphs:
            for t, cols in (sg or {}).items():
                for c in (cols or []):
                    if c not in out.setdefault(t, []):
                        out[t].append(c)
                if not cols and t not in out:
                    out[t] = []
        return out

    def _build_schema_str(self, subgraph: Dict[str, List[str]], db_id: Optional[str]) -> str:
        helper = XiYanFilter.__new__(XiYanFilter)
        helper.db_dir = self.db_dir
        helper.num_examples = self.num_examples
        return helper._build_mschema_with_values(subgraph, db_id or "")

    def _call_prompt(self, section: str, **template_kwargs) -> str:
        prompt = self.prompt_manager.load_prompt(
            file_name='filter', section=section, **template_kwargs,
        )
        return self.client.generate_text(
            prompt=prompt, model=self.model_name, temperature=self.temperature,
        )

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def refine(
        self, query: str, subgraph: Dict[str, List[str]], db_id: Optional[str] = None,
        gold: Optional[Dict[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        empty_tok = {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
        if not subgraph:
            self.last_info = AgentUtils.build_filter_info(
                filter_type="BidirectionalVerifyLoopFilter",
                input_subgraph={}, final_nodes=[], status="Unanswerable",
                token_before=empty_tok, token_after=empty_tok, t_start=t_start,
                model=self.model_name,
                d3_max_rounds=self.d3_max_rounds,
                verify_success_rate=0.0, avg_rounds_used=0,
                d3_llm_calls=0, m4_baseline_count=0,
            )
            return {
                "status": "Unanswerable", "final_nodes": [],
                "reasoning": "Empty input subgraph",
                "filter_info": dict(self.last_info),
            }
        token_before = AgentUtils.token_snapshot()

        # M4 baseline (anchor 그대로)
        m4_result = self.m4.refine(
            query=query, subgraph=subgraph, db_id=db_id, gold=gold, **kwargs,
        )
        m4_nodes = list(m4_result.get("final_nodes") or [])
        current_sg = self._final_nodes_to_subgraph(m4_nodes)
        m4_baseline_size = len(m4_nodes)

        d3_llm_calls = 0
        verify_log: List[Dict[str, Any]] = []
        recovered_total = 0
        rounds_used = 0

        for round_idx in range(self.d3_max_rounds):
            rounds_used = round_idx + 1
            # Sketch SQL prompt = current schema string + query
            schema_str = self._build_schema_str(current_sg, db_id)
            try:
                sketch_resp = self._call_prompt(
                    self.sketch_section, schema_str=schema_str, query=query,
                )
                d3_llm_calls += 1
            except Exception as e:
                logger.warning(f"[D3 sketch round {round_idx+1}] LLM call failed: {e}")
                sketch_resp = ""
            success, err = self._execute_sketch_sql(sketch_resp, db_id)
            verify_log.append({
                "round": round_idx + 1,
                "sketch_sql_preview": (sketch_resp or "")[:200],
                "success": bool(success),
                "error": err,
            })
            if success:
                break
            hints = self.parse_missing_from_error(err or "")
            if not hints:
                # 학술 agent §3.3 — 파싱 가능 힌트 없음 → 조기 종료
                break
            recovered = self.recover_from_extractor(hints, subgraph, current_sg)
            if not recovered:
                # 학술 agent §3.3 — Extractor 후보에도 없음 → 종료
                break
            # sanitize (recovered 가 subgraph 안의 col 만 포함하므로 hallucination 0 이지만
            # 일관성 위해 적용)
            if self.sanitize_output:
                recovered, _ = XiYanFilter.sanitize_filter_output(recovered, subgraph)
            current_sg = self._union_subgraphs(current_sg, recovered)
            recovered_total += sum(len(v) for v in recovered.values())

        final_nodes = sorted(
            f"{t}.{c}" for t, cols in current_sg.items() for c in (cols or [])
        )
        status = "Answerable" if final_nodes else "Unanswerable"
        verify_success_rate = (
            sum(1 for v in verify_log if v["success"]) / max(len(verify_log), 1)
        )
        token_after = AgentUtils.token_snapshot()
        self.last_info = AgentUtils.build_filter_info(
            filter_type="BidirectionalVerifyLoopFilter",
            input_subgraph=subgraph, final_nodes=final_nodes, status=status,
            token_before=token_before, token_after=token_after, t_start=t_start,
            model=self.model_name,
            d3_max_rounds=self.d3_max_rounds,
            d3_db_timeout_s=self.d3_db_timeout_s,
            verify_success_rate=float(verify_success_rate),
            avg_rounds_used=int(rounds_used),
            d3_llm_calls=int(d3_llm_calls),
            m4_baseline_count=int(m4_baseline_size),
            recovered_count=int(recovered_total),
            verify_log=verify_log,
        )
        return {
            "status": status, "final_nodes": final_nodes,
            "reasoning": (
                f"[D3 verify-loop] rounds_used={rounds_used}, "
                f"d3_llm_calls={d3_llm_calls}, "
                f"verify_success_rate={verify_success_rate:.2f}, "
                f"m4_baseline={m4_baseline_size}, recovered={recovered_total}"
            ),
            "stats": {
                "verify_success_rate": verify_success_rate,
                "avg_rounds_used": rounds_used,
                "d3_llm_calls": d3_llm_calls,
                "m4_baseline_count": m4_baseline_size,
                "recovered_count": recovered_total,
            },
            "filter_info": dict(self.last_info),
        }
