"""D4 Value Hint 기반 Forward 강화 (학술 agent §4, Wave 8 2026-05-18).

DECISIONS 2026-05-18 §2 D4 spec 정합. M4 BidirectionalFilter 위 wrapper —
question 에서 값 언급 추출 → Extractor 후보의 column example values 와 매칭 →
Value-Hint Enhanced Forward 프롬프트 (학술 agent §4.3).

설계 (학술 agent §4.4):
  Step 0  M4 baseline 의 Backward 만 호출 (Forward 는 강화 prompt 로 대체)
  Step 1  Value extract (d4_value_extract) — 1 LLM call → List[value_mention]
  Step 2  match_values_to_columns(values, extractor_output) → evidence dict
          {"table.col": {"matched_values": [...], "confidence": "high"|"medium"}}
  Step 3  v1 (default): Value-Hint Forward (d4_forward) — 1 LLM call
          v3 (forced_include=True): high-confidence column 강제 retain (+ M4 그대로)

LLM/q: M4 의 2 + 1 (value extract) + 1 (enhanced forward) = **4** (v1)
        v3 의 경우 M4 의 2 + 1 (value extract only, forward 추가 없음) = **3**

학술 frame: paper §V.5.x.M.18 candidate — Value Evidence axis (Ma 2026 정합).

핵심 제약 (Wave 8 정합):
  - LLM 입력에 Full Schema 포함 금지 — Extractor 후보 (subgraph) 만
  - match_values_to_columns 는 DB sample values 사용 — example values 가 subgraph
    안 column 의 실제 DB 값에서 추출됨 (LLM hallucination 아님)
  - sanitize_filter_output default-on
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


@register("filter", "BidirectionalValueHintFilter")
class BidirectionalValueHintFilter(BaseFilter):
    """D4 — Value Hint 기반 Forward 강화 wrapper."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        db_dir: str = "./data/raw/BIRD_dev/dev_databases",
        num_examples: int = 3,
        sanitize_output: bool = True,
        # D4-specific
        d4_forced_include: bool = False,  # v3 mode
        d4_value_extract_section: str = "d4_value_extract",
        d4_forward_section: str = "d4_forward",
        d4_value_sample_limit: int = 20,  # match 시 column 당 sample value 수
        # M4 base (Backward 만 사용 — v1 mode 에서 Forward 는 d4_forward 가 대체)
        m4_backward_section: str = "bidirectional_backward",
        provider: Optional[str] = "glm",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.temperature = float(temperature)
        self.db_dir = db_dir
        self.num_examples = max(0, int(num_examples))
        self.sanitize_output = bool(sanitize_output)
        self.d4_forced_include = bool(d4_forced_include)
        self.d4_value_extract_section = d4_value_extract_section
        self.d4_forward_section = d4_forward_section
        self.d4_value_sample_limit = max(1, int(d4_value_sample_limit))
        self.m4_backward_section = m4_backward_section
        self.provider = provider
        self.prompt_manager = PromptManager()
        self.client = self._make_llm_client(
            api_key=api_key, base_url=base_url, provider=provider,
        )
        # v3 forced-include 시 M4 그대로 호출. v1 에서는 M4 의 Backward 만 사용 (Forward 는
        # d4_forward 가 대체). M4 instance 는 Backward 호출용으로 보유.
        self.m4 = BidirectionalFilter(
            model_name=model_name, temperature=temperature, db_dir=db_dir,
            num_examples=num_examples, sanitize_output=sanitize_output,
            bidirectional_forward_prompt_mode="recall_biased_mild",  # M4 anchor mild
            backward_section=m4_backward_section,
            provider=provider, api_key=api_key, base_url=base_url,
        )
        logger.info(
            "Initialized BidirectionalValueHintFilter (D4) "
            f"(model={model_name}, forced_include={self.d4_forced_include}, "
            f"value_sample_limit={self.d4_value_sample_limit}, "
            f"sanitize={self.sanitize_output})"
        )

    # ------------------------------------------------------------------
    # Value extraction / matching (학술 agent §4.2)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json_array(response: str) -> List[str]:
        if not isinstance(response, str) or not response.strip():
            return []
        cleaned = response.replace("```json", "").replace("```", "").strip()
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start == -1 or end == -1 or start >= end:
            return []
        try:
            arr = json.loads(cleaned[start : end + 1])
        except Exception:
            return []
        if not isinstance(arr, list):
            return []
        return [str(s).strip() for s in arr if isinstance(s, (str, int, float)) and str(s).strip()]

    @staticmethod
    def _parse_json_dict(response: str) -> Dict[str, List[str]]:
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
                out[t] = [c for c in v.keys() if isinstance(c, str)]
        return out

    @staticmethod
    def _is_numeric(s: Any) -> bool:
        try:
            float(s)
            return True
        except (TypeError, ValueError):
            return False

    def _fetch_column_examples(
        self, subgraph: Dict[str, List[str]], db_id: Optional[str],
    ) -> Dict[str, List[str]]:
        """{table.col: [sample_val, ...]} — DB 에서 num_examples 만큼 fetch.

        d4_value_sample_limit 만큼 더 넓게 sample (matching 용도).
        """
        col_samples: Dict[str, List[str]] = {}
        if not db_id:
            return col_samples
        db_path = os.path.join(self.db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            return col_samples
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            for tbl, cols in (subgraph or {}).items():
                safe = tbl.replace("'", "''")
                for col in (cols or []):
                    try:
                        cur.execute(
                            f'SELECT DISTINCT "{col}" FROM "{safe}" '
                            f'WHERE "{col}" IS NOT NULL LIMIT {self.d4_value_sample_limit}'
                        )
                        samples = [str(r[0]) for r in cur.fetchall()]
                        if samples:
                            col_samples[f"{tbl}.{col}"] = samples
                    except Exception as e:
                        logger.debug(f"[D4 fetch] sample fail {tbl}.{col}: {e}")
            conn.close()
        except Exception as e:
            logger.warning(f"[D4 fetch] DB connect fail {db_id}: {e}")
        return col_samples

    @staticmethod
    def match_values_to_columns(
        value_mentions: List[str],
        col_samples: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, Any]]:
        """학술 agent §4.2 정합. {"table.col": {"matched_values", "confidence"}}.

        confidence: "high" (exact match) / "medium" (partial or numeric hint).
        """
        evidence: Dict[str, Dict[str, Any]] = {}
        for col_key, examples in (col_samples or {}).items():
            examples_lower = [str(e).lower() for e in examples]
            matched: List[Dict[str, str]] = []
            for val in value_mentions or []:
                val_lower = str(val).lower().strip()
                if not val_lower:
                    continue
                # Exact match
                if val_lower in examples_lower:
                    matched.append({"value": val, "type": "exact"})
                    continue
                # Partial: value substring 또는 example substring
                if any(val_lower in ex or ex in val_lower for ex in examples_lower):
                    matched.append({"value": val, "type": "partial"})
                    continue
                # Numeric hint: val 이 숫자고 example 에 숫자 column 이면 (느슨한 hint)
                if BidirectionalValueHintFilter._is_numeric(val) and any(
                    BidirectionalValueHintFilter._is_numeric(ex) for ex in examples
                ):
                    matched.append({"value": val, "type": "numeric_hint"})
            if matched:
                conf = "high" if any(m["type"] == "exact" for m in matched) else "medium"
                evidence[col_key] = {
                    "matched_values": [m["value"] for m in matched],
                    "confidence": conf,
                }
        return evidence

    @staticmethod
    def format_value_evidence(evidence: Dict[str, Dict[str, Any]]) -> str:
        """학술 agent §4.2 정합."""
        if not evidence:
            return "(No value evidence found)"
        lines = ["[Value Evidence — Strong Inclusion Signals]"]
        # high → medium 정렬
        sorted_keys = sorted(
            evidence.keys(),
            key=lambda k: (0 if evidence[k]["confidence"] == "high" else 1, k),
        )
        for k in sorted_keys:
            info = evidence[k]
            conf = info["confidence"].upper()
            vals = ", ".join(f"'{v}'" for v in info["matched_values"])
            lines.append(f"  {conf}: {k} contains {vals}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Subgraph helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _union_subgraphs(*sgs: Dict[str, List[str]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for sg in sgs:
            for t, cols in (sg or {}).items():
                for c in (cols or []):
                    if c not in out.setdefault(t, []):
                        out[t].append(c)
                if not cols and t not in out:
                    out[t] = []
        return out

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

    @staticmethod
    def _example_json_for(subgraph: Dict[str, List[str]]) -> str:
        example_tables = list(subgraph.keys())[:2]
        obj: Dict[str, List[str]] = {}
        for idx, t in enumerate(example_tables):
            obj[t] = (subgraph[t] or [])[: (2 if idx == 0 else 1)]
        return json.dumps(obj)

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
                filter_type="BidirectionalValueHintFilter",
                input_subgraph={}, final_nodes=[], status="Unanswerable",
                token_before=empty_tok, token_after=empty_tok, t_start=t_start,
                model=self.model_name,
                d4_forced_include=self.d4_forced_include,
                evidence_size=0, evidence_high_count=0,
                d4_llm_calls=0, forced_count=0,
            )
            return {
                "status": "Unanswerable", "final_nodes": [],
                "reasoning": "Empty input subgraph",
                "filter_info": dict(self.last_info),
            }

        token_before = AgentUtils.token_snapshot()
        d4_llm_calls = 0

        # Step 1: Value extraction
        try:
            raw_values = self._call_prompt(self.d4_value_extract_section, query=query)
            d4_llm_calls += 1
        except Exception as e:
            logger.warning(f"[D4 value extract] LLM call failed: {e}")
            raw_values = ""
        value_mentions = self._parse_json_array(raw_values)

        # Step 2: Match values to subgraph columns
        col_samples = self._fetch_column_examples(subgraph, db_id)
        evidence = self.match_values_to_columns(value_mentions, col_samples)
        value_evidence_str = self.format_value_evidence(evidence)

        # Step 3: branch v1 (enhanced forward) vs v3 (forced include)
        if self.d4_forced_include:
            # v3: high-confidence column 강제 retain → M4 baseline 그대로 + forced
            m4_result = self.m4.refine(
                query=query, subgraph=subgraph, db_id=db_id, gold=gold, **kwargs,
            )
            m4_nodes = list(m4_result.get("final_nodes") or [])
            m4_sg = self._final_nodes_to_subgraph(m4_nodes)
            forced_sg: Dict[str, List[str]] = {}
            for col_key, info in evidence.items():
                if info["confidence"] == "high" and "." in col_key:
                    t, c = col_key.split(".", 1)
                    if c in (subgraph.get(t, []) or []):  # double-check Extractor 후보
                        if c not in forced_sg.setdefault(t, []):
                            forced_sg[t].append(c)
            final_sg = self._union_subgraphs(m4_sg, forced_sg)
            forced_count = sum(len(v) for v in forced_sg.values())
            m4_baseline_size = len(m4_nodes)
        else:
            # v1: Value-Hint Enhanced Forward (M4 Forward 대체) + M4 Backward Union
            schema_str = self._build_schema_str(subgraph, db_id)
            example_json_str = self._example_json_for(subgraph)
            try:
                raw_fwd = self._call_prompt(
                    self.d4_forward_section,
                    value_evidence_str=value_evidence_str,
                    schema_str=schema_str,
                    query=query,
                    example_json_str=example_json_str,
                )
                d4_llm_calls += 1
            except Exception as e:
                logger.warning(f"[D4 enhanced forward] LLM call failed: {e}")
                raw_fwd = ""
            fwd_parsed = self._parse_json_dict(raw_fwd)
            if self.sanitize_output:
                fwd_clean, _ = XiYanFilter.sanitize_filter_output(fwd_parsed, subgraph)
            else:
                fwd_clean = fwd_parsed

            # M4 Backward 만 별도 호출 (M4 Forward 는 d4_forward 가 대체)
            try:
                bwd_resp = self.m4._call_prompt(
                    self.m4.backward_section, query, schema_str,
                )
            except Exception as e:
                logger.warning(f"[D4 backward] LLM call failed: {e}")
                bwd_resp = ""
            bwd_parsed = self._parse_json_dict(bwd_resp)
            if self.sanitize_output:
                bwd_clean, _ = XiYanFilter.sanitize_filter_output(bwd_parsed, subgraph)
            else:
                bwd_clean = bwd_parsed

            final_sg = self._union_subgraphs(fwd_clean, bwd_clean)
            forced_count = 0
            m4_baseline_size = None  # v1 모드: M4 baseline 별도 산출 없음

        final_nodes = sorted(
            f"{t}.{c}" for t, cols in final_sg.items() for c in (cols or [])
        )
        status = "Answerable" if final_nodes else "Unanswerable"
        token_after = AgentUtils.token_snapshot()

        evidence_high_count = sum(
            1 for v in evidence.values() if v["confidence"] == "high"
        )
        # evidence_gold_precision (gold 있을 때만 — runtime evaluate)
        evidence_gold_precision: Optional[float] = None
        if gold is not None and evidence:
            gold_cols = {
                f"{t}.{c}" for t, cols in (gold or {}).items() for c in (cols or [])
            }
            high_keys = {k for k, v in evidence.items() if v["confidence"] == "high"}
            if high_keys:
                evidence_gold_precision = (
                    len(high_keys & gold_cols) / len(high_keys)
                )

        self.last_info = AgentUtils.build_filter_info(
            filter_type="BidirectionalValueHintFilter",
            input_subgraph=subgraph, final_nodes=final_nodes, status=status,
            token_before=token_before, token_after=token_after, t_start=t_start,
            model=self.model_name,
            d4_forced_include=self.d4_forced_include,
            evidence_size=len(evidence),
            evidence_high_count=evidence_high_count,
            evidence_gold_precision=evidence_gold_precision,
            d4_llm_calls=int(d4_llm_calls),
            forced_count=int(forced_count),
            m4_baseline_count=m4_baseline_size,
            value_mentions_count=len(value_mentions),
            evidence_preview=list(evidence.keys())[:5],
        )
        return {
            "status": status, "final_nodes": final_nodes,
            "reasoning": (
                f"[D4 value-hint{'/forced' if self.d4_forced_include else ''}] "
                f"value_mentions={len(value_mentions)}, "
                f"evidence={len(evidence)} (high={evidence_high_count}), "
                f"forced={forced_count}, d4_llm_calls={d4_llm_calls}"
            ),
            "stats": {
                "evidence_size": len(evidence),
                "evidence_high_count": evidence_high_count,
                "evidence_gold_precision": evidence_gold_precision,
                "d4_llm_calls": d4_llm_calls,
                "forced_count": forced_count,
                "value_mentions_count": len(value_mentions),
            },
            "filter_info": dict(self.last_info),
        }
