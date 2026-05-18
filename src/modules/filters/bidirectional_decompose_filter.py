"""D1 Question Decomposition → Multi-Backward (학술 agent §1, Wave 8 2026-05-18).

DECISIONS 2026-05-18 (Wave 8 M4 Bidirectional 발전 4 Direction) §2 D1 spec 정합.
M4 BidirectionalFilter (commit 88ad47e+60b6988+7a07a6b) 위 wrapper — anchor 변경
없이 composition 패턴.

설계:
  Step 1  Question Decomposer (d1_decompose) → List[sub_q]  — 1 LLM call
  Step 2  per sub_q Backward (d1_backward_sub) — N LLM call (N = len(sub_q))
  Step 3  Union: M4 baseline ∪ ∪_i sub_q_backward_i
          - v1 (default): Forward = M4 original Forward, Backward = sub_q union
          - v2 (forward_per_sub_q=True): Forward 도 sub_q 별 multi-call

LLM/q: M4 의 2 + 1 (decomposer) + N (sub_q backward)
        v2 는 추가 N (sub_q forward 각각)

학술 frame: paper §V.5.x.M.16 candidate — Multi-Backward Question Decomposition
mechanism (Filter Dominance narrative 의 question-decomposition axis 추가).

핵심 제약 (Wave 8 정합):
  - LLM 입력에 Full Schema 포함 금지 — Extractor 출력 (subgraph) 만 schema_str 로
  - sanitize_filter_output() default-on — XiYanFilter static method 재사용
  - Decomposition 실패 (parse fail / empty list) 시 M4 baseline 그대로 (fallback)
"""
import json
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


@register("filter", "BidirectionalDecomposeFilter")
class BidirectionalDecomposeFilter(BaseFilter):
    """D1 — M4 위에 Question Decomposer + Multi-Backward Union wrapper."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        db_dir: str = "./data/raw/BIRD_dev/dev_databases",
        num_examples: int = 3,
        sanitize_output: bool = True,
        # D1-specific
        d1_max_sub_questions: int = 5,
        d1_forward_per_sub_q: bool = False,  # v2 mode
        decompose_section: str = "d1_decompose",
        backward_sub_section: str = "d1_backward_sub",
        # M4 base params (forwarded)
        m4_bidirectional_forward_prompt_mode: Optional[str] = None,
        m4_bidirectional_forward_voting_strategy: str = "MAJORITY",
        m4_backward_section: str = "bidirectional_backward",
        provider: Optional[str] = "glm",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        if int(d1_max_sub_questions) < 1:
            raise ValueError(
                f"d1_max_sub_questions must be ≥ 1, got {d1_max_sub_questions}."
            )
        self.model_name = model_name
        self.temperature = float(temperature)
        self.db_dir = db_dir
        self.num_examples = max(0, int(num_examples))
        self.sanitize_output = bool(sanitize_output)
        self.d1_max_sub_questions = int(d1_max_sub_questions)
        self.d1_forward_per_sub_q = bool(d1_forward_per_sub_q)
        self.decompose_section = decompose_section
        self.backward_sub_section = backward_sub_section
        self.provider = provider
        self.prompt_manager = PromptManager()
        self.client = self._make_llm_client(
            api_key=api_key, base_url=base_url, provider=provider,
        )

        # M4 base composition (anchor 변경 없이 그대로 호출)
        self.m4 = BidirectionalFilter(
            model_name=model_name, temperature=temperature, db_dir=db_dir,
            num_examples=num_examples, sanitize_output=sanitize_output,
            bidirectional_forward_prompt_mode=m4_bidirectional_forward_prompt_mode,
            bidirectional_forward_voting_strategy=m4_bidirectional_forward_voting_strategy,
            backward_section=m4_backward_section,
            provider=provider, api_key=api_key, base_url=base_url,
        )
        logger.info(
            "Initialized BidirectionalDecomposeFilter (D1) "
            f"(model={model_name}, max_sub_q={self.d1_max_sub_questions}, "
            f"forward_per_sub_q={self.d1_forward_per_sub_q}, "
            f"sanitize={self.sanitize_output})"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_json_array(response: str, cap: int) -> List[str]:
        """Parse JSON array of sub-question strings. Fallback to []."""
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
        out = [s.strip() for s in arr if isinstance(s, str) and s.strip()]
        return out[:cap] if cap > 0 else out

    @staticmethod
    def _parse_json_dict(response: str) -> Dict[str, List[str]]:
        """Parse {table: [col,...]} dict. Fallback to {}."""
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
    def _final_nodes_to_subgraph(nodes: List[str]) -> Dict[str, List[str]]:
        """['t.c', 't.c2'] → {'t': ['c', 'c2']}."""
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
                filter_type="BidirectionalDecomposeFilter",
                input_subgraph={}, final_nodes=[], status="Unanswerable",
                token_before=empty_tok, token_after=empty_tok, t_start=t_start,
                model=self.model_name,
                d1_max_sub_questions=self.d1_max_sub_questions,
                d1_forward_per_sub_q=self.d1_forward_per_sub_q,
                num_sub_questions=0,
                added_by_multi_backward=0,
                d1_llm_calls=0,
                m4_baseline_count=0,
            )
            return {
                "status": "Unanswerable", "final_nodes": [],
                "reasoning": "Empty input subgraph",
                "filter_info": dict(self.last_info),
            }

        token_before = AgentUtils.token_snapshot()

        # Step 0: M4 baseline 호출 (anchor 변경 없음, 그대로 사용)
        m4_result = self.m4.refine(
            query=query, subgraph=subgraph, db_id=db_id, gold=gold, **kwargs,
        )
        m4_nodes = list(m4_result.get("final_nodes") or [])
        m4_baseline_sg = self._final_nodes_to_subgraph(m4_nodes)

        # Step 1: Question Decomposition
        d1_llm_calls = 0
        try:
            decomp_resp = self._call_prompt(self.decompose_section, query=query)
            d1_llm_calls += 1
        except Exception as e:
            logger.warning(f"[D1 decompose] LLM call failed: {e}")
            decomp_resp = ""
        sub_questions = self._parse_json_array(decomp_resp, cap=self.d1_max_sub_questions)

        if not sub_questions:
            # Fallback: M4 baseline 그대로 (학술 agent §1.4 주의 1)
            final_sg = m4_baseline_sg
            multi_backward_total_added = 0
            sub_q_results: List[Dict[str, List[str]]] = []
            sub_q_forward_results: List[Dict[str, List[str]]] = []
            decomp_failed = True
        else:
            decomp_failed = False
            # Step 2: per sub-q Backward (and optionally Forward)
            schema_str = self._build_schema_str(subgraph, db_id)
            sub_q_results = []
            sub_q_forward_results = []
            for sq in sub_questions:
                # Backward per sub-q
                try:
                    bsub_resp = self._call_prompt(
                        self.backward_sub_section,
                        schema_str=schema_str, sub_query=sq,
                    )
                    d1_llm_calls += 1
                except Exception as e:
                    logger.warning(f"[D1 bwd sub-q] LLM call failed for '{sq}': {e}")
                    bsub_resp = ""
                bsub_parsed = self._parse_json_dict(bsub_resp)
                if self.sanitize_output:
                    bsub_clean, _ = XiYanFilter.sanitize_filter_output(
                        bsub_parsed, subgraph,
                    )
                else:
                    bsub_clean = bsub_parsed
                sub_q_results.append(bsub_clean)

                # v2 (optional): Forward per sub-q
                if self.d1_forward_per_sub_q:
                    try:
                        fsub_resp = self._call_prompt(
                            self.m4.forward_section,
                            schema_str=schema_str, query=sq,
                            example_json_str=self._example_json_for(subgraph),
                        )
                        d1_llm_calls += 1
                    except Exception as e:
                        logger.warning(f"[D1 fwd sub-q] LLM call failed for '{sq}': {e}")
                        fsub_resp = ""
                    fsub_parsed = self._parse_json_dict(fsub_resp)
                    if self.sanitize_output:
                        fsub_clean, _ = XiYanFilter.sanitize_filter_output(
                            fsub_parsed, subgraph,
                        )
                    else:
                        fsub_clean = fsub_parsed
                    sub_q_forward_results.append(fsub_clean)

            # Step 3: Union — M4 baseline + all sub-q backward (+ optional sub-q forward)
            all_subgraphs = [m4_baseline_sg] + sub_q_results + sub_q_forward_results
            final_sg = self._union_subgraphs(*all_subgraphs)
            # added by multi-backward
            multi_backward_total_added = sum(
                1 for t, cols in final_sg.items() for c in (cols or [])
                if not (t in m4_baseline_sg and c in m4_baseline_sg.get(t, []))
            )

        final_nodes = sorted(f"{t}.{c}" for t, cols in final_sg.items() for c in (cols or []))
        status = "Answerable" if final_nodes else "Unanswerable"
        token_after = AgentUtils.token_snapshot()

        self.last_info = AgentUtils.build_filter_info(
            filter_type="BidirectionalDecomposeFilter",
            input_subgraph=subgraph, final_nodes=final_nodes, status=status,
            token_before=token_before, token_after=token_after, t_start=t_start,
            model=self.model_name,
            d1_max_sub_questions=self.d1_max_sub_questions,
            d1_forward_per_sub_q=self.d1_forward_per_sub_q,
            num_sub_questions=len(sub_questions),
            added_by_multi_backward=int(multi_backward_total_added),
            d1_llm_calls=int(d1_llm_calls),
            m4_baseline_count=len(m4_nodes),
            decompose_failed=decomp_failed,
            sub_questions_preview=sub_questions[:3],
        )
        return {
            "status": status, "final_nodes": final_nodes,
            "reasoning": (
                f"[D1 multi-backward] sub_q={len(sub_questions)}, "
                f"d1_llm_calls={d1_llm_calls}, "
                f"m4_baseline={len(m4_nodes)}, added={multi_backward_total_added}, "
                f"forward_per_sub_q={self.d1_forward_per_sub_q}, "
                f"decompose_failed={decomp_failed}"
            ),
            "stats": {
                "num_sub_questions": len(sub_questions),
                "added_by_multi_backward": multi_backward_total_added,
                "d1_llm_calls": d1_llm_calls,
                "m4_baseline_count": len(m4_nodes),
                "decompose_failed": decomp_failed,
            },
            "filter_info": dict(self.last_info),
        }

    @staticmethod
    def _example_json_for(subgraph: Dict[str, List[str]]) -> str:
        """XiYan-style example_json_str — 첫 2 table 의 일부 cols."""
        example_tables = list(subgraph.keys())[:2]
        obj: Dict[str, List[str]] = {}
        for idx, t in enumerate(example_tables):
            obj[t] = (subgraph[t] or [])[: (2 if idx == 0 else 1)]
        return json.dumps(obj)
