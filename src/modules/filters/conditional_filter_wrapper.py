"""ConditionalFilterWrapper — voluntary cost-effective Filter skip based on TCR(q).

학술 Agent Improving Plan §Phase 4.2 (planning/improving_exp_plan_by_scholar_agent_2026-05-15.md)
+ DECISIONS 2026-05-16 학술 agent plan Phase 3+4 활성 entry §3 Phase 4.2 spec.

설계:
  - 정의   : `TCR(q) = |filter input columns| / |full schema columns|`
            (작을수록 extractor 가 schema 를 잘 trim 했다는 신호 → Filter 호출 불필요)
  - rule   : `TCR(q) < tcr_threshold` → inner Filter 호출 skip,
             extractor output (subgraph) 의 모든 column 을 final_nodes 로 그대로 반환
  - 외부 TCR (kwargs `tcr`) 가 들어오면 우선 사용 (root pipeline 이 미리 계산한 값).
            없으면 본 wrapper 가 metadata['col_to_id'] 기준으로 자체 계산.
  - voluntary  : 5/14 anchor 의 6.32% involuntary skip (filter 자체가 빈 결과 반환 등)
                과 별개 mechanism. 본 wrapper 는 voluntary cost-effective skip 만 담당.

학술 frame (DECISIONS §"Phase 4.2 활성"):
  - paper §V.5.x.M.3 production deployment + §V.5.x.M.11 Filter Short-Circuit
    voluntary vs involuntary mechanism 분리 narrative 확장 evidence.

Config (yaml):
  filter:
    name: "ConditionalFilterWrapper"
    params:
      call_mode: "conditional"     # "conditional" | "always" (baseline, no-op)
      tcr_threshold: 0.5
      inner_filter:
        name: "XiYanFilter"
        params: { provider: "glm", model_name: "...", ... }
"""
import time
from typing import Any, Dict, List, Optional, Set

from modules.registry import register, build
from modules.base import BaseFilter
from modules.filters.agents import AgentUtils
from utils.logger import get_logger

logger = get_logger(__name__)

_VALID_CALL_MODES = ("conditional", "always")


@register("filter", "ConditionalFilterWrapper")
class ConditionalFilterWrapper(BaseFilter):
    """TCR-gated voluntary Filter skip wrapper.

    Args:
        inner_filter : 감쌀 Filter config (registry name + params).
        call_mode    : "conditional" → TCR gate 적용. "always" → 항상 호출 (baseline).
        tcr_threshold: TCR(q) < threshold 면 skip. range [0, 1].
    """

    def __init__(
        self,
        inner_filter: Dict[str, Any],
        call_mode: str = "conditional",
        tcr_threshold: float = 0.5,
        **kwargs,
    ):
        if call_mode not in _VALID_CALL_MODES:
            raise ValueError(
                f"call_mode='{call_mode}' invalid. Expected one of {_VALID_CALL_MODES}."
            )
        if not (0.0 <= float(tcr_threshold) <= 1.0):
            raise ValueError(
                f"tcr_threshold={tcr_threshold} out of [0, 1]."
            )
        self.inner = build("filter", inner_filter)
        self.call_mode = call_mode
        self.tcr_threshold = float(tcr_threshold)
        self.inner_filter_name = inner_filter.get("name", "<unknown>")
        logger.info(
            "Initialized ConditionalFilterWrapper "
            f"(call_mode={self.call_mode}, tcr_threshold={self.tcr_threshold}, "
            f"inner={self.inner_filter_name})"
        )

    # ------------------------------------------------------------------
    # TCR computation
    # ------------------------------------------------------------------
    @staticmethod
    def _count_subgraph_columns(subgraph: Dict[str, List[str]]) -> int:
        """subgraph 의 "table.col" pair 수 + table-only entry 도 1 로 셈."""
        n = 0
        for t, cols in (subgraph or {}).items():
            if cols:
                n += len(cols)
            else:
                n += 1  # table-only node (col 없는 entry)
        return n

    @staticmethod
    def _count_full_schema_columns(metadata: Optional[Dict[str, Any]]) -> int:
        """metadata['col_to_id'] 기준 full schema column 수.

        col_to_id 키 형식: 'table.col' (HeteroGraphBuilder 의 출력).
        없으면 0 반환 → caller 가 fallback 처리.
        """
        if not metadata:
            return 0
        col_to_id = metadata.get("col_to_id") or {}
        return sum(
            1 for k in col_to_id.keys()
            if isinstance(k, str) and "." in k
        )

    def _compute_tcr(
        self,
        subgraph: Dict[str, List[str]],
        metadata: Optional[Dict[str, Any]],
        tcr_override: Optional[float] = None,
    ) -> Optional[float]:
        """TCR(q) = |subgraph cols| / |full schema cols|.

        Priority:
          1. tcr_override (root pipeline 이 미리 계산한 값) — 가용 시 그대로 사용
          2. metadata['col_to_id'] 기준 자체 계산
          3. 둘 다 불가 → None (caller 가 "tcr unknown → 안전 path 로 always-call" 처리)
        """
        if tcr_override is not None:
            try:
                v = float(tcr_override)
                if 0.0 <= v <= 1.0:
                    return v
                logger.warning(
                    f"[ConditionalFilter] tcr_override={v} out of [0,1] — ignored."
                )
            except (TypeError, ValueError):
                logger.warning(
                    f"[ConditionalFilter] tcr_override={tcr_override!r} not float — ignored."
                )

        n_sub = self._count_subgraph_columns(subgraph)
        n_full = self._count_full_schema_columns(metadata)
        if n_full <= 0:
            return None  # full schema 알 수 없음 → safe fallback (caller decides)
        # n_sub > n_full 인 경우 (extractor 가 schema 외 노드 추가? 비정상) — clamp to 1.0
        ratio = float(n_sub) / float(n_full)
        if ratio > 1.0:
            ratio = 1.0
        return ratio

    # ------------------------------------------------------------------
    # Skip path — extractor output 그대로 반환
    # ------------------------------------------------------------------
    @staticmethod
    def _subgraph_to_final_nodes(subgraph: Dict[str, List[str]]) -> List[str]:
        out: List[str] = []
        for t, cols in (subgraph or {}).items():
            if not cols:
                out.append(t)
                continue
            for c in cols:
                out.append(f"{t}.{c}")
        return out

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def refine(
        self,
        query: str,
        subgraph: Dict[str, List[str]],
        db_id: Optional[str] = None,
        tier2_pool: Optional[Any] = None,
        gat_scores: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tcr: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        token_before = AgentUtils.token_snapshot()

        tcr_value = self._compute_tcr(subgraph, metadata, tcr_override=tcr)
        # voluntary skip 결정
        if self.call_mode == "conditional" and tcr_value is not None \
                and tcr_value < self.tcr_threshold:
            voluntary_skipped = True
            final_nodes = self._subgraph_to_final_nodes(subgraph)
            inner_status = None
            inner_info: Dict[str, Any] = {}
            status = "Answerable" if final_nodes else "Unanswerable"
            reasoning = (
                f"[ConditionalFilterWrapper] voluntary skip — "
                f"TCR(q)={tcr_value:.4f} < threshold={self.tcr_threshold:.4f}. "
                f"inner={self.inner_filter_name} NOT called. "
                f"Extractor output preserved ({len(final_nodes)} nodes)."
            )
        else:
            voluntary_skipped = False
            inner_result = self.inner.refine(
                query=query, subgraph=subgraph, db_id=db_id,
                tier2_pool=tier2_pool, gat_scores=gat_scores,
                metadata=metadata, **kwargs,
            )
            final_nodes = list(inner_result.get("final_nodes") or [])
            inner_status = inner_result.get("status")
            inner_info = inner_result.get("filter_info") or {}
            status = inner_status or ("Answerable" if final_nodes else "Unanswerable")
            tcr_str = (
                f"{tcr_value:.4f}" if tcr_value is not None else "unknown (no metadata)"
            )
            why = (
                "always-call mode"
                if self.call_mode == "always"
                else (
                    f"TCR(q)={tcr_str} ≥ threshold={self.tcr_threshold:.4f}"
                    if tcr_value is not None
                    else f"TCR unknown → safe path (call inner)"
                )
            )
            reasoning = (
                f"[ConditionalFilterWrapper] called inner={self.inner_filter_name} "
                f"({why}). inner_status={inner_status}. "
                + (inner_result.get("reasoning") or "")
            )

        token_after = AgentUtils.token_snapshot()
        # Filter LLM call delta 측정 — voluntary skip 시 0 (cost saving 정량)
        token_delta = AgentUtils.token_delta(token_before, token_after)

        self.last_info = AgentUtils.build_filter_info(
            filter_type="ConditionalFilterWrapper",
            input_subgraph=subgraph,
            final_nodes=final_nodes,
            status=status,
            token_before=token_before,
            token_after=token_after,
            t_start=t_start,
            call_mode=self.call_mode,
            tcr_threshold=float(self.tcr_threshold),
            tcr_value=(float(tcr_value) if tcr_value is not None else None),
            tcr_source=("override" if tcr is not None else
                        ("computed" if tcr_value is not None else "unavailable")),
            voluntary_skipped=bool(voluntary_skipped),
            inner_filter_name=self.inner_filter_name,
            inner_status=inner_status,
            inner_called=not voluntary_skipped,
            llm_calls_saved=int(0 if not voluntary_skipped else token_delta.get("calls", 0) == 0),
        )
        # inner filter 의 추가 진단도 보존 (prefix 충돌 회피)
        for k, v in (inner_info or {}).items():
            inner_key = f"inner_{k}" if not k.startswith("filter_") else f"inner_{k}"
            if inner_key not in self.last_info:
                self.last_info[inner_key] = v

        return {
            "status": status,
            "final_nodes": final_nodes,
            "reasoning": reasoning,
            "stats": {
                "call_mode": self.call_mode,
                "tcr_threshold": self.tcr_threshold,
                "tcr_value": tcr_value,
                "voluntary_skipped": bool(voluntary_skipped),
                "inner_called": not voluntary_skipped,
                "inner_filter_name": self.inner_filter_name,
                "n_input_columns": self._count_subgraph_columns(subgraph),
                "n_full_schema_columns": self._count_full_schema_columns(metadata),
                "n_final_nodes": len(final_nodes),
            },
            "filter_info": dict(self.last_info),
        }
