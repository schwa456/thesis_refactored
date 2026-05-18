"""D2 — M4 Bidirectional + FK/PK Connectivity Steiner Closure wrapper (Wave 8).

DECISIONS 2026-05-18 (Wave 8 M4 Bidirectional 발전 4 Direction) §2 D2 정합 +
학술 agent improving_m4_plan_scholar_agent_2026-05-18.md §2 spec 정합.

Composition:
  Step 1  M4 BidirectionalFilter.refine() → forward ∪ backward = m4_union
  Step 2  Steiner closure (직접 FK 또는 1-hop bridge) — Extractor 후보 안에서만
  Step 3  closed_union → sanitize → final_nodes (sorted "table.col")

LLM 호출:
  - M4 가 2× (Forward + Backward, voting 모드 시 4×) — D2 자체는 LLM 0× 추가

핵심 제약 (Wave 8 정합):
  - LLM 입력에 Full Schema 포함 금지 (M4 와 동일)
  - DB DDL 직접 조회 (FK/PK metadata) — LLM 입력 아님
  - Steiner closure 의 검색 범위는 Extractor 후보 (subgraph) 만
  - d2_steiner_closure=False 시 M4 결과 그대로 (디버깅용 backward compat)

측정 메타 (filter_info):
  - d2_added_count: Steiner 가 추가한 FK 컬럼 수
  - d2_added_gold_count: gold 가 들어왔으면 그 중 gold 인 수 (else None)
  - d2_steiner_precision: gold/added (added>0 시), else None
  - d2_variant: "direct_fk" | "bridge_1hop"
  - d2_added_cols: {table: [col, ...]} per-table 추가 컬럼 (debug)
  - m4_filter_info: nested M4 filter_info 전체
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from modules.registry import register
from modules.base import BaseFilter
from modules.filters.agents import AgentUtils
from modules.filters.bidirectional_filter import BidirectionalFilter
from modules.filters.xiyan_filter import XiYanFilter
from modules.filters.steiner_closure import (
    count_added_gold,
    steiner_closure,
    steiner_closure_with_bridge,
)
from utils.logger import get_logger

logger = get_logger(__name__)


@register("filter", "D2SteinerFilter")
class D2SteinerFilter(BaseFilter):
    """M4 Bidirectional + FK Steiner Closure (Wave 8 D2)."""

    # Class-level cache for db_fk_metadata (loaded lazily, shared across instances)
    _DB_FK_METADATA_CACHE: Optional[Dict[str, Any]] = None
    _DB_FK_METADATA_CACHE_PATH: Optional[str] = None

    def __init__(
        self,
        model_name: str,
        max_iteration: int = 1,
        temperature: float = 0.0,
        db_dir: str = "./data/raw/BIRD_dev/dev_databases",
        num_examples: int = 3,
        sanitize_output: bool = True,
        forward_section: str = "recall_biased_mild",
        backward_section: str = "bidirectional_backward",
        bidirectional_forward_prompt_mode: Optional[str] = None,
        bidirectional_forward_voting_strategy: str = "MAJORITY",
        provider: Optional[str] = "glm",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        # D2 신규 옵션
        d2_steiner_closure: bool = True,
        d2_1hop_bridge: bool = False,
        db_fk_metadata_path: str = "data/processed/db_fk_metadata.json",
        **kwargs,
    ):
        self.model_name = model_name
        self.db_dir = db_dir
        self.sanitize_output = bool(sanitize_output)
        self.provider = provider
        # D2 옵션
        self.d2_steiner_closure = bool(d2_steiner_closure)
        self.d2_1hop_bridge = bool(d2_1hop_bridge)
        self.db_fk_metadata_path = db_fk_metadata_path
        self.d2_variant = "bridge_1hop" if self.d2_1hop_bridge else "direct_fk"

        # M4 BidirectionalFilter — composition (M4 self-contained 호출)
        self._m4 = BidirectionalFilter(
            model_name=model_name,
            max_iteration=max_iteration,
            temperature=temperature,
            db_dir=db_dir,
            num_examples=num_examples,
            sanitize_output=sanitize_output,
            forward_section=forward_section,
            backward_section=backward_section,
            bidirectional_forward_prompt_mode=bidirectional_forward_prompt_mode,
            bidirectional_forward_voting_strategy=bidirectional_forward_voting_strategy,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )

        logger.info(
            "Initialized D2SteinerFilter "
            f"(model={model_name}, steiner={self.d2_steiner_closure}, "
            f"variant={self.d2_variant}, db_fk_metadata_path={db_fk_metadata_path}, "
            f"forward_mode={bidirectional_forward_prompt_mode or 'legacy(forward_section)'})"
        )

    # ------------------------------------------------------------------
    # DB FK metadata (lazy load + class-level cache)
    # ------------------------------------------------------------------
    @classmethod
    def _load_db_fk_metadata(cls, path: str) -> Dict[str, Any]:
        if (
            cls._DB_FK_METADATA_CACHE is not None
            and cls._DB_FK_METADATA_CACHE_PATH == path
        ):
            return cls._DB_FK_METADATA_CACHE
        # Local import to avoid heavyweight import-time deps
        from modules.builders.db_fk_extractor import load_db_fk_metadata

        try:
            payload = load_db_fk_metadata(path)
        except RuntimeError as e:
            raise RuntimeError(
                f"[D2SteinerFilter] FK metadata cache 미존재: {e}\n"
                f"Run: python -m modules.builders.db_fk_extractor --out {path}"
            )
        cls._DB_FK_METADATA_CACHE = payload
        cls._DB_FK_METADATA_CACHE_PATH = path
        logger.info(
            f"[D2SteinerFilter] Loaded FK metadata for {len(payload)} DBs from {path}"
        )
        return payload

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _final_nodes_to_dict(
        final_nodes: List[str],
        subgraph: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """['table.col', 'table'] → {table: [col, ...]} (subgraph 기준 정상화)."""
        out: Dict[str, List[str]] = {}
        valid_tables = set(subgraph.keys())
        for node in final_nodes or []:
            if not isinstance(node, str):
                continue
            if "." in node:
                table, col = node.split(".", 1)
                if table in valid_tables:
                    out.setdefault(table, []).append(col)
            else:
                # table-only entry (rare) — subgraph 의 모든 column 으로 확장
                if node in valid_tables:
                    out.setdefault(node, [])
        for t in out:
            out[t] = sorted(set(out[t]))
        return out

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def refine(
        self,
        query: str,
        subgraph: Dict[str, List[str]],
        db_id: Optional[str] = None,
        gold: Optional[Dict[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        empty_tok = {
            "calls": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
        }

        if not subgraph:
            self.last_info = AgentUtils.build_filter_info(
                filter_type="D2SteinerFilter",
                input_subgraph={},
                final_nodes=[],
                status="Unanswerable",
                token_before=empty_tok,
                token_after=empty_tok,
                t_start=t_start,
                model=self.model_name,
                d2_steiner_closure=self.d2_steiner_closure,
                d2_variant=self.d2_variant,
                d2_added_count=0,
                d2_added_gold_count=None,
                d2_steiner_precision=None,
                d2_added_cols={},
                m4_filter_info=None,
            )
            return {
                "status": "Unanswerable",
                "final_nodes": [],
                "reasoning": "Empty input subgraph",
                "filter_info": dict(self.last_info),
            }

        # Step 1: M4 BidirectionalFilter (Forward ∪ Backward)
        m4_result = self._m4.refine(
            query=query, subgraph=subgraph, db_id=db_id, gold=gold, **kwargs
        )
        m4_filter_info = m4_result.get("filter_info") or {}
        m4_final_nodes = m4_result.get("final_nodes") or []
        m4_union_dict = self._final_nodes_to_dict(m4_final_nodes, subgraph)

        # Step 2: Steiner closure (옵션이 켜진 경우만)
        added_only: Dict[str, List[str]] = {}
        closed_union: Dict[str, List[str]] = dict(m4_union_dict)
        skipped_reason: Optional[str] = None

        if not self.d2_steiner_closure:
            skipped_reason = "d2_steiner_closure=False (debug)"
        else:
            try:
                db_fk_metadata = self._load_db_fk_metadata(self.db_fk_metadata_path)
            except RuntimeError as e:
                # 캐시 미존재 → 명확한 에러 (학술 agent 정합 — pre-build 필요)
                logger.error(str(e))
                raise

            per_db_meta = db_fk_metadata.get(db_id or "")
            if not per_db_meta:
                logger.warning(
                    f"[D2SteinerFilter] db_id='{db_id}' not in db_fk_metadata — "
                    f"Steiner closure skipped (M4 결과 그대로 반환)."
                )
                skipped_reason = f"db_id={db_id} not in metadata"
            else:
                try:
                    if self.d2_1hop_bridge:
                        closed_union, added_only = steiner_closure_with_bridge(
                            m4_union_dict, subgraph, per_db_meta, max_bridge_depth=1
                        )
                    else:
                        closed_union, added_only = steiner_closure(
                            m4_union_dict, subgraph, per_db_meta
                        )
                except Exception as e:
                    logger.warning(
                        f"[D2SteinerFilter] Steiner closure failed: {e} — "
                        f"falling back to M4 result"
                    )
                    closed_union = dict(m4_union_dict)
                    added_only = {}
                    skipped_reason = f"steiner exception: {e}"

        # Step 3: sanitize (extractor_output = subgraph 기준)
        # Steiner 결과 자체가 이미 extractor 안에서만 만들어졌으므로 sanitize 는
        # 잔여 안전장치 (m4_union 에 hallucination 이 sanitize 통과한 경우 등).
        if self.sanitize_output:
            sanitized, halluc = XiYanFilter.sanitize_filter_output(
                closed_union, subgraph
            )
        else:
            sanitized = closed_union
            halluc = 0

        final_nodes = sorted(
            f"{t}.{c}" for t, cols in sanitized.items() for c in cols
        )
        status = "Answerable" if final_nodes else "Unanswerable"

        # 측정: added count / gold precision
        added_count = sum(len(v) for v in added_only.values())
        added_gold_count = count_added_gold(added_only, gold)
        steiner_precision: Optional[float] = None
        if added_count > 0 and added_gold_count is not None:
            steiner_precision = added_gold_count / added_count
        elif added_count == 0:
            steiner_precision = None

        # filter_info 구성 (M4 filter_info 도 nested 포함)
        self.last_info = AgentUtils.build_filter_info(
            filter_type="D2SteinerFilter",
            input_subgraph=subgraph,
            final_nodes=final_nodes,
            status=status,
            token_before=empty_tok,
            token_after=empty_tok,
            t_start=t_start,
            model=self.model_name,
            d2_steiner_closure=self.d2_steiner_closure,
            d2_variant=self.d2_variant,
            d2_added_count=added_count,
            d2_added_gold_count=added_gold_count,
            d2_steiner_precision=steiner_precision,
            d2_added_cols=added_only,
            d2_skipped_reason=skipped_reason,
            db_fk_metadata_path=self.db_fk_metadata_path,
            m4_filter_info=m4_filter_info,
            sanitize_output=self.sanitize_output,
            hallucination_removed_count=halluc,
        )

        return {
            "status": status,
            "final_nodes": final_nodes,
            "reasoning": (
                f"[D2 Steiner] variant={self.d2_variant}, "
                f"m4_union={sum(len(v) for v in m4_union_dict.values())}, "
                f"added={added_count}, "
                f"added_gold={added_gold_count}, "
                f"steiner_precision={steiner_precision}, "
                f"skipped={skipped_reason or 'none'}; "
                f"[M4] {m4_result.get('reasoning', '')}"
            ),
            "stats": {
                "d2_added_count": added_count,
                "d2_added_gold_count": added_gold_count,
                "d2_steiner_precision": steiner_precision,
                "d2_added_cols": added_only,
                "d2_variant": self.d2_variant,
                "m4_stats": m4_result.get("stats"),
            },
            "filter_info": dict(self.last_info),
        }
