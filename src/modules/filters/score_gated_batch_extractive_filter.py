"""SGBE — Score-Gated Batch Extractive Filter.

XiYan 의 prune-only recall 손실 (~0.15) 을 column-level routing 으로 해소하는
하이브리드 필터.

설계 (planning/filtering_suggestion_by_scholar_agent_2026-05-12.md §"Score-Gated
Batch Extractive Filter"):

  Step 0  Structural Hard Keep        0 LLM calls, instant
          S_struct = FK/PK columns in S_pcst

  Step 1  Score-Gate                  0 LLM calls, O(n)
          θ_keep = 0.65, θ_drop = 0.40
          S_keep_hard = {v | s_v ≥ θ_keep}     -> 즉시 keep
          S_drop_hard = {v | s_v < θ_drop}     -> 즉시 drop
          S_uncertain = {v | θ_drop ≤ s_v < θ_keep}  -> LLM 대상

  Step 2  Extractive LLM (per-column binary)   1 LLM call, S_uncertain 만
          출력: S_lm_keep ⊆ S_uncertain

  final_nodes = S_keep_hard ∪ S_lm_keep ∪ S_struct

근거: Yuan 2025 (TP mean 0.7108 / Filter✗ mean 0.6394 / TN mean ~0.40);
Glass 2025 (extractive 방식); Talaei 2024 (CHESS hardcode rule);
Hoang 2025 (lightweight LLM input).
"""
import os
import json
import re
import sqlite3
import time
from typing import Dict, List, Any, Optional, Set, Tuple

import numpy as np

from modules.registry import register
from modules.base import BaseFilter
from modules.filters.agents import AgentUtils
from prompts.prompt_manager import PromptManager
from utils.logger import get_logger

logger = get_logger(__name__)

_VALID_STEP_MODES: Tuple[str, ...] = ("step_0", "step_0+1", "step_0+1+2")


@register("filter", "ScoreGatedBatchExtractiveFilter")
class ScoreGatedBatchExtractiveFilter(BaseFilter):
    """Score-Gated Batch Extractive Filter (SGBE).

    PCST subgraph 의 column 들을 GAT score 분포에 따라 3-way routing 한 뒤,
    중간 confidence 구간만 LLM 으로 보내 per-column binary 판단을 받는 hybrid filter.

    Args:
        step_mode: SGBE 의 어느 step 까지 진행할지.
            - "step_0":       FK/PK Hardcode 만 (LLM call 0)
                              → Phase 5 ablation 의 step contribution decomposition
            - "step_0+1":     Step 0 + Score-Gate (LLM call 0, S_uncertain 전부 drop)
                              → Phase 3 calibration sweep 의 "LLM call 없는 Step 0+1 만 평가"
            - "step_0+1+2"    Full SGBE (default, LLM call 1)
                              → Phase 4 final SGBE 평가
        score_collapse_threshold: candidate score 들의 std 가 이 값 미만이면
            score 분포가 collapse 한 것으로 간주, 모두 S_uncertain 로 라우팅하여
            LLM 판단에 위임 (학술 Agent §"한계" 의 over-smoothing era 대비).
            None 이면 collapse 감지 비활성화.
    """

    def __init__(
        self,
        model_name: str = "zai-org/glm-4.7",
        theta_keep: float = 0.65,
        theta_drop: float = 0.40,
        temperature: float = 0.0,
        db_dir: str = "./data/raw/BIRD_dev/dev_databases",
        num_examples: int = 3,
        fk_pk_hardcode: bool = True,
        step_mode: str = "step_0+1+2",
        score_collapse_threshold: Optional[float] = 0.05,
        provider: Optional[str] = "glm",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        if theta_drop > theta_keep:
            raise ValueError(
                f"theta_drop ({theta_drop}) must be ≤ theta_keep ({theta_keep})."
            )
        if step_mode not in _VALID_STEP_MODES:
            raise ValueError(
                f"step_mode='{step_mode}' invalid. "
                f"Expected one of {_VALID_STEP_MODES}."
            )
        self.model_name = model_name
        self.theta_keep = float(theta_keep)
        self.theta_drop = float(theta_drop)
        self.temperature = float(temperature)
        self.db_dir = db_dir
        self.num_examples = max(0, int(num_examples))
        self.fk_pk_hardcode = bool(fk_pk_hardcode)
        self.step_mode = step_mode
        # score_collapse_threshold=None 이면 collapse 감지 비활성화 (legacy 동작).
        self.score_collapse_threshold: Optional[float] = (
            None if score_collapse_threshold is None
            else float(score_collapse_threshold)
        )
        self.provider = provider
        self.prompt_manager = PromptManager()
        # step_mode="step_0" 은 LLM 호출이 절대 없으므로 client init 도 lazy 처리하지 않고
        # 일관성 위해 동일하게 호출 (cost 무시 가능, env 미설정 시 to-be-raised).
        self.client = self._make_llm_client(
            api_key=api_key, base_url=base_url, provider=provider,
        )
        logger.info(
            "Initialized ScoreGatedBatchExtractiveFilter "
            f"(model={model_name}, provider={provider or 'auto'}, "
            f"step_mode={self.step_mode}, "
            f"θ_keep={self.theta_keep}, θ_drop={self.theta_drop}, "
            f"fk_pk_hardcode={self.fk_pk_hardcode}, "
            f"score_collapse_threshold={self.score_collapse_threshold}, "
            f"num_examples={self.num_examples})"
        )

    # ------------------------------------------------------------------
    # Step 0: Structural Hard Keep (FK/PK)
    # ------------------------------------------------------------------
    def _extract_fk_columns(
        self, subgraph: Dict[str, List[str]], metadata: Optional[Dict[str, Any]],
    ) -> Set[str]:
        """metadata['fk_to_id'] 키 (`"a.x->b.y"`) 양쪽 column 을 추출, subgraph 내만 유지."""
        if not metadata:
            return set()
        fk_to_id = metadata.get("fk_to_id", {}) or {}
        subgraph_keys = {
            f"{t}.{c}" for t, cols in (subgraph or {}).items() for c in (cols or [])
        }
        fk_columns: Set[str] = set()
        for fk_key in fk_to_id.keys():
            if not isinstance(fk_key, str) or "->" not in fk_key:
                continue
            left, right = fk_key.split("->", 1)
            for side in (left.strip(), right.strip()):
                if "." not in side:
                    continue
                if side in subgraph_keys:
                    fk_columns.add(side)
        return fk_columns

    def _extract_pk_columns(
        self, subgraph: Dict[str, List[str]], db_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Set[str]:
        """PK columns 추출. metadata['primary_keys'] 우선, 없으면 SQLite PRAGMA 직접 조회 (best effort).

        PRAGMA table_info row 형식: (cid, name, type, notnull, dflt_value, pk).
        builders 가 현재 PK 를 metadata 에 노출하지 않으므로 db_id 가 주어지면 직접 fetch.
        """
        pk_columns: Set[str] = set()
        # metadata fallback (builder 가 추후 노출하면 사용)
        if metadata:
            md_pk = metadata.get("primary_keys")
            if isinstance(md_pk, (list, set, tuple)):
                pk_columns.update(p for p in md_pk if isinstance(p, str) and "." in p)
            elif isinstance(md_pk, dict):
                for tbl, cols in md_pk.items():
                    for c in (cols or []):
                        pk_columns.add(f"{tbl}.{c}")

        if pk_columns or not db_id:
            return pk_columns

        # PRAGMA 직접 조회 (best effort, exception silent)
        db_path = os.path.join(self.db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            return pk_columns
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            for tbl in (subgraph or {}).keys():
                safe = tbl.replace("'", "''")
                try:
                    cursor.execute(f"PRAGMA table_info('{safe}');")
                    for row in cursor.fetchall():
                        # row: (cid, name, type, notnull, dflt_value, pk)
                        if len(row) >= 6 and int(row[5] or 0) > 0:
                            pk_columns.add(f"{tbl}.{row[1]}")
                except sqlite3.Error as e:
                    logger.debug(f"[SGBE] PRAGMA table_info failed on {tbl}: {e}")
            conn.close()
        except Exception as e:
            logger.debug(f"[SGBE] PK fetch failed for {db_id}: {e}")

        subgraph_keys = {
            f"{t}.{c}" for t, cols in (subgraph or {}).items() for c in (cols or [])
        }
        return {p for p in pk_columns if p in subgraph_keys}

    def _structural_hard_keep(
        self,
        subgraph: Dict[str, List[str]],
        db_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Set[str]:
        if not self.fk_pk_hardcode:
            return set()
        fk = self._extract_fk_columns(subgraph, metadata)
        pk = self._extract_pk_columns(subgraph, db_id, metadata)
        return fk | pk

    # ------------------------------------------------------------------
    # Step 1: Score-Gate
    # ------------------------------------------------------------------
    def _score_gate(
        self,
        candidates: List[str],
        gat_scores: Optional[Dict[str, float]],
    ) -> Tuple[Set[str], Set[str], Set[str], Dict[str, Any]]:
        """Return (S_keep_hard, S_drop_hard, S_uncertain, diag).

        gat_scores 가 None 이거나 비어 있으면 모든 candidates 를 S_uncertain 로 처리
        (XiYan-equivalent fallback, score 정보 없을 때 안전망).

        Score collapse fallback (학술 Agent §"한계"):
        - candidate score 들의 표준편차가 self.score_collapse_threshold 미만이면
          분포가 collapse 한 상황 (over-smoothing era V4 score 균일화 등)으로 간주.
        - 이 경우 θ_keep / θ_drop 이 의미 없으므로 모든 column 을 S_uncertain
          으로 라우팅하여 LLM 판단에 위임 (XiYan-equivalent recall-safe path).
        """
        diag: Dict[str, Any] = {
            "score_gate_active": bool(gat_scores),
            "score_collapse_detected": False,
            "score_std": None,
        }
        if not gat_scores:
            return set(), set(), set(candidates), diag

        observed_scores: List[float] = []
        for v in candidates:
            s = self._lookup_score(v, gat_scores)
            if s is not None:
                observed_scores.append(s)
        if observed_scores:
            score_std = float(np.std(np.asarray(observed_scores, dtype=np.float64)))
            diag["score_std"] = score_std
            if (
                self.score_collapse_threshold is not None
                and score_std < self.score_collapse_threshold
            ):
                logger.warning(
                    f"[SGBE] Score collapse detected (std={score_std:.4f} < "
                    f"threshold={self.score_collapse_threshold}). "
                    "Falling back to XiYan-equivalent: all candidates → S_uncertain."
                )
                diag["score_collapse_detected"] = True
                return set(), set(), set(candidates), diag

        keep_hard: Set[str] = set()
        drop_hard: Set[str] = set()
        uncertain: Set[str] = set()
        for v in candidates:
            s = self._lookup_score(v, gat_scores)
            if s is None:
                # score 미존재 column 은 안전하게 uncertain 으로 라우팅
                uncertain.add(v)
                continue
            if s >= self.theta_keep:
                keep_hard.add(v)
            elif s < self.theta_drop:
                drop_hard.add(v)
            else:
                uncertain.add(v)
        return keep_hard, drop_hard, uncertain, diag

    @staticmethod
    def _lookup_score(node: str, gat_scores: Dict[str, float]) -> Optional[float]:
        """gat_scores 의 키 형식이 'table.col' 또는 'col' 또는 'table' 인 경우 모두 try."""
        if node in gat_scores:
            try:
                return float(gat_scores[node])
            except (TypeError, ValueError):
                return None
        # table.col -> col 만으로도 시도 (selector 가 column-only 키를 줄 수 있음)
        if "." in node:
            _, col = node.split(".", 1)
            if col in gat_scores:
                try:
                    return float(gat_scores[col])
                except (TypeError, ValueError):
                    return None
        return None

    # ------------------------------------------------------------------
    # Step 2: Extractive LLM (per-column binary, S_uncertain 만)
    # ------------------------------------------------------------------
    def _extractive_binary_call(
        self,
        query: str,
        uncertain: Set[str],
        db_id: Optional[str],
    ) -> Tuple[Set[str], Dict[str, Any]]:
        """S_uncertain 의 column 들에 대해 per-column binary 판단."""
        if not uncertain:
            return set(), {"called": False, "reason": "empty_uncertain"}

        # column metadata + value samples 구성
        col_blocks, col_id_to_node = self._build_column_blocks(uncertain, db_id)
        if not col_blocks:
            # value retrieval 도 metadata 도 없으면 그냥 column 이름만 라벨
            for idx, node in enumerate(sorted(uncertain), start=1):
                col_blocks.append(f"{idx}. {node}")
                col_id_to_node[str(idx)] = node
                col_id_to_node[node] = node

        candidate_str = "\n".join(col_blocks)

        prompt = self.prompt_manager.load_prompt(
            file_name='filter',
            section='sgbe_extractive',
            query=query,
            candidate_str=candidate_str,
        )

        response = self.client.generate_text(
            prompt=prompt, model=self.model_name, temperature=self.temperature,
        )
        logger.debug(f"[SGBE] LLM response: {response[:500]}")

        lm_keep, parse_info = self._parse_binary_response(
            response, uncertain, col_id_to_node,
        )
        parse_info["called"] = True
        parse_info["response_preview"] = response[:200]
        return lm_keep, parse_info

    def _build_column_blocks(
        self,
        uncertain: Set[str],
        db_id: Optional[str],
    ) -> Tuple[List[str], Dict[str, str]]:
        """각 column 에 대해 'idx. table.column (samples: v1, v2, ...)' 블록 + lookup map.

        Lookup map 은 LLM 이 idx / table.col / 단순 col 중 무엇으로 응답해도 매칭되게 함.
        """
        blocks: List[str] = []
        lookup: Dict[str, str] = {}

        # table 별 column 묶기 (단일 connection 재사용)
        by_table: Dict[str, List[str]] = {}
        for node in sorted(uncertain):
            if "." in node:
                t, c = node.split(".", 1)
                by_table.setdefault(t, []).append(c)
            else:
                by_table.setdefault(node, [])

        conn = None
        if db_id and self.num_examples > 0:
            db_path = os.path.join(self.db_dir, db_id, f"{db_id}.sqlite")
            if os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path)
                except Exception as e:
                    logger.debug(f"[SGBE] DB connect failed for {db_id}: {e}")
                    conn = None

        idx = 1
        for table, cols in by_table.items():
            if not cols:
                node = table
                blocks.append(f"{idx}. {node}  (table-level node, no column)")
                lookup[str(idx)] = node
                lookup[node] = node
                idx += 1
                continue
            for col in cols:
                node = f"{table}.{col}"
                samples_str = ""
                if conn is not None:
                    try:
                        cur = conn.cursor()
                        cur.execute(
                            f'SELECT DISTINCT "{col}" FROM "{table}" '
                            f'WHERE "{col}" IS NOT NULL LIMIT {self.num_examples}'
                        )
                        values = [str(r[0]) for r in cur.fetchall()]
                        if values:
                            samples_str = f" samples=[{', '.join(values)}]"
                    except Exception as e:
                        logger.debug(f"[SGBE] sample fetch failed for {node}: {e}")
                blocks.append(f"{idx}. {node}{samples_str}")
                lookup[str(idx)] = node
                lookup[node] = node
                lookup[col] = node  # column-only fallback (collision 시 마지막이 이김 — 통상 unique)
                idx += 1

        if conn is not None:
            conn.close()
        return blocks, lookup

    def _parse_binary_response(
        self,
        response: str,
        uncertain: Set[str],
        lookup: Dict[str, str],
    ) -> Tuple[Set[str], Dict[str, Any]]:
        """JSON list 응답을 파싱해 keep=True 인 column 만 집합으로 반환.

        Fallback: 파싱 실패 시 S_uncertain 전부 keep (recall-safe, a05_01 교훈).
        """
        clean = response.replace("```json", "").replace("```", "").strip()

        # 우선 list 추출 (단계 A)
        list_start = clean.find('[')
        list_end = clean.rfind(']')
        decisions: List[Dict[str, Any]] = []
        parse_ok = False
        if list_start != -1 and list_end != -1 and list_start < list_end:
            try:
                decisions = json.loads(clean[list_start : list_end + 1])
                if isinstance(decisions, list):
                    parse_ok = True
            except json.JSONDecodeError as e:
                logger.warning(f"[SGBE] JSON list parse failed: {e}")

        if not parse_ok:
            # 한 줄 패턴 fallback ('col: yes/no' or '"col": true/false')
            decisions = self._line_pattern_fallback(clean)
            parse_ok = bool(decisions)

        if not parse_ok:
            logger.warning(
                "[SGBE] Could not parse any binary decisions. "
                "Recall-safe fallback: keep all S_uncertain."
            )
            return set(uncertain), {"parse_ok": False, "kept_via_fallback": True}

        kept: Set[str] = set()
        unresolved = 0
        for d in decisions:
            if not isinstance(d, dict):
                continue
            keep_val = d.get("keep")
            if isinstance(keep_val, str):
                keep_val = keep_val.strip().lower() in {"yes", "true", "1", "y"}
            if not keep_val:
                continue
            col_key = d.get("column") or d.get("col") or d.get("id") or d.get("name")
            if col_key is None:
                continue
            node = lookup.get(str(col_key).strip())
            if node and node in uncertain:
                kept.add(node)
            else:
                unresolved += 1

        return kept, {
            "parse_ok": True,
            "kept_via_fallback": False,
            "kept_count": len(kept),
            "decisions_count": len(decisions),
            "unresolved_keys": unresolved,
        }

    @staticmethod
    def _line_pattern_fallback(text: str) -> List[Dict[str, Any]]:
        """단일 list 파싱 실패 시 줄 단위 'col: yes' 패턴 추출 (best effort).

        지원 형식:
          - "table.col: yes (reason)"
          - "1. table.col -> keep"
          - '"table.col": true'
        """
        out: List[Dict[str, Any]] = []
        line_re = re.compile(
            r'(?:["\'`]?(?P<col>[A-Za-z_][\w.]*)["\'`]?)\s*[:=>\-]+\s*'
            r'(?P<verdict>yes|no|true|false|keep|drop|y|n)\b',
            re.IGNORECASE,
        )
        for m in line_re.finditer(text):
            verdict = m.group("verdict").lower()
            out.append({
                "column": m.group("col"),
                "keep": verdict in {"yes", "true", "keep", "y"},
            })
        return out

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def refine(
        self,
        query: str,
        subgraph: Dict[str, List[str]],
        db_id: Optional[str] = None,
        tier2_pool: Optional[Any] = None,  # unused (SGBE 는 Tier-1 routing 만)
        gat_scores: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()

        # 빈 입력 short-circuit
        if not subgraph:
            empty_tokens = {"calls": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
            self.last_info = AgentUtils.build_filter_info(
                filter_type="ScoreGatedBatchExtractiveFilter",
                input_subgraph={},
                final_nodes=[],
                status="Unanswerable",
                token_before=empty_tokens,
                token_after=empty_tokens,
                t_start=t_start,
                model=self.model_name,
                theta_keep=self.theta_keep,
                theta_drop=self.theta_drop,
                step_mode=self.step_mode,
                step_keep_hard=0,
                step_drop_hard=0,
                step_uncertain=0,
                step_struct=0,
                step_lm_keep=0,
                llm_called=False,
                score_collapse_detected=False,
            )
            return {
                "status": "Unanswerable",
                "final_nodes": [],
                "reasoning": "Empty input subgraph",
                "stats": {
                    "step_mode": self.step_mode,
                    "keep_hard": 0, "drop_hard": 0, "uncertain": 0,
                    "lm_keep": 0, "struct": 0,
                    "score_collapse_detected": False,
                },
                "filter_info": dict(self.last_info),
            }

        token_before = AgentUtils.token_snapshot()
        candidates: List[str] = [
            f"{t}.{c}" for t, cols in subgraph.items() for c in (cols or [])
        ]
        # table-only 노드 (cols 없는 경우)
        candidates.extend(t for t, cols in subgraph.items() if not cols)

        # Step 0 — 항상 실행 (3 mode 공통)
        s_struct = self._structural_hard_keep(subgraph, db_id, metadata)

        # Default zero-counts for mode-specific skips
        s_keep_hard: Set[str] = set()
        s_drop_hard: Set[str] = set()
        s_uncertain: Set[str] = set()
        s_lm_keep: Set[str] = set()
        score_diag: Dict[str, Any] = {
            "score_gate_active": False,
            "score_collapse_detected": False,
            "score_std": None,
        }
        parse_info: Dict[str, Any] = {"called": False, "reason": "skipped_by_step_mode"}

        if self.step_mode == "step_0":
            # Step 0 만 — S_struct 만 keep, 나머지는 모두 drop
            pass
        else:
            # step_0+1 또는 step_0+1+2: Step 1 실행
            s_keep_hard, s_drop_hard, s_uncertain, score_diag = self._score_gate(
                candidates, gat_scores,
            )
            if self.step_mode == "step_0+1+2":
                # Step 2 — LLM
                s_lm_keep, parse_info = self._extractive_binary_call(
                    query, s_uncertain, db_id,
                )
            # step_0+1: s_lm_keep 그대로 빈 set (S_uncertain 전부 drop)

        # Output 합치기
        final_set: Set[str] = s_keep_hard | s_lm_keep | s_struct
        final_nodes: List[str] = sorted(final_set)
        status = "Answerable" if final_nodes else "Unanswerable"

        token_after = AgentUtils.token_snapshot()
        stats = {
            "step_mode": self.step_mode,
            "keep_hard": len(s_keep_hard),
            "drop_hard": len(s_drop_hard),
            "uncertain": len(s_uncertain),
            "lm_keep": len(s_lm_keep),
            "struct": len(s_struct),
            "score_collapse_detected": bool(score_diag.get("score_collapse_detected")),
        }
        reasoning = self._build_reasoning(
            s_keep_hard, s_drop_hard, s_uncertain, s_lm_keep, s_struct,
            parse_info, self.step_mode, score_diag,
        )
        self.last_info = AgentUtils.build_filter_info(
            filter_type="ScoreGatedBatchExtractiveFilter",
            input_subgraph=subgraph,
            final_nodes=final_nodes,
            status=status,
            token_before=token_before,
            token_after=token_after,
            t_start=t_start,
            model=self.model_name,
            theta_keep=self.theta_keep,
            theta_drop=self.theta_drop,
            fk_pk_hardcode=self.fk_pk_hardcode,
            step_mode=self.step_mode,
            step_keep_hard=stats["keep_hard"],
            step_drop_hard=stats["drop_hard"],
            step_uncertain=stats["uncertain"],
            step_struct=stats["struct"],
            step_lm_keep=stats["lm_keep"],
            llm_called=bool(parse_info.get("called", False)),
            parse_ok=bool(parse_info.get("parse_ok", True)),
            kept_via_fallback=bool(parse_info.get("kept_via_fallback", False)),
            score_gate_active=bool(score_diag.get("score_gate_active")),
            score_collapse_detected=bool(score_diag.get("score_collapse_detected")),
            score_std=score_diag.get("score_std"),
        )
        return {
            "status": status,
            "final_nodes": final_nodes,
            "reasoning": reasoning,
            "stats": stats,
            "filter_info": dict(self.last_info),
        }

    @staticmethod
    def _build_reasoning(
        keep_hard: Set[str], drop_hard: Set[str], uncertain: Set[str],
        lm_keep: Set[str], struct: Set[str], parse_info: Dict[str, Any],
        step_mode: str, score_diag: Dict[str, Any],
    ) -> str:
        collapse = (
            " [score-collapse → XiYan-equivalent fallback]"
            if score_diag.get("score_collapse_detected") else ""
        )
        if step_mode == "step_0":
            return (
                f"SGBE[step_0]: FK/PK hardcode only — "
                f"|S_struct|={len(struct)}, all non-struct dropped."
            )
        if step_mode == "step_0+1":
            return (
                f"SGBE[step_0+1]: no LLM. "
                f"|S_keep_hard|={len(keep_hard)}, "
                f"|S_drop_hard|={len(drop_hard)}, "
                f"|S_uncertain (dropped)|={len(uncertain)}, "
                f"|S_struct|={len(struct)}.{collapse}"
            )
        # step_0+1+2 (full)
        return (
            f"SGBE[step_0+1+2]: "
            f"|S_keep_hard|={len(keep_hard)}, "
            f"|S_drop_hard|={len(drop_hard)}, "
            f"|S_uncertain|={len(uncertain)} -> |S_lm_keep|={len(lm_keep)} "
            f"(LLM {'called' if parse_info.get('called') else 'skipped'}, "
            f"parse_ok={parse_info.get('parse_ok', True)}), "
            f"|S_struct|={len(struct)}.{collapse}"
        )
