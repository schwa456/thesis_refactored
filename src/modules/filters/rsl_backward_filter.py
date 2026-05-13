"""RSL-SQL Backward Filter (Direction A, Cao 2024).

학술 Agent Phase 3 (2026-05-13) Direction A GO 확정 → 학위 논문 §V.5.x 의
Filter Dominance 8 → 9 axis 진입 후보. Detailed spec:
[planning/DECISIONS.md "2026-05-13 학술 Agent Phase 3 Response — Direction A
GO 확정" entry §"Direction A `RSLBackwardFilter` Implementation Spec"].

Pipeline (per query, 2 LLM calls):

  Step 1  XiYan forward (의존성)
          S_fwd = XiYanFilter.refine(query, subgraph, db_id).final_nodes

  Step 2  Preliminary SQL backward (신규, GLM 4.7)
          input: full DB schema (PCST subgraph 가 아님) + query + (선택) evidence
          output: prelim_sql (plain SQL string)
          L_bwd = sqlglot.extract_columns(prelim_sql, col-only-distinct)
                  -- Phase 2 bug fix 후 normalization 정합 (alias-distinct -> col-only)

  Step 3  S_restore + 조건부 DB-level guard
          S_restore_col = L_bwd - col_only(S_fwd)
          if db_id in risky_dbs:   # 학술 Agent margin 1.07× 좁아짐 caveat 대비
              S_restore = ∅        # toxicology 같은 outlier DB skip
          else:
              S_restore = expand_to_full_paths(S_restore_col, full_schema)
                          # col_name 이 여러 table 에 있으면 모두 후보 — Cao 2024 RSL-SQL 정합

  Step 4  S_struct FK/PK hardcode (CHESS Talaei 2024)
          S_struct = FK + PK columns in subgraph

  Output: final_nodes = S_fwd ∪ S_restore ∪ S_struct

학술 Agent Phase 2 (fix) 측정:
  - mean(S_restore_precision)        = 0.6434   (>= 0.60 PASS, margin 1.07×)
  - mean(Δrecall_union vs fwd)        = +0.0771  (>= +0.05 PASS, margin 1.54×)
  - mean(recall_gained_by_restore)    = 0.5709   ("forward 가 놓친 gold 의 57% 회복")

Caveat (planner Phase 3 §"Direction A 배포 Margin Caveat"):
  - margin 1.20× -> 1.07× 좁아짐 (Phase 2 bug fix 후).
  - toxicology 외 추가 low-precision DB 발견 시 risky_dbs 갱신 권장.
"""
import os
import re
import sqlite3
import time
from typing import Dict, List, Any, Optional, Set, Tuple

from modules.registry import register
from modules.base import BaseFilter
from modules.filters.agents import AgentUtils
from modules.filters.xiyan_filter import XiYanFilter
from prompts.prompt_manager import PromptManager
from utils.logger import get_logger

logger = get_logger(__name__)


def _extract_columns_from_sql(sql_text: str) -> Set[str]:
    """sqlglot 으로 SQL 에서 column 이름 집합을 추출 (col-only-distinct).

    Phase 2 bug fix 의 정합: alias 무시, table prefix 무시, column 이름만 distinct.
    Parse 실패 시 빈 set (recall-safe — restore path 가 비어지지만 forward 는 보존).
    """
    if not sql_text or not isinstance(sql_text, str):
        return set()
    try:
        import sqlglot
        from sqlglot import expressions as exp
    except ImportError:
        logger.warning("[RSLBackward] sqlglot not available — returning empty column set")
        return set()

    cleaned = sql_text.strip()
    # ```sql ... ``` fences 제거
    cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    if not cleaned:
        return set()

    # sqlglot 가 임의 텍스트도 partial-parse 하여 spurious column 을 만들 수 있음.
    # SQL DML/DDL keyword 가 없으면 SQL 이 아니므로 즉시 빈 set (recall-safe — forward 보존).
    if not re.search(
        r"\b(SELECT|WITH|INSERT|UPDATE|DELETE|REPLACE)\b",
        cleaned, re.IGNORECASE,
    ):
        return set()

    cols: Set[str] = set()
    try:
        # dialect=sqlite — BIRD-Dev DB 가 SQLite. 실패 시 generic parse.
        tree = sqlglot.parse_one(cleaned, dialect="sqlite", error_level="ignore")
        if tree is None:
            return cols
        for node in tree.find_all(exp.Column):
            name = node.name
            if name and name != "*":
                cols.add(name)
    except Exception as e:
        logger.warning(f"[RSLBackward] sqlglot parse failed: {e}")
        try:
            # generic fallback
            tree = sqlglot.parse_one(cleaned, error_level="ignore")
            if tree is not None:
                for node in tree.find_all(exp.Column):
                    name = node.name
                    if name and name != "*":
                        cols.add(name)
        except Exception as e2:
            logger.warning(f"[RSLBackward] sqlglot generic parse failed: {e2}")
    return cols


@register("filter", "RSLBackwardFilter")
class RSLBackwardFilter(BaseFilter):
    """RSL-SQL Backward Filter — XiYan forward + preliminary SQL backward + restore."""

    def __init__(
        self,
        model_name: str = "zai-org/glm-4.7",
        temperature: float = 0.0,
        # XiYan forward params (Step 1)
        xiyan_max_iteration: int = 1,
        xiyan_model_name: Optional[str] = None,  # None 이면 model_name 공유
        xiyan_num_examples: int = 3,
        # DB / schema
        db_dir: str = "./data/raw/BIRD_dev/dev_databases",
        num_examples: int = 3,
        # FK/PK hardcode (Step 4)
        fk_pk_hardcode: bool = True,
        # DB-level precision guard (Step 3)
        risky_dbs: Optional[List[str]] = None,
        # LLM provider
        provider: Optional[str] = "glm",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.temperature = float(temperature)
        self.db_dir = db_dir
        self.num_examples = max(0, int(num_examples))
        self.fk_pk_hardcode = bool(fk_pk_hardcode)
        self.risky_dbs: Set[str] = set(risky_dbs or [])
        self.provider = provider
        self.prompt_manager = PromptManager()
        self.client = self._make_llm_client(
            api_key=api_key, base_url=base_url, provider=provider,
        )

        # XiYan forward (동일 provider 공유, model 만 다를 수 있음)
        self.xiyan = XiYanFilter(
            model_name=xiyan_model_name or model_name,
            max_iteration=xiyan_max_iteration,
            temperature=self.temperature,
            db_dir=db_dir,
            num_examples=xiyan_num_examples,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )
        logger.info(
            "Initialized RSLBackwardFilter "
            f"(model={model_name}, provider={provider or 'auto'}, "
            f"xiyan_max_iter={xiyan_max_iteration}, "
            f"risky_dbs={sorted(self.risky_dbs) if self.risky_dbs else 'none'}, "
            f"fk_pk_hardcode={self.fk_pk_hardcode})"
        )

    # ------------------------------------------------------------------
    # Step 2 helpers — full schema build + preliminary SQL
    # ------------------------------------------------------------------
    def _build_full_schema_map(
        self,
        db_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        fallback_subgraph: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """Return {table: [col, ...]} for ENTIRE DB (not PCST subgraph).

        우선순위:
          1) metadata['col_to_id'] keys ("table.col") — builder 가 노출하는 표준 형식
          2) SQLite PRAGMA table_info (db_id + db_dir 필요)
          3) fallback_subgraph (PCST 출력) — 마지막 수단, full schema 가 아님
        """
        # 1) metadata
        if metadata:
            col_to_id = metadata.get("col_to_id") or {}
            if col_to_id:
                full: Dict[str, List[str]] = {}
                for key in col_to_id.keys():
                    if not isinstance(key, str) or "." not in key:
                        continue
                    t, c = key.split(".", 1)
                    full.setdefault(t, []).append(c)
                if full:
                    # table_to_id 가 있으면 empty-cols table 도 노출
                    table_to_id = metadata.get("table_to_id") or {}
                    for t in table_to_id.keys():
                        full.setdefault(t, [])
                    return full

        # 2) SQLite PRAGMA
        if db_id:
            db_path = os.path.join(self.db_dir, db_id, f"{db_id}.sqlite")
            if os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path)
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name NOT LIKE 'sqlite_%';"
                    )
                    tables = [r[0] for r in cur.fetchall()]
                    full = {}
                    for t in tables:
                        safe = t.replace("'", "''")
                        try:
                            cur.execute(f"PRAGMA table_info('{safe}');")
                            full[t] = [r[1] for r in cur.fetchall()]
                        except sqlite3.Error as e:
                            logger.debug(f"[RSLBackward] PRAGMA failed on {t}: {e}")
                            full[t] = []
                    conn.close()
                    return full
                except Exception as e:
                    logger.warning(f"[RSLBackward] full schema fetch failed ({db_id}): {e}")

        # 3) fallback — PCST subgraph
        logger.warning(
            "[RSLBackward] Falling back to PCST subgraph as full schema "
            "(restore search space limited)."
        )
        return {t: list(cols or []) for t, cols in (fallback_subgraph or {}).items()}

    def _build_full_schema_str(
        self,
        full_schema: Dict[str, List[str]],
        db_id: Optional[str],
        fk_keys: List[str],
    ) -> str:
        """Compose full-schema text for the preliminary-SQL prompt.

        Format mirrors XiYan 의 M-Schema 양식이되 PCST subgraph 가 아닌 전체 schema +
        FK relations 까지 포함. Value samples 는 num_examples > 0 에서만.
        """
        lines: List[str] = []
        # FK relations 먼저 노출
        if fk_keys:
            lines.append("# Foreign Keys:")
            for fk in fk_keys[:200]:  # safety cap
                lines.append(f"  {fk}")
            lines.append("")

        db_path = (
            os.path.join(self.db_dir, db_id, f"{db_id}.sqlite") if db_id else None
        )
        conn = None
        if db_path and os.path.exists(db_path) and self.num_examples > 0:
            try:
                conn = sqlite3.connect(db_path)
            except Exception as e:
                logger.debug(f"[RSLBackward] DB connect failed: {e}")
                conn = None

        for tbl, cols in full_schema.items():
            lines.append(f"# Table: {tbl}")
            col_details: List[str] = []
            for col in cols:
                col_str = col
                if conn is not None:
                    try:
                        cur = conn.cursor()
                        cur.execute(
                            f'SELECT DISTINCT "{col}" FROM "{tbl}" '
                            f'WHERE "{col}" IS NOT NULL LIMIT {self.num_examples}'
                        )
                        samples = [str(r[0]) for r in cur.fetchall()]
                        if samples:
                            col_str += f" (Examples: {', '.join(samples)})"
                    except Exception as e:
                        logger.debug(f"[RSLBackward] sample fetch failed {tbl}.{col}: {e}")
                col_details.append(col_str)
            lines.append("  Columns: " + " | ".join(col_details))
            lines.append("")

        if conn is not None:
            conn.close()
        return "\n".join(lines)

    @staticmethod
    def _fk_keys_from_metadata(metadata: Optional[Dict[str, Any]]) -> List[str]:
        if not metadata:
            return []
        fk_to_id = metadata.get("fk_to_id") or {}
        return [k for k in fk_to_id.keys() if isinstance(k, str) and "->" in k]

    def _generate_preliminary_sql(
        self, query: str, schema_str: str, evidence: Optional[str],
    ) -> str:
        """LLM 호출 → preliminary SQL plain text."""
        prompt = self.prompt_manager.load_prompt(
            file_name='filter',
            section='rsl_backward_preliminary_sql',
            query=query,
            schema_str=schema_str,
            evidence=(evidence or "(no extra evidence provided)"),
        )
        response = self.client.generate_text(
            prompt=prompt, model=self.model_name, temperature=self.temperature,
        )
        logger.debug(f"[RSLBackward] preliminary SQL response: {response[:300]}")
        return response or ""

    # ------------------------------------------------------------------
    # Step 3/4 helpers — S_restore expansion + struct
    # ------------------------------------------------------------------
    @staticmethod
    def _col_names(qualified_nodes: List[str]) -> Set[str]:
        """{'t.c1', 't2.c2', 'bare'} → {'c1', 'c2', 'bare'} (col-only-distinct)."""
        out: Set[str] = set()
        for n in qualified_nodes or []:
            if not isinstance(n, str):
                continue
            out.add(n.split(".", 1)[1] if "." in n else n)
        return out

    @staticmethod
    def _expand_to_full_paths(
        col_names: Set[str], full_schema: Dict[str, List[str]],
    ) -> Set[str]:
        """col_names 의 각 col 을 full_schema 의 해당 table 에 매핑해 'table.col' 집합 반환.

        col 이 여러 table 에 중복이면 모든 후보 채택 (Cao 2024 RSL-SQL 정합 — backward
        가 잠재적으로 wrong-table-prefix SQL 을 만들어도 col 이름 기준으로 schema 후보
        들을 candidate restore set 에 포함).
        """
        if not col_names:
            return set()
        out: Set[str] = set()
        for tbl, cols in (full_schema or {}).items():
            for c in (cols or []):
                if c in col_names:
                    out.add(f"{tbl}.{c}")
        return out

    def _extract_fk_pk_columns(
        self, subgraph: Dict[str, List[str]], db_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Set[str]:
        """Step 4 — FK + PK columns within subgraph (SGBE 의 패턴 재사용)."""
        if not self.fk_pk_hardcode:
            return set()

        subgraph_keys = {
            f"{t}.{c}" for t, cols in (subgraph or {}).items() for c in (cols or [])
        }
        struct: Set[str] = set()

        # FK columns from metadata fk_to_id keys "src.x->dst.y"
        if metadata:
            for fk_key in (metadata.get("fk_to_id") or {}).keys():
                if not isinstance(fk_key, str) or "->" not in fk_key:
                    continue
                left, right = fk_key.split("->", 1)
                for side in (left.strip(), right.strip()):
                    if "." in side and side in subgraph_keys:
                        struct.add(side)

        # PK columns — metadata fallback + PRAGMA best-effort
        if metadata:
            md_pk = metadata.get("primary_keys")
            if isinstance(md_pk, (list, set, tuple)):
                struct.update(
                    p for p in md_pk
                    if isinstance(p, str) and "." in p and p in subgraph_keys
                )
            elif isinstance(md_pk, dict):
                for tbl, cols in md_pk.items():
                    for c in (cols or []):
                        key = f"{tbl}.{c}"
                        if key in subgraph_keys:
                            struct.add(key)

        if db_id and self.fk_pk_hardcode:
            db_path = os.path.join(self.db_dir, db_id, f"{db_id}.sqlite")
            if os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path)
                    cur = conn.cursor()
                    for tbl in (subgraph or {}).keys():
                        safe = tbl.replace("'", "''")
                        try:
                            cur.execute(f"PRAGMA table_info('{safe}');")
                            for row in cur.fetchall():
                                if len(row) >= 6 and int(row[5] or 0) > 0:
                                    key = f"{tbl}.{row[1]}"
                                    if key in subgraph_keys:
                                        struct.add(key)
                        except sqlite3.Error as e:
                            logger.debug(f"[RSLBackward] PK PRAGMA failed {tbl}: {e}")
                    conn.close()
                except Exception as e:
                    logger.debug(f"[RSLBackward] PK fetch error {db_id}: {e}")

        return struct

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def refine(
        self,
        query: str,
        subgraph: Dict[str, List[str]],
        db_id: Optional[str] = None,
        tier2_pool: Optional[Any] = None,  # unused
        gat_scores: Optional[Dict[str, float]] = None,  # unused (SGBE only)
        metadata: Optional[Dict[str, Any]] = None,
        evidence: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        token_before = AgentUtils.token_snapshot()

        # ── Step 1: XiYan forward ─────────────────────────────────────
        fwd_result = self.xiyan.refine(query=query, subgraph=subgraph, db_id=db_id)
        s_fwd: List[str] = list(fwd_result.get("final_nodes") or [])
        s_fwd_set: Set[str] = set(s_fwd)
        s_fwd_col_only: Set[str] = self._col_names(s_fwd)
        xiyan_status: str = fwd_result.get("status", "Answerable")
        xiyan_info: Dict[str, Any] = fwd_result.get("filter_info", {}) or {}

        # ── Step 2: Preliminary SQL backward ─────────────────────────
        full_schema = self._build_full_schema_map(db_id, metadata, subgraph)
        fk_keys = self._fk_keys_from_metadata(metadata)
        schema_str = self._build_full_schema_str(full_schema, db_id, fk_keys)
        prelim_sql: str = ""
        l_bwd_col_names: Set[str] = set()
        sql_parse_ok: bool = False
        try:
            prelim_sql = self._generate_preliminary_sql(query, schema_str, evidence)
            l_bwd_col_names = _extract_columns_from_sql(prelim_sql)
            sql_parse_ok = bool(l_bwd_col_names) or not prelim_sql.strip()
        except Exception as e:
            logger.warning(f"[RSLBackward] preliminary SQL backward failed: {e}")

        # ── Step 3: S_restore + DB-level guard ───────────────────────
        s_restore_col: Set[str] = l_bwd_col_names - s_fwd_col_only
        db_guard_active = bool(db_id) and (db_id in self.risky_dbs)
        if db_guard_active:
            logger.info(
                f"[RSLBackward] DB-level guard active for db_id='{db_id}' — "
                "skipping restore (Phase 3 margin caveat)."
            )
            s_restore: Set[str] = set()
        else:
            s_restore = self._expand_to_full_paths(s_restore_col, full_schema)

        # ── Step 4: S_struct FK/PK hardcode ──────────────────────────
        s_struct = self._extract_fk_pk_columns(subgraph, db_id, metadata)

        # ── Output ───────────────────────────────────────────────────
        final_set: Set[str] = s_fwd_set | s_restore | s_struct
        final_nodes: List[str] = sorted(final_set)
        status = "Answerable" if final_nodes else "Unanswerable"

        token_after = AgentUtils.token_snapshot()
        stats = {
            "fwd_nodes": len(s_fwd_set),
            "bwd_col_names": len(l_bwd_col_names),
            "restore_col_diff": len(s_restore_col),
            "restore_expanded": len(s_restore),
            "struct": len(s_struct),
            "final": len(final_set),
            "db_guard_active": db_guard_active,
            "sql_parse_ok": sql_parse_ok,
            "restore_is_empty": len(s_restore) == 0,
        }
        reasoning = (
            f"RSL-SQL Backward (Cao 2024): "
            f"|S_fwd|={len(s_fwd_set)} -> "
            f"|L_bwd col-names|={len(l_bwd_col_names)} -> "
            f"|S_restore_col diff|={len(s_restore_col)} -> "
            f"|S_restore expanded|={len(s_restore)}"
            + (" [DB-guard skipped restore]" if db_guard_active else "")
            + f" + |S_struct|={len(s_struct)} = |final|={len(final_set)}"
        )

        self.last_info = AgentUtils.build_filter_info(
            filter_type="RSLBackwardFilter",
            input_subgraph=subgraph,
            final_nodes=final_nodes,
            status=status,
            token_before=token_before,
            token_after=token_after,
            t_start=t_start,
            model=self.model_name,
            xiyan_status=xiyan_status,
            xiyan_nodes_after=int(xiyan_info.get("filter_nodes_after", len(s_fwd))),
            bwd_col_names=stats["bwd_col_names"],
            restore_col_diff=stats["restore_col_diff"],
            restore_expanded=stats["restore_expanded"],
            struct=stats["struct"],
            final=stats["final"],
            db_guard_active=db_guard_active,
            risky_dbs=sorted(self.risky_dbs) if self.risky_dbs else [],
            sql_parse_ok=sql_parse_ok,
            restore_is_empty=stats["restore_is_empty"],
            fk_pk_hardcode=self.fk_pk_hardcode,
        )
        return {
            "status": status,
            "final_nodes": final_nodes,
            "reasoning": reasoning,
            "stats": stats,
            "preliminary_sql": prelim_sql,
            "filter_info": dict(self.last_info),
        }
