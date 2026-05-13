"""GRAST-FD Filter — Direction C (Hoang 2025, GRAST-SQL FD graph).

학술 Agent Phase 3 의 Direction C 권고 + Analyzer Direction A sweep (2026-05-14)
의 trigger 발효 (ΔF1 = -0.2832 < +0.02) 에 따라 Steiner-tree based selective
schema restoration 을 구현한다. RSL Backward (Direction A) 의 "full backward
union → P drop -0.4345" 한계를 FD graph 의 structural constraint 로 제어하는
algorithm-only 보강 path.

핵심 가설 (DECISIONS 2026-05-14 §2.2):
  - Phase 2 C-2 의 multi-table 13.76% join col 미회복 (financial 13.21%,
    debit_card 10.59%, card_games 9.33%) — FK declaration 부족 → join col miss.
  - Steiner-tree 가 query mentioned cols 를 terminal 로 한 connectivity 회복
    cols 만 restore → backward noise 의 column 단위 filter mechanism.

Pipeline (per query, 1 LLM call by default — XiYan 만):

  Step 1  XiYan forward (anchor 정합)
          S_fwd = XiYanFilter.refine(query, subgraph, db_id).final_nodes

  Step 2  FD Graph 구성 (algorithm-only)
          - nodes: full schema 의 모든 "table.col" + "table"
          - edges:
              (i) belongs_to  : table -- column   (intra-table grouping)
              (ii) FK         : src.col -- dst.col   (declared FK, metadata)
              (iii) inferred  : src.col -- dst.col   (yaml configurable, Analyzer
                                후속 — GPT-4.1-mini debit_card / card_games 보완)

  Step 3  Steiner Tree based Selective Restore
          terminals = _resolve_terminals(query, subgraph, gat_scores, S_fwd, ...)
                      mode: "forward" (default, no extra LLM)
                          | "gat_topk" (top-K of gat_scores)
                          | "prelim_sql" (RSLBackward 와 동일 prompt, +1 LLM call)
          steiner = nx.approximation.steiner_tree(FD_graph, terminals)
          S_steiner_cols = {n for n in steiner.nodes() if "." in n} − S_fwd
          # connectivity 회복에 필요한 column 만 restore — backward union 처럼
          # noise 폭증 안 함.

  Step 4  S_struct FK/PK hardcode (anchor 정합, CHESS Talaei 2024)
          RSLBackwardFilter 의 _extract_fk_pk_columns 와 동일 패턴 재사용.

  Output: final_nodes = S_fwd ∪ S_steiner_restore ∪ S_struct

Anchor framework 준수:
  - Builder / Selector / Extractor / SQL Gen 모두 anchor 와 동일 — Filter 만 교체.
  - terminal_source="forward" 일 때 LLM calls/query = 1 (XiYan 만). RSL Backward
    의 2 (+200%) 대비 cost 절반.
"""
import os
import re
import sqlite3
import time
from typing import Dict, List, Any, Optional, Set, Tuple

import networkx as nx
from networkx.algorithms.approximation.steinertree import steiner_tree

from modules.registry import register
from modules.base import BaseFilter
from modules.filters.agents import AgentUtils
from modules.filters.xiyan_filter import XiYanFilter
from modules.filters.rsl_backward_filter import _extract_columns_from_sql
from prompts.prompt_manager import PromptManager
from utils.logger import get_logger

logger = get_logger(__name__)

_VALID_TERMINAL_SOURCES: Tuple[str, ...] = ("forward", "gat_topk", "prelim_sql")
_VALID_STEINER_METHODS: Tuple[str, ...] = ("default", "mehlhorn", "kou")


@register("filter", "GRASTFDFilter")
class GRASTFDFilter(BaseFilter):
    """GRAST-SQL FD Filter — XiYan forward + Steiner-tree restore + FK/PK hardcode."""

    def __init__(
        self,
        model_name: str = "zai-org/glm-4.7",
        temperature: float = 0.0,
        # XiYan forward (Step 1)
        xiyan_max_iteration: int = 1,
        xiyan_model_name: Optional[str] = None,
        xiyan_num_examples: int = 3,
        # DB / schema
        db_dir: str = "./data/raw/BIRD_dev/dev_databases",
        num_examples: int = 3,
        # FD graph (Step 2)
        inferred_fk: Optional[List[str]] = None,  # ["src.col->dst.col", ...] 형식
        include_belongs_to: bool = True,
        # Steiner tree (Step 3)
        terminal_source: str = "forward",
        top_k: int = 10,
        steiner_method: str = "default",
        max_restore: int = 30,
        # FK/PK hardcode (Step 4)
        fk_pk_hardcode: bool = True,
        # LLM provider (used when terminal_source="prelim_sql")
        provider: Optional[str] = "glm",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        if terminal_source not in _VALID_TERMINAL_SOURCES:
            raise ValueError(
                f"terminal_source='{terminal_source}' invalid. "
                f"Expected one of {_VALID_TERMINAL_SOURCES}."
            )
        if steiner_method not in _VALID_STEINER_METHODS:
            raise ValueError(
                f"steiner_method='{steiner_method}' invalid. "
                f"Expected one of {_VALID_STEINER_METHODS}."
            )
        self.model_name = model_name
        self.temperature = float(temperature)
        self.db_dir = db_dir
        self.num_examples = max(0, int(num_examples))
        self.inferred_fk: List[str] = list(inferred_fk or [])
        self.include_belongs_to = bool(include_belongs_to)
        self.terminal_source = terminal_source
        self.top_k = int(top_k)
        self.steiner_method = steiner_method
        self.max_restore = int(max_restore)
        self.fk_pk_hardcode = bool(fk_pk_hardcode)
        self.provider = provider
        self.prompt_manager = PromptManager()
        self.client = self._make_llm_client(
            api_key=api_key, base_url=base_url, provider=provider,
        )

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
            "Initialized GRASTFDFilter "
            f"(model={model_name}, provider={provider or 'auto'}, "
            f"terminal_source={terminal_source}, top_k={self.top_k}, "
            f"steiner_method={steiner_method}, max_restore={self.max_restore}, "
            f"|inferred_fk|={len(self.inferred_fk)}, "
            f"include_belongs_to={self.include_belongs_to}, "
            f"fk_pk_hardcode={self.fk_pk_hardcode})"
        )

    # ------------------------------------------------------------------
    # Step 2 — FD Graph 구성
    # ------------------------------------------------------------------
    def _build_full_schema_map(
        self,
        db_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        fallback_subgraph: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """Return {table: [col, ...]} for ENTIRE DB. metadata 우선, SQLite PRAGMA fallback."""
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
                    table_to_id = metadata.get("table_to_id") or {}
                    for t in table_to_id.keys():
                        full.setdefault(t, [])
                    return full

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
                            logger.debug(f"[GRAST-FD] PRAGMA failed {t}: {e}")
                            full[t] = []
                    conn.close()
                    return full
                except Exception as e:
                    logger.warning(f"[GRAST-FD] full schema fetch failed ({db_id}): {e}")

        return {t: list(cols or []) for t, cols in (fallback_subgraph or {}).items()}

    @staticmethod
    def _parse_fk_key(fk_key: str) -> Optional[Tuple[str, str]]:
        """'a.x->b.y' → ('a.x', 'b.y'). Malformed 시 None."""
        if not isinstance(fk_key, str) or "->" not in fk_key:
            return None
        left, right = fk_key.split("->", 1)
        left, right = left.strip(), right.strip()
        if "." not in left or "." not in right:
            return None
        return left, right

    def _build_fd_graph(
        self,
        full_schema: Dict[str, List[str]],
        metadata: Optional[Dict[str, Any]],
    ) -> nx.Graph:
        """Bipartite FD graph: column nodes "table.col" + table nodes "table".

        Edges:
          - belongs_to (column -- table)         intra-table grouping
          - FK         (column -- column)        declared (metadata) + inferred (yaml)
        """
        G = nx.Graph()
        for tbl, cols in full_schema.items():
            G.add_node(tbl, kind="table")
            for c in cols:
                node = f"{tbl}.{c}"
                G.add_node(node, kind="column", table=tbl)
                if self.include_belongs_to:
                    G.add_edge(node, tbl, kind="belongs_to")

        declared_fks: List[str] = []
        if metadata:
            declared_fks = list((metadata.get("fk_to_id") or {}).keys())

        for fk_key in list(declared_fks) + list(self.inferred_fk):
            parsed = self._parse_fk_key(fk_key)
            if parsed is None:
                continue
            src, dst = parsed
            # 그래프에 노드가 없으면 미연방 — schema 와 매칭 안 되는 FK 는 skip
            if not (G.has_node(src) and G.has_node(dst)):
                continue
            kind = "fk_inferred" if fk_key in self.inferred_fk else "fk_declared"
            G.add_edge(src, dst, kind=kind)
        return G

    # ------------------------------------------------------------------
    # Step 3 — terminal resolution + Steiner tree
    # ------------------------------------------------------------------
    def _resolve_terminals(
        self,
        query: str,
        s_fwd: List[str],
        full_schema: Dict[str, List[str]],
        db_id: Optional[str],
        gat_scores: Optional[Dict[str, float]],
        evidence: Optional[str],
        diag: Dict[str, Any],
    ) -> Set[str]:
        """terminal_source 정책에 따라 Steiner tree 의 terminal column 집합 반환.

        모든 terminal 은 full_schema 에 실제 존재하는 "table.col" 로 정규화.
        """
        full_keys: Set[str] = {
            f"{t}.{c}" for t, cols in full_schema.items() for c in (cols or [])
        }

        def _filter_existing(candidates: Set[str]) -> Set[str]:
            return {n for n in candidates if n in full_keys}

        if self.terminal_source == "forward":
            return _filter_existing({n for n in s_fwd if isinstance(n, str) and "." in n})

        if self.terminal_source == "gat_topk":
            terms: Set[str] = set()
            if gat_scores:
                ranked = sorted(
                    ((k, v) for k, v in gat_scores.items() if isinstance(k, str)),
                    key=lambda kv: (kv[1] if isinstance(kv[1], (int, float)) else 0.0),
                    reverse=True,
                )
                for k, _ in ranked[: max(1, self.top_k)]:
                    if "." in k:
                        terms.add(k)
                    else:
                        # column-only key 의 경우 schema 의 모든 후보 확장
                        for t, cols in full_schema.items():
                            if k in cols:
                                terms.add(f"{t}.{k}")
            else:
                logger.warning(
                    "[GRAST-FD] terminal_source='gat_topk' but gat_scores=None — "
                    "falling back to forward."
                )
                diag["terminal_fallback"] = "forward (gat_topk unavailable)"
                return _filter_existing(
                    {n for n in s_fwd if isinstance(n, str) and "." in n}
                )
            # GAT top-K 가 모두 forward 안에 있는 경우, forward 도 함께 terminal 로
            # 두어 Steiner tree 가 connectivity 회복에 더 풍부한 입력을 받게.
            terms.update(n for n in s_fwd if isinstance(n, str) and "." in n)
            return _filter_existing(terms)

        # "prelim_sql" — RSL Backward 의 prompt 재사용 (LLM call +1)
        terms = set()
        try:
            schema_str = self._build_schema_str_for_prompt(full_schema, db_id)
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
            diag["prelim_sql_response_preview"] = (response or "")[:200]
            col_names = _extract_columns_from_sql(response or "")
            for t, cols in full_schema.items():
                for c in (cols or []):
                    if c in col_names:
                        terms.add(f"{t}.{c}")
        except Exception as e:
            logger.warning(f"[GRAST-FD] prelim_sql terminal resolution failed: {e}")
            diag["terminal_fallback"] = "forward (prelim_sql failed)"
            return _filter_existing(
                {n for n in s_fwd if isinstance(n, str) and "." in n}
            )
        terms.update(n for n in s_fwd if isinstance(n, str) and "." in n)
        return _filter_existing(terms)

    def _build_schema_str_for_prompt(
        self, full_schema: Dict[str, List[str]], db_id: Optional[str],
    ) -> str:
        """LLM prompt 용 schema text (prelim_sql terminal source 에서만 사용)."""
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
        for tbl, cols in full_schema.items():
            lines.append(f"# Table: {tbl}")
            details = []
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
                    except Exception:
                        pass
                details.append(col_str)
            lines.append("  Columns: " + " | ".join(details))
            lines.append("")
        if conn is not None:
            conn.close()
        return "\n".join(lines)

    def _compute_steiner_restore(
        self,
        graph: nx.Graph,
        terminals: Set[str],
        s_fwd_set: Set[str],
        diag: Dict[str, Any],
    ) -> Set[str]:
        """Steiner tree → column-only nodes − S_fwd. Disconnected component 안전 처리."""
        if not terminals:
            diag["steiner_skipped"] = "no_terminals"
            return set()
        # terminal 이 graph 에 모두 있어야 steiner_tree 가 동작
        present = {t for t in terminals if graph.has_node(t)}
        diag["terminals_present"] = len(present)
        diag["terminals_missing"] = len(terminals) - len(present)
        if len(present) < 2:
            # Steiner tree 는 2 개 이상의 terminal 필요. 단일 terminal 이면 restore 없음.
            diag["steiner_skipped"] = "single_terminal_or_empty"
            return set()

        # 연결된 component 별로 grouping — 모두 같은 component 가 아니면 component 별로
        # Steiner tree 를 따로 계산해 합친다 (network 의 steiner_tree 가 disconnected
        # 입력에 대해 NetworkXPointlessConcept 또는 비정상 동작).
        try:
            components = list(nx.connected_components(graph))
        except Exception as e:
            logger.warning(f"[GRAST-FD] connected_components failed: {e}")
            diag["steiner_error"] = str(e)
            return set()
        comp_of: Dict[str, int] = {}
        for i, comp in enumerate(components):
            for n in comp:
                comp_of[n] = i
        groups: Dict[int, Set[str]] = {}
        for t in present:
            groups.setdefault(comp_of[t], set()).add(t)

        restore_cols: Set[str] = set()
        steiner_kwargs: Dict[str, Any] = {}
        if self.steiner_method in ("mehlhorn", "kou"):
            steiner_kwargs["method"] = self.steiner_method

        for ci, terms in groups.items():
            if len(terms) < 2:
                continue
            try:
                sub_g = graph.subgraph(components[ci]).copy()
                tree = steiner_tree(sub_g, list(terms), **steiner_kwargs)
            except Exception as e:
                logger.warning(
                    f"[GRAST-FD] steiner_tree failed on component {ci}: {e}"
                )
                diag.setdefault("component_errors", []).append(
                    f"comp{ci}: {type(e).__name__}: {e}"
                )
                continue
            for n in tree.nodes():
                # column node 만 restore 후보로 채택 (table 은 무시)
                if isinstance(n, str) and "." in n and n not in s_fwd_set:
                    restore_cols.add(n)

        # max_restore cap — overly large Steiner tree 에 의한 noise 폭증 방지
        if self.max_restore >= 0 and len(restore_cols) > self.max_restore:
            diag["restore_capped_from"] = len(restore_cols)
            # deterministic order 로 자르기
            restore_cols = set(sorted(restore_cols)[: self.max_restore])
        return restore_cols

    # ------------------------------------------------------------------
    # Step 4 — FK/PK hardcode (RSLBackward 와 동일 패턴)
    # ------------------------------------------------------------------
    def _extract_fk_pk_columns(
        self, subgraph: Dict[str, List[str]], db_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Set[str]:
        if not self.fk_pk_hardcode:
            return set()

        subgraph_keys = {
            f"{t}.{c}" for t, cols in (subgraph or {}).items() for c in (cols or [])
        }
        struct: Set[str] = set()
        if metadata:
            for fk_key in (metadata.get("fk_to_id") or {}).keys():
                parsed = self._parse_fk_key(fk_key)
                if parsed is None:
                    continue
                for side in parsed:
                    if side in subgraph_keys:
                        struct.add(side)
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

        if db_id:
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
                            logger.debug(f"[GRAST-FD] PK PRAGMA failed {tbl}: {e}")
                    conn.close()
                except Exception as e:
                    logger.debug(f"[GRAST-FD] PK fetch error {db_id}: {e}")
        return struct

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
        evidence: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        token_before = AgentUtils.token_snapshot()
        diag: Dict[str, Any] = {}

        # ── Step 1: XiYan forward ─────────────────────────────────────
        fwd_result = self.xiyan.refine(query=query, subgraph=subgraph, db_id=db_id)
        s_fwd: List[str] = list(fwd_result.get("final_nodes") or [])
        s_fwd_set: Set[str] = set(s_fwd)
        xiyan_status = fwd_result.get("status", "Answerable")

        # ── Step 2: FD graph ──────────────────────────────────────────
        full_schema = self._build_full_schema_map(db_id, metadata, subgraph)
        graph = self._build_fd_graph(full_schema, metadata)
        diag["graph_nodes"] = graph.number_of_nodes()
        diag["graph_edges"] = graph.number_of_edges()
        diag["declared_fk_count"] = len(
            (metadata or {}).get("fk_to_id", {}) or {}
        )
        diag["inferred_fk_count"] = len(self.inferred_fk)

        # ── Step 3: Steiner tree restore ──────────────────────────────
        terminals = self._resolve_terminals(
            query, s_fwd, full_schema, db_id, gat_scores, evidence, diag,
        )
        diag["terminal_source_used"] = (
            diag.get("terminal_fallback") or self.terminal_source
        )
        diag["terminal_count"] = len(terminals)

        s_steiner = self._compute_steiner_restore(graph, terminals, s_fwd_set, diag)

        # ── Step 4: FK/PK hardcode ────────────────────────────────────
        s_struct = self._extract_fk_pk_columns(subgraph, db_id, metadata)

        # ── Output ────────────────────────────────────────────────────
        final_set: Set[str] = s_fwd_set | s_steiner | s_struct
        final_nodes: List[str] = sorted(final_set)
        status = "Answerable" if final_nodes else "Unanswerable"
        token_after = AgentUtils.token_snapshot()

        stats = {
            "fwd_nodes": len(s_fwd_set),
            "terminal_count": diag.get("terminal_count", 0),
            "steiner_restore": len(s_steiner),
            "struct": len(s_struct),
            "final": len(final_set),
            "graph_nodes": diag["graph_nodes"],
            "graph_edges": diag["graph_edges"],
            "declared_fk_count": diag["declared_fk_count"],
            "inferred_fk_count": diag["inferred_fk_count"],
            "terminal_source_used": diag["terminal_source_used"],
            "restore_is_empty": len(s_steiner) == 0,
            "restore_capped_from": diag.get("restore_capped_from"),
            "steiner_skipped": diag.get("steiner_skipped"),
        }
        reasoning = (
            f"GRAST-FD (Hoang 2025): "
            f"|S_fwd|={len(s_fwd_set)} -> "
            f"FD graph (|V|={stats['graph_nodes']}, |E|={stats['graph_edges']}, "
            f"declared_fk={stats['declared_fk_count']}"
            + (f", inferred_fk={stats['inferred_fk_count']}" if stats['inferred_fk_count'] else "")
            + f") -> "
            f"terminals={stats['terminal_count']} (src={stats['terminal_source_used']}) -> "
            f"|S_steiner_restore|={len(s_steiner)} + |S_struct|={len(s_struct)} "
            f"= |final|={len(final_set)}"
        )
        if stats.get("restore_capped_from"):
            reasoning += f" [capped from {stats['restore_capped_from']} → {self.max_restore}]"
        if stats.get("steiner_skipped"):
            reasoning += f" [steiner skipped: {stats['steiner_skipped']}]"

        self.last_info = AgentUtils.build_filter_info(
            filter_type="GRASTFDFilter",
            input_subgraph=subgraph,
            final_nodes=final_nodes,
            status=status,
            token_before=token_before,
            token_after=token_after,
            t_start=t_start,
            model=self.model_name,
            xiyan_status=xiyan_status,
            fwd_nodes=stats["fwd_nodes"],
            terminal_count=stats["terminal_count"],
            steiner_restore=stats["steiner_restore"],
            struct=stats["struct"],
            final=stats["final"],
            graph_nodes=stats["graph_nodes"],
            graph_edges=stats["graph_edges"],
            declared_fk_count=stats["declared_fk_count"],
            inferred_fk_count=stats["inferred_fk_count"],
            terminal_source=self.terminal_source,
            terminal_source_used=stats["terminal_source_used"],
            steiner_method=self.steiner_method,
            max_restore=self.max_restore,
            include_belongs_to=self.include_belongs_to,
            fk_pk_hardcode=self.fk_pk_hardcode,
            restore_is_empty=stats["restore_is_empty"],
            restore_capped_from=stats.get("restore_capped_from"),
            steiner_skipped=stats.get("steiner_skipped"),
        )
        return {
            "status": status,
            "final_nodes": final_nodes,
            "reasoning": reasoning,
            "stats": stats,
            "filter_info": dict(self.last_info),
        }
