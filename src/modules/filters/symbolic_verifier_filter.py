"""FL-III. Symbolic Verifier Filter — Deterministic FK-graph connectivity guardrail.

Wraps a base Filter (XiYan / Reflection / Verifier / Stacked) and performs
a post-hoc, purely-symbolic check on the LLM's node selection:

  1. Extract the set of selected tables from `final_nodes`.
  2. Check whether they form a connected subgraph under the FK graph
     induced by those tables only.
  3. If disconnected and `auto_repair=True`, find the shortest FK path in
     the *full* FK graph between components and inject the intermediate
     bridge tables (+ FK join columns) into `final_nodes`.
  4. If a component is unreachable even in the full FK graph, it is
     flagged as unrepairable (no bridge possible).

Rationale: LLM filters can select semantically relevant tables that are not
FK-reachable without intermediate bridges, making the selection unusable for
SQL JOIN construction. A deterministic topology check is orthogonal to
NL unit tests (VerifierFilter / F2) and provides a graph-topology guardrail.

Dependencies: `metadata['fk_to_id']` whose keys have the format
`"{from_table}.{from_col}->{to_table}.{to_col}"` — already emitted by
`HeteroGraphBuilder.build()`.

Modes:
  - detect-only (`auto_repair=False`): final_nodes is unchanged; the filter
    only annotates the result with `connectivity_valid` and
    `connectivity_issues` for downstream measurement.
  - auto_repair (`auto_repair=True`): additively extends final_nodes with
    bridge tables / FK columns. Never removes base-filter decisions.
"""
import time
from collections import deque
from typing import Dict, List, Any, Iterable, Optional, Set, Tuple

from modules.registry import register, build
from modules.base import BaseFilter
from modules.filters.agents import AgentUtils
from utils.logger import get_logger

logger = get_logger(__name__)


@register("filter", "SymbolicVerifierFilter")
class SymbolicVerifierFilter(BaseFilter):
    def __init__(
        self,
        base_filter: Dict[str, Any],
        auto_repair: bool = True,
        add_fk_columns: bool = True,
        max_bridges: int = 3,
        **kwargs,
    ):
        self.base: BaseFilter = build("filter", base_filter)
        self.auto_repair: bool = bool(auto_repair)
        self.add_fk_columns: bool = bool(add_fk_columns)
        self.max_bridges: int = int(max_bridges)
        logger.info(
            f"Initialized SymbolicVerifierFilter "
            f"(base={base_filter.get('name')}, auto_repair={auto_repair}, "
            f"add_fk_columns={add_fk_columns}, max_bridges={max_bridges})"
        )

    @staticmethod
    def _extract_tables(final_nodes: Iterable[str]) -> Set[str]:
        tables: Set[str] = set()
        for n in final_nodes:
            if not isinstance(n, str):
                continue
            if "." in n:
                tables.add(n.split(".", 1)[0])
            else:
                tables.add(n)
        return tables

    @staticmethod
    def _build_fk_adj(
        metadata: Dict[str, Any],
    ) -> Dict[str, List[Tuple[str, str, str, str]]]:
        """Return table -> list of (neighbor, fk_key, from_col_full, to_col_full)."""
        adj: Dict[str, List[Tuple[str, str, str, str]]] = {}
        fk_to_id = metadata.get("fk_to_id", {}) or {}
        for fk_key in fk_to_id.keys():
            if not isinstance(fk_key, str) or "->" not in fk_key:
                continue
            left, right = fk_key.split("->", 1)
            left, right = left.strip(), right.strip()
            if "." not in left or "." not in right:
                continue
            src_tbl, src_col = left.split(".", 1)
            dst_tbl, dst_col = right.split(".", 1)
            if not src_tbl or not dst_tbl:
                continue
            src_full = f"{src_tbl}.{src_col}"
            dst_full = f"{dst_tbl}.{dst_col}"
            adj.setdefault(src_tbl, []).append(
                (dst_tbl, fk_key, src_full, dst_full)
            )
            adj.setdefault(dst_tbl, []).append(
                (src_tbl, fk_key, dst_full, src_full)
            )
        return adj

    @staticmethod
    def _induced_components(
        tables: Set[str],
        adj: Dict[str, List[Tuple[str, str, str, str]]],
    ) -> List[Set[str]]:
        visited: Set[str] = set()
        components: List[Set[str]] = []
        for start in tables:
            if start in visited:
                continue
            comp: Set[str] = set()
            stack = [start]
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.add(cur)
                for (nbr, _, _, _) in adj.get(cur, []):
                    if nbr in tables and nbr not in visited:
                        stack.append(nbr)
            components.append(comp)
        return components

    @staticmethod
    def _bfs_shortest_path(
        adj: Dict[str, List[Tuple[str, str, str, str]]],
        src: str,
        dst: str,
    ) -> Optional[List[str]]:
        if src == dst:
            return [src]
        parents: Dict[str, Optional[str]] = {src: None}
        queue: deque = deque([src])
        while queue:
            cur = queue.popleft()
            for (nbr, _, _, _) in adj.get(cur, []):
                if nbr in parents:
                    continue
                parents[nbr] = cur
                if nbr == dst:
                    path: List[str] = [nbr]
                    while parents[path[-1]] is not None:
                        path.append(parents[path[-1]])
                    path.reverse()
                    return path
                queue.append(nbr)
        return None

    def _shortest_between_sets(
        self,
        adj: Dict[str, List[Tuple[str, str, str, str]]],
        srcs: Iterable[str],
        dsts: Set[str],
    ) -> Optional[List[str]]:
        if not srcs or not dsts:
            return None
        visited: Set[str] = set(srcs)
        parents: Dict[str, Optional[str]] = {s: None for s in srcs}
        queue: deque = deque(list(srcs))
        while queue:
            cur = queue.popleft()
            for (nbr, _, _, _) in adj.get(cur, []):
                if nbr in visited:
                    continue
                visited.add(nbr)
                parents[nbr] = cur
                if nbr in dsts:
                    path: List[str] = [nbr]
                    while parents[path[-1]] is not None:
                        path.append(parents[path[-1]])
                    path.reverse()
                    return path
                queue.append(nbr)
        return None

    @staticmethod
    def _find_fk_columns(
        adj: Dict[str, List[Tuple[str, str, str, str]]],
        src: str,
        dst: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        for (nbr, _, from_col, to_col) in adj.get(src, []):
            if nbr == dst:
                return from_col, to_col
        return None, None

    def _finalize_info(
        self,
        base_result: Dict[str, Any],
        input_subgraph: Dict[str, List[str]],
        *,
        connectivity_valid: bool,
        num_components_before: int,
        num_components_after: int,
        added_tables: List[str],
        added_fk_cols: List[str],
        unrepairable: List[str],
        issues: List[str],
        tables_selected: int,
        repair_attempted: bool,
        t_start: Optional[float] = None,
        token_before: Optional[Dict[str, int]] = None,
    ) -> None:
        base_info = getattr(self.base, "last_info", None) or base_result.get("filter_info") or {}
        token_after = AgentUtils.token_snapshot() if token_before is not None else None
        self.last_info = AgentUtils.build_filter_info(
            filter_type="SymbolicVerifierFilter",
            input_subgraph=input_subgraph,
            final_nodes=base_result.get("final_nodes", []) or [],
            status=base_result.get("status"),
            token_before=token_before,
            token_after=token_after,
            t_start=t_start,
            base_filter_type=base_info.get("filter_type"),
            base_time_s=float(base_info.get("filter_time_s", 0.0) or 0.0),
            base_info=dict(base_info),
            auto_repair=self.auto_repair,
            add_fk_columns=self.add_fk_columns,
            max_bridges=int(self.max_bridges),
            tables_selected=tables_selected,
            connectivity_valid=connectivity_valid,
            components_before_repair=num_components_before,
            components_after_repair=num_components_after,
            repair_attempted=repair_attempted,
            repaired_tables=list(added_tables),
            repaired_fk_cols=list(added_fk_cols),
            repaired_tables_count=len(added_tables),
            repaired_fk_cols_count=len(added_fk_cols),
            unrepairable_components=list(unrepairable),
            unrepairable_count=len(unrepairable),
            connectivity_issues=list(issues),
        )
        base_result["filter_info"] = dict(self.last_info)

    def refine(
        self,
        query: str,
        subgraph: Dict[str, List[str]],
        db_id: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        token_before = AgentUtils.token_snapshot()
        base_result = self.base.refine(
            query=query, subgraph=subgraph, db_id=db_id, **kwargs
        )
        final_nodes: List[str] = list(base_result.get("final_nodes", []) or [])

        if not final_nodes:
            base_result["connectivity_valid"] = True
            base_result["connectivity_issues"] = []
            self._finalize_info(
                base_result, subgraph,
                connectivity_valid=True,
                num_components_before=0,
                num_components_after=0,
                added_tables=[], added_fk_cols=[], unrepairable=[],
                issues=[], tables_selected=0, repair_attempted=False,
                t_start=t_start, token_before=token_before,
            )
            return base_result

        metadata: Dict[str, Any] = kwargs.get("metadata", {}) or {}
        tables = self._extract_tables(final_nodes)

        if len(tables) <= 1:
            base_result["connectivity_valid"] = True
            base_result["connectivity_issues"] = []
            self._finalize_info(
                base_result, subgraph,
                connectivity_valid=True,
                num_components_before=1 if tables else 0,
                num_components_after=1 if tables else 0,
                added_tables=[], added_fk_cols=[], unrepairable=[],
                issues=[], tables_selected=len(tables), repair_attempted=False,
                t_start=t_start, token_before=token_before,
            )
            return base_result

        adj = self._build_fk_adj(metadata)
        induced_comps = self._induced_components(tables, adj)
        is_valid = len(induced_comps) == 1
        base_result["connectivity_valid"] = is_valid
        issues: List[str] = []
        num_comps_before = len(induced_comps)

        if is_valid:
            base_result["connectivity_issues"] = issues
            self._finalize_info(
                base_result, subgraph,
                connectivity_valid=True,
                num_components_before=num_comps_before,
                num_components_after=num_comps_before,
                added_tables=[], added_fk_cols=[], unrepairable=[],
                issues=issues, tables_selected=len(tables),
                repair_attempted=False,
                t_start=t_start, token_before=token_before,
            )
            return base_result

        comp_summary = "; ".join(
            ",".join(sorted(c)) for c in sorted(induced_comps, key=len, reverse=True)
        )
        issues.append(f"FK-disconnected[n={len(induced_comps)}]:{comp_summary}")

        if not self.auto_repair:
            base_result["connectivity_issues"] = issues
            base_result["reasoning"] = (
                base_result.get("reasoning", "")
                + f" | SymVerify(detect): disconnected({len(induced_comps)})"
            )
            self._finalize_info(
                base_result, subgraph,
                connectivity_valid=False,
                num_components_before=num_comps_before,
                num_components_after=num_comps_before,
                added_tables=[], added_fk_cols=[], unrepairable=[],
                issues=issues, tables_selected=len(tables),
                repair_attempted=False,
                t_start=t_start, token_before=token_before,
            )
            return base_result

        repaired: Set[str] = set(final_nodes)
        added_tables: Set[str] = set()
        added_fk_cols: Set[str] = set()
        unrepairable: List[str] = []

        def _outbound(comp: Set[str]) -> int:
            return sum(
                1
                for t in comp
                for (nbr, _, _, _) in adj.get(t, [])
                if nbr not in comp
            )

        comps_sorted = sorted(
            induced_comps,
            key=lambda c: (
                -len(c),
                -_outbound(c),
                sorted(c)[0] if c else "",
            ),
        )
        merged: Set[str] = set(comps_sorted[0])

        for comp in comps_sorted[1:]:
            if len(added_tables) >= self.max_bridges:
                unrepairable.append(
                    f"bridge_budget_exceeded@{len(added_tables)}:{sorted(comp)}"
                )
                break

            path = self._shortest_between_sets(adj, merged, comp)
            if path is None:
                unrepairable.append(f"no_fk_path:{sorted(comp)}")
                continue

            for t in path[1:-1]:
                if t not in tables:
                    added_tables.add(t)

            if self.add_fk_columns:
                for i in range(len(path) - 1):
                    from_col, to_col = self._find_fk_columns(adj, path[i], path[i + 1])
                    if from_col:
                        added_fk_cols.add(from_col)
                    if to_col:
                        added_fk_cols.add(to_col)

            merged |= set(path) | comp

        for col in added_fk_cols:
            repaired.add(col)
        for t in added_tables:
            if not any(n == t or n.startswith(f"{t}.") for n in repaired):
                repaired.add(t)

        if unrepairable:
            issues.extend(unrepairable)

        if added_tables or added_fk_cols:
            base_result["final_nodes"] = sorted(repaired)
            base_result["status"] = "repaired"
            base_result["repair_added_tables"] = sorted(added_tables)
            base_result["repair_added_fk_cols"] = sorted(added_fk_cols)
            base_result["reasoning"] = (
                base_result.get("reasoning", "")
                + f" | SymVerify(repair): +{len(added_tables)}t +{len(added_fk_cols)}c"
            )
        else:
            base_result["reasoning"] = (
                base_result.get("reasoning", "")
                + f" | SymVerify(repair): no bridges found"
            )

        base_result["connectivity_issues"] = issues
        tables_after = self._extract_tables(base_result.get("final_nodes", []) or [])
        comps_after = self._induced_components(tables_after, adj) if tables_after else []
        self._finalize_info(
            base_result, subgraph,
            connectivity_valid=(len(comps_after) <= 1),
            num_components_before=num_comps_before,
            num_components_after=len(comps_after),
            added_tables=sorted(added_tables),
            added_fk_cols=sorted(added_fk_cols),
            unrepairable=list(unrepairable),
            issues=issues,
            tables_selected=len(tables),
            repair_attempted=True,
            t_start=t_start, token_before=token_before,
        )
        return base_result
