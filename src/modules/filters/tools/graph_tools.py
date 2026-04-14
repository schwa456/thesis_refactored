"""Graph context helpers for F3 TieredBidirectionalAgentFilter.

These are "static tools" — functions that build ready-to-read graph context
snippets from the pipeline metadata. They intentionally avoid a full ReAct
tool-calling loop to keep the filter deterministic; the LLM sees the graph
evidence upfront as part of the prompt rather than requesting it on demand.
"""
import os
import sqlite3
from typing import Dict, List, Any, Iterable, Set


class GraphTools:
    def __init__(self, metadata: Dict[str, Any], db_dir: str = None):
        self.metadata = metadata or {}
        self.db_dir = db_dir
        self.node_meta: Dict[Any, str] = self.metadata.get("node_metadata", {}) or {}
        self.fk_descriptions: List[str] = self.metadata.get("fk_descriptions", []) or []
        self._name_to_idx: Dict[str, Any] = {
            str(v): k for k, v in self.node_meta.items()
        }

    def get_tier(self, node: str, tier1: Set[str], tier2: Set[str]) -> str:
        if node in tier1:
            return "Tier-1"
        if node in tier2:
            return "Tier-2"
        return "Unknown"

    def get_gat_score(self, node: str, gat_scores: Dict[str, float]) -> float:
        return float(gat_scores.get(node, 0.0))

    def get_fk_paths(self, nodes: Iterable[str]) -> List[str]:
        node_set = set(nodes)
        relevant = []
        for desc in self.fk_descriptions:
            s = str(desc)
            if "->" not in s:
                continue
            parts = s.split("->")
            if len(parts) != 2:
                continue
            src, dst = parts[0].strip(), parts[1].strip()
            src_tbl = src.split(".")[0] if "." in src else src
            dst_tbl = dst.split(".")[0] if "." in dst else dst
            if (
                src in node_set
                or dst in node_set
                or src_tbl in {n.split(".")[0] for n in node_set if "." in n}
                or dst_tbl in {n.split(".")[0] for n in node_set if "." in n}
            ):
                relevant.append(s)
        return relevant

    def get_neighbors(self, node: str) -> List[str]:
        if "." not in node:
            return []
        tbl, _ = node.split(".", 1)
        neighbors: List[str] = []
        for name in self.node_meta.values():
            s = str(name)
            if s == node:
                continue
            if s.startswith(f"{tbl}."):
                neighbors.append(s)
        return neighbors

    def get_column_examples(
        self, node: str, db_id: str, limit: int = 3
    ) -> List[str]:
        if not (self.db_dir and db_id and "." in node):
            return []
        db_path = os.path.join(self.db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            return []
        tbl, col = node.split(".", 1)
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                f'SELECT DISTINCT "{col}" FROM "{tbl}" '
                f'WHERE "{col}" IS NOT NULL LIMIT {limit}'
            )
            vals = [str(r[0]) for r in cursor.fetchall()]
            conn.close()
            return vals
        except Exception:
            return []

    def format_tier_context(
        self,
        tier1_dropped: Iterable[str],
        tier2_pool: Iterable[str],
        gat_scores: Dict[str, float],
        max_items: int = 40,
    ) -> Dict[str, str]:
        def _fmt(nodes: Iterable[str]) -> str:
            items = list(nodes)[:max_items]
            if not items:
                return "(none)"
            lines = []
            for n in items:
                sc = gat_scores.get(n)
                sc_str = f" score={sc:.3f}" if sc is not None else ""
                lines.append(f"- {n}{sc_str}")
            return "\n".join(lines)

        def _fmt_scores(nodes: Iterable[str]) -> str:
            items = [(n, gat_scores.get(n, 0.0)) for n in nodes]
            items = sorted(items, key=lambda x: -x[1])[:max_items]
            if not items:
                return "(none)"
            return "\n".join(f"- {n}: {sc:.3f}" for n, sc in items)

        all_candidates = list(tier1_dropped) + list(tier2_pool)
        return {
            "tier1_dropped": _fmt(tier1_dropped),
            "tier2_pool": _fmt(tier2_pool),
            "gat_scores_snippet": _fmt_scores(all_candidates),
        }
