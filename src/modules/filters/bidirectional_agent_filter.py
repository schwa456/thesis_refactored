"""F3. Tiered Bidirectional Agent Filter.

Two-tier prior structure:
  * Tier-1 — Extractor (PCST) output: strong prior (connectivity verified).
  * Tier-2 — Selector-positive but PCST-rejected: weak prior
             (semantic relevance but connectivity NOT verified).

Pipeline:
  1) Prune Agent runs XiYan-style filter over Tier-1 → initial_keep.
  2) Restore Agent sees Tier-1 dropped + Tier-2 pool + GAT scores, and
     returns {restore: [...], promote: [...]} applying different evidence
     thresholds per tier (restore is liberal, promote is conservative).
  3) Final selection = initial_keep ∪ restored ∪ promoted.

Inspired by AutoLink (AAAI'26), RSL-SQL bidirectional, MAG-SQL soft linker.
Differentiation: explicit tier-aware evidence hierarchy.
"""
import json
import os
import sqlite3
from typing import Dict, List, Any, Set, Iterable

from modules.registry import register
from modules.base import BaseFilter
from modules.filters.agents import AgentUtils
from modules.filters.tools.graph_tools import GraphTools
from llm_client.api_handler import APIClient
from prompts.prompt_manager import PromptManager
from utils.logger import get_logger

logger = get_logger(__name__)


@register("filter", "TieredBidirectionalAgentFilter")
class TieredBidirectionalAgentFilter(BaseFilter):
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        use_graph_context: bool = True,
        db_dir: str = "./data/raw/BIRD_dev/dev_databases",
        api_key: str = None,
        base_url: str = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.use_graph_context = use_graph_context
        self.db_dir = db_dir
        self.prompt_manager = PromptManager()
        self.client = APIClient(api_key=api_key, base_url=base_url)
        logger.info(
            f"Initialized TieredBidirectionalAgentFilter "
            f"(tools={use_graph_context}, model={model_name})"
        )

    def _schema_with_values(
        self, schema: Dict[str, List[str]], db_id: str
    ) -> str:
        db_path = (
            os.path.join(self.db_dir, db_id, f"{db_id}.sqlite") if db_id else None
        )
        conn = None
        cursor = None
        if db_path and os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
            except Exception as e:
                logger.warning(f"DB connect failed for {db_id}: {e}")

        lines = []
        for tbl, cols in schema.items():
            col_parts = []
            for col in cols:
                s = col
                if cursor:
                    try:
                        cursor.execute(
                            f'SELECT DISTINCT "{col}" FROM "{tbl}" '
                            f'WHERE "{col}" IS NOT NULL LIMIT 3'
                        )
                        vals = [str(r[0]) for r in cursor.fetchall()]
                        if vals:
                            s += f" (Examples: {', '.join(vals)})"
                    except Exception:
                        pass
                col_parts.append(s)
            lines.append(f"# Table: {tbl}\n  Columns: " + " | ".join(col_parts))
        if conn:
            conn.close()
        return "\n".join(lines)

    def _flatten(self, schema: Dict[str, List[str]]) -> List[str]:
        return [f"{t}.{c}" for t, cols in schema.items() for c in cols]

    def _parse_schema_json(self, response: str) -> Dict[str, List[str]]:
        clean = response.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}")
        if start == -1 or end == -1 or start >= end:
            return {}
        try:
            obj = json.loads(clean[start : end + 1])
            if isinstance(obj, dict):
                return {str(k): list(v) for k, v in obj.items() if isinstance(v, list)}
        except Exception as e:
            logger.warning(f"[TieredFilter] JSON parse failed: {e}")
        return {}

    def _prune(
        self, query: str, tier1_schema: Dict[str, List[str]], db_id: str
    ) -> Dict[str, List[str]]:
        schema_str = self._schema_with_values(tier1_schema, db_id)
        example_tables = list(tier1_schema.keys())[:2]
        example_obj = {
            t: tier1_schema[t][: (2 if i == 0 else 1)]
            for i, t in enumerate(example_tables)
        }
        prompt = self.prompt_manager.load_prompt(
            file_name="filter",
            section="xiyan_filter",
            schema_str=schema_str,
            query=query,
            example_json_str=json.dumps(example_obj),
        )
        resp = self.client.generate_text(
            prompt=prompt, model=self.model_name, temperature=self.temperature
        )
        parsed = self._parse_schema_json(resp)
        return parsed if parsed else tier1_schema

    def _restore(
        self,
        query: str,
        current: Dict[str, List[str]],
        tier1_dropped: List[str],
        tier2_pool: List[str],
        gat_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        current_str = "\n".join(
            f"{t}.{c}" for t, cols in current.items() for c in cols
        )
        tools = GraphTools(metadata=None, db_dir=self.db_dir)
        ctx = tools.format_tier_context(tier1_dropped, tier2_pool, gat_scores)
        prompt = self.prompt_manager.load_prompt(
            file_name="filter",
            section="restore_agent",
            query=query,
            current_selection=current_str,
            tier1_dropped=ctx["tier1_dropped"],
            tier2_pool=ctx["tier2_pool"],
            gat_scores_snippet=ctx["gat_scores_snippet"],
        )
        resp = self.client.generate_text(
            prompt=prompt, model=self.model_name, temperature=self.temperature
        )
        return AgentUtils.extract_json(resp), resp

    def _apply_additions(
        self,
        current: Dict[str, List[str]],
        additions: Iterable[str],
        valid: Set[str],
    ) -> Dict[str, List[str]]:
        updated = {t: list(cols) for t, cols in current.items()}
        for node in additions:
            if node not in valid or "." not in node:
                continue
            tbl, col = node.split(".", 1)
            updated.setdefault(tbl, [])
            if col not in updated[tbl]:
                updated[tbl].append(col)
        return updated

    def refine(
        self,
        query: str,
        subgraph: Dict[str, List[str]],
        db_id: str = None,
        tier2_pool: List[str] = None,
        gat_scores: Dict[str, float] = None,
        metadata: Dict[str, Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if not subgraph:
            return {
                "status": "Unanswerable",
                "final_nodes": [],
                "reasoning": "Empty input subgraph",
            }

        tier1_flat = set(self._flatten(subgraph))
        tier2_set = set(tier2_pool or [])
        tier2_set -= tier1_flat
        gat_scores = gat_scores or {}

        current = self._prune(query, subgraph, db_id)
        current_flat = set(self._flatten(current))
        tier1_dropped = sorted(tier1_flat - current_flat)

        trace = [
            f"Prune: {len(tier1_flat)}→{len(current_flat)} "
            f"(Tier-1 dropped={len(tier1_dropped)}, Tier-2 pool={len(tier2_set)})"
        ]
        trace_detail: List[Dict[str, Any]] = [{
            "step": "prune",
            "kept": sorted(current_flat),
            "dropped": tier1_dropped,
            "tier2_pool": sorted(tier2_set),
        }]

        if not tier1_dropped and not tier2_set:
            final_nodes = self._flatten(current)
            return {
                "status": "Answerable" if final_nodes else "Unanswerable",
                "final_nodes": final_nodes,
                "reasoning": " | ".join(trace),
                "trace": trace_detail,
            }

        restore_res, restore_raw = self._restore(
            query=query,
            current=current,
            tier1_dropped=tier1_dropped,
            tier2_pool=sorted(tier2_set),
            gat_scores=gat_scores,
        )
        restore = restore_res.get("restore", []) or []
        promote = restore_res.get("promote", []) or []

        valid_restore = set(tier1_dropped)
        valid_promote = tier2_set
        restored = [n for n in restore if n in valid_restore]
        promoted = [n for n in promote if n in valid_promote]

        updated = self._apply_additions(
            current, restored + promoted, valid_restore | valid_promote | current_flat
        )
        trace.append(
            f"Restore agent: restored={len(restored)} promoted={len(promoted)}"
        )
        trace_detail.append({
            "step": "restore_agent",
            "restored": restored,
            "promoted": promoted,
            "raw": restore_raw,
        })

        final_nodes = self._flatten(updated)
        return {
            "status": "Answerable" if final_nodes else "Unanswerable",
            "final_nodes": final_nodes,
            "reasoning": " | ".join(trace),
            "trace": trace_detail,
            "tier1_dropped_count": len(tier1_dropped),
            "tier2_pool_count": len(tier2_set),
            "restored_count": len(restored),
            "promoted_count": len(promoted),
        }
