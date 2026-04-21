"""Neurosymbolic Layer 1 Selector (proposal #9 — Layer 1).

Adds a deterministic FK-reachability prior to EnsembleSelector's score:

    boosted = ensemble_scores + λ · reach_mask

`reach_mask[v] = 1.0` if v belongs to a table reachable (over the undirected FK
graph) from any query-identified anchor table, else 0.0.

Builder B-III populates `metadata["fk_reachability"]` (T,T bool). If absent
(older cache / non-enriched builder), the hook is a no-op and the selector
degrades gracefully to EnsembleSelector behavior.
"""
from typing import Any, Dict, List, Set

import re
import numpy as np
import torch

from modules.registry import register
from modules.selectors.ensemble_selector import EnsembleSelector
from utils.logger import get_logger

logger = get_logger(__name__)


@register("selector", "NeurosymbolicL1Selector")
class NeurosymbolicL1Selector(EnsembleSelector):
    """Ensemble + FK-reachability prior (proposal #9 Layer 1).

    λ·reach_mask is *added* to normalized ensemble_scores. Since both operands
    live in [0, 1], λ acts as a direct additive bonus — λ=0.1 means anchor-
    reachable nodes get a +0.1 bump.
    """

    def __init__(
        self,
        weight_path: str,
        lambda_sym: float = 0.1,
        anchor_min_token_len: int = 3,
        **kwargs,
    ):
        super().__init__(weight_path=weight_path, **kwargs)
        self.lambda_sym = float(lambda_sym)
        self.anchor_min_token_len = int(anchor_min_token_len)
        self.last_info: Dict[str, Any] = {}
        logger.info(
            f"Initialized NeurosymbolicL1Selector "
            f"(lambda_sym={self.lambda_sym}, alpha={self.alpha}, top_k={self.top_k})"
        )

    # ------------------------------------------------------------------ hook
    def _post_ensemble_hook(
        self,
        ensemble_scores: torch.Tensor,
        question: str,
        graph_data,
        metadata: Dict[str, Any],
    ) -> torch.Tensor:
        reach = metadata.get("fk_reachability")
        table_to_id = metadata.get("table_to_id", {})

        if reach is None or len(table_to_id) == 0:
            logger.debug("[NSL1] fk_reachability missing — falling back to ensemble")
            self.last_info = {"nsl1_active": False, "nsl1_reason": "no_fk_reachability"}
            return ensemble_scores

        anchors = self._identify_anchors(question, metadata)
        if not anchors:
            logger.debug("[NSL1] no anchor table matched — no boost applied")
            self.last_info = {
                "nsl1_active": False, "nsl1_reason": "no_anchor_matched",
                "nsl1_num_tables": len(table_to_id),
            }
            return ensemble_scores

        reach_mask = self._build_reach_mask(anchors, metadata, ensemble_scores.numel())
        reach_pos = int(reach_mask.sum().item())
        if reach_pos == 0:
            self.last_info = {
                "nsl1_active": False, "nsl1_reason": "reach_mask_empty",
                "nsl1_num_anchors": len(anchors),
            }
            return ensemble_scores

        logger.debug(
            f"[NSL1] anchors={len(anchors)}/{len(table_to_id)} "
            f"reach_mask_pos={reach_pos}/{reach_mask.numel()}"
        )
        self.last_info = {
            "nsl1_active": True,
            "nsl1_num_anchors": len(anchors),
            "nsl1_num_tables": len(table_to_id),
            "nsl1_reach_mask_pos": reach_pos,
            "nsl1_reach_mask_total": int(reach_mask.numel()),
            "nsl1_lambda_sym": self.lambda_sym,
        }
        return ensemble_scores + self.lambda_sym * reach_mask

    # ------------------------------------------------------------- internals
    def _question_tokens(self, question: str) -> Set[str]:
        return {
            t for t in re.findall(r"[a-zA-Z0-9]+", question.lower())
            if len(t) >= self.anchor_min_token_len
        }

    @staticmethod
    def _snake_words(name: str) -> List[str]:
        # "retail_complains" -> ["retail", "complains"]; "orderDate" -> ["order", "date"]
        s = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
        return [w for w in s.lower().replace("_", " ").split() if w]

    def _identify_anchors(self, question: str, metadata: Dict[str, Any]) -> Set[int]:
        """Return table indices whose name (or one of their columns) hits the question."""
        table_to_id: Dict[str, int] = metadata.get("table_to_id", {})
        col_to_id: Dict[str, int] = metadata.get("col_to_id", {})
        q_tokens = self._question_tokens(question)
        if not q_tokens:
            return set()

        anchors: Set[int] = set()
        for tname, t_id in table_to_id.items():
            words = self._snake_words(tname)
            if any(
                w in q_tokens
                for w in words
                if len(w) >= self.anchor_min_token_len
            ):
                anchors.add(int(t_id))

        for full_col, _ in col_to_id.items():
            if "." not in full_col:
                continue
            tpart, cpart = full_col.rsplit(".", 1)
            if tpart not in table_to_id:
                continue
            words = self._snake_words(cpart)
            if any(
                w in q_tokens
                for w in words
                if len(w) >= self.anchor_min_token_len
            ):
                anchors.add(int(table_to_id[tpart]))

        return anchors

    def _build_reach_mask(
        self,
        anchors: Set[int],
        metadata: Dict[str, Any],
        num_nodes: int,
    ) -> torch.Tensor:
        """Build a flat (num_nodes,) mask: 1.0 for table/column/fk nodes whose
        owning table is reachable from any anchor, 0.0 otherwise.
        """
        reach = metadata["fk_reachability"]  # (T, T) bool
        table_to_id: Dict[str, int] = metadata["table_to_id"]
        col_to_id: Dict[str, int] = metadata.get("col_to_id", {})
        fk_to_id: Dict[str, int] = metadata.get("fk_to_id", {})

        T = len(table_to_id)
        num_t = T
        num_c = len(col_to_id)
        num_fk = len(fk_to_id)

        # Reachable table set (bool vector length T)
        if isinstance(reach, np.ndarray) and reach.size > 0:
            anchor_idx = np.array(sorted(anchors), dtype=np.int64)
            # any-anchor reachability; diagonal of fk_reachability is True by construction
            reachable_tables = reach[anchor_idx].any(axis=0)  # (T,)
        else:
            reachable_tables = np.zeros(T, dtype=bool)
            for a in anchors:
                if 0 <= a < T:
                    reachable_tables[a] = True

        mask = torch.zeros(num_nodes, dtype=torch.float32)

        # Tables
        for t_idx in range(num_t):
            if t_idx < num_nodes and reachable_tables[t_idx]:
                mask[t_idx] = 1.0

        # Columns inherit from their owning table
        for full_col, c_local in col_to_id.items():
            flat_idx = int(c_local) + num_t
            if flat_idx >= num_nodes:
                continue
            tpart = full_col.rsplit(".", 1)[0]
            t_idx = table_to_id.get(tpart)
            if t_idx is not None and reachable_tables[t_idx]:
                mask[flat_idx] = 1.0

        # FK nodes: reachable if *either* endpoint table is reachable from anchors
        for fk_key, fk_local in fk_to_id.items():
            flat_idx = int(fk_local) + num_t + num_c
            if flat_idx >= num_nodes:
                continue
            # fk_key: "from_table.from_col->to_table.to_col"
            try:
                left, right = fk_key.split("->")
                from_t = left.split(".", 1)[0]
                to_t = right.split(".", 1)[0]
            except ValueError:
                continue
            from_id = table_to_id.get(from_t)
            to_id = table_to_id.get(to_t)
            if (from_id is not None and reachable_tables[from_id]) or (
                to_id is not None and reachable_tables[to_id]
            ):
                mask[flat_idx] = 1.0

        return mask
