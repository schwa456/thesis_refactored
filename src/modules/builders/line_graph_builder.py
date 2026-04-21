"""LineGraphBuilder — proposal #3 (EHGAT) infrastructure.

Wraps a base HeteroGraphBuilder; promotes each edge in the PCST-flat graph to
a node and connects two edge-nodes when they share an original node. The
output is a HeteroData with a single node type `edge_node` that downstream
EHGAT-style selectors can consume.

Returned metadata keeps a pointer to the original HeteroData/metadata so
joint losses (node-level + edge-level) remain feasible.
"""
import time
from typing import Dict, List, Tuple, Any

import torch
from torch_geometric.data import HeteroData

from modules.registry import register
from modules.base import BaseGraphBuilder
from modules.builders.graph_builder import (
    HeteroGraphBuilder,
    EnrichedHeteroGraphBuilder,
    TripletGraphBuilder,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Fixed type order keeps edge-type one-hot consistent across DBs.
EDGE_TYPE_ORDER: List[str] = [
    "belongs_to", "is_source_of", "points_to", "table_to_table",
]


@register("builder", "LineGraphBuilder")
class LineGraphBuilder(BaseGraphBuilder):
    """Convert (HeteroData, metadata) → LineGraphHeteroData.

    Args:
        base: dotted name of base builder
              ("HeteroGraphBuilder"|"EnrichedHeteroGraphBuilder"|"TripletGraphBuilder").
        base_params: kwargs forwarded to the base builder.
        include_endpoint_diff: include `src - dst` as a feature dimension.
        skip_macro_edges: drop `table_to_table` edges before line-graph
                          conversion (controls combinatorial blow-up on large
                          schemas with many FK macro edges).
    """

    _BASE_REGISTRY = {
        "HeteroGraphBuilder": HeteroGraphBuilder,
        "EnrichedHeteroGraphBuilder": EnrichedHeteroGraphBuilder,
        "TripletGraphBuilder": TripletGraphBuilder,
    }

    def __init__(
        self,
        base: str = "EnrichedHeteroGraphBuilder",
        base_params: Dict[str, Any] = None,
        include_endpoint_diff: bool = True,
        skip_macro_edges: bool = False,
        **kwargs,
    ):
        super().__init__()
        if base not in self._BASE_REGISTRY:
            raise ValueError(
                f"Unknown base builder '{base}'. "
                f"Choose from {list(self._BASE_REGISTRY)}."
            )
        bp = dict(base_params or {})
        bp.update(kwargs)  # passthrough
        self.base = self._BASE_REGISTRY[base](**bp)
        self.base_name = base
        self.include_endpoint_diff = include_endpoint_diff
        self.skip_macro_edges = skip_macro_edges
        self.last_info: Dict[str, Any] = {}
        logger.info(
            f"LineGraphBuilder initialized over {base} "
            f"(diff={include_endpoint_diff}, skip_macro={skip_macro_edges})"
        )

    def _flat_node_embeddings(self, data: HeteroData) -> Tuple[torch.Tensor, int]:
        embs = [data["table"].x, data["column"].x]
        if "fk_node" in data.node_types and data["fk_node"].x.size(0) > 0:
            embs.append(data["fk_node"].x)
        else:
            dim = data["table"].x.size(1)
            embs.append(torch.empty((0, dim), dtype=data["table"].x.dtype))
        flat = torch.cat(embs, dim=0)
        return flat, flat.size(1)

    def build(self, db_id: str, db_dir: str) -> Tuple[HeteroData, Dict[str, Any]]:
        t_total = time.perf_counter()
        t = time.perf_counter()
        data, meta = self.base.build(db_id=db_id, db_dir=db_dir)
        t_base = time.perf_counter() - t

        flat_emb, embed_dim = self._flat_node_embeddings(data)
        n_t = len(meta["table_to_id"])
        n_c = len(meta["col_to_id"])
        n_f = len(meta["fk_to_id"])
        n_total = n_t + n_c + n_f
        if flat_emb.size(0) != n_total:
            logger.warning(
                f"[{db_id}] flat embedding count {flat_emb.size(0)} != "
                f"expected {n_total}. Padding."
            )
            if flat_emb.size(0) < n_total:
                pad = torch.zeros((n_total - flat_emb.size(0), embed_dim))
                flat_emb = torch.cat([flat_emb, pad], dim=0)
            else:
                flat_emb = flat_emb[:n_total]

        type_to_idx = {t: i for i, t in enumerate(EDGE_TYPE_ORDER)}
        n_types = len(EDGE_TYPE_ORDER)

        triplet_emb = meta.get("edge_embeddings")
        triplet_dim = triplet_emb.size(1) if triplet_emb is not None and triplet_emb.numel() else 0

        edges = meta["edges"]
        edge_types = meta["edge_types"]
        if self.skip_macro_edges:
            kept = [(e, t, i) for i, (e, t) in enumerate(zip(edges, edge_types))
                    if t != "table_to_table"]
            edges = [e for e, _, _ in kept]
            edge_types = [t for _, t, _ in kept]
            kept_indices = [i for _, _, i in kept]
        else:
            kept_indices = list(range(len(edges)))

        feat_per_edge: List[torch.Tensor] = []
        edge_node_to_orig: Dict[int, Tuple[int, int, str]] = {}
        orig_node_to_edges: Dict[int, List[int]] = {}

        for new_idx, ((src, dst), et) in enumerate(zip(edges, edge_types)):
            type_oh = torch.zeros(n_types)
            type_oh[type_to_idx.get(et, 0)] = 1.0
            src_emb = flat_emb[src] if src < flat_emb.size(0) else torch.zeros(embed_dim)
            dst_emb = flat_emb[dst] if dst < flat_emb.size(0) else torch.zeros(embed_dim)
            mean_emb = 0.5 * (src_emb + dst_emb)
            parts = [type_oh, mean_emb]
            if self.include_endpoint_diff:
                parts.append(src_emb - dst_emb)
            if triplet_dim > 0:
                orig_idx = kept_indices[new_idx]
                if orig_idx < triplet_emb.size(0):
                    parts.append(triplet_emb[orig_idx])
                else:
                    parts.append(torch.zeros(triplet_dim))
            feat_per_edge.append(torch.cat(parts, dim=0))
            edge_node_to_orig[new_idx] = (int(src), int(dst), et)
            orig_node_to_edges.setdefault(int(src), []).append(new_idx)
            orig_node_to_edges.setdefault(int(dst), []).append(new_idx)

        if feat_per_edge:
            edge_x = torch.stack(feat_per_edge, dim=0)
        else:
            feat_dim = n_types + embed_dim * (2 if self.include_endpoint_diff else 1) + triplet_dim
            edge_x = torch.empty((0, feat_dim))

        line_src, line_dst = [], []
        for shared_node, e_list in orig_node_to_edges.items():
            for i, ei in enumerate(e_list):
                for ej in e_list[i + 1:]:
                    line_src.extend([ei, ej])
                    line_dst.extend([ej, ei])

        line_data = HeteroData()
        line_data["edge_node"].x = edge_x
        if line_src:
            line_data["edge_node", "shares_node", "edge_node"].edge_index = torch.tensor(
                [line_src, line_dst], dtype=torch.long
            )
        else:
            line_data["edge_node", "shares_node", "edge_node"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            )

        line_meta: Dict[str, Any] = {
            "edge_node_to_orig": edge_node_to_orig,
            "orig_node_to_edges": orig_node_to_edges,
            "edge_type_order": EDGE_TYPE_ORDER,
            "edge_type_to_idx": type_to_idx,
            "edge_feature_dim": edge_x.size(1) if edge_x.numel() else 0,
            "edge_label_rule": "both_endpoints_gold",
            "orig_data": data,
            "orig_metadata": meta,
        }
        # Forward FK reachability + base ids so EHGAT trainers don't have to
        # reach through `orig_metadata`.
        for k in (
            "table_to_id", "col_to_id", "fk_to_id", "node_metadata",
            "fk_adjacency", "fk_adjacency_undirected", "fk_reachability",
            "fk_distance", "fk_shortest_paths", "fk_components",
            "fk_num_components", "fk_edge_lookup",
        ):
            if k in meta:
                line_meta[k] = meta[k]

        orig_builder_info = meta.get("builder_info", {}) or {}
        orig_edges_count = len(meta.get("edges", []) or [])
        edge_type_counts: Dict[str, int] = {}
        for et in edge_types:
            edge_type_counts[et] = edge_type_counts.get(et, 0) + 1

        self.last_info = {
            "builder_type": "LineGraphBuilder",
            "builder_db_id": db_id,
            "builder_base_name": self.base_name,
            "builder_include_endpoint_diff": bool(self.include_endpoint_diff),
            "builder_skip_macro_edges": bool(self.skip_macro_edges),
            "builder_orig_edges_total": int(orig_edges_count),
            "builder_line_edge_nodes": int(edge_x.size(0)),
            "builder_line_edge_feature_dim": int(line_meta.get("edge_feature_dim", 0) or 0),
            "builder_line_shares_edges": int(len(line_src) // 2),
            "builder_line_edge_type_counts": edge_type_counts,
            "builder_triplet_dim_used": int(triplet_dim),
            "builder_embed_dim_flat": int(embed_dim),
            "builder_base_info": dict(orig_builder_info),
            "builder_timings": {
                "base_build_s": float(t_base),
                "total_s": float(time.perf_counter() - t_total),
            },
        }
        line_meta["builder_info"] = dict(self.last_info)
        return line_data, line_meta
