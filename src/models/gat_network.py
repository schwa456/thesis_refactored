import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
from torch_geometric.data import HeteroData
from typing import Any, Dict, Optional, List, Tuple


SUPERNODE_EDGE_DIRECTIONS = {"bidirectional", "directed_from_sn"}
SUPERNODE_TOPK_CRITERIA = {"raw", "cosine", "ce"}
SUPERNODE_THRESHOLD_MODES = {"top_k", "percentile", "abs_tau"}
SUPERNODE_SCORE_NORMS = {"minmax", "none"}


def _tensor_stats(t: torch.Tensor, dead_eps: float = 1e-4) -> Dict[str, float]:
    """Cheap summary statistics for a 2D embedding tensor [N, D]."""
    if t is None or t.numel() == 0:
        return {"n": 0}
    with torch.no_grad():
        norms = t.detach().norm(dim=-1)
        abs_t = t.detach().abs()
        dead = (abs_t < dead_eps).float().mean().item()
        return {
            "n": int(t.shape[0]),
            "d": int(t.shape[-1]),
            "norm_mean": float(norms.mean().item()),
            "norm_std": float(norms.std().item()) if norms.numel() > 1 else 0.0,
            "abs_mean": float(abs_t.mean().item()),
            "abs_p50": float(abs_t.median().item()),
            "abs_p95": float(abs_t.flatten().kthvalue(max(1, int(abs_t.numel() * 0.95))).values.item()) if abs_t.numel() > 0 else 0.0,
            "dead_ratio": dead,
        }


class SchemaHeteroGAT(nn.Module):
    """
    이종 그래프(Heterogeneous Graph) 구조의 DB 스키마를 학습하는 모델.
    Table, Column, FK_Node가 서로 Attention 기반의 메시지 패싱을 수행하여,
    텍스트 의미(Semantic)와 구조적 위치(Structure)가 융합된 임베딩을 출력합니다.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 3, heads: int = 4, query_conditioned: bool = False,
                 query_supernode: bool = False, enable_stats: bool = False,
                 supernode_edge_direction: str = "bidirectional",
                 supernode_topk: Optional[int] = None,
                 supernode_topk_criterion: str = "raw",
                 supernode_threshold_mode: str = "top_k",
                 supernode_threshold_value: Optional[float] = None,
                 supernode_score_normalization: str = "minmax"):
        super(SchemaHeteroGAT, self).__init__()
        self.num_layers = num_layers
        self.query_conditioned = query_conditioned
        self.query_supernode = query_supernode
        self.enable_stats = enable_stats
        self.last_layer_stats: Dict[str, Any] = {}

        # V-2/V-3 옵션 (SuperNode v2)
        if supernode_edge_direction not in SUPERNODE_EDGE_DIRECTIONS:
            raise ValueError(f"supernode_edge_direction must be in {SUPERNODE_EDGE_DIRECTIONS}")
        if supernode_topk_criterion not in SUPERNODE_TOPK_CRITERIA:
            raise ValueError(f"supernode_topk_criterion must be in {SUPERNODE_TOPK_CRITERIA}")
        if supernode_topk is not None and not query_supernode:
            raise ValueError("supernode_topk 는 query_supernode=True 에서만 의미가 있음")
        if supernode_threshold_mode not in SUPERNODE_THRESHOLD_MODES:
            raise ValueError(
                f"supernode_threshold_mode must be in {SUPERNODE_THRESHOLD_MODES}, got {supernode_threshold_mode}"
            )
        if supernode_score_normalization not in SUPERNODE_SCORE_NORMS:
            raise ValueError(
                f"supernode_score_normalization must be in {SUPERNODE_SCORE_NORMS}, got {supernode_score_normalization}"
            )
        if supernode_threshold_mode in ("percentile", "abs_tau"):
            if not query_supernode:
                raise ValueError(f"supernode_threshold_mode={supernode_threshold_mode} requires query_supernode=True")
            if supernode_threshold_value is None:
                raise ValueError(
                    f"supernode_threshold_mode={supernode_threshold_mode} requires supernode_threshold_value"
                )
        self.supernode_edge_direction = supernode_edge_direction
        self.supernode_topk = supernode_topk
        self.supernode_topk_criterion = supernode_topk_criterion
        self.supernode_threshold_mode = supernode_threshold_mode
        self.supernode_threshold_value = (
            float(supernode_threshold_value) if supernode_threshold_value is not None else None
        )
        self.supernode_score_normalization = supernode_score_normalization

        # query_conditioned=True 시 node feature에 query를 concat → 입력 차원 2배
        effective_in = in_channels * 2 if query_conditioned else in_channels

        # 1. 초기 차원 축소 및 통일 (PLM의 384 차원 -> GAT의 256 차원 등)
        node_types = ['table', 'column', 'fk_node']
        if query_supernode:
            node_types.append('query_node')

        self.lin_dict = nn.ModuleDict({
            nt: Linear(effective_in, hidden_channels) for nt in node_types
        })

        # 2. Heterogeneous GAT 레이어 정의
        base_edge_types = {
            # A. Table <-> Column 상호작용
            ('table', 'has_column', 'column'): GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False),
            ('column', 'belongs_to', 'table'): GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False),

            # B. Column <-> FK_Node 상호작용 (논문의 핵심 기여 포인트)
            ('column', 'is_source_of', 'fk_node'): GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False),
            ('fk_node', 'points_to', 'column'): GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False),

            # C. Table <-> Table 거시적 상호작용
            ('table', 'table_to_table', 'table'): GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False)
        }

        # Query Super Node edges.
        # V-2: supernode_edge_direction ∈ {bidirectional, directed_from_sn}.
        #   - bidirectional (default, 기존): query→schema + schema→query 양방향 등록
        #   - directed_from_sn: query→schema 만. schema→query 제거. SN 은 self-loop 로 보존.
        # V-3: supernode_topk > 0 시 forward 시점에 query→schema edge_index 를 top-k 로 제한.
        supernode_edge_types = {}
        if query_supernode:
            for schema_nt in ['table', 'column', 'fk_node']:
                supernode_edge_types[('query_node', f'attends_to_{schema_nt}', schema_nt)] = \
                    GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False)
                if self.supernode_edge_direction == "bidirectional":
                    supernode_edge_types[(schema_nt, f'attended_by_{schema_nt}', 'query_node')] = \
                        GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False)
            # V-2: directed_from_sn 의 경우 query_node 가 HeteroConv 메시지를 받지 못해 drop 됨.
            # self-loop edge type 을 등록해 두고, forward 시 identity edge_index 를 주입하여 보존.
            if self.supernode_edge_direction == "directed_from_sn":
                supernode_edge_types[('query_node', 'self_loop', 'query_node')] = \
                    GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            all_edge_types = {**base_edge_types, **supernode_edge_types}
            # 각 layer마다 새로운 GATv2Conv 인스턴스 생성
            conv_dict = {}
            for edge_type, _ in all_edge_types.items():
                conv_dict[edge_type] = GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False)
            conv = HeteroConv(conv_dict, aggr='mean')
            self.convs.append(conv)

        # 3. 최종 출력 차원 변환 (Alignment Layer와 연결될 차원)
        self.out_lin_dict = nn.ModuleDict({
            nt: Linear(hidden_channels * heads, out_channels) for nt in node_types
        })

        self.skip_dict = nn.ModuleDict({
            nt: Linear(effective_in, out_channels) for nt in node_types
        })

    def forward(self, x_dict: dict, edge_index_dict: dict,
                query_emb: torch.Tensor = None,
                active_num_layers: Optional[int] = None) -> dict:
        """
        x_dict: {'table': Tensor, 'column': Tensor, 'fk_node': Tensor}
        edge_index_dict: {('table', 'has_column', 'column'): Tensor, ...}
        query_emb: Optional[Tensor] — [1, dim] or [dim]. query_conditioned=True 시 사용.
        active_num_layers: Optional[int] — inference-time early-exit depth. None 시
            self.num_layers 전체 사용. 값이 주어지면 clamp([1, self.num_layers]) 후
            `self.convs[:L']` 까지만 message passing 수행. Proposal C H2 per-DB dynamic
            num_layers 를 checkpoint 재학습 없이 측정하기 위한 hook.
        """
        stats_on = self.enable_stats
        t_start = time.perf_counter() if stats_on else 0.0
        layer_stats: Dict[str, Any] = {"layers": []} if stats_on else {}

        # V-3 / V-3-ext: SuperNode threshold filtering (pre-GAT, per-query cosine).
        # Selective 연결: SN → 선별된 schema node 만 유지. mode 별 dispatch (top_k / percentile / abs_tau).
        if self.query_supernode and query_emb is not None and self._supernode_filter_active():
            mask = self._compute_supernode_mask(query_emb, x_dict)
            edge_index_dict = self._apply_topk_to_edges(edge_index_dict, mask)

        # V-2: directed_from_sn 시 query_node 가 HeteroConv drop 되지 않도록 self-loop edge 주입.
        if (self.query_supernode
                and self.supernode_edge_direction == "directed_from_sn"
                and 'query_node' in x_dict):
            edge_index_dict = self._inject_sn_self_loop(edge_index_dict, x_dict)

        # Query Conditioning: query_emb를 모든 노드 feature에 concat
        if self.query_conditioned and query_emb is not None:
            if query_emb.dim() == 1:
                query_emb = query_emb.unsqueeze(0)
            augmented = {}
            for nt, x in x_dict.items():
                q_exp = query_emb.expand(x.size(0), -1)
                augmented[nt] = torch.cat([x, q_exp], dim=-1)
            x_dict = augmented

        if stats_on:
            layer_stats["input"] = {nt: _tensor_stats(x) for nt, x in x_dict.items()}

        # Step 1: Input Projection (Linear 통과 및 Activation)
        out_dict = {}
        for node_type, x in x_dict.items():
            out_dict[node_type] = F.leaky_relu(self.lin_dict[node_type](x))

        if stats_on:
            layer_stats["after_input_proj"] = {nt: _tensor_stats(x) for nt, x in out_dict.items()}

        # Step 2: Message Passing (GAT Layers)
        if active_num_layers is None:
            depth = self.num_layers
        else:
            depth = max(1, min(int(active_num_layers), self.num_layers))
        for i in range(depth):
            pre = out_dict
            out_dict = self.convs[i](out_dict, edge_index_dict)
            out_dict = {node_type: F.elu(x) for node_type, x in out_dict.items()}

            if stats_on:
                per_layer = {}
                for nt, x_post in out_dict.items():
                    stat = _tensor_stats(x_post)
                    x_pre = pre.get(nt)
                    if x_pre is not None and x_post.shape == x_pre.shape:
                        with torch.no_grad():
                            delta = (x_post - x_pre).norm(dim=-1)
                            stat["delta_norm_mean"] = float(delta.mean().item())
                    per_layer[nt] = stat
                layer_stats["layers"].append(per_layer)

        # Step 3: Output Projection
        final_dict = {}
        skip_norms = {} if stats_on else None
        out_lin_norms = {} if stats_on else None
        for node_type, x in out_dict.items():
            out_lin = self.out_lin_dict[node_type](x)
            skip = self.skip_dict[node_type](x_dict[node_type])
            final_dict[node_type] = out_lin + skip
            if stats_on:
                with torch.no_grad():
                    out_lin_norms[node_type] = float(out_lin.detach().norm(dim=-1).mean().item())
                    skip_norms[node_type] = float(skip.detach().norm(dim=-1).mean().item())

        if stats_on:
            layer_stats["output"] = {nt: _tensor_stats(x) for nt, x in final_dict.items()}
            layer_stats["out_lin_norm_mean"] = out_lin_norms
            layer_stats["skip_norm_mean"] = skip_norms
            # Skip contribution ratio: skip_norm / (skip_norm + out_lin_norm)
            layer_stats["skip_ratio"] = {
                nt: (skip_norms[nt] / (skip_norms[nt] + out_lin_norms[nt] + 1e-12))
                for nt in skip_norms
            }
            layer_stats["forward_time_s"] = float(time.perf_counter() - t_start)
            self.last_layer_stats = layer_stats

        return final_dict

    # ──────────────────────────────────────────────────────────────
    # V-2 / V-3 helper methods
    # ──────────────────────────────────────────────────────────────
    def _supernode_filter_active(self) -> bool:
        if self.supernode_threshold_mode == "top_k":
            return self.supernode_topk is not None and self.supernode_topk > 0
        return self.supernode_threshold_value is not None

    def _compute_supernode_mask(self, query_emb: torch.Tensor,
                                x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """V-3 / V-3-ext: pre-GAT cosine score 기반 schema 노드 mask 산출.

        threshold_mode 별 dispatch:
          - top_k:        torch.topk on (norm 적용된) score 로 K 개.
          - percentile:   torch.quantile cutoff (value=80 → P80).
          - abs_tau:      score >= value (norm 후 절대 cutoff).

        score_normalization='minmax' 시 모든 mode 가 per-query [0,1] 정규화 후 비교. analyzer
        raw_score_distribution_for_directed_topk.md 의 정의와 동일 (학위 논문 Part III base).
        """
        q = query_emb if query_emb.dim() == 2 else query_emb.unsqueeze(0)
        if q.size(0) > 1:
            q = q.mean(dim=0, keepdim=True)
        q_norm = F.normalize(q, dim=-1)

        schema_types = ['table', 'column', 'fk_node']
        scores_list: List[torch.Tensor] = []
        type_tags: List[str] = []
        local_idx: List[int] = []
        for nt in schema_types:
            if nt not in x_dict or x_dict[nt].size(0) == 0:
                continue
            n_norm = F.normalize(x_dict[nt], dim=-1)
            sc = (n_norm @ q_norm.t()).squeeze(-1)
            scores_list.append(sc)
            type_tags.extend([nt] * sc.size(0))
            local_idx.extend(range(sc.size(0)))

        masks: Dict[str, torch.Tensor] = {}
        for nt in schema_types:
            if nt in x_dict:
                masks[nt] = torch.zeros(x_dict[nt].size(0), dtype=torch.bool,
                                        device=x_dict[nt].device)

        if not scores_list:
            return masks

        combined = torch.cat(scores_list, dim=0)
        if self.supernode_score_normalization == "minmax" and combined.numel() > 0:
            cmin, cmax = combined.min(), combined.max()
            combined_norm = (combined - cmin) / (cmax - cmin) if cmax > cmin else combined
        else:
            combined_norm = combined

        if self.supernode_threshold_mode == "top_k":
            k = min(int(self.supernode_topk), combined_norm.size(0))
            keep_idx = torch.topk(combined_norm, k=k).indices.tolist()
            for i in keep_idx:
                masks[type_tags[i]][local_idx[i]] = True
        elif self.supernode_threshold_mode == "percentile":
            p = float(self.supernode_threshold_value) / 100.0
            cutoff = torch.quantile(combined_norm.to(torch.float32), p)
            keep = (combined_norm >= cutoff).nonzero(as_tuple=False).flatten().tolist()
            for i in keep:
                masks[type_tags[i]][local_idx[i]] = True
        else:  # abs_tau
            cutoff = float(self.supernode_threshold_value)
            keep = (combined_norm >= cutoff).nonzero(as_tuple=False).flatten().tolist()
            for i in keep:
                masks[type_tags[i]][local_idx[i]] = True
        return masks

    # Backward-compat alias — 기존 분석/유틸 코드가 _compute_topk_mask 를 직접 호출하면 dispatch.
    def _compute_topk_mask(self, query_emb: torch.Tensor,
                           x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._compute_supernode_mask(query_emb, x_dict)

    def _apply_topk_to_edges(self, edge_index_dict: dict,
                             masks: Dict[str, torch.Tensor]) -> dict:
        """SN → schema (+ 역방향 bidirectional 시) edge 를 top-k 로 제한.
        caller 의 edge_index_dict 는 보존하고 shallow-copy 를 반환."""
        filtered = dict(edge_index_dict)
        for nt in ['table', 'column', 'fk_node']:
            mask = masks.get(nt)
            if mask is None:
                continue
            fwd_key = ('query_node', f'attends_to_{nt}', nt)
            if fwd_key in filtered:
                ei = filtered[fwd_key]
                if ei.numel() > 0:
                    keep = mask[ei[1]]
                    filtered[fwd_key] = ei[:, keep]
            rev_key = (nt, f'attended_by_{nt}', 'query_node')
            if rev_key in filtered:
                ei = filtered[rev_key]
                if ei.numel() > 0:
                    keep = mask[ei[0]]
                    filtered[rev_key] = ei[:, keep]
        return filtered

    def _inject_sn_self_loop(self, edge_index_dict: dict,
                             x_dict: Dict[str, torch.Tensor]) -> dict:
        """directed_from_sn 시 query_node 를 HeteroConv 에서 보존하기 위한 self-loop 주입."""
        num_q = x_dict['query_node'].size(0)
        if num_q == 0:
            return edge_index_dict
        dev = x_dict['query_node'].device
        idx = torch.arange(num_q, device=dev, dtype=torch.long)
        new = dict(edge_index_dict)
        new[('query_node', 'self_loop', 'query_node')] = torch.stack([idx, idx], dim=0)
        return new