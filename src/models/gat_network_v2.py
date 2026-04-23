"""
SchemaHeteroGAT v2 — s06 ablation용 확장 모델.

기존 SchemaHeteroGAT 대비 추가된 옵션:
  (a) pairnorm_mode: 'none' | 'pairnorm' — 각 GAT layer 후 PairNorm (Zhao & Akoglu, ICLR 2020)
  (b) initial_residual_alpha: float ∈ [0, 1) — APPNP/GCNII 스타일 초기 residual 주입
      h_l = (1 - α) · GAT(h_{l-1}) + α · h_0  (h_0 = lin_dict 출력)
  (c) jumping_knowledge: 'none' | 'concat' | 'max' — 모든 layer 출력 융합 (JK-Net, ICML 2018)
  (d) dual_stream: bool — schema-only stream + query-only stream 분리. True일 때:
      - schema 노드는 query concat 없이 GAT 통과
      - query는 별도 MLP로 인코딩
      - 최종 출력 = MLP_head(concat(h_schema, z_q, h_schema⊙z_q))

V-Track (2026-04-21 QCondGAT v2 계열):
  V-1 (num_layers_mode): 'fixed' | 'D_max' | 'D_max_plus1'
      — Per-DB dynamic GAT depth. num_layers 는 upper bound (num_layers_max),
        실제 사용 깊이는 diameter_dict[db_name] 에서 resolve.
  V-2 (supernode_edge_direction): 'bidirectional' | 'directed_from_sn'
      — directed_from_sn 시 schema→SN edge 제거, SN 을 self-loop 로 보존.
  V-3 (supernode_topk + supernode_topk_criterion): SN → top-k schema 만 연결.
      criterion='raw' (Phase 1) = pre-GAT cosine.

모든 옵션은 default에서 off이므로 기존 실험과 호환.
"""
from __future__ import annotations

import os
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear


SUPERNODE_EDGE_DIRECTIONS = {"bidirectional", "directed_from_sn"}
SUPERNODE_TOPK_CRITERIA = {"raw", "cosine", "ce"}
NUM_LAYERS_MODES = {"fixed", "D_max", "D_max_plus1"}


class PairNorm(nn.Module):
    """Zhao & Akoglu (ICLR 2020): feature의 pairwise distance 합을 상수로 유지.
    mode='PN-SI' (scale-individually) 구현 — 각 노드의 L2 norm을 1로 맞춘 뒤 s*sqrt(N)로 스케일.
    """

    def __init__(self, scale: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0:
            return x
        x = x - x.mean(dim=0, keepdim=True)
        row_norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        x = self.scale * x / row_norm
        return x


class SchemaHeteroGATv2(nn.Module):
    """s06 용 확장판. 인터페이스는 기존 SchemaHeteroGAT 과 호환.

    Args:
        in_channels: PLM embedding 차원 (e.g., 384)
        hidden_channels: GAT hidden 차원 (e.g., 256)
        out_channels: 출력 차원 (e.g., 256)
        num_layers: GAT 레이어 수 (default 3)
        heads: attention head 수 (default 4)
        query_conditioned: True면 query concat (기존 동작)
        query_supernode: True면 query_node 추가 edge (기존 동작)
        pairnorm_mode: 'none' | 'pairnorm'
        pairnorm_scale: PairNorm scale factor
        initial_residual_alpha: 0이면 비활성. >0이면 매 layer마다 h_0 주입 비율.
        jumping_knowledge: 'none' | 'concat' | 'max'
        dual_stream: True면 schema/query stream 분리 (query_conditioned 과 배타)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 4,
        query_conditioned: bool = False,
        query_supernode: bool = False,
        pairnorm_mode: str = "none",
        pairnorm_scale: float = 1.0,
        initial_residual_alpha: float = 0.0,
        jumping_knowledge: str = "none",
        dual_stream: bool = False,
        # V-1
        num_layers_mode: str = "fixed",
        num_layers_fallback: int = 3,
        diameter_path: Optional[str] = None,
        diameter_dict: Optional[Dict[str, int]] = None,
        # V-2
        supernode_edge_direction: str = "bidirectional",
        # V-3
        supernode_topk: Optional[int] = None,
        supernode_topk_criterion: str = "raw",
    ):
        super().__init__()
        if dual_stream and query_conditioned:
            raise ValueError("dual_stream 과 query_conditioned 는 배타적입니다.")
        if num_layers_mode not in NUM_LAYERS_MODES:
            raise ValueError(f"num_layers_mode must be in {NUM_LAYERS_MODES}")
        if supernode_edge_direction not in SUPERNODE_EDGE_DIRECTIONS:
            raise ValueError(f"supernode_edge_direction must be in {SUPERNODE_EDGE_DIRECTIONS}")
        if supernode_topk_criterion not in SUPERNODE_TOPK_CRITERIA:
            raise ValueError(f"supernode_topk_criterion must be in {SUPERNODE_TOPK_CRITERIA}")
        if supernode_topk is not None and not query_supernode:
            raise ValueError("supernode_topk 는 query_supernode=True 에서만 의미가 있음")

        # num_layers 는 여기서부터 'upper bound' 역할 — V-1 mode!='fixed' 시 실제 active depth 는
        # resolve_num_layers(db_name) 에서 결정. conv/pairnorm 등 파라미터는 upper bound 까지 allocate.
        self.num_layers = num_layers  # = num_layers_max
        self.num_layers_max = num_layers
        self.num_layers_mode = num_layers_mode
        self.num_layers_fallback = int(num_layers_fallback)
        self.heads = heads
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.query_conditioned = query_conditioned
        self.query_supernode = query_supernode
        self.pairnorm_mode = pairnorm_mode
        self.initial_residual_alpha = float(initial_residual_alpha)
        self.jumping_knowledge = jumping_knowledge
        self.dual_stream = dual_stream
        self.supernode_edge_direction = supernode_edge_direction
        self.supernode_topk = supernode_topk
        self.supernode_topk_criterion = supernode_topk_criterion

        # V-1: diameter dict 로드 (path 우선, 없으면 넘겨받은 dict, 둘 다 없으면 fixed 로만 동작)
        loaded: Optional[Dict[str, int]] = None
        if diameter_path is not None and os.path.exists(diameter_path):
            loaded = torch.load(diameter_path, map_location="cpu")
            if isinstance(loaded, dict) and "diameters" in loaded:
                loaded = loaded["diameters"]
        self.diameter_dict: Dict[str, int] = (
            dict(diameter_dict) if diameter_dict is not None
            else (dict(loaded) if isinstance(loaded, dict) else {})
        )

        # 입력 차원 결정
        # - dual_stream=True: schema stream은 query concat 안 함
        # - query_conditioned=True: 모든 노드에 query concat → 2x
        # - 그 외: 그대로
        effective_in = in_channels * 2 if query_conditioned else in_channels

        node_types = ["table", "column", "fk_node"]
        if query_supernode:
            node_types.append("query_node")

        self.node_types = node_types

        # 1. Input projection — "pre-GAT MLP" 역할
        self.lin_dict = nn.ModuleDict(
            {nt: Linear(effective_in, hidden_channels) for nt in node_types}
        )

        # 2. Heterogeneous GAT layers
        base_edge_types = [
            ("table", "has_column", "column"),
            ("column", "belongs_to", "table"),
            ("column", "is_source_of", "fk_node"),
            ("fk_node", "points_to", "column"),
            ("table", "table_to_table", "table"),
        ]
        # V-2: bidirectional vs directed_from_sn
        supernode_edge_types = []
        if query_supernode:
            for schema_nt in ["table", "column", "fk_node"]:
                supernode_edge_types.append(
                    ("query_node", f"attends_to_{schema_nt}", schema_nt)
                )
                if self.supernode_edge_direction == "bidirectional":
                    supernode_edge_types.append(
                        (schema_nt, f"attended_by_{schema_nt}", "query_node")
                    )
            # directed_from_sn: schema→SN 제거. SN 을 HeteroConv 에서 보존하기 위한 self-loop.
            if self.supernode_edge_direction == "directed_from_sn":
                supernode_edge_types.append(
                    ("query_node", "self_loop", "query_node")
                )
        all_edge_types = base_edge_types + supernode_edge_types

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                et: GATv2Conv(-1, hidden_channels, heads=heads, add_self_loops=False)
                for et in all_edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))

        # 3. PairNorm layers (layer 수만큼)
        if pairnorm_mode == "pairnorm":
            self.pairnorms = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {nt: PairNorm(scale=pairnorm_scale) for nt in node_types}
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.pairnorms = None

        # 4. Initial residual projection: h_0 (hidden_channels) → hidden_channels*heads
        # 각 GAT layer output이 hidden*heads 차원이므로 주입 시 차원 맞춰줘야 함.
        if self.initial_residual_alpha > 0.0:
            self.res_proj = nn.ModuleDict(
                {
                    nt: Linear(hidden_channels, hidden_channels * heads)
                    for nt in node_types
                }
            )
        else:
            self.res_proj = None

        # 5. Jumping Knowledge 융합
        if jumping_knowledge == "concat":
            # hidden*heads * (num_layers+1)  → out_channels  (+1은 h_0 포함)
            jk_in_dim = hidden_channels * heads * num_layers + hidden_channels
            self.jk_lin = nn.ModuleDict(
                {nt: Linear(jk_in_dim, out_channels) for nt in node_types}
            )
        elif jumping_knowledge == "max":
            # h_l 차원을 맞춰놓고 max-pool
            self.jk_lin = nn.ModuleDict(
                {nt: Linear(hidden_channels * heads, out_channels) for nt in node_types}
            )
        else:
            self.jk_lin = None

        # 6. 기본 출력 경로 (JK가 off일 때 사용)
        if jumping_knowledge == "none":
            self.out_lin_dict = nn.ModuleDict(
                {nt: Linear(hidden_channels * heads, out_channels) for nt in node_types}
            )
        else:
            self.out_lin_dict = None

        # 7. Skip connection — dual_stream 일 때는 query-free lin_dict 출력을 쓴다.
        self.skip_dict = nn.ModuleDict(
            {nt: Linear(effective_in, out_channels) for nt in node_types}
        )

        # 8. Dual-stream용 query encoder
        if dual_stream:
            self.query_encoder = nn.Sequential(
                Linear(in_channels, hidden_channels),
                nn.LeakyReLU(0.1),
                Linear(hidden_channels, out_channels),
            )
            # head: concat(h, z_q, h⊙z_q) → out
            self.fusion_head = nn.ModuleDict(
                {
                    nt: nn.Sequential(
                        Linear(out_channels * 3, out_channels),
                        nn.LeakyReLU(0.1),
                        Linear(out_channels, out_channels),
                    )
                    for nt in node_types
                }
            )
        else:
            self.query_encoder = None
            self.fusion_head = None

    # ──────────────────────────────────────────────────────────────
    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        query_emb: torch.Tensor | None = None,
        node_batch_dict: dict | None = None,
        db_name: Optional[str] = None,
        active_num_layers: Optional[int] = None,
    ) -> dict:
        """
        query_emb:
            - 단일 그래프: [d_q] 또는 [1, d_q]
            - 배치 그래프: [B, d_q]
        node_batch_dict:
            - 배치 처리 시 각 node type 별로 노드가 속한 graph index 를 담은 tensor.
            - None 이면 단일 그래프로 간주하고 query 를 모든 노드에 broadcast.
        db_name / active_num_layers: V-1. num_layers_mode != 'fixed' 일 때 실제 사용할 depth.
            active_num_layers 가 명시되면 우선. 아니면 db_name + diameter_dict 로 resolve.
        """
        # V-1: active depth 결정. 최대 self.num_layers_max 로 clamp.
        depth = self.resolve_num_layers(
            db_name=db_name, active_num_layers=active_num_layers
        )

        # V-3: SuperNode top-k filtering (pre-GAT raw).
        if (self.query_supernode and self.supernode_topk is not None
                and self.supernode_topk > 0 and query_emb is not None):
            topk_masks = self._compute_topk_mask(query_emb, x_dict)
            edge_index_dict = self._apply_topk_to_edges(edge_index_dict, topk_masks)

        # V-2: directed_from_sn 시 SN self-loop edge 주입.
        if (self.query_supernode
                and self.supernode_edge_direction == "directed_from_sn"
                and "query_node" in x_dict):
            edge_index_dict = self._inject_sn_self_loop(edge_index_dict, x_dict)

        # --- 1. query concat (query_conditioned) ---
        raw_x_dict = x_dict  # dual_stream skip 용 원본
        if self.query_conditioned and query_emb is not None:
            if query_emb.dim() == 1:
                query_emb = query_emb.unsqueeze(0)
            augmented = {}
            for nt, x in x_dict.items():
                if node_batch_dict is not None and nt in node_batch_dict and query_emb.size(0) > 1:
                    q_per_node = query_emb[node_batch_dict[nt]]
                else:
                    q_per_node = query_emb[:1].expand(x.size(0), -1)
                augmented[nt] = torch.cat([x, q_per_node], dim=-1)
            x_dict = augmented

        # --- 2. Input projection → h_0 ---
        h0_dict = {
            nt: F.leaky_relu(self.lin_dict[nt](x)) for nt, x in x_dict.items()
        }
        out_dict = h0_dict

        # --- 3. GAT layers (+ PairNorm, + Initial Residual) ---
        # V-1: depth 만큼만 순회 (depth ≤ num_layers_max). JK concat 은 num_layers_max 만큼의
        # slot 이 필요하므로 부족한 depth 는 마지막 layer 출력을 패딩.
        layer_outputs = []  # JK 용. 각 원소는 {nt: tensor}
        for i in range(depth):
            conv_out = self.convs[i](out_dict, edge_index_dict)
            # ELU
            conv_out = {nt: F.elu(x) for nt, x in conv_out.items()}
            # HeteroConv는 메시지를 받지 못한 node type을 탈락시킴.
            # JK=concat에서 layer 간 dim을 맞추려면 누락분을 res_proj(h_0)로 채워야 함.
            if self.res_proj is not None:
                for nt in self.node_types:
                    if nt not in conv_out and nt in h0_dict:
                        conv_out[nt] = self.res_proj[nt](h0_dict[nt])
            # Initial residual: (1-α)·GAT + α·h_0_proj
            if self.res_proj is not None:
                mixed = {}
                for nt, x in conv_out.items():
                    h0_proj = self.res_proj[nt](h0_dict[nt])
                    mixed[nt] = (1.0 - self.initial_residual_alpha) * x + self.initial_residual_alpha * h0_proj
                conv_out = mixed
            # PairNorm
            if self.pairnorms is not None:
                conv_out = {
                    nt: self.pairnorms[i][nt](x) for nt, x in conv_out.items()
                }
            out_dict = conv_out
            layer_outputs.append(out_dict)

        # V-1: depth < num_layers_max 일 때, JK concat dim 을 맞추기 위해 마지막 output 반복 패딩.
        # (max / none 은 depth 에 무관)
        if self.jumping_knowledge == "concat" and depth < self.num_layers_max:
            last = layer_outputs[-1] if layer_outputs else {}
            pad = self.num_layers_max - depth
            for _ in range(pad):
                layer_outputs.append(last)

        # --- 4. 출력 경로: JK vs 기본 ---
        if self.jumping_knowledge == "concat":
            final = {}
            for nt in self.node_types:
                parts = [h0_dict[nt]]  # h_0 (hidden)
                for l in range(self.num_layers_max):
                    if nt in layer_outputs[l]:
                        parts.append(layer_outputs[l][nt])
                concat = torch.cat(parts, dim=-1)
                final[nt] = self.jk_lin[nt](concat) + self.skip_dict[nt](x_dict[nt])
        elif self.jumping_knowledge == "max":
            final = {}
            for nt in self.node_types:
                stacked = torch.stack(
                    [layer_outputs[l][nt] for l in range(depth) if nt in layer_outputs[l]],
                    dim=0,
                )
                pooled = stacked.max(dim=0).values
                final[nt] = self.jk_lin[nt](pooled) + self.skip_dict[nt](x_dict[nt])
        else:
            final = {}
            for nt, x in out_dict.items():
                final[nt] = self.out_lin_dict[nt](x) + self.skip_dict[nt](x_dict[nt])

        # --- 5. Dual-stream fusion ---
        if self.dual_stream and query_emb is not None:
            if query_emb.dim() == 1:
                query_emb = query_emb.unsqueeze(0)
            z_q = self.query_encoder(query_emb)  # [B, out] 또는 [1, out]
            fused = {}
            for nt in self.node_types:
                if nt not in final:
                    continue
                h = final[nt]
                if node_batch_dict is not None and nt in node_batch_dict and z_q.size(0) > 1:
                    z_q_per_node = z_q[node_batch_dict[nt]]  # [N_nt, out]
                else:
                    z_q_per_node = z_q[:1].expand(h.size(0), -1)
                concat = torch.cat([h, z_q_per_node, h * z_q_per_node], dim=-1)
                fused[nt] = self.fusion_head[nt](concat)
            final = fused

        return final

    # ──────────────────────────────────────────────────────────────
    # V-Track helper methods
    # ──────────────────────────────────────────────────────────────
    def resolve_num_layers(
        self,
        db_name: Optional[str] = None,
        active_num_layers: Optional[int] = None,
    ) -> int:
        """V-1: runtime depth 결정.

        우선순위: explicit active_num_layers > diameter_dict lookup > fallback/num_layers_max.
        """
        # 1) explicit override
        if active_num_layers is not None:
            return max(1, min(int(active_num_layers), self.num_layers_max))

        # 2) fixed mode
        if self.num_layers_mode == "fixed":
            return self.num_layers_max

        # 3) lookup by db_name
        diam: Optional[int] = None
        if db_name is not None and db_name in self.diameter_dict:
            try:
                diam = int(self.diameter_dict[db_name])
            except (TypeError, ValueError):
                diam = None

        if diam is None:
            return max(1, min(self.num_layers_fallback, self.num_layers_max))

        if self.num_layers_mode == "D_max":
            d = diam
        else:  # D_max_plus1
            d = diam + 1
        return max(1, min(d, self.num_layers_max))

    def _compute_topk_mask(
        self,
        query_emb: torch.Tensor,
        x_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """V-3: pre-GAT raw cosine score 기준 schema 노드 top-k (전 schema 통합)."""
        q = query_emb if query_emb.dim() == 2 else query_emb.unsqueeze(0)
        if q.size(0) > 1:
            q = q.mean(dim=0, keepdim=True)
        q_norm = F.normalize(q, dim=-1)

        schema_types = ["table", "column", "fk_node"]
        scores_list: List[torch.Tensor] = []
        type_tags: List[str] = []
        local_idx: List[int] = []
        for nt in schema_types:
            if nt not in x_dict or x_dict[nt].size(0) == 0:
                continue
            n_norm = F.normalize(x_dict[nt], dim=-1)
            sc = (n_norm @ q_norm.t()).squeeze(-1)  # [N_nt]
            scores_list.append(sc)
            type_tags.extend([nt] * sc.size(0))
            local_idx.extend(range(sc.size(0)))

        masks: Dict[str, torch.Tensor] = {}
        for nt in schema_types:
            if nt in x_dict:
                masks[nt] = torch.zeros(
                    x_dict[nt].size(0), dtype=torch.bool, device=x_dict[nt].device
                )

        if not scores_list:
            return masks

        combined = torch.cat(scores_list, dim=0)
        k = min(int(self.supernode_topk), combined.size(0))
        topk = torch.topk(combined, k=k).indices.tolist()
        for i in topk:
            nt = type_tags[i]
            masks[nt][local_idx[i]] = True
        return masks

    def _apply_topk_to_edges(
        self,
        edge_index_dict: dict,
        masks: Dict[str, torch.Tensor],
    ) -> dict:
        filtered = dict(edge_index_dict)
        for nt in ["table", "column", "fk_node"]:
            mask = masks.get(nt)
            if mask is None:
                continue
            fwd_key = ("query_node", f"attends_to_{nt}", nt)
            if fwd_key in filtered:
                ei = filtered[fwd_key]
                if ei.numel() > 0:
                    keep = mask[ei[1]]
                    filtered[fwd_key] = ei[:, keep]
            rev_key = (nt, f"attended_by_{nt}", "query_node")
            if rev_key in filtered:
                ei = filtered[rev_key]
                if ei.numel() > 0:
                    keep = mask[ei[0]]
                    filtered[rev_key] = ei[:, keep]
        return filtered

    def _inject_sn_self_loop(
        self,
        edge_index_dict: dict,
        x_dict: Dict[str, torch.Tensor],
    ) -> dict:
        num_q = x_dict["query_node"].size(0)
        if num_q == 0:
            return edge_index_dict
        dev = x_dict["query_node"].device
        idx = torch.arange(num_q, device=dev, dtype=torch.long)
        new = dict(edge_index_dict)
        new[("query_node", "self_loop", "query_node")] = torch.stack([idx, idx], dim=0)
        return new
