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

모든 옵션은 default에서 off이므로 기존 실험과 호환.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear


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
    ):
        super().__init__()
        if dual_stream and query_conditioned:
            raise ValueError("dual_stream 과 query_conditioned 는 배타적입니다.")

        self.num_layers = num_layers
        self.heads = heads
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.query_conditioned = query_conditioned
        self.query_supernode = query_supernode
        self.pairnorm_mode = pairnorm_mode
        self.initial_residual_alpha = float(initial_residual_alpha)
        self.jumping_knowledge = jumping_knowledge
        self.dual_stream = dual_stream

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
        supernode_edge_types = []
        if query_supernode:
            for schema_nt in ["table", "column", "fk_node"]:
                supernode_edge_types.append(
                    ("query_node", f"attends_to_{schema_nt}", schema_nt)
                )
                supernode_edge_types.append(
                    (schema_nt, f"attended_by_{schema_nt}", "query_node")
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
    ) -> dict:
        """
        query_emb:
            - 단일 그래프: [d_q] 또는 [1, d_q]
            - 배치 그래프: [B, d_q]
        node_batch_dict:
            - 배치 처리 시 각 node type 별로 노드가 속한 graph index 를 담은 tensor.
            - None 이면 단일 그래프로 간주하고 query 를 모든 노드에 broadcast.
        """
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
        layer_outputs = []  # JK 용. 각 원소는 {nt: tensor}
        for i in range(self.num_layers):
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

        # --- 4. 출력 경로: JK vs 기본 ---
        if self.jumping_knowledge == "concat":
            final = {}
            for nt in self.node_types:
                parts = [h0_dict[nt]]  # h_0 (hidden)
                for l in range(self.num_layers):
                    if nt in layer_outputs[l]:
                        parts.append(layer_outputs[l][nt])
                concat = torch.cat(parts, dim=-1)
                final[nt] = self.jk_lin[nt](concat) + self.skip_dict[nt](x_dict[nt])
        elif self.jumping_knowledge == "max":
            final = {}
            for nt in self.node_types:
                stacked = torch.stack(
                    [layer_outputs[l][nt] for l in range(self.num_layers) if nt in layer_outputs[l]],
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
