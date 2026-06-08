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
from torch_geometric.nn import HeteroConv, GATv2Conv, GINConv, Linear


SUPERNODE_EDGE_DIRECTIONS = {"bidirectional", "directed_from_sn"}
SUPERNODE_TOPK_CRITERIA = {"raw", "cosine", "ce"}
SUPERNODE_THRESHOLD_MODES = {"top_k", "percentile", "abs_tau"}
SUPERNODE_SCORE_NORMS = {"minmax", "none"}
NUM_LAYERS_MODES = {"fixed", "D_max", "D_max_plus1"}
HETEROCONV_AGGR_TYPES = {"mean", "sum", "max", "min", "mul"}
# Mitigation v3 (DECISIONS 2026-05-08 §1(c)) — GIN-style aggregation 추가.
# 'gin' 은 HeteroConv aggr 가 아닌 inner conv 자체를 GINConv 로 교체. HeteroConv aggr 는 'mean' fix.
AGGREGATION_TYPES = HETEROCONV_AGGR_TYPES | {"gin"}


# ──────────────────────────────────────────────────────────────────────
# Mitigation v2 — GATv2Conv subclass variants
# (DECISIONS 2026-05-07 §1(C)/(D), 학위 본 심사 전 #1+#3+#2 mitigation 후보 학습용)
# ──────────────────────────────────────────────────────────────────────

class DropMessageGATv2Conv(GATv2Conv):
    """Mitigation v2 #1 PRIMARY — message-level dropout.

    GATv2Conv 의 `message(x_j, alpha)` 에 dropout 을 씌워, 학습 시 random subset 의 edge 가
    propagate 단계에서 zero-out 된다. attention 가중치 (alpha) 자체는 그대로 두고
    attended-to neighbor 의 feature contribution 만 분산 → mech(ii) edge softmax over-concentration
    완화 시도. backward compat: `drop_message_p=0.0` 시 super class 와 완전 동일.
    """

    def __init__(self, *args, drop_message_p: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_message_p = float(drop_message_p)

    def message(self, x_j, alpha):  # type: ignore[override]
        msg = x_j * alpha.unsqueeze(-1)
        if self.training and self.drop_message_p > 0.0:
            msg = F.dropout(msg, p=self.drop_message_p, training=self.training)
        return msg


class LayerNormGATv2Conv(GATv2Conv):
    """Mitigation v2 #3 SECONDARY — LayerNorm before edge softmax.

    GATv2Conv 의 `edge_update` 에서 raw alpha (= (LeakyReLU(x_i+x_j) · att).sum) 를 산출 직후,
    softmax 직전에 LayerNorm 을 적용. softmax 의 sharp peaking 효과를 완화 → mech(ii) edge
    softmax over-concentration mitigation 시도. LayerNorm 은 heads dim 에 적용 (alpha 의 shape
    = [num_edges, heads]). backward compat: 외부 동일 인터페이스.

    근거: PyG GATv2Conv.edge_update source — `alpha = (x * self.att).sum(dim=-1)` 후
    `softmax(alpha, ...)` 직전 위치에 LayerNorm 삽입.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # alpha shape: [num_edges, heads]. LayerNorm(heads) — 마지막 dim 에 정규화.
        self.alpha_layernorm = nn.LayerNorm(self.heads)

    def edge_update(self, x_j, x_i, edge_attr, index, ptr, dim_size):  # type: ignore[override]
        from torch_geometric.utils import softmax  # local import to mirror PyG source layout
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        # ← 신규: softmax 직전 LayerNorm (heads 축 정규화)
        alpha = self.alpha_layernorm(alpha)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha


class LNGINGATv2Conv(LayerNormGATv2Conv):
    """Mitigation V4-A — LN+GIN Combo (Pre-softmax LayerNorm + GIN-style post-aggregation MLP).

    Architectural intervention. boundary:
      - attention path: LayerNormGATv2Conv 와 동일 (raw α 산출 후 LN → softmax). row-stochasticity 유지.
      - aggregation post-process: alpha-weighted aggregation 결과 (heads × out_channels) 를 2-layer MLP
        로 transform → GIN 의 `MLP(sum)` 와 동일 구조.

    근거: planning/oversmoothing_solution_methodology_2026-05-11_apa.md §C-1 "LN + GIN Combo (최단 경로 검증)":

        > LayerNorm pre-softmax(시도 5) + GIN sum+MLP(시도 7)를 동시 적용하면 두 partial mitigation 의 합이
        > 나타나는지 검증할 수 있다. 변경 코드량: GATv2Conv 의 pre-softmax 에 LN 추가 + aggregation 을 sum+MLP 로 교체.

    이 클래스는 attention 을 유지하면서 (LN 으로 mitigate) aggregation 의 magnitude/dimensionality
    transform 을 MLP 로 학습 가능하게 만들어, v2 #3 LN 단독 (R=0.6011) + v3 #1 GIN 단독 (R=0.5954) 의
    combo 효과 (보고서 §4.1) 가 발현되는지 정량 검증.

    Note: PyG GATv2Conv 의 forward 는 `concat=True` (default) 시 [N, heads*out_channels] 반환.
    concat=False 시 [N, out_channels] (heads 평균). 본 class 는 concat=True 가정 (SchemaHeteroGATv2 default).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # out_channels × heads 차원의 MLP — GIN propagation 의 MLP 와 같은 역할.
        out_dim = self.heads * self.out_channels if self.concat else self.out_channels
        self.gin_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):  # type: ignore[override]
        # super().forward → tensor [N, heads*out_channels] (concat=True) or [N, out_channels].
        out = super().forward(x, edge_index, edge_attr=edge_attr,
                              return_attention_weights=return_attention_weights)
        if isinstance(out, tuple):
            tensor, *rest = out
            tensor = self.gin_mlp(tensor)
            return (tensor, *rest)
        return self.gin_mlp(out)


class SoftplusGATv2Conv(GATv2Conv):
    """Mitigation V4-B — Softplus + (optional) Symmetric Normalization (AERO-GNN style).

    Architectural intervention. row-stochasticity 자체를 파괴 — Wu et al. 2023 의 over-smoothing 이론적
    증명의 핵심 가정 위반.

    수정:
      - softmax 제거 → softplus(α) = log(1 + exp(α)) — non-negative 단 row-sum != 1.
      - (옵션) edge-symmetric normalization: α_ij / sqrt(d_i · d_j) — degree 기반.

    근거: planning/oversmoothing_solution_methodology_2026-05-11_apa.md §A-1 + §C-2 AERO-GNN style.

      > Wu et al. (2023) 이 증명한 over-smoothing 의 이론적 기제:
      >  1. Softmax normalization 이 aggregation operator P 를 row-stochastic matrix 로 만듦
      >  2. 이런 matrix 들의 무한 곱은 ergodic 하게 rank-1 matrix 로 수렴 → 모든 노드 표현 붕괴
      >  3. Joint Spectral Radius (JSR) < 1 → embedding 차이 $\|h_i - h_j\|$ 가 layer 따라 지수적 감소
      >
      > **Softplus + Symmetric Normalization (AERO-GNN 방식)** 은 softmax 를 softplus activation +
      > symmetric normalization 으로 교체. Softmax 의 row-stochastic 특성이 사라지므로 S(T(k)) 수렴 조건이
      > 깨진다 (Theorem 3, AERO-GNN Lee et al. 2023).

    구현 노트:
      - softplus 는 비음수 (softmax 와 동일) 단 row-sum 가 normalize 안 됨
      - symmetric_norm=True 시 message 의 magnitude 가 degree 로 제어됨 (GCN-style)
      - symmetric_norm=False 시 raw softplus (가장 row-stochasticity 파괴 강함)
    """

    def __init__(self, *args, symmetric_norm: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetric_norm = bool(symmetric_norm)

    def edge_update(self, x_j, x_i, edge_attr, index, ptr, dim_size):  # type: ignore[override]
        # GATv2 의 attention compute pattern 그대로 (LeakyReLU + att projection).
        x = x_i + x_j
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)  # [E, heads] raw attention score

        # ← V4-B 핵심: softmax 대신 softplus. row-stochastic 파괴.
        alpha = F.softplus(alpha)

        # (옵션) Symmetric normalization — degree 기반 GCN-style.
        # dst node degree (= row-sum count) 로 normalize → magnitude 제어 단 sum=1 보장 X.
        if self.symmetric_norm and dim_size is not None and dim_size > 0:
            deg = torch.zeros(dim_size, device=alpha.device, dtype=alpha.dtype)
            ones = torch.ones(index.size(0), device=alpha.device, dtype=alpha.dtype)
            deg.scatter_add_(0, index, ones)
            deg_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)
            alpha = alpha * deg_inv_sqrt[index].unsqueeze(-1)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha


# ──────────────────────────────────────────────────────────────────────
# Mitigation V5 — Tier 1+2 4-Direction (DECISIONS 2026-05-12)
# 학술 Agent 의 oversmoothing_v5_plan.md 채택:
#  V5-A 'gate'      : Mustafa & Burkholz 2024 — attention parameter `a` 를 `a_s` (self) + `a_t` (neighbor)
#                     로 분리 → conservation law 수정. task-irrelevant neighbor aggregation 의 switch-off
#                     가능. mech(ii-b) "softmax × weighted-mean propagation combo" 의 task-irrelevant
#                     aggregation 차원 직접 표적.
#  V5-B 'gcnii'     : Chen 2020 + Peng 2024 — Identity Mapping. 매 layer 의 output 에
#                     `(1-β_l) · I + β_l · W^{(l)}` 적용, β_l = log(λ/l + 1). gradient flow upper bound
#                     를 안정화 → Paradox 2 (skip dependency ρ_skip ≈ 3) 의 trainability 해석 검증.
#                     Initial Residual (α) 는 기존 SchemaHeteroGATv2.initial_residual_alpha 와 결합.
#  V5-C 'aero_full' : Lee 2023 — V4-B (Softplus + Symmetric Norm) + Node-Adaptive Hop Attention.
#                     SchemaHeteroGATv2 의 `aero_hop_attention=True` 와 함께 사용. AERO Theorem 3
#                     SR2OS guarantee 의 본 도메인 transfer 가 Hop Attention 동반 시 회복되는지 검증
#                     (V4-B H10.1c 직접 표적).
# ──────────────────────────────────────────────────────────────────────


class GATEGATv2Conv(GATv2Conv):
    """V5-A — GATE (Mustafa & Burkholz NeurIPS 2024) — Conservation Law 수정.

    GATv2 의 단일 attention parameter `a` (shape [1, heads, out_channels]) 를 self-attention
    coefficient `a_s` 와 neighbor-attention coefficient `a_t` 로 분리. 이 분리 덕에 두 parameter
    가 별도 budget 으로 학습되며, 작은 `||a_s||, ||a_t||` 만으로도 task-irrelevant neighbor
    aggregation 을 끌 수 있는 구조 (Mustafa & Burkholz NeurIPS 2024 §3.2 Theorem 1).

    원본 GATv2 attention:
        e_ij = a^T LeakyReLU(W [h_i || h_j])  (이미 PyG 내부에서 split 되어 lin_l/lin_r 적용)
        => x_i = lin_l(h_i), x_j = lin_r(h_j), e_ij = a^T LeakyReLU(x_i + x_j)
    GATE 수정 (paper §3.2 Eq. 4):
        e_ij = a_s^T LeakyReLU(W h_i) + a_t^T LeakyReLU(W h_j)
        ⇒ self / neighbor 활성 후 dot product 가 분리됨. Conservation law 의 directional decoupling.

    Conservation law (paper §3.1 Theorem 1):
        ‖a‖² conservation 으로 small task-irrelevant neighbor 의 switch-off 가 큰 norm 변동 필요.
        a → (a_s, a_t) 분리 후 두 parameter 사이의 budget 교환만으로 a_t → 0 (neighbor mute) 가능.
        결과: OGB-arxiv 에서 GATv2 대비 SOTA (paper Table 1, +0.5%~+1.0%). Heterophilic dataset
        (Texas, Wisconsin, Cornell) 에서 GAT 대비 큰 margin 으로 상회 (paper Table 3).

    Implementation note (PyG):
      - GATE 원 paper 의 W (input linear projection) 는 self/neighbor 공유 (W_s = W_t = W).
        본 구현은 parent GATv2Conv 의 lin_l/lin_r 을 그대로 사용 — W 공유 = paper form 일치.
        (User-spec sketch 의 lin_self_attn 분리 form 은 W_s ≠ W_t 의 일반화; 본 구현은 paper minimal form.)
      - row-stochasticity 유지 (softmax 그대로) — Wu et al. 2023 의 row-stochastic 가정은 위반하지 않음.
        본 시도의 mitigation target 은 neighbor aggregation 의 task-irrelevance switch-off 이지,
        row-stochasticity 자체 파괴가 아님 (V4-B / V5-C 와 다른 axis).
      - 신규 parameter `att_self` = Parameter([1, heads, out_channels]), Xavier init (gain=1.414).

    Reference:
        Mustafa, N., & Burkholz, R. (2024). GATE: How to Keep Out Intrusive Neighbors.
        NeurIPS 2024. arXiv:2406.00418.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Self-attention parameter — neighbor-attention (self.att) 와 분리.
        self.att_self = nn.Parameter(torch.empty(1, self.heads, self.out_channels))
        nn.init.xavier_uniform_(self.att_self.data, gain=1.414)

    def edge_update(self, x_j, x_i, edge_attr, index, ptr, dim_size):  # type: ignore[override]
        from torch_geometric.utils import softmax  # local import (PyG source pattern)
        # x_i, x_j 는 이미 lin_l/lin_r 적용 결과: [E, heads, out_channels]
        x_i_act = F.leaky_relu(x_i, self.negative_slope)
        x_j_act = F.leaky_relu(x_j, self.negative_slope)

        # Edge attr 는 neighbor side 에 합산 (관행, PyG GATv2 와 동일 위치).
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x_j_act = x_j_act + edge_attr

        # GATE: e_ij = a_s^T LRELU(x_i) + a_t^T LRELU(x_j)
        alpha = (x_i_act * self.att_self).sum(dim=-1) + (x_j_act * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha


class GCNIIGATv2Conv(GATv2Conv):
    """V5-B — GCNII-style Identity Mapping (Chen et al. ICML 2020 + Peng et al. CIKM 2024).

    GCN-II 핵심 식 (Chen 2020 §3.2 Eq. 6):
      h^{(l+1)} = σ( ((1-α) P h^{(l)} + α h^{(0)}) · ((1-β_l) I + β_l W^{(l)}) )
        - α ∈ [0.1, 0.3]: Initial Residual 비율 (outer SchemaHeteroGATv2.initial_residual_alpha 처리)
        - β_l = log(λ/l + 1): identity mapping 강도. layer index l 따라 감소 (λ ∈ [0.5, 1.0])
        - W^{(l)}: layer-별 학습 weight; **identity 에 가까운 init 이 핵심** (eye_init)
    본 class 는 Identity Mapping (β_l 부분) 만 담당. Initial Residual 은 outer SchemaHeteroGATv2 의
    initial_residual_alpha 가 이미 처리 → 두 component 모두 활성화하려면 model config 에서
    `initial_residual_alpha=0.2` (Chen 2020 default) + `gat_layer_type='gcnii'` 동시 지정.

    Peng 2024 의 trainability 해석 (CIKM 2024):
      Over-smoothing 이 deep GNN 성능 저하의 **dominant cause 가 아님** — deep MLP trainability 가
      실제 병목 (paper §4 gradient flow upper bound 분석). GCNII 의 (α, β) 두 parameter 가 smoothing
      term 과 training term 을 동시에 적절히 제어 → gradient flow 안정화 → trainability 개선.
      본 framework 의 Paradox 2 (skip dependency ρ_skip ≈ 3) 의 trainability 해석을 직접 검증.

    구현 노트:
      - layer index l 은 SchemaHeteroGATv2.__init__ 시 enumerate(self.convs) 로 주입 (1-indexed,
        gcnii_layer_idx 로 forward).
      - W^{(l)} = `gcnii_w = Linear(out_dim, out_dim, bias=False)`, **Identity init (nn.init.eye_)**
        — Chen 2020 §3.3 에 명시된 핵심 초기화. 학습 시작점이 identity perturbation.
      - GATv2 attention path 자체는 표준 (row-stochasticity 유지). Wu et al. 2023 의 row-stochastic
        가정은 위반하지 않음. 본 시도의 mitigation target 은 trainability (gradient flow upper bound).
      - L = 2/4/6 sweep — depth 별 효과 측정 (paper Table 2 의 L=2/4/8/16/32/64 패턴).

    Reference:
        Chen, M., Wei, Z., Huang, Z., Ding, B., & Li, Y. (2020). Simple and Deep Graph Convolutional
        Networks (GCNII). ICML 2020. arXiv:2007.02133.
        Peng, J., Lei, R., & Wei, Z. (2024). Beyond Over-smoothing: Uncovering the Trainability
        Challenges in Deep Graph Neural Networks. CIKM 2024. DOI:10.1145/3627673.3679776.
    """

    def __init__(self, *args, gcnii_beta_lambda: float = 0.5, gcnii_layer_idx: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        out_dim = self.heads * self.out_channels if self.concat else self.out_channels
        self._gcnii_out_dim = out_dim
        self.gcnii_w = nn.Linear(out_dim, out_dim, bias=False)
        nn.init.eye_(self.gcnii_w.weight)
        self.gcnii_beta_lambda = float(gcnii_beta_lambda)
        self.gcnii_layer_idx = int(gcnii_layer_idx)  # 1-indexed

    def _beta(self) -> float:
        import math
        return math.log(self.gcnii_beta_lambda / max(1, self.gcnii_layer_idx) + 1.0)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):  # type: ignore[override]
        out = super().forward(x, edge_index, edge_attr=edge_attr,
                              return_attention_weights=return_attention_weights)
        beta = self._beta()
        if isinstance(out, tuple):
            tensor, *rest = out
            tensor = (1.0 - beta) * tensor + beta * self.gcnii_w(tensor)
            return (tensor, *rest)
        return (1.0 - beta) * out + beta * self.gcnii_w(out)


class FullAEROGATv2Conv(SoftplusGATv2Conv):
    """V5-C — Full AERO-GNN (Lee et al. ICML 2023, arXiv:2306.02376) Theorem 3 SR2OS guarantee.

    V4-B (Softplus + Symmetric Norm) 가 AERO 의 Theorem 3 의 일부 component (row-stochasticity 파괴)
    만 실현. V5-C = V4-B + **(b) Cumulative Attention** + **(c) Node-Adaptive Hop Attention** 의
    완전체. 세 component 가 모두 동시 동작 시 AERO 원 paper Theorem 3 SR2OS guarantee 가 발동.

    SR2OS (Stochastic Rank-One Stability) — 누적 attention matrix `T(k)` 의 smoothness score
    `S(T(k))` 가 layer depth 따라 0 으로 수렴하는 보장이 row-stochasticity 파괴 + cumulative
    residual + hop attention 의 combo 에서만 깨진다 (Theorem 3, Lee 2023 §4.2).

    Mathematical form (Lee 2023 §3.2):
        Layer-l edge attention with cumulative residual:
            e_ij^(l) = a^T LRELU(x_i + x_j + edge_attr)
            α_ij^(l) = α_ij^(l-1) + softplus(e_ij^(l))      ← Cumulative residual (V5-C 신규)
        Symmetric normalization (V4-B):
            ~α_ij^(l) = α_ij^(l) / sqrt(d_i · d_j)
        Node-Adaptive Hop attention (outer SchemaHeteroGATv2.forward 에서 처리):
            h_v^out = Σ_l ω_v^(l) · h_v^(l)

    Three components:
      (a) Softplus + Symmetric Norm    — parent `SoftplusGATv2Conv` 가 처리 ✅
      (b) **Cumulative Attention**     — 본 class 가 신규 처리 (buffer `cum_alpha_buffer`)
      (c) Node-Adaptive Hop Attention  — outer `SchemaHeteroGATv2.forward` 의 `aero_hop_attention=True`
                                          path 가 처리 (per-node simplex weight + h_0 포함 L+1 hops)

    Cumulative buffer protocol:
      - `register_buffer('cum_alpha_buffer', None, persistent=False)` — non-persistent (학습 후 ckpt
        에 저장 X). 매 query forward 시작 시 outer 가 `reset_cumulative()` 호출.
      - `cumulative_attention=False` 시 V4-B AEROGATConv 와 동치 (parent 만 사용).
      - HeteroConv 안에서 동일 conv 인스턴스가 same query 의 여러 layer 에 걸쳐 호출되지 않음
        (각 layer 가 별도 inst). 따라서 cumulative 는 conv 인스턴스 내부의 sequential forward 호출
        에서만 유의 — 본 framework 는 layer 별 별개 conv 라 cumulative 효과 미발현.
        → **실제 cumulative 누적은 outer SchemaHeteroGATv2.forward 가 conv 출력 alpha 를 layer 간
        pass** 하는 방식으로 구현. 본 class 는 단일 layer 의 raw softplus(e^(l)) 만 반환 +
        outer 가 누적.

    Reference:
        Lee, S. Y., Bu, F., Yoo, J., & Shin, K. (2023). Towards Deep Attention in Graph Neural
        Networks: Problems and Remedies (AERO-GNN). arXiv:2306.02376.
    """

    def __init__(self, *args, cumulative_attention: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.cumulative_attention = bool(cumulative_attention)
        # Layer-local accumulator — outer model 이 layer 간 alpha 누적을 처리할 때 buffer 로 사용.
        # 본 class 의 edge_update 는 raw softplus(e^(l)) 만 반환하고, outer 가 cum 처리.
        # cum_alpha_buffer 는 inspection / debug 용 — 실제 cum 은 outer 가 attribute 로 주입 가능.
        self.register_buffer("cum_alpha_buffer", torch.zeros(0), persistent=False)

    def reset_cumulative(self) -> None:
        """Outer model 이 매 query forward 시작 시 호출. cum_alpha_buffer 를 빈 tensor 로 reset."""
        self.cum_alpha_buffer = torch.zeros(0, device=self.cum_alpha_buffer.device)


# ──────────────────────────────────────────────────────────────────────
# Mitigation V6-W1 — Drop-in 3종 격리 단독 + 조합 ablation
# (DECISIONS 2026-06-01 §V6-W1 격하 retract + 활성 launch)
#
# V6-W0 retrospective (analyzer §6.2) 의 predicted prior 위 actual measurement 의 confirm/refute
# evidence base 강화 motivation. 학위 본 심사 §III chapter §III.3 (Mitigation null) / §III.4
# (mech(ii-b) DOMINANT deep dive) / §III.9 (V5 deep extension) 의 single-cell evidence 추가 강화.
#
# V6-W1 ablation cells (격리 단독 + 조합):
#   P1a `PairNormGATv2Conv`   — Zhao & Akoglu (ICLR 2020). conv-output 뒤 PairNorm scale-individually
#                               normalize. scale 스윕 ∈ {0.5, 1.0, 2.0}.
#   P1b `GCNIIIRGATv2Conv`    — Chen et al. (ICML 2020) + Peng et al. (CIKM 2024). V5-B
#                               `GCNIIGATv2Conv` 와 본질 동일 단 격리 단독 측정 cell 구분 위
#                               별개 class name 유지. α (initial residual) ∈ {0.05, 0.1, 0.2} 는
#                               outer SchemaHeteroGATv2.initial_residual_alpha 가 처리,
#                               β = log(λ/l+1) (identity mapping) 는 inner class 가 처리.
#   P1c `JKGATv2Conv`         — Xu et al. (ICML 2018). marker subclass — JK aggregation 은
#                               multi-layer outer process 이므로 conv-level 에서 standard GATv2Conv
#                               와 indistinguishable. 본 class 는 ablation 격리 cell 의 axis
#                               classification + outer JK 활성화 신호. `jk_mode` ∈ {concat, max}.
#   P1d `DropInComboGATv2Conv` — P1a + P1b + P1c 한 forward 안 sequential application.
#                                 GCNIIIRGATv2Conv 의 GCNII forward 위 PairNorm scale 적용 +
#                                 JK marker 강제 — outer JK aggregation 동시 활성화 가정.
#
# V6-W0 retrospective 정합 layer-wise output capture (`_captured_output`) + attention weights
# (`_captured_alpha`) hook — analyzer 의 Dirichlet/MAD/attention entropy 측정 위. 매 forward 후
# detached buffer 위 직전 forward call 의 hidden state + softmax 직후 alpha 보관.
# ──────────────────────────────────────────────────────────────────────


class PairNormGATv2Conv(GATv2Conv):
    """V6-W1 P1a — Drop-in PairNorm wrapper on standard GATv2Conv output.

    GATv2 attention path 표준 (softmax 유지) + conv output 직후 PairNorm scale-individually
    normalize. node feature 의 pairwise distance 합을 상수로 유지 → mech(ii-b) softmax ×
    weighted-mean propagation 위 row-stochastic collapse 의 hidden-state 측 mitigation 시도.

    근거 (Zhao & Akoglu, ICLR 2020):
      h_v^{out} = s · (h_v - mean) / ||h_v - mean||   (PN-SI mode, scale-individually)
      → pairwise distance Σ_{u,v} ||h_u - h_v||^2 = constant 유지 → over-smoothing 의 norm collapse 차단.

    V1~V5 chain 와의 격리:
      - 본 class 는 conv-level subclass — outer SchemaHeteroGATv2.pairnorm_mode='pairnorm' 의
        layer-level wrapping 과 동일 effect. V6-W1 격리 단독 ablation cell 의 axis classification 위
        별개 class.
      - V2_LayerNorm (-0.0086 mit best) 와 다른 axis: PairNorm 은 hidden state 의 pairwise
        normalization, LN 은 pre-softmax α 의 heads-axis normalization.

    Reference:
        Zhao, L., & Akoglu, L. (2020). PairNorm: Tackling Oversmoothing in GNNs. ICLR 2020.
        arXiv:1909.12223.
    """

    def __init__(self, *args, pairnorm_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if pairnorm_scale <= 0.0:
            raise ValueError(f"pairnorm_scale must be > 0, got {pairnorm_scale}")
        # PairNorm class 정의는 본 module 의 line 510 — instantiation time 에 resolve.
        self.pairnorm = PairNorm(scale=float(pairnorm_scale))
        self.pairnorm_scale = float(pairnorm_scale)
        # V6 capture buffers — analyzer Dirichlet/MAD/attention entropy 측정 hook.
        # detach() 후 보관, 학습 ckpt 비저장 (non-Parameter).
        self._captured_output: Optional[torch.Tensor] = None
        self._captured_alpha: Optional[torch.Tensor] = None

    def edge_update(self, x_j, x_i, edge_attr, index, ptr, dim_size):  # type: ignore[override]
        # Standard GATv2 edge_update — capture softmax-normalized α 위 hook 만 추가.
        alpha = super().edge_update(x_j, x_i, edge_attr, index, ptr, dim_size)
        self._captured_alpha = alpha.detach()
        return alpha

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):  # type: ignore[override]
        out = super().forward(x, edge_index, edge_attr=edge_attr,
                              return_attention_weights=return_attention_weights)
        if isinstance(out, tuple):
            tensor, *rest = out
            tensor = self.pairnorm(tensor)
            self._captured_output = tensor.detach()
            return (tensor, *rest)
        tensor = self.pairnorm(out)
        self._captured_output = tensor.detach()
        return tensor


class GCNIIIRGATv2Conv(GCNIIGATv2Conv):
    """V6-W1 P1b — GCNII Initial Residual + Identity Mapping 격리 단독 측정 cell.

    V5-B `GCNIIGATv2Conv` 와 본질 동일 — class 본체는 inner Identity Mapping (β_l = log(λ/l+1))
    만 처리 + V6 capture hook 추가. Initial Residual (α) 는 outer SchemaHeteroGATv2 의
    `initial_residual_alpha` parameter 가 처리.

    V6-W1 ablation cell 구분 위 별개 class name 유지 (analyzer reporting 위 axis classification).
    V5-B GCNIIGATv2Conv 의 inner forward + super().forward 위 capture hook 만 추가.

    Hyperparameter 명세 (V6-W1 P1b):
      - α (initial_residual_alpha) ∈ {0.05, 0.1, 0.2} — outer SchemaHeteroGATv2 handle.
      - β (identity mapping 강도) optional — gcnii_beta_lambda hyperparameter (default 0.5,
        β_l = log(λ/l + 1) → layer index l 따라 감소). β=0 시 inner identity mapping 비활성.

    Reference:
        Chen, M., Wei, Z., Huang, Z., Ding, B., & Li, Y. (2020). Simple and Deep Graph
        Convolutional Networks (GCNII). ICML 2020. arXiv:2007.02133.
        Peng, J., Lei, R., & Wei, Z. (2024). Beyond Over-smoothing: Uncovering the Trainability
        Challenges in Deep Graph Neural Networks. CIKM 2024.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # V6 capture buffers.
        self._captured_output: Optional[torch.Tensor] = None
        self._captured_alpha: Optional[torch.Tensor] = None

    def edge_update(self, x_j, x_i, edge_attr, index, ptr, dim_size):  # type: ignore[override]
        alpha = super().edge_update(x_j, x_i, edge_attr, index, ptr, dim_size)
        self._captured_alpha = alpha.detach()
        return alpha

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):  # type: ignore[override]
        # V5-B GCNIIGATv2Conv.forward 가 (1-β)·out + β·W·out 처리 후 반환.
        out = super().forward(x, edge_index, edge_attr=edge_attr,
                              return_attention_weights=return_attention_weights)
        if isinstance(out, tuple):
            tensor, *rest = out
            self._captured_output = tensor.detach()
            return (tensor, *rest)
        self._captured_output = out.detach()
        return out


class JKGATv2Conv(GATv2Conv):
    """V6-W1 P1c — Jumping Knowledge marker subclass on standard GATv2Conv.

    JK aggregation (Xu et al. ICML 2018) 은 **multi-layer outer process** — 각 layer 의 output
    h^{(l)} 을 concat 또는 max-pool 하여 final representation 산출. conv-level 단일 layer
    에서는 standard GATv2Conv 와 indistinguishable.

    본 class 는:
      - V6-W1 ablation cell 의 axis classification 위 marker subclass.
      - `jk_mode` hyperparameter ∈ {concat, max} — outer SchemaHeteroGATv2 가 lookup 후
        `jumping_knowledge` parameter 활성화 신호로 사용.
      - V6 capture hook 추가 (analyzer Dirichlet/MAD/attention entropy 위).

    근거 (Xu et al., ICML 2018):
      h_v^{final} = JK(h_v^{(0)}, h_v^{(1)}, ..., h_v^{(L)})
        concat: h_v^{final} = [h_v^{(0)} || h_v^{(1)} || ... || h_v^{(L)}]
        max:    h_v^{final} = max_l h_v^{(l)}      (element-wise)

    V1~V5 chain 와의 격리:
      - V5-C aero_full 의 Hop Attention 과 다른 axis: AERO Hop Attention 은 node-adaptive
        learned weight Σ_l γ_v^{(l)} · h_v^{(l)}, JK 는 fixed concat / max-pool.
      - V1~V5 chain 미수행 격리 cell — JK 단독 ablation 의 column differentiation 회복
        (V6-W0 §5 L_out 위 MAD intra 0.65-0.71 회복 정합) 확인 위.

    Reference:
        Xu, K., Li, C., Tian, Y., Sonobe, T., Kawarabayashi, K., & Jegelka, S. (2018).
        Representation Learning on Graphs with Jumping Knowledge Networks. ICML 2018.
        arXiv:1806.03536.
    """

    JK_MODES = {"concat", "max"}

    def __init__(self, *args, jk_mode: str = "concat", **kwargs):
        super().__init__(*args, **kwargs)
        if jk_mode not in self.JK_MODES:
            raise ValueError(f"jk_mode must be in {self.JK_MODES}, got {jk_mode}")
        self.jk_mode = jk_mode
        # V6 capture buffers.
        self._captured_output: Optional[torch.Tensor] = None
        self._captured_alpha: Optional[torch.Tensor] = None

    def edge_update(self, x_j, x_i, edge_attr, index, ptr, dim_size):  # type: ignore[override]
        alpha = super().edge_update(x_j, x_i, edge_attr, index, ptr, dim_size)
        self._captured_alpha = alpha.detach()
        return alpha

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):  # type: ignore[override]
        out = super().forward(x, edge_index, edge_attr=edge_attr,
                              return_attention_weights=return_attention_weights)
        if isinstance(out, tuple):
            tensor, *rest = out
            self._captured_output = tensor.detach()
            return (tensor, *rest)
        self._captured_output = out.detach()
        return out


class DropInComboGATv2Conv(GCNIIIRGATv2Conv):
    """V6-W1 P1d — PairNorm + GCNII IR + JK marker 한 forward 안 sequential combo.

    Inheritance: GCNIIIRGATv2Conv (P1b) ← GCNIIGATv2Conv (V5-B). super().forward 가 GCNII inner
    identity mapping 처리 → 본 class 가 conv output 위 PairNorm 추가 적용 + JK marker 보관.

    Sequential application order (한 forward call 안):
      1. Standard GATv2 attention (softmax 유지) — GATv2Conv.forward
      2. GCNII Identity Mapping: tensor = (1-β_l)·tensor + β_l·W·tensor — GCNIIGATv2Conv.forward
      3. PairNorm scale-individually: tensor = s · (tensor - mean) / ||tensor - mean||  — 본 class
      4. JK aggregation: outer SchemaHeteroGATv2 가 jumping_knowledge 활성화 시 multi-layer 위 처리

    Hyperparameter 명세 (V6-W1 P1d, 본 class 기준):
      - pairnorm_scale ∈ {0.5, 1.0, 2.0}
      - gcnii_beta_lambda ∈ default 0.5 (β_l = log(λ/l+1))
      - α (outer initial residual): SchemaHeteroGATv2.initial_residual_alpha 가 처리
      - jk_mode ∈ {concat, max}: outer SchemaHeteroGATv2.jumping_knowledge 가 활성화

    V1~V5 chain 의 Phase2_B5 (PN+IR+JK+DS+L=2+AC+ListNet) subset → R@15 ≈ 0.6018 (Phase2_B5)
    유사 예상 (DECISIONS 2026-06-01 §V6-W1 §근거 P1d).
    """

    def __init__(self, *args, pairnorm_scale: float = 1.0, jk_mode: str = "concat", **kwargs):
        super().__init__(*args, **kwargs)
        if pairnorm_scale <= 0.0:
            raise ValueError(f"pairnorm_scale must be > 0, got {pairnorm_scale}")
        if jk_mode not in JKGATv2Conv.JK_MODES:
            raise ValueError(f"jk_mode must be in {JKGATv2Conv.JK_MODES}, got {jk_mode}")
        self.pairnorm = PairNorm(scale=float(pairnorm_scale))
        self.pairnorm_scale = float(pairnorm_scale)
        self.jk_mode = jk_mode

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):  # type: ignore[override]
        # super().forward (= GCNIIIRGATv2Conv ← GCNIIGATv2Conv) 가 GCNII IM 처리 후 반환.
        out = super().forward(x, edge_index, edge_attr=edge_attr,
                              return_attention_weights=return_attention_weights)
        if isinstance(out, tuple):
            tensor, *rest = out
            tensor = self.pairnorm(tensor)
            self._captured_output = tensor.detach()
            return (tensor, *rest)
        tensor = self.pairnorm(out)
        self._captured_output = tensor.detach()
        return tensor


# ──────────────────────────────────────────────────────────────────────
# Mitigation V6-W2 — Edge-Type 분리 메시지 패싱
# (DECISIONS 2026-06-04 §V6-W2~W4 활성 launch + RFP §4 Phase 2)
#
# RFP Phase 2 목적: "단일 평균을 관계별로 분리하여 hub 의 무차별 1/d 희석을 완화."
#
# ── 구조적 진단 (V6-W2 가 메우는 gap) ──
# 기존 SchemaHeteroGATv2 는 이미 PyG HeteroConv + 관계별 GATv2Conv (5 base edge type 각각
# 별도 파라미터) 사용 — 즉 base relation 분리는 이미 존재. 단 두 미탐색 gap:
#   (g1) `add_self_loops=False` — conv 내부에 self-loop 가 전무. 테이블 hub 가 컬럼 30개를
#        belongs_to 로 평균할 때 자기 신호 weight = 0 (within-conv). 자기 정보는 오직 model-level
#        skip_dict / initial_residual 로만 보존. → hub identity collapse 의 직접 원인.
#   (g2) cross-relation aggregation 이 `aggr='mean'` 고정 단 self-loop 가 그 평균에 미포함.
#
# V6-W2 의 핵심 개입: **per-node-type self-loop 관계를 별도 파라미터로 명시 주입**. 이로써
# 각 노드가 degree-diluted neighbor aggregation 과 분리된, 희석되지 않은 self-channel 을
# 획득. 테이블 hub 의 cross-relation mean 이 {belongs_to(컬럼 평균), table_to_table, self_loop}
# 의 평균 → self 가 1/(관계 수) weight (1/degree 가 아님) → hub 1/d 희석 완화의 직접 기제.
#
# ── HeteroConv vs RGATConv 선택 근거 ──
# **HeteroConv (관계별 GATv2Conv) 채택**. RGATConv 은 basis / block-diagonal decomposition 으로
# 관계 간 파라미터를 *공유* (parameter-efficient) → RFP 의 "관계별 분리 극대화" 의도와 반대.
# HeteroConv 의 관계별 fully-separate 파라미터가 edge-type 분리의 maximal form. 또한 본 모델은
# heterograph-native (x_dict / edge_index_dict) — HeteroConv route 가 V-track / supernode /
# capture / Phase 1 (PairNorm·IR·JK, model-level) 와 무손실 합성.
#
# ── self-loop 처리 방침 (RFP §1.3 / §4 Phase 2 "self-loop 처리 방침 명시") ──
#   - 방침: per-node-type 명시 self-loop 관계 `(nt, "self_loop_<nt>", nt)` 를 별도 GATv2Conv
#     파라미터로 forward 시점 주입. GATv2Conv 의 `add_self_loops` 는 미사용 (bipartite 관계
#     table↔column 등에서 src≠dst node type 이므로 PyG auto self-loop 적용 불가 + 의미 없음).
#   - 근거: 희석되지 않은 self-channel → hub 1/d 희석 완화.
#
# ── 역방향 엣지 처리 방침 (RFP §4 Phase 2 "rev_belongs_to 등 정의") ──
#   - base graph 가 이미 forward/reverse 쌍을 별도 관계로 보유 + 각각 별도 파라미터:
#       has_column (table→column)      ↔ belongs_to (column→table)   [rev_belongs_to 역할]
#       is_source_of (column→fk_node)  ↔ points_to (fk_node→column)  [FK 역방향]
#       table_to_table (table→table)   = FK-derived inter-table (대칭)
#   - 따라서 별도 rev_* 관계 추가 불필요 — 기존 쌍이 이미 분리 파라미터. 본 class 는 이를
#     명시 문서화 + self-loop 분리만 추가.
#
# ── PK/FK 세분 관계 typing (RFP `belongs_to`/`PK`/`FK` 분리) ──
#   현 graph 는 PK 컬럼을 일반 컬럼과 구분 안 함 (모두 belongs_to). PK 별도 관계 typing 은
#   Builder edge type metadata (per-column PK/FK mask) 의존 → cross-module (root 경유 escalation).
#   `edge_type_split_key_relations=True` 는 Builder metadata 미배선 상태에서 NotImplementedError —
#   forward-compat flag 로 유지하되 honest fail.
# ──────────────────────────────────────────────────────────────────────


class _CaptureGATv2Conv(GATv2Conv):
    """V6-W2 내부 per-relation conv — standard GATv2 attention + softmax 직후 alpha capture.

    EdgeTypeSplitHeteroConv 의 관계별 conv 로 사용. V6-W1 capture convs 와 동일 `_captured_alpha`
    프로토콜 (analyzer attention entropy 측정 위). attention path 표준 (row-stochastic softmax 유지) —
    edge-type 분리는 관계 단위 분리 + self-loop axis 이지, within-conv attention 수정이 아님.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._captured_alpha: Optional[torch.Tensor] = None

    def edge_update(self, x_j, x_i, edge_attr, index, ptr, dim_size):  # type: ignore[override]
        alpha = super().edge_update(x_j, x_i, edge_attr, index, ptr, dim_size)
        self._captured_alpha = alpha.detach()
        return alpha


class EdgeTypeSplitHeteroConv(nn.Module):
    """V6-W2 — Edge-Type 분리 메시지 패싱 (RFP Phase 2). HeteroConv drop-in 대체.

    PyG `HeteroConv` (관계별 fully-separate `GATv2Conv` 파라미터) 를 wrap 하고, 그 위에
    **per-node-type self-loop 관계를 별도 파라미터로 주입**. call signature 는 HeteroConv 와 동일
    (`conv(x_dict, edge_index_dict) -> out_dict`) → 기존 `SchemaHeteroGATv2.forward` 의 conv loop
    에 무수정 drop-in. PairNorm / Initial Residual / JK (Phase 1, model-level) 는 conv 외부에서
    적용되므로 자동 합성.

    파라미터:
        node_types: 그래프 node type 목록 (table / column / fk_node [/ query_node]).
        edge_types: base + supernode 관계 목록 (각각 별도 GATv2Conv 파라미터).
        hidden_channels / heads: GATv2 차원.
        self_loops: True 시 per-node-type self-loop 관계 주입 (V6-W2 핵심 개입).
        aggr: cross-relation aggregation ('mean' / 'sum' / 'max' / 'min' / 'mul') — HeteroConv 동일.
        capture: True 시 관계별 conv 가 `_captured_alpha` 기록 (analyzer hook 호환).

    self-loop 주입 (forward 시점):
        각 node type nt (x_dict 에 존재 + size>0) 에 대해 `(nt, "self_loop_<nt>", nt)` 관계의
        edge_index = [[0..N-1],[0..N-1]] 를 주입. HeteroConv 의 same-type 분기 (src==dst) 가
        single-tensor 모드로 처리 → 각 노드가 자기 자신만 attend → undiluted self-transform.

    capture 호환:
        `convs` property 가 inner HeteroConv 의 ModuleDict 를 노출 →
        `SchemaHeteroGATv2.forward` 의 model-level capture loop 이 관계별 `_captured_alpha` 를 읽음.
    """

    def __init__(self, node_types, edge_types, hidden_channels, heads, *,
                 self_loops: bool = True, aggr: str = "mean", capture: bool = True):
        super().__init__()
        self.node_types = list(node_types)
        self.heads = heads
        self.self_loops = bool(self_loops)
        self.aggr_mode = aggr
        self.capture = bool(capture)
        # per-node-type self-loop 관계 (별도 파라미터). schema + (옵션) query_node 모두.
        self.self_loop_rel: Dict[str, tuple] = (
            {nt: (nt, f"self_loop_{nt}", nt) for nt in self.node_types}
            if self.self_loops else {}
        )

        _ConvCls = _CaptureGATv2Conv if capture else GATv2Conv
        conv_dict: Dict[tuple, nn.Module] = {}
        for et in edge_types:
            conv_dict[et] = _ConvCls(
                -1, hidden_channels, heads=heads, add_self_loops=False)
        for nt, sl in self.self_loop_rel.items():
            conv_dict[sl] = _ConvCls(
                -1, hidden_channels, heads=heads, add_self_loops=False)

        # inner HeteroConv — 관계별 메시지 패싱 + cross-relation aggregation (battle-tested PyG).
        self._hetero = HeteroConv(conv_dict, aggr=aggr)

    @property
    def convs(self):
        """capture 호환 alias — model-level capture loop 이 inner conv 의 _captured_alpha 를 읽음.

        property 로 노출 (재할당 시 파라미터 double-register 회피). inner HeteroConv 의 ModuleDict
        (string key '__'.join(edge_type)) 반환.
        """
        return self._hetero.convs

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        ei = dict(edge_index_dict)
        if self.self_loops:
            for nt, sl in self.self_loop_rel.items():
                x = x_dict.get(nt)
                if x is not None and x.size(0) > 0:
                    idx = torch.arange(x.size(0), device=x.device, dtype=torch.long)
                    ei[sl] = torch.stack([idx, idx], dim=0)
        return self._hetero(x_dict, ei)


# Alias for user-facing naming convention (DECISIONS 2026-05-13 entry).
# Existing root-implementation class names retained for ckpt backward compat; aliases below
# match user spec naming (Step 1.1/1.3 sketch).
GATEConv = GATEGATv2Conv
FullAEROGATConv = FullAEROGATv2Conv
# V6-W1 aliases (DECISIONS 2026-06-01 §V6-W1).
PairNormGATConv = PairNormGATv2Conv
GCNIIIRGATConv = GCNIIIRGATv2Conv
JKGATConv = JKGATv2Conv
DropInComboGATConv = DropInComboGATv2Conv
# V6-W2 alias (DECISIONS 2026-06-04 §V6-W2~W4).
EdgeTypeSplitConv = EdgeTypeSplitHeteroConv


# Mitigation V4 + V5 + V6-W1 layer type 선택.
# V4: 2026-05-11 / V5: 2026-05-12 / V6-W1: 2026-06-01.
GAT_LAYER_TYPES = {
    "standard", "lngin", "softplus", "gate", "gcnii", "aero_full",
    # V6-W1 drop-in 3종 격리 + 조합 (DECISIONS 2026-06-01)
    "pairnorm", "gcnii_ir", "jk", "dropin_combo",
}

# Mitigation V6-W3 — Phase 3 hub 차수 축소 (DECISIONS 2026-06-06). Builder 3 variants
# (module:builders v6w3_builders.py, commit fce866c) 가 graph 구조를 확장하므로, 모델은
# 신규 node type + edge type 만 등록하면 forward (node_types/all_edge_types 동적 순회) 가
# 자동 처리. classifier (table/column/fk_node) 는 신규 type 무시 (selection target 아님).
#   A `V6W3VirtualSummaryBuilder` — `table_summary` virtual node (table↔summary↔column 2-hop)
#   B `V6W3ColumnPoolingBuilder`  — table.x = column pooling override (신규 node/edge 없음)
#   C `V6W3HubLocalVNBuilder`     — `local_vn` (hub-only Local VN_G)
V6W3_VARIANTS = {None, "A", "B", "C"}
# variant → 신규 node type (B 는 구조 변경 없음 → 항목 없음).
V6W3_NODE_TYPE = {"A": "table_summary", "C": "local_vn"}
# variant → 신규 edge type 리스트 (Builder 가 graph_data 에 채우는 relation 키와 정확 일치).
V6W3_EDGE_TYPES = {
    "A": [
        ("table", "has_summary", "table_summary"),
        ("table_summary", "summary_of", "table"),
        ("table_summary", "summarizes", "column"),
        ("column", "aggregated_by", "table_summary"),
    ],
    "C": [
        ("table", "has_local_vn", "local_vn"),
        ("local_vn", "serves_table", "table"),
        ("local_vn", "aggregates", "column"),
        ("column", "feeds_into", "local_vn"),
    ],
}

# Mitigation V6-W5 — Phase 5 GAT layer-level intervention (DECISIONS 2026-06-07).
# collapse origin 진단 (analyzer §1+§4, 가설 B 확정): hub 컬럼 99.4% 가 incoming edge 위 단일
# table 노드만 수신 (self-loop=False + per-layer residual 없음) → 첫 conv 에서 단일 공유 소스
# `W·(table L0)` 동일 메시지 → intra-MAD L0 0.4201 → L1 0.0136 (−96.8%) 강제 수렴.
# V6-W5 = 이 single-shared-source mechanism 을 정확히 겨냥한 decisive final wave.
#   a `column self-loop`     : ('column','self_loop_column','column') identity relation 주입 —
#                              컬럼 자기 L0 self-channel 보존 (단일 table 소스 독점 완화). 가장 직접적.
#   b `per-layer residual`   : out = ELU(conv(h)) + h_prev (현재 skip_dict 는 L_out 에만) —
#                              input differentiation layer 별 보존.
#   c `a + b`                : 두 mechanism 결합.
# M4 anchor standard arch 위 최소 개입 — gat_layer_type='standard' + edge_type_split=False +
# v6w3_variant=None 강제 (축 격리). bipartite relation (table→column) 은 add_self_loops 무의미하므로
# column→column self relation 으로 구현 (V6-W2 self-loop 패턴, column 전용).
V6W5_VARIANTS = {None, "a", "b", "c"}
V6W5_COLUMN_SELF_LOOP_REL = ("column", "self_loop_column", "column")

# Mitigation V6-W6 — Phase 6 directed query SuperNode + V6-W5 self-loop 조합 (DECISIONS 2026-06-07 #3,
# 지도교수 trigger 로 V6 closure 일시 reopening). 새 conv 기제가 아니라 기존 두 축의 결합:
#   directed query SuperNode (query_supernode=True + supernode_edge_direction='directed_from_sn')
#   + V6-W5 column self-loop (a/b) [+ per-layer residual (b)].
# 가설: 이전 SuperNode 실패는 collapse 아키텍처(self-loop 부재) 위 측정 — self-loop 로 컬럼 정체성
# 유지 상태 + directed query-SN broadcast 조합은 미측정 (column gold-alignment 새 경로 가능성).
# v6w6_variant 는 self-contained — v6w5_variant 동시 설정 없이도 self-loop/residual 활성화
# (config 가 둘 다 설정해도 OR 결합으로 일관). directed SuperNode 자체는 query_supernode /
# supernode_edge_direction flag 가 담당 (validation 으로 결합 강제).
#   a: directed SN + column self-loop
#   b: directed SN + column self-loop + per-layer residual
V6W6_VARIANTS = {None, "a", "b"}


def _make_gatv2_conv(in_channels, out_channels, heads, add_self_loops,
                    *, drop_message_p: float = 0.0,
                    use_layernorm_pre_softmax: bool = False,
                    gat_layer_type: str = "standard",
                    softplus_symmetric_norm: bool = True,
                    gcnii_beta_lambda: float = 0.5,
                    gcnii_layer_idx: int = 1,
                    # V6-W1 P1a/P1c/P1d hyperparameters (DECISIONS 2026-06-01)
                    pairnorm_scale: float = 1.0,
                    jk_mode: str = "concat") -> GATv2Conv:
    """Mitigation v2/V4/V5/V6-W1 옵션에 따라 적절한 GATv2Conv 변형 instance 생성.

    Priority (선택 우선순위, 한 번에 하나만 활성화 권장):
      - gat_layer_type='pairnorm'      → PairNormGATv2Conv     (V6-W1 P1a, Zhao & Akoglu 2020)
      - gat_layer_type='gcnii_ir'      → GCNIIIRGATv2Conv      (V6-W1 P1b, alias of V5-B)
      - gat_layer_type='jk'            → JKGATv2Conv           (V6-W1 P1c, marker for outer JK)
      - gat_layer_type='dropin_combo'  → DropInComboGATv2Conv  (V6-W1 P1d, P1a+P1b+P1c)
      - gat_layer_type='gate'          → GATEGATv2Conv         (V5-A, Mustafa & Burkholz 2024)
      - gat_layer_type='gcnii'         → GCNIIGATv2Conv        (V5-B, Chen 2020 + Peng 2024)
      - gat_layer_type='aero_full'     → FullAEROGATv2Conv     (V5-C, Lee 2023 + Hop Attention outer)
      - gat_layer_type='lngin'         → LNGINGATv2Conv        (V4-A)
      - gat_layer_type='softplus'      → SoftplusGATv2Conv     (V4-B)
      - drop_message_p > 0.0           → DropMessageGATv2Conv  (v2 #1)
      - use_layernorm_pre_softmax      → LayerNormGATv2Conv    (v2 #3)
      - 둘 다 활성화 → 두 옵션 결합 (fallback class 우선 LayerNorm + drop_message)
      - 둘 다 비활성화 → 기존 GATv2Conv (backward compat)
    """
    # V6-W1 P1a: PairNorm 단독 (drop-in)
    if gat_layer_type == "pairnorm":
        return PairNormGATv2Conv(
            in_channels, out_channels, heads=heads, add_self_loops=add_self_loops,
            pairnorm_scale=pairnorm_scale)
    # V6-W1 P1b: GCNII Initial Residual + Identity Mapping (V5-B subclass, alias cell)
    if gat_layer_type == "gcnii_ir":
        return GCNIIIRGATv2Conv(
            in_channels, out_channels, heads=heads, add_self_loops=add_self_loops,
            gcnii_beta_lambda=gcnii_beta_lambda,
            gcnii_layer_idx=gcnii_layer_idx)
    # V6-W1 P1c: JK marker (outer JK aggregation 활성화 신호)
    if gat_layer_type == "jk":
        return JKGATv2Conv(
            in_channels, out_channels, heads=heads, add_self_loops=add_self_loops,
            jk_mode=jk_mode)
    # V6-W1 P1d: PairNorm + GCNII IR + JK marker combo
    if gat_layer_type == "dropin_combo":
        return DropInComboGATv2Conv(
            in_channels, out_channels, heads=heads, add_self_loops=add_self_loops,
            gcnii_beta_lambda=gcnii_beta_lambda,
            gcnii_layer_idx=gcnii_layer_idx,
            pairnorm_scale=pairnorm_scale,
            jk_mode=jk_mode)
    # V5-A: GATE (Conservation Law 수정)
    if gat_layer_type == "gate":
        return GATEGATv2Conv(
            in_channels, out_channels, heads=heads, add_self_loops=add_self_loops)
    # V5-B: GCNII-style Identity Mapping
    if gat_layer_type == "gcnii":
        return GCNIIGATv2Conv(
            in_channels, out_channels, heads=heads, add_self_loops=add_self_loops,
            gcnii_beta_lambda=gcnii_beta_lambda,
            gcnii_layer_idx=gcnii_layer_idx)
    # V5-C: Full AERO (Softplus + Sym-Norm + Hop Attention is outer-handled)
    if gat_layer_type == "aero_full":
        return FullAEROGATv2Conv(
            in_channels, out_channels, heads=heads, add_self_loops=add_self_loops,
            symmetric_norm=softplus_symmetric_norm)
    # V4-A: LN+GIN Combo
    if gat_layer_type == "lngin":
        return LNGINGATv2Conv(
            in_channels, out_channels, heads=heads, add_self_loops=add_self_loops)
    # V4-B: Softplus + Symmetric Norm
    if gat_layer_type == "softplus":
        return SoftplusGATv2Conv(
            in_channels, out_channels, heads=heads, add_self_loops=add_self_loops,
            symmetric_norm=softplus_symmetric_norm)
    if use_layernorm_pre_softmax and drop_message_p > 0.0:
        # 두 옵션 동시 적용 — multiple inheritance 보다 LayerNorm subclass 에 drop_message 추가.
        class _LayerNormDropMessageGATv2Conv(LayerNormGATv2Conv):
            def __init__(self, *a, drop_message_p: float = 0.0, **kw):
                super().__init__(*a, **kw)
                self.drop_message_p = float(drop_message_p)
            def message(self, x_j, alpha):
                msg = x_j * alpha.unsqueeze(-1)
                if self.training and self.drop_message_p > 0.0:
                    msg = F.dropout(msg, p=self.drop_message_p, training=self.training)
                return msg
        return _LayerNormDropMessageGATv2Conv(
            in_channels, out_channels, heads=heads,
            add_self_loops=add_self_loops, drop_message_p=drop_message_p)
    if use_layernorm_pre_softmax:
        return LayerNormGATv2Conv(
            in_channels, out_channels, heads=heads, add_self_loops=add_self_loops)
    if drop_message_p > 0.0:
        return DropMessageGATv2Conv(
            in_channels, out_channels, heads=heads, add_self_loops=add_self_loops,
            drop_message_p=drop_message_p)
    return GATv2Conv(in_channels, out_channels, heads=heads, add_self_loops=add_self_loops)


def _make_gin_conv(in_channels: int, hidden_channels: int, heads: int,
                   train_eps: bool = False) -> GINConv:
    """Mitigation v3 #1 — GIN-style aggregation.

    GIN propagation: `h' = MLP((1+ε)·h + Σ_{j∈N(i)} h_j)` (within-edge aggr='add' fix).
    softmax + weighted-mean 의 attention path 를 완전히 우회 → mech(ii) edge softmax
    over-concentration 의 root cause 차단 시도. attention 자체가 없으므로 #1 DropMessage /
    #3 LayerNorm pre-softmax 와 결합 불가.

    output 차원: hidden_channels × heads (기존 GATv2Conv out 동일 → PairNorm/JK/skip 호환).
    in_channels=-1 시 LazyLinear 로 첫 dim 자동 결정 (기존 GATv2Conv(-1,...) 동치).
    """
    out_dim = hidden_channels * heads
    if in_channels < 0:
        first_lin = nn.LazyLinear(out_dim)
    else:
        first_lin = nn.Linear(in_channels, out_dim)
    mlp = nn.Sequential(
        first_lin,
        nn.LeakyReLU(0.1),
        nn.Linear(out_dim, out_dim),
    )
    return GINConv(mlp, eps=0.0, train_eps=train_eps)


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
        # V-3-ext: Directed Top-K SuperNode (Part III, advisor 제안)
        # threshold_mode='top_k' (기존 V-3 = supernode_topk 사용),
        # 'percentile' (per-query Pn — value=80 → P80),
        # 'abs_tau' (per-query min-max norm 후 절대 cutoff — value=0.7 → score>=0.7).
        # score_normalization='minmax' 시 percentile/abs_tau 모두 norm 후 score 기준.
        supernode_threshold_mode: str = "top_k",
        supernode_threshold_value: Optional[float] = None,
        supernode_score_normalization: str = "minmax",
        # Mitigation v2 (DECISIONS 2026-05-07 §1(C)/(D)) — 학위 본 심사 전 mech(ii) mitigation
        # 후보 학습용. 기본값은 모두 OFF (Phase 2 b8 backward compat).
        # #1: drop_message_p > 0 시 DropMessageGATv2Conv 활성화.
        # #3: use_layernorm_pre_softmax 시 LayerNormGATv2Conv 활성화.
        # #2: aggregation_type ∈ {'mean'(default), 'sum', 'max'} — HeteroConv aggr.
        drop_message_p: float = 0.0,
        use_layernorm_pre_softmax: bool = False,
        aggregation_type: str = "mean",
        # Mitigation V4 — Architectural Intervention (DECISIONS 2026-05-11)
        # 보고서 §C-1 + §C-2 채택. row-stochasticity 자체를 변경하는 architectural intervention.
        # 'lngin' = V4-A (Pre-softmax LN + GIN-style post-aggregation MLP)
        # 'softplus' = V4-B (Softplus + Symmetric Normalization, row-stochasticity 파괴)
        # 'standard' (default) = v2 #1/#3 옵션 또는 unmodified GATv2Conv (backward compat).
        gat_layer_type: str = "standard",
        softplus_symmetric_norm: bool = True,
        # Mitigation V5 — Tier 1+2 4-Direction (DECISIONS 2026-05-12 + 2026-05-13)
        # 'gate'      = V5-A (Mustafa & Burkholz NeurIPS 2024, attention parameter 분리)
        # 'gcnii'     = V5-B (Chen et al. ICML 2020 + Peng et al. CIKM 2024, Identity Mapping)
        # 'aero_full' = V5-C (Lee et al. ICML 2023, V4-B + Hop Attention + Cumulative Attention)
        gcnii_beta_lambda: float = 0.5,
        aero_hop_attention: bool = False,
        # V5-C 의 paper Theorem 3 SR2OS guarantee 의 (b) Cumulative Attention residual.
        # 본 framework 의 layer-별 별도 conv 구조에서는 conv-level α^(l) 누적이 직접 불가.
        # 대신 hidden state level outer residual `H^(l) += λ_cum · H^(l-1)` 로 simulate.
        # paper §4.2 Theorem 4 의 hop attention sum form 이 cumulative residual 의 충분 form 이므로
        # aero_hop_attention=True 와 결합 시 효과 중복 가능 — 본 옵션은 hop attention OFF 시 단독 사용.
        aero_cumulative_attention: bool = False,
        aero_cumulative_decay: float = 1.0,
        # Mitigation V6-W1 — Drop-in 3종 격리 + 조합 (DECISIONS 2026-06-01 §V6-W1)
        # P1a `pairnorm`      : PairNorm scale ∈ {0.5, 1.0, 2.0}
        # P1b `gcnii_ir`      : α ∈ {0.05, 0.1, 0.2} (= initial_residual_alpha) + optional
        #                       β = log(λ/l+1) via gcnii_beta_lambda
        # P1c `jk`            : jk_mode ∈ {concat, max} — outer jumping_knowledge 자동 활성화
        # P1d `dropin_combo`  : P1a + P1b + P1c 한 forward 안 sequential
        v6w1_pairnorm_scale: float = 1.0,
        v6w1_jk_mode: str = "concat",
        # V6-W1 layer-wise output capture toggle (analyzer 의 Dirichlet/MAD/attention entropy 측정 위).
        # True 시 forward 매 호출 후 `self._captured_layer_outputs` (list[dict[nt, tensor]]) 에
        # layer 별 detached hidden state 보관 — analyzer 가 hook 으로 꺼냄.
        capture_layerwise_outputs: bool = False,
        # Mitigation V6-W2 — Edge-Type 분리 메시지 패싱 (DECISIONS 2026-06-04 §V6-W2~W4)
        # edge_type_split=True 시 conv 를 EdgeTypeSplitHeteroConv 로 교체 — 관계별 GATv2Conv
        # (base relation 분리 retain) + per-node-type self-loop 별도 파라미터 주입 (핵심 개입).
        # 전제 (RFP Phase 2): Phase 1 (PairNorm + IR + JK, model-level) 조합 위 적용 가능 —
        # pairnorm_mode / initial_residual_alpha / jumping_knowledge 와 직교 합성. 또는 standard
        # arch 위 edge-type 분리 단독 (Phase 1 flags off) — 두 변형 비교 위 hyperparameter flag.
        edge_type_split: bool = False,
        edge_type_split_self_loops: bool = True,
        edge_type_split_aggr: str = "mean",
        # PK/FK 세분 관계 typing — Builder edge type metadata (per-column key mask) 의존.
        # 미배선 상태에서 True 시 NotImplementedError (cross-module, root escalation).
        edge_type_split_key_relations: bool = False,
        # Mitigation V6-W3 — Phase 3 hub 차수 축소 (DECISIONS 2026-06-06).
        # None (default, backward compat) | "A" (table_summary) | "B" (column pooling, 구조 무변경)
        # | "C" (local_vn hub-only). A/C 는 신규 node type + edge type 을 node_types/all_edge_types
        # 에 주입 → lin_dict/convs/skip/JK/res_proj/pairnorm 자동 allocate. Builder 가 graph_data 에
        # 채우는 relation 키와 정확 일치 (v6w3_builders.py). classifier 는 신규 type 무시.
        v6w3_variant: Optional[str] = None,
        # Mitigation V6-W5 — Phase 5 GAT layer-level intervention (DECISIONS 2026-06-07).
        # None (backward compat) | "a" (column self-loop) | "b" (per-layer residual) | "c" (a+b).
        # single-shared-source collapse mechanism 을 conv 자체에서 수정. builder 변경 없음 (enriched).
        v6w5_variant: Optional[str] = None,
        # Mitigation V6-W6 — Phase 6 directed SuperNode + self-loop 조합 (DECISIONS 2026-06-07 #3).
        # None (backward compat) | "a" (directed SN + column self-loop) | "b" (a + per-layer residual).
        # query_supernode=True + supernode_edge_direction='directed_from_sn' 와 결합 강제 (validation).
        v6w6_variant: Optional[str] = None,
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
        # Mitigation v2 / v3 검증
        if not (0.0 <= float(drop_message_p) < 1.0):
            raise ValueError(f"drop_message_p must be in [0.0, 1.0), got {drop_message_p}")
        if aggregation_type not in AGGREGATION_TYPES:
            raise ValueError(
                f"aggregation_type must be in {AGGREGATION_TYPES}, got {aggregation_type}"
            )
        # Mitigation v3 GIN 모드는 attention 자체가 없으므로 v2 #1/#3 와 결합 의미 없음.
        if aggregation_type == "gin" and (drop_message_p > 0.0 or use_layernorm_pre_softmax):
            raise ValueError(
                "aggregation_type='gin' is incompatible with drop_message_p>0 or "
                "use_layernorm_pre_softmax=True (GIN has no attention/softmax to mitigate)"
            )
        # Mitigation V4 + V5 validation
        if gat_layer_type not in GAT_LAYER_TYPES:
            raise ValueError(f"gat_layer_type must be in {GAT_LAYER_TYPES}, got {gat_layer_type}")
        # V4-A (lngin) 는 LN + GIN combo 가 layer 내부에서 동작 — aggregation_type='gin' 과 결합하면
        # 의미 중복 + dim 충돌 가능. V4-B (softplus) 는 softmax 자체를 대체하므로 LN/dropmessage 와 결합 무의미.
        # V6-W1 4 변형 도 같은 격리 원칙: 기존 v2/v3 mitigation 과 결합 금지.
        _v4v5v6_types = (
            "lngin", "softplus", "gate", "gcnii", "aero_full",
            "pairnorm", "gcnii_ir", "jk", "dropin_combo",
        )
        if gat_layer_type in _v4v5v6_types:
            if aggregation_type == "gin":
                raise ValueError(
                    f"gat_layer_type='{gat_layer_type}' is incompatible with aggregation_type='gin' "
                    f"(V4/V5/V6-W1 layers handle aggregation internally)")
            if drop_message_p > 0.0 or use_layernorm_pre_softmax:
                raise ValueError(
                    f"gat_layer_type='{gat_layer_type}' supersedes drop_message_p / use_layernorm_pre_softmax "
                    f"(V4/V5/V6-W1 layers already incorporate these mitigations or replace softmax)")
        # V5-C 의 Hop Attention 은 aero_full 에서만 의미가 있다.
        if aero_hop_attention and gat_layer_type != "aero_full":
            raise ValueError(
                f"aero_hop_attention=True 는 gat_layer_type='aero_full' (V5-C) 에서만 의미가 있음 — "
                f"got gat_layer_type='{gat_layer_type}'")
        # V5-C Cumulative Attention 도 aero_full 에서만 의미.
        if aero_cumulative_attention and gat_layer_type != "aero_full":
            raise ValueError(
                f"aero_cumulative_attention=True 는 gat_layer_type='aero_full' (V5-C) 에서만 의미가 있음 — "
                f"got gat_layer_type='{gat_layer_type}'")
        if gcnii_beta_lambda <= 0.0:
            raise ValueError(f"gcnii_beta_lambda must be > 0.0, got {gcnii_beta_lambda}")
        if aero_cumulative_decay < 0.0 or aero_cumulative_decay > 2.0:
            raise ValueError(
                f"aero_cumulative_decay must be in [0.0, 2.0], got {aero_cumulative_decay}")
        # V6-W1 validation
        if v6w1_pairnorm_scale <= 0.0:
            raise ValueError(f"v6w1_pairnorm_scale must be > 0.0, got {v6w1_pairnorm_scale}")
        if v6w1_jk_mode not in JKGATv2Conv.JK_MODES:
            raise ValueError(
                f"v6w1_jk_mode must be in {JKGATv2Conv.JK_MODES}, got {v6w1_jk_mode}")
        # V6-W1 P1c/P1d 의 JK marker layer 가 활성화되면, outer jumping_knowledge 도 동일 mode 로
        # 자동 활성화 — JK aggregation 은 multi-layer outer process 이므로 marker conv 만으로는
        # aggregation 효과가 발현되지 않음.
        if gat_layer_type in ("jk", "dropin_combo"):
            if jumping_knowledge == "none":
                jumping_knowledge = v6w1_jk_mode
            elif jumping_knowledge != v6w1_jk_mode:
                raise ValueError(
                    f"gat_layer_type='{gat_layer_type}' with v6w1_jk_mode='{v6w1_jk_mode}' "
                    f"conflicts with jumping_knowledge='{jumping_knowledge}' "
                    f"(must be 'none' for auto-activation or match v6w1_jk_mode)")

        # V6-W2 validation
        if edge_type_split:
            # edge-type 분리는 별도 axis — per-conv V4/V5/V6-W1 layer type 과 결합 금지 (ablation
            # 축 혼동 방지). 관계별 conv 는 standard GATv2 (+ self-loop 분리) 만.
            if gat_layer_type != "standard":
                raise ValueError(
                    f"edge_type_split=True requires gat_layer_type='standard' "
                    f"(edge-type 분리는 별도 axis — V4/V5/V6-W1 per-conv type 과 미결합), "
                    f"got gat_layer_type='{gat_layer_type}'")
            if aggregation_type == "gin":
                raise ValueError(
                    "edge_type_split=True is incompatible with aggregation_type='gin' "
                    "(GIN replaces attention; edge-type 분리는 관계별 GATv2 attention 기반)")
            if edge_type_split_aggr not in HETEROCONV_AGGR_TYPES:
                raise ValueError(
                    f"edge_type_split_aggr must be in {HETEROCONV_AGGR_TYPES}, "
                    f"got {edge_type_split_aggr}")
            if edge_type_split_key_relations:
                # PK/FK 세분 관계 typing 은 Builder per-column key mask metadata 의존 (미배선).
                raise NotImplementedError(
                    "edge_type_split_key_relations=True (PK/FK 세분 관계 typing) 은 Builder "
                    "edge type metadata (per-column PK/FK mask) 의존 — 현재 미배선. "
                    "cross-module 작업 (root 경유 module:builders escalation) 필요. "
                    "self-loop 분리 (edge_type_split_self_loops) + 기존 forward/reverse 쌍 분리는 "
                    "metadata 무관하게 활성 — edge_type_split_key_relations=False 로 진행.")

        # V6-W3 validation
        if v6w3_variant not in V6W3_VARIANTS:
            raise ValueError(
                f"v6w3_variant must be in {V6W3_VARIANTS}, got {v6w3_variant!r}")

        # V6-W5 validation — M4 anchor standard arch 위 최소 개입 (축 격리).
        if v6w5_variant not in V6W5_VARIANTS:
            raise ValueError(
                f"v6w5_variant must be in {V6W5_VARIANTS}, got {v6w5_variant!r}")
        if v6w5_variant is not None:
            if gat_layer_type != "standard":
                raise ValueError(
                    f"v6w5_variant={v6w5_variant!r} requires gat_layer_type='standard' "
                    f"(V6-W5 는 standard conv 위 self-loop/residual 최소 개입 — V4/V5/V6-W1 per-conv "
                    f"type 과 미결합), got gat_layer_type={gat_layer_type!r}")
            if edge_type_split:
                raise ValueError(
                    "v6w5_variant 은 edge_type_split=True 와 결합 금지 — V6-W5 는 standard HeteroConv "
                    "위 column self-loop relation 추가 (별도 axis). edge_type_split=False 로 진행.")
            if v6w3_variant is not None:
                raise ValueError(
                    f"v6w5_variant 은 v6w3_variant 와 결합 금지 (별도 wave/axis), "
                    f"got v6w3_variant={v6w3_variant!r}")
            if aggregation_type == "gin":
                raise ValueError(
                    "v6w5_variant 은 aggregation_type='gin' 과 비결합 (GIN 은 attention 없음 — "
                    "self-loop relation 은 GATv2 attention 기반).")

        # V6-W6 validation — directed SuperNode + self-loop 결합 강제.
        if v6w6_variant not in V6W6_VARIANTS:
            raise ValueError(
                f"v6w6_variant must be in {V6W6_VARIANTS}, got {v6w6_variant!r}")
        if v6w6_variant is not None:
            if not query_supernode:
                raise ValueError(
                    "v6w6_variant 은 directed query SuperNode 와의 조합 — query_supernode=True 필요 "
                    "(directed SN broadcast 가 column gold-alignment 새 경로의 핵심).")
            if supernode_edge_direction != "directed_from_sn":
                raise ValueError(
                    f"v6w6_variant 은 supernode_edge_direction='directed_from_sn' 필요 (DSN 정합), "
                    f"got {supernode_edge_direction!r}")
            if gat_layer_type != "standard":
                raise ValueError(
                    f"v6w6_variant requires gat_layer_type='standard' (V6-W5 정합), "
                    f"got gat_layer_type={gat_layer_type!r}")
            if edge_type_split:
                raise ValueError(
                    "v6w6_variant 은 edge_type_split=True 와 결합 금지 (별도 axis).")
            if v6w3_variant is not None:
                raise ValueError(
                    f"v6w6_variant 은 v6w3_variant 와 결합 금지, got v6w3_variant={v6w3_variant!r}")
            if aggregation_type == "gin":
                raise ValueError(
                    "v6w6_variant 은 aggregation_type='gin' 과 비결합 (attention 기반 self-loop).")

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
        self.supernode_threshold_mode = supernode_threshold_mode
        self.supernode_threshold_value = (
            float(supernode_threshold_value) if supernode_threshold_value is not None else None
        )
        self.supernode_score_normalization = supernode_score_normalization
        # Mitigation v2 state
        self.drop_message_p = float(drop_message_p)
        self.use_layernorm_pre_softmax = bool(use_layernorm_pre_softmax)
        self.aggregation_type = aggregation_type
        # Mitigation V4 state
        self.gat_layer_type = gat_layer_type
        self.softplus_symmetric_norm = bool(softplus_symmetric_norm)
        # Mitigation V5 state
        self.gcnii_beta_lambda = float(gcnii_beta_lambda)
        self.aero_hop_attention = bool(aero_hop_attention)
        self.aero_cumulative_attention = bool(aero_cumulative_attention)
        self.aero_cumulative_decay = float(aero_cumulative_decay)
        # Mitigation V6-W1 state
        self.v6w1_pairnorm_scale = float(v6w1_pairnorm_scale)
        self.v6w1_jk_mode = v6w1_jk_mode
        # Mitigation V6-W2 state
        self.edge_type_split = bool(edge_type_split)
        self.edge_type_split_self_loops = bool(edge_type_split_self_loops)
        self.edge_type_split_aggr = edge_type_split_aggr
        self.edge_type_split_key_relations = bool(edge_type_split_key_relations)
        # Mitigation V6-W3 state
        self.v6w3_variant = v6w3_variant
        # Mitigation V6-W5/W6 state — variant → 두 mechanism boolean 분해.
        # V6-W6 은 self-contained: a/b 모두 column self-loop, b 는 추가로 per-layer residual.
        # v6w5_variant 와 OR 결합 (config 가 둘 다 설정해도 일관). forward/construction 은 아래
        # 두 boolean 만 참조하므로 W5/W6 공통 경로 — 추가 분기 불필요.
        self.v6w5_variant = v6w5_variant
        self.v6w6_variant = v6w6_variant
        _w5_self = v6w5_variant in ("a", "c")
        _w5_res = v6w5_variant in ("b", "c")
        _w6_self = v6w6_variant in ("a", "b")
        _w6_res = v6w6_variant == "b"
        self.v6w5_column_self_loop = _w5_self or _w6_self
        self.v6w5_per_layer_residual = _w5_res or _w6_res
        self.capture_layerwise_outputs = bool(capture_layerwise_outputs)
        # V6 analyzer capture buffer — forward 매 호출 시 layer 별 detached hidden state 저장.
        # detach() 후 list 위 append, ckpt 비저장 (non-Parameter, non-buffer).
        self._captured_layer_outputs: Optional[List[Dict[str, torch.Tensor]]] = None
        # V6 analyzer attention capture — forward 매 호출 후 conv 별 _captured_alpha 의 snapshot.
        # layer_idx × edge_type → tensor 의 nested dict.
        self._captured_layerwise_alpha: Optional[List[Dict[tuple, torch.Tensor]]] = None

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
        # V6-W3 A/C: 신규 node type 등록 (B 는 구조 무변경). lin_dict/skip/JK/res_proj/pairnorm
        # 등 node_types 로부터 build 되는 모든 모듈이 자동 allocate.
        if self.v6w3_variant in V6W3_NODE_TYPE:
            node_types.append(V6W3_NODE_TYPE[self.v6w3_variant])

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
        # V6-W3 A/C: 신규 relation 등록. HeteroConv / EdgeTypeSplitHeteroConv 모두 all_edge_types
        # 기반으로 conv_dict 를 구성하므로 두 경로 자동 적용. Builder relation 키와 정확 일치.
        if self.v6w3_variant in V6W3_EDGE_TYPES:
            all_edge_types = all_edge_types + V6W3_EDGE_TYPES[self.v6w3_variant]
        # V6-W5-a/c: column self-loop relation 등록 (별도 GATv2Conv 파라미터). edge_index 는
        # forward 에서 컬럼 수만큼 identity (i→i) 로 주입. column 자기 L0 를 단일 table 소스와
        # 분리된 self-channel 로 보존 → single-shared-source 독점 완화.
        if self.v6w5_column_self_loop:
            all_edge_types = all_edge_types + [V6W5_COLUMN_SELF_LOOP_REL]
        # 추론/분석 introspection 용 (selector lazy init dummy graph 구성 + analyzer hook).
        self.all_edge_types = all_edge_types

        self.convs = nn.ModuleList()
        for layer_idx_zero in range(num_layers):
            if self.edge_type_split:
                # V6-W2 — Edge-Type 분리 메시지 패싱. HeteroConv (관계별 GATv2Conv, base relation
                # 분리 retain) + per-node-type self-loop 별도 파라미터 주입. gat_layer_type='standard'
                # 강제 (validation) — per-conv V4/V5/V6-W1 와 직교. Phase 1 (PairNorm·IR·JK) 는
                # model-level (아래 3/4/5 블록) 에서 적용 → 자동 합성.
                self.convs.append(EdgeTypeSplitHeteroConv(
                    node_types=node_types,
                    edge_types=all_edge_types,
                    hidden_channels=hidden_channels,
                    heads=heads,
                    self_loops=self.edge_type_split_self_loops,
                    aggr=self.edge_type_split_aggr,
                    capture=True,
                ))
            elif self.aggregation_type == "gin":
                # Mitigation v3 #1 — inner conv 를 GINConv 로 교체. within-edge aggr='add' (GIN 정의).
                # HeteroConv 의 cross-edge-type aggr 는 'mean' (default) — GIN 의 within-edge sum 과
                # cross-type 의 mean 을 분리.
                conv_dict = {
                    et: _make_gin_conv(-1, hidden_channels, heads=heads)
                    for et in all_edge_types
                }
                self.convs.append(HeteroConv(conv_dict, aggr="mean"))
            else:
                # V5-B (gcnii) 는 layer-index dependent β_l = log(λ/l + 1) — 1-indexed.
                # V6-W1 P1b (gcnii_ir) / P1d (dropin_combo) 도 동일 GCNII β_l 처리.
                conv_dict = {
                    et: _make_gatv2_conv(
                        -1, hidden_channels, heads=heads, add_self_loops=False,
                        drop_message_p=self.drop_message_p,
                        use_layernorm_pre_softmax=self.use_layernorm_pre_softmax,
                        gat_layer_type=self.gat_layer_type,
                        softplus_symmetric_norm=self.softplus_symmetric_norm,
                        gcnii_beta_lambda=self.gcnii_beta_lambda,
                        gcnii_layer_idx=layer_idx_zero + 1,
                        # V6-W1
                        pairnorm_scale=self.v6w1_pairnorm_scale,
                        jk_mode=self.v6w1_jk_mode,
                    )
                    for et in all_edge_types
                }
                # Mitigation v2 #2 — HeteroConv aggr 변경 (mean / sum / max / min / mul).
                # V4-A/B / V5-A/B/C 는 aggregation_type='mean' 강제 (V4/V5 layer 내부에서 aggregation 처리).
                self.convs.append(HeteroConv(conv_dict, aggr=self.aggregation_type))

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

        # 7c. V6-W5-b/c — per-layer residual projection. layer 0 의 input h_0 (hidden_channels) 을
        # conv output (hidden*heads) 차원으로 맞추는 Linear (layer≥1 은 input/output 모두 hidden*heads
        # 라 projection 불필요 → identity add). initial_residual 의 res_proj 와 역할은 같으나
        # initial_residual_alpha 와 독립이므로 (V6-W5 cells 는 IR off) 별도 모듈로 둔다.
        if self.v6w5_per_layer_residual:
            self.v6w5_res_proj = nn.ModuleDict(
                {nt: Linear(hidden_channels, hidden_channels * heads) for nt in node_types}
            )
        else:
            self.v6w5_res_proj = None

        # 7b. V5-C — Node-Adaptive Hop Attention (Lee 2023 AERO-GNN).
        # 각 layer l 의 H^{(l)}_v 에 대해 node-adaptive scalar score β^{(l)}_v 산출:
        #   β^{(l)}_v = w_hop^T H^{(l)}_v          (shape [N, 1])
        # Softplus + per-node normalization 으로 γ^{(l)}_v ∈ [0, 1] 산출, Σ_l γ^{(l)}_v = 1.
        # Output: H^{out}_v = Σ_l γ^{(l)}_v · H^{(l)}_v
        # h_0 의 차원 (hidden_channels) 과 layer output 의 차원 (hidden_channels × heads) 이 다르므로
        # h_0 는 res_proj (이미 존재 시) 또는 별도 projection 으로 차원 맞춰서 hop sum 에 포함.
        if self.aero_hop_attention:
            self.hop_attention_lin = nn.ModuleDict(
                {nt: Linear(hidden_channels * heads, 1) for nt in node_types}
            )
            # h_0 projection: hidden_channels → hidden_channels*heads. res_proj 와 동일 역할이지만
            # initial_residual_alpha=0 일 때도 항상 필요하므로 별도 모듈로 둔다 (학습 weight 공유 안 함).
            self.hop_h0_proj = nn.ModuleDict(
                {nt: Linear(hidden_channels, hidden_channels * heads) for nt in node_types}
            )
            # Output projection: hop-aggregated H^{out} (hidden*heads) → out_channels
            self.hop_out_lin = nn.ModuleDict(
                {nt: Linear(hidden_channels * heads, out_channels) for nt in node_types}
            )
        else:
            self.hop_attention_lin = None
            self.hop_h0_proj = None
            self.hop_out_lin = None

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

        # V-3 / V-3-ext: SuperNode threshold filtering (pre-GAT, per-query cosine).
        if self.query_supernode and query_emb is not None and self._supernode_filter_active():
            mask = self._compute_supernode_mask(query_emb, x_dict)
            edge_index_dict = self._apply_topk_to_edges(edge_index_dict, mask)

        # V-2: directed_from_sn 시 SN self-loop edge 주입.
        if (self.query_supernode
                and self.supernode_edge_direction == "directed_from_sn"
                and "query_node" in x_dict):
            edge_index_dict = self._inject_sn_self_loop(edge_index_dict, x_dict)

        # V6-W5-a/c: column self-loop relation 의 identity edge (i→i) 주입. 컬럼 수만큼 self edge —
        # 단일 table 소스와 분리된 self-channel 로 컬럼 L0 보존 (collapse mechanism 직접 완화).
        if self.v6w5_column_self_loop and "column" in x_dict:
            n_col = x_dict["column"].size(0)
            if n_col > 0:
                idx = torch.arange(n_col, device=x_dict["column"].device, dtype=torch.long)
                edge_index_dict = dict(edge_index_dict)
                edge_index_dict[V6W5_COLUMN_SELF_LOOP_REL] = torch.stack([idx, idx], dim=0)

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

        # --- 3. GAT layers (+ PairNorm, + Initial Residual + V5-C Cumulative) ---
        # V-1: depth 만큼만 순회 (depth ≤ num_layers_max). JK concat 은 num_layers_max 만큼의
        # slot 이 필요하므로 부족한 depth 는 마지막 layer 출력을 패딩.
        layer_outputs = []  # JK 용. 각 원소는 {nt: tensor}
        prev_out: Optional[dict] = None  # V5-C Cumulative Attention 용 — 직전 layer 의 출력.
        # V6 analyzer capture init (capture_layerwise_outputs=True 시 매 forward 마다 재초기화).
        if self.capture_layerwise_outputs:
            self._captured_layer_outputs = []
            self._captured_layerwise_alpha = []
        for i in range(depth):
            conv_out = self.convs[i](out_dict, edge_index_dict)
            # ELU
            conv_out = {nt: F.elu(x) for nt, x in conv_out.items()}
            # V6-W5-b/c: per-layer residual — out = ELU(conv(h)) + h_prev. layer input (out_dict)
            # 을 더해 input differentiation 을 layer 별 보존 (skip_dict 는 L_out 에만 적용되므로
            # 중간 layer 의 collapse 를 막지 못함). layer 0 은 h_0 (hidden) → conv (hidden*heads)
            # 차원 불일치 → v6w5_res_proj projection. layer≥1 은 동차원 identity add.
            if self.v6w5_per_layer_residual:
                for nt in conv_out:
                    prev = out_dict.get(nt)
                    if prev is None:
                        continue
                    if prev.shape[-1] != conv_out[nt].shape[-1]:
                        prev = self.v6w5_res_proj[nt](prev)
                    conv_out[nt] = conv_out[nt] + prev
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
            # V5-C Cumulative Attention residual (Lee 2023 Theorem 3, simulated at hidden-state level):
            #   H^(l) ← H^(l) + λ_cum · H^(l-1)
            # paper 의 edge-level α^(l) = α^(l-1) + softplus(e^(l)) 누적의 outer simulation —
            # layer-별 별도 conv 인스턴스 구조에서 conv-level α 누적이 불가하므로 hidden state 누적으로
            # 대체. 동일 over-smoothing 차단 효과 (residual path 가 layer 간 정보 누적).
            if self.aero_cumulative_attention and prev_out is not None:
                cum_out = {}
                for nt, x in conv_out.items():
                    if nt in prev_out and prev_out[nt].shape == x.shape:
                        cum_out[nt] = x + self.aero_cumulative_decay * prev_out[nt]
                    else:
                        cum_out[nt] = x
                conv_out = cum_out
            prev_out = conv_out
            out_dict = conv_out
            layer_outputs.append(out_dict)
            # V6 analyzer capture — layer 별 detached hidden state + attention alpha snapshot.
            # V6-W1 4 classes (pairnorm/gcnii_ir/jk/dropin_combo) 의 _captured_alpha 도 함께 수집.
            if self.capture_layerwise_outputs:
                self._captured_layer_outputs.append(
                    {nt: t.detach() for nt, t in out_dict.items()}
                )
                # HeteroConv 의 inner conv 별 alpha snapshot (edge type → tensor).
                alpha_snapshot: Dict[tuple, torch.Tensor] = {}
                hetero_conv = self.convs[i]
                # PyG HeteroConv 는 .convs (ModuleDict) 위 edge_type → inner conv 보관.
                inner_convs = getattr(hetero_conv, "convs", None)
                if inner_convs is not None:
                    for et_key, inner_conv in inner_convs.items():
                        cap_a = getattr(inner_conv, "_captured_alpha", None)
                        if cap_a is not None:
                            alpha_snapshot[et_key] = cap_a
                self._captured_layerwise_alpha.append(alpha_snapshot)

        # V-1: depth < num_layers_max 일 때, JK concat dim 을 맞추기 위해 마지막 output 반복 패딩.
        # (max / none 은 depth 에 무관)
        if self.jumping_knowledge == "concat" and depth < self.num_layers_max:
            last = layer_outputs[-1] if layer_outputs else {}
            pad = self.num_layers_max - depth
            for _ in range(pad):
                layer_outputs.append(last)

        # --- 4. 출력 경로: V5-C Hop Attention vs JK vs 기본 ---
        # V5-C: aero_hop_attention=True 시, layer_outputs + h_0 의 node-adaptive weighted sum.
        # JK 는 무시 (aero_hop_attention 가 cross-layer aggregation 의 대체 경로).
        if self.aero_hop_attention:
            final = {}
            for nt in self.node_types:
                # 1) h_0 projection → hidden*heads. layer outputs 와 차원 통일.
                h0_proj = self.hop_h0_proj[nt](h0_dict[nt])
                # 2) layer outputs (depth 길이) + h_0 → stack [L+1, N_nt, hidden*heads]
                layer_tensors: List[torch.Tensor] = [h0_proj]
                for l in range(depth):
                    if nt in layer_outputs[l]:
                        layer_tensors.append(layer_outputs[l][nt])
                    else:
                        # 누락 layer 는 h_0 proj 로 패딩 (HeteroConv 에서 일부 node type 누락 시).
                        layer_tensors.append(h0_proj)
                stacked = torch.stack(layer_tensors, dim=0)  # [L+1, N, D]
                # 3) Hop attention score: β^{(l)}_v = w_hop^T H^{(l)}_v, shape [L+1, N]
                hop_scores = self.hop_attention_lin[nt](stacked).squeeze(-1)
                # 4) Softplus + per-node normalization (Σ_l γ_v = 1 — node-adaptive).
                #    AERO 원본은 softplus 정규화 (row-stochasticity 파괴된 layer-level 의 normalization).
                hop_pos = F.softplus(hop_scores)  # [L+1, N]
                hop_weights = hop_pos / hop_pos.sum(dim=0, keepdim=True).clamp(min=1e-12)  # [L+1, N]
                # 5) Weighted sum across hops: Σ_l γ^{(l)}_v · H^{(l)}_v
                hop_agg = (hop_weights.unsqueeze(-1) * stacked).sum(dim=0)  # [N, D]
                final[nt] = self.hop_out_lin[nt](hop_agg) + self.skip_dict[nt](x_dict[nt])
        elif self.jumping_knowledge == "concat":
            final = {}
            for nt in self.node_types:
                parts = [h0_dict[nt]]  # h_0 (hidden)
                # Defensive: HeteroConv 위 일부 node type (예 fk_node) 이 어떤 layer 에서도
                # message 를 받지 못해 layer_outputs[l] 위 missing 일 수 있음 — jk_lin 의 입력 dim
                # 보장 위 zero-tensor 로 패딩.
                _hxh = self.hidden_channels * self.heads
                _h0 = h0_dict[nt]
                for l in range(self.num_layers_max):
                    if nt in layer_outputs[l]:
                        parts.append(layer_outputs[l][nt])
                    else:
                        parts.append(torch.zeros(_h0.size(0), _hxh,
                                                  dtype=_h0.dtype, device=_h0.device))
                concat = torch.cat(parts, dim=-1)
                final[nt] = self.jk_lin[nt](concat) + self.skip_dict[nt](x_dict[nt])
        elif self.jumping_knowledge == "max":
            final = {}
            for nt in self.node_types:
                layer_tensors = [layer_outputs[l][nt] for l in range(depth)
                                 if nt in layer_outputs[l]]
                _h0 = h0_dict[nt]
                _hxh = self.hidden_channels * self.heads
                if not layer_tensors:
                    # Defensive: 모든 layer 위 nt missing — zero tensor 로 fallback (skip path 만 작동)
                    pooled = torch.zeros(_h0.size(0), _hxh,
                                          dtype=_h0.dtype, device=_h0.device)
                else:
                    stacked = torch.stack(layer_tensors, dim=0)
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
    # V6-W1 analyzer hooks (DECISIONS 2026-06-01)
    # ──────────────────────────────────────────────────────────────
    def get_captured_layer_outputs(self) -> Optional[List[Dict[str, torch.Tensor]]]:
        """Layer 별 hidden state snapshot 반환 (capture_layerwise_outputs=True 일 때).

        Returns: `[{nt: tensor [N_nt, hidden*heads]}, ...]` (length = depth) 또는 None.
        analyzer 의 Dirichlet energy / MAD 측정 위 — 직전 forward call 의 layer 별 H 보관.
        """
        return self._captured_layer_outputs

    def get_captured_layerwise_alpha(self) -> Optional[List[Dict[tuple, torch.Tensor]]]:
        """Layer 별 edge type 별 attention alpha snapshot 반환.

        Returns: `[{(src_nt, rel, dst_nt): alpha [E, heads]}, ...]` 또는 None.
        analyzer 의 attention entropy 측정 위 — 직전 forward call 의 layer 별 softmax 직후 α 보관.
        V6-W1 4 classes (pairnorm/gcnii_ir/jk/dropin_combo) 에서만 capture (V4/V5 는 capture
        대상 외 — separate axis).
        """
        return self._captured_layerwise_alpha

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

    def _supernode_filter_active(self) -> bool:
        if self.supernode_threshold_mode == "top_k":
            return self.supernode_topk is not None and self.supernode_topk > 0
        return self.supernode_threshold_value is not None

    def _compute_supernode_mask(
        self,
        query_emb: torch.Tensor,
        x_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """V-3 / V-3-ext: pre-GAT cosine score 기반 schema 노드 mask 산출.

        threshold_mode 별 dispatch:
          - top_k:        torch.topk on (norm 적용된) score 로 K 개.
          - percentile:   torch.quantile cutoff (value=80 → P80).
          - abs_tau:      score >= value (norm 후 절대 cutoff).

        score_normalization='minmax' 시 모든 mode 가 per-query [0,1] 정규화 후 비교. analyzer
        raw_score_distribution_for_directed_topk.md 의 정의와 동일.
        """
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
            sc = (n_norm @ q_norm.t()).squeeze(-1)
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

    # Backward-compat alias — 기존 호출자 (예: v1 분석 스크립트) 가 v2 모델의 동일 method 를
    # 호출할 때 깨지지 않도록 유지. 신규 코드는 _compute_supernode_mask 를 사용.
    def _compute_topk_mask(
        self,
        query_emb: torch.Tensor,
        x_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self._compute_supernode_mask(query_emb, x_dict)

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
