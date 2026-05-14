"""Relation-aware Graph Transformer Encoder for Direction C-GT (Hoang 2025).

학술 Agent Phase 5 Q1 (planning/filter_proposal_by_scholar_agent_phase5_2026-05-14.md §1.1)
의 원문 확정값 구현:

  - Layers     : 3 (BIRD 최적, 0~4 grid search)
  - Hidden dim : 1024 (PR-AUC 최적, default)
  - Heads      : 미명시 — 본 구현 default 8
  - Edge types : R = {foreign_key, column→FK, column→PK} directed + reverse
                 → 6 distinct relation channels
  - PE         : Relation-specific attention coefficient ψ^(ℓ)(i,j) — 표준 RPE 아님,
                 타입별 학습 가능 bias (layer-별 + head-별)
  - Loss       : margin-based contrastive (gold > non-gold), lr 5e-5, 40 epochs, batch 32

학술 frame (학술 Agent Phase 5 §0): "Filter-Invariant 경계 확정 실험" — Direction A/C 의
F1 -0.28 + EX sub-noise (±0.003) 결과 위에 GT 의 query-aware encoding 이 R-P trade-off
mitigation 의 유일 candidate. positive/null 결과 모두 학술적 가치.

본 모듈은 architecture + training utility 까지. 실제 BIRD-Train fine-tune 은 별도 Root
chain 책임 (학술 Agent Q5 protocol: 5 epochs / 12.5% smoke + early stop + fallback).
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# 6 edge types (학술 Agent §1.1: R = {fk, col→fk, col→pk} × {forward, reverse})
EDGE_TYPE_FK_FORWARD = 0
EDGE_TYPE_FK_REVERSE = 1
EDGE_TYPE_COL_TO_FK_FORWARD = 2
EDGE_TYPE_COL_TO_FK_REVERSE = 3
EDGE_TYPE_COL_TO_PK_FORWARD = 4
EDGE_TYPE_COL_TO_PK_REVERSE = 5
NUM_EDGE_TYPES = 6


class RelationAwareGTLayer(nn.Module):
    """Single Relation-aware Graph Transformer layer (Hoang 2025 §3.3).

    Standard multi-head attention 의 score 에 relation-specific bias ψ^(ℓ)(i,j) 를
    추가한 형태. ψ 는 layer 별 + head 별 + edge_type 별 학습 가능 scalar.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_edge_types: int = NUM_EDGE_TYPES,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        # ψ^(ℓ)(i,j) = relation_bias[edge_type, head]
        self.relation_bias = nn.Parameter(
            torch.zeros(num_edge_types, num_heads)
        )
        nn.init.normal_(self.relation_bias, std=0.02)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_mult, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,            # [N, H]
        edge_index: torch.Tensor,   # [2, E] long
        edge_type: torch.Tensor,    # [E] long, values in [0, NUM_EDGE_TYPES)
    ) -> torch.Tensor:
        """Sparse relation-aware attention over edges.

        x[N,H], edges (src,dst) with edge_type[E]. Returns updated x[N,H].
        Each dst node aggregates from src nodes connected by any edge.
        """
        N, H = x.shape
        device = x.device
        residual = x
        x_norm = self.norm1(x)

        q = self.W_q(x_norm).view(N, self.num_heads, self.head_dim)
        k = self.W_k(x_norm).view(N, self.num_heads, self.head_dim)
        v = self.W_v(x_norm).view(N, self.num_heads, self.head_dim)

        if edge_index.numel() == 0:
            # No edges — pass through (with self-attention degenerate to identity via residual)
            return residual + self.ff(self.norm2(residual))

        src = edge_index[0]
        dst = edge_index[1]

        # Compute attention logits per edge: q_dst · k_src * scale + ψ[edge_type]
        # shapes: q_dst[E, H, D], k_src[E, H, D], logits[E, H]
        q_dst = q.index_select(0, dst)
        k_src = k.index_select(0, src)
        v_src = v.index_select(0, src)

        logits = (q_dst * k_src).sum(dim=-1) * self.scale  # [E, H]
        # Relation bias: gather by edge_type
        psi = self.relation_bias.index_select(0, edge_type)  # [E, H]
        logits = logits + psi

        # Softmax per (dst, head). 안정성 위해 numerically stable max-subtraction.
        # dst 별 max 계산.
        # scatter_max approximation via two passes (without torch_scatter dep)
        max_logits = torch.full(
            (N, self.num_heads), float("-inf"), device=device, dtype=logits.dtype
        )
        max_logits.scatter_reduce_(0, dst.unsqueeze(1).expand_as(logits), logits,
                                    reduce="amax", include_self=False)
        # node 가 incoming edge 가 없는 경우 max_logits 이 -inf 유지 — 0 으로 대체.
        max_logits = torch.where(
            torch.isinf(max_logits),
            torch.zeros_like(max_logits),
            max_logits,
        )

        exp_logits = torch.exp(logits - max_logits.index_select(0, dst))  # [E, H]
        denom = torch.zeros((N, self.num_heads), device=device, dtype=logits.dtype)
        denom.index_add_(0, dst, exp_logits)
        denom_safe = denom.clamp_min(1e-9)
        alpha = exp_logits / denom_safe.index_select(0, dst)  # [E, H]

        # Weighted sum of values: out[dst] += alpha * v_src
        weighted = v_src * alpha.unsqueeze(-1)  # [E, H, D]
        out = torch.zeros((N, self.num_heads, self.head_dim),
                          device=device, dtype=v.dtype)
        out.index_add_(0, dst, weighted)

        out = out.reshape(N, H)
        out = self.W_o(out)
        x = residual + self.dropout(out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class GraphTransformerEncoder(nn.Module):
    """Relation-aware Graph Transformer Encoder + final column scorer.

    학술 Agent §1.1 + §1.2 spec:
      - Two-step decoupled (h^0 frozen input from LLM-Reranker / 본 도메인 = anchor scorer)
      - GT internal cross-attention 없음 (query 신호는 h^0 에서 pre-computed)
      - Final layer: per-column relevance score (1-d output)

    Args:
        in_dim: h^0 input dimension (anchor scorer 출력 dimension)
        hidden_dim: GT hidden dim (default 1024 = PR-AUC optimum)
        num_layers: 3 (BIRD 최적)
        num_heads: 8 (default; 원문 미명시)
        dropout: 0.1
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        num_heads: int = 8,
        num_edge_types: int = NUM_EDGE_TYPES,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            RelationAwareGTLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_edge_types=num_edge_types,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        # per-column relevance score (1-d)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        h0: torch.Tensor,           # [N, in_dim] — frozen anchor scorer embedding
        edge_index: torch.Tensor,   # [2, E] long
        edge_type: torch.Tensor,    # [E] long
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (refined_node_repr[N, hidden_dim], column_scores[N]).

        Column score 는 final layer 의 linear + sigmoid 출력 (margin loss 입력).
        """
        if h0.dim() != 2 or h0.size(1) != self.in_dim:
            raise ValueError(
                f"h0 shape mismatch: expected [N, {self.in_dim}], got {tuple(h0.shape)}"
            )
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index shape mismatch: expected [2, E], got {tuple(edge_index.shape)}"
            )
        if edge_type.dim() != 1 or edge_type.size(0) != edge_index.size(1):
            raise ValueError(
                f"edge_type shape mismatch: expected [{edge_index.size(1)}], "
                f"got {tuple(edge_type.shape)}"
            )

        x = self.input_proj(h0)
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
        x = self.norm_out(x)
        scores = self.score_head(x).squeeze(-1)  # [N]
        return x, scores


# ----------------------------------------------------------------------
# Training utility — margin-based contrastive loss (Hoang 2025 §1.3)
# ----------------------------------------------------------------------
def margin_contrastive_loss(
    scores: torch.Tensor,       # [N]
    gold_mask: torch.Tensor,    # [N] bool — True = gold column
    margin: float = 0.1,
) -> torch.Tensor:
    """Pairwise margin loss: every gold should score above every non-gold by at least `margin`.

    학술 Agent §1.3: "gold SQL columns ranked above irrelevant" margin-based contrastive.
    """
    if scores.dim() != 1:
        scores = scores.view(-1)
    gold_mask = gold_mask.view(-1).bool()
    pos = scores[gold_mask]
    neg = scores[~gold_mask]
    if pos.numel() == 0 or neg.numel() == 0:
        # No positive or no negative — well-defined loss is 0 (학습 step skip 가능)
        return scores.new_zeros((), requires_grad=True)
    # pairwise: hinge(margin - (pos - neg))
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)        # [P, N]
    loss = F.relu(margin - diff).mean()
    return loss


# ----------------------------------------------------------------------
# Smoke-train protocol (학술 Agent Q5)
# ----------------------------------------------------------------------
def smoke_train_protocol(
    model: GraphTransformerEncoder,
    batches: List[Dict[str, torch.Tensor]],
    val_batches: List[Dict[str, torch.Tensor]],
    *,
    num_epochs: int = 5,
    lr: float = 5e-5,
    margin: float = 0.1,
    pass_loss_threshold: float = 0.3,
    pass_pr_auc_delta: float = 0.01,
    plateau_patience: int = 2,
) -> Dict[str, object]:
    """학술 Agent Q5 smoke test protocol implementation.

    Pass 조건 (둘 다):
      - train margin loss < pass_loss_threshold (default 0.3)
      - val PR-AUC Δ ≥ pass_pr_auc_delta (default +0.01)
    Plateau (loss 가 patience epoch 동안 개선 없음) 시 early stop + fallback flag.

    batches / val_batches 각 element:
      {"h0": [N, in_dim], "edge_index": [2, E], "edge_type": [E],
       "gold_mask": [N] bool}

    Returns:
      {"passed": bool, "stopped_early": bool, "final_train_loss": float,
       "final_pr_auc": float, "initial_pr_auc": float,
       "epoch_losses": List[float]}
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    initial_pr_auc = _eval_pr_auc(model, val_batches)
    epoch_losses: List[float] = []
    best_loss = float("inf")
    plateau_count = 0
    stopped_early = False

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_steps = 0
        for batch in batches:
            optimizer.zero_grad()
            _, scores = model(batch["h0"], batch["edge_index"], batch["edge_type"])
            loss = margin_contrastive_loss(scores, batch["gold_mask"], margin=margin)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(
                    f"[GT smoke] NaN/Inf loss at epoch {epoch} — divergence, abort early."
                )
                stopped_early = True
                break
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_steps += 1
        avg_loss = epoch_loss / max(1, n_steps)
        epoch_losses.append(avg_loss)
        if stopped_early:
            break
        if avg_loss + 1e-6 < best_loss:
            best_loss = avg_loss
            plateau_count = 0
        else:
            plateau_count += 1
            if plateau_count >= plateau_patience:
                logger.info(
                    f"[GT smoke] Plateau at epoch {epoch} (loss {avg_loss:.4f}) — early stop."
                )
                stopped_early = True
                break

    final_pr_auc = _eval_pr_auc(model, val_batches)
    final_train_loss = epoch_losses[-1] if epoch_losses else float("inf")
    passed = (
        final_train_loss < pass_loss_threshold
        and (final_pr_auc - initial_pr_auc) >= pass_pr_auc_delta
    )
    return {
        "passed": bool(passed),
        "stopped_early": bool(stopped_early),
        "final_train_loss": float(final_train_loss),
        "final_pr_auc": float(final_pr_auc),
        "initial_pr_auc": float(initial_pr_auc),
        "epoch_losses": list(epoch_losses),
    }


def _eval_pr_auc(
    model: GraphTransformerEncoder, batches: List[Dict[str, torch.Tensor]],
) -> float:
    """Average PR-AUC over batches (per-batch precision-recall area-under-curve, simple).

    sklearn 의존 회피 — trapezoidal approximation 구현.
    """
    if not batches:
        return 0.0
    model.eval()
    aucs: List[float] = []
    with torch.no_grad():
        for batch in batches:
            _, scores = model(batch["h0"], batch["edge_index"], batch["edge_type"])
            aucs.append(_pr_auc(scores.detach().cpu(), batch["gold_mask"].detach().cpu()))
    return float(sum(aucs) / len(aucs))


def _pr_auc(scores: torch.Tensor, gold_mask: torch.Tensor) -> float:
    """Trapezoidal PR-AUC (no sklearn). gold_mask: bool [N]."""
    if scores.numel() == 0:
        return 0.0
    gold_mask = gold_mask.view(-1).bool()
    n_pos = int(gold_mask.sum().item())
    if n_pos == 0 or n_pos == gold_mask.numel():
        return 0.0
    order = torch.argsort(scores.view(-1), descending=True)
    sorted_gold = gold_mask[order]
    cumulative_tp = torch.cumsum(sorted_gold.float(), dim=0)
    ranks = torch.arange(1, sorted_gold.numel() + 1, dtype=torch.float32)
    precision = cumulative_tp / ranks
    recall = cumulative_tp / n_pos
    # Trapezoidal AUC over (recall, precision)
    # Prepend (0, precision[0]) so the curve starts at recall=0
    recall = torch.cat([torch.zeros(1), recall])
    precision = torch.cat([precision[:1], precision])
    dr = recall[1:] - recall[:-1]
    avg_p = 0.5 * (precision[1:] + precision[:-1])
    auc = float((dr * avg_p).sum().item())
    return auc
