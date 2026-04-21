"""
s06 ablation용 loss 함수 모음.

  - ListNetLoss: listwise ranking loss (Cao et al., ICML 2007)
      L = -sum_i softmax(y)_i * log(softmax(z)_i)
      per-query 로 계산. binary label을 soft target으로 사용.

  - AntiCollapseRegularizer: schema-aware over-smoothing 방지 regularizer (본 연구 기여)
      같은 table 에 속한 column 쌍들의 cosine similarity가 τ_max 를 초과하면 penalty.
      L_ac = sum_(i,j ∈ same_table, i<j) max(0, cos(h_i, h_j) - τ_max)^2
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ListNetLoss(nn.Module):
    """ListNet (Cao et al., ICML 2007) — per-query listwise ranking loss.

    binary gold label y ∈ {0, 1}^N 를 soft target 으로 변환해 score 분포와 KL.

    Args:
        eps: softmax stability
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,  # [N]
        labels: torch.Tensor,  # [N] (binary)
    ) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.sum() * 0.0
        y = labels.float()
        if y.sum() <= 0:
            # positive가 전혀 없으면 ranking 대상 없음 — 0 반환
            return logits.sum() * 0.0
        # soft target: positive에만 질량을 균등 분배
        target = y / (y.sum() + self.eps)
        # score 분포
        log_p = F.log_softmax(logits, dim=-1)
        # cross-entropy
        loss = -(target * log_p).sum()
        return loss


class AntiCollapseRegularizer(nn.Module):
    """Schema-aware anti-collapse regularizer.

    같은 table에 속한 column 쌍의 cosine similarity 가 τ_max 를 초과하면 squared hinge penalty.

    Args:
        tau_max: allowed max cosine similarity (default 0.85 — over-smoothing critical)
        reduction: 'mean' | 'sum'
    """

    def __init__(self, tau_max: float = 0.85, reduction: str = "mean"):
        super().__init__()
        self.tau_max = tau_max
        self.reduction = reduction

    def forward(
        self,
        col_embs: torch.Tensor,  # [N_col, D]
        col_to_table_edge_index: torch.Tensor,  # [2, E] (col→table)
    ) -> torch.Tensor:
        if col_embs.size(0) < 2 or col_to_table_edge_index.numel() == 0:
            return col_embs.sum() * 0.0

        col_ids = col_to_table_edge_index[0]
        tab_ids = col_to_table_edge_index[1]

        # table별 column 그룹화 (python-level, gradient는 embedding 에만 흐름)
        unique_tabs = tab_ids.unique().tolist()
        normed = F.normalize(col_embs, dim=-1)

        penalties = []
        for t in unique_tabs:
            mask = (tab_ids == t)
            cols = col_ids[mask]
            if cols.numel() < 2:
                continue
            embs = normed[cols]  # [k, D]
            sim = embs @ embs.T  # [k, k]
            k = cols.numel()
            # 상삼각 indices (i < j)
            iu = torch.triu_indices(k, k, offset=1, device=sim.device)
            pairs = sim[iu[0], iu[1]]
            excess = F.relu(pairs - self.tau_max)
            penalties.append((excess ** 2).sum())

        if not penalties:
            return col_embs.sum() * 0.0

        total = torch.stack(penalties).sum()
        if self.reduction == "mean":
            # 평균은 쌍(pair) 단위로 — 쌍 수는 table 마다 다르지만 단순화
            total = total / max(len(penalties), 1)
        return total


def combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str = "bce",
    pos_weight: Optional[torch.Tensor] = None,
    listnet_fn: Optional[ListNetLoss] = None,
) -> torch.Tensor:
    """loss_type dispatcher.

    Args:
        loss_type: 'bce' | 'listnet' | 'bce_listnet'
    """
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)
    elif loss_type == "listnet":
        return listnet_fn(logits, labels)
    elif loss_type == "bce_listnet":
        bce = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)
        listnet = listnet_fn(logits, labels)
        return 0.5 * bce + 0.5 * listnet
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
