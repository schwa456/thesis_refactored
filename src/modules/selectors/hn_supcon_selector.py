"""Direction B — Hard Negative Supervised Contrastive (HN-SupCon) Selector.

Per DECISIONS 2026-05-14 (학술 Agent Phase 5 Response §4 Direction B HN-SupCon Spec).
Piao et al. 2025 LitE-SQL 의 HN-SupCon 목적함수를 anchor framework 의 EnsembleSelector
backbone 에 적용. 학술 frame = "Filter-Invariant 경계 확정 실험" (학술 Agent Phase 5 §0).

Backbone choice (학술 Agent Phase 5 §4.5 본 도메인 transfer 권장):
  - **Primary**: 현 anchor embedding model (`sentence-transformers/all-MiniLM-L6-v2`) 위에
    HN-SupCon 목적함수 적용 — 구현 속도 우세
  - **Fallback** (smoke fail 시): Qwen3-0.6B-Embedding backbone 교체 — 원문 재현 충실

Anchor framework 정합:
  - Direction B 의 fine-tuned encoder 가 anchor 의 EnsembleSelector 의 encoder 만 교체
  - α blending + top_k + GAT projector 등 기존 logic 그대로 유지
  - Filter (XiYanFilter GLM 4.7) / Extractor (MSTKruskalExtractor) / SQL Gen anchor 정합

Hard Negative Mining (Static, Phase 5 §4.1):
  m_ij = 1[s(q_i, n_ij) > s(q_i, p_i) - 0.1]
  → positive 대비 cosine similarity 0.1 이내 non-gold column 만 hard negative,
    나머지는 학습에서 제외. Static (pre-computed embeddings 기반), Dynamic mining 없음.

Loss (Phase 5 §4.2, NT-Xent multi-positive):
  L = log[exp(s(q_i, p_i)/τ) / (exp(s(q_i, p_i)/τ) + Σ_j m_ij × exp(s(q_i, n_ij)/τ))]
  각 gold column 이 별도 anchor 로 사용.

Reference:
  Piao et al. 2025. LitE-SQL: Lightweight Embedding for Text-to-SQL with Hard Negative
  Supervised Contrastive Learning. (학술 Agent Phase 5 §4 원문 확정값)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from modules.encoders.local_encoder import LocalPLMEncoder
from modules.encoders.token_encoder import TokenEncoder
from modules.registry import register
from modules.selectors.ensemble_selector import EnsembleSelector
from utils.logger import get_logger

logger = get_logger(__name__)


@register("selector", "HNSupConSelector")
class HNSupConSelector(EnsembleSelector):
    """Direction B — anchor EnsembleSelector 의 encoder 만 HN-SupCon fine-tuned backbone 으로 교체.

    EnsembleSelector 의 모든 logic (α blending + top_k + GAT projector 호출 +
    raw_gat_scores / raw_cos_scores 노출) 그대로 유지. 차이는 `self.encoder` 인스턴스가
    HN-SupCon fine-tuned ckpt 를 load 한다는 것뿐.

    Args:
        hn_supcon_ckpt_path: Fine-tuned SentenceTransformer checkpoint 경로 (디렉토리 또는 HF hub id).
                             None 또는 미존재 path 시 anchor backbone (`sentence-transformers/all-MiniLM-L6-v2`)
                             fallback — `b06` config 가 smoke 전에도 valid 하도록.
        **kwargs: EnsembleSelector 와 동일 (weight_path, alpha, top_k, query_conditioned, encoder_type, ...)

    Backwards compat: weight_path (GAT ckpt) 는 anchor 와 동일 `best_gat_qcond_nl3.pt` 권장.
                      encoder backbone 만 교체.
    """

    def __init__(
        self,
        hn_supcon_ckpt_path: Optional[str] = None,
        encoder_type: str = "plm",
        **kwargs,
    ):
        # EnsembleSelector 의 __init__ 가 self.encoder 를 LocalPLMEncoder() / TokenEncoder() 로 초기화하므로,
        # super().__init__ 를 먼저 호출한 뒤 self.encoder 를 fine-tuned backbone 으로 교체.
        super().__init__(encoder_type=encoder_type, **kwargs)

        self.hn_supcon_ckpt_path = hn_supcon_ckpt_path
        if hn_supcon_ckpt_path is not None and os.path.exists(hn_supcon_ckpt_path):
            logger.info(
                f"[HNSupConSelector] Loading HN-SupCon fine-tuned encoder from {hn_supcon_ckpt_path}"
            )
            if encoder_type == "plm":
                self.encoder = LocalPLMEncoder(model_name=hn_supcon_ckpt_path)
            else:
                self.encoder = TokenEncoder(model_name=hn_supcon_ckpt_path)
        else:
            if hn_supcon_ckpt_path is None:
                logger.warning(
                    "[HNSupConSelector] hn_supcon_ckpt_path is None — using anchor backbone "
                    "(no HN-SupCon effect). Set hn_supcon_ckpt_path after running src/train_hn_supcon.py."
                )
            else:
                logger.warning(
                    f"[HNSupConSelector] hn_supcon_ckpt_path={hn_supcon_ckpt_path} 미존재 — "
                    f"anchor backbone fallback. Run src/train_hn_supcon.py 후 재실행."
                )
        logger.info(
            f"Initialized HNSupConSelector (hn_supcon_ckpt={hn_supcon_ckpt_path}, alpha={self.alpha}, "
            f"top_k={self.top_k})"
        )


# ──────────────────────────────────────────────────────────────────────
# HN-SupCon Loss + Hard Negative Mining utility (training-time, no torch grad sphere change).
# 학습 entry (src/train_hn_supcon.py) 가 본 utility 를 사용.
# ──────────────────────────────────────────────────────────────────────


def build_hard_negative_mask(
    sim_qp: torch.Tensor,  # [B, P_max] — query-positive cosine
    sim_qn: torch.Tensor,  # [B, N] — query-negative cosine
    margin: float = 0.1,
) -> torch.Tensor:
    """학술 Agent Phase 5 §4.1: m_ij = 1[s(q_i, n_ij) > s(q_i, p_i) - margin].

    Each query i 에 대해 positive cosine similarity 의 max 와 비교 (multi-positive 일 때).

    Args:
        sim_qp: query-positive cosine similarity [B, P_max]. P_max = max num positives in batch.
                Padded entries 는 -inf 로 처리됨 (max 무영향).
        sim_qn: query-negative cosine similarity [B, N]. N = negative pool size.
        margin: positive 대비 cosine similarity gap threshold. 0.1 (학술 Agent §4.3 default).

    Returns:
        mask: bool tensor [B, N] — True if negative is "hard" (≥ p_max - margin).
    """
    if sim_qp.numel() == 0:
        return torch.ones_like(sim_qn, dtype=torch.bool)
    # Per-query positive cosine sim 의 max (multi-positive 시 가장 강한 positive)
    p_max = sim_qp.max(dim=1, keepdim=True).values  # [B, 1]
    mask = sim_qn > (p_max - float(margin))
    return mask


def hn_supcon_loss(
    sim_qp: torch.Tensor,  # [B, P_max]
    sim_qn: torch.Tensor,  # [B, N]
    pos_mask: torch.Tensor,  # [B, P_max] bool — True at valid positives
    hard_neg_mask: torch.Tensor,  # [B, N] bool — True at hard negatives
    tau: float = 0.07,
    n_per_query: int = 8,
) -> torch.Tensor:
    """학술 Agent Phase 5 §4.2: NT-Xent (InfoNCE with margin masking), multi-positive.

    L = -log[exp(s(q, p)/τ) / (exp(s(q, p)/τ) + Σ_j m_ij × exp(s(q, n_ij)/τ))]

    Multi-positive: 각 gold column 이 별도 anchor 로 사용. 한 query 안에서 valid positives 들은
    평균 NT-Xent 로 처리 (per-positive log-loss 의 mean).

    Args:
        sim_qp: query-positive cosine [B, P_max] (already cosine, range [-1, 1])
        sim_qn: query-negative cosine [B, N]
        pos_mask: valid-positive mask [B, P_max]
        hard_neg_mask: hard-negative mask [B, N] (from `build_hard_negative_mask`)
        tau: temperature (학술 Agent §4.3 default 0.07)
        n_per_query: max hard negatives per query (학술 Agent §4.3 default 8). 더 많으면 top-N truncate.

    Returns:
        scalar loss
    """
    B = sim_qp.size(0)
    if B == 0:
        return sim_qp.new_zeros(())

    # Hard negative truncation — N_i ≤ n_per_query (학술 Agent §4.3 default 8)
    # mask out non-hard negatives by setting their sim to -inf (exp = 0)
    neg_sim = sim_qn.masked_fill(~hard_neg_mask, float("-inf"))  # [B, N]
    # truncate to top-n_per_query per row (학술 Agent §3.4 static N_i=8 selection)
    if neg_sim.size(1) > n_per_query:
        top_vals, _ = neg_sim.topk(n_per_query, dim=1)
    else:
        top_vals = neg_sim
    # neg log-sum-exp denominator term (over hard negatives), zero if all negatives masked.
    neg_logsumexp = torch.logsumexp(top_vals / tau, dim=1)  # [B], -inf 가 있으면 잘 처리됨

    # Per-positive NT-Xent loss — exp(s(q,p)/τ) vs (exp(s(q,p)/τ) + Σ exp(s(q,n)/τ))
    # Stable: -log(exp(a)/(exp(a)+exp(b))) = log(1 + exp(b-a))
    losses = []
    for b in range(B):
        valid_p = pos_mask[b]
        if not valid_p.any():
            continue
        p_sims = sim_qp[b][valid_p] / tau  # [P_b]
        nb = neg_logsumexp[b]
        # per-positive loss: log(exp(p_sim) + exp(neg_logsumexp)) - p_sim
        # if all negatives masked out (nb = -inf), -inf 회피 위해 fallback 0 loss
        if torch.isinf(nb) and nb < 0:
            continue
        # logsumexp 안정 form
        combined = torch.stack([p_sims, nb.expand_as(p_sims)], dim=0)  # [2, P_b]
        denom = torch.logsumexp(combined, dim=0)  # [P_b]
        per_pos = denom - p_sims  # [P_b]
        losses.append(per_pos.mean())
    if not losses:
        return sim_qp.new_zeros(())
    return torch.stack(losses).mean()
