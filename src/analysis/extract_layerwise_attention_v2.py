"""extract_layerwise_attention_v2 — V-3-ext directed_from_sn 호환 attention extract.

학위 논문 Part III mechanism deep dive evidence (DSN attention 의 directed edge 영향 정량) 를
위한 base 작업. 기존 v1 (analysis/gat_bottleneck_analysis.py:extract_layerwise_attention) 가
HeteroConv 의 inner conv 를 manual 호출 → V-3-ext 의 supernode_threshold + directed_from_sn 의
self-loop injection 처리 미반영 (matrix shape mismatch 로 attention extract 실패, dsn_oversmoothing_analysis.md §3.3 / §7).

본 v2 는 **forward hook (monkey-patch wrap)** 기반:
  - 각 GATv2Conv (HeteroConv 내부 inner conv) 의 forward 를 wrap 하여 호출 시
    `return_attention_weights=True` 강제 + alpha capture.
  - HeteroConv 자체는 정상 forward → V-3-ext 의 모든 mask/edge filter/self-loop 가 그대로 적용.
  - Layer × edge_type 별 attention map 분리 추출 + entropy / top-K concentration metric 산출.

Outputs:
  - per-layer per-edge-type attention entropy (mean over dst nodes)
  - per-layer per-edge-type top-K=5 concentration (sum of top-5 alpha / sum of all alpha, dst-node 별)
  - raw alpha tensor 보존 (per-query × per-layer × per-edge-type, optional return)

근거:
  - planning/DECISIONS.md 2026-05-06 §1(B) (2)-A Attention 호환성 selector 위임 confirm
  - notebooks/analysis_results/dsn_oversmoothing_analysis.md §7 Caveat (attention entropy 미측정 한계)
  - src/modules/selectors/EXPERIMENT_PLAN_selectors.md §V-3-ext 단계 4 후속
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

EdgeType = Tuple[str, str, str]
EdgeAttn = Tuple[Tensor, Tensor]  # (edge_index, alpha)
LayerAttn = Dict[EdgeType, EdgeAttn]


# ──────────────────────────────────────────────────────────────────────
# Core: AttentionCapture (monkey-patch wrap GATv2Conv.forward)
# ──────────────────────────────────────────────────────────────────────

class AttentionCapture:
    """Context manager that wraps every GATv2Conv inside a SchemaHeteroGAT[v2] model
    so the next `model(...)` invocation captures attention weights.

    각 inner conv 의 forward 를 monkey-patch 로 wrap. 호출 시 `return_attention_weights=True`
    강제 + alpha 를 layer × edge_type 별로 layer_attentions 에 저장. HeteroConv 자체는 정상
    동작 (out 만 받음) → V-3-ext directed_from_sn / supernode_threshold mask / self-loop 모두
    그대로 적용된다.

    Usage:
        cap = AttentionCapture(model)
        with cap:
            with torch.no_grad():
                _ = model(x_dict, edge_index_dict, query_emb=q)
        # cap.layer_attentions: List[Dict[EdgeType, (edge_index, alpha)]]
        # len(cap.layer_attentions) == model.num_layers
    """

    def __init__(self, model: torch.nn.Module):
        if not hasattr(model, "convs") or not isinstance(model.convs, torch.nn.ModuleList):
            raise TypeError(
                "AttentionCapture expects model.convs to be a ModuleList of HeteroConv"
            )
        self.model = model
        self.layer_attentions: List[LayerAttn] = []
        self._restores: List[Tuple[torch.nn.Module, Any]] = []

    def __enter__(self) -> "AttentionCapture":
        # 매번 enter 시 초기화 — 동일 instance 를 여러 query 에 재사용할 때 reset 가능.
        self.layer_attentions = []
        self._restores = []

        for layer_idx, hetero_conv in enumerate(self.model.convs):
            layer_dict: LayerAttn = {}
            self.layer_attentions.append(layer_dict)
            inner_convs = getattr(hetero_conv, "convs", None)
            if inner_convs is None:
                continue
            for edge_type, conv in inner_convs.items():
                self._wrap_conv(conv, edge_type, layer_dict)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Restore: wrap 시 instance attribute 로 'forward' 를 등록했으므로 깔끔히 제거하면
        # class method (원본 GATv2Conv.forward) 로 자연 복귀. instance dict 에 잔존하지 않음.
        for conv, _original in self._restores:
            try:
                if "forward" in conv.__dict__:
                    del conv.__dict__["forward"]
            except Exception:
                # 안전망: instance attribute 가 아닌 경우 fallback (메소드 set 시도).
                conv.forward = _original  # type: ignore[assignment]
        self._restores.clear()

    def _wrap_conv(self, conv: torch.nn.Module, edge_type: EdgeType,
                   layer_dict: LayerAttn) -> None:
        # GATv2Conv 가 아닌 conv (예: 다른 message passing) 는 attention 미반환 가능 → silent skip.
        original_forward = conv.forward
        self._restores.append((conv, original_forward))

        def wrapped(*args, **kwargs):
            kwargs.pop("return_attention_weights", None)
            try:
                result = original_forward(*args, return_attention_weights=True, **kwargs)
            except TypeError:
                # Conv 가 return_attention_weights 를 지원하지 않음. fallback.
                return original_forward(*args, **kwargs)
            # GATv2Conv 가 return_attention_weights=True 시 (out, (edge_index, alpha)) 반환.
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], tuple):
                out, attn = result
                if isinstance(attn, tuple) and len(attn) == 2:
                    att_ei, alpha = attn
                    layer_dict[edge_type] = (
                        att_ei.detach().cpu(),
                        alpha.detach().cpu(),
                    )
                return out
            return result

        conv.forward = wrapped


# ──────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────

def _alpha_per_dst_node(att_ei: Tensor, alpha: Tensor) -> Dict[int, Tensor]:
    """Group alpha rows by dst node index. alpha may be [E] or [E, heads]. Returns
    a dict mapping dst node id → tensor of shape [in_degree] (heads-averaged)."""
    if alpha.numel() == 0 or att_ei.numel() == 0:
        return {}
    if alpha.dim() > 1:
        alpha_flat = alpha.mean(dim=-1)  # [E]
    else:
        alpha_flat = alpha
    dst = att_ei[1].tolist()
    grouped: Dict[int, List[float]] = {}
    for e_idx, d in enumerate(dst):
        grouped.setdefault(d, []).append(float(alpha_flat[e_idx].item()))
    return {d: torch.tensor(vals, dtype=torch.float32) for d, vals in grouped.items()}


def attention_entropy(att_ei: Tensor, alpha: Tensor, eps: float = 1e-12) -> float:
    """Mean over dst nodes of per-dst entropy `H(α) = -Σ p_i log p_i`.

    PyG GATv2Conv 의 alpha 는 dst-node 별로 정규화 (softmax over incoming edges) 되어 있으므로
    바로 entropy 계산. dst 노드 별 entropy 를 평균하여 layer × edge_type 단일 스칼라로 환원.
    in_degree=1 인 dst 는 entropy=0 (집중) — dst-node 평균에 자연 포함.
    """
    grouped = _alpha_per_dst_node(att_ei, alpha)
    if not grouped:
        return float("nan")
    entropies: List[float] = []
    for _, p in grouped.items():
        # 이미 dst-별로 softmax 정규화되어 있다고 가정. 안전을 위해 재정규화.
        s = p.sum()
        if s.item() <= eps:
            continue
        p_norm = p / s
        ent = -(p_norm * (p_norm + eps).log()).sum().item()
        entropies.append(ent)
    return float(np.mean(entropies)) if entropies else float("nan")


def topk_concentration(att_ei: Tensor, alpha: Tensor, k: int = 5,
                       eps: float = 1e-12) -> float:
    """Mean over dst nodes of (sum of top-k alpha) / (sum of all alpha).

    1.0 = 모든 attention 이 top-k 노드에 집중 (in_degree ≤ k 인 경우 자동 1.0).
    낮을수록 distribution 균일. `k` 가 in_degree 보다 크면 자동 1.0.
    """
    grouped = _alpha_per_dst_node(att_ei, alpha)
    if not grouped:
        return float("nan")
    ratios: List[float] = []
    for _, p in grouped.items():
        s = p.sum().item()
        if s <= eps:
            continue
        kk = min(k, p.numel())
        topk_sum = torch.topk(p, k=kk).values.sum().item()
        ratios.append(topk_sum / s)
    return float(np.mean(ratios)) if ratios else float("nan")


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def extract_layerwise_attention_v2(
    model: torch.nn.Module,
    batch: Any,
    query_emb: Optional[Tensor] = None,
    *,
    topk: int = 5,
    return_raw: bool = False,
    forward_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """V-3-ext / directed_from_sn 호환 layer-wise attention extract.

    Args:
        model: SchemaHeteroGAT or SchemaHeteroGATv2 (eval mode 권장).
        batch: HeteroData (single graph) — `batch.x_dict`, `batch.edge_index_dict` 사용.
        query_emb: Optional query embedding ([d] or [1, d]). model.forward 에 직접 전달.
        topk: top-K concentration 의 K (default 5).
        return_raw: True 시 결과 dict 의 'raw' 키에 per-layer per-edge-type
            (edge_index, alpha) tensor 보존.
        forward_kwargs: model.forward 에 추가 전달할 kwargs (예: `active_num_layers`).

    Returns:
        {
            "num_layers": int,
            "entropy":   {f"L{i+1}": {edge_type_str: float, ...}, ...},
            "topk_conc": {f"L{i+1}": {edge_type_str: float, ...}, ...},
            "raw":       (optional) per-layer per-edge-type (att_ei, alpha) tensors
        }
        edge_type_str 은 "src→rel→dst" 형식 (plot 호환).
    """
    if forward_kwargs is None:
        forward_kwargs = {}

    cap = AttentionCapture(model)
    with cap:
        with torch.no_grad():
            # query_emb 처리는 model.forward 에 위임 — V-3-ext directed_from_sn / threshold mask
            # 모두 모델 측에서 자동 적용.
            _ = model(batch.x_dict, batch.edge_index_dict, query_emb=query_emb,
                      **forward_kwargs)

    num_layers = len(cap.layer_attentions)
    entropy: Dict[str, Dict[str, float]] = {}
    topk_conc: Dict[str, Dict[str, float]] = {}

    for layer_idx, layer_dict in enumerate(cap.layer_attentions):
        layer_key = f"L{layer_idx + 1}"
        ent_per_et: Dict[str, float] = {}
        top_per_et: Dict[str, float] = {}
        for edge_type, (att_ei, alpha) in layer_dict.items():
            et_str = "→".join(edge_type)
            ent_per_et[et_str] = attention_entropy(att_ei, alpha)
            top_per_et[et_str] = topk_concentration(att_ei, alpha, k=topk)
        entropy[layer_key] = ent_per_et
        topk_conc[layer_key] = top_per_et

    out: Dict[str, Any] = {
        "num_layers": num_layers,
        "entropy": entropy,
        "topk_conc": topk_conc,
    }
    if return_raw:
        out["raw"] = cap.layer_attentions
    return out


# ──────────────────────────────────────────────────────────────────────
# Multi-query aggregation helper (analyzer-friendly)
# ──────────────────────────────────────────────────────────────────────

def aggregate_attention_metrics(
    per_query_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Aggregate output of `extract_layerwise_attention_v2` over many queries.

    각 query 의 결과를 layer × edge_type 별로 모아 mean / std 산출.
    Returns: {
        "num_queries": int,
        "num_layers":  int,
        "entropy_mean":   {Lx: {et: float}}, "entropy_std":   {...},
        "topk_conc_mean": {Lx: {et: float}}, "topk_conc_std": {...},
    }
    """
    if not per_query_results:
        return {"num_queries": 0, "num_layers": 0,
                "entropy_mean": {}, "entropy_std": {},
                "topk_conc_mean": {}, "topk_conc_std": {}}

    n = len(per_query_results)
    num_layers = max(r.get("num_layers", 0) for r in per_query_results)

    def _collect(metric_key: str) -> Dict[str, Dict[str, List[float]]]:
        coll: Dict[str, Dict[str, List[float]]] = {f"L{i+1}": {} for i in range(num_layers)}
        for r in per_query_results:
            metric = r.get(metric_key, {})
            for layer_key, et_map in metric.items():
                bucket = coll.setdefault(layer_key, {})
                for et, v in et_map.items():
                    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                        continue
                    bucket.setdefault(et, []).append(float(v))
        return coll

    def _summarize(coll: Dict[str, Dict[str, List[float]]], op: str
                   ) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        fn = np.mean if op == "mean" else np.std
        for layer_key, bucket in coll.items():
            row: Dict[str, float] = {}
            for et, vals in bucket.items():
                row[et] = float(fn(vals)) if vals else float("nan")
            out[layer_key] = row
        return out

    ent_coll = _collect("entropy")
    top_coll = _collect("topk_conc")
    return {
        "num_queries": n,
        "num_layers": num_layers,
        "entropy_mean": _summarize(ent_coll, "mean"),
        "entropy_std": _summarize(ent_coll, "std"),
        "topk_conc_mean": _summarize(top_coll, "mean"),
        "topk_conc_std": _summarize(top_coll, "std"),
    }


# ──────────────────────────────────────────────────────────────────────
# Plot helpers (optional, matplotlib)
# ──────────────────────────────────────────────────────────────────────

def plot_attention_entropy_heatmap(agg: Dict[str, Any], out_path: str,
                                   title: Optional[str] = None) -> str:
    """Per-layer × per-edge-type heatmap (rows=edge_type, cols=layer)."""
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    ent = agg.get("entropy_mean", {})
    if not ent:
        return out_path
    layers = sorted(ent.keys(), key=lambda k: int(k.lstrip("L")))
    edge_types = sorted({et for layer in ent.values() for et in layer.keys()})
    if not edge_types:
        return out_path

    mat = np.full((len(edge_types), len(layers)), np.nan, dtype=float)
    for j, lk in enumerate(layers):
        row_map = ent[lk]
        for i, et in enumerate(edge_types):
            mat[i, j] = row_map.get(et, np.nan)

    fig_h = max(3.0, 0.4 * len(edge_types) + 1)
    fig, ax = plt.subplots(figsize=(max(5, len(layers) * 1.2), fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(edge_types)))
    ax.set_yticklabels(edge_types, fontsize=8)
    ax.set_title(title or "Attention entropy (mean over dst nodes, mean over queries)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(edge_types)):
        for j in range(len(layers)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7,
                        color=("white" if v < (np.nanmax(mat) * 0.55) else "black"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_topk_concentration_heatmap(agg: Dict[str, Any], out_path: str,
                                    title: Optional[str] = None) -> str:
    """Per-layer × per-edge-type heatmap of top-K concentration."""
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    top = agg.get("topk_conc_mean", {})
    if not top:
        return out_path
    layers = sorted(top.keys(), key=lambda k: int(k.lstrip("L")))
    edge_types = sorted({et for layer in top.values() for et in layer.keys()})
    if not edge_types:
        return out_path

    mat = np.full((len(edge_types), len(layers)), np.nan, dtype=float)
    for j, lk in enumerate(layers):
        for i, et in enumerate(edge_types):
            mat[i, j] = top[lk].get(et, np.nan)

    fig_h = max(3.0, 0.4 * len(edge_types) + 1)
    fig, ax = plt.subplots(figsize=(max(5, len(layers) * 1.2), fig_h))
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(range(len(edge_types)))
    ax.set_yticklabels(edge_types, fontsize=8)
    ax.set_title(title or "Top-K attention concentration (mean over dst, mean over queries)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(edge_types)):
        for j in range(len(layers)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v < 0.6 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
