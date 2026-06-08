"""Per-epoch Dirichlet energy + MAD logging for V6-W0/V6-W1 GAT training.

RFP §6.1 (Dirichlet energy) + §6.2 (MAD) — `src/analysis/v6_oversmoothing_diagnostics.py`
와 동일한 정의. 학습 중 매 epoch 마다 validation 배치 한 개 위 산출 → wandb 로깅
(`oversmoothing/energy/L{1..L}`, `oversmoothing/energy/mean`, 동등 `mad`).

지원 모델:
  - SchemaHeteroGAT (v1, train_gat.py) — forward_hook 로 layer 별 H 캡처
  - SchemaHeteroGATv2 (v2, train_gat_s06.py) — `capture_layerwise_outputs=True` API 사용

근거: 사용자 trigger 2026-06-02 "Dirichlet Energy 와 MAD 도 함께 logging, wandb 위 oversmoothing/energy + oversmoothing/mad".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


def dirichlet_energy(h: torch.Tensor, edge_index: torch.Tensor,
                     deg: Optional[torch.Tensor] = None,
                     eps: float = 1e-12) -> float:
    """RFP §6.1: $\\frac{1}{2}\\sum_{(i,j)\\in E}\\|h_i/\\sqrt{1+d_i}-h_j/\\sqrt{1+d_j}\\|^2$.

    edge_index 는 양방향 (caller 책임).
    """
    if h is None or h.numel() == 0 or edge_index.numel() == 0:
        return 0.0
    src, dst = edge_index[0], edge_index[1]
    if deg is None:
        N = h.size(0)
        deg = torch.zeros(N, device=h.device, dtype=h.dtype)
        deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=h.device, dtype=h.dtype))
        deg = deg + 1.0
    norm = (deg + eps).rsqrt()
    h_norm = h * norm.unsqueeze(-1)
    diff = h_norm[src] - h_norm[dst]
    return 0.5 * float((diff * diff).sum(dim=-1).sum().item())


def mad(h: torch.Tensor, mask_pairs: Optional[torch.Tensor] = None) -> float:
    """RFP §6.2: 평균 cosine 거리 (1 − sim). 0 에 가까울수록 over-smoothing.

    sample 노드 수 위 N×N 메모리 — 학습 logging 위 별도 sub-sample 권장.
    """
    if h is None or h.size(0) < 2:
        return float("nan")
    h_n = F.normalize(h, p=2, dim=-1)
    sim = h_n @ h_n.t()
    dist = 1.0 - sim
    if mask_pairs is not None:
        if not mask_pairs.any():
            return float("nan")
        return float(dist[mask_pairs].mean().item())
    N = h.size(0)
    offdiag = ~torch.eye(N, dtype=torch.bool, device=h.device)
    return float(dist[offdiag].mean().item())


def _build_global_edge_index(edge_index_dict, type_offsets: Dict[str, int],
                              device: str) -> torch.Tensor:
    """모든 edge_type 을 global node id 좌표계 (table → column → fk_node) 로 concat
    + 무방향 대칭화 (Dirichlet 정의)."""
    parts: List[torch.Tensor] = []
    for et, ei in edge_index_dict.items():
        if ei.numel() == 0:
            continue
        if isinstance(et, tuple) and len(et) == 3:
            src_t, _rel, dst_t = et
        else:
            continue
        src_off = type_offsets.get(src_t)
        dst_off = type_offsets.get(dst_t)
        if src_off is None or dst_off is None:
            continue
        gi = torch.stack([ei[0] + src_off, ei[1] + dst_off], dim=0)
        parts.append(gi.to(device))
    if not parts:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    ei_dir = torch.cat(parts, dim=1)
    return torch.cat([ei_dir, ei_dir.flip(0)], dim=1)


# ─────────────────────────────────────────────────────────────
# v1 (SchemaHeteroGAT) — forward_hook 기반 H 캡처
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_metrics_v1_gat(gat_model, x_dict: Dict[str, torch.Tensor],
                            edge_index_dict, query_emb: Optional[torch.Tensor],
                            device: str = "cuda") -> Dict[str, Any]:
    """SchemaHeteroGAT v1 위 forward_hook 로 H_l 캡처 → Dirichlet + MAD layer 별 산출.

    Returns: {'energy': {1: e1, 2: e2, ..., 'mean': em},
              'mad': {1: m1, 2: m2, ..., 'mean': mm}}
    """
    H_per_layer: List[Dict[str, torch.Tensor]] = []

    def make_hook():
        def hook(_module, _input, output):
            if isinstance(output, dict):
                H_per_layer.append({nt: F.elu(t).detach() for nt, t in output.items()})
        return hook

    handles = [conv.register_forward_hook(make_hook())
               for conv in getattr(gat_model, 'convs', [])]
    try:
        _ = gat_model(x_dict, edge_index_dict, query_emb=query_emb)
    finally:
        for h in handles:
            h.remove()

    return _metrics_from_H_list(H_per_layer, x_dict, edge_index_dict, device)


# ─────────────────────────────────────────────────────────────
# v2 (SchemaHeteroGATv2) — capture_layerwise_outputs API
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_metrics_v2_gat(gat_model, x_dict: Dict[str, torch.Tensor],
                            edge_index_dict, query_emb: Optional[torch.Tensor],
                            device: str = "cuda") -> Dict[str, Any]:
    """SchemaHeteroGATv2 위 capture_layerwise_outputs=True + get_captured_layer_outputs API.

    호출 전 gat_model.capture_layerwise_outputs=True 셋팅 권장 (caller 책임).
    """
    prev = getattr(gat_model, 'capture_layerwise_outputs', False)
    gat_model.capture_layerwise_outputs = True
    try:
        _ = gat_model(x_dict, edge_index_dict, query_emb=query_emb)
    finally:
        gat_model.capture_layerwise_outputs = prev

    # API: get_captured_layer_outputs() returns List[Dict[node_type → Tensor]]
    H_per_layer: List[Dict[str, torch.Tensor]] = []
    try:
        H_per_layer = gat_model.get_captured_layer_outputs() or []
    except Exception:
        H_per_layer = []
    return _metrics_from_H_list(H_per_layer, x_dict, edge_index_dict, device)


def _metrics_from_H_list(H_per_layer: List[Dict[str, torch.Tensor]],
                          x_dict: Dict[str, torch.Tensor],
                          edge_index_dict,
                          device: str) -> Dict[str, Any]:
    """H_per_layer (layer 별 dict[nt → tensor]) 위 Dirichlet + MAD layer 별 산출."""
    out_energy: Dict[Any, float] = {}
    out_mad: Dict[Any, float] = {}
    if not H_per_layer:
        return {'energy': {'mean': float('nan')}, 'mad': {'mean': float('nan')}}

    # Global node-type offset (table → column → fk_node 순)
    n_t = x_dict['table'].size(0) if 'table' in x_dict else 0
    n_c = x_dict['column'].size(0) if 'column' in x_dict else 0
    type_offsets = {'table': 0, 'column': n_t, 'fk_node': n_t + n_c}
    global_ei = _build_global_edge_index(edge_index_dict, type_offsets, device)

    energies: List[float] = []
    mads: List[float] = []
    for li, H_dict in enumerate(H_per_layer):
        Hs = [H_dict[nt] for nt in ['table', 'column', 'fk_node']
              if nt in H_dict and H_dict[nt].size(0) > 0]
        if not Hs:
            continue
        H_all = torch.cat(Hs, dim=0).to(device)
        e = dirichlet_energy(H_all, global_ei)
        m = mad(H_all)
        layer_idx = li + 1
        out_energy[layer_idx] = e
        out_mad[layer_idx] = m
        energies.append(e)
        if m == m:  # not NaN
            mads.append(m)

    out_energy['mean'] = float(sum(energies) / len(energies)) if energies else float('nan')
    out_mad['mean'] = float(sum(mads) / len(mads)) if mads else float('nan')
    return {'energy': out_energy, 'mad': out_mad}


def metrics_to_wandb_log(metrics: Dict[str, Any]) -> Dict[str, float]:
    """`oversmoothing/energy/L{1..L}` + `oversmoothing/energy/mean` 형식 wandb dict.

    사용:
        m = compute_metrics_v1_gat(...)
        wandb.log(metrics_to_wandb_log(m), step=...)
    """
    out: Dict[str, float] = {}
    for k, v in metrics.get('energy', {}).items():
        key = f"oversmoothing/energy/{'L' + str(k) if isinstance(k, int) else k}"
        out[key] = float(v) if v == v else 0.0
    for k, v in metrics.get('mad', {}).items():
        key = f"oversmoothing/mad/{'L' + str(k) if isinstance(k, int) else k}"
        out[key] = float(v) if v == v else 0.0
    # Top-level scalar alias (사용자 명세 `oversmoothing/energy`, `oversmoothing/mad`)
    if 'mean' in metrics.get('energy', {}):
        out['oversmoothing/energy'] = float(metrics['energy']['mean']) if metrics['energy']['mean'] == metrics['energy']['mean'] else 0.0
    if 'mean' in metrics.get('mad', {}):
        out['oversmoothing/mad'] = float(metrics['mad']['mean']) if metrics['mad']['mean'] == metrics['mad']['mean'] else 0.0
    return out
