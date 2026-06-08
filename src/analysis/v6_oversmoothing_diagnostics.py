"""V6-W0 Phase 0 — HGAT Over-Smoothing 진단 계측 (RFP 2026-06-01 §6 정합).

M4 anchor QCond GAT NL3 위 forward hook + GATv2Conv attention capture 를 설치해
layer-wise Dirichlet energy (§6.1) / MAD (§6.2) / Attention entropy (§6.3) 와
테이블 내 컬럼 분산 (H2) + 고차수 hub 분리 통계 (H1) 를 BIRD-Dev 전체 위 측정한다.

근거:
- planning/oversmoothing/oversmoothing_rfp_2026-06-01.md §6 (코드 spec)
- planning/oversmoothing/oversmoothing_v6_plan_2026-06-01.md §1 Phase 0 + §3 진단 protocol
- planning/DECISIONS.md 2026-06-01 V6 chain launch entry

사용:
    PYTHONPATH=src python src/analysis/v6_oversmoothing_diagnostics.py \\
        --ckpt outputs/checkpoints/best_gat_qcond_nl3.pt \\
        --output_dir outputs/v6_w0/diagnostics_s0_reference \\
        [--num_queries 1534] [--device cuda] [--hi_deg_thr 30]

산출 (output_dir 아래):
    per_query.jsonl          — 쿼리별 layer-wise 메트릭 (분석 raw input)
    layer_summary.json       — layer 별 mean / std / p50 (전체 쿼리 위 집계)
    degree_stats.json        — BIRD-Dev 차수/gold 통계 (H1/H2 해석 기준)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_geometric.nn import GATv2Conv  # noqa: E402

from modules.builders.graph_builder import EnrichedHeteroGraphBuilder  # noqa: E402
from modules.selectors.ensemble_selector import EnsembleSelector  # noqa: E402
from utils.logger import get_logger, setup_logger  # noqa: E402


# ─────────────────────────────────────────────────────────────
# 1. RFP §6 metric primitives — 코드 그대로 이식 + 안전 가드
# ─────────────────────────────────────────────────────────────
def dirichlet_energy(h: torch.Tensor, edge_index: torch.Tensor,
                     deg: Optional[torch.Tensor] = None,
                     eps: float = 1e-12) -> float:
    """RFP §6.1: $\\frac{1}{2}\\sum_{(i,j)\\in E}\\|h_i/\\sqrt{1+d_i}-h_j/\\sqrt{1+d_j}\\|^2$.

    edge_index 는 양방향으로 미리 대칭화돼 있어야 한다 (caller 책임).
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

    mask_pairs: [N, N] bool — None 이면 off-diagonal 전체.
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


def attention_entropy(alpha: torch.Tensor, index: torch.Tensor,
                      num_nodes: int, eps: float = 1e-12) -> float:
    """RFP §6.3: destination node 별 attention 분포 엔트로피의 평균.

    이웃이 2개 이상인 destination 만 집계 (단일-이웃 destination 은 정의상 0).
    값이 클수록 균일 (변별력 상실), 작을수록 집중.
    """
    if alpha is None or alpha.numel() == 0:
        return float("nan")
    if alpha.dim() == 2:
        alpha = alpha.mean(dim=1)
    ent_per_edge = -alpha * (alpha + eps).log()
    ent = torch.zeros(num_nodes, device=alpha.device, dtype=alpha.dtype)
    ent.scatter_add_(0, index, ent_per_edge)
    deg = torch.zeros(num_nodes, device=alpha.device, dtype=alpha.dtype)
    deg.scatter_add_(0, index,
                     torch.ones(alpha.size(0), device=alpha.device, dtype=alpha.dtype))
    valid = deg > 1
    if not valid.any():
        return float("nan")
    return float(ent[valid].mean().item())


# ─────────────────────────────────────────────────────────────
# 2. Layer 입력 capture + post-forward attention re-run (HeteroConv 호환)
# ─────────────────────────────────────────────────────────────
def install_input_capture(model) -> List[Dict[str, Any]]:
    """각 HeteroConv 의 입력 x_dict 를 pre-hook 으로 capture.

    이후 `compute_attention_per_layer` 가 capture 된 x_dict + edge_index_dict 위
    각 inner GATv2Conv 를 `return_attention_weights=True` 로 재호출해 alpha 회수.
    PyG HeteroConv 의 ModuleDict 키 형식 (tuple vs string) 변환을 우회한다.
    """
    captures: List[Dict[str, Any]] = []

    def make_prehook(slot):
        def prehook(_module, args):
            # args = (x_dict, edge_index_dict) — clone to detach from grad graph
            if len(args) >= 2:
                slot['x_dict'] = {nt: t.detach() for nt, t in args[0].items()}
                slot['edge_index_dict'] = dict(args[1])
        return prehook

    handles = []
    for hetero in model.convs:
        slot: Dict[str, Any] = {'x_dict': None, 'edge_index_dict': None}
        captures.append(slot)
        h = hetero.register_forward_pre_hook(make_prehook(slot))
        handles.append(h)
    # caller 가 unwrap 할 수 있도록 (handles, captures) 반환
    return handles, captures


@torch.no_grad()
def compute_attention_per_layer(model, captures: List[Dict[str, Any]]
                                ) -> List[Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]]]:
    """capture 된 layer 입력으로 inner GATv2Conv 를 재호출해 attention 산출.

    Returns: layer index → dict[edge_type tuple → (ei_aug, alpha)]
    """
    out: List[Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]]] = []
    for li, hetero in enumerate(model.convs):
        cap = captures[li]
        layer_x = cap.get('x_dict') or {}
        layer_ei = cap.get('edge_index_dict') or {}
        per_et: Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]] = {}
        for edge_type, conv in hetero.convs.items():
            if not isinstance(conv, GATv2Conv):
                continue
            ei = layer_ei.get(edge_type)
            if ei is None or ei.numel() == 0:
                continue
            src_t, _, dst_t = edge_type
            if src_t == dst_t:
                x = layer_x.get(dst_t)
                if x is None:
                    continue
                _o, attn = conv(x, ei, return_attention_weights=True)
            else:
                x_s = layer_x.get(src_t)
                x_d = layer_x.get(dst_t)
                if x_s is None or x_d is None:
                    continue
                _o, attn = conv((x_s, x_d), ei, return_attention_weights=True)
            per_et[edge_type] = (attn[0].detach(), attn[1].detach())
        out.append(per_et)
    return out


# ─────────────────────────────────────────────────────────────
# 3. 노드 metadata: 같은 테이블 컬럼 mask + 고차수 hub 분리
# ─────────────────────────────────────────────────────────────
def build_column_metadata(metadata: Dict[str, Any], hi_deg_thr: int = 30
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """metadata['node_metadata'] 위 column 노드 분석.

    metadata['node_metadata']: dict[flat_idx → name]. name 형식:
      - "table"            : table 노드
      - "table.column"     : column 노드 (.split('.', 1))
      - "src.col -> dst.col": fk_node (edge label, '->' 포함 → 제외)

    Returns:
      intra_mask:       [N_col, N_col] bool  같은 테이블 컬럼 쌍 (off-diag)
      col_global_idx:   [N_col] long         column 노드의 flat_idx
      hi_deg_col_mask:  [N_col] bool         차수 > hi_deg_thr 테이블의 컬럼
      tbl_col_count:    dict[table → count]  테이블별 컬럼 수
    """
    node_meta = metadata.get('node_metadata', {})
    col_entries: List[Tuple[int, str, str]] = []
    tbl_col_count: Dict[str, int] = {}
    for gid, name in node_meta.items():
        if not isinstance(name, str) or '->' in name:
            continue
        if '.' not in name:
            continue
        tbl, col = name.split('.', 1)
        col_entries.append((int(gid), tbl, col))
        tbl_col_count[tbl] = tbl_col_count.get(tbl, 0) + 1

    col_entries.sort(key=lambda x: x[0])
    col_global_idx = torch.tensor([e[0] for e in col_entries], dtype=torch.long)
    N_col = len(col_entries)
    if N_col == 0:
        return (torch.zeros((0, 0), dtype=torch.bool),
                col_global_idx,
                torch.zeros(0, dtype=torch.bool),
                tbl_col_count)

    tables = [e[1] for e in col_entries]
    intra = torch.zeros((N_col, N_col), dtype=torch.bool)
    for i in range(N_col):
        for j in range(N_col):
            if i != j and tables[i] == tables[j]:
                intra[i, j] = True

    hi_deg_col = torch.tensor(
        [tbl_col_count.get(t, 0) > hi_deg_thr for t in tables],
        dtype=torch.bool,
    )
    return intra, col_global_idx, hi_deg_col, tbl_col_count


# ─────────────────────────────────────────────────────────────
# 4. Global edge_index — heterogeneous graph 를 단일 좌표계로 평탄화
# ─────────────────────────────────────────────────────────────
def flatten_edges(graph_data, type_offsets: Dict[str, int], device: str) -> torch.Tensor:
    """모든 edge type 을 global node id 좌표계 (table → column → fk_node 순) 로 concat."""
    parts: List[torch.Tensor] = []
    for et, ei in graph_data.edge_index_dict.items():
        if ei.numel() == 0:
            continue
        src_t, _, dst_t = et
        src_off = type_offsets.get(src_t)
        dst_off = type_offsets.get(dst_t)
        if src_off is None or dst_off is None:
            continue
        gi = torch.stack([ei[0] + src_off, ei[1] + dst_off], dim=0)
        parts.append(gi.to(device))
    if not parts:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    return torch.cat(parts, dim=1)


# ─────────────────────────────────────────────────────────────
# 5. 단일 query 위 진단 metric 산출
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def diagnose_query(
    *, db_id: str, question: str,
    selector: EnsembleSelector, graph_data, metadata: Dict[str, Any],
    input_captures: List[Dict[str, Any]],
    hi_deg_thr: int = 30,
    device: str = "cuda",
) -> Dict[str, Any]:

    # --- 1) Encode query ---
    encoded = selector.encoder.encode([question])
    q_emb = encoded[0] if isinstance(encoded, tuple) else encoded
    if q_emb.dim() == 3:
        q_emb = q_emb.mean(dim=1)
    elif q_emb.dim() == 1:
        q_emb = q_emb.unsqueeze(0)
    elif q_emb.dim() == 2 and q_emb.size(0) > 1:
        q_emb = q_emb.mean(dim=0, keepdim=True)
    q_emb = q_emb.to(device)

    # --- 2) Forward hook: layer 별 출력 dict 캡처 (HeteroConv post-conv post-elu) ---
    H_per_layer: List[Dict[str, torch.Tensor]] = []

    # Reset input_captures (이전 query 의 capture clear)
    for slot in input_captures:
        slot['x_dict'] = None
        slot['edge_index_dict'] = None

    def make_hook():
        def hook(module, _input, output):
            H_per_layer.append({nt: F.elu(t).detach() for nt, t in output.items()})
        return hook

    handles = [conv.register_forward_hook(make_hook())
               for conv in selector.gat_model.convs]
    try:
        _ = selector.gat_model(
            graph_data.x_dict, graph_data.edge_index_dict,
            query_emb=q_emb,
        )
    finally:
        for h in handles:
            h.remove()

    # --- 2b) Post-forward attention re-run on captured layer inputs ---
    attn_per_layer = compute_attention_per_layer(selector.gat_model, input_captures)

    # --- 3) Node offset & global edge_index ---
    num_t = graph_data['table'].num_nodes
    num_c = graph_data['column'].num_nodes
    num_fk = (graph_data['fk_node'].num_nodes
              if 'fk_node' in graph_data.node_types else 0)
    type_offsets = {'table': 0, 'column': num_t, 'fk_node': num_t + num_c}
    N_total = num_t + num_c + num_fk

    global_ei_dir = flatten_edges(graph_data, type_offsets, device=device)
    if global_ei_dir.numel() > 0:
        # 무방향 대칭 (Dirichlet energy 정의 기준)
        global_ei = torch.cat([global_ei_dir, global_ei_dir.flip(0)], dim=1)
    else:
        global_ei = global_ei_dir

    # --- 4) intra-table + 고차수 mask ---
    intra_mask_cpu, col_global_idx_cpu, hi_deg_col_cpu, tbl_col_count = \
        build_column_metadata(metadata, hi_deg_thr)
    col_global_idx = col_global_idx_cpu.to(device)
    intra_mask = intra_mask_cpu.to(device)
    hi_deg_col_mask = hi_deg_col_cpu.to(device)

    if intra_mask.numel() > 0:
        hi_idx = hi_deg_col_mask
        intra_hi = intra_mask & hi_idx.unsqueeze(0) & hi_idx.unsqueeze(1)
        intra_lo = intra_mask & (~hi_idx).unsqueeze(0) & (~hi_idx).unsqueeze(1)
    else:
        intra_hi = intra_mask
        intra_lo = intra_mask

    # --- 5) Layer-wise metrics ---
    out_layers: List[Dict[str, Any]] = []
    for li, H_dict in enumerate(H_per_layer):
        Hs: List[torch.Tensor] = []
        for nt in ['table', 'column', 'fk_node']:
            if nt in H_dict and H_dict[nt].size(0) > 0:
                Hs.append(H_dict[nt])
        if not Hs:
            continue
        H_all = torch.cat(Hs, dim=0)

        # Dirichlet energy (전체 heterogeneous graph)
        e_full = dirichlet_energy(H_all, global_ei)
        # Normalize per edge for cross-graph comparability
        e_per_edge = (e_full / max(global_ei.size(1) // 2, 1)) if global_ei.size(1) > 0 else 0.0

        # MAD (전체 노드 쌍)
        mad_all_val = mad(H_all)

        # Column-only metrics
        if col_global_idx.numel() > 0:
            H_col = H_all[col_global_idx]
            mad_col = mad(H_col)
            mad_intra = mad(H_col, mask_pairs=intra_mask)
            mad_intra_hi = mad(H_col, mask_pairs=intra_hi) if intra_hi.any() else float("nan")
            mad_intra_lo = mad(H_col, mask_pairs=intra_lo) if intra_lo.any() else float("nan")
        else:
            mad_col = float("nan")
            mad_intra = float("nan")
            mad_intra_hi = float("nan")
            mad_intra_lo = float("nan")

        # Attention entropy per edge type
        attn_ent_per_et: Dict[str, float] = {}
        per_et = attn_per_layer[li] if li < len(attn_per_layer) else {}
        for edge_type, (ei_aug, alpha) in per_et.items():
            _src_t, _rel, dst_t = edge_type
            n_dst = H_dict.get(dst_t, torch.zeros(0)).size(0)
            if n_dst == 0:
                continue
            label = f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"
            attn_ent_per_et[label] = attention_entropy(alpha, ei_aug[1], n_dst)

        attn_vals = [v for v in attn_ent_per_et.values() if v == v]
        attn_mean = float(np.mean(attn_vals)) if attn_vals else float('nan')

        out_layers.append({
            'layer': li + 1,
            'dirichlet_energy': e_full,
            'dirichlet_energy_per_edge': float(e_per_edge),
            'mad_all': mad_all_val,
            'mad_column': mad_col,
            'mad_intra_table': mad_intra,
            'mad_intra_table_hi_deg': mad_intra_hi,
            'mad_intra_table_lo_deg': mad_intra_lo,
            'attn_entropy_per_edge_type': attn_ent_per_et,
            'attn_entropy_mean': attn_mean,
        })

    return {
        'db_id': db_id,
        'num_node_table': int(num_t),
        'num_node_column': int(num_c),
        'num_node_fk': int(num_fk),
        'num_node_total': int(N_total),
        'num_column_hi_deg': int(hi_deg_col_mask.sum().item()),
        'num_table': len(tbl_col_count),
        'num_table_hi_deg': sum(1 for c in tbl_col_count.values() if c > hi_deg_thr),
        'layers': out_layers,
    }


# ─────────────────────────────────────────────────────────────
# 6. BIRD-Dev degree statistics (H1/H2 해석 기준)
# ─────────────────────────────────────────────────────────────
SQL_TOKEN_RE = re.compile(
    r"`?[A-Za-z_][A-Za-z0-9_]*`?\s*\.\s*`?[A-Za-z_][A-Za-z0-9_]*`?"
)


def compute_degree_stats(dev_tables_json: str, dev_json: str,
                         hi_deg_thr: int = 30) -> Dict[str, Any]:
    with open(dev_tables_json, 'r') as f:
        tables = json.load(f)
    with open(dev_json, 'r') as f:
        dev = json.load(f)

    db_to_tbl_counts: Dict[str, List[int]] = {}
    for tbl in tables:
        db_id = tbl['db_id']
        tbl_names = tbl.get('table_names_original', [])
        col_table = tbl.get('column_names_original', [])
        per_tbl: Dict[int, int] = {i: 0 for i in range(len(tbl_names))}
        for ti, _ in col_table:
            if ti == -1:
                continue
            per_tbl[ti] = per_tbl.get(ti, 0) + 1
        db_to_tbl_counts[db_id] = list(per_tbl.values())

    all_counts: List[int] = []
    for c in db_to_tbl_counts.values():
        all_counts.extend(c)
    arr = np.array(all_counts) if all_counts else np.array([0])

    gold_counts_rough: List[int] = []
    for e in dev:
        sql = e.get('SQL', '') or e.get('query', '') or ''
        tokens = SQL_TOKEN_RE.findall(sql)
        cleaned = {t.replace('`', '').replace(' ', '').lower() for t in tokens}
        gold_counts_rough.append(len(cleaned))
    g_arr = np.array(gold_counts_rough) if gold_counts_rough else np.array([0])

    return {
        'hi_deg_thr': hi_deg_thr,
        'num_dbs': len(db_to_tbl_counts),
        'num_tables_total': int(len(arr)),
        'columns_per_table': {
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'min': int(arr.min()),
            'p25': float(np.percentile(arr, 25)),
            'p50': float(np.percentile(arr, 50)),
            'p75': float(np.percentile(arr, 75)),
            'p90': float(np.percentile(arr, 90)),
            'max': int(arr.max()),
        },
        'fraction_tables_over_thr': float((arr > hi_deg_thr).mean()),
        'count_tables_over_thr': int((arr > hi_deg_thr).sum()),
        'histogram_columns_per_table': {
            '0-9': int(((arr >= 0) & (arr < 10)).sum()),
            '10-19': int(((arr >= 10) & (arr < 20)).sum()),
            '20-29': int(((arr >= 20) & (arr < 30)).sum()),
            '30-49': int(((arr >= 30) & (arr < 50)).sum()),
            '50+': int((arr >= 50).sum()),
        },
        'per_db_column_count_stats': {
            db_id: {
                'num_tables': len(counts),
                'mean': float(np.mean(counts)) if counts else 0.0,
                'max': int(max(counts)) if counts else 0,
                'num_hi_deg_tables': sum(1 for c in counts if c > hi_deg_thr),
            } for db_id, counts in db_to_tbl_counts.items()
        },
        'num_queries': len(dev),
        'gold_columns_per_query_rough': {
            'mean': float(g_arr.mean()),
            'std': float(g_arr.std()),
            'min': int(g_arr.min()),
            'p50': float(np.percentile(g_arr, 50)),
            'p75': float(np.percentile(g_arr, 75)),
            'p90': float(np.percentile(g_arr, 90)),
            'max': int(g_arr.max()),
            'note': (
                "SQL 문자열의 'table.col' 패턴 token 추출 위 rough estimate. "
                "정확한 gold column count 는 analyzer 가 evaluator 정합 위 재산출 권장."
            ),
        },
    }


# ─────────────────────────────────────────────────────────────
# 7. Per-DB graph cache (build cost 분할)
# ─────────────────────────────────────────────────────────────
def _build_or_get_graph(builder, cache: Dict[str, Tuple[Any, Dict[str, Any]]],
                        db_id: str, db_dir: str, device: str):
    if db_id in cache:
        graph, meta = cache[db_id]
    else:
        graph, meta = builder.build(db_id=db_id, db_dir=db_dir)
        if isinstance(meta, dict):
            meta.setdefault('db_id', db_id)
        cache[db_id] = (graph, meta)
    return graph.to(device), meta


# ─────────────────────────────────────────────────────────────
# 8. CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="V6-W0 Phase 0 over-smoothing diagnostics")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dev_json", default="data/raw/BIRD_dev/dev.json")
    parser.add_argument("--dev_db_dir", default="data/raw/BIRD_dev/dev_databases")
    parser.add_argument("--dev_tables_json", default="data/raw/BIRD_dev/dev_tables.json")
    parser.add_argument("--num_queries", type=int, default=-1, help="-1 = 전체")
    parser.add_argument("--hi_deg_thr", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--in_channels", type=int, default=384)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--out_channels", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--encoder_model",
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--include_views", action="store_true")
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(output_dir), exp_name="v6_w0_diagnostics", sub_dir="")
    logger = get_logger(__name__)

    # --- 1) Degree stats (cheap, 비-GAT) ---
    logger.info(f"[1/3] Computing BIRD-Dev degree statistics (hi_deg_thr={args.hi_deg_thr})")
    degree_stats = compute_degree_stats(
        args.dev_tables_json, args.dev_json, hi_deg_thr=args.hi_deg_thr,
    )
    with open(output_dir / "degree_stats.json", 'w', encoding='utf-8') as f:
        json.dump(degree_stats, f, indent=2, ensure_ascii=False)
    logger.info(
        f"  columns/table mean={degree_stats['columns_per_table']['mean']:.2f}, "
        f"max={degree_stats['columns_per_table']['max']}, "
        f"fraction > {args.hi_deg_thr}: {degree_stats['fraction_tables_over_thr']:.3f}, "
        f"count > {args.hi_deg_thr}: {degree_stats['count_tables_over_thr']}"
    )

    # --- 2) Model + builder + encoder (EnsembleSelector 가 cleanly assemble) ---
    logger.info(f"[2/3] Loading QCond GAT NL3 from {args.ckpt} on {args.device}")
    selector = EnsembleSelector(
        weight_path=args.ckpt,
        alpha=args.alpha, top_k=args.top_k,
        in_channels=args.in_channels, hidden_channels=args.hidden_channels,
        out_channels=args.out_channels, num_layers=args.num_layers,
        query_conditioned=True, encoder_type="plm",
        encoder_model_name=args.encoder_model,
    )
    builder = EnrichedHeteroGraphBuilder(
        plm_model_name=args.encoder_model,
        tables_json_path=args.dev_tables_json,
        include_views=args.include_views,
        run_leiden_clustering=True,
    )
    input_handles, input_captures = install_input_capture(selector.gat_model)
    logger.info(
        f"  Installed layer-input pre-hooks on {len(input_captures)} HeteroConv layers"
    )

    # --- 3) Iterate dev queries ---
    with open(args.dev_json, 'r') as f:
        dev = json.load(f)
    if args.num_queries > 0:
        dev = dev[: args.num_queries]
    logger.info(f"[3/3] Processing {len(dev)} BIRD-Dev queries")

    out_path = output_dir / "per_query.jsonl"
    graph_cache: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
    layer_acc: Dict[int, Dict[str, List[float]]] = {}
    n_ok = 0
    n_err = 0
    t0 = time.perf_counter()

    METRIC_KEYS = [
        'dirichlet_energy', 'dirichlet_energy_per_edge',
        'mad_all', 'mad_column', 'mad_intra_table',
        'mad_intra_table_hi_deg', 'mad_intra_table_lo_deg',
        'attn_entropy_mean',
    ]

    with open(out_path, 'w', encoding='utf-8') as f_out:
        for qi, entry in enumerate(dev):
            try:
                graph_data, metadata = _build_or_get_graph(
                    builder, graph_cache,
                    db_id=entry['db_id'], db_dir=args.dev_db_dir, device=args.device,
                )
                rec = diagnose_query(
                    db_id=entry['db_id'],
                    question=entry['question'],
                    selector=selector,
                    graph_data=graph_data,
                    metadata=metadata,
                    input_captures=input_captures,
                    hi_deg_thr=args.hi_deg_thr,
                    device=args.device,
                )
                rec['question_id'] = entry.get('question_id', qi)
                rec['difficulty'] = entry.get('difficulty', '')
                # 누적 — layer 별 metric 평균/표준편차 위
                for L in rec['layers']:
                    li = L['layer']
                    acc = layer_acc.setdefault(li, {k: [] for k in METRIC_KEYS})
                    for k in METRIC_KEYS:
                        v = L.get(k)
                        if v is None:
                            continue
                        try:
                            fv = float(v)
                        except Exception:
                            continue
                        if fv == fv:  # NaN 제외
                            acc[k].append(fv)
                f_out.write(json.dumps(rec, ensure_ascii=False) + '\n')
                f_out.flush()
                n_ok += 1
            except Exception as e:
                n_err += 1
                err_rec = {
                    'question_id': entry.get('question_id', qi),
                    'db_id': entry.get('db_id', ''),
                    'error': str(e),
                    'trace': traceback.format_exc().splitlines()[-5:],
                }
                f_out.write(json.dumps(err_rec, ensure_ascii=False) + '\n')
                logger.error(f"  query {qi} ({entry.get('db_id')}) failed: {e}")

            if (qi + 1) % args.log_every == 0 or (qi + 1) == len(dev):
                elapsed = time.perf_counter() - t0
                rate = (qi + 1) / max(elapsed, 1e-6)
                eta = (len(dev) - qi - 1) / max(rate, 1e-6)
                logger.info(
                    f"  [{qi+1}/{len(dev)}] ok={n_ok} err={n_err} "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f}q/s eta={eta:.1f}s"
                )

    # --- 4) Layer summary ---
    summary: Dict[str, Any] = {}
    for li, acc in sorted(layer_acc.items()):
        summary[f"layer_{li}"] = {}
        for k, vs in acc.items():
            if not vs:
                summary[f"layer_{li}"][k] = {'mean': None, 'std': None, 'n': 0}
                continue
            a = np.array(vs)
            summary[f"layer_{li}"][k] = {
                'mean': float(a.mean()),
                'std': float(a.std()),
                'p25': float(np.percentile(a, 25)),
                'p50': float(np.percentile(a, 50)),
                'p75': float(np.percentile(a, 75)),
                'n': int(a.size),
            }

    layer_summary_path = output_dir / "layer_summary.json"
    with open(layer_summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'ckpt': args.ckpt,
            'num_queries_processed': n_ok,
            'num_queries_errored': n_err,
            'hi_deg_thr': args.hi_deg_thr,
            'num_layers': args.num_layers,
            'encoder_model': args.encoder_model,
            'per_layer': summary,
            'wall_time_s': time.perf_counter() - t0,
        }, f, indent=2, ensure_ascii=False)

    elapsed = time.perf_counter() - t0
    logger.info(
        f"✅ V6-W0 진단 완료 — ok={n_ok}, err={n_err}, wall={elapsed:.1f}s. "
        f"산출 → {output_dir}"
    )
    # Pre-hook cleanup
    for h in input_handles:
        h.remove()

    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return "N/A"
        try:
            return f"{float(v):.4f}"
        except Exception:
            return "N/A"

    for li in sorted(layer_acc.keys()):
        s = summary[f"layer_{li}"]
        de = s.get('dirichlet_energy', {})
        mc = s.get('mad_column', {})
        ae = s.get('attn_entropy_mean', {})
        logger.info(
            f"  Layer {li}: "
            f"DirichletEnergy μ={_fmt(de.get('mean'))}±{_fmt(de.get('std'))} | "
            f"MAD(col) μ={_fmt(mc.get('mean'))}±{_fmt(mc.get('std'))} | "
            f"AttnEnt μ={_fmt(ae.get('mean'))}±{_fmt(ae.get('std'))}"
        )


if __name__ == "__main__":
    main()
