"""V1~V5 retrospective V6 diagnostic protocol (Dirichlet energy + MAD + attention entropy).

planning/DECISIONS.md 2026-06-01 (V6 chain launch) §추가 필요 분석 정합:
  "V1~V5 의 14-trial cells 의 retrospective Dirichlet/MAD/attention entropy 재산출
   (선행 cells 의 stored embeddings 가 가용 시 — 가용성 확인 위 root 위임 후 analyzer 진행)"

목적:
  V6-W0 의 신규 진단 protocol (Dirichlet energy + MAD + attention entropy, RFP §6) 위
  V1~V5 chain 의 14-trial mitigation cells 가 신규 metric 위 retain 되는지 confirm —
  paper §V.5.4 narrative (14-trial null + mech(ii-b) DOMINANT 5/5) 의 신규 metric 위 정합.

산출:
  - notebooks/analysis_results/v1_v5_retrospective_v6_metrics_2026-06-01.md
  - outputs/analysis/v1_v5_retrospective_v6_metrics_2026-06-01.csv (per-cell × per-layer rows)
  - outputs/analysis/v1_v5_retrospective_v6_metrics_2026-06-01.json (full summary)
  - outputs/analysis/v1_v5_retrospective_v6_metrics_per_query_2026-06-01.jsonl

Spec:
  - 데이터: BIRD-Dev (1534q × 11 DB) 위 stratified 5/DB × 11 = **55 queries** (seed=42)
  - Encoder: LocalPLMEncoder("sentence-transformers/all-MiniLM-L6-v2") — 모든 cells 공통 (384-dim)
  - Builder: EnrichedHeteroGraphBuilder (PLM=MiniLM, default)
  - Cells (14-trial chain + extras):
      Phase 1 baseline (qcond_nl3, v1 SchemaHeteroGAT, NL3)
      Phase 2 B5 (B5 fusion: PN+IR+JK+DS+L=2+AC+ListNet)
      V2 #1 DropMessage
      V2 #2 Sum
      V2 #3 LayerNorm (mit best)
      V3 #1 GIN
      V4-A LN+GIN combo
      V4-B AERO Softplus+Sym-Norm
      V5-A GATE
      V5-B GCNII L=2 / L=4 / L=6
      V5-C cum_only / hop_only / full
  - Forward: per-query (on-the-fly) — dev_enriched_plm_graphs.pt cache 또는 builder-rebuild
  - Per-layer:
      L0_PLM   = input PLM features (pre-projection)
      L1_GAT   = HeteroConv layer 1 output (post F.elu)
      ...
      L_N_GAT  = HeteroConv layer N output (N = num_layers, varies by cell)
      L_out    = final output (out_lin + skip 또는 JK/Hop-attention)
  - V6 metric:
      Dirichlet energy on column intra-table induced edges (column-table-column path 의
        column-column 동테이블 페어 edge set) — scale-invariant 위 per-edge + normalized 둘 다 리포트
      MAD overall (= mean(1-cos sim) over all column pairs i<j)
      MAD intra-table / MAD inter-table (테이블 동일/다른 페어 분리)
      Attention entropy per layer per edge type (via extract_layerwise_attention_v2)
  - Cross-check: 기존 intra_table_sims (L_GAT cos sim) 와의 ranking 정합

근거:
  - planning/oversmoothing/oversmoothing_v6_plan_2026-06-01.md §3 진단 protocol
  - planning/oversmoothing/oversmoothing_rfp_2026-06-01.md §6 신규 metric
  - planning/DECISIONS.md 2026-06-01 §추가 필요 분석 (V1~V5 retrospective)
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/hyeonjin/thesis_refactored")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.builders.graph_builder import EnrichedHeteroGraphBuilder  # noqa: E402
from modules.encoders.local_encoder import LocalPLMEncoder  # noqa: E402
from models.gat_network import SchemaHeteroGAT  # noqa: E402
from models.gat_network_v2 import SchemaHeteroGATv2  # noqa: E402
from analysis.extract_layerwise_attention_v2 import (  # noqa: E402
    extract_layerwise_attention_v2,
)
from analysis.gat_bottleneck_analysis import intra_table_sims, COL_TO_TAB_EDGE  # noqa: E402

DEV_JSON = ROOT / "data/raw/BIRD_dev/dev.json"
DEV_DB_DIR = ROOT / "data/raw/BIRD_dev/dev_databases"
TABLES_JSON = ROOT / "data/raw/BIRD_dev/dev_tables.json"

OUT_DIR = ROOT / "outputs/analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DATE = "2026-06-01"

SAMPLES_PER_DB = 5
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ────────────────────────────────────────────────────────────────────────
# 14-trial chain cells (planning/oversmoothing/README.md §4.1 + V6 plan §0.1)
# ────────────────────────────────────────────────────────────────────────
CELLS: List[Dict[str, Any]] = [
    {
        "tag": "Phase1_baseline",
        "label": "Phase 1 baseline — qcond_nl3 (NL=3, no mitigation)",
        "ckpt_path": "outputs/checkpoints/best_gat_qcond_nl3.pt",
        "model_class": "v1",
        "v1_kwargs": {
            "in_channels": 384, "hidden_channels": 256, "out_channels": 256,
            "num_layers": 3, "heads": 4,
            "query_conditioned": True, "query_supernode": False,
        },
        "r15_reported": 0.6097,
        "narrative": "baseline (no mitigation)",
    },
    {
        "tag": "Phase2_B5",
        "label": "Phase 2 B5 fusion (PN+IR+JK+DS+L=2+AC+ListNet)",
        "ckpt_path": "outputs/checkpoints/best_gat_directed_supernode_p80_b5_mitigation.pt",
        "model_class": "v2",
        "r15_reported": 0.6018,
        "narrative": "B5 fusion, baseline of V2/V3/V4/V5 ablations",
    },
    {
        "tag": "V2_DropMessage",
        "label": "V2 #1 DropMessage (p=0.2, B5 + DM)",
        "ckpt_path": "outputs/checkpoints/best_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.pt",
        "model_class": "v2",
        "r15_reported": 0.5974,
        "narrative": "mech(ii-a) regularizer",
    },
    {
        "tag": "V2_Sum",
        "label": "V2 #2 Sum aggregation (B5 + Sum)",
        "ckpt_path": "outputs/checkpoints/best_gat_directed_supernode_p80_b5_mitigation_v2_sum_aggr.pt",
        "model_class": "v2",
        "r15_reported": 0.5761,
        "narrative": "mech(i) sum-only, worst mit (-0.0336)",
    },
    {
        "tag": "V2_LayerNorm",
        "label": "V2 #3 LayerNorm pre-softmax (B5 + LN) ★ mit best",
        "ckpt_path": "outputs/checkpoints/best_gat_directed_supernode_p80_b5_mitigation_v2_layernorm.pt",
        "model_class": "v2",
        "r15_reported": 0.6011,
        "narrative": "mech(ii-a) pre-softmax LN, mit best (-0.0086)",
    },
    {
        "tag": "V3_GIN",
        "label": "V3 #1 GIN-style aggregation (B5 + GIN)",
        "ckpt_path": "outputs/checkpoints/best_gat_directed_supernode_p80_b5_mitigation_v3_gin.pt",
        "model_class": "v2",
        "r15_reported": 0.5954,
        "narrative": "mech(i) sum+MLP, hierarchy partial fail",
    },
    {
        "tag": "V4A_LNGIN",
        "label": "V4-A LN+GIN combo (PN+IR+JK + LN-pre-softmax + GIN)",
        "ckpt_path": "outputs/checkpoints/best_gat_directed_supernode_p80_v4a_lngin_combo.pt",
        "model_class": "v2",
        "r15_reported": 0.5929,
        "narrative": "V4 architectural intervention #A (LN+GIN fusion)",
    },
    {
        "tag": "V4B_AERO",
        "label": "V4-B AERO Softplus+Sym-Norm (row-stoch 가정 위반)",
        "ckpt_path": "outputs/checkpoints/best_gat_directed_supernode_p80_v4b_aero.pt",
        "model_class": "v2",
        "r15_reported": 0.5951,
        "narrative": "V4 architectural intervention #B (AERO Softplus)",
    },
    {
        "tag": "V5A_GATE",
        "label": "V5-A GATE (att_self+parent decoupling)",
        "ckpt_path": "outputs/checkpoints/best_gat_directed_supernode_p80_v5a_gate.pt",
        "model_class": "v2",
        "r15_reported": 0.5571,
        "narrative": "Mustafa & Burkholz NeurIPS 2024",
    },
    {
        "tag": "V5B_GCNII_L2",
        "label": "V5-B GCNII L=2 (IR α + identity mapping)",
        "ckpt_path": "outputs/checkpoints/best_gat_directed_supernode_p80_v5b_gcnii_L2.pt",
        "model_class": "v2",
        "r15_reported": 0.6072,
        "narrative": "Chen et al. ICML 2020 GCNII, NL=2",
    },
    {
        "tag": "V5B_GCNII_L4",
        "label": "V5-B GCNII L=4 (depth scaling)",
        "ckpt_path": "outputs/checkpoints/best_gat_directed_supernode_p80_v5b_gcnii_L4.pt",
        "model_class": "v2",
        "r15_reported": 0.5969,
        "narrative": "GCNII, NL=4",
    },
    {
        "tag": "V5B_GCNII_L6",
        "label": "V5-B GCNII L=6 (depth scaling 추가)",
        "ckpt_path": "outputs/checkpoints/best_gat_directed_supernode_p80_v5b_gcnii_L6.pt",
        "model_class": "v2",
        "r15_reported": 0.5845,
        "narrative": "GCNII, NL=6",
    },
    {
        "tag": "V5C_cum_only",
        "label": "V5-C cum_only (AERO+Cumulative Residual)",
        "ckpt_path": "outputs/checkpoints/best_gat_directed_supernode_p80_v5c_cum_only.pt",
        "model_class": "v2",
        "r15_reported": 0.5993,
        "narrative": "AERO + cumulative residual only",
    },
    {
        "tag": "V5C_hop_only",
        "label": "V5-C hop_only (AERO+Hop Attention)",
        "ckpt_path": "outputs/checkpoints/best_gat_directed_supernode_p80_v5c_hop_only.pt",
        "model_class": "v2",
        "r15_reported": 0.6076,
        "narrative": "AERO + hop attention only",
    },
    {
        "tag": "V5C_full",
        "label": "V5-C full (AERO+Cumulative+Hop)",
        "ckpt_path": "outputs/checkpoints/best_gat_directed_supernode_p80_v5c_full.pt",
        "model_class": "v2",
        "r15_reported": 0.5887,
        "narrative": "AERO Full (V4-B + Hop + Cumulative)",
    },
]

# Whitelist of valid SchemaHeteroGATv2.__init__ kwargs (filter ckpt config to these)
V2_KW_WHITELIST = {
    "in_channels", "hidden_channels", "out_channels", "num_layers", "heads",
    "query_conditioned", "query_supernode",
    "pairnorm_mode", "pairnorm_scale",
    "initial_residual_alpha",
    "jumping_knowledge",
    "dual_stream",
    "num_layers_mode", "num_layers_fallback",
    "diameter_path", "diameter_dict",
    "supernode_edge_direction",
    "supernode_topk", "supernode_topk_criterion",
    "supernode_threshold_mode", "supernode_threshold_value",
    "supernode_score_normalization",
    "drop_message_p", "use_layernorm_pre_softmax", "aggregation_type",
    "gat_layer_type", "softplus_symmetric_norm",
    "gcnii_beta_lambda",
    "aero_hop_attention",
    "aero_cumulative_attention", "aero_cumulative_decay",
}


# ────────────────────────────────────────────────────────────────────────
# Data + sampling
# ────────────────────────────────────────────────────────────────────────

def load_dev() -> List[Dict]:
    with DEV_JSON.open() as f:
        return json.load(f)


def stratified_qids(dev: List[Dict], per_db: int = SAMPLES_PER_DB,
                    seed: int = RANDOM_SEED) -> Tuple[List[int], Dict[int, str]]:
    qid_by_db: Dict[str, List[int]] = defaultdict(list)
    for i, d in enumerate(dev):
        qid_by_db[d["db_id"]].append(i)
    rng = random.Random(seed)
    qids: List[int] = []
    for db in sorted(qid_by_db):
        sample = rng.sample(qid_by_db[db], min(per_db, len(qid_by_db[db])))
        qids.extend(sample)
    qids.sort()
    qid_to_db = {qid: dev[qid]["db_id"] for qid in qids}
    return qids, qid_to_db


# ────────────────────────────────────────────────────────────────────────
# Model loading — v1 vs v2 dispatch
# ────────────────────────────────────────────────────────────────────────

def _flatten_config(cfg: Any) -> Dict[str, Any]:
    """ckpt['config'] 는 보통 dict 또는 dict-of-dict (sub-section: model/train/...). 평면화."""
    if not isinstance(cfg, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            out.update(_flatten_config(v))
        else:
            out[k] = v
    return out


def load_model_for_cell(cell: Dict[str, Any]) -> torch.nn.Module:
    ckpt_path = ROOT / cell["ckpt_path"]
    raw = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = raw["gat_state_dict"] if isinstance(raw, dict) and "gat_state_dict" in raw else raw

    if cell["model_class"] == "v1":
        model = SchemaHeteroGAT(**cell["v1_kwargs"]).to(DEVICE)
    else:
        cfg_raw = raw.get("config", {}) if isinstance(raw, dict) else {}
        cfg_flat = _flatten_config(cfg_raw)
        v2_kwargs = {k: v for k, v in cfg_flat.items() if k in V2_KW_WHITELIST}
        if "num_layers" not in v2_kwargs:
            v2_kwargs["num_layers"] = 2
        model = SchemaHeteroGATv2(**v2_kwargs).to(DEVICE)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [load] {cell['tag']}: missing={len(missing)} unexpected={len(unexpected)}")
        if missing[:3]:
            print(f"    first missing: {missing[:3]}")
        if unexpected[:3]:
            print(f"    first unexpected: {unexpected[:3]}")
    model.eval()
    return model


# ────────────────────────────────────────────────────────────────────────
# Layerwise forward via hook
# ────────────────────────────────────────────────────────────────────────

def inject_query_supernode(
    x_dict: Dict[str, torch.Tensor],
    edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    q_emb: torch.Tensor,
    edge_direction: str = "directed_from_sn",
) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor]]:
    """V1+ cells 위 query_node 주입 (training-time BIRDSuperNodeDataset 와 동일 logic).

    Adds:
      - x_dict['query_node'] = q_emb (shape [1, d])
      - edge_index_dict[('query_node', 'attends_to_<nt>', <nt>)] for each schema_nt
      - (if bidirectional) reverse edges
    Returns new dicts (does not mutate input).
    """
    x_out = dict(x_dict)
    e_out = dict(edge_index_dict)
    # ensure q_emb is 2D [1, d]
    if q_emb.dim() == 1:
        q_emb = q_emb.unsqueeze(0)
    elif q_emb.dim() == 3:
        q_emb = q_emb.mean(dim=1)
    if q_emb.size(0) != 1:
        q_emb = q_emb[:1]
    x_out["query_node"] = q_emb
    dev = q_emb.device
    for schema_nt in ("table", "column", "fk_node"):
        if schema_nt not in x_out:
            continue
        num_nodes = x_out[schema_nt].size(0)
        if num_nodes == 0:
            e_out[("query_node", f"attends_to_{schema_nt}", schema_nt)] = \
                torch.zeros((2, 0), dtype=torch.long, device=dev)
            if edge_direction == "bidirectional":
                e_out[(schema_nt, f"attended_by_{schema_nt}", "query_node")] = \
                    torch.zeros((2, 0), dtype=torch.long, device=dev)
            continue
        src = torch.zeros(num_nodes, dtype=torch.long, device=dev)
        dst = torch.arange(num_nodes, dtype=torch.long, device=dev)
        e_out[("query_node", f"attends_to_{schema_nt}", schema_nt)] = \
            torch.stack([src, dst], dim=0)
        if edge_direction == "bidirectional":
            e_out[(schema_nt, f"attended_by_{schema_nt}", "query_node")] = \
                torch.stack([dst, src], dim=0)
    return x_out, e_out


def extract_layerwise_via_hook(
    model: torch.nn.Module,
    x_dict: Dict[str, torch.Tensor],
    edge_index_dict: Dict,
    query_emb: torch.Tensor,
) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    """
    Returns:
        embeddings: List of per-layer dict { node_type: Tensor [N_t, D_l] (CPU) }
            Order: [L0_PLM_raw, L1_GAT, L2_GAT, ..., L_N_GAT, L_out_final]
        num_active_layers: int — depth actually used (handles v2 num_layers_mode)
    """
    embeddings: List[Dict[str, torch.Tensor]] = []
    embeddings.append({nt: x.detach().clone().cpu() for nt, x in x_dict.items()})

    captured: List[Dict[str, torch.Tensor]] = []

    def _hook(module, inputs, output):
        if isinstance(output, dict):
            captured.append({nt: x.detach().clone().cpu() for nt, x in output.items()})

    handles = [model.convs[i].register_forward_hook(_hook) for i in range(len(model.convs))]
    try:
        with torch.no_grad():
            final = model(x_dict, edge_index_dict, query_emb=query_emb)
    finally:
        for h in handles:
            h.remove()

    for layer_out in captured:
        # post-elu activation as applied in model.forward
        embeddings.append({nt: F.elu(x).detach().clone() for nt, x in layer_out.items()})
    embeddings.append({nt: x.detach().clone().cpu() for nt, x in final.items()})

    return embeddings, len(captured)


# ────────────────────────────────────────────────────────────────────────
# V6 metrics on column embeddings
# ────────────────────────────────────────────────────────────────────────

def _build_col_table_index(cb_edge: torch.Tensor) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    col_ids = cb_edge[0].tolist()
    tab_ids = cb_edge[1].tolist()
    col_to_tab: Dict[int, int] = {c: t for c, t in zip(col_ids, tab_ids)}
    table_to_cols: Dict[int, List[int]] = defaultdict(list)
    for c, t in zip(col_ids, tab_ids):
        table_to_cols[t].append(c)
    # de-dup
    for t in table_to_cols:
        table_to_cols[t] = sorted(set(table_to_cols[t]))
    return col_to_tab, table_to_cols


def compute_dirichlet_energy_columns(
    col_emb: torch.Tensor,
    cb_edge: torch.Tensor,
) -> Optional[Dict[str, float]]:
    """Dirichlet energy on column-column edges (same-table pairs, undirected, self-loops excluded).

    E_raw = (1/2) Σ_{(i,j) ∈ E_cc} || h_i / sqrt(1+d_i) - h_j / sqrt(1+d_j) ||²
    where d_i = degree of col i in E_cc.

    Returns: {energy_total, energy_per_edge, energy_normalized, num_edges,
              h_norm_sq_mean (mean ‖h‖² of column nodes used in edges)}
    """
    if col_emb is None or col_emb.numel() == 0 or cb_edge.numel() == 0:
        return None
    _, table_to_cols = _build_col_table_index(cb_edge)
    edges: List[Tuple[int, int]] = []
    for cols in table_to_cols.values():
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                edges.append((cols[i], cols[j]))
    if not edges:
        return None
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    edge_u = torch.tensor([u for u, _ in edges], dtype=torch.long)
    edge_v = torch.tensor([v for _, v in edges], dtype=torch.long)
    deg_u = torch.tensor([deg[u] for u, _ in edges], dtype=torch.float)
    deg_v = torch.tensor([deg[v] for _, v in edges], dtype=torch.float)
    # Normalize: h_i / sqrt(1+d_i)
    h_u = col_emb[edge_u] / torch.sqrt(1.0 + deg_u).unsqueeze(-1)
    h_v = col_emb[edge_v] / torch.sqrt(1.0 + deg_v).unsqueeze(-1)
    diff = h_u - h_v
    edge_sq = (diff ** 2).sum(dim=-1)
    energy_total = 0.5 * float(edge_sq.sum().item())
    num_edges = len(edges)
    energy_per_edge = energy_total / num_edges

    # Normalize by mean squared norm of involved cols (scale-invariance for cross-cell)
    used_cols = sorted(set(edge_u.tolist()) | set(edge_v.tolist()))
    used = col_emb[torch.tensor(used_cols, dtype=torch.long)]
    h_norm_sq_mean = float((used ** 2).sum(dim=-1).mean().item())
    energy_normalized = energy_per_edge / max(h_norm_sq_mean, 1e-12)

    return {
        "energy_total": float(energy_total),
        "energy_per_edge": float(energy_per_edge),
        "energy_normalized": float(energy_normalized),
        "num_edges": int(num_edges),
        "h_norm_sq_mean": h_norm_sq_mean,
    }


def compute_mad_columns(
    col_emb: torch.Tensor,
    cb_edge: torch.Tensor,
    max_pairs_sample: int = 50000,
) -> Optional[Dict[str, float]]:
    """MAD = mean(1 - cos sim) over column pairs i<j.

    Returns: {mad_overall, mad_intra_table, mad_inter_table, n_pairs_total, n_intra, n_inter}.
    For graphs with > max_pairs_sample inter pairs, uniformly subsample inter-table pairs
    (intra-table는 보통 작아 그대로 사용).
    """
    if col_emb is None or col_emb.numel() == 0 or cb_edge.numel() == 0:
        return None
    n = col_emb.size(0)
    if n < 2:
        return None
    normed = F.normalize(col_emb, dim=-1)
    sim_matrix = normed @ normed.T  # [N, N]
    dist_matrix = 1.0 - sim_matrix
    iu = torch.triu_indices(n, n, offset=1)
    all_dists = dist_matrix[iu[0], iu[1]]
    mad_overall = float(all_dists.mean().item())

    col_to_tab, _ = _build_col_table_index(cb_edge)
    tab_arr = torch.tensor([col_to_tab.get(i, -1) for i in range(n)], dtype=torch.long)
    same_table = (tab_arr.unsqueeze(0) == tab_arr.unsqueeze(1)) & (tab_arr.unsqueeze(0) >= 0)
    same_table_upper = same_table[iu[0], iu[1]]
    intra_dists = all_dists[same_table_upper]
    inter_dists = all_dists[~same_table_upper]

    mad_intra = float(intra_dists.mean().item()) if intra_dists.numel() > 0 else float("nan")
    mad_inter = float(inter_dists.mean().item()) if inter_dists.numel() > 0 else float("nan")

    return {
        "mad_overall": mad_overall,
        "mad_intra_table": mad_intra,
        "mad_inter_table": mad_inter,
        "n_pairs_total": int(iu.shape[1]),
        "n_intra": int(intra_dists.numel()),
        "n_inter": int(inter_dists.numel()),
    }


# ────────────────────────────────────────────────────────────────────────
# Per-cell analyzer
# ────────────────────────────────────────────────────────────────────────

def analyze_one_cell(
    cell: Dict[str, Any],
    dev: List[Dict],
    qids: List[int],
    encoder: LocalPLMEncoder,
    builder: EnrichedHeteroGraphBuilder,
    db_graph_cache: Dict[str, Tuple[Any, Dict]],
) -> Dict[str, Any]:
    print("=" * 90)
    print(f"Cell [{cell['tag']}] — {cell['label']}")
    print(f"  ckpt: {cell['ckpt_path']}")
    print(f"  reported R@15: {cell['r15_reported']:.4f}")
    print("=" * 90)

    ckpt_path = ROOT / cell["ckpt_path"]
    if not ckpt_path.exists():
        print(f"  ✗ ckpt missing: {ckpt_path}")
        return {"missing": True, "reason": "ckpt_missing", "tag": cell["tag"]}

    try:
        model = load_model_for_cell(cell)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ✗ model load fail: {e}")
        return {"missing": True, "reason": f"load_fail: {e!r}", "tag": cell["tag"]}

    num_layers_max = len(model.convs)
    layer_names: List[str] = ["L0_PLM"] + [f"L{i+1}_GAT" for i in range(num_layers_max)] + ["L_out"]

    # Per-layer aggregators (across queries)
    sims_by_layer: List[List[float]] = [[] for _ in layer_names]  # intra_table_sims (existing L_GAT cos sim)
    dirichlet_by_layer: List[List[Dict]] = [[] for _ in layer_names]
    mad_by_layer: List[List[Dict]] = [[] for _ in layer_names]

    entropy_by_layer: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    topk_by_layer: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    per_query_records: List[Dict] = []
    n_skipped = 0
    n_attn_fail = 0
    t_start = time.time()
    for idx, qid in enumerate(qids):
        item = dev[qid]
        db_id = item["db_id"]
        question = item["question"]

        if db_id not in db_graph_cache:
            try:
                graph_data, metadata = builder.build(db_id=db_id, db_dir=str(DEV_DB_DIR))
            except Exception as e:
                print(f"  [qid={qid}] graph build fail: {e}")
                n_skipped += 1
                continue
            db_graph_cache[db_id] = (graph_data, metadata)
        graph_data, metadata = db_graph_cache[db_id]
        data = graph_data.clone()

        try:
            enc_result = encoder.encode([question])
            q_emb = enc_result[0] if isinstance(enc_result, tuple) else enc_result
            if q_emb.dim() == 3:
                q_emb = q_emb.mean(dim=1)
            if q_emb.dim() == 1:
                q_emb = q_emb.unsqueeze(0)
            q_emb = q_emb.to(DEVICE)
        except Exception as e:
            print(f"  [qid={qid}] encode fail: {e}")
            n_skipped += 1
            continue

        x_dict = {nt: x.to(DEVICE) for nt, x in data.x_dict.items()}
        edge_index_dict = {et: ei.to(DEVICE) for et, ei in data.edge_index_dict.items()}
        # V1+ cells (model_class='v2' with query_supernode=True) require query_node injection
        if cell["model_class"] == "v2" and getattr(model, "query_supernode", False):
            edge_dir = getattr(model, "supernode_edge_direction", "bidirectional")
            x_dict, edge_index_dict = inject_query_supernode(
                x_dict, edge_index_dict, q_emb, edge_direction=edge_dir
            )
        cb_edge = edge_index_dict.get(COL_TO_TAB_EDGE)
        if cb_edge is not None:
            cb_edge_cpu = cb_edge.cpu()
        else:
            cb_edge_cpu = None

        # ── Layer-wise forward ──
        try:
            layer_embs, n_active = extract_layerwise_via_hook(
                model, x_dict, edge_index_dict, query_emb=q_emb,
            )
        except Exception as e:
            print(f"  [qid={qid}] forward fail: {e}")
            import traceback; traceback.print_exc()
            n_skipped += 1
            continue

        # Layer count handling — for v2 cells with num_layers_mode != 'fixed',
        # n_active may differ from num_layers_max. Pad with empty layers if needed.
        if len(layer_embs) != len(layer_names):
            # rebuild layer_names to match actual capture
            layer_names = ["L0_PLM"] + [f"L{i+1}_GAT" for i in range(n_active)] + ["L_out"]
            if len(layer_names) != len(layer_embs):
                # Mismatch — fall back to generic names
                layer_names = [f"layer_{i}" for i in range(len(layer_embs))]

            # rebuild per-layer aggregators (preserve previous if same size; otherwise restart)
            if len(sims_by_layer) != len(layer_names):
                sims_by_layer = [[] for _ in layer_names]
                dirichlet_by_layer = [[] for _ in layer_names]
                mad_by_layer = [[] for _ in layer_names]

        # ── V6 metrics on column embeddings ──
        per_q_layer: Dict[str, Any] = {"qid": int(qid), "db_id": db_id,
                                       "difficulty": item.get("difficulty", "unknown")}
        for l, ed in enumerate(layer_embs):
            col_emb = ed.get("column")
            if col_emb is None or cb_edge_cpu is None:
                per_q_layer[f"L{l}"] = None
                continue
            col_emb = col_emb.float()  # safety cast
            # Existing L_GAT cos sim (intra_table_sims) — cross-check anchor
            sims = intra_table_sims(col_emb, cb_edge_cpu)
            sims_by_layer[l].extend(sims)
            # V6: Dirichlet
            de = compute_dirichlet_energy_columns(col_emb, cb_edge_cpu)
            if de is not None:
                dirichlet_by_layer[l].append(de)
            # V6: MAD
            ma = compute_mad_columns(col_emb, cb_edge_cpu)
            if ma is not None:
                mad_by_layer[l].append(ma)

            per_q_layer[layer_names[l]] = {
                "intra_table_sim_mean": float(np.mean(sims)) if sims else None,
                "dirichlet": de,
                "mad": ma,
            }

        per_query_records.append(per_q_layer)

        # ── Attention entropy (extract_layerwise_attention_v2) ──
        try:
            class _SimpleBatch: pass
            batch_proxy = _SimpleBatch()
            batch_proxy.x_dict = x_dict
            batch_proxy.edge_index_dict = edge_index_dict
            attn_res = extract_layerwise_attention_v2(
                model, batch_proxy, query_emb=q_emb, topk=5, return_raw=False
            )
            for layer_key, et_map in attn_res["entropy"].items():
                for et_str, v in et_map.items():
                    if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                        entropy_by_layer[layer_key][et_str].append(float(v))
            for layer_key, et_map in attn_res["topk_conc"].items():
                for et_str, v in et_map.items():
                    if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                        topk_by_layer[layer_key][et_str].append(float(v))
        except Exception as e:
            n_attn_fail += 1
            if n_attn_fail <= 2:
                print(f"  [qid={qid}] attention fail (#{n_attn_fail}): {e}")

        if (idx + 1) % 10 == 0:
            print(f"  progress: {idx+1}/{len(qids)} qs  elapsed={time.time()-t_start:.1f}s "
                  f"skipped={n_skipped} attn_fail={n_attn_fail}")

    print(f"  done. elapsed={time.time()-t_start:.1f}s skipped={n_skipped} attn_fail={n_attn_fail}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Aggregate over queries ──
    def _agg_scalar(vals: List[float]) -> Dict:
        if not vals:
            return {"n": 0, "mean": None, "std": None}
        a = np.array(vals, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return {"n": 0, "mean": None, "std": None}
        return {"n": int(a.size), "mean": float(a.mean()),
                "std": float(a.std()),
                "q25": float(np.percentile(a, 25)),
                "q50": float(np.percentile(a, 50)),
                "q75": float(np.percentile(a, 75))}

    def _agg_field(layer_dicts: List[List[Dict]], field: str) -> List[Dict]:
        return [_agg_scalar([d[field] for d in lst if d is not None and field in d
                             and d[field] is not None]) for lst in layer_dicts]

    intra_sim_layer = [_agg_scalar(s) for s in sims_by_layer]
    dirichlet_e_total = _agg_field(dirichlet_by_layer, "energy_total")
    dirichlet_e_per_edge = _agg_field(dirichlet_by_layer, "energy_per_edge")
    dirichlet_e_normalized = _agg_field(dirichlet_by_layer, "energy_normalized")
    mad_overall = _agg_field(mad_by_layer, "mad_overall")
    mad_intra = _agg_field(mad_by_layer, "mad_intra_table")
    mad_inter = _agg_field(mad_by_layer, "mad_inter_table")

    def _agg_attn_dict(d: Dict[str, Dict[str, List[float]]]) -> Dict:
        out: Dict[str, Dict[str, Dict]] = {}
        for layer_key, et_map in d.items():
            out[layer_key] = {}
            for et_str, vals in et_map.items():
                a = np.array(vals, dtype=float)
                if a.size == 0:
                    continue
                out[layer_key][et_str] = {
                    "n": int(a.size),
                    "mean": float(a.mean()),
                    "std": float(a.std()),
                }
        return out

    return {
        "tag": cell["tag"], "label": cell["label"],
        "r15_reported": cell["r15_reported"],
        "narrative": cell["narrative"],
        "num_layers_max": num_layers_max,
        "n_qids": len(qids),
        "n_skipped": n_skipped,
        "n_attn_fail": n_attn_fail,
        "layer_names": layer_names,
        "intra_table_sim_per_layer": intra_sim_layer,
        "dirichlet_energy_total_per_layer": dirichlet_e_total,
        "dirichlet_energy_per_edge_per_layer": dirichlet_e_per_edge,
        "dirichlet_energy_normalized_per_layer": dirichlet_e_normalized,
        "mad_overall_per_layer": mad_overall,
        "mad_intra_table_per_layer": mad_intra,
        "mad_inter_table_per_layer": mad_inter,
        "attention_entropy_per_layer_edge_type": _agg_attn_dict(entropy_by_layer),
        "topk_concentration_per_layer_edge_type": _agg_attn_dict(topk_by_layer),
        "per_query_records": per_query_records,
    }


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Encoder model: {ENCODER_MODEL_NAME}")
    dev = load_dev()
    qids, qid_to_db = stratified_qids(dev)
    print(f"Stratified {len(qids)} qids = 5/DB × 11 DBs (seed={RANDOM_SEED})")
    print(f"Cells: {len(CELLS)}")
    print()

    t0 = time.time()
    encoder = LocalPLMEncoder(model_name=ENCODER_MODEL_NAME)
    print(f"Encoder loaded in {time.time()-t0:.1f}s on {DEVICE}")
    builder = EnrichedHeteroGraphBuilder(
        plm_model_name=ENCODER_MODEL_NAME,
        tables_json_path=str(TABLES_JSON),
    )
    print(f"Builder ready (PLM={ENCODER_MODEL_NAME})")
    print()

    db_graph_cache: Dict[str, Tuple[Any, Dict]] = {}
    results: Dict[str, Dict] = {}
    overall_start = time.time()
    for ci, cell in enumerate(CELLS):
        print(f"\n[{ci+1}/{len(CELLS)}] starting {cell['tag']} (elapsed total={time.time()-overall_start:.1f}s)")
        res = analyze_one_cell(cell, dev, qids, encoder, builder, db_graph_cache)
        results[cell["tag"]] = res

    # ── Save outputs ──
    pq_path = OUT_DIR / f"v1_v5_retrospective_v6_metrics_per_query_{REPORT_DATE}.jsonl"
    with pq_path.open("w") as f:
        for tag, res in results.items():
            if res.get("missing"):
                continue
            for r in res.get("per_query_records", []):
                rec = {"cell_tag": tag, **r}
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"\n→ per-query jsonl: {pq_path}")

    # CSV summary — cell × layer rows (V6 metrics + cross-check)
    csv_path = OUT_DIR / f"v1_v5_retrospective_v6_metrics_{REPORT_DATE}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "cell_tag", "r15_reported", "num_layers_max",
            "layer_name",
            "intra_table_sim_mean", "intra_table_sim_std",
            "dirichlet_energy_per_edge_mean", "dirichlet_energy_per_edge_std",
            "dirichlet_energy_normalized_mean", "dirichlet_energy_normalized_std",
            "mad_overall_mean", "mad_overall_std",
            "mad_intra_table_mean", "mad_intra_table_std",
            "mad_inter_table_mean", "mad_inter_table_std",
            "n_query_records",
        ])
        for tag, res in results.items():
            if res.get("missing"):
                w.writerow([tag, "", "", "MISSING:" + res.get("reason", ""),
                            "", "", "", "", "", "", "", "", "", "", "", "", ""])
                continue
            ln = res["layer_names"]
            for li, layer in enumerate(ln):
                its = res["intra_table_sim_per_layer"][li]
                dep = res["dirichlet_energy_per_edge_per_layer"][li]
                den = res["dirichlet_energy_normalized_per_layer"][li]
                mo = res["mad_overall_per_layer"][li]
                mi = res["mad_intra_table_per_layer"][li]
                mn = res["mad_inter_table_per_layer"][li]
                def _f(s, k):
                    v = s.get(k)
                    return round(v, 4) if isinstance(v, (int, float)) else ""
                w.writerow([
                    tag, res["r15_reported"], res["num_layers_max"], layer,
                    _f(its, "mean"), _f(its, "std"),
                    _f(dep, "mean"), _f(dep, "std"),
                    _f(den, "mean"), _f(den, "std"),
                    _f(mo, "mean"), _f(mo, "std"),
                    _f(mi, "mean"), _f(mi, "std"),
                    _f(mn, "mean"), _f(mn, "std"),
                    its.get("n", 0),
                ])
    print(f"→ csv: {csv_path}")

    # JSON full summary (without per-query bulk)
    json_path = OUT_DIR / f"v1_v5_retrospective_v6_metrics_{REPORT_DATE}.json"
    summary_for_json = {
        tag: {k: v for k, v in res.items() if k != "per_query_records"}
        for tag, res in results.items()
    }
    summary_for_json["_meta"] = {
        "report_date": REPORT_DATE,
        "encoder": ENCODER_MODEL_NAME,
        "n_qids": len(qids),
        "samples_per_db": SAMPLES_PER_DB,
        "seed": RANDOM_SEED,
        "device": str(DEVICE),
        "n_cells": len(CELLS),
        "cells_available": [t for t, r in results.items() if not r.get("missing")],
        "cells_missing": [t for t, r in results.items() if r.get("missing")],
    }
    with json_path.open("w") as f:
        json.dump(summary_for_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"→ json: {json_path}")

    # ── Concise terminal summary ──
    print("\n" + "=" * 100)
    print("V1~V5 Retrospective V6 Metric Summary")
    print("=" * 100)
    print(f"{'cell_tag':25s} {'R@15':8s} {'NL':4s} {'L_GAT_max_sim':14s} "
          f"{'Dirichlet_per_edge_L_last':25s} {'MAD_intra_L_last':17s}")
    print("-" * 100)
    for tag, res in results.items():
        if res.get("missing"):
            print(f"{tag:25s} MISSING  reason={res.get('reason')}")
            continue
        nl = res["num_layers_max"]
        last_gat = f"L{nl}_GAT"
        last_idx = res["layer_names"].index(last_gat) if last_gat in res["layer_names"] else -2
        its = res["intra_table_sim_per_layer"][last_idx].get("mean")
        de = res["dirichlet_energy_per_edge_per_layer"][last_idx].get("mean")
        mi = res["mad_intra_table_per_layer"][last_idx].get("mean")
        its_s = f"{its:.4f}" if its is not None else "n/a"
        de_s = f"{de:.4f}" if de is not None else "n/a"
        mi_s = f"{mi:.4f}" if mi is not None else "n/a"
        print(f"{tag:25s} {res['r15_reported']:.4f}  {nl:<3d} {its_s:14s} {de_s:25s} {mi_s:17s}")

    print(f"\nTotal wall: {time.time()-overall_start:.1f}s")


if __name__ == "__main__":
    main()
