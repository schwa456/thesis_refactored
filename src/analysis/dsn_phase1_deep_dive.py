"""DSN Phase 1 후속 3 deep dive: A1 v2_LN decomposition + A2 noise sensitivity + A3 per-DB stratified.

근거: planning/DECISIONS.md 2026-05-08 §1(b) (사용자 결정 A+B 통합)
선행: dsn_mitigation_v2_results.md (7-trial mech(ii) 5/5 dominant)

A1: v2_LN 의 LayerNormGATv2Conv raw alpha tensor capture (forward hook)
    - per-head magnitude / variance / sign distribution (softmax pre/post)
    - alpha → message → aggregation 단계별 decompose
    - L1_GAT cosine 1.0 collapse 가 어디서 발생하는지

A2: 7 ckpt × Gaussian noise (σ ∈ {0.01, 0.05, 0.1}) on column 노드 input
    - alpha tensor variance 측정 (clean vs noisy)
    - v2_LN noise robustness vs Phase 2/3

A3: 55 stratified queries (5 per DB × 11 DBs, seed=42) → 7 ckpt × 4 mechanism
    - mech(ii) 11 DBs invariance
    - toxicology vs european_football_2 mech(ii) 차이

산출물:
  outputs/analysis/dsn_phase1_deep_dive/
    a1_v2_layernorm_decomposition.json + plots
    a2_noise_sensitivity_7ckpt.json
    a3_per_db_stratified_7ckpt.json
"""
from __future__ import annotations

import sys
import json
import yaml
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_geometric.loader import DataLoader
from data.bird_dataset import BIRDGraphDataset, BIRDSuperNodeDataset
from modules.builders.graph_builder import HeteroGraphBuilder, EnrichedHeteroGraphBuilder
from modules.encoders.local_encoder import LocalPLMEncoder
from models.gat_network import SchemaHeteroGAT
from models.gat_network_v2 import SchemaHeteroGATv2
from analysis.extract_layerwise_attention_v2 import (
    extract_layerwise_attention_v2, AttentionCapture,
)
from analysis.gat_bottleneck_analysis import (
    intra_table_sims, COL_TO_TAB_EDGE,
)
from utils.logger import get_logger

# Reuse 7-trial CKPTS + builders
from analysis.dsn_mitigation_v2_7trial import (
    CKPTS, build_dataset, build_model, _resolve_query_emb,
    compute_gradient_flow_compat, extract_layerwise_via_hook,
    analyze_top5_raw_cosine,
)

logger = get_logger(__name__)

OUT_DIR = ROOT / "outputs/analysis/dsn_phase1_deep_dive"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEV_JSON = ROOT / "data/raw/BIRD_dev/dev.json"


# ──────────────────────────────────────────────────────────────
# Helpers — stratified sampling, qid → db lookup
# ──────────────────────────────────────────────────────────────

def load_dev() -> List[Dict]:
    with open(DEV_JSON, "r") as f:
        return json.load(f)


def build_stratified_qids(dev: List[Dict], per_db: int = 5,
                          seed: int = 42) -> Tuple[List[int], Dict[int, str]]:
    """Per-DB stratified sample. seed 고정 reproducibility."""
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


# ──────────────────────────────────────────────────────────────
# A1: v2_LN raw alpha decomposition
# ──────────────────────────────────────────────────────────────

def find_v2_layernorm_ckpt() -> dict:
    for c in CKPTS:
        if c["name"] == "v2_layernorm":
            return c
    raise RuntimeError("v2_layernorm ckpt def not found")


def find_phase2_ckpt() -> dict:
    for c in CKPTS:
        if c["name"] == "phase2_b8":
            return c
    raise RuntimeError("phase2_b8 ckpt def not found")


class RawAlphaCapture:
    """LayerNormGATv2Conv 의 raw alpha (pre-LayerNorm + pre-softmax) capture.

    GATv2Conv 의 edge_update 가 raw alpha 산출 — LayerNormGATv2Conv 는 그 결과에 LN 적용.
    `edge_update` (pre-LN raw) + `edge_update_post_ln` (post-LN, pre-softmax) + `aggregate`
    output (post-softmax) 모두 caputre.

    단순화: LayerNormGATv2Conv 의 forward 안에서 alpha_layernorm 모듈 결과 차이를 보면 됨.
    여기서는 forward hook 으로 conv.alpha_layernorm 의 input/output 동시 capture.
    """
    def __init__(self, model):
        self.model = model
        # layer × edge_type → list of (raw_alpha [E,heads], normed_alpha [E,heads])
        self.captures: List[Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]]] = []
        self._handles = []

    def __enter__(self):
        self.captures = [{} for _ in range(self.model.num_layers)]
        self._handles = []
        for layer_idx, hetero in enumerate(self.model.convs):
            inner = getattr(hetero, "convs", None)
            if inner is None:
                continue
            for et, conv in inner.items():
                ln = getattr(conv, "alpha_layernorm", None)
                if ln is None:
                    continue
                self._wrap(ln, layer_idx, et)
        return self

    def _wrap(self, ln_module, layer_idx, et):
        captures = self.captures

        def _hook(mod, inp, out):
            # inp: tuple, [E, heads] 형태 — pre-LN raw alpha
            raw = inp[0].detach().cpu()
            normed = out.detach().cpu()
            captures[layer_idx][et] = (raw, normed)

        h = ln_module.register_forward_hook(_hook)
        self._handles.append(h)

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles = []


def analyze_a1_v2_layernorm(qids: List[int], qid_to_db: Dict[int, str]) -> Dict[str, Any]:
    """v2_LN 의 raw alpha capture → per-head decomposition.

    같은 query 에서 phase2_b8 (no LN) vs v2_layernorm (with LN) 비교.
    pre-LN raw alpha distribution: magnitude / variance / sign
    post-LN normed alpha distribution: 동일
    softmax post (실제 attention) distribution: extract_layerwise_attention_v2 활용
    """
    logger.info("=" * 60)
    logger.info("A1: v2_LN raw alpha decomposition")
    logger.info("=" * 60)

    cv2 = find_v2_layernorm_ckpt()
    cp2 = find_phase2_ckpt()

    device = torch.device("cpu")

    # 같은 dataset 사용 (둘 다 query_supernode=true, EnrichedHeteroGraphBuilder)
    with open(cv2["config"]) as f:
        cfg_v2 = yaml.safe_load(f)
    with open(cp2["config"]) as f:
        cfg_p2 = yaml.safe_load(f)
    dataset = build_dataset(cfg_v2)
    model_v2 = build_model(cfg_v2, cv2["ckpt"], "v2", device)
    model_p2 = build_model(cfg_p2, cp2["ckpt"], "v2", device)

    # Aggregators (col→tab edge 위주)
    target_et = ("column", "belongs_to", "table")

    # v2_LN: pre-LN raw alpha vs post-LN normed alpha (per-head 통계)
    raw_alpha_stats: Dict[str, List[Dict[str, float]]] = {f"L{i+1}": [] for i in range(model_v2.num_layers)}
    normed_alpha_stats: Dict[str, List[Dict[str, float]]] = {f"L{i+1}": [] for i in range(model_v2.num_layers)}
    # post-softmax (실제 alpha) — extract_layerwise_attention_v2 활용
    post_softmax_stats: Dict[str, List[Dict[str, float]]] = {f"L{i+1}": [] for i in range(model_v2.num_layers)}
    p2_post_softmax_stats: Dict[str, List[Dict[str, float]]] = {f"L{i+1}": [] for i in range(model_p2.num_layers)}

    # 추가: layer-wise column embedding cosine sim (collapse 단계)
    col_cosine_v2: Dict[str, List[float]] = {f"L{i+1}_GAT": [] for i in range(model_v2.num_layers)}
    col_cosine_p2: Dict[str, List[float]] = {f"L{i+1}_GAT": [] for i in range(model_p2.num_layers)}

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    qid_set = set(qids[:30])  # A1 은 30 queries 만 (forward 빠름)

    for idx, batch in enumerate(loader):
        if idx not in qid_set:
            continue
        batch = batch.to(device)
        q_emb = _resolve_query_emb(batch)
        cb_edge = batch.edge_index_dict.get(COL_TO_TAB_EDGE)

        # === v2_LN forward with RawAlphaCapture + AttentionCapture ===
        try:
            with RawAlphaCapture(model_v2) as rcap:
                attn_v2 = extract_layerwise_attention_v2(
                    model_v2, batch, query_emb=q_emb, topk=5, return_raw=True)
            # rcap.captures: layer × edge_type → (raw, normed)
            for li, et_map in enumerate(rcap.captures):
                if target_et not in et_map:
                    continue
                raw, normed = et_map[target_et]
                # per-head stats
                lk = f"L{li+1}"
                raw_a = raw.numpy()  # [E, heads]
                normed_a = normed.numpy()
                if raw_a.size == 0:
                    continue
                # heads 평균 + per-head spread
                raw_alpha_stats[lk].append({
                    "raw_mean": float(raw_a.mean()),
                    "raw_std": float(raw_a.std()),
                    "raw_abs_mean": float(np.abs(raw_a).mean()),
                    "raw_max": float(raw_a.max()),
                    "raw_min": float(raw_a.min()),
                    "raw_pos_frac": float((raw_a > 0).mean()),
                    "raw_per_head_var": float(raw_a.var(axis=0).mean()),
                })
                normed_alpha_stats[lk].append({
                    "normed_mean": float(normed_a.mean()),
                    "normed_std": float(normed_a.std()),
                    "normed_abs_mean": float(np.abs(normed_a).mean()),
                    "normed_max": float(normed_a.max()),
                    "normed_min": float(normed_a.min()),
                    "normed_pos_frac": float((normed_a > 0).mean()),
                    "normed_per_head_var": float(normed_a.var(axis=0).mean()),
                })
            # post-softmax alpha
            for lk, et_map in attn_v2.get("topk_conc", {}).items():
                v = et_map.get("→".join(target_et))
                if v is not None:
                    post_softmax_stats[lk].append({"top5_conc": v,
                                                   "entropy": attn_v2["entropy"][lk].get(
                                                       "→".join(target_et), float("nan"))})
        except Exception as e:
            if idx in (qids[0], qids[1]):
                logger.warning(f"  [v2_LN q{idx}] {e}")

        # === phase2_b8 forward with AttentionCapture only (no LN) ===
        try:
            attn_p2 = extract_layerwise_attention_v2(
                model_p2, batch, query_emb=q_emb, topk=5, return_raw=True)
            for lk, et_map in attn_p2.get("topk_conc", {}).items():
                v = et_map.get("→".join(target_et))
                if v is not None:
                    p2_post_softmax_stats[lk].append({"top5_conc": v,
                                                      "entropy": attn_p2["entropy"][lk].get(
                                                          "→".join(target_et), float("nan"))})
        except Exception as e:
            if idx in (qids[0], qids[1]):
                logger.warning(f"  [p2 q{idx}] {e}")

        # === Layer-wise cosine sim on both models ===
        if cb_edge is not None:
            try:
                lv2 = extract_layerwise_via_hook(model_v2, batch, q_emb)
                for li, ed in enumerate(lv2[1:-1]):  # GAT layers
                    if "column" in ed:
                        col_cosine_v2[f"L{li+1}_GAT"].extend(intra_table_sims(ed["column"], cb_edge))
            except Exception:
                pass
            try:
                lp2 = extract_layerwise_via_hook(model_p2, batch, q_emb)
                for li, ed in enumerate(lp2[1:-1]):
                    if "column" in ed:
                        col_cosine_p2[f"L{li+1}_GAT"].extend(intra_table_sims(ed["column"], cb_edge))
            except Exception:
                pass

    # Aggregate
    def _agg_dict_list(d: Dict[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
        out = {}
        for lk, items in d.items():
            if not items:
                continue
            keys = items[0].keys()
            out[lk] = {k: float(np.mean([i[k] for i in items])) for k in keys}
        return out

    summary = {
        "n_queries": len(qid_set),
        "raw_alpha_stats": _agg_dict_list(raw_alpha_stats),
        "normed_alpha_stats": _agg_dict_list(normed_alpha_stats),
        "v2_post_softmax": _agg_dict_list(post_softmax_stats),
        "p2_post_softmax": _agg_dict_list(p2_post_softmax_stats),
        "col_cosine_v2": {lk: float(np.mean(vs)) if vs else float("nan")
                          for lk, vs in col_cosine_v2.items()},
        "col_cosine_p2": {lk: float(np.mean(vs)) if vs else float("nan")
                          for lk, vs in col_cosine_p2.items()},
    }

    with open(OUT_DIR / "a1_v2_layernorm_decomposition.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  → A1 saved")
    return summary


# ──────────────────────────────────────────────────────────────
# A2: Softmax noise sensitivity (7 ckpt × σ ∈ {0.01, 0.05, 0.1})
# ──────────────────────────────────────────────────────────────

def analyze_a2_noise_sensitivity(qids: List[int]) -> Dict[str, Any]:
    """7 ckpt × column input perturbation σ × alpha variance."""
    logger.info("=" * 60)
    logger.info("A2: Softmax noise sensitivity")
    logger.info("=" * 60)

    sigmas = [0.0, 0.01, 0.05, 0.1]
    target_et = ("column", "belongs_to", "table")
    device = torch.device("cpu")

    # 각 ckpt 별 측정
    summary: Dict[str, Any] = {}
    qid_set = set(qids[:25])  # 25 queries (속도 위해)

    for c in CKPTS:
        if not c["ckpt"].exists():
            continue
        with open(c["config"]) as f:
            cfg = yaml.safe_load(f)
        dataset = build_dataset(cfg)
        model = build_model(cfg, c["ckpt"], c["model_class"], device)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # σ 별 alpha measurement
        # per-(layer, sigma): list of (top5_conc, entropy)
        per_sigma: Dict[float, Dict[str, List[Dict[str, float]]]] = {
            s: defaultdict(list) for s in sigmas
        }

        for idx, batch in enumerate(loader):
            if idx not in qid_set:
                continue
            batch = batch.to(device)
            q_emb = _resolve_query_emb(batch)

            for sigma in sigmas:
                # Perturb column input
                if sigma > 0:
                    orig_col = batch.x_dict["column"].clone()
                    noise = torch.randn_like(orig_col) * sigma
                    batch["column"].x = orig_col + noise

                try:
                    attn = extract_layerwise_attention_v2(
                        model, batch, query_emb=q_emb, topk=5, return_raw=False)
                    for lk, et_map in attn.get("topk_conc", {}).items():
                        v = et_map.get("→".join(target_et))
                        e = attn.get("entropy", {}).get(lk, {}).get("→".join(target_et))
                        if v is not None and not np.isnan(v):
                            per_sigma[sigma][lk].append({
                                "top5_conc": float(v),
                                "entropy": float(e) if e is not None and not np.isnan(e) else float("nan"),
                            })
                except Exception:
                    pass

                # Restore
                if sigma > 0:
                    batch["column"].x = orig_col

        # Aggregate σ × layer
        ckpt_summary = {}
        for sigma in sigmas:
            ckpt_summary[f"sigma_{sigma}"] = {}
            for lk, items in per_sigma[sigma].items():
                if not items:
                    continue
                top5 = [i["top5_conc"] for i in items]
                ent = [i["entropy"] for i in items if not np.isnan(i["entropy"])]
                ckpt_summary[f"sigma_{sigma}"][lk] = {
                    "top5_conc_mean": float(np.mean(top5)) if top5 else float("nan"),
                    "top5_conc_std": float(np.std(top5)) if top5 else float("nan"),
                    "entropy_mean": float(np.mean(ent)) if ent else float("nan"),
                    "entropy_std": float(np.std(ent)) if ent else float("nan"),
                    "n": len(items),
                }
        # Δ noise sensitivity: σ=0.1 의 top5 mean - σ=0.0 의 top5 mean (절대 변동)
        clean = ckpt_summary.get("sigma_0.0", {})
        noisy = ckpt_summary.get("sigma_0.1", {})
        delta_top5: Dict[str, float] = {}
        for lk in clean:
            if lk in noisy:
                delta_top5[lk] = abs(noisy[lk]["top5_conc_mean"] - clean[lk]["top5_conc_mean"])

        summary[c["name"]] = {
            "label": c["label"],
            "best_val_recall": c["best_val_recall"],
            "n_queries": len(qid_set),
            "per_sigma": ckpt_summary,
            "delta_top5_at_sigma_01": delta_top5,
        }
        logger.info(f"  [{c['name']}] done — Δtop5(σ=0.1) L2 col→tab = {delta_top5.get('L2', float('nan')):.4f}")

    with open(OUT_DIR / "a2_noise_sensitivity_7ckpt.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("  → A2 saved")
    return summary


# ──────────────────────────────────────────────────────────────
# A3: Per-DB stratified (55 queries) × 7 ckpt × 4 mechanism
# ──────────────────────────────────────────────────────────────

def analyze_a3_per_db_stratified(qids: List[int],
                                  qid_to_db: Dict[int, str]) -> Dict[str, Any]:
    """7 ckpt × 55 queries × 4 mechanism + per-DB 분해."""
    logger.info("=" * 60)
    logger.info(f"A3: Per-DB stratified ({len(qids)} qids × 7 ckpt × 4 mech)")
    logger.info("=" * 60)

    target_et = ("column", "belongs_to", "table")
    target_et_str = "→".join(target_et)
    device = torch.device("cpu")
    qid_set = set(qids)

    summary: Dict[str, Any] = {}

    for c in CKPTS:
        if not c["ckpt"].exists():
            continue
        logger.info(f"  [{c['name']}] starting...")
        with open(c["config"]) as f:
            cfg = yaml.safe_load(f)
        dataset = build_dataset(cfg)
        model = build_model(cfg, c["ckpt"], c["model_class"], device)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        layer_names = ["L0_PLM"] + [f"L{i+1}_GAT" for i in range(model.num_layers)] + ["L_out"]
        # per-DB aggregators
        sims_by_layer_db: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))  # db → layer → list
        attn_top5_by_db: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))   # db → layer → list (col→tab)
        attn_ent_by_db: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        l0_intra_by_db: Dict[str, List[float]] = defaultdict(list)

        for idx, batch in enumerate(loader):
            if idx not in qid_set:
                continue
            db = qid_to_db[idx]
            batch = batch.to(device)
            q_emb = _resolve_query_emb(batch)
            cb_edge = batch.edge_index_dict.get(COL_TO_TAB_EDGE)

            # Mech (iv) L0
            if cb_edge is not None and "column" in batch.x_dict:
                try:
                    sims = intra_table_sims(batch.x_dict["column"], cb_edge)
                    l0_intra_by_db[db].extend(sims)
                except Exception:
                    pass

            # Step 2: layer-wise
            try:
                layer_embs = extract_layerwise_via_hook(model, batch, q_emb)
                if cb_edge is not None:
                    for li, ed in enumerate(layer_embs):
                        if "column" in ed:
                            ln_name = layer_names[li] if li < len(layer_names) else f"L{li}"
                            sims_by_layer_db[db][ln_name].extend(intra_table_sims(ed["column"], cb_edge))
            except Exception:
                pass

            # Step 3: attention col→tab
            try:
                attn = extract_layerwise_attention_v2(model, batch, query_emb=q_emb, topk=5)
                for lk, et_map in attn.get("topk_conc", {}).items():
                    v = et_map.get(target_et_str)
                    e = attn.get("entropy", {}).get(lk, {}).get(target_et_str)
                    if v is not None and not np.isnan(v):
                        attn_top5_by_db[db][lk].append(float(v))
                    if e is not None and not np.isnan(e):
                        attn_ent_by_db[db][lk].append(float(e))
            except Exception:
                pass

        # Aggregate per-DB
        per_db: Dict[str, Dict[str, Any]] = {}
        all_dbs = set(qid_to_db.values())
        for db in sorted(all_dbs):
            n_q = sum(1 for qid in qids if qid_to_db[qid] == db)
            entry: Dict[str, Any] = {"n_queries": n_q}
            # Layer-wise sims
            entry["layer_sims_mean"] = {
                ln: float(np.mean(sims_by_layer_db[db][ln])) if sims_by_layer_db[db].get(ln) else float("nan")
                for ln in layer_names
            }
            # col→tab attention
            entry["col_to_tab_top5"] = {
                lk: float(np.mean(attn_top5_by_db[db][lk])) if attn_top5_by_db[db].get(lk) else float("nan")
                for lk in attn_top5_by_db[db]
            }
            entry["col_to_tab_entropy"] = {
                lk: float(np.mean(attn_ent_by_db[db][lk])) if attn_ent_by_db[db].get(lk) else float("nan")
                for lk in attn_ent_by_db[db]
            }
            entry["l0_intra_table_mean"] = float(np.mean(l0_intra_by_db[db])) if l0_intra_by_db[db] else float("nan")
            per_db[db] = entry

        # Overall (모든 55 queries 결합)
        overall_sims: Dict[str, List[float]] = defaultdict(list)
        overall_top5: Dict[str, List[float]] = defaultdict(list)
        overall_ent: Dict[str, List[float]] = defaultdict(list)
        overall_l0: List[float] = []
        for db in all_dbs:
            for ln, vs in sims_by_layer_db[db].items():
                overall_sims[ln].extend(vs)
            for lk, vs in attn_top5_by_db[db].items():
                overall_top5[lk].extend(vs)
            for lk, vs in attn_ent_by_db[db].items():
                overall_ent[lk].extend(vs)
            overall_l0.extend(l0_intra_by_db[db])

        overall = {
            "n_queries": len(qids),
            "layer_sims_mean": {ln: float(np.mean(vs)) if vs else float("nan")
                                for ln, vs in overall_sims.items()},
            "col_to_tab_top5": {lk: float(np.mean(vs)) if vs else float("nan")
                                for lk, vs in overall_top5.items()},
            "col_to_tab_entropy": {lk: float(np.mean(vs)) if vs else float("nan")
                                    for lk, vs in overall_ent.items()},
            "l0_intra_table_mean": float(np.mean(overall_l0)) if overall_l0 else float("nan"),
        }

        summary[c["name"]] = {
            "label": c["label"],
            "best_val_recall": c["best_val_recall"],
            "model_class": c["model_class"],
            "overall": overall,
            "per_db": per_db,
        }
        logger.info(f"  [{c['name']}] overall L1_GAT={overall['layer_sims_mean'].get('L1_GAT', float('nan')):.4f}, "
                    f"L2 top5={overall['col_to_tab_top5'].get('L2', float('nan')):.4f}")

    with open(OUT_DIR / "a3_per_db_stratified_7ckpt.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("  → A3 saved")
    return summary


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_db", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analyses", nargs="+", default=["A1", "A2", "A3"],
                        choices=["A1", "A2", "A3"])
    args = parser.parse_args()

    dev = load_dev()
    qids, qid_to_db = build_stratified_qids(dev, per_db=args.per_db, seed=args.seed)
    logger.info(f"Stratified qids: n={len(qids)} ({args.per_db}/DB × 11 DBs, seed={args.seed})")

    if "A1" in args.analyses:
        analyze_a1_v2_layernorm(qids, qid_to_db)
    if "A2" in args.analyses:
        analyze_a2_noise_sensitivity(qids)
    if "A3" in args.analyses:
        analyze_a3_per_db_stratified(qids, qid_to_db)

    logger.info(f"\nAll analyses complete → {OUT_DIR}")


if __name__ == "__main__":
    main()
