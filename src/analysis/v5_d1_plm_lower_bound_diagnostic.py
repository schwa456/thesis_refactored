"""V5-D-1 PLM Lower Bound Diagnostic.

근거:
  - planning/oversmoothing_v5_plan.md §4.4 (Direction D — PLM Lower Bound 파괴)
  - planning/oversmoothing_v5_plan.md §5.3 Tier 1 (PLM lower bound 파괴를 위한 column-specific embedding 재학습)

목적 (2 진단):
  1. anchor stack (DSN p80) 의 c_L0 (PLM 단계 intra-table cosine) + c_L3 (GAT layer 3 후) 정량
  2. Plain HeteroGraphBuilder vs EnrichedHeteroGraphBuilder 의 c_L0 비교 — Enriched 가 PLM lower bound 를
     얼마나 낮추는가? (V5-D-2 contrastive pre-training 의 ROI 정량 evidence)

Protocol:
  - Multi-DB stratified sampling: per_db=5, seed=42, 11 BIRD-Dev DBs → 55 queries
  - intra_table_sims: ('column', 'belongs_to', 'table') edge 의 dst-별 column embedding pairwise cosine sim
  - Plain vs Enriched: 같은 query (qid) 에서 두 builder cache 모두 사용 (`data/processed/dev_plm_graphs.pt`
    + `data/processed/dev_enriched_plm_graphs.pt`)

Outputs:
  - outputs/analysis/v5_d1_plm_lower_bound_diagnostic/
      ├── per_query_c_l0.csv               # qid, db, plain_c_l0, enriched_c_l0, delta
      ├── per_db_summary.csv               # db, plain_mean, plain_std, enriched_mean, enriched_std, delta
      ├── anchor_c_layers.csv              # qid, db, c_L0..c_L_out (DSN p80, enriched)
      ├── anchor_c_layers_per_db.csv       # per-DB layer-wise mean
      ├── summary.json                     # global metrics
      └── plots/
            ├── plain_vs_enriched_c_l0_per_db.png
            ├── plain_vs_enriched_c_l0_scatter.png
            └── anchor_c_layers_trajectory.png

CPU-friendly (1-2h on dev set 55 queries). V5 sweep 와 자원 충돌 없음.
"""
from __future__ import annotations

import os
import sys
import json
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

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

from analysis.dsn_phase1_deep_dive import build_stratified_qids, load_dev
from analysis.gat_bottleneck_analysis import intra_table_sims, COL_TO_TAB_EDGE
from analysis.dsn_oversmoothing_analysis import _build_model_dsn, extract_layerwise_dsn, _resolve_query_emb
from utils.logger import get_logger

logger = get_logger(__name__)

OUT_DIR = ROOT / "outputs/analysis/v5_d1_plm_lower_bound_diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

DEV_JSON = ROOT / "data/raw/BIRD_dev/dev.json"
PLAIN_CACHE = ROOT / "data/processed/dev_plm_graphs.pt"
ENRICHED_CACHE = ROOT / "data/processed/dev_enriched_plm_graphs.pt"

ANCHOR_CFG = ROOT / "configs/training/train_gat_directed_supernode_p80.yaml"
ANCHOR_CKPT = ROOT / "outputs/checkpoints/best_gat_directed_supernode_p80.pt"


# ──────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────

def load_cache(path: Path) -> list:
    logger.info(f"loading {path.name} ...")
    return torch.load(path, weights_only=False)


def inject_query_supernode(data):
    """BIRDSuperNodeDataset 의 핵심 logic 재현 (직전 PyG SuperNode 학습 wrap)."""
    from data.bird_dataset import BIRDSuperNodeDataset
    return BIRDSuperNodeDataset._inject_supernode(data) if hasattr(BIRDSuperNodeDataset, '_inject_supernode') else None


# ──────────────────────────────────────────────────────────────
# c_L0 measurement helper
# ──────────────────────────────────────────────────────────────

def measure_c_l0_one(data) -> Optional[float]:
    """단일 graph 의 column 노드 intra-table cosine sim 평균.
    data: HeteroData (PLM embedding 이 batch.x_dict['column'] 에 있음)
    """
    if 'column' not in data.node_types:
        return None
    if COL_TO_TAB_EDGE not in data.edge_index_dict:
        return None
    cb = data.edge_index_dict[COL_TO_TAB_EDGE]
    col_x = data['column'].x
    if col_x is None or col_x.size(0) == 0:
        return None
    sims = intra_table_sims(col_x, cb)
    if not sims:
        return None
    return float(np.mean(sims))


# ──────────────────────────────────────────────────────────────
# Diagnostic 1 — Anchor c_L0 / c_L3 (DSN p80 with Enriched)
# ──────────────────────────────────────────────────────────────

def diagnostic_1_anchor(enriched_graphs: list, qids: List[int],
                        qid_to_db: Dict[int, str]) -> Dict:
    """DSN p80 ckpt 으로 layer-wise intra-table cosine 측정.
    Enriched dataset 사용 (학습 시와 동일 builder).
    BIRDSuperNodeDataset wrap 적용 (DSN p80 의 query_supernode=true 와 동일).
    """
    logger.info("=" * 60)
    logger.info("Diagnostic 1 — Anchor c_L0 / c_L3 (DSN p80, Enriched + SuperNode)")
    logger.info("=" * 60)

    from data.bird_dataset import BIRDSuperNodeDataset

    # Build supernode dataset on top of enriched cache (in-memory wrap)
    class _ListDataset:
        def __init__(self, data_list):
            self.data_list = data_list
        def __len__(self):
            return len(self.data_list)
        def __getitem__(self, idx):
            return self.data_list[idx]
        def get(self, idx):
            return self.data_list[idx]

    base = _ListDataset(enriched_graphs)
    sn_dataset = BIRDSuperNodeDataset(base)

    # Load anchor model
    with open(ANCHOR_CFG) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cpu")
    model = _build_model_dsn(cfg, ANCHOR_CKPT, device)
    model.eval()
    n_layers = cfg["model"]["num_layers"]
    layer_names = ["L0_PLM"] + [f"L{i+1}_GAT" for i in range(n_layers)] + ["L_out"]

    per_query_records = []  # list of dict (qid, db, c_L0..c_L_out)
    skipped = 0
    for qid in qids:
        try:
            data = sn_dataset[qid]
        except Exception as e:
            logger.warning(f"  qid={qid} skip: {e}")
            skipped += 1
            continue
        db = qid_to_db.get(qid, "?")
        q_emb = _resolve_query_emb(data)
        try:
            embs = extract_layerwise_dsn(model, data, q_emb)
        except Exception as e:
            logger.warning(f"  qid={qid} forward fail: {e}")
            skipped += 1
            continue
        # measure intra-table cosine per layer
        cb = data.edge_index_dict[COL_TO_TAB_EDGE]
        rec = {"qid": qid, "db": db}
        for li, (name, emb) in enumerate(zip(layer_names, embs)):
            col_x = emb.get("column")
            if col_x is None or col_x.size(0) == 0:
                rec[name] = float("nan")
                continue
            sims = intra_table_sims(col_x, cb)
            rec[name] = float(np.mean(sims)) if sims else float("nan")
        per_query_records.append(rec)

    logger.info(f"  measured {len(per_query_records)} / {len(qids)} (skipped {skipped})")

    # global mean per layer
    layer_means = {}
    layer_stds = {}
    for name in layer_names:
        vals = [r[name] for r in per_query_records if not np.isnan(r.get(name, float("nan")))]
        layer_means[name] = float(np.mean(vals)) if vals else float("nan")
        layer_stds[name] = float(np.std(vals)) if vals else float("nan")

    # per-DB
    per_db = defaultdict(lambda: defaultdict(list))
    for r in per_query_records:
        for name in layer_names:
            v = r.get(name, float("nan"))
            if not np.isnan(v):
                per_db[r["db"]][name].append(v)
    per_db_summary = {db: {name: float(np.mean(vs)) for name, vs in d.items()} for db, d in per_db.items()}

    return {
        "layer_names": layer_names,
        "layer_means": layer_means,
        "layer_stds": layer_stds,
        "per_query": per_query_records,
        "per_db": per_db_summary,
        "n_queries": len(per_query_records),
    }


# ──────────────────────────────────────────────────────────────
# Diagnostic 2 — Plain vs Enriched c_L0 비교
# ──────────────────────────────────────────────────────────────

def diagnostic_2_plain_vs_enriched(plain_graphs: list, enriched_graphs: list,
                                    qids: List[int], qid_to_db: Dict[int, str]) -> Dict:
    """Same qids, two builders → c_L0 측정 비교.
    BIRDSuperNodeDataset wrap 안 함 (PLM-level cosine 만 필요, SuperNode 는 query_node 만 추가).
    """
    logger.info("=" * 60)
    logger.info("Diagnostic 2 — Plain vs Enriched c_L0 (PLM-level intra-table cosine)")
    logger.info("=" * 60)

    if len(plain_graphs) != len(enriched_graphs):
        logger.warning(f"  cache length mismatch: plain={len(plain_graphs)} vs "
                       f"enriched={len(enriched_graphs)} — clamped to min")

    n_min = min(len(plain_graphs), len(enriched_graphs))
    records = []
    for qid in qids:
        if qid >= n_min:
            continue
        db = qid_to_db.get(qid, "?")
        plain_c = measure_c_l0_one(plain_graphs[qid])
        enriched_c = measure_c_l0_one(enriched_graphs[qid])
        if plain_c is None or enriched_c is None:
            continue
        records.append({
            "qid": qid,
            "db": db,
            "plain_c_l0": plain_c,
            "enriched_c_l0": enriched_c,
            "delta": enriched_c - plain_c,
        })

    logger.info(f"  measured {len(records)} queries")

    # global summary
    plain_vals = [r["plain_c_l0"] for r in records]
    enriched_vals = [r["enriched_c_l0"] for r in records]
    deltas = [r["delta"] for r in records]

    summary = {
        "n_queries": len(records),
        "plain_mean": float(np.mean(plain_vals)),
        "plain_std": float(np.std(plain_vals)),
        "enriched_mean": float(np.mean(enriched_vals)),
        "enriched_std": float(np.std(enriched_vals)),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas)),
        "delta_median": float(np.median(deltas)),
    }

    # per-DB
    per_db = defaultdict(lambda: {"plain": [], "enriched": [], "delta": []})
    for r in records:
        per_db[r["db"]]["plain"].append(r["plain_c_l0"])
        per_db[r["db"]]["enriched"].append(r["enriched_c_l0"])
        per_db[r["db"]]["delta"].append(r["delta"])
    per_db_summary = {
        db: {
            "n": len(d["plain"]),
            "plain_mean": float(np.mean(d["plain"])),
            "plain_std": float(np.std(d["plain"])),
            "enriched_mean": float(np.mean(d["enriched"])),
            "enriched_std": float(np.std(d["enriched"])),
            "delta_mean": float(np.mean(d["delta"])),
            "delta_std": float(np.std(d["delta"])),
        } for db, d in per_db.items()
    }

    return {
        "summary": summary,
        "per_db": per_db_summary,
        "per_query": records,
    }


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def plot_plain_vs_enriched_per_db(diag2: Dict, save: Path):
    per_db = diag2["per_db"]
    dbs = sorted(per_db.keys(), key=lambda x: per_db[x]["plain_mean"])
    plain_means = [per_db[d]["plain_mean"] for d in dbs]
    plain_stds = [per_db[d]["plain_std"] for d in dbs]
    enriched_means = [per_db[d]["enriched_mean"] for d in dbs]
    enriched_stds = [per_db[d]["enriched_std"] for d in dbs]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dbs))
    w = 0.38
    ax.bar(x - w/2, plain_means, w, yerr=plain_stds, label="Plain HeteroGraphBuilder",
           color="#3357FF", alpha=0.85, capsize=3)
    ax.bar(x + w/2, enriched_means, w, yerr=enriched_stds, label="EnrichedHeteroGraphBuilder",
           color="#FF8C33", alpha=0.85, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(dbs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(r"$\bar{c}_{L_0}$ — Intra-table column cosine")
    ax.set_title("c_L0 per DB — Plain vs Enriched HeteroGraphBuilder (BIRD-Dev n=55 stratified)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save, dpi=120)
    plt.close()
    logger.info(f"  saved {save.name}")


def plot_plain_vs_enriched_scatter(diag2: Dict, save: Path):
    rec = diag2["per_query"]
    plain = [r["plain_c_l0"] for r in rec]
    enriched = [r["enriched_c_l0"] for r in rec]
    dbs = sorted(set(r["db"] for r in rec))
    db_colors = plt.cm.tab20(np.linspace(0, 1, len(dbs)))
    db_to_c = {d: db_colors[i] for i, d in enumerate(dbs)}

    fig, ax = plt.subplots(figsize=(8, 7))
    for r in rec:
        ax.scatter(r["plain_c_l0"], r["enriched_c_l0"], c=[db_to_c[r["db"]]], alpha=0.7, s=40)
    lim_lo = min(min(plain), min(enriched)) - 0.05
    lim_hi = max(max(plain), max(enriched)) + 0.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', alpha=0.4, label='y=x (no change)')
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Plain c_L0")
    ax.set_ylabel("Enriched c_L0")
    ax.set_title("Plain vs Enriched c_L0 (n=55 queries)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # DB legend (small)
    for d, c in db_to_c.items():
        ax.scatter([], [], c=[c], label=d, s=30)
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(save, dpi=120)
    plt.close()
    logger.info(f"  saved {save.name}")


def plot_anchor_layers(diag1: Dict, save: Path):
    layer_names = diag1["layer_names"]
    means = [diag1["layer_means"][n] for n in layer_names]
    stds = [diag1["layer_stds"][n] for n in layer_names]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(layer_names))
    ax.errorbar(x, means, yerr=stds, fmt='-o', color='#D62728', lw=2, markersize=8, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.set_ylabel(r"$\bar{c}$ — Intra-table column cosine")
    ax.set_title(f"Anchor stack (DSN p80) layer-wise intra-table cosine — n={diag1['n_queries']}")
    ax.set_ylim(0.4, 1.02)
    ax.axhline(y=0.51, color='gray', lw=1, ls='--', alpha=0.4, label='Phase 1 baseline c_L0 ≈ 0.51')
    ax.axhline(y=0.30, color='green', lw=1, ls='--', alpha=0.4, label='V5-D-2 target c_L0 ≤ 0.30')
    ax.grid(True, alpha=0.3)
    ax.legend()
    # value annotation
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.annotate(f'{m:.4f}', xy=(xi, m), xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save, dpi=120)
    plt.close()
    logger.info(f"  saved {save.name}")


# ──────────────────────────────────────────────────────────────
# CSV writers
# ──────────────────────────────────────────────────────────────

def write_csv_diag1(diag1: Dict, out_dir: Path):
    layer_names = diag1["layer_names"]
    # per-query
    p = out_dir / "anchor_c_layers.csv"
    with open(p, 'w') as f:
        f.write("qid,db," + ",".join(layer_names) + "\n")
        for r in diag1["per_query"]:
            row = [str(r["qid"]), r["db"]] + [f'{r.get(n, float("nan")):.6f}' for n in layer_names]
            f.write(",".join(row) + "\n")
    logger.info(f"  wrote {p}")
    # per-DB
    p = out_dir / "anchor_c_layers_per_db.csv"
    with open(p, 'w') as f:
        f.write("db," + ",".join(layer_names) + "\n")
        for db, d in sorted(diag1["per_db"].items()):
            row = [db] + [f'{d.get(n, float("nan")):.6f}' for n in layer_names]
            f.write(",".join(row) + "\n")
    logger.info(f"  wrote {p}")


def write_csv_diag2(diag2: Dict, out_dir: Path):
    # per-query
    p = out_dir / "per_query_c_l0.csv"
    with open(p, 'w') as f:
        f.write("qid,db,plain_c_l0,enriched_c_l0,delta\n")
        for r in diag2["per_query"]:
            f.write(f'{r["qid"]},{r["db"]},{r["plain_c_l0"]:.6f},'
                    f'{r["enriched_c_l0"]:.6f},{r["delta"]:.6f}\n')
    logger.info(f"  wrote {p}")
    # per-DB
    p = out_dir / "per_db_summary.csv"
    with open(p, 'w') as f:
        f.write("db,n,plain_mean,plain_std,enriched_mean,enriched_std,delta_mean,delta_std\n")
        for db, d in sorted(diag2["per_db"].items(), key=lambda x: x[1]["plain_mean"]):
            f.write(f'{db},{d["n"]},{d["plain_mean"]:.6f},{d["plain_std"]:.6f},'
                    f'{d["enriched_mean"]:.6f},{d["enriched_std"]:.6f},'
                    f'{d["delta_mean"]:.6f},{d["delta_std"]:.6f}\n')
    logger.info(f"  wrote {p}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(per_db: int = 5, seed: int = 42, skip_anchor: bool = False):
    # Load both caches
    plain_graphs = load_cache(PLAIN_CACHE)
    enriched_graphs = load_cache(ENRICHED_CACHE)

    # Stratified qids
    dev = load_dev()
    qids, qid_to_db = build_stratified_qids(dev, per_db=per_db, seed=seed)
    logger.info(f"sampled {len(qids)} qids across {len(set(qid_to_db.values()))} DBs (per_db={per_db}, seed={seed})")

    # Diagnostic 2 first (cheap — pure CPU, no GAT forward)
    diag2 = diagnostic_2_plain_vs_enriched(plain_graphs, enriched_graphs, qids, qid_to_db)
    logger.info(f"  global: plain c_L0={diag2['summary']['plain_mean']:.4f}±{diag2['summary']['plain_std']:.4f}, "
                f"enriched c_L0={diag2['summary']['enriched_mean']:.4f}±{diag2['summary']['enriched_std']:.4f}, "
                f"Δ={diag2['summary']['delta_mean']:+.4f}")
    write_csv_diag2(diag2, OUT_DIR)
    plot_plain_vs_enriched_per_db(diag2, OUT_DIR / "plots" / "plain_vs_enriched_c_l0_per_db.png")
    plot_plain_vs_enriched_scatter(diag2, OUT_DIR / "plots" / "plain_vs_enriched_c_l0_scatter.png")

    # Diagnostic 1 — anchor c_L0 / c_L3
    if skip_anchor:
        diag1 = None
        logger.info("Skipping Diagnostic 1 (anchor forward)")
    else:
        diag1 = diagnostic_1_anchor(enriched_graphs, qids, qid_to_db)
        logger.info(f"  anchor: c_L0={diag1['layer_means'].get('L0_PLM', float('nan')):.4f}, "
                    f"c_L1_GAT={diag1['layer_means'].get('L1_GAT', float('nan')):.4f}, "
                    f"c_L3_GAT={diag1['layer_means'].get('L3_GAT', float('nan')):.4f}, "
                    f"c_L_out={diag1['layer_means'].get('L_out', float('nan')):.4f}")
        write_csv_diag1(diag1, OUT_DIR)
        plot_anchor_layers(diag1, OUT_DIR / "plots" / "anchor_c_layers_trajectory.png")

    # summary.json
    summary = {
        "protocol": {
            "per_db": per_db,
            "seed": seed,
            "n_qids": len(qids),
            "n_dbs": len(set(qid_to_db.values())),
            "dev_json": str(DEV_JSON),
            "plain_cache": str(PLAIN_CACHE),
            "enriched_cache": str(ENRICHED_CACHE),
            "anchor_ckpt": str(ANCHOR_CKPT),
            "anchor_config": str(ANCHOR_CFG),
        },
        "diagnostic_2_plain_vs_enriched": diag2["summary"] if diag2 else None,
        "diagnostic_2_per_db": diag2["per_db"] if diag2 else None,
        "diagnostic_1_anchor": {
            "layer_means": diag1["layer_means"],
            "layer_stds": diag1["layer_stds"],
            "n_queries": diag1["n_queries"],
        } if diag1 else None,
        "diagnostic_1_per_db": diag1["per_db"] if diag1 else None,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"  wrote {OUT_DIR / 'summary.json'}")

    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_db", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_anchor", action="store_true",
                    help="Skip Diagnostic 1 (anchor forward) — Diagnostic 2 만 측정")
    args = ap.parse_args()
    main(per_db=args.per_db, seed=args.seed, skip_anchor=args.skip_anchor)
