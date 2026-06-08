"""V6-W1 Phase 1 drop-in 3종 ablation — V6 metric 측정 (V6-W0 retrospective script reuse).

planning/oversmoothing/oversmoothing_v6_plan_2026-06-01.md §1 Phase 1 (🟢 활성 launch) +
DECISIONS 2026-06-01 §V6-W1 격하 retract + 활성 launch + §seed 정정 (2026-06-04 single-seed s11) 정합.

목적:
  V6-W1 의 17 single-seed (s11) cells (P1a PairNorm scale{0.5,1.0,2.0} + P1b GCNII IR α{0.05,0.1,0.2}
  + P1c JK {concat,max} + P1d combo + 병행 sweep {temp×3, hn×2, bceinfo×3}) 의 layer-wise V6 metric
  (Dirichlet energy + MAD + attention entropy + Top-5 concentration) 을 V6-W0 와 동일 55 q stratified
  subsample (seed=42 fixed measurement) 위 측정 — V6-W0 의 15 cells (V1~V5) 와 직접 비교 가능하도록.

reuse: src/analysis/v1_v5_retrospective_v6_metrics.py 의 측정 함수 그대로 import
  (compute_dirichlet_energy_columns / compute_mad_columns / extract_layerwise_via_hook /
   inject_query_supernode / analyze_one_cell / load_model_for_cell / stratified_qids 등).

산출:
  - outputs/analysis/v6_phase1_dropin_metrics_2026-06-04.csv (cell × layer rows)
  - outputs/analysis/v6_phase1_dropin_metrics_2026-06-04.json (full summary + attention entropy)
  - outputs/analysis/v6_phase1_dropin_metrics_per_query_2026-06-04.jsonl
"""
from __future__ import annotations

import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/hyeonjin/thesis_refactored")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse all measurement infra from the V6-W0 retrospective script
import analysis.v1_v5_retrospective_v6_metrics as v6  # noqa: E402

OUT_DIR = ROOT / "outputs/analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DATE = "2026-06-04"

# V6-W1 cells use SchemaHeteroGATv2 with new gat_layer_type + v6w1 params — extend whitelist.
v6.V2_KW_WHITELIST |= {
    "v6w1_pairnorm_scale", "v6w1_jk_mode", "capture_layerwise_outputs",
}

CKPT = "outputs/checkpoints"

# Canonical s11 cells — newest valid checkpoint per cell (root rerun where OOM-recovered, else v6_phase1/).
# r15_reported = best Val Recall@15 stored in checkpoint['recall'] (verified == log "Best Recall").
CELLS: List[Dict[str, Any]] = [
    # ── P1a PairNorm (scale sweep) ──
    {"tag": "P1a_pairnorm_s0.5", "label": "P1a PairNorm scale=0.5",
     "ckpt_path": f"{CKPT}/v6_phase1/best_gat_v6w1_p1a_pairnorm_scale0_5_s11.pt",
     "model_class": "v2", "r15_reported": 0.5643, "narrative": "P1a PairNorm scale 0.5"},
    {"tag": "P1a_pairnorm_s1.0", "label": "P1a PairNorm scale=1.0",
     "ckpt_path": f"{CKPT}/v6_phase1/best_gat_v6w1_p1a_pairnorm_scale1_0_s11.pt",
     "model_class": "v2", "r15_reported": 0.5651, "narrative": "P1a PairNorm scale 1.0"},
    {"tag": "P1a_pairnorm_s2.0", "label": "P1a PairNorm scale=2.0",
     "ckpt_path": f"{CKPT}/v6_phase1/best_gat_v6w1_p1a_pairnorm_scale2_0_s11.pt",
     "model_class": "v2", "r15_reported": 0.5631, "narrative": "P1a PairNorm scale 2.0"},
    # ── P1b GCNII IR (alpha sweep) ──
    {"tag": "P1b_gcnii_a0.05", "label": "P1b GCNII IR alpha=0.05",
     "ckpt_path": f"{CKPT}/v6_phase1/best_gat_v6w1_p1b_gcnii_alpha0_05_s11.pt",
     "model_class": "v2", "r15_reported": 0.5707, "narrative": "P1b GCNII IR alpha 0.05"},
    {"tag": "P1b_gcnii_a0.1", "label": "P1b GCNII IR alpha=0.1",
     "ckpt_path": f"{CKPT}/v6_phase1/best_gat_v6w1_p1b_gcnii_alpha0_1_s11.pt",
     "model_class": "v2", "r15_reported": 0.5715, "narrative": "P1b GCNII IR alpha 0.1"},
    {"tag": "P1b_gcnii_a0.2", "label": "P1b GCNII IR alpha=0.2",
     "ckpt_path": f"{CKPT}/v6_phase1/best_gat_v6w1_p1b_gcnii_alpha0_2_s11.pt",
     "model_class": "v2", "r15_reported": 0.5721, "narrative": "P1b GCNII IR alpha 0.2"},
    # ── P1c JK (concat / max) ──
    {"tag": "P1c_jk_concat", "label": "P1c JK concat",
     "ckpt_path": f"{CKPT}/best_gat_v6w1_p1c_jk_concat_s11.pt",
     "model_class": "v2", "r15_reported": 0.5732, "narrative": "P1c JK concat"},
    {"tag": "P1c_jk_max", "label": "P1c JK max",
     "ckpt_path": f"{CKPT}/best_gat_v6w1_p1c_jk_max_s11.pt",
     "model_class": "v2", "r15_reported": 0.5653, "narrative": "P1c JK max"},
    # ── P1d combo (PN+IR+JK) ──
    {"tag": "P1d_combo", "label": "P1d combo (PN s1.0 + IR a0.1 + JK concat)",
     "ckpt_path": f"{CKPT}/v6_phase1/best_gat_v6w1_p1d_combo_s11.pt",
     "model_class": "v2", "r15_reported": 0.5736, "narrative": "P1d PN+IR+JK combo"},
    # ── 병행 sweep: InfoNCE temperature ──
    {"tag": "sweep_temp0.05", "label": "sweep InfoNCE temp=0.05",
     "ckpt_path": f"{CKPT}/best_gat_v6w1_sweep_temp0_05_s11.pt",
     "model_class": "v1", "v1_kwargs": {"in_channels":384,"hidden_channels":256,"out_channels":256,"num_layers":3,"heads":4,"query_conditioned":True,"query_supernode":False}, "r15_reported": 0.5654, "narrative": "loss sweep temp 0.05"},
    {"tag": "sweep_temp0.1", "label": "sweep InfoNCE temp=0.1",
     "ckpt_path": f"{CKPT}/best_gat_v6w1_sweep_temp0_1_s11.pt",
     "model_class": "v1", "v1_kwargs": {"in_channels":384,"hidden_channels":256,"out_channels":256,"num_layers":3,"heads":4,"query_conditioned":True,"query_supernode":False}, "r15_reported": 0.5662, "narrative": "loss sweep temp 0.1"},
    {"tag": "sweep_temp0.2", "label": "sweep InfoNCE temp=0.2",
     "ckpt_path": f"{CKPT}/best_gat_v6w1_sweep_temp0_2_s11.pt",
     "model_class": "v1", "v1_kwargs": {"in_channels":384,"hidden_channels":256,"out_channels":256,"num_layers":3,"heads":4,"query_conditioned":True,"query_supernode":False}, "r15_reported": 0.5650, "narrative": "loss sweep temp 0.2"},
    # ── 병행 sweep: hard negative ──
    {"tag": "sweep_hn_on", "label": "sweep hard-negative on",
     "ckpt_path": f"{CKPT}/v6_phase1/best_gat_v6w1_sweep_hn_on_s11.pt",
     "model_class": "v1", "v1_kwargs": {"in_channels":384,"hidden_channels":256,"out_channels":256,"num_layers":3,"heads":4,"query_conditioned":True,"query_supernode":False}, "r15_reported": 0.5682, "narrative": "loss sweep hard-neg on"},
    {"tag": "sweep_hn_off", "label": "sweep hard-negative off",
     "ckpt_path": f"{CKPT}/best_gat_v6w1_sweep_hn_off_s11.pt",
     "model_class": "v1", "v1_kwargs": {"in_channels":384,"hidden_channels":256,"out_channels":256,"num_layers":3,"heads":4,"query_conditioned":True,"query_supernode":False}, "r15_reported": 0.5659, "narrative": "loss sweep hard-neg off"},
    # ── 병행 sweep: BCE:InfoNCE weight ──
    {"tag": "sweep_bce0.5/0.5", "label": "sweep BCE:InfoNCE 0.5:0.5",
     "ckpt_path": f"{CKPT}/v6_phase1/best_gat_v6w1_sweep_bceinfo_0_5__0_5_s11.pt",
     "model_class": "v1", "v1_kwargs": {"in_channels":384,"hidden_channels":256,"out_channels":256,"num_layers":3,"heads":4,"query_conditioned":True,"query_supernode":False}, "r15_reported": 0.5691, "narrative": "loss sweep BCE:InfoNCE 0.5:0.5"},
    {"tag": "sweep_bce0.7/0.3", "label": "sweep BCE:InfoNCE 0.7:0.3",
     "ckpt_path": f"{CKPT}/v6_phase1/best_gat_v6w1_sweep_bceinfo_0_7__0_3_s11.pt",
     "model_class": "v1", "v1_kwargs": {"in_channels":384,"hidden_channels":256,"out_channels":256,"num_layers":3,"heads":4,"query_conditioned":True,"query_supernode":False}, "r15_reported": 0.5654, "narrative": "loss sweep BCE:InfoNCE 0.7:0.3"},
    {"tag": "sweep_bce0.3/0.7", "label": "sweep BCE:InfoNCE 0.3:0.7",
     "ckpt_path": f"{CKPT}/best_gat_v6w1_sweep_bceinfo_0_3__0_7_s11.pt",
     "model_class": "v1", "v1_kwargs": {"in_channels":384,"hidden_channels":256,"out_channels":256,"num_layers":3,"heads":4,"query_conditioned":True,"query_supernode":False}, "r15_reported": 0.5685, "narrative": "loss sweep BCE:InfoNCE 0.3:0.7"},
]

# Excluded: P1c_jk_concat_s99 (recall=0.4654 @ epoch 2 — incomplete/killed run, not a valid 2nd seed).


def main():
    print(f"Device: {v6.DEVICE}")
    print(f"Encoder: {v6.ENCODER_MODEL_NAME}")
    dev = v6.load_dev()
    qids, qid_to_db = v6.stratified_qids(dev)
    print(f"Stratified {len(qids)} qids = 5/DB × 11 DBs (seed={v6.RANDOM_SEED}) — V6-W0 identical subsample")
    print(f"V6-W1 cells: {len(CELLS)}")
    print()

    t0 = time.time()
    encoder = v6.LocalPLMEncoder(model_name=v6.ENCODER_MODEL_NAME)
    print(f"Encoder loaded in {time.time()-t0:.1f}s on {v6.DEVICE}")
    builder = v6.EnrichedHeteroGraphBuilder(
        plm_model_name=v6.ENCODER_MODEL_NAME,
        tables_json_path=str(v6.TABLES_JSON),
    )
    print(f"Builder ready (PLM={v6.ENCODER_MODEL_NAME})\n")

    db_graph_cache: Dict[str, Any] = {}
    results: Dict[str, Dict] = {}
    overall_start = time.time()
    for ci, cell in enumerate(CELLS):
        print(f"\n[{ci+1}/{len(CELLS)}] {cell['tag']} (elapsed={time.time()-overall_start:.1f}s)")
        res = v6.analyze_one_cell(cell, dev, qids, encoder, builder, db_graph_cache)
        results[cell["tag"]] = res

    # ── Save outputs ──
    pq_path = OUT_DIR / f"v6_phase1_dropin_metrics_per_query_{REPORT_DATE}.jsonl"
    with pq_path.open("w") as f:
        for tag, res in results.items():
            if res.get("missing"):
                continue
            for r in res.get("per_query_records", []):
                f.write(json.dumps({"cell_tag": tag, **r}, ensure_ascii=False, default=str) + "\n")
    print(f"\n→ per-query jsonl: {pq_path}")

    csv_path = OUT_DIR / f"v6_phase1_dropin_metrics_{REPORT_DATE}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "cell_tag", "r15_reported", "num_layers_max", "layer_name",
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
                w.writerow([tag, "", "", "MISSING:" + res.get("reason", ""), *[""] * 13])
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

    json_path = OUT_DIR / f"v6_phase1_dropin_metrics_{REPORT_DATE}.json"
    summary = {tag: {k: v for k, v in res.items() if k != "per_query_records"}
               for tag, res in results.items()}
    summary["_meta"] = {
        "report_date": REPORT_DATE, "encoder": v6.ENCODER_MODEL_NAME,
        "n_qids": len(qids), "samples_per_db": v6.SAMPLES_PER_DB, "seed": v6.RANDOM_SEED,
        "device": str(v6.DEVICE), "n_cells": len(CELLS),
        "cells_available": [t for t, r in results.items() if not r.get("missing")],
        "cells_missing": [t for t, r in results.items() if r.get("missing")],
        "single_seed": "s11", "excluded": "P1c_jk_concat_s99 (epoch 2, incomplete)",
    }
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"→ json: {json_path}")

    # ── Terminal summary ──
    print("\n" + "=" * 100)
    print("V6-W1 drop-in ablation — Last-GAT-layer V6 metrics (single seed s11)")
    print("=" * 100)
    print(f"{'cell':22s} {'R@15':8s} {'NL':3s} {'L_GAT_sim':10s} {'MAD_intra':10s} {'MAD_overall':12s} {'Dirichlet_pe':12s}")
    for tag, res in results.items():
        if res.get("missing"):
            print(f"{tag:22s} MISSING {res.get('reason')}")
            continue
        nl = res["num_layers_max"]
        idx = res["layer_names"].index(f"L{nl}_GAT") if f"L{nl}_GAT" in res["layer_names"] else -2
        its = res["intra_table_sim_per_layer"][idx].get("mean")
        mi = res["mad_intra_table_per_layer"][idx].get("mean")
        mo = res["mad_overall_per_layer"][idx].get("mean")
        de = res["dirichlet_energy_per_edge_per_layer"][idx].get("mean")
        def s(x): return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"
        print(f"{tag:22s} {res['r15_reported']:.4f}  {nl:<3d} {s(its):10s} {s(mi):10s} {s(mo):12s} {s(de):12s}")
    print(f"\nTotal wall: {time.time()-overall_start:.1f}s")


if __name__ == "__main__":
    main()
