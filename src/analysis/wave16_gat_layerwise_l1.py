"""Wave 16 — Qwen3 GAT Best Ckpt Layer-wise L1 Saturation + Top-5 Attention Concentration.

DECISIONS 2026-05-21 (Wave 16) §7.4 + 자매 리포트 wave16_encoder_backbone_m4_2026-05-22.md
§5 §7.5 matrix "L1 saturation row pending — post-measurement extension" 정합.

목적:
  1) 최적 Qwen3 ckpt (best_gat_enriched_qwen3_0.6b_qcond.pt, ep281, R@15=0.6030) 의 layer-wise
     intra-table cosine similarity (= L1 saturation) 측정 — Wave 5 V-3-ext L1=1.0 collapse
     pattern retain 여부 확인.
  2) Top-5 attention concentration + entropy per-layer/edge-type — Wave 5 V-3-ext top5_conc
     pattern retain 여부 확인.
  3) MiniLM nl3 baseline (best_gat_qcond_nl3.pt, ep59, R@15=0.6061) 와 동일 metric 비교 — PLM
     backbone swap 의 학습 dynamics 영향 정량 (Wave 16 scenario 1 marginal sub-noise
     finding 의 mechanism-level cross-evidence).

Spec:
  - 데이터: BIRD-Dev (1534q × 11 DB) 위 stratified 5/DB × 11 = **55 queries** (seed=42)
  - Builder: EnrichedHeteroGraphBuilder (양 ckpt 모두 enriched 학습)
  - Encoder: Qwen3 ckpt → LocalPLMEncoder("Qwen/Qwen3-Embedding-0.6B"), MiniLM ckpt → default
  - Model: SchemaHeteroGAT (v1, query_conditioned=True, num_layers=3, heads=4, hidden=256)
  - Forward: per-query (on-the-fly) — Qwen3 dev cache symlink target 부재 위 매번 encode
  - Layer-wise:
      L0_PLM    = input PLM features (post lin_dict, pre conv)
      L1_GAT    = HeteroConv layer 1 output (post F.elu)
      L2_GAT    = HeteroConv layer 2 output
      L3_GAT    = HeteroConv layer 3 output
      L_out     = out_lin_dict + skip (final)
  - Metric:
      intra_table_sims(col_embs, cb_edge_index) → per-table mean cosine similarity (gat_bottleneck_analysis)
      L1 saturation indicator = mean over all (q, table) intra-table sim, 0..1, 1.0 = collapse
      Top-5 attention concentration = sum(top-5 alpha)/sum(all alpha) per dst-node, mean over dst+query

산출:
  - notebooks/analysis_results/wave16_gat_layerwise_l1_2026-05-22.md
  - outputs/analysis/wave16_gat_layerwise_l1_2026-05-22.csv (per-ckpt × per-layer rows)
  - outputs/analysis/wave16_gat_layerwise_l1_2026-05-22.json (full summary + per-query raw)
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
from analysis.extract_layerwise_attention_v2 import extract_layerwise_attention_v2  # noqa: E402
from analysis.gat_bottleneck_analysis import intra_table_sims, COL_TO_TAB_EDGE  # noqa: E402

DEV_JSON = ROOT / "data/raw/BIRD_dev/dev.json"
DEV_DB_DIR = ROOT / "data/raw/BIRD_dev/dev_databases"

OUT_DIR = ROOT / "outputs/analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPTS = [
    {
        "tag": "qwen3",
        "label": "Qwen3-Embedding-0.6B (1024-dim, 600M) + QCond GAT",
        "ckpt_path": ROOT / "outputs/checkpoints/best_gat_enriched_qwen3_0.6b_qcond.pt",
        "encoder_model_name": "Qwen/Qwen3-Embedding-0.6B",
        "in_channels": 1024, "hidden_channels": 256, "out_channels": 256,
        "num_layers": 3, "heads": 4, "query_conditioned": True,
        "best_epoch": 281, "best_r15": 0.6030,
    },
    {
        "tag": "minilm_nl3",
        "label": "all-MiniLM-L6-v2 (384-dim, 22M) + QCond GAT (Wave 6 anchor base)",
        "ckpt_path": ROOT / "outputs/checkpoints/best_gat_qcond_nl3.pt",
        "encoder_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "in_channels": 384, "hidden_channels": 256, "out_channels": 256,
        "num_layers": 3, "heads": 4, "query_conditioned": True,
        "best_epoch": 59, "best_r15": 0.6061,
    },
]

SAMPLES_PER_DB = 5
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_model(c: Dict) -> SchemaHeteroGAT:
    model = SchemaHeteroGAT(
        in_channels=c["in_channels"],
        hidden_channels=c["hidden_channels"],
        out_channels=c["out_channels"],
        num_layers=c["num_layers"],
        heads=c["heads"],
        query_conditioned=c["query_conditioned"],
    ).to(DEVICE)
    raw = torch.load(c["ckpt_path"], map_location=DEVICE, weights_only=False)
    state = raw["gat_state_dict"] if isinstance(raw, dict) and "gat_state_dict" in raw else raw
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [load] {c['tag']}: missing={len(missing)}, unexpected={len(unexpected)} "
              f"(first missing: {missing[:3]})")
    model.eval()
    return model


def extract_layerwise_via_hook(model, x_dict, edge_index_dict, query_emb
                               ) -> List[Dict[str, torch.Tensor]]:
    """Return list of per-layer embeddings:
       [L0_PLM_raw, after_input_proj, after_conv_L1, after_conv_L2, after_conv_L3, L_out_final]
       — 본 분석 위 L0_PLM_raw + L1~L3_GAT + L_out 만 사용."""
    embeddings: List[Dict[str, torch.Tensor]] = []
    embeddings.append({nt: x.detach().clone().cpu() for nt, x in x_dict.items()})  # L0_PLM (pre-projection)

    captured: List[Dict[str, torch.Tensor]] = []

    def _hook(module, inputs, output):
        if isinstance(output, dict):
            captured.append({nt: x.detach().clone().cpu() for nt, x in output.items()})

    handles = [model.convs[i].register_forward_hook(_hook) for i in range(model.num_layers)]
    try:
        with torch.no_grad():
            final = model(x_dict, edge_index_dict, query_emb=query_emb)
    finally:
        for h in handles:
            h.remove()

    for layer_out in captured:
        # HeteroConv output, post-F.elu in model.forward — apply ELU to capture
        embeddings.append({nt: F.elu(x).detach().clone() for nt, x in layer_out.items()})
    embeddings.append({nt: x.detach().clone().cpu() for nt, x in final.items()})
    return embeddings


def analyze_one_ckpt(c: Dict, dev: List[Dict], qids: List[int],
                     qid_to_db: Dict[int, str]) -> Dict[str, Any]:
    print("=" * 80)
    print(f"Analyzing [{c['tag']}] ({c['label']})")
    print(f"  ckpt: {c['ckpt_path']}")
    print(f"  best epoch={c['best_epoch']}, R@15={c['best_r15']:.4f}")
    print("=" * 80)
    if not c["ckpt_path"].exists():
        print(f"  ✗ ckpt missing: {c['ckpt_path']}")
        return {"missing": True}

    # Setup builder + encoder (per-ckpt)
    # Builder 의 plm_model_name 은 node feature 의 PLM dim 결정 — query_conditioned 시
    # ckpt 의 expected input dim 위 양 PLM 동일 dim 정합 필수 (e.g., Qwen3 1024 + Qwen3 1024 = 2048).
    builder = EnrichedHeteroGraphBuilder(
        plm_model_name=c["encoder_model_name"],
        tables_json_path=str(ROOT / "data/raw/BIRD_dev/dev_tables.json"),
    )
    t0 = time.time()
    encoder = LocalPLMEncoder(model_name=c["encoder_model_name"])
    print(f"  encoder loaded in {time.time()-t0:.1f}s  on device={DEVICE}")

    model = load_model(c)
    print(f"  GAT loaded — num_layers={model.num_layers}, query_conditioned={model.query_conditioned}")

    layer_names = ["L0_PLM"] + [f"L{i+1}_GAT" for i in range(model.num_layers)] + ["L_out"]
    sims_by_layer: List[List[float]] = [[] for _ in layer_names]
    entropy_by_layer: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    topk_conc_by_layer: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    # Per-DB graph cache (build once per DB) — within this ckpt run
    db_graph_cache: Dict[str, Tuple[Any, Dict]] = {}

    per_query_records: List[Dict] = []
    n_skipped = 0
    t_start = time.time()
    for idx, qid in enumerate(qids):
        item = dev[qid]
        db_id = item["db_id"]
        question = item["question"]

        # Build graph (per-DB)
        if db_id not in db_graph_cache:
            graph_data, metadata = builder.build(db_id=db_id, db_dir=str(DEV_DB_DIR))
            db_graph_cache[db_id] = (graph_data, metadata)
        else:
            graph_data, metadata = db_graph_cache[db_id]
        data = graph_data.clone()

        # Encode query
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

        # Move to device
        x_dict = {nt: x.to(DEVICE) for nt, x in data.x_dict.items()}
        edge_index_dict = {et: ei.to(DEVICE) for et, ei in data.edge_index_dict.items()}
        cb_edge = edge_index_dict.get(COL_TO_TAB_EDGE)

        # Layer-wise intra-table sims (oversmoothing trajectory)
        try:
            layer_embs = extract_layerwise_via_hook(model, x_dict, edge_index_dict, query_emb=q_emb)
            assert len(layer_embs) == len(layer_names), \
                f"layer count mismatch: {len(layer_embs)} vs {len(layer_names)}"
            if cb_edge is not None:
                cb_edge_cpu = cb_edge.cpu()
                for l, ed in enumerate(layer_embs):
                    col_emb = ed.get("column")
                    if col_emb is None:
                        continue
                    sims = intra_table_sims(col_emb, cb_edge_cpu)
                    sims_by_layer[l].extend(sims)
        except Exception as e:
            print(f"  [qid={qid}] layerwise fail: {e}")
            import traceback; traceback.print_exc()
            n_skipped += 1
            continue

        # Attention extraction
        try:
            class _SimpleBatch:
                pass
            batch_proxy = _SimpleBatch()
            batch_proxy.x_dict = x_dict
            batch_proxy.edge_index_dict = edge_index_dict
            attn_res = extract_layerwise_attention_v2(model, batch_proxy, query_emb=q_emb,
                                                     topk=5, return_raw=False)
            for layer_key, et_map in attn_res["entropy"].items():
                for et_str, v in et_map.items():
                    if not (np.isnan(v) or np.isinf(v)):
                        entropy_by_layer[layer_key][et_str].append(float(v))
            for layer_key, et_map in attn_res["topk_conc"].items():
                for et_str, v in et_map.items():
                    if not (np.isnan(v) or np.isinf(v)):
                        topk_conc_by_layer[layer_key][et_str].append(float(v))
        except Exception as e:
            print(f"  [qid={qid}] attention fail: {e}")
            # continue — layer-wise sims OK 만으로 진행

        # Per-query record: L1~L3 intra-table sim mean per-query (computed locally)
        per_q_sims = []
        for l, ed in enumerate(layer_embs):
            col_emb = ed.get("column")
            if col_emb is None or cb_edge is None:
                per_q_sims.append(None)
                continue
            sim_list = intra_table_sims(col_emb, cb_edge.cpu())
            per_q_sims.append(float(np.mean(sim_list)) if sim_list else None)
        per_query_records.append({
            "qid": qid, "db_id": db_id,
            "difficulty": item.get("difficulty", "unknown"),
            **{layer_names[l]: per_q_sims[l] for l in range(len(layer_names))},
        })

        if (idx + 1) % 10 == 0:
            print(f"  progress: {idx+1}/{len(qids)} queries  elapsed={time.time()-t_start:.1f}s")

    print(f"  done. elapsed={time.time()-t_start:.1f}s, skipped={n_skipped}")

    # Free up GPU memory (encoder + model)
    del encoder
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Aggregate
    def _agg(vals: List[float]) -> Dict:
        if not vals:
            return {"n": 0, "mean": None, "std": None}
        a = np.array(vals)
        return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max()),
                "q25": float(np.percentile(a, 25)),
                "q50": float(np.percentile(a, 50)),
                "q75": float(np.percentile(a, 75))}

    layer_summary: Dict[str, Dict] = {}
    for l, name in enumerate(layer_names):
        layer_summary[name] = _agg(sims_by_layer[l])

    # Attention summary per-layer per-edge-type
    def _agg_dict(d: Dict[str, Dict[str, List[float]]]) -> Dict:
        out: Dict[str, Dict[str, Dict]] = {}
        for layer_key, et_map in d.items():
            out[layer_key] = {}
            for et_str, vals in et_map.items():
                a = np.array(vals)
                out[layer_key][et_str] = {
                    "n": int(a.size),
                    "mean": float(a.mean()) if a.size else None,
                    "std": float(a.std()) if a.size else None,
                }
        return out

    return {
        "tag": c["tag"], "label": c["label"],
        "best_epoch": c["best_epoch"], "best_r15": c["best_r15"],
        "n_qids_sampled": len(qids),
        "n_skipped": n_skipped,
        "layer_names": layer_names,
        "intra_table_sim_per_layer": layer_summary,
        "attention_entropy_per_layer_edge_type": _agg_dict(entropy_by_layer),
        "topk_concentration_per_layer_edge_type": _agg_dict(topk_conc_by_layer),
        "per_query_records": per_query_records,
    }


def main():
    print(f"Device: {DEVICE}")
    dev = load_dev()
    qids, qid_to_db = stratified_qids(dev)
    print(f"Stratified {len(qids)} qids = 5/DB × 11 DBs (seed={RANDOM_SEED})")
    print()

    results: Dict[str, Dict] = {}
    for c in CKPTS:
        res = analyze_one_ckpt(c, dev, qids, qid_to_db)
        results[c["tag"]] = res
        print()

    # Save per-query records (jsonl)
    for tag, res in results.items():
        if res.get("missing"):
            continue
        pq_path = OUT_DIR / f"wave16_gat_layerwise_l1_per_query_{tag}_2026-05-22.jsonl"
        with pq_path.open("w") as f:
            for r in res["per_query_records"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  per-query → {pq_path}  ({len(res['per_query_records'])} records)")

    # CSV summary
    csv_path = OUT_DIR / "wave16_gat_layerwise_l1_2026-05-22.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt_tag", "layer", "n_table_query_pairs",
                    "intra_sim_mean", "intra_sim_std", "intra_sim_q25", "intra_sim_q50", "intra_sim_q75"])
        for tag, res in results.items():
            if res.get("missing"):
                continue
            for name in res["layer_names"]:
                s = res["intra_table_sim_per_layer"][name]
                if s.get("n", 0) == 0:
                    continue
                w.writerow([tag, name, s["n"],
                            round(s["mean"], 4), round(s["std"], 4),
                            round(s["q25"], 4), round(s["q50"], 4), round(s["q75"], 4)])
    print(f"\n→ csv: {csv_path}")

    # JSON full summary (without per-query bulk to keep small)
    json_path = OUT_DIR / "wave16_gat_layerwise_l1_2026-05-22.json"
    summary_for_json = {
        tag: {k: v for k, v in res.items() if k != "per_query_records"}
        for tag, res in results.items()
    }
    with json_path.open("w") as f:
        json.dump(summary_for_json, f, indent=2, ensure_ascii=False)
    print(f"→ json: {json_path}")

    # Display summary
    print()
    print("=" * 100)
    print("Layer-wise Intra-Table Cosine Similarity (L1 saturation indicator)")
    print("=" * 100)
    for tag, res in results.items():
        if res.get("missing"):
            continue
        print(f"\n--- {tag} ({res['label']}) — best ep{res['best_epoch']}, R@15={res['best_r15']:.4f} ---")
        for name in res["layer_names"]:
            s = res["intra_table_sim_per_layer"][name]
            if s.get("n", 0) == 0:
                continue
            print(f"  {name:10s}  n={s['n']:6d}  mean={s['mean']:.4f}  std={s['std']:.4f}  "
                  f"q25/50/75={s['q25']:.4f}/{s['q50']:.4f}/{s['q75']:.4f}")

    print()
    print("=" * 100)
    print("Top-5 Attention Concentration (mean over dst nodes, mean over queries)")
    print("=" * 100)
    for tag, res in results.items():
        if res.get("missing"):
            continue
        print(f"\n--- {tag} ---")
        tk = res.get("topk_concentration_per_layer_edge_type", {})
        for layer_key in sorted(tk.keys(), key=lambda k: int(k.lstrip("L"))):
            et_map = tk[layer_key]
            for et_str, stats in et_map.items():
                if stats.get("n", 0) == 0:
                    continue
                print(f"  {layer_key:4s} {et_str:55s}  n={stats['n']:5d}  conc={stats['mean']:.4f} ± {stats['std']:.4f}")


if __name__ == "__main__":
    main()
