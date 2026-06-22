#!/usr/bin/env python
"""G-S2-1 Spider2.0-Lite cross-dataset generalization inference (zero-shot).

BIRD M4 anchor (best_gat_qcond_nl3.pt + EnsembleSelector α=0.5 QCond GAT NL=3)
→ MSTPCSTUnion extractor 위 Spider2 schema linking. 학습 0 (zero-shot).
DDL.csv → Spider2GraphBuilder → selector → extractor → predicted_schema.jsonl.

skip: max_columns 초과 (RuntimeError) / DDL 미존재 (FileNotFoundError) → log + 제외.
사용: PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1 python scripts/run_g_s2_1_spider2_inference.py
"""
import os, sys, json, time, traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SPIDER2_ROOT = PROJECT_ROOT / "data" / "Spider2" / "spider2-lite" / "resource" / "databases"
LITE_JSONL = PROJECT_ROOT / "data" / "Spider2" / "spider2-lite" / "spider2-lite.jsonl"
OUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "g_s2_1_spider2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from modules.registry import build
from modules.builders.spider2_builder import Spider2GraphBuilder
from utils.logger import get_logger
logger = get_logger("g_s2_1")

LIMIT = int(os.environ.get("G_S2_1_LIMIT", "0"))  # >0 이면 첫 N개만 (smoke)

def main():
    instances = [json.loads(l) for l in open(LITE_JSONL)]
    if LIMIT > 0:
        instances = instances[:LIMIT]
    logger.info(f"[g_s2_1] {len(instances)} instances 로드 (LIMIT={LIMIT})")

    builder = Spider2GraphBuilder(max_columns=5000, plm_batch_size=256)
    selector = build("selector", {"name": "EnsembleSelector", "params": {
        "weight_path": "outputs/checkpoints/best_gat_qcond_nl3.pt",
        "alpha": 0.5, "top_k": 200, "query_conditioned": True, "encoder_type": "plm",
    }})
    extractor = build("extractor", {"name": "MSTPCSTUnionExtractor",
                                    "params": {"score_threshold": 0.1}})

    out_path = OUT_DIR / "predicted_schema.jsonl"
    skip_log = OUT_DIR / "skipped.jsonl"
    n_ok = n_skip = 0
    skip_reasons = {}
    with open(out_path, "w") as fout, open(skip_log, "w") as fskip:
        for i, inst in enumerate(instances):
            iid = inst["instance_id"]; db = inst["db"]; q = inst["question"]
            backend = ("bq" if iid.startswith("bq") else "sf" if iid.startswith("sf")
                       else "ga" if iid.startswith("ga") else "local" if iid.startswith("local") else "other")
            try:
                graph_data, meta = builder.build(db_id=iid, db_dir=str(SPIDER2_ROOT), spider2_db_field=db)
                node_meta = meta.get("node_metadata", {})
                cand = list(range(len(node_meta)))
                seeds = selector.select(scores=None, candidates=cand, question=q,
                                        graph_data=graph_data, metadata=meta)
                scores_list = (selector.latest_scores if getattr(selector, "latest_scores", None)
                               else [1.0] * len(cand))
                sel_idx, _ = extractor.extract(graph_data=meta, node_scores=scores_list, seed_nodes=seeds)
                pred_tables, pred_cols = [], []
                for nid in sel_idx:
                    k = int(nid) if (isinstance(nid, (int, float)) or (isinstance(nid, str) and str(nid).isdigit())) else nid
                    name = node_meta.get(k, str(k))
                    (pred_cols if "." in name else pred_tables).append(name)
                fout.write(json.dumps({
                    "instance_id": iid, "db": db, "backend": meta.get("spider2_backend", backend),
                    "predicted_tables": sorted(set(pred_tables)),
                    "predicted_columns": sorted(set(pred_cols)),
                    "n_total_columns": meta.get("spider2_total_columns"),
                    "parse_errors": len(meta.get("spider2_parse_errors", [])),
                }) + "\n"); fout.flush()
                n_ok += 1
            except Exception as e:
                reason = type(e).__name__ + ": " + str(e)[:120]
                rkey = "max_columns" if "max_columns" in str(e) or "RuntimeError" in type(e).__name__ else \
                       "not_found" if "FileNotFound" in type(e).__name__ else "other"
                skip_reasons[rkey] = skip_reasons.get(rkey, 0) + 1
                fskip.write(json.dumps({"instance_id": iid, "db": db, "backend": backend, "reason": reason}) + "\n"); fskip.flush()
                n_skip += 1
            if (i + 1) % 25 == 0:
                logger.info(f"[g_s2_1] {i+1}/{len(instances)} | ok={n_ok} skip={n_skip}")
    logger.info(f"[g_s2_1] DONE — ok={n_ok} skip={n_skip} skip_reasons={skip_reasons}")
    logger.info(f"[g_s2_1] 출력: {out_path}")
    json.dump({"n_ok": n_ok, "n_skip": n_skip, "skip_reasons": skip_reasons,
               "total": len(instances)}, open(OUT_DIR / "run_summary.json", "w"), indent=2)

if __name__ == "__main__":
    main()
