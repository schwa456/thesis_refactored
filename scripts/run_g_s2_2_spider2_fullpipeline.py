#!/usr/bin/env python
"""G-S2-2 Spider2.0-Lite full-pipeline (selector→extractor→GLM BiFilter→SQL) — 135 local SQLite.

G-S2-1 (selector+extractor only, Filter 없는 zero-shot) 의 precision 붕괴를 Filter 가 회복시키는지 검증.
M4 anchor selector + MSTPCSTUnion + BidirectionalFilter(GLM) + LLMSQLGenerator(GLM) → predicted SQL.
EX 채점은 별도 (Spider2 evaluation_suite/evaluate.py 재사용, predicted SQL → SQLite 실행 vs gold).
★ GLM 라인 필요 — STE(V7-W5) 완료 후 launch (concurrency=2). 첫 launch 시 GLM 단계 wiring 검증.

사용: PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/run_g_s2_2_spider2_fullpipeline.py
"""
import os, sys, json, traceback
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
SPIDER2_ROOT = PROJECT_ROOT / "data" / "Spider2" / "spider2-lite" / "resource" / "databases"
LITE_JSONL = PROJECT_ROOT / "data" / "Spider2" / "spider2-lite" / "spider2-lite.jsonl"
LOCALDB = SPIDER2_ROOT / "spider2-localdb"   # <db>.sqlite (EX 실행용)
OUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "g_s2_2_spider2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from modules.registry import build
from modules.builders.spider2_builder import Spider2GraphBuilder
from utils.logger import get_logger
logger = get_logger("g_s2_2")
LIMIT = int(os.environ.get("G_S2_2_LIMIT", "0"))
# backbone: 2026-06-11 전면 Sonnet 전환 (기본 sonnet, env 로 GLM 복원 가능)
PROVIDER = os.environ.get("G_S2_2_PROVIDER", "sonnet")
MODEL = os.environ.get("G_S2_2_MODEL", "claude-sonnet-4-6")

def main():
    instances = [json.loads(l) for l in open(LITE_JSONL)]
    instances = [d for d in instances if d["instance_id"].startswith("local")]  # 135 local SQLite
    if LIMIT > 0: instances = instances[:LIMIT]
    logger.info(f"[g_s2_2] {len(instances)} local instances (LIMIT={LIMIT})")

    builder = Spider2GraphBuilder(max_columns=5000, plm_batch_size=256)
    selector = build("selector", {"name": "EnsembleSelector", "params": {
        "weight_path": "outputs/checkpoints/best_gat_qcond_nl3.pt",
        "alpha": 0.5, "top_k": 200, "query_conditioned": True, "encoder_type": "plm"}})
    extractor = build("extractor", {"name": "MSTPCSTUnionExtractor", "params": {"score_threshold": 0.1}})
    flt = build("filter", {"name": "BidirectionalFilter", "params": {
        "provider": PROVIDER, "model_name": MODEL, "max_iteration": 1, "temperature": 0.0,
        "sanitize_output": True, "forward_section": "recall_biased_mild",
        "backward_section": "bidirectional_backward"}})
    sqlgen = build("generator", {"name": "LLMSQLGenerator", "params": {
        "provider": PROVIDER, "llm_model": MODEL, "temperature": 0.0}})

    out_path = OUT_DIR / "predicted_sql.jsonl"
    n_ok = n_skip = 0
    with open(out_path, "w") as fout:
        for i, inst in enumerate(instances):
            iid, db, q = inst["instance_id"], inst["db"], inst["question"]
            try:
                graph_data, meta = builder.build(db_id=iid, db_dir=str(SPIDER2_ROOT), spider2_db_field=db)
                node_meta = meta.get("node_metadata", {})
                cand = list(range(len(node_meta)))
                seeds = selector.select(scores=None, candidates=cand, question=q, graph_data=graph_data, metadata=meta)
                scores_list = selector.latest_scores if getattr(selector, "latest_scores", None) else [1.0]*len(cand)
                sel_idx, _ = extractor.extract(graph_data=meta, node_scores=scores_list, seed_nodes=seeds)
                # subgraph_dict {table: [cols]}
                subgraph = {}
                gat_scores = {}
                for nid in sel_idx:
                    k = int(nid) if (isinstance(nid,(int,float)) or (isinstance(nid,str) and str(nid).isdigit())) else nid
                    name = str(node_meta.get(k, str(k)))
                    if "." in name and "->" not in name:
                        t, c = name.split(".", 1); subgraph.setdefault(t, []).append(c)
                    elif "->" not in name:
                        subgraph.setdefault(name, [])
                # BiFilter (over-extract 흡수 검증 대상)
                fr = flt.refine(query=q, subgraph=subgraph, db_id=db, tier2_pool=[],
                                gat_scores=gat_scores, raw_gat_scores={}, raw_cos_scores={}, metadata=meta)
                # filter 출력(final_nodes: "t.c"/"t" list) → {table:[cols]} 재구성.
                # (2026-06-11 analyzer 진단 key-mismatch 버그 수정: refine()은 filtered_schema/subgraph 키를
                #  반환하지 않고 final_nodes 만 반환 → 기존 fallback 이 extractor subgraph 를 그대로 전달해 filter 무효화)
                final_nodes = fr.get("final_nodes") or []
                if fr.get("status") == "Unanswerable" or not final_nodes:
                    filtered = subgraph  # recall 보호 fallback (filters/CLAUDE.md 규약)
                else:
                    filtered = {}
                    for nm in final_nodes:
                        nm = str(nm)
                        if "->" in nm:
                            continue
                        if "." in nm:
                            t, c = nm.split(".", 1); filtered.setdefault(t, []).append(c)
                        else:
                            filtered.setdefault(nm, [])
                    if not filtered:
                        filtered = subgraph
                # SQL gen
                sql = sqlgen.generate(query=q, subgraph=filtered, evidence="")
                fout.write(json.dumps({"instance_id": iid, "db": db,
                    "predicted_sql": sql, "n_filtered_tables": len(filtered),
                    "n_extracted_tables": len(subgraph)}) + "\n"); fout.flush()
                n_ok += 1
            except Exception as e:
                logger.warning(f"[g_s2_2] {iid} skip: {type(e).__name__}: {str(e)[:120]}")
                n_skip += 1
            if (i+1) % 10 == 0: logger.info(f"[g_s2_2] {i+1}/{len(instances)} ok={n_ok} skip={n_skip}")
    logger.info(f"[g_s2_2] DONE ok={n_ok} skip={n_skip} → {out_path}")
    json.dump({"n_ok": n_ok, "n_skip": n_skip, "total": len(instances)}, open(OUT_DIR/"run_summary.json","w"), indent=2)
    logger.info("[g_s2_2] ★ EX 채점: data/Spider2/.../evaluation_suite/evaluate.py 로 predicted_sql vs gold exec_result (별도 step)")

if __name__ == "__main__":
    main()
