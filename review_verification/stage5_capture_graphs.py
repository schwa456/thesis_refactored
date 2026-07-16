"""
Stage 5 [1/2] — graph_data + node_scores 캡처 (LLM 0, CPU). 이후 sweep 은 캐시로 순수 재계산.

m4_canonical front-half(builder+selector, 결정론적) 구동으로 추출기가 받는 graph_data
(edges/edge_types/node_metadata) + node_scores 를 질의별로 캡처하여 .pt 로 저장.
파이프라인은 1회만 구동, sweep(θ·cost)은 stage5_sweep.py 에서 이 캐시로 추출기만 재적용.

실행: CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python review_verification/stage5_capture_graphs.py
출력: review_verification/stage5_graph_cache.pt  (질의별 dict)
"""
import os, sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch

sys.argv = ["main", "--config", "experiments/sonnet_rebaseline_2026_06_10/m4_canonical_sonnet"]
from utils.config_parser import get_args_and_config
from utils.logger import setup_logger
from utils.evaluator import parse_sql_elements
from pipeline import SchemaLinkingPipeline

_, config = get_args_and_config()
setup_logger(log_dir="./logs/", exp_name="stage5_capture")
pipeline = SchemaLinkingPipeline(config)
if getattr(pipeline, "filter", None) is not None and getattr(pipeline.filter, "client", None):
    pipeline.filter.client.generate_text = lambda prompt=None, model=None, temperature=None, **k: "{}"
pipeline.generator.client.generate_text = lambda prompt=None, model=None, temperature=None, **k: "SELECT 1"

DEV = json.load(open(ROOT / "data/raw/BIRD_dev/dev.json", encoding="utf-8"))

CACHE = {}
CUR = {}
orig = pipeline.extractor.extract
def cap(graph_data, node_scores, seed_nodes=None, **kw):
    qid = CUR["qid"]
    node_meta = graph_data.get("node_metadata", {}) or {}
    # node_metadata 는 idx→name; 저장 최소화 위해 name list (idx 순) 로
    n = len(node_scores)
    names = [str(node_meta.get(i, node_meta.get(str(i), i))) for i in range(n)]
    CACHE[qid] = {
        "edges": [tuple(int(x) for x in e) for e in graph_data.get("edges", [])],
        "edge_types": list(graph_data.get("edge_types", [])),
        "node_scores": [float(s) for s in node_scores],
        "node_names": names,
    }
    return orig(graph_data, node_scores, seed_nodes, **kw)
pipeline.extractor.extract = cap

print(f"[stage5-capture] {len(DEV)} queries (CPU, LLM stub)")
t0 = time.time()
for k, item in enumerate(DEV):
    qid = item.get("question_id", k)
    CUR["qid"] = qid
    try:
        pipeline.run(db_id=item.get("db_id"), query=item.get("question"), evidence=item.get("evidence", ""))
    except Exception as e:
        print(f"  qid={qid} err: {e}")
        continue
    gt, gc = parse_sql_elements(item.get("SQL", item.get("query", "")))
    CACHE[qid]["gold_tables"] = sorted(set(t.lower() for t in gt))
    CACHE[qid]["gold_cols"] = sorted(set(c.lower() for c in gc))
    CACHE[qid]["difficulty"] = item.get("difficulty")
    if (k + 1) % 300 == 0:
        print(f"  {k+1}/{len(DEV)} ({time.time()-t0:.0f}s)")

out = ROOT / "review_verification/stage5_graph_cache.pt"
torch.save(CACHE, out)
print(f"[stage5-capture] done {len(CACHE)}q in {time.time()-t0:.0f}s → {out}")
# edge_type 통계
from collections import Counter
et = Counter()
for v in CACHE.values():
    et.update(v["edge_types"])
print(f"  edge_type 총계: {dict(et)}")
