"""
Stage 2 — V(T_MST) 와 V(union) 로컬 재계산 (LLM 0 호출, CPU).

방법: m4_canonical 파이프라인 front-half (builder+selector, 결정론적, LLM 없음) 를 CPU 로 구동하여
추출기가 받는 정확한 graph_data + node_scores 를 캡처하고, 그 위에 MSTKruskal(θ=0.1) 과
MSTPCSTUnion(θ=0.1) 을 각각 적용 → V(T_MST), V(union) = V(T_MST) ∪ V(PCST).
filter/generator 는 benign stub (API 0). node score 는 score_analysis 캐시와 대조 검증 가능.
seed=고정(파이프라인 결정론적; 랜덤성 없음).

실행: CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python review_verification/stage2_recompute_extractors.py --limit N
출력: review_verification/stage2_extractor_outputs.jsonl
"""
import os, sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)   # 0 = 전체 1534
args = ap.parse_args()

sys.argv = ["main", "--config", "experiments/sonnet_rebaseline_2026_06_10/m4_canonical_sonnet"]
from utils.config_parser import get_args_and_config
from utils.logger import setup_logger
from utils.evaluator import parse_sql_elements
from pipeline import SchemaLinkingPipeline
from modules.extractors.mst_kruskal import MSTKruskalExtractor
from modules.extractors.mst_pcst_union import MSTPCSTUnionExtractor

_, config = get_args_and_config()
setup_logger(log_dir="./logs/", exp_name="stage2_recompute")

pipeline = SchemaLinkingPipeline(config)
if getattr(pipeline, "filter", None) is not None and getattr(pipeline.filter, "client", None):
    pipeline.filter.client.generate_text = lambda prompt, model=None, temperature=None: "{}"
pipeline.generator.client.generate_text = lambda prompt, model=None, temperature=None: "SELECT 1"

# 두 추출기 (θ=0.1 통일, canonical 설정)
mst_ex = MSTKruskalExtractor(score_threshold=0.1)
union_ex = MSTPCSTUnionExtractor(score_threshold=0.1)   # PCST cost default (bt0.01/fk0.05/macro0.5)

DEV = json.load(open(ROOT / "data/raw/BIRD_dev/dev.json", encoding="utf-8"))
if args.limit and 0 < args.limit < len(DEV):
    stride = len(DEV) / args.limit
    DEV = [DEV[int(i * stride)] for i in range(args.limit)]

def gold_sets(gold_sql):
    t, c = parse_sql_elements(gold_sql)
    return set(x.lower() for x in t), set(x.lower() for x in c)

def name_is_gold(name, gt, gc):
    nm = str(name)
    if "->" in nm:   # fk_node — gold 컬럼 판정에서 제외 (R/P/F1 관례)
        return False
    if "." in nm:
        t, c = nm.split(".", 1)
        return c.lower() in gc
    return nm.lower() in gt

CAP = {}   # qid -> dict
def wrap_extract(graph_data, node_scores, seed_nodes=None, **kw):
    qid = CAP["_qid"]
    node_meta = graph_data.get("node_metadata", {}) or {}
    def names(idxs):
        return [str(node_meta.get(int(i), node_meta.get(str(i), i))) for i in idxs]
    mst_nodes, _ = mst_ex.extract(graph_data, node_scores)
    union_nodes, _ = union_ex.extract(graph_data, node_scores)
    mst_set, union_set = set(int(x) for x in mst_nodes), set(int(x) for x in union_nodes)
    pcst_added = sorted(union_set - mst_set)
    gt, gc = CAP["_gold"]
    added_names = names(pcst_added)
    added_gold = [n for n in added_names if name_is_gold(n, gt, gc)]
    CAP[qid] = {
        "n_mst": len(mst_set), "n_union": len(union_set),
        "n_pcst_added": len(pcst_added),
        "pcst_added_names": added_names,
        "pcst_added_gold_count": len(added_gold),
        "pcst_added_gold_names": added_gold,
        "in_D": len(pcst_added) > 0,
    }
    # 파이프라인 계속 진행용: union 반환 (실제 anchor extractor 출력)
    return union_nodes, _
pipeline.extractor.extract = wrap_extract

print(f"[stage2] {len(DEV)} queries (CPU, LLM stub)")
import time
t0 = time.time()
for k, item in enumerate(DEV):
    qid = item.get("question_id", k)
    CAP["_qid"] = qid
    CAP["_gold"] = gold_sets(item.get("SQL", item.get("query", "")))
    try:
        pipeline.run(db_id=item.get("db_id"), query=item.get("question"), evidence=item.get("evidence", ""))
    except Exception as e:
        print(f"  qid={qid} err: {e}")
        continue
    CAP[qid]["db_id"] = item.get("db_id")
    CAP[qid]["difficulty"] = item.get("difficulty")
    gt, gc = CAP["_gold"]
    CAP[qid]["gold_table_count"] = len(gt)
    if (k + 1) % 200 == 0:
        print(f"  {k+1}/{len(DEV)} ({time.time()-t0:.0f}s)")

# 저장
out = ROOT / "review_verification/stage2_extractor_outputs.jsonl"
rows = [{"question_id": q, **v} for q, v in CAP.items() if isinstance(q, int)]
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
n = len(rows)
nD = sum(1 for r in rows if r["in_D"])
print(f"\n[stage2] done {n}q in {time.time()-t0:.0f}s")
print(f"  |D| (PCST가 노드 추가) = {nD} ({nD/max(n,1)*100:.1f}%)")
print(f"  avg n_mst={sum(r['n_mst'] for r in rows)/max(n,1):.1f}  avg n_union={sum(r['n_union'] for r in rows)/max(n,1):.1f}")
print(f"  D 중 PCST-added 가 gold 포함: {sum(1 for r in rows if r['in_D'] and r['pcst_added_gold_count']>0)}")
print(f"  saved: {out}")
