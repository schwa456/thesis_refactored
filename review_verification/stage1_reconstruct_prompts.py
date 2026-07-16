"""
Stage 1 — 프롬프트 재조립 & 토큰 계측 (LLM 호출 없음, CPU 전용).

목적:
  1) m4_canonical_sonnet 파이프라인을 실제 코드로 구동(단, filter/generator LLM 은
     benign stub 로 치환 → API 0 호출)하여 각 stage 의 '실제 조립 프롬프트'를 캡처.
  2) 조건별 쿼리당 입력 토큰을 tiktoken(cl100k, 근사)으로 계측:
       - full_schema  : 전체 DB 스키마 → generator
       - filter_fwd   : recall_biased_mild (extractor 출력 83노드 mschema+values)
       - filter_bwd   : bidirectional_backward (동일 schema_str)
       - gen_actual   : generator 가 '실제로' 받는 subgraph (코드상 extractor 출력)
       - gen_filtered : filter 출력(7.4노드) 로 재조립한 generator (논문 의도 조건 b)
  3) generator 입력이 extractor 출력인지 filter 출력인지 직접 계측하여 검증.

실행: CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python review_verification/stage1_reconstruct_prompts.py --limit N
출력: review_verification/stage1_cost_per_query.csv (+ stdout 요약)
시드: BIRD-dev stride 샘플링 (run_sonnet_batch_e2e.py 와 동일 방식, deterministic).
"""
import os, sys, json, time, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""      # GPU 사용 금지 (타 연구자 점유)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=3)
ap.add_argument("--config", default="experiments/sonnet_rebaseline_2026_06_10/m4_canonical_sonnet")
args_cli = ap.parse_args()

import tiktoken
ENC = tiktoken.get_encoding("cl100k_base")
def ntok(s: str) -> int:
    return len(ENC.encode(s or "", disallowed_special=()))

sys.argv = ["main", "--config", args_cli.config]
from utils.config_parser import get_args_and_config
from utils.logger import setup_logger
from pipeline import SchemaLinkingPipeline
from prompts.prompt_manager import PromptManager
from modules.filters.agents import AgentUtils

args, config = get_args_and_config()
setup_logger(log_dir=config['paths']['log_dir'], exp_name="stage1_recon")

# ── benign stub 로 LLM 치환 (API 0 호출) ─────────────────────────────
pipeline = SchemaLinkingPipeline(config)
if getattr(pipeline, "filter", None) is not None and getattr(pipeline.filter, "client", None):
    pipeline.filter.client.generate_text = lambda prompt, model=None, temperature=None: "{}"
pipeline.generator.client.generate_text = lambda prompt, model=None, temperature=None: "SELECT 1"

# ── PromptManager.load_prompt 래핑: 조립 프롬프트 캡처 ────────────────
CUR = {"qid": None}
CAP = []  # (qid, section, prompt_str, ntok)
_orig_load = PromptManager.load_prompt
def _wrapped_load(self, file_name, section, **kwargs):
    p = _orig_load(self, file_name, section, **kwargs)
    CAP.append({"qid": CUR["qid"], "file": file_name, "section": section,
                "ntok": ntok(p), "prompt": p})
    return p
PromptManager.load_prompt = _wrapped_load

# ── generator.generate 래핑: 실제 받는 subgraph 노드 수 캡처 ──────────
GEN = []  # (qid, call_idx, n_nodes, flat_set)
_orig_gen = pipeline.generator.generate
_gc = {"i": 0}
def _wrapped_gen(query, subgraph, evidence="", **kw):
    flat = set()
    for t, cols in (subgraph or {}).items():
        if cols:
            for c in cols: flat.add(f"{t}.{c}")
        else:
            flat.add(t)
    GEN.append({"qid": CUR["qid"], "call_idx": _gc["i"], "n_nodes": len(flat), "flat": flat})
    _gc["i"] += 1
    return _orig_gen(query=query, subgraph=subgraph, evidence=evidence, **kw)
pipeline.generator.generate = _wrapped_gen

# ── filter.refine 래핑: 입력(extractor 출력) vs 출력(final_nodes) 노드 수 ──
FILT = []
if getattr(pipeline, "filter", None) is not None:
    _orig_refine = pipeline.filter.refine
    def _wrapped_refine(query, subgraph, db_id=None, **kw):
        n_in = sum(len(v) if v else 1 for v in (subgraph or {}).values())
        res = _orig_refine(query=query, subgraph=subgraph, db_id=db_id, **kw)
        FILT.append({"qid": CUR["qid"], "filter_in_nodes": n_in,
                     "filter_out_nodes": len(res.get("final_nodes", []))})
        return res
    pipeline.filter.refine = _wrapped_refine

# ── dev set stride 샘플 (batch script 와 동일) ───────────────────────
data_path = config['paths'].get('dev_json', 'data/raw/BIRD_dev/dev.json')
dataset = json.load(open(data_path, encoding='utf-8'))
N = args_cli.limit
if N and 0 < N < len(dataset):
    stride = len(dataset) / N
    dataset = [dataset[int(i * stride)] for i in range(N)]
print(f"[stage1] {len(dataset)} queries (CPU, benign LLM)")

# ── full-schema 프롬프트 재조립용: dev_tables.json ───────────────────
tables_json = config['graph_builder']['params']['tables_json_path']
DEVT = {d['db_id']: d for d in json.load(open(tables_json, encoding='utf-8'))}
def full_schema_subgraph(db_id):
    d = DEVT[db_id]
    tabs = d['table_names_original']
    sg = {t: [] for t in tabs}
    for (ti, col) in d['column_names_original']:
        if ti < 0: continue
        sg[tabs[ti]].append(col)
    return sg

# ── 실행 ─────────────────────────────────────────────────────────────
rows = []
t0 = time.time()
for k, item in enumerate(dataset):
    qid = item.get("question_id", k)
    db_id = item.get("db_id")
    CUR["qid"] = qid
    _gc["i"] = 0
    t_q = time.time()
    try:
        pipeline.run(db_id=db_id, query=item.get("question"), evidence=item.get("evidence", ""))
    except Exception as e:
        print(f"  qid={qid} run err: {e}")
        continue
    # full-schema generator prompt (benign, 별도 캡처 — CAP 에 section=sql_generator 로 추가됨)
    CUR["qid"] = f"{qid}__FULL"
    try:
        pipeline.generator.generate(query=item.get("question"),
                                    subgraph=full_schema_subgraph(db_id),
                                    evidence=item.get("evidence", ""))
    except Exception as e:
        print(f"  qid={qid} full err: {e}")
    print(f"  [{k+1}/{len(dataset)}] qid={qid} db={db_id} ({time.time()-t_q:.1f}s)")

elapsed = time.time() - t0
print(f"[stage1] done {len(dataset)}q in {elapsed:.1f}s ({elapsed/max(len(dataset),1):.1f}s/q)")

# ── 집계 ─────────────────────────────────────────────────────────────
import csv
# 프롬프트를 qid+section 별로 정리
def sec_tokens(qid, section, occurrence=0):
    hits = [c for c in CAP if c["qid"] == qid and c["section"] == section]
    return hits[occurrence]["ntok"] if len(hits) > occurrence else None

per_q = {}
for item in dataset:
    qid = item.get("question_id")
    fwd = sec_tokens(qid, "recall_biased_mild")
    bwd = sec_tokens(qid, "bidirectional_backward")
    # sql_generator 호출들: 0=main(extractor출력), 1=selector-only, 2=extractor-only
    gen_main = sec_tokens(qid, "sql_generator", 0)
    gen_sel = sec_tokens(qid, "sql_generator", 1)
    gen_ext = sec_tokens(qid, "sql_generator", 2)
    gen_full = sec_tokens(f"{qid}__FULL", "sql_generator", 0)
    f_rec = next((f for f in FILT if f["qid"] == qid), {})
    g_main = next((g for g in GEN if g["qid"] == qid and g["call_idx"] == 0), {})
    per_q[qid] = {
        "qid": qid, "db_id": item.get("db_id"),
        "filter_in_nodes": f_rec.get("filter_in_nodes"),
        "filter_out_nodes": f_rec.get("filter_out_nodes"),
        "gen_main_input_nodes": g_main.get("n_nodes"),
        "tok_filter_fwd": fwd, "tok_filter_bwd": bwd,
        "tok_gen_main_actual": gen_main, "tok_gen_selector_only": gen_sel,
        "tok_gen_extractor_only": gen_ext, "tok_gen_full_schema": gen_full,
        "gen_main_eq_extractor": (gen_main == gen_ext) if (gen_main and gen_ext) else None,
    }

csv_path = ROOT / "review_verification" / "stage1_cost_per_query.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(next(iter(per_q.values())).keys()))
    w.writeheader()
    for r in per_q.values():
        w.writerow(r)

def avg(key):
    vals = [r[key] for r in per_q.values() if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None

print("\n=== 노드수 검증 (generator 실제 입력 vs filter 출력) ===")
print(f"  filter_in (extractor 출력) 평균: {avg('filter_in_nodes')}")
print(f"  filter_out (final_nodes)  평균: {avg('filter_out_nodes')}")
print(f"  gen_main 실제 입력 노드   평균: {avg('gen_main_input_nodes')}")
eqs = [r['gen_main_eq_extractor'] for r in per_q.values() if r['gen_main_eq_extractor'] is not None]
print(f"  gen_main 프롬프트 == extractor-only 프롬프트: {sum(eqs)}/{len(eqs)}")
print("\n=== 조건별 쿼리당 평균 입력 토큰 (tiktoken cl100k 근사) ===")
print(f"  full_schema (→gen)         : {avg('tok_gen_full_schema')}")
print(f"  filter_fwd (83노드)         : {avg('tok_filter_fwd')}")
print(f"  filter_bwd (83노드)         : {avg('tok_filter_bwd')}")
print(f"  gen_actual (실제=extractor) : {avg('tok_gen_main_actual')}")
print(f"  gen_selector_only          : {avg('tok_gen_selector_only')}")
print(f"CSV → {csv_path}")
