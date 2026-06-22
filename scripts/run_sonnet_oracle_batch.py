#!/usr/bin/env python
"""
Oracle EX (Sonnet) — gold subschema → Sonnet SQLgen → EX (DECISIONS 2026-06-10 #6 §confirm Task 1).
B1 Full / B2 Gold Table / B3 Gold Column. B3 = perfect-SL upper bound (vs GLM EX 0.6239).
단일 LLM stage(SQLgen) → collect-replay batch. R/P/F1는 정의적(LLM 무관, B3=1.0)이라 EX만 측정.
사용: python scripts/run_sonnet_oracle_batch.py [--limit N]
"""
import os, sys, json, time, argparse, concurrent.futures
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
except ImportError: pass
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--model", default="claude-sonnet-4-6")
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--poll", type=int, default=20)
ns = ap.parse_args()

from utils.evaluator import parse_sql_elements
from utils.executor import evaluate_ex
from modules.generators.sql_generator import LLMSQLGenerator
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

DEV_JSON = ROOT / "data/raw/BIRD_dev/dev.json"
DEV_TABLES = ROOT / "data/raw/BIRD_dev/dev_tables.json"
DB_DIR = ROOT / "data/raw/BIRD_dev/dev_databases"
OUT = ROOT / "outputs/experiments/sonnet_rebaseline_2026_06_10/oracle_sonnet"
OUT.mkdir(parents=True, exist_ok=True)

def load_db_lookup():
    out = {}
    for db in json.load(open(DEV_TABLES)):
        tnames = [t.lower() for t in db.get("table_names_original", [])]
        cbt = {t: [] for t in tnames}
        for ti, col in db.get("column_names_original", []):
            if ti == -1: continue
            if 0 <= ti < len(tnames): cbt[tnames[ti]].append(col.lower())
        out[db["db_id"]] = cbt
    return out

db_lookup = load_db_lookup()
dataset = json.load(open(DEV_JSON, encoding="utf-8"))
if ns.limit and 0 < ns.limit < len(dataset):
    stride = len(dataset)/ns.limit
    dataset = [dataset[int(i*stride)] for i in range(ns.limit)]

def build_subgraph(cell, db_id, gold_sql):
    cbt = db_lookup.get(db_id, {})
    if cell == "B1_full":
        return {t: list(c) for t, c in cbt.items() if c}
    gt, gc = parse_sql_elements(gold_sql)
    gt = {t.lower() for t in gt}; gcs = {c.lower().split(".")[-1] for c in gc}
    if cell == "B2_gold_table":
        return {t: list(cbt.get(t, [])) for t in gt if cbt.get(t)}
    if cell == "B3_gold_column":
        sg = {}
        for t in gt:
            cols = [c for c in cbt.get(t, []) if c in gcs]
            if cols: sg[t] = cols
        return sg or {t: list(cbt.get(t, [])) for t in gt if cbt.get(t)}  # fallback

client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
gen = LLMSQLGenerator(llm_model=ns.model, temperature=0.0, provider="sonnet")

results_map, p2cid, cid2p, submitted = {}, {}, {}, set()
USAGE = {"in":0,"out":0,"cache":0,"calls":0}
COLLECT = {"on": False}
def register(p):
    if p not in p2cid:
        c=f"r{len(p2cid)}"; p2cid[p]=c; cid2p[c]=p
def patched(prompt, model=None, temperature=None):
    if prompt in results_map: return results_map[prompt]
    if COLLECT["on"]: register(prompt)
    return "SELECT 1"
gen.client.generate_text = patched

CELLS = ["B1_full","B2_gold_table","B3_gold_column"]
# query×cell 단위로 prompt 수집/replay (subgraph 결정적)
items = [(it, cell) for it in dataset for cell in CELLS]

def run_all():
    for it, cell in items:
        sg = build_subgraph(cell, it.get("db_id"), it.get("SQL", it.get("query","")))
        try: gen.generate(query=it.get("question"), subgraph=sg, evidence=it.get("evidence",""))
        except Exception as e: pass

def submit_batch():
    pending=[c for c in cid2p if c not in submitted]
    if not pending: return
    reqs=[Request(custom_id=c, params=MessageCreateParamsNonStreaming(
        model=ns.model, max_tokens=ns.max_tokens,
        system=[{"type":"text","text":"You are a helpful database expert."}],
        messages=[{"role":"user","content":cid2p[c]}], temperature=0.0)) for c in pending]
    print(f"[oracle] submit batch: {len(reqs)} reqs", flush=True)
    b=client.messages.batches.create(requests=reqs)
    while True:
        bb=client.messages.batches.retrieve(b.id)
        if bb.processing_status=="ended": break
        print(f"[oracle] {bb.processing_status} done={bb.request_counts.succeeded}+{bb.request_counts.errored}", flush=True); time.sleep(ns.poll)
    for res in client.messages.batches.results(b.id):
        if res.result.type=="succeeded":
            m=res.result.message; txt="".join(x.text for x in m.content if x.type=="text")
            results_map[cid2p[res.custom_id]]=txt
            u=m.usage; USAGE["in"]+=u.input_tokens; USAGE["out"]+=u.output_tokens
            USAGE["cache"]+=int(getattr(u,"cache_read_input_tokens",0) or 0); USAGE["calls"]+=1
        else: results_map[cid2p[res.custom_id]]="SELECT 'API ERROR'"
        submitted.add(res.custom_id)
    print(f"[oracle] batch done ok={USAGE['calls']}", flush=True)

t0=time.time()
COLLECT["on"]=True; run_all(); COLLECT["on"]=False
submit_batch()
# replay + EX 평가
per_cell={c:{"ex":0,"n":0} for c in CELLS}
for it, cell in items:
    qid=it.get("question_id"); db_id=it.get("db_id"); gold=it.get("SQL", it.get("query",""))
    dbp=str(DB_DIR/db_id/f"{db_id}.sqlite")
    sg=build_subgraph(cell, db_id, gold)
    try: sql=gen.generate(query=it.get("question"), subgraph=sg, evidence=it.get("evidence",""))
    except Exception: sql=""
    ex=0
    if sql and gold and os.path.exists(dbp):
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex_pool:
                ex=ex_pool.submit(evaluate_ex, pred_sql=sql, gold_sql=gold, db_path=dbp).result(timeout=15.0)
        except Exception: ex=0
    per_cell[cell]["ex"]+=ex; per_cell[cell]["n"]+=1

rep={"n_per_cell":len(dataset),"wall_s":round(time.time()-t0,1),"usage":USAGE,
     "cost_usd_batch50pct":round((USAGE["in"]*3+USAGE["out"]*15)/1e6*0.5,4),
     "cells":{c:{"EX":round(per_cell[c]["ex"]/max(per_cell[c]["n"],1),4),"EX_count":f"{per_cell[c]['ex']}/{per_cell[c]['n']}"} for c in CELLS}}
json.dump(rep, open(OUT/"oracle_gate_report.json","w"), indent=2)
print(json.dumps(rep, indent=2, ensure_ascii=False))
