#!/usr/bin/env python
"""Ideal envelope oracle sweep (Sonnet) — P=1 / R=1 의 EX vs candidate width 실측.

run_sonnet_oracle_batch.py 확장. dome envelope (docs/figs/fig_ex_width_ideal_envelope.png) 의
좌측 팔(P=1)·우측 팔(R=1) 을 이론선 → 실측선으로 채운다.

cells:
  P=1 (gold-subset, noise=0, recall<1):  P1_20 / P1_40 / P1_60 / P1_80   (gold col 의 fraction)
  apex (R=1 & P=1):                       B3_gold_column                  (= P1_100 = R1_n0)
  R=1 (gold + m noise col, recall=1):     R1_n2 / R1_n5 / R1_n10 / R1_n20 / R1_n40

각 cell: EX + avg width(tables+cols) + avg recall + avg precision (idealization 검증) 기록.
SQLgen 단일 stage(temperature 0.0) collect→batch→replay. 결정적 subgraph.
사용: python scripts/run_ideal_envelope_oracle.py [--limit N]
"""
import os, sys, json, time, argparse, concurrent.futures
from math import ceil
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
OUT = ROOT / "outputs/experiments/sonnet_rebaseline_2026_06_10/oracle_ideal_envelope"
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

# fraction / noise 정의
P1_FRAC = {"P1_20":0.2, "P1_40":0.4, "P1_60":0.6, "P1_80":0.8}
R1_NOISE = {"R1_n2":2, "R1_n5":5, "R1_n10":10, "R1_n20":20, "R1_n40":40}
CELLS = ["P1_20","P1_40","P1_60","P1_80","B3_gold_column",
         "R1_n2","R1_n5","R1_n10","R1_n20","R1_n40"]

def gold_pairs_of(cbt, gold_sql):
    """gold (table,col) 쌍 — B3 정의와 동일 (deterministic sorted)."""
    gt, gc = parse_sql_elements(gold_sql)
    gt = {t.lower() for t in gt}
    gcs = {c.lower().split(".")[-1] for c in gc}
    pairs = []
    for t in sorted(gt):
        for c in sorted(cbt.get(t, [])):
            if c in gcs:
                pairs.append((t, c))
    return pairs

def build_subgraph(cell, db_id, gold_sql):
    cbt = db_lookup.get(db_id, {})
    gp = gold_pairs_of(cbt, gold_sql)
    if not gp:  # fallback: gold table 전체 (B3 fallback 과 동일 취지)
        gt, _ = parse_sql_elements(gold_sql)
        return {t.lower(): list(cbt.get(t.lower(), [])) for t in gt if cbt.get(t.lower())}
    gold_set = set(gp)
    if cell == "B3_gold_column":
        keep = gp
    elif cell in P1_FRAC:                      # P=1: gold 의 prefix fraction (noise 0)
        n = len(gp); k = max(1, ceil(P1_FRAC[cell] * n))
        keep = gp[:k]
    elif cell in R1_NOISE:                      # R=1: gold 전체 + m noise col
        m = R1_NOISE[cell]
        noise = [(t, c) for t in sorted(cbt) for c in sorted(cbt[t]) if (t, c) not in gold_set]
        keep = gp + noise[:m]
    else:
        keep = gp
    sg = {}
    for t, c in keep:
        sg.setdefault(t, []).append(c)
    return sg

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

items = [(it, cell) for it in dataset for cell in CELLS]

def run_all():
    for it, cell in items:
        sg = build_subgraph(cell, it.get("db_id"), it.get("SQL", it.get("query","")))
        try: gen.generate(query=it.get("question"), subgraph=sg, evidence=it.get("evidence",""))
        except Exception: pass

def submit_batch():
    pending=[c for c in cid2p if c not in submitted]
    if not pending: return
    reqs=[Request(custom_id=c, params=MessageCreateParamsNonStreaming(
        model=ns.model, max_tokens=ns.max_tokens,
        system=[{"type":"text","text":"You are a helpful database expert."}],
        messages=[{"role":"user","content":cid2p[c]}], temperature=0.0)) for c in pending]
    print(f"[envelope] submit batch: {len(reqs)} reqs", flush=True)
    b=client.messages.batches.create(requests=reqs)
    while True:
        bb=client.messages.batches.retrieve(b.id)
        if bb.processing_status=="ended": break
        print(f"[envelope] {bb.processing_status} done={bb.request_counts.succeeded}+{bb.request_counts.errored}", flush=True); time.sleep(ns.poll)
    for res in client.messages.batches.results(b.id):
        if res.result.type=="succeeded":
            m=res.result.message; txt="".join(x.text for x in m.content if x.type=="text")
            results_map[cid2p[res.custom_id]]=txt
            u=m.usage; USAGE["in"]+=u.input_tokens; USAGE["out"]+=u.output_tokens
            USAGE["cache"]+=int(getattr(u,"cache_read_input_tokens",0) or 0); USAGE["calls"]+=1
        else: results_map[cid2p[res.custom_id]]="SELECT 'API ERROR'"
        submitted.add(res.custom_id)
    print(f"[envelope] batch done ok={USAGE['calls']}", flush=True)

t0=time.time()
COLLECT["on"]=True; run_all(); COLLECT["on"]=False
submit_batch()

# replay + EX 평가 + width/R/P 집계 (idealization 검증)
per={c:{"ex":0,"n":0,"width":0.0,"rec":0.0,"prec":0.0} for c in CELLS}
for it, cell in items:
    db_id=it.get("db_id"); gold=it.get("SQL", it.get("query",""))
    dbp=str(DB_DIR/db_id/f"{db_id}.sqlite")
    cbt=db_lookup.get(db_id, {})
    sg=build_subgraph(cell, db_id, gold)
    # width / recall / precision (col-level)
    sg_cols={(t,c) for t,cs in sg.items() for c in cs}
    gold_set=set(gold_pairs_of(cbt, gold))
    inter=len(gold_set & sg_cols)
    width=len(sg)+sum(len(v) for v in sg.values())   # tables + columns 노드 수
    rec=inter/max(len(gold_set),1); prec=inter/max(len(sg_cols),1)
    try: sql=gen.generate(query=it.get("question"), subgraph=sg, evidence=it.get("evidence",""))
    except Exception: sql=""
    ex=0
    if sql and gold and os.path.exists(dbp):
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex_pool:
                ex=ex_pool.submit(evaluate_ex, pred_sql=sql, gold_sql=gold, db_path=dbp).result(timeout=15.0)
        except Exception: ex=0
    d=per[cell]; d["ex"]+=ex; d["n"]+=1; d["width"]+=width; d["rec"]+=rec; d["prec"]+=prec

def fin(d):
    n=max(d["n"],1)
    return {"EX":round(d["ex"]/n,4),"EX_count":f"{d['ex']}/{d['n']}",
            "avg_width":round(d["width"]/n,2),"avg_recall":round(d["rec"]/n,4),
            "avg_precision":round(d["prec"]/n,4)}

rep={"n_per_cell":len(dataset),"wall_s":round(time.time()-t0,1),"usage":USAGE,
     "cost_usd_batch50pct":round((USAGE["in"]*3+USAGE["out"]*15)/1e6*0.5,4),
     "cells":{c:fin(per[c]) for c in CELLS}}
json.dump(rep, open(OUT/"ideal_envelope_report.json","w"), indent=2)
print(json.dumps(rep, indent=2, ensure_ascii=False))
