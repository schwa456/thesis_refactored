#!/usr/bin/env python
"""Sonnet M4 anchor + baseline 3종 difficulty별 R/P/F1/EX (2026-06-11 사용자 요청).
R/P/F1 정의 = run_sonnet_batch_e2e.py 와 동일 (gold_all=gold_tables∪gold_cols, per-query mean).
M4: predictions.jsonl 의 pred_tables/pred_cols/ex 사용. baseline: final_nodes→pred 변환 + ex_score.
"""
import os, sys, json
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils.evaluator import parse_sql_elements

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEV = json.load(open(os.path.join(ROOT, "data/raw/BIRD_dev/dev.json")))
# qid → (difficulty, gold_tables, gold_cols)
GOLD = {}
for it in DEV:
    gt, gc = parse_sql_elements(it.get("SQL", ""))
    GOLD[it["question_id"]] = (it.get("difficulty", "?"),
                               set(t.lower() for t in gt), set(c.lower() for c in gc))

def rpf(gold_tables, gold_cols, pred_tables, pred_cols):
    gold_all = gold_tables | gold_cols; pred_all = pred_tables | pred_cols
    if not gold_all: return None
    inter = len(gold_all & pred_all)
    rec = inter / len(gold_all)
    prec = inter / len(pred_all) if pred_all else 0.0
    f1 = 2*rec*prec/(rec+prec) if rec+prec else 0.0
    return rec, prec, f1

def nodes_to_pred(final_nodes, gold_cols):
    pt, pc = set(), set()
    for node in final_nodes:
        if not isinstance(node, str) or '->' in node: continue
        if '.' in node:
            tbl, col = node.split('.', 1); pt.add(tbl.lower())
            tcl = f"{tbl.lower()}.{col.lower()}"
            pc.add(tcl if tcl in gold_cols else col.lower())
        else:
            pt.add(node.lower())
    return pt, pc

BASE = os.path.join(ROOT, "outputs/experiments/sonnet_rebaseline_2026_06_10")
EXPS = [
    ("M4 anchor",   f"{BASE}/m4_canonical_sonnet/predictions.jsonl",        "m4"),
    ("G-Retriever", f"{BASE}/baseline_g_retriever_sonnet/predictions.jsonl", "base"),
    ("LinkAlign",   f"{BASE}/baseline_linkalign_sonnet/predictions.jsonl",   "base"),
    ("XiYan-SQL",   f"{BASE}/baseline_xiyansql_sonnet/predictions.jsonl",    "base"),
]
DIFFS = ["simple", "moderate", "challenging", "ALL"]

print("\n## Sonnet difficulty별 R / P / F1 / EX (BIRD-Dev 1534: simple 925 / moderate 464 / challenging 145)\n")
for name, path, kind in EXPS:
    if not os.path.exists(path):
        print(f"### {name}: (predictions 없음)\n"); continue
    agg = {d: {"R":0.0,"P":0.0,"F1":0.0,"EX":0.0,"n":0} for d in DIFFS}
    for line in open(path):
        d = json.loads(line); qid = d.get("question_id")
        if qid not in GOLD: continue
        diff, gt, gc = GOLD[qid]
        if kind == "m4":
            pt = set(t.lower() for t in d.get("pred_tables", []))
            pc = set(c.lower() for c in d.get("pred_cols", []))
            ex = d.get("ex", 0)
        else:
            pt, pc = nodes_to_pred(d.get("final_nodes", []), gc)
            ex = d.get("ex_score", 0)
        r = rpf(gt, gc, pt, pc)
        if r is None: continue
        for tgt in (diff, "ALL"):
            a = agg[tgt]; a["R"]+=r[0]; a["P"]+=r[1]; a["F1"]+=r[2]; a["EX"]+=int(bool(ex)); a["n"]+=1
    print(f"### {name}")
    print("| difficulty | n | Recall | Precision | F1 | EX |")
    print("|---|---|---|---|---|---|")
    for d in DIFFS:
        a = agg[d]; n = max(a["n"],1)
        print(f"| {d} | {a['n']} | {a['R']/n:.4f} | {a['P']/n:.4f} | {a['F1']/n:.4f} | {a['EX']/n:.4f} |")
    print()
