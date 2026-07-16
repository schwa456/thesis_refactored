"""
Stage 3 addendum — Recall 정의 통일 (element-level: 테이블 ∪ 컬럼). LLM 0 호출, seed=42.

element-level 정의:
  gold_all  = {gold tables (lower)} ∪ {gold cols (lower, bare)}
  pred_all  = {pred tables (lower)} ∪ {pred cols (lower, bare)}
  recall    = |gold_all ∩ pred_all| / |gold_all|     (per-query, 이후 macro mean)
  precision = |gold_all ∩ pred_all| / |pred_all|
  F1        = 2PR/(P+R) per-query 후 macro mean (main.py 관례와 동일: per-query 평균)
col-only 정의(대조): gold_cols/pred_cols 만 (main.py:205 방식).

입력(전부 기존): predictions.jsonl (Proposed: pred_tables/pred_cols; baseline: final_nodes),
  dev.json (gold SQL, difficulty), stage_bprime_per_query.csv (b′ per-query EX).
출력: stage3_unified_metrics.csv (그림6 좌표 겸용) + stage3_addendum_summary.json
"""
import os, sys, json, csv
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.evaluator import parse_sql_elements
from scipy.stats import binomtest

BASE = ROOT / "outputs/experiments/sonnet_rebaseline_2026_06_10"
DEV = json.load(open(ROOT / "data/raw/BIRD_dev/dev.json", encoding="utf-8"))
gold = {}      # qid -> (gt set, gc set) lower
diff = {}      # qid -> difficulty
for i, d in enumerate(DEV):
    qid = d.get("question_id", i)
    gt, gc = parse_sql_elements(d.get("SQL", d.get("query", "")))
    gold[qid] = (set(gt), set(gc))
    diff[qid] = str(d.get("difficulty", "")).lower()

def fn_to_tc(fn):
    pt, pc = set(), set()
    for n in fn or []:
        s = str(n)
        if "->" in s:
            continue
        if "." in s:
            t, c = s.split(".", 1); pt.add(t.lower()); pc.add(c.lower())
        else:
            pt.add(s.lower())
    return pt, pc

# method -> qid -> (pt set, pc set)
methods = {}
prop = {}
for l in open(BASE / "m4_canonical_sonnet/predictions.jsonl"):
    r = json.loads(l)
    prop[r["question_id"]] = (set(t.lower() for t in r.get("pred_tables", [])),
                              set(c.lower() for c in r.get("pred_cols", [])))
methods["Proposed"] = prop
for name, d in [("XiYanSQL", "baseline_xiyansql_sonnet"),
                ("G-Retriever", "baseline_g_retriever_sonnet"),
                ("LinkAlign", "baseline_linkalign_sonnet")]:
    m = {}
    for l in open(BASE / d / "predictions.jsonl"):
        r = json.loads(l)
        m[r["question_id"]] = fn_to_tc(r.get("final_nodes", []))
    methods[name] = m

# EX per-query: Proposed as-run(ex) + b′(stage_bprime) ; baseline ex_score
ex_asrun = {json.loads(l)["question_id"]: json.loads(l)["ex"]
            for l in open(BASE / "m4_canonical_sonnet/predictions.jsonl")}
ex_bprime = {}
for row in csv.DictReader(open(ROOT / "review_verification/stage_bprime_per_query.csv")):
    v = row["ex_bprime_filter"]
    ex_bprime[int(row["qid"])] = int(v) if v not in ("", "None") else None
base_ex = {}
for name, d in [("XiYanSQL", "baseline_xiyansql_sonnet"),
                ("G-Retriever", "baseline_g_retriever_sonnet"),
                ("LinkAlign", "baseline_linkalign_sonnet")]:
    base_ex[name] = {json.loads(l)["question_id"]: json.loads(l).get("ex_score")
                     for l in open(BASE / d / "predictions.jsonl")}

def rpf_query(pt, pc, gt, gc, level):
    if level == "element":
        g = gt | gc; p = pt | pc
    else:  # col-only
        g = set(gc); p = set(pc)
    if not g and not p:
        return 1.0, 1.0, 1.0
    inter = len(g & p)
    r = inter / len(g) if g else 0.0
    pr = inter / len(p) if p else 0.0
    f1 = 2 * r * pr / (r + pr) if (r + pr) > 0 else 0.0
    return r, pr, f1

def macro(method_map, ids, level):
    R = P = F = 0.0; n = 0
    for q in ids:
        gt, gc = gold[q]; pt, pc = method_map[q]
        r, pr, f = rpf_query(pt, pc, gt, gc, level)
        R += r; P += pr; F += f; n += 1
    return round(R/n, 4), round(P/n, 4), round(F/n, 4)

ids_all = sorted(set(prop) & set.intersection(*[set(m) for m in methods.values()]))

# ── 1. 전 method element vs col-only ──────────────────────────────────
table3 = []
for name, m in methods.items():
    er, ep, ef = macro(m, ids_all, "element")
    cr, cp, cf = macro(m, ids_all, "col")
    ex_v = round(sum(ex_asrun[q] for q in ids_all)/len(ids_all), 4) if name == "Proposed" else \
           round(sum(base_ex[name][q] for q in ids_all)/len(ids_all), 4)
    table3.append({"method": name, "R_elem": er, "P_elem": ep, "F1_elem": ef,
                   "R_col": cr, "P_col": cp, "F1_col": cf, "EX": ex_v})

# ── 2. 난이도별 (element-level) ───────────────────────────────────────
table5 = {}
for lv in ["simple", "moderate", "challenging"]:
    dids = [q for q in ids_all if diff[q] == lv]
    table5[lv] = {"n": len(dids)}
    for name, m in methods.items():
        er, ep, ef = macro(m, dids, "element")
        table5[lv][name] = {"R": er, "P": ep, "F1": ef}

# ── 4. McNemar (baseline vs Proposed b′ EX 0.6089) ───────────────────
def mcnemar(bex, pex, ids):
    b = sum(1 for q in ids if bex.get(q) == 1 and pex.get(q) == 0)
    c = sum(1 for q in ids if bex.get(q) == 0 and pex.get(q) == 1)
    p = binomtest(min(b, c), b+c, 0.5).pvalue if (b+c) > 0 else 1.0
    return b, c, p
def stars(p): return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

partA_bprime = []
for name in ["XiYanSQL", "G-Retriever", "LinkAlign"]:
    ids = [q for q in ids_all if base_ex[name].get(q) is not None and ex_bprime.get(q) is not None]
    b, c, p = mcnemar(base_ex[name], ex_bprime, ids)
    exb = sum(base_ex[name][q] for q in ids)/len(ids)
    partA_bprime.append({"method": name, "EX": round(exb,4), "dEX_vs_prop_bprime": round(exb-0.6089,4),
                         "b": b, "c": c, "p": p, "sig": stars(p), "n": len(ids)})

# ── 5. 그림6 좌표 CSV (element-level R/P/EX) ──────────────────────────
with open(ROOT / "review_verification/stage3_unified_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "R_element", "P_element", "F1_element", "R_col_only", "P_col_only", "F1_col_only", "EX_asrun_or_baseline", "EX_bprime"])
    for x in table3:
        exbp = 0.6089 if x["method"] == "Proposed" else ""
        w.writerow([x["method"], x["R_elem"], x["P_elem"], x["F1_elem"], x["R_col"], x["P_col"], x["F1_col"], x["EX"], exbp])

summary = {"n_queries": len(ids_all), "table3_element_vs_col": table3,
           "table5_by_difficulty_element": table5,
           "partA_mcnemar_bprime": partA_bprime,
           "note_table4": "oracle R/P/F1 = 정의적(Full=R1 all, GoldCol=1/1/1); batch 경로 내부일관.",
           "note_table5_source": "현행 표5 소스=paper draft; 채점경로 미기록 → 본 재계산(element-level, macro-per-query)로 대체 권장."}
json.dump(summary, open(ROOT / "review_verification/stage3_addendum_summary.json", "w"), indent=2, ensure_ascii=False)

print("=== 1. 표3 element vs col-only (전 method) ===")
for x in table3:
    print(f"  {x['method']:12s} | elem R/P/F1={x['R_elem']}/{x['P_elem']}/{x['F1_elem']} | col R/P/F1={x['R_col']}/{x['P_col']}/{x['F1_col']} | EX={x['EX']}")
print("\n=== 2. 표5 난이도별 (element-level) ===")
for lv, d in table5.items():
    print(f"  [{lv} n={d['n']}]")
    for name in methods:
        v = d[name]; print(f"    {name:12s} R/P/F1={v['R']}/{v['P']}/{v['F1']}")
print("\n=== 4. McNemar (baseline vs Proposed b′ 0.6089) ===")
for x in partA_bprime:
    print(f"  {x['method']:12s} EX={x['EX']} ΔEX={x['dEX_vs_prop_bprime']:+.4f} b={x['b']} c={x['c']} p={x['p']:.2e} {x['sig']}")
print("\nsaved: stage3_unified_metrics.csv, stage3_addendum_summary.json")
