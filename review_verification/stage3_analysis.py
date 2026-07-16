"""
Stage 3 — 메인 비교 유의성(McNemar) + strict recall(전 방법) + gold 파싱 규칙 검증.
LLM 0 호출, 순수 로컬 계산. seed=42 (Part C 샘플링).
출력: stage3_summary.json (+ 보고서에 반영). CSV: stage3_strict_recall_per_query.csv
"""
import os, sys, json, csv, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.evaluator import parse_sql_elements
from scipy.stats import binomtest, spearmanr
import sqlglot
from sqlglot.expressions import Star, Select, Alias, Column, Table

BASE = ROOT / "outputs/experiments/sonnet_rebaseline_2026_06_10"
DEV = json.load(open(ROOT / "data/raw/BIRD_dev/dev.json", encoding="utf-8"))
gold_sql_by_id = {d.get("question_id", i): d.get("SQL", d.get("query", "")) for i, d in enumerate(DEV)}
gold_parsed = {qid: parse_sql_elements(s) for qid, s in gold_sql_by_id.items()}  # (tables,cols) lower

def load_pred(path, ex_key, schema_key):
    out = {}
    for l in open(path):
        r = json.loads(l)
        out[r["question_id"]] = {"ex": r.get(ex_key), "schema": r.get(schema_key)}
    return out

# Proposed as-run (m4_canonical): ex + pred_tables/pred_cols
prop = {}
for l in open(BASE / "m4_canonical_sonnet/predictions.jsonl"):
    r = json.loads(l)
    prop[r["question_id"]] = {"ex": r["ex"], "pt": r.get("pred_tables", []), "pc": r.get("pred_cols", [])}

# baselines: ex_score + final_nodes ("table.Col")
BASELINES = {
    "XiYanSQL": "baseline_xiyansql_sonnet",
    "G-Retriever": "baseline_g_retriever_sonnet",
    "LinkAlign": "baseline_linkalign_sonnet",
}
base_pred = {}
for name, d in BASELINES.items():
    m = {}
    for l in open(BASE / d / "predictions.jsonl"):
        r = json.loads(l)
        m[r["question_id"]] = {"ex": r.get("ex_score"), "fn": r.get("final_nodes", []) or []}
    base_pred[name] = m

def strict_and_recall_from_tabcol(pt, pc, gold):
    gt, gc = gold
    gold_all = set(gt) | set(gc)
    pred_all = set(t.lower() for t in pt) | set(c.lower() for c in pc)
    if not gold_all:
        return None, None
    inter = len(gold_all & pred_all)
    return inter / len(gold_all), (1 if inter == len(gold_all) else 0)

def final_nodes_to_tabcol(fn):
    pt, pc = set(), set()
    for n in fn:
        nm = str(n)
        if "->" in nm:  # fk node
            continue
        if "." in nm:
            t, c = nm.split(".", 1); pt.add(t); pc.add(c)
        else:
            pt.add(nm)
    return list(pt), list(pc)

# ── Part A: McNemar (baseline vs Proposed as-run) ────────────────────
def mcnemar(base_ex, prop_ex, ids):
    b = sum(1 for q in ids if base_ex[q] == 1 and prop_ex[q] == 0)   # baseline only
    c = sum(1 for q in ids if base_ex[q] == 0 and prop_ex[q] == 1)   # proposed only
    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    return b, c, p

def stars(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

prop_ex = {q: prop[q]["ex"] for q in prop}
partA = []
for name, m in base_pred.items():
    ids = [q for q in m if q in prop_ex and m[q]["ex"] is not None]
    base_ex = {q: m[q]["ex"] for q in ids}
    ex_mean = sum(base_ex.values()) / len(ids)
    b, c, p = mcnemar(base_ex, prop_ex, ids)
    partA.append({"method": name, "EX": round(ex_mean, 4), "dEX_vs_prop": round(ex_mean - 0.6030, 4),
                  "mcn_b_baseline_only": b, "mcn_c_prop_only": c, "p": p, "sig": stars(p), "n": len(ids)})

# ── Part B: strict recall 전 방법 + Spearman ──────────────────────────
def method_recall_strict(schema_fn, ids):
    rec, strict = [], []
    for q in ids:
        pt, pc = schema_fn(q)
        r, s = strict_and_recall_from_tabcol(pt, pc, gold_parsed[q])
        if r is not None:
            rec.append(r); strict.append(s)
    return sum(rec) / len(rec), sum(strict) / len(strict)

partB = []
# Proposed: 표기값 사용 (task 지정) — recall 0.9539 / strict 0.8012 / EX 0.6089(b′). 재계산도 병기.
prop_ids = list(prop.keys())
prop_r, prop_s = method_recall_strict(lambda q: (prop[q]["pt"], prop[q]["pc"]), prop_ids)
partB.append({"method": "Proposed", "recall": 0.9539, "strict_recall": 0.8012, "EX": 0.6089,
              "recall_recomputed": round(prop_r, 4), "strict_recomputed": round(prop_s, 4)})
for name, m in base_pred.items():
    ids = [q for q in m if q in gold_parsed]
    r, s = method_recall_strict(lambda q: final_nodes_to_tabcol(m[q]["fn"]), ids)
    ex = sum(m[q]["ex"] for q in ids if m[q]["ex"] is not None) / len(ids)
    partB.append({"method": name, "recall": round(r, 4), "strict_recall": round(s, 4), "EX": round(ex, 4)})

# Spearman across methods (n=4): strict_recall vs EX, recall vs EX
methods_order = [x["method"] for x in partB]
EXs = [x["EX"] for x in partB]
strs_ = [x["strict_recall"] for x in partB]
recs = [x["recall"] for x in partB]
sp_strict = spearmanr(strs_, EXs)
sp_recall = spearmanr(recs, EXs)

# ── Part C: gold 파싱 규칙 검증 (100 샘플, seed=42) + 전역 SELECT* 비율 ──
def sql_features(sql):
    try:
        p = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return {"star": False, "n_select": 0, "alias_collision": False, "parse_err": True}
    has_star = any(True for _ in p.find_all(Star))
    n_select = sum(1 for _ in p.find_all(Select))
    alias_names = set(a.alias.lower() for a in p.find_all(Alias) if a.alias)
    # 실제 컬럼 참조명 (table.col 형태로 쓰인 것 = 진짜 DB 컬럼)
    real_qualified_cols = set(c.name.lower() for c in p.find_all(Column) if c.name and c.table)
    alias_collision = bool(alias_names & real_qualified_cols)
    return {"star": has_star, "n_select": n_select, "alias_collision": alias_collision, "parse_err": False}

# 전역 (전체 1534)
glob_star = glob_nested = glob_collision = glob_err = 0
for qid, s in gold_sql_by_id.items():
    f = sql_features(s)
    glob_star += f["star"]; glob_nested += (f["n_select"] >= 2); glob_collision += f["alias_collision"]; glob_err += f["parse_err"]
N = len(gold_sql_by_id)

# 100 샘플 (seed=42)
rng = random.Random(42)
sample_ids = sorted(rng.sample(list(gold_sql_by_id.keys()), 100))
flags = []
for qid in sample_ids:
    s = gold_sql_by_id[qid]; f = sql_features(s)
    tags = []
    if f["star"]: tags.append("SELECT_STAR")
    if f["n_select"] >= 2: tags.append("NESTED")
    if f["alias_collision"]: tags.append("ALIAS_COLLISION")
    if tags:
        gt, gc = gold_parsed[qid]
        flags.append({"qid": qid, "flags": tags, "sql": s[:200], "gold_tables": sorted(gt), "gold_cols": sorted(gc)})

summary = {
    "partA_mcnemar": partA,
    "partB_strict_recall": partB,
    "partB_spearman_strict_vs_EX": {"rho": round(sp_strict.correlation, 4), "p": round(sp_strict.pvalue, 4), "n_methods": len(partB)},
    "partB_spearman_recall_vs_EX": {"rho": round(sp_recall.correlation, 4), "p": round(sp_recall.pvalue, 4)},
    "partC_global": {"n": N, "select_star": glob_star, "select_star_pct": round(glob_star/N*100, 2),
                     "nested_subquery": glob_nested, "nested_pct": round(glob_nested/N*100, 2),
                     "alias_collision": glob_collision, "parse_error": glob_err},
    "partC_sample_seed": 42, "partC_sample_n": 100,
    "partC_sample_flag_counts": {
        "SELECT_STAR": sum(1 for x in flags if "SELECT_STAR" in x["flags"]),
        "NESTED": sum(1 for x in flags if "NESTED" in x["flags"]),
        "ALIAS_COLLISION": sum(1 for x in flags if "ALIAS_COLLISION" in x["flags"]),
        "total_flagged": len(flags)},
    "partC_flags": flags,
}
json.dump(summary, open(ROOT / "review_verification/stage3_summary.json", "w"), indent=2, ensure_ascii=False)

# CSV (strict recall per query, 전 방법)
with open(ROOT / "review_verification/stage3_strict_recall_per_query.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["qid", "method", "recall", "strict_recall", "ex"])
    for name, m in base_pred.items():
        for q in m:
            if q not in gold_parsed: continue
            pt, pc = final_nodes_to_tabcol(m[q]["fn"])
            r, s = strict_and_recall_from_tabcol(pt, pc, gold_parsed[q])
            if r is not None: w.writerow([q, name, round(r, 4), s, m[q]["ex"]])
    for q in prop:
        r, s = strict_and_recall_from_tabcol(prop[q]["pt"], prop[q]["pc"], gold_parsed[q])
        if r is not None: w.writerow([q, "Proposed", round(r, 4), s, prop[q]["ex"]])

# stdout
print("=== Part A: McNemar (baseline vs Proposed as-run 0.6030) ===")
for x in partA: print(f"  {x['method']:12s} EX={x['EX']} ΔEX={x['dEX_vs_prop']:+.4f} b={x['mcn_b_baseline_only']} c={x['mcn_c_prop_only']} p={x['p']:.2e} {x['sig']}")
print("\n=== Part B: strict recall 전 방법 ===")
for x in partB: print(f"  {x['method']:12s} recall={x['recall']} strict={x['strict_recall']} EX={x['EX']}" + (f" (재계산 r={x.get('recall_recomputed')}/s={x.get('strict_recomputed')})" if 'recall_recomputed' in x else ""))
print(f"  Spearman(strict, EX)={sp_strict.correlation:.4f} (p={sp_strict.pvalue:.4f}) | Spearman(recall, EX)={sp_recall.correlation:.4f} (p={sp_recall.pvalue:.4f}) [n={len(partB)}]")
print("\n=== Part C: gold 파싱 검증 ===")
print(f"  전역(1534): SELECT* {glob_star}({glob_star/N*100:.1f}%) | nested {glob_nested}({glob_nested/N*100:.1f}%) | alias충돌 {glob_collision} | parse오류 {glob_err}")
print(f"  100샘플(seed=42) flag: {summary['partC_sample_flag_counts']}")
print("saved: stage3_summary.json, stage3_strict_recall_per_query.csv")
