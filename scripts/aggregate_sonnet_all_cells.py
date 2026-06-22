#!/usr/bin/env python
"""
Sonnet 전체 cell 통합 표 (2026-06-11 사용자 directive).
모든 cell의 Recall / Precision / F1 / EX 를 하나의 markdown 표로.
- batch cell (M4 anchor, STE k015~k090): batch_gate_report.json → R_mean/P_mean/F1_mean/EX (per-query mean)
- baseline (g_retriever/xiyansql/linkalign, main.py): metrics.txt → recall/precision/ex, F1=2PR/(P+R) (조화평균, HISTORY baseline 표 컨벤션)
- oracle (B1/B2/B3): oracle_gate_report.json → EX. R/P/F1 는 정의적(schema-membership) → oracle_baseline_rpf1.py 산출 있으면 병합, 없으면 '—'
사용: python scripts/aggregate_sonnet_all_cells.py
"""
import os, sys, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/experiments/sonnet_rebaseline_2026_06_10"

def f4(x):
    try: return f"{float(x):.4f}"
    except: return "—"

def read_batch(d):
    p = BASE / d / "batch_gate_report.json"
    if not p.exists(): return None
    j = json.load(open(p))
    return dict(R=j.get("R_mean"), P=j.get("P_mean"), F1=j.get("F1_mean"),
               EX=j.get("EX"), exc=j.get("EX_count",""), n=j.get("n_queries"))

def read_metrics(d):
    p = BASE / d / "metrics.txt"
    if not p.exists() or p.stat().st_size == 0: return None
    kv = {}
    for line in open(p):
        m = re.match(r"^(recall|precision|ex)\s*:\s*([\d.]+)", line.strip())
        if m: kv[m.group(1)] = float(m.group(2))
    if "recall" not in kv or "precision" not in kv: return None
    R, P = kv["recall"], kv["precision"]
    F1 = (2*R*P/(R+P)) if (R+P) > 0 else 0.0
    pred = BASE / d / "predictions.jsonl"
    n = sum(1 for _ in open(pred)) if pred.exists() else None
    return dict(R=R, P=P, F1=F1, EX=kv.get("ex"), exc="", n=n)

def read_oracle():
    p = BASE / "oracle_sonnet" / "oracle_gate_report.json"
    if not p.exists(): return {}
    j = json.load(open(p))
    return {k: v for k, v in j.get("cells", {}).items()}, j.get("n_per_cell")

# oracle R/P/F1 (정의적) — analyzer 산출(oracle_baseline_rpf1) 있으면 병합
ORACLE_RPF1 = {  # schema-membership 정의값 (B3=perfect). 산출 파일 있으면 override.
    "B1_full":       dict(R="1.0000", P="—", F1="—"),
    "B2_gold_table": dict(R="1.0000", P="—", F1="—"),
    "B3_gold_column":dict(R="1.0000", P="1.0000", F1="1.0000"),
}

ROWS = []  # (group, label, R, P, F1, EX, EX_count, n, note)

# 1) M4 canonical anchor
r = read_batch("m4_canonical_sonnet")
if r: ROWS.append(("Anchor", "M4 canonical (MSTPCSTUnion + BiFilter)", f4(r["R"]), f4(r["P"]), f4(r["F1"]), f4(r["EX"]), r["exc"], r["n"], "새 anchor"))
else: ROWS.append(("Anchor", "M4 canonical", "pending","pending","pending","pending","","","미완"))

# 2) STE frontier k015~k090
for k in ["015","020","025","030","040","050","060","070","080","090"]:
    r = read_batch(f"ste/ste_k{k}_sonnet")
    if r: ROWS.append(("STE frontier", f"STE k={int(k)}", f4(r["R"]), f4(r["P"]), f4(r["F1"]), f4(r["EX"]), r["exc"], r["n"], ""))
    else: ROWS.append(("STE frontier", f"STE k={int(k)}", "pending","pending","pending","pending","","","미완"))

# 3) Oracle B1/B2/B3
oc = read_oracle()
ocells, on = (oc if isinstance(oc, tuple) else ({}, None)) if oc else ({}, None)
for bc, lbl in [("B1_full","Oracle B1 (Full schema)"),("B2_gold_table","Oracle B2 (Gold table)"),("B3_gold_column","Oracle B3 (Gold column, perfect-SL)")]:
    rp = ORACLE_RPF1.get(bc, {})
    if ocells.get(bc):
        ex = ocells[bc]; full = (on==1534)
        ROWS.append(("Oracle", lbl, rp.get("R","—"), rp.get("P","—"), rp.get("F1","—"), f4(ex.get("EX")), ex.get("EX_count",""), on, "" if full else f"⚠ {on}q 검증본"))
    else:
        ROWS.append(("Oracle", lbl, rp.get("R","—"), rp.get("P","—"), rp.get("F1","—"), "pending","","","미완"))

# 4) Baselines
for d, lbl in [("baseline_g_retriever_sonnet","G-Retriever (PCST)"),
               ("baseline_linkalign_sonnet","LinkAlign"),
               ("baseline_xiyansql_sonnet","XiYan-SQL")]:
    r = read_metrics(d)
    if r: ROWS.append(("Baseline", lbl, f4(r["R"]), f4(r["P"]), f4(r["F1"]), f4(r["EX"]), "", r["n"], "F1=조화평균"))
    else:
        pred = BASE / d / "predictions.jsonl"
        n = sum(1 for _ in open(pred)) if pred.exists() else 0
        ROWS.append(("Baseline", lbl, "pending","pending","pending","pending","", f"{n}/1534", "진행중"))

# ── markdown 출력 ──
print("\n## Sonnet 4.6 전체 cell — Recall / Precision / F1 / EX\n")
print("| Group | Cell | Recall | Precision | F1 | EX | EX count | n | 비고 |")
print("|---|---|---|---|---|---|---|---|---|")
prev = None
for g, lbl, R, P, F1, EX, exc, n, note in ROWS:
    gcol = g if g != prev else ""
    prev = g
    print(f"| {gcol} | {lbl} | {R} | {P} | {F1} | {EX} | {exc} | {n} | {note} |")
print("\n★ cross-backbone(vLLM/GLM/Sonnet) 절대수치 직접비교 금지 — 결론(Filter Dominance, EX∝후보폭) 불변성만. STE/anchor=per-query mean, baseline F1=조화평균(2PR/(P+R)).")
