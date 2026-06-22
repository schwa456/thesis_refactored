#!/usr/bin/env python3
"""Oracle 갭 분해 (paper novelty 방어선 iii, DECISIONS 2026-06-12 risk 3).
anchor m4_canonical_sonnet (EX 0.6030) vs B3 perfect-SL oracle (EX 0.7190), gap 0.1160.
query-level: anchor full-recall vs missed-gold 버킷 EX 교차표 → recall-loss vs precision/SQLgen-loss 분해.
"""
import os, sys, json, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # src/ (utils.evaluator)
from utils.evaluator import parse_sql_elements

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")  # thesis_refactored/
OUT = os.path.join(ROOT, "outputs/analysis")
ANCHOR = os.path.join(ROOT, "outputs/experiments/sonnet_rebaseline_2026_06_10/m4_canonical_sonnet/predictions.jsonl")
ORACLE_B3 = 0.7190  # oracle_sonnet gate_report B3_gold_column EX (per-query 미기록 → aggregate)

DEV = json.load(open(os.path.join(ROOT, "data/raw/BIRD_dev/dev.json")))
GOLD = {}
for it in DEV:
    gt, gc = parse_sql_elements(it.get("SQL", ""))
    GOLD[it["question_id"]] = (set(t.lower() for t in gt), set(c.lower() for c in gc))

fr, mr = [], []  # full-recall EX list, missed-gold EX list
for line in open(ANCHOR):
    d = json.loads(line); qid = d["question_id"]
    if qid not in GOLD:
        continue
    gt, gc = GOLD[qid]; gold = gt | gc
    if not gold:
        continue
    pred = set(t.lower() for t in d.get("pred_tables", [])) | set(c.lower() for c in d.get("pred_cols", []))
    rec = len(gold & pred) / len(gold)
    ex = int(bool(d.get("ex", 0)))
    (fr if rec >= 0.999 else mr).append(ex)

N = len(fr) + len(mr)
E_fr, E_mr = statistics.mean(fr), statistics.mean(mr)
w_mr = len(mr) / N
anchor_ex = (sum(fr) + sum(mr)) / N
gap = ORACLE_B3 - anchor_ex
recall_loss = w_mr * (E_fr - E_mr)          # missed-gold 버킷의 EX 결손
prec_loss = ORACLE_B3 - E_fr                # full-recall 인데도 oracle 미달 = extra-noise(precision)+residual
sqlgen_floor = 1 - ORACLE_B3                # 완벽 SL 에도 실패 (SL-irrecoverable)

print(f"query-level cross-table (anchor, N={N}):")
print(f"  full-recall (R=1.0) : n={len(fr)} ({len(fr)/N*100:.1f}%)  EX={E_fr:.4f}")
print(f"  missed-gold (R<1.0) : n={len(mr)} ({len(mr)/N*100:.1f}%)  EX={E_mr:.4f}")
print(f"  anchor overall EX   : {anchor_ex:.4f}")
print()
print(f"oracle B3 (perfect SL) EX = {ORACLE_B3}")
print(f"gap = {gap:.4f}")
print(f"=== 분해 (gap = recall_loss + precision_loss, 정확 identity) ===")
print(f"  recall-loss     = w_mr*(E_fr-E_mr) = {w_mr:.4f}*{E_fr-E_mr:.4f} = {recall_loss:.4f}  ({recall_loss/gap*100:.0f}% of gap)")
print(f"  precision-loss  = oracleB3 - E_fr  = {prec_loss:.4f}  ({prec_loss/gap*100:.0f}% of gap)")
print(f"  (sum = {recall_loss+prec_loss:.4f} vs gap {gap:.4f})")
print(f"SQLgen-floor (SL-irrecoverable) = 1-oracleB3 = {sqlgen_floor:.4f} (28.1%)")
print(f"전체 anchor 실패 {1-anchor_ex:.4f} = SQLgen-floor {sqlgen_floor:.4f} + SL-gap {gap:.4f}")

json.dump(dict(N=N, n_fr=len(fr), n_mr=len(mr), E_fr=E_fr, E_mr=E_mr, anchor_ex=anchor_ex,
               oracle_b3=ORACLE_B3, gap=gap, recall_loss=recall_loss, precision_loss=prec_loss,
               sqlgen_floor=sqlgen_floor),
          open(os.path.join(OUT, "oracle_gap_decomp_sonnet_2026-06-12.json"), "w"), indent=2)
print(f"\nWrote {OUT}/oracle_gap_decomp_sonnet_2026-06-12.json")
