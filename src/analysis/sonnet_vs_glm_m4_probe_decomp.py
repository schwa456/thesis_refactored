#!/usr/bin/env python3
"""Sonnet 4.6 probe vs GLM-4.7 — M4 anchor EX 하락 원인 stage 분해 (DECISIONS 2026-06-10 #6).

판별: EX 하락이 (a) parser/config/harness artifact 인가 (b) genuine SQL dialect / filter pruning 인가.

핵심 비교축:
- 같은 extractor cell(STE k015/k020) + 같은 raw EX 플래그 harness 에서 Sonnet vs GLM final EX (paired, 1534q)
  → SQLgen+filter backbone 효과를 노드셋 고정 상태에서 분리.
- GLM stage-wise EX(selector_only / extractor_only / final) → filter EX cost vs extractor EX cost 분해.
- Sonnet m4_anchor(MSTPCSTUnion) final R/P + gold 제거 → filter pruning 성향.
- McNemar paired test + Wilson CI → 차이 유의성/noise band.
"""
import json, os, statistics, math
from collections import Counter

ROOT = "/home/hyeonjin/thesis_refactored"
SON = os.path.join(ROOT, "outputs/experiments/sonnet_rebaseline_2026_06_10")
GLM = os.path.join(ROOT, "outputs/experiments/ste_topk_e2e_confirm_2026_06_09")
OUT = os.path.join(ROOT, "outputs/analysis")


def load(path, key_ex):
    rows = {}
    for l in open(path):
        r = json.loads(l)
        rows[r["question_id"]] = r
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar(pairs):
    """pairs: list of (a_ex, b_ex) 0/1.  a=sonnet b=glm.  return b01(a win),b10(b win),p."""
    b01 = sum(1 for a, b in pairs if a == 1 and b == 0)  # sonnet correct, glm wrong
    b10 = sum(1 for a, b in pairs if a == 0 and b == 1)  # glm correct, sonnet wrong
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    # exact binomial two-sided (p=0.5)
    from math import comb
    k = min(b01, b10)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return b01, b10, min(1.0, p)


SON_STE = {k: os.path.join(SON, f"ste/ste_k{k}_sonnet/predictions.jsonl") for k in ["015", "020", "025"]}
GLM_STE = {k: os.path.join(GLM, f"m4_ste_k{k}_e2e/predictions.jsonl") for k in ["015", "020", "025", "030"]}

# 카논 GLM M4 anchor (EnrichedHeteroGraphBuilder + selector alpha=0.5, EX=0.5300 출처)
GLM_M4 = os.path.join(ROOT, "outputs/experiments/abl/wave6_recall_biased/w6_p2_m4_bidirectional")

print("=" * 90)
print("[측정 0 — 핵심] M4 anchor 직접 비교 (같은 metric harness, 같은 300 qid) + config diff")
print("=" * 90)
son_o = {json.loads(l)["question_id"]: json.loads(l)
         for l in open(os.path.join(SON, "m4_anchor_sonnet/output_m4_anchor_sonnet.jsonl"))}
glm_o = {json.loads(l)["question_id"]: json.loads(l)
         for l in open(os.path.join(GLM_M4, "output_w6_p2_m4_bidirectional.jsonl"))}
glm_p = {json.loads(l)["question_id"]: json.loads(l) for l in open(os.path.join(GLM_M4, "predictions.jsonl"))}
common_m4 = sorted(set(son_o) & set(glm_o))


def agg3(rows):
    return (statistics.mean(r["recall"] for r in rows), statistics.mean(r["precision"] for r in rows),
            statistics.mean(r["ex"] for r in rows))


sR, sP, sE = agg3([son_o[q] for q in common_m4])
gR, gP, gE = agg3([glm_o[q] for q in common_m4])
# 후보폭 (extractor 출력 노드)
glm_ext = statistics.mean(glm_p[q]["extractor_info"]["extractor_num_selected_nodes"]
                          for q in common_m4 if glm_p.get(q, {}).get("extractor_info"))
print(f"matched qids = {len(common_m4)} (Sonnet 300q ∩ GLM 1534q), 동일 output_*.jsonl R/P/EX 정의")
print(f"  config       | builder                    | sel alpha | extractor 출력노드")
print(f"  GLM  M4      | EnrichedHeteroGraphBuilder | 0.5       | ~{glm_ext:.0f}")
print(f"  Sonnet M4    | HeteroGraphBuilder (plain) | 0.0       | ~9.13")
print(f"  {'':12s} {'R':>8} {'P':>8} {'EX':>8}")
print(f"  GLM  M4      {gR:>8.4f} {gP:>8.4f} {gE:>8.4f}")
print(f"  Sonnet M4    {sR:>8.4f} {sP:>8.4f} {sE:>8.4f}")
print(f"  Δ(Son-GLM)   {sR-gR:>+8.4f} {sP-gP:>+8.4f} {sE-gE:>+8.4f}")
from math import comb
pm = [(son_o[q]["ex"], glm_o[q]["ex"]) for q in common_m4]
b01 = sum(1 for a, b in pm if a == 1 and b == 0); b10 = sum(1 for a, b in pm if a == 0 and b == 1)
nn = b01 + b10; kk = min(b01, b10)
pval = min(1, sum(comb(nn, i) for i in range(kk + 1)) / 2 ** nn * 2) if nn else 1
print(f"  EX paired McNemar: Son-only={b01} GLM-only={b10} p={pval:.3g}  → config(builder+alpha) 교란, backbone isolation 아님")
m4_direct = dict(n=len(common_m4), glm=dict(R=gR, P=gP, EX=gE, ext=glm_ext),
                 son=dict(R=sR, P=sP, EX=sE, ext=9.13), mcnemar_p=pval, b01=b01, b10=b10)


print("=" * 90)
print("[측정 1] parser/empty-SQL artifact 검사")
print("=" * 90)


def ex_empty(rows, exkey, sqlkey):
    ex = [r[exkey] for r in rows.values()]
    empty = sum(1 for r in rows.values() if not (r.get(sqlkey) or "").strip())
    return statistics.mean(ex), empty, len(ex)


print(f"{'cell':12s} {'backbone':8s} {'n':>5} {'EX':>8} {'empty_sql':>10} {'parser_fail':>12}")
# Sonnet m4_anchor full
son_m4 = load(os.path.join(SON, "m4_anchor_sonnet/predictions.jsonl"), "ex")
exm, em, n = ex_empty(son_m4, "ex", "predicted_sql")
print(f"{'m4_anchor':12s} {'sonnet':8s} {n:>5} {exm:>8.4f} {em:>10} {'0.0 (gate)':>12}")
for k in ["015", "020", "025"]:
    rows = load(SON_STE[k], "ex")
    exm, em, n = ex_empty(rows, "ex", "predicted_sql")
    print(f"{'ste_k'+k:12s} {'sonnet':8s} {n:>5} {exm:>8.4f} {em:>10} {'0.0 (gate)':>12}")
for k in ["015", "020", "025", "030"]:
    if not os.path.exists(GLM_STE[k]):
        continue
    rows = load(GLM_STE[k], "ex_score")
    exm, em, n = ex_empty(rows, "ex_score", "generated_sql")
    print(f"{'ste_k'+k:12s} {'glm':8s} {n:>5} {exm:>8.4f} {em:>10} {'(see status)':>12}")

print("\n" + "=" * 90)
print("[측정 3+핵심] 같은 extractor cell, 같은 harness — Sonnet vs GLM final EX (paired, 노드셋 고정)")
print("=" * 90)
print(f"{'cell':10s} {'n_common':>9} {'Son EX':>8} {'GLM EX':>8} {'ΔEX(S-G)':>9} "
      f"{'S>G':>5} {'G>S':>5} {'McNemar p':>10} {'Δ 95%CI':>22}")
paired_summary = {}
for k in ["015", "020", "025"]:
    if not os.path.exists(GLM_STE[k]):
        continue
    sr = load(SON_STE[k], "ex")
    gr = load(GLM_STE[k], "ex_score")
    common = sorted(set(sr) & set(gr))
    pairs = [(sr[q]["ex"], gr[q]["ex_score"]) for q in common]
    sEX = statistics.mean(a for a, b in pairs)
    gEX = statistics.mean(b for a, b in pairs)
    b01, b10, p = mcnemar(pairs)
    d = sEX - gEX
    # paired diff CI (normal approx on per-query diff)
    diffs = [a - b for a, b in pairs]
    sd = statistics.pstdev(diffs)
    se = sd / math.sqrt(len(diffs))
    lo, hi = d - 1.96 * se, d + 1.96 * se
    paired_summary[k] = dict(n=len(common), son=sEX, glm=gEX, d=d, p=p, ci=[lo, hi], b01=b01, b10=b10)
    print(f"{'ste_k'+k:10s} {len(common):>9} {sEX:>8.4f} {gEX:>8.4f} {d:>+9.4f} "
          f"{b01:>5} {b10:>5} {p:>10.4f} {f'[{lo:+.4f},{hi:+.4f}]':>22}")

print("\n" + "=" * 90)
print("[측정 2] GLM stage-wise EX 분해 — filter EX cost vs extractor EX cost (어디서 깎이나)")
print("=" * 90)
print(f"{'cell':10s} {'sel_only':>9} {'ext_only':>9} {'final':>9} "
      f"{'Δext(e-s)':>10} {'Δfilter(f-e)':>12} {'status_unans':>12}")
glm_stage = {}
for k in ["015", "020"]:
    rows = [json.loads(l) for l in open(GLM_STE[k])]
    sel = statistics.mean(r["ex_score_selector_only"] for r in rows if r.get("ex_score_selector_only") is not None)
    ext = statistics.mean(r["ex_score_extractor_only"] for r in rows if r.get("ex_score_extractor_only") is not None)
    fin = statistics.mean(r["ex_score"] for r in rows)
    unans = sum(1 for r in rows if r.get("status") == "Unanswerable")
    glm_stage[k] = dict(sel=sel, ext=ext, fin=fin, dext=ext - sel, dfil=fin - ext, unans=unans)
    print(f"{'ste_k'+k:10s} {sel:>9.4f} {ext:>9.4f} {fin:>9.4f} {ext-sel:>+10.4f} {fin-ext:>+12.4f} {unans:>12}")

print("\n" + "=" * 90)
print("[측정 2-b] Sonnet m4_anchor(300q) filter 최종 R/P + gold 제거 — pruning 성향")
print("=" * 90)
son_out = [json.loads(l) for l in open(os.path.join(SON, "m4_anchor_sonnet/output_m4_anchor_sonnet.jsonl"))]
R = statistics.mean(r["recall"] for r in son_out)
P = statistics.mean(r["precision"] for r in son_out)
# table-level gold removal (final): fraction of gold tables missing
miss_t = [len(r.get("missing_tables", [])) for r in son_out]
gold_t = [len(r.get("gold_tables", [])) for r in son_out]
miss_c = [len(r.get("missing_cols", [])) for r in son_out]
gold_c = [len(r.get("gold_cols", [])) for r in son_out]
ex300 = statistics.mean(r["ex"] for r in son_out)
print(f"Sonnet m4_anchor 300q: R={R:.4f} P={P:.4f} EX={ex300:.4f}")
print(f"  gold table 제거율(missing/gold)={sum(miss_t)/max(sum(gold_t),1):.4f}  "
      f"gold col 제거율={sum(miss_c)/max(sum(gold_c),1):.4f}")

print("\n" + "=" * 90)
print("[측정 4] 300q EX noise band (Wilson 95% CI) + 1534q 대비")
print("=" * 90)
k300 = round(ex300 * len(son_out))
lo, hi = wilson(k300, len(son_out))
print(f"Sonnet m4_anchor 300q: EX={ex300:.4f} ({k300}/{len(son_out)}) Wilson95=[{lo:.4f},{hi:.4f}] half-width=±{(hi-lo)/2:.4f}")
k1534 = round(exm := statistics.mean(r["ex"] for r in son_m4.values()) * len(son_m4))
exm2 = statistics.mean(r["ex"] for r in son_m4.values())
lo2, hi2 = wilson(round(exm2 * len(son_m4)), len(son_m4))
print(f"Sonnet m4_anchor 1534q: EX={exm2:.4f} Wilson95=[{lo2:.4f},{hi2:.4f}] half-width=±{(hi2-lo2)/2:.4f}")

# save
out = dict(
    m4_direct=m4_direct,
    paired_ste=paired_summary, glm_stage=glm_stage,
    sonnet_m4_300q=dict(R=R, P=P, EX=ex300, ci=[lo, hi],
                        gold_t_removal=sum(miss_t)/max(sum(gold_t), 1),
                        gold_c_removal=sum(miss_c)/max(sum(gold_c), 1)),
    sonnet_m4_1534q=dict(EX=exm2, ci=[lo2, hi2]),
)
with open(os.path.join(OUT, "sonnet_vs_glm_m4_probe_decomp_2026-06-10.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {OUT}/sonnet_vs_glm_m4_probe_decomp_2026-06-10.json")
