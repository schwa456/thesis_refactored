#!/usr/bin/env python3
"""Module-ablation B: 각 변형 vs anchor(m4_canonical_sonnet) per-query EX 유의성 검정.

목적: 'plateau 내 변형들이 anchor 와 EX 유의차 없음(특히 α=0.85 ΔEX +0.0117)' 정량 입증.
- 검정 1: McNemar paired test (per-query EX 0/1, exact binomial — discordant 쌍 b/c)
- 검정 2: paired bootstrap 95% CI(ΔEX) (B=10000, percentile)

Bonferroni family = 11셀 = module_ablation_b 8 + m4_ablation 정준 leave-one-out 3
  (−Builder Plain Hetero / −Selector α=1.0 cosine / −Filter None).
sel_alpha00(α=0, GAT-only degenerate, ΔEX −0.2132)은 family 제외 — 검정력 대조군.

데이터: outputs/experiments/sonnet_rebaseline_2026_06_10/
  - m4_canonical_sonnet/predictions.jsonl                 (anchor, EX 0.6030)
  - module_ablation_b/{stage}/{cell}/predictions.jsonl    (9 cells)
  - m4_ablation/m4_abl_{builder,selector,filter}_sonnet/predictions.jsonl  (3 cells)
predictions.jsonl 레코드: {question_id, db_id, pred_tables, pred_cols, predicted_sql, ex(0/1)}
모두 n=1534 per-query ex.

산출:
  outputs/analysis/module_ablation_b_mcnemar_2026-06-20.json
  notebooks/analysis_results/05_ablation_waves/module_ablation_b_significance.md (별도 작성)
"""
from __future__ import annotations

import json
import os
import zlib

import numpy as np
from scipy import stats

ROOT = "/home/hyeonjin/thesis_refactored"
BASE = f"{ROOT}/outputs/experiments/sonnet_rebaseline_2026_06_10"
ANCHOR_F = f"{BASE}/m4_canonical_sonnet/predictions.jsonl"
OUT_JSON = f"{ROOT}/outputs/analysis/module_ablation_b_mcnemar_2026-06-20.json"

# (stage, subdir[BASE 기준], label, family?)  — 파이프라인 순서
#   family=True : Bonferroni multiple-comparison family (= plateau/null 후보, 11셀)
#   family=False: sel_alpha00 (α=0 GAT-only degenerate) = 검정력 대조군 (family 제외)
# module_ablation_b/* = 세분화 design 변형, m4_ablation/* = 정준 leave-one-out(−Module)
CELLS = [
    ("Builder",   "module_ablation_b/builder/bld_no_t2t_sonnet",      "no_t2t (table↔table edge 제거)",   True),
    ("Builder",   "module_ablation_b/builder/bld_rfm_tokens_sonnet",  "rfm_tokens (RFM 토큰 노드텍스트)",  True),
    ("Builder",   "m4_ablation/m4_abl_builder_sonnet",                "−Builder (Enriched→Plain Hetero)", True),
    ("Selector",  "module_ablation_b/selector/sel_alpha085_sonnet",   "α=0.85 (cosine-heavy ensemble)",   True),
    ("Selector",  "m4_ablation/m4_abl_selector_sonnet",               "−Selector (α=1.0 cosine-only)",    True),
    ("Selector",  "module_ablation_b/selector/sel_alpha00_sonnet",    "α=0.0 (GAT-only, degenerate)",     False),  # 대조군
    ("Extractor", "module_ablation_b/extractor/ext_mst_only_sonnet",  "MST-only (PCST 확장 제거)",          True),
    ("Extractor", "module_ablation_b/extractor/ext_pcst_only_sonnet", "PCST-only (MST 연결보장 제거)",      True),
    ("Filter",    "module_ablation_b/filter/flt_cot_sonnet",          "CoT filter",                        True),
    ("Filter",    "module_ablation_b/filter/flt_two_stage_sonnet",    "two-stage filter",                  True),
    ("Filter",    "module_ablation_b/filter/flt_voting_sonnet",       "voting filter",                     True),
    ("Filter",    "m4_ablation/m4_abl_filter_sonnet",                 "−Filter (BiFilter→None)",          True),
]

B_BOOT = 10000
SEED = 20260620


def load_ex(path: str) -> dict:
    d = {}
    for ln in open(path):
        r = json.loads(ln)
        d[r["question_id"]] = int(r["ex"])
    return d


def mcnemar_exact(b: int, c: int) -> dict:
    """exact binomial McNemar (two-sided) + continuity-corrected chi2 (참고)."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0,
                "p_exact": 1.0, "p_chi2_cc": 1.0, "chi2_cc": 0.0}
    k = min(b, c)
    p_exact = float(stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    chi2_cc = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    p_chi2 = float(stats.chi2.sf(chi2_cc, df=1))
    return {"b": int(b), "c": int(c), "n_discordant": int(n),
            "p_exact": p_exact, "p_chi2_cc": p_chi2, "chi2_cc": float(chi2_cc)}


def boot_ci(var: np.ndarray, anc: np.ndarray, cell_key: str) -> dict:
    """paired bootstrap 95% CI of ΔEX = mean(var - anc).

    셀별 독립 시드(SEED ⊕ crc32(cell_key)) → CELLS 순서/개수와 무관하게 CI 재현.
    """
    rng = np.random.default_rng([SEED, zlib.crc32(cell_key.encode()) & 0xFFFFFFFF])
    diff = var.astype(float) - anc.astype(float)
    n = diff.size
    idx = rng.integers(0, n, size=(B_BOOT, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta_ex": float(diff.mean()),
            "ci95_lo": float(lo), "ci95_hi": float(hi),
            "boot_se": float(boot.std(ddof=1)),
            "ci_excludes_0": bool(lo > 0 or hi < 0)}


def main() -> None:
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    anchor = load_ex(ANCHOR_F)
    anchor_ex = sum(anchor.values()) / len(anchor)

    bonf_n = sum(1 for *_, fam in CELLS if fam)        # Bonferroni family 크기 (=11)
    bonf_thr = 0.05 / bonf_n

    rows = []
    for stage, subdir, label, family in CELLS:
        d = load_ex(f"{BASE}/{subdir}/predictions.jsonl")
        qids = sorted(set(anchor) & set(d))
        anc = np.array([anchor[q] for q in qids])
        var = np.array([d[q] for q in qids])
        var_ex = var.mean()
        # discordant: b = anchor 1 & var 0 ; c = anchor 0 & var 1
        b = int(((anc == 1) & (var == 0)).sum())
        c = int(((anc == 0) & (var == 1)).sum())
        mc = mcnemar_exact(b, c)
        bc = boot_ci(var, anc, subdir)
        rows.append({
            "stage": stage, "cell": subdir.split("/")[-1], "label": label,
            "source": subdir.split("/")[0], "family": family, "n": len(qids),
            "ex_variant": float(var_ex), "ex_anchor": float(anchor_ex),
            "delta_ex": bc["delta_ex"],
            "ci95_lo": bc["ci95_lo"], "ci95_hi": bc["ci95_hi"],
            "boot_se": bc["boot_se"], "ci_excludes_0": bc["ci_excludes_0"],
            "mcnemar_b": mc["b"], "mcnemar_c": mc["c"],
            "n_discordant": mc["n_discordant"],
            "p_mcnemar_exact": mc["p_exact"],
            "p_mcnemar_chi2_cc": mc["p_chi2_cc"],
            "significant_005": bool(mc["p_exact"] < 0.05),
            "significant_bonferroni": bool(family and mc["p_exact"] < bonf_thr),
        })

    out = {
        "anchor": "m4_canonical_sonnet", "anchor_ex": float(anchor_ex),
        "n_queries": len(anchor), "n_bootstrap": B_BOOT, "seed": SEED,
        "test": "McNemar exact-binomial (two-sided) + paired bootstrap 95% CI(ΔEX)",
        "bonferroni_n": bonf_n, "bonferroni_thr": float(bonf_thr),
        "family_definition": (
            "Bonferroni family = 11셀 (module_ablation_b 8 + m4_ablation leave-one-out 3); "
            "sel_alpha00(α=0 GAT-only)은 검정력 대조군으로 family 제외"
        ),
        "cells": rows,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ── 콘솔 표 ──
    print(f"anchor m4_canonical_sonnet EX={anchor_ex:.4f}  (N={len(anchor)}, "
          f"bootstrap B={B_BOOT}, seed={SEED}, Bonferroni N={bonf_n} thr={bonf_thr:.5f})\n")
    hdr = (f"{'stage':9s} {'cell':26s} {'EX':>7s} {'ΔEX':>8s} "
           f"{'95% CI(ΔEX)':>20s} {'b':>4s} {'c':>4s} {'p(McNemar)':>11s} {'.05':>4s} {'Bonf':>5s}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        tag = "" if r["family"] else " [대조]"
        pstr = (f"{r['p_mcnemar_exact']:.4f}" if r["p_mcnemar_exact"] >= 1e-4
                else f"{r['p_mcnemar_exact']:.1e}")
        ci = f"[{r['ci95_lo']:+.4f},{r['ci95_hi']:+.4f}]"
        bonf = ("SIG" if r["significant_bonferroni"] else
                ("-" if r["family"] else "n/a"))
        print(f"{r['stage']:9s} {r['cell'].replace('_sonnet',''):26s} "
              f"{r['ex_variant']:.4f} {r['delta_ex']:+.4f} {ci:>20s} "
              f"{r['mcnemar_b']:4d} {r['mcnemar_c']:4d} {pstr:>11s} "
              f"{'YES' if r['significant_005'] else 'no':>4s} {bonf:>5s}{tag}")

    fam = [r for r in rows if r["family"]]
    nsig = sum(r["significant_005"] for r in fam)
    nbonf = sum(r["significant_bonferroni"] for r in fam)
    print(f"\nfamily {bonf_n}셀: 유의(p<0.05) {nsig}/{bonf_n}, "
          f"Bonferroni 유의(p<{bonf_thr:.5f}) {nbonf}/{bonf_n}, "
          f"bootstrap CI 0 포함 {sum(not r['ci_excludes_0'] for r in fam)}/{bonf_n}")
    print(f"→ JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
