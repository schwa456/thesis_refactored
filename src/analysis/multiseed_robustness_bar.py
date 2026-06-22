#!/usr/bin/env python3
"""Multiseed robustness bar — 4 cells × {s12,s13} (monitor=recall_at_15, builder=enriched).

R@15: 로그 'Best recall_at_15=' (s12/s13 primary, s11 caveat 별도)
ROC-AUC / gold_p50 / gold-nongold separation: score_analysis (column 노드, EnrichedHeteroGraphBuilder)
cell별 mean±std (n=2) → paper robustness bar.
"""
import json, os, re, statistics
from collections import defaultdict

ROOT = "/home/hyeonjin/thesis_refactored"
LOG = os.path.join(ROOT, "logs/multiseed_robustness_gpu2")
SO = os.path.join(ROOT, "outputs/experiments/multiseed_selector_only_enriched")
OUT = os.path.join(ROOT, "outputs/analysis")
CELLS = ["ma2_a_p50_r15", "qcond_r15", "w2_sum", "v6w6_a_r15"]
LABEL = {"ma2_a_p50_r15": "ma2_a_p50 (gold-margin)", "qcond_r15": "qcond = M4 anchor",
         "w2_sum": "w2_sum (high-ROC)", "v6w6_a_r15": "v6w6_a (DSN+SL)"}
SEEDS = [12, 13]
BEST_RE = re.compile(r"Best recall_at_15=([0-9.]+)")


def r15_from_log(cell, seed):
    f = os.path.join(LOG, f"{cell}_s{seed}.log")
    if not os.path.exists(f):
        return None
    best = None
    for line in open(f, errors="ignore"):
        m = BEST_RE.search(line)
        if m:
            best = float(m.group(1))
    return best


def iscol(n):
    return "." in n.split("->")[0]


def pct(vs, p):
    vs = sorted(vs)
    return vs[min(len(vs)-1, int(round((len(vs)-1)*p/100)))] if vs else float("nan")


def roc_q(items):
    pos = [s for s, g in items if g]; neg = [s for s, g in items if not g]
    if not pos or not neg:
        return None
    allv = sorted(s for s, _ in items); n = len(allv); rk = {}; i = 0
    while i < n:
        j = i
        while j+1 < n and allv[j+1] == allv[i]:
            j += 1
        rk[allv[i]] = (i+j)/2+1; i = j+1
    return (sum(rk[s] for s in pos)-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))


def sep_from_score(cell, seed):
    d = os.path.join(SO, f"{cell}_s{seed}")
    f = None
    if os.path.isdir(d):
        for fn in os.listdir(d):
            if fn.startswith("score_analysis_") and fn.endswith(".jsonl"):
                f = os.path.join(d, fn); break
    if not f:
        return None
    pq = defaultdict(list)
    with open(f) as fh:
        for line in fh:
            r = json.loads(line)
            if iscol(r["node_name"]):
                pq[r["query_id"]].append((r["score"], bool(r.get("is_gold", False))))
    gold = [s for it in pq.values() for s, g in it if g]
    nong = [s for it in pq.values() for s, g in it if not g]
    rocs = [roc_q(it) for it in pq.values()]; rocs = [x for x in rocs if x is not None]
    return dict(gold_p50=pct(gold, 50), nong_p50=pct(nong, 50), gap=pct(gold, 50)-pct(nong, 50),
                roc_auc=statistics.mean(rocs))


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s, vals


def main():
    rows = {}
    print("=== per (cell, seed) raw ===")
    print(f"{'cell':22s} {'seed':>4} {'R@15':>7} {'gold_p50':>8} {'nong_p50':>8} {'gap':>7} {'ROC':>7}")
    for c in CELLS:
        per = {}
        for s in SEEDS:
            r15 = r15_from_log(c, s)
            sep = sep_from_score(c, s)
            per[s] = dict(r15=r15, **(sep or {}))
            print(f"{c:22s} {s:>4} {r15 if r15 else 0:>7.4f} "
                  f"{(sep or {}).get('gold_p50',0):>8.4f} {(sep or {}).get('nong_p50',0):>8.4f} "
                  f"{(sep or {}).get('gap',0):>7.4f} {(sep or {}).get('roc_auc',0):>7.4f}")
        rows[c] = per

    print("\n=== ROBUSTNESS BAR (mean±std, n=2 seeds s12/s13) ===")
    print(f"{'cell':22s} {'R@15':>16} {'ROC-AUC':>16} {'gold_p50':>16} {'gap':>16}")
    summary = {}
    for c in CELLS:
        r15 = ms([rows[c][s]["r15"] for s in SEEDS])
        roc = ms([rows[c][s].get("roc_auc") for s in SEEDS])
        gp = ms([rows[c][s].get("gold_p50") for s in SEEDS])
        gap = ms([rows[c][s].get("gap") for s in SEEDS])
        summary[c] = dict(r15=r15, roc=roc, gold_p50=gp, gap=gap)
        print(f"{c:22s} {f'{r15[0]:.4f}±{r15[1]:.4f}':>16} {f'{roc[0]:.4f}±{roc[1]:.4f}':>16} "
              f"{f'{gp[0]:.4f}±{gp[1]:.4f}':>16} {f'{gap[0]:.4f}±{gap[1]:.4f}':>16}")

    with open(os.path.join(OUT, "multiseed_robustness_bar_2026-06-10.json"), "w") as f:
        json.dump({c: {k: (v[:2] if v else None) for k, v in summary[c].items()} for c in CELLS}, f, indent=2)
    print(f"\nWrote {OUT}/multiseed_robustness_bar_2026-06-10.json")
    return rows, summary


if __name__ == "__main__":
    main()
