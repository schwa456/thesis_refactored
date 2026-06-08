#!/usr/bin/env python3
"""V6-W3 Phase 3 — L2(c) e2e EX + threshold-pass mechanism cross-check.

e2e predictions (ex_score 0/1) → per-cell EX (overall/per-DB/per-difficulty/european_football_2).
selector-only score_analysis → pass@0.1 count + gold/nongold score dist.
Spearman(EX, pass@0.1) / (EX, ext_nodes) — V6-W2 threshold-pass mechanism (ρ=+1.0) 정합 검증.
"""
import json, os, statistics
from collections import defaultdict

ROOT = "/home/hyeonjin/thesis_refactored"
E2E = os.path.join(ROOT, "outputs/experiments/s06_gat_bottleneck_fix/w3_hub_reduction_e2e")
SO = os.path.join(ROOT, "outputs/experiments/s06_gat_bottleneck_fix/w3_hub_reduction")
DEV = os.path.join(ROOT, "data/raw/BIRD_dev/dev.json")
OUTDIR = os.path.join(ROOT, "outputs/analysis")

CELLS = ["a", "b", "c"]
LABEL = {"a": "VirtualSummary", "b": "ColumnPooling", "c": "HubLocalVN"}
EXT_NODES = {"a": 36.2164, "b": 33.2373, "c": 42.8038}  # metrics.txt extractor_selected_nodes_mean
BASE = {"M4_anchor": 0.5300, "V6W2_no_selfloop_thr010": 0.3331, "V6W2_no_selfloop_thr005": 0.3422}

dev = json.load(open(DEV))
diff = {i: q.get("difficulty", "?") for i, q in enumerate(dev)}


def load_e2e(c):
    """qid -> (ex_score, db_id)"""
    out = {}
    with open(os.path.join(E2E, f"v6w3_{c}_e2e_s11", "predictions.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            out[r["question_id"]] = (int(r.get("ex_score") or 0), r["db_id"])
    return out


def load_scores(c):
    pq = defaultdict(list)
    with open(os.path.join(SO, f"v6w3_{c}_s11", f"score_analysis_v6w3_{c}_s11.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            pq[r["query_id"]].append((r["score"], bool(r.get("is_gold", False))))
    return pq


def pct(vs, p):
    vs = sorted(vs)
    return vs[min(len(vs)-1, int(round(p/100*(len(vs)-1))))] if vs else float("nan")


def spearman(xs, ys):
    def rank(vs):
        o = sorted(range(len(vs)), key=lambda i: vs[i]); r = [0]*len(vs)
        for rk, i in enumerate(o): r[i] = rk+1
        return r
    rx, ry = rank(xs), rank(ys); n = len(xs)
    return 1 - 6*sum((a-b)**2 for a, b in zip(rx, ry))/(n*(n*n-1))


results = {}
print("=== L2(c) e2e EX (overall / per-difficulty / european_football_2) ===")
ex_overall = {}
for c in CELLS:
    e = load_e2e(c)
    ex = [v[0] for v in e.values()]
    ex_overall[c] = statistics.mean(ex)
    by_diff = defaultdict(list); by_db = defaultdict(list)
    for qid, (s, db) in e.items():
        by_diff[diff.get(qid, "?")].append(s); by_db[db].append(s)
    ef = by_db.get("european_football_2", [])
    results[c] = dict(
        ex_overall=ex_overall[c],
        ex_simple=statistics.mean(by_diff["simple"]) if by_diff["simple"] else None,
        ex_moderate=statistics.mean(by_diff["moderate"]) if by_diff["moderate"] else None,
        ex_challenging=statistics.mean(by_diff["challenging"]) if by_diff["challenging"] else None,
        ex_european_football_2=statistics.mean(ef) if ef else None,
        n_ef=len(ef),
        per_db={db: round(statistics.mean(v), 4) for db, v in sorted(by_db.items())},
    )
    r = results[c]
    print(f"  v6w3_{c} ({LABEL[c]:14s}) EX={r['ex_overall']:.4f} | simple={r['ex_simple']:.4f} "
          f"moderate={r['ex_moderate']:.4f} challenging={r['ex_challenging']:.4f} | EF2={r['ex_european_football_2']:.4f} (n={r['n_ef']})")

print(f"\n  baseline: M4 anchor={BASE['M4_anchor']:.4f} | V6W2 no_selfloop thr010={BASE['V6W2_no_selfloop_thr010']:.4f} thr005={BASE['V6W2_no_selfloop_thr005']:.4f}")
for c in CELLS:
    e = results[c]["ex_overall"]
    print(f"  v6w3_{c}: ΔM4={e-BASE['M4_anchor']:+.4f}  ΔV6W2best={e-BASE['V6W2_no_selfloop_thr010']:+.4f}")

print("\n=== threshold-pass mechanism cross-check ===")
print(f"  {'cell':14s} {'EX':>7} {'ext_nodes':>9} {'pass@0.1':>9} {'gold_pass':>9} {'nong_pass':>9} {'gold_p50':>8} {'nong_mu':>8}")
passdata = {}
for c in CELLS:
    pq = load_scores(c)
    pass_tot = []; gp = []; npss = []
    gold_all = []; nong_all = []
    for items in pq.values():
        gpass = sum(1 for sc, g in items if g and sc >= 0.1)
        npass = sum(1 for sc, g in items if (not g) and sc >= 0.1)
        pass_tot.append(gpass + npass); gp.append(gpass); npss.append(npass)
        gold_all += [sc for sc, g in items if g]; nong_all += [sc for sc, g in items if not g]
    passdata[c] = dict(pass_mean=statistics.mean(pass_tot), gold_pass=statistics.mean(gp),
                       nong_pass=statistics.mean(npss), gold_p50=pct(gold_all, 50),
                       nong_mu=statistics.mean(nong_all), gold_mu=statistics.mean(gold_all))
    p = passdata[c]
    print(f"  v6w3_{c} ({LABEL[c]:10s}) {results[c]['ex_overall']:>7.4f} {EXT_NODES[c]:>9.2f} "
          f"{p['pass_mean']:>9.2f} {p['gold_pass']:>9.2f} {p['nong_pass']:>9.2f} {p['gold_p50']:>8.4f} {p['nong_mu']:>8.4f}")

exs = [results[c]["ex_overall"] for c in CELLS]
print(f"\n  Spearman(EX, ext_nodes)  = {spearman(exs, [EXT_NODES[c] for c in CELLS]):+.4f}")
print(f"  Spearman(EX, pass@0.1)   = {spearman(exs, [passdata[c]['pass_mean'] for c in CELLS]):+.4f}")
print(f"  Spearman(EX, nong_pass)  = {spearman(exs, [passdata[c]['nong_pass'] for c in CELLS]):+.4f}")
print("\n  V6-W2 no_selfloop ref: gold p50=0.1469, nongold μ=0.2314 (threshold-friendly elevated nongold)")

out = dict(baseline=BASE, ext_nodes=EXT_NODES,
           cells={c: {**results[c], **passdata[c]} for c in CELLS},
           spearman_ex_extnodes=spearman(exs, [EXT_NODES[c] for c in CELLS]),
           spearman_ex_passk=spearman(exs, [passdata[c]["pass_mean"] for c in CELLS]))
with open(os.path.join(OUTDIR, "v6_phase3_l2c_e2e_2026-06-06.json"), "w") as f:
    json.dump(out, f, indent=2, default=lambda o: round(o, 4) if isinstance(o, float) else o)
print(f"\nWrote {OUTDIR}/v6_phase3_l2c_e2e_2026-06-06.json")
