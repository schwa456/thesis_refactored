#!/usr/bin/env python3
"""V6-W2 Phase 2 oversmoothing 분석 — edge_type split message-passing 분리 학습.
4 cells (standalone / phase1 / standalone_no_selfloop / sum) × 300 epoch × seed s11.

로그 파싱: 'Epoch N | Loss: L | Main: M | AC: A | Val Recall@15: Y | oversmoothing/energy: E | oversmoothing/mad: D'
산출: epoch-level CSV + 상관 정량 (mad-R@15, energy-R@15 Pearson) + trajectory 통계.
"""
import re, os, csv, json, math

ROOT = "/home/hyeonjin/thesis_refactored"
LOGDIR = os.path.join(ROOT, "logs/v6_phase2")
OUTDIR = os.path.join(ROOT, "outputs/analysis")
os.makedirs(OUTDIR, exist_ok=True)

CELLS = ["standalone", "phase1", "standalone_no_selfloop", "sum"]
# design diff (config 정합)
DESIGN = {
    "standalone":             dict(pairnorm="none", ir_alpha=0.0, jk="none", self_loops=True,  aggr="mean"),
    "phase1":                 dict(pairnorm="pairnorm", ir_alpha=0.1, jk="concat", self_loops=True, aggr="mean"),
    "standalone_no_selfloop": dict(pairnorm="none", ir_alpha=0.0, jk="none", self_loops=False, aggr="mean"),
    "sum":                    dict(pairnorm="none", ir_alpha=0.0, jk="none", self_loops=True,  aggr="sum"),
}

LINE = re.compile(
    r"Epoch (\d+) \| Loss: ([\d.eE+-]+) \| Main: ([\d.eE+-]+) \| AC: ([\d.eE+-]+) \| "
    r"Val Recall@15: ([\d.eE+-]+) \| oversmoothing/energy: ([\d.eE+-]+) \| oversmoothing/mad: ([\d.eE+-]+)"
)


def parse_cell(cell):
    rows = []
    with open(os.path.join(LOGDIR, f"p2_{cell}_s11.log"), errors="ignore") as f:
        for line in f:
            m = LINE.search(line)
            if m:
                rows.append(dict(
                    epoch=int(m.group(1)), loss=float(m.group(2)), main=float(m.group(3)),
                    ac=float(m.group(4)), r15=float(m.group(5)),
                    energy=float(m.group(6)), mad=float(m.group(7)),
                ))
    rows.sort(key=lambda r: r["epoch"])
    return rows


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def spearman(xs, ys):
    def rank(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def stats(rows, key, last_n=50):
    vs = [r[key] for r in rows]
    last = vs[-last_n:]
    mean_last = sum(last) / len(last)
    std_last = math.sqrt(sum((v - mean_last) ** 2 for v in last) / len(last))
    # peak (max for r15, min for energy/mad context-dependent — report both)
    peak_v = max(vs); peak_ep = rows[vs.index(peak_v)]["epoch"]
    return dict(final=vs[-1], peak=peak_v, peak_ep=peak_ep,
                mean_last=mean_last, std_last=std_last, min=min(vs), max=max(vs))


def convergence_epoch(rows, key="r15", tol_frac=0.01, window=20):
    """첫 epoch 이후 값이 final mean 의 tol_frac 이내로 안정되는 epoch."""
    vs = [r[key] for r in rows]
    final_mean = sum(vs[-window:]) / window
    band = abs(final_mean) * tol_frac
    for i in range(len(vs) - window):
        seg = vs[i:i + window]
        if all(abs(v - final_mean) <= band for v in seg):
            return rows[i]["epoch"]
    return rows[-1]["epoch"]


def main():
    data = {c: parse_cell(c) for c in CELLS}
    # write per-epoch CSV (long format)
    csvp = os.path.join(OUTDIR, "v6_phase2_epoch_trajectory_2026-06-05.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "epoch", "loss", "r15", "energy", "mad"])
        for c in CELLS:
            for r in data[c]:
                w.writerow([c, r["epoch"], round(r["loss"], 4), round(r["r15"], 4),
                            f"{r['energy']:.6e}", round(r["mad"], 4)])
    print(f"Wrote {csvp}\n")

    summary = {}
    for c in CELLS:
        rows = data[c]
        r15 = [r["r15"] for r in rows]; mad = [r["mad"] for r in rows]; en = [r["energy"] for r in rows]
        # log energy for correlation (6 orders span)
        logen = [math.log10(max(e, 1e-12)) for e in en]
        s = dict(
            n=len(rows),
            r15=stats(rows, "r15"), mad=stats(rows, "mad"), energy=stats(rows, "energy"),
            conv_r15=convergence_epoch(rows, "r15"),
            pearson_mad_r15=pearson(mad, r15),
            spearman_mad_r15=spearman(mad, r15),
            pearson_logenergy_r15=pearson(logen, r15),
            spearman_energy_r15=spearman(en, r15),
            design=DESIGN[c],
        )
        summary[c] = s
        print(f"=== p2_{c} === {DESIGN[c]}")
        print(f"  R@15 final={s['r15']['final']:.4f} peak={s['r15']['peak']:.4f}@ep{s['r15']['peak_ep']} "
              f"mean_last50={s['r15']['mean_last']:.4f} std_last50={s['r15']['std_last']:.4f} conv_ep={s['conv_r15']}")
        print(f"  mad  final={s['mad']['final']:.4f} min={s['mad']['min']:.4f} max={s['mad']['max']:.4f} "
              f"mean_last50={s['mad']['mean_last']:.4f}")
        print(f"  energy final={s['energy']['final']:.4e} min={s['energy']['min']:.4e} max={s['energy']['max']:.4e}")
        print(f"  Pearson(mad,R@15)={s['pearson_mad_r15']:+.4f}  Spearman={s['spearman_mad_r15']:+.4f}")
        print(f"  Pearson(log10 energy,R@15)={s['pearson_logenergy_r15']:+.4f}  Spearman(energy,R@15)={s['spearman_energy_r15']:+.4f}")
        print()

    # pooled across all 4 cells × 300 epoch
    allr15 = [r["r15"] for c in CELLS for r in data[c]]
    allmad = [r["mad"] for c in CELLS for r in data[c]]
    alllogen = [math.log10(max(r["energy"], 1e-12)) for c in CELLS for r in data[c]]
    print("=== POOLED (4 cells × 300 epoch = %d points) ===" % len(allr15))
    print(f"  Pearson(mad,R@15)={pearson(allmad, allr15):+.4f}  Spearman={spearman(allmad, allr15):+.4f}")
    print(f"  Pearson(log10 energy,R@15)={pearson(alllogen, allr15):+.4f}  Spearman(energy,R@15)={spearman([r['energy'] for c in CELLS for r in data[c]], allr15):+.4f}")

    # cross-cell final comparison (point estimate)
    print("\n=== CROSS-CELL final epoch 300 (point) ===")
    for c in CELLS:
        rows = data[c]
        fr = rows[-1]
        print(f"  p2_{c:24s} R@15={fr['r15']:.4f} energy={fr['energy']:.4e} mad={fr['mad']:.4f}")

    summary["_pooled"] = dict(
        pearson_mad_r15=pearson(allmad, allr15), spearman_mad_r15=spearman(allmad, allr15),
        pearson_logenergy_r15=pearson(alllogen, allr15),
        spearman_energy_r15=spearman([r["energy"] for c in CELLS for r in data[c]], allr15),
        n=len(allr15),
    )
    jp = os.path.join(OUTDIR, "v6_phase2_oversmoothing_2026-06-05.json")
    with open(jp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {jp}")


if __name__ == "__main__":
    main()
