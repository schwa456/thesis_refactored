"""V6-W2 Phase 2 oversmoothing trajectory parser + correlation analyzer.

Inputs: logs/v6_phase2/p2_{standalone, phase1, standalone_no_selfloop, sum}_s11.log
Outputs: epoch-level CSV + summary stats (printed to stdout).

근거: 사용자 trigger 2026-06-05 — Phase 2 4 cells × 300 epochs analysis.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
from typing import Dict, List, Tuple

EPOCH_RE = re.compile(
    r"Epoch (\d+) \| Loss: ([0-9.]+) \| Main: ([0-9.]+) \| AC: ([0-9.]+) "
    r"\| Val Recall@15: ([0-9.]+) \| oversmoothing/energy: ([0-9.eE+\-]+) "
    r"\| oversmoothing/mad: ([0-9.eE+\-]+)"
)

CELLS = [
    "p2_standalone",
    "p2_phase1",
    "p2_standalone_no_selfloop",
    "p2_sum",
]


def parse_log(log_path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if not m:
                continue
            rows.append({
                "epoch": int(m.group(1)),
                "loss": float(m.group(2)),
                "main": float(m.group(3)),
                "ac": float(m.group(4)),
                "r15": float(m.group(5)),
                "energy": float(m.group(6)),
                "mad": float(m.group(7)),
            })
    return rows


def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    den = math.sqrt(sxx * syy)
    if den == 0:
        return float("nan")
    return sxy / den


def spearman(xs: List[float], ys: List[float]) -> float:
    def rankify(vs: List[float]) -> List[float]:
        idx = sorted(range(len(vs)), key=lambda i: vs[i])
        ranks = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[idx[j + 1]] == vs[idx[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[idx[k]] = avg_rank
            i = j + 1
        return ranks
    return pearson(rankify(xs), rankify(ys))


def summarize(rows: List[Dict[str, float]], name: str) -> Dict[str, float]:
    r15s = [r["r15"] for r in rows]
    energies = [r["energy"] for r in rows]
    mads = [r["mad"] for r in rows]
    peak_r15 = max(r15s)
    peak_epoch = rows[r15s.index(peak_r15)]["epoch"]
    final_r15 = r15s[-1]
    early_r15 = r15s[9] if len(r15s) >= 10 else r15s[-1]  # epoch 10
    return {
        "cell": name,
        "n_epochs": len(rows),
        "peak_r15": peak_r15,
        "peak_epoch": peak_epoch,
        "final_r15": final_r15,
        "final_energy": energies[-1],
        "final_mad": mads[-1],
        "energy_mean": sum(energies) / len(energies),
        "energy_std": statistics.pstdev(energies),
        "mad_mean": sum(mads) / len(mads),
        "mad_std": statistics.pstdev(mads),
        "r15_std": statistics.pstdev(r15s),
        "r15_at_epoch10": early_r15,
        "pearson_mad_r15": pearson(mads, r15s),
        "spearman_mad_r15": spearman(mads, r15s),
        "pearson_energy_r15": pearson(energies, r15s),
        "spearman_energy_r15": spearman(energies, r15s),
        "pearson_logEnergy_r15": pearson([math.log10(max(e, 1e-12)) for e in energies], r15s),
    }


def trajectory_milestones(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Snapshot at epochs 1, 10, 50, 100, 150, 200, 250, 300."""
    targets = [1, 10, 50, 100, 150, 200, 250, 300]
    by_epoch = {r["epoch"]: r for r in rows}
    out = []
    for t in targets:
        if t in by_epoch:
            out.append(by_epoch[t])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs/v6_phase2")
    parser.add_argument("--out-csv", default="notebooks/analysis_results/v6_phase2_oversmoothing_trajectory.csv")
    parser.add_argument("--summary-csv", default="notebooks/analysis_results/v6_phase2_oversmoothing_summary.csv")
    args = parser.parse_args()

    all_rows: List[Tuple[str, Dict[str, float]]] = []
    all_summaries: List[Dict[str, float]] = []
    milestones_per_cell: Dict[str, List[Dict[str, float]]] = {}

    for cell in CELLS:
        log_path = os.path.join(args.log_dir, f"{cell}_s11.log")
        rows = parse_log(log_path)
        print(f"[{cell}] parsed {len(rows)} epochs from {log_path}")
        for r in rows:
            all_rows.append((cell, r))
        all_summaries.append(summarize(rows, cell))
        milestones_per_cell[cell] = trajectory_milestones(rows)

    # Trajectory CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "epoch", "loss", "main", "ac", "r15", "energy", "mad"])
        for cell, r in all_rows:
            w.writerow([cell, r["epoch"], r["loss"], r["main"], r["ac"],
                        r["r15"], r["energy"], r["mad"]])
    print(f"Wrote trajectory CSV → {args.out_csv}")

    # Summary CSV
    fields = list(all_summaries[0].keys())
    with open(args.summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in all_summaries:
            w.writerow(s)
    print(f"Wrote summary CSV → {args.summary_csv}")

    # Print summary to stdout
    print("\n=== Summary per cell ===")
    for s in all_summaries:
        print(f"\n[{s['cell']}]")
        for k, v in s.items():
            if k == "cell":
                continue
            if isinstance(v, float):
                if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0):
                    print(f"  {k:25s} = {v:.4e}")
                else:
                    print(f"  {k:25s} = {v:.4f}")
            else:
                print(f"  {k:25s} = {v}")

    # Milestone trajectory print
    print("\n=== Milestones (epoch 1,10,50,100,150,200,250,300) ===")
    for cell, milestones in milestones_per_cell.items():
        print(f"\n[{cell}]")
        print(f"  {'epoch':>6} {'r15':>8} {'energy':>14} {'mad':>8} {'loss':>8}")
        for r in milestones:
            print(f"  {r['epoch']:>6} {r['r15']:>8.4f} {r['energy']:>14.4e} {r['mad']:>8.4f} {r['loss']:>8.4f}")


if __name__ == "__main__":
    main()
