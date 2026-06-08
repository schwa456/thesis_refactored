"""Wave 9 Baseline Relog 분석 — G-Retriever / LinkAlign / XiYan-SQL × new SQL Gen prompt.

3 baseline cells × 1534 queries × per-difficulty + final_nodes size 분석:
1. Per-cell overall + per-difficulty EX (metrics.txt 보정 + cross-check)
2. final_nodes size 분포 + size-band 별 EX
3. anchor c01_01 + M4 per-difficulty EX 비교 (paper §10 ΔEX 갱신)
4. prompt-axis confounder ΔΔ 분리 (anchor +0.1780 vs baseline 평균 ΔEX)
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("/home/hyeonjin/thesis_refactored")
WAVE9 = ROOT / "outputs/baselines/wave9_relog"
ANCHOR_OUT = ROOT / "outputs/experiments/abl/c01_threshold_sweep/c01_01_wave7_relog/output_c01_01_wave7_relog.jsonl"
M4_OUT = ROOT / "outputs/experiments/abl/wave6_recall_biased/w6_p2_m4_bidirectional/output_w6_p2_m4_bidirectional.jsonl"

# Prior outdated baseline EX (from EXPERIMENT_HISTORY Wave 9 entry)
OUTDATED = {
    "g_retriever": {"overall": 0.2490, "simple": 0.3211, "moderate": 0.1315, "challenging": 0.1655},
    "linkalign":   {"overall": 0.2001, "simple": 0.2789, "moderate": 0.0754, "challenging": 0.0966},
    "xiyansql":    {"overall": 0.1969, "simple": 0.2757, "moderate": 0.0668, "challenging": 0.1103},
}


def _read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def _parse_difficulty_from_dev(question_id: int, dev_json_cache: dict | None = None) -> str | None:
    """Look up difficulty from BIRD-Dev metadata if it's missing in the per-record JSONL."""
    if dev_json_cache is None:
        return None
    return dev_json_cache.get(question_id)


def load_difficulty_lookup() -> dict[int, str]:
    """BIRD-Dev metadata: question_id → difficulty (simple/moderate/challenging)."""
    dev_path = ROOT / "data/raw/BIRD_dev/dev.json"
    if not dev_path.exists():
        return {}
    with dev_path.open() as f:
        items = json.load(f)
    return {it["question_id"]: it.get("difficulty", "unknown") for it in items}


def analyze_wave9(diff_lookup: dict[int, str]) -> dict:
    cells = ["g_retriever", "linkalign", "xiyansql"]
    summary = {}
    for cell in cells:
        path = WAVE9 / f"{cell}_relog" / "predictions.jsonl"
        rows = _read_jsonl(path)
        if not rows:
            print(f"[SKIP] {cell} — empty")
            continue
        # per-difficulty + final_nodes size
        by_diff: dict[str, list] = {"simple": [], "moderate": [], "challenging": []}
        size_buckets: dict[str, list] = {"0-5": [], "6-15": [], "16-30": [], "31-60": [], "61+": []}
        sizes_per_diff: dict[str, list] = {"simple": [], "moderate": [], "challenging": []}
        ex_by_size_diff: dict[tuple, list] = {}
        ex_all = []
        size_all = []
        for r in rows:
            qid = r["question_id"]
            ex = int(r.get("ex_score", 0))
            diff = r.get("difficulty") or diff_lookup.get(qid)
            if not diff:
                continue
            nodes = r.get("final_nodes", [])
            n_cols = sum(1 for n in nodes if "." in n and "->" not in n)
            ex_all.append(ex)
            size_all.append(n_cols)
            by_diff.setdefault(diff, []).append(ex)
            sizes_per_diff.setdefault(diff, []).append(n_cols)
            if n_cols <= 5:
                size_buckets["0-5"].append(ex)
            elif n_cols <= 15:
                size_buckets["6-15"].append(ex)
            elif n_cols <= 30:
                size_buckets["16-30"].append(ex)
            elif n_cols <= 60:
                size_buckets["31-60"].append(ex)
            else:
                size_buckets["61+"].append(ex)
            ex_by_size_diff.setdefault((diff, "0-5" if n_cols <= 5 else
                                              "6-15" if n_cols <= 15 else
                                              "16-30" if n_cols <= 30 else
                                              "31-60" if n_cols <= 60 else "61+"), []).append(ex)

        def _mean(xs):
            return sum(xs) / len(xs) if xs else None

        summary[cell] = {
            "n": len(ex_all),
            "ex_overall": _mean(ex_all),
            "mean_cols_per_q": _mean(size_all),
            "median_cols_per_q": (statistics.median(size_all) if size_all else None),
            "ex_by_difficulty": {d: {"n": len(v), "ex": _mean(v)} for d, v in by_diff.items()},
            "mean_cols_by_difficulty": {d: _mean(v) for d, v in sizes_per_diff.items()},
            "ex_by_size_bucket": {b: {"n": len(v), "ex": _mean(v)} for b, v in size_buckets.items()},
        }
    return summary


def compare_baseline_anchor(diff_lookup: dict[int, str]) -> dict:
    """Per-difficulty EX for anchor c01_01_wave7_relog + M4 bidirectional from output_*.jsonl."""
    result = {}
    for tag, path in [("anchor_c01_01", ANCHOR_OUT), ("m4_bidirectional", M4_OUT)]:
        rows = _read_jsonl(path)
        if not rows:
            continue
        by_diff: dict[str, list] = {"simple": [], "moderate": [], "challenging": []}
        ex_all = []
        for r in rows:
            qid = r["question_id"]
            diff = diff_lookup.get(qid)
            ex = int(r.get("ex", 0))
            ex_all.append(ex)
            if diff:
                by_diff.setdefault(diff, []).append(ex)
        result[tag] = {
            "n": len(ex_all),
            "ex_overall": sum(ex_all) / len(ex_all),
            "ex_by_difficulty": {d: {"n": len(v), "ex": (sum(v) / len(v)) if v else None} for d, v in by_diff.items()},
        }
    return result


def confounder_decomposition(wave9: dict, ref: dict) -> dict:
    """ΔEX decomposition — Wave 9 new prompt 효과 vs anchor prompt-axis 효과 ΔΔ."""
    # anchor prior EX = 0.3396 (5/1 prior), anchor new EX = 0.5117 (Wave 7) → ΔEX_anchor = +0.1721 (Wave 7 base)
    # Using anchor's prompt-axis ΔEX = +0.1780 (from 0.3396 5/1 → 0.5176 c01_01 baseline; or Wave 7 0.5117 → 0.5176)
    anchor_prompt_dEX = 0.5117 - 0.3396  # Wave 7 vs 5/1 prior
    baseline_dEX = {
        cell: (wave9[cell]["ex_overall"] - OUTDATED[cell]["overall"])
        for cell in ("g_retriever", "linkalign", "xiyansql") if cell in wave9
    }
    baseline_dEX_mean = sum(baseline_dEX.values()) / len(baseline_dEX)
    return {
        "anchor_prompt_dEX": anchor_prompt_dEX,
        "baseline_dEX_per_cell": baseline_dEX,
        "baseline_dEX_mean": baseline_dEX_mean,
        "ddEX_anchor_minus_baseline_avg": anchor_prompt_dEX - baseline_dEX_mean,
    }


def main():
    diff = load_difficulty_lookup()
    wave9 = analyze_wave9(diff)
    refs = compare_baseline_anchor(diff)
    conf = confounder_decomposition(wave9, refs)

    print("\n=== Wave 9 Baseline Relog — per-cell summary ===")
    for cell, s in wave9.items():
        print(f"\n--- {cell} (n={s['n']}) ---")
        print(f"  Overall EX:               {s['ex_overall']:.4f}")
        print(f"  Mean cols/q:              {s['mean_cols_per_q']:.2f} (median={s['median_cols_per_q']:.0f})")
        for d, v in s["ex_by_difficulty"].items():
            mean_cols = s["mean_cols_by_difficulty"].get(d)
            mc = f"{mean_cols:.1f}" if mean_cols is not None else "n/a"
            print(f"  {d:12s} EX={v['ex']:.4f} (n={v['n']:4d}, mean_cols={mc})")
        print("  by size bucket:")
        for b, v in s["ex_by_size_bucket"].items():
            ex_s = f"{v['ex']:.4f}" if v["ex"] is not None else "n/a"
            print(f"     {b:8s} n={v['n']:4d} EX={ex_s}")

    print("\n=== Comparison anchor + M4 (from output_*.jsonl) ===")
    for tag, s in refs.items():
        print(f"\n--- {tag} (n={s['n']}) ---")
        print(f"  Overall EX:        {s['ex_overall']:.4f}")
        for d, v in s["ex_by_difficulty"].items():
            ex_s = f"{v['ex']:.4f}" if v["ex"] is not None else "n/a"
            print(f"  {d:12s} EX={ex_s} (n={v['n']:4d})")

    print("\n=== ΔEX vs anchor c01_01 (Wave 7 EX=0.5117) + M4 (EX=0.5300) ===")
    anchor_ex = refs.get("anchor_c01_01", {}).get("ex_overall", 0.5117)
    m4_ex = refs.get("m4_bidirectional", {}).get("ex_overall", 0.5300)
    print(f"\n  Anchor EX = {anchor_ex:.4f}, M4 EX = {m4_ex:.4f}")
    for cell, s in wave9.items():
        d_anchor = anchor_ex - s["ex_overall"]
        d_m4 = m4_ex - s["ex_overall"]
        print(f"  {cell:15s} EX={s['ex_overall']:.4f}  Δ vs anchor=+{d_anchor:.4f}  Δ vs M4=+{d_m4:.4f}")

    print("\n=== Per-difficulty ΔEX (anchor vs each baseline) ===")
    anchor_diff = refs.get("anchor_c01_01", {}).get("ex_by_difficulty", {})
    for d in ("simple", "moderate", "challenging"):
        a = anchor_diff.get(d, {}).get("ex")
        if a is None:
            continue
        print(f"\n  {d}: anchor EX={a:.4f}")
        for cell, s in wave9.items():
            b = s["ex_by_difficulty"].get(d, {}).get("ex")
            if b is None:
                continue
            print(f"     {cell:15s} EX={b:.4f}  ΔEX_anchor=+{a-b:.4f}")

    print("\n=== Prompt-axis Confounder ΔΔ Decomposition ===")
    print(f"  Anchor ΔEX (Wave 7 0.5117 − 5/1 prior 0.3396) = {conf['anchor_prompt_dEX']:+.4f}")
    print("  Wave 9 baseline ΔEX (new prompt − outdated):")
    for cell, dEX in conf["baseline_dEX_per_cell"].items():
        print(f"     {cell:15s} ΔEX = {dEX:+.4f}")
    print(f"  Baseline avg ΔEX = {conf['baseline_dEX_mean']:+.4f}")
    print(f"  ΔΔ (anchor − baseline avg) = {conf['ddEX_anchor_minus_baseline_avg']:+.4f}")
    print("  → ΔΔ 가 본 framework 의 schema linking effect 의 정량 evidence (prompt-axis 분리 후)")

    # Save JSON
    out_path = ROOT / "outputs/analysis/wave9_baseline_relog_2026-05-18.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"wave9": wave9, "anchor_m4_ref": refs, "confounder_decomposition": conf}
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n→ saved: {out_path}")


if __name__ == "__main__":
    main()
