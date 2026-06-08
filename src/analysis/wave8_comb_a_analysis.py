"""Wave 8 Comb-A (D4 v1 + D3 v2 직렬 stacking) mechanism 분해 분석.

Comb-A 의 F1-axis +0.0314 vs M4 (dramatic) + EX-axis −0.0183 paradox 의 mechanism 분해:
1. Per-stage telemetry (Stage 0 = D4 v1 / Stage 1 = D3 v2)
2. F1-axis component breakdown (D4 P-lift + D3 retain + stacking synergy)
3. EX-axis paradox (D4 schema modification 이 D3 verify base 변경 mechanism)
4. Per-difficulty cross-tab (Comb-A vs M4 vs M1-B vs D4 v1 alone vs D3 v2 alone)
5. F1-EX decoupling 정량 evidence (paper §V.5.x.M.12)
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path("/home/hyeonjin/thesis_refactored")
COMB_A_DIR = ROOT / "outputs/experiments/abl/wave8_m4_extensions/comb_a/abl_wave8_comb_a_value_hint_verify2round"

REFS = {
    "M4 anchor": ROOT / "outputs/experiments/abl/wave6_recall_biased/w6_p2_m4_bidirectional",
    "anchor c01_01 (Wave 5)": ROOT / "outputs/experiments/abl/c01_threshold_sweep/c01_01_theta_0.1",
    "anchor c01_01 (Wave 7 relog)": ROOT / "outputs/experiments/abl/c01_threshold_sweep/c01_01_wave7_relog",
    "M1-B strong (Wave 6)": ROOT / "outputs/experiments/abl/wave6_recall_biased/wave6_p1_recall_biased_strong",
    "D4 v1 alone (Wave 8)": ROOT / "outputs/experiments/abl/wave8_m4_extensions/d4_value_hint/abl_wave8_d4v1_value_hint_forward",
    "D3 v2 alone (Wave 8)": ROOT / "outputs/experiments/abl/wave8_m4_extensions/d3_verify/abl_wave8_d3v2_verify2round",
}

COMB_A_LABEL = "Comb-A (D4 v1 + D3 v2 Stacked)"

KNOWN_F1 = {
    "M4 anchor": 0.8370,
    "anchor c01_01 (Wave 5)": 0.8664,
    "anchor c01_01 (Wave 7 relog)": 0.8638,
    "M1-B strong (Wave 6)": 0.8655,
    "D4 v1 alone (Wave 8)": 0.8393,
    "D3 v2 alone (Wave 8)": 0.8353,
    COMB_A_LABEL: 0.8684,
}


def _read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def parse_metrics_txt(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for ln in path.read_text().splitlines():
        if ":" not in ln:
            continue
        k, v = ln.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k in ("recall", "precision", "ex"):
            try:
                out[k] = float(v)
            except ValueError:
                pass
        elif k == "llm_calls":
            try:
                out["llm_calls"] = int(v)
            except ValueError:
                pass
        elif k == "filter_samples":
            try:
                out["n"] = int(v)
            except ValueError:
                pass
        elif k == "filter_time_mean_s":
            try:
                out["filter_time_mean_s"] = float(v)
            except ValueError:
                pass
    return out


def load_difficulty_lookup() -> dict[int, str]:
    dev_path = ROOT / "data/raw/BIRD_dev/dev.json"
    if not dev_path.exists():
        return {}
    with dev_path.open() as f:
        items = json.load(f)
    return {it["question_id"]: it.get("difficulty", "unknown") for it in items}


def per_difficulty_metrics(cell_dir: Path, diff_lookup: dict) -> dict:
    output = next(cell_dir.glob("output_*.jsonl"), None)
    if not output or not output.exists():
        return {}
    per_diff = defaultdict(list)
    for r in _read_jsonl(output):
        qid = r["question_id"]
        d = diff_lookup.get(qid)
        if not d:
            continue
        per_diff[d].append((r.get("recall", 0.0), r.get("precision", 0.0), int(r.get("ex", 0))))

    out = {}
    for d, vs in per_diff.items():
        if not vs:
            continue
        Rs = [v[0] for v in vs]
        Ps = [v[1] for v in vs]
        EXs = [v[2] for v in vs]
        R = _mean(Rs)
        P = _mean(Ps)
        out[d] = {
            "n": len(vs),
            "R": R,
            "P": P,
            "F1": 2 * R * P / (R + P) if (R + P) > 0 else 0.0,
            "EX": _mean(EXs),
        }
    return out


def analyze_comb_a_stages() -> dict:
    """Per-stage telemetry of Comb-A (StackedFilter)."""
    pred = COMB_A_DIR / "predictions.jsonl"
    rows = _read_jsonl(pred)
    stage0_evidence_size = []  # D4
    stage0_evidence_high_count = []
    stage0_forced_count = []
    stage0_nodes_in, stage0_nodes_out = [], []
    stage1_verify_success_rate = []  # D3
    stage1_avg_rounds_used = []
    stage1_recovered_count = []
    stage1_nodes_in, stage1_nodes_out = [], []
    overall_nodes_after = []

    for r in rows:
        fi = r.get("filter_info", {})
        si = fi.get("filter_stage_infos") or []
        overall_nodes_after.append(fi.get("filter_nodes_after"))
        if len(si) >= 1:
            s0 = si[0]
            info0 = s0.get("info", {})
            stage0_evidence_size.append(info0.get("filter_evidence_size"))
            stage0_evidence_high_count.append(info0.get("filter_evidence_high_count"))
            stage0_forced_count.append(info0.get("filter_forced_count"))
            stage0_nodes_in.append(s0.get("nodes_in"))
            stage0_nodes_out.append(s0.get("nodes_out"))
        if len(si) >= 2:
            s1 = si[1]
            info1 = s1.get("info", {})
            stage1_verify_success_rate.append(info1.get("filter_verify_success_rate"))
            stage1_avg_rounds_used.append(info1.get("filter_avg_rounds_used"))
            stage1_recovered_count.append(info1.get("filter_recovered_count"))
            stage1_nodes_in.append(s1.get("nodes_in"))
            stage1_nodes_out.append(s1.get("nodes_out"))

    def _safe_mean(values):
        nums = [v for v in values if isinstance(v, (int, float))]
        return _mean(nums)

    return {
        "n_rows": len(rows),
        "stage0_D4": {
            "evidence_size_mean": _safe_mean(stage0_evidence_size),
            "evidence_high_count_mean": _safe_mean(stage0_evidence_high_count),
            "forced_count_mean": _safe_mean(stage0_forced_count),
            "nodes_in_mean": _safe_mean(stage0_nodes_in),
            "nodes_out_mean": _safe_mean(stage0_nodes_out),
            "n_records": sum(1 for x in stage0_evidence_size if x is not None),
        },
        "stage1_D3": {
            "verify_success_rate_mean": _safe_mean(stage1_verify_success_rate),
            "avg_rounds_used_mean": _safe_mean(stage1_avg_rounds_used),
            "recovered_count_mean": _safe_mean(stage1_recovered_count),
            "recovered_count_sum": sum(v for v in stage1_recovered_count if isinstance(v, (int, float))),
            "nodes_in_mean": _safe_mean(stage1_nodes_in),
            "nodes_out_mean": _safe_mean(stage1_nodes_out),
            "n_records": sum(1 for x in stage1_verify_success_rate if x is not None),
        },
        "overall_nodes_after_mean": _safe_mean(overall_nodes_after),
    }


def analyze_pairwise_ex_diff(comb_a_dir: Path, d3v2_dir: Path, diff_lookup: dict) -> dict:
    """Per-query EX 비교 (Comb-A vs D3 v2 alone) — 어느 query 가 EX-down 인지 분해."""
    out_a = next(comb_a_dir.glob("output_*.jsonl"), None)
    out_b = next(d3v2_dir.glob("output_*.jsonl"), None)
    if not out_a or not out_b:
        return {}

    rows_a = {r["question_id"]: r for r in _read_jsonl(out_a)}
    rows_b = {r["question_id"]: r for r in _read_jsonl(out_b)}
    common = set(rows_a.keys()) & set(rows_b.keys())

    # EX(1, 0) 패턴 카운트
    counts = Counter()
    by_diff_counts = defaultdict(lambda: Counter())
    delta_R_for_ex_down = []
    delta_P_for_ex_down = []
    for qid in common:
        a = rows_a[qid]
        b = rows_b[qid]
        ex_a = int(a.get("ex", 0))
        ex_b = int(b.get("ex", 0))
        diff = diff_lookup.get(qid, "unknown")
        key = (ex_a, ex_b)  # (Comb-A EX, D3 v2 EX)
        counts[key] += 1
        by_diff_counts[diff][key] += 1
        if ex_a == 0 and ex_b == 1:
            delta_R_for_ex_down.append(a.get("recall", 0) - b.get("recall", 0))
            delta_P_for_ex_down.append(a.get("precision", 0) - b.get("precision", 0))

    return {
        "n_common": len(common),
        "ex_pattern_counts": dict(counts),  # {(combA_ex, d3v2_ex): count}
        "ex_pattern_by_difficulty": {d: dict(c) for d, c in by_diff_counts.items()},
        "ex_down_n": counts.get((0, 1), 0),  # Comb-A 0 but D3 v2 1
        "ex_up_n": counts.get((1, 0), 0),    # Comb-A 1 but D3 v2 0
        "mean_dR_for_ex_down": _mean(delta_R_for_ex_down) if delta_R_for_ex_down else None,
        "mean_dP_for_ex_down": _mean(delta_P_for_ex_down) if delta_P_for_ex_down else None,
    }


def main():
    diff_lookup = load_difficulty_lookup()

    # Cell-by-cell metrics + per-difficulty
    cells = {COMB_A_LABEL: COMB_A_DIR, **REFS}
    summary = {}
    for label, d in cells.items():
        m = parse_metrics_txt(d / "metrics.txt")
        if "recall" in m and "precision" in m:
            R = m["recall"]
            P = m["precision"]
            m["F1"] = 2 * R * P / (R + P) if (R + P) > 0 else 0.0
        m["per_difficulty"] = per_difficulty_metrics(d, diff_lookup)
        summary[label] = m

    # Per-stage telemetry of Comb-A
    stage_info = analyze_comb_a_stages()

    # EX-axis paradox decomposition (Comb-A vs D3 v2 alone)
    paradox = analyze_pairwise_ex_diff(COMB_A_DIR, REFS["D3 v2 alone (Wave 8)"], diff_lookup)

    print("=" * 100)
    print("Wave 8 Comb-A — Overall + per-difficulty")
    print("=" * 100)
    print(f"\n{'Cell':35s} {'R':>7s} {'P':>7s} {'F1':>7s} {'EX':>7s} {'n':>6s}")
    print("-" * 80)
    for label, m in summary.items():
        if "recall" not in m:
            continue
        print(f"{label:35s} {m['recall']:.4f} {m['precision']:.4f} {m['F1']:.4f} {m['ex']:.4f} {m.get('n', '-')}")

    print("\n--- Per-difficulty ---")
    for label, m in summary.items():
        pd = m.get("per_difficulty", {})
        if not pd:
            continue
        print(f"\n{label}:")
        for d in ("simple", "moderate", "challenging"):
            v = pd.get(d)
            if v:
                print(f"  {d:12s} n={v['n']:4d}  R={v['R']:.4f}  P={v['P']:.4f}  F1={v['F1']:.4f}  EX={v['EX']:.4f}")

    print("\n" + "=" * 100)
    print("Comb-A Stage Telemetry")
    print("=" * 100)
    print(f"\nn_rows: {stage_info['n_rows']}")
    print("\n--- Stage 0 (D4 v1 value_hint, Pre-Filter) ---")
    s0 = stage_info["stage0_D4"]
    print(f"  nodes_in mean:                {s0['nodes_in_mean']:.2f}  (= Extractor output)")
    print(f"  nodes_out mean:               {s0['nodes_out_mean']:.2f}")
    print(f"  evidence_size mean/q:         {s0['evidence_size_mean']:.2f}")
    print(f"  evidence_high_count mean/q:   {s0['evidence_high_count_mean']:.2f}")
    print(f"  forced_count mean/q:          {s0['forced_count_mean']:.4f}")
    print("\n--- Stage 1 (D3 v2 verify2round, Post-Filter) ---")
    s1 = stage_info["stage1_D3"]
    print(f"  nodes_in mean (= D4 nodes_out): {s1['nodes_in_mean']:.2f}")
    print(f"  nodes_out mean:                 {s1['nodes_out_mean']:.2f}")
    print(f"  verify_success_rate mean:       {s1['verify_success_rate_mean']:.4f}")
    print(f"  avg_rounds_used mean:           {s1['avg_rounds_used_mean']:.4f}")
    print(f"  recovered_count mean/q:         {s1['recovered_count_mean']:.6f}")
    print(f"  recovered_count sum (total):    {s1['recovered_count_sum']:.0f}")

    print(f"\nOverall (post-Stack) nodes_after mean: {stage_info['overall_nodes_after_mean']:.2f}")

    print("\n" + "=" * 100)
    print("EX-axis Paradox Decomposition (Comb-A vs D3 v2 alone)")
    print("=" * 100)
    print(f"\nn_common queries: {paradox['n_common']}")
    print(f"\nEX pattern (Comb-A EX, D3 v2 EX) → count:")
    for k, v in sorted(paradox["ex_pattern_counts"].items()):
        print(f"  {k}: {v}")
    print(f"\nEX-down (Comb-A 0, D3 v2 1): {paradox['ex_down_n']}")
    print(f"EX-up   (Comb-A 1, D3 v2 0): {paradox['ex_up_n']}")
    print(f"Net EX delta: {paradox['ex_up_n'] - paradox['ex_down_n']}")
    print(f"Mean ΔR for EX-down queries:  {paradox['mean_dR_for_ex_down']:+.4f}" if paradox.get('mean_dR_for_ex_down') is not None else "Mean ΔR for EX-down: n/a")
    print(f"Mean ΔP for EX-down queries:  {paradox['mean_dP_for_ex_down']:+.4f}" if paradox.get('mean_dP_for_ex_down') is not None else "Mean ΔP for EX-down: n/a")

    print("\nPer-difficulty EX pattern breakdown:")
    for d, c in paradox["ex_pattern_by_difficulty"].items():
        print(f"  {d}:")
        for k, v in sorted(c.items()):
            print(f"     {k}: {v}")

    # Save JSON
    out_path = ROOT / "outputs/analysis/wave8_comb_a_2026-05-19.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert tuple keys to strings for JSON
    paradox_serializable = dict(paradox)
    paradox_serializable["ex_pattern_counts"] = {f"{k[0]}_{k[1]}": v for k, v in paradox["ex_pattern_counts"].items()}
    paradox_serializable["ex_pattern_by_difficulty"] = {
        d: {f"{k[0]}_{k[1]}": v for k, v in c.items()}
        for d, c in paradox["ex_pattern_by_difficulty"].items()
    }
    payload = {
        "summary": summary,
        "stage_telemetry": stage_info,
        "ex_paradox": paradox_serializable,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n→ saved: {out_path}")


if __name__ == "__main__":
    main()
