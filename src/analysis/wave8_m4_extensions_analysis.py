"""Wave 8 M4 Bidirectional 발전 분석 — D1 + D2 + D3 + D4 × 8 cells.

8 cells × R/P/F1/EX + per-difficulty + direction-specific telemetry.
Source: outputs/experiments/abl/wave8_m4_extensions/{d1_decompose,d2_steiner,d3_verify,d4_value_hint}/abl_wave8_*/
M4 anchor: outputs/experiments/abl/wave6_recall_biased/w6_p2_m4_bidirectional/ (R=0.9325, P=0.7593, F1=0.8370, EX=0.5300)
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/hyeonjin/thesis_refactored")
WAVE8 = ROOT / "outputs/experiments/abl/wave8_m4_extensions"
M4_ANCHOR = ROOT / "outputs/experiments/abl/wave6_recall_biased/w6_p2_m4_bidirectional"

# Cell list with telemetry-specific fields
CELLS = [
    ("d1_decompose/abl_wave8_d1v1_multi_backward", "D1 v1 multi_backward"),
    ("d1_decompose/abl_wave8_d1v2_full_decompose", "D1 v2 full_decompose"),
    ("d2_steiner/abl_wave8_d2v1_direct_fk", "D2 v1 direct_fk"),
    ("d2_steiner/abl_wave8_d2v2_bridge_1hop", "D2 v2 bridge_1hop"),
    ("d3_verify/abl_wave8_d3v1_verify1round", "D3 v1 verify1round"),
    ("d3_verify/abl_wave8_d3v2_verify2round", "D3 v2 verify2round"),
    ("d4_value_hint/abl_wave8_d4v1_value_hint_forward", "D4 v1 value_hint"),
    ("d4_value_hint/abl_wave8_d4v3_forced_include", "D4 v3 forced_include"),
]

M4_METRICS = {"R": 0.9325, "P": 0.7593, "F1": 0.8370, "EX": 0.5300}


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
    """Read metrics.txt for overall R/P/EX + llm_calls."""
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


def analyze_cell(cell_rel: str, diff_lookup: dict) -> dict:
    cell_dir = WAVE8 / cell_rel
    name = cell_dir.name
    pred = cell_dir / "predictions.jsonl"
    output = next(cell_dir.glob("output_*.jsonl"), None)
    metrics = parse_metrics_txt(cell_dir / "metrics.txt")

    # per-difficulty R/EX
    per_diff = defaultdict(list)
    if output and output.exists():
        for r in _read_jsonl(output):
            qid = r["question_id"]
            d = diff_lookup.get(qid)
            if not d:
                continue
            per_diff[d].append((r.get("recall", 0.0), r.get("precision", 0.0), int(r.get("ex", 0))))

    diff_summary = {}
    for d, vals in per_diff.items():
        if not vals:
            continue
        Rs = [v[0] for v in vals]
        Ps = [v[1] for v in vals]
        EXs = [v[2] for v in vals]
        diff_summary[d] = {"n": len(vals), "R": _mean(Rs), "P": _mean(Ps), "EX": _mean(EXs)}

    # Direction-specific telemetry
    rows = _read_jsonl(pred)
    tele = analyze_telemetry(cell_rel, rows)

    return {
        "name": name,
        "metrics": metrics,
        "per_difficulty": diff_summary,
        "telemetry": tele,
    }


def analyze_telemetry(cell_rel: str, rows: list) -> dict:
    """Per-direction telemetry summary."""
    if not rows:
        return {}

    def grab(key):
        return [r.get("filter_info", {}).get(key) for r in rows if r.get("filter_info", {}).get(key) is not None]

    def safe_mean(values, predicate=None):
        if predicate is not None:
            values = [v for v in values if predicate(v)]
        nums = [v for v in values if isinstance(v, (int, float))]
        return _mean(nums)

    if "d1_" in cell_rel:
        sub_q = grab("filter_num_sub_questions")
        added = grab("filter_added_by_multi_backward")
        d1_calls = grab("filter_d1_llm_calls")
        failed = grab("filter_decompose_failed")
        return {
            "filter_num_sub_questions": {"mean": safe_mean(sub_q), "n": len(sub_q)},
            "filter_added_by_multi_backward": {"mean": safe_mean(added), "n": len(added)},
            "filter_d1_llm_calls": {"mean": safe_mean(d1_calls), "n": len(d1_calls)},
            "decompose_failed_count": sum(1 for f in failed if f),
        }
    if "d2_" in cell_rel:
        added = grab("filter_d2_added_count")
        gold_added = grab("filter_d2_added_gold_count")
        precision = grab("filter_d2_steiner_precision")
        variant = grab("filter_d2_variant")
        skipped = grab("filter_d2_skipped_reason")
        return {
            "filter_d2_added_count": {"mean": safe_mean(added), "n": len(added)},
            "filter_d2_added_gold_count": {"mean": safe_mean(gold_added), "n": len(gold_added)},
            "filter_d2_steiner_precision": {"mean": safe_mean(precision), "n": len(precision)},
            "variant_distribution": {v: sum(1 for x in variant if x == v) for v in set(variant)} if variant else {},
            "skipped_reason_distribution": {v: sum(1 for x in skipped if x == v) for v in set(skipped) if v} if skipped else {},
        }
    if "d3_" in cell_rel:
        success = grab("filter_verify_success_rate")
        rounds = grab("filter_avg_rounds_used")
        recovered = grab("filter_recovered_count")
        d3_calls = grab("filter_d3_llm_calls")
        return {
            "filter_verify_success_rate": {"mean": safe_mean(success), "n": len(success)},
            "filter_avg_rounds_used": {"mean": safe_mean(rounds), "n": len(rounds)},
            "filter_recovered_count": {"mean": safe_mean(recovered), "n": len(recovered)},
            "filter_d3_llm_calls": {"mean": safe_mean(d3_calls), "n": len(d3_calls)},
        }
    if "d4_" in cell_rel:
        ev_size = grab("filter_evidence_size")
        ev_high = grab("filter_evidence_high_count")
        forced = grab("filter_forced_count")
        ev_prec = grab("filter_evidence_gold_precision")
        return {
            "filter_evidence_size": {"mean": safe_mean(ev_size), "n": len(ev_size)},
            "filter_evidence_high_count": {"mean": safe_mean(ev_high), "n": len(ev_high)},
            "filter_evidence_gold_precision": {"mean": safe_mean(ev_prec), "n": len(ev_prec)},
            "filter_forced_count": {"mean": safe_mean(forced), "n": len(forced)},
        }
    return {}


def m4_anchor_per_difficulty(diff_lookup: dict) -> dict:
    output = next(M4_ANCHOR.glob("output_*.jsonl"), None)
    rows = _read_jsonl(output) if output and output.exists() else []
    per_diff = defaultdict(list)
    for r in rows:
        qid = r["question_id"]
        d = diff_lookup.get(qid)
        if not d:
            continue
        per_diff[d].append((r.get("recall", 0.0), r.get("precision", 0.0), int(r.get("ex", 0))))
    return {
        d: {"n": len(vs), "R": _mean([v[0] for v in vs]), "P": _mean([v[1] for v in vs]), "EX": _mean([v[2] for v in vs])}
        for d, vs in per_diff.items() if vs
    }


def main():
    diff = load_difficulty_lookup()
    results = []
    for cell_rel, label in CELLS:
        res = analyze_cell(cell_rel, diff)
        res["label"] = label
        res["cell_rel"] = cell_rel
        results.append(res)
    m4_diff = m4_anchor_per_difficulty(diff)

    print("\n" + "=" * 100)
    print("Wave 8 M4 Extensions — 8 cells × R/P/F1/EX vs M4 anchor")
    print("=" * 100)
    print(f"\nM4 anchor: R={M4_METRICS['R']:.4f}, P={M4_METRICS['P']:.4f}, F1={M4_METRICS['F1']:.4f}, EX={M4_METRICS['EX']:.4f}")
    print()
    print(f"{'Cell':35s} {'R':>7s} {'P':>7s} {'F1':>7s} {'EX':>7s} | {'ΔR':>8s} {'ΔP':>8s} {'ΔF1':>8s} {'ΔEX':>8s}")
    print("-" * 110)
    for res in results:
        m = res["metrics"]
        R = m.get("recall")
        P = m.get("precision")
        EX = m.get("ex")
        F1 = (2 * R * P / (R + P)) if (R is not None and P is not None and (R + P) > 0) else None
        dR = R - M4_METRICS["R"] if R is not None else None
        dP = P - M4_METRICS["P"] if P is not None else None
        dF1 = F1 - M4_METRICS["F1"] if F1 is not None else None
        dEX = EX - M4_METRICS["EX"] if EX is not None else None
        print(f"{res['label']:35s} {R:.4f} {P:.4f} {F1:.4f} {EX:.4f} | {dR:+.4f} {dP:+.4f} {dF1:+.4f} {dEX:+.4f}")

    print("\n" + "=" * 100)
    print("Per-difficulty R / EX (per cell)")
    print("=" * 100)
    print(f"\nM4 anchor per-difficulty:")
    for d in ("simple", "moderate", "challenging"):
        v = m4_diff.get(d)
        if v:
            print(f"  {d:12s} n={v['n']:4d}  R={v['R']:.4f}  EX={v['EX']:.4f}")
    print()
    print(f"{'Cell':35s} {'simple R':>9s} {'simple EX':>9s} {'mod R':>9s} {'mod EX':>9s} {'chall R':>9s} {'chall EX':>9s}")
    print("-" * 110)
    for res in results:
        pd = res["per_difficulty"]
        sR = pd.get("simple", {}).get("R")
        sEX = pd.get("simple", {}).get("EX")
        mR = pd.get("moderate", {}).get("R")
        mEX = pd.get("moderate", {}).get("EX")
        cR = pd.get("challenging", {}).get("R")
        cEX = pd.get("challenging", {}).get("EX")
        def fmt(x):
            return f"{x:.4f}" if x is not None else "n/a"
        print(f"{res['label']:35s} {fmt(sR):>9s} {fmt(sEX):>9s} {fmt(mR):>9s} {fmt(mEX):>9s} {fmt(cR):>9s} {fmt(cEX):>9s}")

    print("\n" + "=" * 100)
    print("Direction-specific Telemetry")
    print("=" * 100)
    for res in results:
        print(f"\n--- {res['label']} ---")
        for k, v in (res["telemetry"] or {}).items():
            print(f"  {k}: {v}")

    # save JSON
    out_path = ROOT / "outputs/analysis/wave8_m4_extensions_2026-05-19.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "m4_anchor": {"overall": M4_METRICS, "per_difficulty": m4_diff},
        "cells": results,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n→ saved: {out_path}")


if __name__ == "__main__":
    main()
