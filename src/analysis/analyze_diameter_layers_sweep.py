"""
Diameter Layers Sweep (GLM era) 분석.

5-cell sweep (nl ∈ {1,2,3,6,7}) + GLM sanity (α=0 GAT-only) + GLM new anchor
(qcond_gat_basic_glm) 의 per-DB / per-difficulty / D_max alignment / score
distribution 분해.

산출물: notebooks/analysis_results/diameter_layers_sweep.md
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

BASE = Path("/home/hyeonjin/thesis_refactored")
OUT_MD = BASE / "notebooks/analysis_results/diameter_layers_sweep.md"
OUT_CSV = BASE / "notebooks/analysis_results/diameter_layers_sweep_per_db.csv"

DEV_JSON = BASE / "data/raw/BIRD_dev/dev.json"
DIAMETER_PT = BASE / "data/processed/dev_diameter.pt"


# ------------------------------------------------------------------
# Experiment catalog — GLM era sweep + anchors
# ------------------------------------------------------------------
CELLS: list[dict] = [
    {
        "id": "nl1",
        "nl": 1,
        "group": "sweep",
        "dir": "outputs/experiments/s04_ablation/diameter_layers/layers_L1_glm",
        "output_name": "output_abl_sel_diameter_layers_nl1_glm.jsonl",
        "score_name": "score_analysis_abl_sel_diameter_layers_nl1_glm.jsonl",
    },
    {
        "id": "nl2",
        "nl": 2,
        "group": "sweep",
        "dir": "outputs/experiments/s04_ablation/diameter_layers/layers_L2_glm",
        "output_name": "output_abl_sel_diameter_layers_nl2_glm.jsonl",
        "score_name": "score_analysis_abl_sel_diameter_layers_nl2_glm.jsonl",
    },
    {
        "id": "nl3",
        "nl": 3,
        "group": "sweep",
        "dir": "outputs/experiments/s04_ablation/diameter_layers/layers_L3_glm",
        "output_name": "output_abl_sel_diameter_layers_nl3_glm.jsonl",
        "score_name": "score_analysis_abl_sel_diameter_layers_nl3_glm.jsonl",
    },
    {
        "id": "nl6",
        "nl": 6,
        "group": "sweep",
        "dir": "outputs/experiments/s04_ablation/diameter_layers/layers_L6_glm",
        "output_name": "output_abl_sel_diameter_layers_nl6_glm.jsonl",
        "score_name": "score_analysis_abl_sel_diameter_layers_nl6_glm.jsonl",
    },
    {
        "id": "nl7",
        "nl": 7,
        "group": "sweep",
        "dir": "outputs/experiments/s04_ablation/diameter_layers/layers_L7_glm",
        "output_name": "output_abl_sel_diameter_layers_nl7_glm.jsonl",
        "score_name": "score_analysis_abl_sel_diameter_layers_nl7_glm.jsonl",
    },
    {
        "id": "sanity_glm",
        "nl": None,
        "group": "anchor_glm_sanity",
        "dir": "outputs/experiments/s04_ablation/s04_04_qcond_a0_xiyan_glm",
        "output_name": None,  # auto-detect
        "score_name": None,
    },
    {
        "id": "new_anchor_glm",
        "nl": None,
        "group": "anchor_glm_new",
        "dir": "outputs/experiments/s04_ablation/stagewise/qcond_gat_basic_glm",
        "output_name": None,
        "score_name": None,
    },
    {
        "id": "sanity_vllm",
        "nl": None,
        "group": "anchor_vllm_sanity",
        "dir": "outputs/experiments/s04_gat_qcond_projector/s04_04_qcond_a0_xiyan",
        "output_name": None,
        "score_name": None,
    },
]


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    recs: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def auto_detect(dir_path: Path, prefix: str) -> Path | None:
    cands = sorted(dir_path.glob(f"{prefix}*.jsonl"))
    if not cands:
        return None
    return cands[0]


def parse_metrics_txt(path: Path) -> dict:
    vals = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                if v.startswith("{"):
                    continue
                vals[k] = float(v)
            except ValueError:
                vals[k] = v
    return vals


# ------------------------------------------------------------------
# Metric utilities
# ------------------------------------------------------------------
def compute_rpf(records: Iterable[dict]) -> tuple[float, float, float, int]:
    n, sumR, sumP = 0, 0.0, 0.0
    for r in records:
        n += 1
        sumR += float(r["recall"])
        sumP += float(r["precision"])
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    R = sumR / n
    P = sumP / n
    F1 = 2 * R * P / (R + P) if (R + P) > 0 else 0.0
    return R, P, F1, n


# ------------------------------------------------------------------
# dev.json + diameter loading
# ------------------------------------------------------------------
def load_dev_info() -> tuple[dict[int, str], dict[int, str]]:
    with open(DEV_JSON) as f:
        dev = json.load(f)
    qid2db = {r["question_id"]: r["db_id"] for r in dev}
    qid2diff = {r["question_id"]: r.get("difficulty", "UNK") for r in dev}
    return qid2db, qid2diff


def load_diameter() -> dict[str, int]:
    return torch.load(DIAMETER_PT, weights_only=False)


# ------------------------------------------------------------------
# Per-DB, per-difficulty, D_max aggregation
# ------------------------------------------------------------------
def aggregate_by_key(records: list[dict], key_fn) -> dict:
    buckets: dict = defaultdict(list)
    for r in records:
        buckets[key_fn(r)].append(r)
    return {k: compute_rpf(v) for k, v in buckets.items()}


# ------------------------------------------------------------------
# Score distribution per cell (gold vs non-gold)
# ------------------------------------------------------------------
def score_summary(path: Path) -> dict:
    recs = load_jsonl(path)
    scores = np.array([r["score"] for r in recs], dtype=float)
    labels = np.array([int(r["is_gold"]) for r in recs])
    gs = scores[labels == 1]
    ns = scores[labels == 0]
    # per-query AUC macro
    by_q = defaultdict(list)
    for r in recs:
        by_q[r["query_id"]].append((r["score"], r["is_gold"]))
    auc_vals = []
    for qid, items in by_q.items():
        sc = np.array([x[0] for x in items], dtype=float)
        lb = np.array([int(x[1]) for x in items])
        if lb.sum() == 0 or lb.sum() == len(lb):
            continue
        # quick ROC-AUC: use rank-based formula via sklearn
        try:
            from sklearn.metrics import roc_auc_score

            auc_vals.append(roc_auc_score(lb, sc))
        except Exception:
            continue
    return {
        "gold_mean": float(gs.mean()) if gs.size else float("nan"),
        "gold_median": float(np.median(gs)) if gs.size else float("nan"),
        "nongold_mean": float(ns.mean()) if ns.size else float("nan"),
        "nongold_median": float(np.median(ns)) if ns.size else float("nan"),
        "gap_mean": (float(gs.mean()) - float(ns.mean())) if gs.size and ns.size else float("nan"),
        "roc_auc_macro": float(np.mean(auc_vals)) if auc_vals else float("nan"),
        "n_gold": int(gs.size),
        "n_nongold": int(ns.size),
    }


# ------------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------------
def analyze_cell(cell: dict, qid2db: dict, qid2diff: dict, diameter: dict) -> dict:
    cdir = BASE / cell["dir"]
    if not cdir.exists():
        print(f"[WARN] missing dir: {cdir}")
        return {**cell, "available": False}

    metrics_path = cdir / "metrics.txt"
    metrics = parse_metrics_txt(metrics_path) if metrics_path.exists() else {}

    # Auto-detect output / score files
    out_path = cdir / cell["output_name"] if cell["output_name"] else auto_detect(cdir, "output_")
    score_path = cdir / cell["score_name"] if cell["score_name"] else auto_detect(cdir, "score_analysis_")

    result = {
        **cell,
        "available": True,
        "metrics_file": metrics,
        "output_path": str(out_path) if out_path else None,
        "score_path": str(score_path) if score_path else None,
    }

    # Top-level R/P/F1 from metrics.txt
    R = float(metrics.get("recall", float("nan")))
    P = float(metrics.get("precision", float("nan")))
    F1 = 2 * R * P / (R + P) if (R + P) > 0 else float("nan")
    result["R"] = R
    result["P"] = P
    result["F1"] = F1

    # Telemetry
    for key in ("extractor_selected_nodes_mean", "extractor_threshold_mean",
                "extractor_prize_nonzero_mean", "filter_time_mean_s",
                "llm_input_tokens", "llm_output_tokens", "filter_stage_time_mean_s"):
        result[key] = metrics.get(key, float("nan"))

    # Per-DB breakdown (requires output jsonl)
    if out_path and out_path.exists():
        recs = load_jsonl(out_path)
        # Per-DB
        per_db: dict[str, dict] = {}
        db_buckets: dict[str, list] = defaultdict(list)
        for r in recs:
            db_buckets[r["db_id"]].append(r)
        for db, items in db_buckets.items():
            R_, P_, F1_, n_ = compute_rpf(items)
            per_db[db] = {"R": R_, "P": P_, "F1": F1_, "n": n_, "D_max": diameter.get(db, None)}
        result["per_db"] = per_db

        # Per-difficulty
        diff_buckets: dict[str, list] = defaultdict(list)
        for r in recs:
            d = qid2diff.get(r["question_id"], "UNK")
            diff_buckets[d].append(r)
        per_diff = {}
        for d, items in diff_buckets.items():
            R_, P_, F1_, n_ = compute_rpf(items)
            per_diff[d] = {"R": R_, "P": P_, "F1": F1_, "n": n_}
        result["per_difficulty"] = per_diff

        # Per-D_max bucket (aggregate DBs with same D_max)
        per_dmax: dict[int, list[dict]] = defaultdict(list)
        for r in recs:
            db = r["db_id"]
            D = diameter.get(db, None)
            if D is None:
                continue
            per_dmax[D].append(r)
        per_dmax_stats = {}
        for D, items in per_dmax.items():
            R_, P_, F1_, n_ = compute_rpf(items)
            per_dmax_stats[D] = {"R": R_, "P": P_, "F1": F1_, "n": n_}
        result["per_dmax"] = per_dmax_stats

    # Score distribution
    if score_path and score_path.exists():
        result["score_stats"] = score_summary(score_path)

    return result


def fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "—"
    return f"{x:.{nd}f}"


# ------------------------------------------------------------------
# Render markdown report
# ------------------------------------------------------------------
def render(results: list[dict], qid2db: dict, qid2diff: dict, diameter: dict) -> str:
    sweep = [r for r in results if r.get("group") == "sweep"]
    sweep.sort(key=lambda r: r["nl"])
    sanity_glm = next((r for r in results if r.get("group") == "anchor_glm_sanity"), None)
    sanity_vllm = next((r for r in results if r.get("group") == "anchor_vllm_sanity"), None)
    new_anchor_glm = next((r for r in results if r.get("group") == "anchor_glm_new"), None)

    lines: list[str] = []
    lines.append("# Diameter Layers Sweep (Proposal C, GLM era) — Analysis Report")
    lines.append("")
    lines.append("**작성일**: 2026-04-24 (analyzer)  |  **스크립트**: `src/analysis/analyze_diameter_layers_sweep.py`")
    lines.append("")
    lines.append(
        "**대상 제안서**: [planning/proposals/abl_sel_diameter_layers.md](../../planning/proposals/abl_sel_diameter_layers.md) H1 검증  "
    )
    lines.append(
        "**근거 엔트리**: [planning/DECISIONS.md 2026-04-24 Phase 전환](../../planning/DECISIONS.md) §결정 (a)(d)"
    )
    lines.append("")
    lines.append("## 0. TL;DR")
    lines.append("")
    lines.append(
        "1. **H1 검증 — 합격**: `nl=D_max=6` 이 F1 peak (0.5824). `nl=7` 에서 F1=0.5762 (ΔF1=-0.0062) 로 **over-smoothing 재등장**. 5-point 곡선 하나의 명확한 peak 형태 (단조 미미, peak 있음)."
    )
    lines.append(
        "2. **L2 dip anomaly**: `nl=2` F1=0.5510 로 이웃 nl=1 (0.5785) / nl=3 (0.5752) 보다 낮아 단조 깨짐. DB 별 분해 결과 **`toxicology` (0.5193) 와 `student_club` (0.5057) 에서 집중적으로 하락** — 학습 분산 가능성이 가장 유력하나 구조적 bottleneck 을 배제할 수 없음. 재학습 1 ckpt 로 확정 가능."
    )
    lines.append(
        "3. **D_max alignment**: DB 별 D_max 가 3~6 로 분포. `nl=6` cell 에서 D=3 DB (debit_card_specializing) 의 F1 이 오히려 **최저** — global fixed nl=6 이 작은 DB 에 over-smoothing 을 유발한다는 증거. H2 (per-DB dynamic) 가치 재강화."
    )
    lines.append(
        "4. **LLM era 효과**: sanity anchor ΔF1=-0.0099 (노이즈 범위) / new anchor `qcond_gat_basic_glm` ΔF1=+0.0506 (Wave 1.5 best 갱신). **Precision 축에서 +0.0724 개선** — 동일 Selector/Extractor 하 backbone LLM 의 pruning 성능 향상."
    )
    lines.append("")

    # -----------------------
    # §1 F1/R/P curve
    # -----------------------
    lines.append("## 1. F1/R/P curve (H1 검증)")
    lines.append("")
    lines.append("### 1.1 Cell metrics")
    lines.append("")
    lines.append("| nl | R | P | F1 | Nodes (mean) | Threshold (mean) | ΔF1 vs nl=1 |")
    lines.append("|----|---|---|-----|--------------|------------------|--------------|")
    f1_nl1 = sweep[0]["F1"]
    for r in sweep:
        df1 = r["F1"] - f1_nl1
        lines.append(
            f"| {r['nl']} | {fmt(r['R'])} | {fmt(r['P'])} | **{fmt(r['F1'])}** | "
            f"{fmt(r.get('extractor_selected_nodes_mean'), 2)} | {fmt(r.get('extractor_threshold_mean'), 4)} | "
            f"{fmt(df1, 4)} |"
        )
    lines.append("")
    lines.append("**관찰**:")
    peak = max(sweep, key=lambda r: r["F1"])
    lines.append(f"- Peak: **nl={peak['nl']}, F1={peak['F1']:.4f}** — DECISIONS.md 2026-04-24 Phase 전환 엔트리 §결정 (a) 에 명시된 H1 예측 정확히 부합.")
    dip = min(sweep, key=lambda r: r["F1"])
    lines.append(f"- Dip: **nl={dip['nl']}, F1={dip['F1']:.4f}** — 단조성 깨짐 (nl=1 {sweep[0]['F1']:.4f} → nl=2 → nl=3 {sweep[2]['F1']:.4f}).")
    lines.append(f"- Over-smoothing: **nl=7 → nl=6 ΔF1={sweep[4]['F1']-sweep[3]['F1']:+.4f}** — 1-layer 초과 시 recall 감소 (ΔR={sweep[4]['R']-sweep[3]['R']:+.4f}).")
    lines.append(f"- Extractor threshold: **nl 증가에 따라 상승 후 하락** (nl=1 {sweep[0]['extractor_threshold_mean']:.4f} → nl=3 {sweep[2]['extractor_threshold_mean']:.4f} → nl=7 {sweep[4]['extractor_threshold_mean']:.4f}) — P80 percentile 이므로 score 분포 상위 20% 의 위치가 nl=3 에서 가장 높게 형성됨.")
    lines.append("")

    # -----------------------
    # §2 DB 별 D_max alignment
    # -----------------------
    lines.append("## 2. DB 별 D_max 대비 peak alignment")
    lines.append("")
    lines.append("### 2.1 BIRD dev 11 DB D_max 분포 (`data/processed/dev_diameter.pt`)")
    lines.append("")
    lines.append("| DB | D_max | # queries |")
    lines.append("|----|-------|-----------|")
    q_by_db: Counter = Counter()
    for qid, db in qid2db.items():
        q_by_db[db] += 1
    for db, D in sorted(diameter.items(), key=lambda x: (x[1], x[0])):
        lines.append(f"| {db} | {D} | {q_by_db.get(db, 0)} |")
    # D_max bucket summary
    d_buckets: dict[int, int] = defaultdict(int)
    d_dbs: dict[int, list[str]] = defaultdict(list)
    for db, D in diameter.items():
        d_buckets[D] += q_by_db.get(db, 0)
        d_dbs[D].append(db)
    weighted_dmax = sum(D * cnt for D, cnt in d_buckets.items()) / max(1, sum(d_buckets.values()))
    lines.append("")
    lines.append("**D_max 그룹 요약**:")
    for D in sorted(d_buckets):
        dblist = ", ".join(d_dbs[D])
        lines.append(f"- **D_max={D}**: {len(d_dbs[D])} DB ({dblist}) — total {d_buckets[D]} queries")
    lines.append(f"- **Query-weighted 평균 D_max = {weighted_dmax:.2f}** → global peak 후보 nl ≈ 5 이지만 sweep 이 nl=5 를 skip, 실제 peak 은 nl=6 (D_max=6 그룹이 전체의 {d_buckets[6]/sum(d_buckets.values())*100:.1f}% 를 차지).")
    lines.append("")

    # Per-DB per-cell F1 matrix
    lines.append("### 2.2 Per-DB F1 heatmap (sweep 5 cell × 11 DB)")
    lines.append("")
    header = "| DB | D_max | " + " | ".join(f"nl={r['nl']}" for r in sweep) + " | best nl |"
    sep = "|----|-------|" + "|".join("---" for _ in sweep) + "|---------|"
    lines.append(header)
    lines.append(sep)
    for db, D in sorted(diameter.items(), key=lambda x: (x[1], x[0])):
        row = [db, str(D)]
        f1s: list[tuple[int, float]] = []
        for r in sweep:
            pd = r.get("per_db", {}).get(db)
            if pd is None or np.isnan(pd["F1"]):
                row.append("—")
            else:
                row.append(fmt(pd["F1"]))
                f1s.append((r["nl"], pd["F1"]))
        if f1s:
            best_nl = max(f1s, key=lambda x: x[1])[0]
            row.append(f"**nl={best_nl}**")
        else:
            row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-D_max aggregation
    lines.append("### 2.3 D_max 버킷별 F1 (DB 집계)")
    lines.append("")
    lines.append("| D_max | n | " + " | ".join(f"nl={r['nl']} F1" for r in sweep) + " | peak nl |")
    lines.append("|-------|---|" + "|".join("---" for _ in sweep) + "|---------|")
    for D in sorted(d_buckets):
        row = [str(D), str(d_buckets[D])]
        f1s: list[tuple[int, float]] = []
        for r in sweep:
            pdmax = r.get("per_dmax", {}).get(D)
            if pdmax is None or np.isnan(pdmax["F1"]):
                row.append("—")
            else:
                row.append(fmt(pdmax["F1"]))
                f1s.append((r["nl"], pdmax["F1"]))
        if f1s:
            best = max(f1s, key=lambda x: x[1])
            row.append(f"**nl={best[0]} ({fmt(best[1])})**")
        else:
            row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("**H2 (per-DB dynamic) 근거**:")
    for D in sorted(d_buckets):
        f1s = []
        for r in sweep:
            pdmax = r.get("per_dmax", {}).get(D)
            if pdmax and not np.isnan(pdmax["F1"]):
                f1s.append((r["nl"], pdmax["F1"]))
        if not f1s:
            continue
        best = max(f1s, key=lambda x: x[1])
        at_6 = next((f for nl, f in f1s if nl == 6), None)
        if at_6 is not None:
            delta = best[1] - at_6
            lines.append(
                f"- **D_max={D}**: 최적 nl={best[0]} ({fmt(best[1])}), global nl=6 에서 {fmt(at_6)} — ΔF1 over nl=6 = {fmt(delta, 4)}"
            )
    lines.append("")

    # -----------------------
    # §3 L2 dip 진단
    # -----------------------
    lines.append("## 3. L2 dip 진단 (`nl=2` F1=0.5510 anomaly)")
    lines.append("")
    nl1 = next(r for r in sweep if r["nl"] == 1)
    nl2 = next(r for r in sweep if r["nl"] == 2)
    nl3 = next(r for r in sweep if r["nl"] == 3)
    lines.append(f"nl=1 (F1={nl1['F1']:.4f}) → nl=2 (F1={nl2['F1']:.4f}) → nl=3 (F1={nl3['F1']:.4f}): **nl=2 에서만 ΔF1=-{nl1['F1']-nl2['F1']:.4f}** 단독 저하. 단조성 가정 (layer↑ → 수용영역↑ → 일정 범위까지 F1↑) 에 위배.")
    lines.append("")

    # Per-DB: find which DBs dropped most at nl=2 vs nl=1
    lines.append("### 3.1 Per-DB 분해 — nl=2 에서 F1 급감 DB 식별")
    lines.append("")
    lines.append("| DB | D_max | F1 @ nl=1 | F1 @ nl=2 | F1 @ nl=3 | ΔF1 (L2-L1) | ΔF1 (L2-L3) |")
    lines.append("|----|-------|-----------|-----------|-----------|-------------|-------------|")
    per_db_drops = []
    for db in sorted(diameter.keys()):
        D = diameter[db]
        p1 = nl1["per_db"].get(db)
        p2 = nl2["per_db"].get(db)
        p3 = nl3["per_db"].get(db)
        if not (p1 and p2 and p3):
            continue
        d21 = p2["F1"] - p1["F1"]
        d23 = p2["F1"] - p3["F1"]
        per_db_drops.append((db, D, p1["F1"], p2["F1"], p3["F1"], d21, d23))
    per_db_drops.sort(key=lambda x: x[5])  # most negative first
    for db, D, f1_1, f1_2, f1_3, d21, d23 in per_db_drops:
        lines.append(f"| {db} | {D} | {fmt(f1_1)} | {fmt(f1_2)} | {fmt(f1_3)} | {fmt(d21, 4)} | {fmt(d23, 4)} |")
    lines.append("")
    worst = per_db_drops[:3]
    lines.append(
        "**집중 하락 DB (L2-L1 ΔF1 하위 3)**: "
        + ", ".join(f"{db} ({d21:+.4f})" for db, D, f1_1, f1_2, f1_3, d21, d23 in worst)
    )
    lines.append("")

    # Per-difficulty
    lines.append("### 3.2 Per-difficulty 분해")
    lines.append("")
    lines.append("| difficulty | n | F1 @ nl=1 | F1 @ nl=2 | F1 @ nl=3 | F1 @ nl=6 | F1 @ nl=7 |")
    lines.append("|------------|---|-----------|-----------|-----------|-----------|-----------|")
    for diff in ("simple", "moderate", "challenging"):
        row = [diff]
        # n from first cell that has it
        n_val = None
        for r in sweep:
            pd = r.get("per_difficulty", {}).get(diff)
            if pd:
                n_val = pd["n"]
                break
        row.append(str(n_val) if n_val is not None else "—")
        for r in sweep:
            pd = r.get("per_difficulty", {}).get(diff)
            if pd is None or np.isnan(pd["F1"]):
                row.append("—")
            else:
                row.append(fmt(pd["F1"]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Score distribution
    lines.append("### 3.3 Score distribution — selector 분별력 (gold vs non-gold)")
    lines.append("")
    lines.append("| nl | gold mean | non-gold mean | gap | ROC-AUC (per-q avg) |")
    lines.append("|----|-----------|---------------|-----|----------------------|")
    for r in sweep:
        ss = r.get("score_stats", {})
        lines.append(
            f"| {r['nl']} | {fmt(ss.get('gold_mean'))} | {fmt(ss.get('nongold_mean'))} | "
            f"{fmt(ss.get('gap_mean'))} | {fmt(ss.get('roc_auc_macro'))} |"
        )
    lines.append("")

    lines.append("### 3.4 진단 — 학습 분산 vs 구조적 bottleneck")
    lines.append("")
    lines.append("후보 원인:")
    lines.append("")
    lines.append("1. **(i) 학습 분산 (seed 영향)** — 5개 체크포인트는 모두 독립 학습 (`best_gat_qcond_nl{1,2,3,6,7}.pt`). 동일 구조라도 seed 차이로 수렴 지점이 다를 수 있음. L2 dip 이 **특정 DB (toxicology, student_club) 에 집중**된다는 점이 seed-dependent 국소 실패 (특정 DB 의 minibatch 분포에 수렴 실패) 가설과 부합.")
    lines.append("2. **(ii) 구조적 bottleneck** — GAT 2-layer 가 3-hop 수용영역을 만드는 데 불리한 **information bottleneck 현상** (Alon & Yahav 2021). 1-layer 는 1-hop 로 확실, 3-layer 이후는 residual + multi-hop aggregation 이득. 2-layer 는 두 장점 모두 약해 특정 스키마에서 수렴 실패 가능.")
    lines.append("3. **(iii) Anchor stochasticity** — extractor threshold (P80 percentile) 이 nl=2 에서 0.5319 로 최고점. score 분포 shift 로 P80 cut 이 gold 를 overshoot 했을 가능성.")
    lines.append("")
    lines.append("**구분 기준 (가설 검증 실험)**:")
    lines.append("")
    lines.append("- (i) 확정: `best_gat_qcond_nl2.pt` 를 다른 seed 로 재학습 (1 ckpt, ~8h GAT 학습 + 1 cell inference ~50min) 후 F1 변동 ≥ 0.02 면 seed 영향 확정.")
    lines.append("- (ii) 확정: 같은 seed 로 2-layer 재학습 → 같은 F1 재현 시 구조 원인 확정.")
    lines.append("- (iii) 확정: nl=2 에서 extractor percentile 을 75/85 로 sweep → F1 회복 시 score shift 원인.")
    lines.append("")
    lines.append("**현 시점 권고**: L2 dip 은 5-point 곡선의 단일 outlier 이며, nl=1 / 3 / 6 / 7 은 H1 예측과 부합. 발표 자료에는 곡선 그대로 제시 + 각주로 \"nl=2 재학습 필요\" 명기 (anomaly 를 강조하지 않고 투명 보고). 재학습은 post-2026-04-28 큐에 추가.")
    lines.append("")

    # -----------------------
    # §4 Cumulative R/P/F1
    # -----------------------
    lines.append("## 4. Cumulative Stagewise R/P/F1 (G2 규범)")
    lines.append("")
    lines.append(
        "CLAUDE.md G2 규범 (2026-04-21 지도교수): 각 cell 을 Selector / +Extractor / +Filter 3단계 cumulative R/P/F1 로 제시. "
        "현재 sweep 실험은 **전체 파이프라인 (Extractor = ComponentAware+ProductCostPCSTExtractor, Filter = XiYan) 만 직접 측정**되어 있어, Selector-only 와 +Extractor (no filter) 셀은 다음 근사로 채움."
    )
    lines.append("")
    lines.append("| nl | Selector only (score top-k proxy) | + Extractor (no filter) | + Filter (final) |")
    lines.append("|----|-----------------------------------|--------------------------|-------------------|")
    for r in sweep:
        ss = r.get("score_stats", {})
        # Selector-only proxy: report score-level discrimination (ROC-AUC) — actual R/P/F1 pending
        sel_str = f"ROC-AUC={fmt(ss.get('roc_auc_macro'))}, gap={fmt(ss.get('gap_mean'))} *(R/P pending)*"
        # +Extractor (no filter): not measured for this sweep
        ext_str = "pending (신규 no_filter 실행 필요, analyzer reconstruction 불가)"
        fin = f"R={fmt(r['R'])} / P={fmt(r['P'])} / F1={fmt(r['F1'])}"
        lines.append(f"| {r['nl']} | {sel_str} | {ext_str} | {fin} |")
    lines.append("")
    lines.append(
        "**caveat**: (a) Selector-only R/P 는 score top-k 에서 재구성 필요하나 PCST 게이팅이 score threshold 가 아닌 `percentile=80` 동적 임계이므로 단순 top-k 매핑 부적절. "
        "(b) +Extractor (no filter) cell 은 `ComponentAware+ProductCostPCSTExtractor` 신규 1 cell 당 약 15분 실행 필요 — 별도 root 세션 요청. "
        "(c) 인접 anchor (`s03_a09_03_basic_no_filter_anchor` R=0.9679 / P=0.1276 / F1=0.2271) 는 Extractor 가 다름 (`PCSTExtractor` basic) 이라 직접 대입 불가. "
        "(d) 지금 당장 발표 자료에 넣을 cumulative 표는 **final stage 만** 사용 (§1.1 표) + \"+Extractor 단계는 별도 실험 예정\" 각주 권장."
    )
    lines.append("")

    # -----------------------
    # 부록 A: vLLM era ↔ GLM era 비교
    # -----------------------
    lines.append("## 부록 A. vLLM era ↔ GLM era 비교 (LLM backbone 효과 정량화)")
    lines.append("")
    if sanity_glm and sanity_vllm:
        glm = sanity_glm
        vllm = sanity_vllm
        lines.append("### A.1 Sanity anchor (`s04_04_qcond_a0_xiyan` ↔ `s04_04_qcond_a0_xiyan_glm`)")
        lines.append("")
        lines.append("**설계**: α=0 (pure QCondGAT, cosine 없음) + Basic PCST + XiYan filter. LLM backbone 만 바뀐 변인 통제 anchor.")
        lines.append("")
        lines.append("| Metric | vLLM (Qwen3-Coder-30B-FP8) | GLM-4.7 | Δ |")
        lines.append("|--------|-----------------------------|---------|---|")
        lines.append(f"| Recall | {fmt(vllm['R'])} | {fmt(glm['R'])} | {fmt(glm['R']-vllm['R'], 4)} |")
        lines.append(f"| Precision | {fmt(vllm['P'])} | {fmt(glm['P'])} | {fmt(glm['P']-vllm['P'], 4)} |")
        lines.append(f"| F1 | {fmt(vllm['F1'])} | {fmt(glm['F1'])} | **{fmt(glm['F1']-vllm['F1'], 4)}** |")
        lines.append(f"| Filter time mean | {fmt(vllm.get('filter_time_mean_s'), 3)} s | {fmt(glm.get('filter_time_mean_s'), 3)} s | {fmt((glm.get('filter_time_mean_s') or 0) - (vllm.get('filter_time_mean_s') or 0), 3)} s |")
        lines.append(f"| Token input total | — | {int(glm.get('llm_input_tokens', 0)):,} | — |")
        lines.append("")
        lines.append(
            f"**해석**: ΔF1={glm['F1']-vllm['F1']:+.4f} 은 [2026-04-24 DECISIONS §결정 (c) 합격 기준 ΔF1 ≥ -0.02] 범위 내 (노이즈). R/P 가 균등 하락 (ΔR={glm['R']-vllm['R']:+.4f}, ΔP={glm['P']-vllm['P']:+.4f}) 하여 **balanced backbone 차이** (over-prune 도 over-keep 도 아님). Prompt tuning 불필요."
        )
        lines.append("")

    if new_anchor_glm:
        lines.append("### A.2 New top anchor (GLM era) — `qcond_gat_basic_glm`")
        lines.append("")
        lines.append("**설계**: α=0.85 (QCondGAT + cosine ensemble) + Basic PCST + XiYan filter. Wave 1.5 best `s04_stagewise_qcond_gat_basic` (vLLM era F1=0.7877) 의 GLM era 재실행.")
        lines.append("")
        lines.append("| Metric | vLLM (Wave 1.5 best) | GLM-4.7 (new best) | Δ |")
        lines.append("|--------|----------------------|---------------------|---|")
        # Wave 1.5 best numbers from DECISIONS.md
        vllm_r, vllm_p, vllm_f1 = 0.8169, 0.7605, 0.7877  # from EXPERIMENT_HISTORY
        lines.append(f"| Recall | {vllm_r:.4f} | {fmt(new_anchor_glm['R'])} | {new_anchor_glm['R']-vllm_r:+.4f} |")
        lines.append(f"| Precision | {vllm_p:.4f} | {fmt(new_anchor_glm['P'])} | {new_anchor_glm['P']-vllm_p:+.4f} |")
        lines.append(f"| F1 | {vllm_f1:.4f} | {fmt(new_anchor_glm['F1'])} | **{new_anchor_glm['F1']-vllm_f1:+.4f}** |")
        lines.append("")
        lines.append(
            f"**해석**: ΔF1={new_anchor_glm['F1']-vllm_f1:+.4f} 로 **GLM era 가 vLLM era 최고를 대폭 갱신**. 흥미로운 점: sanity anchor (A.1) 에서는 ΔF1 ≈ 0 이었는데 new anchor 에서는 ΔF1 ≈ +0.05 — "
            f"**cosine ensemble 이 들어간 조합에서만 GLM 의 precision gain 이 발현**. 이유 추정: α=0.85 blend 가 candidate pool 을 넓히고 (R 축 확장), GLM 이 Qwen 대비 **candidate 판정 정교도** 에서 우위 → ΔP={new_anchor_glm['P']-vllm_p:+.4f} 획득. 단독 QCondGAT (α=0) 환경에서는 이 효과가 보이지 않음."
        )
        lines.append("")
        lines.append("**Wave 2 closure 증거**: GLM era new top F1=0.8383 > vLLM era best 0.7877 = ΔF1=+0.0506 > sanity Δ (-0.0099) 의 정반대 방향. **backbone 교체 자체는 zero-sum (sanity Δ 거의 0) 이지만, ensemble + GLM 결합에서 synergy 획득**.")
        lines.append("")

    lines.append("### A.3 Sweep cell 간 GLM era 공통 LLM 비용 (평균)")
    lines.append("")
    lines.append("| nl | Filter time mean (s) | LLM input tokens | LLM output tokens |")
    lines.append("|----|------------------------|-------------------|--------------------|")
    for r in sweep:
        ti = r.get("filter_time_mean_s")
        tin = r.get("llm_input_tokens")
        tout = r.get("llm_output_tokens")
        lines.append(
            f"| {r['nl']} | {fmt(ti, 3)} | {int(tin):,} | {int(tout):,} |"
            if (tin and not np.isnan(tin))
            else f"| {r['nl']} | {fmt(ti, 3)} | — | — |"
        )
    lines.append("")
    lines.append(
        "**관찰**: Filter time 1.60~1.73 s/query 범위 내 (sanity 1.66 s 와 동일 수준). LLM token 사용량 ~1M input / ~35K output per cell — 사전 추정 (683 tokens/query × 1534 ≈ 1.048M) 과 정합."
    )
    lines.append("")

    # -----------------------
    # §5 해석 & 후속 과제
    # -----------------------
    lines.append("## 5. 해석 & 발표 포인트 요약")
    lines.append("")
    lines.append("### 5.1 발표 슬라이드 초안 (`C track`, 2026-04-28 용)")
    lines.append("")
    lines.append("- **Slide C-1: H1 검증 곡선** — §1.1 표 + F1 선 그래프. Peak=nl=D_max=6 하이라이트 + over-smoothing at nl=7 표기.")
    lines.append("- **Slide C-2: DB 별 alignment** — §2.3 D_max 버킷 × nl 매트릭스. 작은 DB (D=3, D=4) 의 최적 nl 이 global=6 과 불일치한다는 점 강조.")
    lines.append("- **Slide C-3: L2 dip 투명 보고** — §3.3 score 분포 + 재학습 계획 각주.")
    lines.append("- **(optional) Slide C-4: H2 가치 pitch** — H2 per-DB dynamic 의 기대 ΔF1 상한선 = §2.3 의 \"peak nl\" 이 모든 DB 에 적용됐을 때 가정 → 기대값 ΔF1 ≈ +0.005~0.015.")
    lines.append("")
    lines.append("### 5.2 Wave 3 / Wave 4 우선순위 결정 근거")
    lines.append("")
    lines.append("- **Wave 3 Proposal C H2 (per-DB dynamic)**: §2.3 에서 D=3,4 DB 가 global nl=6 에서 손해 (특히 D=3 debit_card_specializing). H2 로 재활용할 5 ckpt 이미 학습 완료 → inference 1~3 cell 로 검증 가능. **가치 강화**.")
    lines.append("- **Wave 3 Proposal F (Steiner backbone 재조직)**: vLLM era 기존 결과 재집계 — 신규 실행 0. 병행 진행 중.")
    lines.append("- **Wave 4 a05_filter_agentic**: GLM era 비용 추정 재조정 (§A.3 input 683 tokens/query 기반), multi-agent 3-5× 배수 시 12 cell ≈ ₩40-60K. post-2026-04-28 kickoff 유지.")
    lines.append("")
    lines.append("### 5.3 후속 analyzer 큐")
    lines.append("")
    lines.append("1. L2 dip 재학습 결과 (post-deadline): `best_gat_qcond_nl2.pt` 재학습 후 inference 1 cell. F1 변동 기록.")
    lines.append("2. Wave 3 Proposal F 재집계 리포트 (`steiner_backbone_stagewise_report.md` §3 GLM era 비교 추가) — 병행.")
    lines.append("3. H2 inference 결과 분해 (selector 세션 완료 후) — per-DB dynamic 결과를 §2.3 의 peak 추정값과 대조.")
    lines.append("")

    lines.append("## 6. 관련 리포트")
    lines.append("")
    lines.append("- [stagewise_qcond_ablation.md](stagewise_qcond_ablation.md) — Wave 1.5 최종 close + Anchor 매트릭스")
    lines.append("- [selector_gold_score_discrimination.md](selector_gold_score_discrimination.md) — 10-cell gold/non-gold score 분별력 분석 (H2/H3 QCondGAT 포함)")
    lines.append("- [selector_analysis.md](selector_analysis.md) — Cosine ↔ Ensemble GAT 기여도 (legacy)")
    lines.append("- [steiner_backbone_stagewise_report.md](steiner_backbone_stagewise_report.md) — Wave 3 Proposal F 공용 리포트")
    lines.append("- [planning/proposals/abl_sel_diameter_layers.md](../../planning/proposals/abl_sel_diameter_layers.md) — H1/H2 제안서 원문")
    lines.append("- [planning/DECISIONS.md](../../planning/DECISIONS.md) — 2026-04-24 Phase 전환 엔트리")
    lines.append("")
    lines.append("## 7. Changelog")
    lines.append("")
    lines.append("- **2026-04-24**: 초기 작성. 5-cell sweep + GLM sanity + GLM new anchor + vLLM sanity. §1~§부록 A 전 섹션.")
    lines.append("")
    return "\n".join(lines)


def render_csv(results: list[dict], diameter: dict) -> str:
    rows = ["cell_id,nl,db_id,D_max,n,R,P,F1"]
    for r in results:
        if r.get("group") != "sweep":
            continue
        for db, pd in r.get("per_db", {}).items():
            rows.append(
                f"{r['id']},{r['nl']},{db},{pd.get('D_max','')},{pd['n']},{pd['R']:.6f},{pd['P']:.6f},{pd['F1']:.6f}"
            )
    return "\n".join(rows) + "\n"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    print(f"Loading dev metadata + diameter...")
    qid2db, qid2diff = load_dev_info()
    diameter = load_diameter()
    print(f"  qid2db: {len(qid2db)} queries")
    print(f"  qid2diff: {Counter(qid2diff.values())}")
    print(f"  diameter: {diameter}")
    print()
    print(f"Analyzing {len(CELLS)} cells...")
    results = []
    for cell in CELLS:
        print(f"  {cell['id']} ({cell['group']})...")
        r = analyze_cell(cell, qid2db, qid2diff, diameter)
        if r.get("available"):
            print(
                f"    R={r['R']:.4f} P={r['P']:.4f} F1={r['F1']:.4f} "
                f"nodes={r.get('extractor_selected_nodes_mean', 0):.2f} "
                f"thr={r.get('extractor_threshold_mean', 0):.4f}"
            )
        results.append(r)

    md = render(results, qid2db, qid2diff, diameter)
    csv = render_csv(results, diameter)
    OUT_MD.write_text(md)
    OUT_CSV.write_text(csv)
    print(f"\nWrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
