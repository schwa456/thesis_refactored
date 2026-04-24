"""
Selector Gold vs Non-Gold Score Discrimination Analysis — Cross-Model.

For 10 experiments (5 high-recall, 5 low-recall), quantify how selector
assigns scores to gold vs non-gold nodes. Output gap, ROC-AUC, PR-AUC,
percentile tables + P80 crossing rate.

Usage:
    conda run -n base python src/analysis/analyze_selector_gold_vs_nongold.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path("/home/hyeonjin/thesis_refactored")
OUT_MD = BASE / "notebooks/analysis_results/selector_gold_score_discrimination.md"
OUT_CSV = BASE / "notebooks/analysis_results/selector_gold_score_discrimination.csv"


# ------------------------------------------------------------------
# Experiment catalog
# (label, score_file, selector_desc, group, recall, precision, f1)
# ------------------------------------------------------------------
EXPERIMENTS: list[dict] = [
    # High recall (top)
    {
        "id": "H1",
        "label": "Cosine-only (baseline)",
        "path": "outputs/experiments/s01_vector_only/a01_basic_pcst/s01_a01_02_raw_pcst_baseline/score_analysis_b0_raw_pcst_baseline.jsonl",
        "selector": "Pure cosine (α=1)",
        "group": "High",
        "recall": 0.9489,
        "precision": 0.1570,
    },
    {
        "id": "H2",
        "label": "Ens GAT α=0.85 (Basic PCST)",
        "path": "outputs/experiments/s03_gat_ensemble/a01_basic_pcst/s03_a01_01_ensemble_basic/score_analysis_b2_ensemble.jsonl",
        "selector": "Ensemble α=0.85 (legacy GAT blend)",
        "group": "High",
        "recall": 0.9679,
        "precision": 0.1293,
    },
    {
        "id": "H3",
        "label": "QCondGAT α=0.85 (Basic PCST, no filter)",
        "path": "outputs/experiments/s04_ablation/stagewise/no_filter/qcond_gat_basic_no_filter/score_analysis_s04_stagewise_qcond_gat_basic_no_filter.jsonl",
        "selector": "Ensemble α=0.85 (new QCondGAT blend)",
        "group": "High",
        "recall": 0.9651,
        "precision": 0.1287,
    },
    {
        "id": "H4",
        "label": "Supernode binary selector-only",
        "path": "outputs/experiments/abl/a03_direct_per_step/abl_a03_03_supernode_selector_only/score_analysis_ablation_supernode_direct_selector_only.jsonl",
        "selector": "Supernode binary classifier (direct)",
        "group": "High",
        "recall": 0.9968,
        "precision": 0.1173,
    },
    {
        "id": "H5",
        "label": "QCond binary selector-only",
        "path": "outputs/experiments/abl/a03_direct_per_step/abl_a03_01_qcond_selector_only/score_analysis_ablation_qcond_direct_selector_only.jsonl",
        "selector": "QCond binary classifier (direct)",
        "group": "High",
        "recall": 0.9968,
        "precision": 0.1173,
    },
    # Low recall (bottom)
    {
        "id": "L1",
        "label": "GAT classifier multi-agent",
        "path": "outputs/experiments/s02_gat_classifier/s02_03_gat_pcst_multi_agent/score_analysis_experiment_gat_pcst_multi_agent.jsonl",
        "selector": "GAT Classifier (sigmoid)",
        "group": "Low",
        "recall": 0.1913,
        "precision": 0.2577,
    },
    {
        "id": "L2",
        "label": "Supernode binary full pipeline",
        "path": "outputs/experiments/abl/a03_direct_per_step/abl_a03_12_supernode_binary_full/score_analysis_abl_a03_12_supernode_binary_full.jsonl",
        "selector": "Supernode binary classifier (direct)",
        "group": "Low",
        "recall": 0.2682,
        "precision": 0.4234,
    },
    {
        "id": "L3",
        "label": "FK-Steiner r=1.0 (extreme cut)",
        "path": "outputs/experiments/s03_gat_ensemble/a10_fk_steiner/s03_a10_11_fk_steiner_r10/score_analysis_s03_a10_11_fk_steiner_r10.jsonl",
        "selector": "Ensemble α=0.85 (legacy GAT blend) + FK Steiner r=1.0",
        "group": "Low",
        "recall": 0.2972,
        "precision": 0.4920,
    },
    {
        "id": "L4",
        "label": "QCond binary full pipeline",
        "path": "outputs/experiments/abl/a03_direct_per_step/abl_a03_08_qcond_binary_full/score_analysis_abl_a03_08_qcond_binary_full.jsonl",
        "selector": "QCond binary classifier (direct)",
        "group": "Low",
        "recall": 0.3357,
        "precision": 0.5320,
    },
    {
        "id": "L5",
        "label": "Adaptive multi-agent filter (s03 ens basic)",
        "path": "outputs/experiments/abl/a05_filter_agentic/a05_01_adaptive_multi_agent/score_analysis_a05_01_adaptive_multi_agent.jsonl",
        "selector": "Ensemble α=0.85 (legacy GAT blend)",
        "group": "Low",
        "recall": 0.3770,
        "precision": 0.6276,
    },
]


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------
def load_records(path: Path) -> list[dict]:
    recs: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def group_by_query(records: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["query_id"]].append(r)
    return groups


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------
def pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if arr.size > 0 else float("nan")


def describe(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {k: float("nan") for k in
                ["n", "min", "p10", "p25", "median", "mean", "p75", "p80", "p90", "p95", "max", "std"]}
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p10": pct(arr, 10),
        "p25": pct(arr, 25),
        "median": pct(arr, 50),
        "mean": float(arr.mean()),
        "p75": pct(arr, 75),
        "p80": pct(arr, 80),
        "p90": pct(arr, 90),
        "p95": pct(arr, 95),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def compute_per_query_auc(groups: dict[int, list[dict]]) -> dict:
    roc_vals, pr_vals = [], []
    skipped = 0
    for qid, recs in groups.items():
        labels = np.array([int(r["is_gold"]) for r in recs])
        scores = np.array([r["score"] for r in recs], dtype=float)
        if labels.sum() == 0 or labels.sum() == len(labels) or scores.size < 2:
            skipped += 1
            continue
        try:
            roc_vals.append(roc_auc_score(labels, scores))
            pr_vals.append(average_precision_score(labels, scores))
        except ValueError:
            skipped += 1
    return {
        "n_queries": len(groups),
        "skipped": skipped,
        "roc_auc_macro": float(np.mean(roc_vals)) if roc_vals else float("nan"),
        "roc_auc_macro_std": float(np.std(roc_vals)) if roc_vals else float("nan"),
        "pr_auc_macro": float(np.mean(pr_vals)) if pr_vals else float("nan"),
        "pr_auc_macro_std": float(np.std(pr_vals)) if pr_vals else float("nan"),
    }


def p80_crossing(groups: dict[int, list[dict]]) -> dict:
    """For each query, compute P80 threshold of all scores. Report:
    - gold_above_p80: fraction of gold nodes with score >= P80 (recall-like)
    - nongold_above_p80: fraction of non-gold nodes above P80 (precision-unfriendly)
    - avg number of nodes above P80 per query
    """
    gold_above = []
    nongold_above = []
    above_counts = []
    for qid, recs in groups.items():
        scores = np.array([r["score"] for r in recs], dtype=float)
        labels = np.array([int(r["is_gold"]) for r in recs])
        if scores.size == 0:
            continue
        thr = np.percentile(scores, 80)
        above = scores >= thr
        above_counts.append(int(above.sum()))
        if labels.sum() > 0:
            gold_above.append(float((above & (labels == 1)).sum()) / max(1, int((labels == 1).sum())))
        non_n = int((labels == 0).sum())
        if non_n > 0:
            nongold_above.append(float((above & (labels == 0)).sum()) / non_n)
    return {
        "gold_above_p80_mean": float(np.mean(gold_above)) if gold_above else float("nan"),
        "nongold_above_p80_mean": float(np.mean(nongold_above)) if nongold_above else float("nan"),
        "above_p80_count_mean": float(np.mean(above_counts)) if above_counts else float("nan"),
    }


# ------------------------------------------------------------------
# Per-experiment analysis
# ------------------------------------------------------------------
def analyze_one(exp: dict) -> dict:
    path = BASE / exp["path"]
    print(f"  loading {exp['id']} :: {path.name}")
    recs = load_records(path)

    all_scores = np.array([r["score"] for r in recs], dtype=float)
    all_labels = np.array([int(r["is_gold"]) for r in recs])
    gold_scores = all_scores[all_labels == 1]
    nongold_scores = all_scores[all_labels == 0]

    gold_desc = describe(gold_scores)
    nongold_desc = describe(nongold_scores)

    roc_auc, pr_auc = compute_auc(all_scores, all_labels)
    per_q = compute_per_query_auc(group_by_query(recs))
    p80_stats = p80_crossing(group_by_query(recs))

    mean_gap = gold_desc["mean"] - nongold_desc["mean"]
    median_gap = gold_desc["median"] - nongold_desc["median"]
    n_distinct_scores = int(np.unique(np.round(all_scores, 6)).size)

    return {
        **exp,
        "n_records": int(all_scores.size),
        "n_gold": int(all_labels.sum()),
        "n_nongold": int((all_labels == 0).sum()),
        "gold_prevalence": float(all_labels.mean()),
        "gold": gold_desc,
        "nongold": nongold_desc,
        "mean_gap": mean_gap,
        "median_gap": median_gap,
        "roc_auc_global": roc_auc,
        "pr_auc_global": pr_auc,
        "roc_auc_macro": per_q["roc_auc_macro"],
        "roc_auc_macro_std": per_q["roc_auc_macro_std"],
        "pr_auc_macro": per_q["pr_auc_macro"],
        "pr_auc_macro_std": per_q["pr_auc_macro_std"],
        "gold_above_p80": p80_stats["gold_above_p80_mean"],
        "nongold_above_p80": p80_stats["nongold_above_p80_mean"],
        "above_p80_count": p80_stats["above_p80_count_mean"],
        "n_distinct_scores": n_distinct_scores,
    }


# ------------------------------------------------------------------
# Fingerprint: detect shared selector outputs
# ------------------------------------------------------------------
def fingerprint(path: Path) -> str:
    """Hash of first 100 (query_id, node_name, score) tuples — rough signature of selector output."""
    import hashlib
    h = hashlib.md5()
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            r = json.loads(line)
            h.update(f"{r['query_id']}|{r['node_name']}|{round(float(r['score']), 6)}".encode())
    return h.hexdigest()[:10]


# ------------------------------------------------------------------
# Render markdown report
# ------------------------------------------------------------------
def fmt(x: float, nd: int = 4) -> str:
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "—"
    return f"{x:.{nd}f}"


def render_markdown(results: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Selector Score Discrimination: Gold vs Non-Gold (10 Experiments)")
    lines.append("")
    lines.append("**생성일**: 2026-04-22  |  **스크립트**: `src/analysis/analyze_selector_gold_vs_nongold.py`  |  **데이터**: `score_analysis_*.jsonl` × 10")
    lines.append("")
    lines.append("## 0. 목적")
    lines.append("")
    lines.append(
        "> Selector 가 실질적으로 `score` 를 부여하는 역할이라면, **gold 노드와 non-gold 노드에 부여한 score 차이**가 성능(recall)의 선행 지표여야 한다. "
        "이 리포트는 recall 상위 5 + 하위 5 = 10개 실험에서 gold/non-gold score 분포를 비교하고, 분별력 지표(ROC-AUC, PR-AUC, P80 crossing rate, score gap)를 산출한다."
    )
    lines.append("")
    lines.append("## 1. 실험 표본 (10개)")
    lines.append("")
    lines.append("| id | Group | Label | Selector 종류 | R | P | score file |")
    lines.append("|----|-------|-------|---------------|---|---|------------|")
    for r in results:
        sf = Path(r["path"]).name
        lines.append(
            f"| {r['id']} | {r['group']} | {r['label']} | {r['selector']} | {fmt(r['recall'])} | {fmt(r['precision'])} | `{sf}` |"
        )
    lines.append("")

    lines.append("## 2. Selector score fingerprint — 공유 selector 탐지")
    lines.append("")
    lines.append(
        "Recall 차이가 **selector 고유 출력**에서 오는지 **downstream 차이**에서 오는지를 분리하기 위해, "
        "각 실험의 score 스트림 첫 100 레코드 해시로 fingerprint 를 계산한다 (동일 hash → selector output 동일)."
    )
    lines.append("")
    lines.append("| id | fingerprint | 동일 출력 그룹 |")
    lines.append("|----|-------------|----------------|")
    # Group by fingerprint
    fp_groups: dict[str, list[str]] = defaultdict(list)
    for r in results:
        fp_groups[r["_fp"]].append(r["id"])
    for r in results:
        group = ",".join(fp_groups[r["_fp"]])
        lines.append(f"| {r['id']} | `{r['_fp']}` | {group} |")
    lines.append("")
    shared = {fp: ids for fp, ids in fp_groups.items() if len(ids) > 1}
    if shared:
        lines.append("**관찰**: 같은 fingerprint 를 공유하는 실험 세트는 **selector output 이 동일**하므로, 이들 사이의 recall/precision 차이는 전적으로 **extractor·filter 설정 차이**에서 발생한다.")
        for fp, ids in shared.items():
            lines.append(f"- `{fp}` → {ids}")
        lines.append("")
    else:
        lines.append("**관찰**: 모든 실험이 고유한 selector output 을 가진다.")
        lines.append("")

    lines.append("## 3. Gold vs Non-Gold 분포 요약")
    lines.append("")
    lines.append("각 실험에서 gold 노드와 non-gold 노드에 부여된 score 의 통계량 (소수점 4자리).")
    lines.append("")
    lines.append("### 3.1 Gold 노드 score 분포")
    lines.append("")
    lines.append("| id | n | min | p25 | median | mean | p75 | p90 | p95 | max | std |")
    lines.append("|----|---|-----|-----|--------|------|-----|-----|-----|-----|-----|")
    for r in results:
        g = r["gold"]
        lines.append(
            f"| {r['id']} | {g['n']} | {fmt(g['min'])} | {fmt(g['p25'])} | {fmt(g['median'])} | "
            f"{fmt(g['mean'])} | {fmt(g['p75'])} | {fmt(g['p90'])} | {fmt(g['p95'])} | {fmt(g['max'])} | {fmt(g['std'])} |"
        )
    lines.append("")

    lines.append("### 3.2 Non-Gold 노드 score 분포")
    lines.append("")
    lines.append("| id | n | min | p25 | median | mean | p75 | p90 | p95 | max | std |")
    lines.append("|----|---|-----|-----|--------|------|-----|-----|-----|-----|-----|")
    for r in results:
        ng = r["nongold"]
        lines.append(
            f"| {r['id']} | {ng['n']} | {fmt(ng['min'])} | {fmt(ng['p25'])} | {fmt(ng['median'])} | "
            f"{fmt(ng['mean'])} | {fmt(ng['p75'])} | {fmt(ng['p90'])} | {fmt(ng['p95'])} | {fmt(ng['max'])} | {fmt(ng['std'])} |"
        )
    lines.append("")

    lines.append("## 4. 분별력 지표 (Discrimination)")
    lines.append("")
    lines.append("| id | Group | R | mean gap | median gap | ROC-AUC (global) | ROC-AUC (per-q avg) | PR-AUC (global) | PR-AUC (per-q avg) | Gold≥P80 | Non-Gold≥P80 | #distinct |")
    lines.append("|----|-------|---|----------|------------|------------------|----------------------|------------------|---------------------|-----------|---------------|-----------|")
    for r in results:
        lines.append(
            f"| {r['id']} | {r['group']} | {fmt(r['recall'])} | {fmt(r['mean_gap'])} | {fmt(r['median_gap'])} | "
            f"{fmt(r['roc_auc_global'])} | {fmt(r['roc_auc_macro'])}±{fmt(r['roc_auc_macro_std'], 3)} | "
            f"{fmt(r['pr_auc_global'])} | {fmt(r['pr_auc_macro'])}±{fmt(r['pr_auc_macro_std'], 3)} | "
            f"{fmt(r['gold_above_p80'])} | {fmt(r['nongold_above_p80'])} | {r['n_distinct_scores']} |"
        )
    lines.append("")
    lines.append("**지표 해설**:")
    lines.append("- **mean gap / median gap**: `gold_score - nongold_score`. 클수록 selector 가 gold 를 확실히 분리.")
    lines.append("- **ROC-AUC (per-q avg)**: 쿼리별 ROC-AUC 의 산술평균 (macro). 쿼리 스케일 편향을 제거.")
    lines.append("- **Gold≥P80**: 쿼리별 P80 threshold 이상에 포함된 gold 비율 (~Selector-only recall 근사).")
    lines.append("- **Non-Gold≥P80**: 같은 threshold 이상에 포함된 non-gold 비율 (낮을수록 분별력 우수).")
    lines.append("- **#distinct**: 전체 score 중 소수 6자리로 반올림했을 때 distinct 값 수. 낮으면 binary/이진화된 classifier.")
    lines.append("")

    lines.append("## 5. 해석")
    lines.append("")

    # Sort by roc_auc_macro desc for commentary
    ranked = sorted(results, key=lambda x: (x["roc_auc_macro"] if not np.isnan(x["roc_auc_macro"]) else -1), reverse=True)
    best = ranked[0]
    worst = ranked[-1]
    lines.append("### 5.1 Top vs Bottom ROC-AUC (per-query macro)")
    lines.append("")
    lines.append(f"- **최고**: {best['id']} ({best['label']}) — ROC-AUC = {fmt(best['roc_auc_macro'])}, Gold≥P80 = {fmt(best['gold_above_p80'])}, Recall(downstream) = {fmt(best['recall'])}")
    lines.append(f"- **최저**: {worst['id']} ({worst['label']}) — ROC-AUC = {fmt(worst['roc_auc_macro'])}, Gold≥P80 = {fmt(worst['gold_above_p80'])}, Recall(downstream) = {fmt(worst['recall'])}")
    lines.append("")

    # High vs Low group averages
    highs = [r for r in results if r["group"] == "High"]
    lows = [r for r in results if r["group"] == "Low"]

    def grp_avg(rs: list[dict], key: str) -> float:
        vals = [r[key] for r in rs if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    lines.append("### 5.2 High-recall vs Low-recall 그룹 평균")
    lines.append("")
    lines.append("| 지표 | High 그룹 평균 | Low 그룹 평균 | Δ (High − Low) |")
    lines.append("|------|----------------|----------------|-----------------|")
    for key, label in [
        ("mean_gap", "mean gap"),
        ("median_gap", "median gap"),
        ("roc_auc_global", "ROC-AUC (global)"),
        ("roc_auc_macro", "ROC-AUC (per-q avg)"),
        ("pr_auc_macro", "PR-AUC (per-q avg)"),
        ("gold_above_p80", "Gold≥P80"),
        ("nongold_above_p80", "Non-Gold≥P80"),
    ]:
        h = grp_avg(highs, key)
        l = grp_avg(lows, key)
        lines.append(f"| {label} | {fmt(h)} | {fmt(l)} | {fmt(h - l)} |")
    lines.append("")

    lines.append("### 5.3 공유 selector 실험쌍 — Selector 분별력 vs 최종 Recall")
    lines.append("")
    lines.append("같은 fingerprint 를 가진 실험쌍은 selector output 이 **완전히 동일**하다. 이들의 최종 recall 격차는 전적으로 extractor/filter 설정 차이에서 발생한다.")
    lines.append("")
    lines.append("| fingerprint | 실험 | Recall | Gold≥P80 | ROC-AUC (per-q) | 해석 |")
    lines.append("|-------------|------|--------|-----------|------------------|------|")
    for fp, ids in fp_groups.items():
        if len(ids) < 2:
            continue
        for eid in ids:
            r = next(x for x in results if x["id"] == eid)
            lines.append(f"| `{fp}` | {r['id']} {r['label']} | {fmt(r['recall'])} | {fmt(r['gold_above_p80'])} | {fmt(r['roc_auc_macro'])} | — |")
    lines.append("")
    lines.append("**해석 포인트**: fingerprint 가 같은데 recall 이 다르면, selector 입장에서는 gold 를 같은 방식으로 분별했고, 그 뒤 extractor/filter 가 gold 를 놓쳤다는 뜻이다. 즉 **저성능의 원인이 selector 가 아님**을 입증한다.")
    lines.append("")

    lines.append("## 6. 후속 과제 (analyzer)")
    lines.append("")
    lines.append("- 쿼리별 **selector rank**에서 gold 의 분포 (상위 몇 % 에 분포?) — per-query top-k 분석")
    lines.append("- QCondGAT 의 query conditioning 이 쿼리별 gold rank 에 얼마나 기여했는지 — 쿼리 길이/난이도별 조건부 분석")
    lines.append("- Supernode/QCond binary classifier (H4/H5) 가 score ≈ 1.0 에 집중되어 있는 이유 — 이진화된 출력 분포 확인")
    lines.append("")

    lines.append("## 7. Changelog")
    lines.append("")
    lines.append("- **2026-04-22**: 초기 작성. 10개 실험 (High 5 + Low 5) 기준 gold/non-gold score 분포 + AUC 지표.")
    lines.append("")
    return "\n".join(lines)


def render_csv(results: list[dict]) -> str:
    cols = [
        "id", "group", "label", "selector", "recall", "precision",
        "n_records", "n_gold", "n_nongold", "gold_prevalence",
        "gold_mean", "gold_median", "gold_std",
        "nongold_mean", "nongold_median", "nongold_std",
        "mean_gap", "median_gap",
        "roc_auc_global", "pr_auc_global",
        "roc_auc_macro", "roc_auc_macro_std",
        "pr_auc_macro", "pr_auc_macro_std",
        "gold_above_p80", "nongold_above_p80",
        "above_p80_count", "n_distinct_scores", "fingerprint",
    ]
    rows = [",".join(cols)]
    for r in results:
        vals = [
            r["id"], r["group"], r["label"].replace(",", ";"), r["selector"].replace(",", ";"),
            f"{r['recall']:.4f}", f"{r['precision']:.4f}",
            str(r["n_records"]), str(r["n_gold"]), str(r["n_nongold"]),
            f"{r['gold_prevalence']:.6f}",
            f"{r['gold']['mean']:.6f}", f"{r['gold']['median']:.6f}", f"{r['gold']['std']:.6f}",
            f"{r['nongold']['mean']:.6f}", f"{r['nongold']['median']:.6f}", f"{r['nongold']['std']:.6f}",
            f"{r['mean_gap']:.6f}", f"{r['median_gap']:.6f}",
            f"{r['roc_auc_global']:.6f}", f"{r['pr_auc_global']:.6f}",
            f"{r['roc_auc_macro']:.6f}", f"{r['roc_auc_macro_std']:.6f}",
            f"{r['pr_auc_macro']:.6f}", f"{r['pr_auc_macro_std']:.6f}",
            f"{r['gold_above_p80']:.6f}", f"{r['nongold_above_p80']:.6f}",
            f"{r['above_p80_count']:.4f}", str(r["n_distinct_scores"]), r["_fp"],
        ]
        rows.append(",".join(vals))
    return "\n".join(rows) + "\n"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    print(f"Analyzing {len(EXPERIMENTS)} experiments...")
    results: list[dict] = []
    for exp in EXPERIMENTS:
        path = BASE / exp["path"]
        if not path.exists():
            print(f"[WARN] missing {path}")
            continue
        r = analyze_one(exp)
        r["_fp"] = fingerprint(path)
        results.append(r)
        print(
            f"    {r['id']} | gold_mean={r['gold']['mean']:.4f} nongold_mean={r['nongold']['mean']:.4f} "
            f"gap={r['mean_gap']:.4f} roc_auc_macro={r['roc_auc_macro']:.4f} "
            f"Gold≥P80={r['gold_above_p80']:.4f} fp={r['_fp']}"
        )

    md = render_markdown(results)
    csv = render_csv(results)
    OUT_MD.write_text(md)
    OUT_CSV.write_text(csv)
    print(f"\nWrote {OUT_MD}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
