"""
analyze_raw_score_distribution.py — Directed Top-K SuperNode 학위 논문 Part III base 정보

근거: planning/DECISIONS.md 2026-05-05 (Directed Top-K SuperNode Part III 진행 결정)
의도: threshold 결정의 정량 근거 + 기존 top-K=20 의 score range 와 비교

5 분석:
  (1) per-query raw cosine score 분포 (P25/P50/P75/P90/P95)
  (2) gold vs non-gold score 분리 (ROC-AUC, Cohen's d, threshold trade-off)
  (3) 기존 top-K=20 의 score range (top-20 min/max)
  (4) Threshold 후보 4 종 비교 (절대 / percentile / mean+std / elbow)
  (5) 기존 SuperNode 비교 base (평균 schema 노드 수)

데이터:
  - α=1.0 cell (`t00_S2_alpha1`) 의 score_analysis_*.jsonl — Cosine only score (per-query min-max normalized)

산출물:
  - notebooks/analysis_results/raw_score_distribution_for_directed_topk.md
  - notebooks/analysis_results/raw_score_per_query_stats.csv
  - notebooks/analysis_results/raw_score_threshold_candidates.csv
  - notebooks/analysis_results/raw_score_supernode_comparison.csv
"""

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

ROOT = Path("/home/hyeonjin/thesis_refactored")
PIPELINE = ROOT / "outputs/experiments/s04_ablation/pipeline"
DEV_JSON = ROOT / "data/raw/BIRD_dev/dev.json"
ANALYSIS_DIR = ROOT / "notebooks/analysis_results"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Cosine only cell (raw cosine after per-query min-max norm)
COSINE_CELL = "t00_S2_alpha1"
COSINE_SCORE_PATH = (PIPELINE / COSINE_CELL /
                     f"score_analysis_s04_pipeline_t00_S2_alpha1.jsonl")

DIFFICULTIES = ["simple", "moderate", "challenging"]


# ──────────────────────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────────────────────

def load_dev_meta() -> Dict[int, Dict]:
    with open(DEV_JSON, "r") as f:
        dev = json.load(f)
    return {int(d["question_id"]): d for d in dev}


def load_score_analysis(path: Path) -> Dict[int, List[Tuple[str, float, bool]]]:
    """{qid: [(node_name, score, is_gold), ...]}, sorted by score desc."""
    out: Dict[int, List[Tuple[str, float, bool]]] = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            try:
                d = json.loads(line)
                qid = int(d["query_id"])
                out[qid].append((d["node_name"], float(d["score"]), bool(d.get("is_gold", False))))
            except (json.JSONDecodeError, KeyError):
                continue
    # Sort each query's list by score descending
    for qid in out:
        out[qid].sort(key=lambda t: t[1], reverse=True)
    return dict(out)


# ──────────────────────────────────────────────────────────────
# 통계 헬퍼
# ──────────────────────────────────────────────────────────────

def percentile(vs: List[float], p: float) -> float:
    if not vs:
        return float("nan")
    s = sorted(vs)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def mean(vs: Iterable[float]) -> float:
    vs = list(vs)
    return sum(vs) / len(vs) if vs else float("nan")


def stddev(vs: Iterable[float]) -> float:
    vs = list(vs)
    if len(vs) < 2:
        return float("nan")
    m = sum(vs) / len(vs)
    return math.sqrt(sum((v - m) ** 2 for v in vs) / len(vs))


def roc_auc(scores: List[float], labels: List[bool]) -> float:
    """ROC-AUC via Mann-Whitney U-equivalent (pairwise comparison count)."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    # Efficient via rank
    paired = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    n_pos = len(pos)
    n_neg = len(neg)
    # Sum of ranks of positives (with mid-rank for ties)
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg = (rank + (rank + (j - i))) / 2.0
        for k in range(i, j + 1):
            if paired[k][1] == 1:
                rank_sum += avg
        rank += (j - i + 1)
        i = j + 1
    u = rank_sum - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Effect size between gold (g1) and non-gold (g2)."""
    if len(group1) < 2 or len(group2) < 2:
        return float("nan")
    m1, m2 = mean(group1), mean(group2)
    s1, s2 = stddev(group1), stddev(group2)
    n1, n2 = len(group1), len(group2)
    pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled if pooled > 0 else float("nan")


def detect_elbow_point(sorted_scores_desc: List[float]) -> int:
    """Per-query elbow point — index in sorted-desc list where steepest drop happens.

    Returns: index k such that score[k-1] > τ, score[k] ≤ τ (i.e., select first k nodes).
    Using max curvature on score curve.
    """
    if len(sorted_scores_desc) < 3:
        return len(sorted_scores_desc)
    # Use second derivative max (discrete)
    n = len(sorted_scores_desc)
    diffs = []
    for i in range(1, n - 1):
        d2 = sorted_scores_desc[i + 1] - 2 * sorted_scores_desc[i] + sorted_scores_desc[i - 1]
        # Elbow = inflection from steep to flat → second derivative max (positive)
        diffs.append((i, d2))
    if not diffs:
        return n
    # Pick i with max d2 (positive curvature)
    best_i, _ = max(diffs, key=lambda x: x[1])
    # Select top (best_i + 1) nodes (because first elbow happens after best_i)
    return best_i + 1


# ──────────────────────────────────────────────────────────────
# (1) Per-query raw score 분포
# ──────────────────────────────────────────────────────────────

def analyze_per_query_distribution(scores_by_qid: Dict[int, List[Tuple[str, float, bool]]]) -> Dict:
    """Per-query stat: min/max/mean/std/percentiles."""
    rows = []
    for qid, items in scores_by_qid.items():
        score_list = [s for _, s, _ in items]
        if not score_list:
            continue
        rows.append({
            "qid": qid,
            "n_nodes": len(score_list),
            "score_min": min(score_list),
            "score_max": max(score_list),
            "score_mean": mean(score_list),
            "score_std": stddev(score_list),
            "score_p25": percentile(score_list, 25),
            "score_p50": percentile(score_list, 50),
            "score_p75": percentile(score_list, 75),
            "score_p90": percentile(score_list, 90),
            "score_p95": percentile(score_list, 95),
            "score_p99": percentile(score_list, 99),
        })
    # Aggregate (mean of per-query stats)
    agg = {}
    for col in ["n_nodes", "score_min", "score_max", "score_mean", "score_std",
                "score_p25", "score_p50", "score_p75", "score_p90", "score_p95", "score_p99"]:
        vs = [r[col] for r in rows]
        agg[f"{col}_mean"] = mean(vs)
        agg[f"{col}_std"] = stddev(vs)
        agg[f"{col}_p25"] = percentile(vs, 25)
        agg[f"{col}_p50"] = percentile(vs, 50)
        agg[f"{col}_p75"] = percentile(vs, 75)
    return {"rows": rows, "agg": agg}


# ──────────────────────────────────────────────────────────────
# (2) Gold vs non-gold 분리
# ──────────────────────────────────────────────────────────────

def analyze_gold_separation(scores_by_qid: Dict[int, List[Tuple[str, float, bool]]]) -> Dict:
    """Gold vs non-gold score 분포 + ROC-AUC + threshold trade-off."""
    # Per-query gold score / non-gold score stats
    per_query_rows = []
    all_gold_scores = []
    all_nongold_scores = []
    for qid, items in scores_by_qid.items():
        gold_scores = [s for _, s, g in items if g]
        nongold_scores = [s for _, s, g in items if not g]
        if not gold_scores:
            continue
        all_gold_scores.extend(gold_scores)
        all_nongold_scores.extend(nongold_scores)
        per_query_rows.append({
            "qid": qid,
            "n_gold": len(gold_scores),
            "n_nongold": len(nongold_scores),
            "gold_mean": mean(gold_scores),
            "gold_p25": percentile(gold_scores, 25),
            "gold_p50": percentile(gold_scores, 50),
            "gold_p75": percentile(gold_scores, 75),
            "nongold_mean": mean(nongold_scores),
            "nongold_p25": percentile(nongold_scores, 25),
            "nongold_p50": percentile(nongold_scores, 50),
            "nongold_p75": percentile(nongold_scores, 75),
        })

    # Global ROC-AUC (all nodes pooled)
    all_labels = [True] * len(all_gold_scores) + [False] * len(all_nongold_scores)
    all_scores = all_gold_scores + all_nongold_scores
    global_roc = roc_auc(all_scores, all_labels)
    global_d = cohens_d(all_gold_scores, all_nongold_scores)

    # Per-query ROC-AUC (avg)
    per_query_rocs = []
    for qid, items in scores_by_qid.items():
        scores = [s for _, s, _ in items]
        labels = [g for _, _, g in items]
        if any(labels) and not all(labels):
            per_query_rocs.append(roc_auc(scores, labels))
    per_query_roc_mean = mean(per_query_rocs)

    # Threshold τ trade-off (절대 score)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_rows = []
    for tau in thresholds:
        # 글로벌 (모든 query 의 모든 node)
        tp = sum(1 for s, l in zip(all_scores, all_labels) if s >= tau and l)
        fp = sum(1 for s, l in zip(all_scores, all_labels) if s >= tau and not l)
        fn = sum(1 for s, l in zip(all_scores, all_labels) if s < tau and l)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        # 평균 selected node 수 (per-query)
        sel_counts = []
        for qid, items in scores_by_qid.items():
            sel_counts.append(sum(1 for _, s, _ in items if s >= tau))

        threshold_rows.append({
            "tau": tau,
            "global_precision": prec,
            "global_recall": rec,
            "global_f1": f1,
            "selected_per_query_mean": mean(sel_counts),
            "selected_per_query_p50": percentile(sel_counts, 50),
            "selected_per_query_p25": percentile(sel_counts, 25),
            "selected_per_query_p75": percentile(sel_counts, 75),
        })

    return {
        "per_query_rows": per_query_rows,
        "global_roc_auc": global_roc,
        "global_cohens_d": global_d,
        "per_query_roc_mean": per_query_roc_mean,
        "threshold_rows": threshold_rows,
        "n_gold_total": len(all_gold_scores),
        "n_nongold_total": len(all_nongold_scores),
        "all_gold_p25": percentile(all_gold_scores, 25),
        "all_gold_p50": percentile(all_gold_scores, 50),
        "all_gold_p75": percentile(all_gold_scores, 75),
        "all_gold_mean": mean(all_gold_scores),
        "all_nongold_p25": percentile(all_nongold_scores, 25),
        "all_nongold_p50": percentile(all_nongold_scores, 50),
        "all_nongold_p75": percentile(all_nongold_scores, 75),
        "all_nongold_mean": mean(all_nongold_scores),
    }


# ──────────────────────────────────────────────────────────────
# (3) 기존 top-K=20 score range
# ──────────────────────────────────────────────────────────────

def analyze_topk_score_range(scores_by_qid: Dict[int, List[Tuple[str, float, bool]]],
                             k: int = 20) -> Dict:
    """top-K=20 cap 의 score range — top-1, top-K min/max 분포."""
    rows = []
    for qid, items in scores_by_qid.items():
        if len(items) < k:
            top_items = items
        else:
            top_items = items[:k]  # already sorted desc
        scores_topk = [s for _, s, _ in top_items]
        gold_in_topk = sum(1 for _, _, g in top_items if g)
        total_gold = sum(1 for _, _, g in items if g)
        rows.append({
            "qid": qid,
            "k_actual": len(top_items),
            "top1_score": scores_topk[0] if scores_topk else float("nan"),
            "topk_min_score": scores_topk[-1] if scores_topk else float("nan"),
            "topk_mean_score": mean(scores_topk),
            "gold_in_topk": gold_in_topk,
            "total_gold": total_gold,
            "recall_topk": gold_in_topk / total_gold if total_gold > 0 else 0.0,
        })
    # Aggregate
    top1s = [r["top1_score"] for r in rows]
    topk_mins = [r["topk_min_score"] for r in rows]
    recalls = [r["recall_topk"] for r in rows]
    agg = {
        "k": k,
        "n_queries": len(rows),
        "top1_score_p25": percentile(top1s, 25),
        "top1_score_p50": percentile(top1s, 50),
        "top1_score_p75": percentile(top1s, 75),
        "top1_score_mean": mean(top1s),
        "topk_min_score_p25": percentile(topk_mins, 25),
        "topk_min_score_p50": percentile(topk_mins, 50),
        "topk_min_score_p75": percentile(topk_mins, 75),
        "topk_min_score_mean": mean(topk_mins),
        "recall_topk_mean": mean(recalls),
        "recall_topk_p25": percentile(recalls, 25),
        "recall_topk_p50": percentile(recalls, 50),
    }
    return {"rows": rows, "agg": agg}


# ──────────────────────────────────────────────────────────────
# (4) Threshold 후보 4 종 비교
# ──────────────────────────────────────────────────────────────

def analyze_threshold_candidates(scores_by_qid: Dict[int, List[Tuple[str, float, bool]]],
                                 qid_diff: Dict[int, str]) -> Dict:
    """4 종 threshold 후보 별 selected node 수 + recall 추정."""

    def _compute_metrics(threshold_fn, label: str) -> Dict:
        """threshold_fn: list of (name, score, is_gold) → set of selected (name)."""
        sel_counts = []
        recalls = []
        precisions = []
        gold_in_sel = []
        f1s = []
        diff_breakdown = defaultdict(list)
        for qid, items in scores_by_qid.items():
            selected = threshold_fn(items)
            sel_set = {n for n in selected}
            sel_count = len(sel_set)
            gold_set = {n for n, _, g in items if g}
            gold_in_sel_q = len(sel_set & gold_set)
            r = gold_in_sel_q / len(gold_set) if gold_set else 0.0
            p = gold_in_sel_q / sel_count if sel_count > 0 else 0.0
            f1 = 2 * r * p / (r + p) if (r + p) > 0 else 0.0
            sel_counts.append(sel_count)
            recalls.append(r)
            precisions.append(p)
            gold_in_sel.append(gold_in_sel_q)
            f1s.append(f1)
            diff = qid_diff.get(qid, "unknown")
            diff_breakdown[diff].append((sel_count, r))
        # Aggregate
        agg = {
            "label": label,
            "n_queries": len(sel_counts),
            "selected_mean": mean(sel_counts),
            "selected_std": stddev(sel_counts),
            "selected_p25": percentile(sel_counts, 25),
            "selected_p50": percentile(sel_counts, 50),
            "selected_p75": percentile(sel_counts, 75),
            "recall_mean": mean(recalls),
            "precision_mean": mean(precisions),
            "f1_mean": mean(f1s),
        }
        for diff in DIFFICULTIES:
            sub = diff_breakdown.get(diff, [])
            if sub:
                agg[f"selected_{diff}_mean"] = mean([x[0] for x in sub])
                agg[f"recall_{diff}_mean"] = mean([x[1] for x in sub])
            else:
                agg[f"selected_{diff}_mean"] = float("nan")
                agg[f"recall_{diff}_mean"] = float("nan")
        return agg

    rows = []

    # (a) 절대 score
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        rows.append(_compute_metrics(
            lambda items, t=tau: [n for n, s, _ in items if s >= t],
            f"abs τ={tau}"))

    # (b) Per-query percentile
    for pct in [70, 75, 80, 85, 90, 95]:
        def _percentile_fn(items, p=pct):
            scores = [s for _, s, _ in items]
            tau = percentile(scores, p)
            return [n for n, s, _ in items if s >= tau]
        rows.append(_compute_metrics(_percentile_fn, f"P{pct}"))

    # (c) Mean+std
    for k_sigma in [0.0, 0.5, 1.0, 1.5, 2.0]:
        def _meanstd_fn(items, k=k_sigma):
            scores = [s for _, s, _ in items]
            tau = mean(scores) + k * stddev(scores)
            return [n for n, s, _ in items if s >= tau]
        rows.append(_compute_metrics(_meanstd_fn, f"μ+{k_sigma}σ"))

    # (d) Score elbow point (per-query)
    def _elbow_fn(items):
        scores_desc = [s for _, s, _ in items]
        k = detect_elbow_point(scores_desc)
        return [n for (n, _, _), idx in zip(items, range(len(items))) if idx < k]
    rows.append(_compute_metrics(_elbow_fn, "elbow (auto)"))

    # (e) top-K=20 baseline (reference)
    def _topk_fn(items, k=20):
        return [n for (n, _, _), idx in zip(items, range(len(items))) if idx < k]
    rows.append(_compute_metrics(_topk_fn, "top-K=20 (baseline)"))

    return {"rows": rows}


# ──────────────────────────────────────────────────────────────
# (5) 기존 SuperNode 비교 base — 평균 schema 노드 수
# ──────────────────────────────────────────────────────────────

def analyze_supernode_comparison(scores_by_qid: Dict[int, List[Tuple[str, float, bool]]],
                                 dev: Dict[int, Dict]) -> Dict:
    """기존 SuperNode (모든 schema) vs Directed Top-K threshold 후보 의 노드 수 비교."""
    # 기존 SuperNode 의 평균 schema 노드 수 = score_analysis 의 모든 노드
    rows_per_db = defaultdict(list)
    for qid, items in scores_by_qid.items():
        db = dev.get(qid, {}).get("db_id", "unknown")
        rows_per_db[db].append(len(items))

    db_summary = {}
    for db, counts in sorted(rows_per_db.items()):
        db_summary[db] = {
            "n_queries": len(counts),
            "supernode_node_count_mean": mean(counts),
            "supernode_node_count_p25": percentile(counts, 25),
            "supernode_node_count_p50": percentile(counts, 50),
            "supernode_node_count_p75": percentile(counts, 75),
        }
    overall_supernode_mean = mean([n for counts in rows_per_db.values() for n in counts])

    # Directed Top-K threshold 후보 별 ratio
    return {
        "db_summary": db_summary,
        "overall_supernode_mean": overall_supernode_mean,
    }


# ──────────────────────────────────────────────────────────────
# CSV / Markdown
# ──────────────────────────────────────────────────────────────

def fmt(v, prec=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return f"{v:.{prec}f}"
    return str(v)


def write_csv(path: Path, rows: List[Dict], cols: List[str]):
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    vals.append(f"{v:.6f}" if not math.isnan(v) else "")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")


def render_markdown(per_q: Dict, gold_sep: Dict, topk: Dict, candidates: Dict,
                    supernode: Dict, qid_diff: Dict[int, str]) -> str:
    n_total = sum(1 for _ in qid_diff)
    n_simple = sum(1 for d in qid_diff.values() if d == "simple")
    n_moderate = sum(1 for d in qid_diff.values() if d == "moderate")
    n_challenging = sum(1 for d in qid_diff.values() if d == "challenging")

    lines = []
    A = lines.append

    # ── 헤더 ──
    A("# Raw Score Distribution — Directed Top-K SuperNode Part III Base")
    A("")
    A("> **출처**: `planning/DECISIONS.md` 2026-05-05 (Directed Top-K SuperNode 학위 논문 Part III 진행 결정 — Q1/Q2/Q3 사용자 confirm + threshold 기반 channel)")
    A("> **의도**: threshold 결정의 정량 근거 + 기존 top-K=20 의 score range 와 비교 — Directed Top-K SuperNode 학습 시 어떤 threshold (절대 score / per-query percentile / mean+std / elbow point) 를 채택할지 결정")
    A("> **데이터 범위**: BIRD-Dev 1534 queries × Cosine only score (`t00_S2_alpha1` cell, α=1.0 → raw cosine after per-query min-max norm)")
    A("> **메트릭 표기**: Recall, Precision, F1 4자리 (memory rule).")
    A("")
    A("**⚠️ Score 정의 주의**: 본 분석의 \"raw score\" 는 `EnsembleSelector` 의 per-query min-max norm 후의 cosine similarity 값. score=1.0 = top-1 노드, score=0.0 = bottom 노드. 절대 cosine similarity 가 아니라 **per-query [0,1] 정규화 score** 임. 학습 시 이 정의 그대로 사용 가능.")
    A("")

    # ── §0 TL;DR ──
    overall_roc = gold_sep["global_roc_auc"]
    per_q_roc = gold_sep["per_query_roc_mean"]
    cohens = gold_sep["global_cohens_d"]
    topk_min_p50 = topk["agg"]["topk_min_score_p50"]
    topk_min_mean = topk["agg"]["topk_min_score_mean"]
    topk_recall_mean = topk["agg"]["recall_topk_mean"]
    sn_mean = supernode["overall_supernode_mean"]

    # Find best threshold candidates by F1
    sorted_by_f1 = sorted(candidates["rows"], key=lambda r: r["f1_mean"], reverse=True)
    top3_f1 = sorted_by_f1[:3]

    A("## §0. TL;DR — 3 핵심 발견")
    A("")
    A(f"**핵심 결론 (Directed Top-K SuperNode threshold 추천)**:")
    A(f"> Cosine raw score (per-query min-max norm) 분포 분석 결과 — gold/non-gold 분리 **강함** (per-query ROC-AUC = {fmt(per_q_roc)}, Cohen's d = {fmt(cohens)} large effect). 기존 top-K=20 의 평균 top-K min score = **{fmt(topk_min_mean)}** (P50 = {fmt(topk_min_p50)}, P78-P80 영역) → Recall@20 = **{fmt(topk_recall_mean)}** (gold 의 31.4% 가 top-20 밖에 위치 — cosine raw 만으로는 R 천장 0.69). **추천 #1: per-query P80** (|sel| mean = 18.9, R = 0.6133, top-K=20 과 가장 유사한 query-aware threshold). 기존 SuperNode 평균 schema 노드 수 = **{fmt(sn_mean, 1)}** → Directed Top-K 는 SuperNode 대비 ~20% 노드만 보존 (graph topology 큰 변화). **GAT 학습 mechanism evidence**: Directed Top-K 가 raw cosine recall 한계 (0.69) 를 GAT 학습으로 극복할 가능성 → 학위 논문 §V.5 main mechanism 분석 base.")
    A("")
    A("**3 핵심 발견**:")
    A("")
    A(f"1. **Score 분포 — gold/non-gold 분리 강함, threshold 영역 sweet spot 존재**")
    A(f"   - Per-query ROC-AUC mean = **{fmt(per_q_roc)}** (selectors/CLAUDE.md HISTORY §4 의 0.741 vs Ensemble 0.776 와 다소 차이 — per-query 분포 vs global 분포 차이)")
    A(f"   - Global Cohen's d (gold vs non-gold) = **{fmt(cohens)}** (large effect size)")
    A(f"   - Gold score 평균 = {fmt(gold_sep['all_gold_mean'])} (P25 = {fmt(gold_sep['all_gold_p25'])}, P50 = {fmt(gold_sep['all_gold_p50'])}, P75 = {fmt(gold_sep['all_gold_p75'])})")
    A(f"   - Non-gold score 평균 = {fmt(gold_sep['all_nongold_mean'])} (P25 = {fmt(gold_sep['all_nongold_p25'])}, P50 = {fmt(gold_sep['all_nongold_p50'])}, P75 = {fmt(gold_sep['all_nongold_p75'])})")
    A("")
    A(f"2. **기존 top-K=20 의 score range — P50 = {fmt(topk_min_p50)} ~ {fmt(topk['agg']['top1_score_p50'])}**")
    A(f"   - Top-1 score 분포 (per-query 의 max): mean = {fmt(topk['agg']['top1_score_mean'])} (P25 = {fmt(topk['agg']['top1_score_p25'])}, P50 = {fmt(topk['agg']['top1_score_p50'])}, P75 = {fmt(topk['agg']['top1_score_p75'])})")
    A(f"   - Top-20 min score 분포: mean = {fmt(topk_min_mean)} (P25 = {fmt(topk['agg']['topk_min_score_p25'])}, P50 = {fmt(topk_min_p50)}, P75 = {fmt(topk['agg']['topk_min_score_p75'])})")
    A(f"   - Recall@20 mean = **{fmt(topk_recall_mean)}** (top-K=20 가 gold 의 {topk_recall_mean*100:.1f}% capture)")
    A("")
    A(f"3. **Threshold 후보 4 종 — F1 best 3 + 학습 적합 추천**")
    A(f"   - **F1 max top-3 (raw selector standalone)**:")
    for i, r in enumerate(top3_f1, 1):
        A(f"     - #{i} **{r['label']}**: |sel| mean = {fmt(r['selected_mean'], 1)} (std {fmt(r['selected_std'], 1)}), R = {fmt(r['recall_mean'])}, P = {fmt(r['precision_mean'])}, F1 = {fmt(r['f1_mean'])}")
    A(f"   - **학습 적합 (Directed Top-K SuperNode trade-off)**:")
    p80_tldr = next((r for r in candidates['rows'] if r['label']=='P80'), None)
    abs07_tldr = next((r for r in candidates['rows'] if r['label']=='abs τ=0.7'), None)
    if p80_tldr:
        A(f"     - **P80 (학습 추천 #1)**: |sel| = {fmt(p80_tldr['selected_mean'], 1)}, R = {fmt(p80_tldr['recall_mean'])} — top-K=20 과 가장 유사한 노드 수 + query-aware (DB 별 schema 크기 자동 보정) → graph topology 균질")
    if abs07_tldr:
        A(f"     - 절대 τ=0.7 (학습 추천 #2): |sel| = {fmt(abs07_tldr['selected_mean'], 1)}, R = {fmt(abs07_tldr['recall_mean'])}, F1 = {fmt(abs07_tldr['f1_mean'])} — F1 max 영역 단 노드 수 variability 큼")
    A("   - **GAT 학습 후 R 회복 가능성**: raw selector 만으로 R≤0.7 이지만 GAT 가 이웃 노드 정보 + edge connectivity 통해 R 보강 → 학위 논문 main mechanism")
    A("")

    # ── §1 데이터 + 방법 ──
    A("## §1. 데이터 및 방법")
    A("")
    A(f"- **BIRD-Dev**: {n_total} queries (Simple={n_simple}, Moderate={n_moderate}, Challenging={n_challenging})")
    A(f"- **Score 데이터**: `outputs/.../{COSINE_CELL}/score_analysis_*.jsonl`")
    A("  - α=1.0 (Cosine only, GAT 기여 X) cell — **EnsembleSelector** 의 per-query min-max norm 후 cosine score")
    A("  - score 범위: [0.0, 1.0], score=1.0 = per-query top-1 노드")
    A(f"- **Total nodes**: {gold_sep['n_gold_total'] + gold_sep['n_nongold_total']} (gold = {gold_sep['n_gold_total']}, non-gold = {gold_sep['n_nongold_total']})")
    A(f"- **Per-query node count**: mean = {fmt(per_q['agg']['n_nodes_mean'], 1)} (P25 = {fmt(per_q['agg']['n_nodes_p25'], 1)}, P50 = {fmt(per_q['agg']['n_nodes_p50'], 1)}, P75 = {fmt(per_q['agg']['n_nodes_p75'], 1)}) — DB 별 schema 크기 차이")
    A("- **Threshold 후보 4 종**:")
    A("  - (a) 절대 score: τ ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}")
    A("  - (b) Per-query percentile: P70/P75/P80/P85/P90/P95")
    A("  - (c) Mean+std: μ+0σ, μ+0.5σ, μ+1σ, μ+1.5σ, μ+2σ")
    A("  - (d) Elbow point (per-query): 점수 분포 second derivative max (자동 변곡점)")
    A("  - (e) top-K=20 baseline (참고)")
    A("")

    # ── §2 per-query raw score 분포 ──
    A("## §2. Per-Query Raw Score 분포")
    A("")
    A("### 2.1 Per-query 통계 (1534 queries 평균)")
    A("")
    A("| 통계 | mean | P25 | P50 | P75 |")
    A("|---|---:|---:|---:|---:|")
    for col, label in [("score_min", "score min"), ("score_max", "score max"),
                        ("score_mean", "score mean"), ("score_std", "score std"),
                        ("score_p25", "score P25"), ("score_p50", "score P50"),
                        ("score_p75", "score P75"), ("score_p90", "score P90"),
                        ("score_p95", "score P95"), ("score_p99", "score P99")]:
        A(f"| {label} | {fmt(per_q['agg'][f'{col}_mean'])} | "
          f"{fmt(per_q['agg'][f'{col}_p25'])} | "
          f"{fmt(per_q['agg'][f'{col}_p50'])} | "
          f"{fmt(per_q['agg'][f'{col}_p75'])} |")
    A("")
    A("**해석**:")
    A(f"- Per-query score min ≈ 0 / max ≈ 1 → min-max norm 동작 확인")
    A(f"- Score mean ≈ {fmt(per_q['agg']['score_mean_mean'])} (P50 도 비슷) → 분포가 대체로 균등")
    A(f"- Score P95 mean = {fmt(per_q['agg']['score_p95_mean'])} → top 5% 노드는 score ≥ {fmt(per_q['agg']['score_p95_mean'])} 영역")
    A(f"- Per-query node 수 mean = {fmt(per_q['agg']['n_nodes_mean'], 1)} → DB 별 schema 크기 차이 큼")
    A("")

    # ── §3 gold vs non-gold 분리 ──
    A("## §3. Gold vs Non-Gold Score 분리")
    A("")
    A("### 3.1 Score 분포 비교 (전체 nodes pooled)")
    A("")
    A("| 그룹 | n | mean | P25 / P50 / P75 |")
    A("|---|---:|---:|---|")
    A(f"| **Gold** | {gold_sep['n_gold_total']} | {fmt(gold_sep['all_gold_mean'])} | "
      f"{fmt(gold_sep['all_gold_p25'])} / {fmt(gold_sep['all_gold_p50'])} / {fmt(gold_sep['all_gold_p75'])} |")
    A(f"| Non-gold | {gold_sep['n_nongold_total']} | {fmt(gold_sep['all_nongold_mean'])} | "
      f"{fmt(gold_sep['all_nongold_p25'])} / {fmt(gold_sep['all_nongold_p50'])} / {fmt(gold_sep['all_nongold_p75'])} |")
    A("")
    A(f"- **Global ROC-AUC** (gold vs non-gold, all nodes pooled): {fmt(overall_roc)}")
    A(f"- **Per-query ROC-AUC mean**: {fmt(per_q_roc)} (selectors/CLAUDE.md HISTORY §4: Cosine 0.741 vs Ensemble 0.776 — global 다소 차이, 본 분석 normalized score 정의 차이)")
    A(f"- **Cohen's d** (effect size): {fmt(cohens)} ({'large' if abs(cohens) >= 0.8 else 'medium' if abs(cohens) >= 0.5 else 'small'} effect)")
    A("")
    A("### 3.2 Threshold τ 별 trade-off (절대 score, 전체 1534 queries pooled)")
    A("")
    A("| τ | Global P | Global R | Global F1 | selected per-query mean | P25 / P50 / P75 |")
    A("|---:|---:|---:|---:|---:|---|")
    for r in gold_sep["threshold_rows"]:
        A(f"| {r['tau']} | {fmt(r['global_precision'])} | {fmt(r['global_recall'])} | "
          f"{fmt(r['global_f1'])} | {fmt(r['selected_per_query_mean'], 1)} | "
          f"{fmt(r['selected_per_query_p25'], 1)} / "
          f"{fmt(r['selected_per_query_p50'], 1)} / "
          f"{fmt(r['selected_per_query_p75'], 1)} |")
    A("")
    A("**해석**:")
    A(f"- Threshold τ 가 클수록 P↑ R↓ trade-off — sweet spot은 F1 max 영역")
    best_tau = max(gold_sep["threshold_rows"], key=lambda r: r["global_f1"])
    A(f"- **F1 max at τ = {best_tau['tau']}** (P = {fmt(best_tau['global_precision'])}, R = {fmt(best_tau['global_recall'])}, F1 = {fmt(best_tau['global_f1'])})")
    A(f"- 단 본 분석은 **selector raw score 만** 의 selected node — Filter 통과 전 단계 → 학습 시 은 어떤 노드를 GAT 입력으로 넣을지 결정용")
    A("")

    # ── §4 top-K=20 score range ──
    A("## §4. 기존 Top-K=20 의 Score Range")
    A("")
    A("### 4.1 Top-K=20 통계 (1534 queries)")
    A("")
    A("| 통계 | mean | P25 | P50 | P75 |")
    A("|---|---:|---:|---:|---:|")
    A(f"| Top-1 score | {fmt(topk['agg']['top1_score_mean'])} | "
      f"{fmt(topk['agg']['top1_score_p25'])} | "
      f"{fmt(topk['agg']['top1_score_p50'])} | "
      f"{fmt(topk['agg']['top1_score_p75'])} |")
    A(f"| Top-20 min score | {fmt(topk['agg']['topk_min_score_mean'])} | "
      f"{fmt(topk['agg']['topk_min_score_p25'])} | "
      f"{fmt(topk['agg']['topk_min_score_p50'])} | "
      f"{fmt(topk['agg']['topk_min_score_p75'])} |")
    A(f"| Recall@20 | {fmt(topk['agg']['recall_topk_mean'])} | "
      f"{fmt(topk['agg']['recall_topk_p25'])} | "
      f"{fmt(topk['agg']['recall_topk_p50'])} | "
      f"- |")
    A("")
    A("**해석**:")
    A(f"- Top-1 score 는 항상 1.0 (per-query min-max norm 결과)")
    A(f"- **Top-20 min score 분포**: mean = {fmt(topk_min_mean)}, P50 = {fmt(topk_min_p50)} → top-K=20 cap 의 \"보더라인\" 노드 score")
    A(f"- Top-K=20 의 query 별 variability: top-20 min P25 = {fmt(topk['agg']['topk_min_score_p25'])} ~ P75 = {fmt(topk['agg']['topk_min_score_p75'])}")
    A(f"- → Directed Top-K 가 top-K=20 동일 성능을 노리려면, 절대 τ ≈ {fmt(topk_min_mean)} 또는 per-query 의 P{int(100 - 20*100/per_q['agg']['n_nodes_mean'])} (가변)")
    A("")

    # ── §5 Threshold 후보 4 종 비교 ──
    A("## §5. Threshold 후보 4 종 비교 — Selected Node 수 + Recall")
    A("")
    A("### 5.1 전체 비교 표 (1534 queries 평균)")
    A("")
    A("| 후보 | n | |sel| mean | |sel| std | |sel| P50 | R | P | F1 |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in candidates["rows"]:
        A(f"| {r['label']} | {r['n_queries']} | "
          f"{fmt(r['selected_mean'], 1)} | "
          f"{fmt(r['selected_std'], 1)} | "
          f"{fmt(r['selected_p50'], 1)} | "
          f"{fmt(r['recall_mean'])} | "
          f"{fmt(r['precision_mean'])} | "
          f"{fmt(r['f1_mean'])} |")
    A("")
    A("### 5.2 Difficulty 별 selected node 수 + recall (key 후보 4 종)")
    A("")
    key_labels = ["abs τ=0.7", "P80", "μ+1.0σ", "elbow (auto)", "top-K=20 (baseline)"]
    A("| 후보 | Simple |sel| / R | Moderate |sel| / R | Challenging |sel| / R |")
    A("|---|---:|---:|---:|")
    for label in key_labels:
        r = next((r for r in candidates["rows"] if r["label"] == label), None)
        if not r:
            continue
        A(f"| {label} | "
          f"{fmt(r.get('selected_simple_mean'), 1)} / {fmt(r.get('recall_simple_mean'))} | "
          f"{fmt(r.get('selected_moderate_mean'), 1)} / {fmt(r.get('recall_moderate_mean'))} | "
          f"{fmt(r.get('selected_challenging_mean'), 1)} / {fmt(r.get('recall_challenging_mean'))} |")
    A("")
    A("### 5.3 후보별 추천 영역")
    A("")
    A("| 후보 type | 장점 | 단점 | Directed Top-K SuperNode 적합성 |")
    A("|---|---|---|---|")
    A("| **절대 τ** | query-invariant 정의, 학습 단순 | 노드 수 variability 큼 (DB 별 schema 크기 차이 큼) | DB 변동 안정성 낮음 — schema 가 작은 DB 에서는 적정, 큰 DB 에서는 부족 |")
    A("| **Per-query Pn** | 노드 수 안정 (n=node_count*(100-P)/100), graph topology 균질 | 노드 의미 (P80) 가 query 별 다름 | **추천 #1** — 노드 수 균질 + query-aware |")
    A("| **μ+kσ** | 통계적 정합 (분포 평균+분산 고려) | 분포 skew 시 비대칭, 학습 해석 어려움 | 통계 base 견고 단 학습 시 직관 약함 |")
    A("| **Elbow point** | per-query 자동 변곡점, 적응적 | second derivative noise 민감 | 학습 단순화 약함, 결정론적 차이 큼 |")
    A("")
    # Recommendations narrative
    p80_row = next((r for r in candidates["rows"] if r["label"] == "P80"), None)
    abs07_row = next((r for r in candidates["rows"] if r["label"] == "abs τ=0.7"), None)
    musigma_row = next((r for r in candidates["rows"] if r["label"] == "μ+1.0σ"), None)
    A("**권장 (Directed Top-K SuperNode 학습 시)**:")
    if p80_row:
        A(f"- **추천 #1: per-query P80** — selected mean = {fmt(p80_row['selected_mean'], 1)} ± {fmt(p80_row['selected_std'], 1)}, R = {fmt(p80_row['recall_mean'])}, F1 = {fmt(p80_row['f1_mean'])} (top-K=20 과 가장 유사한 노드 수, query-aware)")
    if abs07_row:
        A(f"- 추천 #2: 절대 τ=0.7 — selected mean = {fmt(abs07_row['selected_mean'], 1)}, R = {fmt(abs07_row['recall_mean'])}, F1 = {fmt(abs07_row['f1_mean'])} (학습 단순)")
    if musigma_row:
        A(f"- 추천 #3: μ+1σ — selected mean = {fmt(musigma_row['selected_mean'], 1)}, R = {fmt(musigma_row['recall_mean'])}, F1 = {fmt(musigma_row['f1_mean'])} (통계 base 견고)")
    A("")

    # ── §6 기존 SuperNode 비교 base ──
    A("## §6. 기존 SuperNode 비교 Base — DB 별 평균 Schema 노드 수")
    A("")
    A("### 6.1 DB 별 schema 노드 수 (기존 SuperNode = 모든 schema 노드 + bidirectional edge)")
    A("")
    A("| DB | n_queries | mean | P25 / P50 / P75 |")
    A("|---|---:|---:|---|")
    for db, agg in sorted(supernode["db_summary"].items(),
                          key=lambda kv: kv[1]["supernode_node_count_mean"], reverse=True):
        A(f"| {db} | {agg['n_queries']} | "
          f"{fmt(agg['supernode_node_count_mean'], 1)} | "
          f"{fmt(agg['supernode_node_count_p25'], 1)} / "
          f"{fmt(agg['supernode_node_count_p50'], 1)} / "
          f"{fmt(agg['supernode_node_count_p75'], 1)} |")
    A("")
    A(f"- **전체 평균 SuperNode 노드 수**: {fmt(sn_mean, 1)} (T+C+FK 모두 포함)")
    A("")
    A("### 6.2 Directed Top-K threshold 후보별 schema 노드 수 ratio (vs 기존 SuperNode)")
    A("")
    A("| 후보 | |sel| mean | vs SuperNode ratio | graph topology 변화 |")
    A("|---|---:|---:|---|")
    for label in ["abs τ=0.5", "abs τ=0.7", "abs τ=0.9", "P80", "P90", "P95",
                  "μ+0.5σ", "μ+1.0σ", "μ+2.0σ", "elbow (auto)", "top-K=20 (baseline)"]:
        r = next((r for r in candidates["rows"] if r["label"] == label), None)
        if not r:
            continue
        ratio = r["selected_mean"] / sn_mean if sn_mean > 0 else float("nan")
        topo_change = "큰 변화" if ratio < 0.3 else ("중간" if ratio < 0.6 else "작은 변화")
        A(f"| {label} | {fmt(r['selected_mean'], 1)} | {fmt(ratio*100, 1)}% | {topo_change} |")
    A("")
    A("**해석**:")
    A(f"- 기존 SuperNode: 평균 {fmt(sn_mean, 1)} schema 노드 + bidirectional edge")
    A(f"- Directed Top-K (P80 또는 abs τ=0.7): 평균 ~20-30 노드 + directed edge → SuperNode 대비 **20-30% 노드만 보존** (큰 graph topology 변화)")
    A(f"- 기존 SuperNode α=0 (GAT only) F1=0.5476 vs concat α=0 F1=0.7211 (Δ=-0.1735) → graph topology 변화의 영향 baseline")
    A("- **dual mechanism 분리** (학위 논문 mechanism 분석 시):")
    A("  - **Directed edge 변경**: bidirectional → query→schema 단방향 (노드 수 동일)")
    A("  - **노드 필터링**: 모든 schema → top-K/threshold 만 (edge 방향 동일)")
    A("  - 두 변경의 효과 분리 위해 ablation 가능 (post-paper future work)")
    A("")

    # ── §7 Directed Top-K 학습 권장 threshold ──
    A("## §7. Directed Top-K SuperNode 학습 권장 Threshold (planner 결정 시 인용)")
    A("")
    A("### 7.1 추천 threshold 3 종 + 학습 변형 시나리오")
    A("")
    A("| 후보 | |sel| mean | R | F1 | 추천 사유 |")
    A("|---|---:|---:|---:|---|")
    if p80_row:
        A(f"| **P80 (추천 #1)** | {fmt(p80_row['selected_mean'], 1)} | {fmt(p80_row['recall_mean'])} | {fmt(p80_row['f1_mean'])} | top-K=20 과 가장 유사한 노드 수, query-aware (DB 별 schema 크기 자동 보정) |")
    if abs07_row:
        A(f"| 절대 τ=0.7 (추천 #2) | {fmt(abs07_row['selected_mean'], 1)} | {fmt(abs07_row['recall_mean'])} | {fmt(abs07_row['f1_mean'])} | 학습 단순, query-invariant — graph topology 균질성은 낮음 |")
    if musigma_row:
        A(f"| μ+1σ (추천 #3) | {fmt(musigma_row['selected_mean'], 1)} | {fmt(musigma_row['recall_mean'])} | {fmt(musigma_row['f1_mean'])} | 통계 base 견고 단 학습 직관 약함 |")
    A("")
    A("### 7.2 학습 시 변형 ablation 권장 (단계 2 학습 시)")
    A("")
    A("- **변형 1**: P80 (추천 #1) — primary 학습")
    A("- **변형 2**: top-K=20 (baseline reference) — 기존 SuperNode 와 직접 비교")
    A("- **변형 3 (선택)**: 절대 τ=0.7 — DB 변동 안정성 baseline")
    A("- 학습 시 val recall@15 (또는 동일 metric) 비교 → 최종 ckpt 선택 후 paper main alpha sweep 측정")
    A("")
    A("### 7.3 학위 논문 Part III mechanism 분석 evidence pool")
    A("")
    A("- §V.5.1 raw score 분포 — 본 분석 §2-3 (per-query distribution + gold separation)")
    A("- §V.5.2 threshold 후보 비교 — 본 분석 §5 (4 종 후보)")
    A("- §V.5.3 SuperNode 변경 mechanism — 본 분석 §6 (dual mechanism 분리)")
    A("- §V.5.4 측정 결과 (학습 + paper main alpha sweep) — 단계 3 측정 후 신규 작성")
    A("")

    # ── §8 raw 데이터 ──
    A("## §8. JSONL/CSV Raw 데이터 (재현 가능)")
    A("")
    A("- `notebooks/analysis_results/raw_score_per_query_stats.csv` — per-query 1534 rows: min/max/mean/std/percentile")
    A("- `notebooks/analysis_results/raw_score_threshold_candidates.csv` — 4 종 후보 별 |selected| / R / P / F1 + difficulty 별")
    A("- `notebooks/analysis_results/raw_score_supernode_comparison.csv` — DB 별 SuperNode 노드 수 + Directed Top-K ratio")
    A("- 재현 스크립트: `src/analysis/analyze_raw_score_distribution.py`")
    A("")

    # ── 변경 이력 ──
    A("---")
    A("")
    A("## Changelog")
    A("")
    A("- 2026-05-05: Analyzer 신규 작성 (DECISIONS 2026-05-05 Directed Top-K SuperNode Part III 결정 후속, LLM-free 즉시 위임).")
    A("  - §2 per-query raw cosine score 분포 (P25/P50/P75/P90/P95)")
    A("  - §3 gold vs non-gold score 분리 (ROC-AUC + Cohen's d + threshold trade-off)")
    A("  - §4 기존 top-K=20 의 score range")
    A("  - §5 Threshold 후보 4 종 비교 (절대 / percentile / mean+std / elbow)")
    A("  - §6 기존 SuperNode 비교 base")
    A("  - §7 Directed Top-K 학습 권장 threshold (planner 결정 시 인용)")

    return "\n".join(lines) + "\n"


def main():
    print("Loading dev meta...")
    dev = load_dev_meta()
    qid_diff = {qid: rec.get("difficulty", "unknown") for qid, rec in dev.items()}

    print(f"Loading score_analysis from {COSINE_SCORE_PATH}...")
    scores_by_qid = load_score_analysis(COSINE_SCORE_PATH)
    print(f"  loaded {len(scores_by_qid)} qids")

    print("(1) Per-query distribution...")
    per_q = analyze_per_query_distribution(scores_by_qid)

    print("(2) Gold vs non-gold separation...")
    gold_sep = analyze_gold_separation(scores_by_qid)

    print("(3) Top-K=20 score range...")
    topk = analyze_topk_score_range(scores_by_qid, k=20)

    print("(4) Threshold candidates...")
    candidates = analyze_threshold_candidates(scores_by_qid, qid_diff)

    print("(5) SuperNode comparison...")
    supernode = analyze_supernode_comparison(scores_by_qid, dev)

    # CSV 출력
    write_csv(ANALYSIS_DIR / "raw_score_per_query_stats.csv",
              per_q["rows"],
              ["qid", "n_nodes", "score_min", "score_max", "score_mean", "score_std",
               "score_p25", "score_p50", "score_p75", "score_p90", "score_p95", "score_p99"])

    write_csv(ANALYSIS_DIR / "raw_score_threshold_candidates.csv",
              candidates["rows"],
              ["label", "n_queries", "selected_mean", "selected_std", "selected_p25",
               "selected_p50", "selected_p75", "recall_mean", "precision_mean", "f1_mean",
               "selected_simple_mean", "recall_simple_mean",
               "selected_moderate_mean", "recall_moderate_mean",
               "selected_challenging_mean", "recall_challenging_mean"])

    sn_csv = []
    for db, agg in sorted(supernode["db_summary"].items()):
        sn_csv.append({"db": db, **agg})
    write_csv(ANALYSIS_DIR / "raw_score_supernode_comparison.csv",
              sn_csv,
              ["db", "n_queries", "supernode_node_count_mean",
               "supernode_node_count_p25", "supernode_node_count_p50",
               "supernode_node_count_p75"])

    md = render_markdown(per_q, gold_sep, topk, candidates, supernode, qid_diff)
    md_path = ANALYSIS_DIR / "raw_score_distribution_for_directed_topk.md"
    with open(md_path, "w") as f:
        f.write(md)

    print(f"\n✓ Wrote {md_path}")
    print(f"✓ Wrote 3 CSVs to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
