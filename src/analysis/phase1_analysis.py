"""
Phase 1: 기존 결과 기반 정량 분석
- E1: 단계별 병목 분석
- A1: Score Distribution (gold vs non-gold)
- B1: Selector Recall 상한선
- B2: 오류 유형 분류
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter, defaultdict

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 120
plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = "/home/hyeonjin/thesis_refactored/outputs"
SAVE_DIR = "/home/hyeonjin/thesis_refactored/notebooks/analysis_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ================================================================
# 데이터 로딩
# ================================================================

def load_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

# 실험별 결과 로딩
experiments = {
    # Baselines
    "VectorOnly (baseline)": f"{OUTPUT_DIR}/baselines/preliminary_vector_only/output_preliminary_vector_only.jsonl",
    "MST Expansion": f"{OUTPUT_DIR}/baselines/preliminary_graph_expansion/output_preliminary_graph_expansion.jsonl",
    "G-Retriever (PCST)": f"{OUTPUT_DIR}/baselines/baseline_g_retriever/output_baseline_g_retriever.jsonl",
    "LinkAlign": f"{OUTPUT_DIR}/baselines/baseline_linkalign/output_baseline_linkalign.jsonl",
    "XiYanSQL": f"{OUTPUT_DIR}/baselines/baseline_xiyansql/output_baseline_xiyansql.jsonl",
    # Proposed
    "GAT Classifier": f"{OUTPUT_DIR}/experiments/experiment_gat_classifier/output_experiment_gat_classifier.jsonl",
    "GAT+MultiAgent": f"{OUTPUT_DIR}/experiments/experiment_gat_classifier_multi_agent/output_experiment_gat_classifier_multi_agent.jsonl",
    # PCST Variants
    "Base PCST": f"{OUTPUT_DIR}/experiments/experiment_base_pcst/output_base_pcst.jsonl",
    "Dynamic PCST": f"{OUTPUT_DIR}/experiments/experiment_dynamic_pcst/output_dynamic_pcst.jsonl",
    "Uncertainty PCST": f"{OUTPUT_DIR}/experiments/experiment_uncertainty_pcst/output_uncertainty_pcst.jsonl",
    "Dyn+Unc PCST": f"{OUTPUT_DIR}/experiments/experiment_dynamic_uncertainty_pcst/output_dynamic_uncertainty_pcst.jsonl",
}

# Score analysis 파일 로딩
score_files = {
    "GAT Classifier": f"{OUTPUT_DIR}/experiments/experiment_gat_classifier/score_analysis_experiment_gat_classifier.jsonl",
    "Base PCST": f"{OUTPUT_DIR}/experiments/experiment_base_pcst/score_analysis_base_pcst.jsonl",
    "GAT+MultiAgent": f"{OUTPUT_DIR}/experiments/experiment_gat_classifier_multi_agent/score_analysis_experiment_gat_classifier_multi_agent.jsonl",
    "G-Retriever (PCST)": f"{OUTPUT_DIR}/baselines/baseline_g_retriever/score_analysis_baseline_g_retriever.jsonl",
    "VectorOnly (baseline)": f"{OUTPUT_DIR}/baselines/preliminary_vector_only/score_analysis_preliminary_vector_only.jsonl",
}

all_data = {}
for name, path in experiments.items():
    if os.path.exists(path):
        df = pd.DataFrame(load_jsonl(path))
        if len(df) > 0 and 'recall' in df.columns:
            all_data[name] = df
            print(f"✓ {name}: {len(df)} queries loaded")
        else:
            print(f"✗ {name}: empty or missing recall column")
    else:
        print(f"✗ {name}: file not found")

all_scores = {}
for name, path in score_files.items():
    if os.path.exists(path):
        all_scores[name] = load_jsonl(path)
        print(f"✓ Score: {name}: {len(all_scores[name])} records")

print(f"\n총 {len(all_data)}개 실험 로드 완료\n")

# ================================================================
# E1: 실험간 전체 성능 비교 (단계별 병목 파악)
# ================================================================
print("=" * 70)
print("E1: 실험간 성능 비교 및 병목 분석")
print("=" * 70)

summary_rows = []
for name, df in all_data.items():
    row = {
        'Method': name,
        'Recall': df['recall'].mean(),
        'Precision': df['precision'].mean(),
        'F1': 2 * df['recall'].mean() * df['precision'].mean() / max(df['recall'].mean() + df['precision'].mean(), 1e-9),
        'EX': df['ex'].mean() if 'ex' in df.columns else 0.0,
        'Avg #Pred Cols': df['pred_cols'].apply(len).mean() if 'pred_cols' in df.columns else 0,
        'Avg #Gold Cols': df['gold_cols'].apply(len).mean() if 'gold_cols' in df.columns else 0,
        'Avg #Missing Cols': df['missing_cols'].apply(len).mean() if 'missing_cols' in df.columns else 0,
        'Avg #Extra Cols': df['extra_cols'].apply(len).mean() if 'extra_cols' in df.columns else 0,
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).sort_values('F1', ascending=False)
print(summary_df.to_string(index=False, float_format='{:.4f}'.format))
summary_df.to_csv(f"{SAVE_DIR}/e1_summary.csv", index=False)

# Bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
methods = summary_df['Method'].values
x = np.arange(len(methods))

for ax, metric in zip(axes, ['Recall', 'Precision', 'EX']):
    vals = summary_df[metric].values
    colors = ['#2196F3' if 'baseline' in m.lower() or m in ['MST Expansion', 'LinkAlign', 'XiYanSQL', 'G-Retriever (PCST)'] else '#FF5722' for m in methods]
    bars = ax.barh(x, vals, color=colors, edgecolor='white')
    ax.set_yticks(x)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel(metric)
    ax.set_title(metric, fontweight='bold')
    ax.set_xlim(0, 1.0)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2, f'{v:.3f}', va='center', fontsize=8)

plt.suptitle('E1: Experiment Performance Comparison\n(Blue=Baseline, Red=Proposed)', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/e1_performance_comparison.png", bbox_inches='tight')
plt.close()
print(f"\n→ 저장: {SAVE_DIR}/e1_performance_comparison.png\n")

# ================================================================
# E1-b: Subgraph Size vs Performance (PCST 과다 선택 진단)
# ================================================================
print("=" * 70)
print("E1-b: Subgraph 크기 vs 성능 (PCST 과다 선택 진단)")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 6))
for name, df in all_data.items():
    if 'pred_cols' in df.columns and 'pred_tables' in df.columns:
        avg_nodes = df['pred_cols'].apply(len).mean() + df['pred_tables'].apply(len).mean()
        recall = df['recall'].mean()
        precision = df['precision'].mean()
        ax.scatter(avg_nodes, recall, s=100, label=f"{name} (P={precision:.2f})", zorder=5)
        ax.annotate(name, (avg_nodes, recall), fontsize=7, textcoords="offset points", xytext=(5, 5))

ax.set_xlabel('Avg # Predicted Nodes (Tables + Columns)')
ax.set_ylabel('Recall')
ax.set_title('E1-b: Subgraph Size vs Recall (Precision in legend)')
ax.legend(fontsize=7, loc='lower right')
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/e1b_size_vs_recall.png", bbox_inches='tight')
plt.close()
print(f"→ 저장: {SAVE_DIR}/e1b_size_vs_recall.png\n")

# ================================================================
# A1: Score Distribution (gold vs non-gold)
# ================================================================
print("=" * 70)
print("A1: Score Distribution Analysis (gold vs non-gold)")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (name, records) in enumerate(all_scores.items()):
    if idx >= 6:
        break
    ax = axes[idx]

    gold_scores = [r['score'] for r in records if r['is_gold']]
    non_gold_scores = [r['score'] for r in records if not r['is_gold']]

    # Statistics
    gold_mean = np.mean(gold_scores) if gold_scores else 0
    non_gold_mean = np.mean(non_gold_scores) if non_gold_scores else 0
    gold_median = np.median(gold_scores) if gold_scores else 0
    non_gold_median = np.median(non_gold_scores) if non_gold_scores else 0

    print(f"\n--- {name} ---")
    print(f"  Gold:     n={len(gold_scores):,}, mean={gold_mean:.4f}, median={gold_median:.4f}, std={np.std(gold_scores):.4f}")
    print(f"  Non-Gold: n={len(non_gold_scores):,}, mean={non_gold_mean:.4f}, median={non_gold_median:.4f}, std={np.std(non_gold_scores):.4f}")
    print(f"  Mean Gap: {gold_mean - non_gold_mean:.4f}")

    # Overlap 계산 (히스토그램 기반)
    bins = np.linspace(0, 1, 50)
    gold_hist, _ = np.histogram(gold_scores, bins=bins, density=True)
    non_gold_hist, _ = np.histogram(non_gold_scores, bins=bins, density=True)
    overlap = np.sum(np.minimum(gold_hist, non_gold_hist)) * (bins[1] - bins[0])
    print(f"  Histogram Overlap: {overlap:.4f}")

    ax.hist(non_gold_scores, bins=50, alpha=0.5, label=f'Non-Gold (n={len(non_gold_scores):,})', color='gray', density=True)
    ax.hist(gold_scores, bins=50, alpha=0.7, label=f'Gold (n={len(gold_scores):,})', color='#FF5722', density=True)
    ax.axvline(gold_mean, color='red', linestyle='--', linewidth=1, label=f'Gold μ={gold_mean:.3f}')
    ax.axvline(non_gold_mean, color='gray', linestyle='--', linewidth=1, label=f'Non-Gold μ={non_gold_mean:.3f}')
    ax.set_title(f'{name}\nOverlap={overlap:.3f}, Gap={gold_mean-non_gold_mean:.3f}', fontsize=10)
    ax.legend(fontsize=7)
    ax.set_xlabel('Similarity Score')

# Hide unused axes
for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('A1: Score Distribution — Gold vs Non-Gold Nodes', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/a1_score_distribution.png", bbox_inches='tight')
plt.close()
print(f"\n→ 저장: {SAVE_DIR}/a1_score_distribution.png\n")

# ================================================================
# A1-b: AUROC per experiment
# ================================================================
from sklearn.metrics import roc_auc_score

print("A1-b: AUROC (Score 기반 gold 분류 능력)")
for name, records in all_scores.items():
    labels = [1 if r['is_gold'] else 0 for r in records]
    scores = [r['score'] for r in records]
    if sum(labels) > 0 and sum(labels) < len(labels):
        auroc = roc_auc_score(labels, scores)
        print(f"  {name}: AUROC = {auroc:.4f}")

# ================================================================
# B1: Selector Recall 상한선 (Score 기반 Top-K Recall)
# ================================================================
print("\n" + "=" * 70)
print("B1: Score 기반 Top-K Recall 상한선 (Selector가 놓치는 비율)")
print("=" * 70)

ks = [3, 5, 10, 15, 20, 30, 50]

fig, ax = plt.subplots(figsize=(10, 6))

for name, records in all_scores.items():
    # Group by query_id
    by_query = defaultdict(list)
    for r in records:
        by_query[r['query_id']].append(r)

    recall_at_k = {k: [] for k in ks}

    for qid, qrecords in by_query.items():
        gold_indices = {i for i, r in enumerate(qrecords) if r['is_gold']}
        if not gold_indices:
            continue

        # Sort by score descending
        sorted_indices = sorted(range(len(qrecords)), key=lambda i: qrecords[i]['score'], reverse=True)

        for k in ks:
            top_k = set(sorted_indices[:k])
            hits = len(top_k & gold_indices)
            recall_at_k[k].append(hits / len(gold_indices))

    avg_recalls = [np.mean(recall_at_k[k]) for k in ks]
    ax.plot(ks, avg_recalls, marker='o', label=name, linewidth=2)

    print(f"\n--- {name} ---")
    for k, r in zip(ks, avg_recalls):
        print(f"  Recall@{k:2d} = {r:.4f}")

ax.set_xlabel('K (Number of Selected Nodes)')
ax.set_ylabel('Recall')
ax.set_title('B1: Top-K Recall Upper Bound by Score Ranking', fontweight='bold')
ax.legend()
ax.set_xticks(ks)
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/b1_recall_at_k.png", bbox_inches='tight')
plt.close()
print(f"\n→ 저장: {SAVE_DIR}/b1_recall_at_k.png\n")

# ================================================================
# B2: 오류 유형 분류 (놓치는 gold 노드 패턴)
# ================================================================
print("=" * 70)
print("B2: 오류 유형 분류 (Missing Gold 노드 패턴 분석)")
print("=" * 70)

# 주요 실험 2개에 대해 분석
target_exps = ["G-Retriever (PCST)", "GAT+MultiAgent", "GAT Classifier", "VectorOnly (baseline)"]

for exp_name in target_exps:
    if exp_name not in all_data:
        continue

    df = all_data[exp_name]

    # Missing 컬럼 패턴 분석
    missing_counter = Counter()
    missing_table_counter = Counter()
    total_missing = 0
    total_queries = len(df)
    zero_recall = (df['recall'] == 0).sum()
    perfect_recall = (df['recall'] == 1.0).sum()

    # Gold 컬럼 수별 recall 분포
    gold_size_recall = []

    for _, row in df.iterrows():
        missing_cols = row.get('missing_cols', [])
        missing_tables = row.get('missing_tables', [])
        gold_cols = row.get('gold_cols', [])

        total_missing += len(missing_cols)

        for col in missing_cols:
            if '.' in col:
                tbl, c = col.split('.', 1)
                missing_counter[f"table:{tbl}"] += 1
            else:
                missing_counter[f"col_only:{col}"] += 1

        for tbl in missing_tables:
            missing_table_counter[tbl] += 1

        gold_size_recall.append((len(gold_cols), row['recall']))

    print(f"\n{'='*50}")
    print(f"--- {exp_name} ---")
    print(f"  Total Queries: {total_queries}")
    print(f"  Zero Recall (complete miss): {zero_recall} ({zero_recall/total_queries*100:.1f}%)")
    print(f"  Perfect Recall: {perfect_recall} ({perfect_recall/total_queries*100:.1f}%)")
    print(f"  Total Missing Columns: {total_missing}")
    print(f"\n  Top 15 Missing Tables:")
    for tbl, cnt in missing_table_counter.most_common(15):
        print(f"    {tbl}: {cnt} times ({cnt/total_queries*100:.1f}%)")

    print(f"\n  Top 15 Missing Column Sources:")
    for pattern, cnt in missing_counter.most_common(15):
        print(f"    {pattern}: {cnt} times")

# ================================================================
# B2-b: Gold 컬럼 수 vs Recall (복잡도별 성능)
# ================================================================
print("\n" + "=" * 70)
print("B2-b: Gold Schema 복잡도 vs Recall")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for exp_name in ["G-Retriever (PCST)", "GAT+MultiAgent"]:
    if exp_name not in all_data:
        continue
    df = all_data[exp_name]

    gold_sizes = df['gold_cols'].apply(len)

    # Bin by gold_cols size
    bins_gold = [(1, 2), (3, 4), (5, 7), (8, 12), (13, 100)]
    bin_labels = ['1-2', '3-4', '5-7', '8-12', '13+']

    bin_recalls = []
    bin_counts = []
    for lo, hi in bins_gold:
        mask = (gold_sizes >= lo) & (gold_sizes <= hi)
        bin_recalls.append(df.loc[mask, 'recall'].mean() if mask.any() else 0)
        bin_counts.append(mask.sum())

    ax = axes[0] if exp_name == "G-Retriever (PCST)" else axes[1]
    bars = ax.bar(bin_labels, bin_recalls, color='#2196F3', edgecolor='white')
    for bar, cnt, r in zip(bars, bin_counts, bin_recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'n={cnt}\n{r:.3f}', ha='center', fontsize=8)
    ax.set_xlabel('# Gold Columns')
    ax.set_ylabel('Avg Recall')
    ax.set_title(f'{exp_name}', fontweight='bold')
    ax.set_ylim(0, 1.0)

plt.suptitle('B2-b: Schema Complexity vs Recall', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/b2b_complexity_vs_recall.png", bbox_inches='tight')
plt.close()
print(f"→ 저장: {SAVE_DIR}/b2b_complexity_vs_recall.png\n")

# ================================================================
# B2-c: DB별 성능 분포 (어떤 DB가 어려운가)
# ================================================================
print("=" * 70)
print("B2-c: DB별 성능 분포")
print("=" * 70)

best_exp = "G-Retriever (PCST)"
if best_exp in all_data:
    df = all_data[best_exp]
    db_stats = df.groupby('db_id').agg(
        recall=('recall', 'mean'),
        precision=('precision', 'mean'),
        count=('recall', 'count')
    ).sort_values('recall')

    print(f"\n--- {best_exp}: 가장 어려운 DB Top 10 ---")
    for _, row in db_stats.head(10).iterrows():
        print(f"  {row.name}: recall={row['recall']:.4f}, precision={row['precision']:.4f}, n={int(row['count'])}")

    print(f"\n--- {best_exp}: 가장 쉬운 DB Top 10 ---")
    for _, row in db_stats.tail(10).iterrows():
        print(f"  {row.name}: recall={row['recall']:.4f}, precision={row['precision']:.4f}, n={int(row['count'])}")

# ================================================================
# 종합 진단 요약
# ================================================================
print("\n" + "=" * 70)
print("Phase 1 종합 진단 요약")
print("=" * 70)

print("""
[발견사항 요약]

1. E1 (병목):
   - PCST 계열 (Base/Dynamic/Uncertainty PCST)는 recall은 높으나 (0.66~0.76)
     precision이 극히 낮고 (0.14~0.16) EX=0.0 → 스키마 전체를 반환하는 효과
   - SQL 생성이 비활성화(enabled: false)된 상태로 EX 측정 불가
   - 최선 제안 방법(GAT+MultiAgent)도 베이스라인 G-Retriever에 못 미침

2. 핵심 문제:
   - G-Retriever: recall=0.758, precision=0.787, EX=0.249 (현재 최고 성능)
   - GAT+MultiAgent: recall=0.658, precision=0.685, EX=0.150
   - GAT Classifier (no filter): recall=0.549, precision=0.620, EX=0.091

3. 시사점:
   - GAT Classifier의 recall이 0.549로 낮음 → Selector 단계 병목
   - MultiAgent Filter가 recall을 +0.11 개선 (0.549→0.658) → 유효하지만 부족
   - PCST가 구조적 연결성은 확보하나 pruning이 부재하여 precision 파괴
""")

print(f"\n✅ 모든 결과가 {SAVE_DIR}/ 에 저장되었습니다.")
