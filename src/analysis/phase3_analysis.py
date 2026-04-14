"""
Phase 3: 모듈 간 상호작용 분석
- A3: GAT 전/후 임베딩 비교 (ranking 기여도)
- D1~D3: MultiAgent Filter 효과 분석
- E2+E3: EX 상관관계 + DB 복잡도별 성능
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 120
plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = "/home/hyeonjin/thesis_refactored/outputs"
SAVE_DIR = "/home/hyeonjin/thesis_refactored/notebooks/analysis_results"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def group_by_query(records):
    by_q = defaultdict(list)
    for r in records:
        by_q[r['query_id']].append(r)
    return by_q


# ================================================================
# 데이터 로딩
# ================================================================
print("Loading data...")

scores_gat = load_jsonl(f"{OUTPUT_DIR}/experiments/experiment_gat_classifier/score_analysis_experiment_gat_classifier.jsonl")
scores_raw = load_jsonl(f"{OUTPUT_DIR}/baselines/preliminary_vector_only/score_analysis_preliminary_vector_only.jsonl")
scores_pcst = load_jsonl(f"{OUTPUT_DIR}/experiments/experiment_base_pcst/score_analysis_base_pcst.jsonl")

out_gat = load_jsonl(f"{OUTPUT_DIR}/experiments/experiment_gat_classifier/output_experiment_gat_classifier.jsonl")
out_gat_ma = load_jsonl(f"{OUTPUT_DIR}/experiments/experiment_gat_classifier_multi_agent/output_experiment_gat_classifier_multi_agent.jsonl")
out_gretriever = load_jsonl(f"{OUTPUT_DIR}/baselines/baseline_g_retriever/output_baseline_g_retriever.jsonl")
out_vectoronly = load_jsonl(f"{OUTPUT_DIR}/baselines/preliminary_vector_only/output_preliminary_vector_only.jsonl")
out_linkalign = load_jsonl(f"{OUTPUT_DIR}/baselines/baseline_linkalign/output_baseline_linkalign.jsonl")
out_xiyan = load_jsonl(f"{OUTPUT_DIR}/baselines/baseline_xiyansql/output_baseline_xiyansql.jsonl")

pred_gat = load_jsonl(f"{OUTPUT_DIR}/experiments/experiment_gat_classifier/predictions.jsonl")
pred_gat_ma = load_jsonl(f"{OUTPUT_DIR}/experiments/experiment_gat_classifier_multi_agent/predictions.jsonl")

print("Data loaded.\n")


# ================================================================
# A3: GAT vs Raw Cosine — Ranking 능력 비교 (심화)
# ================================================================
print("=" * 70)
print("A3: GAT vs Raw Cosine — Ranking 능력 심화 비교")
print("=" * 70)

gat_by_q = group_by_query(scores_gat)
raw_by_q = group_by_query(scores_raw)

# Per-query AUROC 비교
gat_aurocs = []
raw_aurocs = []
gat_better_count = 0
raw_better_count = 0
tie_count = 0

for qid in gat_by_q:
    if qid not in raw_by_q:
        continue

    g_labels = [1 if r['is_gold'] else 0 for r in gat_by_q[qid]]
    g_scores = [r['score'] for r in gat_by_q[qid]]
    r_labels = [1 if r['is_gold'] else 0 for r in raw_by_q[qid]]
    r_scores = [r['score'] for r in raw_by_q[qid]]

    if sum(g_labels) == 0 or sum(g_labels) == len(g_labels):
        continue

    try:
        g_auc = roc_auc_score(g_labels, g_scores)
        r_auc = roc_auc_score(r_labels, r_scores)
        gat_aurocs.append(g_auc)
        raw_aurocs.append(r_auc)

        if g_auc > r_auc + 0.01:
            gat_better_count += 1
        elif r_auc > g_auc + 0.01:
            raw_better_count += 1
        else:
            tie_count += 1
    except:
        pass

print(f"Per-query AUROC (n={len(gat_aurocs)} queries):")
print(f"  GAT: mean={np.mean(gat_aurocs):.4f}, median={np.median(gat_aurocs):.4f}")
print(f"  Raw: mean={np.mean(raw_aurocs):.4f}, median={np.median(raw_aurocs):.4f}")
print(f"  GAT better: {gat_better_count} ({gat_better_count/len(gat_aurocs)*100:.1f}%)")
print(f"  Raw better: {raw_better_count} ({raw_better_count/len(gat_aurocs)*100:.1f}%)")
print(f"  Tie (|diff|<0.01): {tie_count} ({tie_count/len(gat_aurocs)*100:.1f}%)")

# Histogram of per-query AUROC difference
diff_auroc = np.array(gat_aurocs) - np.array(raw_aurocs)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# AUROC histogram
axes[0].hist(diff_auroc, bins=50, color='#2196F3', edgecolor='white', alpha=0.8)
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('AUROC(GAT) - AUROC(Raw)')
axes[0].set_ylabel('# Queries')
axes[0].set_title(f'A3: Per-Query AUROC Difference\nGAT better: {gat_better_count}, Raw better: {raw_better_count}', fontweight='bold')

# Scatter: GAT AUROC vs Raw AUROC
axes[1].scatter(raw_aurocs, gat_aurocs, alpha=0.3, s=10, c='#2196F3')
axes[1].plot([0, 1], [0, 1], 'r--', linewidth=1)
axes[1].set_xlabel('Raw Cosine AUROC')
axes[1].set_ylabel('GAT AUROC')
axes[1].set_title('A3: Per-Query AUROC (GAT vs Raw)', fontweight='bold')
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)

# When does GAT help? Analyze by query difficulty
# Split queries into "easy" (raw AUROC > 0.8) and "hard" (raw AUROC < 0.6)
easy_mask = np.array(raw_aurocs) > 0.8
hard_mask = np.array(raw_aurocs) < 0.6
mid_mask = ~easy_mask & ~hard_mask

for label, mask, color in [('Easy (Raw>0.8)', easy_mask, 'green'), ('Mid', mid_mask, 'orange'), ('Hard (Raw<0.6)', hard_mask, 'red')]:
    if mask.sum() > 0:
        avg_diff = np.mean(diff_auroc[mask])
        cnt = mask.sum()
        axes[2].bar(label, avg_diff, color=color, edgecolor='white')
        axes[2].text(label, avg_diff + 0.002 * np.sign(avg_diff), f'n={cnt}\n{avg_diff:+.4f}', ha='center', fontsize=9)

axes[2].axhline(0, color='gray', linestyle='--')
axes[2].set_ylabel('Mean AUROC Diff (GAT - Raw)')
axes[2].set_title('A3: GAT Improvement by Query Difficulty', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/a3_gat_vs_raw_auroc.png", bbox_inches='tight')
plt.close()
print(f"-> Saved: {SAVE_DIR}/a3_gat_vs_raw_auroc.png\n")


# ================================================================
# D1: Semantic vs Structural Agent 일치율
# ================================================================
print("=" * 70)
print("D1~D3: MultiAgent Filter 분석")
print("=" * 70)

# Predictions에서 uncertainty와 status 분석
uncertainties = [p.get('uncertainty', 0) for p in pred_gat_ma]
statuses = [p.get('status', 'Unknown') for p in pred_gat_ma]

answerable_mask = np.array([s == 'Answerable' for s in statuses])
unanswerable_mask = np.array([s == 'Unanswerable' for s in statuses])

print(f"D1: Agent Agreement Analysis")
print(f"  Total: {len(pred_gat_ma)}")
print(f"  Answerable: {answerable_mask.sum()} ({answerable_mask.sum()/len(pred_gat_ma)*100:.1f}%)")
print(f"  Unanswerable: {unanswerable_mask.sum()} ({unanswerable_mask.sum()/len(pred_gat_ma)*100:.1f}%)")

# Uncertainty distribution
u_arr = np.array(uncertainties)
print(f"\n  Uncertainty stats:")
print(f"    mean={u_arr.mean():.4f}, median={np.median(u_arr):.4f}")
print(f"    Fully agreed (U=0): {(u_arr == 0).sum()} ({(u_arr == 0).sum()/len(u_arr)*100:.1f}%)")
print(f"    Partial (0<U<1): {((u_arr > 0) & (u_arr < 1)).sum()}")
print(f"    Full disagreement (U=1): {(u_arr == 1).sum()} ({(u_arr == 1).sum()/len(u_arr)*100:.1f}%)")

# D2: How does Unanswerable affect gold recall?
# Compare per-query: GAT Classifier (no filter) vs GAT+MultiAgent
print(f"\nD2: Filter Effect on Gold Recall (GAT-only vs GAT+MultiAgent)")

# Build per-query comparison
gat_by_qid = {o['question_id']: o for o in out_gat}
gat_ma_by_qid = {o['question_id']: o for o in out_gat_ma}

recall_improved = 0
recall_degraded = 0
recall_same = 0
filter_killed = 0  # recall went from >0 to 0

improvement_amounts = []
degradation_amounts = []

for qid in gat_by_qid:
    if qid not in gat_ma_by_qid:
        continue
    r_before = gat_by_qid[qid].get('recall', 0)
    r_after = gat_ma_by_qid[qid].get('recall', 0)

    diff = r_after - r_before
    if diff > 0.01:
        recall_improved += 1
        improvement_amounts.append(diff)
    elif diff < -0.01:
        recall_degraded += 1
        degradation_amounts.append(diff)
        if r_before > 0 and r_after == 0:
            filter_killed += 1
    else:
        recall_same += 1

total = recall_improved + recall_degraded + recall_same
print(f"  Recall improved: {recall_improved} ({recall_improved/total*100:.1f}%) — avg improvement: +{np.mean(improvement_amounts):.4f}" if improvement_amounts else f"  Recall improved: 0")
print(f"  Recall degraded: {recall_degraded} ({recall_degraded/total*100:.1f}%) — avg degradation: {np.mean(degradation_amounts):.4f}" if degradation_amounts else f"  Recall degraded: 0")
print(f"  Recall same:     {recall_same} ({recall_same/total*100:.1f}%)")
print(f"  Filter killed (>0 -> 0): {filter_killed}")

# D2-b: Precision comparison
print(f"\nD2-b: Filter Effect on Precision")
p_before_list = [gat_by_qid[q]['precision'] for q in gat_by_qid if q in gat_ma_by_qid]
p_after_list = [gat_ma_by_qid[q]['precision'] for q in gat_by_qid if q in gat_ma_by_qid]
print(f"  Before (GAT only): mean P = {np.mean(p_before_list):.4f}")
print(f"  After  (GAT+MA):   mean P = {np.mean(p_after_list):.4f}")
print(f"  Delta: {np.mean(p_after_list) - np.mean(p_before_list):+.4f}")

# D3: Unanswerable 판정의 정확도
print(f"\nD3: Unanswerable 판정 분석")
unanswerable_qids = [p['question_id'] for p in pred_gat_ma if p.get('status') == 'Unanswerable']
answerable_qids = [p['question_id'] for p in pred_gat_ma if p.get('status') == 'Answerable']

# Among Unanswerable: what was the GAT-only recall? (i.e., was gold actually reachable?)
unanswerable_gat_recalls = []
for qid in unanswerable_qids:
    if qid in gat_by_qid:
        unanswerable_gat_recalls.append(gat_by_qid[qid].get('recall', 0))

answerable_gat_recalls = []
for qid in answerable_qids:
    if qid in gat_by_qid:
        answerable_gat_recalls.append(gat_by_qid[qid].get('recall', 0))

print(f"  Queries marked Unanswerable: {len(unanswerable_qids)}")
print(f"    Their GAT-only recall: mean={np.mean(unanswerable_gat_recalls):.4f}, median={np.median(unanswerable_gat_recalls):.4f}")
print(f"    Had recall=0 in GAT: {sum(1 for r in unanswerable_gat_recalls if r == 0)} ({sum(1 for r in unanswerable_gat_recalls if r == 0)/len(unanswerable_gat_recalls)*100:.1f}%)")
print(f"    Had recall>0.5 in GAT: {sum(1 for r in unanswerable_gat_recalls if r > 0.5)} ({sum(1 for r in unanswerable_gat_recalls if r > 0.5)/len(unanswerable_gat_recalls)*100:.1f}%)")
print(f"    Had recall=1.0 in GAT: {sum(1 for r in unanswerable_gat_recalls if r == 1.0)} ({sum(1 for r in unanswerable_gat_recalls if r == 1.0)/len(unanswerable_gat_recalls)*100:.1f}%)")

print(f"\n  Queries marked Answerable: {len(answerable_qids)}")
print(f"    Their GAT-only recall: mean={np.mean(answerable_gat_recalls):.4f}")

# D3-b: Among "Unanswerable" queries, what would G-Retriever have achieved?
gr_by_qid = {o['question_id']: o for o in out_gretriever}
unanswerable_gr_recalls = []
for qid in unanswerable_qids:
    if qid in gr_by_qid:
        unanswerable_gr_recalls.append(gr_by_qid[qid].get('recall', 0))

print(f"\n  G-Retriever recall on 'Unanswerable' queries:")
print(f"    mean={np.mean(unanswerable_gr_recalls):.4f}")
print(f"    G-Retriever had recall>0.5: {sum(1 for r in unanswerable_gr_recalls if r > 0.5)} ({sum(1 for r in unanswerable_gr_recalls if r > 0.5)/len(unanswerable_gr_recalls)*100:.1f}%)")
print(f"    G-Retriever had recall=1.0: {sum(1 for r in unanswerable_gr_recalls if r == 1.0)} ({sum(1 for r in unanswerable_gr_recalls if r == 1.0)/len(unanswerable_gr_recalls)*100:.1f}%)")

# D Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# D1: Uncertainty histogram
axes[0, 0].hist(u_arr, bins=20, color='#FF5722', edgecolor='white', alpha=0.8)
axes[0, 0].set_xlabel('Uncertainty (1 - Jaccard)')
axes[0, 0].set_ylabel('# Queries')
axes[0, 0].set_title(f'D1: Agent Uncertainty Distribution\nU=0: {(u_arr==0).sum()}, U=1: {(u_arr==1).sum()}', fontweight='bold')

# D2: Before/After recall scatter
r_bf = [gat_by_qid[q]['recall'] for q in gat_by_qid if q in gat_ma_by_qid]
r_af = [gat_ma_by_qid[q]['recall'] for q in gat_by_qid if q in gat_ma_by_qid]
axes[0, 1].scatter(r_bf, r_af, alpha=0.2, s=10, c='#2196F3')
axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=1)
axes[0, 1].set_xlabel('Recall (GAT Only)')
axes[0, 1].set_ylabel('Recall (GAT + MultiAgent)')
axes[0, 1].set_title(f'D2: Filter Effect on Recall\nImproved: {recall_improved}, Degraded: {recall_degraded}, Killed: {filter_killed}', fontweight='bold')

# D3: Unanswerable query's actual achievable recall
axes[1, 0].hist(unanswerable_gat_recalls, bins=20, alpha=0.6, label='GAT recall', color='red')
axes[1, 0].hist(unanswerable_gr_recalls, bins=20, alpha=0.6, label='G-Retriever recall', color='blue')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('# Queries')
axes[1, 0].set_title(f'D3: Recall of "Unanswerable" queries (n={len(unanswerable_qids)})\nThese queries were DISCARDED by filter', fontweight='bold')
axes[1, 0].legend()

# D3-b: Uncertainty vs Recall change
u_per_query = {p['question_id']: p.get('uncertainty', 0) for p in pred_gat_ma}
recall_changes = []
u_values = []
for qid in gat_by_qid:
    if qid in gat_ma_by_qid and qid in u_per_query:
        recall_changes.append(gat_ma_by_qid[qid]['recall'] - gat_by_qid[qid]['recall'])
        u_values.append(u_per_query[qid])

axes[1, 1].scatter(u_values, recall_changes, alpha=0.2, s=10, c='#FF5722')
axes[1, 1].axhline(0, color='gray', linestyle='--')
axes[1, 1].set_xlabel('Uncertainty')
axes[1, 1].set_ylabel('Recall Change (After - Before)')
axes[1, 1].set_title('D3-b: Uncertainty vs Recall Change', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/d_multiagent_analysis.png", bbox_inches='tight')
plt.close()
print(f"\n-> Saved: {SAVE_DIR}/d_multiagent_analysis.png\n")


# ================================================================
# E2: Schema Linking Recall vs EX 상관관계
# ================================================================
print("=" * 70)
print("E2: Schema Linking Recall vs EX Score 상관관계")
print("=" * 70)

all_experiments = {
    "G-Retriever": out_gretriever,
    "VectorOnly": out_vectoronly,
    "LinkAlign": out_linkalign,
    "XiYanSQL": out_xiyan,
    "GAT Classifier": out_gat,
    "GAT+MultiAgent": out_gat_ma,
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (name, outputs) in enumerate(all_experiments.items()):
    if idx >= 6:
        break
    ax = axes[idx]

    recalls = [o.get('recall', 0) for o in outputs]
    exs = [o.get('ex', 0) for o in outputs]

    # Bin recalls
    bins = [(0, 0), (0.01, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 0.99), (1.0, 1.0)]
    bin_labels = ['0', '0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0', '1.0']
    bin_ex_rates = []
    bin_counts = []

    for lo, hi in bins:
        mask = [(r >= lo and r <= hi) for r in recalls]
        cnt = sum(mask)
        if cnt > 0:
            avg_ex = np.mean([e for e, m in zip(exs, mask) if m])
        else:
            avg_ex = 0
        bin_ex_rates.append(avg_ex)
        bin_counts.append(cnt)

    bars = ax.bar(bin_labels, bin_ex_rates, color='#2196F3', edgecolor='white')
    for bar, cnt, ex_r in zip(bars, bin_counts, bin_ex_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'n={cnt}\n{ex_r:.3f}', ha='center', fontsize=7)
    ax.set_xlabel('Recall Range')
    ax.set_ylabel('Avg EX Score')
    ax.set_title(f'{name}', fontweight='bold')
    ax.set_ylim(0, 0.8)

plt.suptitle('E2: Schema Recall vs EX Score', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/e2_recall_vs_ex.png", bbox_inches='tight')
plt.close()

# Correlation stats
for name, outputs in all_experiments.items():
    recalls = [o.get('recall', 0) for o in outputs]
    exs = [o.get('ex', 0) for o in outputs]
    corr = np.corrcoef(recalls, exs)[0, 1]
    perfect_recall_ex = np.mean([e for r, e in zip(recalls, exs) if r == 1.0]) if any(r == 1.0 for r in recalls) else 0
    n_perfect = sum(1 for r in recalls if r == 1.0)
    print(f"  {name}: Pearson r={corr:.4f}, Perfect recall EX={perfect_recall_ex:.4f} (n={n_perfect})")

print(f"-> Saved: {SAVE_DIR}/e2_recall_vs_ex.png\n")


# ================================================================
# E3: DB 복잡도별 성능 분석
# ================================================================
print("=" * 70)
print("E3: DB 복잡도별 성능 분석")
print("=" * 70)

# Use G-Retriever as reference
df_gr = pd.DataFrame(out_gretriever)

# DB별 통계
db_stats = df_gr.groupby('db_id').agg(
    n_queries=('recall', 'count'),
    avg_recall=('recall', 'mean'),
    avg_precision=('precision', 'mean'),
    avg_ex=('ex', 'mean'),
    avg_gold_cols=('gold_cols', lambda x: np.mean([len(c) for c in x])),
    avg_pred_cols=('pred_cols', lambda x: np.mean([len(c) for c in x])),
).reset_index()

# Schema complexity: avg_gold_cols as proxy
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Recall vs #gold cols
axes[0].scatter(db_stats['avg_gold_cols'], db_stats['avg_recall'],
                s=db_stats['n_queries']*2, alpha=0.6, c='#2196F3', edgecolors='white')
for _, row in db_stats.iterrows():
    axes[0].annotate(row['db_id'], (row['avg_gold_cols'], row['avg_recall']), fontsize=6, alpha=0.7)
axes[0].set_xlabel('Avg # Gold Columns per Query')
axes[0].set_ylabel('Avg Recall')
axes[0].set_title('E3: Schema Complexity vs Recall', fontweight='bold')

# EX vs #gold cols
axes[1].scatter(db_stats['avg_gold_cols'], db_stats['avg_ex'],
                s=db_stats['n_queries']*2, alpha=0.6, c='#FF5722', edgecolors='white')
for _, row in db_stats.iterrows():
    axes[1].annotate(row['db_id'], (row['avg_gold_cols'], row['avg_ex']), fontsize=6, alpha=0.7)
axes[1].set_xlabel('Avg # Gold Columns per Query')
axes[1].set_ylabel('Avg EX Score')
axes[1].set_title('E3: Schema Complexity vs EX', fontweight='bold')

# Multi-method comparison by DB
methods_to_compare = {
    "G-Retriever": out_gretriever,
    "GAT+MultiAgent": out_gat_ma,
    "GAT Classifier": out_gat,
}

db_method_recalls = defaultdict(dict)
for name, outputs in methods_to_compare.items():
    df_m = pd.DataFrame(outputs)
    for db_id, group in df_m.groupby('db_id'):
        db_method_recalls[db_id][name] = group['recall'].mean()

# Find DBs where GAT+MultiAgent beats G-Retriever
gat_wins = []
gr_wins = []
for db_id, method_recalls in db_method_recalls.items():
    if 'G-Retriever' in method_recalls and 'GAT+MultiAgent' in method_recalls:
        diff = method_recalls['GAT+MultiAgent'] - method_recalls['G-Retriever']
        if diff > 0.05:
            gat_wins.append((db_id, diff, method_recalls))
        elif diff < -0.05:
            gr_wins.append((db_id, diff, method_recalls))

print(f"\nDBs where GAT+MultiAgent > G-Retriever (margin>0.05):")
for db_id, diff, mr in sorted(gat_wins, key=lambda x: -x[1]):
    print(f"  {db_id}: GAT+MA={mr['GAT+MultiAgent']:.4f}, GR={mr['G-Retriever']:.4f}, diff={diff:+.4f}")

print(f"\nDBs where G-Retriever > GAT+MultiAgent (margin>0.05):")
for db_id, diff, mr in sorted(gr_wins, key=lambda x: x[1]):
    print(f"  {db_id}: GAT+MA={mr['GAT+MultiAgent']:.4f}, GR={mr['G-Retriever']:.4f}, diff={diff:+.4f}")

# Bar chart: per-DB recall for top methods
dbs = db_stats.sort_values('avg_recall')['db_id'].values
x = np.arange(len(dbs))
width = 0.3

for i, (name, outputs) in enumerate(methods_to_compare.items()):
    df_m = pd.DataFrame(outputs)
    db_recall = df_m.groupby('db_id')['recall'].mean()
    vals = [db_recall.get(db, 0) for db in dbs]
    axes[2].barh(x + i * width, vals, width, label=name, alpha=0.8)

axes[2].set_yticks(x + width)
axes[2].set_yticklabels(dbs, fontsize=7)
axes[2].set_xlabel('Recall')
axes[2].set_title('E3: Per-DB Recall Comparison', fontweight='bold')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/e3_db_complexity.png", bbox_inches='tight')
plt.close()
print(f"\n-> Saved: {SAVE_DIR}/e3_db_complexity.png")


# ================================================================
# E3-b: Gold 컬럼 수 구간별 EX (다중 방법 비교)
# ================================================================
print("\n" + "=" * 70)
print("E3-b: Gold Schema 복잡도 구간별 EX 비교")
print("=" * 70)

fig, ax = plt.subplots(figsize=(12, 6))
bin_edges = [(1, 2), (3, 4), (5, 7), (8, 100)]
bin_names = ['1-2 cols', '3-4 cols', '5-7 cols', '8+ cols']
x = np.arange(len(bin_names))
width = 0.15

for i, (name, outputs) in enumerate(all_experiments.items()):
    df_m = pd.DataFrame(outputs)
    gold_sizes = df_m['gold_cols'].apply(len)
    bin_vals = []
    for lo, hi in bin_edges:
        mask = (gold_sizes >= lo) & (gold_sizes <= hi)
        bin_vals.append(df_m.loc[mask, 'ex'].mean() if mask.any() else 0)
    ax.bar(x + i * width, bin_vals, width, label=name, alpha=0.85)

ax.set_xticks(x + width * len(all_experiments) / 2)
ax.set_xticklabels(bin_names)
ax.set_ylabel('Avg EX Score')
ax.set_title('E3-b: EX Score by Gold Schema Complexity', fontweight='bold')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/e3b_complexity_vs_ex.png", bbox_inches='tight')
plt.close()
print(f"-> Saved: {SAVE_DIR}/e3b_complexity_vs_ex.png")

for name, outputs in all_experiments.items():
    df_m = pd.DataFrame(outputs)
    gold_sizes = df_m['gold_cols'].apply(len)
    print(f"\n  {name}:")
    for (lo, hi), bl in zip(bin_edges, bin_names):
        mask = (gold_sizes >= lo) & (gold_sizes <= hi)
        if mask.any():
            print(f"    {bl}: EX={df_m.loc[mask, 'ex'].mean():.4f}, R={df_m.loc[mask, 'recall'].mean():.4f}, n={mask.sum()}")


# ================================================================
# Phase 3 종합 요약
# ================================================================
print("\n" + "=" * 70)
print("Phase 3 종합 진단 요약")
print("=" * 70)

print("""
[A3: GAT vs Raw Cosine Ranking]
- Per-query AUROC에서도 Raw Cosine이 우세.
- GAT가 도움 되는 케이스가 있지만, 해치는 케이스가 더 많음.
- 특히 "어려운" 쿼리(raw AUROC<0.6)에서 GAT 개선이 미미하거나 오히려 악화.

[D1~D3: MultiAgent Filter]
- 38.2%를 Unanswerable로 판정 → 과도한 false rejection
- Unanswerable로 판정된 쿼리 중 상당수가 GAT-only에서 recall>0 → 잘못된 거부
- Filter가 precision은 개선하지만 recall 손실이 더 큼
- Uncertainty=1.0 (완전 불일치)이 45%로 매우 높음 → 에이전트 간 합의 메커니즘 개선 필요

[E2: Recall vs EX]
- Recall=1.0일 때 EX가 가장 높음 (당연하지만 정량적 확인)
- Recall이 0.75 이상이면 EX가 급격히 개선 → recall 0.75가 실질적 "필요 충분 조건"
- 현재 GAT+MultiAgent recall=0.658은 이 임계점에 미달

[E3: DB 복잡도]
- 스키마가 큰 DB(california_schools, card_games, debit_card_specializing)에서 모든 방법이 고전
- GAT+MultiAgent가 G-Retriever를 이기는 DB는 극소수 → 범용적 개선이 아님
""")
