import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

PROJ_OUTPUT = "outputs/experiments/experiment_qcond_idea24_a0_xiyan/output_qcond_idea24_a0_xiyan.jsonl"
DIRECT_OUTPUT = "outputs/experiments/experiment_qcond_direct_idea24_xiyan/output_qcond_direct_idea24_xiyan.jsonl"
PROJ_SCORE = "outputs/experiments/experiment_qcond_idea24_a0_xiyan/score_analysis_qcond_idea24_a0_xiyan.jsonl"
DIRECT_SCORE = "outputs/experiments/experiment_qcond_direct_idea24_xiyan/score_analysis_qcond_direct_idea24_xiyan.jsonl"

def load_jsonl(path):
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

proj_out = {r['question_id']: r for r in load_jsonl(PROJ_OUTPUT) if 'recall' in r}
direct_out = {r['question_id']: r for r in load_jsonl(DIRECT_OUTPUT) if 'recall' in r}

proj_scores_raw = load_jsonl(PROJ_SCORE)
direct_scores_raw = load_jsonl(DIRECT_SCORE)

# Build per-(qid, node_name) score maps
proj_score_map = {}
for r in proj_scores_raw:
    proj_score_map[(r['query_id'], r['node_name'])] = (r['score'], r['is_gold'])

direct_score_map = {}
for r in direct_scores_raw:
    direct_score_map[(r['query_id'], r['node_name'])] = (r['score'], r['is_gold'])

# Extract gold/non-gold scores
proj_gold_scores, proj_nongold_scores = [], []
direct_gold_scores, direct_nongold_scores = [], []

for (qid, name), (score, is_gold) in proj_score_map.items():
    if is_gold:
        proj_gold_scores.append(score)
    else:
        proj_nongold_scores.append(score)

for (qid, name), (score, is_gold) in direct_score_map.items():
    if is_gold:
        direct_gold_scores.append(score)
    else:
        direct_nongold_scores.append(score)

# Paired gold node scores
paired_proj, paired_direct = [], []
for key, (p_score, p_gold) in proj_score_map.items():
    if p_gold and key in direct_score_map:
        d_score, d_gold = direct_score_map[key]
        if d_gold:
            paired_proj.append(p_score)
            paired_direct.append(d_score)

paired_proj = np.array(paired_proj)
paired_direct = np.array(paired_direct)

# Recall arrays
common_qids = sorted(set(proj_out.keys()) & set(direct_out.keys()))
proj_recalls = np.array([proj_out[qid]['recall'] for qid in common_qids])
direct_recalls = np.array([direct_out[qid]['recall'] for qid in common_qids])

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Direct (BCE only) vs Projector (BCE+InfoNCE): Why Direct Fails', fontsize=15, fontweight='bold', y=0.98)

# ===== Plot 1: Gold Node Score Distribution =====
ax = axes[0, 0]
bins = np.linspace(0, 1, 21)
ax.hist(proj_gold_scores, bins=bins, alpha=0.6, label=f'Projector (n={len(proj_gold_scores)})', color='#2196F3', edgecolor='white')
ax.hist(direct_gold_scores, bins=bins, alpha=0.6, label=f'Direct (n={len(direct_gold_scores)})', color='#F44336', edgecolor='white')
ax.set_xlabel('Score')
ax.set_ylabel('Count')
ax.set_title('(a) Gold Node Score Distribution')
ax.legend(fontsize=9)
ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='threshold')

# ===== Plot 2: Score Bucket Comparison =====
ax = axes[0, 1]
buckets = ['< 0.2', '0.2-0.5', '0.5-0.8', '>= 0.8']
proj_pcts = [
    np.mean(np.array(proj_gold_scores) < 0.2) * 100,
    np.mean((np.array(proj_gold_scores) >= 0.2) & (np.array(proj_gold_scores) < 0.5)) * 100,
    np.mean((np.array(proj_gold_scores) >= 0.5) & (np.array(proj_gold_scores) < 0.8)) * 100,
    np.mean(np.array(proj_gold_scores) >= 0.8) * 100,
]
direct_pcts = [
    np.mean(np.array(direct_gold_scores) < 0.2) * 100,
    np.mean((np.array(direct_gold_scores) >= 0.2) & (np.array(direct_gold_scores) < 0.5)) * 100,
    np.mean((np.array(direct_gold_scores) >= 0.5) & (np.array(direct_gold_scores) < 0.8)) * 100,
    np.mean(np.array(direct_gold_scores) >= 0.8) * 100,
]
x = np.arange(len(buckets))
w = 0.35
bars1 = ax.bar(x - w/2, proj_pcts, w, label='Projector', color='#2196F3', edgecolor='white')
bars2 = ax.bar(x + w/2, direct_pcts, w, label='Direct', color='#F44336', edgecolor='white')
for bar, val in zip(bars1, proj_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, direct_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(buckets)
ax.set_ylabel('% of Gold Nodes')
ax.set_title('(b) Gold Node Score Buckets')
ax.legend(fontsize=9)

# ===== Plot 3: Paired Score Scatter =====
ax = axes[0, 2]
rescued = (paired_proj > 0.5) & (paired_direct < 0.2)
normal = ~rescued
ax.scatter(paired_proj[normal], paired_direct[normal], alpha=0.08, s=8, c='gray', rasterized=True)
ax.scatter(paired_proj[rescued], paired_direct[rescued], alpha=0.4, s=12, c='#F44336', label=f'Rescued by InfoNCE ({rescued.sum()})', rasterized=True)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
ax.set_xlabel('Projector Score (BCE+InfoNCE)')
ax.set_ylabel('Direct Score (BCE only)')
ax.set_title(f'(c) Paired Gold Node Scores (n={len(paired_proj)})')
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

# ===== Plot 4: Per-Query Recall Comparison =====
ax = axes[1, 0]
ax.scatter(proj_recalls, direct_recalls, alpha=0.15, s=10, c='#607D8B', rasterized=True)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
proj_better = np.sum(proj_recalls > direct_recalls)
direct_better = np.sum(direct_recalls > proj_recalls)
tied = np.sum(proj_recalls == direct_recalls)
ax.set_xlabel('Projector Recall')
ax.set_ylabel('Direct Recall')
ax.set_title(f'(d) Per-Query Recall (Proj>{proj_better}, Direct>{direct_better}, Tied={tied})')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

# ===== Plot 5: Recall Distribution =====
ax = axes[1, 1]
recall_bins = np.linspace(0, 1, 11)
ax.hist(proj_recalls, bins=recall_bins, alpha=0.6, label='Projector', color='#2196F3', edgecolor='white')
ax.hist(direct_recalls, bins=recall_bins, alpha=0.6, label='Direct', color='#F44336', edgecolor='white')
ax.set_xlabel('Recall')
ax.set_ylabel('# Queries')
ax.set_title('(e) Per-Query Recall Distribution')
ax.legend(fontsize=9)

# ===== Plot 6: Recall Delta by Gold Node Count =====
ax = axes[1, 2]
delta_by_gold = defaultdict(list)
for qid in common_qids:
    p = proj_out[qid]
    d = direct_out[qid]
    n_gold = len(p.get('gold_tables', [])) + len(p.get('gold_cols', []))
    delta = p['recall'] - d['recall']
    bucket = min(n_gold, 15)
    delta_by_gold[bucket].append(delta)

buckets_sorted = sorted(delta_by_gold.keys())
means = [np.mean(delta_by_gold[b]) for b in buckets_sorted]
stds = [np.std(delta_by_gold[b]) / np.sqrt(len(delta_by_gold[b])) for b in buckets_sorted]
counts = [len(delta_by_gold[b]) for b in buckets_sorted]
labels = [str(b) if b < 15 else '15+' for b in buckets_sorted]

bars = ax.bar(range(len(buckets_sorted)), means, yerr=stds, capsize=3, color='#FF9800', edgecolor='white', alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(range(len(buckets_sorted)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_xlabel('# Gold Nodes in Query')
ax.set_ylabel('Recall Delta (Proj - Direct)')
ax.set_title('(f) Projector Advantage by Query Complexity')
for i, (m, c) in enumerate(zip(means, counts)):
    ax.text(i, max(m, 0) + 0.01, f'n={c}', ha='center', va='bottom', fontsize=7, color='gray')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('outputs/analysis/direct_vs_projector.png', dpi=150, bbox_inches='tight')
plt.savefig('outputs/analysis/direct_vs_projector.pdf', bbox_inches='tight')
print("Saved to outputs/analysis/direct_vs_projector.png and .pdf")
