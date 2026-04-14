# Selector Score Analysis

## Data Sources
- **Cosine model**: `experiment_b0_raw_pcst_baseline` (142,071 node-scores, 1534 queries)
- **Ensemble model**: `experiment_b2_ensemble` (142,071 node-scores, 1534 queries)
- Ensemble formula: `0.85 * cosine + 0.15 * GAT`

---
## Analysis 1: Score Discrimination (ROC-AUC / PR-AUC)

### Global Metrics (all nodes pooled)

| Metric | Cosine | Ensemble | Delta |
|--------|--------|----------|-------|
| ROC-AUC | 0.7409 | 0.7762 | 0.0353 |
| PR-AUC  | 0.2434 | 0.3172 | 0.0738 |

### Per-Query Metrics (macro-averaged)

| Metric | Cosine (mean +/- std) | Ensemble (mean +/- std) | Delta (mean) |
|--------|-----------------------|-------------------------|--------------|
| ROC-AUC | 0.7811 +/- 0.1584 | 0.7998 +/- 0.1485 | 0.0186 |
| PR-AUC  | 0.4641 +/- 0.2177 | 0.4793 +/- 0.2207 | 0.0152 |

- Queries evaluated: 1534 (skipped 0 with all-gold or no-gold)

---
## Analysis 2: Score Distribution & PCST Prize Impact (P80 Threshold)

### Score Distribution: Gold Nodes

| Stat | Cosine | Ensemble |
|------|--------|----------|
| min | -0.0825 | 0.0001 |
| p10 | 0.1236 | 0.3217 |
| p25 | 0.1971 | 0.4685 |
| median | 0.2888 | 0.6453 |
| mean | 0.2902 | 0.6321 |
| p75 | 0.3802 | 0.8130 |
| p80 | 0.4014 | 0.8500 |
| p90 | 0.4583 | 0.9274 |
| p95 | 0.5034 | 0.9841 |
| max | 0.7222 | 1.0000 |
| std | 0.1279 | 0.2232 |

### Score Distribution: Non-Gold Nodes

| Stat | Cosine | Ensemble |
|------|--------|----------|
| min | -0.1460 | 0.0000 |
| p10 | 0.0525 | 0.1657 |
| p25 | 0.1075 | 0.2604 |
| median | 0.1758 | 0.3896 |
| mean | 0.1825 | 0.4048 |
| p75 | 0.2504 | 0.5365 |
| p80 | 0.2693 | 0.5723 |
| p90 | 0.3207 | 0.6661 |
| p95 | 0.3636 | 0.7409 |
| max | 0.7144 | 1.0000 |
| std | 0.1042 | 0.1908 |

### Gold Nodes Above P80 Threshold (Positive PCST Prize)

| Metric | Cosine | Ensemble | Delta |
|--------|--------|----------|-------|
| Gold nodes above P80 (global) | 5715/10252 (55.7%) | 5929/10252 (57.8%) | 2.1% |
| Per-query rate (mean +/- std) | 59.3% +/- 24.9% | 61.4% +/- 24.8% | 2.0% |
| Per-query rate (median) | 57.1% | 61.5% | 4.4% |
| Queries where ALL gold above P80 | 222/1534 (14.5%) | 241/1534 (15.7%) | 1.2% |

> **Interpretation**: P80 threshold means top 20% of nodes per query get positive PCST prize.
> A higher 'gold above P80' rate means the scoring method better separates gold nodes from non-gold.

---
## Analysis 3: GAT Marginal Contribution (Ensemble vs Cosine at P80 Threshold)

Using per-method P80 thresholds (cosine P80 for cosine, ensemble P80 for ensemble).

### Threshold Crossing Summary

| Category | Count | % of Gold |
|----------|-------|-----------|
| GAT rescued (cos < P80, ens >= P80) | 544 | 5.3% |
| GAT hurt (cos >= P80, ens < P80) | 330 | 3.2% |
| Neutral: both above P80 | 5385 | 52.5% |
| Neutral: both below P80 | 3993 | 38.9% |
| **Total gold nodes** | **10252** | **100%** |

- Net rescued: 214 gold nodes (2.1%)

### Score Characteristics of Rescued vs Hurt Nodes

| Metric | GAT Rescued | GAT Hurt |
|--------|-------------|----------|
| Mean cosine score | 0.2518 | 0.2811 |
| Mean ensemble score | 0.6230 | 0.5497 |
| Mean implied GAT score | 2.7265 | 2.0720 |
| Median implied GAT score | 2.7268 | 2.0759 |

### Per-Query Breakdown

| Metric | Value |
|--------|-------|
| Total queries analyzed | 1534 |
| Queries where GAT rescued >= 1 gold node | 443 (28.9%) |
| Queries where GAT hurt >= 1 gold node | 292 (19.0%) |
| Mean rescued per query (among all queries) | 0.35 |
| Mean rescued per query (among queries with rescue) | 1.23 |

### Score Separation: Gold vs Non-Gold

| Metric | Cosine | Ensemble |
|--------|--------|----------|
| Mean gold score | 0.2902 | 0.6321 |
| Mean non-gold score | 0.1825 | 0.4048 |
| Gap (gold - non-gold) | 0.1077 | 0.2273 |
