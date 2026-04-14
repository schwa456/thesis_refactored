# Per-Stage Failure Analysis (Corrected Pipeline)

**Pipeline stages**: Score computation -> PCST extraction (uses ALL node scores) -> Filter (optional)

**FN categories**:
- **PCST X**: gold node not in PCST output (determined from no-filter counterpart)
- **Filter X**: gold node in PCST output but removed by XiYan filter

## Table 1 -- Per-Stage Failure Breakdown

| Model | Gold | TP | FP | FN | PCST X | Filter X | Recall | Precision | F1 |
|-------|-----:|---:|---:|---:|-------:|---------:|-------:|----------:|---:|
| #1 C+B+N | 10252 | 6666 | 92094 | 3586 | 3586 | 0 | 0.6502 | 0.0675 | 0.1223 |
| #2 C+B+X | 10252 | 5093 | 23335 | 5159 | 3573 | 1586 | 0.4968 | 0.1792 | 0.2633 |
| #3 C+A+N | 10252 | 4303 | 15747 | 5949 | 5949 | 0 | 0.4197 | 0.2146 | 0.2840 |
| #4 C+A+X | 10252 | 3542 | 1422 | 6710 | 5896 | 814 | 0.3455 | 0.7135 | 0.4656 |
| #5 E+B+N | 10252 | 6883 | 113787 | 3369 | 3369 | 0 | 0.6714 | 0.0570 | 0.1051 |
| #6 E+B+X | 10252 | 5270 | 29235 | 4982 | 3367 | 1615 | 0.5140 | 0.1527 | 0.2355 |
| #7 E+A+N | 10252 | 4894 | 22368 | 5358 | 5358 | 0 | 0.4774 | 0.1795 | 0.2609 |
| #8 E+A+X | 10252 | 3914 | 1694 | 6338 | 5325 | 1013 | 0.3818 | 0.6979 | 0.4936 |

## Table 2 -- Score Distribution by Category

| Model | TP mean | TP median | FN:PCST X mean | FN:PCST X median | FN:Filter X mean | FN:Filter X median |
|-------|--------:|----------:|---------------:|-----------------:|-----------------:|-------------------:|
| #1 C+B+N | 0.3012 | 0.2970 | 0.2697 | 0.2661 | 0.0000 | 0.0000 |
| #2 C+B+X | 0.3071 | 0.3043 | 0.2705 | 0.2666 | 0.2801 | 0.2710 |
| #3 C+A+N | 0.3325 | 0.3372 | 0.2596 | 0.2482 | 0.0000 | 0.0000 |
| #4 C+A+X | 0.3362 | 0.3410 | 0.2598 | 0.2486 | 0.3099 | 0.3166 |
| #5 E+B+N | 0.6351 | 0.6481 | 0.6260 | 0.6337 | 0.0000 | 0.0000 |
| #6 E+B+X | 0.6526 | 0.6678 | 0.6264 | 0.6337 | 0.5774 | 0.5848 |
| #7 E+A+N | 0.6980 | 0.7206 | 0.5720 | 0.5593 | 0.0000 | 0.0000 |
| #8 E+A+X | 0.7108 | 0.7375 | 0.5730 | 0.5600 | 0.6394 | 0.6513 |

## Table 3 -- PCST Recovery Analysis

For PCST X (false negative) nodes: what fraction have scores above various fractions of the PCST threshold?

- Basic PCST: fixed node_threshold = 0.1
- Adaptive PCST: per-query P80 of all node scores

| Model | N(PCST X) | Mean threshold | >=25% thr | >=50% thr | >=75% thr | >=100% thr | >=125% thr | >=150% thr |
|-------|----------:|---------------:|------------:|------------:|------------:|------------:|------------:|------------:|
| #1 C+B+N | 3586 | 0.1000 | 3487/3586 (97.2%) | 3391/3586 (94.6%) | 3265/3586 (91.0%) | 3106/3586 (86.6%) | 2920/3586 (81.4%) | 2735/3586 (76.3%) |
| #2 C+B+X | 3573 | 0.1000 | 3478/3573 (97.3%) | 3384/3573 (94.7%) | 3259/3573 (91.2%) | 3106/3573 (86.9%) | 2920/3573 (81.7%) | 2735/3573 (76.5%) |
| #3 C+A+N | 5949 | 0.2802 | 5713/5949 (96.0%) | 5134/5949 (86.3%) | 4112/5949 (69.1%) | 2211/5949 (37.2%) | 1083/5949 (18.2%) | 546/5949 (9.2%) |
| #4 C+A+X | 5896 | 0.2802 | 5663/5896 (96.0%) | 5084/5896 (86.2%) | 4071/5896 (69.0%) | 2203/5896 (37.4%) | 1083/5896 (18.4%) | 546/5896 (9.3%) |
| #5 E+B+N | 3369 | 0.1000 | 3361/3369 (99.8%) | 3357/3369 (99.6%) | 3352/3369 (99.5%) | 3335/3369 (99.0%) | 3328/3369 (98.8%) | 3297/3369 (97.9%) |
| #6 E+B+X | 3367 | 0.1000 | 3360/3367 (99.8%) | 3356/3367 (99.7%) | 3352/3367 (99.6%) | 3335/3367 (99.0%) | 3328/3367 (98.8%) | 3297/3367 (97.9%) |
| #7 E+A+N | 5358 | 0.6043 | 5232/5358 (97.6%) | 4743/5358 (88.5%) | 3754/5358 (70.1%) | 2107/5358 (39.3%) | 1172/5358 (21.9%) | 564/5358 (10.5%) |
| #8 E+A+X | 5325 | 0.6046 | 5201/5325 (97.7%) | 4715/5325 (88.5%) | 3736/5325 (70.2%) | 2104/5325 (39.5%) | 1172/5325 (22.0%) | 564/5325 (10.6%) |

## Table 4 -- Ensemble vs Cosine Scoring Quality

Fraction of gold nodes ranked in top-20 by score per query (even though top-k is not used as a gate).

| Model | Scoring | Gold total | Gold in top-20 | Fraction |
|-------|---------|----------:|--------------:|---------:|
| #1 C+B+N | Cosine | 10252 | 6540 | 0.6379 |
| #2 C+B+X | Cosine | 10252 | 6540 | 0.6379 |
| #3 C+A+N | Cosine | 10252 | 6540 | 0.6379 |
| #4 C+A+X | Cosine | 10252 | 6540 | 0.6379 |
| #5 E+B+N | Ensemble | 10252 | 6712 | 0.6547 |
| #6 E+B+X | Ensemble | 10252 | 6712 | 0.6547 |
| #7 E+A+N | Ensemble | 10252 | 6712 | 0.6547 |
| #8 E+A+X | Ensemble | 10252 | 6712 | 0.6547 |

### Summary by Scoring Method

Note: Since PCST and Filter do not affect scoring, we also show per-scoring-method stats from one representative model each:

- **Cosine** (#1 C+B+N): 6540/10252 = 63.8% of gold nodes in top-20
- **Ensemble** (#5 E+B+N): 6712/10252 = 65.5% of gold nodes in top-20

### Gold Node Score Statistics by Scoring Method

| Scoring | Model | Gold mean score | Gold median score | Non-gold mean | Non-gold median |
|---------|-------|----------------:|------------------:|--------------:|----------------:|
| Cosine | #1 C+B+N | 0.2902 | 0.2888 | 0.1825 | 0.1758 |
| Ensemble | #5 E+B+N | 0.6321 | 0.6453 | 0.4048 | 0.3896 |
