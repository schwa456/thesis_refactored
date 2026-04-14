# Enriched GAT 실험 결과 정리

## 1. 실험 개요

기존 HeteroGraphBuilder는 노드 텍스트를 단순하게 구성:
- Table: `"Table: frpm"`
- Column: `"Column: CDSCode in table frpm, type TEXT. Example values: ..."`

EnrichedHeteroGraphBuilder는 database_description/*.csv와 dev_tables.json에서 추가 메타정보를 주입:
- Table: `"Table: frpm (Free or Reduced Price Meals)"`
- Column: `"Column: CDSCode in table frpm, type TEXT. Meaning: California Department of Schools code. Description: unique identifier for schools. Values info: concatenated county, district, school codes. Example values: ..."`

이 enriched features로 GAT를 재학습(300 epochs, BCE+InfoNCE)하고, 전체 파이프라인(Ensemble+AdaptivePCST+XiYanFilter)에서 평가.

## 2. GAT 학습 결과

| Model | Best Val Recall@15 | Checkpoint |
|---|---|---|
| Original GAT | ~0.510 | best_gat_model.pt |
| Enriched GAT | 0.5119 | best_gat_enriched.pt |

GAT 단독 Recall@15는 거의 동등하나, enriched features가 cosine similarity 품질을 높여 Ensemble 점수에서 차이 발생.

## 3. 전체 파이프라인 성능 비교 (BIRD-Dev 1,534 queries)

### 3.1 Overall

| Method | Recall | Precision | F1 | Avg Cols |
|---|---|---|---|---|
| G-Retriever | 0.7577 | 0.7866 | 0.7719 | 3.7 |
| LinkAlign | 0.6940 | 0.7641 | 0.7274 | 3.4 |
| XiYanSQL | 0.6832 | 0.7408 | 0.7108 | 3.4 |
| **Proposed (Original GAT)** | 0.6244 | 0.7930 | 0.6987 | 3.2 |
| **Proposed (Enriched GAT)** | **0.6658** | **0.8147** | **0.7328** | 3.3 |
| EXP-A (Cos+Adaptive+XiYan) | 0.5835 | 0.7829 | 0.6687 | 2.9 |
| EXP-B (Ens+Basic+XiYan) | 0.8149 | 0.7597 | 0.7863 | 18.4 |
| EXP-C (Cos+Basic+XiYan) | 0.7987 | 0.7694 | 0.7838 | 15.4 |

### 3.2 Enriched GAT 개선폭 (vs Original GAT)

| Metric | Original | Enriched | Delta |
|---|---|---|---|
| Recall | 0.6244 | 0.6658 | **+4.14%p** |
| Precision | 0.7930 | 0.8147 | **+2.17%p** |
| F1 | 0.6987 | 0.7328 | **+3.41%p** |

### 3.3 난이도별 성능

#### Simple (N=925)

| Method | Recall | Precision | F1 |
|---|---|---|---|
| G-Retriever | 0.7698 | 0.8064 | 0.7877 |
| LinkAlign | 0.7288 | 0.7960 | 0.7609 |
| XiYanSQL | 0.7192 | 0.7800 | 0.7484 |
| Proposed (Original GAT) | 0.6563 | 0.8016 | 0.7217 |
| **Proposed (Enriched GAT)** | **0.6980** | **0.8182** | **0.7533** |
| EXP-B (Ens+Basic+XiYan) | 0.8209 | 0.7715 | 0.7954 |

#### Moderate (N=464)

| Method | Recall | Precision | F1 |
|---|---|---|---|
| G-Retriever | 0.7441 | 0.7605 | 0.7522 |
| LinkAlign | 0.6448 | 0.7192 | 0.6800 |
| XiYanSQL | 0.6275 | 0.6827 | 0.6540 |
| Proposed (Original GAT) | 0.5916 | 0.7842 | 0.6744 |
| **Proposed (Enriched GAT)** | **0.6246** | **0.8106** | **0.7056** |
| EXP-B (Ens+Basic+XiYan) | 0.7984 | 0.7381 | 0.7671 |

#### Challenging (N=145)

| Method | Recall | Precision | F1 |
|---|---|---|---|
| G-Retriever | 0.7246 | 0.7437 | 0.7340 |
| LinkAlign | 0.6296 | 0.7045 | 0.6649 |
| XiYanSQL | 0.6318 | 0.6759 | 0.6531 |
| Proposed (Original GAT) | 0.5261 | 0.7667 | 0.6240 |
| **Proposed (Enriched GAT)** | **0.5916** | **0.8060** | **0.6823** |
| EXP-B (Ens+Basic+XiYan) | 0.8289 | 0.7540 | 0.7897 |

## 4. 분석

1. **Enriched GAT는 전 난이도에서 Original GAT 대비 일관적 개선**: Challenging에서 Recall +6.55%p, F1 +5.83%p로 가장 큰 폭 개선.
2. **Precision이 0.8147로 모든 실험 중 최고**: Column description 주입으로 노드 임베딩의 의미 표현력이 향상되어, PCST 단계에서 더 정확한 노드에 높은 점수가 부여됨.
3. **Recall 갭은 여전히 존재**: G-Retriever(0.7577) 대비 0.6658로 약 9%p 차이. Adaptive PCST의 percentile-based pruning이 주 원인.
4. **Avg Cols 3.3으로 타이트한 선택 유지**: EXP-B(18.4), EXP-C(15.4) 대비 5~6배 적은 컬럼 수로 높은 Precision 달성.
