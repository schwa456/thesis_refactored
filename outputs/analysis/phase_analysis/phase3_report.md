# Phase 3: 모듈 간 상호작용 분석 보고서

## A3: GAT vs Raw Cosine — Per-Query Ranking 능력

### Per-Query AUROC (n=1,534 queries)

| Metric | GAT Classifier | Raw Cosine |
|--------|---------------|------------|
| Mean AUROC | 0.692 | **0.780** |
| Median AUROC | 0.713 | **0.812** |
| GAT better | 512 (33.4%) | — |
| Raw better | — | **973 (63.4%)** |
| Tie | 49 (3.2%) | — |

### 쿼리 난이도별 GAT 기여

| 난이도 | 기준 | GAT-Raw 차이 |
|--------|------|-------------|
| Easy | Raw AUROC > 0.8 | GAT가 오히려 악화 |
| Medium | 0.6~0.8 | 미미한 차이 |
| Hard | Raw AUROC < 0.6 | GAT 개선 미미 또는 악화 |

**결론**: GAT는 **63.4%의 쿼리에서 Raw Cosine보다 ranking을 악화**시킴. 어떤 난이도 구간에서도 일관된 개선이 없음. 현재 GAT의 message passing + DualTower 구조는 schema linking ranking에 부적합.

---

## D1~D3: MultiAgent Filter 심층 분석

### D1: 에이전트 합의 현황

| 합의 수준 | 쿼리 수 | 비율 |
|-----------|---------|------|
| 완전 합의 (U=0) | 417 | 27.2% |
| 부분 합의 (0<U<1) | 427 | 27.8% |
| **완전 불일치 (U=1)** | **690** | **45.0%** |

에이전트 간 **45%가 완전 불일치** -> Semantic/Structural 에이전트의 판단 기준이 크게 다름. 합의 메커니즘이 효과적으로 작동하지 않음.

### D2: Filter가 Recall에 미치는 영향

| 변화 유형 | 쿼리 수 | 비율 | 평균 변화량 |
|-----------|---------|------|-----------|
| Recall 개선 | 683 | 44.5% | +0.368 |
| Recall 악화 | 216 | 14.1% | -0.390 |
| Recall 동일 | 635 | 41.4% | — |
| **Filter 완전 삭제 (>0->0)** | **80** | **5.2%** | — |

- Precision: +0.065 개선 (0.620 -> 0.685)
- **Recall 개선 건의 평균이 +0.368로 크지만, 악화 건도 -0.390으로 거의 대칭**
- 80건에서 recall이 ���전 소멸 -> filter가 유효한 스키마를 전부 제거

### D3: "Unanswerable" 판정의 위험성 (핵심 문제)

| 지표 | 값 |
|------|-----|
| Unanswerable 판정 | 586건 (38.2%) |
| 이 중 GAT recall > 0 | 495건 (84.5%) |
| 이 중 GAT recall > 0.5 | 233건 (39.8%) |
| 이 중 GAT recall = 1.0 | 73건 (12.5%) |
| **G-Retriever recall on these** | **mean=0.725** |
| G-Retriever recall > 0.5 on these | **431건 (73.5%)** |
| G-Retriever recall = 1.0 on these | **235건 (40.1%)** |

**치명적 문제**: Filter가 Unanswerable로 판정한 586건 중, G-Retriever 기준으로 **73.5%가 recall>0.5, 40.1%가 perfect recall**. 즉 filter가 **실제로 풀 수 있는 쿼리의 대다수를 잘못 버리고 있음**.

---

## E2: Schema Recall vs EX 상관관계

### Recall 구간별 EX Score (G-Retriever)

| Recall Range | Avg EX | n |
|-------------|--------|---|
| 0 (complete miss) | 0.000 | 71 |
| 0~0.25 | 0.014 | 40 |
| 0.25~0.50 | 0.053 | 105 |
| 0.50~0.75 | 0.088 | 253 |
| 0.75~1.0 | 0.182 | 369 |
| **1.0 (perfect)** | **0.417** | **696** |

- Pearson r = 0.315 (중간 수준 양의 상관)
- **Recall=1.0에서 EX=0.417** -> perfect schema에서도 EX는 42%에 불과
  - 이는 SQL 생성(Generator)의 한계를 의미
  - Schema Linking이 완벽해도 EX의 상한은 ~42%
- **Recall >= 0.75 이상이면 EX가 급격히 개선** -> recall 0.75가 실용적 목표선

### 방법별 Perfect Recall EX

| Method | Perfect Recall n | EX@perfect |
|--------|-----------------|------------|
| G-Retriever | 696 | 0.417 |
| LinkAlign | 548 | 0.409 |
| VectorOnly | 524 | 0.378 |
| XiYanSQL | 570 | 0.388 |
| GAT+MultiAgent | 455 | 0.365 |
| GAT Classifier | 221 | 0.362 |

---

## E3: DB 복잡도별 성능

### G-Retriever가 GAT+MultiAgent를 크게 이기는 DB

| DB | GAT+MA Recall | GR Recall | Gap |
|----|--------------|-----------|-----|
| european_football_2 | 0.498 | 0.728 | -0.230 |
| california_schools | 0.421 | 0.643 | -0.223 |
| debit_card_specializing | 0.360 | 0.575 | -0.214 |
| superhero | 0.631 | 0.832 | -0.201 |
| codebase_community | 0.678 | 0.781 | -0.104 |

**GAT+MultiAgent가 G-Retriever를 이기는 DB는 0개.**

### Gold Schema 복잡도 구간별 EX

| #Gold Cols | G-Retriever EX | GAT+MA EX | VectorOnly EX |
|-----------|---------------|-----------|---------------|
| 1-2 | 0.358 | 0.240 | 0.323 |
| 3-4 | 0.249 | 0.140 | 0.178 |
| 5-7 | 0.193 | 0.125 | 0.104 |
| 8+ | 0.100 | 0.050 | 0.000 |

- 모든 복잡도 구간에서 G-Retriever가 최고
- 복잡도가 높아질수록 모든 방법의 EX가 급락
- 흥미: GAT+MA는 5-7/8+ 구간에서 VectorOnly보다 우수 -> 복잡한 스키마에서 filter가 약간 도움

---

## Phase 3 종합 진단

### 현 시스템의 근본 문제 3가지

**1. GAT의 Ranking 실패**
- Raw Cosine (all-MiniLM-L6-v2)의 ranking이 GAT보다 우수
- Message passing이 non-gold 노드까지 활성화하여 ranking 오염
- InfoNCE + BCE 학습이 contrastive ranking 목적에 부적합

**2. MultiAgent Filter의 과도한 Rejection**
- 38.2%를 Unanswerable로 판정하나, 이 중 73.5%가 실제 solvable
- Precision +6.5%p 개선 vs Recall -11%p 손실 -> 순손실
- 에이전트 간 45% 완전 불일치 -> 합의 메커니즘 실패

**3. EX 상한 제약**
- Perfect recall에서도 EX=42% -> SQL Generator 품질이 EX의 병목
- Schema Linking만으로는 EX를 0.5 이상 끌어올리기 어려움

### 개선 전략 최종 정리 (Phase 1~3 종합)

| 순위 | 개선안 | 기대 효과 | 난이도 |
|------|-------|----------|-------|
| 1 | **GAT 폐기, Raw Cosine backbone 채택** | 즉시 recall +10%p, AUROC +9%p | 낮음 |
| 2 | **Filter를 "pruning only" 모드로 전환** (Unanswerable 판정 제거) | recall 손실 방지 | 낮음 |
| 3 | **Raw Cosine + PCST + Pruning Filter** (G-Retriever + Filter) | recall 0.76 유지 + precision 개선 | 중간 |
| 4 | **Adaptive PCST threshold** (per-query percentile) | subgraph 크기 최적화 | 중간 |
| 5 | **Score Ensemble** (0.85*raw + 0.15*gat) | F1 +4%p (보조적) | 낮음 |
| 6 | **SQL Generator 업그레이드** | EX 상한 돌파 | 높음 |
| 7 | **JOIN key 자동 포함** (PK/FK 기반) | id/code 누락 해결 | 중간 |
