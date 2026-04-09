# Phase 2: 하이퍼파라미터 민감도 분석 보고서

## B3: GATClassifier Threshold / Top-K Sweep

### Threshold 기반 선택

| Threshold | GAT Recall | GAT Precision | GAT F1 | Raw Recall | Raw Precision | Raw F1 |
|-----------|-----------|--------------|--------|-----------|--------------|--------|
| 0.10 | 0.718 | 0.154 | 0.253 | 0.914 | 0.138 | 0.240 |
| 0.20 | 0.664 | 0.164 | 0.263 | 0.717 | 0.227 | 0.345 |
| **0.30** | 0.602 | 0.170 | 0.265 | **0.436** | **0.377** | **0.404** |
| 0.40 | 0.555 | 0.174 | 0.265 | 0.186 | 0.366 | 0.247 |
| 0.50 | 0.503 | 0.183 | 0.268 | 0.044 | 0.135 | 0.066 |

- **GAT Best F1**: 0.270 at t=0.35 (매우 낮음)
- **Raw Cosine Best F1**: 0.404 at t=0.30
- GAT의 F1이 Raw Cosine보다 **어떤 threshold에서도 낮음** -> GAT 점수의 ranking 품질 문제 재확인

### Top-K 기반 선택

| Method | Best K | Best F1 | Recall@Best | Precision@Best |
|--------|--------|---------|-------------|----------------|
| GAT Classifier | K=13 | 0.300 | 0.449 | 0.225 |
| Raw Cosine | **K=7** | **0.403** | 0.438 | 0.373 |

- Raw Cosine은 K=7에서 F1=0.403으로, K가 적어도 효율적인 선택 가능
- GAT는 K를 크게 잡아야 recall이 올라가지만 precision이 급락

### Score Ensemble 시뮬레이션 (alpha * raw_cosine + (1-alpha) * gat_score)

| K | Best alpha | Best F1 | Recall | Precision |
|---|-----------|---------|--------|-----------|
| 5 | **0.85** | **0.446** | 0.411 | 0.488 |
| 10 | 0.80 | 0.439 | 0.573 | 0.356 |
| 15 | 0.80 | 0.397 | 0.671 | 0.282 |
| 20 | 0.85 | 0.355 | 0.736 | 0.234 |

**핵심 발견:**
- 최적 alpha가 0.80~0.85 -> **Raw Cosine에 80~85% 가중치**, GAT에 15~20%만
- 그럼에도 K=5에서 F1=0.446으로 순수 Raw Cosine(0.403)보다 **+4.3%p 개선**
- GAT는 reranker로서 소량의 보조 신호로만 유효. 주(main) scorer로는 부적합

---

## C2+C4: PCST Node Threshold 분석

### Threshold별 Prize 노드 수 및 Gold 커버리지

| Threshold | Raw: Avg Nodes | Raw: Gold Recall | GAT: Avg Nodes | GAT: Gold Recall |
|-----------|---------------|-----------------|----------------|-----------------|
| 0.05 | 79.6 | 0.966 | 50.8 | 0.795 |
| 0.10 | 65.3 | 0.914 | 44.2 | 0.718 |
| **0.15** | **48.5** | **0.828** | 40.6 | 0.691 |
| 0.20 | 32.9 | 0.717 | 38.6 | 0.664 |
| 0.30 | 10.3 | 0.436 | 33.3 | 0.602 |
| 0.50 | 0.4 | 0.044 | 23.4 | 0.503 |

**핵심 발견:**
- **현재 PCST 실험은 threshold=0.15 사용** -> Raw Cosine 기준 평균 48.5개 노드가 prize를 받음
- 이는 전체 스키마(~92개)의 53%에 해당 -> PCST가 거의 전체를 선택하는 원인
- threshold를 0.25~0.30으로 올리면 노드 수가 10~20개로 줄지만, gold recall도 0.44~0.58로 급락
- **Dilemma**: 낮은 threshold = 높은 recall + 파괴적 precision / 높은 threshold = 적절한 size + recall 손실

### 적응형 Threshold의 필요성
- 고정 threshold가 근본 문제: score 분포는 DB마다 크게 다름
- 해결책: 각 query의 score 분포 상위 percentile(예: top 20%)로 adaptive하게 설정

---

## C1: PCST Bridge 복원율

### Seed Top-K vs PCST Final Recall (G-Retriever)

| Seed (Top-K) | Seed Recall | Final Recall (PCST) | Delta (Bridge 복원) |
|-------------|-------------|--------------------|--------------------|
| Top-5 | 0.367 | 0.758 | **+0.390** |
| Top-10 | 0.517 | 0.758 | **+0.241** |
| Top-15 | 0.608 | 0.758 | **+0.150** |
| Top-20 | 0.673 | 0.758 | **+0.085** |
| Top-30 | 0.753 | 0.758 | +0.005 |

**핵심 발견:**
- PCST는 Top-20 seed 대비 **+8.5%p recall 복원** -> bridge 노드 복원 효과 확인
- 단, Top-30 이상에서는 거의 효과 없음 (seed가 이미 충분히 커서)
- PCST의 가치: **seed가 적을 때(K=5~15) 극적인 recall 향상** (최대 +39%p)
- 문제: 이 bridge 복원이 precision을 대폭 희생하며 이루어짐 (final avg nodes = 56.9)

---

## Phase 2 종합 진단

### 확인된 핵심 문제

1. **GAT 점수가 Raw Cosine보다 ranking 능력이 열등**
   - 모든 threshold, 모든 K에서 Raw Cosine이 우세
   - 앙상블 시에도 GAT 가중치는 15~20%가 최적 -> 보조 신호에 불과

2. **PCST의 딜레마**
   - threshold가 낮으면 과다 선택 (precision 파괴)
   - threshold가 높으면 gold 손실 (recall 파괴)
   - 고정 threshold로는 해결 불가 -> adaptive threshold 필수

3. **PCST Bridge 복원은 실질적으로 유효하나 pruning이 부재**
   - Top-20 -> Final에서 +8.5%p recall 복원은 의미 있음
   - 그러나 final nodes가 56.9개로 과다 -> post-PCST pruning 필요

### 개선 전략 업데이트 (Phase 1 + Phase 2 종합)

| 우선순위 | 개선안 | 근거 | 예상 효과 |
|---------|-------|------|----------|
| 1 | Raw Cosine Top-K + PCST + MultiAgent Filter | G-Retriever backbone에 filter 추가 | recall 유지 + precision 대폭 개선 |
| 2 | Adaptive PCST threshold (per-query percentile) | 고정 threshold의 한계 극복 | subgraph size 최적화 |
| 3 | Score ensemble (0.85*raw + 0.15*gat) reranking | GAT를 보조 신호로만 활용 | Top-K F1 +4.3%p |
| 4 | JOIN key 자동 포함 (PK/FK 기반) | `id` 컬럼 누락 해결 | missing cols 대폭 감소 |
