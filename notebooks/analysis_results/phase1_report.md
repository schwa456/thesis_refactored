# Phase 1: 기존 결과 기반 정량 분석 보고서

## E1: 전체 성능 비교 (병목 진단)

| Method | Recall | Precision | F1 | EX | Avg #Pred |
|--------|--------|-----------|-----|-----|-----------|
| **G-Retriever (PCST)** | **0.758** | **0.787** | **0.772** | **0.249** | 3.68 |
| LinkAlign | 0.694 | 0.764 | 0.727 | 0.200 | 3.44 |
| VectorOnly (baseline) | 0.683 | 0.747 | 0.713 | 0.179 | 3.46 |
| XiYanSQL | 0.683 | 0.741 | 0.711 | 0.197 | 3.45 |
| **GAT+MultiAgent** | **0.658** | **0.685** | **0.671** | **0.150** | 3.59 |
| GAT Classifier | 0.549 | 0.620 | 0.582 | 0.091 | 3.40 |
| Base/Dynamic PCST | 0.757 | 0.141 | 0.238 | 0.000 | **26.8** |

**핵심 발견: 제안 방법이 모든 베이스라인보다 뒤처짐.**

## A1: Score Distribution

| Scoring Method | Gold Mean | Non-Gold Mean | Gap | AUROC |
|----------------|-----------|---------------|-----|-------|
| Raw cosine (VectorOnly/G-Retriever) | 0.265 | 0.158 | 0.107 | **0.743** |
| GAT Classifier scores | 0.447 | 0.239 | 0.208 | 0.690 |

**역설적 결과**: GAT가 mean gap을 0.107->0.208로 벌렸지만, **AUROC는 오히려 하락** (0.743->0.690). GAT가 gold 점수는 올렸으나 non-gold도 같이 올려서 **순위(ranking) 능력은 약화**됨. Histogram overlap도 ~0.70으로 두 분포가 여전히 심하게 겹침.

## B1: Selector Recall 상한선

| K | Raw Cosine | GAT Classifier |
|---|------------|----------------|
| 5 | 0.367 | 0.221 |
| 10 | 0.517 | 0.365 |
| 15 | 0.608 | 0.484 |
| 20 | **0.673** | 0.564 |
| 30 | 0.753 | 0.669 |
| 50 | 0.859 | 0.793 |

**Raw cosine 기반이 모든 K에서 GAT보다 우수**. GAT의 ranking 능력이 오히려 baseline보다 나쁨.

## B2: 오류 패턴

1. **가장 많이 놓치는 컬럼**: `id` (모든 방법에서 1위, 128~209회) -- 범용적 이름이라 semantic similarity가 낮음
2. **어려운 DB**: `debit_card_specializing` (recall 0.57), `california_schools` (0.64), `card_games` (0.65)
3. **GAT Classifier의 zero recall**: 10.0% (153건) vs G-Retriever 4.6% (71건)
4. **복잡도 영향**: gold 컬럼이 많아질수록 recall 급락 (예상 가능하나 정량화됨)

---

## 이론적 개선점 (Phase 1에서 도출된)

### 문제 1: GAT가 Ranking을 악화시킴
- **원인**: GAT는 contrastive learning(InfoNCE + BCE)으로 학습되었지만, message passing이 non-gold 노드의 점수도 끌어올려 ranking을 오염
- **개선안 A**: GAT 후 score를 raw cosine과 **앙상블** (`a * raw_cosine + (1-a) * gat_score`)
- **개선안 B**: GAT 학습 시 **margin-based ranking loss** (e.g., triplet loss with margin) 추가
- **개선안 C**: GAT output을 classifier가 아닌 **reranker**로 사용 (초기 top-K를 raw cosine으로 뽑고, GAT로 rerank)

### 문제 2: PCST가 Precision을 파괴
- **원인**: `node_threshold=0.15`로 거의 모든 노드가 prize를 받아 전체 스키마 반환 (평균 26.8 노드)
- **개선안 D**: Threshold를 adaptive하게 설정 (score 분포의 상위 percentile 기반)
- **개선안 E**: PCST 후 **LLM pruning** 추가 (현재 PCST 실험은 filter=None)
- **개선안 F**: PCST + MultiAgent Filter 조합 실험 (현재 미수행)

### 문제 3: "id", "name" 같은 범용 컬럼 누락
- **원인**: `id`는 semantic embedding에서 어떤 테이블의 `id`인지 구분 불가
- **개선안 G**: 컬럼명에 **테이블 컨텍스트 강화** (`"Column: id in table users, primary key"` 등)
- **개선안 H**: FK/PK 관계를 활용한 **구조적 필수 컬럼 자동 포함** (JOIN에 필요한 id 컬럼 자동 추가)

### 문제 4: 최선 제안 방법이 베이스라인에 미달
- **근본 원인**: G-Retriever는 raw cosine + PCST(적절한 threshold) 조합인데, 이것이 이미 충분히 강력
- **전략적 개선**: G-Retriever를 baseline이 아닌 **backbone**으로 사용하고, 그 위에 GAT reranking 또는 MultiAgent filter를 추가하는 구조로 전환

---

## 개선 우선순위

1. **PCST + MultiAgent Filter 조합** (현재 미실험, 빠르게 검증 가능)
2. **GAT를 reranker로 전환** (raw cosine top-K -> GAT rerank)
3. **JOIN key 자동 포함 로직** (id/code 누락 해결)
