# Graph-Based Schema Linking 종합 분석 및 향후 전략 보고서

> BIRD Dev Set 1,534 queries 기준 | 11개 실험 결과 종합 | Phase 1~3 분석 기반

---

## 1. 현재 시스템 진단 요약

### 1.1 전체 성능 맵

| 방법 | Recall | Precision | F1 | EX | Avg #Pred |
|------|--------|-----------|-----|-----|-----------|
| **G-Retriever (baseline)** | **0.758** | **0.787** | **0.772** | **0.249** | 3.68 |
| LinkAlign (baseline) | 0.694 | 0.764 | 0.727 | 0.200 | 3.44 |
| VectorOnly (baseline) | 0.683 | 0.747 | 0.713 | 0.179 | 3.46 |
| XiYanSQL (baseline) | 0.683 | 0.741 | 0.711 | 0.197 | 3.45 |
| MST Expansion (baseline) | 0.642 | 0.727 | 0.682 | 0.147 | 3.34 |
| GAT+MultiAgent (proposed) | 0.658 | 0.685 | 0.671 | 0.150 | 3.59 |
| GAT Classifier (proposed) | 0.549 | 0.620 | 0.582 | 0.091 | 3.40 |
| Base PCST (proposed) | 0.757 | 0.141 | 0.238 | 0.000 | 26.8 |
| Dynamic PCST (proposed) | 0.757 | 0.142 | 0.238 | 0.000 | 26.7 |
| Uncertainty PCST (proposed) | 0.663 | 0.161 | 0.259 | 0.000 | 19.3 |
| Dyn+Unc PCST (proposed) | 0.665 | 0.161 | 0.259 | 0.000 | 19.4 |

**핵심 문제: 제안 방법 중 단 하나도 최약 베이스라인(MST Expansion, F1=0.682)조차 넘지 못함.**

### 1.2 문제의 구조

현재 시스템의 실패는 개별 모듈의 문제가 아니라, **모듈 간 조합의 실패**에 기인한다.

```
[문제 구조 요약]

GAT Classifier → Ranking 능력이 Raw Cosine보다 열등 (AUROC 0.692 vs 0.780)
       ↓
  낮은 Seed 품질 → Recall@20 = 0.564 (Raw Cosine은 0.673)
       ↓
  PCST → 낮은 seed를 보상하려 과다 확장 → Avg 26.8 nodes (Precision 0.14)
       ↓
  MultiAgent Filter → 과도한 Rejection (38.2% Unanswerable 판정)
       ↓
  최종 성능 열화: F1=0.671 (vs G-Retriever F1=0.772)
```

---

## 2. 모듈별 상세 진단

### 2.1 GAT + DualTower Projector

#### 증상
- Per-query AUROC: GAT 0.692 vs Raw Cosine **0.780** (Raw가 +8.8%p 우위)
- **63.4%**의 쿼리에서 Raw Cosine이 더 나은 ranking 제공
- GAT가 우세한 쿼리는 33.4%에 불과하며, "어려운" 쿼리에서도 개선 미미

#### 근본 원인 분석
1. **Message Passing의 Over-Smoothing**: 3-layer GATv2Conv가 이웃 노드의 특징을 과도하게 혼합. 특히 `belongs_to` (Column→Table) 엣지로 인해 같은 테이블의 모든 컬럼 임베딩이 수렴하여, 개별 컬럼의 의미적 차별성이 소실됨.
2. **학습 목표와 평가 목표의 불일치**: InfoNCE + BCE 손실은 "gold/non-gold 이진 분류"를 최적화하지만, 실제 필요한 것은 "gold 노드를 상위에 ranking"하는 능력. 분류 정확도가 높아도 ranking 품질은 나빠질 수 있음.
3. **DualTower의 차원 병목**: PLM 384d → GAT 256d → Joint 256d로 차원 축소 과정에서 원본 의미 임베딩의 세밀한 차이가 소실.
4. **Score Distribution 특성**: GAT score의 gold mean=0.447, non-gold mean=0.239로 gap은 0.208이지만, 히스토그램 overlap이 0.697로 매우 높음. Raw Cosine은 gap이 0.107로 작지만 overlap이 0.645로 오히려 낮음. 즉 GAT가 전체적으로 점수를 올렸지만 **분리 능력은 악화**.

#### 수치 근거
| 지표 | GAT | Raw Cosine |
|------|-----|------------|
| Gold score mean | 0.447 | 0.265 |
| Non-gold score mean | 0.239 | 0.158 |
| Mean gap | 0.208 | 0.107 |
| Histogram overlap | 0.697 | **0.645** |
| Global AUROC | 0.690 | **0.743** |
| Per-query AUROC median | 0.713 | **0.812** |

### 2.2 PCST Extractor

#### 증상
- Base PCST: recall 0.757 / precision **0.141** / EX **0.000**
- 평균 26.8개 노드 반환 (gold 평균 3.85개 대비 7배)
- SQL 생성 비활성화 상태(EX 측정 불가)였지만, 이 크기의 subgraph로는 유효한 SQL 생성이 불가능

#### 근본 원인 분석
1. **고정 Threshold의 한계**: `node_threshold=0.15`에서 Raw Cosine 기준 평균 48.5개 노드가 prize > 0. PCST는 prize가 있는 노드를 edge cost보다 이득이면 모두 포함하므로, 사실상 스키마 절반 이상을 선택.
2. **Edge Cost 체계의 문제**: `belongs_to=0.01`, `fk=0.05`로 너무 저렴하여 PCST가 테이블 전체 컬럼과 FK chain을 모두 포함. 하나의 테이블을 선택하면 해당 테이블의 모든 컬럼이 cost 0.01로 딸려옴.
3. **Dynamic/Uncertainty 변형의 한계**: Dynamic PCST(hub discount)는 이미 과다 선택 상태에서 추가 노드를 더 쉽게 포함시킴. Uncertainty PCST는 threshold를 0.55로 올려 노드 수를 줄였으나(19.3개), 여전히 과다.

#### 수치 근거
| Threshold | Avg Prize > 0 Nodes | Gold Recall in Prizes |
|-----------|--------------------|-----------------------|
| 0.10 | 65.3 | 0.914 |
| **0.15 (현재)** | **48.5** | **0.828** |
| 0.20 | 32.9 | 0.717 |
| 0.30 | 10.3 | 0.436 |
| 0.50 | 0.4 | 0.044 |

Threshold를 올리면 과다 선택은 해결되지만 gold 커버리지가 급락하는 딜레마.

#### PCST의 실질적 가치 (Bridge 복원)
| Seed (Top-K) | Seed Recall | PCST Final Recall | Bridge 복원분 |
|-------------|-------------|-------------------|--------------|
| Top-5 | 0.367 | 0.758 | **+0.390** |
| Top-10 | 0.517 | 0.758 | **+0.241** |
| Top-15 | 0.608 | 0.758 | +0.150 |
| Top-20 | 0.673 | 0.758 | +0.085 |

PCST는 seed가 적을 때(Top-5~10) **극적인 bridge 복원 효과**를 보임. 문제는 과다 선택이지 PCST 알고리즘 자체가 아님.

### 2.3 MultiAgent Filter

#### 증상
- **38.2% (586건)을 Unanswerable로 판정** → 해당 쿼리에 0개 노드 반환
- 에이전트 간 **45.0%가 완전 불일치**(Uncertainty=1.0)
- Precision +6.5%p 개선 vs Recall -11%p 손실

#### 근본 원인 분석
1. **Unanswerable 판정의 잘못된 전제**: Filter가 GAT Classifier의 seed를 입력받는데, seed 품질이 이미 낮음(recall 0.549). 입력에 gold 컬럼이 부족하니 LLM이 "답할 수 없다"고 판정하는 것은 논리적으로 맞지만, **문제의 원인은 filter가 아니라 seed 단계**.
2. **Skeptic 에이전트의 보수적 경향**: 두 에이전트가 불일치하면 Skeptic이 호출되는데, Skeptic이 보수적으로 판단하여 "Unanswerable" 쪽으로 기울어짐.
3. **컨텍스트 부족**: Filter에 전달되는 DDL은 seed에 포함된 노드만 포함. Gold 노드가 seed에 없으면 filter가 아무리 정확해도 복원 불가.

#### 수치 근거 (Unanswerable 판정의 위험성)
| 지표 | 값 |
|------|-----|
| Unanswerable 판정 쿼리 | 586건 |
| 이 중 GAT recall > 0 | 495건 (84.5%) |
| 이 중 GAT recall = 1.0 | 73건 (12.5%) |
| **G-Retriever recall on these** | **mean=0.725** |
| **G-Retriever recall > 0.5 on these** | **431건 (73.5%)** |
| **G-Retriever recall = 1.0 on these** | **235건 (40.1%)** |

Filter가 "풀 수 없다"고 판정한 쿼리의 **40.1%가 G-Retriever 기준으로 perfect recall**. 이는 filter의 false rejection 비율이 치명적으로 높음을 의미.

### 2.4 EX Score의 구조적 한계

#### 발견
| Recall Range | G-Retriever EX | n |
|-------------|---------------|---|
| 0 | 0.000 | 71 |
| 0~0.25 | 0.014 | 40 |
| 0.25~0.50 | 0.053 | 105 |
| 0.50~0.75 | 0.088 | 253 |
| 0.75~1.0 | 0.182 | 369 |
| **1.0 (perfect)** | **0.417** | 696 |

- Perfect recall에서도 EX는 0.417에 불과 → **SQL Generator가 EX의 실질적 상한을 결정**
- Pearson r(recall, EX) = 0.315 → 중간 수준의 양의 상관. Schema Linking이 EX의 필요조건이지만 충분조건은 아님.
- **Recall >= 0.75가 EX 급상승의 임계점**: 이 구간에서 EX가 0.088 → 0.182로 2배 이상 점프.

---

## 3. 향후 전략

### 전략 1: Raw Cosine Backbone 전환 + PCST 유지 (즉시 적용 가능)

#### 개요
현재 최고 성능인 G-Retriever의 구조를 그대로 채택하되, 그 위에 개선 모듈을 적층하는 전략.

#### 구체적 변경
```
[현재]  TokenEncoder → GAT → DualTower → GATClassifier → PCST → MultiAgent Filter
[변경]  APIEncoder → Raw Cosine Similarity → VectorOnlySelector(top_k=20) → PCST → (개선된 Filter)
```

#### 근거
- Raw Cosine AUROC **0.780** vs GAT 0.692 → 즉시 +8.8%p ranking 개선
- G-Retriever가 이 구조로 이미 F1=0.772, EX=0.249를 달성
- 추가 학습이나 모델 변경 없이 **config 변경만으로 적용 가능**

#### 예상 효과
- Recall: 0.658 → ~0.758 (+10%p)
- Precision: 0.685 → ~0.787 (+10%p)
- EX: 0.150 → ~0.249 (+10%p)

---

### 전략 2: Adaptive PCST Threshold (핵심 개선)

#### 개요
고정 `node_threshold` 대신, 각 query의 score 분포에 맞춰 동적으로 threshold를 설정.

#### 구체적 방법
```python
# 현재: 고정 threshold
prizes = max(scores - 0.15, 0)  # → 평균 48.5개 노드

# 개선: per-query adaptive threshold
sorted_scores = sorted(node_scores, reverse=True)
adaptive_threshold = sorted_scores[min(k, len(sorted_scores)-1)]  # top-K의 최소 점수
prizes = max(scores - adaptive_threshold * decay_factor, 0)
```

또는 percentile 기반:
```python
adaptive_threshold = np.percentile(node_scores, 80)  # 상위 20%만 prize
```

#### 근거
- 현재 threshold=0.15에서 48.5개 → threshold=0.30에서 10.3개, gold recall 0.436
- DB마다 score 분포가 크게 다름: `debit_card_specializing`(노드 수 많음)과 `toxicology`(적음)에 같은 threshold 적용이 부적절
- Percentile 기반이면 DB 크기에 무관하게 상위 N%만 선택하여 **subgraph 크기를 일정하게 유지**

#### 예상 효과
- Subgraph size: 26.8 → ~8-12개 (적정 수준)
- Precision: 0.14 → ~0.5-0.7 (대폭 개선)
- Recall: 0.76 → ~0.65-0.70 (다소 하락하나 bridge 복원이 보상)

---

### 전략 3: Filter를 "Pruning-Only" 모드로 전환

#### 개요
현재 filter의 "Unanswerable" 판정을 제거하고, 입력 subgraph에서 불필요한 노드만 제거하는 순수 pruning 역할로 한정.

#### 구체적 변경
```python
# 현재: Unanswerable이면 0개 노드 반환
if uncertainty > threshold:
    return {"status": "Unanswerable", "final_nodes": []}

# 개선: Unanswerable 판정 제거, 항상 union 반환
# Skeptic 에이전트가 pruning만 수행
final_nodes = agent_a_nodes | agent_b_nodes  # union (보수적 유지)
# 또는 Skeptic이 union에서 명확한 noise만 제거
```

#### 근거
- 현재 Unanswerable 586건 중 **73.5%가 실제 solvable** → false rejection 비용이 극도로 높음
- Filter가 "없는 것보다 나쁜" 상태: GAT-only(no filter) recall 0.549 → GAT+MA(filter) recall 0.658이지만, 이는 filter가 recall을 올린 것이 아니라 **PCST가 올린 것**
- Pruning-only 모드는 "불필요한 노드 제거"에 집중하므로 recall 손실 없이 precision만 개선

#### 예상 효과
- Recall: 유지 또는 소폭 개선 (Unanswerable → 항상 응답)
- Precision: 개선 (noise 노드 제거)
- EX: 개선 (SQL Generator에 clean한 schema 전달)

---

### 전략 4: Score Ensemble Reranking

#### 개요
Raw Cosine을 주(primary) scorer로, GAT score를 보조(secondary) 신호로 사용하는 앙상블.

#### 구체적 방법
```python
final_score = alpha * raw_cosine_score + (1 - alpha) * gat_score
# alpha = 0.85 (Phase 2 실험에서 도출된 최적값)
```

#### 근거
| K | Pure Raw F1 | Pure GAT F1 | Ensemble (a=0.85) F1 | 개선 |
|---|-------------|-------------|----------------------|------|
| 5 | 0.399 | 0.162 | **0.446** | +4.7%p |
| 10 | 0.389 | 0.261 | **0.439** | +5.0%p |
| 15 | 0.358 | 0.269 | **0.397** | +3.9%p |
| 20 | 0.315 | 0.270 | **0.355** | +4.0%p |

- 모든 K에서 앙상블이 단독 사용보다 우수
- GAT가 단독으로는 열등하지만, **15%의 가중치로 보조 신호 역할은 유효**
- GAT가 구조적 정보(message passing)를 반영하므로, 순수 의미 유사도가 놓치는 "FK를 통해 간접 연결된 테이블"에 대한 약간의 boost 제공

#### 예상 효과
- Top-K F1: +4~5%p (전략 1 위에 추가 적용)

---

### 전략 5: JOIN Key 자동 포함 (Post-Processing)

#### 개요
SQL에 필수적인 PK/FK 컬럼을 seed 선택 후 자동으로 포함하는 후처리 로직.

#### 구체적 방법
```python
# PCST 결과에서 2개 이상의 테이블이 선택된 경우:
selected_tables = extract_tables(selected_nodes)
if len(selected_tables) > 1:
    for table_pair in combinations(selected_tables, 2):
        fk_cols = find_fk_columns(table_pair, graph_metadata)
        selected_nodes.extend(fk_cols)  # JOIN에 필요한 id/key 자동 추가
```

#### 근거
- 가장 많이 놓치는 컬럼 1위: `id` (128~209회, 모든 방법에서 1위)
- `id`, `code`, `cdscode` 등 범용 이름의 PK/FK는 semantic similarity가 낮아 ranking에서 밀림
- 이들은 SQL에서 JOIN 조건에 반드시 필요하므로, 의미적 유사도가 아닌 **구조적 역할**로 선택되어야 함
- Graph 메타데이터에 FK 정보가 이미 포함되어 있으므로 추가 비용 없음

#### 예상 효과
- Missing `id` 계열 컬럼 대폭 감소
- JOIN이 필요한 multi-table 쿼리의 EX 개선

---

### 전략 6: SQL Generator 업그레이드

#### 개요
Perfect recall에서도 EX=0.417인 현 SQL Generator의 한계를 돌파.

#### 구체적 방법
- **모델 업그레이드**: Qwen3-Coder-30B → 더 강력한 모델 (GPT-4o, Claude 등) 또는 fine-tuned Text-to-SQL 전문 모델
- **Self-Consistency**: 여러 번 SQL 생성 후 다수결
- **Self-Debugging**: 생성된 SQL을 실행하고, 에러 시 에러 메시지와 함께 재생성

#### 근거
| Schema Recall | G-Retriever EX |
|--------------|---------------|
| 1.0 (perfect) | 0.417 |
| 0.75~1.0 | 0.182 |

- Schema가 완벽해도 EX 42% → **Generator가 EX의 실질적 병목**
- Schema Linking 개선만으로는 EX를 0.5 이상으로 끌어올리기 어려움
- BIRD 벤치마크 최상위 방법들은 EX 0.65+ 달성 → Generator 품질 차이가 핵심

#### 예상 효과
- EX 상한: 0.42 → 0.55~0.65 (Generator 수준에 따라)

---

### 전략 7: GAT 학습 목표 재설계 (중장기)

#### 개요
현재 GAT의 ranking 실패를 근본적으로 해결하기 위한 학습 방법론 변경.

#### 구체적 방법

**방법 A: ListMLE / LambdaRank 도입**
```python
# 현재: BCE + InfoNCE (분류 기반)
loss = bce_loss + lambda * infonce_loss

# 개선: Ranking Loss (순위 기반)
loss = listMLE_loss(predicted_ranking, gold_ranking) + margin_triplet_loss
```
- 노드의 이진 분류가 아닌 **순위 최적화**를 직접 학습

**방법 B: GAT를 Reranker로 재정의**
```python
# 현재: GAT가 raw 임베딩 → score 생성 (end-to-end)
# 개선: Raw Cosine으로 Top-K 후보 추출 → GAT가 후보 내에서만 reranking
candidates = raw_cosine_topk(query, all_nodes, k=30)
reranked = gat_rerank(query, candidates, graph)
final = reranked[:k_final]
```

**방법 C: Residual Connection 강화**
```python
# 현재 skip connection: out + skip(input)
# 개선: raw PLM 임베딩의 비중을 학습 가능하게
alpha = sigmoid(learnable_alpha)
final = alpha * gat_output + (1 - alpha) * raw_plm_embedding
```
- Message passing으로 인한 over-smoothing을 원본 임베딩 보존으로 완화

#### 근거
- GAT의 문제는 아키텍처가 아니라 **학습 목표**: 분류(gold/non-gold)와 ranking(상위에 gold 배치)은 다른 문제
- Phase 2의 앙상블 실험에서 alpha=0.85(raw 85%)가 최적 → raw 임베딩을 대부분 보존하면서 GAT 정보를 소량 추가하는 것이 최적 전략임을 확인
- 이를 학습 단계에서 구현하면 앙상블보다 효율적

---

## 4. 실행 로드맵

### Phase A: 즉시 적용 (config 변경만으로 가능)

```
소요: 1일 (실험 실행 시간 포함)
```

| 단계 | 작업 | 내용 |
|------|------|------|
| A-1 | Raw Cosine backbone 전환 | `seed_selector: VectorOnlySelector(top_k=20)`, `projection: disabled` |
| A-2 | PCST + Filter 조합 실험 | G-Retriever 구조에 `AdaptiveMultiAgentFilter` 추가 |
| A-3 | Filter pruning-only 모드 | Unanswerable 판정 제거, union 반환 |

**예상 결과**: F1 0.671 → ~0.77+, EX 0.150 → ~0.25+

### Phase B: 단기 개선 (코드 수정 필요)

```
소요: 3~5일
```

| 단계 | 작업 | 내용 |
|------|------|------|
| B-1 | Adaptive PCST threshold | Per-query percentile 기반 threshold 구현 |
| B-2 | Score ensemble | `0.85*raw + 0.15*gat` reranking 구현 |
| B-3 | JOIN key 자동 포함 | PK/FK 기반 post-processing 구현 |
| B-4 | Filter 개선 | Pruning-only + schema에 value 예시 포함 |

**예상 결과**: F1 ~0.77 → ~0.80+, EX ~0.25 → ~0.28+

### Phase C: 중기 개선 (재학습 필요)

```
소요: 1~2주
```

| 단계 | 작업 | 내용 |
|------|------|------|
| C-1 | GAT 학습 목표 변경 | ListMLE/LambdaRank 기반 ranking loss |
| C-2 | GAT Reranker 전환 | Raw Top-30 → GAT rerank → Top-K |
| C-3 | Residual connection 강화 | 학습 가능한 raw/GAT 비율 파라미터 |
| C-4 | SQL Generator 업그레이드 | 더 강력한 LLM 또는 self-consistency |

**예상 결과**: EX ~0.28 → ~0.35+

### Phase D: 장기/논문 기여 (신규 방법론)

```
소요: 2~4주
```

| 단계 | 작업 | 내용 |
|------|------|------|
| D-1 | Structure-Aware Adaptive PCST | GAT attention weight를 PCST edge cost로 직접 활용 |
| D-2 | Confidence-Calibrated Filter | Filter의 "확신도"를 calibration하여 rejection 정밀도 향상 |
| D-3 | End-to-End Graph RAG | Selector→Extractor→Filter를 하나의 differentiable pipeline으로 통합 |

---

## 5. 논문 서사(Narrative) 재구성 제안

현재 실험 결과만으로는 "제안 방법이 베이스라인보다 우수하다"는 주장이 불가능하다. 대신, 분석 결과를 기반으로 다음과 같은 논문 서사를 제안한다.

### 서사 옵션 A: "진단과 처방" 프레임

> "기존 Graph-based Schema Linking 방법들의 체계적 분석을 통해, 각 모듈(Encoder, Selector, Extractor, Filter)의 병목을 정량적으로 진단하고, 이를 기반으로 설계한 개선된 파이프라인이 SOTA를 달성함을 보인다."

- Phase 1~3의 분석 자체가 contribution
- 진단 → 처방 → 검증의 과학적 절차

### 서사 옵션 B: "PCST의 올바른 사용법" 프레임

> "PCST는 schema linking에서 bridge node 복원에 강력하지만, 고정 threshold와 pruning 부재로 실패한다. Adaptive threshold + pruning filter를 결합한 Controlled PCST가 recall과 precision을 동시에 달성함을 보인다."

- PCST bridge 복원 효과(+8.5~39%p)는 실질적 기여
- "왜 PCST가 지금까지 잘 안 됐는지"를 설명하고 해결하는 프레임

### 서사 옵션 C: "모듈 조합 최적화" 프레임

> "Schema Linking 파이프라인의 최적 모듈 조합을 ablation study를 통해 탐색한 결과, Raw Cosine + Adaptive PCST + Pruning Filter의 조합이 기존 개별 기법 대비 우수함을 보인다."

- 11개 실험의 ablation 자체가 체계적 기여
- 각 모듈의 "단독 효과"와 "조합 효과"를 분리하여 분석

---

## 6. 부록: 생성된 분석 자료 목록

### 보고서
| 파일 | 내용 |
|------|------|
| `phase1_report.md` | 성능 비교, score 분포, selector recall, 오류 패턴 |
| `phase2_report.md` | threshold sweep, top-K sweep, ensemble, PCST threshold |
| `phase3_report.md` | GAT AUROC, MultiAgent filter, EX 상관, DB 복잡도 |
| `comprehensive_strategy_report.md` | 본 문서 (종합 전략) |

### 차트
| 파일 | 내용 |
|------|------|
| `e1_performance_comparison.png` | 11개 실험 성능 비교 bar chart |
| `e1b_size_vs_recall.png` | Subgraph 크기 vs Recall scatter |
| `a1_score_distribution.png` | Gold vs Non-gold score 히스토그램 |
| `a3_gat_vs_raw_auroc.png` | Per-query AUROC 비교 |
| `b1_recall_at_k.png` | Top-K Recall 상한선 곡선 |
| `b2b_complexity_vs_recall.png` | Schema 복잡도 vs Recall |
| `b3_threshold_sweep.png` | Threshold sweep PR 곡선 |
| `b3b_topk_sweep.png` | Top-K sweep PR 곡선 |
| `b3c_ensemble_sweep.png` | Score ensemble alpha sweep |
| `c4_threshold_vs_size.png` | PCST threshold vs subgraph 크기 |
| `d_multiagent_analysis.png` | MultiAgent uncertainty/recall 분석 |
| `e2_recall_vs_ex.png` | Recall 구간별 EX score |
| `e3_db_complexity.png` | DB 복잡도별 성능 scatter |
| `e3b_complexity_vs_ex.png` | Gold 복잡도 구간별 EX 비교 |

### 분석 스크립트
| 파일 | 내용 |
|------|------|
| `phase1_analysis.py` | E1, A1, B1, B2 분석 |
| `phase2_analysis.py` | B3, C1, C2, C4 분석 |
| `phase3_analysis.py` | A3, D1~D3, E2, E3 분석 |
