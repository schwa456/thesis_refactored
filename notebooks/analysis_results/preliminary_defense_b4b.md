# Graph-Based Adaptive Schema Linking for Text-to-SQL

**예비 심사 발표용 실험 결과 정리**  
평가 기준: BIRD Benchmark (1,534 dev queries, 11 databases)

---

## 1. 제안 프레임워크 (B-4b Pipeline)

### 1.1 전체 아키텍처

```
NL Query
   |
   v
+-------------------------------------------------------------+
|  Stage 1. Schema Graph Construction (HeteroGraphBuilder)    |
|  - DB 스키마를 이종 그래프(Heterogeneous Graph)로 변환       |
|  - Node Types: Table, Column, FK_Node                       |
|  - Edge Types: belongs_to, fk_link, macro_edge              |
|  - Node Feature: sentence-transformers/all-MiniLM-L6-v2     |
+--------------------------+----------------------------------+
                           |
                           v
+-------------------------------------------------------------+
|  Stage 2. Query Encoding (LocalPLMEncoder)                  |
|  - NL Query -> 384-dim embedding (MiniLM-L6-v2)            |
|  - Cosine similarity로 모든 노드에 대한 raw score 계산       |
+--------------------------+----------------------------------+
                           |
                           v
+-------------------------------------------------------------+
|  Stage 3. Score Ensemble Selector [제안 B-2]                |
|  - GAT 기반 구조적 점수 + Raw Cosine 점수를 가중 결합        |
|  - ensemble = a*raw_norm + (1-a)*gat_norm  (a=0.85)        |
|  - GATv2Conv 3-layer, 4-head, skip connection               |
|  - Top-K=20 seed nodes 선택                                 |
+--------------------------+----------------------------------+
                           |
                           v
+-------------------------------------------------------------+
|  Stage 4. Adaptive PCST Subgraph Extraction [제안 B-1]      |
|  - 기존: 고정 threshold(0.1) -> 과잉 선택 문제              |
|  - 제안: per-query percentile 기반 adaptive threshold        |
|    - threshold = P80(score distribution)                     |
|    - min_prize_nodes=3, max_prize_nodes=25                  |
|  - Prize-Collecting Steiner Tree로 연결 서브그래프 추출       |
+--------------------------+----------------------------------+
                           |
                           v
+-------------------------------------------------------------+
|  Stage 5. Auto JOIN Key Injection [제안 B-3]                |
|  - 선택된 2+ 테이블 간 FK 컬럼 자동 보강                    |
|  - JOIN 누락으로 인한 SQL 오류 방지                          |
+--------------------------+----------------------------------+
                           |
                           v
+-------------------------------------------------------------+
|  Stage 6. XiYan-style LLM Filter [제안 B-4]                |
|  - DB에서 컬럼별 Example Value 3개를 직접 조회               |
|  - M-Schema + Value 정보를 포함한 프롬프트로 LLM 호출        |
|  - 불필요한 테이블/컬럼 제거 -> 최종 스키마 결정             |
|  - Model: Qwen3-Coder-30B-A3B-Instruct-FP8                 |
+--------------------------+----------------------------------+
                           |
                           v
            Final Schema (Table.Column list)
```

### 1.2 제안 기법 요약

| 기법 | 문제 인식 | 해결 방법 |
|---|---|---|
| **Adaptive PCST** (B-1) | 고정 threshold(0.1)가 query score 분포와 무관 -> 평균 47.7 cols 과잉 선택 | Per-query percentile(P80) 기반 동적 threshold. Score 분포에 따라 자동 조정 |
| **Score Ensemble** (B-2) | Raw cosine만으로는 구조적 관계(FK, 테이블 소속) 반영 불가 | GATv2Conv로 학습한 구조적 점수와 raw cosine을 a=0.85로 결합 |
| **Auto JOIN Keys** (B-3) | 서브그래프에 테이블은 있지만 FK 컬럼이 누락 -> JOIN 불가 | 선택된 테이블 쌍의 FK 컬럼을 자동 보강 |
| **XiYan-style Filter** (B-4) | Adaptive PCST 후에도 평균 12.1 cols (gold 3.8의 3배) | DB value example 포함 프롬프트로 LLM이 정밀 필터링 |

---

## 2. 실험 결과

### 2.1 Baseline 비교

| Method | Recall | Precision | F1 | Avg Cols |
|---|---|---|---|---|
| G-Retriever (재현) | 0.7577 | 0.7866 | 0.7719 | 3.7 |
| LinkAlign (재현) | 0.6940 | 0.7641 | 0.7274 | 3.4 |
| XiYanSQL (재현) | 0.6832 | 0.7408 | 0.7108 | 3.4 |
| **제안 (B-4b)** | 0.6244 | **0.7930** | 0.6987 | **3.2** |

- **Precision: 제안 방법이 모든 baseline 대비 최고** (0.7930)
- G-Retriever 대비 Precision +0.64%p 개선, 선택 컬럼 수도 3.7->3.2로 감소
- **F1은 G-Retriever 대비 -7.3%p** -- Recall 하락(0.76->0.62)이 주요 원인

### 2.2 Ablation Study (누적 개선 효과)

| 단계 | Recall | Precision | F1 | 기여 |
|---|---|---|---|---|
| B-0 (Raw PCST baseline) | 0.9489 | 0.1570 | 0.2694 | -- |
| + Adaptive PCST (B-1) | 0.6719 | 0.3745 | 0.4809 | **F1 +0.21** |
| + Ensemble + JoinKeys (B-combined) | 0.7210 | 0.3471 | 0.4686 | Recall +4.9%p |
| + XiYanFilter (B-4b) | 0.6244 | 0.7930 | 0.6987 | **F1 +0.23** |

### 2.3 DB별 성능

| DB | B-4b Recall | B-4b Precision | B-4b F1 | G-Retriever F1 | 차이 |
|---|---|---|---|---|---|
| thrombosis_prediction | 0.7417 | 0.8490 | **0.7917** | 0.7181 | **+7.4%p** |
| superhero | 0.6909 | 0.9057 | 0.7839 | 0.8622 | -7.8%p |
| formula_1 | 0.7563 | 0.7987 | 0.7769 | 0.8020 | -2.5%p |
| codebase_community | 0.6874 | 0.8393 | 0.7558 | 0.8146 | -5.9%p |
| student_club | 0.6961 | 0.7908 | 0.7404 | 0.7935 | -5.3%p |
| financial | 0.6744 | 0.8076 | 0.7350 | 0.7422 | -0.7%p |
| european_football_2 | 0.5968 | 0.7119 | 0.6493 | 0.7511 | -10.2%p |
| card_games | 0.5139 | 0.7100 | 0.5962 | 0.6944 | -9.8%p |
| toxicology | 0.4387 | 0.8322 | 0.5745 | 0.8966 | -32.2%p |
| california_schools | 0.4806 | 0.6349 | 0.5471 | 0.6465 | -9.9%p |
| debit_card_specializing | 0.3961 | 0.7969 | 0.5292 | 0.6464 | -11.7%p |

- **thrombosis_prediction에서 G-Retriever를 상회** (+7.4%p F1) — 유일한 F1 승리 DB
- **Precision은 11개 DB 중 4개에서 G-Retriever를 상회**: thrombosis_prediction (+18.6%p), financial (+6.2%p), debit_card_specializing (+5.8%p), superhero (+1.1%p)
- **Recall은 11개 DB 전부에서 열세** — Recall 손실이 구조적·전면적임을 시사
- F1 하락이 큰 DB: toxicology(-32.2%p), debit_card(-11.7%p), european_football_2(-10.2%p) -- Recall 부족이 원인

---

## 3. 한계점 분석

### 3.1 Recall 하락의 구조적 원인

B-4b의 Recall(0.6244)이 G-Retriever(0.7577)보다 **13.3%p 낮은** 이유를 단계별로 분해:

**Recall 손실 원인 분류 (1,534 queries)**

| 원인 | Query 수 | 비율 | 설명 |
|---|---:|---:|---|
| Filter harmless (recall 유지) | 1,020 | **66.49%** | Filter가 필요한 컬럼을 유지 |
| Filter improved (recall 상승) | 14 | 0.91% | 드문 엣지케이스 |
| Filter partial drop | 439 | **28.62%** | 필요 컬럼 일부 제거 |
| Filter removed all | 30 | 1.96% | 필요 노드 전부 제거 (완전 오판) |
| PCST already missed (Filter 전 Recall=0) | 31 | 2.02% | Adaptive threshold가 공격적 |
| **합계** | **1,534** | **100%** | |

**핵심**: Recall 손실이 발생한 500개 query 중 **93.8%(469개)가 Filter 단계에서 손실**되었으며, PCST 단계 손실은 6.2%(31개)에 불과. Filter 전(B-combined)의 Recall은 0.7210 → 개선 target이 Filter 단계로 명확히 좁혀진다.

### 3.2 DB별 Recall 하락 패턴

Filter에 의한 평균 Recall 하락이 큰 DB:

| DB | Filter 전->후 Recall Drop | 분석 |
|---|---|---|
| codebase_community | -0.179 | 대규모 스키마(186 cols), 유사 컬럼명 다수 |
| card_games | -0.163 | 대규모 스키마(191 cols), 도메인 특수 용어 |
| student_club | -0.114 | 비슷한 컬럼명이 여러 테이블에 분산 |
| california_schools | -0.111 | 약어/코드 형태 컬럼명 (CDSCode 등) |

**공통점**: 컬럼 수가 많고, 컬럼명이 유사하거나 도메인 특수적인 DB에서 LLM이 필요 컬럼을 잘못 제거

### 3.3 Encoder 차이 문제

- **G-Retriever**: base_config의 `APIEncoder` 사용 (vLLM 기반)
- **B-4b**: `LocalPLMEncoder` 사용 (sentence-transformers 직접 로드)
- 동일 모델(all-MiniLM-L6-v2)이지만 encoding 경로가 달라 미세한 차이 가능
- 공정 비교를 위해 동일 encoder로 통일 실험 필요

### 3.4 Latency

| Stage | 평균 시간 |
|---|---|
| Graph Build | 0.095s |
| Query Encoding | 0.004s |
| Score Ensemble (GAT) | 0.090s |
| Adaptive PCST | 0.001s | 
| **XiYan Filter (LLM)** | **1.132s** |
| **Total** | **1.323s/query** |

- LLM Filter가 전체 시간의 **85.6%** 차지
- Filter 없이는 0.19s/query로 매우 빠름

---

## 4. 개선 방향 (향후 연구)

### 4.1 Recall 개선 -- Filter 단계 정밀화

| 방안 | 기대 효과 | 난이도 |
|---|---|---|
| **Filter 프롬프트 보수화**: "확실히 불필요한 것만 제거" 지침 추가 | Recall 상승, Precision 소폭 하락 | 낮음 |
| **Recall Floor 보장**: Filter 후에도 seed top-K 노드는 강제 포함 | Recall 상승 (하한선 설정) | 낮음 |
| **2-Pass Filter**: 1차 넓게 선택 -> 2차 정밀 제거 | Recall 상승, Precision 유지 | 중간 |
| **Confidence-aware Filter**: LLM 출력에 confidence를 포함시켜 낮은 confidence 노드는 유지 | Recall 상승 | 중간 |

### 4.2 Recall 개선 -- PCST 단계 최적화

| 방안 | 기대 효과 | 난이도 |
|---|---|---|
| **Percentile 하향 (P80->P70)**: 더 많은 노드에 prize 부여 | Filter 입력 Recall 상승 (현재 0.72->예상 0.78+) | 낮음 |
| **DB 크기별 동적 percentile**: 대규모 DB에서 더 관대한 threshold | DB간 성능 편차 감소 | 중간 |
| **max_prize_nodes 확대 (25->35)**: 대규모 DB에서 커버리지 확보 | 대규모 DB Recall 상승 | 낮음 |

### 4.3 Precision 추가 개선

| 방안 | 기대 효과 | 난이도 |
|---|---|---|
| **Multi-iteration XiYan**: 현재 1회 -> 2회 반복 정제 | Precision 상승 (단, Recall 주의) | 낮음 |
| **Value-aware PCST**: DB value를 PCST prize에 반영 | 불필요 노드 조기 제거 | 높음 |

### 4.4 구조적 개선

| 방안 | 기대 효과 | 난이도 |
|---|---|---|
| **GAT 재학습**: 현재 AUROC 0.69 -> DualTower + 개선된 negative sampling | Ensemble 효과 극대화 | 높음 |
| **Encoder 통일**: APIEncoder vs LocalPLMEncoder 공정 비교 | 정확한 성능 비교 | 낮음 |
| **경량 Filter**: LLM 대신 학습된 classifier로 대체 | Latency 대폭 감소 | 높음 |

---

## 5. 발표 핵심 메시지

### 주장

> 기존 Graph-RAG 기반 Schema Linking(G-Retriever)의 고정 threshold PCST는 query별 score 분포를 무시하여 과잉 선택 문제를 야기한다. 본 연구는 **Adaptive PCST + Score Ensemble + LLM Filter**의 3단계 정제 파이프라인을 통해, **Precision 0.7930으로 G-Retriever(0.7866) 대비 최고 정밀도를 달성**하였다.

### 현재 한계

> 그러나 Recall이 0.6244로 G-Retriever(0.7577) 대비 하락하여 F1 기준으로는 열세(0.70 vs 0.77)이다. Recall 손실의 93%가 LLM Filter 단계에서 발생하며, 이는 개선 가능한 영역이다.

### 향후 방향

> Filter 프롬프트 보수화, Percentile 하향, Recall Floor 보장 등의 기법으로 Recall을 0.72+ 수준으로 회복하면, **F1 0.75+ 달성이 가능**할 것으로 기대한다. 이는 Precision 우위를 유지하면서 G-Retriever를 전면적으로 상회하는 결과가 될 것이다.

---

## Appendix: 실험 환경

- **Hardware**: NVIDIA RTX 3090 x 4 (vLLM: GPU 2장, 실험: CPU)
- **LLM**: Qwen3-Coder-30B-A3B-Instruct-FP8 (vLLM serving)
- **Embedding**: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **GAT**: SchemaHeteroGAT (3-layer GATv2Conv, 4-head, hidden=128)
- **PCST**: pcst_fast library
- **Benchmark**: BIRD dev set (1,534 queries, 11 databases)
- **Metrics**: Column-level Recall, Precision, F1 (EX 미측정)
