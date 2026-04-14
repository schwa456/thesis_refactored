# 석사 학위 예비 심사 발표 구성안

**주제**: Graph-Based Adaptive Schema Linking for Text-to-SQL
**발표 분량**: 20–25분 / 22 슬라이드 (+ optional 3)
**스토리 프레이밍**: Graph 기반 Schema Linking의 **구조적 기여** 강조 (Adaptive PCST · Score Ensemble · XiYan Filter)
**벤치마크**: BIRD dev (1,534 queries, 11 databases)

---

## 목차 (Table of Contents)

| Part | 슬라이드 | 제목 |
|---|---|---|
| 1. 연구 배경 | S1–S4 | Title / SL 개요 / 문제 제기 / 목표·기여 |
| 2. 관련 연구 | S5–S6 | Baseline Landscape / Gap Analysis |
| 3. 제안 방법 | S7–S13 | Pipeline / Graph·Encoder / Adaptive PCST / Ensemble / GAT / JoinKeys / XiYan Filter |
| 4. 실험·결과 | S14–S19 | 설정 / Main / Ablation / DB별 / Case / Latency |
| 5. 한계·향후 | S20–S22 | Recall 분석 / 향후 연구 / Conclusion |
| Appendix | S23–S27 | Recall 손실 분석 / PCST 수식 / 프롬프트 예시 / GAT 학습 곡선 / DB별 Precision 비교 |

---

## Part 1. 연구 배경

### S1. Title Slide

- **Key Message**: 본 연구는 Graph 기반 Schema Linking에 3가지 구조적 기법을 도입한 예비 심사 발표이다.
- **본문**
  - 제목: Graph-Based Adaptive Schema Linking for Text-to-SQL
  - 발표자 / 지도교수 / 소속
  - 발표일: 2026-04
- **발표 노트**: 인사와 함께 "Text-to-SQL에서 Schema Linking 문제를 그래프 구조 관점에서 접근한 연구"라고 한 문장으로 요약한다.

---

### S2. Text-to-SQL & Schema Linking 개요

- **Key Message**: Schema Linking은 Text-to-SQL 파이프라인의 실질적 병목이며, SL 오류가 그대로 SQL 오류로 이어진다.
- **본문**
  - Text-to-SQL 파이프라인: NL Query → Schema Linking → SQL Generation → Execution
  - SL이란: NL의 의도를 DB 스키마의 table/column 집합으로 축약하는 단계
  - 실패 예시: 관련 없는 컬럼 과다 선택 → LLM이 잘못된 JOIN/WHERE 생성
  - BIRD gold schema 평균 3.8 columns → SL이 이 수준으로 압축해야 함
- **삽입 자료**: (개념도 — 직접 작성)
- **발표 노트**: "아무리 강력한 LLM이라도 잘못된 스키마를 주면 올바른 SQL을 만들 수 없다"는 점을 강조.

---

### S3. 문제 제기: 기존 Graph-RAG SL의 한계

- **Key Message**: G-Retriever 류의 고정 threshold PCST는 query score 분포를 무시하여 과잉 선택을 초래한다.
- **본문**
  - 기존 G-Retriever: fixed threshold = 0.1로 PCST prize 부여
  - 문제 1. Query마다 score 분포가 달라 고정값이 부적절 → B-0(Raw PCST) 평균 47.7 cols 선택
  - 문제 2. Raw cosine만으로는 FK·테이블 소속 같은 **구조 정보** 반영 불가
  - 문제 3. 서브그래프 추출 후 FK 컬럼 누락 시 JOIN 실패
- **삽입 자료**: `b1_recall_at_k.png` (고정 threshold의 Recall-Precision trade-off)
- **발표 노트**: "PCST는 좋은 아이디어지만, threshold가 static이면 query-agnostic하다는 구조적 한계가 있다."

---

### S4. 연구 목표 & 기여

- **Key Message**: 본 연구의 4가지 구조적 기여가 기존 Graph-RAG SL의 한계를 단계별로 해결한다.
- **본문**
  1. **Adaptive PCST** — per-query percentile(P80) 기반 동적 threshold
  2. **Score Ensemble** — GAT 구조 점수 + Raw cosine 결합 (α=0.85)
  3. **Auto JOIN Keys** — 선택된 테이블 쌍의 FK 컬럼 자동 보강
  4. **XiYan-style Filter** — DB value example 주입 LLM 정밀 필터
- **발표 노트**: "본 발표는 이 네 가지 기법을 정의하고, 각각이 파이프라인 내에서 어떤 역할을 하는지, 그리고 실험적으로 어떤 효과를 갖는지를 보여준다."

---

## Part 2. 관련 연구

### S5. Baseline Landscape

- **Key Message**: SL 연구는 Graph-RAG, LLM-centric, Hybrid 세 계열로 나뉘며, 본 연구는 Graph-RAG 계열의 정교화에 해당한다.
- **본문 (비교표)**

  | Method | 접근 | 대표 구성 |
  |---|---|---|
  | G-Retriever (2024) | Graph-RAG | Cosine + fixed PCST |
  | LinkAlign (2024) | LLM-centric | Self-reflection + Agent |
  | XiYanSQL (2024) | Hybrid | M-Schema + Value-aware LLM Filter |
- **발표 노트**: "본 연구는 G-Retriever 계열을 기반으로 XiYanSQL의 value-aware filter 아이디어를 흡수한 hybrid에 가깝다."

---

### S6. Gap Analysis

- **Key Message**: 본 연구는 기존 세 계열의 약점을 한 파이프라인에서 동시에 해결하는 것을 목표로 한다.
- **본문 (Gap ↔ 기여 매핑)**
  - G-Retriever의 static threshold → **Adaptive PCST**
  - G-Retriever의 구조 무시 → **Score Ensemble (GAT)**
  - Graph-RAG의 JOIN 누락 → **Auto JOIN Keys**
  - LLM-only의 value-blindness → **XiYan-style Filter**
- **발표 노트**: "이 슬라이드는 본 발표 Part 3의 로드맵이기도 하다."

---

## Part 3. 제안 방법 — 발표의 핵심

### S7. Overall Pipeline (6-Stage)

- **Key Message**: 제안 파이프라인은 Graph 구성 → 인코딩 → Ensemble → Adaptive PCST → JoinKeys → XiYan Filter의 6단계로 구성된다.
- **본문**: `preliminary_defense_b4b.md` §1.1의 ASCII 아키텍처 다이어그램 그대로 사용
  - Stage 1. HeteroGraphBuilder
  - Stage 2. LocalPLMEncoder (MiniLM-L6-v2)
  - Stage 3. Score Ensemble Selector (GAT + Raw)
  - Stage 4. Adaptive PCST Extractor
  - Stage 5. Auto JOIN Key Injection
  - Stage 6. XiYan-style LLM Filter (Qwen3-Coder-30B)
- **발표 노트**: "이후 4장에 걸쳐 각 Stage의 핵심 기여를 설명한다."

---

### S8. Stage 1–2. Heterogeneous Graph & Query Encoding

- **Key Message**: DB 스키마를 이종 그래프로 변환하고, NL query와 노드를 동일 공간에 임베딩한다.
- **본문**
  - **Node Types**: table / column / **fk_node** (FK 관계를 explicit 노드화)
  - **Edge Types**: has_column · belongs_to · is_source_of · points_to · table_to_table(macro_edge)
  - **Encoder**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
  - **Query Encoding**: spaCy 기반 불용어·POS(NOUN/PROPN/NUM/VERB/ADJ) masking 후 토큰 임베딩
- **발표 노트**: "FK를 edge가 아닌 node로 모델링한 것이 이후 GAT에서 구조 점수를 학습하기 위한 장치다."

---

### S9. [기여 ①] Adaptive PCST

- **Key Message**: Per-query percentile(P80) 기반 동적 threshold가 고정 threshold의 과잉 선택 문제를 해결한다.
- **본문**
  - **Prize**: `max(score − P80(score), 0)` → query-adaptive ReLU
  - **Clamp**: `min_prize_nodes=3`, `max_prize_nodes=25`
  - **Cost**: belongs_to=0.01 / fk_link=0.05 / macro_edge=0.5 (타입별 상수)
  - 효과: B-0 → B-1에서 **Precision +0.22, F1 +0.21**
- **삽입 자료**: `b3_threshold_sweep.png`, `c4_threshold_vs_size.png`
- **발표 노트**: "Prize를 ReLU 형태로 만든 덕분에 낮은 score 노드는 자연스럽게 제외되고, P80 이상인 노드만 예산 경쟁에 들어간다."

---

### S10. [기여 ②] Score Ensemble Selector

- **Key Message**: GAT 기반 구조 점수와 Raw cosine 점수를 α=0.85로 가중 결합하면 Recall이 향상된다.
- **본문**
  - **Raw score**: NL embedding ↔ node name embedding cosine
  - **GAT score**: GAT 노드 임베딩을 DualTowerProjector로 공유 공간에 투영 후 query와 cosine
  - **Ensemble**: `score = α · raw_norm + (1 − α) · gat_norm`, α=0.85 (sweep으로 결정)
  - Node type(table / column / fk_node)별 독립 정규화 → type 간 scale 편향 제거
- **삽입 자료**: `b3c_ensemble_sweep.png` (α sweep), `a3_gat_vs_raw_auroc.png`
- **발표 노트**: "α=0.85는 Raw가 여전히 주인공이지만, 구조 점수가 boundary 케이스에서 recall을 끌어올리는 보조 역할임을 뜻한다."

---

### S11. GAT Network 구조 (Score Ensemble의 백본)

- **Key Message**: SchemaHeteroGAT는 5개 edge type을 HeteroConv로 동시에 처리하며 InfoNCE+BCE 공동 학습된다.
- **본문**
  - **구조**: 3-layer GATv2Conv, heads=4, hidden=128, skip connection
  - **학습 손실**: `BCE(pos_weight) + λ · InfoNCE(hard-neg top-15)`
  - **Projector**: DualTowerProjector — text(384) / graph(256) → joint(256) 대칭 MLP, learnable temperature
  - **Validation**: Recall@15 (BIRD train 9:1 split)
- **발표 노트**: "Projection이 두 개의 독립 MLP이지만 대조 손실을 공유하기 때문에 같은 latent 공간으로 정렬된다."

---

### S12. [기여 ③] Auto JOIN Key Injection

- **Key Message**: 선택된 테이블 쌍 사이의 FK 컬럼을 자동 보강하여 JOIN 누락으로 인한 SQL 실패를 방지한다.
- **본문**
  - PCST 출력: 테이블 2개 이상이 선택됐는데 연결 FK 컬럼이 빠지는 경우가 빈번
  - 해결: 선택 테이블 집합의 모든 pairwise FK를 조회해 컬럼 집합에 추가
  - 효과: recall 소폭 상승, downstream SQL generation에서 JOIN error 감소
- **발표 노트**: "이 단계는 비용이 거의 없는 reliable한 개선이므로 pipeline 기본값으로 포함한다."

---

### S13. [기여 ④] XiYan-style LLM Filter

- **Key Message**: DB에서 직접 조회한 value example을 프롬프트에 주입해 LLM이 의미 기반으로 정밀 필터링한다.
- **본문**
  - **M-Schema + Value**: 컬럼마다 DB에서 example value 3개 동적 조회
  - **Prompt**: `src/prompts/filter.md` `xiyan_filter` 템플릿 (M-Schema 블록 + few-shot)
  - **Robustness**: 3-stage JSON parsing 방어 (code fence strip → json.loads → regex fallback)
  - **Fallback**: parse 실패 시 직전 스키마를 그대로 반환
  - **Model**: `Qwen3-Coder-30B-A3B-Instruct-FP8` via vLLM (GPU 2장)
- **발표 노트**: "value example 3개가 LLM에게 컬럼의 '의미'를 짐작할 결정적 단서가 된다 — 예: `gender` 컬럼이 M/F/NULL인지, 1/0인지."

---

## Part 4. 실험 및 결과

### S14. 실험 설정

- **Key Message**: BIRD dev 전체(1,534 queries, 11 DBs)를 column-level metric으로 평가한다.
- **본문**
  - **Benchmark**: BIRD dev 1,534 queries / 11 databases
  - **Metrics**: column-level Recall / Precision / F1, Avg Cols (EX 미측정)
  - **Hardware**: NVIDIA RTX 3090 × 4 (vLLM: GPU 2장, 실험: CPU pipeline)
  - **Encoder**: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
  - **GAT**: 3-layer GATv2Conv, 4-head, hidden=128
  - **LLM Filter**: Qwen3-Coder-30B-A3B-Instruct-FP8 (vLLM)

---

### S15. Main Result — 3-Baseline vs 제안 (B-4b)

- **Key Message**: 제안 기법은 **Precision 0.7930으로 모든 baseline을 상회**하며 평균 3.2 컬럼으로 가장 타이트한 선택을 달성한다.
- **본문 (표)**

  | Method | Recall | Precision | F1 | Avg Cols |
  |---|---|---|---|---|
  | G-Retriever (재현) | 0.7577 | 0.7866 | 0.7719 | 3.7 |
  | LinkAlign (재현) | 0.6940 | 0.7641 | 0.7274 | 3.4 |
  | XiYanSQL (재현) | 0.6832 | 0.7408 | 0.7108 | 3.4 |
  | **제안 B-4b** | 0.6244 | **0.7930** | 0.6987 | **3.2** |
- **삽입 자료**: `e1_performance_comparison.png`
- **발표 노트**: "Precision 측면에서 기존 baseline 모두를 상회했다는 점, 그리고 선택 컬럼 수도 gold(3.8)에 가장 가깝다는 점을 강조."

---

### S16. Ablation Study — 누적 개선 효과

- **Key Message**: 각 기법은 독립적으로 기여하며, 순차 적용 시 F1이 0.27 → 0.70로 **0.43** 상승한다.
- **본문 (표)**

  | 단계 | Recall | Precision | F1 | 기여 |
  |---|---|---|---|---|
  | B-0 Raw PCST baseline | 0.9489 | 0.1570 | 0.2694 | — |
  | + Adaptive PCST (B-1) | 0.6719 | 0.3745 | 0.4809 | F1 +0.21 |
  | + Ensemble + JoinKeys (B-combined) | 0.7210 | 0.3471 | 0.4686 | Recall +4.9%p |
  | + XiYan Filter (B-4b) | 0.6244 | 0.7930 | 0.6987 | F1 +0.23 |
- **발표 노트**: "Ensemble은 F1이 아닌 Recall을 끌어올리는 역할, 최종 Precision boost는 XiYan Filter에서 나온다는 것을 분명히 설명."

---

### S17. DB별 성능 분석

- **Key Message**: 11개 DB 중 Precision은 4개 DB에서, F1은 1개(thrombosis_prediction)에서 G-Retriever를 상회하며, Recall은 전 DB에서 열세이다.
- **본문**
  - **Precision Win (4개)**: thrombosis_prediction (+18.6%p), financial (+6.2%p), debit_card_specializing (+5.8%p), superhero (+1.1%p)
  - **F1 Win (1개)**: thrombosis_prediction (+7.4%p) — 유일한 F1 승리, Precision이 압도적으로 상승한 DB
  - **F1 Loss**: toxicology (−32.2%p), debit_card_specializing (−11.7%p), european_football_2 (−10.2%p)
  - **Recall 전 DB 열세**: 11/11 DB에서 Recall 하락 → Recall 손실이 구조적·전면적임을 시사
  - **패턴**: G-Retriever가 상대적으로 과다 선택(Recall > Precision)이던 DB(예: thrombosis_prediction)에서 본 연구의 Precision 보강이 효과적으로 작동; 반대로 약어·유사 컬럼명이 많은 DB(toxicology)에서는 Filter가 과다 제거
- **삽입 자료**: `e3_db_complexity.png`
- **발표 노트**: "Precision 승리 DB 4개 중 3개는 소폭(+1~6%p), thrombosis_prediction만 압도적(+18.6%p). 'Precision 보강 장치'가 과다 선택 DB에서 가장 잘 작동한다는 해석으로 연결. 하락 DB의 공통점이 다음 슬라이드(Case Study)와 자연스럽게 이어진다."

---

### S18. Case Study

- **Key Message**: B-4b는 tight selection에 강하지만, 유사 컬럼명이 많은 DB에서 필요 컬럼을 잘못 제거하는 실패를 보인다.
- **본문**
  - **성공 사례 (예: thrombosis_prediction)**: Adaptive PCST로 30+ 후보 → Filter 후 3개 정확 선택
  - **실패 사례 (예: codebase_community)**: 유사 `user_*` 컬럼 다수 → Filter가 필요한 컬럼 제거
- **삽입 자료**: `b4b_dashboard.png` (종합 대시보드에서 해당 사례 발췌)
- **발표 노트**: "실패 사례는 뒤에 이어질 Recall 손실 93%가 Filter 단계에서 발생한다는 분석의 근거다."

---

### S19. Latency 분석

- **Key Message**: 전체 latency의 85.6%가 LLM Filter에 집중되어 있어, Filter 최적화가 곧 속도 개선으로 이어진다.
- **본문 (표)**

  | Stage | 평균 시간 |
  |---|---|
  | Graph Build | 0.095s |
  | Query Encoding | 0.004s |
  | Score Ensemble (GAT) | 0.090s |
  | Adaptive PCST | 0.001s |
  | **XiYan Filter (LLM)** | **1.132s** |
  | **Total** | **1.323s/query** |
- **발표 노트**: "Filter 없이는 0.19s/query로 매우 빠르다 — 향후 경량 classifier로 대체할 여지를 암시."

---

## Part 5. 한계 및 향후 연구

### S20. Recall 하락의 구조적 원인

- **Key Message**: Recall 손실이 발생한 500개 query 중 **93.8%가 LLM Filter 단계에서 발생**하며, PCST 단계 손실은 6.2%에 불과 → 개선 target이 Filter 단계로 명확히 좁혀진다.
- **본문 (B-combined vs B-4b query-level 비교, 1,534 queries)**

  | 원인 | Query 수 | 비율 |
  |---|---:|---:|
  | Filter harmless (recall 유지) | 1,020 | 66.5% |
  | Filter improved (recall 상승) | 14 | 0.9% |
  | Filter partial drop | 439 | 28.6% |
  | Filter removed all | 30 | 2.0% |
  | PCST already missed | 31 | 2.0% |

  → **손실 원인 집중도**: Filter 단계 469건 (93.8%) vs PCST 단계 31건 (6.2%)
- **삽입 자료**: Appendix A (S23) 상세 분석 방법 참조
- **발표 노트**: "Filter 전(B-combined) Recall이 이미 0.72로 회복 가능성이 높다는 점을 강조. 상세 분석 방법은 Appendix A 참조."

---

### S21. 향후 연구 방향

- **Key Message**: Filter 정밀화·Multi-Agent Selector·PCST 재조정·GAT 재학습·Encoder 통일의 다섯 개 축으로 Recall 회복과 구조적 개선을 병행한다.
- **본문**
  - **(a) Filter 보수화**: 프롬프트 강화 / Recall Floor (seed top-K 강제 포함) / 2-Pass Filter / Confidence-aware
  - **(b) Multi-Agent XiYan Selector**: 단일 에이전트를 Semantic · Structural · Skeptic 에이전트로 확장하여 불확실성 기반 라우팅으로 의사결정의 해석 가능성 확보 (프로토타입 `AdaptiveMultiAgentFilter` 존재, 정량 평가 예정)
  - **(c) PCST 재조정**: Percentile 하향 (P80→P70) / DB 크기별 동적 percentile / max_prize_nodes 확대
  - **(d) GAT 재학습**: 현재 AUROC 0.69 → DualTower + 개선된 hard-negative sampling → Ensemble 효과 극대화
  - **(e) Encoder 통일**: APIEncoder vs LocalPLMEncoder 공정 비교로 재현 편차 제거
- **발표 노트**: "(a)·(c)는 저난이도·high-impact 단기 과제, (b)·(d)는 구조적 확장을 위한 중장기 과제. Multi-Agent는 필터 결정의 해석 가능성을 확보하는 측면에서도 의미가 있다."

---

### S22. 기대 효과 & Conclusion

- **Key Message**: 본 연구는 Graph 기반 SL의 4가지 구조적 기여로 **Precision 최고**를 달성했고, 향후 Recall 회복을 통해 F1 0.75+로 G-Retriever를 전면 상회할 것으로 기대된다.
- **본문**
  - **Contribution 요약**: Adaptive PCST + Score Ensemble + Auto JOIN + XiYan Filter
  - **현재 성과**: Precision 0.7930 (최고), Avg Cols 3.2 (gold 3.8에 가장 근접)
  - **향후**: Filter 정밀화 / Multi-Agent Selector / PCST 재조정 / GAT 재학습 등으로 Recall 0.62 → 0.72+ 회복 시 F1 0.75+ 기대
  - **광의의 기여**: Graph-RAG 계열 SL이 과잉 선택 문제를 해결할 수 있음을 실증
- **삽입 자료**: `b4b_dashboard.png` (종합 요약)
- **발표 노트**: "Precision 우위라는 확정 결과 + Recall 회복이라는 명확한 후속 target을 동시에 강조하면서 마무리."

---

## Appendix 슬라이드

### S23 (App-A). Recall 손실 원인 분석 — 방법론

- **Key Message**: B-combined(Filter 적용 전)와 B-4b(Filter 적용 후)의 query-level 출력을 1:1 매칭하여 Recall 손실이 어느 단계에서 발생했는지 결정론적으로 분류하였다.
- **데이터 소스**
  - **Filter 전**: `outputs/experiments/experiment_b_combined/output_b_combined.jsonl` (Adaptive PCST + Ensemble + JoinKeys까지 적용)
  - **Filter 후**: `outputs/experiments/experiment_b4_xiyan_filter/output_b4_xiyan_filter.jsonl`
  - **매칭 키**: `question_id` (BIRD dev 1,534 queries 전체)
  - **비교 변수**: 각 query의 `recall` 값(column-level)
- **분류 규칙 (5-way)**

  | 조건 | 분류 | 의미 |
  |---|---|---|
  | `recall_before == 0` | PCST already missed | Filter 이전 단계에서 이미 gold 노드 누락 |
  | `recall_after == recall_before` | Filter harmless | Filter가 recall 유지 (유해하지 않음) |
  | `recall_after > recall_before` | Filter improved | Filter가 recall 향상 (드문 엣지케이스) |
  | `recall_after == 0` (& before ≠ 0) | Filter removed all | Filter가 필요 노드 전부 제거 (완전 오판) |
  | 그 외 (recall 감소) | Filter partial drop | Filter가 필요 노드 일부 제거 |
- **손실 집중도 지표 산출**
  - 전체 손실 query 수 = `filter_partial_drop + filter_removed_all + pcst_already_missed`
  - Filter 단계 기여도 = `(partial_drop + removed_all) / 전체 손실 query 수`
- **발표 노트**: "결정론적 rule-based 분류이므로 실험 재현 시 동일 수치 재산출 가능. 코드는 약 20 lines Python."

---

### S24 (App-B). Recall 손실 원인 분석 — 결과

- **Key Message**: 1,534 queries 중 손실이 발생한 500 queries의 93.8%가 Filter 단계에서 발생, 6.2%가 PCST 단계에서 발생한다.
- **본문 (분류별 결과)**

  | # | 분류 | Query 수 | 전체 대비 | 손실 query 대비 |
  |---|---|---:|---:|---:|
  | 1 | Filter harmless | 1,020 | 66.49% | — |
  | 2 | Filter improved | 14 | 0.91% | — |
  | 3 | **Filter partial drop** | **439** | **28.62%** | **87.80%** |
  | 4 | **Filter removed all** | **30** | **1.96%** | **6.00%** |
  | 5 | **PCST already missed** | **31** | **2.02%** | **6.20%** |
  | | **합계** | **1,534** | **100%** | |

  | 집계 | Query 수 | 비율 |
  |---|---:|---:|
  | Filter가 Recall 유지·개선 (1+2) | 1,034 | 67.40% |
  | **손실 발생 query (3+4+5)** | **500** | **32.60%** |
  | ↳ Filter 단계 원인 (3+4) | 469 | **93.80% (손실 query 중)** |
  | ↳ PCST 단계 원인 (5) | 31 | **6.20% (손실 query 중)** |
- **해석 포인트**
  - **(1) 67.4% 유지**: Filter는 과반 query에서 recall을 해치지 않음 → Filter 자체를 버릴 이유가 없음
  - **(2) 손실의 88%는 부분 제거**: 완전 오판(2%)보다 부분 오판(29%)이 훨씬 빈번 → "보수화(under-filtering)" 기조로 회복 가능
  - **(3) PCST 손실 6.2%**: Adaptive threshold 공격성의 부작용 → percentile 하향(P80→P70) 여지
  - **(4) 개선 target 명확화**: 전체 1,534 query의 **28.62%만 손본다면** Recall을 G-Retriever 수준으로 회복 가능
- **발표 노트**: "'Filter를 전부 걷어내지 말고, 28.6% partial-drop 케이스만 구제하면 된다'는 메시지로 마무리. 이것이 Part 5 향후 연구(Filter 보수화, Recall Floor, Multi-Agent)의 정량적 근거."

---

### S25 (App-C). PCST Prize/Cost 수식 상세

- **Key Message**: Adaptive PCST의 Prize/Cost 설계 디테일.
- **본문**
  - `prize(v) = max(score(v) − P80(scores), 0)` — ReLU 형태
  - `cost(e)`: belongs_to=0.01 / fk_link=0.05 / macro_edge=0.5 (경로 선호도)
  - pcst_fast library 사용, min/max prize node clamp 정책

### S26 (App-D). XiYanFilter 프롬프트 예시

- **Key Message**: 실제 `xiyan_filter` 템플릿의 입력·출력 구조.
- **본문**: `src/prompts/filter.md` 발췌 — M-Schema 블록 + value example 3개 + few-shot + JSON 출력 규격

### S27 (App-E). DB별 Precision 비교 상세

- **Key Message**: 11개 DB 전체의 B-4b vs G-Retriever Precision 비교.
- **본문**: S17에서 요약한 Precision win 4개 DB(thrombosis_prediction / financial / debit_card_specializing / superhero)의 상세 수치 + Precision loss DB 7개
- (S17에서 지면 부족으로 생략된 full table 배치)

---

## 시각 자료 매핑

| 파일 | 배치 | 용도 |
|---|---|---|
| `b1_recall_at_k.png` | S3 | 고정 threshold 한계 |
| `b3_threshold_sweep.png` | S9 | Adaptive threshold 효과 |
| `c4_threshold_vs_size.png` | S9 | threshold vs 선택 크기 |
| `a3_gat_vs_raw_auroc.png` | S10 | GAT vs Raw 점수 |
| `b3c_ensemble_sweep.png` | S10 | α sweep |
| `e1_performance_comparison.png` | S15 | 11-method 비교 |
| `e3_db_complexity.png` | S17 | DB별 성능 |
| `b4b_dashboard.png` | S18, S22 | 종합 대시보드 |
| `preliminary_defense_b4b.md` ASCII diagram | S7 | 6-stage 파이프라인 |

모든 자산은 `notebooks/analysis_results/` 에 이미 존재하며, 본 발표를 위해 신규 생성할 시각 자료는 없다.

---

## 준비 체크리스트

- [ ] 22 슬라이드 본편 + 3 optional 슬라이드 완성
- [ ] 각 슬라이드 Key Message 1문장 명시
- [ ] 3개 baseline 수치가 [preliminary_defense_b4b.md](preliminary_defense_b4b.md) §2.1과 일치
- [ ] 모든 PNG 자산 파일 경로 유효성 확인
- [ ] Part 3(제안 방법)에 "구조적 기여" 프레이밍이 일관되게 드러남
- [ ] 발표 리허설에서 Part 3을 12분, Part 4를 7분으로 배분
