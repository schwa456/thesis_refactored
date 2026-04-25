# Experiment History

Schema Linking 파이프라인 연구의 전체 실험 이력.
BIRD-Dev 1,534 queries 기준. EX=0.0000인 실험은 SQL Generator를 비활성화하고 Schema Linking만 평가한 것.

> **ID 체계 (2026-04-14 재정비)**: 모델 구조(특히 Seed Selector) 기준으로 실험을 재분류. `b0_baselines/`, `s01~s05/`(Selector별), `abl/`(ablation studies) 폴더로 구조화. 전체 매핑은 [`EXPERIMENT_ID_MIGRATION.md`](EXPERIMENT_ID_MIGRATION.md) 참조. 본 문서의 기존 ID(B#, A#, I#, Q#, T# 등)는 이력 가독성을 위해 유지하며, migration doc이 신규 폴더 경로의 단일 진실 원본(single source of truth).

---

## 1. Baselines (외부 방법 재현)

| # | Method | Recall | Precision | F1 | EX | 비고 |
|---|--------|--------|-----------|------|-----|------|
| B1 | G-Retriever (PCST) | 0.7577 | 0.7866 | 0.7719 | 0.2490 | 우리 파이프라인의 출발점. NeurIPS 2024 |
| B2 | LinkAlign | 0.6940 | 0.7641 | 0.7274 | 0.2001 | EMNLP 2025, multi-DB retrieval |
| B3 | XiYan-SQL | 0.6832 | 0.7408 | 0.7108 | 0.1969 | XiYan Filter만 단독 사용 |
| B4 | Vector-Only (ours) | 0.6825 | 0.7470 | 0.7133 | 0.1786 | PLM cosine similarity만 사용 |
| B5 | Graph Expansion (ours) | 0.6417 | 0.7265 | 0.6815 | 0.1467 | 초기 그래프 확장 시도 |
| B6 | Graph + Agent (ours) | 0.6413 | 0.7252 | 0.6807 | 0.1454 | Multi-Agent 필터 추가 |

**Insight**: G-Retriever가 가장 균형 잡힌 baseline. Recall 0.76으로 높지만, fixed PCST cost 구조의 한계가 있음.

---

## 2. Phase A: 초기 아키텍처 탐색 (2026-02 ~ 03)

### 2-1. GAT Classifier + Multi-Agent Filter

| # | Experiment | Recall | Precision | F1 | EX | Config |
|---|-----------|--------|-----------|------|-----|--------|
| A1 | gat_classifier | 0.5489 | 0.6203 | 0.5824 | 0.0906 | GATClassifierSelector only |
| A2 | gat_classifier_multi_agent | 0.6580 | 0.6848 | 0.6712 | 0.1499 | + AdaptiveMultiAgentFilter |
| A3 | gat_pcst_multi_agent | 0.1913 | 0.2577 | 0.2196 | 0.0078 | GATAwarePCST + MultiAgent |
| A4 | gat_multi_agent | 0.2848 | 0.3651 | 0.3201 | 0.0261 | GAT Projection + MultiAgent |

**Insight**: GATClassifier + MultiAgent(A2)가 EX 0.15로 가장 높았으나, precision/recall 자체가 baseline보다 크게 낮음. GAT를 classifier로 직접 사용하면 PLM cosine 대비 약화됨. GATAwarePCST(A3)는 실패 — GAT score를 PCST prize로 직접 사용하면 scale mismatch 발생.

### 2-2. PCST Variant 탐색

| # | Experiment | Recall | Precision | F1 | Config |
|---|-----------|--------|-----------|------|--------|
| A5 | base_pcst | 0.7571 | 0.1411 | 0.2378 | PCSTExtractor (fixed cost) |
| A6 | dynamic_pcst | 0.7567 | 0.1415 | 0.2384 | DynamicPCSTExtractor (hub discount) |
| A7 | uncertainty_pcst | 0.6633 | 0.1608 | 0.2589 | + uncertainty margin |
| A8 | dynamic_uncertainty_pcst | 0.6646 | 0.1606 | 0.2587 | dynamic + uncertainty |

**Insight**: PCST는 recall을 0.75+로 끌어올리지만 precision이 0.14~0.16으로 급락. Fixed cost PCST는 너무 많은 노드를 포함시킴. Uncertainty 변형은 recall을 소폭 낮추지만 precision 개선은 미미 — cost 구조 자체가 문제.

---

## 3. Phase B: 파이프라인 단계별 발전 (2026-03 ~ 04 초)

단계적으로 모듈을 교체하며 기여도를 검증한 실험 시리즈.

| # | Experiment | Selector | Extractor | Filter | Recall | Precision | F1 |
|---|-----------|----------|-----------|--------|--------|-----------|------|
| B0 | b0_raw_pcst_baseline | VectorOnly | PCST(fixed) | None | 0.9489 | 0.1570 | 0.2694 |
| B1 | b1_adaptive_pcst | VectorOnly | AdaptivePCST | None | 0.6719 | 0.3745 | 0.4810 |
| B2 | b2_ensemble | Ensemble(α=0.85) | PCST(fixed) | None | 0.9679 | 0.1293 | 0.2281 |
| B-c | b_combined | Ensemble(α=0.85) | AdaptivePCST | None | 0.7210 | 0.3471 | 0.4685 |
| B4a | b4_single_filter | Ensemble(α=0.85) | AdaptivePCST | SingleAgent | 0.5720 | 0.7795 | 0.6598 |
| B4b | b4_xiyan_filter | Ensemble(α=0.85) | AdaptivePCST | XiYan | 0.6244 | 0.7930 | 0.6987 |

**Insight**:
- B0→B1: AdaptivePCST(P80 threshold)가 precision을 0.16→0.37로 +0.22 향상 (recall은 -0.28 trade-off)
- B1→B-c: Ensemble selector 추가 시 recall +0.05, precision 소폭 감소
- B-c→B4b: **XiYan Filter가 결정적** — precision을 0.35→0.79로 +0.44 폭등. Recall은 -0.10 감소
- XiYan > SingleAgent: precision +0.01, recall +0.05 (XiYan의 iterative refinement 효과)

---

## 4. Phase C: 2×2×2 Ablation Matrix (2026-04-07)

Selector(Cosine/Ensemble) × Extractor(Basic/Adaptive) × Filter(None/XiYan) 조합 비교.

| # | Selector | PCST | Filter | Recall | Precision | F1 |
|---|----------|------|--------|--------|-----------|------|
| 1 | Cosine | Basic | None | 0.9489 | 0.1570 | 0.2694 |
| 2 | Cosine | Adaptive | None | 0.6719 | 0.3745 | 0.4810 |
| 3 | Ensemble | Basic | None | 0.9679 | 0.1293 | 0.2281 |
| 4 | Ensemble | Adaptive | None | 0.7210 | 0.3471 | 0.4685 |
| 5 | Cosine | Basic | XiYan | 0.7987 | 0.7694 | 0.7838 |
| 6 | Ensemble | Basic | XiYan | 0.8149 | 0.7597 | 0.7863 |
| 7 | Cosine | Adaptive | XiYan | 0.5835 | 0.7829 | 0.6687 |
| 8 | Ensemble | Adaptive | XiYan | 0.6244 | 0.7930 | 0.6987 |

**Insight**:
- **Best F1**: #6 Ensemble+Basic+XiYan (0.7863) — Basic PCST가 넓게 포함 → XiYan이 정밀 pruning
- **Best Precision**: #8 Ensemble+Adaptive+XiYan (0.793) — 하지만 recall 0.62로 낮음
- **XiYan 유무가 가장 큰 차이**: Filter 없이는 P<0.38, 있으면 P>0.76
- **Adaptive PCST의 역설**: Filter 없이는 도움(P +0.22), Filter와 함께 쓰면 오히려 recall 손실 — PCST가 미리 잘라낸 노드를 XiYan이 복구 못 함
- **Ensemble vs Cosine**: Filter 있을 때 차이 미미(+0.01~0.02) — α=0.85에서 GAT 기여 15%만

---

## 5. GAT Training History

| # | Model | Loss | Best Recall@15 | Epochs | Date | Checkpoint |
|---|-------|------|----------------|--------|------|-----------|
| T1 | GAT v1 baseline | BCE | 0.5885 | 122 (early stop) | 03-20 | gat_classifier_best.pt |
| T2 | MLP Classifier | BCE | - | 300 | 03-24 | mlp_classifier_train_best_recall.pt |
| T3 | MLP + GAT | BCE | - | 300 | 03-24 | mlp_classifier_with_gat_train_best_recall.pt |
| T4 | GAT + InfoNCE | BCE+InfoNCE | 0.4876 | 300 | 04-01 | best_gat_model.pt |
| T5 | Enriched GAT | BCE+InfoNCE | - | 300 | 04-10 | best_gat_enriched.pt |
| T6 | Query-Cond (Projector) | BCE+InfoNCE | - | 300 | 04-11 | best_gat_query_conditioned.pt |
| T7 | Query-Supernode (Projector) | BCE+InfoNCE | - | 300 | 04-11 | best_gat_query_supernode.pt |
| T8 | Query-Cond Direct | BCE only | **0.5914** | 300 | 04-12 | best_gat_query_conditioned_direct.pt |
| T9 | Query-Supernode Direct | BCE only | 0.5548 | 300 | 04-12 | best_gat_query_supernode_direct.pt |

**Insight**:
- InfoNCE 추가(T4) 시 recall 0.59→0.49로 오히려 하락 — contrastive loss가 schema 분류에 부적합할 수 있음
- Query-Conditioned Direct(T8)가 최고 0.5914 — query 정보 주입이 유효
- Concat(T8) > SuperNode(T9): 0.5914 vs 0.5548 — supernode의 over-smoothing 가능성
- Direct(BCE only)가 Projector(BCE+InfoNCE)보다 단순하면서 효과적

---

## 6. 지도교수 면담 아이디어 실험 (2026-04-10 ~ 현재)

### 6-1. Idea 1: Alpha 조정 (GAT 기여도 증폭)

| # | Alpha | Recall | Precision | F1 | 비고 |
|---|-------|--------|-----------|------|------|
| I1a | 0.85 (baseline) | 0.7210 | 0.3471 | 0.4685 | b_combined 결과 |
| I1b | 0.75 | 0.6937 | 0.3417 | 0.4577 | 미미한 차이 |
| I1c | 0.70 | 0.6714 | 0.3299 | 0.4423 | 오히려 약간 하락 |

**Insight**: GAT 가중치를 높여도 filter 없이는 큰 차이 없음. GAT 자체의 판별력이 cosine과 크게 다르지 않기 때문.

### 6-2. Idea 2: Product Cost PCST (Score-Driven Edge Cost)

| # | Experiment | Recall | Precision | F1 | 비고 |
|---|-----------|--------|-----------|------|------|
| I2a | idea2_product_cost | 0.7349 | 0.3453 | 0.4698 | Filter 없이, Adaptive 대비 recall +0.01 |
| I2b | idea2_product_cost_xiyan | 0.6141 | 0.7963 | 0.6935 | + XiYan |

**Insight**: Product cost가 fixed cost 대비 recall 소폭 개선. XiYan과 결합 시 precision 0.80 도달.

### 6-3. Idea 3: Steiner Backbone + PCST Expansion

| # | Experiment | Recall | Precision | F1 |
|---|-----------|--------|-----------|------|
| I3a | idea3_steiner_backbone | 0.8208 | 0.2330 | 0.3630 |
| I3b | idea3_steiner_backbone_xiyan | 0.6806 | 0.7917 | 0.7320 |

**Insight**: Filter 없이는 recall 높지만 precision 급락 (backbone이 중간 노드를 과도 포함). **XiYan 추가 시 F1 0.7320으로 크게 개선** — XiYan이 backbone의 노이즈를 효과적으로 제거. 기존 best인 enriched_gat(0.7327)과 거의 동등한 수준.

### 6-4. Idea 4: Connected Component 분리

| # | Experiment | Recall | Precision | F1 |
|---|-----------|--------|-----------|------|
| I4 | idea4_component_aware | 0.7563 | 0.3529 | 0.4813 |

**Insight**: Adaptive PCST(0.3471/0.7210) 대비 recall +0.04, precision +0.01. Component별 독립 threshold가 소규모 component에서 더 정확한 pruning을 수행.

### 6-5. Idea 2+4 결합 (Product Cost + Component Aware)

| # | Experiment | Recall | Precision | F1 | 비고 |
|---|-----------|--------|-----------|------|------|
| I24a | idea24_product_component | 0.7633 | 0.3538 | 0.4835 | Filter 없이 |
| I24b | idea24_product_component_xiyan | 0.6304 | **0.8028** | 0.7063 | + XiYan |

**Insight**: Idea 2+4 결합이 개별 적용보다 우수. **XiYan 포함 시 precision 0.80 달성** — 현재까지 best pipeline 후보 중 하나.

### 6-6. Bayesian Optimization for PCST Cost Ratios

| # | Experiment | Recall | Precision | F1 | 비고 |
|---|-----------|--------|-----------|------|------|
| BO1 | bo_fixed_cost | 0.4793 | 0.7468 | 0.5839 | BO가 찾은 cost: bt=0.195, fk=0.346, macro=0.044 |
| BO2 | bo_score_driven | 0.5910 | 0.7867 | 0.6751 | BO가 찾은 weight: bt=1.955, fk=2.779, macro=3.439 |

**Insight**: BO-score-driven(BO2)이 수동 설정보다 precision +0.01, recall -0.02. BO가 macro_weight를 3.44로 높게 잡아 불필요한 table 연결을 억제. 그러나 기대만큼의 큰 개선은 아님.

### 6-7. Enriched Node Features

| # | Experiment | Recall | Precision | F1 |
|---|-----------|--------|-----------|------|
| E1 | enriched_gat | 0.6658 | **0.8147** | 0.7327 |
| E2 | edge_prize | 0.6823 | **0.8139** | 0.7424 |

**Insight**: **Enriched GAT가 전체 실험 중 최고 precision(0.8147).** Column description, value_description, NL name을 node text에 포함시켜 PLM 임베딩 품질이 향상됨. Edge Prize(triplet relation 기반)도 유사한 성능 — 풍부한 node feature가 핵심.

### 6-8. Query-Conditioned GAT (α=0.85, Projector 기반)

| # | Experiment | α | Recall | Precision | F1 |
|---|-----------|---|--------|-----------|------|
| Q1 | qcond_idea24_xiyan | 0.85 | 0.6236 | **0.8056** | 0.7032 |
| Q2 | supernode_idea24_xiyan | 0.70 | 0.6089 | 0.7922 | 0.6886 |
| Q3 | supernode_idea24_a085_xiyan | 0.85 | 0.6154 | 0.8005 | 0.6958 |

**Insight**: Query-Conditioned Concat(Q1)이 α=0.85에서 precision 0.8056으로 높음. SuperNode(Q2, Q3)는 α 값과 무관하게 약간 낮음. Query 정보 주입은 Projector 기반에서도 효과적.

### 6-9. Query-Conditioned GAT (α=0.0, GAT-only Score)

| # | Experiment | α | Recall | Precision | F1 |
|---|-----------|---|--------|-----------|------|
| Q4 | qcond_idea24_a0_xiyan | 0.0 | 0.5015 | 0.7065 | 0.5867 |
| Q5 | supernode_idea24_a0_xiyan | 0.0 | 0.5237 | 0.7155 | 0.6048 |

**Insight**: α=0.0(cosine 제거, GAT score만 사용)에서는 SuperNode(Q5)가 Concat(Q4)보다 우수 (+0.9%p P, +2.2%p R). α=0.85에서는 반대 — cosine과 결합할 때는 Concat이, GAT 단독일 때는 SuperNode가 더 효과적. 그러나 α=0.0의 절대 성능은 α=0.85 대비 크게 낮아 cosine baseline이 여전히 중요함.

### 6-10. Direct Variant 평가 (BCE only, Projector 제거) — 2026-04-13

DualTowerProjector + InfoNCE를 제거하고 DirectClassifierHead(BCE only)로 학습한 체크포인트로 full pipeline 평가.

| # | Experiment | Loss | Recall | Precision | F1 |
|---|-----------|------|--------|-----------|------|
| Q6 | qcond_direct_idea24_xiyan | BCE only | 0.4384 | 0.6578 | 0.5261 |
| Q7 | supernode_direct_idea24_xiyan | BCE only | 0.4369 | 0.6553 | 0.5243 |

비교 (동일 architecture, loss만 다름):

| 비교 | Loss | Recall | Precision | F1 |
|------|------|--------|-----------|------|
| Q4 qcond Projector | BCE+InfoNCE | 0.5015 | 0.7065 | 0.5867 |
| Q6 qcond Direct | BCE only | 0.4384 | 0.6578 | 0.5261 |
| Q5 supernode Projector | BCE+InfoNCE | 0.5237 | 0.7155 | 0.6048 |
| Q7 supernode Direct | BCE only | 0.4369 | 0.6553 | 0.5243 |

**Insight**: Direct(BCE only)가 Projector(BCE+InfoNCE) 대비 **P -0.05~0.06, R -0.06~0.09로 일관되게 하락**. "query 중복 제거로 GAT 독립 판별력 향상" 가설이 기각됨. 오히려 DualTowerProjector가 query-node joint embedding 공간을 추가 학습하면서 score ranking 품질에 기여하고 있었고, InfoNCE의 contrastive signal이 hard negative mining을 통해 판별력을 강화한 것으로 해석됨. 이는 T4(InfoNCE 추가 시 단독 recall 하락)와 모순되는 결과로, InfoNCE가 단독 recall에는 부정적이지만 **Projector를 경유한 최종 score ranking에는 긍정적**일 수 있음을 시사.

### 6-11. Direct Variant Per-Step Ablation — 2026-04-13

Direct variant의 각 파이프라인 단계별 metric 변화. Subgraph Extractor는 기존 ablation과의 비교를 위해 AdaptivePCSTExtractor 사용.

**QCond Concat Direct:**

| Step | Pipeline | Recall | Precision | F1 |
|------|----------|--------|-----------|------|
| 1 | DirectGATSelector only | 0.9968 | 0.1173 | 0.2098 |
| 2 | + AdaptivePCST + AutoJoinKeys | 0.3904 | 0.2391 | 0.2966 |
| 3 | + XiYan Filter (Full) | 0.4384 | 0.6578 | 0.5261 |

**SuperNode Direct:**

| Step | Pipeline | Recall | Precision | F1 |
|------|----------|--------|-----------|------|
| 1 | DirectGATSelector only | 0.9968 | 0.1173 | 0.2098 |
| 2 | + AdaptivePCST + AutoJoinKeys | 0.3168 | 0.1757 | 0.2261 |
| 3 | + XiYan Filter (Full) | 0.4369 | 0.6553 | 0.5243 |

**Delta 분석 (QCond Concat)**:
- Step 1→2: R **-60.64%p**, P +12.18%p — AdaptivePCST가 과도하게 pruning
- Step 2→3: R +4.80%p, P **+41.87%p** — XiYan이 precision을 3배 가까이 향상

**Delta 분석 (SuperNode Direct)**:
- Step 1→2: R **-68.00%p**, P +5.84%p — QCond보다 더 심한 pruning
- Step 2→3: R +12.01%p, P **+47.96%p** — XiYan의 회복 효과가 더 큼

**QCond vs SuperNode 비교**:
- Step 1: 동일 — 둘 다 top-k 없이 전체 schema를 반환
- Step 2: QCond이 우세 (R +7.36%p, F1 +7.05%p) — Concat 방식이 AdaptivePCST에 더 적합한 score 분포 생성
- Step 3: 거의 동일 (F1 차이 0.18%p) — XiYan Filter가 Extractor 단계의 차이를 상쇄

**Insight**: DirectGATSelector는 top-k 없이 전체 후보를 반환(R=99.68%). AdaptivePCST의 고정 macro_cost(0.5)가 Direct variant의 bimodal score 분포와 불일치하여 과도한 pruning 발생. SuperNode 방식이 특히 더 심한 R 손실을 보이는 것은 SuperNode의 score 분포가 PCST fixed cost와 더 큰 불일치를 가짐을 시사. 최종적으로 XiYan Filter가 두 variant 모두에서 핵심 역할을 하며, Extractor 단계의 차이를 거의 완전히 보상.

### 6-12. Direct Variant Binary Threshold Per-Step Ablation — 2026-04-13

DirectGATSelector에 binary threshold(≥0.5)를 적용한 per-step ablation. 기존(전체 반환)과 비교.

**QCond Concat Direct (Binary threshold=0.5):**

| Step | Pipeline | Recall | Precision | F1 | vs 전체반환 F1 |
|------|----------|--------|-----------|------|--------------|
| 1 | Binary Selector (≥0.5) | 0.4871 | 0.2517 | 0.3319 | +0.1221 |
| 2 | + AdaptivePCST + AutoJoinKeys | 0.3904 | 0.2391 | 0.2966 | 동일 |
| 3 | + XiYan Filter (Full) | 0.4384 | 0.6578 | 0.5261 | 동일 |

**SuperNode Direct (Binary threshold=0.5):**

| Step | Pipeline | Recall | Precision | F1 | vs 전체반환 F1 |
|------|----------|--------|-----------|------|--------------|
| 1 | Binary Selector (≥0.5) | 0.6261 | 0.1885 | 0.2898 | +0.0800 |
| 2 | + AdaptivePCST + AutoJoinKeys | 0.3168 | 0.1757 | 0.2260 | 동일 |
| 3 | + XiYan Filter (Full) | 0.4369 | 0.6553 | 0.5243 | 동일 |

**핵심 발견**:
- **Step 1에서 Binary threshold 적용 시 F1 대폭 개선**: QCond +12.2%p, SuperNode +8.0%p. 전체 반환(R=0.9968/P=0.1173)에서 불필요한 노드가 대량 포함되던 문제 해소.
- **SuperNode이 Binary Recall에서 우세** (0.6261 vs 0.4871): 실제 이진 분류 능력은 SuperNode이 더 높음.
- **Step 2부터 결과 동일**: `AdaptivePCSTExtractor`가 `seed_nodes`를 무시하고 전체 `node_scores`로 prize를 계산하기 때문. Binary filtering의 이점이 Extractor 단계에서 무효화됨.
- **역설적 패턴**: Step 1→2에서 Binary의 경우 F1 하락 (QCond: 0.3319→0.2966) — Selector의 binary decision이 PCST의 score 기반 prize 계산에 의해 덮어씌워짐.
- **구조적 시사점**: seed_nodes를 활용하는 Extractor(예: SteinerBackbonePCST)와 결합해야 Binary threshold의 이점이 downstream에 전파됨.

### 6-13. Binary Threshold × SteinerBackbone Sweep (Selector/Extractor만) — 2026-04-14

Binary threshold 값과 SteinerBackbonePCSTExtractor 결합의 R/P/F1 관계. GAT 추론은 기존 score_analysis를 재사용하여 오프라인으로 계산.

**QCond Concat Direct (Steiner only, no filter):**

| Thresh | Sel R | Sel F1 | +Steiner R | +Steiner P | +Steiner F1 | R delta |
|--------|-------|--------|------------|------------|-------------|---------|
| 0.05 | 0.5738 | 0.3687 | 0.6821 | 0.2520 | 0.3680 | +0.1083 |
| 0.10 | 0.5534 | 0.3729 | 0.6650 | 0.2589 | 0.3727 | +0.1116 |
| 0.20 | 0.5286 | 0.3802 | 0.6463 | 0.2693 | 0.3802 | +0.1177 |
| 0.50 | 0.4862 | 0.3847 | 0.6081 | 0.2867 | 0.3897 | +0.1219 |

**SuperNode Direct (Steiner only, no filter):**

| Thresh | Sel R | Sel F1 | +Steiner R | +Steiner P | +Steiner F1 | R delta |
|--------|-------|--------|------------|------------|-------------|---------|
| **0.05** | 0.7133 | 0.2821 | **0.7860** | 0.1793 | 0.2920 | +0.0726 |
| **0.10** | 0.6921 | 0.2848 | **0.7709** | 0.1836 | 0.2966 | +0.0788 |
| **0.15** | 0.6799 | 0.2874 | **0.7609** | 0.1860 | 0.2990 | +0.0810 |
| **0.20** | 0.6694 | 0.2892 | **0.7535** | 0.1889 | 0.3020 | +0.0840 |
| 0.30 | 0.6514 | 0.2907 | 0.7399 | 0.1923 | 0.3053 | +0.0885 |
| 0.50 | 0.6248 | 0.2957 | 0.7202 | 0.2004 | 0.3135 | +0.0954 |

**핵심 발견**:
- **SuperNode + Steiner가 R≥0.75 달성**: threshold 0.05~0.20에서 R=0.7535~0.7860. Direct variant 중 최초로 recall ceiling 도달.
- **QCond는 R 0.75 불가**: 최고 0.6821. GAT 분류 성능 자체의 한계로 threshold tuning으로 해결 불가.
- **Steiner R 기여 일정**: QCond ~+0.12, SuperNode ~+0.08. Threshold에 거의 독립적 — bridge node 복원 효과가 안정적.
- **Precision은 낮음** (0.18~0.29): Steiner backbone이 무관한 중간 노드를 다수 포함. XiYan Filter 필수.

### 6-14. Binary Threshold × SteinerBackbone + XiYan Full Pipeline — 2026-04-14

6-13에서 R≥0.75 확보된 SuperNode 저-threshold 구간에 XiYan Filter 추가. 최종 full pipeline 성능 평가.

**SuperNode Direct Binary + SteinerBackbone + XiYan:**

| Thresh | Steiner R | +XiYan R | +XiYan P | +XiYan F1 |
|--------|-----------|----------|----------|-----------|
| **0.05** | 0.7860 | **0.6353** | **0.7054** | **0.6684** |
| 0.10 | 0.7709 | 0.6272 | 0.7011 | 0.6621 |
| 0.15 | 0.7609 | 0.6196 | 0.6988 | 0.6569 |
| 0.20 | 0.7535 | 0.6122 | 0.6936 | 0.6508 |

**핵심 발견**:
- **Threshold 낮을수록 우세**: t=0.05가 R/P/F1 모두 최고. 더 많은 seed를 주면 Steiner backbone이 더 많은 gold bridge를 포함하고, XiYan이 그 중 옳은 것만 남겨 최종 성능 향상.
- **Direct variant 최고 F1=0.6684**: 기존 Direct 최고(SuperNode Idea2+4+XiYan, F1=0.5243) 대비 **+14.4%p 대폭 개선**. Binary threshold + Steiner backbone 조합이 DirectGATSelector의 약한 recall을 보완.
- **XiYan의 R 손실 일정** (~-0.15): Steiner가 R=0.786까지 끌어올려도 XiYan 후 0.635로 하락. XiYan이 Steiner backbone에 포함된 정상 gold 노드도 일부 제거함 — Steiner의 P 저하(0.18)가 XiYan의 false negative 원인.
- **Ensemble 기반 best(F1=0.7863)에는 여전히 미달**: GAT 자체 분류 성능(val recall 0.5548)이 Ensemble의 cosine+GAT 조합보다 약하다는 근본 한계.

---

### 6-15. Direct Variant Extractor Ablation Consolidated (a03_06 ~ a03_18) — 2026-04-14

DirectGATSelector (binary, threshold=0.5) 위에서 Selector → Extractor → Filter 단계별 효과를 QCond/SuperNode 양쪽으로 계통적 비교. a03_13~15는 QCond + Fixed/Steiner + (no-filter/XiYan) 조합, a03_16~18은 SuperNode 대칭 조합.

| ID | Selector | Extractor | Filter | Recall | Precision | F1 |
|----|----------|-----------|--------|--------|-----------|-----|
| a03_06 | QCond | AdaptivePCST | — | 0.3904 | 0.2391 | 0.2966 |
| a03_07 | QCond | SteinerBackbone | — | 0.6072 | 0.2154 | 0.3180 |
| a03_08 | QCond | AdaptivePCST | XiYan | 0.3357 | 0.5320 | 0.4116 |
| a03_09 | SuperNode | — | — | 0.6261 | 0.1885 | 0.2898 |
| a03_10 | SuperNode | AdaptivePCST | — | 0.3168 | 0.1757 | 0.2260 |
| a03_11 | SuperNode | SteinerBackbone | — | 0.7120 | 0.1798 | 0.2871 |
| a03_12 | SuperNode | AdaptivePCST | XiYan | 0.2682 | 0.4234 | 0.3284 |
| a03_13 | QCond | PCST (fixed) | — | 0.6748 | 0.1979 | 0.3060 |
| a03_14 | QCond | PCST (fixed) | XiYan | 0.5843 | 0.6929 | 0.6340 |
| a03_15 | QCond | SteinerBackbone | XiYan | 0.5247 | 0.6824 | 0.5932 |
| a03_16 | SuperNode | PCST (fixed) | — | 0.7982 | 0.1587 | 0.2648 |
| **a03_17** | **SuperNode** | **PCST (fixed)** | **XiYan** | **0.6761** | **0.7128** | **0.6940** |
| a03_18 | SuperNode | SteinerBackbone | XiYan | 0.5855 | 0.6871 | 0.6322 |

**핵심 발견**:
- **a03_17 (SuperNode + Fixed PCST + XiYan) F1=0.6940, Direct variant 신기록**: 기존 Direct 최고(6-14 SuperNode Steiner+XiYan t=0.05, F1=0.6684) 대비 +2.6%p. Fixed PCST가 SuperNode의 강한 recall(selector-only R=0.6261, +PCST R=0.7982)을 손실 없이 유지하면서 XiYan이 precision을 0.71까지 끌어올림.
- **Fixed PCST > SteinerBackbone (+XiYan)**: QCond (a03_14 0.6340 > a03_15 0.5932) / SuperNode (a03_17 0.6940 > a03_18 0.6322) 양쪽 모두 동일 경향. Steiner의 backbone_bonus(0.5)가 저점수 bridge를 강제 포함시켜 XiYan 후 noise로 남음.
- **Fixed PCST > AdaptivePCST (+XiYan)**: QCond (a03_14 0.6340 vs a03_08 0.4116), SuperNode (a03_17 0.6940 vs a03_12 0.3284) 양쪽 모두. Adaptive의 P80 per-query threshold가 binary-classified score 분포에서 과도한 pruning 유발.
- **SuperNode > QCond (동일 extractor+XiYan)**: a03_17 > a03_14, a03_18 > a03_15 일관. SuperNode가 recall-heavy하므로 XiYan filtering과 상보적.
- **Filter 없으면 SuperNode는 P<0.20**: a03_09/10/11/16 모두 precision 0.15~0.19로 저조. XiYan이 사실상 필수.

---

### 6-16. a05 Agentic Filter Ablation — 2026-04-15

Anchor: a03_17 (SuperNode Direct + Fixed PCST). Filter만 교체하여 agentic refinement 효과 비교. Backbone: Qwen3-Coder-30B-A3B-Instruct-FP8 (vLLM, GPUs 2+3). a05_13/14/15/17만 backbone 민감도 측정용 gpt-4o-mini (OpenAI API).

| ID | Filter | Recall | Precision | F1 | Runtime |
|----|--------|--------|-----------|------|---------|
| a03_17 (anchor) | XiYan | 0.6761 | 0.7128 | **0.6940** | — |
| a05_01 | AdaptiveMultiAgent (Semantic+Structural+Skeptic) | 0.3770 | 0.6276 | 0.4713 | 10h 23m |
| a05_02 | ReflectionFilter (1 iter, propose→critique→revise) | **0.7320** | 0.6833 | **0.7068** | 3h 18m (7.3s/q) |
| a05_04 | VerifierFilter (XiYan + NL Unit Tester) | 0.7093 | 0.6676 | 0.6878 | 6h 48m (16.0s/q) |
| a05_13 | XiYan (gpt-4o-mini backbone, prune-only) | 0.6037 | **0.7317** | 0.6616 | 38m (1.10s/q) |
| a05_14 | AdaptiveMultiAgent (gpt-4o-mini backbone) | 0.3992 | 0.7576 | 0.5230 | 176m (6.9s/q) |
| a05_15 | ReflectionFilter 1iter (gpt-4o-mini backbone) | 0.6827 | 0.6620 | 0.6722 | 131m (5.1s/q) |
| a05_17 | VerifierFilter (gpt-4o-mini backbone) | 0.7055 | 0.6385 | 0.6706 | 206m (8.1s/q) |

**관찰**:
- **a05_01 F1=0.4713, anchor 대비 −22.3%p**: 3-agent consensus가 지나치게 보수적으로 교집합화 — Recall 0.38로 anchor 대비 -30%p 대폭 손실. Precision도 anchor XiYan보다 낮음 (0.63 < 0.71).
- JSON Parsing failed warning 빈발: agents.py fallback이 Unanswerable로 처리되어 빈 선택 누적 → Recall 파괴.
- **a05_02 F1=0.7068, anchor 대비 +1.3%p (신기록)**: Critique-revise가 Recall을 0.68→0.73으로 밀어올림. Precision은 0.71→0.68로 소폭 하락하나 net F1 상승. **Restore path 확보가 실제로 Recall 천장을 돌파**함을 실증.
- **a05_04 F1=0.6878, anchor 대비 −0.6%p**: XiYan-style 초기 필터 + NL unit test 생성 + missing_nodes 복원. Recall 0.7093로 anchor 대비 +3.3%p 회복되나 Precision 0.6676으로 -4.5%p 하락 → net F1 하락. Unit tester의 missing 판정이 recall 확보에는 효과 있으나 정교하지 않은 복원으로 noise 유입. ReflectionFilter 대비 F1 -1.9%p (0.6878 vs 0.7068) — critique-revise의 통합 추론이 generate-then-check 분리 파이프라인보다 우월.
- **a05_13 F1=0.6616, anchor 대비 −3.2%p**: XiYanFilter의 LLM만 Qwen3-Coder-30B → gpt-4o-mini로 교체 (구조·프롬프트 동일, 1534쿼리 38분, ~$0.41). Precision +0.0189 (0.7128→0.7317)로 소폭 개선되나 Recall −0.0724 (0.6761→0.6037)로 크게 손실 — gpt-4o-mini가 schema-value 판단 시 더 보수적으로 컬럼을 제거. Prune-only 구조에서는 backbone 교체가 recall 병목을 오히려 악화. 토큰 사용: input 2.70M (fresh 2.56M + cached 0.14M, 캐시 히트율 5.15%) / output 30K. F3/F4(a05_11/12)는 restore path가 있어 민감도 다를 수 있음.
- **a05_14 F1=0.5230 (AdaptiveMultiAgent + gpt-4o-mini), a05_01(Qwen) 대비 +5.2%p / anchor 대비 −17.1%p**: 3-agent consensus 구조는 backbone 교체로도 구조적 한계 유지. Recall 0.3992 (a05_01 0.3770 대비 +0.022) / Precision 0.7576 (0.6276 대비 +0.130). gpt-4o-mini가 agent별 JSON parsing을 더 안정적으로 수행하여 parsing 실패로 인한 빈 선택 손실은 일부 해소되나, consensus 교집합화로 인한 recall 파괴가 여전. 토큰 1.43M input / 503K output / cached 0, 비용 ~$0.52. 3-agent 구조 자체가 prune-only에서는 recall 천장 상실 근본 원인.
- **a05_15 F1=0.6722 (Reflection 1iter + gpt-4o-mini), a05_02(Qwen) 대비 −3.5%p / anchor 대비 −2.2%p**: 구조는 유지 (propose→critique→revise, 1iter) 하나 backbone 교체로 F1 손실. Recall 0.6827 (a05_02 0.7320 대비 −0.049) / Precision 0.6620 (0.6833 대비 −0.021). gpt-4o-mini의 critique가 Qwen 대비 원 subgraph 밖 노드 재도입을 덜 공격적으로 수행 → recall 천장 돌파 효과 약화. 토큰 8.13M input / 193K output / cached 144K (1.8%), 비용 ~$1.32. Reflection 구조는 a05_13/14 대비 backbone 민감도가 가장 큼 (F1 −3.5%p vs −3.2%p / +5.2%p) — critique의 질이 backbone 능력에 비례하는 structural 특성.
- **a05_17 F1=0.6706 (Verifier + gpt-4o-mini), a05_04(Qwen) 대비 −1.7%p / anchor 대비 −2.3%p**: XiYan 초기 필터 + NL unit test 생성·검증 구조 유지. Recall 0.7055 (a05_04 0.7093 대비 −0.004로 거의 동일) / Precision 0.6385 (0.6676 대비 −0.029). Verifier는 NL unit test로 missing을 복원하는 recall path 의존도가 높아 backbone 영향이 상대적으로 작음 — Reflection 대비 민감도 절반 수준. 토큰 8.06M input / 496K output / cached 87K (1.1%), 비용 ~$1.50. JSON Parsing failed warning 일부 발생하나 fallback이 Initial nodes 유지로 recall 보존.
- **GPT-4o-mini backbone 민감도 종합**: prune-only(a05_13 −3.2%p) < Verifier(a05_17 −1.7%p) < AdaptiveMultiAgent(a05_14 +5.2%p vs Qwen 최악치) / Reflection(a05_15 −3.5%p). **Recall path 유무와 무관하게 전반적으로 F1 2~3.5%p 하락**, 3-agent 구조만 parsing 안정성 이득으로 Qwen 대비 개선. Qwen3-Coder-30B가 schema linking에서 gpt-4o-mini 대비 구조적 우위. 4개 실험 총 비용 ~$3.76 (a05_13 $0.41 + a05_14 $0.52 + a05_15 $1.32 + a05_17 $1.50).
- 향후 agentic filter는 (1) prune-only 대신 restore 경로 확보, (2) parsing robustness, (3) fallback 시 XiYan 결과 유지가 필수.

---

### 6-17. a09 TopologyCost PCST Ablation (edge-type param-free, NoFilter) — 2026-04-16

**동기**: BO (6-6)가 edge-type cost weight (bt/fk/macro)를 튜닝했지만 F1=0.6751로 plateau 도달. 튜닝 공간 자체가 한계일 가능성 — 방향 2로 edge-type 파라미터를 제거하고 **그래프 토폴로지 (degree) 기반 cost**로 전환. Filter가 Precision을 담당한다는 전제 하에 extractor는 Implicit Bridge Table 확보 (Recall-oriented) 에 집중. 본 ablation은 **Filter 없이** raw extractor 순효과만 측정.

**Cost 공식**: `c(u,v) = cost_scale × (1 / (1 + γ·log(1+max(deg_u,deg_v)))) × (1 / (1 + λ·(norm_p_u+norm_p_v)))`
- 고차원 노드(테이블)를 지나는 bridge를 저렴화 → belongs_to/FK 경로 확보 촉진
- Prize는 tiebreaker로만 작용 (λ=0.3, 약한 term)
- Edge-type 파라미터 (bt/fk/macro) 완전 제거

**결과** (Ensemble α=0.85 + 동일 GAT checkpoint + NoFilter 고정):

| ID | Extractor | Recall | Precision | F1 | Δ vs a09_05 |
|----|-----------|--------|-----------|------|-------------|
| a09_05 (parent) | AdaptivePCST | 0.7210 | **0.3471** | **0.4686** | — |
| **a09_01** | **TopologyCost (γ=1, λ=0.3)** | **0.7318** | 0.3412 | 0.4654 | **−0.0032** |
| a09_04 (I24a 계열) | CA-ProductCost | 0.7489 | 0.3333 | 0.4613 | −0.0073 |
| a09_02 | CA-TopologyCost | 0.7463 | 0.3313 | 0.4590 | −0.0096 |
| a09_03 | Basic PCST (fixed) | **0.9679** | 0.1276 | 0.2255 | −0.2431 |

**핵심 발견**:
- **TopologyCost 순효과: F1 −0.0032** (직계 부모 AdaptivePCST 대비 미세 후퇴). Recall +0.0108 얻었으나 Precision −0.0059 더 떨어져 상쇄 — topology cost의 bridge 확보 효과는 관측되지만 추가 노드 유입이 precision 손실을 압도하지 못함.
- **Edge-type 파라미터 3개 제거해도 F1 동등**: TopologyCost는 bt/fk/macro 3개 파라미터를 완전히 삭제하고도 Adaptive와 −0.3% 내에서 동작 → BO plateau가 tuning 공간 한계가 아니라 **extractor 자체의 plateau**임을 시사. edge-type cost를 포기해도 잃는 것이 없음 (단, 더 얻는 것도 없음).
- **CA mixin이 TopologyCost에서는 역효과** (a09_01 → a09_02, F1 −0.0064). ProductCost에서 CA가 얻었던 F1 +0.02 이득과 **반대 패턴** — component 분해된 국소 threshold가 이미 prize-aware cost를 덮어쓰는 중복 작용 가능성.
- **Basic PCST의 recall (0.9679)이 가장 높지만 F1 최하 (0.2255)**: NoFilter에서는 recall-only 전략이 실패. 과거 Basic+XiYan F1=0.7863 기록을 고려하면 recall 대량 확보 전략은 Filter와 결합해야만 가치를 가짐.
- **"Filter가 precision 담당" 가설의 한계**: NoFilter 시점에서 순수 extractor recall 개선 폭이 +0.0108에 불과 → TopologyCost는 bridge 확보보다는 Adaptive 대비 marginal variant 수준. Filter 추가 시 topology cost의 micro-recall 이점이 살아나는지는 별도 pass 필요.

**다음 의사결정**:
1. TopologyCost + XiYanFilter 조합 평가 (a09_01 + XiYan) — recall 이득이 Filter 이후 net F1 양수로 전환되는지 확인
2. γ/λ/cost_scale grid 또는 small BO — 본 실험의 (1.0, 0.3, 0.1)이 sweet spot인지 검증
3. topology signal 교체 실험 — PageRank, clustering coefficient 등 degree 외 대안

---

### 6-18. a10 FK-Backbone Steiner Closure (Graph-Structure 기반 Recall ≥ 0.85 달성) — 2026-04-16

**동기**: a09 TopologyCost가 F1 ±0.003 수준 plateau를 확증 — 기존 단일-단계 PCST는 edge cost 공간을 아무리 바꿔도 한계 달성. 근본적인 전환이 필요: "**Filter 이전에 Recall 0.85 이상 확보**"를 목표로, DB schema의 2-레벨 구조 (Table FK backbone + Column membership) 를 명시적으로 활용.

**설계**: 단일 PCST를 두 단계로 분해
1. **FK Backbone Steiner Tree** (table 수준): Selector가 고점수를 준 테이블 + high-score 컬럼의 parent 테이블을 terminal로, `table_to_table` FK edge로 구성된 G_fk 그래프에서 Kou-Markowsky-Berman 2-근사 Steiner tree를 구한다. **Implicit Bridge Table이 구조적으로 보장됨** (PCST는 bridge table의 prize가 낮으면 드랍했던 것을 여기선 Steiner 알고리즘이 필연적으로 포함).
2. **Column Recovery**: Closed backbone에 속한 테이블 안에서 `score ≥ θ_r` 인 컬럼 복원 + `force_fk_columns=True` 로 closed 테이블 간 FK 컬럼 강제 포함.

**Extractor**: `FKBackboneSteinerExtractor` (AdaptivePCST 상속, percentile=80, min/max_prize=3/25 재사용 → terminal 선정)

**결과** — 11-point θ_r sweep (Ensemble α=0.85 + 동일 GAT checkpoint + NoFilter 고정, 2026-04-16 완료):

| ID | θ_r | Recall | Precision | F1 | Δ F1 vs a09_05 |
|----|-----|--------|-----------|------|----------------|
| a09_05 (anchor) | — (AdaptivePCST) | 0.7210 | 0.3471 | 0.4686 | — |
| a10_01 | 0.0 | **0.9492** | 0.1567 | 0.2690 | −0.1996 |
| a10_04 | 0.1 | 0.9481 | 0.1582 | 0.2711 | −0.1975 |
| a10_05 | 0.2 | 0.9418 | 0.1644 | 0.2800 | −0.1886 |
| a10_02 | 0.3 | 0.9293 | 0.1812 | 0.3033 | −0.1653 |
| a10_06 | 0.4 | 0.9014 | 0.2125 | 0.3439 | −0.1247 |
| a10_03 | 0.5 | 0.8565 | 0.2627 | 0.4021 | −0.0665 |
| a10_07 | 0.6 | 0.7789 | 0.3341 | 0.4677 | −0.0009 |
| a10_08 | 0.7 | 0.6662 | 0.4245 | 0.5185 | +0.0499 |
| **a10_09 ★** | **0.8** | **0.5455** | **0.5044** | **0.5241** | **+0.0555** |
| a10_10 | 0.9 | 0.4083 | **0.5300** | 0.4612 | −0.0074 |
| a10_11 | 1.0 | 0.2972 | 0.4920 | 0.3706 | −0.0980 |

**핵심 발견**:
- **Recall ≥ 0.85 목표 달성 구간**: θ_r ∈ [0.0, 0.5] 의 6개 config 모두 0.85 상회, 최고 0.9492 (θ_r=0.0). "0.1+ Recall jump" 요구는 구조적으로 해결됨 (기존 anchor 0.7210 대비 +0.2282). PCST의 cost 조정으로는 도달 불가능한 영역.
- **F1 Peak: θ_r=0.8 (F1=0.5241)**: a09_05 AdaptivePCST anchor (F1=0.4686) 대비 **+0.0555**, NoFilter 체제에서 순수 Extractor 최고치. θ_r=0.7 (0.5185), 0.8 (0.5241), 0.9 (0.4612) 의 inverted-U 형태로 피크 명확.
- **P > R Crossover**: θ_r ≈ 0.8 부근 (a10_09 에서 R=0.5455 / P=0.5044 거의 균형, θ_r=0.9 에서 P=0.5300 > R=0.4083 로 완전 역전).
- **Precision 노이즈 천장**: P는 θ_r=0.9 에서 0.5300 이 한계, θ_r=1.0 에서는 오히려 **0.4920 으로 하락** — FK 강제 컬럼만 포함해도 bridge FK 자체가 gold 에 없는 경우가 있어 Steiner closure 에는 구조적 precision 상한이 존재.
- **FK-Backbone Steiner의 Dual Operating Point**:
  1. **Recall-first (θ_r=0.5)**: R=0.8565, P=0.2627, F1=0.4021 → Filter 전 recall 담보용
  2. **F1-first (θ_r=0.8)**: R=0.5455, P=0.5044, F1=0.5241 → NoFilter 단독 최고 F1
- **구조적 vs Cost-기반 대비**: a09 TopologyCost는 edge cost 완화로 Recall +0.0108 에 그쳤으나, a10 은 Steiner closure 강제로 Recall +0.2282 확보 — **"cost 튜닝 plateau" 를 돌파하는 길은 구조 이용뿐**임이 실증.

**다음 의사결정**:
1. **XiYan Filter 결합** (GPU 여유 시) — a10_03 (θ_r=0.5, R=0.8565, P=0.2627) 자리에서 Filter 가 P 를 얼마나 끌어올리는지가 최종 판정. 목표: 기존 chain top (a03_17 F1=0.6940 / abl_ens_basic_xiyan F1=0.7863) 돌파. Filter 투입 후 optimal θ_r 은 별도 sweep 필요할 수 있음 (P 여유가 Filter 부담과 trade).
2. **force_fk_columns / fallback_to_parent 플래그 ablation** — 두 구조적 강제 요소 중 어느 쪽이 recall jump 를 주로 견인했는지 분해 (force_fk_columns=False / fallback_to_parent=False 각각 실험).
3. **θ_r=0.8 근방 세밀 sweep** — F1 피크 0.05 간격 (0.75, 0.85) 추가 확인으로 최적점 정밀화.
4. **Selector 교체 영향** — 본 실험은 `best_gat_model.pt` 고정. s06 bottleneck fix GAT checkpoint 로 교체 시 seed 품질 향상이 Extractor-stage recall/precision 에 어떻게 전파되는지 별도 pass.

---

### 6-19. FK-Backbone Steiner Column Recovery Percentile Sweep (Offline) — 2026-04-17

**동기**: a10 FKBackboneSteiner의 `column_recovery_threshold θ_r`은 **절댓값**이다. Ensemble Selector(α=0.85)는 raw cosine이 지배적이므로 DB/query별 score 분포가 이동 → "θ_r=0.8"이 쿼리에 따라 상위 10~40% 사이에서 움직인다. Per-query **percentile 기준**으로 재정의하면 이 변동을 흡수할 수 있는지 검증 (α 조정/GAT 교체는 selector 세션의 재설계를 기다리기로 하고, extractor-side normalization만 먼저 시도).

**방법 (오프라인 재평가)**: `FKBackboneSteinerExtractor` 에 `column_recovery_percentile` + `column_recovery_percentile_scope` 파라미터 추가. **a10_09 의 `score_analysis_*.jsonl` (per-query node scores) + dev graph cache를 재사용** 해 Selector를 재실행하지 않고 extractor stage만 재평가. 각 config마다 1534 쿼리 × (Step 4a column recovery 기준만 변경) → R/P/F1 macro 계산 (auto_join_keys 포함). Smoke test로 `abs θ_r=0.8` config 가 a10_09 출력과 bit-exact 일치함을 확인 (R=0.5455 / P=0.5044 / F1=0.5242).

**4 scopes × 21 percentiles (0, 5, 10, …, 95, 100) = 84 configs + abs anchor (θ_r=0.8) = 85 configs**. Total runtime ~90s.

**Scope 정의**:

| Scope | Percentile 계산 모집단 |
|---|---|
| `global` | 쿼리의 모든 노드 (table + column + fk) score |
| `all_cols` | 쿼리의 모든 컬럼 노드 score |
| `closed_cols` | Steiner closure로 확정된 테이블 내부 컬럼 score (candidate 한정) |
| `per_table` | closed table 각각 독립 (테이블별 percentile) |

**결과 — 각 scope F1 peak**:

| Scope | 최적 p | Recall | Precision | F1 | Δ vs abs_anchor |
|---|---|---|---|---|---|
| abs_anchor (θ_r=0.8) | — | 0.5455 | 0.5044 | **0.5242** | — |
| `global` | p=95 | 0.5754 | 0.4776 | 0.5219 | −0.0023 |
| **`all_cols` ★** | **p=95** | **0.6167** | 0.4626 | **0.5287** | **+0.0045** |
| `closed_cols` | p=95 | 0.5688 | 0.4899 | 0.5264 | +0.0022 |
| `per_table` | p=100 | 0.5471 | 0.4928 | 0.5185 | −0.0057 |

**High-Recall 운영점 (R ≥ 0.85, Filter-앞 후보)**:

| Scope | p | Recall | Precision |
|---|---|---|---|
| `global` | 50 | 0.8998 | 0.2058 |
| `all_cols` | 55 | 0.8801 | 0.2151 |
| **`closed_cols`** | 50 | **0.8522** | **0.2389** |
| `per_table` | 50 | 0.8293 | 0.2173 |

`closed_cols` 가 R~0.85 구간에서 P 최고 — Steiner closure 로 후보군이 이미 필터된 pool 에서 선별 → 과도 포함 억제. 절댓값 anchor 대비 같은 R 수준에서 P 가 높은 단일 config 운영점 확보.

**핵심 발견**:
- **개선 폭 작지만 양 효과**: `all_cols p=95` 가 abs_anchor 대비 **+0.0045 F1**. 큰 jump 는 아니어도 per-query calibration 이 제한적 이득을 준다.
- **`global` ≈ `all_cols`**: 쿼리 노드 중 컬럼이 ~95% 를 차지해 두 분포가 거의 동일. `global` scope 는 사실상 중복 정의.
- **`closed_cols` 는 targeted interpretation**: "복원 후보군에서 상위 p%" 로 해석이 깔끔. F1 peak 도 `all_cols` 에 근접(0.5264 vs 0.5287, −0.0023). **고-Recall 구간(R>0.85)에서 P 최고 → Filter-앞 단계에서 선호되는 scope**.
- **`per_table` 은 floor 도 천장도 낮음**: p=5 에서도 R=0.9041(타 scope는 0.94+), F1 peak 도 최하(0.5185). 작은 테이블에서 percentile noise 로 gold 손실. **추천 X**.
- **모든 scope가 고-percentile (p=90~100)에서 peak**: `θ_r=0.8` 절댓값이 실제로 상위 ~5% 의 고정 cut 에 해당. 현재 regime 이 이미 dev set 평균 최적 근처임을 재확인.
- **per-query calibration의 이득이 제한적인 이유**:
  1. Score 분포가 쿼리 간 의외로 일관적 (all-MiniLM cos 가 0.2~0.9 범위에 안정).
  2. 이미 `adaptive_threshold P80` 이 seed_tables 단계에서 per-query normalize 중.
  3. Column recovery 단계의 변동 흡수 이득은 주변부에 국한.

**권장사항**:
- **F1 최대화**: `all_cols p=95` (+0.0045 F1, interpretability 향상).
- **High-Recall 운영점(Filter-앞)**: `closed_cols p=50` (R=0.8522, P=0.2389) — `all_cols p=55` 의 R=0.8801 보다 R 은 낮지만 P 는 더 높아 Filter 부담 최소화.
- **`per_table` 비추천**, **`global` 불필요** (`all_cols` 와 중복).

**다음 의사결정**:
1. **`all_cols p=95` + XiYan Filter 조합** — 절댓값 θ_r=0.8 + XiYan 대비 net F1 유지 되는지.
2. **`closed_cols p=50` + XiYan 조합** — high-Recall operating point 가 Filter 결합 후 최종 F1 peak 갱신하는지.
3. **GAT 재학습 후 재실행** — selector 세션의 새 GAT checkpoint 배포 후 동일 sweep 재수행. Percentile 방식이 score scale 변화에 robust 한지 검증.
4. **Micro-averaged vs macro-averaged 비교** — 본 분석은 query macro. 쿼리 크기 편향 보정을 위해 micro 도 병행 계산 시 해석 명확해질 수 있음.

**산출물**:
- 분석 MD: [notebooks/analysis_results/fk_steiner_percentile_sweep.md](notebooks/analysis_results/fk_steiner_percentile_sweep.md)
- CSV: [notebooks/analysis_results/fk_steiner_percentile_sweep.csv](notebooks/analysis_results/fk_steiner_percentile_sweep.csv) (85 rows)
- 스크립트: [src/analysis/fk_steiner_percentile_sweep.py](src/analysis/fk_steiner_percentile_sweep.py)
- Extractor 확장: `FKBackboneSteinerExtractor` (pcst.py) — `column_recovery_percentile`, `column_recovery_percentile_scope ∈ {global, all_cols, closed_cols, per_table}` 추가. 미지정(None) 시 기존 절댓값 모드 유지 (backward compatible).

---

### 6-20. Builder Phase A — Infrastructure (B-I / B-II / B-III) — 2026-04-20

Builder 모듈 세션에서 3개 인프라 제안을 일괄 구현. 모두 **인프라 layer** — 후속 모듈(Selector S-II/S-III/S-V, Extractor E-III, Filter FL-III)이 활용해야 end-to-end 효과 측정 가능. 본 항목은 빌더 단독 검증(스모크 테스트)까지의 기록이며, full pipeline 메트릭은 후속 작업에서 anchor (E1/E2) 대비 noise 수준 일치 확인 후 갱신.

#### B-III. FK reachability precompute (★ 최우선, 임계 경로)

**구현**: `HeteroGraphBuilder._compute_fk_reachability()` 추가. 모든 빌더(`HeteroGraphBuilder` / `EnrichedHeteroGraphBuilder` / `TripletGraphBuilder`)의 `build()` 종료 직전 `metadata.update(self._compute_fk_reachability(...))`. `scipy.sparse.csgraph.shortest_path` + `connected_components` + BFS predecessor walk 기반.

**메타데이터 추가 키 (8종)**:
- `fk_adjacency` — `np.ndarray[T,T] int8` (방향성 FK 인접)
- `fk_adjacency_undirected` — 조인 경로용 무방향 버전
- `fk_reachability` — `np.ndarray[T,T] bool` (transitive closure, undirected)
- `fk_distance` — `np.ndarray[T,T] float32` (BFS hop, 비도달 = `inf`)
- `fk_shortest_paths` — `Dict[(i,j), {distance, table_path, edge_path, fk_edge_ids}]`
- `fk_components` — `Dict[table_idx, comp_id]`
- `fk_num_components` — int
- `fk_edge_lookup` — `Dict[(src_tbl, dst_tbl), List[fk_edge_id]]` (멀티-FK 처리)

**스모크 테스트 결과**: `scripts/smoke_test_b3_fk_reach.py`
- california_schools: T=3, FK=2, reach=1.000, components=1, paths=6 (3×3 - diag)
- BIRD-Dev 11개 DB 전체: 1,171 multi-table queries, 1,793 gold join pairs 중 1,677 covered
  - **Pair coverage 0.9353, Query coverage 0.9445**
- 미스 분포 (top): debit_card_specializing 47, card_games 31. 모두 **선언되지 않은 shared-column join** (e.g., `cards.setCode = set_translations.setCode`) — 인프라 한계가 아닌 BIRD 스키마 자체의 비-FK 조인.

**해석**: 95% target에 약간 미달이지만 root cause는 BIRD의 schema 선언 불완전성. 후속 작업으로 (1) 컬럼명 매칭 기반 implicit FK 추론을 metadata에 보조키로 추가하거나, (2) Filter 단계에서 LLM이 join key를 추가 추론하는 방식으로 보완 가능. 현 단계에서는 인프라로 충분.

**호환성**: metadata_dict는 추가만 하고 기존 키(`table_to_id`, `col_to_id`, `fk_to_id`, `node_metadata`, `edges`, `edge_types`)는 변경 없음. 캐시(`BIRDGraphDataset`)는 graph data만 저장하고 metadata를 저장하지 않으므로 inference 시 fresh metadata가 흐른다 (캐시 무효화 불필요).

**ID**: `abl_build_01_fk_reach` — anchor `s03_a07_01_enriched_gat` (E1, F1 0.7327). Pipeline은 metadata key를 무시하므로 noise 수준 일치 예상. (full pipeline 실행은 selector S-V/extractor E-III/filter FL-III 와 묶어 진행 예정)

#### B-II. LineGraph builder (EHGAT 인프라)

**구현**: `src/modules/builders/line_graph_builder.py` (신규). `LineGraphBuilder(BaseGraphBuilder)`. base builder를 wrapping (`_BASE_REGISTRY` dict로 `EnrichedHeteroGraphBuilder` / `TripletGraphBuilder` / 기본 셋 모두 지원).

**구조 변환**:
- 원본 그래프의 모든 PCST 후보 edge → 새 그래프의 노드 (`edge_node`)
- 두 edge가 동일 노드를 공유 → `(edge_node, shares_node, edge_node)`
- `EDGE_TYPE_ORDER = ["belongs_to", "is_source_of", "points_to", "table_to_table"]` (cross-DB 일관성)

**Edge feature** (총 dim 772 또는 1156):
- type one-hot (4) + endpoint mean embedding (384) + (`include_endpoint_diff=True` 시) abs diff (384) + (Triplet base 시) triplet edge embedding (384)

**Params**: `base="EnrichedHeteroGraphBuilder"`, `base_params`, `include_endpoint_diff=True`, `skip_macro_edges=False`

**Metadata 추가 키**: `edge_node_to_orig`, `orig_node_to_edges`, `edge_type_order`, `edge_type_to_idx`, `edge_feature_dim`, `edge_label_rule="both_endpoints_gold"`, `orig_data` (원본 HeteroData 보존), `orig_metadata`. FK reachability 키들도 forward.

**스모크 테스트**: `scripts/smoke_test_b2_linegraph.py`
- california_schools: 97 edge_nodes, 3,856 line_edges (feat_dim 772)
- financial: 87 edge_nodes, 1,060 line_edges
- Triplet base: feat_dim 1,156 (+384 triplet embedding) 정상 확인

**한계**: end-to-end 파이프라인은 **Selector S-III (EHGAT)** 가 `edge_node` HeteroData를 소비할 수 있어야 동작. 현재는 builder-side smoke marker로만 등록.

**ID**: `abl_build_02_linegraph` — anchor `s03_a07_02_edge_prize` (E2, F1 0.7424). S-III 후 재실행.

#### B-I. RFM-compatible serialize API

**구현**: `RFMCompatibleBuilder(EnrichedHeteroGraphBuilder)` 추가 (graph_builder.py). Enriched와 동일한 graph + metadata에 RFM 직렬화 텍스트/토큰 추가.

**Special tokens**: `[DB] [TAB] [/TAB] [COL] [TYPE] [PK] [DESC] [VAL] [FKS] [FK→]`

**직렬화 포맷**:
```
[DB] db_id [TAB] tab_name (NL_name) [COL] col1 [TYPE] type [PK] [DESC] desc [VAL] v1 | v2 | v3 [/TAB] ... [FKS] tab1.col1 [FK→] tab2.col2 ...
```

**Params**: `include_values=True`, `max_values=3`, `value_max_chars=50`, `max_desc_chars=200`

**Metadata 추가 키**: `rfm_text`, `rfm_tokens` (List[str], 화이트스페이스 split), `rfm_special_tokens`

**API**: `build()` (그래프 + 직렬화) 또는 `serialize(db_id, db_dir)` (오프라인 텍스트만)

**스모크 테스트**: `scripts/smoke_test_b1_rfm.py`
- BIRD-Dev 11 DB 토큰 수: min 203 / median 1,041 / mean 1,177 / max 2,578 (european_football_2)
- 4K context 충분히 수용. 별도 토큰화기 wrapping 시 special_tokens는 vocab 추가 필요 — 후속 RFM 인코더 (S-II) 작업에서 처리.

**한계**: 현재 파이프라인의 GAT/PCST/XiYan stack은 `rfm_text`/`rfm_tokens` 키를 무시 → behavioral identical to Enriched. 실제 효과는 **Selector S-II (RFM encoder)** 가 wired 된 후 측정.

**ID**: `abl_build_03_rfm_tokens` — anchor `s03_a07_01_enriched_gat` (E1, F1 0.7327). S-II 후 재실행.

#### 공통 (인터페이스 / 캐시)

- **인터페이스 계약 보존**: `(HeteroData, metadata_dict)` 반환 형태 유지. metadata 키 추가만, 기존 키 변경 없음. Selector/Extractor/Filter는 무지(無知) 상태로 동작.
- **캐시 라우팅**: `bird_dataset.py`에서 `type(builder).__name__ == "EnrichedHeteroGraphBuilder"` 조건을 `isinstance(builder, EnrichedHeteroGraphBuilder)`로 변경 → 서브클래스(Triplet/RFM)도 `_enriched` cache 공유.
- **모듈 export**: `src/modules/builders/__init__.py`에서 6개 빌더 모두 노출 (Hetero/Enriched/Triplet/RFM/LineGraph/Cached).

#### 다음 의사결정

1. **B-III full-pipeline anchor 검증**: `abl_build_01_fk_reach` 실행해 E1과 noise 일치(±0.5pp) 확인. 차이 발생 시 metadata pass-through에 사이드 이펙트 의심.
2. **Selector 세션과 인터페이스 정렬**: S-V (FK metadata 활용 게이트), S-III (LineGraph encoder), S-II (RFM encoder) 작업 순서 조율.
3. **Implicit FK 보강 검토**: 컬럼명 + 타입 매칭 기반 후보 FK를 `metadata['fk_implicit']`로 추가하는 안. 현재 6.47% 미스를 추가로 회수할 잠재력. (선행 의사결정: explicit FK만으로 부족하다고 판정될 때)

**산출물**:
- 빌더 코드: [src/modules/builders/graph_builder.py](src/modules/builders/graph_builder.py) (FK reach + RFM), [src/modules/builders/line_graph_builder.py](src/modules/builders/line_graph_builder.py) (LineGraph)
- 스모크 스크립트: [scripts/smoke_test_b3_fk_reach.py](scripts/smoke_test_b3_fk_reach.py), [scripts/smoke_test_b2_linegraph.py](scripts/smoke_test_b2_linegraph.py), [scripts/smoke_test_b1_rfm.py](scripts/smoke_test_b1_rfm.py)
- 실험 config: [configs/experiments/abl/build/fk_reach/abl_build_01_fk_reach.yaml](configs/experiments/abl/build/fk_reach/abl_build_01_fk_reach.yaml), [configs/experiments/abl/build/linegraph/abl_build_02_linegraph.yaml](configs/experiments/abl/build/linegraph/abl_build_02_linegraph.yaml), [configs/experiments/abl/build/rfm_tokens/abl_build_03_rfm_tokens.yaml](configs/experiments/abl/build/rfm_tokens/abl_build_03_rfm_tokens.yaml)

---

### 6-20.b. Builder Phase A 보강 — B-II.b T2T toggle / B-III.b Schema diameter — 2026-04-21

지도교수 2026-04-21 미팅 §4 의견 2 처방을 Builder 인프라에 흡수. 두 항목 모두 빌더 단독 인프라 (downstream consumer 가 추후 활용).

#### B-II.b — Base heterograph T2T edge toggle

**구현**: `HeteroGraphBuilder.__init__(add_t2t_edges: bool = True)` 추가. False 시 base graph 와 PCST flat 표현 모두에서 `(table, table_to_table, table)` macro edges 가 제거됨. `EnrichedHeteroGraphBuilder` / `TripletGraphBuilder` / `RFMCompatibleBuilder` 모두 super().__init__(**kwargs) 통해 자동 전파. `LineGraphBuilder.skip_macro_edges` 와 **직교** (base on/off × line-graph on/off → 4 조합).

**캐시 라우팅**: `bird_dataset.py` 가 `getattr(builder, "add_t2t_edges", True) is False` 시 cache suffix `_no_t2t` 추가 → 기존 cache 와 충돌 없음.

**스모크 결과** (`scripts/smoke_test_b2b_no_t2t.py`, california_schools):
- `HeteroGraphBuilder`: T2T edges 4 → 0, total PCST 97 → 93, FK reachability 동일, schema_diameter 4 → 8 (FK→column→FK 우회 거리)
- `EnrichedHeteroGraphBuilder`: 동일 결과
- HeteroData T2T edge type properly absent when off
- `metadata['add_t2t_edges']` 가 사용 중인 값 그대로 노출

**해석**: schema_diameter 가 4→8 로 정확히 두 배 증가하는 것이 핵심 신호. T2T 가 짧은 경로를 제공해 메시지 패싱이 빠르게 over-smoothing 으로 수렴할 가능성 시사. QCondGAT 재학습 시 num_layers 효과 분리 가능.

**ID**: `abl_build_05_no_t2t` — anchor `s03_a07_01_enriched_gat` (E1, F1 0.7327). **주의**: anchor checkpoint 는 T2T 포함 그래프로 학습됨 → distribution shift 가능, recall 하락 시 GAT 재학습 필요.

#### B-III.b — Full-hetero Schema Graph Diameter precompute

**구현**: `HeteroGraphBuilder._compute_schema_diameter(table_to_id, col_to_id, fk_to_id, pcst_edges)` 추가. PCST flat indexing 의 모든 edge 를 무방향 sparse matrix 로 구성 → `scipy.sparse.csgraph.shortest_path(directed=False, unweighted=True)` 로 all-pairs 거리 계산 → per-node eccentricity (최대 finite 거리) 계산 → `schema_diameter = max(eccentricity)`. Disconnected component 자연스럽게 처리 (component 별 max 의 max).

**메타키 (모든 빌더 공통)**:
| 키 | 타입 | 설명 |
|----|------|------|
| `schema_diameter` | int | 전체 hetero graph 무방향 D_max |
| `schema_eccentricity` | `Dict[flat_idx, int]` | 노드별 max finite shortest-path |

**Table-only FK subgraph diameter** 는 의미가 다르므로 (join-path receptive field vs GAT depth) 별도 sub-task 로 분리, 본 라운드 미포함.

**스모크 결과** (`scripts/smoke_test_b3b_diameter.py`): BIRD-Dev 11 DB D_max 프로파일링 완료.

| db_id | T | C | FK | D_max | ecc_med | ecc_max |
|-------|---|----|----|------|---------|---------|
| debit_card_specializing | 5 | 21 | 1 | **3** | 2 | 3 |
| california_schools | 3 | 89 | 2 | **4** | 3 | 4 |
| card_games | 6 | 115 | 4 | **4** | 3 | 4 |
| thrombosis_prediction | 3 | 64 | 2 | **4** | 4 | 4 |
| financial | 8 | 55 | 8 | **5** | 5 | 5 |
| toxicology | 4 | 11 | 5 | **5** | 4 | 5 |
| codebase_community | 8 | 71 | 13 | **5** | 4 | 5 |
| formula_1 | 13 | 94 | 19 | **6** | 5 | 6 |
| european_football_2 | 7 | 199 | 31 | **6** | 4 | 6 |
| student_club | 8 | 48 | 8 | **6** | 5 | 6 |
| superhero | 10 | 31 | 11 | **6** | 5 | 6 |

**D_max 분포**: min=3 / median=5 / mean=4.91 / max=6.

**핵심 시사점**:
- 현재 GAT default `num_layers=3` 은 11 DB 중 **debit_card_specializing 단 1개** 에만 충분 (D_max=3). 나머지 10 개 DB 에서 distant gold 노드의 NLQ 신호 흐름이 차단될 수 있음. QCondGAT 의 over-smoothing 진단(s06)과 별개로, **shallow-bias** 도 동시 작용 가능성.
- 분포가 [3, 6] 구간으로 좁음 → advisor proposal C 의 `num_layers ∈ {1, 2, 3, D_max, D_max+1}` 스윕은 사실상 [1, 7] 범위로 한정됨. DB 별 자동 튜닝 효과를 측정하기 위해 fixed-vs-adaptive 비교 셀 필요.
- ecc_med ≈ ecc_max 인 DB (financial, thrombosis_prediction, toxicology) 은 그래프가 길쭉(linear-like) → adaptive depth 효과 제한적. 반면 superhero (T=10, ecc_med=5, ecc_max=6) 처럼 wide 구조에서는 eccentricity 기반 per-node depth control 가능성.

**활용 (하류)**: Selector QCondGAT 의 `num_layers ∈ {1, 2, 3, D_max, D_max+1}` 자동 스윕 (advisor proposal C). 큰 D_max DB 에서는 over-smoothing 재등장 위험 → Selector 세션이 별도 ablation.

**ID**: `abl_build_06_diameter_meta` — anchor `s03_a07_01_enriched_gat` (E1, F1 0.7327). 메타키 추가만, 파이프라인은 무시 → behavioral identical to E1 (regression marker).

#### 공통

- 인터페이스 계약 유지: 두 항목 모두 metadata 키 추가만, 기존 키 변경 없음.
- 1 패스 원칙: `_compute_fk_reachability` 와 `_compute_schema_diameter` 가 build() 종료 직전 연속 호출. 비용 microsec 단위 (BIRD T<20).
- LineGraphBuilder / RFMCompatibleBuilder 는 super().build() 통해 두 메타키 모두 자동 forward.

#### 산출물

- 코드: [src/modules/builders/graph_builder.py](src/modules/builders/graph_builder.py) (`add_t2t_edges` 인자 + `_compute_schema_diameter`), [src/data/bird_dataset.py](src/data/bird_dataset.py) (cache suffix `_no_t2t`)
- 스모크: [scripts/smoke_test_b2b_no_t2t.py](scripts/smoke_test_b2b_no_t2t.py), [scripts/smoke_test_b3b_diameter.py](scripts/smoke_test_b3b_diameter.py)
- Configs: [configs/experiments/abl/build/no_t2t/abl_build_05_no_t2t.yaml](configs/experiments/abl/build/no_t2t/abl_build_05_no_t2t.yaml), [configs/experiments/abl/build/diameter_meta/abl_build_06_diameter_meta.yaml](configs/experiments/abl/build/diameter_meta/abl_build_06_diameter_meta.yaml)
- Selector 편의 캐시 writer: [scripts/build_diameter_cache.py](scripts/build_diameter_cache.py) → `data/processed/<split>_diameter.pt` (NAS symlink)
- PLAN: [src/modules/builders/EXPERIMENT_PLAN_builders.md](src/modules/builders/EXPERIMENT_PLAN_builders.md) §B-II.b / §B-III.b
- Advisor 근거: [planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md](planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md) §4 의견 2

#### Naming alignment (2026-04-21 사용자 결정)
- B-II.b flag: `include_table_to_table` → **`add_t2t_edges`** (proposal `abl_bld_t2t_edge` §4.1 명칭 정합). default `True` 유지, 시맨틱 동일.
- B-III.b: 기존 metadata 주입 + **별도 selector 편의 캐시** `data/processed/<split>_diameter.pt` (`{db_id: D_max}` dict). Enriched/triplet cache 와 동일 NAS-symlink 패턴.

#### Cache writer scope (2026-04-21 사용자 결정)
- **전체 11 DB build 는 GAT 학습 trigger 시점에 수행** (NAS 경합으로 첫 실행 시 7 min+ Dl wait, vLLM serve 와 충돌). 
- 본 라운드 verification 은 **1-DB 미니멀 smoke** ([scripts/smoke_test_diameter_cache.py](scripts/smoke_test_diameter_cache.py)) 로 writer 로직만 검증 (california_schools D_max=4, payload 직렬화/symlink 왕복 OK).
- `scripts/build_diameter_cache.py` 는 idempotent (캐시 존재 시 skip, `--force` 로 재빌드). GAT 학습 스크립트 (혹은 launcher) 가 학습 시작 시 한 번 호출하도록 분리 — root 세션에서 후속 정리.

---

### 6-21. Selector S-V — Neurosymbolic Layer 1 (FK-reachability prior) — 2026-04-20

Builder B-III (`metadata['fk_reachability']`) 직후 착수한 Selector 축 첫 구현. EnsembleSelector 에 **최소 침습**(single hook override) 으로 symbolic prior 를 주입.

#### 설계

```
boosted_scores = ensemble_scores + λ · reach_mask
ensemble_scores = α · raw_cosine + (1−α) · gat_score       # α=0.85 (기존)
reach_mask[v]  = 1.0 if owning_table(v) ∈ reachable(anchors)
```

- **Anchor 식별** (deterministic, token-level): question 을 `re.findall(r"[a-zA-Z0-9]+", lower())` 로 토큰화 (min_len=3). Table 이름과 column 이름을 snake_case/camelCase 로 분해 후 word intersection. 테이블 이름 hit 또는 해당 테이블의 컬럼 이름 hit 시 table_id 를 anchor 에 추가.
- **Reach computation**: `fk_reachability[anchor_idx].any(axis=0)` — 여러 anchor 의 reachable set 합집합. Undirected 이므로 FK 방향과 무관.
- **Mask 확장**: 테이블 → owning table 반영, 컬럼 → owning table 상속, FK 노드 → src/dst table 중 하나라도 reachable 이면 1.
- **Graceful fallback** 3단계: `fk_reachability` 없음 → ensemble; anchor 없음 → ensemble; reach_mask sum=0 → ensemble.

#### 구현 파일

- `src/modules/selectors/neurosymbolic_l1_selector.py` (신규, 195 lines)
- `src/modules/selectors/ensemble_selector.py`: `_post_ensemble_hook` method 추가 (default no-op) → `select()` 에서 top-k 직전 호출
- `src/modules/selectors/__init__.py`: `NeurosymbolicL1Selector` export
- `configs/experiments/abl/sel/ns_l1/abl_sel_ns_l1_01.yaml`: pilot config (λ=0.1)

#### 스모크 검증 (real GAT weights)

1. **B-III 메타데이터 shape**: california_schools (T=3, C=89) → `fk_reachability.shape=(3,3) bool`, 전부 True (connected) ✅
2. **Anchor 식별** 3개 질문:
   - "math score" → satscores + schools (column word hit)
   - "FRPM" → frpm
   - "nonsense xyz" → ∅ (graceful fallback)
3. **Disconnected FK graph** (debit_card_specializing, 4 components):
   - "customer purchases" → anchors [0, 3, 4], mask 19/27 nodes
   - "gas prices" → anchors [1, 3], mask 15/27
   - "how many transactions" → anchors [3], mask 10/27
   - 각 결과가 해당 component 소속 노드만 정확히 커버하는 것 확인
4. **End-to-end (Real ensemble, random top-k)**: max|Δscore|=0.1000 정확히 일치 (λ=0.1), boosted count 24/27 nodes

#### 의의

- **첫 Selector 축 진입**: 루트 PLAN 의 5개 축 중 **S-V 가 가장 낮은 엔지니어링 비용** (metadata 소비 + 1 hook). Builder B-III 완료가 선행 조건이었고 2026-04-20 에 해결됨.
- **Zero regression 경로**: fk_reachability 없는 구설정에서도 parent class (EnsembleSelector) 동작을 그대로 유지하므로 기존 실험 결과와 noise 수준 일치해야 정상.
- **가설**: anchor table 에서 FK-reachable 한 bridge table 의 recall 이 개선된다. 특히 3-table JOIN 이 필요한 쿼리에서 **AdaptivePCST 의 P80 threshold 하위로 떨어지던 FK 노드**를 λ bonus 로 prize 보존권에 끌어올림.

#### 보류 작업

- **End-to-end F1 측정**: `abl_sel_ns_l1_01` — vLLM 서버 (Qwen3-Coder-30B) 가동 후 실행 예정. Anchor `s03_a02_03_xiyan_filter` (F1 baseline) 과 직접 A/B.
- **λ sweep**: 0.05 / 0.1 / 0.2 — pilot 결과 기반으로 선정. 각각 `abl_sel_ns_l1_{02,03,04}` 예약.
- **Builder switch**: 현재 best_gat_model.pt (basic builder 로 학습) 사용. Enriched 쪽은 `best_gat_enriched.pt` + `EnrichedHeteroGraphBuilder` 조합으로 별도 실험 (`abl_sel_ns_l1_enriched_*`) 예정.

**산출물**:
- 셀렉터 코드: [src/modules/selectors/neurosymbolic_l1_selector.py](src/modules/selectors/neurosymbolic_l1_selector.py), hook in [src/modules/selectors/ensemble_selector.py](src/modules/selectors/ensemble_selector.py)
- 실험 config: [configs/experiments/abl/sel/ns_l1/abl_sel_ns_l1_01.yaml](configs/experiments/abl/sel/ns_l1/abl_sel_ns_l1_01.yaml)

---

## 7. 전체 실험 순위 (Recall 기준 Top 10)

NoFilter(Extractor-only) 실험과 Full-pipeline 실험이 혼재됨에 주의. **†** 표시는 NoFilter (extractor 단독 recall).

| Rank | Experiment | Recall | Precision | F1 | Key Components |
|------|-----------|--------|-----------|------|----------------|
| 1 | a09_03_basic_no_filter_anchor † | **0.9679** | 0.1276 | 0.2255 | Ensemble + BasicPCST + **NoFilter** |
| 2 | s03_a10_01_fk_steiner_full_col † | **0.9492** | 0.1567 | 0.2690 | Ensemble + **FKBackboneSteiner(θ_r=0.0)** + NoFilter |
| 3 | s03_a10_04_fk_steiner_r01 † | 0.9481 | 0.1582 | 0.2711 | Ensemble + FKBackboneSteiner(θ_r=0.1) + NoFilter |
| 4 | s03_a10_05_fk_steiner_r02 † | 0.9418 | 0.1644 | 0.2800 | Ensemble + FKBackboneSteiner(θ_r=0.2) + NoFilter |
| 5 | s03_a10_02_fk_steiner_mid_col † | 0.9293 | 0.1812 | 0.3033 | Ensemble + FKBackboneSteiner(θ_r=0.3) + NoFilter |
| 6 | s03_a10_06_fk_steiner_r04 † | 0.9014 | 0.2125 | 0.3439 | Ensemble + FKBackboneSteiner(θ_r=0.4) + NoFilter |
| 7 | s03_a10_03_fk_steiner_high_col † | **0.8565** | 0.2627 | 0.4021 | Ensemble + **FKBackboneSteiner(θ_r=0.5)** + NoFilter |
| 8 | abl_ens_basic_xiyan | 0.8149 | 0.7597 | **0.7863** | Ensemble + BasicPCST + XiYan |
| 9 | abl_cos_basic_xiyan | 0.7987 | 0.7694 | 0.7838 | Cosine + BasicPCST + XiYan |
| 10 | s03_a10_07_fk_steiner_r06 † | 0.7789 | 0.3341 | 0.4677 | Ensemble + FKBackboneSteiner(θ_r=0.6) + NoFilter |

## 8. 전체 실험 순위 (F1 기준 Top 10)

| Rank | Experiment | Recall | Precision | F1 | Key Components |
|------|-----------|--------|-----------|------|----------------|
| 1 | abl_ens_basic_xiyan | 0.8149 | 0.7597 | **0.7863** | Ensemble + BasicPCST + XiYan |
| 2 | abl_cos_basic_xiyan | 0.7987 | 0.7694 | 0.7838 | Cosine + BasicPCST + XiYan |
| 3 | edge_prize | 0.6823 | 0.8139 | 0.7424 | TripletBuilder + EdgePrizePCST + XiYan |
| 4 | enriched_gat | 0.6658 | 0.8147 | 0.7327 | EnrichedBuilder + Ensemble + Adaptive + XiYan |
| 5 | **a05_02_reflection_1iter** | 0.7320 | 0.6833 | **0.7068** | Ensemble + AdaptivePCST + **ReflectionFilter(1iter)** |
| 6 | idea24_product_component_xiyan | 0.6304 | 0.8028 | 0.7063 | Ensemble + ProductCost+Component + XiYan |
| 7 | qcond_idea24_xiyan | 0.6236 | 0.8056 | 0.7032 | QueryCond(α=0.85) + Idea2+4 + XiYan |
| 8 | b4_xiyan_filter | 0.6244 | 0.7930 | 0.6987 | Ensemble + AdaptivePCST + XiYan |
| 9 | supernode_idea24_a085_xiyan | 0.6154 | 0.8005 | 0.6958 | SuperNode(α=0.85) + Idea2+4 + XiYan |
| 10 | abl_a03_17_supernode_binary_fixed_xiyan | 0.6761 | 0.7128 | 0.6940 | SuperNode-Direct(binary, τ=0.5) + BasicPCST + XiYan |
| 11 | **a05_04_verifier** | 0.7093 | 0.6676 | 0.6878 | SuperNode-Direct + BasicPCST + **VerifierFilter** |

---

## 9. 핵심 발견 요약

### 파이프라인 모듈별 기여도 (정량적)

1. **XiYan Filter**: 가장 큰 단일 기여. Precision +0.40~0.45 향상 (0.35→0.79). 없으면 어떤 조합도 P<0.40.
2. **Adaptive PCST**: Filter 없이는 P +0.22 기여. Filter와 함께 쓸 때는 R -0.17 손실 발생 (과도한 pruning).
3. **Enriched Node Features**: P 0.81로 최고 precision. PLM 임베딩 품질 향상이 모든 downstream 모듈에 전파.
4. **Product Cost PCST (Idea 2)**: Fixed cost 대비 R +0.01~0.04. Prize-cost scale 일치 원리 적용.
5. **Component Aware (Idea 4)**: R +0.03~0.04 추가. 이론적 기여 명확 (component별 독립 threshold).
6. **Query-Conditioned GAT**: α=0.85에서 P +0.01. α=0.0에서는 cosine 없이도 P 0.71 달성.
7. **GAT Ensemble (α=0.85)**: Cosine 대비 P/R +0.01~0.02. 기여가 미미한 이유는 GAT 자체 판별력 한계.
8. **ReflectionFilter (a05_02)**: XiYan 대비 R +0.11 (0.6244→0.7320), P −0.11 (0.7930→0.6833), F1 +0.008. Propose→critique→revise 루프가 XiYan의 recall 천장을 돌파한 최초 사례. Critique가 원래 subgraph 밖 노드 재도입을 허용하는 구조적 차별.
9. **AdaptiveMultiAgentFilter (a05_01)**: R=0.3770으로 매우 낮음 — agent consensus 과보수적, JSON parsing 실패 다수. 추가 튜닝 필요.
10. **VerifierFilter (a05_04)**: anchor(a03_17) 대비 R +0.03 (0.6761→0.7093), P −0.05 (0.7128→0.6676), F1 −0.006. XiYan 초기 필터 + NL unit test → missing 복원. Recall 회복은 성공하나 Reflection 대비 열등 (F1 0.6878 vs 0.7068) — generate-then-check 분리 구조가 통합 critique-revise보다 약함. ReAct-style 단일 agent 통합 reasoning의 우월성 시사.

### 구조적 패턴

- **Basic PCST + XiYan > Adaptive PCST + XiYan**: F1 기준 0.7863 vs 0.6987. XiYan이 pruning을 더 잘하므로 PCST는 넓게 포함시키는 게 유리.
- **Precision과 Recall의 trade-off**: 모든 실험에서 일관되게 나타남. Filter가 precision을 올리면 recall이 내려감. **ReflectionFilter는 이 trade-off를 명시적으로 recall 방향으로 이동**시킨 첫 개입.
- **BO 결과**: `bo_score_driven` (P=0.7867, R=0.5910) > `bo_fixed_cost` (P=0.7468, R=0.4793). Score-driven cost weights가 dev F1 기준으로 fixed cost보다 우월.
- **Edge Prize PCST**: F1=0.7424로 XiYan 조합 중 Top-3. Triplet edge embedding → edge prize가 connectivity-aware pruning에 유효.
- **GAT 기여가 제한적인 이유**: (1) α=0.85에서 15%만 반영 (2) query-agnostic attention (3) FK 노드 label 부재

### 다음 단계 제안

1. **a05_03 Reflection 3iter** (진행 중, 2026-04-15 기준 ~34%): 1iter 대비 추가 recall 회복 또는 수렴 여부 확인.
2. **a05_04 VerifierFilter** ✅ 완료 — R=0.7093 / P=0.6676 / F1=0.6878. Reflection(a05_02) 대비 F1 열등 (-1.9%p) — generate-then-check 분리 구조 한계 확인.
3. **a05_05/06 Tiered Bidirectional Agent (F3)**: Tier-1(PCST) vs Tier-2(selector-only) 구분 + graph-native tools.
4. **a05_07 Uncertainty-gated adaptive depth (F4)**: GAT confidence 기반 agentic compute 조절.
5. **a05_09/10 Extraction-retry (F5)**: Unanswerable verdict → Extractor 완화 재호출 loop.
6. **Backbone 민감도 (gpt-4o-mini)** ✅ 4개 실험 완료 — `a05_13` XiYan F1=0.6616 (−3.2%p), `a05_14` AdaptiveMultiAgent F1=0.5230 (Qwen 대비 +5.2%p이나 anchor −17.1%p), `a05_15` Reflection 1iter F1=0.6722 (−3.5%p), `a05_17` Verifier F1=0.6706 (−1.7%p). **전반적으로 gpt-4o-mini가 Qwen3-Coder-30B 대비 F1 1.7~3.5%p 열세** (AdaptiveMultiAgent만 parsing 안정성 이득). Reflection이 backbone 민감도 최고 (critique 질이 backbone 능력에 비례), Verifier가 최저 (unit test recall path 의존). 총 비용 ~$3.76 / 누적 런타임 ~9h. 결론: **prune-only / agentic 전반에서 Qwen 유지가 유리**; a05_11/12 (F3/F4 + gpt-4o-mini)는 우선순위 하향.
7. **Enriched + Query-Conditioned + Reflection 결합**: 각각의 최고점 결합 시 시너지 기대.
8. **FK 노드 supervised training (Idea 1)**: GAT의 bridge table 인식 능력 강화.
9. **Direct variant 결론**: BCE only Direct가 Projector(BCE+InfoNCE) 대비 열등 — DualTowerProjector + InfoNCE 유지가 유리.

---

## 10. 모듈 카탈로그 (Selector / Extractor / Filter 하이퍼파라미터)

### 10-1. Graph Builder

| Builder | 핵심 파라미터 | 비고 |
|---------|--------------|------|
| `HeteroGraphBuilder` (default) | `include_views=false`, `run_leiden_clustering=true` | 기본 table-column-FK 그래프 |
| `EnrichedHeteroGraphBuilder` | 동일 + column description/value_description/NL name을 node text에 병합 | P 0.81 |
| `TripletGraphBuilder` | + `triplet_path="data/processed/triplet_relations.json"` | Edge prize용 |

### 10-2. Seed Selector

| Selector | 핵심 파라미터 | 용도 |
|----------|--------------|------|
| `VectorOnlySelector` | `top_k=20` | Cosine similarity only |
| `EnsembleSelector` | `weight_path`, `alpha`, `top_k=20` | `alpha`=cosine 비중 (0.85/0.70/0.0), `1-alpha`=GAT 비중. Top-k 상위 선택 |
| `GATClassifierSelector` | `weight_path`, `top_k=20` | GAT score 단독 (legacy) |
| `DirectGATSelector` | `weight_path`, `query_conditioned`, `query_supernode`, `threshold=0.5`, `apply_threshold=false`, `in_channels=384`, `hidden/out/classifier_hidden=256` | Projector 없이 BCE만. `apply_threshold=true` 시 sigmoid ≥ threshold만 반환 |

**Weight paths**:
- `best_gat_model.pt`: GAT v4 (BCE+InfoNCE, T4)
- `best_gat_enriched.pt`: Enriched features (T5)
- `best_gat_query_conditioned.pt` / `best_gat_query_supernode.pt`: Projector 기반 query-cond (T6, T7)
- `best_gat_query_conditioned_direct.pt` / `best_gat_query_supernode_direct.pt`: DirectClassifierHead (T8, T9)

### 10-3. Connectivity Extractor (PCST)

공통 cost 기본값: `base_cost=0.05`, `belongs_to_cost=0.01`, `fk_cost=0.05`, `macro_cost=0.5`

| Extractor | 핵심 파라미터 (공통 외) | 설명 |
|-----------|----------------------|------|
| `None` | - | Pass-through (seed_nodes 그대로) |
| `TopK` | `top_k=15` | Score 상위 k개 |
| `PCSTExtractor` (Basic) | `node_threshold=0.1` | Fixed cost PCST |
| `AdaptivePCSTExtractor` | `percentile=80.0`, `min/max_prize_nodes=3/25`, `node_threshold=0.0` | Score P80 threshold, prize 개수 clamp |
| `DynamicPCSTExtractor` | 동일 + hub discount | Hub 노드 cost 감소 |
| `EdgePrizePCSTExtractor` | Adaptive + `topk_e=5`, `edge_cost=0.05` | Triplet edge embedding 기반 edge prize |
| `ProductCostPCSTExtractor` | `bt_weight=0.1`, `fk_weight=0.2`, `macro_weight=0.5`, `min_cost=0.0001`, `percentile=80` | Edge cost를 양 노드 score의 곱으로 정의 (Idea 2) |
| `ComponentAwareProductCostPCSTExtractor` | ProductCost + Component 분해 | Idea 2+4 결합 |
| `ScoreDrivenPCSTExtractor` | `belongs_to/fk/macro_weight`, `epsilon` | BO로 튜닝한 weights |
| `SteinerBackbonePCSTExtractor` | Adaptive + `backbone_bonus=0.5` | Seed 간 Steiner tree 2-근사 → PCST expansion (Idea 3) |
| `MSTExtractor` | - | Metric closure 기반 Steiner 2-근사 (단독 사용 드묾) |

### 10-4. Filter

| Filter | 핵심 파라미터 | 용도 |
|--------|--------------|------|
| `None` | - | Pass-through |
| `SingleAgentFilter` | LLM 모델명 | 1회 LLM pruning |
| `AdaptiveMultiAgentFilter` | `model_name`, `uncertainty_threshold=0.6` | Semantic+Structural+Skeptic agent voting. a05_01에서 R=0.3770 (과보수적) |
| `XiYanFilter` | `model_name="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"`, `max_iteration=1`, `temperature=0.0` | XiYan-SQL pruning |
| `ReflectionFilter` | `model_name`, `max_iteration=1~3`, `temperature=0.0` | Propose→Critique→Revise. 원 subgraph 외 노드 재도입 허용. a05_02 (1iter): R=0.7320 F1=0.7068 |
| `VerifierFilter` | `model_name`, `max_iteration=1`, `temperature=0.0` | XiYan-style 초기 필터 + NL unit test 생성/검증 + missing_nodes 복원. a05_04: R=0.7093 F1=0.6878 |
| `BidirectionalAgentFilter` (예정) | tier1_subgraph, tier2_pool, gat_scores, tools | Tier-aware prune + restore with graph-native tools. a05_05/06 |
| `AdaptiveDepthFilter` (예정) | uncertainty_threshold | GAT confidence 기반 depth 선택 (단일/Reflection/Bidirectional). a05_07 |

### 10-5. Post-processing

- `auto_join_keys: true`: 2개 이상 table이 선택되면 FK 컬럼 자동 추가

---

## 11. 실험별 구성 매핑

모든 실험은 `LocalPLMEncoder` (MiniLM-L6-v2), `auto_join_keys=true` (대부분), `XiYanFilter`의 경우 Qwen3-Coder-30B FP8 / max_iter=1 / temp=0.0 사용.

### Phase A (초기 실험)

| # | Experiment | Selector | Extractor | Filter |
|---|-----------|----------|-----------|--------|
| A1 | gat_classifier | GATClassifier(top_k=20) | None | None |
| A2 | gat_classifier_multi_agent | GATClassifier | None | AdaptiveMultiAgent |
| A3 | gat_pcst_multi_agent | GATClassifier | GATAwarePCST | AdaptiveMultiAgent |
| A4 | gat_multi_agent | GATProjection | None | AdaptiveMultiAgent |
| A5 | base_pcst | VectorOnly | PCST(fixed, threshold=0.1) | None |
| A6 | dynamic_pcst | VectorOnly | DynamicPCST | None |
| A7 | uncertainty_pcst | VectorOnly | UncertaintyPCST | None |
| A8 | dynamic_uncertainty_pcst | VectorOnly | Dynamic+Uncertainty | None |

### Phase B (단계별 발전)

| # | Experiment | Selector | Extractor | Filter |
|---|-----------|----------|-----------|--------|
| B0 | b0_raw_pcst_baseline | VectorOnly(top_k=20) | PCST(threshold=0.1) | None |
| B1 | b1_adaptive_pcst | VectorOnly | AdaptivePCST | None |
| B2 | b2_ensemble | Ensemble(α=0.85) | PCST(threshold=0.1) | None |
| B-c | b_combined | Ensemble(α=0.85) | AdaptivePCST | None |
| B4a | b4_single_filter | Ensemble(α=0.85) | AdaptivePCST | SingleAgent |
| B4b | b4_xiyan_filter | Ensemble(α=0.85) | AdaptivePCST | XiYan |

### Phase C (2×2×2 Ablation)

| # | Experiment | Selector | Extractor | Filter |
|---|-----------|----------|-----------|--------|
| 1 | abl_cos_basic (none filter) | VectorOnly | PCST(basic, threshold=0.1) | None |
| 2 | abl_cos_adaptive | VectorOnly | AdaptivePCST | None |
| 3 | abl_ens_basic | Ensemble(α=0.85) | PCST(basic) | None |
| 4 | abl_ens_adaptive | Ensemble(α=0.85) | AdaptivePCST | None |
| 5 | **abl_cos_basic_xiyan** | VectorOnly | PCST(basic) | XiYan |
| 6 | **abl_ens_basic_xiyan** | Ensemble(α=0.85) | PCST(basic) | XiYan |
| 7 | abl_cos_adaptive_xiyan | VectorOnly | AdaptivePCST | XiYan |
| 8 | abl_ens_adaptive_xiyan (=b4_xiyan_filter) | Ensemble(α=0.85) | AdaptivePCST | XiYan |

### 아이디어 실험 (6-1 ~ 6-14)

| # | Experiment | Builder | Selector | Extractor | Filter |
|---|-----------|---------|----------|-----------|--------|
| I1a-c | alpha_sweep (α=0.85/0.75/0.70) | Default | Ensemble(α=변수) | AdaptivePCST | None |
| I2a | idea2_product_cost | Default | Ensemble(α=0.85) | ProductCost(bt=0.1, fk=0.2, macro=0.5) | None |
| I2b | idea2_product_cost_xiyan | Default | Ensemble(α=0.85) | ProductCost | XiYan |
| I3a | idea3_steiner_backbone | Default | Ensemble(α=0.85) | SteinerBackbone(bonus=0.5) | None |
| I3b | idea3_steiner_backbone_xiyan | Default | Ensemble(α=0.85) | SteinerBackbone(bonus=0.5) | XiYan |
| I4 | idea4_component_aware | Default | Ensemble(α=0.85) | ComponentAwareAdaptivePCST | None |
| I24a | idea24_product_component | Default | Ensemble(α=0.85) | ComponentAwareProductCost(bt=0.1, fk=0.2, macro=0.5) | None |
| I24b | idea24_product_component_xiyan | Default | Ensemble(α=0.85) | ComponentAwareProductCost | XiYan |
| BO1 | bo_fixed_cost | Default | Ensemble(α=0.85) | AdaptivePCST(bt=0.195, fk=0.346, macro=0.044) | XiYan |
| BO2 | bo_score_driven | Default | Ensemble(α=0.85) | ScoreDrivenPCST(bt=1.955, fk=2.779, macro=3.439, ε=0.009) | XiYan |
| E1 | enriched_gat | **Enriched** | Ensemble(α=0.85, `best_gat_enriched.pt`) | AdaptivePCST | XiYan |
| E2 | edge_prize | **Triplet** | Ensemble(α=0.85, `best_gat_enriched.pt`) | EdgePrizePCST(topk_e=5, edge_cost=0.05) | XiYan |
| Q1 | qcond_idea24_xiyan | Default | Ensemble(α=0.85, `best_gat_query_conditioned.pt`) | ComponentAwareProductCost | XiYan |
| Q2 | supernode_idea24_xiyan | Default | Ensemble(α=0.70, `best_gat_query_supernode.pt`) | ComponentAwareProductCost | XiYan |
| Q3 | supernode_idea24_a085_xiyan | Default | Ensemble(α=0.85, `best_gat_query_supernode.pt`) | ComponentAwareProductCost | XiYan |
| Q4 | qcond_idea24_a0_xiyan | Default | Ensemble(α=0.0, `best_gat_query_conditioned.pt`) | ComponentAwareProductCost | XiYan |
| Q5 | supernode_idea24_a0_xiyan | Default | Ensemble(α=0.0, `best_gat_query_supernode.pt`) | ComponentAwareProductCost | XiYan |
| Q6 | qcond_direct_idea24_xiyan | Default | Direct(qcond, `_direct.pt`) | ComponentAwareProductCost | XiYan |
| Q7 | supernode_direct_idea24_xiyan | Default | Direct(supernode, `_direct.pt`) | ComponentAwareProductCost | XiYan |

### Direct Variant Per-Step Ablation (6-11, 6-12, 6-13, 6-14)

모든 Direct 실험은 `DirectGATSelector` 사용. `apply_threshold=true` (Binary) / `false` (전체 반환).

| Section | Experiment | Selector (Direct mode) | apply_threshold | threshold | Extractor | Filter |
|---------|-----------|----------------------|-----------------|-----------|-----------|--------|
| 6-11 | ablation_{qcond,supernode}_direct_selector_only | Concat/SuperNode | false | - | None | None |
| 6-11 | ablation_{qcond,supernode}_direct_selector_extractor | Concat/SuperNode | false | - | AdaptivePCST | None |
| 6-11 | Full (= Q6/Q7) | Concat/SuperNode | false | - | ComponentAwareProductCost | XiYan |
| 6-12 | ablation_{qcond,supernode}_direct_binary_selector_only | Concat/SuperNode | true | 0.5 | None | None |
| 6-12 | ablation_{qcond,supernode}_direct_binary_selector_extractor | Concat/SuperNode | true | 0.5 | AdaptivePCST | None |
| 6-13 | threshold sweep (offline, Steiner) | Concat/SuperNode | true | 0.05~0.50 | SteinerBackbone (offline) | None |
| 6-14 | ablation_supernode_binary_t{005,010,015,020}_steiner_xiyan | SuperNode | true | 0.05~0.20 | SteinerBackbone(bonus=0.5) | XiYan |

**공통 Extractor 하이퍼파라미터** (specified subset):
- AdaptivePCST / SteinerBackbone: `base_cost=0.05, belongs_to_cost=0.01, fk_cost=0.05, macro_cost=0.5, percentile=80.0, min/max_prize_nodes=3/25, node_threshold=0.0`
- SteinerBackbone 추가: `backbone_bonus=0.5`
- ComponentAwareProductCost: `bt_weight=0.1, fk_weight=0.2, macro_weight=0.5, min_cost=0.0001, percentile=80.0, min/max_prize_nodes=3/25`

**Direct Selector 공통**: `in_channels=384, hidden_channels=out_channels=classifier_hidden=256, encoder_type="plm"`

### Phase D — Filter Module Agentic Refinement (a05 series, 2026-04-14~)

모든 a05 실험은 `configs/experiments/abl/a05_filter_agentic/` 하위. Backbone: Qwen3-Coder-30B-A3B-Instruct-FP8 (vLLM, GPU 2+3). Extractor는 best Direct anchor(`abl_a03_17_supernode_binary_fixed_xiyan`)의 `PCSTExtractor(base=0.05, bt=0.01, fk=0.05, macro=0.5, threshold=0.0)` 고정, Filter만 교체.

| # | Experiment ID | Filter | Backbone | 상태 | Recall | Precision | F1 |
|---|---------------|--------|----------|------|--------|-----------|-----|
| a05_01 | a05_01_adaptive_multi_agent | `AdaptiveMultiAgentFilter(uncertainty=0.6)` | Qwen | ✅ | 0.3770 | 0.6276 | 0.4713 |
| a05_02 | a05_02_reflection_1iter | `ReflectionFilter(max_iter=1)` | Qwen | ✅ | **0.7320** | 0.6833 | **0.7068** |
| a05_03 | a05_03_reflection_3iter | `ReflectionFilter(max_iter=3)` | Qwen | 🏃 (~34%) | - | - | - |
| a05_04 | a05_04_verifier | `VerifierFilter` (XiYan + Unit Tester) | Qwen | ✅ | 0.7093 | 0.6676 | 0.6878 |
| a05_05 | a05_05_bidirectional_notool | `BidirectionalAgentFilter` (tier1+tier2, no tools) | Qwen | ⏸ | - | - | - |
| a05_06 | a05_06_bidirectional_fulltool | `BidirectionalAgentFilter` + graph tools ★ | Qwen | ⏸ | - | - | - |
| a05_07 | a05_07_adaptive_depth | `AdaptiveDepthFilter` (uncertainty gating) | Qwen | ⏸ | - | - | - |
| a05_08 | a05_08_verifier_bidirectional | F3 + F2 stacked | Qwen | ⏸ | - | - | - |
| a05_09 | a05_09_extraction_retry | F5 Extractor retry (K=2) + F3 | Qwen | ⏸ | - | - | - |
| a05_10 | a05_10_retry_gated | F5 + F4 gating | Qwen | ⏸ | - | - | - |
| a05_11 | a05_11_bidirectional_gpt4omini | F3 full tools | **GPT-4o-mini** | ⏸ | - | - | - |
| a05_12 | a05_12_retry_gpt4omini | F5+F4 stack | **GPT-4o-mini** | ⏸ | - | - | - |

---

## 7. Phase E — GAT Bottleneck Fix (s06 series, 2026-04-16~)

**Motivation**: `outputs/analysis/gat_bottleneck{,_qcond}/` 의 3-step 병목 진단(2026-04-15~16)
결과를 처방으로 반영하는 ablation 사이클. 관찰된 4가지 병리:

1. **Layer-1 catastrophic over-smoothing**: QCond L1 cosine 0.89 / SN 0.97 — 단 1홉 만에 critical(0.85) 초과
2. **Skip/Input-dominated learning**: SN `skip_dict` gradient 5.0 압도적 / QCond `lin_dict` gradient 2.0 압도적 → GAT layer 실질 기여 약함
3. **Attention uniformity**: 3 layer entropy 패턴 완전 동일 → layer별 고유 relational view 미학습
4. **BCE–Recall divergence**: QCond ep75 / SN ep79 이후 loss↓ / Recall plateau (~221 epoch 낭비)

### Phase E-1 (s06_a01): Forward-Additive Ablation on QCond

- Anchor: QCond Direct (T8, `best_gat_query_conditioned_direct.pt`, Val Recall@15=0.591)
- Filter 없이 Selector only 측정 (Val Recall@15 기준)
- Compute budget 제약으로 QCond 라인만 우선 검증. SN은 추후 대칭 검증 예정.

| # | ID | 처방 | 예상 Val R@15 | **실제 Val R@15** | B0 대비 Δ | 학습시간 | 상태 |
|---|----|------|---------------|-------------------|-----------|---------|------|
| s06-B0 | s06_a01_01_b0_baseline | 현행 QCond Direct (reference) | 0.591 | **0.5738** | — | 3h 48m | ✅ 2026-04-16 |
| s06-B1 | s06_a01_02_b1_pairnorm | + PairNorm | 0.605 | **0.5707** | −0.0031 | 4h 12m | ✅ 2026-04-16 |
| s06-B2 | s06_a01_03_b2_initial_residual | + Initial Residual (APPNP α=0.2) | 0.615 | **0.5986** | +0.0248 | 3h 57m | ✅ 2026-04-17 |
| s06-B3 | s06_a01_04_b3_listnet | BCE → ListNet | 0.625 | **0.5745** | +0.0007 | 5h 36m | ✅ 2026-04-17 |
| s06-B4 | s06_a01_05_b4_anti_collapse | + Schema-Aware Anti-Collapse (λ=0.3) | 0.640 | **0.5894** | +0.0156 | 9h 23m | ✅ 2026-04-17 |
| s06-B5 | s06_a01_06_b5_dual_stream | Full Dual-Stream + JK concat, 2 layers | 0.670 | **0.6073** | **+0.0335** | ~29h (rerun) | ✅ 2026-04-19 |

**Hypothesis vs 실측**: 각 처방이 독립적으로 Recall +0.01~0.015 씩 기여하고, Dual-Stream 구조 변경이 +0.03 추가하리라 예상 → 최종 B5 ≈ 0.67 목표. **실측**: B5 = 0.6073 (예상보다 -0.063 낮음). 개별 처방 기여도는 **비단조** (B1 오히려 후퇴, B3 무영향, B2/B4/B5만 유의미). 과잉 예측의 원인: 누적 효과 가정이 비현실적 (처방 간 상호작용, anti-collapse 가 listnet 과 동일 축 신호 공유).

**특이사항**:
- B5 첫 run 은 2026-04-17 19:59 에 1초만에 crash (fk_node 관련 버그) → 2026-04-17 21:32 rerun. rerun 이 정상. 체크포인트 best epoch 은 rerun 초반 (04-18 04:35 저장).
- B1 (PairNorm 단독) 은 **오히려 성능 후퇴** → PairNorm 이 단독으론 oversmoothing 제어는 되지만 discriminative capacity 를 제거해버림. B2 (+Initial Residual) 가 pair 로 붙어야 회복.
- B3 (ListNet) 변화 무영향 — joint-train 에서는 ListNet 의 list-wise signal 이 AntiCollapse/BCE 와 섞이며 희석되는 것으로 추정. 별도 post-hoc head retrain (§7-2 exp C) 에서는 ListNet 이 실제로 Dev AUC +0.024 기여함이 확인됨. 즉 "joint training 에서만 signal 이 뭉개짐".
- B5 에서 `jumping_knowledge=concat + 2 layers + dual_stream` 구조 변경이 가장 큰 단일 기여 (+0.0179 vs B4).

**논문 기여 매핑**:
- Observation → Empirical motivation (§IV-A 병목 진단)
- B1~B3 → 기존 원리 적용 (standard baselines for over-smoothing / ranking)
- B4 (SACR) → 본 연구 기여 1 "Schema-Aware Anti-Collapse Regularization"
- B5 (Dual-Stream) → 본 연구 기여 2 "Query-Schema Stream Disentanglement"
- Per-component Recall 차 → Ablation Table (§IV-B)

**구현 산출물** (2026-04-16):
- `src/models/gat_network_v2.py` — PairNorm, Initial Residual, JK, Dual-Stream 플래그
- `src/models/losses.py` — ListNet, Anti-Collapse regularizer
- `src/train_gat_s06.py` — s06 전용 학습 스크립트
- `configs/experiments/s06_gat_bottleneck_fix/a01_additive_ablation/*.yaml` — 6개 config

**실행 명령** (GPU 확보 후):
```bash
for cfg in configs/experiments/s06_gat_bottleneck_fix/a01_additive_ablation/*.yaml; do
  PYTHONPATH=src python src/train_gat_s06.py --config "$cfg"
done
```

### Phase E-2 (B5 verification & head retrain diagnostic, 2026-04-20)

**Motivation**: B5 Val R@15=0.6073 / Dev Recall 품질의 병목이 (a) GAT representation 자체인지 (b) joint-train 된 classifier head 인지 분리 진단. 동일 데이터셋에서 frozen L_out (마지막 GAT layer 출력) 을 cache 한 뒤 head 만 재학습.

**Verification 사전 관찰** (`src/analysis/b5_verification.py`):
- Dev 기준 `cosine(gold, gold) μ=0.2242`, `cosine(gold, non) μ=0.1060`, `cosine(non, non) μ=0.2165` → gold↔non 구분은 있으나 margin 얇음.
- Linear Probe (5-fold CV LogisticRegression on dev L_out) **AUC 0.9195 ± 0.002, holdout 0.9178** → representation 자체는 매우 잘 분리됨.
- 원본 B5 joint classifier dev AUC 0.7067 — linear probe 0.92 대비 **-0.20 의 큰 gap** → **joint-trained head 가 bottleneck** 가설.

**정정**: linear probe 0.92 는 *within-dev* CV (dev 데이터로 학습+평가 분리). 우리가 실제로 할 "train 으로 head 학습 → dev 평가" 와 개념 다름. 실제 bottleneck 은 **train→dev 분포 shift** 였음 (§7-2 결과).

### §7-2. B5 Post-Hoc Head Retrain 2×2 Ablation (Offline, 2026-04-20)

**Setup**: B5 frozen L_out cache (`outputs/analysis/s06_bottleneck/B5/retrain/lout_cache_{train,dev}.pt`, 학습 GAT state 그대로 freeze) 위에 head 만 50 epoch 재학습.
- Split: train query 10% 를 val, 나머지 90% 학습 (query-random split, seed=42).
- Grid: head ∈ {linear, mlp(256→256→128→1)} × loss ∈ {bce, listnet} × normalize ∈ {none, per-query zscore}.
- Script: `src/analysis/b5_head_retrain.py`, runner: `scripts/run_b5_head_retrain{,_CDE}.sh`.

**Matrix (head=mlp):**

| Exp | Loss | Norm | val R@15 | Dev AUC (val-ES) | Dev R@15 (val-ES) | best Dev AUC (oracle) | best Dev R@15 (oracle) |
|-----|------|------|----------|------------------|---------------------|----------------------|------------------------|
| A (linear) | bce | none | 0.9945 | 0.6483 | 0.5856 | — | — |
| B | bce | none | 0.9962 | 0.6648 | 0.6042 | — | — |
| **C** | **listnet** | none | 0.9962 | **0.6891** | 0.6184 | **0.7548 @ep3** | 0.6497 @ep3 |
| **D** | bce | **zscore** | 0.9953 | 0.6724 | **0.6228** | 0.7027 @ep1 | **0.6571 @ep2** |
| E | listnet | zscore | 0.9948 | 0.6687 | 0.6146 | 0.7292 @ep3 | 0.6514 @ep2 |

참조: 원본 B5 joint classifier Dev AUC=0.7067, Val R@15=0.6073.

**관찰**:
1. **Val R@15 전부 ≥ 0.99** → head 가 train L_out 에 overfit 쉬움. 남은 차이는 전부 train→dev 분포 shift 에서 발생.
2. **개선 강도 ListNet >> zscore >> combine(sub-additive)**: listnet alone Dev AUC +0.024 (vs B) / zscore alone +0.008 / 두 기법 combine 은 C 단독보다 -0.020 (상쇄).
3. **Metric 별 승자 상이**: Dev AUC 1등 C / Dev R@15 1등 D.
4. **Oracle dev-ES 로 보면 retrain 이 joint 를 앞섬**: C best_dev_auc 0.7548 (+0.048 vs 0.7067), D best_dev_r15 0.6571 (+0.050 vs 0.6073). 즉 joint-train 의 head 가 suboptimal.
5. **극단적 early peak**: C/D/E 모두 ep1~3 에서 best_dev. Train val AUC 0.99 에 2~3 epoch 만에 도달 → head 에게 L_out 은 거의 linear-separable.

**결론**: "head 가 bottleneck 이다" 는 oracle dev-ES 관점에서만 참. Realistic val-ES (train-internal val split) 로 고르면 retrain 이 joint 를 소폭 밑돌기도 함 → **진짜 병목은 train→dev domain shift**. 이를 진단하기 위해 §7-3 LDBO 진행.

### §7-3. B5 Head-Only LDBO Diagnostic (Offline, 2026-04-20)

**Motivation**: §7-2 는 val split 이 *query-random* 이라 같은 DB 의 다른 query 가 val 에 섞임 → val = "본 적 있는 DB 의 새 query", dev = "본 적 없는 DB". Val-ES 기준과 실제 dev 성능이 어긋나는 근본 원인. 이를 완화하려면 val 도 "본 적 없는 DB" scenario 여야 함 → **Leave-DB-Out (LDBO)**.

**Setup**: Train 의 69 unique DB 중 11 개 (≈ dev 크기, 16%) 를 seed=42 로 홀드아웃 → `proxy_dev`. 나머지 58 DB 로 head 학습. GAT 는 재학습 없음 (이미 69 DB 전체를 본 checkpoint 사용 → **진단용**이지 해결책 아님을 명확히 인지).
- 홀드아웃 DB: `['bike_share_1', 'book_publishing_company', 'coinmarketcap', 'ice_hockey_draft', 'movie_3', 'movies_4', 'restaurant', 'shooting', 'talkingdata', 'university', 'video_games']`
- Implementation: `b5_head_retrain.py` 에 `--ldbo_frac`, `--train_json` 인자 추가; runner `scripts/run_b5_ldbo_diagnostic.sh` (GPU 0 순차, 4 cells B/C/D/E).

**결과 (LDBO val-ES vs Query-Random val-ES 비교)**:

| Exp | Loss | Norm | LDBO val R@15 | QR val R@15 | **LDBO Dev AUC(ES)** | **QR Dev AUC(ES)** | Δ(L−Q) |
|-----|------|------|---------------|-------------|----------------------|--------------------|--------|
| B | bce | none | 0.9963 | 0.9962 | 0.6614 | 0.6648 | −0.003 |
| C | listnet | none | 0.9958 | 0.9962 | 0.6594 | 0.6891 | **−0.030** |
| D | bce | zscore | 0.9947 | 0.9953 | 0.6660 | 0.6724 | −0.006 |
| E | listnet | zscore | 0.9950 | 0.9948 | **0.6761** | 0.6687 | **+0.007** |

LDBO oracle best_dev: B 0.6980@1 / C 0.6722@4 / D 0.7029@1 / E 0.6877@10.

**핵심 진단** (negative result, **본 실험의 가장 중요한 발견**):

1. **LDBO val R@15 여전히 0.99+** → 홀드아웃된 11 train DB 가 "unseen" 역할을 충분히 수행하지 못함. `proxy_dev` 에서도 head 는 여전히 거의 완벽히 recall 함.
2. **val↔dev R@15 gap 변화 없음**: query-random 0.99-0.62=0.37 vs LDBO 0.99-0.62=0.37.
3. **val-ES 기준 Dev AUC 도 LDBO 에서 개선 없음** (B/C/D 약간 악화, E만 +0.007). 즉 LDBO-ES 가 oracle dev-ES 보다 나은 가이드 못 됨.
4. 해석: **BIRD train DB 간 domain 다양성 ≪ BIRD train↔dev domain gap**. Dev 의 11 DB 는 schema/column naming style 이 train 에서 전혀 안 보이는 영역 → train 내부에서 아무리 DB 홀드아웃해도 그 shift 를 simulate 할 수 없음.

**함의**:
- **LDBO-ES 전략 무용** (최소한 head-only 수준에서는). 더 근본적인 intervention 필요.
- 가능한 방향:
  1. **GAT 재학습 with LDBO split** (이 실험은 GAT frozen; GAT 까지 LDBO 로 재학습하면 representation 자체가 "unseen DB" 를 본 적 있게 됨 — 비용 ~8h)
  2. **Encoder 개선**: L_out 의 근본은 sentence encoder 출력. Dev DB 의 novel column name 에 대한 encoder OOD 가 실제 병목일 가능성.
  3. **Domain-adversarial training** (DANN-style): DB-id 를 gradient-reversal 로 invariance 신호
  4. **Cross-DB contrastive pretraining of encoder** on BIRD-wide column corpus
- Val-ES 와 dev-ES 는 여전히 큰 gap (ex. C oracle 0.7548 vs val-ES 0.6891 = 0.066). "dev label 을 early stop 기준으로 쓰는 것" 을 피하려면 별도 meta-dev 분할 필요.

**논문 기여 매핑**:
- §7-3 의 negative result 는 "naive LDBO 로는 BIRD 의 train-dev shift 해결 불가" 라는 **방법론 경고** 형태로 논문 §V (limitations / future work) 에 인용 가능.
- §7-2 의 "joint training 의 head 가 oracle 대비 suboptimal" 발견은 §III-C 학습 레시피 개선 근거로 사용 가능.

**구현 산출물**:
- `src/analysis/b5_verification.py` — cosine / classifier logit / linear probe 진단
- `src/analysis/b5_extract_frozen_lout.py` — frozen L_out 캐시 추출 (train/dev)
- `src/analysis/b5_head_retrain.py` — head-only retrain (query-random + LDBO 양쪽 지원)
- `scripts/run_b5_head_retrain.sh` — A/B (linear vs mlp, bce, none)
- `scripts/run_b5_head_retrain_CDE.sh` — C/D/E (mlp × listnet/zscore grid)
- `scripts/run_b5_ldbo_diagnostic.sh` — LDBO 4 cells, GPU 0 순차

### §7-4. B5 Enriched — Training + 3축 병목 분석 확장 (2026-04-21)

**Motivation**: §7-2/§7-3 결론이 "L_out 이후 head 학습은 포화, 진짜 병목은 train→dev domain shift". Dev 의 novel column naming 에 대한 encoder OOD 완화를 시도하려면 **입력 텍스트 자체를 풍부하게** — `tables.json` 의 자연어 테이블/컬럼명과 `database_description/*.csv` 주입 (EnrichedHeteroGraphBuilder). B5 구조 그대로 유지하여 **"enrichment 순수 효과"** 만 측정.

**Setup**:
- Config: `configs/experiments/s06_gat_bottleneck_fix/a01_additive_ablation/s06_a01_07_b5_enriched_dual_stream.yaml`
- Builder: `EnrichedHeteroGraphBuilder(tables_json_path=/SSL_NAS/peoples/khj/thesis/train/train_tables.json)`
- Model: B5 동일 (PN + IR α=0.2 + JK concat + Dual-Stream + ListNet + AC 0.3, L=2)
- `batch_size=8` (기존 B5 batch=1 → 29h 에서 병목, batched dual_stream 코드로 가속)
- 300 epoch, pos_weight=100, lr=1e-4

**Training 결과** (2026-04-20 21:49 ~ 04-21 07:03):

| 항목 | 값 |
|---|---|
| 총 학습 시간 | **9h 14m** (B5 ~29h 대비 **3.1× 단축**) |
| Best Val R@15 | **0.6016 @ Epoch 60** (B0 0.5738 대비 +0.0278, **B5 0.6073 대비 −0.0057**) |
| Best 갱신 에폭 | E60 (23:46:37 04-20) 이후 240 epoch 무갱신 → early saturation |
| Final (E300) | Loss 1.1382, Val R@15 0.5969 |
| Checkpoint | `/SSL_NAS/peoples/khj/thesis/checkpoints/s06_gat_bottleneck_fix/best_gat_s06_a01_07_b5_enriched.pt` (67 MB) |

**핵심 관찰**:
1. **Enriched features 가 R@15 를 개선하지 못함** (−0.0057 vs B5). Final train loss 는 소폭 낮음 (1.138 vs 1.162) → train fit 개선, dev 일반화 실패. Train 내부 DB 의 NL/description 에 과적합 가능성.
2. **수렴 속도는 B5 와 동일 (Best @ E60)** — 60 epoch 면 충분, 300 epoch 무용. 향후 학습은 80~100 epoch 로 단축 권장.
3. **Batched dual_stream 코드 이득** — batch_size=1 → 8 로 학습시간 3.1× 단축.

**3축 병목 분석** (CPU, dev 1534 queries, 스크립트: `src/analysis/gat_bottleneck_analysis_v2.py --models B5E`):

| ID | L0_PLM | L1_GAT | L2_GAT | L_out | grad_ratio |
|----|--------|--------|--------|-------|-----------|
| B5 | 0.657 | 0.373 | 0.920 | 0.357 | 0.687 |
| **B5E** | **0.636** | 0.430 | **0.978** | **0.329** | 0.244 |

**Step 2 관찰** (Over-smoothing):
- **L0 더 분산 (0.657 → 0.636)**: Enriched 텍스트가 column 간 원본 임베딩을 의도대로 분산시킴 — 시작점은 B5 보다 좋음.
- **L2 더 collapse (0.920 → 0.978, 거의 완전 동질화)**: 2-layer GAT 후엔 오히려 sibling column 이 더 비슷해짐. `column→belongs_to→table` attention (entropy ≈1.95, B5 와 동일) 이 richer features 를 table 중심으로 과도하게 pooling. **Enriched info 가 table-centric homogenization 을 강화** 하는 역설.
- **L_out 더 분산 (0.357 → 0.329)**: Dual-Stream fusion 이 L2 collapse 를 뚫고 최종 표현을 오히려 더 분리. **"Fusion head 가 GAT 병리를 사후 교정"** — B5E 에서 이 분업이 더 뚜렷.

**Step 3 관찰** (Gradient flow):
- **모든 파라미터 그룹 gradient 가 2~4× 증가**: `lin_dict` 0.43→1.13, `conv_L1` 0.043→0.171, `jk_lin` 0.144→0.515, `fusion_head` **0.59→1.83**, `query_encoder` 0.63→1.35. 학습 신호 양적으로 증가하나 R@15 로 전환 실패.
- **`grad_ratio` 0.244 (B5 0.687)**: conv_L2 gradient 가 conv_L1 의 1/4 수준. 2-layer 에선 L2 가 조기 수렴한 것으로 해석 가능 (vanish 위기 아님 — 절대값은 B5 보다 큼).
- **Fusion 이 최대 gradient 담당 (1.83)**: B5 대비 3.1× 증가. **"Enriched 로 인한 L2 collapse 는 Fusion 이 보완"** — 그러나 보완이 완벽하지 못해 R@15 회복까진 못 함.
- **Attention entropy 는 B5 와 거의 동일** — edge weight 학습엔 영향 못 미침.

**핵심 발견**:
- **Enriched features 의 순효과는 neutral~slightly negative** (−0.0057 R@15). train-internal val 에선 fit 향상, dev 일반화 실패. §7-3 의 "BIRD dev shift 는 encoder-level 개선 없이 풀리지 않음" 가설과 정합 — Enriched text 로 input 만 풍부하게 해도 representation gap 해소 안 됨.
- **2-layer 구조 + Fusion 이 L2 collapse 를 사후 교정하는 패턴 강화**. Enriched 의 더 풍부한 input 은 오히려 GAT 를 통과하며 homogenize → Fusion 부담 증가.
- **학습 시간 3.1× 단축 이득은 명확** (batched dual_stream code). Code 는 유지가치 있음.

**함의 (후속 실험)**:
1. **B5E L=3 재학습**: Enriched 에 추가 hop 이 필요한지 검증 (현재는 2-layer 가설 차용).
2. **"Enriched only" 효과 분리**: B0 또는 B4 에 Enriched 만 투입 (Dual-Stream/JK 없이) → 순수 enrichment 기여도 측정.
3. **Downstream E2E F1 재평가**: R@15 는 −0.0057 이지만 L_out 품질 (−0.028) 은 개선. Extractor/Filter 거치면 F1 에서 다를 수 있음.
4. **B5E + Neurosymbolic L1 (§6-21)**: FK-reachability prior 가 Enriched representation 과 결합 시 시너지 측정. `abl_sel_ns_l1_enriched_*` 으로 예약.
5. **Train-time domain-adversarial regularizer**: DB-id 에 대한 gradient reversal 로 domain-invariance 강제 (§7-3 의 방향 #3 후속).

**논문 기여 매핑**:
- "Enriched input features alone insufficient for BIRD train→dev shift" — §V (limitations) 에 §7-3 와 함께 인용 가능한 second evidence.
- "Fusion head as last-stage over-smoothing corrector" — §III-C (model design) 에 dual-stream architecture 의 역할 해석 근거.
- Batched dual_stream 학습가속 (3.1×) — reproducibility 섹션에 실무 노트.

**산출물**:
- Training log: `logs/train/s06_a01_07_b5_enriched_dual_stream_20260420_214928.log`
- Checkpoint: `/SSL_NAS/peoples/khj/thesis/checkpoints/s06_gat_bottleneck_fix/best_gat_s06_a01_07_b5_enriched.pt`
- 3축 분석 결과: `outputs/analysis/s06_bottleneck_b5_enriched/` (B5E 단독), `outputs/analysis/s06_bottleneck_merged/` (B0~B5+B5E cross-model 플롯)
- 분석 문서: [notebooks/analysis_results/s06_bottleneck_b5_enriched_extension.md](notebooks/analysis_results/s06_bottleneck_b5_enriched_extension.md)
- 스크립트: `src/analysis/gat_bottleneck_analysis_v2.py` (+ `--models` filter), `src/analysis/merge_b5e_bottleneck.py`

---

## 8. Wave 1.5 Stagewise Backfill — Extractor 축 통일 (2026-04-22)

**Motivation**: Proposal A stagewise ablation matrix ([`notebooks/analysis_results/stagewise_qcond_ablation.md`](notebooks/analysis_results/stagewise_qcond_ablation.md) §1.1) 에서 EnsembleSelector (legacy cosine-only) vs QCond Projector GAT 대조 시 Extractor 축이 `ComponentAwareProductCostPCSTExtractor`(s04 원본) 와 `PCSTExtractor(Basic)`(legacy baseline) 사이에서 섞여 있어 **순수 Selector 축 기여 분리 불가**. 지도교수 피드백 ([`planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md`](planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md) §9 Root 행) 에 따라 Extractor 를 `PCSTExtractor(Basic)` 로 통일한 3 개 실험을 번들로 재실행.

**Setup** (3 실험 공통):
- Connectivity Extractor: `PCSTExtractor(Basic)` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `node_threshold`=0.1
- Filter: `XiYanFilter` — `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`, `max_iteration`=1, `temperature`=0.0, vLLM serving
- Post-processing: `auto_join_keys=True`
- BIRD-Dev 1,534 queries

| # | Config ID | Selector | α | Recall | Precision | F1 | filter_mean |
|---|-----------|----------|---|--------|-----------|------|-------------|
| W1 | `s04_stagewise_ensemble_raw_a0` | EnsembleSelector (legacy cosine-only) | 0 | **0.6676** | **0.7236** | **0.6944** | 7.6959s |
| W2 | `s04_stagewise_qcond_raw_basic` | EnsembleSelector (QCond encoder) | 0 | **0.6622** | **0.7539** | **0.7051** | 8.6349s |
| W3 | `s04_stagewise_qcond_gat_basic` ★ | EnsembleSelector (QCond + GAT blend) | 0.85 | **0.8169** | **0.7605** | **0.7877** | 1.3908s |

**Stagewise cumulative R/P/F1** (memory rule §4 G2):

| Stage | ensemble_raw_a0 | qcond_raw_basic | qcond_gat_basic |
|-------|-----------------|------------------|------------------|
| Selector only | pending (analyzer reconstruction) | pending (analyzer reconstruction) | pending (analyzer reconstruction) |
| + Extractor (no filter) | R=0.7785 P=0.1330 F1=0.2272 | R=0.7813 P=0.1752 F1=0.2862 | R=0.9651 P=0.1287 F1=0.2271 |
| + Filter (final) | R=0.6676 P=0.7236 F1=0.6944 | R=0.6622 P=0.7539 F1=0.7051 | R=0.8169 P=0.7605 F1=0.7877 |

**No-filter cell backfill (2026-04-22 16:29~17:04)**: NoneFilter 를 pass-through 로 세 config 를 재실행하여 Extractor 직후 상태 측정. W2 (GPU 0) 와 W3 (GPU 1) 은 vLLM 종료 후 병렬 실행 (약 7 분 단축). Config: `configs/experiments/s04_ablation/stagewise/no_filter/{ensemble_raw_a0,qcond_raw_basic,qcond_gat_basic}_no_filter.yaml`, Script: `scripts/run_wave15_no_filter.sh` (CUDA_VISIBLE_DEVICES=0,1). Output: `outputs/experiments/s04_ablation/stagewise/no_filter/<config>/metrics.txt`.

**Filter 가 stage 에 기여한 Δ(F1)**:
- W1: 0.2272 → 0.6944 (**+0.4672**)
- W2: 0.2862 → 0.7051 (**+0.4189**)
- W3: 0.2272 → 0.7877 (**+0.5605**) — 최대. XiYan Filter 가 Precision 0.1287 → 0.7605 (**+0.6318**) 끌어올리며 Recall 0.9651 → 0.8169 (−0.1482) 로만 손실. Selector+Extractor 단계에서 Recall ceiling 이 높을수록 Filter 의 Precision 정제 폭이 커진다는 관찰.

Selector-only stage (W1/W2/W3 raw_seeds 행 3 cell) 는 `notebooks/analysis_results/stagewise_qcond_ablation.md` §4.2 **Open queue** 로 이관됨 (analyzer 3차 개정 2026-04-22 기준 §1.1 15 cell 중 **14 확정**, Selector-only 1 행만 pending). 재구성 방법: `output_*.jsonl` 의 `raw_seeds` 필드에서 gold 대비 R/P/F1 집계. 2026-04-28 발표 전에는 deferred — §5 3-stage 점진 gain 서사에 영향 없음. 발표 후 analyzer 다음 턴으로 지시 시 완성 가능.

**핵심 관찰**:
1. **QCond GAT (W3, α=0.85) F1=0.7877 — 현 시점 최고 tied** (abl_ens_basic_xiyan F1=0.7863 대비 +0.0014). Recall 이 **0.8169** 로 급등 (W1/W2 0.66대 → **+0.15**) 하면서 Precision 은 유지 (0.7605).
2. **α=0 축 순수 QCond encoder 효과** (W1→W2): **F1 +0.0107** (0.6944→0.7051). Extractor 통일 조건에서 QCond encoder 가 Legacy cosine-only 대비 Precision +0.0303 개선 (0.7236→0.7539), Recall 동등 (0.6676→0.6622, −0.005).
3. **GAT blend 기여** (W2→W3): α=0→α=0.85 로 **F1 +0.0826** (0.7051→0.7877). GAT score 블렌드가 Recall 증폭 — Basic PCST 가 noise 노드도 넓게 포함하는 상황에서 GAT positive 신호 노드가 seed 로 승격되어 downstream subgraph 에 올바른 노드 더 많이 포함.
4. **W3 Filter 호출 감소 (91.6%)**: `filter_llm_calls_mean`=0.9159 vs W1/W2 ~1.0 (100%). `extractor_selected_nodes_mean`=83.84 (W1 51.25, W2 43.45) — Subgraph 자체는 크지만 일부 쿼리에서 filter skip (추가 분석 필요 — 조건 확인 후 §1.1 재작성에 반영).

**Operational note — NAS folio wait stall (2026-04-22 01:17 ~ 13:25)**:
- W3 첫 실행 시 쿼리 #505 부터 **filtering time 3800~3900s/query** 폭증, NAS (96% 포화, 1.1 TB 여유) `folio_wait_bit_common` 커널 대기로 1534 중 508 개에서 정체. vLLM 은 HTTP 200 정상.
- 원인: `data/raw/BIRD_dev/` 가 `/SSL_NAS/peoples/khj/thesis/dev/` symlink. XiYan filter 의 `_build_mschema_with_values` 가 쿼리마다 NAS sqlite 읽기 → NFS folio 대기.
- **해결**: symlink 제거 후 NAS → 로컬 `/home/hyeonjin/thesis_refactored/data/raw/BIRD_dev/` rsync 1.4 GB 복사 (~62 분). 재시작 후 `filter_mean` 7.70s → **1.39s** (5.5× 가속), `filter_max` 1908s → **7.15s**. 전체 실험 **46m 39s** (14:38:01 → 15:24:40) 에 완료.
- 교훈: dev sqlite 가 NAS 에 있으면 NFS 포화 상태에서 XiYan filter 심각 stall. 향후 BIRD dev 는 로컬에 유지 (SSD `/home/hyeonjin/thesis_refactored/data/raw/BIRD_dev/` 1.4 GB).

**논문 기여 매핑**:
- "α=0 축 순수 Selector encoder 교체 효과 정량화" — §IV-B ablation 표에 W1 vs W2 cell 추가 근거.
- "QCond + GAT blend (α=0.85) + Basic PCST + XiYan — **new top F1=0.7877**" — §IV-A main results 상단 수치 업데이트.
- "Extractor 축 통일 후에도 QCond+GAT 의 Recall dominance (0.82 vs α=0 series 0.66)" — §V discussion 에서 "GAT 가 Filter recall floor 를 끌어올리는 메커니즘" 근거.

**산출물**:
- Configs: `configs/experiments/s04_ablation/stagewise/ensemble_raw_a0.yaml`, `qcond_raw_basic.yaml`, `qcond_gat_basic.yaml`
- Script: `scripts/run_wave15_backfill.sh` (2026-04-22 audit 에서 `CUDA_VISIBLE_DEVICES` 2,3 → 0,1 정정 완료)
- No-filter 변형: `configs/experiments/s04_ablation/stagewise/no_filter/{ensemble_raw_a0,qcond_raw_basic,qcond_gat_basic}_no_filter.yaml`, `scripts/run_wave15_no_filter.sh`
- Metrics: `outputs/experiments/s04_ablation/stagewise/<config>/metrics.txt`, `outputs/experiments/s04_ablation/stagewise/no_filter/<config>/metrics.txt`
- Logs: `logs/experiments/s04_ablation/stagewise/<config>/`, `logs/experiments/s04_ablation/stagewise/no_filter/<config>/`
- Partial run backup (W3 stall 시점): `/tmp/qcond_gat_basic_partial_backup_20260422/` (508 쿼리, NAS stall 검증용)

---

## 9. Bug Fixes & Reproducibility Notes

### §8-1. BIRDSuperNodeDataset split-order bug (fixed 2026-04-21)

**요약**: `src/train_gat.py` 에서 `BIRDSuperNodeDataset(...)` 래핑이 `random_split(...)` **이후** 적용되어, train/val DataLoader 가 실제로는 **래핑되지 않은 원본** dataset (query_node 미주입, supernode edges 없음) 을 순회했음. `query_supernode=True` config 로 학습된 체크포인트는 SuperNode 경로를 학습 단계에서 전혀 exercise 하지 않은 상태.

**원인**: PyTorch `torch.utils.data.random_split` 은 `Subset` 객체를 반환하며 생성 시점의 dataset reference 를 캡처한다. 따라서 `full_train_dataset = BIRDSuperNodeDataset(full_train_dataset)` 으로 변수에 재바인딩해도 이미 생성된 `Subset.dataset` 참조에는 전파되지 않는다. Dummy batch 는 wrap 직후 생성되어 모델 초기화만 SuperNode 구조로 수행, 정작 학습 루프는 vanilla GAT 로 흘렀다.

**증상**: 학습 로그에 `Query Super Node mode` 가 찍히고 체크포인트도 `SN=True` 플래그로 저장되지만, 실제 파라미터는 SuperNode 유도 signal 을 학습하지 못함. Post-hoc inference 시 SuperNode 경로를 타더라도 해당 파라미터는 "초기화 상태 + vanilla GAT gradient 로만 간접 영향" 수준.

**수정 (2026-04-21)**: wrap 블록을 split 이전으로 이동, flag 추출도 dataset load 이전으로 끌어올림. `src/train_gat.py` 라인 214~243 참고. s06 라인 학습 스크립트인 `src/train_gat_s06.py` 는 설계 당시부터 wrap-before-split 순서가 올바랐음 → **s06 a01_* 시리즈 (B0~B5/B5E) 는 영향 없음**.

**재현성 의심 (suspect) 체크포인트 / 실험**:
- **GAT 학습**: `T7` (`best_gat_query_supernode.pt`), `T9` (`best_gat_query_supernode_direct.pt`) — 둘 다 `src/train_gat.py` + `query_supernode=True` 로 학습.
- **Enriched 라인의 SuperNode 변형**: `configs/training/train_gat_enriched_query_supernode.yaml` 기반 체크포인트가 존재한다면 동일 조건 — 재학습 필요.
- **Downstream eval 파생 실험**: 위 체크포인트를 그대로 사용한 Q2 (`supernode_idea24_xiyan`), Q3 (`supernode_idea24_a085_xiyan`), Q5 (`supernode_idea24_a0_xiyan`), Q7 (`supernode_direct_idea24_xiyan`) — Selector/Extractor/Filter 수치는 버그 있는 weights 위에서 측정된 값이므로 수정본으로 재실행 전까지 "reproducibility suspect" 로 표기.

**영향 없음 (unaffected)**:
- s06 시리즈 전체 (`s06_a01_01` ~ `s06_a01_07` B0/B1/B2/B3/B4/B5/B5E) — `train_gat_s06.py` 로 학습.
- `query_supernode=False` 계열 (T1~T6, T8, s06 B0~B5E 등).
- V-2 smoke 체크포인트 `best_gat_enriched_v2_smoke.pt` (2026-04-21 재학습, 수정본 train_gat.py 사용) — `Val R@15=0.4471`, query_node stats 가 train 단계 로그에 찍혀 정상 동작 확인.

**후속 조치**:
1. V-2 `directed_from_sn` full-epoch 재학습 (수정본 코드, 20 epoch) → `best_gat_enriched_v2_directed.pt` 생성 후 과거 T7/T9 를 대체.
2. V-3 `supernode_topk∈{3,5,10,20}` 실험은 수정본 코드 + top-k 기준 `raw` 로 새로 학습.
3. Q2/Q3/Q5/Q7 결과는 V-2 full / V-3 peak 확정 이후 해당 체크포인트로 재실행하여 덮어쓸 예정. 그 전까지 논문 표/그래프에서 SuperNode 계열 수치는 인용 자제.

**커밋 검증**: `src/train_gat.py` (line 214~243) wrap-before-split 순서 + flag 선추출 반영. Validation 로그 `logs/gat_enriched_v2_smoke/train/train_step.jsonl` 의 step 0~1060 query_node 항목(skip_ratio=0.5647, out_norm_mean, last_layer_delta=2.039 등) 이 실제 학습 step 동안 갱신되었음을 확인.

---

## Wave 2 Proposal C GLM era kickoff (2026-04-24)

**LLM backbone 전환**: vLLM `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` → **GLM-4.7 (`zai-org/glm-4.7`) via Elice ML API (OpenAI-compatible)**.
근거: [planning/DECISIONS.md](planning/DECISIONS.md) 2026-04-24 엔트리들 (LLM 전환 + Sanity 합격 기준 재정의).
합격 기준 (재정의): vLLM era 동일 anchor 대비 **ΔF1 ≥ −0.02**.

### 7 cells 최종 결과 (R / P / F1, 4자리)

| 실험 ID | num_layers | Recall | Precision | F1 | vLLM era 대응 F1 | ΔF1 |
|---------|-----------:|-------:|----------:|---:|----------------:|----:|
| `s04_04_qcond_a0_xiyan_glm` (sanity) | 3 | 0.4922 | 0.6965 | 0.5768 | 0.5866 | **−0.0098** ✅ |
| `abl_sel_diameter_layers_nl1_glm` | 1 | 0.4897 | 0.7067 | 0.5785 | — (new sweep) | — |
| `abl_sel_diameter_layers_nl2_glm` | 2 | 0.4632 | 0.6800 | 0.5510 | — (new sweep) | — |
| `abl_sel_diameter_layers_nl3_glm` | 3 | 0.4901 | 0.6961 | 0.5752 | — (new sweep) | — |
| **`abl_sel_diameter_layers_nl6_glm` (= D_max)** | **6** | **0.5018** | **0.6939** | **0.5824 ← sweep peak** | — (new sweep) | — |
| `abl_sel_diameter_layers_nl7_glm` (= D_max+1) | 7 | 0.4920 | 0.6952 | 0.5762 | — (new sweep) | — |
| **🚀 `s04_stagewise_qcond_gat_basic_glm`** (new anchor) | 3 | **0.8438** | **0.8329** | **0.8383** | **0.7877** | **+0.0506** |
| `layers_Ldbmax_glm` (H2 truncate, 2026-04-25) | 6 (truncate) | 0.5036 | 0.7031 | **0.5869** | — (new) | vs L6_glm: **+0.0045** partial neutral |
| `layers_Ldbmax_plus1_glm` (H2 truncate, 2026-04-25) | 7 (truncate) | 0.4778 | 0.6776 | **0.5604** | — (new) | vs L6_glm: **−0.0220** 기각 확고 |

**3단계 Selector/+Extractor/+Filter cumulative R/P/F1**: pending — `notebooks/analysis_results/diameter_layers_sweep.md` GLM era 작성 시 analyzer 재집계 (memory rule G2 적용).

### 주요 발견
1. **🚀 GLM era 새 전체 최고**: `s04_stagewise_qcond_gat_basic_glm` F1=0.8383 — Wave 1.5 vLLM best (F1=0.7877) 대비 **ΔF1=+0.0506** (ΔR=+0.0269, ΔP=+0.0724). Precision 주 개선축.
2. **H1 peak 가설 검증 완료**: diameter_layers sweep 에서 nl=D_max(6) 이 peak (F1=0.5824), nl=D_max+1(7) 에서 0.5762 로 **over-smoothing 재등장 관측** (ΔF1=−0.0062).
3. **α=0 anchor (sanity) ΔF1=−0.0098**: LLM backbone 교체로 인한 미세 하락, planner 재정의 기준 ΔF1 ≥ −0.02 통과 → sweep 진행 승인.
4. **L2 dip 이상치**: nl=2 에서 F1=0.5510 으로 nl=1(0.5785), nl=3(0.5752) 보다 낮음 — GAT 2-layer 특이 패턴 (analyzer 후속 분석 대상).

### 비용
- 7 cells 실제 token: input ≈ 7.3M + output ≈ 0.24M = **~₩5,350 (~$3.8 USD)**.
- 초기 추정 ₩26,740 의 약 **1/5** (실제 input/query ≈ 683 tokens, 추정 3K/query 의 1/5). Extractor 평균 18.58 nodes 선택 → M-Schema 간결.

### 운영 기록
- **Phase 2 1회 full failure** (17:05~17:24): (a) `num_layers` 파라미터 yaml 미지정 → L1/L2/L6/L7 체크포인트 weight shape mismatch (`EnsembleSelector` default `num_layers=3` vs nl=1/2/6/7 ckpt); (b) `/home` 파티션 100% full → `card_games.sqlite` WAL write 불가 → I/O error 전파.
- **복구 (17:24~17:36)**:
  - `api_handler.py` + `base.py` + 7 filter classes provider 분기 (filter 세션 완료)
  - `.env.example` + 7개 GLM yaml configs + scripts vLLM→GLM 헬스체크 (root, 2026-04-24 이전 turn)
  - `scripts/run_vllm_server.sh` nohup + setsid + disown 강화 (SSH 끊김 대비, 향후 vLLM 재사용 시)
  - **disk cleanup**: `outputs/archive/` 582 MB → NAS 이동 (`/SSL_NAS/peoples/khj/thesis/outputs_archive_20260424`) + symlink, `layers_L3_partial_*` 19 MB 삭제, `card_games.sqlite-{shm,wal}` 재생성용 제거
  - **5 yaml 수정**: `configs/.../layers_L{1,2,3,6,7}_glm.yaml` 에 `num_layers: N` 파라미터 추가 (N=1/2/3/6/7)
- **Sweep 재실행 (17:36~23:00) 정상 완료** — 다른 user 의 `/home` 사용량 변동에 의한 일시적 disk 급감 (486 MB → 168 MB) 도 자동 복구 (→ 7.6 GB).

### 산출물
- **Configs (7)**: `configs/experiments/s04_ablation/{s04_04_qcond_a0_xiyan_glm.yaml, diameter_layers/layers_L{1,2,3,6,7}_glm.yaml, stagewise/qcond_gat_basic_glm.yaml}`
- **Script**: `scripts/run_glm_era_kickoff.sh` (sanity 합격 후 6 cells sequential runner, setsid+nohup+disown)
- **Logs**: `logs/experiments/s04_ablation/{.../layers_L{1,2,3,6,7}_glm/, stagewise/qcond_gat_basic_glm/, s04_04_qcond_a0_xiyan_glm/}`
- **Outputs**: 동일 경로 미러 (metrics.txt, predictions.jsonl, output_*.jsonl, profiling_*.jsonl, score_analysis_*.jsonl, stage_aggregates.json, token_usage.json)
- **이전 실패 백업**: `outputs/experiments/s04_ablation/diameter_layers/layers_L{1,2,3}_glm_failed_20260424_1724/`, `stagewise/qcond_gat_basic_glm_failed_20260424_1724/`

### 2026-04-25 H2 truncate 검증 (analyzer recon ≠ selector impl mechanism 실측)

**근거**: [planning/DECISIONS.md](planning/DECISIONS.md) 2026-04-26 "Selector H2 inference 2 cell 실측 승인" 엔트리 + 2026-04-25 "H2 원래 가설 기각" 엔트리.

**Mechanism 차이**:
- **Analyzer reconstruction** (F1=0.5805, sweep 5 cell 재조합): D_max=4/5 query (95.8%) 가 nl=6 cell 결과 그대로 fallback (nl=4/5 ckpt 부재).
- **Selector impl** (이번 실측, EnsembleSelector v2 + `_resolve_active_depth` hook): nl=6/7 ckpt 의 layer 수 **동적 truncate forward** (D_max<6 DB 도 per-DB 동적 depth).

**2 cells 결과**:

| Cell | num_layers_mode | ckpt | R | P | F1 | ΔF1 vs L6_glm (0.5824) | 4-way 분기 |
|------|-----------------|------|---|---|---|---|---|
| `layers_Ldbmax_glm` | `D_max` | nl=6 | 0.5036 | 0.7031 | **0.5869** | **+0.0045** | **(3) partial neutral** |
| `layers_Ldbmax_plus1_glm` | `D_max_plus1` | nl=7 | 0.4778 | 0.6776 | **0.5604** | **−0.0220** | **(1) 기각 확고 (training mismatch)** |

**판정** (2026-04-26 DECISIONS §영향 범위 표 사전 합의 기준):
- **Ldbmax**: 분기 (3) −0.002 ~ +0.005 — partial neutral. 발표 슬라이드 C-3 minor mention 권장. nl=6 ckpt truncate 가 analyzer recon (0.5805) 대비 +0.0064 로 mechanism 차이가 **미세 positive** 효과는 확인되나 global fixed nl=6 (0.5824) 대비는 noise 범위.
- **Ldbmax_plus1**: 분기 (1) < −0.005 — H2 기각 확고. nl=7 ckpt 이 7-layer 로 학습됐는데 D_max<7 DB 에서 4~6 layer 로 truncate → training-inference depth mismatch 가 분기 (1) 예측대로 큰 손실 발생.
- **종합**: H2 truncate mechanism 은 nl=6 ckpt 에서만 미미한 positive (+0.0064 vs analyzer recon). 2026-04-25 "H2 원래 가설 기각" 결정 **유지**, 발표 슬라이드 C-3 narrative 분기 불필요 (minor mention 수준).

**Mechanism 비교 (analyzer recon 대비)**:
- Ldbmax (F1=0.5869 vs recon 0.5805, +0.0064) = per-DB 동적 depth 가 nl=6 fallback 대비 약간 유리. D_max<6 버킷 (62%) 에서 truncate forward 가 효과.
- Ldbmax_plus1 (F1=0.5604 vs recon 0.5805, −0.0201) = nl=7 ckpt 의 7-layer training 과 truncate forward 의 depth mismatch 가 negative dominates.

**비용**: 2 cell × ~₩780 = **~₩1,560** (병렬 ~55min wall clock, wrapper PID 1074671, 01:36:06~02:33:12).
**산출물**:
- Configs (selector 세션 산출, 2026-04-25 01:05/01:06): `configs/experiments/s04_ablation/diameter_layers/layers_Ldbmax{,_plus1}_glm.yaml`
- Script: `scripts/run_h2_truncate.sh` (GLM health check + 2 cell 병렬 launch + wait + metrics summary)
- Logs: `logs/experiments/s04_ablation/diameter_layers/layers_Ldbmax{,_plus1}_glm/`
- Outputs: `outputs/experiments/s04_ablation/diameter_layers/layers_Ldbmax{,_plus1}_glm/` (metrics.txt 포함 전체 5종)

### 후속 (핸드오프)
- **Planner**: analyzer 큐 추가 — `notebooks/analysis_results/diameter_layers_sweep.md` (H1 검증 곡선 + vLLM era ↔ GLM era 비교 부록 + L2 dip 진단)
- **추후**: Wave 3 Proposal F (analyzer 단독 — SteinerBackbone 재조직), Wave 4 `a05_filter_agentic` (post-2026-04-28) — 모두 GLM era backbone 유지 가능 (합격 기준 ΔF1 ≥ −0.02 적용).
