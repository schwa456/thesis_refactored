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

Anchor: a03_17 (SuperNode Direct + Fixed PCST). Filter만 교체하여 agentic refinement 효과 비교. Backbone: Qwen3-Coder-30B-A3B-Instruct-FP8 (vLLM, GPUs 2+3).

| ID | Filter | Recall | Precision | F1 | Runtime |
|----|--------|--------|-----------|------|---------|
| a03_17 (anchor) | XiYan | 0.6761 | 0.7128 | **0.6940** | — |
| a05_01 | AdaptiveMultiAgent (Semantic+Structural+Skeptic) | 0.3770 | 0.6276 | 0.4713 | 10h 23m |
| a05_02 | ReflectionFilter (1 iter, propose→critique→revise) | **0.7320** | 0.6833 | **0.7068** | 3h 18m (7.3s/q) |

**관찰**:
- **a05_01 F1=0.4713, anchor 대비 −22.3%p**: 3-agent consensus가 지나치게 보수적으로 교집합화 — Recall 0.38로 anchor 대비 -30%p 대폭 손실. Precision도 anchor XiYan보다 낮음 (0.63 < 0.71).
- JSON Parsing failed warning 빈발: agents.py fallback이 Unanswerable로 처리되어 빈 선택 누적 → Recall 파괴.
- **a05_02 F1=0.7068, anchor 대비 +1.3%p (신기록)**: Critique-revise가 Recall을 0.68→0.73으로 밀어올림. Precision은 0.71→0.68로 소폭 하락하나 net F1 상승. **Restore path 확보가 실제로 Recall 천장을 돌파**함을 실증.
- 향후 agentic filter는 (1) prune-only 대신 restore 경로 확보, (2) parsing robustness, (3) fallback 시 XiYan 결과 유지가 필수.

---

## 7. 전체 실험 순위 (Recall 기준 Top 10)

| Rank | Experiment | Recall | Precision | F1 | Key Components |
|------|-----------|--------|-----------|------|----------------|
| 1 | abl_ens_basic_xiyan | **0.8149** | 0.7597 | 0.7863 | Ensemble + BasicPCST + XiYan |
| 2 | abl_cos_basic_xiyan | 0.7987 | 0.7694 | 0.7838 | Cosine + BasicPCST + XiYan |
| 3 | **a05_02_reflection_1iter** | **0.7320** | 0.6833 | 0.7068 | Ensemble + AdaptivePCST + **ReflectionFilter(1iter)** |
| 4 | edge_prize | 0.6823 | 0.8139 | 0.7424 | TripletBuilder + EdgePrizePCST + XiYan |
| 5 | abl_a03_17_supernode_binary_fixed_xiyan | 0.6761 | 0.7128 | 0.6940 | SuperNode-Direct(binary, τ=0.5) + BasicPCST + XiYan |
| 6 | enriched_gat | 0.6658 | 0.8147 | 0.7327 | EnrichedBuilder + Ensemble + Adaptive + XiYan |
| 7 | abl_a04_01_supernode_t005_steiner_xiyan | 0.6353 | 0.7054 | 0.6685 | SuperNode-Direct(τ=0.05) + SteinerBackbone + XiYan |
| 8 | idea24_product_component_xiyan | 0.6304 | 0.8028 | 0.7063 | Ensemble + ProductCost+Component + XiYan |
| 9 | b4_xiyan_filter | 0.6244 | 0.7930 | 0.6987 | Ensemble + AdaptivePCST + XiYan |
| 10 | qcond_idea24_xiyan | 0.6236 | 0.8056 | 0.7032 | QueryCond(α=0.85) + Idea2+4 + XiYan |

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

### 구조적 패턴

- **Basic PCST + XiYan > Adaptive PCST + XiYan**: F1 기준 0.7863 vs 0.6987. XiYan이 pruning을 더 잘하므로 PCST는 넓게 포함시키는 게 유리.
- **Precision과 Recall의 trade-off**: 모든 실험에서 일관되게 나타남. Filter가 precision을 올리면 recall이 내려감. **ReflectionFilter는 이 trade-off를 명시적으로 recall 방향으로 이동**시킨 첫 개입.
- **BO 결과**: `bo_score_driven` (P=0.7867, R=0.5910) > `bo_fixed_cost` (P=0.7468, R=0.4793). Score-driven cost weights가 dev F1 기준으로 fixed cost보다 우월.
- **Edge Prize PCST**: F1=0.7424로 XiYan 조합 중 Top-3. Triplet edge embedding → edge prize가 connectivity-aware pruning에 유효.
- **GAT 기여가 제한적인 이유**: (1) α=0.85에서 15%만 반영 (2) query-agnostic attention (3) FK 노드 label 부재

### 다음 단계 제안

1. **a05_03 Reflection 3iter** (진행 중, 2026-04-15 기준 ~34%): 1iter 대비 추가 recall 회복 또는 수렴 여부 확인.
2. **a05_04 VerifierFilter (CHESS-style Unit Tester)**: NL unit test로 선택 검증, 실패 시 expansion.
3. **a05_05/06 Tiered Bidirectional Agent (F3)**: Tier-1(PCST) vs Tier-2(selector-only) 구분 + graph-native tools.
4. **a05_07 Uncertainty-gated adaptive depth (F4)**: GAT confidence 기반 agentic compute 조절.
5. **a05_09/10 Extraction-retry (F5)**: Unanswerable verdict → Extractor 완화 재호출 loop.
6. **a05_11/12 GPT-4o-mini backbone**: Qwen3-Coder-30B 대비 backbone 민감도 검증.
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
| `VerifierFilter` (예정) | `model_name`, unit test 개수 | CHESS-style NL unit tester + Expander. a05_04 |
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
| a05_04 | a05_04_verifier | `VerifierFilter` (XiYan + Unit Tester) | Qwen | ⏸ | - | - | - |
| a05_05 | a05_05_bidirectional_notool | `BidirectionalAgentFilter` (tier1+tier2, no tools) | Qwen | ⏸ | - | - | - |
| a05_06 | a05_06_bidirectional_fulltool | `BidirectionalAgentFilter` + graph tools ★ | Qwen | ⏸ | - | - | - |
| a05_07 | a05_07_adaptive_depth | `AdaptiveDepthFilter` (uncertainty gating) | Qwen | ⏸ | - | - | - |
| a05_08 | a05_08_verifier_bidirectional | F3 + F2 stacked | Qwen | ⏸ | - | - | - |
| a05_09 | a05_09_extraction_retry | F5 Extractor retry (K=2) + F3 | Qwen | ⏸ | - | - | - |
| a05_10 | a05_10_retry_gated | F5 + F4 gating | Qwen | ⏸ | - | - | - |
| a05_11 | a05_11_bidirectional_gpt4omini | F3 full tools | **GPT-4o-mini** | ⏸ | - | - | - |
| a05_12 | a05_12_retry_gpt4omini | F5+F4 stack | **GPT-4o-mini** | ⏸ | - | - | - |
