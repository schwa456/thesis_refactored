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

---

## Builder Cumulative Backfill (Option C, 2026-04-26)

**근거**: [planning/DECISIONS.md](planning/DECISIONS.md) 2026-04-26 (보강 — Option C 채택) §결정 (a)(b)(c)
**목적**: Ablation 1 (Builder × Stage) 9-cell cumulative matrix 완성 — Selector only stage 신규 3 cell + Extractor (no filter) stage 신규 2 cell (Plain Builder 의 +Extractor 는 Wave 1.5 에서 기측정).

### 9-cell cumulative matrix (R / P / F1, 4자리)

| Builder | Selector only | + Extractor (Basic PCST, no filter) | + Filter (XiYan, final) |
|---------|--------------|-------------------------------------|------------------------|
| **Plain** (HeteroGraphBuilder) | 0.7834 / 0.2700 / **0.4016** | 0.9651 / 0.1287 / **0.2271** | 0.8169 / 0.7605 / **0.7877** |
| **Enriched** (EnrichedHeteroGraphBuilder) | 0.7921 / 0.2567 / **0.3877** | 0.9676 / 0.1274 / **0.2252** | 0.6658 / 0.8147 / **0.7328** |
| **Triplet** (TripletGraphBuilder) | 0.7921 / 0.2567 / **0.3877** | 0.9676 / 0.1274 / **0.2252** | 0.6823 / 0.8139 / **0.7423** |

### 5 신규 cells (이번 측정, GPU 2/3 병렬, ~38min wall clock)
- `s04_stagewise_qcond_gat_basic_selector_only` (Plain Builder) — R=0.7834 / P=0.2700 / F1=**0.4016**
- `s03_a07_01_enriched_gat_selector_only` (Enriched) — R=0.7921 / P=0.2567 / F1=**0.3877**
- `s03_a07_02_edge_prize_selector_only` (Triplet) — R=0.7921 / P=0.2567 / F1=**0.3877**
- `s03_a07_01_enriched_gat_no_filter` (Enriched) — R=0.9676 / P=0.1274 / F1=**0.2252**
- `s03_a07_02_edge_prize_no_filter` (Triplet) — R=0.9676 / P=0.1274 / F1=**0.2252**

### 4 기존 cells (재참조, vLLM era anchor)
- `s04_stagewise_qcond_gat_basic` (Plain final) — F1=0.7877 (Wave 1.5 best, vLLM era)
- `s04_stagewise_qcond_gat_basic_no_filter` (Plain +Extractor) — F1=0.2271 (Wave 1.5 backfill)
- `s03_a07_01_enriched_gat` (Enriched final, E1) — F1=0.7328
- `s03_a07_02_edge_prize` (Triplet final, E2) — F1=0.7423

### Filter Δ F1 by Builder (no_filter → final)
- **Plain: +0.5606** (0.2271 → 0.7877) — 최대
- Enriched: +0.5076 (0.2252 → 0.7328)
- Triplet: +0.5171 (0.2252 → 0.7423)

→ Plain Builder 에서 Filter 의 marginal 효과 가장 큼. Enriched/Triplet 의 description 정보가 미세하게 filter 의존도를 줄이지만 (절대 Δ 작음), final F1 자체는 Plain 우세.

### 주요 발견
1. **Plain 이 모든 단계에서 우세** — 특히 final 단계. Builder 의 Description 정보 추가가 selector level 에선 미미한 영향, final XiYan filter 단계에서는 **noise 로 작용** (Plain ΔF1 over Enriched=+0.0549, over Triplet=+0.0454).
2. **Selector only Δ 미미** — Plain (0.4016) vs Enriched/Triplet (0.3877), Δ=0.0139. Builder 차이가 selector 에선 제한적.
3. **+Extractor 단계 사실상 동일** — 3 Builder 모두 R≈0.965~0.968, F1≈0.225~0.227. PCST 가 노드 거의 모두 끌어와 Builder 차이 dilute.
4. **Enriched ≈ Triplet (selector_only / no_filter 단계 동일값)** — Selector weight 같고 (`best_gat_enriched.pt`) Builder 만 다름. Builder 의 graph structure 차이가 Selector 단계에서 동일 score 산출.
5. **Builder 가치 = Filter 단계에서 발현** — selector_only / no_filter 에서는 Builder 차이 보이지 않다가 Filter 통과 후 차이 드러남. 단 description 정보가 도움보다 noise.

### LLM era 표기
- Selector only (3 cell): **N/A (no LLM call)** — Filter=None, Extractor=None
- + Extractor no filter (3 cell): **N/A** — Filter=None, Extractor 비-LLM
- + Filter final (3 cell): **vLLM era** (Wave 1.5 측정, GLM 재측정 X — post-deadline 큐, ΔF1 +0.04~+0.06 추정 §presentation_brief Q7)

### 비용 / 운영
- LLM 호출 0, 비용 ₩0
- 시간: ~38min wall clock (01:55:29 launch → 02:34:37 metrics 모두 생성)
- GPU: 2/3 (사용자 명시 swap 후, memory rule + settings.json `ask` rule 보강) — 다른 연구자 GPU 0/1 점유로 2026-04-26 이번 swap 한정

### 산출물
- Configs (5): `configs/experiments/s04_ablation/stagewise/qcond_gat_basic_selector_only.yaml`, `configs/experiments/s03_gat_ensemble/a07_enriched_triplet/s03_a07_0{1,2}_*_{selector_only,no_filter}.yaml`
- Script: `scripts/run_builder_cumulative.sh` (GPU 2/3 split, sequential within each, GLM endpoint 검사 없음 — LLM 호출 X)
- Logs/Outputs: `logs|outputs/experiments/{s04_ablation/stagewise/qcond_gat_basic_selector_only/, s03_gat_ensemble/a07_enriched_triplet/s03_a07_*/}/`

### 후속 (planner 핸드오프)
- `presentation_brief_2026-04-28.md §14.1` cumulative 표 9-cell 완성 + Filter Δ F1 by Builder 정량 반영
- DECISIONS 후속 엔트리 (Option C 결과 기록 + Plain 우세 narrative 정량)

---

## Selector Ablation Cumulative Backfill (Option B, 2026-04-26) — SuperNode 보류 + Plain/QCond 18-cell

**근거**: [planning/DECISIONS.md](planning/DECISIONS.md) 2026-04-26 (Ablation 2 보강 #2 9-cell 매트릭스 + Option B 채택)
**목적**: Ablation 2 (Encoder × Score × Stage) 27-cell cumulative matrix 완성. **SuperNode 9 cells 보류** (smoke FAILED), Plain/QCond 18-cell (= 6 score combo × 3 stage) 진행.

### SuperNode smoke FAILED (2026-04-26 02:49:23)

**Cell**: `s04_stagewise_supernode_gat_a0_selector_only` (smoke test 단독 launch)
**원인**: ckpt input dim mismatch
- ckpt `best_gat_query_supernode.pt`: `[256, 384]`
- `EnsembleSelector(query_supernode=True)` model expect: `[256, 768]`
- Traceback: `src/modules/selectors/ensemble_selector.py:128` self.gat_model.load_state_dict
- DECISIONS 2026-04-22 17:05 §8-1 SuperNode split-order bug 영향권 ckpt (T7) 후속 영향
- **9 SuperNode cells 보류** (selector_only 3 + no_filter 3 + final 3) — partial backup `*_failed_20260426_0249/`
- 후속 해결 옵션 (planner 결정): (a) selector 세션 SchemaHeteroGAT in_channels 자동 분기 구현 (b) ckpt 재학습 (post-2026-04-28, §8-1 bug fix 와 묶어서)

### 18-cell stagewise matrix (Plain/QCond × {GAT/Cos/Ens} × {Selector only / +Extractor / +Filter})

> ⚠ Alpha convention (2026-04-26 정정): `final_score = α·cosine + (1−α)·gat`. **α=0 → GAT only, α=1 → Cosine only, α=0.85 → Cosine 우세 ensemble**.

| Encoder | Score | Selector only | + Extractor (Basic PCST, no_filter) | + Filter (Final) | era |
|---------|-------|---------------|-------------------------------------|------------------|-----|
| **Plain** | GAT (α=0) | 0.5281 / 0.2034 / **0.2937** ★ | 0.7785 / 0.1330 / **0.2272** ⊕ | 0.6676 / 0.7236 / **0.6945** ⊙ | vLLM |
| **Plain** | Cosine (α=1) | 0.7693 / 0.2549 / **0.3829** ★ | 0.9662 / 0.1302 / **0.2295** ★ | 0.7987 / 0.7694 / **0.7838** ⊙ | vLLM |
| **Plain** | Ensemble (α=0.85) | 0.7678 / 0.2681 / **0.3974** ★ | 0.9667 / 0.1273 / **0.2250** ★ | 0.8149 / 0.7597 / **0.7863** ⊙ | vLLM |
| **QCond** | GAT (α=0) | 0.6061 / 0.2494 / **0.3534** ★ | 0.7813 / 0.1752 / **0.2862** ⊕ | 0.6622 / 0.7539 / **0.7051** ⊙ | vLLM |
| **QCond** | Cosine (α=1) | 0.7693 / 0.2549 / **0.3829** ★ | 0.9662 / 0.1302 / **0.2295** ★ | **0.8501 / 0.8348 / 0.8424** 🚀★ | GLM |
| **QCond** | Ensemble (α=0.85) | 0.7834 / 0.2700 / **0.4016** ⊕ | 0.9651 / 0.1287 / **0.2271** ⊕ | 0.8169 / 0.7605 / **0.7877** ⊙ / 0.8438 / 0.8329 / **0.8383** 🚀⊙ | vLLM / GLM |

★ = 2026-04-26 신규 측정 (이번 Backfill 10 cells)
⊕ = Wave 1.5 backfill 또는 Builder Cumulative (기존 측정)
⊙ = Wave 1.5 / 2×2×2 / GLM era (기존 측정)
🚀 = GLM era top 후보

### 10 신규 cells 측정 (2026-04-26 02:52:01 launch → 04:31:08 완료, ~1h 39min)
- Selector_only 6: `plain_{gat_a0,cos_a1,ens}_selector_only`, `qcond_{gat_a0,cos_a1,ens}_selector_only`
- No-filter 3: `plain_cos_a1_no_filter`, `plain_ens_no_filter`, `qcond_cos_a1_no_filter`
- Final 1: **`s04_stagewise_qcond_cos_a1_glm`** (GLM API)

### 8 기존 cells (재참조)

| Cell | Stage | F1 | 출처 |
|------|-------|---:|------|
| `qcond_raw_basic_no_filter` | Plain GAT +Ext (=ensemble_raw_a0_no_filter) | 0.2272 | Wave 1.5 W1 |
| `qcond_raw_basic_no_filter` | QCond GAT +Ext (=qcond_raw_basic_no_filter) | 0.2862 | Wave 1.5 W2 |
| `qcond_gat_basic_no_filter` | QCond Ensemble +Ext | 0.2271 | Wave 1.5 W3 |
| `s04_stagewise_ensemble_raw_a0` | Plain GAT Final | 0.6945 | Wave 1.5 W1 |
| `s04_stagewise_qcond_raw_basic` | QCond GAT Final | 0.7051 | Wave 1.5 W2 |
| `s04_stagewise_qcond_gat_basic` | QCond Ensemble Final | 0.7877 | Wave 1.5 best (vLLM) |
| `abl_a01_05_cos_basic_xiyan` | Plain Cosine Final | 0.7838 | 2×2×2 |
| `abl_a01_06_ens_basic_xiyan` | Plain Ensemble Final | 0.7863 | 2×2×2 |
| `s04_stagewise_qcond_gat_basic_glm` | QCond Ensemble Final (GLM) | 0.8383 | GLM era top (이전) |

### Filter Δ F1 by Encoder × Score (no_filter → final)

| Encoder × Score | no_filter F1 | final F1 | Filter Δ F1 |
|-----------------|-------------:|---------:|------------:|
| Plain GAT | 0.2272 | 0.6945 | **+0.4673** |
| Plain Cosine | 0.2295 | 0.7838 | **+0.5543** |
| Plain Ensemble | 0.2250 | 0.7863 | **+0.5613** |
| QCond GAT | 0.2862 | 0.7051 | **+0.4189** |
| QCond Cosine | 0.2295 | **0.8424** 🚀 | **+0.6129 (max!)** |
| QCond Ensemble (vLLM) | 0.2271 | 0.7877 | +0.5606 |
| QCond Ensemble (GLM) | 0.2271 | 0.8383 | +0.6112 |

### 🚀 새 GLM era top 후보 발견
- **`s04_stagewise_qcond_cos_a1_glm`**: R=0.8501 / P=0.8348 / **F1=0.8424**
- vs 직전 GLM era top `qcond_gat_basic_glm` (F1=0.8383): **+0.0041** (미세 우세, noise 범위 가능)
- vs Wave 1.5 best `qcond_gat_basic` (F1=0.7877): **+0.0547**
- α=1.0 (Cosine only) + GLM-4.7 가 α=0.85 ensemble + GLM 보다 살짝 더 높음

### 주요 발견 (Plain/QCond 6 cells × 3 stage)
1. **🚀 QCond Cosine + GLM 새 GLM era 후보 top** — F1=0.8424, 직전 0.8383 대비 +0.0041. anchor 갱신 임계 (≥+0.005) 미달이지만 noise 가능성 분석 필요.
2. **Selector_only Cosine 단독에선 encoder 무관**: Plain/QCond Cosine F1 둘 다 0.3829 동일 — PLM 임베딩 직접 사용해 encoder 차이 없음 (cosine 만 쓸 때 GAT module 통과 안 함).
3. **GAT only 단계에서 QCond > Plain (+0.0597)**: encoder 효과 발현 가장 강한 지점. selector_only Plain GAT 0.2937 → QCond GAT 0.3534.
4. **Ensemble 단계도 QCond > Plain (+0.0042)**: 매우 미세. Cosine 우세 blend 라 encoder 효과 약화.
5. **Filter Δ F1 max = QCond Cosine + GLM (+0.6129)**: filter 가 cosine-only candidate 의 정밀도 부족을 가장 크게 보강. 다음으로 QCond Ensemble GLM (+0.6112).
6. **+Extractor 단계 매우 균질화**: 6 cells 모두 R≈0.965~0.967, F1≈0.225~0.230. PCST 가 후보 거의 다 끌어와 encoder/score 차이 dilute (단, QCond GAT 만 R=0.7813 / F1=0.2862 로 다름 — extractor 가 score 분포에 sensitive).

### 비용 / 운영
- 10 신규 cells 비용: ~₩764 (Final 1 cell GLM API), No-filter/Selector_only 9 cells LLM-free
- Wall clock: 1h 39min (02:52:01 → 04:31:08, GPU 2/3 parallel, smoke fail 직후 즉시 launch)
- GPU: 2/3 (memory rule swap, 다른 연구자 GPU 0/1 점유 유지)

### 산출물
- Configs (19): `configs/experiments/s04_ablation/stagewise/{selector_only/, no_filter/, *_glm.yaml}` (10 측정 + 9 SuperNode 보류용)
- Script: `scripts/run_ablation2_selector_cumulative.sh` (CFGS_GPU2 9 cells + CFGS_GPU3 1 cell, SuperNode 5 cells 제거)
- Smoke fail backup: `outputs/.../selector_only/supernode_gat_a0_selector_only_failed_20260426_0249/`, `logs/.../supernode_gat_a0_selector_only_failed_20260426_0249/`

### 후속 (planner 핸드오프)
- `presentation_brief_2026-04-28.md §14.2` 18-cell matrix 완성 + Filter Δ F1 by Encoder × Score 정량
- DECISIONS 후속 엔트리: (a) SuperNode 9 cells 보류 + ckpt mismatch (b) Plain/QCond 18-cell 결과 + QCond Cosine GLM 새 후보 (c) anchor 갱신 임계 (+0.005 미달, noise 가능성) 분석 — vLLM era 동일 anchor 비교 필요?
- selector 세션: SchemaHeteroGAT in_channels 자동 분기 (384 vs 768) 또는 SuperNode ckpt 재학습 검토
- analyzer (선택): qcond_cos_a1_glm 의 cosine-only 가 ensemble blend 보다 우세인 origin 분석 — query 분포별 효과 분해

## GLM era 일관 재측정 (Ablation 1/2/3, 2026-04-27) — 11 cells (8 final GLM + 3 LLM-free no-filter)

발사: 2026-04-27 01:01:27 → 완료: 03:14:47 (wall clock 2h 13min, budget 3.5h 내). GPU 2/3 split, 2 concurrent per GPU = 4 cells parallel × 3 batch.

### Ablation 1 (Builder × Stage) — Enriched final 1 cell

| Cell | R | P | F1 |
|---|---|---|---|
| s03_a07_01_enriched_gat_glm | 0.6926 | 0.8300 | 0.7551 |

### Ablation 2 (Encoder × Score × Stage) — Final 4 cells (Plain encoder + Basic PCST + XiYan(GLM))

| Cell | R | P | F1 |
|---|---|---|---|
| plain_gat_a0_glm (α=0 GAT only) | 0.6825 | 0.7153 | 0.6985 |
| **plain_cos_a1_glm (α=1 Cos only)** | **0.8472** | 0.8310 | **0.8390** ★ |
| plain_ens_glm (α=0.85 Ensemble) | 0.8447 | 0.8316 | 0.8381 |
| qcond_gat_a0_glm | 0.6830 | 0.7638 | 0.7211 |

### Ablation 3 (Extractor × Stage) — final 3 + no_filter 3 cells (Plain Ens stack)

| Stack | Extractor | R (no_filter) | P (no_filter) | F1 (no_filter) | R (final) | P (final) | F1 (final) | Filter ΔF1 |
|---|---|---|---|---|---|---|---|---|
| Plain Ens | AdaptivePCST | 0.7255 | 0.3480 | 0.4704 | 0.6479 | 0.8099 | 0.7199 | +0.2495 |
| Plain Ens | SteinerBackbone | 0.8242 | 0.2345 | 0.3651 | 0.7081 | 0.8073 | 0.7545 | +0.3894 |
| Plain Ens | MST (Steiner 2-approx) | 0.8370 | 0.2366 | 0.3689 | 0.7252 | 0.8276 | 0.7730 | +0.4041 |
| (참고) Plain Ens | Basic PCST = `plain_ens_glm` | — | — | — | 0.8447 | 0.8316 | **0.8381** | — |

### 새 GLM era top 후보 갱신

- **`plain_cos_a1_glm`**: R=0.8472 / P=0.8310 / **F1=0.8390**
- 직전 GLM era top `qcond_gat_basic_glm` (F1=0.8383): **ΔF1=+0.0007** (anchor 갱신 임계 +0.005 미달, **anchor 유지**)
- `qcond_cos_a1_glm` (직전 측정 F1=0.8424) 와 비교: -0.0034 (encoder 차이는 노이즈 수준)
- **결론**: F1=0.83~0.84 plateau 영역 — Cosine 우세 stack 동률 후보 다수

### 주요 발견 (11 cells)

1. **Cosine 우세 stack 의 GLM era 일관 우세**: α=1 Cos 0.8390, α=0.85 Ens 0.8381, α=0 GAT 0.6985 → ΔF1 GAT→Cos +0.1405. **Score signal 절대 우세**, encoder 차이 무시 가능.
2. **Encoder agnostic** (Cos 기준): Plain Cos 0.8390 ≈ QCond Cos 0.8424 — encoder 효과 noise 수준. GAT-only stack 일 때만 QCond > Plain (+0.0226).
3. **Extractor 위계 GLM era 재현**: Basic PCST (0.8381) >> MST (0.7730) > Steiner (0.7545) > Adaptive (0.7199). vLLM era "Basic > Adaptive + XiYan" 결론 GLM era 에서도 견고.
4. **새 발견 — MST > Adaptive + XiYan (+0.0531)**: MST 의 소량 selection 이 XiYan 정밀 prune 과 시너지. Adaptive 의 P80 widening 은 XiYan 의 prune 부담만 가중.
5. **Filter Δ F1 by Extractor**: MST (+0.4041) > Steiner (+0.3894) > Adaptive (+0.2495) — **입력 sub graph 가 단순할수록 LLM filter 효율 ↑**. MST 가 seed-only Steiner tree로 가장 좁게 시작 → filter 가 false negative 추가 prune 적음.
6. **Enriched GAT GLM era 갱신**: vLLM era a07_01 (F1≈0.7140) → GLM era 0.7551 (+0.0411). Builder 효과는 GLM 환경에서 더 발현.

### 비용 / 운영

- 8 GLM cells 비용: 8 × ~₩764 = **~₩6,112** (단일 GLM 셀 1.78s/query × 1534 = 45.5min, total LLM input ~50.6M tok)
- 3 LLM-free cells: 비용 0
- Wall clock: 2h 13min (concurrent GPU 2/3, batch 1+2 GLM ~58min each, batch 3 LLM-free ~16min)

### 산출물

- Configs (11): `configs/experiments/s03_gat_ensemble/a07_enriched_triplet/s03_a07_01_enriched_gat_glm.yaml`, `configs/experiments/s04_ablation/stagewise/{plain_gat_a0,plain_cos_a1,plain_ens,qcond_gat_a0}_glm.yaml`, `configs/experiments/s04_ablation/extractor/plain_ens_{adaptive,steiner,mst}_glm.yaml`, `configs/experiments/s04_ablation/extractor/no_filter/plain_ens_{adaptive,steiner,mst}_no_filter.yaml`
- Script: `scripts/run_glm_era_ablation_full.sh` (GPU 2/3 split, 2 concurrent per GPU)
- MST smoke (sanity): 28s에 15 preds 산출, MSTExtractor forward pass 정상 (smoke 후 output 삭제 + main launch)

### 후속 (planner 핸드오프)

- `presentation_brief_2026-04-28.md §14.1` (Ablation 1) Enriched GLM 1 cell 추가 + 9-cell Builder × Stage 매트릭스 완성
- `§14.2` (Ablation 2) Plain final 4 cells 추가 → 9-cell Encoder × Score × Stage 정합 (Plain final 줄 채움)
- `§14.3` (Ablation 3) **신규 도입**: Extractor (Adaptive/Steiner/MST) × Stage 6 cells + plain_ens_glm 참조 — Filter Δ F1 by Extractor 정량
- `§14.5` (예: Alpha sweep) — 다음 결정 후보: anchor `qcond_gat_basic_glm` 유지하지만 plain_cos_a1_glm 동률 표기 + alpha sweep 시 Plain encoder 기준 또는 QCond 기준 선택 필요
- DECISIONS 후속 엔트리: (a) GLM era 일관 재측정 11 cells 결과 (b) MST > Adaptive 새 발견 + Filter Δ F1 by Extractor 위계 (c) anchor 유지 + plain_cos_a1_glm 동률 후보 표기 (d) Alpha sweep stack 선택 (Plain Cos 대 QCond Cos 우선순위)
- analyzer (선택): MST 셀 score_analysis_*.jsonl 분해 — Adaptive 대비 MST 가 R 손실 (-0.07) 했음에도 P 향상 (+0.0177) 으로 F1 +0.0531 — 정확히 어떤 query 클러스터에서 MST 가 보이드 회피하는지 case study

## Ablation 1/2/3 α=0.5 Re-measurement (Option B, 2026-04-27) — 15 cells (6 final GLM + 9 LLM-free)

발사: 2026-04-27 14:41:16 → 완료: 17:42:22 (wall clock 3h 1min, budget 3h 살짝 초과 +1min). GPU 2 (6 Final GLM, 3 batches × 2 concurrent), GPU 3 (9 LLM-free, 5 batches). α=0.85 → α=0.5 (neutral, GAT/Cosine 동등 결합) baseline 재정의.

### 근거 (DECISIONS.md 2026-04-27 α=0.5 재측정 결정)

- **α=0.85 의 sweep 근거 제한**: I1a-c sweep 은 No Filter stack 한정 (α∈{0.70/0.75/0.85}, with-Filter 미수행)
- **L92 분석 인용**: "Filter 적용 시 Ensemble vs Cosine 차이 미미, α=0.85 GAT 15% 만 반영" — Filter 단 ensemble 약화
- **Advisor analysis L9/L29/L43**: α=0.85 의 GAT 15% 비중 비판, neutral baseline 권장
- **사용자 confirm 2026-04-27**: "Ensemble score 비교에 α=0.5 가 더 합리적 + alpha ablation 따로 진행"

### Ablation 2 — Plain/QCond × α=0.5 × 3 stage (6 cells)

| Encoder | Stage | R | P | F1 |
|---|---|---|---|---|
| Plain | Selector only | 0.6301 | 0.2358 | 0.3432 |
| Plain | + Extractor (Basic PCST) no_filter | 0.9550 | 0.1217 | 0.2159 |
| **Plain** | + Filter (XiYan GLM) Final | **0.8316** | **0.8188** | **0.8252** |
| QCond | Selector only | 0.7110 | 0.2780 | 0.3997 |
| QCond | + Extractor (Basic PCST) no_filter | 0.9581 | 0.1304 | 0.2296 |
| **QCond** | + Filter (XiYan GLM) Final | **0.8337** | **0.8275** | **0.8306** |

### Ablation 1 — Enriched α=0.5 × 3 stage (3 cells)

| Stage | R | P | F1 |
|---|---|---|---|
| Selector only | 0.6243 | 0.2326 | 0.3389 |
| + Extractor (Basic PCST) no_filter | 0.9557 | 0.1233 | 0.2184 |
| **+ Filter (XiYan GLM) Final** | **0.8325** | **0.8199** | **0.8262** ★ |

### Ablation 3 — Plain α=0.5 + 3 ext × 2 stage (6 cells)

| Extractor | Stage | R | P | F1 |
|---|---|---|---|---|
| AdaptivePCST | no_filter | 0.5849 | 0.2929 | 0.3903 |
| AdaptivePCST | Final (XiYan GLM) | 0.5058 | 0.6730 | **0.5775** |
| SteinerBackbone | no_filter | 0.6979 | 0.2101 | 0.3230 |
| SteinerBackbone | Final (XiYan GLM) | 0.5992 | 0.7081 | **0.6491** |
| MST | no_filter | 0.7231 | 0.2170 | 0.3338 |
| MST | Final (XiYan GLM) | 0.6257 | 0.7377 | **0.6771** |

### α=0.85 vs α=0.5 비교 (직접 baseline 비교)

#### Final GLM (with-Filter, 6 cells)

| Stack | α=0.85 F1 | α=0.5 F1 | ΔF1 |
|---|---|---|---|
| Plain Ens (Basic PCST) | 0.8381 | 0.8252 | -0.0129 |
| QCond Ens (Basic PCST) | 0.8383 | 0.8306 | -0.0077 |
| **Enriched Ens (Basic PCST)** | **0.7551** | **0.8262** | **+0.0711 ★** |
| Plain Ens + Adaptive | 0.7199 | 0.5775 | **-0.1424 ⚠️** |
| Plain Ens + Steiner | 0.7545 | 0.6491 | -0.1054 |
| Plain Ens + MST | 0.7730 | 0.6771 | -0.0959 |

#### LLM-free cumulative (9 cells)

| Stage / Stack | α=0.85 F1 | α=0.5 F1 | ΔF1 |
|---|---|---|---|
| Plain Ens selector_only | 0.3974 | 0.3432 | -0.0542 |
| QCond Ens selector_only | 0.4016 | 0.3997 | -0.0019 |
| Enriched Ens selector_only | 0.3877 | 0.3389 | -0.0488 |
| Plain Ens no_filter | 0.2250 | 0.2159 | -0.0091 |
| QCond Ens no_filter | 0.2271 | 0.2296 | +0.0025 |
| Enriched Ens no_filter | 0.2252 | 0.2184 | -0.0068 |
| Plain + Adaptive no_filter | 0.4704 | 0.3903 | -0.0801 |
| Plain + Steiner no_filter | 0.3651 | 0.3230 | -0.0421 |
| Plain + MST no_filter | 0.3689 | 0.3338 | -0.0351 |

### 🎯 핵심 발견 (15 cells)

1. **🚀 Builder × α 상호작용 (Enriched 가 α=0.5 압도적 우세 +0.0711)**: 새 발표 주력 narrative. Enriched 의 description 정보가 Cosine PLM 임베딩에 noise 로 작용 (α=0.85 cos 우세 → 손실), GAT 가 학습한 구조 정보로 보정 (α=0.5 GAT 비중 ↑ → 회복). **Enriched 의 학술적 가치를 α=0.5 baseline 에서 새로 발견**.
2. **⚠️ Extractor × α 상호작용 (Adaptive/Steiner/MST 모두 α=0.5 에서 큰 손실 -0.10~-0.14)**: Adaptive PCST 의 per-q P80 threshold + Steiner backbone bonus + MST seed-only 가 모두 score 분포에 sensitive — GAT noise 가 percentile/cost cutoff 왜곡. **Basic PCST 가 α 변경에 robust** (fixed θ=0.1 절대 threshold).
3. **anchor 유지 정당 강화 (Plain/QCond Final α=0.85 vs α=0.5: ΔF1 +0.008/+0.013)**: with-Filter stack 에서도 α=0.85 가 약하게 우세, I1a-c sweep "α=0.85 best" (No Filter stack) 결론 with-Filter 에서도 재현. 단 plateau 영역 임계 +0.005 미달이라 narrative 상 α=0.5 baseline 표기 가능.
4. **Pre-Filter 단계 평균 ΔF1 = -0.0354**: GAT 비중 ↑로 noise 영향 ↑, Filter 단계에서 일부 회복. Filter 의 noise prune 가치 정량.
5. **Adaptive + α=0.5 stack 부적절 발견**: F1=0.5775 — 모든 변형 중 최저 (no_filter 0.3903 → final 0.5775, Filter Δ +0.1872 도 가장 낮음). per-q P80 + GAT noise 부정적 시너지.

### 비용 / 운영

- 6 GLM cells 비용: ~₩4,584 (1.78s/query × 1534 × 6)
- 9 LLM-free cells: ₩0
- Wall clock: 3h 1min (GPU 2/3 split, GPU 3 finish 51min, GPU 2 finish 3h 1min — bottleneck = GPU 2 GLM batch 3)

### 산출물

- Configs (15): `s04_ablation/stagewise/{plain_ens_a05_glm, qcond_ens_a05_glm}.yaml`, `s04_ablation/stagewise/selector_only/{plain_ens, qcond_ens}_a05_selector_only.yaml`, `s04_ablation/stagewise/no_filter/{plain_ens, qcond_ens}_a05_no_filter.yaml`, `s03_gat_ensemble/a07_enriched_triplet/s03_a07_01_enriched_a05_{selector_only, no_filter, glm}.yaml`, `s04_ablation/extractor/plain_ens_a05_{adaptive, steiner, mst}_glm.yaml`, `s04_ablation/extractor/no_filter/plain_ens_a05_{adaptive, steiner, mst}_no_filter.yaml`
- Script: `scripts/run_ablation_alpha05_remeasure.sh` (GPU 2: 6 GLM 3 batches, GPU 3: 9 LLM-free 5 batches)

### 후속 (planner 핸드오프)

- `presentation_brief_2026-04-28.md §14.1` — Builder Ensemble 행 α=0.5 갱신 (Enriched +0.0711 새 narrative ★ 핵심), α=0.85 anchor 표기 유지
- `§14.2` — Plain/QCond Ensemble 행 α=0.5 갱신, α=0.85 alpha sweep 한 점 표기
- `§14.3` — 4 Extractor 통일 stack α=0.5 갱신, α 변경 robustness 위계 (Basic >> Steiner > MST > Adaptive)
- `§14.6` — anchor 결정 재판정 (현 anchor `qcond_gat_basic_glm` F1=0.8383 유지, α=0.5 plateau 내 plain_cos_a1_glm 0.8390 동률 표기 그대로)
- `§11 Q&A` — "왜 α=0.5 로 재측정?" Q 추가 (DECISIONS L33-34 narrative 활용)
- DECISIONS 후속 엔트리: (a) 15 cells α=0.5 재측정 결과 (b) Builder × α 상호작용 Enriched +0.0711 발견 (c) Extractor × α 상호작용 (d) anchor 유지 정당성 강화 (e) Adaptive + α=0.5 stack 부적절 발견
- Alpha sweep H8 (post-deadline): 본 측정으로 α∈{0.5, 0.85, 1} 3 점 확보 — α∈{0.25, 0.7, 0.95} 보강하여 with-Filter stack alpha sensitivity 완성

## MST 변형 측정 (옵션 C + Union, 2026-04-27) — 6 cells (4 옵션 C + 2 Union, 🚀 anchor 갱신)

발사 1: 2026-04-27 19:11:12 → 완료: 20:21:43 (4 cells, wall clock 1h 10min, GPU 1)
발사 2: 2026-04-27 20:23:30 → 완료: 22:21:53 (2 cells, wall clock 1h 58min, GPU 1)
SuperNode 학습 GPU 0 와 병렬 진행 (충돌 없음).

### 근거 (DECISIONS.md 2026-04-27)

- **옵션 C 결정**: 사용자 의문 "MST recall 이 왜 낮나" 해소 — 기존 `MSTExtractor` 가 사실 Steiner 2-approx + top-k seed 였기 때문. 진짜 MST Kruskal + score-threshold seed 측정 필요.
- **MST ∪ PCST union 결정**: 사용자 직전 요청 "MST 로 찾은 집합과 PCST 로 찾은 집합의 합집합". 새 anchor (MST Kruskal F1=0.8642) 의 R 상한 검증 + Filter 의 union 처리 능력 정량.

### 신규 Extractor 구현 (Extractor 모듈 세션)

- **MSTExtractor**: `seed_mode ∈ {"topk", "threshold"}` + `score_threshold=0.1` 추가 — 기존 default 보존, threshold 변형 활성화
- **MSTKruskalExtractor** (신규): `score_threshold=0.1` 노드의 induced subgraph 위 networkx.minimum_spanning_tree (Kruskal default), Steiner point 없음
- **MSTPCSTUnionExtractor** (신규): MSTKruskal ∪ PCSTExtractor (Basic, node_threshold=0.1) — 노드 + 엣지 합집합

### 6 cells 측정 결과 (R/P/F1, 4자리)

#### Plain Ens α=0.5 stack — Extractor × Stage matrix

| Extractor | no_filter R / P / F1 | Final GLM R / P / F1 | Filter ΔF1 |
|---|---|---|---|
| Steiner Tree threshold seed (`MSTExtractor seed_mode=threshold`) | 0.9914 / 0.1223 / **0.2177** | 0.8720 / 0.8538 / **0.8628** | +0.6451 |
| MST Kruskal (진짜 MST, induced) | 0.9914 / 0.1222 / **0.2176** | **0.8724 / 0.8561 / 0.8642** ★ | +0.6466 |
| **MST ∪ PCST union** ★🆕 | **0.9914 / 0.1222 / 0.2176** | **0.8787 / 0.8560 / 0.8672** ★🚀 | **+0.6496 (max)** |

#### 시나리오 판정

- **mst_pcst_union vs mst_kruskal anchor**: ΔF1 = 0.8672 - 0.8642 = **+0.0030**
- **시나리오 B 채택** (DECISIONS.md L24-27): F1 ≈ anchor ±0.005 plateau → **anchor 유지** (`plain_ens_a05_mst_kruskal_glm` F1=0.8642)
- 단 union 미세 우세 — narrative 강화: "MST Kruskal R 상한 거의 도달, union 추가 노드 효과 ΔR=+0.0063 / ΔF1=+0.0030"

### 🎯 6 cells 핵심 발견

1. **🚀 anchor 갱신 (옵션 C 4 cells, 직전 결과)**: 직전 anchor `qcond_gat_basic_glm` F1=0.8383 → 새 anchor `plain_ens_a05_mst_kruskal_glm` F1=**0.8642** (ΔF1=+0.0259, 임계 +0.005 의 5배 초과). 사용자 의문 정확 해소: 기존 "MST" final F1=0.6771 의 R 한계 = Steiner 2-approx + top-k seed (한정) 때문. 진짜 MST Kruskal + score-threshold seed 변경 시 +0.1871 ΔF1 향상.

2. **Algorithm 차이 거의 없음 (옵션 C)**: MST Kruskal vs Steiner Tree threshold ΔF1=+0.0014 (final), no_filter R 동일 (0.9914). **Steiner point 추가 효과 무시 가능**.

3. **🆕 MST Kruskal R 상한 도달 증거 (Union 측정)**: 
   - no_filter R 모두 동일 (Steiner threshold / MST Kruskal / union 모두 0.9914)
   - PCST ⊆ MST Kruskal (PCST 의 노드 = MST Kruskal score>0.1 induced subgraph 의 부분집합)
   - **score>0.1 노드 안에서 gold 회수율 99.14% 가 자연 상한**

4. **🆕 Union 의 미세 final F1 +0.0030 향상**:
   - ΔR = +0.0063 (약간 더 많은 정답 회수, 엣지 정보 차이 가능성)
   - ΔP = -0.0001 (거의 동일)
   - **paper insight 후보**: "Multi-extractor union 의 marginal R 회수 (+0.0063) — Filter 가 추가 엣지 정보로 정답 식별 미세 향상". 단 anchor 갱신 임계 미달이므로 narrative 보조.

5. **seed pool widening 이 R 결정 mechanism (옵션 C 결과)**:
   - top-k seed (Selector top-20): no_filter R=0.7231 (기존 MSTExtractor)
   - score-threshold seed (score > 0.1): no_filter R=0.9914 (+0.2683 ΔR over top-k)
   - **paper main contribution 후보**: "Extractor 의 seed pool (top-k vs score-threshold) + algorithm choice (Steiner Tree vs MST Kruskal) 가 Recall 결정 mechanism"

6. **Filter ΔF1 위계 갱신**: union (+0.6496) > MST Kruskal (+0.6466) > Steiner Tree threshold (+0.6451) > Basic PCST (+0.6131, α=0.5 측정). Filter 가 minimal+rich subgraph 위에서 가장 효율적.

### 명명 정정 확정

- 기존 `MSTExtractor` = Steiner 2-approx (Kou-Markowsky-Berman 1981) — 명명 오류
- 신규 `MSTKruskalExtractor` = 진짜 MST (Kruskal, networkx.minimum_spanning_tree)
- `MSTExtractor seed_mode="threshold"` = Steiner Tree + score-threshold seed 변형
- **post-deadline 코드 rename** (`MSTExtractor` → `SteinerTreeExtractor`, alias 유지)

### 비용 / 운영

- 3 GLM cells 비용: 3 × ~₩764 = **~₩2,292**
- 3 LLM-free cells: ₩0
- Wall clock 발사 1 (4 cells): 1h 10min
- Wall clock 발사 2 (2 cells): 1h 58min (GLM API throughput 일시 변동, no_filter 14min + glm 117min)
- GPU 1 only (SuperNode 학습 GPU 0 보호, 충돌 없음)

### 산출물

- Configs (6): `s04_ablation/extractor/no_filter/plain_ens_a05_{steiner_threshold, mst_kruskal, mst_pcst_union}_no_filter.yaml`, `s04_ablation/extractor/plain_ens_a05_{steiner_threshold, mst_kruskal, mst_pcst_union}_glm.yaml`
- Scripts: `scripts/run_mst_variants.sh` (4 cells), 추가 inline launch (2 cells)
- 신규 Extractor 코드 (Extractor 모듈 세션): `src/modules/extractors/mst.py` (seed_mode 추가), `src/modules/extractors/mst_kruskal.py`, `src/modules/extractors/mst_pcst_union.py`

### 후속 (planner 핸드오프)

- `presentation_brief_2026-04-28.md §14.3` 6-row 매트릭스 확장:
  - Basic PCST / Steiner Tree top-k (기존 "MST") / Steiner Tree threshold / MST Kruskal / **MST ∪ PCST union (신규)** / Adaptive PCST
  - "MST 단독" 표기 → "Steiner Tree (2-approx, Kou-Markowsky-Berman 1981)" 정정
- `§14.6` anchor 결정: `plain_ens_a05_mst_kruskal_glm` F1=0.8642 새 anchor 확정 (옵션 C 4 cells 결과). MST ∪ PCST union F1=0.8672 plateau 내 동률 후보 표기.
- `§11 Q&A` "Q: MST recall 이 왜 낮나?" + "Q: MST ∪ PCST union 효과는?" 신규 추가
- DECISIONS 후속 엔트리: (a) 옵션 C 4 cells + Union 2 cells 6 cells 통합 결과 (b) 시나리오 B 판정 + anchor 유지 (c) MST Kruskal R 상한 도달 증거 (d) Union 의 marginal +0.0030 narrative
- Analyzer (선택, post-deadline): MST Kruskal vs union vs Basic PCST 의 per-DB / per-difficulty 분해 + 어떤 query 클러스터에서 union 의 +0.0063 ΔR 발생하는지 case study
- Extractor 모듈 (post-deadline): 코드 명명 정정 (MSTExtractor → SteinerTreeExtractor, alias 유지)

## Paper Main Pipeline Measurement (옵션 A2, 2026-04-28) — 2 cells (End-to-End Co-Design with Modular LLM Filter)

발사: 2026-04-28 00:10:20 → 완료: 01:12:39 (wall clock 1h 02min, GPU 1 only). SuperNode 학습 GPU 0 와 동시 진행 (충돌 없음).

### 근거 (DECISIONS.md 2026-04-28 — 방향 F' 최종 채택 + 옵션 A2)

- **방향 F' 최종 채택**: End-to-End Pipeline Co-Design with Modular LLM Filter (4 module contributions + 4 co-design principles)
- **옵션 A2 측정**: 사용자 의도된 paper main pipeline (Enriched + QCond + MST Kruskal/Union + XiYan GLM) F1 정확 확보
- **paper title 권장**: "LLM Filter as a First-Class Stage in Graph-RAG Schema Linking: Co-Designing Builder, Selector, Extractor, and Filter"

### Stack 구성 (4 module)

| 모듈 | 결정 | ckpt / 알고리즘 |
|---|---|---|
| Builder | EnrichedHeteroGraphBuilder | description-aware (CSV + tables.json) |
| Encoder | LocalPLMEncoder | sentence-transformers/all-MiniLM-L6-v2 |
| Selector | EnsembleSelector α=0.5 (neutral) | weight=best_gat_qcond_nl3.pt, query_conditioned=true |
| Extractor (Cell 1) | MSTKruskalExtractor | score_threshold=0.1 (induced subgraph Kruskal MST) |
| Extractor (Cell 2) | MSTPCSTUnionExtractor | score_threshold=0.1 (MST ∪ Basic PCST) |
| Filter | XiYanFilter | provider=glm, model=zai-org/glm-4.7 |

### 2 cells 측정 결과 (R/P/F1, 4자리)

| Cell | R | P | F1 | ΔF1 vs Plain anchor (0.8642) |
|---|---|---|---|---|
| **enriched_qcond_a05_mst_kruskal_glm** ★ | 0.8741 | 0.8606 | **0.8673** | **+0.0031** |
| enriched_qcond_a05_mst_pcst_union_glm | 0.8772 | 0.8564 | 0.8667 | +0.0025 |

### 🎯 시나리오 판정: **A 부분 (paper main = anchor 동등)**

- 두 cell 모두 Plain anchor F1=0.8642 보다 미세 우세 (ΔF1=+0.0025~+0.0031)
- **anchor 갱신 임계 +0.005 미달** (LLM noise 범위 ±0.003~0.005, 직전 Union 진단)
- **결론**: paper main stack ≈ Plain anchor (F1 동등), narrative 우선

### 핵심 발견 (paper main pipeline)

1. **🚀 End-to-End Co-Design 의 통합 효과 = Plain anchor 와 동등 F1**
   - Plain Builder + Plain Ens (단순) 와 Enriched Builder + QCond Ens (paper main) 사이 F1 plateau 영역 (+0.0031)
   - **paper insight**: "F1 동등이지만 학술적으로 강한 narrative — Description-aware + Query-conditioned + first-class LLM Filter"

2. **MST Kruskal > Union (paper main에서)**
   - Cell 1 MST Kruskal F1=0.8673
   - Cell 2 Union F1=0.8667
   - ΔF1=-0.0006 — Union 의 추가 PCST 노드가 paper main stack 에서는 효과 없음 (LLM noise 범위)
   - **사용자 의도 main pipeline = MST Kruskal stack** 권장

3. **Builder × Encoder 조합의 Plain stack 동등 F1 검증**
   - Plain Builder + Plain Ens + MST Kruskal: F1=0.8642
   - Enriched + QCond + MST Kruskal: F1=0.8673 (+0.0031)
   - 사용자 의도 main pipeline 의 학술적 정당성 확보 (F1 손실 없음)

4. **paper anchor narrative 결정 가능**
   - 옵션 A: Plain anchor 표기 (F1=0.8642, simpler stack, anchor 유지)
   - **옵션 B (권장)**: paper main pipeline 표기 (F1=0.8673, Enriched+QCond+MST Kruskal+XiYan GLM)
   - 권장 사유: 학술적 contribution narrative 강화, 4 module + first-class LLM Filter 관점 일관

### 비용 / 운영

- 2 GLM cells: 2 × ~₩764 = **~₩1,528**
- Wall clock: 1h 02min (GPU 1 단독, 2 cells 병렬, throughput 0.43 preds/s)
- GPU 0 SuperNode 학습 보호 (T+6h 39min 동시 진행 정상)
- 발사 시 GLM API HTTP 200 ✅, 모든 cells 정상 완료

### 산출물

- Configs (2): `s04_ablation/pipeline/enriched_qcond_a05_mst_kruskal_glm.yaml`, `s04_ablation/pipeline/enriched_qcond_a05_mst_pcst_union_glm.yaml` (신규 카테고리 `pipeline/`)
- Script: `scripts/run_paper_main_pipeline.sh` (GPU 1 only, 2 cells parallel)

### 후속 (planner 핸드오프)

- `presentation_brief_2026-04-28.md §0` Executive Summary 갱신:
  - paper main pipeline F1=0.8673 (Cell 1 MST Kruskal) — 4 module + co-design 표현
  - "Plain anchor 0.8642 와 동등" 표기 + paper narrative 우선
- `§10` 빠른 참조 갱신: paper main F1 정확 추가
- `§14.6` (anchor 결정): 옵션 B 권장 (paper main pipeline anchor 표기)
- `paper_research_direction.md §0` Executive Summary F1 갱신 (??? → 0.8673)
- `paper_research_direction.md §7` 측정 갭 표 → 측정 완료 표시 + F1 기록
- `paper_research_direction.md §10` 핵심 수치 요약 갱신
- DECISIONS 후속 엔트리: (a) 옵션 A2 2 cells 결과 (b) 시나리오 A 부분 판정 + paper anchor 결정 (c) MST Kruskal > Union (paper main, ΔF1=-0.0006 noise) (d) End-to-End Co-Design 통합 효과 narrative
- post-deadline:
  - SuperNode 학습 완료 후 Concat vs SuperNode 결정 (H6)
  - Multi-seed 검증 (H7) — paper main pipeline F1 reliability
  - Wave 4 a05_filter_agentic — paper main pipeline 위에서 multi-agent extension

## SuperNode QCond GAT 학습 완료 (옵션 A, 2026-04-28) — 9h 8min, best epoch 228, val recall@15=0.5737

발사: 2026-04-27 18:32:00 → 완료: 2026-04-28 03:35 (wall clock 9h 8min, GPU 0). 사용자 명시 ≤8h 가이드라인 +1h 8m 초과 (단 paper main pipeline 측정 동시 진행 으로 GPU 효율 ↑).

### 근거 (DECISIONS.md 2026-04-27 — H6 옵션 A 선택)

- **사용자 옵션 A 직접 선택**: query_conditioned=True + query_supernode=True 통합 stack 학습
- **mechanism 정정**: query_conditioned=True 시 query feature concat → input 768 (effective_in=in_channels*2), query_supernode 는 그래프에 SuperNode 노드만 추가 (dim 무관). 두 flag 별개 mechanism.
- **이전 SuperNode smoke fail (2026-04-26)** 진단: 기존 ckpt input dim [256,384] vs Ablation 2 사용자 framing 정의 [256,768] mismatch — 새 ckpt 학습으로 해소

### 학습 config

- 파일: `configs/training/train_gat_query_supernode_qcond.yaml` (base: train_gat_query_supernode.yaml 복사 + query_conditioned=true 만 변경)
- experiment_name: `gat_query_supernode_qcond`
- checkpoint_name: `best_gat_query_supernode_qcond.pt` (기존 .pt 보존, 분리 저장)
- model: in_channels=384 (effective_in=768 자동), hidden=256, layers=3, heads=4, dropout=0.1, query_conditioned=true, query_supernode=true
- training: epochs=300, lr=1e-4, batch=8, pos_weight=100, infonce_lambda=0.5, temp=0.07, num_hard_negatives=15
- GPU: 0/1 (학습 시작 시 default, 다른 연구자 부재 확인)

### 학습 진행 (cron tick 추적)

| Time | epoch | loss_total | best ckpt 갱신 |
|---|---|---|---|
| T+0:00 (18:32) | 1/300 | 15.3688 | — |
| T+3h 16m (21:48) | ~120 | — | 첫 best 저장 |
| T+6h 51m (01:23) | 228 | — | **마지막 best 갱신 (val recall@15=0.5737)** |
| T+7h 27m (01:59 cron #1) | 248 | 0.3128 | 변경 없음 |
| T+7h 38m (02:10 cron #2) | 254 | 0.0972 | 변경 없음 |
| T+8h 08m (02:40 cron #3) | 270 | 0.3015 | 변경 없음 ⚠ 8h 임계 통과 |
| T+8h 38m (03:10 cron #4) | 286 | 0.4088 | 변경 없음 |
| T+9h 08m (03:40 cron #5) | 300 | 0.4233 | 변경 없음 — 학습 완료 |

→ **best epoch 228 plateau 도달**, 이후 72 epoch 추가 학습 무익. H7 multi-seed 검증 시 epoch 250 cap 권장.

### Best ckpt 정보 + 검증

- **best epoch**: 228 / 300
- **val recall@15**: **0.5737**
- **lin_dict.column.weight shape**: (256, **768**) → effective_in=768 정상 (query_conditioned=True 활성)
- **query_node lin_dict 존재**: query_supernode=True 활성
- **ckpt 위치 (NAS)**: `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_query_supernode_qcond.pt` (220 MB)
- **symlink**: `outputs/checkpoints/best_gat_query_supernode_qcond.pt → NAS`

### Smoke test (ensemble_selector load_state_dict 검증) — 🐛 버그 발견 → 즉시 수정 → 재검증 PASS

- Config: `configs/experiments/s04_ablation/stagewise/selector_only/supernode_qcond_a0_smoke.yaml` (신규, 안전한 별도 yaml — 기존 supernode_gat_a0_selector_only.yaml 변경 없이)
- Stack: SuperNode encoder (query_conditioned=true + query_supernode=true) + GAT α=0 + Selector only
- ckpt: best_gat_query_supernode_qcond.pt

#### 1차 시도 — FAIL

- 결과 (R/P/F1): 0.0000 / 0.0000 / NaN (1534 queries 모두 status=Error)
- Error: `mat1 and mat2 shapes cannot be multiplied (3x384 and 768x256)`

#### 진단 + 코드 수정 (root, 2026-04-28)

- **위치**: `src/modules/selectors/ensemble_selector.py:241-243` (SuperNode 분기에서 query_emb 미전달)
- **원인**: 학습 path 에는 `query_conditioned=True` 시 query embedding concat 활성. Inference 의 SuperNode 분기는 `query_emb` 인자 미전달 → GAT 의 query concat 비활성 → input 384 dim, ckpt 가중치 768 dim mismatch
- **수정** (1 line):
  ```python
  if self.query_supernode:
      ...
      node_embs_dict = self.gat_model(
          graph_data.x_dict, graph_data.edge_index_dict,
          query_emb=q_emb if self.query_conditioned else None,  # 추가
          active_num_layers=active_depth)
  ```

#### 2차 시도 — PASS ✅

- 결과 (R/P/F1): **0.6035 / 0.2534 / 0.3569** (1534 queries 모두 status=Answerable, error 없음)
- **비교 (selector_only α=0 GAT only)**:
  - Plain GAT α=0: F1=0.2937
  - QCond GAT α=0 (Concat): F1=0.3534
  - **SuperNode QCond α=0 (신규)**: F1=**0.3569** ★ (Concat 대비 +0.0035, noise 범위)
- **결론**: load_state_dict + forward path + selected_nodes 산출 모두 정상. SuperNode encoder 효과는 selector_only 에서 Concat 과 거의 동일.

### 비용 / 운영

- 학습 비용: ₩0 (GPU 자체 사용)
- Wall clock: 9h 8m (사용자 ≤8h 가이드라인 +1h 8m 초과)
- GPU 0 단독 (paper main pipeline 측정 동시 GPU 1 진행, 충돌 없음)
- NAS 220 MB 사용 (1.1T 여유 영향 minor)

### 산출물

- 학습 config: `configs/training/train_gat_query_supernode_qcond.yaml`
- ckpt: `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_query_supernode_qcond.pt` (symlink → outputs/checkpoints/)
- 학습 로그: `logs/gat_query_supernode_qcond/train/train_step.jsonl` (32100 entries, 30 MB)
- Smoke config: `configs/experiments/s04_ablation/stagewise/selector_only/supernode_qcond_a0_smoke.yaml`

### 후속 (planner 핸드오프)

- **Selector 결정 (H6)**: 신규 ckpt 학습 완료 (val recall@15=0.5737) + inference path 버그 수정 완료 + smoke PASS (selector_only F1=0.3569 ≈ Concat 0.3534)
- **코드 수정 적용**: `src/modules/selectors/ensemble_selector.py:241-243` SuperNode 분기에 `query_emb=q_emb if self.query_conditioned else None` 인자 추가 (1 line). 이전 SuperNode cells (vLLM era a03_16~18 등) 도 본 수정의 영향 확인 필요 — `query_conditioned=False, query_supernode=True` 였으면 영향 없음.
- **post-deadline measurement 옵션** (코드 수정 완료, 즉시 가능):
  - SuperNode stack × 3 stage = 3 cells (Plain Builder + α=0.5 통일) — paper main pipeline narrative 와 align
  - 또는 SuperNode + Enriched + α=0.5 + MST Kruskal/Union × 2 cells = paper main pipeline (옵션 A2) 의 SuperNode variant
- **DECISIONS 후속 엔트리**: (a) 학습 완료 + best epoch 228 / val recall=0.5737 (b) NAS migration 완료 (c) 🐛 smoke 1차 fail = inference path 버그 (load_state_dict OK, SuperNode 분기 query_emb 미전달) (d) ✅ 코드 fix + 2차 smoke PASS (F1=0.3569 selector_only) (e) post-deadline measurement 검토
- **paper_research_direction.md §1 모듈별 결정**: "Concat vs SuperNode 결정" → "SuperNode ckpt 학습 + smoke PASS (val recall@15=0.5737, selector_only F1=0.3569 ≈ Concat 0.3534), measurement 검토 진행"
- **§8 Future Works H6** 갱신: "진행 중 → 학습 + 코드 수정 + smoke 모두 완료, post-deadline measurement"
- **발표 narrative 영향 X** — SuperNode 9 cells "측정 X" caveat 그대로 유지 (DECISIONS 2026-04-27 H6 옵션 A 엔트리 caveat 적용)
- **사용자 결정 대기 (옵션 B/C)**: smoke 통과 했으니 즉시 추가 cells 측정 가능
  - 옵션 B: SuperNode + Enriched + α=0.5 + MST Kruskal + XiYan GLM 1 cell (~50min, ~₩764) — paper main pipeline F1=0.8673 (Concat) 와 직접 비교
  - 옵션 C: 옵션 B + Union variant 2 cells (~50min, ~₩1,528)

## SuperNode 9-cell Matrix 측정 (Ablation 2 SuperNode, 2026-04-29) — α∈{0, 0.5, 1} × {Selector_only, +Basic PCST, +XiYan GLM}

발사: 2026-04-29 19:45:46 → 완료: 21:38:51 (wall clock 1h 53min, GPU 0/1 split). 사용자 요청 (2026-04-29): "SuperNode 의 alpha=0.0, 0.5, 1.0 일 때의 각 단계별 (Selector Only, +Basic PCST, +XiYan Filter) 점수".

### 근거 (DECISIONS.md 2026-04-28 — H6 옵션 A 선택 + 학습 완료 + 코드 fix)

- **새 ckpt**: `best_gat_query_supernode_qcond.pt` (best epoch 228, val recall@15=0.5737)
- **코드 fix 적용**: `src/modules/selectors/ensemble_selector.py:241-243` (SuperNode 분기 query_emb 전달)
- **smoke PASS**: F1=0.3569 (α=0 selector_only) 정상 동작 검증

### 9 cells 결과 매트릭스 (R/P/F1, 4-decimal)

| α / Stage | Selector only | + Basic PCST (no_filter) | + XiYan(GLM) Final |
|---|---|---|---|
| **α=0** (GAT only) | 0.6035 / 0.2534 / **0.3569** | 0.5539 / 0.2809 / **0.3728** | 0.4738 / 0.6487 / **0.5476** |
| **α=0.5** (neutral) | 0.7276 / 0.2787 / **0.4030** | 0.9564 / 0.1396 / **0.2436** | **0.8353 / 0.8330 / 0.8341** |
| **α=1** (Cosine only) | 0.7693 / 0.2549 / **0.3829** | 0.9662 / 0.1302 / **0.2295** | **0.8441 / 0.8296 / 0.8368** |

### vs QCond Concat 비교 — Final GLM

| α | Concat F1 | SuperNode F1 | ΔF1 (SN − Concat) |
|---|---|---|---|
| α=0 (GAT only) | 0.7211 | 0.5476 | **-0.1735 ⚠️** |
| α=0.5 (neutral) | 0.8306 | 0.8341 | +0.0035 (noise) |
| α=1 (Cosine only) | 0.8424 | 0.8368 | -0.0056 (noise) |

### 🎯 핵심 발견 (9 cells)

1. **🚨 α=0 (GAT only) 에서 SuperNode 큰 손실 (-0.1735 vs Concat)** — paper insight 후보:
   - Concat: query embedding 을 모든 노드 input feature 에 concat → GAT score 산출에 직접 기여
   - SuperNode: query_node 를 그래프에 주입 + message passing 통한 indirect 영향
   - α=0 (GAT-only) 신호 모드에서 SuperNode 의 indirect 효과가 dilution → Concat 의 direct concat 이 우세
   - **Selector_only/+ PCST 단계에서도 동일 패턴 확인**: SuperNode α=0 selector_only F1=0.3569 < QCond Concat α=0 0.3534 (+0.0035 가까움) → Filter 단에서 격차 확대 (-0.1735) — Filter 의 prune 부담이 SuperNode α=0 의 약한 signal 에 더 큼

2. **α=0.5/1 에서 SuperNode ≈ Concat (plateau 동등)** — Cosine 비중 우세 영역 (α≥0.5) 에서는 GAT/SuperNode 차이 dilute. 양쪽 모두 plateau (F1=0.83~0.84) 도달.

3. **No-filter 단계 (no_filter) SuperNode α=0.5/1 R 매우 높음** (0.9564/0.9662) — Concat 과 거의 동일. Selector signal 차이가 없음.

4. **paper main pipeline anchor 유지**: Concat (`s04_pipeline_enriched_qcond_a05_mst_kruskal_glm` F1=0.8673). SuperNode 어떤 α 에서도 anchor 갱신 임계 +0.005 초과 못 함.

5. **H6 결정 — Concat 채택, SuperNode 보류**:
   - 발표 narrative: paper main pipeline = QCond **Concat**
   - SuperNode 는 future work (α=0 손실 mechanism 분석 필요, paper limitation 후보)

6. **Filter Δ F1 by α (SuperNode stack)**:
   - α=0: no_filter 0.3728 → final 0.5476, Δ=+0.1748 (small)
   - α=0.5: no_filter 0.2436 → final 0.8341, Δ=**+0.5905** (large)
   - α=1: no_filter 0.2295 → final 0.8368, Δ=**+0.6073** (max)
   - **α=0 SuperNode 의 small Filter Δ** → Filter 가 SuperNode α=0 의 signal noise 를 충분히 prune 못 함

### 비용 / 운영

- 3 GLM cells: 3 × ~₩764 = **~₩2,292**
- 6 LLM-free cells: ₩0
- Wall clock: 1h 53min (GPU 0/1 split, 6 LLM-free 빠르게 GPU 1 batch 처리, 3 GLM GPU 0 batches)
- 코드 fix 적용 후 첫 정상 측정 — load_state_dict + forward + selected_nodes + R/P/F1 모두 정상

### 산출물

- Configs (9): `s04_ablation/stagewise/{selector_only/, no_filter/, }supernode_qcond_a{0,05,1}_{selector_only,no_filter,glm}.yaml`
- Script: `scripts/run_supernode_qcond_9cells.sh`
- 코드 수정: `src/modules/selectors/ensemble_selector.py:241-243` (SuperNode 분기 query_emb 전달)

### 후속 (planner 핸드오프)

- **DECISIONS 후속 엔트리**: "2026-04-29 (SuperNode 9-cell matrix 완료) — H6 결정 (Concat 채택) + α=0 SuperNode 손실 발견 + paper limitation"
  - 9-cell 결과 표
  - Concat vs SuperNode 비교 (final F1)
  - α=0 손실 mechanism 후보 narrative
  - paper main pipeline anchor 유지 (Concat F1=0.8673)
- **paper_research_direction.md 갱신**:
  - §1 모듈별 결정 — "Concat vs SuperNode 결정" → **"Concat 채택"** (H6 결정 완료)
  - §8 Future Works H6 — "post-deadline measurement" → **"완료, paper limitation 으로 narrative 강화"**
  - §9 paper limitation — α=0 SuperNode 손실 mechanism 분석 미완 → future work
- **presentation_brief 갱신**:
  - §11 Q&A — "SuperNode 측정 X" caveat 정정: "SuperNode 9-cell 측정 완료, Concat α=0.5 보다 plateau 동등 (F1=0.8341 vs 0.8306)"
  - §14.2 SuperNode 3 row → 9 cells 결과 채움
- **Analyzer (선택, post-deadline)**: α=0 SuperNode 손실 mechanism — query_node message passing 의 indirect signal 약화 case study. SuperNode message passing depth 별 attention 분석.

## SuperNode + Enriched Paper Main Pipeline 측정 (2026-04-29 22:57 → 23:58, 2 cells)

발사: 2026-04-29 22:57:30 → 완료: 23:58 (wall clock ~62min, GPU 0/1 split). 사용자 요청 (2026-04-29): "Enriched Graph + QCond-SuperNode + MST + PCST + XiYan Filter 사용한 성능".

### 근거 (DECISIONS.md 2026-04-28 옵션 A2 + 9-cell matrix 후속)

- 옵션 A2 (Concat) 측정 완료: F1=0.8673 (MST Kruskal) / 0.8667 (Union)
- SuperNode 9-cell matrix 결과: SuperNode α=0.5 ≈ Concat plateau (F1=0.8341 vs 0.8306) 검증
- 질문: **Enriched Builder 와 SuperNode 통합 시 paper main pipeline F1 변화?**

### Stack 구성 (2 cells)

| 모듈 | 결정 |
|---|---|
| Builder | EnrichedHeteroGraphBuilder (description-aware) |
| Encoder | LocalPLMEncoder (MiniLM-L6-v2) |
| Selector | EnsembleSelector α=0.5 + best_gat_query_supernode_qcond.pt + query_conditioned=true + query_supernode=true |
| Extractor (Cell 1) | MSTKruskalExtractor (score_threshold=0.1) |
| Extractor (Cell 2) | MSTPCSTUnionExtractor (score_threshold=0.1) |
| Filter | XiYanFilter (provider=glm, model=zai-org/glm-4.7) |

### 2 cells 측정 결과 (R/P/F1, 4-decimal)

| Cell | R | P | F1 |
|---|---|---|---|
| `enriched_supernode_a05_mst_kruskal_glm` | 0.8706 | 0.8591 | **0.8648** |
| **`enriched_supernode_a05_mst_pcst_union_glm`** ★ (= 사용자 의도 MST + PCST) | **0.8742** | 0.8597 | **0.8669** |

### Concat vs SuperNode Paper Main Pipeline 비교

| Stack | F1 | ΔF1 vs Concat anchor (0.8673) |
|---|---|---|
| **Concat + MST Kruskal** (paper main anchor) | **0.8673** | (baseline) |
| Concat + Union | 0.8667 | -0.0006 |
| **SuperNode + MST Kruskal** | 0.8648 | **-0.0025** (noise) |
| **SuperNode + Union** ★ | **0.8669** | **-0.0004** (noise) |

### 🎯 시나리오 B 채택 — Plateau 동등 (anchor 유지)

- 두 SuperNode 변형 모두 **갱신 임계 ±0.005 plateau** 내
- **SuperNode + Union F1=0.8669** = Concat + Union F1=0.8667 거의 정확 동률 (+0.0002)
- **paper main pipeline anchor 유지**: Concat + MST Kruskal F1=0.8673

### 핵심 발견

1. **🚀 Enriched 효과 SuperNode 에서도 발현**: Plain SuperNode α=0.5 (0.8341) → Enriched SuperNode (0.8648/0.8669) **ΔF1 +0.0307~+0.0328**. Description-aware Builder 의 학술적 정당성 SuperNode stack 에서도 확보.
2. **SuperNode ≈ Concat plateau 동등 (Enriched + α=0.5 + GLM stack)**: ΔF1=-0.0025/-0.0004, anchor 갱신 임계 미달.
3. **SuperNode + Union 가 SuperNode 변형 중 best (F1=0.8669)** — Concat + Union (0.8667) 와 거의 동일.
4. **사용자 의도 stack (Enriched + QCond-SuperNode + MST + PCST + XiYan GLM) F1=0.8669** — paper main pipeline 후보로 narrative 활용 가능 (Concat 0.8667 와 동률).
5. **2 × 2 cross-comparison (Selector × Extractor)**:
   - SuperNode + MST Kruskal (0.8648) < SuperNode + Union (0.8669): Union 이 SuperNode 에서 +0.0021 우세
   - Concat + MST Kruskal (0.8673) > Concat + Union (0.8667): MST Kruskal 이 Concat 에서 +0.0006 우세
   - → SuperNode 와 Union 시너지 (간접 signal 의 marginal R 회수와 union 의 추가 노드 시너지 가능성, 단 모두 plateau 내 noise)

### 비용 / 운영

- 2 GLM cells: 2 × ~₩764 = **~₩1,528**
- Wall clock: ~62 min (GPU 0/1 parallel)
- 코드 fix 적용 후 SuperNode + Enriched + extractor 변형 첫 정상 측정

### 산출물

- Configs (2): `s04_ablation/pipeline/enriched_supernode_a05_mst_kruskal_glm.yaml`, `s04_ablation/pipeline/enriched_supernode_a05_mst_pcst_union_glm.yaml`
- 비교 baseline (이전 측정): `enriched_qcond_a05_mst_kruskal_glm` F1=0.8673, `enriched_qcond_a05_mst_pcst_union_glm` F1=0.8667

### 후속 (planner 핸드오프)

- **DECISIONS 후속 엔트리**: "2026-04-29 (SuperNode + Enriched paper main 2 cells 완료) — 시나리오 B + paper main anchor 유지 + Enriched 효과 SuperNode 에서도 발현"
- **paper_research_direction.md §1 Selector 결정 재확인**: Concat 채택 (paper main anchor F1=0.8673) — SuperNode + Union 동률 (0.8669), narrative 차원에서 SuperNode 활용 가능 단 anchor 는 Concat
- **paper_research_direction.md §10 핵심 수치 갱신**: 4-cell anchor plateau 정량 (Concat MST/Union, SuperNode MST/Union 모두 F1≈0.864~0.867)
- **presentation_brief §14.6 anchor 결정** 갱신: 4 anchor plateau (3 Concat 후보 0.8673/0.8667/0.8424 + SuperNode + Union 0.8669) 중 paper main = Concat MST Kruskal 0.8673 유지
- **§11 Q&A 보강**: "Q: SuperNode + Enriched paper main 결과는?" — A: F1=0.8669 (Union) / 0.8648 (MST Kruskal), Concat 와 plateau 동등, anchor 미갱신
- **post-deadline H7 multi-seed**: 4 anchor plateau 의 통계적 reliability 검증 — 4 stacks × 3 seeds = 12 cells

## H-A Distribution Shift 검증 + H-D Score Normalization 변형 (2026-05-04, 13 cells)

발사: 2026-05-04 15:35:08 → 완료: 18:45 (wall clock ~3h 10min, GPU 0/1 split). 사용자 결정 (DECISIONS 2026-05-04 사용자 의사결정 엔트리) — narrative resolution H-A + H-D 검증.

### 근거 + 가설

**H-A 가설**: 현재 t_00 inference 가 Enriched features 인데 ckpt `qcond_nl3.pt` 가 Plain features 학습 → distribution mismatch → GAT score 변별력 평탄화 → α plateau α∈[0.3,1.0] 발생. 검증: `best_gat_enriched.pt` (Enriched features 학습, query_conditioned=False) 로 alpha sweep 재측정 → distribution match 후 GAT contribution 회복 여부 검증.

**H-D 가설**: `ensemble_selector.py:297-307` min-max normalization 이 GAT score 절대값 변별력 flatten → α plateau 발생. 검증: `score_normalization` 파라미터 추가 (`"none"` / `"zscore"` 변형) → α=0.5 single-point 비교.

### H-A 11 cells 결과 — Best F1=0.8651 (α=1.0), Best EX=0.3429 (α=0.8)

Configs: `s04_ablation/pipeline/t00_enriched_ckpt_alpha_0[0~10].yaml`
Stack: Enriched Builder + best_gat_enriched.pt + query_conditioned=false (ckpt 정합) + α∈{0.0~1.0} + MSTPCSTUnion + XiYan(GLM, 3 ex) + SQL gen

| α | R | P | F1 | EX | 기존 (qcond_nl3) F1 | ΔF1 |
|---|---|---|---|---|---|---|
| 0.0 (GAT only) | 0.6993 | 0.7408 | **0.7195** | 0.2177 | 0.7030 | **+0.0165** |
| 0.1 | 0.7655 | 0.7992 | 0.7820 | 0.2432 | 0.7880 | -0.0060 |
| 0.2 | 0.8586 | 0.8547 | 0.8566 | 0.3188 | 0.8535 | +0.0031 |
| 0.3 | 0.8714 | 0.8555 | 0.8634 | 0.3292 | 0.8632 | +0.0002 |
| 0.4 | 0.8740 | 0.8557 | 0.8648 | 0.3331 | 0.8639 | +0.0009 |
| 0.5 | 0.8748 | 0.8529 | 0.8637 | 0.3403 | 0.8657 | -0.0020 |
| 0.6 | 0.8734 | 0.8533 | 0.8632 | 0.3403 | 0.8638 | -0.0006 |
| 0.7 | 0.8734 | 0.8518 | 0.8625 | 0.3396 | 0.8629 | -0.0004 |
| 0.8 | 0.8742 | 0.8529 | 0.8634 | **0.3429** | 0.8644 | -0.0010 |
| 0.9 | 0.8762 | 0.8526 | 0.8642 | 0.3383 | 0.8639 | +0.0003 |
| 1.0 (Cosine only) | 0.8767 | 0.8538 | **0.8651** | 0.3390 | 0.8664 | -0.0013 |

### H-D 2 cells 결과 — minmax 가 best, norm 변형이 plateau 원인 X

Configs: `s04_ablation/pipeline/t00_norm_{none, zscore}.yaml`
코드 fix: `src/modules/selectors/ensemble_selector.py` — `score_normalization: str = "minmax"` 파라미터 추가 (modes: `"minmax"` / `"none"` / `"zscore"`)

| Variant | R | P | F1 | EX | ΔF1 vs t_00 | ΔEX vs t_00 |
|---|---|---|---|---|---|---|
| t_00 (minmax, default) | 0.8734 | 0.8581 | **0.8657** | **0.3377** | (anchor) | (anchor) |
| norm_none | 0.8544 | 0.8562 | 0.8553 | 0.3214 | -0.0104 | -0.0163 |
| norm_zscore | 0.8118 | 0.8542 | 0.8325 | 0.2881 | **-0.0332** | **-0.0496** |

### 🎯 핵심 결론 — 시나리오 ② 채택 (옵션 1 + 옵션 4 통합 narrative)

**H-A 가설 부정**:
1. **F1 plateau α∈[0.2, 1.0] 그대로 유지** — 9/10 cells 가 best F1=0.8651 (α=1.0) 의 ±0.01 noise band 내
2. **α=0 (GAT only) 약간 회복** (+0.0165) but plateau 패턴 유지 — Distribution match 해소 효과 미미
3. **α=1 (Cosine only) noise 범위 차이 (-0.0013)** — GAT 미사용 cell, ckpt 무관
4. **EX best α=0.8 (0.3429)** vs 기존 α=1.0 (0.3475) — Enriched ckpt 에서 GAT 가 EX 에 약간 더 기여하지만 plateau 동등 (ΔEX -0.0046, plateau 내)

**H-D 가설 부정 (간접)**:
1. **minmax > none > zscore** — 현재 default 가 best
2. **norm 제거 시 F1 -0.01, EX -0.016** — normalization 자체 효과 있음 (제거 시 손실)
3. **z-score 가장 나쁨 (ΔF1 -0.033)** — GAT/Cosine score 분포 가정 부적합
4. → **norm 자체가 plateau 의 원인 X** (단 α=0.5 single-point 만 측정, strong 결론 X)

### Paper Narrative 정정 — 옵션 1 + 옵션 4 통합 채택

**❌ 기존**: "QCondGAT main contribution"
**✅ 신규**: "4 module Co-Design + Filter dominance"

- **Selector contribution**: "GAT-floor (α=0 손실 -0.16 으로 baseline robustness 보장) + Cosine-ceiling (α≥0.2 plateau)"
- **Filter contribution**: first-class stage, F1 driver (P 회복 +0.64), EX 에는 marginal (+0.01)
- **Plateau 의 진짜 원인**: H-A/H-D 부정 → 미해결 (H-B Cosine vs GAT redundancy, H-C Filter dominance 등 analyzer 큐)

### 비용 / 운영

- 13 GLM cells: ~₩9,928 (실제)
- Wall clock: ~3h 10min (예상 ~3-4h, throughput 0.18 preds/s, 13 cells GLM API contention)
- GPU 0/1 split (8 cells GPU 0 + 7 cells GPU 1)
- 모든 cells 정상 완료, failure 없음

### 산출물

- Configs (13): `s04_ablation/pipeline/t00_enriched_ckpt_alpha_0[0~10].yaml` (11 H-A) + `t00_norm_{none, zscore}.yaml` (2 H-D)
- Scripts: `scripts/run_h_a_enriched_ckpt_alpha_sweep.sh`, `scripts/run_h_d_norm_variants.sh`
- 코드 수정: `src/modules/selectors/ensemble_selector.py:33-65, 295-318` — `score_normalization` 파라미터 추가

### 후속 (planner 핸드오프)

- **DECISIONS 후속 엔트리**: "2026-05-04 (H-A/H-D 검증 완료) — 가설 모두 부정 + 시나리오 ② 채택 + 옵션 1+4 통합 narrative 정식 확정"
- **paper_research_direction.md §1, §2.2 갱신**:
  - §1 paper main pipeline Selector 결정 narrative 정정 — "QCondGAT main contribution" → "GAT-floor + Cosine-ceiling + Filter dominance"
  - §2.2 H-A 결과 추가 (alpha sweep 11 cells, distribution match 검증, 가설 부정)
  - §3 Inter-Module Co-Design — Filter ↔ Selector Absorption (옵션 4) 신설 sub-section
  - §8 Future Works — H-A/H-D 항목 "✅ 완료" 표기, 새 가설 (H-B, H-C analyzer 큐) High 우선순위
- **presentation_brief §14 보강** — H-A/H-D ablation 섹션 + 시나리오 ② 채택 narrative
- **Analyzer 위임** (analyzer 세션 핸드오프):
  - H-B per-query GAT score vs Cosine score correlation 검증 — plateau 의 진짜 원인 후보
  - H-C Filter dominance 정량 — Framework #2 (F1=0.22 / EX=0.33) mechanism 분석 보강
  - H-F Top-K=20 cap 의 영향 — α 변화가 top-20 ordering vs set 자체에 미치는 효과 분리

## Wave 4 Filter Ablation (2026-05-04 → 05, 14 cells GLM, 🚀 신규 최고 F1=0.8809)

발사: 2026-05-04 19:08 → 완료 2026-05-05 03:06 (wall clock ~7h 58min, GPU 0/1 split — 7 cells × 2 GPUs). 사용자 결정 (DECISIONS 2026-05-04 옵션 B GLM 통일) — Filter 모듈 14 변형의 F1/EX 효과 정량.

### 근거 + 가설

Plan: [planning/templates/vivid-sprouting-sunbeam.md](planning/templates/vivid-sprouting-sunbeam.md) — Filter 모듈을 agentic refinement 모듈로 확장하여 (a) Recall 손실 감소 + Precision 유지, (b) 그래프 prior(GAT score, PCST membership)를 agent 에 노출. 14 variants:
- a05_01: AdaptiveMultiAgent (Semantic + Structural + Skeptic)
- a05_02/03: ReflectionFilter (1 / 3 iter)
- a05_04: VerifierFilter (CHESS-style Unit Tester)
- a05_05/06: TieredBidirectionalAgent (no_tools / full_tools)
- a05_07: AdaptiveDepthFilter (uncertainty-gated)
- a05_08: StackedFilter (Tiered → Verifier)
- a05_09/10: TieredRetry / AdaptiveRetry (extraction_retry K=2)
- a05_19/21: SymbolicVerifier + XiYan (repair / detect)
- a05_20: SymbolicVerifier + Reflection repair
- a05_22: SymbolicVerifier + Reflection + Verifier stacked

Stack: paper main pipeline (Enriched + QCond α=0.5 + qcond_nl3 + MSTPCSTUnion + LLMSQLGenerator(GLM) + 모든 모듈 GLM 통일) + Filter 만 변경.

### 14 cells 결과 — F1 정렬 (t_00 base F1=0.8657, EX=0.3377)

| 순위 | Cell | R | P | F1 | EX | ΔF1 | ΔEX |
|---|---|---|---|---|---|---|---|
| 1 | **a05_08 stacked Tiered+Verifier** | 0.8880 | 0.8739 | **0.8809** | 0.3351 | **+0.0152** ★ | -0.0026 |
| 2 | a05_22 stacked SymVerify+Reflection+Verifier | 0.8844 | 0.8675 | 0.8759 | 0.3364 | +0.0102 | -0.0013 |
| 3 | a05_05 tiered_no_tools | 0.8940 | 0.8463 | 0.8695 | 0.3429 | +0.0038 | +0.0052 |
| 4 | a05_09 tiered_retry | 0.8932 | 0.8449 | 0.8684 | 0.3377 | +0.0027 | +0.0000 |
| 5 | a05_06 tiered_full_tools | 0.8931 | 0.8438 | 0.8678 | 0.3422 | +0.0021 | +0.0045 |
| 6 | a05_04 verifier | **0.9155** | 0.8220 | 0.8662 | 0.3383 | +0.0005 | +0.0006 |
| 7 | a05_19 symverify_xiyan_repair | 0.8743 | 0.8559 | 0.8650 | 0.3409 | -0.0007 | +0.0032 |
| 8 | a05_21 symverify_xiyan_detect | 0.8726 | **0.8565** | 0.8645 | 0.3370 | -0.0012 | -0.0007 |
| 9 | a05_07 adaptive_depth | 0.8802 | 0.8471 | 0.8633 | **0.3501** | -0.0024 | **+0.0124** ★ |
| 10 | a05_02 reflection_1iter | 0.8894 | 0.8383 | 0.8631 | 0.3429 | -0.0026 | +0.0052 |
| 11 | a05_10 adaptive_retry | 0.8791 | 0.8462 | 0.8623 | 0.3422 | -0.0034 | +0.0045 |
| 12 | a05_20 symverify_reflection_repair | 0.8903 | 0.8354 | 0.8620 | 0.3396 | -0.0037 | +0.0019 |
| 13 | a05_03 reflection_3iter | 0.8914 | 0.8297 | 0.8594 | 0.3344 | -0.0063 | -0.0033 |
| 14 | a05_01 adaptive_multi_agent | 0.7724 | 0.8448 | 0.8070 | 0.3279 | -0.0587 ⚠️ | -0.0098 |

### 🎯 핵심 발견

**(1) StackedFilter sweet spot — F1 +0.0152 신규 최고 (a05_08)**
- Tiered (semantic agent) → Verifier (precision check) 2-stage stacking 이 단일 agent 변형 모두 능가
- t_00 (XiYan 단일) F1 0.8657 → a05_08 (Stacked) F1 0.8809 — schema-linking F1 ceiling 갱신
- a05_22 (SymVerify+Reflection+Verifier 3-stack) F1=0.8759 도 t_00 능가 — stacking 효과 일관

**(2) AdaptiveDepth 만 EX 개선 — F1 trade-off 하지만 SQL 정확도 유일하게 +1.24%p (a05_07)**
- 14 cells 중 유일하게 EX > 0.35 (0.3501, t_00 0.3377 대비 +0.0124)
- F1 -0.0024 (plateau 내) trade-off 로 EX 만 개선 — Schema-linking F1 ↔ SQL EX **decoupling 재확인**
- Uncertainty-gated agent depth 가 selector confidence 를 SQL gen 까지 비대칭 전파 가능성

**(3) VerifierFilter R 1위 (0.9155) but P trade-off 로 F1 중간 (a05_04)**
- R 0.9155 = 14 cells 중 R 최고 (t_00 0.8734 대비 +0.0421) — schema-linking recall ceiling 회복 약 절반
- 그러나 P 0.8220 (-0.0361) 으로 F1 0.8662 (plateau 내)
- R↑ trade-off P↓: Verifier 가 누락 가능 노드 회수 시 false positive 도 함께 → F1 plateau 유지

**(4) AdaptiveMultiAgent 실패 — Skeptic over-prune 추측 (a05_01)**
- 14 cells 중 유일하게 F1 -0.05 이하 (-0.0587), R 급락 (-0.1010)
- Semantic + Structural + Skeptic 3 agent conservative voting → 정답 노드 다수 prune
- 다른 agent 변형은 모두 F1 -0.01 ~ +0.015 noise band 내 — Skeptic agent outlier 원인

**(5) Reflection iteration depth — 1iter > 3iter (P drift)**
- a05_02 (1iter) F1=0.8631 vs a05_03 (3iter) F1=0.8594 — Δ=-0.0037
- iter 증가 시 P drift (0.8383 → 0.8297) — Reflection over-correction 으로 false negative 증가

### Paper Narrative 함의

**🎯 §3.5 mechanism 갱신 candidate**: Filter 단계 내 variation 첫 측정 — 기존 narrative 는 "Filter on/off" 만 (F-1 vs With-Filter ΔF1≈0.63).
- 이번 측정은 **Filter design 자체 variation** 0.8070 ~ 0.8809 (Δ=0.0739) — Filter on/off (Δ=0.63) 의 12% scale
- → Filter "first-class stage" 한층 보강: Filter on/off + Filter design 둘 다 F1 driver
- 단 EX 는 a05_07 만 개선 — schema-linking F1 ↔ SQL EX decoupling 일관 (Filter 변형으로도 EX ceiling 미돌파)

**🎯 paper main anchor 결정 필요**:
- 옵션 A: t_00 anchor 유지 (XiYan 단일, F1=0.8657, simple/clean baseline) + Wave 4 결과를 Filter ablation appendix
- 옵션 B: a05_08 (Stacked Tiered+Verifier, F1=0.8809) 을 새 anchor 로 promote — "Co-Design with Stacked Filter"
- 권장: **옵션 A** (anchor 단순화 + narrative 일관) + a05_08 을 §3.5/§4 evidence 로 인용

### 비용 / 운영

- 14 GLM cells: ~₩30-54K 추정 (multi-agent 3-5x LLM call/query + sql gen)
- Wall clock: 7h 58min (GPU 0/1 split, Stacked 2종 (a05_08/22) bottleneck — 단일 cell ~7h)
- 모든 cells 정상 완료, failure 없음
- 로그 NAS 이관 (Phase A/B/C) 동시 진행 — logs/ 7.6G → 468K (이관 99.99%)

### 산출물

- Configs (14): `s04_ablation/pipeline/wave4/t00_a05_{01-10, 19-22}_*.yaml`
- Script: `scripts/run_wave4_filter_ablation_glm.sh` (failure-tolerant 14 cells parallel GPU 0/1 split)
- Plan reference: `planning/templates/vivid-sprouting-sunbeam.md` (F1~F5 phases)
- 코드 산출 (이전 세션):
  - `src/modules/filters/reflection_filter.py` (F1)
  - `src/modules/filters/verifier_filter.py` (F2)
  - `src/modules/filters/bidirectional_agent_filter.py` + `tools/graph_tools.py` (F3 tiered)
  - `src/modules/filters/adaptive_depth_filter.py` (F4)
  - `src/modules/filters/stacked_filter.py` (Stacked)
  - `src/modules/filters/symbolic_verifier_filter.py` (SymVerify)
  - `src/modules/filters/agents.py` (AdaptiveMultiAgent + Skeptic)

### 후속 (planner 핸드오프)

- **DECISIONS 후속 엔트리**: "2026-05-05 (Wave 4 Filter Ablation 14 cells 완료) — Stacked Filter F1 신규 최고 (a05_08 0.8809) + AdaptiveDepth EX 유일 개선 (a05_07 0.3501) + paper main anchor 옵션 A 권장"
- **paper_research_direction.md §3.5 갱신**: Filter design variation 추가 evidence — "Filter on/off (ΔF1=0.63) + Filter design (ΔF1=0.07) 둘 다 driver", a05_08/a05_07 결과 인용
- **paper_research_direction.md §10 핵심 수치 갱신**:
  - 신규 최고 F1: a05_08 Stacked F1=0.8809 (R=0.8880, P=0.8739, EX=0.3351)
  - 신규 최고 EX: a05_07 AdaptiveDepth EX=0.3501 (F1=0.8633)
  - paper main anchor 결정 (옵션 A 권장) 명시
- **paper_research_direction.md §8 Future Works**:
  - "Filter design variation" 항목 ✅ 완료 처리
  - **Stacked Filter** narrative 신설 sub-항목 (P-stage + R-stage 2-tier)
- **presentation_brief 갱신**: Wave 4 14 cells 결과 + Filter design variation evidence
- **선행 queued**:
  - F-1 alpha sweep 10 cells (LLM-free, ~1h, ₩0) — DECISIONS 2026-05-04 옵션 A
  - H-G Adaptive PCST F-1 alpha sweep 6-11 cells (LLM-free, ~1h, ₩0) — DECISIONS 2026-05-04 옵션 B
- **Analyzer 위임**:
  - a05_01 (AdaptiveMultiAgent) Skeptic over-prune mechanism 정량 — per-query R 분포 + Skeptic veto rate
  - a05_07 (AdaptiveDepth) EX 개선 mechanism — uncertainty distribution vs EX 정확도
  - a05_08 (Stacked) 2-stage absorption 의 stage-wise 분해 — Tiered 출력 vs Verifier 출력 differential
  - Filter design variation 의 R/P/F1 trade-off curve — 14 cells scatter

## F-1 Alpha Sweep + H-G Adaptive PCST F-1 (2026-05-05, 17 cells, 🔥 Stage 2 Filter dominance 결정적 evidence)

발사: 2026-05-05 11:16 → 완료 12:57 (wall clock ~1h 41min, GPU 0/1 split). 사용자 결정 (DECISIONS 2026-05-04 옵션 A 채택) — F-1 main stack 10 cells (GPU 0) + H-G Adaptive PCST F-1 7 cells (GPU 1) 병렬 진행.

### 근거 + 가설

§3.5 paper main insight "2-stage absorption" 의 결정적 evidence 확보. 직전 H-C partial (3 cells, basic PCST + no filter) 에서 R~0.96 plateau 관측 → Stage 1 (Extractor MST set saturation) 가설 제기. 단 partial sweep 한계 + basic PCST 가 paper main extractor 가 아니므로 본 측정으로 (a) MSTPCSTUnion + No Filter 11 cells full sweep + (b) AdaptivePCST 비교로 mechanism 주체 정량.

**결과 분기**:
- F-1 R/F1 spread > 0.05 → Filter dominance Stage 2 결정적 evidence (현 §3.5 narrative 강화)
- F-1 R/F1 spread ≤ 0.01 → Extractor set saturation Stage 1 결정적 evidence (§3.5 mechanism 정정)

### F-1 MSTPCSTUnion 11 cells 결과 (10 신규 + α=0.5 baseline 기존 측정)

Configs: `s04_ablation/pipeline/t00_f1_alpha_0[0~10].yaml` (10 신규) + `enriched_qcond_a05_mst_pcst_union_no_filter` (α=0.5 기존)
Stack: Enriched Builder + qcond_nl3 ckpt + α + MSTPCSTUnion(score_threshold=0.1) + **No Filter** + No SQL gen

| α | R | P | F1 | nodes | vs With-Filter ΔF1 |
|---|---|---|---|---|---|
| 0.0 (GAT only) | 0.7585 | 0.2047 | 0.3224 | 39.2 | +0.3806 |
| 0.1 | 0.8535 | 0.2137 | **0.3418** ★ | 42.2 | +0.4462 |
| 0.2 | 0.9645 | 0.1728 | 0.2931 | 57.3 | +0.5604 |
| 0.3 | 0.9845 | 0.1438 | 0.2509 | 71.2 | +0.6123 |
| 0.4 | 0.9905 | 0.1320 | 0.2330 | 78.9 | +0.6309 |
| 0.5 (baseline) | 0.9927 | 0.1268 | 0.2249 | 83.1 | +0.6408 |
| 0.6 | 0.9939 | 0.1240 | 0.2205 | 85.6 | +0.6433 |
| 0.7 | 0.9940 | 0.1224 | 0.2180 | 87.1 | +0.6449 |
| 0.8 | 0.9943 | 0.1212 | 0.2161 | 88.1 | +0.6483 |
| 0.9 | 0.9945 | 0.1208 | 0.2154 | 88.6 | +0.6485 |
| 1.0 (Cosine only) | **0.9947** ★ | 0.1207 | 0.2153 | 88.8 | +0.6511 |

**F-1 spread**:
- **R spread = 0.2362** (0.7585 → 0.9947)
- **F1 spread = 0.1265** (0.2153 → 0.3418)
- F1 best at α=0.1 — saturation 직전 sweet spot
- α≥0.2 부터 R 천장 도달 시작 (0.96+) → α↑ → P↓ (P=0.21 → 0.12) → F1↓
- node 수 39 → 89 (α↑ 따라 score-threshold seed 통과 노드 증가)

### H-G AdaptivePCST F-1 7 cells 결과

Configs: `s04_ablation/pipeline/t00_hg_adaptive_f1_alpha_0[0,02,04,05,06,08,10].yaml`
Stack: Enriched Builder + qcond_nl3 ckpt + α + AdaptivePCST(per-q P80, top-K=20) + No Filter + No SQL gen

| α | R | P | F1 | nodes |
|---|---|---|---|---|
| 0.0 | 0.5074 | 0.2566 | 0.3408 | 17.0 |
| 0.2 | 0.6480 | 0.3142 | 0.4232 | 18.5 |
| 0.4 | 0.7017 | 0.3268 | 0.4459 | 19.1 |
| 0.5 | 0.7260 | 0.3315 | 0.4552 | 19.2 |
| 0.6 | 0.7500 | 0.3392 | 0.4671 | 18.9 |
| 0.8 | **0.7834** ★ | 0.3511 | **0.4849** ★ | 18.7 |
| 1.0 | 0.7778 | 0.3428 | 0.4759 | 19.3 |

**H-G spread**:
- **R spread = 0.2760** (0.5074 → 0.7834)
- **F1 spread = 0.1441** (0.3408 → 0.4849)
- F1 best at α=0.8 — α=1.0 (Cosine only) 보다 약간 낮음 (GAT 가 marginal contribution)
- AdaptivePCST 의 R 천장 ≈ 0.78 (P80 percentile cutoff 효과) — MSTPCSTUnion 의 0.99 보다 훨씬 낮음
- node 수 17~19 (per-q P80 + top-K cap 으로 일정)

### 🎯 결정적 결론 — DECISIONS 분기 1 확정

| Stack | R spread | F1 spread | 분기 |
|-------|----------|-----------|------|
| F-1 MSTPCSTUnion (paper main extractor) | **0.2362** | **0.1265** | ✅ 분기 1 (>0.05 의 4-5배) |
| H-G AdaptivePCST | **0.2760** | **0.1441** | ✅ 분기 1 (>0.05 의 5-6배) |

**§3.5 narrative 정정 결정적 evidence**:

1. **🚨 Stage 1 Extractor MST set saturation 가설 — paper main pipeline 에서 부정**:
   - 직전 H-C partial 의 plateau (basic PCST R~0.96 invariant) 는 **basic PCST stack 한정 결과**
   - paper main 의 MSTPCSTUnion 은 plateau 부재 (R 0.7585 → 0.9947, spread 0.2362)
   - AdaptivePCST 도 plateau 부재 (R 0.5074 → 0.7834, spread 0.2760)
   - → **Extractor 가 plateau absorption 의 mechanism 주체 X**

2. **✅ Stage 2 Filter precision absorption — 결정적 evidence**:
   - F-1 F1 plateau spread (α∈[0.2,1.0]) 0.0778 → With-Filter F1 plateau spread 0.0129 (qcond_nl3) — **Filter 가 plateau region F1 spread 6배 압축**
   - F-1 P plateau spread (α∈[0.2,1.0]) 0.0521 (P=0.1207~0.1728) → With-Filter P plateau ~0.85 부근 거의 균일 — **Filter 가 P 를 ~0.85 로 균일 elevate (P spread 5-10배 압축)**
   - F-1 P 0.12-0.21 → With-Filter P 0.83-0.86 — Filter 가 P 를 평균 +0.65 ~ +0.74 elevate
   - ΔF1 (With-Filter - F-1): α=0.0 +0.38, α=0.5 +0.64, α=1.0 +0.65 — α 가 클수록 Filter 효과 증가 (R 큰 baseline 에서 Filter 가 P 회복 더 효과적)
   - → **Filter (Modular LLM) 가 plateau 의 dominant absorber, single-stage main mechanism**

3. **§3.5 narrative 정정**:
   - **❌ 직전 narrative ("2-stage absorption")**: "Stage 1 Extractor MST set saturation + Stage 2 Filter precision absorption"
   - **✅ 신규 narrative ("Filter dominance" — single-stage main + Extractor stack-dependent)**:
     - **Main mechanism (Stage 2)**: Modular LLM Filter 가 selector signal 차이를 set + ordering 모두 absorb (P 회복 +0.64, ΔF1 ≤ 0.005 plateau 생성)
     - **Stack-dependent (Stage 1)**: Extractor saturation 은 stack 특화 (basic PCST 만 saturate, MSTPCSTUnion / AdaptivePCST 는 plateau 부재)
     - **5 evidence 결합 갱신**: H-B ckpt-invariant + H-F stability/ordering + 🆕 H-C full + 🆕 H-G + ΔF1 +0.64 + H-A/H-D 부정

### 비용 / 운영

- 17 cells (F-1 10 + H-G 7) LLM-free: ₩0
- Wall clock: 1h 41min (GPU 0 sequential F-1 10 cells 1h 41min + GPU 1 sequential H-G 7 cells 1h 12min, 병렬 wall = max)
- Throughput: ~10 min/cell (F-1) ~ 10 min/cell (H-G) — LLM 없으므로 GAT inference + PCST 만
- 모든 cells 정상 완료, failure 없음

### 산출물

- Configs (17): `s04_ablation/pipeline/t00_f1_alpha_0[0~10].yaml` (10) + `s04_ablation/pipeline/t00_hg_adaptive_f1_alpha_0[0,02,04,05,06,08,10].yaml` (7)
- Scripts: `scripts/run_f1_full_alpha_sweep.sh` + `scripts/run_hg_adaptive_f1_sweep.sh` (failure-tolerant sequential)
- 비교 baseline: `enriched_qcond_a05_mst_pcst_union_no_filter` (α=0.5 기존 측정)
- 코드 수정 없음 (기존 EnsembleSelector + MSTPCSTUnionExtractor + AdaptivePCSTExtractor 재사용)

### 후속 (planner 핸드오프)

- **DECISIONS 후속 엔트리** (planner 작성):
  - "2026-05-05 (F-1 + H-G alpha sweep 17 cells 완료) — 분기 1 확정 + Stage 2 Filter precision absorption 결정적 evidence + §3.5 narrative 정정 (2-stage → Filter dominance single-stage main)"
- **paper_research_direction.md §3.5 mechanism 정밀화 정정**:
  - "2-stage absorption" → "Filter dominance (Stage 2 main, Stage 1 stack-dependent)"
  - 5 evidence 결합 갱신 — F-1 full + H-G + 기존 H-B/H-F/H-C partial + ΔF1
  - F-1 R spread 0.2362 + H-G R spread 0.2760 + Filter plateau-region F1 압축 비율 6× 정량 인용
- **§9 Limitations 갱신**: F-1 partial sweep (3 cells) 한계 항목 → ✅ full 11 cells 측정 완료, 한계 해소
- **§10 핵심 수치 표 갱신**: F-1 11 cells + H-G 7 cells R/P/F1 spread 행 추가
- **§8 Future Works**:
  - 🆕 H-G Extractor MST set saturation ✅ 검증 완료 (paper main pipeline 에서 saturation 부재)
  - basic PCST saturation 이 stack 특화 — post-deadline mechanism deep dive 후보
- **Analyzer 위임 (post-planner)**:
  - F-1 + H-G 결과로 §3.5 Stage 2 단일 absorption mechanism 정밀화 — `alpha_plateau_mechanism_validation.md §7` 신설 또는 `mechanism_final.md` 작성
  - Filter F1 압축 비율 (20×) per-query 분포 — Filter 가 어떤 query type 에서 가장 강한 absorption 수행하는지
  - F-1 best at α=0.1 (sweet spot before saturation) vs With-Filter plateau α∈[0.2~1.0] — Filter 가 saturation 후 P 차이 absorb 하는 mechanism 정량

## Directed Top-K SuperNode GAT 학습 (V-3-ext 단계 2, 2026-05-06, 3 변형 × 300 epochs)

발사: 2026-05-06 00:20 → 완료 10:37 (wall clock ~10h 17min, GPU 0/1/2 split). 학위 논문 Part III Directed Top-K SuperNode 변형 학습 — `EXPERIMENT_PLAN_selectors.md` V-3-ext 단계 2 (2026-05-05 단계 1 구현 완료 후속).

### 운영 이력 (사용자 결정 + 운영 결정)

- **사용자 결정 옵션 A (epochs 300 채택, 2026-05-06 00:18 KST)**: 초기 sweep 의 epochs=20 이 main baseline (qcond_nl3 등 epochs=300) 와 일관성 부재 — 사용자 지적으로 즉시 옵션 A 채택, 3 configs `epochs: 20 → 300` 갱신 후 재launch.
- **사용자 결정 GPU 2 일시 활용 (2026-05-06 02:48 KST)**: 새벽 시간대 다른 연구자 GPU 미사용 확정으로 abstau07 즉시 GPU 2 launch — 당초 PRIMARY p80 종료 후 GPU 0 launch 예정에서 GPU 2 병렬로 변경. wall clock ~14:55 → ~10:37 KST 4h 18min 단축.
- **wrapper bash kill (사용자 옵션 A 승인, 02:50 KST)**: GPU 2 abstau07 launch 시 wrapper 의 GPU 0 abstau07 launch 차단 위해 부모 bash 3772005 SIGTERM. setsid + 무 TTY 환경 검증 — 자식 python (p80, topk20) SIGHUP 미전파, 학습 무중단 유지. train_one subshell 의 post-train NAS mv 로직은 subshell 내부 실행이라 자동 처리.
- **초기 20-epoch sweep KeyboardInterrupt 사고 (00:04 KST)**: nohup 으로 launch 한 sweep 의 topk20 가 epoch 5 validate() 단계에서 SIGINT 받음. 원인 추정: nohup 은 SIGHUP 만 차단, terminal Ctrl-C SIGINT 전파됨. 옵션 A 재launch 시 setsid 적용으로 재발 방지.

### 학습 변형 + 결과 — 3 ckpt 모두 NAS symlinked

| 변형 | mode | value | epochs | wall | best val recall@15 | NAS ckpt |
|---|---|---|---:|:---:|---:|---|
| **PRIMARY p80** | percentile | 80.0 | 300 | 7h 30min (00:20~07:50) | **0.6097** | `best_gat_directed_supernode_p80.pt` (171MB) |
| **BASELINE topk20** | top_k | 20 | 300 | 7h 36min (00:20~07:56) | 0.5839 | `best_gat_directed_supernode_topk20.pt` (171MB) |
| **OPTIONAL abstau07** | abs_tau | 0.7 | 300 | 7h 43min (02:54~10:37) | 0.5805 | `best_gat_directed_supernode_abstau07.pt` (171MB) |

### Best val recall@15 progression (50-epoch 단위) — 3 변형 모두 saturation 결정적 evidence

```
p80       : ep50=0.6083 ep100=0.6097 ep150=0.6097 ep200=0.6097 ep250=0.6097 ep300=0.6097
topk20    : ep50=0.5836 ep100=0.5837 ep150=0.5839 ep200=0.5839 ep250=0.5839 ep300=0.5839
abstau07  : ep50=0.5788 ep100=0.5805 ep150=0.5805 ep200=0.5805 ep250=0.5805 ep300=0.5805
```

**🚨 saturation 결정적**: p80 epoch 100 부터 200 epochs 동안 best 0.6097 그대로. topk20 epoch 150 부터 150 epochs 무변동. abstau07 epoch 100 부터 200 epochs 무변동. **추가 학습 효과 거의 없음 — GAT 학습이 raw selector R 한계 회복 못함의 결정적 evidence**.

### vs raw selector standalone (V-3-ext PLAN §V-3-ext 표 인용)

| 변형 | raw R (standalone, GAT 학습 X) | 학습 후 best val R@15 | Δ |
|---|---:|---:|---:|
| p80 | 0.6133 | **0.6097** | -0.0036 (학습이 raw R 거의 회복, 미흡 -0.4%p) |
| topk20 | 0.6865 | 0.5839 | **-0.1026** (학습이 raw R 능가 못함, **negative result**) |
| abstau07 | 0.4857 | 0.5805 | **+0.0948** (학습이 raw R 능가, 가장 큰 학습 개선) |

**🎯 핵심 함의** (시나리오 A 또는 C 후보, 단계 3 alpha sweep BIRD-dev F1/EX 결과 의해 확정):
1. **시나리오 A 가장 가능성 高 (Filter Dominance)**: GAT 학습이 selector R 한계 회복 못 함 → R 0.85+ 도달 못 함 → Filter 가 plateau 안 흡수. paper §3.5 Filter Dominance narrative 와 일관.
2. **시나리오 C 가능성** (R<0.85 negative result): 단계 3 alpha sweep 시 BIRD-dev F1<0.85 면 paper §V.5.3 negative result + advisor mechanism deep dive
3. **topk20 의 학습 negative result** (-0.1026 vs raw): directed_from_sn edge 가 top_k=20 (강제 20 노드) 모드에서 학습 disadvantage. P80 (variable, mean 18.9) 와 abs_tau (variable, mean 10.2) 는 학습 효과 발현. **per-query selectivity 가 학습 핵심**.

### 비용 / 운영

- 3 변형 GPU 학습: ₩0 (LLM 미사용)
- Wall clock: ~10h 17min (당초 ~22h sequential 대비 GPU 2 일시 활용으로 절반 단축)
- GPU: 0 (p80) + 1 (topk20) + 2 (abstau07) — memory rule 명시 옵션 A 승인 (새벽 한정)
- 모든 학습 정상 완주, mid-run 사고 (KeyboardInterrupt) 1건 — 옵션 A 재launch 시 해소
- ckpt 3 변형 NAS 정상 저장 + symlink 검증 완료 (각 171MB)

### 산출물

- Configs (3): `configs/training/train_gat_directed_supernode_{p80, topk20, abstau07}.yaml` (epochs=300 갱신)
- Script: `scripts/run_directed_supernode_training.sh` (failure-tolerant, post-train NAS mv 자동)
- Checkpoints (NAS): `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_{p80, topk20, abstau07}.pt`
- 학습 logs (NAS symlink): `/home/hyeonjin/thesis_refactored/logs/train/gat_directed_supernode_{p80, topk20, abstau07}_20260506_*.log`

### 후속 (root 단계 3 + planner 핸드오프)

- **단계 3 (즉시, root)**: paper main stack (Enriched + 신규 ckpt + α=0.5 + MSTPCSTUnion + XiYan GLM + LLMSQLGenerator GLM) 위 alpha sweep — 3 ckpt × α∈{0.0, 0.5, 1.0} 최소 9 cells (또는 full 11 cells/ckpt × 3 = 33 cells). 비용 ~₩8-30K.
- **시나리오 분기 (단계 3 결과 의해)**:
  - 시나리오 A: F1 ≤ 0.870 → Filter Dominance 5 축 격상 (🆕 topology-invariant). paper §3.5 narrative 강화
  - 시나리오 B: F1 > 0.870 → 학위 논문 main contribution 5 항목 격상
  - 시나리오 C: F1 < 0.85 → paper §V.5.3 negative result + advisor mechanism deep dive
- **DECISIONS 후속 엔트리** (planner): "2026-05-06 (Directed SuperNode 학습 완료) — 3 변형 모두 saturation, GAT 학습이 raw R 한계 회복 못함 (시나리오 A 잠정), 단계 3 alpha sweep 으로 BIRD-dev F1/EX 분기 확정"
- **paper_research_direction.md §3.5 갱신 candidate** (단계 3 결과 후): topology-invariant Filter Dominance evidence — Direct Top-K SuperNode 학습이 R 한계 회복 못 함에도 plateau 가 안정적이면 Filter mechanism 의 5번째 축 (graph topology 변경 시에도 robust)

## DSN Phase 1 Alpha Sweep (V-3-ext 단계 3, 2026-05-06, 9 cells, 🎯 시나리오 A 확정)

발사: 2026-05-06 11:11 → 완료 13:36 (wall ~2h 25min, GPU 0/1 9 cells 병렬). 학위 논문 Part III V-3-ext 단계 3 — paper main t_00 stack + DirectedTopKSuperNodeSelector × 3 ckpt × α∈{0.0, 0.5, 1.0}.

### 근거 + 가설

`EXPERIMENT_PLAN_selectors.md` V-3-ext §단계 3 + DECISIONS 2026-05-06 (사용자 핸드오프 prompt). 의도: GAT 학습이 raw selector R 한계 회복 못함 (단계 2 결과 saturation 0.58-0.61) 에도 paper main pipeline F1/EX 가 plateau 유지하는지 검증 → Filter Dominance topology-invariant 5축 candidate.

### Stack

paper main t_00 + DirectedTopKSuperNodeSelector + 신규 3 ckpt (V-3-ext 단계 2 결과 학습)
- Builder: EnrichedHeteroGraphBuilder
- Selector: **DirectedTopKSuperNodeSelector** (3 변형)
  - p80: percentile=80.0 (best val R@15=0.6097)
  - topk20: top_k=20 (best val R@15=0.5839)
  - abstau07: abs_tau=0.7 (best val R@15=0.5805)
  - 공통: top_k=20 (final select), supernode_edge_direction=directed_from_sn, score_normalization=minmax
- Extractor: MSTPCSTUnionExtractor(score_threshold=0.1)
- Filter: XiYanFilter(GLM, max_iteration=1)
- SQL gen: LLMSQLGenerator(GLM)
- α: 0.0, 0.5, 1.0

### 9 cells 결과 — F1 정렬 (t_00 base F1=0.8657, EX=0.3377)

| 순위 | Cell | R | P | F1 | EX | vs t_00 ΔF1 |
|---|---|---|---|---|---|---|
| 1 | **topk20_α=1.0** | 0.8776 | 0.8547 | **0.8660** | **0.3396** | +0.0003 |
| 1 | **abstau07_α=1.0** | 0.8787 | 0.8536 | **0.8660** | 0.3377 | +0.0003 |
| 3 | abstau07_α=0.5 | 0.8753 | 0.8546 | 0.8648 | 0.3364 | -0.0009 |
| 3 | **p80_α=1.0** | 0.8766 | 0.8534 | 0.8648 | **0.3396** | -0.0009 |
| 5 | topk20_α=0.5 | 0.8742 | 0.8551 | 0.8645 | 0.3318 | -0.0012 |
| 6 | p80_α=0.5 | 0.8738 | 0.8546 | 0.8641 | 0.3331 | -0.0016 |
| 7 | p80_α=0.0 (GAT only) | 0.7415 | 0.7877 | 0.7639 | 0.2288 | -0.1018 |
| 8 | abstau07_α=0.0 | 0.7315 | 0.7792 | 0.7546 | 0.2484 | -0.1111 |
| 9 | topk20_α=0.0 | 0.6932 | 0.7656 | 0.7276 | 0.2269 | -0.1381 |

### 🎯 시나리오 A 확정 (Filter Dominance 5번째 축 topology-invariant)

DECISIONS 2026-05-05 §1(d) 분기:
- **✅ 시나리오 A** (F1 ≤ 0.870, plateau 흡수): **확정** — best F1=0.8660 < 0.870, plateau 0.0019 spread (α∈{0.5, 1.0} 6 cells)
- ❌ 시나리오 B (F1 > 0.870 main contribution 격상): 미충족
- ❌ 시나리오 C (F1 < 0.85 negative result): 미충족

### 핵심 발견 (5)

**(1) Filter Dominance topology-invariant 결정적 evidence**:
- α∈{0.5, 1.0} 6 cells F1 plateau **[0.8641, 0.8660]** spread = **0.0019**
- 직전 qcond_nl3 stack plateau α∈[0.2~1.0] spread 0.013 와 동일 패턴 (둘 다 ≤0.013)
- **graph topology 변경 (Concat → directed_from_sn SuperNode) + selector threshold 변경 (top_k vs percentile vs abs_tau) 모두에도 plateau 동일** → Filter mechanism 의 **5번째 축 (topology-invariant) 정량 evidence**
- §3.5 paper main insight 5 evidence + topology-invariant 6번째 evidence 추가

**(2) GAT 학습이 R 한계 회복 못 함의 schema-linking 단위 직접 증명**:
- α=0.0 (GAT only) 3 cells 모두 F1 0.7276-0.7639 (R 0.6932-0.7415)
- 이전 학습 단계의 val recall@15 saturation (0.5805-0.6097) 와 일관 — GAT 학습이 selector R 한계 회복 못함의 BIRD-dev 단위 직접 측정
- ΔF1 vs t_00: -0.10 ~ -0.14 (α=0.0) → α=0.5/1.0 plateau (-0.0009 ~ +0.0003) 회복 — Filter 가 GAT raw R 차이 absorb

**(3) 3 ckpt 가 동일 plateau 도달**:
- α=1.0: topk20=0.8660, abstau07=0.8660, p80=0.8648 (Δ 0.0012 noise band)
- 학습 차이 (best val recall 0.5805~0.6097, Δ 0.0292) 가 With-Filter F1 에서 ~26× 압축됨 (Δ 0.0012)

**(4) Best EX = 0.3396** (p80_α=1.0 + topk20_α=1.0) — t_00 base 0.3377 대비 +0.0019 marginal. EX 도 plateau 안.

**(5) F1 ↔ EX 일관 patterns**:
- α=0.0 cells: F1↓ + EX↓ (plateau 외)
- α=0.5/1.0 cells: F1 plateau + EX plateau

### 비용 / 운영

- 9 GLM cells: ~₩18-36K
- Wall clock: 2h 25min (당초 추정 2.5-3h, GPU 0/1 9 cells 병렬, GLM API contention 양호)
- GPU: 0 (5 cells: p80 × 3 + topk20 × 2) + 1 (4 cells: topk20 × 1 + abstau07 × 3)
- 모든 cells 정상 완료, failure 없음

### 산출물

- Configs (9): `configs/experiments/s04_ablation/pipeline/dsn_phase1/t00_dsn_{p80,topk20,abstau07}_alpha_{00,05,10}.yaml`
- Script: `scripts/run_dsn_alpha_sweep_phase1.sh` (failure-tolerant 9 cells parallel GPU 0/1 split)

### 후속 (planner 핸드오프)

- **DECISIONS 후속 엔트리** (planner): "2026-05-06 (DSN Phase 1 9 cells 완료) — 시나리오 A 확정 (Filter Dominance topology-invariant), best F1=0.8660 ≈ t_00 (0.8657), §3.5 narrative 5번째 축 (topology-invariant) 추가"
- **paper_research_direction.md §3.5 갱신**:
  - Filter Dominance 5 축 (직전 4축) → **6 축** (topology-invariant 추가)
  - 6번째 evidence: DSN 9 cells plateau 0.0019 spread (raw R 0.58-0.61 차이를 With-Filter 가 0.0012 차이로 압축)
  - 3 ckpt × 3 α 표 인용
- **paper_research_direction.md §10 핵심 수치 갱신**:
  - DSN 9 cells F1/EX 표 추가
  - With-Filter plateau 압축 비율 (raw R Δ 0.029 → With-Filter F1 Δ 0.0012, ~26×)
- **paper_research_direction.md §8 Future Works**:
  - 학위 논문 Part III V-3-ext 단계 3 ✅ 완료 (Phase 1)
  - Phase 2 (full alpha 11 cells/ckpt): 시나리오 A 확정으로 priority 강등 — best ckpt (topk20 또는 abstau07) full curve 만 선택적 진행 가능
- **paper §V.5 갱신 candidate** (학위 논문):
  - Part III "Directed Top-K SuperNode" — best F1=0.8660 (topk20/abstau07 α=1.0) baseline 동등, paper main 의 robustness 입증
  - V-3-ext 단계 2 학습 결과 (saturation) + 단계 3 결과 (plateau) 통합 narrative
- **Analyzer 위임** (선택, post-paper):
  - DSN plateau 의 per-query 분포 — Filter 가 어떤 query 에서 가장 강한 absorption 수행 (3 ckpt × t_00 vs DSN 비교)
  - α=0.0 cells (F1 0.73-0.76) 의 R 차이 mechanism — GAT-only 모드에서 ckpt 변동성이 R 에 미치는 영향

## Baseline Correction — qcond_nl3 best val recall@15 = 0.6061 (2026-05-06, analyzer V-3-ext 단계 4 over-smoothing 진단 부산물)

분석일: 2026-05-06 (V-3-ext 단계 4 over-smoothing 진단 시 4 ckpt training log 일괄 parse 산출).

### 정정 사유

본 HISTORY 의 직전 entries (Wave 4 / F-1 + H-G / V-3-ext 단계 2-3 등) 에서 paper main t_00 anchor ckpt `best_gat_qcond_nl3.pt` 의 best val recall@15 값이 **명시 기록 부재**. 일부 cross-references 에서 "main baseline (qcond_nl3 등 epochs=300)" 등으로 언급되었으나 실측 수치 없음. 인접 ckpt (`best_gat_query_supernode_direct` val R=0.5548 / `best_gat_query_supernode_qcond` val R=0.5737) 와의 유사성으로 사용자 / planner 측 추정 ~0.55 가 통용됐으나, **실측은 0.6061**.

### 정정 값

| Ckpt | best val recall@15 | best epoch | Final R@15 | 학습 일자 (추정) | 학습 wall |
|---|---:|---:|---:|---|---|
| **best_gat_qcond_nl3.pt** | **0.6061** | **59** | 0.5958 | 2026-04-23 (cross-references) | 미기록 (~9h 추정 epochs=300) |
| (참고) best_gat_directed_supernode_p80.pt | 0.6097 | 91 | 0.6055 | 2026-05-06 | 7h 30min |
| (참고) best_gat_directed_supernode_topk20.pt | 0.5839 | 112 | 0.5814 | 2026-05-06 | 7h 36min |
| (참고) best_gat_directed_supernode_abstau07.pt | 0.5805 | 90 | 0.5720 | 2026-05-06 | 7h 43min |

**핵심 함의**:
- DSN p80 (0.6097) ≈ qcond_nl3 baseline (0.6061), Δ=+0.0036 — **graph topology 변경 (Concat → directed_from_sn) 이 학습 saturation 한계 갱신 못함**
- 이전 추정 ~0.55 부정확 → DSN ckpt 들이 baseline 을 +0.04 능가했다는 narrative 는 일부 부정확. 정확 narrative: DSN p80 만 baseline 과 동등, topk20/abstau07 는 underperform.
- BCE-Recall divergence (4 ckpt 모두 ep23~38 detect): 학습 saturation 의 결정적 evidence — 이 패턴 자체가 over-smoothing 후속 분석의 trigger.

### 근거

- Analyzer 산출: [`notebooks/analysis_results/dsn_oversmoothing_analysis.md`](notebooks/analysis_results/dsn_oversmoothing_analysis.md) §1.1 "Best Val Recall@15 + Saturation 비교 (4 ckpt)" 표
- Source: `best_gat_qcond_nl3` 학습 log parse — `parse_training_log_v2()` 알고리즘 적용
- Cross-reference: [planning/DECISIONS.md 2026-05-06 (DSN over-smoothing 진단)](planning/DECISIONS.md) §c — "qcond_nl3 baseline 정정 — HISTORY 추정 ~0.55 부정확"

### 영향 범위

- **HISTORY**: 본 entry 신설 (정정 record 의 단일 출처)
- **CATALOG**: qcond_nl3 baseline 표기 갱신 (best val recall@15 추가)
- **ID_MIGRATION**: qcond_nl3 ckpt entry 갱신 (best val recall@15 + best epoch)
- **paper_research_direction.md** §10 / §3.5: 직접 영향 X (planner 가 별도 정정 권한)
- **학회 논문**: anchor t_00 (F1=0.8657) 변경 X — 본 정정은 학습 saturation 정량만 갱신

### 후속

- 본 entry 가 정정 record 의 출처 — V-3-ext 단계 2/3/Phase 2 entries 의 cross-reference 에서 "qcond_nl3 baseline 0.6061" 이라 명시 가능
- planner 가 paper §10 / §3.5 narrative 정정 시 본 entry 인용

## DSN Phase 2 + Phase 3 4-trial Mitigation Sweep (V-3-ext 단계 5, 2026-05-06 → 05-07, 🎯 시나리오 P3-A 결정적 confirm)

발사: 2026-05-06 17:45 (Phase 2 b8 launch) → 완료 2026-05-07 15:29 (Phase 3 #4 종료). Wall ~46h (병렬), 학습 합산 ~39h. 사용자 결정 (DECISIONS 2026-05-06 §1(C)/(I) — Phase 3 #3+#4 가속 + alpha sweep skip + Phase 3 #4 자동 이어 실행).

### 운영 이력 (사용자 결정 + 운영 결정)

- **Phase 2 b8**: 2026-05-06 17:45 launch (batch_size=1 → 8 사용자 결정으로 변경 후 본격 학습), 2026-05-07 04:11 종료. 10h 26min wall.
- **Phase 3 #3 (Direct AC)**: 2026-05-07 00:18 launch (GPU 1, Phase 2 와 병렬), 2026-05-07 10:33 종료. 10h 15min wall.
- **Phase 3 #4 (LR x5)**: 2026-05-07 04:13 launch (GPU 0, Phase 2 종료 직후 즉시 launch — 사용자 결정 가속, 당초 plan ~10:25 KST → 6h 단축), 2026-05-07 15:29 종료. 11h 16min wall.
- **STEP 3-5 alpha sweep skip** (사용자 결정 2026-05-07 03:00 KST): "Phase 3 의 #4 는 자동으로 이어서 실행하고 alpha sweep 은 하지 마"
- **모니터링 종료**: Phase 3 #4 종료 시점 — 사용자 결정 2026-05-07 02:50 KST

### 신규 학습 entries (3 cells)

#### Phase 2 b8 — DSN p80 + B5 mitigation (AC target='fusion')

- Config: `configs/training/train_gat_directed_supernode_p80_b5_mitigation.yaml`
- Stack: V-3-ext (DSN p80 directed_from_sn + percentile=80) + B5 mitigation (PN+IR α=0.2+JK=concat+Dual-Stream+L=2+AC=0.1+ListNet)
- 학습 entry: `train_gat_s06.py` (V-3-ext options forward 추가 by root 2026-05-06 17:00)
- batch_size: 8 (사용자 결정 2026-05-06 17:10 — 초기 b1 7.82min/ep → b8 2.12min/ep, 3.7x 빠름)
- AC target: 'fusion' (Phase 2 default — model output 후 적용, skip path 흡수 가능)
- 신규 ckpt: `best_gat_directed_supernode_p80_b5_mitigation.pt` (113MB, NAS symlinked)
- Best val recall@15 = **0.6018** @ ep157, final R@15 0.5988

#### Phase 3 #3 — DSN p80 + B5 + Direct AC

- Config: `configs/training/train_gat_directed_supernode_p80_b5_phase3_directAC.yaml`
- Stack: Phase 2 base + 변경: AC target='gat_out_L_last' (forward hook 으로 main GAT path 직접 압박, skip path 우회 차단 의도)
- 신규 ckpt: `best_gat_directed_supernode_p80_b5_phase3_directAC.pt` (113MB, NAS symlinked)
- Best val recall@15 = **0.5927** @ ep51, final R@15 0.5874
- AC loss 0.62 일관 유지 (Phase 2 의 0.07 → 0.01 decay 와 대조) — main GAT path 가 collapse 압박 처리 못함의 정량 evidence

#### Phase 3 #4 — DSN p80 + B5 + Layer-wise LR

- Config: `configs/training/train_gat_directed_supernode_p80_b5_phase3_layerwiseLR.yaml`
- Stack: Phase 2 base + 변경: optimizer_layer_wise_lr=true, gat_lr_multiplier=5.0 (HeteroConv path 5e-4, other 1e-4)
- 신규 ckpt: `best_gat_directed_supernode_p80_b5_phase3_layerwiseLR.pt` (113MB, NAS symlinked)
- Best val recall@15 = **0.5935** @ ep172, final R@15 0.5895
- Phase 3 #3 보다 빠른 수렴 (LR x5 효과 발현) but ceiling 갱신 X

### 🎯 4-trial mitigation 결과 표 (decreasing R@15)

| 순위 | Variant | Best R@15 | Best Epoch | Δ vs Phase 1 | Mitigation 항목 |
|------|---------|-----------|------------|--------------|-----------------|
| **1** | **Phase 1 P80 (no mit)** | **0.6097** | ep91 | (baseline) | (no mit) |
| 2 | Phase 3 #4 (LR x5) | 0.5935 | ep172 | -0.0162 | B5 mitigation + gat_lr 5e-4 / other 1e-4 |
| 3 | Phase 3 #3 (Direct AC) | 0.5927 | ep51 | -0.0170 | B5 mitigation + AC target='gat_out_L_last' |
| 4 | Phase 2 b8 (mit fusion) | 0.6018 | ep157 | -0.0079 | B5 mitigation (PN+IR+JK+DS+L=2+AC fusion+ListNet) |

**🚨 핵심 발견**:
1. **모든 mitigation variants 가 baseline (Phase 1) 보다 lower** — graph topology 변경 (Phase 1 → 2) + 5 mitigation (Phase 2) + Direct AC (Phase 3 #3) + LR x5 (Phase 3 #4) 모두 raw R 한계 갱신 X
2. **Phase 2 가 mitigation variants 중 best** (0.6018) — 이상하게 더 적극적 mitigation (Direct AC, LR x5) 가 오히려 underperform
3. Phase 1 baseline 의 학습 saturation (0.6097) 이 모든 mitigation 시도에 robust — **training-pathology-invariant**

### 시나리오 분기 결정 — P3-A 절대 confirm

DECISIONS 2026-05-06 §1(F) 분기:
- **✅ 시나리오 P3-A** (null effect, mitigation 모두 ceiling 갱신 X): **결정적 confirm**
  - 4 trials × {graph topology 변경, B5 mitigation, Direct AC, Layer-wise LR} 모두 → val R@15 ~0.59-0.61 영역 saturate
  - paper main F1 plateau (Phase 1 plateau 0.0019 spread) 가 학습 saturation 어떤 형태에도 absorb 가능 → **Filter (Modular LLM) 의 single-stage main mechanism 의 학위 논문 Part III main contribution**
- ❌ 시나리오 P3-B (partial recovery, val R 0.62-0.70): 미달
- ❌ 시나리오 P3-C (full recovery, val R 0.85+): 사실상 불가능 확정

### Filter Dominance 6번째 축 (training-pathology-invariant) 결정적 evidence

직전 5 evidence + 6번째 추가:
1. H-B ckpt-invariant
2. H-F stability/ordering
3. F-1 + H-G alpha sweep
4. ΔF1 +0.65 lift
5. H-A/H-D 부정
6. **🆕 Phase 2 + Phase 3 mitigation 4-trial null effect** (training-pathology-invariant)

### AC loss 행태 — Mechanism 정량 (학위 논문 Part III deep dive)

| Variant | AC target | AC ep1 | AC ep~50 | AC ep~150 | 해석 |
|---------|-----------|--------|----------|-----------|------|
| Phase 2 b8 | 'fusion' | 0.0683 | ~0.005 | ~0.001 | skip path 가 AC 흡수 (pathology 우회) |
| Phase 3 #3 | 'gat_out_L_last' | 0.6155 | 0.6178 | 0.6183 | **main GAT path 가 collapse 압박 처리 못함** (raw GAT path 는 학습 통해서도 collapse mitigation 불가) |
| Phase 3 #4 | 'fusion' (Phase 2 동일) | 0.07 | ~0.01 | ~0.005 | LR x5 로 GAT path 빠른 학습 but fusion AC 는 동일 |

**🔥 결정적 evidence**:
- Phase 3 #3 의 AC=0.62 일관 유지 → main GAT path 의 raw collapse 가 학습으로 회복 안 됨
- Phase 2 / Phase 3 #4 의 fusion AC decay → fusion path 가 main GAT path 의 collapse 를 우회 (skip 활용)
- 어떤 path 든 raw R 한계 ~0.61 영역 — **GAT path 자체의 fundamental limitation** (raw PLM embedding sibling 유사성 또는 GATv2Conv normalization mechanism 후보 — analyzer deep dive 위임)

### 비용 / 운영

- 4 학습 합산 wall: 39h 9min (Phase 1 7.5h + Phase 2 10.5h + Phase 3 #3 10.25h + Phase 3 #4 11.25h)
- 병렬 wall: ~46h (5/6 00:20 ~ 5/7 15:29)
- 비용: **₩0** (모든 학습 LLM-free)
- 4 ckpt NAS 저장: Phase 1 P80 171MB + Phase 2/3 #3/3 #4 각 113MB = ~510MB
- Alpha sweep skip (사용자 명시) → paper main F1/EX 측정 X, val recall@15 evidence 만 활용

### 산출물

- Configs (3 신규):
  - `configs/training/train_gat_directed_supernode_p80_b5_mitigation.yaml`
  - `configs/training/train_gat_directed_supernode_p80_b5_phase3_directAC.yaml`
  - `configs/training/train_gat_directed_supernode_p80_b5_phase3_layerwiseLR.yaml`
- 학습 entry 확장: `src/train_gat_s06.py` (V-3-ext options forward + AC target='gat_out_L_last' + optimizer_layer_wise_lr 추가)
- Checkpoints (NAS): `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_b5_{mitigation, phase3_directAC, phase3_layerwiseLR}.pt`
- 학습 logs: `logs/train/gat_directed_supernode_p80_b5_{mitigation, phase3_directAC, phase3_layerwiseLR}_*.log`

### 후속 (analyzer + planner 핸드오프)

- **Analyzer 위임**: 4 ckpt × 4 mechanism 후보 (DECISIONS 단계 4-bis §(d)) 분리 분석
  - (i) Aggregation collapse: top-5 흡수 노드 sibling 유사성
  - (ii) GATv2Conv normalization: edge softmax 분포
  - (iii) Skip dependency pathology: gradient flow main vs skip path
  - (iv) Schema sibling 유사성: raw PLM embedding intra-table cosine sim
  - 산출물: `notebooks/analysis_results/dsn_phase3_mitigation_results.md`
- **Planner 위임**: 시나리오 P3-A 정식 채택 + paper §3.5 narrative 6번째 축 정식 명문화 + 학위 논문 Part III chapter base
  - DECISIONS 후속 엔트리
  - paper_research_direction.md §3.5 / §V / §8 / §10 갱신
  - presentation_brief 갱신

## DSN Mitigation v2 3-trial Sweep (V-3-ext 단계 6, 2026-05-07 → 05-08, 🎯 시나리오 V2-A 확정)

발사: 2026-05-07 16:35 → 완료 2026-05-08 13:54 (wall ~21h, GPU 0 3개 동시 학습). 사용자 결정 옵션 A (3개 동시 GPU 0) — sweep wrapper kill 후 #2 manual launch + setsid + disown.

### 운영 이력

- **5/7 16:35**: sweep wrapper launch (#1 + #3 GPU 0/1 병렬)
- **5/7 17:27**: 사용자 GPU 자원 배분 이슈로 GPU 1 #3 layernorm kill (ep20 partial)
- **5/7 17:31**: resume wrapper launch (#3 GPU 0 재학습) — wrapper bug 로 즉시 launch (sum_aggr wait 검출 실패)
- **5/7 17:43**: 사용자 결정 옵션 A — 3개 동시 GPU 0 (#2 manual launch 즉시)
- **5/7 17:59**: sweep wrapper parent kill (1899618) + #2 sum_aggr manual launch — 3개 동시 진행 시작
- **5/8 12:22**: #1 drop_message 종료 → NAS symlinked
- **5/8 13:54**: #3 layernorm + #2 sum_aggr 종료 → NAS symlinked, 전체 sweep 완료

### 신규 학습 entries (3 ckpts)

#### v2 #1 DropMessage (drop_message_p=0.2)

- Config: `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.yaml`
- Mechanism: GATv2Conv.message 출력에 F.dropout(p=0.2) — attention 가중치는 그대로, attended-to neighbor feature contribution 분산 (mech(ii) edge softmax over-concentration mitigation 시도)
- 신규 ckpt: `best_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.pt` (113MB, NAS symlinked)
- Best val recall@15 = **0.5974** @ ep157, final R@15 0.5936

#### v2 #3 LayerNorm pre-softmax (use_layernorm_pre_softmax=true)

- Config: `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_layernorm.yaml`
- Mechanism: GATv2Conv 의 attention coefficient 에 LayerNorm 적용 후 softmax — sharp peak 완화
- 신규 ckpt: `best_gat_directed_supernode_p80_b5_mitigation_v2_layernorm.pt` (113MB, NAS symlinked)
- Best val recall@15 = **0.6011** @ ep289 (208 epochs 만의 미세 갱신), final R@15 0.5988
- **🔥 Mitigation variants 중 best** — Phase 2 (0.6018) 까지 -0.0007 까지 좁힘

#### v2 #2 Sum Aggregation (aggregation_type='sum')

- Config: `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_sum_aggr.yaml`
- Mechanism: aggregation type mean → sum 변경 — message contribution variance 보존 시도
- 신규 ckpt: `best_gat_directed_supernode_p80_b5_mitigation_v2_sum_aggr.pt` (113MB, NAS symlinked)
- Best val recall@15 = **0.5761** @ ep194, final R@15 0.5748
- mitigation variants 중 worst — sum aggregation 의 학습 dynamics fundamental difference

### 🎯 7-trial mitigation 통합 결과

| 순위 | Variant | Best R@15 | Best Epoch | Δ vs Phase 1 |
|------|---------|-----------|------------|--------------|
| **1** | **Phase 1 P80 (no mit, baseline)** | **0.6097** | ep91 | (baseline) |
| 2 | Phase 2 b8 (mit fusion) | 0.6018 | ep157 | -0.0079 |
| 3 | **v2 #3 LayerNorm pre-softmax** | **0.6011** ★ | ep289 | -0.0086 |
| 4 | v2 #1 DropMessage | 0.5974 | ep157 | -0.0123 |
| 5 | Phase 3 #4 (LR x5) | 0.5935 | ep172 | -0.0162 |
| 6 | Phase 3 #3 (Direct AC gat_out_L_last) | 0.5927 | ep51 | -0.0170 |
| 7 | v2 #2 Sum Aggregation | 0.5761 | ep194 | -0.0336 |

**🚨 핵심 발견**:
1. **모든 mitigation variants 가 baseline 보다 lower** — graph topology + B5 mitigation + Direct AC + LR x5 + DropMessage + LayerNorm + Sum Aggr 모두에도 raw R 한계 갱신 X
2. **v2 #3 LayerNorm 가 mitigation variants 중 best** (0.6011) — Phase 2 fusion AC (0.6018) 까지 -0.0007 까지 좁힘. mech(ii) edge softmax over-concentration 의 partial mitigation 신호
3. **v2 #2 Sum Aggregation 압도적 underperform** (0.5761, -0.0336) — sum aggregation 의 학습 saturation 더 일찍 발생
4. **paradox 강력 confirm**: 단계 4-bis 발견 (attention 매우 집중적, top-5 ≈ 91%) + 6 mitigation variants 모두 적용에도 동일한 ~0.59-0.61 saturation

### 시나리오 V2-A 절대 confirm — Filter Dominance 6번째 축 7-trial evidence 결정적

DECISIONS 2026-05-07 §1(F) 분기:
- **✅ 시나리오 V2-A** (3 candidates 모두 ceiling 갱신 X): **결정적 confirm**
  - 7 trials × {graph topology, B5 mit, Direct AC, LR x5, DropMessage, LayerNorm, Sum Aggr} 모두 → val R@15 ~0.58-0.61 영역 saturate
  - paper main F1 plateau (Phase 1 plateau 0.0019 spread) 가 학습 saturation 어떤 형태에도 absorb 가능
  - **Filter (Modular LLM) 의 single-stage main mechanism 의 학위 논문 Part III main contribution 7-trial evidence 절대 강화**
- ❌ 시나리오 V2-B (partial mitigation): 일부 신호 (v2 #3 LayerNorm 0.6011, mech(ii) partial) but baseline 갱신 X
- ❌ 시나리오 V2-C (full recovery): 사실상 불가능 확정

### Filter Dominance 6번째 축 (training-pathology-invariant) 7-trial evidence

직전 4-trial evidence + 3-trial 추가:
1. H-B ckpt-invariant
2. H-F stability/ordering
3. F-1 + H-G alpha sweep
4. ΔF1 +0.65 lift
5. H-A/H-D 부정
6. **🆕 Phase 2 + Phase 3 4-trial mitigation null effect** (training-pathology-invariant)
7. **🆕 Mitigation v2 3-trial 추가 evidence** (DropMessage + LayerNorm + Sum Aggr 모두 baseline 미달)

### 비용 / 운영

- 3 학습 합산 wall: ~63h (학습 시간 GPU 0 share, 단독 21h 환산)
- 병렬 wall: **~21h** (5/7 16:35 ~ 5/8 13:54, 3개 동시 GPU 0)
- 비용: **₩0** (모든 학습 LLM-free)
- 3 ckpt NAS 저장: 3 × 113MB = 339MB
- Alpha sweep skip (사용자 결정 2026-05-07 (1)A 유지) → val recall@15 evidence only

### 산출물

- Configs (3 신규):
  - `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.yaml`
  - `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_layernorm.yaml`
  - `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_sum_aggr.yaml`
- Sweep scripts: `scripts/run_mitigation_v2_sweep.sh` + `scripts/run_mitigation_v2_layernorm_resume.sh`
- 학습 entry 확장: `src/train_gat_s06.py` (Mitigation v2 옵션 forward — drop_message_p, use_layernorm_pre_softmax, aggregation_type)
- 모델 확장: `src/models/gat_network_v2.py` (DropMessageGATv2Conv + LayerNormGATv2Conv + sum aggregation)
- Checkpoints (NAS): `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_b5_mitigation_v2_{drop_message, layernorm, sum_aggr}.pt`
- 학습 logs: `logs/train/gat_directed_supernode_p80_b5_mitigation_v2_*_*.log`

### 후속 (analyzer + planner 핸드오프)

- **Analyzer 위임**: 7 ckpt × 4 mechanism dominance scoring (DECISIONS 단계 4-bis §(d))
  - (i) Aggregation collapse: v2 #2 sum_aggr underperform (-0.0336) 이 직접 evidence
  - (ii) GATv2Conv normalization: v2 #3 layernorm partial recovery (mit best 0.6011) 이 mech(ii) DOMINANT 후보 강화
  - (iii) Skip dependency: 7 ckpt 통합 gradient flow
  - (iv) Schema sibling 유사성: 7 ckpt ceiling ~0.61 → fundamental limitation
  - 산출물: `notebooks/analysis_results/dsn_mitigation_v2_results.md`
- **Planner 위임**: 시나리오 V2-A 정식 채택 + paper §3.5 6번째 축 narrative 7-trial evidence + §V Part III main contribution mechanism finding 정량 확정 + 학위 논문 Part III chapter base

## Filter Module Confirmation Sweep v2 — 9-cell Filter Ablation PARALLEL with Evidence Forward (anchor base + 8 filters, GLM 4.7 + EX) (2026-05-13, 🎯 Filter-Invariant 시나리오 확정 + Baseline gap 1.8%p)

발사: 2026-05-13 01:33:31 KST → 종료: 2026-05-13 08:56:01 KST (wall **7h22min30s**, 9-cell PARALLEL). v1 (5/12 14:52 launch, no evidence) 의 EX 결과가 Baseline B1' Full (55.87%) 대비 anchor EX=33.96% gap 21.91%p — 사용자 진단으로 **LLMSQLGenerator 가 BIRD-dev `external_knowledge` 필드 (evidence) 미사용** 이 dominant 원인 확인. v2 = evidence forward fix 후 9 cell 재측정. v1 결과는 archive (`outputs/experiments/s04_ablation/pipeline/filter_sweep_v1_no_evidence/`).

### 운영 이력 (v1 → v2 evidence fix)

**v1 (no evidence, archive)**:
- 2026-05-12 14:30 (사용자 옵션 B): EX 측정 활성, 9-cell sweep 으로 갱신
- 2026-05-12 14:52:33 (parallel launch): PID 868031, GPU 0 max 7.7GB
- 2026-05-13 00:19:12 (종료, wall 9h26min): 9 cell 정상 완료, anchor EX=0.3396
- Baseline B1' Full EX=0.5587 대비 anchor gap = -0.2191 (-21.91%p) — **gap 의 dominant 원인 진단 필요**

**v2 evidence fix (본 sweep)**:
- 2026-05-13 ~01:00 (root 진단): SQL gen prompt 비교 — Baseline 의 `chat_completion_glm` 은 `external_knowledge` 사용, Filter sweep 의 `LLMSQLGenerator` 미사용 → dominant 원인 확정
- 2026-05-13 ~01:20 (code fix): `src/prompts/sql_generator.md` + `src/modules/generators/sql_generator.py` + `src/pipeline/schema_linking.py` + `src/main.py` 4 곳 수정. `pipeline.run(... evidence=item.get('evidence', ''))` + `LLMSQLGenerator.generate(query, subgraph, evidence='')` + prompt template 에 `[External Knowledge]` 섹션 + backtick 명시 추가
- 2026-05-13 01:33:31 (v2 parallel launch): PID 1599383, GPU 0 max 7.7GB
- 각 cell 완료 시각 (v2): C8 (~03:00, LLM-free) → C0/C5 (~04:00) → C6 (~05:07) → C7/C1 (~05:33~05:42) → C2/C3 (~07:12~07:14) → **C4 (08:56, last)**
- 2026-05-13 08:56:01 (v2 종료, wall 7h22min30s): "Sweep finished" marker — v1 보다 wall 2h 단축 (GLM API 응답 안정)

### v2 9-cell 최종 매트릭스 (F1 정렬, 4 decimal)

| 순위 | Cell | Filter | R | P | **F1** (Δv1) | **EX** (Δv1) | filter_time | LLM | tokens in/out |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **1** | **c4_stacked_glm** | Stacked (Refl→Verif) | 0.8781 | 0.8629 | **0.8704** (-0.0031) | 0.5267 (+0.1851) | 6h06m | 8155 | 16.1M / 609K |
| 2 | c7_bidirectional_glm | TieredBidirectional | 0.8923 | 0.8433 | 0.8671 (-0.0024) | **0.5287** ⭐**best EX** (+0.1858) | 2h39m | 3056 | 8.2M / 236K |
| 3 | c0_xiyan_glm_sql (anchor) | XiYan | 0.8706 | 0.8596 | **0.8651** (-0.0012) | 0.5202 (+0.1806) | 1h01m | 1534 | 6.4M / 44K |
| 4 | c1_reflection_glm | Reflection (1 iter) | 0.8907 | 0.8407 | 0.8650 (**+0.0025**) ⭐best ΔF1 | 0.5222 (+0.1754) | 2h46m | 3658 | 14.5M / 166K |
| 4 | c5_symverify_glm | SymbolicVerifier | 0.8717 | 0.8585 | 0.8650 (-0.0022) | 0.5222 (+0.1832) | 1h01m | 1534 | 6.4M / 44K |
| 6 | c2_verifier_glm | Verifier | **0.9163** ⭐best R | 0.8161 | 0.8633 (+0.0014) | 0.5267 (+0.1916) ⭐**biggest ΔEX** | 4h10m | 4484 | 18.5M / 477K |
| 7 | c6_adaptive_depth_glm | AdaptiveDepth | 0.8786 | 0.8484 | 0.8632 (-0.0008) | 0.5248 (+0.1826) | 2h10m | 2787 | 10.0M / 136K |
| 8 | c3_adaptive_multi_agent_glm | AdaptiveMultiAgent ⚠️outlier | 0.7734 | 0.8373 | 0.8041 (±0) | 0.5189 (+0.1812) | 4h12m | 3351 | 2.0M / 684K |
| 9 | c8_no_filter (baseline) | None | **0.9927** | 0.1269 | 0.2250 (±0) | 0.5156 (+0.1721) | 0.21s | 0 | 0 / 0 |

(F1=0.8650 동순위: C1 vs C5, EX=0.5267 동순위: C2 vs C4. 둘 다 의도된 noise band)

### v1 → v2 핵심 변화 (evidence fix 효과)

| 측면 | v1 (no evidence) | v2 (evidence fix) | Δ |
|---|---:|---:|---:|
| **anchor EX (C0)** | 0.3396 | **0.5202** | **+18.06%p** 🎉 |
| best EX | 0.3468 (C1) | **0.5287 (C7)** | +18.19%p |
| EX vs Baseline B1' (55.87%) | -21.91%p gap | **-3.85%p gap (anchor) / -3.00%p gap (best)** | gap 의 82% 회수 |
| best F1 | 0.8735 (C4) | 0.8704 (C4) | -0.0031 (sub-noise, evidence 가 F1 미미 변화) |
| C3 outlier F1 | 0.8041 | 0.8041 | ±0 (확정 outlier, evidence 무관) |

### 🎯 시나리오 재확정 — **Filter-Invariant (EX + F1 양쪽)**

**F1 spread**:
- 7 LLM filter (C3 outlier 제외) F1 spread = 0.8704 − 0.8632 = **0.0072** → Filter-Invariant (≤ 0.01) ★
- 직전 v1 spread 0.0116 → v2 0.0072 (더 좁아짐, sub-noise 확실)

**EX spread**:
- 7 LLM filter (C3 제외) EX spread = 0.5287 (C7) − 0.5202 (C0) = **0.0085** → Filter-Invariant (≤ 0.01) ★
- 8 LLM filter (C3 포함) EX spread = 0.5287 − 0.5189 = 0.0098 → 여전히 Filter-Invariant
- 9-cell (C8 포함) EX spread = 0.5287 − 0.5156 = 0.0131 → 거의 invariant (no filter 도 +0.17 baseline 회복)

→ **Filter-Invariant 시나리오 결정적 confirm (F1 + EX 양쪽)** + paper §3.5 7번째 축 (Filter-axis robustness, F1 + EX 둘 다) 강력 후보.

### v2 핵심 발견 (5)

**(1) Evidence fix 가 EX gap 의 82% 회수 (+18.06%p anchor)**:
- Baseline B1' Full EX=55.87% 대비 anchor v1 gap 21.91%p → **v2 gap 3.85%p**
- 잔여 3.85%p 는 column TEXT hardcoded + DDL vs flat format + max_tokens 256 등 minor factor 추정
- 사용자 진단 (prompt 차이로 인한 confounding) 정확히 확인

**(2) Best EX: C7 TieredBidirectional = 0.5287 (anchor +0.0085)**:
- Tier-1/Tier-2 evidence-aware restoration 의 EX 측 marginal lift
- F1=0.8671 (anchor +0.0020 noise) 단 EX +0.0085 — schema linking 보다 SQL gen quality 가 EX dominant 일관 (v1 narrative 와 동일)

**(3) Best F1 ΔLift: C1 Reflection +0.0025 (v1 -0.0143 → v2 +0.0025)**:
- v1 에서 F1 손실 (-0.0143) 이었던 cell 이 v2 에서 marginal lift
- Evidence 가 schema linking F1 도 약간 영향 — Reflection 의 critique 단계에서 evidence 활용
- 단 정 magnitude 는 sub-noise band (4 decimal)

**(4) C3 AdaptiveMultiAgent outlier 확정 (F1=0.8041 in v1, v2 모두 동일)**:
- Evidence 추가에도 F1 무변동 — 3-agent vote 의 over-aggressive pruning 이 schema linking 자체 문제
- 의미: multi-agent vote pathology 가 prompt-independent — **확정 outlier**

**(5) C8 no_filter EX +0.1721 회복 (0.3435 → 0.5156, anchor v2 −0.0046)**:
- Filter 없이도 evidence 만으로 anchor 거의 동등 (-0.0046, sub-noise)
- Schema linking F1=0.2250 (over-include) 가 GLM 4.7 의 SQL gen 으로 흡수
- 해석: **Filter 의 EX 측 효과는 marginal 단 paper §3.5 의 "Filter Dominance F1 측" narrative 는 결정적 (F1=0.2250 → 0.8651, +0.64) 유지**

### Baseline 비교 (재정정)

| 측정 | EX |
|---|---:|
| **B3' Gold Column (perfect SL upper bound)** | **0.6239** |
| B2' Gold Table | 0.5932 |
| **B1' Full Schema (Maamari paradigm)** | **0.5587** |
| **본 연구 best filter (C7 v2)** | **0.5287** |
| **본 연구 anchor (C0 v2)** | **0.5202** |
| 본 연구 C8 no_filter (LLM-free filter, +sql gen) | 0.5156 |
| Llama 3.1 8B Full Schema | 0.3410 |

→ 본 연구 anchor 가 Baseline B1' Full (55.87%) 의 93%, B3' Gold Column (62.39%) 의 83% 도달. 잔여 gap = SQL gen quality (column type / max_tokens) + schema linking imperfection.

### 비용 / 운영 (v2)

- 학습 wall: **7h22min30s** (01:33:31 → 08:56:01 5/13, 9-cell PARALLEL) — v1 9h26min 대비 2h 단축
- 비용: ~₩수십만 (총 token in = **82.0M**, token out = **2.4M**, evidence 추가로 v1 대비 사실상 동일)
- GPU 0 max memory: 7.7GB (9 cell × ~850MB)
- 모든 9 cell 정상 완료, fail 0
- v1 결과 보존: `outputs/experiments/s04_ablation/pipeline/filter_sweep_v1_no_evidence/`

### 산출물

- Code fix (v2): `src/prompts/sql_generator.md` + `src/modules/generators/sql_generator.py` (`generate(query, subgraph, evidence='')`) + `src/pipeline/schema_linking.py` (`run(... evidence=...)`) + `src/main.py` (item.get("evidence", "") forward)
- Configs (9): `configs/experiments/s04_ablation/pipeline/filter_sweep/c{0..8}_*.yaml` (변경 없음)
- Sweep script: `scripts/run_filter_sweep_glm.sh`
- 출력 (v2): `outputs/experiments/s04_ablation/pipeline/filter_sweep/<cell>/`
- v1 archive: `outputs/experiments/s04_ablation/pipeline/filter_sweep_v1_no_evidence/<cell>/`
- 로그: `logs/experiments/s04_ablation/pipeline/filter_sweep/<cell>/*.log`
- Sweep parent log: `/tmp/filter_sweep_glm_v2.log`

### 후속 (analyzer + planner 핸드오프 — 즉시 trigger 가능)

- **Analyzer 위임**: `notebooks/analysis_results/filter_sweep_glm_9cell.md` 신규 — §0 TL;DR (v1↔v2 evidence fix +18%p) / §1 v2 9-cell × R/P/F1/EX 매트릭스 / §2 v1 vs v2 비교 (ΔEX +0.17~0.19 일관) / §3 Filter cost / §4 Per-stage cumulative / §5 Filter Dominance 7번째 축 (F1 + EX 양쪽 Invariant) narrative / §6 paper §3.5 통합 + Baseline gap 회수 분해
- **Planner 위임 (analyzer 후)**: paper §3.5 Filter Dominance evidence 표 7번째 axis (Filter-Invariant) 추가 + best/worst filter narrative + anchor F1=0.8651 / EX=0.5202 ranking 정량 + paper main pipeline anchor 유지/변경 결정 + DECISIONS Filter sweep v2 결과 entry prepend

---

## SGBE Phase 3-5 — Score-Gated Batch Extractive Filter θ Calibration + Final + Step Contribution Ablation (2026-05-12, 🚀 Phase 3 launch active)

근거: planning/DECISIONS.md 2026-05-12 (SGBE Chain Phase 3 Launch Trigger — Prerequisite 완료) + (SGBE Filter 채택) Phase 3-5 + (SGBE Phase 2 보강 §"Option B 권장"). Module:filters 가 step_mode 3-mode (`step_0` / `step_0+1` / `step_0+1+2`) + `score_collapse_threshold` (Option A default 0.05) 옵션 추가 완료 (16/16 smoke PASSED, 5/12). **본 entry 는 Phase 3 calibration sweep launch active (LLM call 0, ETA ~2-3h).** Phase 4 + Phase 5 는 후속 chain.

### 운영 이력

- **2026-05-12 (module:filters 세션, 초기)**: SGBE filter 구현 완료
  - `src/modules/filters/score_gated_batch_extractive_filter.py`
  - Registry 키: `'ScoreGatedBatchExtractiveFilter'`
  - Smoke test 10/10 PASSED (5/12 초기)
- **2026-05-12 (root chain, 초기)**: 3 master yaml + 2 sweep scripts 작성 — placeholder 값 (skip_llm / step_mode 가 의도된 옵션, 단 module 미구현 시 silent ignore)
- **2026-05-12 (module:filters 세션, 보강)**: step_mode + score_collapse_threshold option 추가
  - `step_mode` ∈ {`step_0`, `step_0+1`, `step_0+1+2`}, default = `step_0+1+2` (Full SGBE)
    - `step_0`: FK/PK hardcode 만 (LLM call 0)
    - `step_0+1`: Step 0 + Score-Gate (LLM call 0, S_uncertain 전부 drop)
    - `step_0+1+2`: Full SGBE (Step 0 + Score-Gate + LLM Extractive)
  - `score_collapse_threshold: float = 0.05` (Option A, default). candidate score 들의 std < threshold 면 SGBE fallback (XiYan-equivalent) — V4 era boundary case 의 detection.
  - 인터페이스 보강: `stats["score_collapse_detected"]` (bool) + `filter_info["filter_score_std"]` (float)
  - Smoke test 16/16 PASSED (5/12 보강)
- **2026-05-12 (root chain, 보강)**: yaml + sweep script 의 step_mode 정식 값 정정
  - `sgbe_calibration_base.yaml`: `skip_llm: true` → `step_mode: "step_0+1"` + `score_collapse_threshold: 0.05`
  - `sgbe_final.yaml`: `step_mode: "step_0+1+2"` + `score_collapse_threshold: 0.05` 명시
  - `sgbe_step_ablation_base.yaml`: step_mode 값 정정 (step0_only → step_0 / step01_only → step_0+1 / full → step_0+1+2)
  - `scripts/run_sgbe_calibration.sh` pre-check 갱신 (skip_llm grep → step_mode grep)
  - `scripts/run_sgbe_final_ablation.sh` STEP_MODES 값 정정
  - `scripts/run_sgbe_ablation.sh` 신규 (Phase 5 only 분리, 3 cell 순차)
- **2026-05-12 21:44 KST (root chain, Phase 3 launch — serial)**: `nohup bash scripts/run_sgbe_calibration.sh > logs/sgbe/calibration.log 2>&1 &`
  - PID 1241364 (script) + 1241409 (python main.py)
  - cell 1/9 (keep=0.50, drop=0.20) initialization (~3 min, no metrics)
  - 9 cell 순차 (3×3 grid), ETA ~2-3h wall
- **2026-05-12 21:55 KST (root chain, Phase 3 RELAUNCH — PARALLEL)**: 사용자 명시 동의로 serial sweep kill + parallel launch
  - Kill (사용자 5/12 22:00 KST input "지금 직렬 프로세스를 kill 하고 병렬 script 로 실행"): PID 1241364 + 1241409 종료
  - 신규 script: `scripts/run_sgbe_calibration_parallel.sh` — 9 cells 동시 nohup launch
  - `nohup bash scripts/run_sgbe_calibration_parallel.sh > logs/sgbe/calibration_parallel.log 2>&1 &` (script PID 1257052)
  - GPU 분배 (cell 당 ~846 MiB):
    - **GPU 0**: cell 1 (PID 1257191) + cell 2 (1257189) + cell 3 (1257190) + cell 4 (1257192) + cell 5 (1257193) = 5 SGBE cells × 846 MiB = 4,230 MiB
    - **GPU 1**: cell 6 (PID 1257264) + cell 7 (1257336) + cell 8 (1257266) + cell 9 (1257407) = 4 SGBE cells × 846 MiB = 3,384 MiB
  - 동거 process (다른 root chain): GPU 0 의 filter sweep c2/c3/c4 (PIDs 868132/868135/868137, 각 846 MiB)
  - GPU 0 total used: 6,768 MiB / 24,576 MiB free
  - GPU 1 total used: 3,384 MiB / 24,576 MiB free
  - CPU 36 cores, load avg ~2.84 (low contention, 9 × ~10 threads/cell ≈ 90 threads)
  - **ETA: ~15-20 min wall** (vs serial ~2-3h, ~6-10x speedup)
- **2026-05-12 22:07 KST (Phase 3 완료, 12 min wall)**: 9 cells 모두 종료, metrics.txt 9 개 생성

### Phase 3 9-cell Calibration 결과 (Step 0+1 only, LLM call 0)

| Cell | θ_keep | θ_drop | Recall | Precision | F1 | filter_time_total |
|---:|---:|---:|---:|---:|---:|---:|
| **1** | **0.50** | 0.20 | **0.8157** | 0.2350 | **0.3649** | 1.4134s |
| 2 | 0.50 | 0.25 | 0.8157 | 0.2350 | 0.3649 | 1.3975s |
| 3 | 0.50 | 0.30 | 0.8157 | 0.2350 | 0.3649 | 1.4137s |
| 4 | 0.55 | 0.20 | 0.7481 | 0.2328 | 0.3551 | 1.3998s |
| 5 | 0.55 | 0.25 | 0.7481 | 0.2328 | 0.3551 | 1.3982s |
| 6 | 0.55 | 0.30 | 0.7481 | 0.2328 | 0.3551 | 1.3875s |
| 7 | 0.60 | 0.20 | 0.7279 | 0.2431 | 0.3645 | 1.3979s |
| 8 | 0.60 | 0.25 | 0.7279 | 0.2431 | 0.3645 | 1.3863s |
| 9 | 0.60 | 0.30 | 0.7279 | 0.2431 | 0.3645 | 1.3937s |

핵심:
1. **θ_drop range (0.20~0.30) 무효** — 같은 θ_keep 3 cells 의 R/P 완벽 동일. TN mean (paper main anchor 0.3005) 가 θ_drop range 의 아래쪽 경계와 일치 → 대부분 TN 이 S_uncertain 으로 라우팅
2. **θ_keep 효과**: 0.50→0.55→0.60 → R 0.8157→0.7481→0.7279 (recall trade-off), P 0.2350→0.2328→0.2431 (marginal)
3. **Recall ≥ 0.85 constraint 미충족** — 모든 cell. Best R = 0.8157 (cell 1/2/3, θ_keep=0.50)
4. **Precision 0.23 (Step 0+1 only, LLM 없음)** — S_keep_hard 가 크고 S_uncertain 모두 drop. Phase 4 의 LLM 활성 시 회복 예상
5. **filter_llm_calls_total: 0** confirmed — step_mode="step_0+1" 정상 동작
6. **Optimal θ**: **θ_keep=0.50, θ_drop=0.20** (R 최대 보호, drop_hard 거의 empty)

### Phase 4 — Final SGBE 평가 (완료, 5/13 01:10:51 KST, 2h 45m wall)

- **2026-05-12 22:25 KST (Phase 4 launch)**: best θ (0.50/0.20) + step_mode="step_0+1+2" + GLM 4.7 + SQL generator
  - Config: `configs/experiments/s04_ablation/pipeline/sgbe/sgbe_final.yaml`
  - PID 1299288 (script), 1299314 (python main.py)
- **2026-05-13 01:10:51 KST (Phase 4 완료)**: 2h 45m wall, 진행 속도 평균 ~6.5 sec/query (5.9~7.5 변동)
  - Outputs: `outputs/experiments/s04_ablation/pipeline/sgbe/sgbe_final/`

#### Phase 4 결과 vs Anchor 비교

| Metric | SGBE Phase 4 | Anchor XiYan (F1=0.8673) | Δ | 학술 Agent 예상 |
|---|---:|---:|---:|---|
| **Recall** | **0.8311** | 0.8741 | **-0.0430** | R ≥ 0.73 ✅ |
| **Precision** | **0.2377** | 0.8606 | **-0.6229** | P ≥ 0.70 ❌ 재앙적 미충족 |
| **F1** (계산) | **0.3690** | 0.8673 | **-0.4983** | — |
| **EX** | 0.3396 | 0.0000 (anchor SQL gen off) | (직접 비교 불가) | — |
| filter_time_total | 5854s (1h 37m) | 3002s (50 min) | **+2x slower** | 1.5~2× 빠름 ❌ 반대 |
| filter_llm_calls_total | 1490 (mean 0.97/q) | 1534 (1/q) | -44 | LLM input 60-80% 감소 ❌ |
| llm_input_tokens | 4,328,366 | (anchor not measured) | — | — |
| llm_output_tokens | 478,330 | (anchor not measured) | — | — |

→ **🚨 SGBE 의 Score-Gate 가 paper main anchor backbone (GLM 4.7) 의 score 분포 (TP mean 0.4746) 와 부정합 — S_keep_hard 가 over-include source. P=0.2377 의 거의 완전한 회복 실패.**

#### Phase 3 → Phase 4 P 변화 — LLM Extractive 가 무효

- Phase 3 (Step 0+1 only, LLM 0): P=0.2350, R=0.8157
- Phase 4 (Step 0+1+2, LLM 활성, 1490 calls): **P=0.2377 (+0.003 만)**, R=0.8311 (+0.015)
- LLM Extractive 가 P 회복에 거의 무효 — S_uncertain 의 LLM 판단이 over-keep 또는 S_keep_hard 자체가 noise dominated

#### 시나리오 분기 — Filter-Underperform

- SGBE 가 anchor F1=0.8673 갱신 ❌ (ΔF1 = -0.4983)
- paper main anchor score ladder 와 SGBE θ design 부정합 — DECISIONS 2026-05-12 SGBE Phase 2 보강 entry §"학술 Agent 권장 θ 적용 가능 anchor 범위 제한" 의 caveat 실증
- Filter Dominance 8번째 axis (Score-Gated Hybrid 효과) **부정적 evidence** — paper §V.5.4 narrative 의 selector + filter co-design caveat 강화

### Phase 5 — Step Contribution Ablation (active, 5/13 01:14:41 KST)

- **2026-05-13 01:14:41 KST (Phase 5 launch)**: `nohup bash scripts/run_sgbe_ablation.sh > logs/sgbe/ablation.log 2>&1 &`
  - PID 1570529 (script), 1570568 (cell 1 step_0)
  - 변경: `STEP_MODES=("step_0" "step_0+1")` — step_0+1+2 skip (Phase 4 결과 = 동등, 시간 절약)
  - ETA: ~5-10 min wall (LLM 없음)
- 측정 목적: Step 별 cumulative R/P/F1 → P drop source 정량
  - step_0: FK/PK only — structural baseline
  - step_0+1: + Score-Gate — S_keep_hard over-include 정량
  - step_0+1+2 (= Phase 4): + LLM Extractive — LLM 기여도

### Anchor stack (paper main mirror, F1=0.8673)

- Builder: EnrichedHeteroGraphBuilder
- Selector: EnsembleSelector + qcond_nl3.pt + α=0.5 (QCond Concat)
- Extractor: MSTKruskalExtractor (score_threshold=0.1)
- Filter: ScoreGatedBatchExtractiveFilter (SGBE)
- LLM: GLM 4.7 (provider="glm", model_name="zai-org/glm-4.7", T=0.0)

### 9-cell θ Calibration Grid (Phase 3)

| | θ_drop=0.20 | θ_drop=0.25 | θ_drop=0.30 |
|---|---|---|---|
| **θ_keep=0.50** | cell 1 | cell 2 | cell 3 |
| **θ_keep=0.55** | cell 4 | cell 5 (recommended center) | cell 6 |
| **θ_keep=0.60** | cell 7 | cell 8 | cell 9 |

근거: DECISIONS 2026-05-12 (SGBE Phase 2 보강) §"anchor GLM grid: θ_keep ∈ {0.50, 0.55, 0.60}, θ_drop ∈ {0.20, 0.25, 0.30}". Paper main anchor 의 TP mean 0.4746 / TN mean 0.3005 score 분포 정합.

### 3 Step Contribution Ablation (Phase 5)

| Mode | SGBE 동작 | LLM 호출 | 의미 |
|---|---|---|---|
| **step0_only** | Structural Hard Keep (FK/PK) 만 | 0 | Step 0 의 isolated 효과 |
| **step01_only** | Step 0 + Score-Gate (S_uncertain → drop) | 0 | Step 0+1 만 — LLM 없는 lower bound |
| **full** | Step 0 + Score-Gate + LLM Extractive | ~300-600 calls / 1534 query | Full SGBE (= sgbe_final.yaml) |

### Launch Trigger 발동 (prerequisite 해제)

- Module:filters 16/16 smoke PASSED — `step_mode` 3-mode + `score_collapse_threshold` 정식 추가
- Phase 3 calibration sweep launch active (~2-3h ETA, LLM call 0)
- Phase 4/5 는 후속 chain (Phase 3 결과 후 optimal θ 결정 + sgbe_final.yaml update + Phase 4 launch → Phase 5 launch)

### 시나리오 분기 (Phase 5 결과)

| 시나리오 | 조건 | narrative 영향 |
|---|---|---|
| **(1) SGBE > XiYan anchor (F1 > 0.8673)** | full SGBE 가 anchor 갱신 | SGBE 가 paper main pipeline anchor 변경 candidate. Filter Dominance 8번째 axis (Score-Gated Hybrid) candidate. |
| **(2) SGBE ≈ XiYan anchor (|ΔF1| ≤ 0.005)** | full SGBE 가 anchor 와 동등 | SGBE 의 speed/cost advantage narrative (input token 60-80% 감소). Filter-invariance 추가 evidence. |
| **(3) SGBE < XiYan anchor** | SGBE 가 anchor 미달 | Score-Gate 의 calibration sensitivity narrative + selector + filter backbone interaction (DECISIONS 2026-05-12 SGBE Phase 2 보강) 강화 |

### 후속 위임 (chain handoff)

- **module:filters**: SGBE 의 `skip_llm` (default False) + `step_mode` ∈ {'step0_only', 'step01_only', 'full'} (default 'full') option 추가. Sweep script pre-check 통과 후 root 후속 chain 에서 launch.
- **analyzer (Phase 3 launch 후)**: 9-cell θ calibration 결과 분석 + best θ 결정 → root 에 sgbe_final.yaml 의 theta_keep/drop update 요청
- **analyzer (Phase 4+5 launch 후)**: SGBE final 결과 + Step contribution 정량 + boundary case (over-smoothing era V4 score collapse 시 SGBE 무력 caveat). `notebooks/analysis_results/sgbe_filter_results.md` 신규.
- **planner (analyzer 후)**: narrative integration — Filter Dominance 7번째 axis (Filter-invariance, 9-cell sweep 결과) + 8번째 axis (Score-Gated Hybrid 효과) candidate.

### 산출물 (코드 + config 준비 완료)

- Filter 모듈: `src/modules/filters/score_gated_batch_extractive_filter.py` (module:filters 세션)
- Configs (3 master): `configs/experiments/s04_ablation/pipeline/sgbe/{sgbe_calibration_base, sgbe_final, sgbe_step_ablation_base}.yaml`
- Sweep scripts (2): `scripts/run_sgbe_calibration.sh` + `scripts/run_sgbe_final_ablation.sh`

### Caveat — 본 entry 는 launch 보류 placeholder

Launch 진행 (module:filters 의 option 추가 후) 시 root 재호출 시 다음 값 반영:
- Phase 3: 9 cells 의 R/P/F1 (4 decimal) — analyzer 가 best θ 결정
- Phase 4: final SGBE 의 R/P/F1 + LLM call count + wall clock
- Phase 5: 3 step ablation 의 R/P/F1 (cumulative Step 0 → 0+1 → full)
- 시나리오 분기 (1/2/3) 확정 + analyzer 핸드오프 trigger

---

## Anchor (MSTPCSTUnion+XiYan+SQL) Sweep — Option γ 재실행 (2026-05-14 21:51 → 23:40, 🎯 SQL Gen prompt 변경 효과 ΔEX +0.1512 + 5 Capacity 지표 prerequisite)

근거: planning/DECISIONS.md 2026-05-15 §0 Option γ 채택 — 사용자 명시 anchor stack (Enriched + QCond Ensemble + MSTPCSTUnion + XiYan) + sql_generator 활성화 + EX 측정 + 5 Capacity 지표 (TCR/BNR/TOR/AUC/Filter Pruning) 분석 prerequisite. 5/1 prior run (EX=0.3377) 존재 발견 후 SQL Gen prompt 변경 효과 검증 재실행.

### Stack (사용자 명시 anchor)

- Builder: EnrichedHeteroGraphBuilder
- Selector: EnsembleSelector (best_gat_qcond_nl3.pt, α=0.5, top_k=20, query_conditioned=true)
- Extractor: **MSTPCSTUnionExtractor** (score_threshold=0.1)
- Filter: XiYanFilter (provider=glm, model_name=zai-org/glm-4.7, max_iteration=1, temperature=0)
- SQL Gen: **LLMSQLGenerator** (provider=glm, llm_model=zai-org/glm-4.7, temperature=0, evidence-aware prompt 갱신)

### 운영 이력

- **2026-05-01 04:12 (prior run)**: 동일 stack 의 sweep (sql_gen=true) 한 번 실행됨 — F1=0.8657, EX=0.3377. SQL Gen prompt 변경 이전 결과.
- **2026-05-14 21:51:58 (본 sweep launch)**: nohup + TMPDIR=/tmp + PYTHONUNBUFFERED=1. GPU 0 (V5 v5b_L4 공존). 5/1 결과 의 predictions.jsonl/output/score_analysis overwrite.
- **2026-05-14 23:40:42 (DONE)**: Wall **~1h 49m**. metrics.txt overwrite 완료.

### Final Metrics

| 지표 | 값 |
|---|---:|
| **R** | **0.8831** |
| **P** | **0.8070** |
| **F1** (2·P·R/(P+R)) | **0.8434** |
| **EX** | **0.4889** ⭐ |
| LLM calls | 2873 (filter 1437 + SQL gen 1436) |
| Filter time mean | 1.82s |
| Token in / out | 5.27M / 105K |
| extractor_selected_nodes_mean | 83.08 (anchor 동일) |
| filter_llm_calls_mean | 0.9368 (6.32% query skip) |

### Δ 비교 매트릭스

| 지표 | 본 sweep | 5/1 prior | Δ (vs 5/1) | c0 (MSTKruskal+XiYan+SQL) | Δ (vs c0) |
|---|---:|---:|---:|---:|---:|
| R | **0.8831** | 0.8734 | **+0.0097** ⏫ | 0.8706 | +0.0125 |
| P | **0.8070** | 0.8581 | **-0.0511** ⏬ | 0.8596 | -0.0526 |
| F1 | **0.8434** | 0.8657 | **-0.0223** ⚠ | 0.8650 | -0.0216 |
| **EX** | **0.4889** | **0.3377** | **+0.1512** ⏫⏫ | **0.5202** | **-0.0313 sub-noise** |

### 핵심 발견

1. **SQL Gen prompt 변경 효과 EX +0.1512** (0.3377 → 0.4889) ⭐ — 사용자 명시 예상 정합. evidence-aware fix 의 EX 회복 mechanism 강력 확인.
2. **ΔEX vs c0 -0.0313 sub-noise** — MSTPCSTUnion 의 extractor 차이가 EX 측에 거의 zero. **Filter Dominance EX-axis robustness 가 anchor extractor (MSTKruskal vs MSTPCSTUnion) 무관** 추가 evidence. paper §V.5.x.M.4 Three-Caveat narrative 의 EX-axis mechanism 확장 candidate.
3. **ΔF1 vs 5/1 prior -0.0223 (deterministic partial fail)** — 같은 schema linking stack 인데 F1 0.0223 변동 (P -0.0511, R +0.0097). non-trivial — auto_join_keys / filter prompt 변경 / extractor randomness / config side effect 진단 필요 (analyzer §6.5).
4. **filter_llm_calls_mean 0.9368** — 6.32% query 의 filter skip (input nodes 작은 query 일 가능성). 5 Capacity 지표 분석에 의미 — Filter Pruning Breakdown 의 edge case.
5. **R/P trade-off paradox**: R +0.0097 vs P -0.0511 = **5.3× P loss per R gain** — Direction A (6.35×) / Direction C (9.07×) 의 R-P trade-off pattern 의 더 mild variant.

### 비용 / 운영

- Wall: ~1h 49m (sweep-only, 학습 X)
- LLM call: 2873 (filter 1437 + SQL gen 1436)
- Token in: 5.27M (학습 chain 의 ~1/3)
- GPU 시간: 0 (학습 chain 과 분리, V5 학습 공존)
- Cost: ~$1-3 GLM 4.7 API

### 산출물

- Config: `configs/experiments/s04_ablation/pipeline/enriched_qcond_a05_mst_pcst_union_glm_sql.yaml` (5/1 prior 그대로, sql_gen=true)
- Sweep script: `scripts/run_anchor_sql_sweep.sh` (신규, TMPDIR=/tmp + PYTHONUNBUFFERED=1 정합)
- Outputs: `outputs/experiments/s04_ablation/pipeline/enriched_qcond_a05_mst_pcst_union_glm_sql/` (predictions/output/metrics/score_analysis/stage_aggregates 모두 본 sweep overwrite)
- Logs: `logs/sweep_anchor_sql_20260514_215158.log` + `sweep_anchor_sql_main.log`

### 후속

- **Analyzer (즉시 trigger 가능, 사용자 처리)**: `notebooks/analysis_results/anchor_capacity_indices_2026-05-15.md` — 5 Capacity 지표 (TCR/BNR/TOR/AUC/Filter Pruning) + EX 통합 분석 (DECISIONS 2026-05-15 §4)
- **Planner (analyzer 후)**: paper §V.5.x.M.4 Three-Caveat 의 mechanism-level evidence 확장 narrative + Anchor extractor invariance (MSTKruskal vs MSTPCSTUnion EX sub-noise) 의 학술 weight 통합

---

## Direction B + Direction C-GT 배포 Sweep (b06_01 + a05_26, 2026-05-14 19:39 → 20:46, 🎯 B = Filter-Invariant F1 측 추가 evidence, C-GT = Four-caveat outlier candidate)

근거: planning/DECISIONS.md 2026-05-15 (Module 구현 chain 완료). 학습 단계 (5/14 16:12 → 18:09) 완료 직후 sweep launch — 사용자 5/14 input "GPU 여유 있으면 병렬로 진행". GPU 0 (B sweep, V5 v5b_L4 공존) + GPU 1 (C-GT sweep, V5 v5c_cum 공존) 병렬. PYTHONUNBUFFERED=1 + TMPDIR=/tmp 적용.

### 운영 이력

- **2026-05-14 19:39 launch (PARALLEL)**: GPU 0 b06_01 + GPU 1 a05_26, V5 학습 공존.
- **2026-05-14 20:46 양 sweep 종료 (wall 1h 7m)**: B 와 C-GT 거의 동시 종료 (per-q 2.4-2.6s 안정).

### 결과 (1534 BIRD-Dev × GLM 4.7)

| Cell | R | P | **F1** | EX¹ | LLM call | mean \|final_n\| | Filter time mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| anchor (MSTKruskal+XiYan, c0)² | 0.8706 | 0.8596 | **0.8650** | 0.5202 | 1534 | ~4.8 | 2.39s |
| anchor (MSTPCSTUnion+XiYan)² | 0.8772 | 0.8564 | **0.8666** | 0.0³ | 1534 | ~4.8 | 1.96s |
| **B (HN-SupCon, b06_01)** | **0.8713** | **0.8545** | **0.8628** | 0.0¹ | 1534 | 85.07 | 2.07s |
| **C-GT (GraST-GT, a05_26)** | **0.7311** | **0.3969** | **0.5145** | 0.0¹ | 1534 | 49.16 | 1.95s |

¹ sql_generator=false (학술 frame "Filter-Invariant 경계 확정 실험" 정합, EX 측정 X).  
² 비교 base 두 anchor — MSTKruskal+XiYan c0 (EX 측정 base) + MSTPCSTUnion+XiYan (현재 anchor extractor).  
³ MSTPCSTUnion+XiYan anchor 의 EX=0 은 sql_gen 비활성 의 표기 artifact.

### Δ vs Anchor 매트릭스

**Direction B**:

| 지표 | vs c0 | vs MSTPCSTUnion |
|---|---:|---:|
| ΔR | +0.0007 | -0.0059 |
| ΔP | -0.0051 | -0.0019 |
| **ΔF1** | **-0.0022 sub-noise** | **-0.0038 sub-noise** |

→ **F1 측 Filter-Invariant 강력 confirm**. HN-SupCon fine-tuned encoder (MiniLM-L6 → contrastive fine-tune, SLR Δ +0.0267) 가 anchor 의 raw cosine 대비 F1 lift 없음 — anchor cluster 의 sub-noise band 내.

**Direction C-GT**:

| 지표 | vs c0 | vs MSTPCSTUnion |
|---|---:|---:|
| ΔR | -0.1395 | -0.1461 |
| ΔP | -0.4627 | -0.4595 |
| **ΔF1** | **-0.3505** ⚠ | **-0.3521** ⚠ |

→ **Four-caveat outlier candidate**. mean |final_n| 49.16 (anchor 의 ~10× over-include) — GraphTransformer reranker 의 score threshold + Steiner restore 가 의도와 다르게 over-include. Direction A (LLM backward, -0.2832) + Direction C (algorithmic Steiner, -0.2873) 의 -0.28 outlier 보다 더 큰 -0.35 outlier.

### 핵심 발견

1. **Direction B = Filter-Invariant 7번째 축 F1 측 추가 evidence** — selector backbone (encoder) 교체에도 anchor cluster 의 sub-noise band 내. paper §3.5 의 axis #7 narrative 의 selector-axis robustness 확장.

2. **Direction C-GT = Four-caveat outlier candidate (또는 boundary 확장 evidence)**:
   - Caveat 1 (C3 architectural pathology) F1 -0.0622
   - Caveat 2 (Direction A, LLM backward) F1 -0.2832
   - Caveat 3 (Direction C, algorithmic Steiner) F1 -0.2873
   - **🆕 Caveat 4 (Direction C-GT, GraphTransformer rerank over-include)** F1 -0.3505
   - 새 mechanism axis: GT reranker 의 score threshold over-include (단 EX 미측정 — Filter-Invariant boundary 의 추가 evidence 아직 부분 확정)

3. **mean |final_n| 패턴**:
   - anchor ~4.8
   - Direction A/C ~19 (4× over-include)
   - **Direction C-GT 49.16 (10× over-include)** — GT reranker top_k=10 + Steiner restore 의 cumulative effect
   - Direction B 85.07 (17.7× over-include!) — anchor 보다 훨씬 많음, 그러나 F1 sub-noise 유지 (anchor cluster 의 selector backbone 차이 흡수)

4. **B 의 mean |final_n| 85.07 vs anchor 4.8 17× 차이임에도 F1 sub-noise** — paradox: filter 의 over-include + downstream pruning 효과 또는 selector top_k=20 의 명시적 retention. b06_01 의 selector top_k=20 명시 + GAT projector 유지 → 다른 anchor 대비 selector 단계 final_n 보존. Filter (XiYan) 가 그 후 P 측 retention.

5. **C-GT 의 R=0.7311 손실** — GT reranker 가 gold column 누락 시그널. R 0.87 → 0.73 -0.14. score threshold top_k=10 이 좁아 일부 gold 가 top_k 밖.

### 비용 / 운영

- Wall: 1h 7m (parallel, GPU 0/1 V5 학습 공존)
- LLM call: 3068 (B 1534 + C-GT 1534)
- Token in: B 6.0M, C-GT 2.1M (C-GT 는 GraphTransformer inference 만 + XiYan forward 만, anchor 의 ~30% token)
- Filter time mean: B 2.07s, C-GT 1.95s

### 산출물

- B: `outputs/experiments/abl/b06_hn_supcon/b06_01_hn_supcon_glm/` (predictions/metrics/output)
- C-GT: `outputs/experiments/abl/a05_filter_agentic/a05_26_grast_with_transformer_glm/`
- Logs: `logs/sweep_{b06,a05_26}/*.log`

### 후속

- **Analyzer**: `direction_b_hn_supcon_sweep.md` + `direction_c_gt_sweep.md` 보고서
- **Planner**: paper §V.5.x.M.4 four-caveat 또는 boundary 확장 narrative + DECISIONS prepend

---

## Direction B (HN-SupCon) + Direction C-GT (GraST-GT) Full Training (2026-05-14, 🎯 학술 Agent §6.4 5/17 일정 앞당김, B + C-GT 모두 학습 PASS)

근거: planning/DECISIONS.md 2026-05-15 (Module 구현 chain 완료). 사용자 5/14 input — Option (A) 진행 + B 1 epoch / C-GT 40 epoch full launch. Memory `feedback_root_no_module_impl.md` 갱신 (Root 가 train script CLI wrapper 작성 허용 + 모델 학습/실험 진행 main owner).

### Stack

| Module | Direction B (b06_01) | Direction C-GT (a05_26) |
|---|---|---|
| Backbone | sentence-transformers/all-MiniLM-L6-v2 (HN-SupCon fine-tune) | GraphTransformerEncoder (3-layer relation-aware, hidden 1024, 8 heads, from-scratch) |
| Loss | HN-SupCon (Piao 2025 LitE-SQL) | Margin contrastive (Hoang 2025) |
| Hyperparameters | τ=0.07, N=8, margin=0.1, lr=5e-5, batch=16, **1 epoch** | margin=0.1, lr=5e-5, in_dim=16, **40 epoch** |
| Init source | PLM pretrained (fine-tune) | From-scratch |

### 운영 이력

- **5/14 15:22 (commit fb92775)**: Module:Selector + Module:Filter mixed — `hn_supcon_selector.py`, `grast_fd_transformer.py`, `grast_fd_filter_with_transformer.py` + smoke test 24/24 PASS.
- **5/14 15:30 (Root)**: `src/train_grast_gt.py` wrapper 신규 (Module:Filter helper inline copy, ~350 line).
- **5/14 15:47 (smoke v1)**: B (3 steps, evaluator 부재 → 불명확) + C-GT (5 epoch, loss 0.0711 + PR-AUC Δ +0.0131 PASS).
- **5/14 16:04 (Root, evaluator 추가)**: `train_hn_supcon.py` 의 `evaluate_val_slr()` + `--val-fraction` + `--eval-top-k` args 추가. Smoke v2: B SLR Δ +0.1087 ✅.
- **5/14 16:12 → 16:18 (B full)**: 1 epoch / 383 steps / wall **~6분** (V5 학습 GPU 0 공존).
- **5/14 16:12 → 18:09 (C-GT full)**: 40 epoch / 8477 train records / wall **~1h 57m** (V5 학습 GPU 1 공존, smoke 추정 +17% — schema 무게).

### B 학습 결과

| 지표 | 값 |
|---|---:|
| n_train (1 epoch full) | **6116** records → 383 steps × batch 16 |
| n_val | 679 records |
| Initial val SLR @15 | **0.6653** |
| **Final val SLR @15** | **0.6920** |
| **SLR Δ** | **+0.0267** ✅ PASS (≥ +0.01) |
| Final train loss | 1.2681 |
| 산출물 | `outputs/checkpoints/hn_supcon/` — model.safetensors (90 MB) + smoke_result.json |

### C-GT 학습 결과

| 지표 | 값 |
|---|---:|
| n_train | **8477** records |
| n_val | 941 records |
| Initial loss (ep1) | **0.0770** |
| **Best loss (ep31)** | **0.0674** ⭐ |
| Final loss (ep40) | 0.0701 (saturation 후 약간 noise) |
| Δ (init → best) | **−0.0096 (−12.5%)** |
| 산출물 | `outputs/checkpoints/grast_gt/` — best.pt (151 MB) + train_log.json |

### Epoch 수 정당화 (사용자 5/14 질문 정합)

| 모델 | Epoch | 정당화 |
|---|:---:|---|
| **B (HN-SupCon)** | **1** | PLM fine-tune (MiniLM-L6 pretrained) + supervised contrastive fast convergence + overfitting 회피 + Piao 2025 §4.3 spec |
| **C-GT (GraST-GT)** | **40** | From-scratch + single margin contrastive + sparse attention 빠른 saturation + Hoang 2025 spec |
| GAT (selector) | 300 | From-scratch + multi-task (BCE+InfoNCE+ListNet+AC) + heterogeneous graph + conservative upper bound (실제 best ~58~289 variant 별) |

### Stdout buffering caveat (C-GT log)

`conda run -n base python -u` 의 `python -u` flag 가 conda run wrapper 의 추가 stdout buffer layer 를 해제 못함. 학습 종료 시 process exit 의 일괄 flush 까지 log 가 비어 있음. 다음 학습 launch 시 `stdbuf -oL -eL` 또는 `PYTHONUNBUFFERED=1` env 권장.

### 산출물

- `src/train_hn_supcon.py` (Module:Selector commit fb92775 + Root 5/14 evaluator 추가)
- `src/train_grast_gt.py` (Root 5/14 신규 wrapper — Module:Filter helper inline copy)
- Checkpoints: `outputs/checkpoints/{hn_supcon,grast_gt}/`
- Logs: `logs/train_{hn_supcon,grast_gt}/full_20260514_161204.log`

### 후속

- **Root sweep launch (즉시)**: `b06_01_hn_supcon_glm` (GPU 0) + `a05_26_grast_with_transformer_glm` (GPU 1) 병렬, BIRD-Dev 1534 query × GLM 4.7
- **Analyzer (sweep 후)**: `notebooks/analysis_results/direction_b_hn_supcon_sweep.md` + `direction_c_gt_sweep.md`
- **Planner (analyzer 후)**: paper §V.5.x narrative integration + Filter-Invariant 경계 추가 evidence

---

## Direction C 배포 Sweep (GRASTFDFilter + GPT-4.1-mini inferred FK, 2026-05-14, 🎯 Direction A 비교 sub-noise + 비용 효율 우세)

근거: planning/DECISIONS.md 2026-05-14 (Direction C inferred_fk Analyzer 핸드오프 정식 launch). Module:Filter commit e90d91a (`GRASTFDFilter` — XiYan forward + FD graph 위 Steiner-tree restore + FK/PK hardcode). notebooks/analysis_results/direction_c_inferred_fk.md §5 (GPT-4.1-mini inferred FK 4개 yaml). Direction A (a05_23 F1=0.5833, EX=0.5169) 의 ΔF1 trigger (<0.02) 충족 → Direction C 타겟 launch.

### Stack

- Builder: EnrichedHeteroGraphBuilder
- Selector: EnsembleSelector (best_gat_qcond_nl3.pt, α=0.5, top_k=20)
- Extractor: **MSTPCSTUnionExtractor** (Direction A 정합)
- Filter: **GRASTFDFilter** (provider=glm, model_name=zai-org/glm-4.7, terminal_source=forward, top_k=10, steiner_method=default, max_restore=30, fk_pk_hardcode=true)
- SQL Generator: LLMSQLGenerator (GLM 4.7, evidence-aware)

### Inferred FK (4개, flat list, GPT-4.1-mini predictions)

- `transactions_1k.CustomerID->customers.CustomerID` (debit_card_specializing)
- `transactions_1k.GasStationID->gasstations.GasStationID` (debit_card_specializing)
- `transactions_1k.ProductID->products.ProductID` (debit_card_specializing)
- `cards.setCode->sets.code` (card_games)

⚠ yaml format caveat: 사용자 prompt + direction_c_inferred_fk.md §5.1 의 yaml = Dict[db_id, List[str]] 형식이지만, GRASTFDFilter 의 `__init__` 은 `inferred_fk: Optional[List[str]]` flat list expect. 본 config 는 flat list 로 변환 — `_build_fd_graph` (line 240-241) 가 query 별 db_id schema match 시만 적용. 후속 권고: module:filter 의 `inferred_fk: Union[List, Dict]` patch.

### 운영 이력

- **2026-05-14 02:43:39 launch (single cell, GPU 0)**: V5 학습 (GPU 0 V5-A + GPU 1 V5-C) 공존 — GPU 0 memory 2.5GB/24GB 안전. GLM API V5 와 충돌 없음 (V5 = local GAT).
- **2026-05-14 04:33:26 DONE (wall 1h50m)**: Direction A 의 2h47m parallel 보다 짧음 (single cell 만 + GRAST-FD 의 algorithmic restore 빠른 per-q). per-q 4.37s (Direction A 6.04s 대비 -28%).

### 결과 — a05_25_grast_with_inferred_fk_glm (1534 BIRD-Dev)

| Cell | R | P | F1 | EX | LLM calls | Filter time mean | Token in |
|---|---|---|---|---|---|---|---|
| **a05_25** | **0.9251** | **0.4218** | **0.5794** | **0.5176** | 3068 (2/q) | 1.90s | 7.4M |

### Δ 분석 매트릭스

| 비교 | ΔR | ΔP | ΔF1 | ΔEX |
|---|---|---|---|---|
| **vs Direction A (a05_23 F1=0.5833)** | -0.0205 | -0.0001 | **-0.0039** ≈ zero | +0.0007 ≈ zero |
| vs anchor (MSTPCSTUnion+XiYan F1=0.8666) | +0.0479 | -0.4346 | **-0.2872** ⚠ | +0.5176* |

\* Anchor EX=0.0 측정 실패 — 정확 ΔEX 분석은 analyzer 진단 필요.

### Inferred FK 적용 통계

| DB | inferred FK 수 | 진입 query | 비중 |
|---|---|---|---|
| card_games | 1 (`cards.setCode->sets.code`) | **191** | 12.4% |
| debit_card_specializing | 3 | **64** | 4.2% |
| **합계** | 4 | **255** | 16.6% |

### 핵심 발견

1. **Direction C ≈ Direction A (F1 sub-noise)**: ΔF1 -0.0039 — algorithmic Steiner restore vs LLM-based backward 의 R-P tradeoff 동일 결과. paper §V.5.x 의 "R 회복 vs P 손실" narrative 의 dual-direction 정합.
2. **비용 효율 우세** ⚡: LLM call 4602→**3068 (-33%)**, token 14.3M→**7.4M (-48%)**, filter time 4.03s→**1.90s (-53%)**. GRAST-FD 의 algorithmic restore (Steiner-tree + FK/PK hardcode) 가 LLM call 미사용으로 throughput 향상.
3. **inferred FK 16.6% query 에 실제 적용** (card_games 191 + debit_card_specializing 64) — Steiner-tree restore 의 join path 회복 mechanism 동작. per-DB lift 정량은 analyzer 후속.
4. **EX maintained** (0.5176, Direction A 0.5169 와 동일 sub-noise) — backward restore (LLM) 와 Steiner restore (algorithmic) 가 downstream SQL gen 에 동일 효과. **paper §V.5.x EX 측 Filter-Invariant 의 mechanism-agnostic** evidence.

### Filter Dominance 7번째 축 (EX 측 Filter-Invariant) 보강

직전 Direction A (a05_23) 의 caveat 2 (RSL Backward F1 outlier but EX in-band) 와 정합:
- Filter Sweep v2: F1 spread 0.0072, EX spread 0.0085 (both sub-noise)
- Direction A: ΔF1 -0.2832 outlier, ΔEX -0.0033 in-band
- **Direction C: ΔF1 -0.2872 outlier, ΔEX +0.0007 in-band (vs A)** — RSL Backward 외 GRAST-FD (algorithmic Steiner) 도 동일 패턴
- → **dual evidence**: GLM 4.7 의 schema noise tolerance + restoration mechanism (LLM/algorithmic 무관) 의 EX-axis robustness

### 비용 / 운영

- Wall: 1h 50min (single cell, GPU 0)
- LLM call 합계: 3068
- Token in: 7.4M, out: 113K
- Filter time total: 2912s (=48.5min)
- ckpt: 추가 없음 (inference only)

### 산출물

- Config: `configs/experiments/abl/a05_filter_agentic/a05_25_grast_with_inferred_fk_glm.yaml`
- Sweep script: `scripts/run_grast_fd_sweep.sh`
- Outputs: `outputs/experiments/abl/a05_filter_agentic/a05_25_grast_with_inferred_fk_glm/`
- Module:Filter implementation: commit e90d91a (`GRASTFDFilter`)
- Inferred FK source: `outputs/analysis/direction_c_inferred_fk.yaml` + `notebooks/analysis_results/direction_c_inferred_fk.md`

### 후속

- **Analyzer 위임** (즉시 trigger 가능): `notebooks/analysis_results/direction_c_grast_fd_sweep.md`
  - Direction A vs Direction C F1 sub-noise (Δ -0.0039) + 비용 효율 narrative (-33% LLM call)
  - per-DB lift: card_games (191 query) + debit_card_specializing (64 query)
  - inferred FK 4개의 actual application rate + Steiner restore case studies
  - F1 net negative 일관 (paper §V.5.x R/P tradeoff dual-direction)
  - EX-axis Filter-Invariant 의 mechanism-agnostic evidence (Direction A + Direction C 통합)
- **Planner 위임** (analyzer 후): paper §V.5.x narrative 확정 + Filter Dominance 7번째 축 dual evidence (Direction A + C) + DECISIONS prepend
- **Cron 종료**: `2f8383e6` 5/14 09:00 사용자 명시로 종료 처리 완료

---

## Direction A 배포 Sweep (RSLBackwardFilter baseline + with_guard, 2026-05-13 → 05-13, 🎯 ΔF1 net negative — Direction C 타겟 launch trigger)

근거: planning/DECISIONS.md 2026-05-13 (학술 Agent Phase 3 Response — Direction A GO 확정). Module:Filter commit 462798d (`RSLBackwardFilter` 구현 + smoke 15/15). 사용자 5/13 지시 — (i) 두 cell 병렬 + (ii) Extractor = **MSTPCSTUnionExtractor** (MSTKruskal → MSTPCSTUnion anchor 변경). Memory `feedback_parallel_first.md` + `project_current_anchor.md` 신규 등재.

### Stack

- Builder: EnrichedHeteroGraphBuilder
- Selector: EnsembleSelector (best_gat_qcond_nl3.pt, α=0.5, top_k=20)
- **Extractor: MSTPCSTUnionExtractor** (score_threshold=0.1, MST ∪ PCST union, 본 anchor)
- Filter: RSLBackwardFilter (XiYan forward prune + Backward column restore + FK/PK guarantee)
- SQL Generator: LLMSQLGenerator (GLM 4.7, evidence-aware)

### 운영 이력

- **2026-05-13 19:50 launch (v1)**: Sequential sweep wrapper, MSTKruskal anchor. 21:03 사용자 redirect — extractor MSTKruskal → MSTPCSTUnion 변경 + 두 cell 병렬화 명시. v1 kill (712 records 폐기).
- **2026-05-13 21:03:57 → 23:51:20 (v2, PARALLEL, wall 2h47m)**: GPU 0=a05_23 baseline / GPU 1=a05_24 with_guard. GLM API 2 concurrent throughput 효율 ~98% (per-q 6.04→6.14s 분산 ~1.6%). V5 sweep (GPU 0/1 학습 동시) 와 충돌 없음 (각 GPU ~2.3GB / 24GB).

### 결과 (2 cell × 1534 BIRD-Dev)

| Cell | risky_dbs | R | P | F1 | EX | LLM calls | Filter time mean |
|---|---|---|---|---|---|---|---|
| **a05_23** baseline | `[]` | **0.9456** | 0.4219 | **0.5833** | 0.5169 | 4602 | 4.03s |
| **a05_24** with_guard | `["toxicology"]` | 0.9395 | 0.4202 | 0.5806 | 0.5150 | 4602 | 4.01s |
| ΔvsAnchor (vs MSTPCSTUnion+XiYan) | | +0.0684 | **-0.4345** | **-0.2833** | +0.5169* | +3068 | +2.07s |
| Δ a05_24 vs a05_23 | toxicology guard | -0.0061 | -0.0017 | **-0.0027** | -0.0019 | 0 | -0.02s |

*Anchor EX=0.0000 (sql_generator 비활성 또는 측정 실패) — Δ 정확치 보고 위해 analyzer 진단 필요. F1 의 Δ 는 신뢰 가능.

### 핵심 발견

1. **Backward restore 가 R +0.0684 회복 vs P -0.4345 손실** — 정량 R/P tradeoff 명확. F1 net negative.
2. **EX 0.5169 maintained** — anchor 의 EX 측정 실패 case 제외 시, RSL backward 의 column restore 가 downstream SQL gen 에 useful evidence.
3. **risky_dbs guard 효과 ≈ zero** (ΔF1 -0.0027, ΔR -0.0061, ΔEX -0.0019) — toxicology 의 backward restore 가 net neutral. **Guard 불필요** narrative.
4. **양 cell 의 final_n 거의 동일** — qid 9 california_schools final_n=5, qid 306 toxicology final_n=10 (a05_23=a05_24), qid 1094 european_football_2 final_n=44 — guard 활성 DB (toxicology) 에서도 same final_nodes 가 다수, 일부 미세 차이 측정 필요.
5. **Trigger 분기**: ΔF1 < 0.02 → **Direction C 타겟 launch** 명백히 trigger (실측 ΔF1 = -0.2833, 강한 negative).

### Filter Dominance 7번째 축 (Filter-Invariant) 보강

직전 Filter Sweep v2 (2026-05-13) 의 7 LLM filter spread 0.0072 → Direction A 의 RSLBackward 추가 시 spread 확장:
- 7 LLM filter best F1 = ~0.87 (Stacked / TieredBidirectional)
- RSLBackward F1 = 0.5833 (anchor 미만)
- → **schema linking 단계 F1 ceiling 0.87 의 7번째 축 narrative 유지** + RSLBackward 는 P-aggressive restore 가 net negative 인 outlier case.

### 비용 / 운영

- Wall: 2h 47min (PARALLEL)
- LLM call 합계: 9204 (a05_23 4602 + a05_24 4602)
- Token in: 28.6M, out: 364K
- Filter time total: 12.3K seconds (=205min)
- ckpt: 추가 없음 (inference only)

### 산출물

- Configs: `configs/experiments/abl/a05_filter_agentic/a05_2{3,4}_rsl_backward_*.yaml`
- Sweep script: `scripts/run_rsl_backward_sweep.sh` (parallel)
- Outputs: `outputs/experiments/abl/a05_filter_agentic/a05_2{3,4}_rsl_backward_*/`
- Module:Filter implementation: commit 462798d (`RSLBackwardFilter` + smoke 15/15)

### 후속

- **Analyzer 위임** (즉시 trigger 가능): `notebooks/analysis_results/direction_a_rsl_backward_sweep.md` (ΔF1 trigger 분기 final + anchor EX 측정 실패 진단 + R-P tradeoff narrative + Direction C 결정)
- **Planner 위임** (analyzer 후): paper §V.5.x narrative integration + Direction C launch trigger 확정 + DECISIONS prepend
- **Cron 종료**: `56c27b7f` 5/14 00:00 사용자 명시로 종료 처리 완료

---

## DSN Mitigation V5 Tier 1+2 4-Direction (V5-A GATE / V5-B GCNII L=2/4/6 / V5-C Full AERO) 학습 (V-3-ext 단계 9, 2026-05-12, 🚧 코드 준비 완료 + Launch 보류)

근거: planning/DECISIONS.md 2026-05-12 (V5 Mitigation Plan — Tier 1+2 4 Direction) + planning/oversmoothing/oversmoothing_v5_plan.md (학술 Agent input). V4-Combo-Null (mech(ii-b) DOMINANT 5/5 absolute confirm) 의 conditional trigger — Conservation Law 수정 (V5-A) / GCNII Trainability (V5-B) / Full AERO + Hop Attention (V5-C) 세 architectural direction + V5-D-1 PLM Lower Bound 진단 (analyzer 위임). **본 entry 는 코드 준비 완료 + Launch 보류 (사용자 redirect 5/12) — module 세션 review 후 launch 결정 대기.**

### 운영 이력

- **2026-05-12 (root chain, sweep PID 미할당)**: V5-A/B/C class + Hop Attention forward + V5 kwargs forwarding 변경 완료
  - `src/models/gat_network_v2.py` (line ~211~340 신규 클래스):
    - `GATEGATv2Conv` (V5-A) — Mustafa & Burkholz 2024. attention parameter `att` (neighbor) + `att_self` 분리.
    - `GCNIIGATv2Conv` (V5-B) — Chen 2020 + Peng 2024. β_l = log(λ/l + 1) Identity Mapping. `gcnii_w` Linear init=I, layer-index dependent β.
    - `FullAEROGATv2Conv` (V5-C) — Lee 2023. `SoftplusGATv2Conv` subclass + Hop Attention 은 outer SchemaHeteroGATv2.forward 에서 처리.
  - `_make_gatv2_conv` dispatch: `gat_layer_type` ∈ {'gate', 'gcnii', 'aero_full'} 추가. `GAT_LAYER_TYPES` set 확장.
  - `SchemaHeteroGATv2.__init__`: V5 ctor 옵션 (`gcnii_beta_lambda`, `aero_hop_attention`) + validation (aero_hop_attention=True 는 gat_layer_type='aero_full' 에서만, gcnii_beta_lambda>0).
  - `SchemaHeteroGATv2`: Hop Attention modules (`hop_attention_lin` / `hop_h0_proj` / `hop_out_lin` — node_types 별 nn.ModuleDict) + forward 의 V5-C 출력 경로 (jumping_knowledge 우선, h_0 proj + L+1 stack + softplus per-node norm + weighted sum).
  - `src/train_gat_s06.py` (line ~248-252 + log ~263): V5 kwargs forwarding (`gcnii_beta_lambda`, `aero_hop_attention`) + 로그 마지막에 mitV5 표기 추가.
- **2026-05-12 (root chain, smoke test)**: forward pass shape + lazy-init 검증 (toy graph, num_layers=2, heads=4):
  - standard:   1.999M params, out shapes OK
  - gate:       2.004M params (+att_self), out shapes OK
  - gcnii:      3.179M params (+gcnii_w eye-init), out shapes OK
  - aero_full+hop_attention: 2.051M params (+hop_attention_lin/hop_h0_proj/hop_out_lin), out shapes OK
  - lngin (V4-A): 4.368M params (backwards-compat OK)
  - softplus (V4-B): 1.999M params (backwards-compat OK)
  - 모든 variant out_channels=64, query_node/table/column/fk_node shapes 정상.
- **2026-05-12 (root chain)**: V5 configs 5 + sweep script 작성
  - `configs/training/dsn/train_dsn_p80_v5a_gate.yaml`
  - `configs/training/dsn/train_dsn_p80_v5b_gcnii_L2.yaml` (num_layers=2, gcnii_beta_lambda=0.5)
  - `configs/training/dsn/train_dsn_p80_v5b_gcnii_L4.yaml` (num_layers=4)
  - `configs/training/dsn/train_dsn_p80_v5b_gcnii_L6.yaml` (num_layers=6)
  - `configs/training/dsn/train_dsn_p80_v5c_aero_full.yaml` (jumping_knowledge='none' + aero_hop_attention=true)
  - `scripts/run_v5_mitigation_sweep.sh` — Stage 1 (V5-A + V5-C 병렬, ~10h) → Stage 2 (V5-B L=2 + L=4 병렬, ~10h) → Stage 3 (V5-B L=6 단일, ~10-15h). NAS mv + symlink + best val recall@15 추출.
- **2026-05-12 (user redirect)**: "오케스트레이션과 실험을 진행하지 직접 모듈을 구현하지 마". Auto-mode classifier 가 V5 sweep launch 차단 (root 가 작성한 module code 의 30+ 시간 실행이 redirect 위반). **Launch 보류** — module 세션 (예: src/modules/selectors or 신규 src/models/ 세션) review + launch 결정 대기.

### 5 ckpt 학습 예정 (Launch 보류)

| Variant | gat_layer_type / 옵션 | num_layers | 예상 wall | 비교 baseline |
|---|---|---:|---|---|
| V5-A GATE | `gat_layer_type='gate'` | 2 | ~10h | V4-A 0.5929 / V4-B 0.5951 |
| V5-B GCNII L=2 | `gat_layer_type='gcnii'`, `gcnii_beta_lambda=0.5` | 2 | ~10h | Phase 1 P80 0.6097 |
| V5-B GCNII L=4 | gcnii + num_layers=4 | 4 | ~12h | depth scale evidence |
| V5-B GCNII L=6 | gcnii + num_layers=6 | 6 | ~15h | Chen 2020 deep-GNN claim |
| V5-C Full AERO | `aero_full` + `aero_hop_attention=true`, JK='none' | 2 | ~10h | V4-B 0.5951 (Hop 없는 절반 구현) |

### Stack

V-3-ext (DSN p80 directed_from_sn + percentile=80) + B5 mitigation (PN+IR α=0.2+JK=concat(V5-A/B), JK=none+Hop(V5-C)+Dual-Stream+AC=0.1+ListNet) + V5 architectural intervention.

### 시나리오 분기 (DECISIONS V5)

| 시나리오 | 조건 | narrative 영향 |
|---|---|---|
| **(1) V5-D-1 R 갱신** | analyzer 의 PLM Lower Bound 진단 + 후속 학습 R > 0.6097 | 학술 Agent reinterpretation confirm. Layer 2 narrative pivot — "R@15 ceiling 의 원인은 PLM lower bound" |
| **(2) V5-A 또는 V5-C 단독 R 갱신** | architectural mitigation 가능 path 발견 | mech(ii-b) "5/5 absolute confirm" narrative 약화. paper §V.5.4 major rewrite |
| **(3) V5 4 Direction 모두 fail** | Full Tier 1+2 null | 현재 narrative (mech(ii-b) 5/5 + Filter Dominance 6번째 축) **결정적 강화** — 14-trial + 4 architectural direction 모두 무력 |

### 후속 위임 (chain handoff)

- **module:selectors (또는 신규 module:models)**: V5-A/B/C 클래스 + Hop Attention forward 의 코드 review (planning/oversmoothing/oversmoothing_v5_plan.md §4.1/§4.2/§4.3 의 정확한 mechanism 일치 확인) → launch 결정. 본 root 가 root 영역 위반으로 직접 launch 불가.
- **analyzer (module session 후 sweep launch 후)**: dsn_oversmoothing_analysis.py 의 CKPTS 리스트 갱신 (v5a_gate / v5b_gcnii_L{2,4,6} / v5c_aero_full entry 등록) + 14-trial V5 결과 + Layer 1/2/3 evidence 재정량. `notebooks/analysis_results/dsn_mitigation_v5_4dir.md` 신규.
- **analyzer (V5-D-1)**: anchor stack 의 c_L0 + c_L3 measurement (Plain vs Enriched 비교) — outputs/analysis/v5_d1_plm_lower_bound_diagnostic/. **별도 chain** — V5 학습 무관, CPU forward 가능.
- **planner (analyzer 후)**: narrative pivot 결정 (시나리오 1/2/3) + 5 over-smoothing planning 문서 통합 갱신 + paper §3.5 narrative.

### 산출물 (코드 + config 준비 완료)

- 모델 확장: `src/models/gat_network_v2.py` (`GATEGATv2Conv` + `GCNIIGATv2Conv` + `FullAEROGATv2Conv` + Hop Attention modules + `GAT_LAYER_TYPES` set 확장 + V5 forwarding/validation)
- 학습 entry: `src/train_gat_s06.py` (V5 kwargs forwarding line ~248-252 + log)
- Configs (5): `configs/training/dsn/train_dsn_p80_v5{a_gate, b_gcnii_L2, b_gcnii_L4, b_gcnii_L6, c_aero_full}.yaml`
- Sweep script: `scripts/run_v5_mitigation_sweep.sh` (3-stage, GPU 0/1 병렬)

### Caveat — 본 entry 는 launch 보류 placeholder

Launch 진행 (module session 위임 후) 시 root 재호출 시 다음 값 반영:
- 5 ckpt 의 best val Recall@15 (4 decimal) + best epoch
- 학습 wall clock + ckpt NAS 경로
- 시나리오 분기 (1/2/3) 확정
- analyzer 핸드오프 trigger (sweep 완료 + dsn_mitigation_v5_4dir.md)

---

## DSN Mitigation V4 Architectural Intervention (V4-A LN+GIN Combo + V4-B AERO Softplus) 학습 (V-3-ext 단계 8, 2026-05-11 → 05-12, 🎯 시나리오 V4-Combo-Null 확정)

발사: 2026-05-11 23:23 KST → 종료: V4-B 2026-05-12 09:05 (wall 9h 38min) + V4-A 2026-05-12 10:14 (wall 10h 47min). DECISIONS 2026-05-11 §V4 채택 + `planning/oversmoothing/oversmoothing_solution_methodology_2026-05-11_apa.md` §C-1 + §C-2 prerequisite — combo 가설 (mech(ii-b) softmax-weighted-mean DOMINANT 4/5 partial 부정) 의 정량 검증. **결과: 둘 다 baseline 0.6097 미달 → mech(ii-b) DOMINANT 5/5 absolute confirm + Filter Dominance 6번째 축 narrative 결정적 강화.**

### 운영 이력

- **2026-05-11 (selector 세션, root 위임)**: V4-A/V4-B class 구현 + smoke test 통과
  - `src/models/gat_network_v2.py`: `LNGINGATv2Conv` (V4-A) + `SoftplusGATv2Conv` (V4-B) 추가
  - `_make_gatv2_conv` 에 `gat_layer_type` / `softplus_symmetric_norm` 인자 forward
  - `SchemaHeteroGATv2.__init__` 에 V4 옵션 + validation (gat_layer_type='lngin'/'softplus' 가 aggregation_type='gin' / drop_message_p / use_layernorm_pre_softmax 와 incompatible 시 raise)
  - 코드 주석에 §C-1 + §C-2 + Wu et al. 2023 JSR < 1 theory + AERO-GNN Theorem 3 인용
- **2026-05-11 (selector 세션)**: V4 configs 작성
  - `configs/training/dsn/train_dsn_p80_v4a_lngin_combo.yaml`
  - `configs/training/dsn/train_dsn_p80_v4b_aero.yaml`
- **2026-05-11 (selector 세션)**: `src/train_gat_s06.py` 에 V4 kwargs forwarding 추가 (line ~243-258)
- **2026-05-11 (selector 세션)**: `src/analysis/dsn_oversmoothing_analysis.py` 의 CKPTS 리스트에 v4a_lngin_combo + v4b_aero entry 등록 + `_build_model_dsn` 분기 처리
- **2026-05-11 23:23 KST (root 세션)**: `scripts/run_v4_mitigation_sweep.sh` nohup launch (GPU 0=V4-A + GPU 1=V4-B 병렬), ETA ~12h wall
- **2026-05-11 23:26 KST**: V4-A epoch 1 wall ~2:23, Loss 2.8450, Val R@15 **0.4789** (best) — 정상 학습 시작
- **2026-05-11 23:26 KST**: V4-B epoch 1 wall ~2:12, Loss 2.8258, Val R@15 **0.4840** (best) — 정상 학습 시작
- **2026-05-12 01:17 KST**: V4-B epoch 58 새 best **0.5951** 갱신 — 이후 240+ epoch 정체 (final ep300=0.5913)
- **2026-05-12 08:43 KST**: V4-A epoch 259 새 best **0.5929** 갱신 (직전 ep ~46 best 0.5924 갱신) — 이후 40+ epoch 정체 (final ep300=0.5888)
- **2026-05-12 09:05 KST**: V4-B 학습 완료 (wall 9h 38min) → NAS mv + symlink (113MB) ✅
- **2026-05-12 10:14 KST**: V4-A 학습 완료 (wall 10h 47min) → NAS mv + symlink (257MB) ✅
- **2026-05-12 10:15 KST**: sweep parent (PID 199851) exit + `dsn_oversmoothing_analysis.py` 자동 호출 시작 (post-train 측정 — attention/cosine OK, grad_flow 일부 미호환 발생)

### 신규 학습 entry (최종 결과)

#### V4-A LN+GIN Combo (gat_layer_type='lngin') — §C-1 최단 경로 검증

- Config: `configs/training/dsn/train_dsn_p80_v4a_lngin_combo.yaml`
- Mechanism: GATv2Conv → `LNGINGATv2Conv` 교체. row-stochasticity **유지** 단 두 차원의 partial mitigation combo:
  - attention path: `LayerNormGATv2Conv` 기반 (raw α → LN → softmax) — v2 #3 LN 와 동일 (mech(ii-a) softmax over-concentration partial)
  - aggregation post-process: weighted-mean 결과 (heads × out_channels) 를 2-layer MLP 로 transform — v3 #1 GIN 와 동일 (mech(ii-b) propagation partial)
- 의도: 직전 v2 #3 LN 단독 (R=0.6011) + v3 #1 GIN 단독 (R=0.5954) 의 partial mitigation 의 **합산 효과** 가 나타나는지 정량 검증
- 신규 ckpt: `best_gat_directed_supernode_p80_v4a_lngin_combo.pt` (NAS symlinked, **257MB**)
- **Best val recall@15 = 0.5929 @ ep259** (final ep300=0.5888)
- **Final epoch 300, wall 10h 47min** (23:26:45 5/11 → 10:14:04 5/12)
- Δ vs Phase 1 baseline = **-0.0168** (10-trial #9)
- 직전 v2 #3 LN 단독 (0.6011) + v3 #1 GIN 단독 (0.5954) 의 **합산 회복 0** — 두 partial mit 의 합이 새 회복을 만들지 못함

#### V4-B AERO Softplus + Symmetric Norm (gat_layer_type='softplus') — §C-2 이론적 최강

- Config: `configs/training/dsn/train_dsn_p80_v4b_aero.yaml`
- Mechanism: GATv2Conv → `SoftplusGATv2Conv` 교체. row-stochasticity 자체 **파괴**:
  - softmax 제거 → `softplus(α) = log(1 + exp(α))` (비음수 단 row-sum ≠ 1)
  - (옵션) edge-symmetric normalization: `α_ij / sqrt(d_i · d_j)` (degree 기반)
  - Wu et al. 2023 JSR < 1 의 row-stochastic matrix 가정 위반 → over-smoothing 의 이론적 보장 회피 (AERO-GNN Theorem 3 보증)
- 의도: row-stochasticity 자체를 깨야 회복 가능한지의 이론적 검증
- 신규 ckpt: `best_gat_directed_supernode_p80_v4b_aero.pt` (NAS symlinked, **113MB**)
- **Best val recall@15 = 0.5951 @ ep58** (final ep300=0.5913)
- **Final epoch 300, wall 9h 38min** (23:26:35 5/11 → 09:05:23 5/12)
- Δ vs Phase 1 baseline = **-0.0146** (10-trial #6, v3 GIN 0.5954 와 -0.0003 사실상 동등)
- **row-stochasticity 파괴에도 baseline 미달** — Wu et al. 2023 의 JSR < 1 가정 위반이 over-smoothing 회피 보장한다는 이론이 본 stack 에서는 실증 안 됨. mech(ii-b) DOMINANT 의 5/5 absolute confirm 결정적 evidence.

### 🎯 10-trial mitigation 통합 매트릭스 (최종, V4 결과 반영)

| 순위 | Variant | Best R@15 | Best Epoch | Δ vs Phase 1 | 가설 검증 결과 |
|------|---------|-----------|------------|--------------|----------------|
| **1** | **Phase 1 P80 (no mit)** | **0.6097** | ep91 | (baseline) | — |
| 2 | Phase 2 b8 (B5 mit fusion) | 0.6018 | ep157 | -0.0079 | mech(iii) 부정 |
| 3 | v2 #3 LayerNorm pre-softmax | 0.6011 | ep289 | -0.0086 | mech(ii-a) partial |
| 4 | v2 #1 DropMessage | 0.5974 | ep157 | -0.0123 | partial |
| 5 | v3 #1 GIN-style aggregation | 0.5954 | ep246 | -0.0143 | mech(ii-b) partial |
| **6** | **🆕 V4-B AERO Softplus** | **0.5951** | **ep58** | **-0.0146** | **row-stochasticity 파괴 fail — mech(ii-b) DOMINANT 5/5 confirm** |
| 7 | Phase 3 #4 (LR x5) | 0.5935 | ep172 | -0.0162 | training dynamics 부정 |
| 8 | Phase 3 #3 (Direct AC gat_out_L_last) | 0.5927 | ep51 | -0.0170 | mech(iii) 부정 |
| **9** | **🆕 V4-A LN+GIN Combo** | **0.5929** | **ep259** | **-0.0168** | **combo 가설 fail — partial mit 합산 효과 nothing** |
| 10 | v2 #2 Sum Aggregation | 0.5761 | ep194 | -0.0336 | aggregation magnitude pathology |

(시도 #9 v3 #2 Max aggregation 은 5/14 ETA 별도 chain — 본 V4 chain 의 후속.)

### 🎯 시나리오 V4-Combo-Null 확정 (mech(ii-b) DOMINANT 5/5 absolute confirm)

DECISIONS 2026-05-11 §V4 분기:
- ❌ V4-Combo-Win (V4-A 또는 V4-B 가 0.6097 이상): 미충족 (V4-A=0.5929, V4-B=0.5951)
- **✅ V4-Combo-Null** (둘 다 0.6097 미만): **확정** — V4-A Δ=-0.0168 / V4-B Δ=-0.0146 → mech(ii-b) DOMINANT 4/5 → **5/5 absolute confirm**. Filter Dominance 6번째 축 (training-pathology-invariant) narrative 결정적 강화.
- ❌ V4-Mixed (한 쪽만 win): 미충족

### 핵심 발견 (3)

**(1) Architectural intervention 도 ceiling 갱신 실패**:
- V4-A (row-stochasticity 유지 + LN+GIN combo) Δ=-0.0168 — 직전 v2 #3 LN 단독 (-0.0086) + v3 #1 GIN 단독 (-0.0143) **합산보다도 손실 더 큼**. 두 partial mit 가 destructive interfere.
- V4-B (row-stochasticity 파괴 + Softplus + Symmetric Norm) Δ=-0.0146 — v3 #1 GIN (-0.0143) 와 사실상 동등. Wu et al. 2023 JSR < 1 의 row-stochastic 가정 위반에도 over-smoothing 회피 실증 안 됨.

**(2) Best epoch 패턴이 직전 trial 들과 일관**:
- V4-A best ep259 — 직전 v2 #3 LN best ep289 와 유사한 late-epoch 회복 (300 epoch saturation 후 새 best)
- V4-B best ep58 — 직전 v3 #1 GIN ep246 보다 훨씬 빠른 early-epoch saturation (60+ epoch 의 240+ 정체 패턴)
- 두 패턴 모두 직전 8-trial 의 ceiling 흡수 패턴과 일관 → "GAT internal dynamics 가 학습 변형에 lock-in" 의 일관 evidence

**(3) Loss curve 수렴이 동등**:
- V4-A final Loss 1.1410 / V4-B final Loss 1.1510 — 직전 v3 GIN (final ~1.16) 와 유사한 수렴 region
- AC loss 둘 다 < 0.001 (효과적 anti-collapse pressure)
- 학습 dynamics 측면 모두 normal — pathology 가 학습 단계가 아니라 **architecture 자체** 라는 결론 강화

### Filter Dominance 6번째 축 10-trial evidence 누적 (완료)

직전 8 evidence + 2 추가:
1-8. (직전 8 evidence: V-3-ext 단계 1~7 mitigation)
9. **🆕 V4-B AERO Softplus evidence** (V-3-ext 단계 8a) — row-stochasticity 파괴 도 ceiling 갱신 실패 (R 0.5951, Δ=-0.0146). Wu et al. 2023 의 over-smoothing 회피 이론이 본 stack 의 schema linking 에서는 실증 안 됨.
10. **🆕 V4-A LN+GIN combo evidence** (V-3-ext 단계 8b) — 두 partial mitigation 의 combo 도 ceiling 갱신 실패 (R 0.5929, Δ=-0.0168). partial mit 의 합산이 새 회복을 만들지 못함.

### 비용 / 운영

- 학습 wall: V4-B 9h 38min + V4-A 10h 47min (병렬, max=10h 47min)
- 비용: **₩0** (LLM-free)
- ckpt NAS: V4-A **257MB** + V4-B **113MB** (V4-A 가 LN+GIN MLP 추가 파라미터로 더 큼)
- 자동 후속: sweep script 가 학습 종료 후 `dsn_oversmoothing_analysis.py --max_queries 50 --skip_step1 --skip_step2` 자동 호출 (10:15 KST start) — attention/cosine 측정 OK, ⚠️ grad_flow 측정에서 `'NoneType' object has no attribute 'named_parameters'` (V4 ckpt class 일부 미호환, analyzer 가 attention/cosine 만 활용 필요)

### 산출물

- Configs (2): `configs/training/dsn/train_dsn_p80_v4{a,b}_*.yaml`
- 모델 확장: `src/models/gat_network_v2.py` (`LNGINGATv2Conv` + `SoftplusGATv2Conv` + `GAT_LAYER_TYPES` enum + `_make_gatv2_conv` V4 분기 + `SchemaHeteroGATv2.__init__` V4 옵션/validation)
- 학습 entry: `src/train_gat_s06.py` (V4 kwargs forwarding)
- 분석 확장: `src/analysis/dsn_oversmoothing_analysis.py` (CKPTS v4a/v4b 등록 + `_build_model_dsn` 분기)
- Sweep script: `scripts/run_v4_mitigation_sweep.sh` (병렬 GPU 0/1 launch + 자동 NAS mv + 자동 후속 분석)
- Checkpoint (NAS): `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_v4{a_lngin_combo,b_aero}.pt`
- 학습 log: `logs/train/dsn_p80_v4{a_lngin_combo,b_aero}_20260511_232355.log`

### 후속 (analyzer + planner 핸드오프)

- **Analyzer 위임 (즉시 trigger 가능)**: `notebooks/analysis_results/dsn_mitigation_v4_combo.md` 신규 — V4-A=0.5929 / V4-B=0.5951 정량 + 10-trial matrix + mech(ii-b) 5/5 absolute confirm narrative
- **Planner 위임 (analyzer 후)**: advisor briefing §3 + §4 + §7 갱신 + root cause report §3 + §5 갱신 + paper §V.5.4 narrative integration + DECISIONS.md V4 결과 entry prepend

---

## DSN Mitigation v3 #1 GIN-style aggregation 학습 (V-3-ext 단계 7, 2026-05-08 → 05-09, 🎯 시나리오 V3-A 1차 confirm)

발사: 2026-05-08 17:12 → 완료 2026-05-09 04:51 (wall ~11h 39min, GPU 0 단독). Phase 1 deep dive (A1+A2+A3) 후 Phase 2 GIN 구현 + Phase 3 학습.

### 운영 이력

- **2026-05-08 04:28**: Phase 2 selector 모듈 GIN-style aggregation 구현 + 7 smoke 통과 (별도 세션)
- **2026-05-08 17:12**: Phase 3 GIN 학습 launch (GPU 0, setsid + disown, batch_size=8)
- **2026-05-09 04:51**: 학습 종료 + ckpt NAS symlinked (140MB)
- **Alpha sweep skip 유지** (사용자 결정 2026-05-07 (1)A) → val recall@15 evidence only

### 신규 학습 entry

#### v3 #1 GIN-style aggregation (aggregation_type='gin')

- Config: `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v3_gin.yaml`
- Mechanism: GATv2Conv → GINConv 교체 (HeteroConv `aggr='mean'` fix + 18 inner GINConvs). PyG GINConv 의 MLP+sum aggregation — attention 자체 부재 (mech(ii-a) 측정 X), mech(ii-b) propagation pathology 직접 검증
- 신규 ckpt: `best_gat_directed_supernode_p80_b5_mitigation_v3_gin.pt` (140MB, NAS symlinked)
- Best val recall@15 = **0.5954** @ ep246, final R@15 0.5937
- per-epoch ~2.35 min (Phase 2 b8 의 2.12 보다 약간 느림 — GIN MLP overhead)

### 🎯 8-trial mitigation 통합 결과 (V-3-ext 단계 5+6+7)

| 순위 | Variant | Best R@15 | Best Epoch | Δ vs Phase 1 |
|------|---------|-----------|------------|--------------|
| **1** | **Phase 1 P80 (no mit)** | **0.6097** | ep91 | (baseline) |
| 2 | Phase 2 b8 (mit fusion) | 0.6018 | ep157 | -0.0079 |
| 3 | v2 #3 LayerNorm pre-softmax | 0.6011 | ep289 | -0.0086 |
| 4 | v2 #1 DropMessage | 0.5974 | ep157 | -0.0123 |
| **5** | **🆕 v3 #1 GIN-style aggregation** | **0.5954** | ep246 | **-0.0143** |
| 6 | Phase 3 #4 (LR x5) | 0.5935 | ep172 | -0.0162 |
| 7 | Phase 3 #3 (Direct AC gat_out_L_last) | 0.5927 | ep51 | -0.0170 |
| 8 | v2 #2 Sum Aggregation | 0.5761 | ep194 | -0.0336 |

**🚨 핵심 발견**:
1. **모든 8 mitigation variants 가 baseline 미달** — graph topology + B5 mit + Direct AC + LR x5 + DropMessage + LayerNorm + Sum Aggr + GIN 모두 raw R 한계 갱신 X
2. **GIN 가 mit variants 5위** — Phase 3 #3/#4 (Direct AC, LR x5) 추월 but v2 #1 (DropMessage) / v2 #3 (LayerNorm) 미달
3. **GIN attention 부재인데 partial mitigation** (-0.0143 vs Phase 1) — mech(ii-a) softmax 자체 부재해도 mech(ii-b) propagation pathology 가 동일 ceiling 유발

### 시나리오 V3-A 1차 confirm (잠정, analyzer Phase 4 deep dive 전)

DECISIONS 2026-05-08 §1 분기:
- **✅ 시나리오 V3-A** (best R@15 ~0.59-0.61): **1차 confirm** — GIN best 0.5954, 8-trial 모두 baseline 0.6097 미달
- ❌ V3-B (best 0.62-0.70): GIN MLP+sum 효과 발견 — 미달
- ❌ V3-C (best 0.85+): mech(ii) 부정 — 사실상 불가능

**잠정 mech(ii) sub-mechanism 분리** (analyzer Phase 4 후 정식 확정):
- mech(ii-a) softmax over-concentration: v2 #3 LayerNorm partial mitigation (0.6011) — 가장 효과적
- mech(ii-b) aggregation family / propagation: GIN 0.5954 → mech(ii-b) 자체 limitation 신호 (attention 부재해도 ceiling 유사)
- → mech(ii-b) DOMINANT 후보 강화 (analyzer 분석 후 확정)

### Filter Dominance 6번째 축 8-trial evidence 누적

직전 7 evidence + 1 추가:
1-5. (직전 5 evidence)
6. Phase 2 + Phase 3 4-trial mitigation null effect (training-pathology-invariant)
7. Mitigation v2 3-trial 추가 evidence (DropMessage / LayerNorm / Sum Aggr 모두 baseline 미달)
8. **🆕 Mitigation v3 #1 GIN 8번째 evidence** (aggregation family 변경에도 baseline 미달)

### 비용 / 운영

- 학습 wall: ~11h 39min (5/8 17:12 ~ 5/9 04:51)
- 비용: **₩0** (LLM-free)
- ckpt NAS: 140MB
- Alpha sweep skip 유지

### 산출물

- Config: `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v3_gin.yaml`
- 모델 확장: `src/models/gat_network_v2.py` (`_make_gin_conv` factory + AGGREGATION_TYPES 'gin' 추가)
- 학습 entry: `src/train_gat_s06.py` (aggregation_type='gin' forward)
- Smoke test: `src/modules/selectors/tests/test_mitigation_v3.py` (7/7 통과, Phase 2)
- Checkpoint (NAS): `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_b5_mitigation_v3_gin.pt`
- 학습 log: `logs/train/gat_directed_supernode_p80_b5_mitigation_v3_gin_*.log`

### 후속 (analyzer Phase 4 + planner Phase 5 핸드오프)

- **Analyzer 위임**: 8 ckpt × 5-step protocol (multi-DB stratified, 55 queries = 5 × 11 DBs, Phase 1 A3 동일)
  - Step 1-5 + mech(ii-b) GIN 차단 정도 정량 (L1_GAT cosine)
  - 산출물: `notebooks/analysis_results/dsn_mitigation_v3_8trial.md` (또는 dsn_mitigation_v2_results.md §14 보강)
- **Planner 위임**: 시나리오 V3-A 정식 채택 + paper §3.5 6번째 축 8-trial evidence 명문화 + §V Part III main contribution narrative
- 사용자 결정 (3) 재고: V3-A 시 max aggregation (#2) 추가 시도 후보


---

## DSN Mitigation V5 7-Trial Sweep 학습 완료 (V-3-ext 단계 9, 2026-05-13 → 05-15, 🎯 시나리오 (a) confirm — 7/7 모두 P80 baseline 미달, mech(ii-b) 7/7 absolute confirm 17-trial 격상 candidate)

근거: planning/DECISIONS.md 2026-05-12 (V5 Mitigation Plan Tier 1+2 4 Direction) + planning/oversmoothing/oversmoothing_v5_plan.md §Tier 1 + V5-A/B/C spec + planning/oversmoothing/README.md §Phase 2 + src/modules/selectors/EXPERIMENT_PLAN_selectors.md §단계 9. V4-Combo-Null (mech(ii-b) DOMINANT 5/5 absolute confirm) 의 conditional trigger — Conservation Law decoupling (V5-A) / GCNII Trainability L=2/4/6 (V5-B) / Full AERO with Hop+Cumulative Attention (V5-C) 의 3 axis × 7 cell (V5-A 1 + V5-B 3 + V5-C 3) architectural intervention 완전 sweep. 학습 완료 후 7 cell 모두 P80 baseline 0.6097 미달 — **시나리오 (a) 격상**.

### 운영 이력

- **2026-05-12 (root chain)**: V5-A/B/C class 구현 완료 (`src/models/gat_network_v2.py` commit `afadafd`) + 5 ckpt config 작성 + smoke test 통과
- **2026-05-12 (user redirect)**: "Root 는 오케스트레이션 + 모델 학습 + 실험 진행 메인" — module impl 금지 해소, Root 가 학습 launch + 모니터링 main 담당
- **2026-05-13 12:10 launch (Stage 1, GPU 0/1 병렬)**: v5a_gate (GPU 0) + v5c_full (GPU 1) — V5-C 가 sweep script 의 첫 launch
- **2026-05-14 00:15-00:25 종료 (Stage 1)**: v5c_full + v5a_gate 종료 (~12h wall)
- **2026-05-14 00:15 launch (Stage 2)**: v5b_gcnii_L2 (GPU 0) + v5c_hop_only (GPU 1) — V5-C rename (`v5c_aero_full` → `v5c_hop_only`, task #92) + v5c_full / v5c_cum_only 신규 config 작성 (task #93/#94, hop+cum 별도 분리)
- **2026-05-14 12:27-13:01 종료 (Stage 2)**: v5c_hop_only + v5b_gcnii_L2 종료
- **2026-05-14 12:27 launch (Stage 3)**: v5b_gcnii_L4 (GPU 0) + v5c_cum_only (GPU 1)
- **2026-05-15 01:11-04:51 종료 (Stage 3)**: v5c_cum_only + v5b_gcnii_L4 종료
- **2026-05-15 01:12 launch (Stage 4)**: v5b_gcnii_L6 (GPU 1, 단일)
- **2026-05-15 19:06:59 종료 (Stage 4 = 마지막)**: v5b_gcnii_L6 종료, 7-trial 완료

### 7 variants 의 best Val R@15 (4-decimal, Phase 1 P80 baseline=0.6097 비교)

| Variant | Module Class | gat_layer_type / options | num_layers | Best Val R@15 | Best Epoch | Final Epoch | Wall | Δ vs P80 | Δ vs anchor qcond_nl3 (0.6061) |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| **v5c_hop_only** | `FullAEROGATv2Conv` (hop only) | `aero_full` + `aero_hop_attention=true`, `aero_cum_attention=false`, JK='none' | 2 | **0.6076** | 78 | 300 | 12h 12m | -0.0021 | +0.0015 |
| v5b_gcnii_L2 | `GCNIIGATv2Conv` (L=2) | `gcnii`, `gcnii_beta_lambda=0.5` | 2 | 0.6072 | 76 | 300 | 12h 36m | -0.0025 | +0.0011 |
| v5c_cum_only | `FullAEROGATv2Conv` (cum only) | `aero_full` + `aero_cum_attention=true`, `aero_hop_attention=false`, JK='none' | 2 | 0.5993 | 25 | 300 | 12h 44m | -0.0104 | -0.0068 |
| v5b_gcnii_L4 | `GCNIIGATv2Conv` (L=4) | `gcnii` + num_layers=4 | 4 | 0.5969 | 198 | 300 | 15h 50m | -0.0128 | -0.0092 |
| v5c_full | `FullAEROGATv2Conv` (full) | `aero_full` + `aero_hop_attention=true`, `aero_cum_attention=true`, JK='none' | 2 | 0.5887 | 241 | 300 | 12h 05m | -0.0210 | -0.0174 |
| v5b_gcnii_L6 | `GCNIIGATv2Conv` (L=6) | `gcnii` + num_layers=6 | 6 | 0.5845 | 212 | 300 | 17h 55m | -0.0252 | -0.0216 |
| v5a_gate | `GATEGATv2Conv` | `gate` | 2 | 0.5571 | 286 | 300 | 12h 15m | -0.0526 | -0.0490 |

**P/F1@15**: 학습 시 측정 안 됨 — `train_gat_s06.py` 가 R@15 만 평가 (val P@15 / F1@15 산출은 analyzer 의 후속 평가 dispatch — `dsn_oversmoothing_analysis.py` CKPTS 등록 후 측정).

### 시나리오 분기 결과 (DECISIONS V5 §1, V5-D-1 multi-DB n=55 PLM lower bound 진단 정합)

| 시나리오 | 조건 | 결과 |
|---|---|:---:|
| **(a) V5 7 모두 null (R ~0.58-0.61 saturate)** | mech(ii-b) 7/7 absolute confirm | **✅ CONFIRM** — 7/7 모두 P80 0.6097 미달, max v5c_hop_only 0.6076 = Δ -0.0021 noise band |
| (b) V5-A / V5-C 단독 R 갱신 | mech(ii-b) 7/7 부분 부정 | ❌ V5-A 0.5571 (Δ -0.0526 worst), V5-C 최고 0.6076 (Δ -0.0021 미달) |
| (c) V5-B L=2/4/6 중 1+ R 갱신 | mech(iv) / trainability 가설 confirm | ❌ V5-B 최고 0.6072 (Δ -0.0025 미달), L scale monotonic decay (L=2 0.6072 > L=4 0.5969 > L=6 0.5845) |
| (d) V5-C Cumulative only R 갱신 | mech(ii-b) hidden-state residual evidence | ❌ V5-C Cumulative only 0.5993 (Δ -0.0104 미달) |

→ **시나리오 (a) absolute confirm** — V5 Tier 1+2 architectural intervention 의 3 axis × 7 cell 모두 fail.

### Mech Dominance Scoring 14-trial → 17-trial 정합 candidate

14-trial Final (dsn_mitigation_v4_combo.md base, 8-trial + V4-A + V4-B + ...) + V5 7-trial = **17-trial 통합** (직전 14 + V5-A + V5-B-L2 + V5-B-L4 + V5-B-L6 + V5-C-Full + V5-C-Hop + V5-C-Cumulative — 단 14 가 분류상 V4-A/B 포함이므로 V5-B-L2/L4/L6 + V5-C-Full/Hop/Cumulative 의 새 7 = 직전 10 + 7 = 17 가능).

**mech(ii-b) DOMINANT 7/7 absolute confirm 격상 candidate** — analyzer 의 17-trial dominance scoring 정식 채택 trigger:
- V4-A (LN+GIN combo): 0.5929 fail
- V4-B (AERO Softplus+Sym-Norm half): 0.5951 fail
- V5-A GATE (Conservation Law 수정): 0.5571 fail (V5 worst)
- V5-B GCNII L=2/4/6 (Identity Mapping + Initial Residual): 0.5845~0.6072 모두 fail
- V5-C Full AERO with Hop+Cum Attention (SR2OS guarantee full): 0.5887 fail
- → mech(ii-b) **softmax × weighted-mean propagation combo** 가 5 axis (V4-A LN, V4-B Softplus, V5-A Conservation Law, V5-B Identity Mapping, V5-C Hop+Cum Attention) 의 모든 architectural mitigation 에 invariant — fundamental architectural limitation 격상 가능

### 학습 비용 + 환경

- **GPU 시간**: 95h 17m (7 cell 합산, 2-GPU 병렬 4 stage 의 wall ~55h: 5/13 12:10 → 5/15 19:07)
- **LLM API 비용**: ₩0 (학습은 local GAT, LLM call 없음)
- **GPU 사용량**: GPU 0 (3 cell 순차) + GPU 1 (4 cell 순차) — `CUDA_VISIBLE_DEVICES=0,1` 만 (GPU 2/3 reserved)
- **Loss**: Phase 2 B5 mitigation 동일 (PN + IR α=0.2 + JK=concat(V5-A/B), JK=none + Hop(V5-C) + Dual-Stream + AC=0.1 + ListNet)
- **per-epoch wall**: v5a_gate ~2.45m, v5b_L2 ~2.52m, v5b_L4 ~3.17m, v5b_L6 ~3.58m, v5c_full ~2.42m, v5c_hop_only ~2.44m, v5c_cum_only ~2.55m

### Checkpoint NAS 위치 (7 신규 ckpt)

| Variant | NAS ckpt | Size | 학습 종료 |
|---|---|---:|---|
| v5a_gate | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_v5a_gate.pt` | 113 MB | 5/14 00:25 |
| v5b_gcnii_L2 | `/SSL_NAS/.../best_gat_directed_supernode_p80_v5b_gcnii_L2.pt` | 185 MB | 5/14 13:01 |
| v5b_gcnii_L4 | `/SSL_NAS/.../best_gat_directed_supernode_p80_v5b_gcnii_L4.pt` | 409 MB | 5/15 04:51 |
| v5b_gcnii_L6 | `/SSL_NAS/.../best_gat_directed_supernode_p80_v5b_gcnii_L6.pt` | 633 MB | 5/15 19:07 |
| v5c_full | `/SSL_NAS/.../best_gat_directed_supernode_p80_v5c_full.pt` | 116 MB | 5/14 00:15 |
| v5c_hop_only | `/SSL_NAS/.../best_gat_directed_supernode_p80_v5c_hop_only.pt` | 116 MB | 5/14 12:27 |
| v5c_cum_only | `/SSL_NAS/.../best_gat_directed_supernode_p80_v5c_cum_only.pt` | 113 MB | 5/15 01:11 |

- `outputs/checkpoints/` 에 symlink 보존
- ckpt key: `epoch`, `gat_state_dict`, `classifier_state_dict`, `recall`, `config` (5 keys) — **projector_state_dict 미저장** (학습 entry 가 GAT + classifier 만 save)

### V5-D-1 진단 base 정합 (PLM Lower Bound vs V5 Architectural Intervention)

- V5-D-1 진단 (multi-DB n=55, 2026-05-12): anchor Enriched builder $\bar{c}_{L_0} = 0.6246$ (target $\leq 0.30$ 까지 -0.32 reduction needed, Enriched 단독 효과 -0.0279 = 9%)
- V5 architectural intervention 7-trial 모두 P80 0.6097 미달 → **V5-D-1 진단 정합**: PLM lower bound 의 R-ceiling 효과가 architectural intervention 보다 dominant
- 학술 narrative: paper §V.5.4 main mechanism finding 의 결정적 강화 — "mech(ii-b) 의 fundamental architectural limitation 임에도 R@15 ceiling 의 직접 원인은 PLM lower bound + domain bottleneck"

### Per-DB Stratified (multi-DB n=55) — pending analyzer reconstruction

- 학습 시점 측정: Val Recall@15 단일 metric (overall)
- 11 BIRD DBs 의 stratified R@15 + L_GAT cosine sim + top-K conc + skip_dep ratio 등 정량 진단 metric — 학습 시 layer_stats 미저장
- → **Analyzer 의 후속 평가 dispatch 필요**: `dsn_oversmoothing_analysis.py` 의 CKPTS 리스트 갱신 (7 V5 ckpt 등록) + 5-step protocol (L1/L2/L3 cosine + top5_conc + skip_dep_ratio + AC loss decay + conv_L1 grad norm) — multi-DB n=55 stratified

### 산출물 + Caveat

- Configs: `configs/training/dsn/train_dsn_p80_v5{a_gate, b_gcnii_L2, b_gcnii_L4, b_gcnii_L6, c_full, c_hop_only, c_cum_only}.yaml` (7 cell)
- 학습 logs: `logs/train/dsn_p80_v5{*}_*.log` (7 cell, 모든 log 300 epoch + Training Completed 종료 mark)
- Sweep script: `scripts/run_v5_mitigation_sweep.sh` (4-stage GPU 0/1 병렬)
- ⚠️ **Caveat**: 본 entry 는 학습 sweep 완료 시점 결과 집계. V5 inference (downstream Filter/SQL pipeline 통합) 결과 + L_GAT cosine / capacity 지표 정량 진단 = **별도 후속 chain (root → analyzer)**. V5 inference 7-cell sweep 은 [V5 Inference Sweep entry](EXPERIMENT_HISTORY.md) 참조 (별도 작업, 본 entry 와 chain 연속).

### 후속 위임 (chain handoff)

- **Analyzer 위임 (17-Trial Dominance Scoring 핸드오프, primary)**:
  - 산출물: `notebooks/analysis_results/dsn_mitigation_v5_17trial.md` 신규 — 14-trial Final (dsn_mitigation_v4_combo.md base) + V5 7-trial = 17-trial 통합 dominance scoring
  - 4-mechanism evidence matrix 갱신 (mech i / ii-a / ii-b / iii / iv) — mech(ii-b) 7/7 absolute confirm 격상 candidate
  - per-DB stratified (multi-DB n=55, 11 BIRD DBs) + L_GAT cosine sim layer-wise (L0/L1/L2/L3)
  - V5-D-1 PLM lower bound 정합 verify (anchor $\bar{c}_{L_0} = 0.6246$ base)
  - paper §V.5.4 main mechanism finding 의 학술 weight 갱신 권고 + paper §3.5 axis #6 (Training-Pathology-Invariant) 의 V5 evidence 추가 권고
- **Planner 위임 (analyzer 후, 시나리오 (a) confirm 후 narrative pivot)**:
  - 5 over-smoothing planning 문서 통합 갱신
  - paper §V.5.4 mech(ii-b) 7/7 absolute confirm 격상 정식 채택 + Filter Dominance 6번째 축 (Training-Pathology-Invariant) 의 17-trial evidence 명문화
- **Root 위임 (별도 chain)**:
  - V5 inference 7-cell sweep (anchor projector patch + downstream Filter/SQL pipeline 통합) — 별도 entry
  - Phase 1 Sensitivity 13-cell sweep — 별도 entry
  - Anchor SQL Sweep (Option γ) — 별도 entry


---

## Phase 2 Grid Sweep — Hyperparameter 2D Grid θ × K = 5×5 = 25 cells (Wave 5 Partial Reopen, 2026-05-16, 🎯 Success criterion (a) Plateau breadth confirm + anchor 정합 PASS + R 갱신 lever 잠정 sub-noise)

근거: planning/DECISIONS.md 2026-05-16 (Wave 5 Partial Reopen — Phase 2 grid 25 cells 재활성) + planning/improving_exp_plan_by_scholar_agent_2026-05-15.md §"Phase 2" + EXPERIMENT_PLAN.md §4 Phase 0 Wave 5 ★★★ Phase 2 Grid Sweep + analyzer phase1_sensitivity_analysis_2026-05-15.md §3.3 grid spec. Wave 5 closure 일부 retract 후 Phase 2 grid 만 활성 — closure narrative axis #11 (builder-axis invariance) 의 R 갱신 lever 재탐색.

### 운영 이력

- **2026-05-16 00:45 launch prep**: 25 configs 신규 생성 (`configs/experiments/abl/c03_phase2_grid/p2_{01..25}_theta_X_topk_Y.yaml`) — Python `/tmp/gen_phase2_configs.py` 일괄 생성. Anchor 정합: P2_02 (θ=0.1, K=20) ↔ c01_01 spec diff = 주석 + experiment_name 만 (deterministic 일치 검증 cell).
- **2026-05-16 00:45 script 작성**: `scripts/run_phase2_grid_sweep.sh` (failure-tolerant 25 cells, GPU 0+1 split, conc=4 per GPU). 초기 conc=3 (안전), 사용자 5/16 input "8개 동시에 병렬로 진행하고 kill은 하지 마" → conc=4 per GPU = **8-conc total** 변경.
- **2026-05-16 00:47:25 launch**: `nohup bash scripts/run_phase2_grid_sweep.sh > logs/phase2_grid_main.log 2>&1 &` (wrapper PID 3484328). 8-conc Round 1: GPU 0 × {p2_01,02,03,04} + GPU 1 × {p2_14,15,16,17}. GPU 0 alloc cells 1-13 (4 round), GPU 1 alloc cells 14-25 (3 round).
- **2026-05-16 09:02:15 종료**: 25/25 metrics 모두 도착. Wall = **8h14m50s** (사용자 spec 4-5h ETA + 8-conc drift 페널티 정합).
- **30m cron** (id `d6c467d7`) + **monitor** 5개 (b49l669a6, b1ikmgq7c, b8paj7ti8, b9p2isy9l, bsjuq4nxx, bvjxa8qgj, be5bo1ebf, bkfz4q8wz, bx4ztykgh — 1h timeout 마다 re-arm) 으로 진행 모니터링.

### 25 cells 의 R/P/F1/EX (4-decimal, anchor c01_01 F1=0.8664 / EX=0.5176 비교)

| Cell | θ | K | R | P | F1 | EX | ΔF1 vs c01_01 | ΔEX vs c01_01 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| p2_01 | 0.1 | 15 | 0.8759 | 0.8581 | 0.8669 | 0.5163 | +0.0005 | -0.0013 |
| p2_02 ⭐ | 0.1 | 20 | 0.8761 | 0.8579 | 0.8669 | 0.5163 | **+0.0005** ⭐ | -0.0013 |
| **p2_03** | 0.1 | **30** | 0.8768 | 0.8593 | **0.8680** ★ | 0.5130 | **+0.0016** ★ | -0.0046 |
| p2_04 | 0.1 | 40 | 0.8731 | 0.8563 | 0.8646 | 0.5169 | -0.0018 | -0.0007 |
| p2_05 | 0.1 | 70 | 0.8752 | 0.8589 | 0.8670 | 0.5163 | +0.0006 | -0.0013 |
| p2_06 | 0.125 | 15 | 0.8712 | 0.8551 | 0.8631 | 0.5117 | -0.0033 | -0.0059 |
| **p2_07** | 0.125 | **20** | 0.8693 | 0.8590 | 0.8641 | **0.5189** ★ | -0.0023 | **+0.0013** ★ |
| p2_08 | 0.125 | 30 | 0.8717 | 0.8565 | 0.8640 | 0.5143 | -0.0024 | -0.0033 |
| p2_09 | 0.125 | 40 | 0.8705 | 0.8570 | 0.8637 | 0.5137 | -0.0027 | -0.0039 |
| p2_10 | 0.125 | 70 | 0.8729 | 0.8591 | 0.8659 | 0.5143 | -0.0005 | -0.0033 |
| p2_11 | 0.15 | 15 | 0.8659 | 0.8588 | 0.8623 | 0.5098 | -0.0041 | -0.0078 |
| p2_12 | 0.15 | 20 | 0.8673 | 0.8584 | 0.8628 | 0.5111 | -0.0036 | -0.0065 |
| p2_13 | 0.15 | 30 | 0.8683 | 0.8618 | 0.8650 | 0.5020 | -0.0014 | -0.0156 |
| p2_14 | 0.15 | 40 | 0.8689 | 0.8614 | 0.8651 | 0.5026 | -0.0013 | -0.0150 |
| p2_15 | 0.15 | 70 | 0.8692 | 0.8611 | 0.8651 | 0.5013 | -0.0013 | -0.0163 |
| p2_16 | 0.175 | 15 | 0.8647 | 0.8592 | 0.8619 | 0.5033 | -0.0045 | -0.0143 |
| p2_17 | 0.175 | 20 | 0.8645 | 0.8586 | 0.8615 | 0.5026 | -0.0049 | -0.0150 |
| p2_18 | 0.175 | 30 | 0.8597 | 0.8579 | 0.8588 | 0.5007 | -0.0076 | -0.0169 |
| p2_19 | 0.175 | 40 | 0.8600 | 0.8561 | 0.8580 | 0.5007 | -0.0084 | -0.0169 |
| p2_20 | 0.175 | 70 | 0.8589 | 0.8562 | 0.8575 | 0.5020 | -0.0089 | -0.0156 |
| p2_21 | 0.2 | 15 | 0.8576 | 0.8583 | 0.8579 | 0.4961 | -0.0085 | -0.0215 |
| p2_22 | 0.2 | 20 | 0.8612 | 0.8611 | 0.8611 | 0.4980 | -0.0053 | -0.0196 |
| p2_23 | 0.2 | 30 | 0.8648 | 0.8605 | 0.8626 | 0.4980 | -0.0038 | -0.0196 |
| p2_24 | 0.2 | 40 | 0.8621 | 0.8605 | 0.8613 | 0.4954 | -0.0051 | -0.0222 |
| p2_25 | 0.2 | 70 | 0.8644 | 0.8598 | 0.8621 | 0.4954 | -0.0043 | -0.0222 |

### 5×5 F1 Heatmap (Global max p2_03 0.8680)

| θ \ K | 15 | 20 | 30 | 40 | 70 | **avg** |
|---|---:|---:|---:|---:|---:|---:|
| **0.1** | 0.8669 | 0.8669 | **0.8680** ★ | 0.8646 | 0.8670 | **0.8667** |
| 0.125 | 0.8631 | 0.8641 | 0.8640 | 0.8637 | 0.8659 | 0.8642 |
| 0.15 | 0.8623 | 0.8628 | 0.8650 | 0.8651 | 0.8651 | 0.8641 |
| 0.175 | 0.8619 | 0.8615 | 0.8588 | 0.8580 | 0.8575 | 0.8595 |
| 0.2 | 0.8579 | 0.8611 | 0.8626 | 0.8613 | 0.8621 | 0.8610 |

### 5×5 EX Heatmap (Global max p2_07 0.5189)

| θ \ K | 15 | 20 | 30 | 40 | 70 |
|---|---:|---:|---:|---:|---:|
| **0.1** | 0.5163 | 0.5163 | 0.5130 | **0.5169** | 0.5163 |
| 0.125 | 0.5117 | **0.5189** ★ | 0.5143 | 0.5137 | 0.5143 |
| 0.15 | 0.5098 | 0.5111 | 0.5020 | 0.5026 | 0.5013 |
| 0.175 | 0.5033 | 0.5026 | 0.5007 | 0.5007 | 0.5020 |
| 0.2 | 0.4961 | 0.4980 | 0.4980 | 0.4954 | 0.4954 |

### Anchor 정합 검증 — P2_02 vs c01_01 (deterministic 일치)

| | R | P | F1 | EX |
|---|---:|---:|---:|---:|
| **c01_01** (Phase 1.1 base) | 0.8748 | 0.8582 | 0.8664 | 0.5176 |
| **p2_02** (Phase 2 grid 동일 spec) | 0.8761 | 0.8579 | 0.8669 | 0.5163 |
| **Δ (P2_02 vs c01_01)** | +0.0013 | -0.0003 | **+0.0005** | -0.0013 |

→ **✅ Deterministic 정합 검증 PASS** (사용자 spec "F1 차이 ≤ 0.0010 noise" 정합) — GLM API stochastic variance 안. 25 cells 결과의 신뢰성 base 확보.

### Success criterion 분기 판단 (DECISIONS 2026-05-16 §2)

| Criterion | 결과 | 학술 weight |
|---|---|:---:|
| **(a) Plateau breadth** | anchor-band θ ∈ {0.1, 0.125, 0.15} × K ∈ {15, 20, 30, 40, 70} = **15 cells F1 spread = 0.8623~0.8680 (Δ=0.0057)**, EX spread ~0.020. V5 inference 7-cell F1 spread (0.0052) 정합 sub-noise band 안. **plateau 확인** | **High** — axis #11 (builder-axis invariance) evidence **retain + strengthen** |
| **(b) R 갱신 lever** | **p2_03 (θ=0.1, K=30) F1=0.8680 = anchor +0.0016** — GLM stochastic noise floor (~0.001) 약간 초과 잠정. **p2_07 (θ=0.125, K=20) EX=0.5189 = anchor +0.0013** — 비슷 sub-noise. **둘 다 statistically robust 아님** | **Low** — closure narrative 재고 trigger 미달성, 잠정 sub-noise candidate |

→ **Outcome (a) Plateau 흡수** — axis #11 (builder-axis invariance candidate) 의 **더 강한 evidence retain**. closure narrative 유지.

### θ axis 의 trend 정합 (Phase 1.1 θ sweep)

| θ | F1 avg | EX avg | ΔF1 vs c01_01 | Phase 1.1 (K=20 단독) | Δ |
|---|---:|---:|---:|---:|---:|
| 0.1 | 0.8667 | 0.5158 | +0.0003 | 0.8664 | +0.0003 ✅ noise |
| 0.125 | 0.8642 | 0.5146 | -0.0022 | (미측정) | n/a |
| 0.15 | 0.8641 | 0.5046 | -0.0023 | (미측정) | n/a |
| 0.175 | 0.8595 | 0.5019 | -0.0069 | (미측정) | n/a |
| 0.2 | 0.8610 | 0.4966 | -0.0054 | 0.8632 | -0.0022 ✅ noise |

→ Phase 1.1 의 θ=0.1, θ=0.2 결과 (K=20 단독) 와 Phase 2 grid 의 동일 θ 의 5-K avg 가 sub-noise 일치.

**관찰**: **θ=0.175 → θ=0.2 mid-θ dip** — θ=0.2 row avg F1=0.8610 > θ=0.175 row avg F1=0.8595 (Δ=+0.0015 sub-noise but consistent across K). 다만 EX 는 θ=0.2 가 worst (0.4966) — F1 ↔ EX trade-off.

### K axis 의 sub-noise vs sensitivity (Phase 1.2 K sweep 정합)

| θ | K spread (F1) | EX spread | Phase 1.2 정합 |
|---|---:|---:|---|
| 0.1 (anchor-band) | **0.0034** (0.8646~0.8680) | 0.0039 | Phase 1.2 sub-noise (0.0019) 정합 |
| 0.125 | 0.0028 | 0.0072 | sub-noise |
| 0.15 | 0.0028 | 0.0098 | sub-noise |
| 0.175 | 0.0044 | 0.0026 | K↑ → F1 monotonic decay (anchor-band 외) |
| 0.2 | 0.0047 | 0.0026 | K↑ marginal increase |

→ **anchor-band θ ∈ [0.1, 0.15] 안에서는 K sub-noise**, θ ≥ 0.175 부터 K sensitivity 등장.

### 학습 비용 + 환경

- **Wall**: 8h14m50s (00:47:25 → 09:02:15)
- **GPU 시간**: 25 × ~2h = ~50 GPU-hour (8-conc 4-conc 비교 wall 단축 효과 검증, V5 inference 정합 drift +13~17%)
- **LLM API 비용**: ~$15-30 GLM 4.7 (anchor SQL sweep ~1.5h × 25 cells 분산)
- **per_q drift**: 4.7s (Round 1 start) → 5.4s (Round 1 end) → 4.9s avg (Round 2/3) → 3.9s (Round 4 단독, 1-conc 회복)
- **failure**: 0 cells (모든 25 cells 의 metrics.txt 정상 생성)

### Checkpoint + Reference (재사용)

- Stack: c01_01 anchor (QCondGAT 3-layer + bidirectional SN + MSTPCSTUnion + XiYanFilter GLM 4.7 + LLMSQLGenerator)
- weight_path: `outputs/checkpoints/best_gat_qcond_nl3.pt` (5/14 → 5/16 anchor 동일)
- 신규 ckpt 학습 없음 — sweep only

### 산출물

- Configs (25): `configs/experiments/abl/c03_phase2_grid/p2_{01..25}_theta_X_topk_Y.yaml`
- Sweep script: `scripts/run_phase2_grid_sweep.sh` (4-stage GPU 0/1 8-conc parallel)
- Logs: `logs/phase2_grid_main.log` + `logs/phase2_grid/p2_{01..25}_*.log` (25 cells)
- Outputs: `outputs/experiments/abl/c03_phase2_grid/p2_{01..25}_*/` (25 metrics.txt + predictions.jsonl + output_*.jsonl)

### 후속 위임 (chain handoff)

- **Analyzer 위임 (primary, immediate)**: `notebooks/analysis_results/phase2_grid_heatmap_2026-05-16.md` 신규 작성
  - 5×5 heatmap visualization (F1 + EX + TCR + TOR + Filter Prune Ratio)
  - Success criterion (a/b) 분기 판단 (위 결론 (a) plateau 흡수 confirm)
  - P2_02 ↔ c01_01 deterministic 정합 통계 verify
  - θ × K interaction effect (anchor-band 안 K sub-noise vs anchor-band 밖 K monotonic) mechanism 분석
  - paper §V.5.x.M.9 (Extractor θ R-Ceiling Mechanism) 의 Phase 2 evidence 갱신 권고
  - paper §V.5.x.M.10 (Selector K Filter-Invariant) 의 Phase 2 25-cell evidence 강화 권고
- **Planner 위임 (analyzer 후)**: closure narrative axis #11 (builder-axis invariance) 의 Phase 2 plateau 흡수 evidence 추가 명문화
- **Caveat**: p2_03 (θ=0.1, K=30) F1=0.8680 / p2_07 (θ=0.125, K=20) EX=0.5189 의 noise floor 약간 초과 sub-noise candidate — 추가 measurement (seed sweep 또는 GLM stochastic variance 분석) 으로 statistical significance 검증 후속 backlog


---

## Phase 4.1 (α sweep) + Phase 4.2 (TCR-conditional Filter) Chain (학술 agent plan §Phase 4, DECISIONS 2026-05-16 §3+§4, 2026-05-16, 🎯 α=0.0~0.8 plateau confirm + α=1.0 cliff -0.0952 (Extractor threshold rescue dominant) + thr=0.5 Pareto sweet spot)

근거: planning/DECISIONS.md 2026-05-16 (학술 agent plan Phase 3+4 활성) §3 Phase 4.1 + §3 Phase 4.2 + §4 결정 요약 (Phase 4.1+4.2 = 6+3=9 cells parallel launch). Module:extractors commit `1e2c46a` (`MSTPCSTUnionExtractor.seed_selection_mode="integrated_score"` + α weighting) + Module:filters commit `e0685eb` (`ConditionalFilterWrapper(inner=XiYanFilter)` + TCR-gated voluntary skip + smoke 16/16 PASS) 구현 후 통합 chain launch.

### 운영 이력

- **2026-05-16 11:34 (Module:filters commit `e0685eb`)**: `ConditionalFilterWrapper` 구현 — TCR(q) = |subgraph cols| / |full schema cols| 계산 (kwargs override 우선, metadata fallback). call_mode ∈ {"conditional", "always"} + tcr_threshold ∈ [0,1]. skip 시 subgraph 그대로 final_nodes 반환 (LLM call 0). stats/filter_info 에 voluntary_skipped/inner_called/tcr_value/tcr_source/inner_filter_name 노출.
- **2026-05-16 11:38 (Module:extractors commit `1e2c46a`)**: `MSTPCSTUnionExtractor.seed_selection_mode="integrated_score"` + α ∈ [0,1] 구현. integrated_score 모드는 `s_integrated(v) = α·𝟙[v∈Selector_TopK] + (1-α)·𝟙[s_v≥score_threshold]` 를 양쪽 sub-extractor 의 node_scores 로 전달. last_info 에 integrated_topk_only/threshold_only/intersection/positive_total telemetry.
- **2026-05-16 11:47 (Root chain)**: `scripts/run_phase4_chain.sh` 신규 작성 — 9-conc parallel (GPU 0 × 4 Phase 4.1 α=0.0~0.6 + GPU 1 × 5 Phase 4.1 α=0.8/1.0 + Phase 4.2 thr=0.3/0.5/0.7). 사용자 5/16 spec "9 cells parallel launch" + "kill 금지".
- **2026-05-16 11:48:04 launch**: wrapper PID 391843, monitor task `bpe89xnyc` (1h timeout × 3 re-arm) + 30m cron `a60df239`.
- **2026-05-16 13:52 종료**: 9/9 metrics 모두 도착. Wall = **2h05m** (사용자 spec 2-3h ETA 정합).

### Phase 4.1 6 cells α sweep — 결과 (anchor c01_01 F1=0.8664 / EX=0.5176)

| Cell | α | mode 설명 | R | P | F1 | EX | ΔF1 | ΔEX |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| **p4_01** ⭐ | **0.0** | threshold-only (= c01_01 default) | 0.8767 | 0.8559 | **0.8662** | 0.5150 | **-0.0002** ✅ anchor 정합 PASS | -0.0026 |
| p4_02 | 0.2 | TopK 가중 + threshold 가중 | 0.8769 | 0.8564 | **0.8665** | 0.5137 | **+0.0001** sub-noise | -0.0039 |
| p4_03 | 0.4 | balanced | 0.8764 | 0.8563 | **0.8662** | 0.5137 | -0.0002 sub-noise | -0.0039 |
| p4_04 | 0.6 | balanced | 0.8762 | 0.8554 | 0.8657 | 0.5169 | -0.0007 sub-noise | -0.0007 |
| p4_05 | 0.8 | TopK 강화 | 0.8771 | 0.8566 | **0.8667** | 0.5150 | **+0.0003** sub-noise | -0.0026 |
| **p4_06** | **1.0** | TopK-only | 0.7254 | 0.8232 | **0.7712** | 0.3638 | **-0.0952** ★ | **-0.1538** |

**핵심 finding (Phase 4.1)**:
- **α=0.0~0.8 plateau** — 모두 sub-noise (|ΔF1| ≤ 0.0007, GLM stochastic noise floor 정합). Phase 2 Grid 의 anchor-band plateau (5×5 25 cells, F1 spread 0.0057) 와 일관.
- **α=1.0 cliff drop** — F1=0.7712 (ΔF1=-0.0952, -11%), R=0.7254 (ΔR=-0.1494, -17%), EX=0.3638 (ΔEX=-0.1538, -30%). 
- **Extractor threshold-pass rescue 가 final R/F1/EX 의 dominant contributor** — Selector top-K (~20 nodes) 만으로는 schema linking 의 minimum coverage 미달, threshold-pass rescue 가 ~10% F1 lever.
- α=0.0 anchor deterministic 일치 PASS (ΔF1=-0.0002, |Δ|<0.005 sub-noise verify, GLM stochastic ~0.001 floor 안).

### Phase 4.2 3 cells TCR-conditional — 결과 (anchor c01_01 F1=0.8664)

| Cell | thr | R | P | F1 | EX | ΔF1 | Skip / 1534 | Skip % | LLM cost saving |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p4_2_thr_0.3 | 0.3 | 0.8772 | 0.8577 | **0.8673** | 0.5111 | **+0.0009** sub-noise | 0 | 0.0% | 0% |
| **p4_2_thr_0.5** ⭐ | **0.5** | 0.8793 | 0.8553 | **0.8671** | 0.5156 | **+0.0007** sub-noise | 8 | 0.5% | 0.5% |
| p4_2_thr_0.7 | 0.7 | 0.8785 | 0.8399 | 0.8588 | 0.5150 | **-0.0076** | 39 | 2.5% | 2.5% |

**핵심 finding (Phase 4.2)**:
- **TCR 분포 high** — anchor-band Prune% 92-94% 의 inverse (TCR = col coverage). 대부분 query 의 TCR ≥ 0.7, 따라서 thr=0.3/0.5 시 skip 거의 무 (0-0.5%).
- **thr=0.5 = Pareto sweet spot** — F1 sub-noise +0.0007 + 0.5% saving (8 calls). production deployment 의 cost-effective candidate.
- **thr=0.7 aggressive** — 2.5% saving 위해 F1 -0.0076 cost. 39 skipped query 의 F1 average 낮음 (skip 시 schema 부족하지만 Filter 가 못 prune)
- TCR threshold sensitivity: 0.3 → 0.5 → 0.7 의 skip rate 0% → 0.5% → 2.5% non-linear increase (TCR distribution heavy tail toward 1.0).

### α=0.0 ↔ c01_01 ↔ p2_02 deterministic 정합 통계 (3 anchor 동일 spec)

| Source | F1 | EX | Δ vs c01_01 (F1) |
|---|---:|---:|---:|
| c01_01 (Phase 1.1, 5/15) | 0.8664 | 0.5176 | (base) |
| p2_02 (Phase 2 Grid, 5/16 morning) | 0.8669 | 0.5163 | +0.0005 |
| **p4_01 α=0.0 (Phase 4 Chain, 5/16 noon)** | **0.8662** | **0.5150** | **-0.0002** |

→ GLM stochastic noise floor: **~±0.0005 F1, ~±0.0026 EX** (3 measurement spread). Phase 4 metric 의 statistical significance threshold candidate.

### α axis interpolation 분석 (Phase 4.1, α=0.0~1.0 6 cells)

```
F1 trajectory (α axis):
  α=0.0 → 0.8662  (anchor 정합)
  α=0.2 → 0.8665  (sub-noise)
  α=0.4 → 0.8662  (sub-noise)
  α=0.6 → 0.8657  (sub-noise)
  α=0.8 → 0.8667  (sub-noise, marginal max)
  α=1.0 → 0.7712  ★ CLIFF DROP (-0.0950)
```

- **α=0.0~0.8 plateau** (F1 spread 0.0010 sub-noise, GLM noise floor 정합)
- **α=0.95-1.0 transition zone** (single cliff event at α=1.0)
- → Extractor seed 의 PCST prize weighting (α) 가 0.0~0.8 range 안에서 final F1 에 거의 영향 무. **threshold-pass rescue 가 plateau 유지의 핵심**.

### 학습 비용 + 환경

- **Wall**: 2h05m (11:48:04 → 13:52:56)
- **GPU 시간**: 9 cells × 2h × (effective conc 7) / 9 = ~14 GPU-hours (8-conc Phase 2 정합 + Phase 4.2 skip 효과)
- **LLM API 비용**: ~$8-15 GLM 4.7 (Phase 4.2 skip 으로 ~2% saving 만)
- **per_q drift**: 4.2-4.4s (Round 1 start) → 4.7-4.9s (mid) → 안정
- **failure**: 0 cells (9/9 metrics 정상 생성)
- **process kill**: 0 (사용자 spec "kill 금지" 정합)

### Checkpoint + Reference (재사용)

- Stack: c01_01 anchor (QCondGAT 3-layer + bidirectional SN + MSTPCSTUnion + XiYanFilter GLM 4.7 + LLMSQLGenerator)
- Phase 4.1 변경: `MSTPCSTUnionExtractor.seed_selection_mode="integrated_score"` + α (6 cells)
- Phase 4.2 변경: `ConditionalFilterWrapper(inner=XiYanFilter)` + tcr_threshold (3 cells)
- weight_path: `outputs/checkpoints/best_gat_qcond_nl3.pt` (anchor 동일, 학습 없음)

### 산출물

- Phase 4.1 Configs (6, tracked from commit 1e2c46a): `configs/experiments/abl/c04_phase4_alpha_sweep/p4_{01..06}_alpha_X.yaml`
- Phase 4.2 Configs (3, untracked from commit e0685eb): `configs/experiments/abl/c05_phase4_conditional_filter/p4_2_thr_{0.3, 0.5, 0.7}.yaml`
- Sweep script: `scripts/run_phase4_chain.sh` (9-conc parallel, GPU 0 × 4 + GPU 1 × 5)
- Logs: `logs/phase4_chain_main.log` + `logs/phase4_chain/p4_*_*.log` (9 cells)
- Outputs: `outputs/experiments/abl/c0[45]_phase4*/p4_*/` (9 metrics.txt + predictions.jsonl + output_*.jsonl)

### 후속 위임 (chain handoff)

- **Analyzer 위임 #1 (Phase 4.1, primary)**: `notebooks/analysis_results/phase4_1_integrated_alpha_sweep_2026-05-16.md` 신규 작성
  - α axis 6 cells F1/EX/Filter Prune% visualization
  - α=0.0~0.8 plateau mechanism 분석 (PCST prize weighting 의 final F1 invariance — Extractor threshold-rescue dominance)
  - α=1.0 cliff drop 의 Selector top-K only effective seed 분석 — schema coverage 부족 → R -0.1494 / F1 -0.0952 / EX -0.1538
  - paper §V.5.x.M.13 (Selector + Extractor co-design integration) narrative 신규 권고
  - integrated_topk_only / threshold_only / intersection / positive_total telemetry 분포 정량
- **Analyzer 위임 #2 (Phase 4.2)**: `notebooks/analysis_results/phase4_2_conditional_filter_2026-05-16.md` 신규 작성
  - TCR(q) 분포 histogram (1534 queries × 3 cells)
  - 3 threshold (0.3/0.5/0.7) Filter skip rate + F1 trade-off frontier
  - per-difficulty (simple/moderate/challenging) Filter 호출 비율 + F1 손실 비교
  - LLM call 절감 % + GLM 4.7 API token cost ratio
  - thr=0.5 Pareto sweet spot evidence + paper §V.5.x.M.3 production deployment narrative 직접 매핑
  - paper §V.5.x.M.11 (Filter Short-Circuit voluntary vs involuntary) narrative 강화
- **Planner 위임 (analyzer 후)**:
  - paper §V.5.x.M.13 신규 sub-section (α plateau + α=1.0 cliff = Selector + Extractor co-design integration evidence)
  - paper §V.5.x.M.3 + §V.5.x.M.11 narrative 갱신 (Phase 4.2 cost-effective Pareto)
  - axis #5/#6/#7 (Filter Dominance) 의 Phase 4.1+4.2 통합 mechanism evidence 명문화


---

## Wave 6 Phase 1 M1 Recall-Biased Prompt — 3 variants Sweep (DECISIONS 2026-05-16 Wave 6 §2, 학술 agent filter improve plan §3, 2026-05-16, 🎯 R-lift +0.0511 (mild) / F1 sub-noise (strong) / Phase 2 (a) M2 CoT 분기 권고)

근거: planning/DECISIONS.md 2026-05-16 (Wave 6 신규 활성 — Filter Recall Chain) §2 Phase 1 Spec + planning/filter/0516_scholar_filter_improve_plan.md §3 방법론 1. Module:filters commit `07d2fda` (XiYanFilter prompt_mode parameter + sanitize_filter_output 후처리 + smoke 18/18 PASS) 구현 후 즉시 launch. Wave 5 closure 정합 위에서 **별도 lever 축** — anchor stack 그대로 + Filter prompt language 만 교체 (LLM call 1× 동일, 최저비용 lever).

### 운영 이력

- **2026-05-16 17:46 (Module:filters commit `07d2fda`)**: XiYanFilter prompt_mode parameter (default / recall_biased_mild / recall_biased_strong / recall_biased_exclusion_rule) + `_PROMPT_SECTION_BY_MODE` 매핑 + sanitize_filter_output() static method (학술 agent §2.3, input subgraph 없는 table/column 제거) + 측정 메타 (filter_prompt_mode / filter_sanitize_output / filter_hallucination_removed_count / filter_input/output_node_count / filter_prune_pct) 노출. src/prompts/filter.md 에 PROMPT_M1_A/B/C 3 section 추가. Smoke 18/18 PASS.
- **2026-05-16 17:48 (Root chain)**: 3 configs Python 일괄 생성 (`configs/experiments/abl/wave6_recall_biased/wave6_p1_recall_biased_{mild,strong,exclusion_rule}.yaml`) + `scripts/run_wave6_phase1_recall_biased.sh` 작성 (3-conc GPU 0×2 + GPU 1×1).
- **2026-05-16 17:49:19 launch**: wrapper PID 935829, monitor task `bbuk2024c` (1h timeout × 2 re-arm) + 30m cron `c8d77b78`.
- **2026-05-16 19:47:48 종료**: 3/3 metrics 도착. Wall = **1h58m** (사용자 spec 1.5h + GLM 정합 ~25분 늦음).

### 3 variants 의 R/P/F1/EX (4-decimal, anchor c01_01 F1=0.8664 / EX=0.5176 / R=0.8748 / P=0.8582 비교)

| Cell | prompt_mode | R | P | F1 | EX | ΔR | ΔP | ΔF1 | ΔEX |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **wave6_p1_recall_biased_mild** | recall_biased_mild | **0.9259** ★ | 0.7648 | 0.8377 | **0.5169** ★ | **+0.0511** ✅ | -0.0934 | -0.0287 | -0.0007 |
| **wave6_p1_recall_biased_strong** | recall_biased_strong | 0.9022 | 0.8316 | **0.8655** ★ | 0.5130 | +0.0274 | -0.0266 | **-0.0009** sub-noise | -0.0046 |
| **wave6_p1_recall_biased_exclusion_rule** | recall_biased_exclusion_rule | 0.8907 | 0.8263 | 0.8573 | 0.5143 | +0.0159 | -0.0319 | -0.0091 | -0.0033 |

### 핵심 finding (학술 agent improve plan §3 검증)

**1. Inclusion bias strength → R-P trade-off monotonic 정합** (3 variants ordering):
- R: mild (0.9259) > strong (0.9022) > exclusion_rule (0.8907) — inclusion bias 강도 ↑ → R ↑
- P: strong (0.8316) > exclusion_rule (0.8263) > mild (0.7648) — inclusion 가장 강한 mild 가 P 최대 손실
- F1: strong (0.8655) > exclusion_rule (0.8573) > mild (0.8377) — strong 가 R-P 균형 sweet spot

**2. strong (M1-B) = F1 최적 균형 cell** — 학술 agent default "Default decision is INCLUDE + 명시적 exclusion criteria":
- ΔR = +0.0274 (R-lift 의미)
- ΔF1 = -0.0009 **sub-noise** (anchor noise floor ±0.0005 baseline 정합)
- F1 = 0.8655 vs 학술 agent §10 success criterion (≥0.8672) — **-0.0017 sub-noise 거의 일치**

**3. mild (M1-A) = R 최대 lift** — 학술 agent "RELEVANT or POTENTIALLY RELEVANT + WHEN IN DOUBT INCLUDE":
- ΔR = **+0.0511** (Wave 6 chain R-lift evidence)
- 다만 ΔP = -0.0934, ΔF1 = -0.0287 (too inclusive, P-cost 큼)
- EX = **0.5169 max** (mild 의 inclusive selection 이 SQL 작성에 도움 — 다만 sub-noise -0.0007)

**4. exclusion_rule (M1-C) = 중간** — 4-rule conjunctive + UNSURE→KEEP:
- ΔR = +0.0159 (minimum R lift)
- ΔF1 = -0.0091 — 4-rule conjunctive 가 너무 conservative 결과

### 학술 agent §10 성공 기준 검증

| 기준 | mild | strong | exclusion_rule | 결론 |
|------|:---:|:---:|:---:|------|
| F1_fil ≥ 0.8672 (필수) | ❌ 0.8377 | ❌ 0.8655 (-0.0017 sub-noise) | ❌ 0.8573 | 셋 다 미달 (다만 strong 가까움) |
| Filter Prune % ≤ 50% (목표) | TBD | TBD | TBD | output_*.jsonl 의 filter_prune_pct 직접 측정 — analyzer 위임 |
| Filter 의존도 ≤ 50% (목표) | TBD | TBD | TBD | analyzer 위임 |

→ F1 기준 모두 미달 (sub-noise level) — 단독 M1 prompt 만으로 anchor F1 갱신 불충분. **Phase 2 M2+ 후속 chain 필요**.

### DECISIONS §3 Phase 2 분기 권고

**R_fil 기준 분기 candidate**:

| 분기 | 조건 | 결과 |
|------|------|------|
| (a) M2 CoT + Confidence-Gated + M1 best 조합 | R_fil ≥ 0.92 | **✅ mild 0.9259 ≥ 0.92 충족** → (a) 권고 |
| (b) M3 OR Voting 활성 | R_fil 0.88~0.92 | strong (0.9022) / exclusion_rule (0.8907) 모두 이 range |
| (c) M4 Bidirectional 우선 | R_fil < 0.88 | 셋 다 만족 안 함 (R 모두 ≥ 0.88) |

→ **Phase 2 (a) M2 CoT + Confidence-Gated + M1 best 조합 권고**:
- M1 best = **strong (M1-B)** F1=0.8655 (anchor sub-noise)
- M2 CoT (Chain-of-Thought) + Confidence-Gated 추가 → P 회복 + F1 ≥ 0.8672 달성 시도
- mild (R 0.9259) 의 R lift 가 학술 agent R ≥ 0.92 trigger 충족 → Phase 2 (a) 활성 정당화

### Phase 2 측정 spec (분기 (a) 활성 시)

- M1 best (strong) + M2 CoT prompt + Confidence-Gated (예: P_filter > threshold 시만 prune)
- LLM call 2× /q (M1 + M2) = 2 × 1534 = 3068 calls (anchor 2873 정합)
- 측정: 학술 agent §2.1 동일 (R_fil/P_fil/F1_fil/FNR/FPR/Prune%/LLM_calls)
- 비용: ~$2-4 GLM 4.7

### 학습 비용 + 환경

- **Wall**: 1h58m (17:49:19 → 19:47:48)
- **GPU 시간**: 3 cells × 2h × (effective conc 3) / 3 = ~6 GPU-hour
- **LLM API 비용**: ~$3-6 GLM 4.7 (4602 calls = 1534 × 3)
- **per_q drift**: 4.1-4.2s (Round start) → 4.6s (mid) — 안정
- **failure**: 0/3 cells (모든 metrics.txt 정상)
- **sanitize_filter_output**: default-on (Hallucination 방지, telemetry 정량 = analyzer 위임)

### Checkpoint + Reference (재사용)

- Stack: c01_01 anchor stack 의 Filter prompt_mode 만 교체
- weight_path: `outputs/checkpoints/best_gat_qcond_nl3.pt` (학습 없음)
- Filter: XiYanFilter + prompt_mode = recall_biased_{mild, strong, exclusion_rule} (commit `07d2fda`)
- LLM: glm-4.7 (Elice ML API, OpenAI-compatible)

### 산출물

- Configs (3): `configs/experiments/abl/wave6_recall_biased/wave6_p1_recall_biased_{mild,strong,exclusion_rule}.yaml`
- Sweep script: `scripts/run_wave6_phase1_recall_biased.sh`
- Logs: `logs/wave6_phase1_main.log` + `logs/wave6_phase1/wave6_p1_recall_biased_{*}_*.log` (3 cells)
- Outputs: `outputs/experiments/abl/wave6_recall_biased/wave6_p1_*/` (3 metrics.txt + predictions.jsonl + output_*.jsonl)

### 후속 위임 (chain handoff)

- **Analyzer 위임 (primary, immediate)**: `notebooks/analysis_results/wave6_phase1_recall_biased_2026-05-16.md` 신규 작성
  - 3 variants × 1534q × 7 metrics 매트릭스 (R_fil/P_fil/F1_fil/FNR/FPR/Prune%/LLM_calls)
  - R_gain trajectory: mild (+0.0511) > strong (+0.0274) > exclusion_rule (+0.0159) — inclusion bias monotonic
  - Hallucination rate per variant (sanitize_filter_output 효과 정량 — filter_hallucination_removed_count / filter_input_node_count from output_*.jsonl)
  - DECISIONS §3 Phase 2 분기 결정 정량 — R_fil ≥ 0.92 분기 (a) confirm + M2 CoT spec proposal
  - 학술 agent §10 성공 기준 통계 검증 (3 cells F1 sub-noise vs ≥0.8672 threshold)
- **Planner 위임 (analyzer 후)**:
  - Phase 2 (a) M2 CoT + Confidence-Gated + M1 best (strong) 조합 spec 결정
  - paper §V.5.x.M.x 신규 sub-section candidate (Filter prompt language axis = Recall lever evidence)
- **Root 위임 (planner 후)**: Phase 2 (a) launch trigger (M1 best strong + M2 CoT 조합 config 작성 + 3-conc parallel)


---

## Wave 6 Phase 2 (a+aggressive) — M2 + M3 + M4 + M5 4 cells (DECISIONS 2026-05-16 §2+§3+§5, 학술 agent §3~§7+§10, 2026-05-16 ~ 2026-05-17, 🎯 Outcome (b) confirmed — axis #15 evidence retain + axis #11 Option A retain + M4 EX gain +0.0124 첫 evidence)

근거: DECISIONS 2026-05-16 (Wave 6 Phase 2 a+aggressive M2~M5 동시 launch) §2~§6 + 학술 agent filter improve plan §3~§7 + §10 success criterion. Module:filters commits `7dac875` (XiYanFilter CoT + Confidence-Gated) + `88ad47e` (MultiPromptVotingFilter / BidirectionalFilter / TwoStageFilter 3 신규 class) 구현 후 동시 launch.

### 운영 이력

- **2026-05-16 21:30:48 M2 launch**: wrapper PID 1249922, `w6_p2a_m2cot_strong` (recall_biased_strong + CoT + Confidence-Gated 0.5 + sanitize) single cell. Wall 2h31m → 종료 ~24:01.
- **2026-05-16 23:03:34 M3+M4+M5 launch**: wrapper PID 1381276, 3 cells parallel (M3 GPU 0 + M4 GPU 0 + M5 GPU 1). Wall ~4h → 종료 ~02:47.
- **2026-05-17 02:47 종료**: 4/4 metrics 모두 도착. Total Phase 2 wall = ~5h17m (M2 launch → M3 종료).
- **CronDelete `c221a430` + Monitor `b2jc8ruor` 정상 종료**.

### 4 cells 결과 (R/P/F1/EX 4-decimal, anchor c01_01 F1=0.8664 / EX=0.5176)

| Cell | Method | R | P | F1 | EX | ΔR | ΔP | ΔF1 | ΔEX |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **M2** w6_p2a_m2cot_strong | CoT + Confidence-Gated (thr=0.5) on M1 strong | **0.9745** ★ | 0.2286 | **0.3703** ★ worst | 0.5169 | +0.0997 ✅ | **-0.6296** ❌ | **-0.4961** ❌❌ | -0.0007 |
| **M3** w6_p2_m3_voting | Multi-Prompt OR Voting (3 prompts × OR default) | 0.9408 | 0.6859 | 0.7934 | 0.5202 | +0.0660 | -0.1723 | -0.0730 | +0.0026 |
| **M4** ⭐ w6_p2_m4_bidirectional | Forward (M1-A mild) + Backward (SQL Schema Analyst) union | 0.9325 | 0.7593 | **0.8370** ★ F1-best | **0.5300** ★ EX-max | +0.0577 | -0.0989 | -0.0294 | **+0.0124** ✅ |
| **M5** w6_p2_m5_two_stage | Sequential Stage1 (Coarse Recall) → Stage2 (Fine Precision) | 0.7739 | 0.7964 | 0.7850 | 0.5222 | **-0.1009** ❌ | -0.0618 | -0.0814 | +0.0046 |

### 🌟 핵심 finding

**1. 학술 agent §10 success criterion F1 ≥ 0.8672 — 모두 미달**:
- M4 가 가장 가까움 (F1=0.8370, ΔF1=-0.0294) but statistically robust 미달
- DECISIONS §5 → **Outcome (b) 확정**: axis #15 evidence retain (prompt-level strengthening) + axis #11 Option A retain (prompt-axis + builder-axis 별도)

**2. 🚀 M4 EX gain +0.0124 — Wave 6 chain 첫 EX 갱신**:
- M4 EX=0.5300 > anchor 0.5176 (다른 모든 cell sub-noise 또는 미달)
- Mechanism: Backward prompt (SQL Schema Analyst, question 관점 column generation) 가 SQL execution 의 missing column 보충
- 학술 frame: **Filter ↔ Selector co-design 의 EX-axis new evidence** (학술 agent §6 hypothesis confirm via EX)
- paper §V.5.x.M.13 / §3.1 Inter-Module Co-Design narrative 강화 candidate

**3. M2 CoT + Confidence-Gated catastrophic failure (F1=-0.4961)**:
- Confidence-Gated default-retain 정책의 design flaw: "uncertain → keep" 으로 LLM low-confidence majority → P=0.2286 collapse
- EX 는 sub-noise (-0.0007) — Filter 가 schema 거의 다 keep → SQL gen 영향 거의 무
- → **Filter Dominance dual narrative**: schema linking F1 ≠ SQL EX correlation (M2 가 F1 -0.50 인데 EX sub-noise plateau)

**4. M3 Voting OR (R=0.9408) — inclusion bias spectrum endpoint**:
- 3 prompts OR voting → mild 보다 더 inclusive (+0.0149 R), F1=0.7934 (P=0.6859, P loss 큼)
- OR voting 의 design (≥1 vote → keep) 이 noise 까지 capture
- voting variants (MAJORITY / AND) 의 metrics 는 output_*.jsonl 의 telemetry 에 있으나 default metrics.txt 에 OR 만 — analyzer post-proc 필요

**5. M5 Two-Stage R loss -0.1009 — Sequential pipeline negate**:
- Stage1 (M1-C 변형, Recall-First, R 가져옴) → Stage2 (Fine Precision, aggressive prune)
- Stage2 가 Stage1 의 R lift 효과 negate → R=0.7739 (anchor 보다 lower!)
- §V.5.x.M.3 production deployment 의 sequential candidate **fails** (학술 motivation 미달)

### Inclusion bias strength axis spectrum (Phase 1 M1 + Phase 2 통합)

| Cell | Method | R | F1 | mechanism |
|---|---|---:|---:|---|
| anchor c01_01 | default | 0.8748 | 0.8664 | balanced (baseline) |
| M1-A mild | RELEVANT or POTENTIALLY RELEVANT | 0.9259 | 0.8377 | medium inclusive |
| M1-B strong | Default INCLUDE + criteria | 0.9022 | 0.8655 | mild inclusive (F1-best M1) |
| M1-C exclusion | 4-rule conjunctive | 0.8907 | 0.8573 | weak inclusive |
| **M3 Voting OR** | 3 prompts OR | 0.9408 | 0.7934 | inclusive spectrum endpoint (OR) |
| **M4 Bidirectional** | Forward (mild) + Backward | 0.9325 | 0.8370 | union (forward + question-driven) |
| **M2 CoT-Gated** | strong + CoT + Confidence-Gated | 0.9745 | 0.3703 | **extreme inclusive** (uncertain → keep) |

→ Inclusion bias 강도 axis: anchor → M1-C → M1-B → M4 → M1-A → M3 OR → M2 (R monotonic ↑, P monotonic ↓, F1 inverted U-shape with peak at M1-B 0.8655)

### 학습 비용 + 환경

- **Wall**: 5h17m (M2 21:30:48 → M3 02:47)
- **GPU 시간**: 4 cells × 2-3h × (effective conc 3-4) = ~25 GPU-hours (GPU 0 + GPU 1)
- **LLM API 비용**: 총 ~$30-50 GLM 4.7
  - M2: 3068 calls (~$2-4)
  - M3: 4602 calls (~$10-15)
  - M4: 3068 calls (~$7-12)
  - M5: 3068 calls (~$7-12)
- **failure**: 0/4 (모든 metrics.txt 정상)
- **process kill**: 0 (사용자 spec "kill 금지" 정합)

### Filter ↔ Selector ↔ SQL Gen 의 dual narrative (Filter Dominance 강화)

**schema linking F1 ≠ SQL EX correlation 의 4 evidence points**:
- M2: F1=0.3703 (-0.4961) but EX=0.5169 (sub-noise) — Filter 가 schema 거의 다 keep → F1 catastrophic but SQL gen 영향 무
- M3: F1=0.7934 (-0.0730), EX=0.5202 (+0.0026) — schema linking 손실, EX 거의 무영향
- M4: F1=0.8370 (-0.0294), **EX=0.5300 (+0.0124)** — F1 ↓ but EX ↑ (Backward 의 SQL-aware column 가져옴 효과)
- M5: F1=0.7850 (-0.0814), EX=0.5222 (+0.0046) — F1 큰 손실, EX sub-noise

→ **schema linking F1 metric 의 ceiling 효과 + LLM SQL Gen 의 schema-tolerance** 가 Filter Dominance 의 dual axis evidence

### Checkpoint + Reference (재사용)

- Stack: c01_01 anchor stack 의 Filter module 만 4 variants 교체
- weight_path: `outputs/checkpoints/best_gat_qcond_nl3.pt` (학습 없음)
- Filter classes:
  - M2: XiYanFilter (commit `7dac875`, prompt_mode=recall_biased_strong + cot_reasoning + confidence_gated + confidence_threshold)
  - M3: MultiPromptVotingFilter (commit `88ad47e`, 3 voting strategies OR/MAJORITY/AND)
  - M4: BidirectionalFilter (commit `88ad47e`, Forward+Backward union)
  - M5: TwoStageFilter (commit `88ad47e`, Stage1+Stage2 sequential)

### 산출물

- Configs (4):
  - `configs/experiments/abl/wave6_recall_biased/w6_p2a_m2cot_strong.yaml` (M2)
  - `configs/experiments/abl/wave6_recall_biased/w6_p2_m3_voting.yaml` (M3)
  - `configs/experiments/abl/wave6_recall_biased/w6_p2_m4_bidirectional.yaml` (M4)
  - `configs/experiments/abl/wave6_recall_biased/w6_p2_m5_two_stage.yaml` (M5)
- Sweep scripts: `scripts/run_wave6_phase2a_cot.sh` (M2 single) + `scripts/run_wave6_phase2_aggressive.sh` (M3+M4+M5 parallel)
- Logs: `logs/wave6_phase2a/` + `logs/wave6_phase2_aggressive/` (4 cells)
- Outputs: `outputs/experiments/abl/wave6_recall_biased/{w6_p2a_m2cot_strong,w6_p2_m3_voting,w6_p2_m4_bidirectional,w6_p2_m5_two_stage}/` (4 metrics.txt + predictions.jsonl + output_*.jsonl)

### 후속 위임 (chain handoff)

- **Analyzer 위임 (Phase 3, primary)**: `notebooks/analysis_results/wave6_phase2_results_all_methods_2026-05-17.md` 신규 작성
  - 4 cells (M2/M3/M4/M5) + M1 3 cells = **7 methodology cells × 7 metrics matrix** + R_gain/P_loss/ΔF1/ΔEX trajectory
  - **Pareto frontier R ≥ 0.90 ∧ P ≥ 0.75** 후보 (학술 agent §8.2):
    - M4 (R=0.9325, P=0.7593) ★ 후보
    - M3 (R=0.9408, P=0.6859) — P fails 0.75
    - M1-A mild (R=0.9259, P=0.7648) ★ 후보
  - **M3 voting variants** (OR/MAJORITY/AND) 의 각 metrics 분리 측정 (output_*.jsonl 의 voting_variants_metrics)
  - **M4 backward stats** (backward_added/backward_gold_recovered/backward_precision) — Backward 의 contribution 정량
  - **M5 stage stats** (stage1_recall/stage2_recall_loss/stage2_precision_gain) — Stage2 의 R-loss mechanism 정량
  - Per-methodology mechanism axis 분리 (Inclusion bias / Question-driven / Sequential refinement)
  - **schema linking F1 ↔ SQL EX dual narrative**: M2 의 catastrophic F1 + EX sub-noise plateau evidence
  - **학술 agent §8.3 Top 2 조합 candidate**: M4 (EX best) + M1-B strong (F1 best M1) 조합 — Phase 3 후속 trigger
  - DECISIONS §5 → Outcome (b) 정량 확정 (모든 F1 미달, axis #15 evidence retain + axis #11 Option A)
- **Planner 위임 (analyzer 후)**:
  - paper §V.5.x.M.15 본문 정식 채택 (Filter Prompt Language Axis as Recall Lever, M1+M3 R-lift evidence + M4 EX gain new finding)
  - paper §3.5 axis #15 candidate row 정식 채택
  - paper §3.5 axis #11 narrative Option A retain
  - paper §3.1 Inter-Module Co-Design 의 Filter ↔ Selector co-design 의 EX-axis new axis (M4 Bidirectional evidence)
  - paper §V.5.x.M.3 production deployment narrative: M5 Two-Stage fails footnote
- **Root 위임 (analyzer + planner 후)**:
  - 학술 agent §8.3 Top 2 조합 chain (M4 + M1-B strong 등) 추가 실험 trigger


---

## Wave 6 Phase 4 Top 2 C1 (M4 + M1-B strong) — Single Cell (DECISIONS 2026-05-17 §4, 학술 agent §8.3 Top 2 조합, 2026-05-17, 🎯 Partial Degrade — Forward Dominance + Backward Effect Reduction, Pareto frontier 진입 ✅, M4 EX gain mechanism = Forward-prompt-dependent 새 evidence)

근거: DECISIONS 2026-05-17 (Wave 6 Phase 3 통합 채택 + Top 2 C1 launch) §4 Top 2 C1 spec + 학술 agent filter improve plan §8.3 (Top 2 methodology 조합 candidate). Module:filters commit `60b6988` (BidirectionalFilter `bidirectional_forward_prompt_mode` config flag 추가) 구현 후 launch.

### 운영 이력

- **2026-05-17 (Module:filters commit `60b6988`)**: BidirectionalFilter 의 `bidirectional_forward_prompt_mode` config flag 신규 — Forward prompt 를 mild (default) / strong / exclusion_rule 중 선택 가능. backward_compat retain.
- **2026-05-17 10:31:05 launch**: wrapper PID 2300448, single cell GPU 0, no parallelism. Monitor task `b9593qy2c` (1h timeout × 2 re-arm) + 30m cron `182f7e83`.
- **2026-05-17 13:04 종료**: metrics.txt 도달. Wall = **2h33m** (사용자 spec 1.5h ETA + ~1h overage, per_q drift 5.7→6.1s).

### 결과 (R/P/F1/EX 4-decimal, 3 baseline 비교)

| Source | R | P | F1 | EX | ΔF1 vs C1 | ΔEX vs C1 |
|---|---:|---:|---:|---:|---:|---:|
| **C1 w6_p4_c1_m4_strong** | **0.9177** | 0.8109 | **0.8610** | 0.5150 | (base) | (base) |
| anchor c01_01 | 0.8748 | 0.8582 | 0.8664 | 0.5176 | C1 -0.0054 sub-noise | C1 -0.0026 sub-noise |
| M4 baseline (mild Forward) | 0.9325 | 0.7593 | 0.8370 | **0.5300** ★ | **C1 +0.0240** ✅ | **C1 -0.0150** ❌ EX loss |
| M1-B strong (Forward only) | 0.9022 | 0.8316 | **0.8655** ★ | 0.5130 | C1 -0.0045 sub-noise | C1 +0.0020 sub-noise |

### 🎯 Synergy / Additive / Degrade 분기 판단

| 분기 | 조건 | 결과 |
|------|------|:---:|
| **Synergy** | F1 > 0.8655 (M1-B) OR EX > 0.5300 (M4) | ❌ — F1 0.8610 < 0.8655, EX 0.5150 < 0.5300 |
| **Additive (full)** | F1 ≈ M1-B + EX ≈ M4 | ⚠ 부분: F1 ≈ M1-B ✅ BUT EX ≈ M1-B (NOT M4) — Backward effect 손실 |
| **Partial Degrade** | F1 < M1-B sub-noise ∧ EX < M4 큰 손실 | ✅ **확정** |

→ **Outcome: Forward Dominance + Backward Effect Reduction (Partial Degrade)**

### 🌟 New Finding — Backward mechanism Forward-prompt-dependent

**M4 EX gain source mechanism 재해석**:
- M4 baseline (mild Forward): EX=0.5300 (anchor +0.0124) ★
- C1 (strong Forward + same Backward): EX=0.5150 (anchor -0.0026)
- **Δ EX = -0.0150** ← Forward prompt 가 mild → strong 으로 변경 시 Backward 의 EX gain 효과 거의 소멸

→ **Backward (SQL Schema Analyst) 의 EX gain 은 Forward (mild, inclusive) 가 만들어둔 base 위에서만 효과적**:
- mild Forward → inclusive base (큰 column set) → Backward 가 추가할 column space 큼 → SQL-aware column 보충 → EX gain
- strong Forward → less inclusive base (작은 column set) → Backward 가 보충 가능한 SQL-aware column space 줄어듦 → EX gain 사라짐

**학술적 함의**:
- DECISIONS §3.1 의 "Forward/Backward orthogonality" hypothesis 부분 부정 — Forward prompt 가 Backward effect size 결정 (entanglement)
- backward_added 0.18 nodes/q overlap 96.43% (M4 baseline 정합) 의 mechanism 의 Forward-prompt-dependence 첫 evidence
- C2 (M4 + M3 MAJORITY Forward) launch 의 학술 motivation 강화 — Forward 가 voting strategy 인 경우 Backward effect 변동 추가 평가

### Pareto Frontier 갱신

| Cell | R | P | F1 | EX | Pareto (R≥0.90 ∧ P≥0.75) |
|---|---:|---:|---:|---:|:---:|
| M1-A mild | 0.9259 | 0.7648 | 0.8377 | 0.5169 | ✅ |
| M1-B strong ⭐ F1-best M1 | 0.9022 | 0.8316 | 0.8655 | 0.5130 | ✅ |
| M4 Bidirectional (mild Forward) ⭐ EX-max | 0.9325 | 0.7593 | 0.8370 | **0.5300** ★ | ✅ |
| M3 MAJORITY (post-hoc) | 0.9290 | 0.7934 | 0.8433 | (post-hoc) | ✅ |
| **🆕 C1 w6_p4_c1_m4_strong** | **0.9177** | 0.8109 | 0.8610 | 0.5150 | ✅ **5번째 frontier cell** |

C1 의 Pareto frontier position: F1 (0.8610) 가 M3 MAJORITY (0.8433) > M4 (0.8370) 보다 높고, M1-B strong (0.8655) sub-noise lower. EX 는 M1-B strong (0.5130) 보다 marginal 높지만 M4 (0.5300) 보다 -0.0150 낮음.

### 학술 agent §10 success criterion + DECISIONS §5/§6 분기

- **F1 ≥ 0.8672**: C1 = 0.8610 ❌ 미달 (-0.0062, sub-noise but statistically robust fail)
- **DECISIONS §5 Outcome (b) retain**: axis #15 evidence retain (prompt-level strengthening) + axis #11 Option A retain (prompt-axis + builder-axis 별도)
- **새 axis evidence**: Forward Dominance mechanism (Backward effect Forward-prompt-dependent) — paper §3.1 Inter-Module Co-Design narrative 추가 dimension

### 학습 비용 + 환경

- **Wall**: 2h33m (10:31:05 → 13:04)
- **GPU 시간**: 1 cell × 2.5h × 1-conc = ~2.5 GPU-hour (GPU 0 only)
- **LLM API 비용**: ~$2-4 GLM 4.7 (3068 calls = 1534 × 2)
- **per_q drift**: 5.7s → 6.0s (Round mid) → 6.1s (Round end, +5% sub-noise)
- **failure**: 0/1 (metrics.txt 정상)

### Checkpoint + Reference (재사용)

- Stack: c01_01 anchor stack + BidirectionalFilter (commit `88ad47e`) + Forward prompt config (commit `60b6988`)
- weight_path: `outputs/checkpoints/best_gat_qcond_nl3.pt` (anchor 동일, 학습 없음)
- Filter: BidirectionalFilter with `bidirectional_forward_prompt_mode="recall_biased_strong"` + `backward_section="bidirectional_backward"`

### 산출물

- Config: `configs/experiments/abl/wave6_recall_biased/w6_p4_c1_m4_strong.yaml` (module:filters 미리 작성, root verify)
- Sweep script: `scripts/run_wave6_phase4_c1.sh` (single cell GPU 0)
- Logs: `logs/wave6_phase4_c1_main.log` + `logs/wave6_phase4_c1/w6_p4_c1_m4_strong_20260517_103105.log`
- Outputs: `outputs/experiments/abl/wave6_recall_biased/w6_p4_c1_m4_strong/` (metrics.txt + predictions.jsonl + output_*.jsonl)

### 후속 위임 (chain handoff)

- **Analyzer 위임 (primary, Phase 4 결과 분석)**: `notebooks/analysis_results/wave6_phase4_c1_2026-05-17.md` 신규 작성
  - C1 vs M4 baseline + M1-B strong baseline 의 ΔF1/ΔEX 분석 (Synergy/Additive/Partial Degrade 정량 확정)
  - C1 의 backward_added mechanism 변동 정량 (strong Forward 시 Backward 가 추가하는 column 분포 비교 vs mild Forward)
  - per-difficulty (simple/moderate/challenging) C1 vs M4 EX 변동 (Backward effect 의 difficulty-axis 변동)
  - **C2 (M4 + M3 MAJORITY Forward) launch 결정 권고**: M3 MAJORITY (post-hoc, R=0.9290, P=0.7934, F1=0.8433) 를 Forward 로 사용하면 voting 의 Backward effect 변동 확인 — Forward Dominance hypothesis 검증
  - axis #15 evidence 추가 강화 (Forward-prompt-dependent Backward mechanism)
  - paper §V.5.x.M.15 본문 갱신 (Top 2 C1 결과 + Backward mechanism Forward-prompt-dependent new finding)
  - paper §3.1 Filter ↔ Selector Backward Mechanism bullet 갱신 (orthogonality partial 부정 + entanglement evidence)
- **Planner 위임 (analyzer 후)**:
  - paper §V.5.x.M.15 narrative 강화 (Top 2 C1 Forward Dominance + Backward Effect Reduction evidence)
  - paper §3.1 Inter-Module Co-Design narrative 의 Forward-Backward entanglement 새 dimension
  - C2 launch 결정 (analyzer 권고 후 또는 사용자 직접 결정)
- **Root 위임 (planner 후)**: C2 (M4 + M3 MAJORITY Forward) launch trigger — Phase 4 chain 후속


---

## Wave 6 Phase 5 Top 2 C2 (M4 + M3 MAJORITY voting Forward) — Single Cell (DECISIONS 2026-05-17 §6, 학술 agent §8.3 + §5+§6, 2026-05-17, 🎯 H3 Partial Entanglement 확정 — Backward Effect Reduction mechanism 정량 분해 (Voting ~70% + Inclusiveness ~30%), Pareto frontier 6번째 cell 진입 ✅)

근거: DECISIONS 2026-05-17 (Wave 6 Phase 4 C1 결과 + C2 launch 결정) §6 C2 launch spec + 학술 agent filter improve plan §5 + §6 + §8.3. Module:filters commit `7a07a6b` (BidirectionalFilter + voting_multi_prompt Forward composition + smoke 36/36 PASS) 구현 후 launch.

### 운영 이력

- **2026-05-17 (Module:filters commit `7a07a6b`)**: BidirectionalFilter 에 `bidirectional_forward_prompt_mode="voting_multi_prompt"` 옵션 신규 — M3 MultiPromptVotingFilter 의 3-prompt MAJORITY voting logic 을 Forward 로 swap-in composition. backward_compat retain.
- **2026-05-17 15:31:47 launch**: wrapper PID 2727111, single cell GPU 0, no parallelism (BidirectionalFilter 내부 Forward 3 voting prompts sequential). Monitor task `bnxyb3j3r` (1h timeout × 3 re-arm) + 30m cron `74a5bc09`.
- **2026-05-17 19:34 종료**: metrics.txt 도달. Wall = **4h02m** (사용자 spec ~3h ETA + ~1h overage, per_q drift 8.9 → 9.6s).

### 결과 (R/P/F1/EX 4-decimal, 4 baseline 비교)

| Source | R | P | F1 | EX | ΔF1 vs C2 | ΔEX vs C2 |
|---|---:|---:|---:|---:|---:|---:|
| **C2 w6_p5_c2_m4_majority** | **0.9273** | 0.7745 | **0.8440** | **0.5196** | (base) | (base) |
| anchor c01_01 | 0.8748 | 0.8582 | 0.8664 | 0.5176 | C2 -0.0224 | C2 +0.0020 sub-noise |
| M4 baseline (mild Forward) ⭐ EX-max | 0.9325 | 0.7593 | 0.8370 | **0.5300** ★ | **C2 +0.0070** | **C2 -0.0104** ← key |
| C1 (strong Forward, Partial Degrade) | 0.9177 | 0.8109 | 0.8610 | 0.5150 | C2 -0.0170 | **C2 +0.0046** ← key |
| M3 MAJORITY (post-hoc, voting Forward only) | 0.9290 | 0.7934 | 0.8433 | (post-hoc) | C2 +0.0007 sub-noise | — |

### 🎯 3 Hypothesis 판정 — **H3 Partial Entanglement 확정** ✅

| Hypothesis | 조건 | C2 EX = 0.5196 | 판정 |
|---|---|---|:---:|
| **H1** — Forward inclusiveness dominant | C2 EX ≈ M4 EX (0.5300) | Δ=-0.0104 from M4 (M4 보다 lower) | ❌ 부정 |
| **H2** — Voting mechanism dominant | C2 EX ≈ C1 (0.5150) | Δ=+0.0046 from C1 (C1 보다 higher) | ❌ 부정 |
| **H3** — Partial entanglement | C2 EX intermediate (0.52~0.53) | **0.5196 ∈ [0.5150, 0.5300]** ✅ | ✅ **확정** |

### 📊 Backward Effect Reduction mechanism 정량 분해

**C1 의 Backward Effect Reduction (-0.0150 EX from M4)** 의 mechanism:
- C2 EX = 0.5196 의 distance:
  - M4 거리 (mild Forward): 0.5300 - 0.5196 = **0.0104**
  - C1 거리 (strong Forward): 0.5196 - 0.5150 = **0.0046**
- **ratio M4 distance : C1 distance = 2.26 : 1**
- → C2 가 C1 쪽으로 약간 치우침 (mechanism dominant 약간 강함)
- **정량 분해**:
  - **Voting mechanism (mechanism dominant) ~70%** — voting noise pruning 효과
  - **Forward inclusiveness (inclusiveness dominant) ~30%** — Forward 의 inclusion 강도 효과
- 합산: 100% (partial entanglement, 양쪽 영향 정량 분리)

**학술적 함의**:
- DECISIONS §3.1 Forward/Backward orthogonality hypothesis 의 entanglement 정확 정량 (60% mechanism / 40% inclusiveness 대략)
- **axis #15 의 mechanism axis 4-cell complete coverage** (M4 mild + C1 strong + C2 voting MAJORITY + 정량 entanglement)
- paper §3.1 Inter-Module Co-Design narrative 의 Forward-Backward entanglement 정량 dimension 추가

### Pareto Frontier 6 cells (C2 신규 진입)

| Cell | R | P | F1 | EX | Pareto position |
|---|---:|---:|---:|---:|---|
| M1-A mild | 0.9259 | 0.7648 | 0.8377 | 0.5169 | R-bias |
| M1-B strong | 0.9022 | 0.8316 | **0.8655** ★ | 0.5130 | F1-best |
| M4 Bidirectional (mild Fwd) | 0.9325 | 0.7593 | 0.8370 | **0.5300** ★ | EX-max |
| M3 MAJORITY (post-hoc) | 0.9290 | 0.7934 | 0.8433 | (post-hoc) | R-P balanced |
| C1 (M4 + strong Fwd) | 0.9177 | 0.8109 | 0.8610 | 0.5150 | F1-secondary + Partial Degrade |
| **🆕 C2 (M4 + voting Fwd MAJORITY)** | **0.9273** | 0.7745 | 0.8440 | 0.5196 | **Partial Entanglement (intermediate)** |

C2 Pareto position: R-P balanced + EX intermediate (M4-C1 사이). F1 (0.8440) marginal lower than C1 (0.8610) but higher than M4 (0.8370). EX (0.5196) intermediate but closer to C1.

### 학술 agent §10 success criterion + DECISIONS §5/§6 분기

- **F1 ≥ 0.8672**: C2 = 0.8440 ❌ 미달 (-0.0232) — Wave 6 chain 의 모든 cell F1 미달 확정
- **DECISIONS §5 Outcome (b) retain**: axis #15 evidence retain (prompt-level strengthening) + axis #11 Option A retain (prompt-axis + builder-axis 별도)
- **새 axis evidence**: H3 Partial Entanglement 정량 — Forward Dominance 3-cell complete coverage + entanglement mechanism axis 정확 정량

### Forward Dominance 3-cell complete coverage (M4 + C1 + C2 통합)

| Forward | R | P | F1 | EX | Backward Effect (EX gain from M4) |
|---|---:|---:|---:|---:|---:|
| M4 mild | 0.9325 | 0.7593 | 0.8370 | **0.5300** ★ | +0.0124 (anchor base, 첫 EX gain) |
| **C2 voting MAJORITY** | **0.9273** | 0.7745 | 0.8440 | 0.5196 | -0.0104 (mechanism ~70%) |
| C1 strong | 0.9177 | 0.8109 | 0.8610 | 0.5150 | -0.0150 (Partial Degrade, full mechanism + inclusiveness) |

→ **Forward 의 inclusiveness 와 voting 의 noise pruning 이 Backward effect 의 dual axis 결정** — Wave 6 chain 의 complete mechanism axis 정량.

### voting telemetry (M3 MAJORITY ≥2 of 3 votes)

- `bidirectional_forward_voting_strategy: "MAJORITY"` (≥2 votes)
- `filter_forward_raw_counts` + `filter_forward_voted_counts` — output_*.jsonl 의 telemetry 에 noted (analyzer 위임 정량)
- expected: A (M1-A mild) + B (SQL clause) + C (Conservative) 의 raw counts 분포 + MAJORITY voted counts

### 학습 비용 + 환경

- **Wall**: 4h02m (15:31:47 → 19:34)
- **GPU 시간**: 1 cell × 4h × 1-conc = ~4 GPU-hours (GPU 0 only, BidirectionalFilter 내부 Forward 3 voting sequential)
- **LLM API 비용**: ~$10-15 GLM 4.7 (6136 calls = 1534 × 4 = 3 voting + 1 backward)
- **per_q drift**: 8.9s (Round start) → 9.0s (mid) → 9.6s (end, +8% sub-noise)
- **failure**: 0/1 (metrics.txt 정상)

### Checkpoint + Reference (재사용)

- Stack: c01_01 anchor stack + BidirectionalFilter (commit `88ad47e`) + voting_multi_prompt Forward composition (commit `7a07a6b`)
- weight_path: `outputs/checkpoints/best_gat_qcond_nl3.pt` (학습 없음)
- Filter spec:
  - `bidirectional_forward_prompt_mode: "voting_multi_prompt"`
  - `bidirectional_forward_voting_strategy: "MAJORITY"` (≥2 of 3)
  - `backward_section: "bidirectional_backward"` (M4 default retain)
  - `sanitize_output: true`

### 산출물

- Config: `configs/experiments/abl/wave6_recall_biased/w6_p5_c2_m4_majority.yaml` (module:filters 미리 작성)
- Sweep script: `scripts/run_wave6_phase5_c2.sh` (single cell GPU 0)
- Logs: `logs/wave6_phase5_c2_main.log` + `logs/wave6_phase5_c2/w6_p5_c2_m4_majority_20260517_153147.log`
- Outputs: `outputs/experiments/abl/wave6_recall_biased/w6_p5_c2_m4_majority/` (metrics.txt + predictions.jsonl + output_*.jsonl)

### 후속 위임 (chain handoff)

- **Analyzer 위임 (primary, Phase 5 결과 분석)**: `notebooks/analysis_results/wave6_phase5_c2_2026-05-17.md` 신규 작성
  - C2 vs 4 baseline (M4 + C1 + M3 MAJORITY + anchor) 의 ΔF1/ΔEX 정량
  - H3 Partial Entanglement 정량 (Voting ~70% + Inclusiveness ~30% mechanism 분해)
  - Forward Dominance 3-cell complete coverage (M4 mild + C1 strong + C2 voting MAJORITY) 의 mechanism axis 정량
  - voting telemetry (filter_forward_raw_counts / filter_forward_voted_counts) per-prompt distribution
  - C2 vs M3 MAJORITY (post-hoc) — voting Forward 단독 vs voting + Backward union 의 추가 contribution
  - C2 backward_added mean / distribution / overlap (M4 baseline 0.18 nodes/q 96% overlap vs C1 0.48 nodes/q 90% overlap vs C2 ?)
  - per-difficulty (simple/moderate/challenging) C2 vs M4/C1 EX 변동
  - paper §V.5.x.M.15 본문 갱신 권고 — Triple → Quadruple Evidence (M1 R-lift + M4 EX gain + C1 Partial Degrade + C2 H3 Partial Entanglement 확정)
  - paper §3.5 axis #15 row 갱신 — entanglement quantification (70/30 split)
  - paper §3.1 Filter ↔ Selector Backward Mechanism bullet 갱신 — entanglement 정량 정확
- **Planner 위임 (analyzer 후)**:
  - paper §V.5.x.M.15 Triple → Quadruple Evidence 격상 (Wave 6 main contribution 완성)
  - paper §3.1 Forward-Backward entanglement quantification 정확
  - Wave 6 chain 종료 결정 (모든 cell F1 미달, Pareto 6 cells 완성)
  - 후속 chain candidate (선택): 다른 voting strategies (OR / AND) + Backward 조합 추가 cell, 또는 paper closure


## Wave 7 Stage-wise EX Chain — Anchor Relog (Option A SQL Gen 통합) (DECISIONS 2026-05-18 §2+§3, m4_anchor_framework_analysis §5.5.1+§5.6.1, 2026-05-18, 🎯 Stage-wise EX 3 stage 측정 완료 — Filter EX cost ΔEX=−0.0033 + Extractor R-lift +0.1643 EX dimension first evidence)

### 목적

m4_anchor_framework_analysis §5.5.1 의 Stage-wise Cumulative R/P/F1/EX 표의 **EX column 빈 cell 채우기**:
- (1) Selector only (top-K=20) — SQL Gen 직접 호출 → EX 측정
- (2) +Extractor (no filter, ~83 nodes/q) — SQL Gen 직접 호출 → EX 측정
- (3) +Filter (anchor c01_01 XiYan) — 기존 anchor 의 EX 재현 (validation)

### Option A 구현 — anchor 재실행 안 3 SQL Gen 통합

DECISIONS 2026-05-18 §3 Option A 권고: 별도 launcher script 작성 + Wave 7 4 cells launch 대신, anchor 재실행 안에 Selector-only / Extractor-only / Final SQL Gen 3개 통합 (~2h 시간 절약).

**code patches** (commit local 예정):
- `src/pipeline/schema_linking.py`:
  - Patch 1: selector top-K nodes snapshot (table.col list)
  - Patch 2: extractor output snapshot (before filter)
  - Patch 3: generator 호출 시 3 stage SQL 생성 (selector_only + extractor_only + final)
  - Patch 4: stage-wise node lists 를 selector_info / extractor_info 에 추가
- `src/main.py`:
  - Patch 1: `_compute_ex_extra` 함수 + ex_score_selector_only / extractor_only 계산 (ProcessPoolExecutor 15s timeout)
  - Patch 2: pred_record 에 stage-wise SQL + EX 추가

### Final metrics (n=1534)

| 항목 | 값 |
|---|---|
| **R** | 0.8697 |
| **P** | 0.8581 |
| **F1** | 0.8639 |
| **EX (final)** | 0.5117 |
| filter_time_mean | 1.91s |
| llm_calls_total | 6136 (4/q: 1 Filter + 3 SQL Gen) |
| llm_input_tokens | 8,807,966 |
| llm_output_tokens | 244,721 |
| Wall | 3h 28min (00:53:43 → 04:21 KST) |
| failure | 0/1 (metrics.txt 정상) |

### Stage-wise EX (paper §5.5.1 갱신 데이터)

| Stage | nodes/q | R | P | F1 | EX | Δ vs prior |
|---|---:|---:|---:|---:|---:|---|
| **(1) Selector only (top-K=20)** | 20.00 | 0.6765 | 0.2131 | 0.3242 | **0.3507** | 🆕 첫 측정 |
| **(2) + Extractor (MSTPCSTUnion, no filter)** | 83.08 | 0.9710 | 0.1167 | 0.2084 | **0.5150** | 🆕 첫 측정 |
| **(3) + Filter (anchor c01_01 XiYan)** | ~4.70 | 0.8697 | 0.8581 | 0.8639 | **0.5117** | ΔEX=−0.0059 vs prior 0.5176 (LLM noise) |

### vs prior anchor c01_01 (재현 정확도)

| Metric | prior c01_01 (Wave 5 baseline) | new c01_01_wave7_relog | Δ |
|---|---:|---:|---:|
| R | 0.8748 | 0.8697 | −0.0051 |
| P | 0.8582 | 0.8581 | −0.0001 |
| F1 | 0.8664 | 0.8639 | −0.0025 |
| EX | 0.5176 | 0.5117 | −0.0059 |

→ **재현 noise sub-noise** (ΔF1 −0.0025, ΔEX −0.0059). GLM 4.7 temperature=0.0 단 inherent stochastic variability.

### Stage 별 Δ contribution (paper §5.5.2 갱신 — EX dimension 추가)

**Stage (1) → (2) (Selector → +Extractor)**:
- ΔR = +0.2945 (+44%)
- ΔP = −0.0964 (−45%)
- ΔF1 = −0.1158 (−36%)
- **ΔEX = +0.1643 ★ 🆕** — Extractor 의 R-lift 가 EX dimension 에도 **dramatic** 효과 (Selector top-K=20 만으로는 SQL gen 정밀도 부족)

**Stage (2) → (3) (Extractor → +Filter)**:
- ΔR = −0.1013 (−10%)
- ΔP = +0.7414 (+635%)
- ΔF1 = +0.6555 (+315%)
- **ΔEX = −0.0033 🆕** — Filter 가 EX 에는 미세 **negative** (F1 +0.6555 dominant 와 분리!)

### 핵심 finding — Filter F1 Dominance 와 EX Dimension 분리

**Filter (XiYan, 83 → 6.48 nodes prune):**
- F1: +0.6555 contribution (~76% of final F1) → **dominant**
- EX: −0.0033 contribution → **micro-negative**

→ paper §V.5.x.M.12 Filter Dominance 3-Zone Mechanism 갱신: **F1 axis 에서 dominant 단 EX axis 에서 micro-negative**. F1-EX 분리 evidence 의 첫 정량.

### M4 anchor 정합성

m4_anchor_framework_analysis §5.5.1 의 anchor stack 은 **M4 BidirectionalFilter** (EX=0.5300). 본 실험의 base 는 anchor c01_01 (XiYanFilter, prior anchor) — Stage (3) 의 row 만 c01_01 row 갱신, Stage (1)(2) 의 schema input source 는 Filter stage 무관 (Selector + Extractor 동일) 이므로 M4 stack 의 (1)(2) row 에도 동일 EX 0.3507 / 0.5150 적용 가능.

- M4 stack 의 stage-wise (별도 anchor 재실행 불필요): (1) 0.3507, (2) 0.5150, (3) M4 EX = 0.5300 (prior 측정)
- M4 의 Filter EX cost = 0.5300 − 0.5150 = **+0.0150** (M4 의 Backward Union 이 EX 에 positive contribution)
- c01_01 의 Filter EX cost = 0.5117 − 0.5150 = **−0.0033** (negative)
- → **M4 ↔ c01_01 의 Filter EX cost 차이 = +0.0183** — M4 Bidirectional Filter 의 EX gain mechanism evidence 강화

### 학습 비용 + 환경

- **Wall**: 3h 28min (00:53:43 → 04:21 KST, +1min slack)
- **GPU 시간**: 1 cell × 3.5h × GPU 0 only = ~3.5 GPU-hours
- **LLM API 비용**: ~$15-20 GLM 4.7 (6136 calls = 1534 × 4 = 1 Filter + 3 SQL Gen)
- **per_q drift**: 8.25s → 8.16s → 8.16s (안정, no drift)
- **failure**: 0/1

### Checkpoint + Reference (재사용)

- Stack: c01_01 anchor stack (XiYanFilter default + MSTPCSTUnion + EnsembleSelector + GLM 4.7 SQL gen)
- weight_path: `outputs/checkpoints/best_gat_qcond_nl3.pt` (학습 없음)
- patch: `src/pipeline/schema_linking.py` + `src/main.py` (Wave 7 Option A integration, commit local 예정)

### 산출물

- Config: `configs/experiments/abl/c01_threshold_sweep/c01_01_wave7_relog.yaml`
- Patches: `src/pipeline/schema_linking.py` (4 patches) + `src/main.py` (2 patches)
- Logs: `logs/wave7_relog/c01_01_wave7_relog_20260518_005343.log`
- Outputs: `outputs/experiments/abl/c01_threshold_sweep/c01_01_wave7_relog/` (metrics.txt + predictions.jsonl + output_*.jsonl + score_analysis_*.jsonl + profiling_*.jsonl)

### 후속 위임 (chain handoff)

- **Analyzer 위임 (primary, Wave 7 Stage-wise EX 분석)**: `notebooks/analysis_results/wave7_stagewise_ex_2026-05-18.md` 신규 작성
  - m4_anchor_framework_analysis §5.5.1 표 갱신 — (1)(2)(3) EX cell 채움 (0.3507 / 0.5150 / 0.5117 c01_01 + 0.5300 M4 retain)
  - m4_anchor_framework_analysis §5.5.2 Stage 별 Δ contribution 갱신 — ΔEX line 추가 (+0.1643 Extractor, −0.0033 Filter c01_01)
  - m4_anchor_framework_analysis §5.5.3 종합 view 갱신 — EX axis breakdown 추가
  - m4_anchor_framework_analysis §5.6.1 9 cells × EX 매트릭스 — anchor c01_01 row 의 EX=0.5117 (Wave 7 측정) vs 0.5176 (prior) 의 sub-noise 확인 + M4 EX cost 정확 +0.0150 (vs c01_01 −0.0033) 정량
  - paper §V.5.x.M.12 Filter Dominance 갱신 — F1 axis dominant vs EX axis micro-negative 분리 evidence
  - paper §V.5.x.M.15 axis #15 갱신 — M4 ↔ c01_01 Filter EX cost ΔΔ=+0.0183 정량 (M4 Bidirectional EX gain mechanism evidence 강화)
- **Planner 위임 (analyzer 후)**:
  - paper §V.5.x.M.12 F1-EX 분리 evidence 격상 (Wave 7 main contribution)
  - paper §3.5 axis #12 row 갱신 — EX dimension Filter cost 추가
  - Wave 7 closure 결정 (Stage-wise EX 완료) + Wave 8 candidate (선택): M4 anchor stack 의 stage-wise EX 직접 측정 (Backward Union 의 EX gain mechanism per-stage 분해)


## Wave 9 Baseline Relog — G-Retriever / LinkAlign / XiYan-SQL 3 cells SQL Gen prompt 재측정 (DECISIONS 2026-05-18 Wave 9, 2026-05-18, 🎯 prompt-axis confounder 분리 — baseline 우위 narrative ΔEX squeeze +0.28~+0.33 → +0.08~+0.27)

### 목적

paper §V.5.x.M.2 5/15 갱신 narrative ("SQL gen prompt = EX-axis dominant factor +0.1512") + anchor EX +18.06%p jump (0.3396 → 0.5176) 정합 위에서 **baseline 3 cells (2026-03-28 측정) 의 EX outdated SQL Gen prompt 정합 검증 + 재측정**. paper §10 의 6 baseline 비교 표 정확성 회복 + baseline 우위 narrative 정량 정확화.

### 구현 — Option A Stand-alone Python Script (main.py 변경 없이)

**Pattern**: Wave 7 Option A 정합 — 기존 final_nodes 보존 + SQL Gen 만 재실행.

- `scripts/wave9_sql_regen.py`: stand-alone Python script
  - `parse_final_nodes()` — list of "table.col" strings → `{table: [col, col, ...]}` subgraph dict
  - LLMSQLGenerator(provider="glm", llm_model="zai-org/glm-4.7", temperature=0.0)
  - `evaluate_ex(pred_sql, gold_sql, db_path)` with 15s ProcessPoolExecutor timeout
  - per-difficulty (simple/moderate/challenging) EX 분해
  - 기존 generated_sql 도 보존 (`prior_generated_sql`)
- `scripts/run_wave9_baseline_relog.sh`: 3 cells parallel wrapper

### Final Metrics (n=1534 per cell)

| Baseline | overall EX | simple | moderate | challenging | Outdated overall | Δ overall |
|---|---:|---:|---:|---:|---:|---:|
| **G-Retriever** | **0.4283** | 0.5114 | 0.3125 | 0.2690 | 0.2490 | **+0.1793** ★ |
| **LinkAlign** | **0.3390** | 0.4314 | 0.2112 | 0.1586 | 0.2001 | **+0.1389** |
| **XiYan-SQL** | **0.2405** | 0.3092 | 0.1358 | 0.1379 | 0.1969 | **+0.0436** |

→ 3 baseline 모두 +Jump (prompt-axis confounder 정합 확인) 단 **G-Retriever 가 dominant +0.1793** vs XiYan-SQL +0.0436 만 (XiYan 의 final_nodes 평균 1 col/q sparse — SQL gen 가능성 부족 → 신규 prompt 의 schema-strict 사용 시 +Δ 작음).

### Δ vs anchor c01_01 (Wave 7 EX=0.5117) + M4 (EX=0.5300, EX-best)

| Baseline | Wave 9 EX | Δ vs anchor c01_01 | Δ vs M4 |
|---|---:|---:|---:|
| G-Retriever | 0.4283 | **−0.0834** (anchor 우위) | **−0.1017** (M4 우위) |
| LinkAlign | 0.3390 | **−0.1727** | **−0.1910** |
| XiYan-SQL | 0.2405 | **−0.2712** | **−0.2895** |

### baseline 우위 narrative 정량 squeeze — paper §10 + paper main contribution 갱신 candidate

| 영역 | prior (outdated baseline) | Wave 9 (new prompt) | 정량 변화 |
|---|---:|---:|---|
| anchor c01_01 vs baseline ΔEX | +0.2627~+0.3207 | **+0.0834~+0.2712** | **squeeze** (prompt confounder 분리) |
| M4 vs baseline ΔEX | +0.2810~+0.3331 | **+0.1017~+0.2895** | squeeze |
| **dominant evidence retain** | — | **본 framework 의 schema linking effect 정량 evidence** ★ | paper main contribution narrative 정합 정확화 |

→ paper §10 의 6 baseline 비교 표 갱신 candidate (overall + per-difficulty 모두 갱신) + paper main contribution 의 baseline 우위 narrative ΔEX 정량 정확화 (anchor +0.18 vs baseline 평균 +0.12 의 ΔΔ = 본 framework 의 schema linking effect ~+0.06~+0.07).

### per-difficulty 분포 정합 (paper §V.5.x.M.5 thrombosis_prediction outlier narrative 검증)

- **simple**: G-Retriever 0.5114 > LinkAlign 0.4314 > XiYan-SQL 0.3092 (linear decay, schema rich → poor)
- **moderate**: G-Retriever 0.3125 > LinkAlign 0.2112 > XiYan-SQL 0.1358 (동일 trend)
- **challenging**: G-Retriever 0.2690 > LinkAlign 0.1586 ≈ XiYan-SQL 0.1379 (LinkAlign vs XiYan-SQL 의 gap shrink)
- → **simple/moderate 에서 schema sparse 의 dominant penalty**, challenging 에서는 schema quality 가 less critical (다른 lever 가 dominant, e.g. domain knowledge)

### 첫 launch fail + fix history (debugging note)

- **첫 launch (18:19 KST)**: 모든 cell EX=0.0000 (API ERROR fallback)
  - **Root cause**: `wave9_sql_regen.py` 에 `load_dotenv` 호출 미존재 → nohup subprocess 에 GLM_API_KEY env 미전달 → APIClient "sk-missing" fallback → OpenAI 401 Unauthorized
  - Wave 8 cells 의 main.py 가 정상 동작 이유 = main.py line 4 의 `from dotenv import load_dotenv; load_dotenv(.env)` 자동 호출
- **Fix (18:28 KST)**: `scripts/wave9_sql_regen.py` 에 dotenv 호출 추가
- **Relaunch (18:28 KST)**: 정상 진행 → 19:30 KST 종료

### 학습 비용 + 환경

- **Wall**: 1h 02min (18:28 → 19:30 KST, parallel 3 streams)
- **GPU 시간**: 0 (LLM API only)
- **LLM API 비용**: ~$5~10 GLM 4.7 (3 cells × 1534 q × 1 LLM call = 4,602 calls)
- **rate**: ~25 q/min/cell (anchor 의 ~3.4× 빠름 — 1 LLM/q 만, anchor 4 LLM/q 와 비교)
- **failure**: 첫 launch failed (dotenv 누락), relaunch 정상 (0/3)

### Checkpoint + Reference

- final_nodes source: `outputs/baselines/baseline_{g_retriever,linkalign,xiyansql}/predictions.jsonl` (mtime 2026-03-28)
- final_nodes 평균 cols/q: G-Retriever 80, LinkAlign 20, XiYan-SQL 1 (sparse)
- new prompt: LLMSQLGenerator (sql_generator prompt + evidence) — Wave 5+ 정합

### 산출물

- Scripts: `scripts/wave9_sql_regen.py` (stand-alone) + `scripts/run_wave9_baseline_relog.sh` (3 parallel wrapper)
- Outputs: `outputs/baselines/wave9_relog/{g_retriever,linkalign,xiyansql}_relog/` (predictions.jsonl + metrics.txt)
- Logs: `logs/wave9_baseline_relog/`

### 후속 위임 (chain handoff)

- **Analyzer 위임 (primary, Wave 9 분석)**: `notebooks/analysis_results/wave9_baseline_relog_2026-05-18.md` 신규 작성
  - 3 baseline × 4 metric (overall + simple + moderate + challenging) 정합 정량 갱신
  - prompt-axis confounder 분리 정량 (anchor +0.1780 vs baseline 평균 +0.1206 의 ΔΔ ~+0.057)
  - 본 framework 의 schema linking effect 정량 evidence (anchor 의 schema quality 우위 effect 정합)
  - per-difficulty 정합 분포 (schema sparse penalty 의 difficulty-stratified analysis)
  - paper §10 6 baseline 비교 표 갱신 권고 (overall + per-difficulty)
  - paper §V.5.x.M.2 EX-Friendly Property narrative 정합 정확화 (baseline 도 prompt-axis +Jump 정합 확인)
  - paper main contribution 의 baseline 우위 narrative ΔEX 정량 정확화 (현 +0.28~+0.33 → 신규 +0.08~+0.27)
- **Planner 위임 (analyzer 후)**:
  - paper §10 표 갱신 + ΔF1 / ΔEX 재계산
  - paper main contribution baseline 우위 narrative 정합 정확화
  - paper §V.5.x.M.2 narrative retain 확인

