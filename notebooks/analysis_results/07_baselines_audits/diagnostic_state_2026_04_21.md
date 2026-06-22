# 진단 상태 보고서 — 2026-04-21

**작성 세션**: analyzer (읽기 전용, 실험 실행/설계 제안 없음)
**범위**: 현재 `outputs/experiments/` 전체 실험에 대한 cross-matrix, per-stage failure attribution, filter cost/latency profile, GAT 학습 상태 요약
**관련 문서**:
- 개선 후보 목록 → planner로 에스컬레이션: [improvement_opportunities_2026_04_21.md](../10_misc_planning/improvement_opportunities_2026_04_21.md)
- GAT 병목 상세: [s06_bottleneck_comparison.md](../06_selector_encoder_bottleneck/s06_bottleneck_comparison.md), [s06_bottleneck_b5_enriched_extension.md](../06_selector_encoder_bottleneck/s06_bottleneck_b5_enriched_extension.md)
- FK-Steiner sweep: [fk_steiner_percentile_sweep.md](../03_extractor_pcst_steiner/fk_steiner_percentile_sweep.md)

---

## 0. TL;DR

1. **전체 실험 40+ 개 F1 스펙트럼**: no-filter 0.2280–0.4833, with-filter 0.6597–0.7863. 필터가 **+0.45~+0.63 precision / −0.09~−0.16 recall**의 정형화된 tradeoff.
2. **Best anchor 해부**: `abl_a01_06_ens_basic_xiyan` (R=0.8149 / P=0.7597 / **F1=0.7863**). Selector+PCST가 3.21% gold 드롭, XiYan이 추가로 PCST-통과 gold의 **15.81%** 드롭. **Filter가 recall 병목**.
3. **Filter Pareto**: 동일 anchor(supernode+fixed)에서 **XiYan 1.36s/q → F1 0.6939** vs **Reflection 1iter 7.30s/q → F1 0.7069 (+0.0130, 5.4× cost)**. Verifier / AdaptiveMultiAgent는 strictly dominated. Reflection 3iter는 1iter 대비 +0.0003 F1에 2.5× 시간 → 포화.
4. **GAT 포화**: B5 Val R@15 plateau (peak ep 62, 240 epoch 무의미). Enriched는 L0 spread 개선에도 L2 collapse 심화 → **Fusion 의존도 상승**. 학습 비용만 3×.
5. **미측정 조합 다수**: Ensemble×Reflection, Enriched×Reflection, FK-Steiner×Filter, B5 체크포인트×Full pipeline — 전부 빈 셀.

---

## 1. Cross-experiment Matrix (F1 기준)

### 1-1. No-filter 계층 (extractor 순수 성능)

| Extractor | Selector | Recall | Precision | F1 | 출처 |
|---|---|---:|---:|---:|---|
| Basic PCST (θ=0.1) | Ensemble | 0.9679 | 0.1293 | 0.2280 | s03_a01_01_ensemble_basic |
| Basic PCST (θ=0.1, anchor) | Ensemble | 0.9679 | 0.1276 | 0.2253 | s03_a09_03 |
| Raw PCST (no threshold) | Cosine | 0.9489 | 0.1570 | 0.2695 | s01_a01_02_raw_pcst_baseline |
| Basic PCST (θ=0.1) | Cosine | 0.7571 | 0.1411 | 0.2378 | s01_a01_01_basic_pcst |
| Adaptive PCST (P80) | Ensemble | 0.7210 | 0.3471 | 0.4687 | s03_a09_05 / s03_a02_01 |
| Adaptive PCST | Cosine | 0.6719 | 0.3745 | 0.4809 | s01_a02_01_adaptive_pcst |
| Product Cost | Ensemble | 0.7349 | 0.3453 | 0.4701 | s03_a03_01_product_cost |
| Component-Aware Product | Ensemble | 0.7633 | 0.3538 | 0.4833 | s03_a06_01 |
| Topology Cost | Ensemble | 0.7318 | 0.3412 | 0.4655 | s03_a09_01 |
| Component-Aware Topology | Ensemble | 0.7463 | 0.3313 | 0.4591 | s03_a09_02 |
| Component-Aware Product (anchor) | Ensemble | 0.7489 | 0.3333 | 0.4613 | s03_a09_04 |
| Steiner Backbone | Ensemble | 0.8208 | 0.2330 | 0.3628 | s03_a04_01_steiner |
| FK-Backbone Steiner θ=0.1 | Ensemble | 0.9481 | 0.1582 | 0.2712 | s03_a10_04 |
| FK-Backbone Steiner θ=0.2 | Ensemble | 0.9418 | 0.1644 | 0.2800 | s03_a10_05 |
| FK-Backbone Steiner θ=0.3 | Ensemble | 0.9293 | 0.1812 | 0.3033 | s03_a10_02 |
| FK-Backbone Steiner θ=0.4 | Ensemble | 0.9014 | 0.2125 | 0.3439 | s03_a10_06 |
| FK-Backbone Steiner θ=0.5 | Ensemble | 0.8565 | 0.2627 | 0.4019 | s03_a10_03 |
| FK-Backbone Steiner θ=0.6 | Ensemble | 0.7789 | 0.3341 | 0.4673 | s03_a10_07 |
| FK-Backbone Steiner θ=0.7 | Ensemble | 0.6662 | 0.4245 | 0.5186 | s03_a10_08 |
| **FK-Backbone Steiner θ=0.8** | Ensemble | 0.5455 | 0.5044 | **0.5241** | s03_a10_09 (peak) |
| FK-Backbone Steiner θ=0.9 | Ensemble | 0.4083 | 0.5300 | 0.4613 | s03_a10_10 |
| FK-Backbone Steiner θ=1.0 | Ensemble | 0.2972 | 0.4920 | 0.3707 | s03_a10_11 |

**관찰**: no-filter 최고 F1은 **FK-Steiner θ=0.8의 0.5241** — 필터 없이도 R=0.5455 / P=0.5044 balanced. Cost 계열(Product/Component/Topology)은 F1 0.46~0.48로 포화.

### 1-2. Filter 적용 계층

| Extractor | Selector | Filter | Recall | Precision | F1 | 출처 |
|---|---|---|---:|---:|---:|---|
| Basic PCST | Ensemble | XiYan | **0.8149** | 0.7597 | **0.7863** | abl_a01_06 (⭐) |
| Basic PCST | Cosine | XiYan | 0.7987 | 0.7694 | 0.7838 | abl_a01_05 |
| Adaptive PCST | Cosine | XiYan | 0.5835 | 0.7829 | 0.6687 | abl_a01_07 |
| Adaptive PCST | Ensemble | XiYan | 0.6244 | 0.7930 | 0.6988 | s03_a02_03 |
| Adaptive PCST | Ensemble | Single-LLM | 0.5720 | 0.7795 | 0.6599 | s03_a02_02 |
| Product Cost | Ensemble | XiYan | 0.6141 | 0.7963 | 0.6935 | s03_a03_02 |
| Component-Aware Product | Ensemble | XiYan | 0.6304 | 0.8028 | 0.7062 | s03_a06_02 |
| Steiner Backbone | Ensemble | XiYan | 0.6806 | 0.7917 | 0.7320 | s03_a04_02 |
| Enriched GAT (Basic) | Ensemble | XiYan | 0.6658 | **0.8147** | 0.7326 | s03_a07_01 |
| Edge Prize (Basic) | Ensemble | XiYan | 0.6823 | 0.8139 | 0.7423 | s03_a07_02 |
| Basic + BO fixed cost | Ensemble | XiYan | 0.4793 | 0.7468 | 0.5839 | s03_a08_01 |
| Basic + BO score-driven | Ensemble | XiYan | 0.5910 | 0.7867 | 0.6751 | s03_a08_02 |
| SuperNode Direct + Fixed | SuperNode | XiYan | 0.6761 | 0.7128 | 0.6939 | abl_a03_17 |
| SuperNode Direct + Fixed | SuperNode | **Reflection 1iter** | 0.7320 | 0.6833 | **0.7069** | a05_02 |
| SuperNode Direct + Fixed | SuperNode | Reflection 3iter | 0.7405 | 0.6765 | 0.7071 | a05_03 |
| SuperNode Direct + Fixed | SuperNode | Verifier | 0.7093 | 0.6676 | 0.6878 | a05_04 |
| SuperNode Direct + Fixed | SuperNode | AdaptiveMultiAgent (Qwen) | 0.3770 | 0.6276 | 0.4711 | a05_01 |
| SuperNode Direct + Fixed | SuperNode | XiYan (gpt-4o-mini) | 0.6037 | 0.7317 | 0.6617 | a05_13 |
| SuperNode Direct + Fixed | SuperNode | AdaptiveMultiAgent (gpt-4o-mini) | 0.3992 | 0.7576 | 0.5229 | a05_14 |
| SuperNode Direct + Fixed | SuperNode | Reflection 1iter (gpt-4o-mini) | 0.6827 | 0.6620 | 0.6722 | a05_15 |
| SuperNode Direct + Fixed | SuperNode | Verifier (gpt-4o-mini) | 0.7055 | 0.6385 | 0.6703 | a05_17 |

### 1-3. Direct GAT / QCond 계열 (s04, s05, a03, a04)

| Config | Selector | Extractor | Filter | R | P | F1 |
|---|---|---|---|---:|---:|---:|
| s04_01_qcond_a085_xiyan | QCond | Basic | XiYan | 0.6236 | 0.8056 | 0.7030 |
| s04_03_supernode_a085_xiyan | SuperNode | Basic | XiYan | 0.6154 | 0.8005 | 0.6958 |
| s04_02_supernode_a070_xiyan | SuperNode (α=0.70) | Basic | XiYan | 0.6089 | 0.7922 | 0.6885 |
| s04_04_qcond_a0_xiyan | QCond (α=0) | Basic | XiYan | 0.5015 | 0.7065 | 0.5866 |
| s04_05_supernode_a0_xiyan | SuperNode (α=0) | Basic | XiYan | 0.5237 | 0.7155 | 0.6048 |
| s05_a01_01_qcond_direct_xiyan | QCond Direct | — | XiYan | 0.4384 | 0.6578 | 0.5261 |
| s05_a01_02_supernode_direct_xiyan | SuperNode Direct | — | XiYan | 0.4369 | 0.6553 | 0.5243 |
| abl_a03_14_qcond_binary_fixed_xiyan | QCond (binary) | Fixed | XiYan | 0.5843 | 0.6929 | 0.6340 |
| abl_a03_15_qcond_binary_steiner_xiyan | QCond (binary) | Steiner | XiYan | 0.5247 | 0.6824 | 0.5932 |
| abl_a03_18_supernode_binary_steiner_xiyan | SuperNode (binary) | Steiner | XiYan | 0.5855 | 0.6871 | 0.6322 |
| abl_a04_01_supernode_t005_steiner_xiyan | SuperNode (θ=0.05) | Steiner | XiYan | 0.6353 | 0.7054 | 0.6686 |
| abl_a04_02_supernode_t010_steiner_xiyan | SuperNode (θ=0.10) | Steiner | XiYan | 0.6272 | 0.7011 | 0.6620 |
| abl_a04_03_supernode_t015_steiner_xiyan | SuperNode (θ=0.15) | Steiner | XiYan | 0.6196 | 0.6988 | 0.6568 |
| abl_a04_04_supernode_t020_steiner_xiyan | SuperNode (θ=0.20) | Steiner | XiYan | 0.6122 | 0.6936 | 0.6504 |

### 1-4. 빈 셀 (미실행 조합)

| Extractor | Selector | Filter | 상태 |
|---|---|---|---|
| Basic PCST | Ensemble | **Reflection** | ❌ 미실행 (Tier-1 후보) |
| Basic PCST | Ensemble | **Reflection × XiYan stacked** | ❌ 미실행 |
| Enriched GAT / EdgePrize | Ensemble | **Reflection** | ❌ 미실행 |
| FK-Backbone Steiner θ=0.5 | Ensemble | **XiYan / Reflection** | ❌ 미실행 (peak R=0.8565 anchor) |
| SymbolicVerifier | SuperNode | (all 4 variants) | ⚠️ config만 있음 (a05_19~22) |
| Adaptive PCST | Ensemble | Reflection | ❌ 미실행 |
| Component-Aware | Ensemble | Reflection | ❌ 미실행 |
| B5 checkpoint | — | — (Full pipeline) | ❌ 미실행 (R@15만 측정) |

---

## 2. Per-stage Failure Attribution

**방법**: 동일 Selector+Extractor에 대해 filter on/off 페어를 비교하여 각 stage의 gold 유지율을 분해.

### 2-1. Ensemble + Basic PCST 체인 (best F1 anchor)

| Stage | Gold 유지율 | 누적 Recall | 단계별 drop |
|---|---:|---:|---:|
| 입력 | 100.00% | 1.0000 | — |
| Selector + PCST (θ=0.1) | **96.79%** | 0.9679 | −3.21% |
| + XiYan filter | **84.19%** | **0.8149** | −15.81% (PCST 출력 대비) |

**해석**:
- Selector+PCST는 거의 모든 gold를 보존. 이 단계는 recall 병목 아님.
- **XiYan 필터가 PCST-통과 gold의 15.81%를 drop** — 필터가 유일한 recall 병목.
- Precision: no-filter 0.1293 → filter 0.7597 (+0.6304). 필터가 noise 87% 제거.

### 2-2. SuperNode Direct + Fixed PCST 체인 (a05 시리즈 anchor)

| Stage | Gold 유지율 | 누적 Recall | 단계별 drop | 출처 |
|---|---:|---:|---:|---|
| 입력 | 100.00% | 1.0000 | — | — |
| Selector (supernode only) | **99.68%** | 0.9968 | −0.32% | a03_03 |
| + PCST (θ=0.0) | **80.07%** | 0.7982 | −19.92% | a03_16 |
| + XiYan | 84.70% of PCST | **0.6761** | −15.30% | a03_17 |
| + Reflection 1iter | 91.71% of PCST | **0.7320** | **−8.29%** | a05_02 |
| + Reflection 3iter | 92.77% of PCST | 0.7405 | −7.23% | a05_03 |
| + Verifier | 88.86% of PCST | 0.7093 | −11.14% | a05_04 |
| + AdaptiveMultiAgent (Qwen) | 47.23% of PCST | 0.3770 | −52.77% | a05_01 |

**해석**:
- **SuperNode에서는 PCST가 가장 큰 recall-loser** (19.92%). Ensemble에서는 1.9%만 손실되므로, **PCST는 selector score calibration에 민감**.
- XiYan vs Reflection: 같은 PCST-통과 gold 중 Reflection이 **7.01pp 더 보존** (91.71% vs 84.70%).
- Reflection 3iter는 1iter 대비 +1.06pp만 recall 추가 — **iteration 증가 효과 제한적**.
- AdaptiveMultiAgent는 52.77% 드롭 — **recall 측면에서 사용 불가**.

### 2-3. Ensemble-anchor에 Reflection 기대 효과 (외삽)

Reflection on supernode: PCST-통과 gold의 91.71% 보존.
만약 Ensemble-basic anchor에서도 동일 비율이면:
- Projected Reflection recall = 0.9679 × 0.9171 = **0.8877**
- Projected F1 (precision을 Reflection/XiYan P 비율로 가정): ~**0.79~0.80**

**단순 외삽, 실측 아님**. 그러나 "Reflection이 recall 보존 우위, XiYan이 precision 우위"라는 비대칭은 확실. Stacking 가치 있음.

---

## 3. Filter Cost / Latency Pareto

### 3-1. 동일 anchor (SuperNode + Fixed PCST) Pareto

| Filter | Model | Recall | Precision | F1 | time/q (mean) | time/q (p95) | F1/s | 비고 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| XiYan | Qwen3-Coder-30B | 0.6761 | 0.7128 | 0.6939 | 1.36s | 2.66s | **0.510** | Pareto 우위 (속도) |
| Reflection 1iter | Qwen3-Coder-30B | 0.7320 | 0.6833 | **0.7069** | 7.30s | 11.58s | 0.097 | Pareto 우위 (F1) |
| Verifier | Qwen3-Coder-30B | 0.7093 | 0.6676 | 0.6878 | 16.00s | 20.07s | 0.043 | Dominated |
| Reflection 3iter | Qwen3-Coder-30B | 0.7405 | 0.6765 | 0.7071 | 18.34s | 29.10s | 0.039 | 1iter 대비 효과 포화 |
| AdaptiveMultiAgent | Qwen3-Coder-30B | 0.3770 | 0.6276 | 0.4711 | — | — | 낮음 | Strictly dominated |

**Pareto frontier = {XiYan Qwen, Reflection 1iter Qwen}**.
- **XiYan이 F1/s에서 5.3× 효율**. Reflection 1iter는 +0.0130 F1에 5.4× 시간.
- Reflection 3iter / Verifier / AdaptiveMulti는 **전부 dominated**.

### 3-2. Ensemble anchor (best F1) XiYan 프로파일

| Filter | Model | time/q (mean) | time/q (p95) | total |
|---|---|---:|---:|---:|
| XiYan | Qwen3-Coder-30B | **1.25s** | 2.23s | 1917s |
| (paired no-filter s03_a01_01) | — | ~0s | — | — |

**관찰**: Ensemble anchor에서도 XiYan 1.25s/query — supernode anchor의 1.36s와 거의 동일. **XiYan 속도는 anchor에 의존하지 않음** (호출당 비용 상수).

### 3-3. gpt-4o-mini 시리즈 (비용 기록 있음)

| Filter | Model | Recall | Precision | F1 | time/q | llm_calls/q | input_tokens total | output_tokens total | cached_tokens total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XiYan | gpt-4o-mini | 0.6037 | 0.7317 | 0.6617 | 1.10s | 1.00 | 2.70M | 30K | 139K (5.15%) |
| Reflection 1iter | gpt-4o-mini | 0.6827 | 0.6620 | 0.6722 | 4.63s | 2.98 | 8.13M | 193K | 144K (1.78%) |
| Verifier | gpt-4o-mini | 0.7055 | 0.6385 | 0.6703 | 7.52s | 2.92 | 8.06M | 496K | 87K (1.08%) |
| AdaptiveMultiAgent | gpt-4o-mini | 0.3992 | 0.7576 | 0.5229 | 6.38s | 2.30 | 1.43M | 503K | 0 |

**관찰**:
- gpt-4o-mini Reflection/Verifier은 Qwen 대비 F1 −0.03~−0.04 손실. Qwen3-Coder-30B가 스키마 이해도 명백히 우위.
- Cache hit rate 1~5%만 — **cache 최적화 여지 있음** (prompt를 DB-level prefix로 재구성하면 상승 가능).
- AdaptiveMulti는 tokens_out 503K(!!) — **LLM이 긴 추론 생성**, Qwen 버전과 동일한 failure mode (recall 40% 아래).

### 3-4. Filter 3iter 수익 체감

- Reflection 1iter: F1 0.7069, 7.30s
- Reflection 3iter: F1 0.7071, 18.34s
- **+0.0003 F1 / +11.04s** — iteration 2~3의 자기 교정은 효과 없음. 1iter로 충분.

---

## 4. GAT 학습 상태 요약

### 4-1. 구조적 포화 징후

| 모델 | L | Val R@15 | Peak epoch | Plateau 이후 epoch | Total time |
|---|---|---:|---:|---:|---:|
| B0 (baseline) | 3 | 0.5713 | 298 | — | — |
| B1 (+ PairNorm) | 3 | 0.5938 | — | — | — |
| B2 (+ IR α=0.2) | 3 | 0.5956 | — | — | — |
| B3 (+ ListNet) | 3 | 0.5993 | — | — | — |
| B4 (+ AC) | 3 | 0.5993 | — | — | — |
| **B5 (2L Dual-Stream)** | 2 | **0.6073** | 62 | 240 epoch 무효 | ~29h |
| B5E (Enriched + B5) | 2 | 0.6016 | 60 | 240 epoch 무효 | 9h 14m (3.1× 빠름) |

**관찰**:
- **B5 peak at epoch 62**, 이후 240 epoch plateau. **학습 예산 80~100 epoch으로 충분**.
- B5E는 Enriched builder로 batch=8 가능 → 3.1× 학습 속도. 그러나 Val R@15은 −0.0057.
- AC loss 값 0.003~0.004 범위 (전체 loss 1.16의 0.3%) — **Anti-Collapse가 실질적으로 자주 트리거되지 않음**. L2 collapse는 Fusion이 보상.

### 4-2. Residual stream 병목 (s06 분석 재인용)

| 지표 | B0 | B4 | B5 | B5E |
|---|---:|---:|---:|---:|
| L0 (PLM) cosine | 0.657 | 0.657 | 0.657 | **0.636** (Enriched 분산 효과) |
| L1 GAT | 0.851 | 0.386 | 0.373 | 0.430 |
| L2 GAT | 0.891 | 0.947 | 0.920 | **0.978** (collapse 심화) |
| L_out (after Fusion) | 0.833 | 0.562 | 0.357 | **0.329** |
| ΔL0→L_out | +0.176 | −0.095 | −0.300 | −0.307 |

**해석**:
- B0→B5에서 **L_out cosine 0.833→0.357**. Dual-Stream Fusion이 residual stream의 collapse를 L_out에서 −0.30씩 교정.
- B5E에서 L2 collapse는 **더 심각**하지만 Fusion이 더 큰 폭으로 교정. **Fusion head가 압도적 compensator**.
- Fusion gradient (B5→B5E): 0.59 → 1.83 (3.1×). Fusion이 실제로 더 많은 신호를 처리.

### 4-3. Attention entropy

| Edge | L1 | L2 | 해석 |
|---|---:|---:|---|
| `column→belongs_to→table` | 1.945 | 1.860 | max entropy — 분산, table-centric pooling |
| `fk_node→points_to→column` | 0.758 | 0.756 | sharp, 안정 |
| `table→table_to_table→table` | 0.617 | 0.613 | sharp, JOIN 구조 인식 |

**관찰**: belongs_to edge는 **flat attention** — GAT가 table 레벨에서 column을 평균화. column-level discriminability는 downstream Fusion에 맡기는 구조.

---

## 5. 종합 진단

### 5-1. 어디가 확실히 병목인가

1. **Filter Recall**: 현 best anchor에서 **PCST-통과 gold의 15.81% 손실**. XiYan이 과도하게 conservative. **단일 최대 improvement surface**.
2. **Selector encoder OOD**: LDBO에서 확인된 train↔dev gap. Head retrain만으로 oracle Dev AUC +0.048. **encoder-level 개입 필요** (DANN 류).
3. **GAT Val R@15 plateau 0.6073**: 구조적 포화. 추가 depth / features 모두 미미.

### 5-2. 어디가 병목 아닌가

1. **Selector top-k**: Ensemble+Basic에서 Selector+PCST가 96.79% gold 보존 — 거의 perfect.
2. **PCST extractor (Ensemble)**: 3.21% 손실 — 미세 튜닝으로 회수 어렵고 비용 대비 효과 낮음.
3. **Filter iteration 수**: 1iter로 충분. 3iter는 비용 낭비.
4. **Cost tuning (PCST edge cost)**: BO로 F1=0.6751 plateau 확인. 추가 튜닝 무의미.
5. **gpt-4o-mini filter**: Qwen3-Coder에 strictly dominated.

### 5-3. 추가 데이터가 필요한 지점

1. **Ensemble-anchor × Reflection**: 페어 데이터 없음. 외삽 F1 0.79~0.80 가능성은 측정해야 확정.
2. **Enriched/EdgePrize × Reflection**: Precision 0.8147 anchor에 Reflection 부착 시 Pareto 도약 가능성.
3. **FK-Steiner θ=0.5 (R=0.8565) × Filter**: 단순 cost로 filter 비용 상승 여부 프로파일링 필요 (후보 노드 수 많음).
4. **B5 checkpoint × Full pipeline**: Val R@15 +0.0335 이득이 E2E F1로 전환되는지 미측정.
5. **SymbolicVerifier a05_19~22**: config 4개 존재, 실행 필요. Symbolic 제약의 비용/효과 baseline 누락.

### 5-4. Filter 수익 구조 요약

| 지표 | 공통 패턴 |
|---|---|
| Recall drop (XiYan) | 평균 −12~−16pp (no-filter 대비) |
| Precision gain (XiYan) | 평균 +45~+63pp |
| F1 gain (XiYan) | 평균 +0.28~+0.56 (noisy no-filter일수록 큼) |
| Reflection vs XiYan (same anchor) | +0.0559 Recall, −0.0295 Precision, **+0.0130 F1**, 5.4× time |
| 3iter vs 1iter | +0.0003 F1, 2.5× time |
| Qwen3 vs gpt-4o-mini | +0.03~+0.04 F1, 속도 비슷 |

---

## 6. 제약 사항 & 남은 의문

- `filter_info` 표준화 이전 실험(abl_a01_06, s03_a0x 대부분)은 **route distribution / repair count / stage_infos가 비어 있음**. `_aggregate_stage_telemetry()`를 재실행해 backfill 가능하지만, 이는 실험 주체(root 세션) 결정.
- SuperNode Direct 계열 중 **a05_02의 "basic_no_filter" 페어가 엄밀하게 일치하지 않음** (a03_16은 "fixed", a05_02는 "PCST θ=0.0"). 동일 변수는 아니므로 per-stage 분해는 근사값.
- **SQL execution accuracy (EX)**: s02_02(0.1499)만 측정됨, 다른 실험 전부 0. 본 진단은 schema linking F1만 다룸.
- GAT `train_step.jsonl` (CLAUDE.md에 언급된 per-layer diagnostic) 파일은 현재 `logs/train/`에 없음 — .log 형태 텍스트만 존재. **detailed layer health diagnostic은 s06 분석 리포트가 유일한 출처**.

---

## 7. 파일 위치 및 재현

```
# Cross-matrix 원본
outputs/summary_all.csv                                               # 20 rows (주로 a09/a10 + a05 gpt-4o-mini)
outputs/experiments/abl/a05_filter_agentic/summary_all.csv            # 6 rows (a05 Qwen 시리즈)
outputs/experiments/*/*/metrics.txt                                    # 40+ 파일

# Per-stage attribution 원본
outputs/experiments/s03_gat_ensemble/a01_basic_pcst/s03_a01_01_ensemble_basic/  # 9679 R no-filter
outputs/experiments/abl/a01_2x2x2_selector_extractor_filter/abl_a01_06_ens_basic_xiyan/  # best anchor
outputs/experiments/abl/a03_direct_per_step/abl_a03_03_supernode_selector_only/   # selector-only
outputs/experiments/abl/a03_direct_per_step/abl_a03_16_supernode_binary_fixed/    # + PCST
outputs/experiments/abl/a03_direct_per_step/abl_a03_17_supernode_binary_fixed_xiyan/  # + XiYan
outputs/experiments/abl/a05_filter_agentic/a05_02_reflection_1iter/               # + Reflection

# Filter profiling
outputs/experiments/abl/a05_filter_agentic/a05_0[1-4]/metrics.txt     # Qwen 시리즈
outputs/experiments/abl/a05_filter_agentic/a05_1[3-7]/metrics.txt     # gpt-4o-mini 시리즈
outputs/experiments/*/profiling_*.jsonl                                # per-query stage timings

# GAT diagnostic
logs/train/s06_a01_0[1-7]_*.log                                        # 텍스트 로그 (epoch별 loss + Val R@15)
notebooks/analysis_results/s06_bottleneck_*.md                         # residual stream 3-step 분석
```

**재현 커맨드** (각 실험 재실행 아님, 표 재계산만):
- 본 리포트의 F1 = 2 × R × P / (R + P) 로 metrics.txt의 R, P에서 직접 계산
- Per-stage attribution = 페어된 metrics.txt의 recall 차이
- Filter profiling = metrics.txt의 filter_time_* 필드 또는 profiling_*.jsonl의 `filtering` 키 mean/p95

---

## 8. Escalation (읽기 전용 세션이 할 수 없는 것)

- 본 리포트는 **새 실험 제안 / config 생성 / EXPERIMENT_HISTORY·CATALOG·ID_MIGRATION 갱신** 없음.
- 개선 가설(예: Reflection × Enriched, FK-Steiner × Filter)은 [improvement_opportunities_2026_04_21.md](../10_misc_planning/improvement_opportunities_2026_04_21.md) 참조 — **planner / root 세션 책임**.
- 진단 결과로 집계 버그/정정이 필요한 수치가 발견되면 여기에 기록 후 루트로 에스컬레이션.

현재 정정 요청 없음 — 수치 일관성 OK (루트 CLAUDE.md의 "#8 F1=0.4936" vs "#6 F1=0.7863" 이슈는 2026-04-21 루트 세션에서 해결됨).
