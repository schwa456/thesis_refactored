# Experiment ID Migration (2026-04-14)

모델 구조(주로 Selector 아키텍처) 기준으로 실험을 재분류하고 ID를 재부여.
폴더 내 숫자는 실행 순서(chronological).

## 폴더 구조

```
configs/experiments/  (outputs/experiments/, logs/experiments/ 동일 구조)
├── b0_baselines/                우리 baseline → 외부 baseline
├── s01_vector_only/             VectorOnly (cosine)
│   ├── a01_basic_pcst/
│   ├── a02_adaptive_pcst/
│   └── a03_pcst_variants/       dynamic, uncertainty
├── s02_gat_classifier/          GATClassifier (early v1)
├── s03_gat_ensemble/            Ensemble (Projector GAT + cosine, α)
│   ├── a01_basic_pcst/
│   ├── a02_adaptive_pcst/
│   ├── a03_product_cost/
│   ├── a04_steiner_backbone/
│   ├── a05_component_aware/
│   ├── a06_component_product/
│   ├── a07_enriched_triplet/
│   ├── a08_bayesian_opt/
│   ├── a09_topology_cost/       Topology-derived edge cost (edge-type param-free)
│   └── a10_fk_steiner/          FK-Backbone Steiner Closure (θ_r sweep, Recall ≥ 0.85 target)
├── s04_gat_qcond_projector/     Query-Conditioned Projector GAT
├── s05_gat_direct/              DirectGATSelector (BCE-only)
│   └── a01_full_pipeline/
├── s06_gat_bottleneck_fix/      GAT over-smoothing 해소 처방 (PairNorm / GCNII / ListNet / Anti-Collapse / Dual-Stream)
│   └── a01_additive_ablation/
└── abl/                         Ablation studies
    ├── a01_2x2x2_selector_extractor_filter/
    ├── a02_alpha_sweep/
    ├── a03_direct_per_step/
    ├── a04_direct_binary_steiner_sweep/
    ├── a05_filter_agentic/       Filter module agentic refinement (F1-F5)
    ├── a06_ext_fkprior/          [2026-04-20] Extractor E-III: Hybrid FK-Prior PCST
    ├── a07_ext_path_ensemble/    [2026-04-20] Extractor E-II: Pathfinding + PCST Ensemble
    └── a08_ext_louvain/          [2026-04-20] Extractor E-I: Louvain Community PCST
```

## ID 매핑

### b0_baselines/

| 신규 ID | 기존 | 구분 |
|---------|------|------|
| `b0_01_vector_only` | B4 (history) | 우리 |
| `b0_02_graph_expansion` | B5 | 우리 |
| `b0_03_graph_agent` | B6 | 우리 |
| `b0_04_g_retriever` | B1 | 외부 |
| `b0_05_linkalign` | B2 | 외부 |
| `b0_06_xiyan_sql` | B3 | 외부 |

### s01_vector_only/

| 신규 ID | 기존 config/output 이름 |
|---------|------------------------|
| `s01_a01_01_basic_pcst` | `experiment_base_pcst` (A5) |
| `s01_a01_02_raw_pcst_baseline` | `experiment_b0_raw_pcst_baseline` (B0) |
| `s01_a02_01_adaptive_pcst` | `experiment_b1_adaptive_pcst` (B1) |
| `s01_a03_01_dynamic_pcst` | `experiment_dynamic_pcst` (A6) |
| `s01_a03_02_uncertainty_pcst` | `experiment_uncertainty_pcst` (A7) |
| `s01_a03_03_dynamic_uncertainty_pcst` | `experiment_dynamic_uncertainty_pcst` (A8) |

### s02_gat_classifier/

| 신규 ID | 기존 |
|---------|------|
| `s02_01_gat_classifier` | `experiment_gat_classifier` (A1) |
| `s02_02_gat_classifier_multi_agent` | `experiment_gat_classifier_multi_agent` (A2) |
| `s02_03_gat_pcst_multi_agent` | (output only, A3) |
| `s02_04_gat_multi_agent` | `experiment_gat_multi_agent` (A4) |

### s03_gat_ensemble/

| 신규 ID | 기존 |
|---------|------|
| `s03_a01_01_ensemble_basic` | `experiment_b2_ensemble` (B2) |
| `s03_a02_01_combined` | `experiment_b_combined` |
| `s03_a02_02_single_filter` | `experiment_b4_single_filter` (B4a) |
| `s03_a02_03_xiyan_filter` | `experiment_b4_xiyan_filter` (B4b) |
| `s03_a03_01_product_cost` | `experiment_idea2_product_cost` (I2a) |
| `s03_a03_02_product_cost_xiyan` | `experiment_idea2_product_cost_xiyan` (I2b) |
| `s03_a04_01_steiner` | `experiment_idea3_steiner_backbone` (I3a) |
| `s03_a04_02_steiner_xiyan` | `experiment_idea3_steiner_backbone_xiyan` (I3b) |
| `s03_a05_01_component_aware` | `experiment_idea4_component_aware` (I4) |
| `s03_a06_01_product_component` | `experiment_idea24_product_component` (I24a) |
| `s03_a06_02_product_component_xiyan` | `experiment_idea24_product_component_xiyan` (I24b) |
| `s03_a06_03_idea124_combined` | `experiment_idea124_combined` |
| `s03_a06_04_idea124_combined_xiyan` | `experiment_idea124_combined_xiyan` |
| `s03_a07_01_enriched_gat` | `experiment_enriched_gat` (E1) |
| `s03_a07_02_edge_prize` | `experiment_edge_prize` (E2) |
| `s03_a08_01_bo_fixed_cost` | `experiment_bo_fixed_cost` (BO1) |
| `s03_a08_02_bo_score_driven` | `experiment_bo_score_driven` (BO2) |
| `s03_a09_01_topology_no_filter` | (신규, 2026-04-16) TopologyCost 방향 2 prototype |
| `s03_a09_02_ca_topology_no_filter` | (신규) CA + TopologyCost (방향 2+4) |
| `s03_a09_03_basic_no_filter_anchor` | (신규) Basic PCST anchor for a09 비교 |
| `s03_a09_04_ca_product_no_filter_anchor` | (신규) CA-ProductCost anchor (I24a 계열, no filter) |
| `s03_a09_05_adaptive_no_filter_anchor` | (신규) Adaptive PCST anchor = TopologyCost 직계 부모 |
| `s03_a10_01_fk_steiner_full_col` | (신규, 2026-04-16) FK-Backbone Steiner, θ_r=0.0 — R=0.9492, P=0.1567, F1=0.2690 |
| `s03_a10_04_fk_steiner_r01` | (신규) θ_r=0.1 — R=0.9481, P=0.1582, F1=0.2711 |
| `s03_a10_05_fk_steiner_r02` | (신규) θ_r=0.2 — R=0.9418, P=0.1644, F1=0.2800 |
| `s03_a10_02_fk_steiner_mid_col` | (신규) θ_r=0.3 — R=0.9293, P=0.1812, F1=0.3033 |
| `s03_a10_06_fk_steiner_r04` | (신규) θ_r=0.4 — R=0.9014, P=0.2125, F1=0.3439 |
| `s03_a10_03_fk_steiner_high_col` | (신규) θ_r=0.5 — R=0.8565, P=0.2627, F1=0.4021 |
| `s03_a10_07_fk_steiner_r06` | (신규) θ_r=0.6 — R=0.7789, P=0.3341, F1=0.4677 |
| `s03_a10_08_fk_steiner_r07` | (신규) θ_r=0.7 — R=0.6662, P=0.4245, F1=0.5185 |
| `s03_a10_09_fk_steiner_r08` | (신규) **θ_r=0.8 ★ F1 Peak** — R=0.5455, P=0.5044, F1=0.5241 |
| `s03_a10_10_fk_steiner_r09` | (신규) θ_r=0.9 — R=0.4083, P=0.5300, F1=0.4612 |
| `s03_a10_11_fk_steiner_r10` | (신규) θ_r=1.0 (FK-only) — R=0.2972, P=0.4920, F1=0.3706 |
| _(offline)_ `a10_09` percentile sweep | (2026-04-17) 4 scopes × 21 percentiles = 85 configs, offline re-eval using a10_09 score_analysis (no new config ID). Best: `all_cols p=95` R=0.6167 / P=0.4626 / F1=0.5287. HISTORY §6-19. |

### s04_gat_qcond_projector/

| 신규 ID | 기존 |
|---------|------|
| `s04_01_qcond_a085_xiyan` | `experiment_qcond_idea24_xiyan` (Q1) |
| `s04_02_supernode_a070_xiyan` | `experiment_supernode_idea24_xiyan` (Q2) |
| `s04_03_supernode_a085_xiyan` | `experiment_supernode_idea24_a085_xiyan` (Q3) |
| `s04_04_qcond_a0_xiyan` | `experiment_qcond_idea24_a0_xiyan` (Q4) |
| `s04_05_supernode_a0_xiyan` | `experiment_supernode_idea24_a0_xiyan` (Q5) |

### s04_ablation/stagewise/

**Motivation**: Wave 1.5 backfill — Extractor 축을 `PCSTExtractor(Basic)` 로 통일해 Selector 축 순수 기여 분리. 2026-04-22, 지도교수 advisor input §9 Root 행 + stagewise_qcond_ablation.md §1.1 Extractor 불일치 caveat 근거. HISTORY §8 참조.

| 신규 ID | 설명 |
|---------|------|
| `s04_stagewise_ensemble_raw_a0` | (신규, 2026-04-22 W1) Legacy cosine-only EnsembleSelector α=0 + Basic PCST + XiYan — R=0.6676 P=0.7236 F1=0.6944 |
| `s04_stagewise_qcond_raw_basic` | (신규, 2026-04-22 W2) QCond encoder + EnsembleSelector α=0 + Basic PCST + XiYan — R=0.6622 P=0.7539 F1=0.7051 |
| `s04_stagewise_qcond_gat_basic` | (신규, 2026-04-22 W3) ★ QCond + GAT blend α=0.85 + Basic PCST + XiYan — **R=0.8169 P=0.7605 F1=0.7877 (new top)** |

### s05_gat_direct/a01_full_pipeline/

| 신규 ID | 기존 |
|---------|------|
| `s05_a01_01_qcond_direct_xiyan` | `experiment_qcond_direct_idea24_xiyan` (Q6) |
| `s05_a01_02_supernode_direct_xiyan` | `experiment_supernode_direct_idea24_xiyan` (Q7) |

### s06_gat_bottleneck_fix/a01_additive_ablation/

**Motivation**: `outputs/analysis/gat_bottleneck{,_qcond}/` 의 3-step 병목 진단 결과 도출된 처방의 기여 확인 ablation.
- L1 catastrophic over-smoothing (QCond 0.89 / SN 0.97)
- Skip-dominated gradient (SN) / Input-dominated (QCond)
- Attention uniformity (L1=L2=L3 동일 패턴)
- BCE–Recall divergence (QCond ep75, SN ep79)

**Strategy**: Forward-additive ablation. B0 → B1 → ... → B5 로 한 처방씩 누적해 per-component 기여 측정.
**Base anchor**: QCond Direct (T8, `best_gat_query_conditioned_direct.pt`). SuperNode 는 차후 GPU 여유 시 대칭 검증.
**Filter 미적용**: Selector 단독 품질 (Val Recall@15) 기준. PCST/XiYan 은 추후 결합.

| 신규 ID | 처방 (누적) | 근거 논문 | Val R@15 | Status |
|---------|-----------|----------|---------|--------|
| `s06_a01_01_b0_baseline` | 현행 QCond Direct (reference) | — | 0.5738 | ✅ 2026-04-16 (300ep) |
| `s06_a01_02_b1_pairnorm` | + PairNorm (layer-wise) | Zhao & Akoglu (ICLR 2020) | 0.5707 | ✅ 2026-04-16 (-0.0031) |
| `s06_a01_03_b2_initial_residual` | + Initial Residual (α=0.2) | Klicpera et al. (APPNP, ICLR 2019); Chen et al. (GCNII, ICML 2020) | 0.5986 | ✅ 2026-04-17 (+0.0248) |
| `s06_a01_04_b3_listnet` | Loss: BCE → ListNet | Cao et al. (ICML 2007) | 0.5745 | ✅ 2026-04-17 (+0.0007) |
| `s06_a01_05_b4_anti_collapse` | + Schema-Aware Anti-Collapse Reg (λ=0.3, τ_max=0.85) | 본 연구 | 0.5894 | ✅ 2026-04-17 (+0.0156) |
| `s06_a01_06_b5_dual_stream` | Full Dual-Stream (query/schema 분리 + JK concat, 2 layers) | 본 연구 | **0.6073** | ✅ 2026-04-19 (+0.0335, rerun after fk_node fix) |
| `s06_a01_07_b5_enriched_dual_stream` | B5 구조 ⊕ EnrichedHeteroGraphBuilder (tables.json NL + description CSV) | 본 연구 | 0.6016 | ✅ 2026-04-21 (+0.0278 vs B0, **−0.0057 vs B5**), batched dual_stream 으로 9h 14m (B5 ~29h 대비 3.1× 단축) |

**Offline post-hoc analyses on B5 (frozen L_out):**
- _(offline)_ **B5 Head Retrain 2×2** (2026-04-20, `outputs/analysis/s06_bottleneck/B5/retrain/`) — frozen L_out 위에 head 만 재학습. 5 cells: A(linear,bce,none), B(mlp,bce,none), C(mlp,listnet,none), D(mlp,bce,zscore), E(mlp,listnet,zscore). val-ES best: C Dev AUC 0.6891 / D Dev R@15 0.6228. dev-ES oracle best: C Dev AUC **0.7548** (+0.048 vs original B5 joint 0.7067). HISTORY §7-2.
- _(offline)_ **B5 Head-Only LDBO Diagnostic** (2026-04-20, `outputs/analysis/s06_bottleneck/B5/retrain/ldbo/`) — 같은 4 cells (B/C/D/E) 를 train 69 DB 중 11 DB 홀드아웃 (LDBO) 방식으로 재학습. val R@15 여전히 0.99+ (→ held-out train DB와 dev DB 간 domain gap 큼). val-ES dev AUC LDBO vs query-random gap 미미 (-0.003~+0.007). **결론: train 내부 DB 다양성만으로는 realistic BIRD dev shift 를 simulate 불가.** HISTORY §7-3.

### abl/a01_2x2x2_selector_extractor_filter/ (Phase C)

| 신규 ID | 기존 |
|---------|------|
| `abl_a01_01_cos_basic` | (from B0 run, cell 1) |
| `abl_a01_02_cos_adaptive` | (from B1 run, cell 2) |
| `abl_a01_03_ens_basic` | `experiment_abl_ens_basic_xiyan`의 non-xiyan — 별도 run 없음 (history 참조) |
| `abl_a01_04_ens_adaptive` | `experiment_b_combined` = cell 4 (cross-link, s03_a02_01 공유) |
| `abl_a01_05_cos_basic_xiyan` | `experiment_abl_cos_basic_xiyan` |
| `abl_a01_06_ens_basic_xiyan` | `experiment_abl_ens_basic_xiyan` |
| `abl_a01_07_cos_adaptive_xiyan` | `experiment_abl_cos_adaptive_xiyan` |
| `abl_a01_08_ens_adaptive_xiyan` | `experiment_b4_xiyan_filter` = cell 8 (s03_a02_03 공유) |

### abl/a02_alpha_sweep/

| 신규 ID | 기존 |
|---------|------|
| `abl_a02_01_alpha085` | `experiment_b_combined` (α=0.85, cross-link s03_a02_01) |
| `abl_a02_02_alpha075` | `experiment_idea1_alpha075` (I1b) |
| `abl_a02_03_alpha070` | `experiment_idea1_alpha070` (I1c) |

### abl/a03_direct_per_step/

| 신규 ID | 기존 |
|---------|------|
| `abl_a03_01_qcond_selector_only` | `ablation_qcond_direct_selector_only` |
| `abl_a03_02_qcond_selector_extractor` | `ablation_qcond_direct_selector_extractor` |
| `abl_a03_03_supernode_selector_only` | `ablation_supernode_direct_selector_only` |
| `abl_a03_04_supernode_selector_extractor` | `ablation_supernode_direct_selector_extractor` |
| `abl_a03_05_qcond_binary_selector_only` | `ablation_qcond_direct_binary_selector_only` |
| `abl_a03_06_qcond_binary_selector_extractor` | `ablation_qcond_direct_binary_selector_extractor` |
| `abl_a03_07_qcond_binary_steiner` | `ablation_qcond_direct_binary_steiner` |
| `abl_a03_08_qcond_binary_full` | `ablation_qcond_direct_binary_full` |
| `abl_a03_09_supernode_binary_selector_only` | `ablation_supernode_direct_binary_selector_only` |
| `abl_a03_10_supernode_binary_selector_extractor` | `ablation_supernode_direct_binary_selector_extractor` |
| `abl_a03_11_supernode_binary_steiner` | `ablation_supernode_direct_binary_steiner` |
| `abl_a03_12_supernode_binary_full` | `ablation_supernode_direct_binary_full` |
| `abl_a03_13_qcond_binary_fixed` | (신규, 2026-04-14) QCond Direct + Fixed PCST, no filter |
| `abl_a03_14_qcond_binary_fixed_xiyan` | (신규, 2026-04-14) QCond Direct + Fixed PCST + XiYan |
| `abl_a03_15_qcond_binary_steiner_xiyan` | (신규, 2026-04-14) QCond Direct + Steiner PCST + XiYan |
| `abl_a03_16_supernode_binary_fixed` | (신규, 2026-04-14) SuperNode Direct + Fixed PCST, no filter |
| `abl_a03_17_supernode_binary_fixed_xiyan` | (신규, 2026-04-14) SuperNode Direct + Fixed PCST + XiYan |
| `abl_a03_18_supernode_binary_steiner_xiyan` | (신규, 2026-04-14) SuperNode Direct + Steiner PCST + XiYan |

### abl/a04_direct_binary_steiner_sweep/

| 신규 ID | 기존 |
|---------|------|
| `abl_a04_01_supernode_t005_steiner_xiyan` | `ablation_supernode_binary_t005_steiner_xiyan` |
| `abl_a04_02_supernode_t010_steiner_xiyan` | `ablation_supernode_binary_t010_steiner_xiyan` |
| `abl_a04_03_supernode_t015_steiner_xiyan` | `ablation_supernode_binary_t015_steiner_xiyan` |
| `abl_a04_04_supernode_t020_steiner_xiyan` | `ablation_supernode_binary_t020_steiner_xiyan` |
| `abl_a04_offline_sweep` | `src/analysis/threshold_steiner_sweep.py` (offline script, no config) |

### abl/a05_filter_agentic/ (Phase D — rolling execution 2026-04-15; a05_11/12 deferred; a05_13/14/15/17 added 2026-04-16~17 as gpt-4o-mini backbone sensitivity; a05_16 skipped for cost)

Filter 모듈 고도화 (plan: `/home/hyeonjin/.claude/plans/vivid-sprouting-sunbeam.md`).
Anchor: a03_17 components (SuperNode Direct + Fixed PCST). 외부 baseline 없음.

| 신규 ID | Filter 모듈 | 근거 축 | Status |
|---------|------------|---------|--------|
| `a05_01_adaptive_multi_agent` | AdaptiveMultiAgentFilter (existing) | Multi-agent baseline | ✅ R=0.3770 / P=0.6276 / F1=0.4713 |
| `a05_02_reflection_1iter` | ReflectionFilter (F1, 1 iter) | Self-Refine (NeurIPS'23) | ✅ R=0.7320 / P=0.6833 / F1=0.7068 |
| `a05_03_reflection_3iter` | ReflectionFilter (F1, 3 iter) | Iteration depth |
| `a05_04_verifier` | VerifierFilter (F2) | CHESS Unit Tester (ICLR'25) | ✅ R=0.7093 / P=0.6676 / F1=0.6878 |
| `a05_05_tiered_no_tools` | TieredBidirectionalAgent (F3, no tools) | Ablation vs a05_06 |
| `a05_06_tiered_full_tools` | TieredBidirectionalAgent (F3, full tools) | ★ 핵심 기여 |
| `a05_07_adaptive_depth` | AdaptiveDepthFilter (F4) | Uncertainty routing |
| `a05_08_tiered_verifier_stack` | Stacked(F3→F2) | 상한 |
| `a05_09_tiered_retry` | F3 + pipeline F5 (K=2) | Extractor reverse feedback |
| `a05_10_adaptive_retry` | F4 + F5 (K=2) | Selective retry |
| `a05_11_tiered_gpt4omini` | F3, GPT-4o-mini backbone | Backbone 민감도 |
| `a05_12_adaptive_retry_gpt4omini` | F4+F5, GPT-4o-mini | Backbone 민감도 |
| `a05_13_xiyan_gpt4omini` | XiYanFilter (gpt-4o-mini backbone) | Backbone 민감도 (prune-only baseline) | ✅ R=0.6037 / P=0.7317 / F1=0.6616 |
| `a05_14_adaptive_multi_agent_gpt4omini` | AdaptiveMultiAgentFilter (gpt-4o-mini backbone) | Backbone 민감도 (multi-agent) | ✅ R=0.3992 / P=0.7576 / F1=0.5230 |
| `a05_15_reflection_1iter_gpt4omini` | ReflectionFilter 1iter (gpt-4o-mini backbone) | Backbone 민감도 (restore path) | ✅ R=0.6827 / P=0.6620 / F1=0.6722 |
| `a05_16_reflection_3iter_gpt4omini` | ReflectionFilter 3iter (gpt-4o-mini backbone) | Backbone 민감도 (deep critique) | ⊘ Skipped (비용 $3.6 추정, a05_03 Qwen 미완료로 기대 이득 불명확) |
| `a05_17_verifier_gpt4omini` | VerifierFilter (gpt-4o-mini backbone) | Backbone 민감도 (unit test path) | ✅ R=0.7055 / P=0.6385 / F1=0.6706 |

### abl/a06_ext_fkprior/ (Phase — Extractor E-III, 2026-04-20 추가)

Hybrid PCST with FK topology prior. Anchor: s03_a02_01 (Ensemble Selector + AdaptivePCST).
Spec: [src/modules/extractors/EXPERIMENT_PLAN_extractors.md §E-III](src/modules/extractors/EXPERIMENT_PLAN_extractors.md).
On-the-fly FK shortest-path fallback 포함 — Builder B-III 완료 전에도 실행 가능.

| 신규 ID | Extractor 파라미터 | Filter | Status |
|---------|---------------------|--------|--------|
| `a06_01_fkprior_discount03` | discount=0.3, bridge=0.0 | None | pending |
| `a06_02_fkprior_bridge05` | discount=0.3, bridge=0.5 | None | pending |
| `a06_03_fkprior_aggressive` | discount=0.5, bridge=1.0 | None | pending |
| `a06_04_fkprior_discount03_xiyan` | discount=0.3, bridge=0.0 | XiYan | pending (paper main comp) |

### abl/a07_ext_path_ensemble/ (Phase — Extractor E-II, 2026-04-20 추가)

PCST + Steiner pathfinder ensemble (union / 2pass / intersection modes).
Anchor: s03_a02_01 (Ensemble + AdaptivePCST). MSTExtractor 의 `steiner_tree_2approx` 재활용.
Spec: [src/modules/extractors/EXPERIMENT_PLAN_extractors.md §E-II](src/modules/extractors/EXPERIMENT_PLAN_extractors.md).

| 신규 ID | mode / k_anchors / boost | Filter | Status |
|---------|--------------------------|--------|--------|
| `a07_01_path_union_k5` | union / k=5 / boost=0.2 | None | pending |
| `a07_02_path_2pass_k5` | 2pass / k=5 / boost=0.2 | None | pending |
| `a07_03_path_union_k3` | union / k=3 / boost=0.2 | None | pending |
| `a07_04_path_2pass_k10` | 2pass / k=10 / boost=0.3 | None | pending |
| `a07_05_path_union_k5_xiyan` | union / k=5 / boost=0.2 | XiYan | pending (paper main comp) |

### abl/a08_ext_louvain/ (Phase — Extractor E-I, 2026-04-20 추가)

Louvain community masking 후 Base PCST. `networkx.algorithms.community.louvain_communities` 사용.
Anchor: s03_a02_01 (Ensemble + AdaptivePCST). 입력 그래프 community (vs CA 의 PCST 이후 component).
Spec: [src/modules/extractors/EXPERIMENT_PLAN_extractors.md §E-I](src/modules/extractors/EXPERIMENT_PLAN_extractors.md).

| 신규 ID | 파라미터 | Filter | Status |
|---------|---------|--------|--------|
| `a08_01_louvain_top2` | res=1.0, top_m=2 | None | pending |
| `a08_02_louvain_res05` | res=0.5, top_m=2 (coarse cluster) | None | pending |
| `a08_03_louvain_adaptive_top3` | res=1.0, top_m=3, adaptive_coverage=True | None | pending |

### abl/sel/ (Phase — Selector architecture ablation, 2026-04-20 추가)

루트 PLAN Selector 5축(S-I ~ S-V). 우선순위 S-V > S-III > S-II > S-IV > S-I. 본 entries 는 selector 인프라 ID 슬롯을 예약.
Spec: [src/modules/selectors/EXPERIMENT_PLAN_selectors.md](src/modules/selectors/EXPERIMENT_PLAN_selectors.md).

| 신규 ID | Selector | 핵심 파라미터 | Status |
|---------|---------|--------------|--------|
| `abl_sel_ns_l1_01` | NeurosymbolicL1Selector (S-V) — EnsembleSelector + λ·reach_mask hook (FK-reachability additive prior) | λ=0.1, α=0.85, top_k=20, anchor_min_token_len=3 | ✅ 구현 + smoke 통과 (reach_mask 정확도 검증, 4-component FK 그래프 anchor→component 매핑 일치). End-to-end F1 pending (vLLM 서버 필요). Anchor: `s03_a02_03_xiyan_filter` (Ensemble+AdaptivePCST+XiYan) |

### abl/build/ (Phase A — Builder infrastructure, 2026-04-20 추가)

루트 PLAN Phase A 의 Builder 3축(B-I/B-II/B-III). 본 entries 는 builder/metadata 인프라 ID 슬롯을 예약하고, 하류 Selector S-II/S-III/S-V 가 합류한 뒤 end-to-end 결과로 갱신.
Spec: [src/modules/builders/EXPERIMENT_PLAN_builders.md](src/modules/builders/EXPERIMENT_PLAN_builders.md).

| 신규 ID | Builder | 연동 하류 | Status |
|---------|---------|----------|--------|
| `abl_build_01_fk_reach` | EnrichedHeteroGraphBuilder + auto-injected FK reachability metadata (B-III) | 기존 Ensemble + AdaptivePCST + XiYan (Selector S-V / Extractor E-III / Filter FL-III 미합류) | ✅ Builder smoke 통과 (california_schools T=3, FK=2, reach=1.000, comps=1; dev pair coverage 93.53%, query coverage 94.45%). End-to-end run pending — Anchor: `s03_a07_01_enriched_gat` (E1, F1=0.7327) |
| `abl_build_02_linegraph` | LineGraphBuilder(base=EnrichedHeteroGraphBuilder) (B-II) | 하류 Selector S-III(EHGAT) 미구현 | ✅ Builder smoke 통과 (california_schools edge_nodes=97, line_edges=3856, feat_dim=772). End-to-end pending S-III. |
| `abl_build_03_rfm_tokens` | RFMCompatibleBuilder (B-I) — Enriched 위에 RFM 호환 special-token serialization 부착 | 하류 Selector S-II(RFM encoder) 미구현 — 현 stack 에서는 Enriched 와 동일 동작 | ✅ Builder smoke 통과 (dev 11 DB token median 1041 / max 2578). End-to-end pending S-II. Anchor: `s03_a07_01_enriched_gat` (E1, F1=0.7327) |
| `abl_build_05_no_t2t` | EnrichedHeteroGraphBuilder + `add_t2t_edges=False` (B-II.b, advisor 2026-04-21 의견 2) | 기존 Ensemble + AdaptivePCST + XiYan. Anchor checkpoint 가 T2T 포함 그래프로 학습 → distribution shift 가능, recall 하락 시 GAT 재학습 필요 | ✅ Builder smoke 통과 (california_schools: T2T 4→0, FK reachability 동일, schema_diameter 4→8). End-to-end pending. Anchor: `s03_a07_01_enriched_gat` (E1, F1=0.7327) |
| `abl_build_06_diameter_meta` | EnrichedHeteroGraphBuilder + auto-injected `schema_diameter`/`schema_eccentricity` (B-III.b, advisor 2026-04-21 의견 2) | 기존 Ensemble + AdaptivePCST + XiYan (메타키 무시). 후속 QCondGAT `num_layers ∈ {1,2,3,D_max,D_max+1}` 스윕(advisor proposal C) 인프라 | ✅ Builder smoke 통과 (BIRD-Dev 11 DB D_max profile: min=3, median=5, mean=4.91, max=6 — 현 GAT default `num_layers=3` 은 1/11 DB 만 충분). End-to-end regression marker. Anchor: `s03_a07_01_enriched_gat` (E1, F1=0.7327) |

### GAT Checkpoints (별도 네임스페이스)

| 신규 ID | 기존 T# | checkpoint |
|---------|---------|------------|
| `t01_gat_v1` | T1 | `gat_classifier_best.pt` |
| `t02_mlp_classifier` | T2 | `mlp_classifier_train_best_recall.pt` |
| `t03_mlp_gat` | T3 | `mlp_classifier_with_gat_train_best_recall.pt` |
| `t04_gat_infonce` | T4 | `best_gat_model.pt` |
| `t05_enriched_gat` | T5 | `best_gat_enriched.pt` |
| `t06_qcond_projector` | T6 | `best_gat_query_conditioned.pt` |
| `t07_supernode_projector` | T7 | `best_gat_query_supernode.pt` |
| `t08_qcond_direct` | T8 | `best_gat_query_conditioned_direct.pt` |
| `t09_supernode_direct` | T9 | `best_gat_query_supernode_direct.pt` |

## 중복/아카이브 처리

- `outputs/experiments/qcond_idea24_a0_xiyan/`, `outputs/experiments/supernode_idea24_a0_xiyan/`:
  더 이른 run. `experiment_` 접두사 버전이 canonical. 아카이브 → `outputs/archive/legacy_base_runs/`

---

## GLM era `_glm` suffix 규칙 (2026-04-24)

LLM backbone 교체 (vLLM `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` → GLM-4.7 (`zai-org/glm-4.7`) via Elice ML API, OpenAI-compatible) 로 발생한 실험 시리즈. 기존 ID 에 **`_glm` suffix** 를 붙여 vLLM era 원본과 구분.

### 명명 규칙
- vLLM era anchor 가 존재하는 경우: `<original_id>_glm` (예: `s04_04_qcond_a0_xiyan` → `s04_04_qcond_a0_xiyan_glm`)
- 새 sweep cell: 원본 명명 규칙 + `_glm` (예: `abl_sel_diameter_layers_nl{1,2,3,6,7}_glm`)
- 파일 경로도 `_glm` suffix: `configs/.../layers_L{1,2,3,6,7}_glm.yaml`

### GLM era 실험 목록 (2026-04-24 kickoff)

| 신규 ID | vLLM era 대응 | 역할 | F1 |
|---------|--------------|------|---:|
| `s04_04_qcond_a0_xiyan_glm` | `s04_04_qcond_a0_xiyan` | GLM backbone sanity | 0.5768 |
| `abl_sel_diameter_layers_nl1_glm` | — (new) | diameter sweep nl=1 | 0.5785 |
| `abl_sel_diameter_layers_nl2_glm` | — (new) | diameter sweep nl=2 | 0.5510 |
| `abl_sel_diameter_layers_nl3_glm` | — (new) | diameter sweep nl=3 | 0.5752 |
| `abl_sel_diameter_layers_nl6_glm` | — (new) | diameter sweep nl=6 = D_max (peak) | **0.5824** |
| `abl_sel_diameter_layers_nl7_glm` | — (new) | diameter sweep nl=7 = D_max+1 | 0.5762 |
| `s04_stagewise_qcond_gat_basic_glm` | `s04_stagewise_qcond_gat_basic` | GLM era new anchor (전체 최고) | **0.8383** 🚀 |
| `layers_Ldbmax_glm` (H2, 2026-04-25) | — (new, selector v2 `_resolve_active_depth`) | H2 truncate: nl=6 ckpt + D_max mode | 0.5869 |
| `layers_Ldbmax_plus1_glm` (H2, 2026-04-25) | — (new, selector v2 `_resolve_active_depth`) | H2 truncate: nl=7 ckpt + D_max_plus1 mode | 0.5604 |

### 경로 변경
- Sanity (`s04_04_qcond_a0_xiyan_glm`) 와 new anchor (`s04_stagewise_qcond_gat_basic_glm`) 의 `_glm` variant 는 **`configs/experiments/s04_ablation/` 하위** 로 배치. 원본 `s04_gat_qcond_projector/` (sanity) / `s04_ablation/stagewise/` (anchor) 와 달리 GLM era 를 `s04_ablation` 클러스터로 통합 관리 (변경 전 원본 경로는 historical reference 로 보존).

### Config 주의사항 (재현 필수)
- diameter_layers 계열 yaml 은 반드시 `seed_selector.params.num_layers: N` 명시 (N∈{1,2,3,6,7}). `EnsembleSelector` default `num_layers=3` 이라 누락 시 N≠3 체크포인트에서 weight shape mismatch RuntimeError (2026-04-24 1회 full failure 원인).
- `.env` 에 `GLM_BASE_URL=https://mlapi.run/<api_id>/v1` (SDK 표준 `/v1` suffix) + `GLM_API_KEY=<bearer>` 필수. Endpoint 는 `POST {GLM_BASE_URL}/chat/completions` (OpenAI SDK auto-append).
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Wave 2 Proposal C GLM era kickoff (2026-04-24)](EXPERIMENT_HISTORY.md).

---

## `_selector_only` / `_no_filter` suffix 규칙 (2026-04-26, Builder Cumulative Backfill)

**용도**: Ablation 1 (Builder × Stage) cumulative matrix 측정 — 동일 anchor 의 Selector-only / +Extractor 단계를 final stage 와 분리해서 R/P/F1 정량.

### 명명 규칙
- `<anchor_id>_selector_only`: Anchor 의 Builder + Selector(Ensemble α=0.85, top_k=20) 만 활성화. Extractor=None, Filter=None, auto_join_keys=false.
- `<anchor_id>_no_filter`: Anchor 의 Builder + Selector + Extractor (Basic PCST 기본값) 활성화. Filter=None, auto_join_keys=true.
- 동일 폴더 (`stagewise/` 또는 `a07_enriched_triplet/`) 내 anchor 와 같은 위치.

### 등재된 cells (2026-04-26 측정)

| 신규 ID | Anchor (final stage) | Stage | F1 |
|---------|---------------------|-------|---:|
| `s04_stagewise_qcond_gat_basic_selector_only` | `s04_stagewise_qcond_gat_basic` | Selector only (Plain) | 0.4016 |
| `s03_a07_01_enriched_gat_selector_only` | `s03_a07_01_enriched_gat` | Selector only (Enriched) | 0.3877 |
| `s03_a07_02_edge_prize_selector_only` | `s03_a07_02_edge_prize` | Selector only (Triplet) | 0.3877 |
| `s03_a07_01_enriched_gat_no_filter` | `s03_a07_01_enriched_gat` | +Extractor no filter (Enriched) | 0.2252 |
| `s03_a07_02_edge_prize_no_filter` | `s03_a07_02_edge_prize` | +Extractor no filter (Triplet) | 0.2252 |
| `s04_stagewise_qcond_gat_basic_no_filter` (기존) | `s04_stagewise_qcond_gat_basic` | +Extractor no filter (Plain) | 0.2271 (Wave 1.5 backfill) |

### Config 주의사항
- `selector_only` config: `connectivity_extractor.name: "None"` + `filter.name: "None"` (둘 다 빈 params={}) + `post_processing.auto_join_keys: false`. Reference pattern: `configs/experiments/abl/a03_direct_per_step/abl_a03_0{1,3,5,9}_*_selector_only.yaml`.
- `no_filter` config: anchor 의 Extractor 가 ComponentAware/Adaptive/EdgePrize 등이어도 **Basic `PCSTExtractor` 로 통일** (Wave 1.5 backfill 패턴 일관). `filter.name: "None"`, `auto_join_keys: true`.
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Builder Cumulative Backfill (2026-04-26)](EXPERIMENT_HISTORY.md).

---

## Ablation 2 Selector cells 명명 규칙 (2026-04-26)

**용도**: Ablation 2 (Encoder × Score × Stage) cumulative matrix 측정 — Encoder (Plain/QCond/SuperNode) × Score (GAT α=0 / Cosine α=1 / Ensemble α=0.85).

### 명명 규칙
- 패턴: `{encoder}_{score}_{stage}.yaml`
  - encoder ∈ `plain` / `qcond` / `supernode`
  - score ∈ `gat_a0` (α=0) / `cos_a1` (α=1) / `ens` (α=0.85)
  - stage ∈ `selector_only` / `no_filter` / `glm` (final stage with GLM-4.7)
- 폴더: `configs/experiments/s04_ablation/stagewise/{selector_only/,no_filter/,...}`
- α convention: `final_score = α·cosine + (1−α)·gat` ([ensemble_selector.py:28](../src/modules/selectors/ensemble_selector.py#L28))

### Encoder 별 ckpt + flags (selector_only/no_filter/final 모든 stage 공통)
- **Plain**: `weight_path=best_gat_model.pt`, `query_conditioned=false`, `query_supernode=false`
- **QCond**: `weight_path=best_gat_query_conditioned.pt`, `query_conditioned=true`, `query_supernode=false`
- **SuperNode (보류)**: `weight_path=best_gat_query_supernode.pt`, `query_conditioned=true`, `query_supernode=true` — ckpt input dim mismatch 로 측정 불가 (DECISIONS 2026-04-26 SuperNode smoke fail 엔트리 + 2026-04-22 17:05 §8-1 bug)

### 등재된 cells (2026-04-26 측정, 10 신규 + 9 SuperNode 보류)

| 신규 ID | Encoder × Score | Stage | F1 |
|---------|-----------------|-------|---:|
| `s04_stagewise_plain_gat_a0_selector_only` | Plain × GAT(α=0) | Selector only | 0.2937 |
| `s04_stagewise_plain_cos_a1_selector_only` | Plain × Cosine(α=1) | Selector only | 0.3829 |
| `s04_stagewise_plain_ens_selector_only` | Plain × Ensemble(α=0.85) | Selector only | 0.3974 |
| `s04_stagewise_qcond_gat_a0_selector_only` | QCond × GAT(α=0) | Selector only | 0.3534 |
| `s04_stagewise_qcond_cos_a1_selector_only` | QCond × Cosine(α=1) | Selector only | 0.3829 |
| `s04_stagewise_qcond_ens_selector_only` | QCond × Ensemble(α=0.85) | Selector only | 0.4016 |
| `s04_stagewise_plain_cos_a1_no_filter` | Plain × Cosine(α=1) | + Extractor | 0.2295 |
| `s04_stagewise_plain_ens_no_filter` | Plain × Ensemble(α=0.85) | + Extractor | 0.2250 |
| `s04_stagewise_qcond_cos_a1_no_filter` | QCond × Cosine(α=1) | + Extractor | 0.2295 |
| **`s04_stagewise_qcond_cos_a1_glm`** | QCond × Cosine(α=1) | + Filter (GLM) | **0.8424** 🚀 (새 GLM era 후보) |
| `s04_stagewise_supernode_*_{selector_only,no_filter,glm}` (9개) | SuperNode × {GAT/Cos/Ens} | 모든 stage | 보류 (smoke fail 2026-04-26) |

### Config 주의사항
- α convention: yaml `seed_selector.params.alpha: 0.0` (GAT only), `1.0` (Cosine only), `0.85` (Ensemble Cosine 우세).
- Cosine only (α=1) 사용 시 GAT module 학습 가중치 무관 — 같은 α=1 이면 Plain/QCond 결과 동일 (PLM 임베딩 직접 사용).
- SuperNode cells 사용 전: ckpt input dim 정합성 사전 검증 (smoke 1 query) 필수.
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Selector Ablation Cumulative Backfill (Option B, 2026-04-26)](EXPERIMENT_HISTORY.md).

---

## GLM era 일관 재측정 — Ablation 1/2/3 (2026-04-27)

**용도**: Wave 2 Phase 2 + Selector backfill 결과 위에 Ablation 1 (Builder × Stage) / Ablation 2 (Encoder × Score × Stage) / Ablation 3 (Extractor × Stage) 의 missing cells 를 GLM era 로 일관 재측정 — 11 cells (8 final GLM + 3 LLM-free no-filter). 발사 2026-04-27 01:01:27 → 완료 03:14:47 (wall clock 2h 13min).

### Ablation 1 — Builder × Stage Enriched final 1 cell

| 신규 ID | Builder | Stage | F1 |
|---------|---------|-------|---:|
| `s03_a07_01_enriched_gat_glm` | Enriched | Final (XiYan(GLM)) | 0.7551 |

vs vLLM era `s03_a07_01_enriched_gat` (F1≈0.7140): **+0.0411**.

### Ablation 2 — Encoder × Score × Stage final 4 cells

| 신규 ID | Encoder × Score | Stage | F1 |
|---------|-----------------|-------|---:|
| `s04_stagewise_plain_gat_a0_glm` | Plain × GAT(α=0) | Final (GLM) | 0.6985 |
| **`s04_stagewise_plain_cos_a1_glm`** | Plain × Cosine(α=1) | Final (GLM) | **0.8390** ★ |
| `s04_stagewise_plain_ens_glm` | Plain × Ensemble(α=0.85) | Final (GLM) | 0.8381 |
| `s04_stagewise_qcond_gat_a0_glm` | QCond × GAT(α=0) | Final (GLM) | 0.7211 |

**관찰**: Plain Cos α=1 (F1=0.8390) ≈ QCond Cos α=1 (F1=0.8424, 직전 측정) — encoder agnostic. anchor 갱신 임계 (+0.005) 미달 → 직전 anchor `qcond_gat_basic_glm` (F1=0.8383) 유지하되 `plain_cos_a1_glm` 동률 후보 표기.

### Ablation 3 — Extractor × Stage 신규 폴더

**신규 폴더**: `configs/experiments/s04_ablation/extractor/`, `configs/experiments/s04_ablation/extractor/no_filter/`

#### 명명 규칙

- 패턴: `plain_ens_{extractor}_{stage}.yaml`
  - extractor ∈ `adaptive` / `steiner` / `mst`
  - stage ∈ `glm` (final XiYan(GLM)) / `no_filter` (filter.name=None)
- 공통 selector: `Plain encoder + Ensemble (α=0.85) + best_gat_model.pt`

#### 등재된 cells (3 final + 3 no_filter)

| 신규 ID | Extractor | Stage | F1 |
|---------|-----------|-------|---:|
| `s04_extractor_plain_ens_adaptive_glm` | AdaptivePCST | Final (GLM) | 0.7199 |
| `s04_extractor_plain_ens_steiner_glm` | SteinerBackbonePCST | Final (GLM) | 0.7545 |
| `s04_extractor_plain_ens_mst_glm` | MSTExtractor (Steiner 2-approx) | Final (GLM) | 0.7730 |
| `s04_extractor_plain_ens_adaptive_no_filter` | AdaptivePCST | No filter | 0.4704 |
| `s04_extractor_plain_ens_steiner_no_filter` | SteinerBackbonePCST | No filter | 0.3651 |
| `s04_extractor_plain_ens_mst_no_filter` | MSTExtractor (Steiner 2-approx) | No filter | 0.3689 |

**참고**: Basic PCST (`s04_stagewise_plain_ens_glm`, F1=0.8381) 가 Ablation 3 비교 anchor — Basic > MST > Steiner > Adaptive 위계 GLM era 재현.

### Config 주의사항

- 모든 GLM cells 의 filter: `provider: "glm", model_name: "zai-org/glm-4.7", max_iteration: 1, temperature: 0.0`
- MSTExtractor: `params: {}` (kwargs 없음) — seed_nodes 기반 metric closure, prize 무관
- 세부 실행 이력: [EXPERIMENT_HISTORY.md GLM era 일관 재측정 (Ablation 1/2/3, 2026-04-27)](EXPERIMENT_HISTORY.md).

---

## Ablation 1/2/3 α=0.5 재측정 (2026-04-27, 15 cells)

**용도**: Ensemble baseline α=0.85 (Cosine 우세) → α=0.5 (neutral GAT/Cosine 동등) 재정의 측정. 발사 2026-04-27 14:41:16 → 완료 17:42:22 (wall clock 3h 1min).

**근거**: I1a-c sweep "α=0.85 best" 결론이 No Filter stack 한정이고 with-Filter 미수행. Filter 단 ensemble 약화 분석 (HISTORY L92). Advisor analysis (α=0.85 GAT 비중 비판). 사용자 confirm 2026-04-27.

### 명명 규칙 — `_a05` suffix

- 패턴: `<encoder>_<score>_a05_<stage>.yaml` 또는 `<extractor>_a05_<stage>.yaml`
  - α=0.5 cells 는 모두 `_a05` suffix 추가 (α=0.85 cells 는 suffix 없음, default)
- 폴더: 기존 `s04_ablation/stagewise/`, `s04_ablation/extractor/`, `s03_gat_ensemble/a07_enriched_triplet/` 그대로 사용

### 등재된 cells (15 신규, all 측정 완료)

#### Ablation 2 — Plain/QCond × α=0.5 × 3 stage (6 cells)

| 신규 ID | Encoder | Stage | F1 |
|---------|---------|-------|---:|
| `s04_stagewise_plain_ens_a05_selector_only` | Plain | Selector only | 0.3432 |
| `s04_stagewise_plain_ens_a05_no_filter` | Plain | + Extractor (Basic) | 0.2159 |
| `s04_stagewise_plain_ens_a05_glm` | Plain | + Filter (GLM) Final | **0.8252** |
| `s04_stagewise_qcond_ens_a05_selector_only` | QCond | Selector only | 0.3997 |
| `s04_stagewise_qcond_ens_a05_no_filter` | QCond | + Extractor (Basic) | 0.2296 |
| `s04_stagewise_qcond_ens_a05_glm` | QCond | + Filter (GLM) Final | **0.8306** |

#### Ablation 1 — Enriched α=0.5 × 3 stage (3 cells)

| 신규 ID | Stage | F1 |
|---------|-------|---:|
| `s03_a07_01_enriched_a05_selector_only` | Selector only | 0.3389 |
| `s03_a07_01_enriched_a05_no_filter` | + Extractor (Basic) | 0.2184 |
| **`s03_a07_01_enriched_a05_glm`** ★ | + Filter (GLM) Final | **0.8262** |

#### Ablation 3 — Plain α=0.5 + 3 ext × 2 stage (6 cells)

| 신규 ID | Extractor | Stage | F1 |
|---------|-----------|-------|---:|
| `s04_extractor_plain_ens_a05_adaptive_no_filter` | AdaptivePCST | No filter | 0.3903 |
| `s04_extractor_plain_ens_a05_adaptive_glm` | AdaptivePCST | Final (GLM) | 0.5775 |
| `s04_extractor_plain_ens_a05_steiner_no_filter` | SteinerBackbone | No filter | 0.3230 |
| `s04_extractor_plain_ens_a05_steiner_glm` | SteinerBackbone | Final (GLM) | 0.6491 |
| `s04_extractor_plain_ens_a05_mst_no_filter` | MST (Steiner 2-approx) | No filter | 0.3338 |
| `s04_extractor_plain_ens_a05_mst_glm` | MST (Steiner 2-approx) | Final (GLM) | 0.6771 |

### Config 주의사항

- α=0.5 = neutral: GAT/Cosine 동등 결합 (GAT 50% + Cosine 50%)
- 모든 GLM cells: `provider: "glm", model_name: "zai-org/glm-4.7", max_iteration: 1, temperature: 0.0`
- α=0.85 baseline cells (suffix 없음) 와 cell-by-cell 비교 가능 (다른 모든 stack 동일)
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Ablation 1/2/3 α=0.5 Re-measurement (Option B, 2026-04-27)](EXPERIMENT_HISTORY.md).

---

## MST 변형 측정 (옵션 C + Union, 2026-04-27, 6 cells)

**용도**: 사용자 의문 "MST recall 이 왜 낮나" 해소 + 새 anchor (`plain_ens_a05_mst_kruskal_glm` F1=0.8642) 발견 + Union R 상한 검증.

**근거**: planning/DECISIONS.md 2026-04-27 (옵션 C 선택) + (MST ∪ PCST union 변형 측정 결정).

### 명명 규칙 — `_a05_<variant>` suffix

- 패턴: `plain_ens_a05_<variant>_<stage>.yaml`
  - variant ∈ `steiner_threshold` / `mst_kruskal` / `mst_pcst_union`
  - stage ∈ `glm` (final) / `no_filter` (LLM-free)
- 폴더: `s04_ablation/extractor/{no_filter/,}` 그대로

### 신규 Extractor 등록

- `MSTExtractor` (기존, seed_mode 추가): `seed_mode ∈ {topk, threshold}`, `score_threshold=0.1`
- `MSTKruskalExtractor` (신규): `score_threshold=0.1`, Kruskal MST on score>0.1 induced subgraph
- `MSTPCSTUnionExtractor` (신규): `score_threshold=0.1`, MSTKruskal ∪ Basic PCST 합집합

### 등재된 cells (6 신규, all 측정 완료)

| 신규 ID | Extractor | Stage | F1 |
|---------|-----------|-------|---:|
| `s04_extractor_plain_ens_a05_steiner_threshold_no_filter` | MSTExtractor seed_mode=threshold | No filter | 0.2177 |
| `s04_extractor_plain_ens_a05_steiner_threshold_glm` | MSTExtractor seed_mode=threshold | Final (GLM) | 0.8628 |
| `s04_extractor_plain_ens_a05_mst_kruskal_no_filter` | MSTKruskalExtractor | No filter | 0.2176 |
| **`s04_extractor_plain_ens_a05_mst_kruskal_glm`** ★ | MSTKruskalExtractor | Final (GLM) | **0.8642 (anchor)** |
| `s04_extractor_plain_ens_a05_mst_pcst_union_no_filter` | MSTPCSTUnionExtractor | No filter | 0.2176 |
| **`s04_extractor_plain_ens_a05_mst_pcst_union_glm`** 🆕 | MSTPCSTUnionExtractor | Final (GLM) | **0.8672 (plateau)** |

### Anchor 갱신 (2026-04-27 직전)

- **이전 anchor**: `qcond_gat_basic_glm` F1=0.8383
- **새 anchor**: `plain_ens_a05_mst_kruskal_glm` F1=**0.8642** (ΔF1=+0.0259, 임계 +0.005 의 5배 초과)
- Union 동률 후보: `plain_ens_a05_mst_pcst_union_glm` F1=0.8672 (시나리오 B, anchor 유지)

### Config 주의사항

- 모든 cells: Plain encoder + Ensemble α=0.5 + best_gat_model.pt 통일
- 모든 GLM cells: `provider: "glm", model_name: "zai-org/glm-4.7", max_iteration: 1, temperature: 0.0`
- score_threshold=0.1 통일 (3 신규 extractor 모두 동일)
- **명명 정정 (post-deadline)**: 기존 `MSTExtractor` 는 사실 Steiner 2-approx (Kou-Markowsky-Berman 1981). 코드 rename → `SteinerTreeExtractor` (alias 유지).
- 세부 실행 이력: [EXPERIMENT_HISTORY.md MST 변형 측정 (옵션 C + Union, 2026-04-27)](EXPERIMENT_HISTORY.md).

---

## Paper Main Pipeline 측정 (옵션 A2, 2026-04-28, 2 cells) — End-to-End Co-Design with Modular LLM Filter

**용도**: 사용자 의도된 paper main pipeline (방향 F') 의 정확한 F1 측정. paper title 권장: "LLM Filter as a First-Class Stage in Graph-RAG Schema Linking: Co-Designing Builder, Selector, Extractor, and Filter".

**근거**: planning/DECISIONS.md 2026-04-28 (방향 F' 최종 채택 + 옵션 A2 측정 결정).

### 명명 규칙 — `s04_pipeline_*` 신규 카테고리

- 패턴: `enriched_qcond_a05_<extractor>_<stage>.yaml`
  - 4 module 통합 표기: Enriched + QCond + α=0.5 + extractor
  - extractor ∈ `mst_kruskal` / `mst_pcst_union`
  - stage = `glm` (Final, paper main pipeline)
- 폴더: **신규** `configs/experiments/s04_ablation/pipeline/`

### Selector params 공통

- `EnsembleSelector(alpha=0.5, top_k=20, weight_path=best_gat_qcond_nl3.pt, query_conditioned=true, encoder_type=plm)`
- Builder: `EnrichedHeteroGraphBuilder(include_views=false, run_leiden_clustering=true, tables_json_path=data/raw/BIRD_dev/dev_tables.json)`
- Filter: `XiYanFilter(provider=glm, model_name=zai-org/glm-4.7, max_iteration=1, temperature=0.0)`

### 등재된 cells (2 신규, all 측정 완료)

| 신규 ID | Extractor | R | P | F1 |
|---------|-----------|---|---|---|
| **`s04_pipeline_enriched_qcond_a05_mst_kruskal_glm`** ★ paper main | MSTKruskalExtractor (score>0.1 induced Kruskal MST) | 0.8741 | 0.8606 | **0.8673** |
| `s04_pipeline_enriched_qcond_a05_mst_pcst_union_glm` | MSTPCSTUnionExtractor (MST ∪ Basic PCST) | 0.8772 | 0.8564 | 0.8667 |

### 비교 baseline

- Plain anchor (`plain_ens_a05_mst_kruskal_glm`): F1=0.8642
- Plain Union plateau (`plain_ens_a05_mst_pcst_union_glm`): F1=0.8672
- **paper main (Cell 1)** F1=0.8673 — Plain anchor 와 plateau 동등 (ΔF1=+0.0031, 임계 미달)

### Anchor 정리 (2026-04-28)

- **측정 anchor**: `plain_ens_a05_mst_kruskal_glm` F1=0.8642 (단순 stack, 측정 ground truth)
- **paper anchor 권장**: `s04_pipeline_enriched_qcond_a05_mst_kruskal_glm` F1=0.8673 (학술적 narrative)
- 두 anchor 가 plateau 내 동등 → narrative 차원에서 paper anchor 표기 가능

### Config 주의사항

- 모든 cells: Enriched Builder + QCond Encoder + Ensemble α=0.5 + GLM filter 통일
- score_threshold=0.1 통일 (extractor 양쪽 동일)
- weight_path: `best_gat_qcond_nl3.pt` (이전 `best_gat_query_conditioned.pt` 와 다름 — nl3 = num_layers=3, 2026-04-23 학습)
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Paper Main Pipeline Measurement (옵션 A2, 2026-04-28)](EXPERIMENT_HISTORY.md).

---

## SuperNode QCond GAT ckpt 신규 (2026-04-28, 옵션 A 학습)

**용도**: post-deadline H6 (Selector Concat vs SuperNode 결정) measurement 위한 신규 ckpt. paper_research_direction.md §1 paper main pipeline Selector 결정의 base.

**근거**: planning/DECISIONS.md 2026-04-27 (H6 옵션 A 선택).

### 명명 규칙 — 기존 SuperNode ckpt 와 분리

- 기존 (2026-04-11): `best_gat_query_supernode.pt` — `query_conditioned=False, query_supernode=True`, input dim 384
- **신규 (2026-04-28)**: `best_gat_query_supernode_qcond.pt` — `query_conditioned=True, query_supernode=True`, **input dim 768** (effective_in=in_channels*2)

### Ckpt 정보

| 항목 | 값 |
|---|---|
| 파일 | `best_gat_query_supernode_qcond.pt` (220 MB) |
| 위치 (NAS) | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_query_supernode_qcond.pt` |
| Symlink | `outputs/checkpoints/best_gat_query_supernode_qcond.pt → NAS` |
| Best epoch | 228 / 300 |
| Val recall@15 | **0.5737** |
| in_channels (yaml) | 384 |
| effective_in (실제) | **768** (query_conditioned=True 자동 ×2) |

### Smoke verification

- Config: `s04_stagewise_supernode_qcond_a0_smoke` (`configs/experiments/s04_ablation/stagewise/selector_only/supernode_qcond_a0_smoke.yaml`)
- 결과: **load_state_dict + forward pass 정상** (이전 size mismatch 해소)

### Config 주의사항

- 새 ckpt 사용 시 yaml: `query_conditioned: true` + `query_supernode: true` + `weight_path: outputs/checkpoints/best_gat_query_supernode_qcond.pt`
- in_channels=384 그대로 (model 내부 effective_in 자동 처리, gat_network.py:64-65)
- 세부 실행 이력: [EXPERIMENT_HISTORY.md SuperNode QCond GAT 학습 완료 (2026-04-28)](EXPERIMENT_HISTORY.md).

---

## SuperNode 9-cell Matrix (Ablation 2 SuperNode, 2026-04-29, 9 cells)

**용도**: 사용자 요청 (2026-04-29) — "SuperNode 의 alpha=0.0, 0.5, 1.0 일 때의 각 단계별 (Selector Only, +Basic PCST, +XiYan Filter) 점수". H6 (Concat vs SuperNode) 결정 measurement.

**근거**: planning/DECISIONS.md 2026-04-28 (H6 옵션 A SuperNode 학습 완료) + ensemble_selector.py 코드 fix.

### 명명 규칙 — `supernode_qcond_a{0,05,1}_<stage>`

- 패턴: `supernode_qcond_a<alpha_token>_<stage>.yaml`
  - alpha_token: `a0` (α=0.0), `a05` (α=0.5), `a1` (α=1.0)
  - stage: `selector_only` / `no_filter` / `glm` (final XiYan GLM)
- 폴더: `configs/experiments/s04_ablation/stagewise/{selector_only/, no_filter/, }`
- 기존 SuperNode cells (smoke fail 2026-04-26): `supernode_<score>_<stage>.yaml` (구 ckpt, 보류 상태) — 유지

### Selector params 공통

- `EnsembleSelector(weight_path=best_gat_query_supernode_qcond.pt, query_conditioned=true, query_supernode=true, encoder_type=plm, top_k=20)`
- 코드 fix 의존: `ensemble_selector.py:241-243` (SuperNode 분기 query_emb 전달, 2026-04-28 수정)

### 등재된 cells (9 신규, all 측정 완료 2026-04-29)

| 신규 ID | α | Stage | F1 |
|---------|---|-------|---:|
| `s04_stagewise_supernode_qcond_a0_selector_only` | 0.0 | Selector only | 0.3569 |
| `s04_stagewise_supernode_qcond_a05_selector_only` | 0.5 | Selector only | 0.4030 |
| `s04_stagewise_supernode_qcond_a1_selector_only` | 1.0 | Selector only | 0.3829 |
| `s04_stagewise_supernode_qcond_a0_no_filter` | 0.0 | + Basic PCST | 0.3728 |
| `s04_stagewise_supernode_qcond_a05_no_filter` | 0.5 | + Basic PCST | 0.2436 |
| `s04_stagewise_supernode_qcond_a1_no_filter` | 1.0 | + Basic PCST | 0.2295 |
| `s04_stagewise_supernode_qcond_a0_glm` | 0.0 | + XiYan GLM | **0.5476** |
| `s04_stagewise_supernode_qcond_a05_glm` | 0.5 | + XiYan GLM | **0.8341** |
| `s04_stagewise_supernode_qcond_a1_glm` | 1.0 | + XiYan GLM | **0.8368** |

### H6 결정

- **Concat 채택, SuperNode 보류** (paper main pipeline = QCond Concat F1=0.8673)
- α=0 SuperNode -0.1735 손실 발견 (vs QCond Concat α=0 0.7211 → SuperNode 0.5476)
- α=0.5/1 plateau 동등 (ΔF1 ±0.005 미달)

### Config 주의사항

- α=0.5 = neutral (GAT/Cosine 동등 결합), α=1 = Cosine only, α=0 = GAT only
- 모든 GLM cells: `provider: "glm", model_name: "zai-org/glm-4.7", max_iteration: 1, temperature: 0.0`
- ckpt input dim 768 (effective_in=in_channels*2, query_conditioned=true)
- 세부 실행 이력: [EXPERIMENT_HISTORY.md SuperNode 9-cell Matrix 측정 (Ablation 2 SuperNode, 2026-04-29)](EXPERIMENT_HISTORY.md).

---

## SuperNode + Enriched Paper Main Pipeline (2026-04-29, 2 cells)

**용도**: 사용자 요청 (2026-04-29) "Enriched + QCond-SuperNode + MST + PCST + XiYan Filter 성능". paper main pipeline (옵션 A2 Concat anchor F1=0.8673) 의 SuperNode variant 측정.

**근거**: planning/DECISIONS.md 2026-04-28 (옵션 A2) + 2026-04-29 (SuperNode 9-cell matrix) 후속.

### 명명 규칙 — `s04_pipeline_enriched_supernode_*`

- 패턴: `enriched_supernode_a05_<extractor>_glm.yaml`
  - extractor ∈ `mst_kruskal` / `mst_pcst_union`
  - α=0.5 통일 (neutral, paper main pipeline 일관성)
- 폴더: `configs/experiments/s04_ablation/pipeline/`

### Selector params

- `EnsembleSelector(weight_path=best_gat_query_supernode_qcond.pt, query_conditioned=true, query_supernode=true, alpha=0.5, top_k=20, encoder_type=plm)`
- Builder: `EnrichedHeteroGraphBuilder` (CSV + tables.json)
- Filter: `XiYanFilter(provider=glm, model=zai-org/glm-4.7)`

### 등재된 cells (2 신규)

| 신규 ID | Extractor | F1 |
|---------|-----------|---:|
| `s04_pipeline_enriched_supernode_a05_mst_kruskal_glm` | MSTKruskalExtractor | 0.8648 |
| **`s04_pipeline_enriched_supernode_a05_mst_pcst_union_glm`** ★ | MSTPCSTUnionExtractor | **0.8669** |

### vs Concat Paper Main Pipeline

| Selector | Extractor | F1 | 비고 |
|---|---|---|---|
| Concat | MST Kruskal | **0.8673** | paper main anchor |
| Concat | Union | 0.8667 | |
| SuperNode | MST Kruskal | 0.8648 | -0.0025 (noise) |
| SuperNode | Union ★ | 0.8669 | -0.0004 (noise) |

→ **4-cell plateau** — 모두 ΔF1 ±0.005 임계 미달, anchor 유지

### Config 주의사항

- weight_path: `best_gat_query_supernode_qcond.pt` (2026-04-28 학습, val recall@15=0.5737)
- 코드 fix 의존: `ensemble_selector.py:241-243` (SuperNode 분기 query_emb 전달, 2026-04-28 수정)
- 모든 GLM cells: `provider: "glm", model_name: "zai-org/glm-4.7", max_iteration: 1, temperature: 0.0`
- 세부 실행 이력: [EXPERIMENT_HISTORY.md SuperNode + Enriched Paper Main Pipeline 측정 (2026-04-29)](EXPERIMENT_HISTORY.md).

---

## H-A + H-D Ablation (2026-05-04, 13 cells)

**용도**: Alpha sweep plateau (α∈[0.3,1.0] F1≈0.86, ΔF1<0.005) 의 원인 진단 — 사용자 narrative resolution 검증.
**근거**: planning/DECISIONS.md 2026-05-04 사용자 의사결정 H-A + H-D 승인.

### 명명 규칙

- **H-A 11 cells**: `t00_enriched_ckpt_alpha_<XX>` — XX ∈ {00, 01, ..., 10} (α=0.0~1.0, 0.1 단위)
- **H-D 2 cells**: `t00_norm_<variant>` — variant ∈ {none, zscore}

### 등재 cells (13 신규)

#### H-A: Enriched ckpt + Distribution match alpha sweep (11 cells)

| α | 신규 ID | F1 | EX |
|---|---|---|---|
| 0.0 | `s04_pipeline_t00_enriched_ckpt_alpha_00` | 0.7195 | 0.2177 |
| 0.1 | `s04_pipeline_t00_enriched_ckpt_alpha_01` | 0.7820 | 0.2432 |
| 0.2 | `s04_pipeline_t00_enriched_ckpt_alpha_02` | 0.8566 | 0.3188 |
| 0.3 | `s04_pipeline_t00_enriched_ckpt_alpha_03` | 0.8634 | 0.3292 |
| 0.4 | `s04_pipeline_t00_enriched_ckpt_alpha_04` | 0.8648 | 0.3331 |
| 0.5 | `s04_pipeline_t00_enriched_ckpt_alpha_05` | 0.8637 | 0.3403 |
| 0.6 | `s04_pipeline_t00_enriched_ckpt_alpha_06` | 0.8632 | 0.3403 |
| 0.7 | `s04_pipeline_t00_enriched_ckpt_alpha_07` | 0.8625 | 0.3396 |
| 0.8 | `s04_pipeline_t00_enriched_ckpt_alpha_08` | 0.8634 | **0.3429** |
| 0.9 | `s04_pipeline_t00_enriched_ckpt_alpha_09` | 0.8642 | 0.3383 |
| 1.0 | `s04_pipeline_t00_enriched_ckpt_alpha_10` | **0.8651** | 0.3390 |

#### H-D: Score Normalization 변형 (2 cells)

| Variant | 신규 ID | F1 | EX |
|---|---|---|---|
| norm_none | `s04_pipeline_t00_norm_none` | 0.8553 | 0.3214 |
| norm_zscore | `s04_pipeline_t00_norm_zscore` | 0.8325 | 0.2881 |

### Config 주의사항

#### H-A
- weight_path: `best_gat_enriched.pt` (input dim 384, query_conditioned=False 학습 — Enriched Builder 매칭)
- query_conditioned: **false** (ckpt 학습 정합)
- 다른 모듈 t_00 동일 (Enriched Builder, MSTPCSTUnion, XiYan(GLM, num_examples=3), SQL gen)

#### H-D
- weight_path: `best_gat_qcond_nl3.pt` (t_00 default)
- query_conditioned: true
- alpha: 0.5
- score_normalization: "none" 또는 "zscore" (default "minmax")
- 코드 fix 의존: `src/modules/selectors/ensemble_selector.py:33-65, 295-318` (score_normalization 파라미터 추가)

### 결론 — 시나리오 ② 채택

- H-A 가설 부정: Distribution shift 해소가 GAT contribution 회복 못 함 (F1 plateau α∈[0.2,1.0] 유지)
- H-D 가설 부정 (간접): minmax > none > zscore, norm 자체가 plateau 원인 X
- → **paper main contribution narrative 정정**: "QCondGAT main contribution" → "4 module Co-Design + Filter dominance"
- 세부 실행 이력: [EXPERIMENT_HISTORY.md H-A Distribution Shift 검증 + H-D Score Normalization 변형 (2026-05-04)](EXPERIMENT_HISTORY.md).

---

## Wave 4 Filter Ablation (2026-05-04 → 05, 14 cells)

### 명명 규칙 — `s04_pipeline_wave4_a05_*`

| 신규 ID | Filter Variant | F1 | EX |
|---|---|---|---|
| `s04_pipeline_wave4_a05_01_adaptive_multi_agent` | AdaptiveMultiAgent | 0.8070 | 0.3279 |
| `s04_pipeline_wave4_a05_02_reflection_1iter` | ReflectionFilter (1 iter) | 0.8631 | 0.3429 |
| `s04_pipeline_wave4_a05_03_reflection_3iter` | ReflectionFilter (3 iter) | 0.8594 | 0.3344 |
| `s04_pipeline_wave4_a05_04_verifier` | VerifierFilter | 0.8662 | 0.3383 |
| `s04_pipeline_wave4_a05_05_tiered_no_tools` | TieredBidirectionalAgent (no_tools) | 0.8695 | 0.3429 |
| `s04_pipeline_wave4_a05_06_tiered_full_tools` | TieredBidirectionalAgent (full_tools) | 0.8678 | 0.3422 |
| `s04_pipeline_wave4_a05_07_adaptive_depth` | AdaptiveDepthFilter | 0.8633 | **0.3501** |
| `s04_pipeline_wave4_a05_08_tiered_verifier_stack` | StackedFilter (Tiered → Verifier) | **0.8809** | 0.3351 |
| `s04_pipeline_wave4_a05_09_tiered_retry` | TieredBidirectional + Retry K=2 | 0.8684 | 0.3377 |
| `s04_pipeline_wave4_a05_10_adaptive_retry` | AdaptiveDepth + Retry K=2 | 0.8623 | 0.3422 |
| `s04_pipeline_wave4_a05_19_symverify_xiyan_repair` | SymbolicVerifier + XiYan repair | 0.8650 | 0.3409 |
| `s04_pipeline_wave4_a05_20_symverify_reflection_repair` | SymVerify + Reflection repair | 0.8620 | 0.3396 |
| `s04_pipeline_wave4_a05_21_symverify_xiyan_detect` | SymbolicVerifier + XiYan detect | 0.8645 | 0.3370 |
| `s04_pipeline_wave4_a05_22_symverify_reflection_verifier_stacked` | SymVerify + Reflection + Verifier 3-stack | 0.8759 | 0.3364 |

### Config 주의사항

- weight_path: `best_gat_qcond_nl3.pt` (t_00 default)
- query_conditioned: true, alpha: 0.5, top_k: 20
- connectivity_extractor: MSTPCSTUnionExtractor (score_threshold=0.1)
- sql_generator: LLMSQLGenerator (provider=glm, llm_model=zai-org/glm-4.7)
- 모든 Filter agent provider=glm + temperature=0.0
- Plan reference: `planning/templates/vivid-sprouting-sunbeam.md` (F1~F5 phases)

### 결론

- **신규 최고 F1=0.8809 (a05_08 Stacked Tiered+Verifier)** — t_00 base 0.8657 대비 +0.0152
- **신규 최고 EX=0.3501 (a05_07 AdaptiveDepth)** — t_00 base 0.3377 대비 +0.0124
- **R 최고 0.9155 (a05_04 Verifier)** — but P trade-off 로 F1 plateau 내
- AdaptiveMultiAgent (a05_01) 만 outlier 실패 (-0.0587) — Skeptic over-prune
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Wave 4 Filter Ablation (2026-05-04 → 05)](EXPERIMENT_HISTORY.md).

---

## F-1 Alpha Sweep + H-G Adaptive PCST F-1 (2026-05-05, 17 cells, LLM-free)

### 명명 규칙

- **F-1 main stack** (paper main minus Filter+SQL): `s04_pipeline_t00_f1_alpha_<00~10>` (10 신규, α=0.5 기존 cell `enriched_qcond_a05_mst_pcst_union_no_filter` 재활용)
- **H-G Adaptive** (Adaptive Extractor 교체): `s04_pipeline_t00_hg_adaptive_f1_alpha_<00,02,04,05,06,08,10>` (7 신규)

### F-1 MSTPCSTUnion 11 cells

| α | 신규 ID | R | P | F1 |
|---|---|---|---|---|
| 0.0 | `s04_pipeline_t00_f1_alpha_00` | 0.7585 | 0.2047 | 0.3224 |
| 0.1 | `s04_pipeline_t00_f1_alpha_01` | 0.8535 | 0.2137 | **0.3418** |
| 0.2 | `s04_pipeline_t00_f1_alpha_02` | 0.9645 | 0.1728 | 0.2931 |
| 0.3 | `s04_pipeline_t00_f1_alpha_03` | 0.9845 | 0.1438 | 0.2509 |
| 0.4 | `s04_pipeline_t00_f1_alpha_04` | 0.9905 | 0.1320 | 0.2330 |
| 0.5 | `s04_pipeline_enriched_qcond_a05_mst_pcst_union_no_filter` (기존) | 0.9927 | 0.1268 | 0.2249 |
| 0.6 | `s04_pipeline_t00_f1_alpha_06` | 0.9939 | 0.1240 | 0.2205 |
| 0.7 | `s04_pipeline_t00_f1_alpha_07` | 0.9940 | 0.1224 | 0.2180 |
| 0.8 | `s04_pipeline_t00_f1_alpha_08` | 0.9943 | 0.1212 | 0.2161 |
| 0.9 | `s04_pipeline_t00_f1_alpha_09` | 0.9945 | 0.1208 | 0.2154 |
| 1.0 | `s04_pipeline_t00_f1_alpha_10` | **0.9947** | 0.1207 | 0.2153 |

→ **R spread = 0.2362, F1 spread = 0.1265** — DECISIONS 분기 1 (>0.05 의 4-5배)

### H-G AdaptivePCST 7 cells

| α | 신규 ID | R | P | F1 |
|---|---|---|---|---|
| 0.0 | `s04_pipeline_t00_hg_adaptive_f1_alpha_00` | 0.5074 | 0.2566 | 0.3408 |
| 0.2 | `s04_pipeline_t00_hg_adaptive_f1_alpha_02` | 0.6480 | 0.3142 | 0.4232 |
| 0.4 | `s04_pipeline_t00_hg_adaptive_f1_alpha_04` | 0.7017 | 0.3268 | 0.4459 |
| 0.5 | `s04_pipeline_t00_hg_adaptive_f1_alpha_05` | 0.7260 | 0.3315 | 0.4552 |
| 0.6 | `s04_pipeline_t00_hg_adaptive_f1_alpha_06` | 0.7500 | 0.3392 | 0.4671 |
| 0.8 | `s04_pipeline_t00_hg_adaptive_f1_alpha_08` | **0.7834** | 0.3511 | **0.4849** |
| 1.0 | `s04_pipeline_t00_hg_adaptive_f1_alpha_10` | 0.7778 | 0.3428 | 0.4759 |

→ **R spread = 0.2760, F1 spread = 0.1441** — DECISIONS 분기 1 (>0.05 의 5-6배)

### Config 주의사항

- weight_path: `best_gat_qcond_nl3.pt`
- query_conditioned: true, alpha: 위 표, top_k: 20
- score_normalization: minmax (default)
- F-1: connectivity_extractor=MSTPCSTUnionExtractor(score_threshold=0.1)
- H-G: connectivity_extractor=AdaptivePCSTExtractor(percentile=80.0, max_prize_nodes=25, base_cost=0.05, fk_cost=0.05)
- filter: NoneFilter (LLM-free)
- sql_generator: enabled=false (no EX 측정)

### 결론

- **🚨 Stage 1 Extractor MST set saturation 가설 부정** (paper main pipeline 에서)
- **✅ Stage 2 Filter precision absorption 결정적 evidence** — Filter F1 spread 20× 압축
- **§3.5 narrative 정정**: "2-stage absorption" → **"Filter dominance" single-stage main + Extractor stack-dependent**
- 세부 실행 이력: [EXPERIMENT_HISTORY.md F-1 Alpha Sweep + H-G Adaptive PCST F-1 (2026-05-05)](EXPERIMENT_HISTORY.md).

---

## Directed Top-K SuperNode GAT 학습 (V-3-ext 단계 2, 2026-05-06, 3 변형 × 300 epochs)

### 명명 규칙 — `best_gat_directed_supernode_*.pt`

| 변형 | mode | value | NAS ckpt | best val recall@15 |
|---|---|---|---|---:|
| **PRIMARY p80** | percentile | 80.0 | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80.pt` | **0.6097** |
| **BASELINE topk20** | top_k | 20 | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_topk20.pt` | 0.5839 |
| **OPTIONAL abstau07** | abs_tau | 0.7 | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_abstau07.pt` | 0.5805 |

### 명명 규칙 — `outputs/checkpoints/` symlink

- `outputs/checkpoints/best_gat_directed_supernode_p80.pt → /SSL_NAS/.../best_gat_directed_supernode_p80.pt`
- `outputs/checkpoints/best_gat_directed_supernode_topk20.pt → /SSL_NAS/.../best_gat_directed_supernode_topk20.pt`
- `outputs/checkpoints/best_gat_directed_supernode_abstau07.pt → /SSL_NAS/.../best_gat_directed_supernode_abstau07.pt`

### Config 주의사항 (training)

- 공통: `query_supernode: true, supernode_edge_direction: directed_from_sn, supernode_score_normalization: minmax, in_channels: 384, hidden_channels: 256, num_layers: 3, heads: 4, dropout: 0.1, epochs: 300, batch_size: 8, learning_rate: 0.0001, weight_decay: 0.00001, pos_weight: 100.0, val_split: 0.1, recall_k: 15, infonce_lambda: 0.5, temperature: 0.07, num_hard_negatives: 15`
- p80: `supernode_threshold_mode: percentile, supernode_threshold_value: 80.0`
- topk20: `supernode_threshold_mode: top_k, supernode_topk: 20, supernode_topk_criterion: raw`
- abstau07: `supernode_threshold_mode: abs_tau, supernode_threshold_value: 0.7`
- Builder: `EnrichedHeteroGraphBuilder` + tables_json `/SSL_NAS/peoples/khj/thesis/train/train_tables.json`
- Train data: `/SSL_NAS/peoples/khj/thesis/train/train.json` + `/SSL_NAS/peoples/khj/thesis/train/train_databases`

### 결론 — 시나리오 A 잠정 (Filter Dominance 5번째 축 candidate)

- 3 변형 모두 epoch 100~150 부터 saturation (200+ epochs 무변동 evidence)
- p80 (raw R 0.6133 → 학습 0.6097, -0.4%p): GAT 학습이 raw R 거의 회복 (가장 비슷한 노드 수 P80 의 학습 효과 거의 없음 → narrative: GAT 학습이 selector R 한계 회복 못함)
- topk20 (raw R 0.6865 → 학습 0.5839, -10.3%p): **negative result** — directed_from_sn edge 가 top_k=20 강제 모드에서 학습 disadvantage
- abstau07 (raw R 0.4857 → 학습 0.5805, +9.5%p): 학습이 raw R 능가, 가장 큰 학습 개선
- 단계 3 alpha sweep (paper main stack + 신규 ckpt × α∈{0.0~1.0}) BIRD-dev F1/EX 측정으로 시나리오 A/B/C 확정
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Directed Top-K SuperNode GAT 학습 (V-3-ext 단계 2, 2026-05-06)](EXPERIMENT_HISTORY.md).

---

## DSN Phase 1 Alpha Sweep (V-3-ext 단계 3, 2026-05-06, 9 cells, 시나리오 A 확정)

### 명명 규칙 — `s04_pipeline_dsn_phase1_<variant>_alpha_<aa>`

| α | Cell ID | F1 | EX |
|---|---|---:|---:|
| 0.0 | `s04_pipeline_dsn_phase1_p80_alpha_00` | 0.7639 | 0.2288 |
| 0.5 | `s04_pipeline_dsn_phase1_p80_alpha_05` | 0.8641 | 0.3331 |
| 1.0 | `s04_pipeline_dsn_phase1_p80_alpha_10` | 0.8648 | **0.3396** |
| 0.0 | `s04_pipeline_dsn_phase1_topk20_alpha_00` | 0.7276 | 0.2269 |
| 0.5 | `s04_pipeline_dsn_phase1_topk20_alpha_05` | 0.8645 | 0.3318 |
| 1.0 | `s04_pipeline_dsn_phase1_topk20_alpha_10` | **0.8660** | **0.3396** |
| 0.0 | `s04_pipeline_dsn_phase1_abstau07_alpha_00` | 0.7546 | 0.2484 |
| 0.5 | `s04_pipeline_dsn_phase1_abstau07_alpha_05` | 0.8648 | 0.3364 |
| 1.0 | `s04_pipeline_dsn_phase1_abstau07_alpha_10` | **0.8660** | 0.3377 |

### Config 주의사항

- weight_path: `outputs/checkpoints/best_gat_directed_supernode_<variant>.pt` (V-3-ext 단계 2 학습 ckpt 3종)
- selector: `DirectedTopKSuperNodeSelector` (V-3-ext)
- threshold_mode + threshold_value 매핑 (학습 일치):
  - p80: percentile, 80.0
  - topk20: top_k, 20
  - abstau07: abs_tau, 0.7
- 공통: top_k=20 (final select), supernode_edge_direction=directed_from_sn, score_normalization=minmax, encoder_type=plm, alpha=variable
- Extractor: MSTPCSTUnionExtractor(score_threshold=0.1)
- Filter: XiYanFilter(provider=glm, model_name=zai-org/glm-4.7, max_iteration=1)
- SQL gen: LLMSQLGenerator(provider=glm, llm_model=zai-org/glm-4.7)

### 결론

- **🎯 시나리오 A 확정** (F1 ≤ 0.870): best F1 0.8660 < 0.870, plateau 0.0019 spread (α∈{0.5, 1.0} 6 cells)
- **Filter Dominance topology-invariant 5번째 축 evidence**: graph topology + selector threshold 변경에도 plateau 동일 (직전 qcond_nl3 plateau 와 동일 패턴)
- best F1 (0.8660) ≈ t_00 base F1 (0.8657) — paper main pipeline 의 robustness 입증
- 학위 논문 Part III V-3-ext 단계 3 Phase 1 완료
- 세부 실행 이력: [EXPERIMENT_HISTORY.md DSN Phase 1 Alpha Sweep (V-3-ext 단계 3, 2026-05-06)](EXPERIMENT_HISTORY.md).

---

## Baseline Correction — qcond_nl3 best val recall@15 = 0.6061 (2026-05-06)

### 정정 record (paper main t_00 anchor ckpt)

| Ckpt path | best val recall@15 | best epoch | Final R@15 |
|---|---:|---:|---:|
| `outputs/checkpoints/best_gat_qcond_nl3.pt` (NAS symlink) | **0.6061** | **59** | 0.5958 |

이전 본 ID_MIGRATION 에서는 qcond_nl3 ckpt 의 best val recall@15 명시 부재 (paper main 의 anchor ckpt 임에도 불구). 본 entry 가 정정 record.

### Config 출처 (재확인)

- 학습 entry: `configs/training/train_gat_query_supernode_qcond.yaml` 또는 `train_gat_query_conditioned.yaml` (정확한 학습 entry log 미보존, 2026-04-23 학습 추정 기준)
- query_conditioned: true (Concat mode), bidirectional SuperNode
- num_layers: 3 (nl3 명명)

### 결론

- qcond_nl3 baseline best val R@15 = 0.6061 (이전 ~0.55 추정 부정확)
- DSN p80 (0.6097) 와 사실상 동등 (Δ=+0.0036) — graph topology 변경이 학습 saturation 한계 갱신 못함
- 세부 정정 이력: [EXPERIMENT_HISTORY.md Baseline Correction](EXPERIMENT_HISTORY.md)

---

## DSN Phase 2 + Phase 3 4-trial Mitigation Sweep (V-3-ext 단계 5, 2026-05-06 → 05-07, 3 신규 ckpt)

### 명명 규칙 — `best_gat_directed_supernode_p80_b5_*.pt`

| 변형 | AC target | LR config | NAS ckpt | Best R@15 | Best Epoch |
|---|---|---|---|---:|---|
| **Phase 2 b8** | fusion | base 1e-4 | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_b5_mitigation.pt` (113MB) | **0.6018** | ep157 |
| **Phase 3 #3** | gat_out_L_last | base 1e-4 | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_b5_phase3_directAC.pt` (113MB) | 0.5927 | ep51 |
| **Phase 3 #4** | fusion | gat 5e-4 / other 1e-4 | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_b5_phase3_layerwiseLR.pt` (113MB) | 0.5935 | ep172 |

### 명명 규칙 — `outputs/checkpoints/` symlink

- `outputs/checkpoints/best_gat_directed_supernode_p80_b5_mitigation.pt → /SSL_NAS/.../`
- `outputs/checkpoints/best_gat_directed_supernode_p80_b5_phase3_directAC.pt → /SSL_NAS/.../`
- `outputs/checkpoints/best_gat_directed_supernode_p80_b5_phase3_layerwiseLR.pt → /SSL_NAS/.../`

### Config 주의사항 (training)

- 학습 entry: `src/train_gat_s06.py` (root 가 V-3-ext options forward + AC target='gat_out_L_last' + optimizer_layer_wise_lr 추가 2026-05-06)
- 공통 V-3-ext (DSN p80 base): query_supernode=true, supernode_edge_direction='directed_from_sn', supernode_threshold_mode='percentile', supernode_threshold_value=80.0, score_normalization='minmax'
- 공통 B5 mitigation: pairnorm_mode='pairnorm', pairnorm_scale=1.0, initial_residual_alpha=0.2, jumping_knowledge='concat', dual_stream=true, num_layers=2, classifier_hidden=256
- 공통 training: epochs=300, batch_size=8, learning_rate=0.0001, weight_decay=0.00001, pos_weight=100.0, val_split=0.1, recall_k=15, loss_type='listnet', anti_collapse_weight=0.1, anti_collapse_tau_max=0.85
- 차이 차원:
  - Phase 2 b8: anti_collapse_target='fusion' (s06 B5 default)
  - Phase 3 #3: anti_collapse_target='gat_out_L_last' (forward hook 으로 main GAT path 직접 압박)
  - Phase 3 #4: optimizer_layer_wise_lr=true, gat_lr_multiplier=5.0 (HeteroConv path 5e-4 별도 LR)
- Builder: `EnrichedHeteroGraphBuilder` + tables_json `/SSL_NAS/peoples/khj/thesis/train/train_tables.json`
- Train data: `/SSL_NAS/peoples/khj/thesis/train/train.json` + `/SSL_NAS/peoples/khj/thesis/train/train_databases`

### 4-trial 비교 표 (V-3-ext 단계 2 + 단계 5 통합)

| 순위 | Variant | Best R@15 | Best Epoch | Δ vs Phase 1 | 학습 wall |
|------|---------|-----------|------------|--------------|-----------|
| **1** | **Phase 1 P80 (no mit)** | **0.6097** | ep91 | (baseline) | 7h 30min |
| 2 | Phase 3 #4 (LR x5) | 0.5935 | ep172 | -0.0162 | 11h 16min |
| 3 | Phase 3 #3 (Direct AC gat_out_L_last) | 0.5927 | ep51 | -0.0170 | 10h 15min |
| 4 | Phase 2 b8 (mit fusion) | 0.6018 | ep157 | -0.0079 | 10h 26min |

### 결론 — 시나리오 P3-A 절대 confirm + Filter Dominance 6번째 축

- **모든 mitigation variants 가 baseline (Phase 1 P80) 보다 lower** — graph topology + B5 mitigation + Direct AC + Layer-wise LR 모두 raw R 한계 갱신 X
- AC loss 0.62 일관 유지 (Phase 3 #3) → main GAT path 가 collapse 압박 처리 못함의 정량 evidence
- paper §3.5 Filter Dominance narrative 6번째 축 (training-pathology-invariant) 결정적 evidence
- Alpha sweep skip (사용자 명시) → val recall@15 evidence only, paper main F1/EX 측정 X
- 학위 논문 Part III main contribution 후보 — mechanism deep dive analyzer 위임
- 세부 실행 이력: [EXPERIMENT_HISTORY.md DSN Phase 2 + Phase 3 4-trial Mitigation Sweep (V-3-ext 단계 5, 2026-05-06 → 05-07)](EXPERIMENT_HISTORY.md).

---

## DSN Mitigation v2 3-trial Sweep (V-3-ext 단계 6, 2026-05-07 → 05-08, 3 신규 ckpt)

### 명명 규칙 — `best_gat_directed_supernode_p80_b5_mitigation_v2_*.pt`

| 변형 | 옵션 | NAS ckpt | Best R@15 | Best Epoch |
|---|---|---|---:|---|
| **v2 #1 DropMessage** | drop_message_p=0.2 | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.pt` (113MB) | 0.5974 | ep157 |
| **v2 #3 LayerNorm pre-softmax** | use_layernorm_pre_softmax=true | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_b5_mitigation_v2_layernorm.pt` (113MB) | **0.6011** ★ | ep289 |
| **v2 #2 Sum Aggregation** | aggregation_type='sum' | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_b5_mitigation_v2_sum_aggr.pt` (113MB) | 0.5761 | ep194 |

### 7-trial 통합 비교 표 (V-3-ext 단계 2 + 단계 5 + 단계 6)

| 순위 | Variant | Best R@15 | Δ vs Phase 1 | 학습 wall |
|------|---------|-----------|--------------|-----------|
| **1** | **Phase 1 P80 (no mit, baseline)** | **0.6097** | (baseline) | 7h 30min |
| 2 | Phase 2 b8 (mit fusion) | 0.6018 | -0.0079 | 10h 26min |
| 3 | **v2 #3 LayerNorm pre-softmax** | **0.6011** ★ | -0.0086 | ~21h (3 동시) |
| 4 | v2 #1 DropMessage | 0.5974 | -0.0123 | ~21h (3 동시) |
| 5 | Phase 3 #4 (LR x5) | 0.5935 | -0.0162 | 11h 16min |
| 6 | Phase 3 #3 (Direct AC) | 0.5927 | -0.0170 | 10h 15min |
| 7 | v2 #2 Sum Aggregation | 0.5761 | -0.0336 | ~21h (3 동시) |

### Config 주의사항 (training)

- 학습 entry: `src/train_gat_s06.py` (Mitigation v2 옵션 forward 추가됨 by root 2026-05-07)
- 공통 V-3-ext: query_supernode=true, supernode_edge_direction='directed_from_sn', supernode_threshold_mode='percentile', supernode_threshold_value=80.0, score_normalization='minmax'
- 공통 B5 mitigation: pairnorm_mode='pairnorm', initial_residual_alpha=0.2, jumping_knowledge='concat', dual_stream=true, num_layers=2
- 공통 training: epochs=300, batch_size=8, learning_rate=0.0001, weight_decay=0.00001, pos_weight=100.0, val_split=0.1, recall_k=15, loss_type='listnet', anti_collapse_weight=0.1, anti_collapse_target='fusion'
- 변경 차원 (Mitigation v2):
  - v2 #1: drop_message_p=0.2
  - v2 #3: use_layernorm_pre_softmax=true
  - v2 #2: aggregation_type='sum'

### 결론 — 시나리오 V2-A 절대 confirm + Filter Dominance 6번째 축 7-trial evidence

- 7-trial × 4 mitigation 카테고리 모두 raw R 한계 갱신 X
- v2 #3 LayerNorm 가 mitigation variants 중 best (0.6011) — mech(ii) partial mitigation but baseline 미달
- v2 #2 Sum Aggregation 압도적 underperform (-0.0336) — mech(i) 직접 evidence
- 시나리오 V2-A (모든 candidate 가 baseline 미달) 절대 confirm
- paper §3.5 Filter Dominance 6번째 축 (training-pathology-invariant) **7-trial evidence 결정적**
- 세부 실행 이력: [EXPERIMENT_HISTORY.md DSN Mitigation v2 3-trial Sweep (V-3-ext 단계 6, 2026-05-07 → 05-08)](EXPERIMENT_HISTORY.md).

---

## Filter Module Confirmation Sweep v2 — 9-cell with Evidence Forward (2026-05-13, 9 cell, EX 측정 활성, evidence fix)

### 명명 규칙

- C0 (anchor _sql 변형, 신규): `s04_pipeline_enriched_qcond_filter_sweep_c0_xiyan_glm_sql` (anchor `s04_pipeline_enriched_qcond_a05_mst_kruskal_glm` 의 sql_generator 활성 변형)
- C1~C8: `s04_pipeline_enriched_qcond_filter_sweep_c{N}_<filter_short>_glm`

### v2 최종 결과 (F1 순, evidence fix)

| 순위 | Cell | experiment_name | Filter | Best F1 | EX | EX Δ vs v1 |
|---|---|---|---|---:|---:|---:|
| **1** | **C4** | `..._filter_sweep_c4_stacked_glm` | Stacked (Refl+Verif) | **0.8704** ⭐ | 0.5267 | +0.1851 |
| 2 | C7 | `..._filter_sweep_c7_bidirectional_glm` | TieredBidirectional | 0.8671 | **0.5287** ⭐ | +0.1858 |
| 3 | **C0** | `..._filter_sweep_c0_xiyan_glm_sql` (anchor) | XiYan | **0.8651** | 0.5202 | **+0.1806** |
| 4 | C1 | `..._filter_sweep_c1_reflection_glm` | Reflection | 0.8650 | 0.5222 | +0.1754 |
| 4 | C5 | `..._filter_sweep_c5_symverify_glm` | SymbolicVerifier | 0.8650 | 0.5222 | +0.1832 |
| 6 | C2 | `..._filter_sweep_c2_verifier_glm` | Verifier (best R=0.9163) | 0.8633 | 0.5267 | +0.1916 (biggest) |
| 7 | C6 | `..._filter_sweep_c6_adaptive_depth_glm` | AdaptiveDepth | 0.8632 | 0.5248 | +0.1826 |
| 8 | C3 | `..._filter_sweep_c3_adaptive_multi_agent_glm` | AdaptiveMultiAgent ⚠️ | 0.8041 | 0.5189 | +0.1812 |
| 9 | C8 | `..._filter_sweep_c8_no_filter` | None (baseline) | 0.2250 (R=0.9927/P=0.1269) | 0.5156 | +0.1721 |

### v1 vs v2 사실상 동일 experiment_name 사용, 결과만 evidence fix 후 갱신

- v1 (no evidence): `outputs/experiments/s04_ablation/pipeline/filter_sweep_v1_no_evidence/` archive 보존
- v2 (evidence fix, main): `outputs/experiments/s04_ablation/pipeline/filter_sweep/` (현재)
- Code fix: `src/prompts/sql_generator.md` + `src/modules/generators/sql_generator.py` + `src/pipeline/schema_linking.py` + `src/main.py`

### Stack 공통

- Builder: EnrichedHeteroGraphBuilder
- Selector: EnsembleSelector + qcond_nl3.pt + α=0.5 (query_conditioned=true)
- Extractor: MSTKruskalExtractor (score_threshold=0.1)
- LLM (C0~C7): GLM 4.7 (provider="glm", model_name="zai-org/glm-4.7", T=0.0)
- SQL Generator (C0~C8): LLMSQLGenerator + GLM 4.7
- C8: filter.name="None" (LLM-free filter, sql_gen 만)

### 시나리오: Filter-Modest + 단일 C3 outlier

- C3 outlier 제외 7 LLM filter F1 spread = **0.0116** (Filter-Modest)
- 7-cell 평균 F1 = 0.8668 ± 0.0040
- Anchor (C0) cost-effective default 유지 권장

세부 실행 이력: [EXPERIMENT_HISTORY.md Filter Module Confirmation Sweep](EXPERIMENT_HISTORY.md)

---

## SGBE Phase 3-5 — Score-Gated Batch Extractive Filter (2026-05-12, 13 sub-experiment — 🚀 Phase 3 launch active)

### 명명 규칙 — `s04_pipeline_sgbe_{calibration_cell_X_keepK_dropD, final_glm, step_step{0_only, 01_only, full}}`

본 chain 의 13 sub-experiment (9 calibration cells + 1 final + 3 step ablation) 가 모두 `s04_pipeline_sgbe_*` prefix. Sweep script 가 base yaml 을 cp + sed override 로 temp yaml generate (configs/experiments/s04_ablation/pipeline/sgbe/{calibration,step_ablation}/_tmp/...).

### 등재 예정 (13 sub-experiment, Launch 보류)

| Phase | Sub-experiment | Filter params | Status |
|---|---|---|---|
| 3 (9-cell) | `sgbe_calibration_cell_1_keep50_drop20` ~ `cell_9_keep60_drop30` | theta_keep ∈ {0.50, 0.55, 0.60} × theta_drop ∈ {0.20, 0.25, 0.30} + `step_mode="step_0+1"` + `score_collapse_threshold=0.05` | 🚀 active (21:44 KST) |
| 4 | `sgbe_final_glm` | best θ (placeholder default 0.55/0.25) + `step_mode="step_0+1+2"` + SQL generator | ⏸ Phase 3 후 |
| 5 | `sgbe_ablation_step_0` | step_mode=`step_0` + best θ | ⏸ Phase 4 후 |
| 5 | `sgbe_ablation_step_0p1` | step_mode=`step_0+1` + best θ (LLM 없음) | ⏸ Phase 4 후 |
| 5 | `sgbe_ablation_step_0p1p2` | step_mode=`step_0+1+2` + best θ (= sgbe_final) | ⏸ Phase 4 후 |

### Stack 정합

- Builder: EnrichedHeteroGraphBuilder
- Selector: EnsembleSelector + qcond_nl3.pt + α=0.5 (QCond Concat)
- Extractor: MSTKruskalExtractor (score_threshold=0.1)
- Filter: ScoreGatedBatchExtractiveFilter — Phase 별 params 차이
- LLM: GLM 4.7 (paper main backbone)
- Anchor 정합: `s04_pipeline_enriched_qcond_a05_mst_kruskal_glm` (F1=0.8673)

### Module:filters prerequisite

- `skip_llm: bool = False` — Phase 3 θ calibration (LLM 없음)
- `step_mode: str = 'full'` ∈ {'step0_only', 'step01_only', 'full'} — Phase 5 step ablation

### 후속 위임

- **module:filters**: skip_llm + step_mode option 추가
- **root (option 후)**: `bash scripts/run_sgbe_calibration.sh` + `bash scripts/run_sgbe_final_ablation.sh` 순차
- **analyzer**: `notebooks/analysis_results/sgbe_filter_results.md` 신규
- **planner**: Filter Dominance 7번째 axis + 8번째 axis narrative

세부 실행 이력: [EXPERIMENT_HISTORY.md SGBE Phase 3-5 (2026-05-12)](EXPERIMENT_HISTORY.md)

---

## DSN Mitigation V5 Tier 1+2 4-Direction 학습 (V-3-ext 단계 9, 2026-05-12, 5 신규 ckpt 예정 — 🚧 코드 준비 + Launch 보류)

### 명명 규칙 — `best_gat_directed_supernode_p80_v5{a_gate, b_gcnii_L{2,4,6}, c_aero_full}.pt`

V-3-ext 단계 9 의 architectural intervention 4-direction. V4 의 단축 명명 (`p80` 직접 suffix) 그대로 + V5 suffix.

| 변형 | gat_layer_type / 옵션 | NAS ckpt (예정) | Status |
|---|---|---|---|
| **V5-A GATE** | `gate` (att + att_self 분리, Mustafa & Burkholz 2024) | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_v5a_gate.pt` | Launch 보류 |
| **V5-B GCNII L=2** | `gcnii`, `gcnii_beta_lambda=0.5` (Chen 2020 + Peng 2024) | `best_gat_directed_supernode_p80_v5b_gcnii_L2.pt` | Launch 보류 |
| **V5-B GCNII L=4** | gcnii + num_layers=4 | `best_gat_directed_supernode_p80_v5b_gcnii_L4.pt` | Launch 보류 |
| **V5-B GCNII L=6** | gcnii + num_layers=6 | `best_gat_directed_supernode_p80_v5b_gcnii_L6.pt` | Launch 보류 |
| **V5-C Full AERO** | `aero_full` + `aero_hop_attention=true`, JK='none' (Lee 2023 full) | `best_gat_directed_supernode_p80_v5c_aero_full.pt` | Launch 보류 |

### Stack 정합

- Builder: EnrichedHeteroGraphBuilder (train_enriched_plm_graphs.pt 재사용)
- Selector: SchemaHeteroGATv2 + V-3-ext (query_supernode=true, directed_from_sn, percentile=80) + B5 mitigation (PN+IR α=0.2 + JK=concat / V5-C 만 JK=none + Dual-Stream + AC=0.1 + ListNet)
- 차별점: `gat_layer_type ∈ {gate, gcnii, aero_full}` — V4 (lngin/softplus) 과 별개의 architectural axis 4 direction
  - V5-A: Conservation Law 수정 (task-irrelevant aggregation switch-off)
  - V5-B: Trainability (Identity Mapping β_l + Initial Residual α)
  - V5-C: V4-B + Node-Adaptive Hop Attention (V4-B H10.1c 직접 표적)

### 15-trial mitigation 매트릭스 (V5 결과 합산 후 갱신 예정)

(현재 10-trial — V4 결과 까지 [위 V4 entry](#dsn-mitigation-v4-architectural-intervention-학습-v-3-ext-단계-8-2026-05-11--05-12-2-신규-ckpt----시나리오-v4-combo-null-확정) 참조. V5 5 ckpt 종료 시 15-trial 매트릭스로 확장.)

### 후속 위임

- **module:selectors (또는 신규 module:models)**: V5-A/B/C 클래스 + Hop Attention forward code review → launch 결정
- **analyzer (launch 후)**: `notebooks/analysis_results/dsn_mitigation_v5_4dir.md` 신규 (15-trial 매트릭스 + Layer 1/2/3 evidence)
- **analyzer (V5-D-1, 별도 chain)**: PLM Lower Bound 진단
- **planner**: narrative pivot 결정

세부 실행 이력: [EXPERIMENT_HISTORY.md DSN Mitigation V5 Tier 1+2 4-Direction 학습 (V-3-ext 단계 9, 2026-05-12)](EXPERIMENT_HISTORY.md)

---

## DSN Mitigation V4 Architectural Intervention 학습 (V-3-ext 단계 8, 2026-05-11 → 05-12, 2 신규 ckpt — 🎯 시나리오 V4-Combo-Null 확정)

### 명명 규칙 — `best_gat_directed_supernode_p80_v4{a_lngin_combo,b_aero}.pt`

V-3-ext 단계 8 의 architectural intervention. 직전 단계 7 v3 GIN 의 명명 (`p80_b5_mitigation_v3_gin`) 과 분기 — V4 는 `p80` 직접 suffix (B5 mitigation 통합 전제 + V4 layer 자체가 architectural 변경) 의 짧은 형식.

| 변형 | 옵션 | NAS ckpt | Best R@15 | Best Epoch |
|---|---|---|---:|---:|
| **V4-A LN+GIN Combo** | gat_layer_type='lngin' (Pre-softmax LN + GIN MLP) | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_v4a_lngin_combo.pt` (**257MB**) | **0.5929** | **ep259** |
| **V4-B AERO Softplus** | gat_layer_type='softplus' + softplus_symmetric_norm=true | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_v4b_aero.pt` (**113MB**) | **0.5951** | **ep58** |

### Stack 정합

- Builder: EnrichedHeteroGraphBuilder (train_enriched_plm_graphs.pt 재사용)
- Selector: SchemaHeteroGATv2 + V-3-ext (query_supernode=true, directed_from_sn, percentile=80) + B5 mitigation (PN+IR α=0.2 + JK=concat + Dual-Stream + L=2 + AC=0.1 + ListNet)
- 차별점: `gat_layer_type ∈ {lngin, softplus}` — directly attacks mech(ii-b) softmax × weighted-mean propagation combo

### 10-trial mitigation 통합 (최종, V4 결과 반영)

| 순위 | Variant | Best R@15 | Δ vs Phase 1 | wall |
|------|---------|-----------|--------------|------|
| **1** | **Phase 1 P80 (no mit)** | **0.6097** | (baseline) | 7h 30min |
| 2 | Phase 2 b8 (B5 mit fusion) | 0.6018 | -0.0079 | ~14h |
| 3 | v2 #3 LayerNorm pre-softmax | 0.6011 | -0.0086 | ~14h |
| 4 | v2 #1 DropMessage | 0.5974 | -0.0123 | ~14h |
| 5 | v3 #1 GIN-style aggregation | 0.5954 | -0.0143 | ~11h 39min |
| **6** | **🆕 V4-B AERO Softplus** | **0.5951** | **-0.0146** | **9h 38min** |
| 7 | Phase 3 #4 (LR x5) | 0.5935 | -0.0162 | ~10h |
| 8 | Phase 3 #3 (Direct AC) | 0.5927 | -0.0170 | ~10h |
| **9** | **🆕 V4-A LN+GIN Combo** | **0.5929** | **-0.0168** | **10h 47min** |
| 10 | v2 #2 Sum Aggregation | 0.5761 | -0.0336 | ~14h |

🎯 **V4-Combo-Null 확정** — 10-trial 모두 baseline 0.6097 미달. mech(ii-b) DOMINANT 5/5 absolute confirm + Filter Dominance 6번째 축 (training-pathology-invariant) narrative 결정적 강화.

세부 실행 이력: [EXPERIMENT_HISTORY.md DSN Mitigation V4 Architectural Intervention 학습 (V-3-ext 단계 8, 2026-05-11)](EXPERIMENT_HISTORY.md)

---

## DSN Mitigation v3 #1 GIN-style aggregation 학습 (V-3-ext 단계 7, 2026-05-08 → 05-09, 1 신규 ckpt)

### 명명 규칙 — `best_gat_directed_supernode_p80_b5_mitigation_v3_gin.pt`

| 변형 | 옵션 | NAS ckpt | Best R@15 | Best Epoch |
|---|---|---|---:|---|
| **v3 #1 GIN** | aggregation_type='gin' | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_b5_mitigation_v3_gin.pt` (140MB) | **0.5954** | ep246 |

### 8-trial 통합 비교 표

| 순위 | Variant | Best R@15 | Δ vs Phase 1 | 학습 wall |
|------|---------|-----------|--------------|-----------|
| **1** | **Phase 1 P80 (no mit)** | **0.6097** | (baseline) | 7h 30min |
| 2 | Phase 2 b8 (mit fusion) | 0.6018 | -0.0079 | 10h 26min |
| 3 | v2 #3 LayerNorm pre-softmax | 0.6011 | -0.0086 | ~21h (3 동시) |
| 4 | v2 #1 DropMessage | 0.5974 | -0.0123 | ~21h (3 동시) |
| **5** | **🆕 v3 #1 GIN-style aggregation** | **0.5954** | -0.0143 | ~11h 39min |
| 6 | Phase 3 #4 (LR x5) | 0.5935 | -0.0162 | 11h 16min |
| 7 | Phase 3 #3 (Direct AC) | 0.5927 | -0.0170 | 10h 15min |
| 8 | v2 #2 Sum Aggregation | 0.5761 | -0.0336 | ~21h (3 동시) |

### Config 주의사항 (training)

- 학습 entry: `src/train_gat_s06.py` (aggregation_type='gin' forward 추가됨)
- 공통 V-3-ext + B5 mitigation: Phase 2 b8 와 동일
- 변경 차원: aggregation_type='gin' — GATv2Conv → GINConv 교체 (HeteroConv `aggr='mean'` fix + 18 inner GINConvs)
- attention 자체 부재 → mech(ii-a) 측정 X, mech(ii-b) propagation pathology 직접 검증
- Builder + Train data: Phase 2/3 와 동일

### 결론 — 시나리오 V3-A 1차 confirm + Filter Dominance 6번째 축 8-trial evidence

- 8-trial × 5 mitigation 카테고리 모두 raw R 한계 갱신 X
- GIN 가 mit variants 5위 — mech(ii-a) 부재해도 ceiling 유사 → mech(ii-b) aggregation family limitation 강화
- v2 #3 LayerNorm (mech(ii-a) partial mitigation) > GIN (mech(ii-b) 직접) → mech(ii-a) 우위 잠정
- analyzer Phase 4 deep dive 후 mech(ii-a)/(ii-b) sub-mechanism 정식 확정
- paper §3.5 Filter Dominance 6번째 축 (training-pathology-invariant) **8-trial evidence 누적**
- 세부 실행 이력: [EXPERIMENT_HISTORY.md DSN Mitigation v3 #1 GIN-style aggregation 학습 (V-3-ext 단계 7, 2026-05-08 → 05-09)](EXPERIMENT_HISTORY.md).
