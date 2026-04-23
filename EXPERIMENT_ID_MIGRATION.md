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
