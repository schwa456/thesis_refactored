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
│   └── a08_bayesian_opt/
├── s04_gat_qcond_projector/     Query-Conditioned Projector GAT
├── s05_gat_direct/              DirectGATSelector (BCE-only)
│   └── a01_full_pipeline/
└── abl/                         Ablation studies
    ├── a01_2x2x2_selector_extractor_filter/
    ├── a02_alpha_sweep/
    ├── a03_direct_per_step/
    ├── a04_direct_binary_steiner_sweep/
    └── a05_filter_agentic/       Filter module agentic refinement (F1-F5)
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

### s04_gat_qcond_projector/

| 신규 ID | 기존 |
|---------|------|
| `s04_01_qcond_a085_xiyan` | `experiment_qcond_idea24_xiyan` (Q1) |
| `s04_02_supernode_a070_xiyan` | `experiment_supernode_idea24_xiyan` (Q2) |
| `s04_03_supernode_a085_xiyan` | `experiment_supernode_idea24_a085_xiyan` (Q3) |
| `s04_04_qcond_a0_xiyan` | `experiment_qcond_idea24_a0_xiyan` (Q4) |
| `s04_05_supernode_a0_xiyan` | `experiment_supernode_idea24_a0_xiyan` (Q5) |

### s05_gat_direct/a01_full_pipeline/

| 신규 ID | 기존 |
|---------|------|
| `s05_a01_01_qcond_direct_xiyan` | `experiment_qcond_direct_idea24_xiyan` (Q6) |
| `s05_a01_02_supernode_direct_xiyan` | `experiment_supernode_direct_idea24_xiyan` (Q7) |

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

### abl/a05_filter_agentic/ (Phase D — pre-registered 2026-04-14, not yet run)

Filter 모듈 고도화 (plan: `/home/hyeonjin/.claude/plans/vivid-sprouting-sunbeam.md`).
Anchor: a03_17 components (SuperNode Direct + Fixed PCST). 외부 baseline 없음.

| 신규 ID | Filter 모듈 | 근거 축 |
|---------|------------|---------|
| `a05_01_adaptive_multi_agent` | AdaptiveMultiAgentFilter (existing) | Multi-agent baseline |
| `a05_02_reflection_1iter` | ReflectionFilter (F1, 1 iter) | Self-Refine (NeurIPS'23) |
| `a05_03_reflection_3iter` | ReflectionFilter (F1, 3 iter) | Iteration depth |
| `a05_04_verifier` | VerifierFilter (F2) | CHESS Unit Tester (ICLR'25) |
| `a05_05_tiered_no_tools` | TieredBidirectionalAgent (F3, no tools) | Ablation vs a05_06 |
| `a05_06_tiered_full_tools` | TieredBidirectionalAgent (F3, full tools) | ★ 핵심 기여 |
| `a05_07_adaptive_depth` | AdaptiveDepthFilter (F4) | Uncertainty routing |
| `a05_08_tiered_verifier_stack` | Stacked(F3→F2) | 상한 |
| `a05_09_tiered_retry` | F3 + pipeline F5 (K=2) | Extractor reverse feedback |
| `a05_10_adaptive_retry` | F4 + F5 (K=2) | Selective retry |
| `a05_11_tiered_gpt4omini` | F3, GPT-4o-mini backbone | Backbone 민감도 |
| `a05_12_adaptive_retry_gpt4omini` | F4+F5, GPT-4o-mini | Backbone 민감도 |

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
