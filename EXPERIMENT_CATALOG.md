# Experiment Catalog

각 실험의 Selector / Extractor / Filter 모듈과 하이퍼파라미터 정리.
ID 체계 및 폴더 구조: [`EXPERIMENT_ID_MIGRATION.md`](EXPERIMENT_ID_MIGRATION.md).

## 공통 모듈 기본값

**NLQ Encoder** (거의 모든 실험 공통):
- `LocalPLMEncoder` — `sentence-transformers/all-MiniLM-L6-v2` (384-dim)

**Post-processing**: `auto_join_keys: true` (미선택 FK 노드의 양끝 컬럼 자동 포함)

**SQL Generator**: 모든 실험에서 `enabled: false` (Schema Linking만 평가)

**PCST 공통 cost 기본값** (AdaptivePCST/SteinerBackbone/EdgePrize 계열):
`base_cost=0.05, belongs_to_cost=0.01, fk_cost=0.05, macro_cost=0.5, percentile=80.0, min/max_prize_nodes=3/25, node_threshold=0.0`

**Filter 공통** (XiYan):
`model_name=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, max_iteration=1, temperature=0.0`

---

## abl — Ablation Studies

### abl/a01_2x2x2_selector_extractor_filter
*Phase C: Selector × Extractor × Filter 2×2×2*

#### `abl_a01_05_cos_basic_xiyan`

- **Seed Selector**: `VectorOnlySelector` — `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `PCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `node_threshold`=0.1
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `abl_a01_06_ens_basic_xiyan`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `PCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `node_threshold`=0.1
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `abl_a01_07_cos_adaptive_xiyan`

- **Seed Selector**: `VectorOnlySelector` — `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True


### abl/a02_alpha_sweep
*Ensemble α 값 sweep (0.85/0.75/0.70)*

#### `abl_a02_02_alpha075`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.75, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `abl_a02_03_alpha070`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.7, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True


### abl/a03_direct_per_step
*Direct Variant per-step ablation (6-11, 6-12)*

#### `abl_a03_01_qcond_selector_only`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_conditioned_direct.pt, `query_conditioned`=True, `query_supernode`=False, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `None` — -
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=False

#### `abl_a03_02_qcond_selector_extractor`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_conditioned_direct.pt, `query_conditioned`=True, `query_supernode`=False, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `abl_a03_03_supernode_selector_only`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode_direct.pt, `query_conditioned`=False, `query_supernode`=True, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `None` — -
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=False

#### `abl_a03_04_supernode_selector_extractor`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode_direct.pt, `query_conditioned`=False, `query_supernode`=True, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `abl_a03_05_qcond_binary_selector_only`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_conditioned_direct.pt, `query_conditioned`=True, `query_supernode`=False, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `None` — -
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=False

#### `abl_a03_06_qcond_binary_selector_extractor`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_conditioned_direct.pt, `query_conditioned`=True, `query_supernode`=False, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `abl_a03_07_qcond_binary_steiner`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_conditioned_direct.pt, `query_conditioned`=True, `query_supernode`=False, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `SteinerBackbonePCSTExtractor` — `backbone_bonus`=0.5, `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `abl_a03_08_qcond_binary_full`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_conditioned_direct.pt, `query_conditioned`=True, `query_supernode`=False, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `abl_a03_09_supernode_binary_selector_only`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode_direct.pt, `query_conditioned`=False, `query_supernode`=True, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `None` — -
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=False

#### `abl_a03_10_supernode_binary_selector_extractor`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode_direct.pt, `query_conditioned`=False, `query_supernode`=True, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `abl_a03_11_supernode_binary_steiner`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode_direct.pt, `query_conditioned`=False, `query_supernode`=True, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `SteinerBackbonePCSTExtractor` — `backbone_bonus`=0.5, `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `abl_a03_12_supernode_binary_full`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode_direct.pt, `query_conditioned`=False, `query_supernode`=True, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `abl_a03_13_qcond_binary_fixed`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_conditioned_direct.pt, `query_conditioned`=True, `query_supernode`=False, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `PCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `abl_a03_14_qcond_binary_fixed_xiyan`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_conditioned_direct.pt, `query_conditioned`=True, `query_supernode`=False, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `PCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `abl_a03_15_qcond_binary_steiner_xiyan`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_conditioned_direct.pt, `query_conditioned`=True, `query_supernode`=False, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `SteinerBackbonePCSTExtractor` — `backbone_bonus`=0.5, `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `abl_a03_16_supernode_binary_fixed`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode_direct.pt, `query_conditioned`=False, `query_supernode`=True, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `PCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `abl_a03_17_supernode_binary_fixed_xiyan`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode_direct.pt, `query_conditioned`=False, `query_supernode`=True, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `PCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `abl_a03_18_supernode_binary_steiner_xiyan`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode_direct.pt, `query_conditioned`=False, `query_supernode`=True, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `SteinerBackbonePCSTExtractor` — `backbone_bonus`=0.5, `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True


### abl/a04_direct_binary_steiner_sweep
*Direct Binary threshold × Steiner + XiYan (6-14)*

#### `abl_a04_01_supernode_t005_steiner_xiyan`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode_direct.pt, `query_conditioned`=False, `query_supernode`=True, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True, `threshold`=0.05
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `SteinerBackbonePCSTExtractor` — `backbone_bonus`=0.5, `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `abl_a04_02_supernode_t010_steiner_xiyan`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode_direct.pt, `query_conditioned`=False, `query_supernode`=True, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True, `threshold`=0.1
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `SteinerBackbonePCSTExtractor` — `backbone_bonus`=0.5, `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `abl_a04_03_supernode_t015_steiner_xiyan`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode_direct.pt, `query_conditioned`=False, `query_supernode`=True, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True, `threshold`=0.15
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `SteinerBackbonePCSTExtractor` — `backbone_bonus`=0.5, `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `abl_a04_04_supernode_t020_steiner_xiyan`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode_direct.pt, `query_conditioned`=False, `query_supernode`=True, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256, `apply_threshold`=True, `threshold`=0.2
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `SteinerBackbonePCSTExtractor` — `backbone_bonus`=0.5, `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True


## s01 — VectorOnly Selector (Cosine Only)

### s01_vector_only/a01_basic_pcst
*Fixed-cost PCST*

#### `s01_a01_01_basic_pcst`

- **Seed Selector**: `GATClassifierSelector` — `hidden_dim`=256, `threshold`=0.5, `weight_path`=outputs/checkpoints/best_gat_model.pt
- **Connectivity Extractor**: `PCSTExtractor` — `base_cost`=1.0, `belongs_to_cost`=0.01, `node_threshold`=0.15

#### `s01_a01_02_raw_pcst_baseline`

- **Seed Selector**: `VectorOnlySelector` — `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `PCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `node_threshold`=0.1
- **Filter**: `None`


### s01_vector_only/a02_adaptive_pcst
*AdaptivePCST (P80 threshold)*

#### `s01_a02_01_adaptive_pcst`

- **Seed Selector**: `VectorOnlySelector` — `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`


### s01_vector_only/a03_pcst_variants
*Dynamic / Uncertainty PCST 변주*

#### `s01_a03_01_dynamic_pcst`

- **Seed Selector**: `GATClassifierSelector` — `hidden_dim`=256, `threshold`=0.5, `weight_path`=outputs/checkpoints/best_gat_model.pt
- **Connectivity Extractor**: `DynamicPCSTExtractor` — `base_cost`=1.0, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `hub_discount`=0.2, `node_threshold`=0.15

#### `s01_a03_02_uncertainty_pcst`

- **Seed Selector**: `GATClassifierSelector` — `hidden_dim`=256, `threshold`=0.5, `weight_path`=outputs/checkpoints/best_gat_model.pt
- **Connectivity Extractor**: `UncertaintyPCSTExtractor` — `base_cost`=1.0, `belongs_to_cost`=0.01, `node_threshold`=0.15, `alpha`=2.0, `uncertainty_margin`=0.05

#### `s01_a03_03_dynamic_uncertainty_pcst`

- **Seed Selector**: `GATClassifierSelector` — `hidden_dim`=256, `threshold`=0.5, `weight_path`=outputs/checkpoints/best_gat_model.pt
- **Connectivity Extractor**: `DynamicUncertaintyPCSTExtractor` — `base_cost`=1.0, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `hub_discount`=0.2, `node_threshold`=0.15, `alpha`=2.0, `uncertainty_margin`=0.05


## s02 — GATClassifier (Early GAT v1)

#### `s02_01_gat_classifier`

- **Seed Selector**: `GATClassifierSelector` — `hidden_dim`=256, `threshold`=0.5, `weight_path`=outputs/checkpoints/mlp_classifier_with_gat_train_best_re...
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `None` — -
- **Filter**: `None`

#### `s02_02_gat_classifier_multi_agent`

- **Seed Selector**: `GATClassifierSelector` — `hidden_dim`=256, `threshold`=0.5, `weight_path`=outputs/checkpoints/mlp_classifier_with_gat_train_best_re...
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `GATAwarePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `node_threshold`=0.1
- **Filter**: `AdaptiveMultiAgentFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `uncertainty_threshold`=0.4, `temperature`=0.0

#### `s02_04_gat_multi_agent`

- **Seed Selector**: `VectorOnlySelector` — `top_k`=10000
- **Projection**: `enabled=True`, `hidden_channels`=256, `num_layers`=3, `heads`=4, `checkpoint_path`=./outputs/checkpoints/best_gat_model.pt
- **Connectivity Extractor**: `GATAwarePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `node_threshold`=0.1
- **Filter**: `AdaptiveMultiAgentFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `uncertainty_threshold`=0.4, `temperature`=0.0


## s03 — Ensemble (Projector GAT + Cosine, α-weighted)

### s03_gat_ensemble/a01_basic_pcst
*Fixed-cost PCST + Ensemble*

#### `s03_a01_01_ensemble_basic`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `PCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `node_threshold`=0.1
- **Filter**: `None`


### s03_gat_ensemble/a02_adaptive_pcst
*AdaptivePCST + Ensemble (+ Filter)*

#### `s03_a02_01_combined`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `s03_a02_02_single_filter`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `SingleAgentFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `s03_a02_03_xiyan_filter`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True


### s03_gat_ensemble/a03_product_cost
*ProductCostPCST (Idea 2) — edge cost를 노드 점수의 곱으로*

#### `s03_a03_01_product_cost`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ProductCostPCSTExtractor` — `bt_weight`=0.1, `fk_weight`=0.2, `macro_weight`=0.5, `min_cost`=0.0001, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `s03_a03_02_product_cost_xiyan`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ProductCostPCSTExtractor` — `bt_weight`=0.1, `fk_weight`=0.2, `macro_weight`=0.5, `min_cost`=0.0001, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True


### s03_gat_ensemble/a04_steiner_backbone
*SteinerBackbonePCST (Idea 3) — Steiner tree 2-근사 + PCST 확장*

#### `s03_a04_01_steiner`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `SteinerBackbonePCSTExtractor` — `backbone_bonus`=0.5, `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `s03_a04_02_steiner_xiyan`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `SteinerBackbonePCSTExtractor` — `backbone_bonus`=0.5, `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True


### s03_gat_ensemble/a05_component_aware
*ComponentAwareAdaptivePCST (Idea 4) — CC별 독립 threshold*

#### `s03_a05_01_component_aware`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ComponentAwareAdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True


### s03_gat_ensemble/a06_component_product
*ComponentAware + ProductCost (Idea 2+4, 1+2+4)*

#### `s03_a06_01_product_component`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ComponentAwareProductCostPCSTExtractor` — `bt_weight`=0.1, `fk_weight`=0.2, `macro_weight`=0.5, `min_cost`=0.0001, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `s03_a06_02_product_component_xiyan`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ComponentAwareProductCostPCSTExtractor` — `bt_weight`=0.1, `fk_weight`=0.2, `macro_weight`=0.5, `min_cost`=0.0001, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `s03_a06_03_idea124_combined`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.75, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ComponentAwareProductCostPCSTExtractor` — `bt_weight`=0.1, `fk_weight`=0.2, `macro_weight`=0.5, `min_cost`=0.0001, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `s03_a06_04_idea124_combined_xiyan`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.75, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ComponentAwareProductCostPCSTExtractor` — `bt_weight`=0.1, `fk_weight`=0.2, `macro_weight`=0.5, `min_cost`=0.0001, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True


### s03_gat_ensemble/a07_enriched_triplet
*Enriched/Triplet GraphBuilder*

#### `s03_a07_01_enriched_gat`

- **Graph Builder**: `EnrichedHeteroGraphBuilder` — `include_views`=False, `run_leiden_clustering`=True, `tables_json_path`=data/raw/BIRD_dev/dev_tables.json
- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_enriched.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `s03_a07_02_edge_prize`

- **Graph Builder**: `TripletGraphBuilder` — `include_views`=False, `run_leiden_clustering`=True, `tables_json_path`=data/raw/BIRD_dev/dev_tables.json, `triplet_path`=data/processed/triplet_relations.json
- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_enriched.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `EdgePrizePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0, `topk_e`=5, `edge_cost`=0.05
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True


### s03_gat_ensemble/a08_bayesian_opt
*Bayesian-optimized PCST cost*

#### `s03_a08_01_bo_fixed_cost`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.195, `fk_cost`=0.346, `macro_cost`=0.044, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `s03_a08_02_bo_score_driven`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ScoreDrivenPCSTExtractor` — `belongs_to_weight`=1.955, `fk_weight`=2.779, `macro_weight`=3.439, `epsilon`=0.009, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `s03_a08_03_score_driven_manual`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ScoreDrivenPCSTExtractor` — `belongs_to_weight`=0.3, `fk_weight`=0.5, `macro_weight`=1.5, `epsilon`=0.0001, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True


## s04 — Query-Conditioned Projector GAT

#### `s04_01_qcond_a085_xiyan`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_query_conditioned.pt, `alpha`=0.85, `top_k`=20, `query_conditioned`=True, `encoder_type`=plm
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ComponentAwareProductCostPCSTExtractor` — `bt_weight`=0.1, `fk_weight`=0.2, `macro_weight`=0.5, `min_cost`=0.0001, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `s04_02_supernode_a070_xiyan`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode.pt, `alpha`=0.7, `top_k`=20, `query_supernode`=True, `encoder_type`=plm
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ComponentAwareProductCostPCSTExtractor` — `bt_weight`=0.1, `fk_weight`=0.2, `macro_weight`=0.5, `min_cost`=0.0001, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `s04_03_supernode_a085_xiyan`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode.pt, `alpha`=0.85, `top_k`=20, `query_supernode`=True, `encoder_type`=plm
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ComponentAwareProductCostPCSTExtractor` — `bt_weight`=0.1, `fk_weight`=0.2, `macro_weight`=0.5, `min_cost`=0.0001, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `s04_04_qcond_a0_xiyan`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_query_conditioned.pt, `alpha`=0.0, `top_k`=20, `query_conditioned`=True, `encoder_type`=plm
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ComponentAwareProductCostPCSTExtractor` — `bt_weight`=0.1, `fk_weight`=0.2, `macro_weight`=0.5, `min_cost`=0.0001, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `s04_05_supernode_a0_xiyan`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode.pt, `alpha`=0.0, `top_k`=20, `query_supernode`=True, `encoder_type`=plm
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ComponentAwareProductCostPCSTExtractor` — `bt_weight`=0.1, `fk_weight`=0.2, `macro_weight`=0.5, `min_cost`=0.0001, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True


## s05 — DirectGATSelector (BCE-only, No Projector)

### s05_gat_direct/a01_full_pipeline
*Direct Selector 전체 파이프라인*

#### `s05_a01_01_qcond_direct_xiyan`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_conditioned_direct.pt, `query_conditioned`=True, `query_supernode`=False, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ComponentAwareProductCostPCSTExtractor` — `bt_weight`=0.1, `fk_weight`=0.2, `macro_weight`=0.5, `min_cost`=0.0001, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `s05_a01_02_supernode_direct_xiyan`

- **Seed Selector**: `DirectGATSelector` — `weight_path`=outputs/checkpoints/best_gat_query_supernode_direct.pt, `query_conditioned`=False, `query_supernode`=True, `encoder_type`=plm, `in_channels`=384, `hidden_channels`=256, `out_channels`=256, `classifier_hidden`=256
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ComponentAwareProductCostPCSTExtractor` — `bt_weight`=0.1, `fk_weight`=0.2, `macro_weight`=0.5, `min_cost`=0.0001, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True


---

## b0 — Baselines

Config 파일 없이 외부 방법 재현 또는 초기 간단 구현. 실제 실행은 legacy 스크립트 기반.

### 우리 baseline

#### `b0_01_vector_only`
- **Seed Selector**: PLM cosine top-k 만 사용 (VectorOnly 등가)
- **Connectivity Extractor**: 없음
- **Filter**: 없음
- **특징**: 순수 embedding 유사도 상한선. 우리 파이프라인의 Selector 기여 upper-bound 가늠용

#### `b0_02_graph_expansion`
- **Seed Selector**: PLM cosine
- **Connectivity Extractor**: PCST (fixed cost)
- **Filter**: 없음
- **특징**: 초기 그래프 확장 시도. Recall 유지에 실패(0.6417)

#### `b0_03_graph_agent`
- **Seed Selector**: PLM cosine
- **Connectivity Extractor**: PCST
- **Filter**: `AdaptiveMultiAgentFilter`
- **특징**: Multi-agent 정제 추가 — Graph Expansion 대비 개선 없음 (LLM 판단이 fixed-cost PCST 출력의 잡음을 복구하지 못함)

### 외부 baseline

#### `b0_04_g_retriever`
- **참고**: He et al., NeurIPS 2024
- **Connectivity Extractor**: PCST (query-prize)
- **Filter**: 없음
- **특징**: 우리 파이프라인의 출발점 논문

#### `b0_05_linkalign`
- **참고**: EMNLP 2025
- **특징**: Multi-DB retrieval → schema item grounding 2단계

#### `b0_06_xiyan_sql`
- **Filter**: XiYan Filter 단독 적용
- **특징**: LLM 판단만으로 schema linking 수행하는 상한선

---

## t — GAT Training Runs (Checkpoint 카탈로그)

각 training run이 산출하는 checkpoint. Selector는 여기서 학습된 가중치를 로드.

#### `t01_gat_v1` — `gat_classifier_best.pt`
- **모델**: `SchemaHeteroGAT` (GATv2Conv, hidden=256)
- **Loss**: BCE (node-level binary classification)
- **Best Recall@15**: 0.5885 (early stop at epoch 122)
- **사용처**: `s02_gat_classifier` 계열

#### `t02_mlp_classifier` — `mlp_classifier_train_best_recall.pt`
- **모델**: MLP (GAT 없이 PLM 임베딩만)
- **Loss**: BCE / 300 epoch

#### `t03_mlp_gat` — `mlp_classifier_with_gat_train_best_recall.pt`
- **모델**: MLP + GAT 연쇄
- **Loss**: BCE / 300 epoch

#### `t04_gat_infonce` — `best_gat_model.pt`
- **모델**: HeteroGAT + DualTowerProjector
- **Loss**: BCE + InfoNCE (contrastive)
- **Best Recall@15**: 0.4876 (InfoNCE 추가로 오히려 하락)
- **사용처**: `s03_gat_ensemble/a01~a06` 및 `abl/a01~a02` 대부분의 기본 체크포인트

#### `t05_enriched_gat` — `best_gat_enriched.pt`
- **모델**: EnrichedHeteroGraphBuilder 기반 GAT (노드에 확장 feature 추가)
- **Loss**: BCE + InfoNCE
- **사용처**: `s03_a07_01_enriched_gat`, `s03_a07_02_edge_prize`

#### `t06_qcond_projector` — `best_gat_query_conditioned.pt`
- **모델**: Query-Conditioned GAT (Concat 방식) + Projector
- **Loss**: BCE + InfoNCE
- **특징**: 모든 노드 feature에 query embedding concat (384+384=768-dim 입력)
- **사용처**: `s04_01`, `s04_04`

#### `t07_supernode_projector` — `best_gat_query_supernode.pt`
- **모델**: Query-SuperNode GAT + Projector
- **Loss**: BCE + InfoNCE
- **특징**: Query를 virtual node로 추가, 모든 schema 노드와 양방향 연결
- **사용처**: `s04_02`, `s04_03`, `s04_05`

#### `t08_qcond_direct` — `best_gat_query_conditioned_direct.pt`
- **모델**: Query-Cond GAT + Classifier head (Projector 제거)
- **Loss**: BCE only (InfoNCE 제거)
- **Best Recall@15**: **0.5914** (전체 최고)
- **사용처**: `s05_a01_01`, `abl/a03` qcond 계열

#### `t09_supernode_direct` — `best_gat_query_supernode_direct.pt`
- **모델**: Query-SuperNode GAT + Classifier head
- **Loss**: BCE only
- **Best Recall@15**: 0.5548
- **사용처**: `s05_a01_02`, `abl/a03` supernode 계열, `abl/a04`

---

## 모듈별 하이퍼파라미터 참조

### Seed Selector

| 모듈 | 주요 파라미터 | 설명 |
|------|--------------|------|
| `VectorOnlySelector` | `top_k` | PLM cosine 상위 k개 |
| `GATClassifierSelector` | `weight_path`, `top_k` | 학습된 GAT의 sigmoid score 상위 k개 |
| `EnsembleSelector` | `weight_path`, `alpha`, `top_k` | score = α·GAT + (1-α)·cosine |
| `DirectGATSelector` | `weight_path`, `query_conditioned`, `query_supernode`, `apply_threshold`, `threshold`, `in_channels`, `hidden_channels`, `out_channels`, `classifier_hidden`, `encoder_type` | BCE-trained classifier. `apply_threshold=true`일 때 sigmoid≥threshold만 반환 |

### Connectivity Extractor

| 모듈 | 주요 파라미터 | 특징 |
|------|--------------|------|
| `None` | - | seed_nodes 그대로 통과 |
| `TopKExtractor` | `top_k` | score 상위 k개만 |
| `PCSTExtractor` | `base_cost`, `belongs_to_cost`, `fk_cost`, `macro_cost`, `node_threshold` | Fixed-cost PCST (Goemans-Williamson 2-근사) |
| `AdaptivePCSTExtractor` | + `percentile`, `min_prize_nodes`, `max_prize_nodes` | P80 threshold로 prize 동적 계산 |
| `DynamicPCSTExtractor` | + `hub_discount` | Hub 노드 cost 할인 |
| `UncertaintyPCSTExtractor` | + `uncertainty_margin` | score 불확실성 반영 |
| `ScoreDrivenPCSTExtractor` | `bt_weight`, `fk_weight`, `macro_weight`, `epsilon` | cost를 (1 - score) × weight로 |
| `ProductCostPCSTExtractor` | `bt_weight`, `fk_weight`, `macro_weight`, `min_cost` | cost = type × (1-s_u)(1-s_v) (Idea 2) |
| `SteinerBackbonePCSTExtractor` | + `backbone_bonus` | Steiner tree 2-근사(Kou 1981) backbone + PCST 확장 (Idea 3) |
| `ComponentAwareAdaptivePCSTExtractor` | Adaptive와 동일 | 각 CC에 대해 독립 실행 (Idea 4) |
| `ComponentAwareProductCostPCSTExtractor` | ProductCost와 동일 | CC별 독립 + product cost (Idea 2+4) |
| `EdgePrizePCSTExtractor` | + `topk_e`, `edge_cost` | G-Retriever 스타일 edge prize |
| `GATAwarePCSTExtractor` | - | GAT 잠재표현을 prize에 반영 (Phase A 실험적) |

### Filter

| 모듈 | 주요 파라미터 | 설명 |
|------|--------------|------|
| `None` | - | Filter 없음 |
| `SingleAgentFilter` | `model_name`, `temperature` | 단일 LLM agent 정제 |
| `AdaptiveMultiAgentFilter` | `model_name`, `max_iteration`, `temperature` | 다중 agent 반복 정제 |
| `XiYanFilter` | `model_name`, `max_iteration`, `temperature`, `api_key`, `base_url` | XiYan-SQL 방식 (현재 standard) |
| `ReflectionFilter` | `model_name`, `max_iteration`, `temperature` | F1: Self-Refine (propose → critique → revise) |
| `VerifierFilter` | `model_name`, `max_iteration`, `temperature` | F2: XiYan + NL unit test 검증 및 missing node 복원 |
| `TieredBidirectionalAgentFilter` | `model_name`, `temperature`, `use_graph_context` | F3: Tier-1(PCST)/Tier-2(selector-only) prune + restore |
| `AdaptiveDepthFilter` | `model_name`, `high_conf_threshold`, `low_conf_threshold`, `reflection_max_iteration` | F4: 신뢰도 기반 XiYan/Reflection/Tiered 분기 |
| `StackedFilter` | `stages` (list of filter configs) | 필터 체이닝 (e.g. F3→F2) |

### Graph Builder

| 모듈 | 주요 파라미터 | 특징 |
|------|--------------|------|
| `HeteroGraphBuilder` | `include_views`, `run_leiden_clustering`, `tables_json_path` | Default. Tables/Columns/FK 3-type 노드 |
| `EnrichedHeteroGraphBuilder` | + enriched feature 경로 | 노드에 확장 feature 주입 (`t05`) |
| `TripletGraphBuilder` | + `triplet_path` | Triplet relation edge 임베딩 (`s03_a07_02`) |

---

## a05_filter_agentic (2026-04-14 pre-registered, 2026-04-15 rolling execution)

Filter 모듈 agentic 고도화 실험. Anchor = a03_17 (SuperNode Direct + Fixed PCST,
filter만 교체). a05_11/12(GPT-4o-mini)는 이번 라운드 제외. 플랜:
`/home/hyeonjin/.claude/plans/vivid-sprouting-sunbeam.md`.

**진행 결과**:

| ID | Recall | Precision | F1 | Runtime |
|----|--------|-----------|------|---------|
| a03_17 (anchor) | 0.6761 | 0.7128 | **0.6940** | — |
| a05_01 AdaptiveMultiAgent | 0.3770 | 0.6276 | 0.4713 | 10h 23m |
| a05_02 Reflection (1 iter) | **0.7320** | 0.6833 | **0.7068** | 3h 18m |

**구성표**:

| ID | Selector | Extractor | Filter | Backbone | Retry |
|----|----------|-----------|--------|----------|-------|
| `a05_01` | `DirectGATSelector` (t09, QSuperNode) | `PCSTExtractor` (fixed) | `AdaptiveMultiAgentFilter` (thr=0.6) | Qwen3-Coder-30B | - |
| `a05_02` | (동일) | (동일) | `ReflectionFilter` (iter=1) | Qwen | - |
| `a05_03` | (동일) | (동일) | `ReflectionFilter` (iter=3) | Qwen | - |
| `a05_04` | (동일) | (동일) | `VerifierFilter` (iter=1) | Qwen | - |
| `a05_05` | (동일) | (동일) | `TieredBidirectionalAgentFilter` (use_graph_context=false) | Qwen | - |
| `a05_06` | (동일) | (동일) | `TieredBidirectionalAgentFilter` (use_graph_context=true) ★ | Qwen | - |
| `a05_07` | (동일) | (동일) | `AdaptiveDepthFilter` (high=0.20, low=0.05) | Qwen | - |
| `a05_08` | (동일) | (동일) | `StackedFilter` (Tiered→Verifier) | Qwen | - |
| `a05_09` | (동일) | (동일) | `TieredBidirectionalAgentFilter` (full) | Qwen | K=2, widen+steiner |
| `a05_10` | (동일) | (동일) | `AdaptiveDepthFilter` | Qwen | K=2, widen+steiner |
| `a05_11` | (동일) | (동일) | `TieredBidirectionalAgentFilter` (full) | GPT-4o-mini | - |
| `a05_12` | (동일) | (동일) | `AdaptiveDepthFilter` | GPT-4o-mini | K=2, widen+steiner |

공통 하이퍼: `temperature=0.0`, auto_join_keys=true, t09 SuperNode Direct
체크포인트. GPT-4o-mini 실험은 `.env`의 `OPENAI_API_KEY` 필요 (config는
`api_key: null`로 env 폴백). F5 retry는 pipeline 레벨 설정
(`extraction_retry.enabled: true`).

실행: `bash run_a05_filter_agentic.sh [id...]`.
