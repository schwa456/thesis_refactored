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


### abl/sel/ns_l1
*S-V: Neurosymbolic Layer 1 Selector (제안 #9). `EnsembleSelector`의 (α·cos + (1−α)·GAT) 결과에 `λ · reach_mask`를 합산 (reach_mask = question anchor 테이블에서 FK-reachable 한 node). Builder B-III의 `metadata['fk_reachability']`를 소비하며, 키가 없으면 graceful fallback → ensemble 동작.*

#### `abl_sel_ns_l1_01`

- **Seed Selector**: `NeurosymbolicL1Selector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20, `lambda_sym`=0.1, `anchor_min_token_len`=3
  - Base: `EnsembleSelector` + `_post_ensemble_hook` override (boosted = ensemble + 0.1 · reach_mask)
  - Anchor 식별: question 토큰 (`[a-zA-Z0-9]+`, min_len=3) ↔ table 이름 snake_case words + column 이름 snake_case words 매칭
  - Reach mask: 테이블은 `fk_reachability[anchors].any(axis=0)` → 1.0, 컬럼은 owning table 상속, FK 노드는 endpoint table 중 하나라도 reachable이면 1.0
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True
- **Anchor**: `s03_a02_03_xiyan_filter` (Ensemble + AdaptivePCST + XiYan, 2×2×2 best). 가설: 3-table JOIN 등 bridge-table recall 개선.
- **Status**: 구현 + smoke 통과. End-to-end F1 pending (vLLM 서버 필요). λ sweep (0.05 / 0.1 / 0.2) 후속.


### abl/build/fk_reach
*B-III: FK reachability precompute (Symbolic Layer 1, 제안 #9). Builder가 Floyd-Warshall로 table-level FK adjacency / reachability / shortest-paths / connected-components를 계산해 metadata에 자동 주입. 모든 하류 모듈은 새 키를 무시해도 호환.*

#### `abl_build_01_fk_reach`

- **Graph Builder**: `EnrichedHeteroGraphBuilder` — `tables_json_path`=data/raw/BIRD_dev/dev_tables.json, `include_views`=False, `run_leiden_clustering`=True
  - Auto-injected metadata keys: `fk_adjacency`, `fk_adjacency_undirected`, `fk_reachability`, `fk_distance`, `fk_shortest_paths`, `fk_components`, `fk_num_components`, `fk_edge_lookup`
- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_enriched.pt, `alpha`=0.85, `top_k`=20
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True
- **Anchor**: `s03_a07_01_enriched_gat` (E1, F1=0.7327) — 동일 numbers 내 noise


### abl/build/linegraph
*B-II: LineGraph 변환 (EHGAT 인프라, 제안 #3). 5종 edge를 노드로 승격, 공유 노드 기반 line-graph edge. 하류 EHGAT Selector(S-III) 미구현 상태 — 본 config는 builder 단위 smoke 용도.*

#### `abl_build_02_linegraph`

- **Graph Builder**: `LineGraphBuilder` — `base`=EnrichedHeteroGraphBuilder, `base_params.tables_json_path`=data/raw/BIRD_dev/dev_tables.json, `include_endpoint_diff`=True, `skip_macro_edges`=False
  - Output: `HeteroData` with single node type `edge_node` and `(edge_node, shares_node, edge_node)` edges
  - Edge node feature dim = 4 (type one-hot) + 384·2 (mean + diff endpoint emb) [+ 384 if base=Triplet] = 772 (or 1156 with Triplet)
  - Metadata: `edge_node_to_orig`, `orig_node_to_edges`, `edge_type_order`, `edge_type_to_idx`, `orig_data`, `orig_metadata` (+ FK reachability forwarded)
- **Status**: pending Selector S-III (EHGAT). 현 main.py 파이프라인은 `table/column/fk_node` 가정 → end-to-end 미실행.


### abl/build/rfm_tokens
*B-I: RFMCompatibleBuilder (Schema serialization, 제안 #2). 기존 Enriched 노드 텍스트는 그대로 두고 zero-shot RFM encoder용 special-token serialization을 metadata에 추가 노출.*

#### `abl_build_03_rfm_tokens`

- **Graph Builder**: `RFMCompatibleBuilder` — `tables_json_path`=data/raw/BIRD_dev/dev_tables.json, `include_views`=False, `run_leiden_clustering`=True, `include_values`=True, `max_values`=3, `value_max_chars`=50, `max_desc_chars`=200
  - Special tokens: `[DB][TAB][/TAB][COL][TYPE][PK][DESC][VAL][FKS][FK→]`
  - Added metadata: `rfm_text` (str), `rfm_tokens` (List[str]), `rfm_special_tokens`
  - dev DB 11개 token 길이 profile: min 203 / median 1041 / mean 1177 / max 2578 (`european_football_2`)
- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_enriched.pt, `alpha`=0.85, `top_k`=20
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — defaults
- **Filter**: `XiYanFilter` — defaults
- **Post-processing**: `auto_join_keys`=True
- **Anchor**: `s03_a07_01_enriched_gat` (E1, F1=0.7327) — Enriched와 동일 inference path. RFM Selector(S-II) 미장착 상태에서는 결과 동일해야 정상.


### abl/build/no_t2t
*B-II.b: base heterograph T2T edge toggle (advisor 2026-04-21 의견 2). `(table, table_to_table, table)` macro edges 를 base 단계에서 제거. line-graph `skip_macro_edges` 와 직교한 control variable.*

#### `abl_build_05_no_t2t`

- **Graph Builder**: `EnrichedHeteroGraphBuilder` — `tables_json_path`=data/raw/BIRD_dev/dev_tables.json, `include_views`=False, `run_leiden_clustering`=True, **`add_t2t_edges`=False**
  - Cache: `dev_enriched_no_t2t_plm_graphs.pt` (별도 suffix)
  - 검증 (california_schools): T2T edges 4 → 0, FK reachability 동일, schema_diameter 4 → 8 (FK→column→FK 우회 거리)
- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_enriched.pt, `alpha`=0.85, `top_k`=20
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — defaults
- **Filter**: `XiYanFilter` — defaults
- **Anchor**: `s03_a07_01_enriched_gat` (E1, F1=0.7327). **주의**: Enriched checkpoint 는 T2T 포함 그래프 위에서 학습됨 → distribution shift 가능, recall 하락 시 GAT 재학습 필요.


### abl/build/diameter_meta
*B-III.b: full hetero schema diameter precompute (advisor 2026-04-21 의견 2). metadata 에 `schema_diameter`, `schema_eccentricity` 추가만, 파이프라인은 키를 무시 → behavioral identical to E1. 후속 Selector QCondGAT `num_layers ∈ {1,2,3,D_max,D_max+1}` 스윕(advisor proposal C)의 기반 인프라.*

#### `abl_build_06_diameter_meta`

- **Graph Builder**: `EnrichedHeteroGraphBuilder` — defaults (E1 와 동일)
  - Added metadata: `schema_diameter` (int, full hetero undirected D_max, disconnected 시 component max), `schema_eccentricity` (Dict[flat_idx, int])
- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_enriched.pt, `alpha`=0.85, `top_k`=20
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — defaults
- **Filter**: `XiYanFilter` — defaults
- **Anchor**: `s03_a07_01_enriched_gat` (E1, F1=0.7327) — noise 일치(±0.5pp) 확인용 regression marker.


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


### s03_gat_ensemble/a09_topology_cost
*Topology-derived edge cost PCST (edge-type parameter-free). 방향 2: degree 기반 cost signal로 BO가 갇혔던 edge-type tuning 공간 자체에서 탈출. Filter는 전부 None으로 고정하여 extractor 순효과 측정.*

#### `s03_a09_01_topology_no_filter`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `TopologyCostPCSTExtractor` — `gamma`=1.0, `lambda_prize`=0.3, `cost_scale`=0.1, `degree_combination`=max, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `s03_a09_02_ca_topology_no_filter`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ComponentAwareTopologyCostPCSTExtractor` — `gamma`=1.0, `lambda_prize`=0.3, `cost_scale`=0.1, `degree_combination`=max, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `s03_a09_03_basic_no_filter_anchor`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `PCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `node_threshold`=0.1
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `s03_a09_04_ca_product_no_filter_anchor`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `ComponentAwareProductCostPCSTExtractor` — `bt_weight`=0.1, `fk_weight`=0.2, `macro_weight`=0.5, `min_cost`=0.0001, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `s03_a09_05_adaptive_no_filter_anchor`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `AdaptivePCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

### s03_gat_ensemble/a10_fk_steiner
*FK-Backbone Steiner Closure (Graph 2-레벨 분해): Table FK backbone 을 Steiner 2-근사로 강제 closure + Column recovery threshold θ_r 조절. 목표 "Filter 이전 Recall ≥ 0.85".*

#### `s03_a10_01_fk_steiner_full_col`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `FKBackboneSteinerExtractor` — `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0, `column_recovery_threshold`=0.0, `force_fk_columns`=True, `fallback_to_parent`=True
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `s03_a10_02_fk_steiner_mid_col`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `FKBackboneSteinerExtractor` — `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0, `column_recovery_threshold`=0.3, `force_fk_columns`=True, `fallback_to_parent`=True
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `s03_a10_03_fk_steiner_high_col`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_model.pt, `alpha`=0.85, `top_k`=20
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `FKBackboneSteinerExtractor` — `percentile`=80.0, `min_prize_nodes`=3, `max_prize_nodes`=25, `node_threshold`=0.0, `column_recovery_threshold`=0.5, `force_fk_columns`=True, `fallback_to_parent`=True
- **Filter**: `None`
- **Post-processing**: `auto_join_keys`=True

#### `s03_a10_04_fk_steiner_r01` ~ `s03_a10_11_fk_steiner_r10` (θ_r sweep, 진행 중)

동일 구성 — `column_recovery_threshold` 만 변경:
- `s03_a10_04_fk_steiner_r01` → θ_r=0.1
- `s03_a10_05_fk_steiner_r02` → θ_r=0.2
- `s03_a10_06_fk_steiner_r04` → θ_r=0.4
- `s03_a10_07_fk_steiner_r06` → θ_r=0.6
- `s03_a10_08_fk_steiner_r07` → θ_r=0.7
- `s03_a10_09_fk_steiner_r08` → θ_r=0.8
- `s03_a10_10_fk_steiner_r09` → θ_r=0.9
- `s03_a10_11_fk_steiner_r10` → θ_r=1.0

다른 모든 파라미터는 a10_01~03 과 동일. 실행 스크립트: `scripts/run_fk_steiner_sweep.sh`.

#### Offline Percentile Sweep (2026-04-17) — Extractor 확장

`FKBackboneSteinerExtractor` 에 per-query percentile 기반 column recovery 모드 추가 (backward compatible):

- **신규 파라미터**:
  - `column_recovery_percentile` (Optional[float]) — `None` 이면 절댓값 모드 (기존). 값 지정 시 percentile 모드.
  - `column_recovery_percentile_scope` ∈ `{global, all_cols, closed_cols, per_table}` — percentile 계산 모집단.
- **오프라인 재평가**: a10_09 의 `score_analysis_s03_a10_09_fk_steiner_r08.jsonl` 를 재사용해 Selector 재실행 없이 extractor stage 만 4 scopes × 21 percentiles + abs anchor = **85 configs** 평가. Config 기반이 아닌 offline sweep 이므로 신규 `s03_a10_XX` ID 는 발급하지 않음.
- **Best config**: `all_cols p=95` — R=0.6167, P=0.4626, F1=0.5287 (abs_anchor F1=0.5242 대비 +0.0045).
- **High-Recall 운영점**: `closed_cols p=50` — R=0.8522, P=0.2389 (Filter-앞 단계에서 P 최고).
- 세부 결과: HISTORY §6-19, [notebooks/analysis_results/fk_steiner_percentile_sweep.md](notebooks/analysis_results/fk_steiner_percentile_sweep.md).


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


## s04_ablation — Stagewise Backfill (Wave 1.5, Extractor 축 통일)

Basic PCST 로 Extractor 를 통일해 Selector 축 (Legacy Ensemble vs QCond encoder vs QCond+GAT blend) 순수 기여를 분리. 2026-04-22 Wave 1.5 backfill 번들. HISTORY §8 참조.

#### `s04_stagewise_ensemble_raw_a0`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_query_conditioned.pt, `alpha`=0.0, `top_k`=20, `query_conditioned`=False (legacy cosine-only), `encoder_type`=plm
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `PCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `node_threshold`=0.1
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `s04_stagewise_qcond_raw_basic`

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_query_conditioned.pt, `alpha`=0.0, `top_k`=20, `query_conditioned`=True, `encoder_type`=plm
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `PCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `node_threshold`=0.1
- **Filter**: `XiYanFilter` — `model_name`=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, `max_iteration`=1, `temperature`=0.0
- **Post-processing**: `auto_join_keys`=True

#### `s04_stagewise_qcond_gat_basic` ★ (F1 new top 0.7877)

- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_query_conditioned.pt, `alpha`=0.85, `top_k`=20, `query_conditioned`=True, `encoder_type`=plm
- **Projection**: `enabled=False`
- **Connectivity Extractor**: `PCSTExtractor` — `base_cost`=0.05, `belongs_to_cost`=0.01, `fk_cost`=0.05, `macro_cost`=0.5, `node_threshold`=0.1
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
filter만 교체). a05_11/12(GPT-4o-mini F3/F4)는 이번 라운드 제외. a05_13/14/15/17은
GPT-4o-mini backbone 민감도 (2026-04-16~17 추가, a05_16 Reflection 3iter는 비용
$3.6 추정으로 skip). 플랜: `/home/hyeonjin/.claude/plans/vivid-sprouting-sunbeam.md`.

**진행 결과**:

| ID | Recall | Precision | F1 | Runtime |
|----|--------|-----------|------|---------|
| a03_17 (anchor) | 0.6761 | 0.7128 | **0.6940** | — |
| a05_01 AdaptiveMultiAgent | 0.3770 | 0.6276 | 0.4713 | 10h 23m |
| a05_02 Reflection (1 iter) | **0.7320** | 0.6833 | **0.7068** | 3h 18m |
| a05_04 VerifierFilter | 0.7093 | 0.6676 | 0.6878 | 6h 48m |
| a05_13 XiYan (gpt-4o-mini) | 0.6037 | **0.7317** | 0.6616 | 38m |
| a05_14 AdaptiveMultiAgent (gpt-4o-mini) | 0.3992 | 0.7576 | 0.5230 | 176m |
| a05_15 Reflection 1iter (gpt-4o-mini) | 0.6827 | 0.6620 | 0.6722 | 131m |
| a05_17 VerifierFilter (gpt-4o-mini) | 0.7055 | 0.6385 | 0.6706 | 206m |

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
| `a05_13` | (동일) | (동일) | `XiYanFilter` (iter=1) | GPT-4o-mini | - |
| `a05_14` | (동일) | (동일) | `AdaptiveMultiAgentFilter` (thr=0.6) | GPT-4o-mini | - |
| `a05_15` | (동일) | (동일) | `ReflectionFilter` (iter=1) | GPT-4o-mini | - |
| `a05_17` | (동일) | (동일) | `VerifierFilter` (iter=1) | GPT-4o-mini | - |

공통 하이퍼: `temperature=0.0`, auto_join_keys=true, t09 SuperNode Direct
체크포인트. GPT-4o-mini 실험은 `.env`의 `OPENAI_API_KEY` 필요 (config에
`api_key: null` + `base_url: https://api.openai.com/v1`로 명시해 VLLM env
폴백 차단). a05_16(Reflection 3iter + gpt-4o-mini)은 비용 추정 $3.6로 skip.
F5 retry는 pipeline 레벨 설정 (`extraction_retry.enabled: true`).
토큰 사용량은 `outputs/<exp>/token_usage.json` (input/cached/output 분할)으로
자동 기록 — APIClient.TOKEN_USAGE 누적기. 참고값: a05_13 1534 calls / 2.70M
input / 30K output / ~$0.41; a05_14 3531 calls / 1.43M input / 503K output /
~$0.52; a05_15 4569 calls / 8.13M input (cached 144K) / 193K output / ~$1.32;
a05_17 4472 calls / 8.06M input (cached 87K) / 496K output / ~$1.50.

실행: `bash scripts/run_a05_gpt4omini_filters.sh [a05_XX]` (a05_14/15/17 sequential).

---

## s06 — GAT Bottleneck Fix (ablation)

### s06_gat_bottleneck_fix/a01_additive_ablation
*Forward-additive ablation (B0 → B5). Filter 없이 Selector only (Val Recall@15).*

**공통 모델 (B0 대비 추가 옵션만 기술)**:
- Backbone: `SchemaHeteroGATv2` (src/models/gat_network_v2.py)
- `in_channels=384, hidden_channels=256, out_channels=256, heads=4, classifier_hidden=256, dropout=0.1`
- Dataset: BIRD train (`/SSL_NAS/.../train`), val_split=0.1 random
- Optimizer: AdamW (lr=1e-4, wd=1e-5), epochs=300
- 학습 스크립트: `src/train_gat_s06.py`

#### `s06_a01_01_b0_baseline`
- Mode: `query_conditioned=true, query_supernode=false`
- 추가 옵션: 전부 off (`pairnorm_mode=none, initial_residual_alpha=0, jumping_knowledge=none, dual_stream=false`)
- Loss: BCE (pos_weight=100.0), anti_collapse_weight=0
- num_layers=3
- **Checkpoint**: `best_gat_s06_a01_01_b0.pt` (NAS)
- **실측**: Val R@15 = **0.5738** (300ep, 3h 48m, 2026-04-16)

#### `s06_a01_02_b1_pairnorm`
- B0 ⊕ `pairnorm_mode=pairnorm` (scale=1.0)
- 기타 동일
- **Checkpoint**: `best_gat_s06_a01_02_b1.pt` (NAS)
- **실측**: Val R@15 = **0.5707** (300ep, 4h 12m, B0 대비 −0.0031)

#### `s06_a01_03_b2_initial_residual`
- B1 ⊕ `initial_residual_alpha=0.2` (APPNP-style)
- 기타 동일
- **Checkpoint**: `best_gat_s06_a01_03_b2.pt` (NAS)
- **실측**: Val R@15 = **0.5986** (300ep, 3h 57m, B0 대비 +0.0248)

#### `s06_a01_04_b3_listnet`
- B2 ⊕ `loss_type=listnet` (per-query ListNet, BCE 교체)
- 기타 동일 (pos_weight는 fallback 대비 유지)
- **Checkpoint**: `best_gat_s06_a01_04_b3.pt` (NAS)
- **실측**: Val R@15 = **0.5745** (300ep, 5h 36m, B0 대비 +0.0007 — joint 에서는 listnet signal 희석)

#### `s06_a01_05_b4_anti_collapse`
- B3 ⊕ `anti_collapse_weight=0.3, anti_collapse_tau_max=0.85`
- Schema-Aware Anti-Collapse Regularizer: 같은 table 내 column 쌍 cosine > τ_max 시 squared hinge
- **Checkpoint**: `best_gat_s06_a01_05_b4.pt` (NAS)
- **실측**: Val R@15 = **0.5894** (300ep, 9h 23m, B0 대비 +0.0156)

#### `s06_a01_06_b5_dual_stream`
- 구조 변경: `query_conditioned=false, dual_stream=true, jumping_knowledge=concat, num_layers=2, batch_size=1`
- 유지: `pairnorm_mode=pairnorm, initial_residual_alpha=0.2, loss_type=listnet, anti_collapse_weight=0.3`
- Dual-stream: schema는 query-free GAT 통과, query는 별도 MLP → fusion head `MLP(concat(h, z_q, h⊙z_q))`
- JK concat: L0 (lin 출력) + L1 + L2 모든 hidden 융합
- **Checkpoint**: `best_gat_s06_a01_06_b5.pt` (NAS, 70 MB)
- **실측**: Val R@15 = **0.6073** (rerun 2026-04-17 21:32 ~ 04-19 02:25, ~29h, B0 대비 **+0.0335**)
- **특이사항**: 첫 run (04-17 19:59) 은 fk_node 관련 버그로 1초만에 crash → fix 후 rerun. rerun best checkpoint 는 초반 epoch 에서 결정 (04-18 04:35 저장), 이후 ep 286/300 까지 돌다 중단.

#### `s06_a01_07_b5_enriched_dual_stream`
- B5 구조(PN + IR α=0.2 + JK concat + Dual-Stream + ListNet + AC 0.3) 그대로 + **EnrichedHeteroGraphBuilder**
- Builder: `EnrichedHeteroGraphBuilder` with `tables_json_path=/SSL_NAS/peoples/khj/thesis/train/train_tables.json` — tables.json 자연어명 + `database_description/*.csv` 주입
- `batch_size=8` (기존 B5 batch=1 → 29h 소요) — batched dual_stream 코드로 가속
- **Checkpoint**: `best_gat_s06_a01_07_b5_enriched.pt` (NAS, 67 MB)
- **실측**: Val R@15 = **0.6016 @ E60** (2026-04-20 21:49 ~ 04-21 07:03, **9h 14m**, B0 대비 +0.0278, **B5 대비 −0.0057**)
- **특이사항**: E60 이후 240 epoch 무갱신 (early saturation). Final loss 1.1382 < B5 1.1617 (train fit 개선) 이나 dev R@15 오히려 소폭 하락 → Enriched features 가 dev 일반화엔 중립/미세 부정적.
- **3축 병목 분석** (2026-04-21, [notebooks/analysis_results/s06_bottleneck_b5_enriched_extension.md](notebooks/analysis_results/s06_bottleneck_b5_enriched_extension.md)): L0_PLM 0.636 (B5 0.657, 더 분산) → L2_GAT 0.978 (B5 0.920, **더 collapse**) → L_out 0.329 (B5 0.357, Fusion 이 L2 collapse 를 뚫고 최종 더 분산). `fusion_head` gradient 1.83 (B5 0.59) — **Fusion 이 병목이자 구원자**. grad_ratio 0.244.

실행 (GPU 확보 후):
```bash
PYTHONPATH=src python src/train_gat_s06.py \
  --config configs/experiments/s06_gat_bottleneck_fix/a01_additive_ablation/s06_a01_02_b1_pairnorm.yaml
```

### s06 offline post-hoc analyses (B5 Head Retrain + LDBO Diagnostic, 2026-04-20)

Frozen L_out (학습된 B5 GAT 의 마지막 layer 출력) 캐시 위에서 head 만 재학습하는 offline 실험 계열. 새 config ID 는 부여하지 않음. 산출물 경로 `outputs/analysis/s06_bottleneck/B5/retrain/`. 세부 분석은 [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md) §7-2, §7-3 참조.

#### B5 Head Retrain 2×2 (query-random split)
- Cache: `lout_cache_{train,dev}.pt` (B5 ckpt forward 한 번, val_split=0.1 random)
- 5 cells: A(linear,bce,none), B(mlp,bce,none), C(mlp,**listnet**,none), D(mlp,bce,**zscore**), E(mlp,listnet,zscore)
- Head (mlp): `DirectClassifierHead(in=256, hidden=256, dropout=0.1)` 3-layer
- **Best (val-ES)**: C Dev AUC **0.6891** / D Dev R@15 **0.6228**
- **Best (oracle dev-ES)**: C Dev AUC **0.7548** @ep3 (vs 원본 B5 joint 0.7067, **+0.048**)
- Script: `src/analysis/b5_head_retrain.py` / Runner: `scripts/run_b5_head_retrain{,_CDE}.sh`

#### B5 Head-Only LDBO Diagnostic (2026-04-20)
- Train 69 DB 중 11 DB (≈16%) 를 `proxy_dev` 홀드아웃 (seed=42). 나머지 58 DB 로 head 학습.
- 같은 4 cells (B/C/D/E) 를 LDBO 로 재실행, GPU 0 순차.
- **결과**: LDBO val R@15 여전히 0.99+ (홀드아웃 train DB 가 "unseen" 역할 못 함). val-ES Dev AUC gap (LDBO vs query-random): B −0.003 / C −0.030 / D −0.006 / E +0.007.
- **진단**: BIRD train DB 간 domain 다양성 ≪ train↔dev domain gap → **train 내부 LDBO 로는 realistic shift simulate 불가**.
- Script: `src/analysis/b5_head_retrain.py --ldbo_frac 0.16 --train_json ...` / Runner: `scripts/run_b5_ldbo_diagnostic.sh`

