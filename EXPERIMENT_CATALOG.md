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
- **vLLM era**: `model_name=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8, max_iteration=1, temperature=0.0`
- **GLM era** (2026-04-24~): `provider=glm, model_name=zai-org/glm-4.7, max_iteration=1, temperature=0.0` (Elice ML API via `GLM_BASE_URL`)

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

---

## s04_ablation GLM era (2026-04-24)

LLM backbone 전환 series — XiYan Filter 는 `provider=glm, model_name=zai-org/glm-4.7` (Elice ML API, OpenAI-compatible). Selector/Extractor hyperparameter 는 vLLM era 원본과 동일.

#### `s04_04_qcond_a0_xiyan_glm` (sanity)
- **Seed Selector**: `EnsembleSelector` — `weight_path`=outputs/checkpoints/best_gat_query_conditioned.pt, `alpha`=0.0, `top_k`=20, `query_conditioned`=true, `encoder_type`=plm
- **Connectivity Extractor**: `ComponentAwareProductCostPCSTExtractor` — 기본 (bt=0.1, fk=0.2, macro=0.5, percentile=80, min/max_prize_nodes=3/25, node_threshold=0.0)
- **Filter**: `XiYanFilter(provider=glm, model_name=zai-org/glm-4.7, max_iteration=1, temperature=0.0)`

#### `abl_sel_diameter_layers_nl{1,2,3,6,7}_glm`
- **Seed Selector**: `EnsembleSelector(alpha=0.0, num_layers=N, weight_path=outputs/checkpoints/best_gat_qcond_nl{N}.pt, top_k=20, query_conditioned=true, encoder_type=plm)` (N∈{1,2,3,6,7})
- **Connectivity Extractor**: `ComponentAwareProductCostPCSTExtractor` (sanity 와 동일)
- **Filter**: `XiYanFilter(provider=glm, model_name=zai-org/glm-4.7, max_iteration=1, temperature=0.0)`
- **주의**: yaml 에 `num_layers: N` 명시 필수 (default=3 이면 N≠3 체크포인트에서 weight shape mismatch)

#### `s04_stagewise_qcond_gat_basic_glm` (GLM era new anchor — 전체 최고 F1=0.8383)
- **Seed Selector**: `EnsembleSelector(alpha=0.85, weight_path=best_gat_query_conditioned.pt, top_k=20, query_conditioned=true, encoder_type=plm)` (Wave 1.5 vLLM anchor 와 동일)
- **Connectivity Extractor**: `PCSTExtractor` (Basic) — `base_cost=0.05, belongs_to_cost=0.01, fk_cost=0.05, macro_cost=0.5, node_threshold=0.1`
- **Filter**: `XiYanFilter(provider=glm, model_name=zai-org/glm-4.7, max_iteration=1, temperature=0.0)`
- **Post-processing**: `auto_join_keys=True`

#### `layers_Ldbmax_glm` / `layers_Ldbmax_plus1_glm` (H2 truncate, 2026-04-25)
- **Seed Selector**: `EnsembleSelector(alpha=0.0, num_layers=6|7, num_layers_mode=D_max|D_max_plus1, diameter_cache_path=data/processed/dev_diameter.pt, weight_path=best_gat_qcond_nl{6|7}.pt, query_conditioned=true, encoder_type=plm)` — v2 `_resolve_active_depth` hook 활성
- **Connectivity Extractor**: `ComponentAwareProductCostPCSTExtractor` (sweep cell 과 동일)
- **Filter**: `XiYanFilter(provider=glm, model_name=zai-org/glm-4.7, max_iteration=1, temperature=0.0)`
- **Mechanism**: nl=6/7 ckpt 의 layer 수 per-DB 동적 truncate forward (D_max<N ckpt DB 도 동적 depth 로 inference — v2 분기 + `_resolve_active_depth` 사용)
- **결과 (2026-04-25)**:
  - `Ldbmax_glm` (D_max mode, nl=6 ckpt): R=0.5036 / P=0.7031 / **F1=0.5869** (ΔF1 vs L6_glm=+0.0045 partial neutral)
  - `Ldbmax_plus1_glm` (D_max+1 mode, nl=7 ckpt): R=0.4778 / P=0.6776 / **F1=0.5604** (ΔF1 vs L6_glm=−0.0220, training-inference depth mismatch 로 H2 기각 확고)

---

## Builder Cumulative Backfill (2026-04-26, Ablation 1 9-cell)

LLM 호출 0 (Filter=None / Extractor=None or Basic PCST). Selector 통일 stack: `EnsembleSelector(alpha=0.85, top_k=20, query_conditioned=true)` — Builder 별 GAT weight 만 다름.

#### `s04_stagewise_qcond_gat_basic_selector_only` (Plain Builder + Selector only)
- **Builder**: `HeteroGraphBuilder` (default, no graph_builder block)
- **Seed Selector**: `EnsembleSelector(alpha=0.85, top_k=20, weight_path=best_gat_query_conditioned.pt, query_conditioned=true, encoder_type=plm)`
- **Connectivity Extractor**: `None` (params={})
- **Filter**: `None` (params={})
- **Post-processing**: `auto_join_keys=false`

#### `s03_a07_01_enriched_gat_selector_only` (Enriched + Selector only)
- **Builder**: `EnrichedHeteroGraphBuilder` (include_views=false, run_leiden_clustering=true, tables_json_path=data/raw/BIRD_dev/dev_tables.json)
- **Seed Selector**: `EnsembleSelector(alpha=0.85, top_k=20, weight_path=best_gat_enriched.pt)`
- **Connectivity Extractor / Filter**: `None`
- **Post-processing**: `auto_join_keys=false`

#### `s03_a07_02_edge_prize_selector_only` (Triplet + Selector only)
- **Builder**: `TripletGraphBuilder` (include_views=false, run_leiden_clustering=true, triplet_path=data/processed/triplet_relations.json)
- **Seed Selector / Extractor / Filter**: enriched_gat_selector_only 와 동일 (weight_path=best_gat_enriched.pt 공유)

#### `s03_a07_01_enriched_gat_no_filter` (Enriched + Basic PCST + No Filter)
- **Builder**: `EnrichedHeteroGraphBuilder`
- **Seed Selector**: `EnsembleSelector(alpha=0.85, top_k=20, weight_path=best_gat_enriched.pt)`
- **Connectivity Extractor**: `PCSTExtractor` (Basic) — `base_cost=0.05, belongs_to_cost=0.01, fk_cost=0.05, macro_cost=0.5, node_threshold=0.1`
- **Filter**: `None`
- **Post-processing**: `auto_join_keys=true`

#### `s03_a07_02_edge_prize_no_filter` (Triplet + Basic PCST + No Filter)
- **Builder**: `TripletGraphBuilder`
- **나머지**: enriched_gat_no_filter 와 동일

### 9-cell summary (R / P / F1, 4자리)
| Builder | Selector only | + Extractor (Basic PCST, no filter) | + Filter (XiYan, final) |
|---------|--------------|-------------------------------------|------------------------|
| Plain | 0.7834 / 0.2700 / **0.4016** | 0.9651 / 0.1287 / **0.2271** | 0.8169 / 0.7605 / **0.7877** |
| Enriched | 0.7921 / 0.2567 / **0.3877** | 0.9676 / 0.1274 / **0.2252** | 0.6658 / 0.8147 / **0.7328** |
| Triplet | 0.7921 / 0.2567 / **0.3877** | 0.9676 / 0.1274 / **0.2252** | 0.6823 / 0.8139 / **0.7423** |

**Filter Δ F1 by Builder**: Plain +0.5606, Enriched +0.5076, Triplet +0.5171 (Plain 최대).

---

## Selector Ablation Cumulative Backfill (Option B, 2026-04-26, Plain/QCond × 3 score × 3 stage)

LLM 호출: Final 1 cell (qcond_cos_a1_glm, GLM API), 9 cells LLM-free. **SuperNode 9 cells 보류** (smoke fail, ckpt input dim mismatch). α convention: `final_score = α·cosine + (1−α)·gat`.

### Selector params 공통 (Encoder × Score)
- **Plain × GAT (α=0)**: `EnsembleSelector(alpha=0.0, weight_path=best_gat_model.pt)`
- **Plain × Cosine (α=1)**: `EnsembleSelector(alpha=1.0, weight_path=best_gat_model.pt)` — GAT module 가중치 무관 (cosine only)
- **Plain × Ensemble (α=0.85)**: `EnsembleSelector(alpha=0.85, weight_path=best_gat_model.pt)`
- **QCond × {GAT/Cos/Ens}**: `EnsembleSelector(alpha={0,1,0.85}, weight_path=best_gat_query_conditioned.pt, query_conditioned=true)`
- **SuperNode × {GAT/Cos/Ens}**: `EnsembleSelector(alpha={0,1,0.85}, weight_path=best_gat_query_supernode.pt, query_conditioned=true, query_supernode=true)` — **사용 불가, ckpt input dim mismatch 384 vs 768**
- 공통: `top_k=20, encoder_type=plm`

### Stage 별 Extractor/Filter
- **Selector only**: `connectivity_extractor.name=None` + `filter.name=None` + `auto_join_keys=false`
- **+Extractor (no_filter)**: `PCSTExtractor` (Basic, base_cost=0.05/belongs_to=0.01/fk=0.05/macro=0.5/node_threshold=0.1) + `filter.name=None` + `auto_join_keys=true`
- **+Filter (Final)**: 동일 PCSTExtractor + `XiYanFilter(provider=glm, model=zai-org/glm-4.7, max_iteration=1, temperature=0.0)` + `auto_join_keys=true`

### 18-cell summary (R / P / F1, 4자리)

| Encoder | Score | Selector only | + Extractor (no_filter) | + Filter (Final) | era |
|---------|-------|---------------|------------------------|------------------|-----|
| Plain | GAT (α=0) | 0.5281 / 0.2034 / **0.2937** | 0.7785 / 0.1330 / **0.2272** | 0.6676 / 0.7236 / **0.6945** | vLLM |
| Plain | Cosine (α=1) | 0.7693 / 0.2549 / **0.3829** | 0.9662 / 0.1302 / **0.2295** | 0.7987 / 0.7694 / **0.7838** | vLLM |
| Plain | Ensemble (α=0.85) | 0.7678 / 0.2681 / **0.3974** | 0.9667 / 0.1273 / **0.2250** | 0.8149 / 0.7597 / **0.7863** | vLLM |
| QCond | GAT (α=0) | 0.6061 / 0.2494 / **0.3534** | 0.7813 / 0.1752 / **0.2862** | 0.6622 / 0.7539 / **0.7051** | vLLM |
| **QCond** | **Cosine (α=1)** | 0.7693 / 0.2549 / **0.3829** | 0.9662 / 0.1302 / **0.2295** | **0.8501 / 0.8348 / 0.8424** 🚀 | **GLM** |
| QCond | Ensemble (α=0.85) | 0.7834 / 0.2700 / **0.4016** | 0.9651 / 0.1287 / **0.2271** | 0.8169 / 0.7605 / **0.7877** (vLLM) / **0.8438 / 0.8329 / 0.8383** 🚀 (GLM) | vLLM/GLM |

**Filter Δ F1 max**: QCond Cosine + GLM **+0.6129** (no_filter 0.2295 → final 0.8424). Plain GAT min +0.4673.

---

## GLM era 일관 재측정 (Ablation 1/2/3, 2026-04-27, 11 cells)

발사: 2026-04-27 01:01:27 → 완료: 03:14:47 (wall clock 2h 13min). GPU 2/3, 2 concurrent per GPU. LLM: 8 cells GLM API (~₩6,112), 3 cells LLM-free. MST smoke OK (28s에 15 preds) → smoke 종료 후 main 11 cells launch.

### Ablation 1 (Builder × Stage) GLM era 1 cell

| Cell ID | Builder | Stage | R | P | F1 |
|---------|---------|-------|---|---|---|
| `s03_a07_01_enriched_gat_glm` | Enriched | Final (XiYan(GLM)) | 0.6926 | 0.8300 | 0.7551 |

vs 기존 vLLM era a07_01 (F1≈0.7140): **+0.0411** — Builder 효과 GLM 환경에서 더 발현.

### Ablation 2 (Encoder × Score × Stage) GLM era final 4 cells

Stack: Plain encoder + Basic PCST + XiYan(GLM). α convention: `α·cosine + (1−α)·gat`.

| Cell ID | Score | R | P | F1 |
|---------|-------|---|---|---|
| `s04_stagewise_plain_gat_a0_glm` | α=0 GAT only | 0.6825 | 0.7153 | 0.6985 |
| **`s04_stagewise_plain_cos_a1_glm`** ★ | **α=1 Cos only** | **0.8472** | 0.8310 | **0.8390** |
| `s04_stagewise_plain_ens_glm` | α=0.85 Ensemble | 0.8447 | 0.8316 | 0.8381 |
| `s04_stagewise_qcond_gat_a0_glm` | α=0 GAT only (QCond encoder) | 0.6830 | 0.7638 | 0.7211 |

**관찰**: Plain Cos α=1 (F1=0.8390) ≈ QCond Cos α=1 (F1=0.8424) — encoder agnostic. α=1 우세 + GLM 조합이 새 GLM era top 후보 (anchor 갱신 임계 +0.005 미달, 직전 anchor `qcond_gat_basic_glm` 유지).

### Ablation 3 (Extractor × Stage) GLM era 신규 — 6 cells (final 3 + no_filter 3)

Stack: Plain encoder + Ensemble (α=0.85, weight_path=best_gat_model.pt) + Extractor 변형. Final = +XiYan(GLM), No-filter = filter.name=None.

#### Extractor params

- **AdaptivePCST**: `base_cost=0.05, belongs_to_cost=0.01, fk_cost=0.05, macro_cost=0.5, percentile=80.0, min_prize_nodes=3, max_prize_nodes=25, node_threshold=0.0`
- **SteinerBackbonePCST**: `backbone_bonus=0.5` + AdaptivePCST 와 동일 base
- **MSTExtractor**: `params: {}` (Kou-Markowsky-Berman 2-approx, seed_nodes 기반 metric closure)
- (참고) **Basic PCST = `plain_ens_glm`** (Ablation 2 final): `node_threshold=0.1`, fixed cost — 별도 cell

#### 6-cell summary (R / P / F1, 4자리)

| Extractor | + Extractor (no_filter) | + Filter (Final, XiYan(GLM)) | Filter Δ F1 |
|-----------|------------------------|------------------------------|-------------|
| AdaptivePCST | 0.7255 / 0.3480 / **0.4704** | 0.6479 / 0.8099 / **0.7199** | +0.2495 |
| SteinerBackbone | 0.8242 / 0.2345 / **0.3651** | 0.7081 / 0.8073 / **0.7545** | +0.3894 |
| MST (Steiner 2-approx) | 0.8370 / 0.2366 / **0.3689** | 0.7252 / 0.8276 / **0.7730** | +0.4041 |
| (참고) Basic PCST = `plain_ens_glm` | (별도) | 0.8447 / 0.8316 / **0.8381** | — |

**관찰**:

1. **Basic PCST 압도 (0.8381 vs MST 0.7730)** — vLLM era "Basic > Adaptive" 결론 GLM era 재현
2. **MST > Adaptive + XiYan (+0.0531) — 새 발견**. MST 의 좁은 selection 이 XiYan 정밀 prune 과 시너지
3. **Filter Δ F1 by Extractor**: MST (+0.4041) > Steiner (+0.3894) > Adaptive (+0.2495) — 입력 단순할수록 LLM filter 효율 ↑

### 산출물

- Configs (11): `s03_a07_01_enriched_gat_glm.yaml`, `s04_ablation/stagewise/{plain_gat_a0,plain_cos_a1,plain_ens,qcond_gat_a0}_glm.yaml`, `s04_ablation/extractor/plain_ens_{adaptive,steiner,mst}_glm.yaml`, `s04_ablation/extractor/no_filter/plain_ens_{adaptive,steiner,mst}_no_filter.yaml`
- Script: `scripts/run_glm_era_ablation_full.sh`

---

## Ablation 1/2/3 α=0.5 재측정 (Option B, 2026-04-27, 15 cells)

발사: 2026-04-27 14:41:16 → 완료: 17:42:22 (wall clock 3h 1min). GPU 2 (6 Final GLM, 3 batches × 2 concurrent), GPU 3 (9 LLM-free, 5 batches). α convention: `α·cosine + (1−α)·gat`, α=0.5 = neutral GAT/Cosine 동등 결합.

### 근거

- α=0.85 anchor 의 sweep 근거 한계 (I1a-c No Filter stack 한정, with-Filter 미수행)
- Filter 단 ensemble 약화 분석 (HISTORY L92, GAT 15% 만 반영)
- Advisor analysis (α=0.85 GAT 비중 비판, neutral baseline 권장)

### Selector params 공통 (α=0.5 통일)

- **Plain × Ensemble (α=0.5)**: `EnsembleSelector(alpha=0.5, weight_path=best_gat_model.pt)`
- **QCond × Ensemble (α=0.5)**: `EnsembleSelector(alpha=0.5, weight_path=best_gat_query_conditioned.pt, query_conditioned=true)`
- **Enriched × Ensemble (α=0.5)**: `EnsembleSelector(alpha=0.5, weight_path=best_gat_enriched.pt)` + `EnrichedHeteroGraphBuilder`
- 공통: `top_k=20, encoder_type=plm`

### Stage 별 Extractor/Filter

- **Selector_only**: `connectivity_extractor.name=None` + `filter.name=None` + `auto_join_keys=false`
- **+Extractor (no_filter)**: `PCSTExtractor` (Basic, fixed θ=0.1) + `filter.name=None` + `auto_join_keys=true`
- **+Filter (Final, GLM)**: 동일 Extractor + `XiYanFilter(provider=glm, model=zai-org/glm-4.7, max_iteration=1, temperature=0.0)` + `auto_join_keys=true`
- **Ablation 3 별도**: `connectivity_extractor.name ∈ {AdaptivePCSTExtractor, SteinerBackbonePCSTExtractor, MSTExtractor}` + 동일 selector

### 15-cell summary (R / P / F1, 4자리)

#### Ablation 2 — Plain/QCond × α=0.5 × 3 stage (6 cells)

| Encoder | Selector only | + Extractor (Basic, no_filter) | + Filter (Final, GLM) |
|---------|--------------|------------------------------|------------------------|
| Plain | 0.6301 / 0.2358 / **0.3432** | 0.9550 / 0.1217 / **0.2159** | 0.8316 / 0.8188 / **0.8252** |
| QCond | 0.7110 / 0.2780 / **0.3997** | 0.9581 / 0.1304 / **0.2296** | 0.8337 / 0.8275 / **0.8306** |

#### Ablation 1 — Enriched α=0.5 × 3 stage (3 cells)

| Stage | R / P / F1 |
|-------|-----------|
| Selector only | 0.6243 / 0.2326 / **0.3389** |
| + Extractor (Basic, no_filter) | 0.9557 / 0.1233 / **0.2184** |
| **+ Filter (Final, GLM)** | **0.8325 / 0.8199 / 0.8262** ★ |

#### Ablation 3 — Plain α=0.5 + 3 ext × 2 stage (6 cells)

| Extractor | + Extractor (no_filter) | + Filter (Final, GLM) |
|-----------|------------------------|------------------------|
| AdaptivePCST | 0.5849 / 0.2929 / **0.3903** | 0.5058 / 0.6730 / **0.5775** |
| SteinerBackbone | 0.6979 / 0.2101 / **0.3230** | 0.5992 / 0.7081 / **0.6491** |
| MST | 0.7231 / 0.2170 / **0.3338** | 0.6257 / 0.7377 / **0.6771** |

### α=0.85 vs α=0.5 비교 — Final GLM (with-Filter)

| Stack | α=0.85 F1 | α=0.5 F1 | ΔF1 |
|---|---|---|---|
| Plain Ens (Basic PCST) | 0.8381 | 0.8252 | -0.0129 |
| QCond Ens (Basic PCST) | 0.8383 | 0.8306 | -0.0077 |
| **Enriched Ens (Basic PCST)** | **0.7551** | **0.8262** | **+0.0711 ★** |
| Plain Ens + Adaptive | 0.7199 | 0.5775 | -0.1424 ⚠️ |
| Plain Ens + Steiner | 0.7545 | 0.6491 | -0.1054 |
| Plain Ens + MST | 0.7730 | 0.6771 | -0.0959 |

### 핵심 발견

1. **🚀 Builder × α 상호작용 — Enriched α=0.5 +0.0711**: Description 정보가 Cosine PLM 임베딩에 noise → α=0.85 cos 우세에서 손실, α=0.5 GAT 비중 ↑로 회복. **새 발표 narrative 핵심**.
2. **⚠️ Extractor × α 상호작용 — Adaptive/Steiner/MST 모두 α=0.5 큰 손실 -0.10~-0.14**: per-q P80 / backbone bonus / MST seed 가 score 분포에 sensitive, GAT noise 부정적 시너지. **Basic PCST (fixed θ=0.1) 가 α 변경에 robust**.
3. **anchor 유지 정당성 강화**: Plain/QCond Final α=0.85 vs α=0.5 ΔF1 +0.008~+0.013 (plateau 임계 +0.005 미달). I1a-c sweep "α=0.85 best" with-Filter 약하게 재현.
4. **Pre-Filter 평균 ΔF1 = -0.0354**: GAT 비중 ↑로 noise 영향, Filter 단계에서 일부 회복.

### 산출물

- Configs (15): 위 15 yaml
- Script: `scripts/run_ablation_alpha05_remeasure.sh`

---

## MST 변형 측정 (옵션 C + Union, 2026-04-27, 6 cells)

발사 1: 2026-04-27 19:11:12 → 완료 20:21:43 (4 cells, wall clock 1h 10min, GPU 1)
발사 2: 2026-04-27 20:23:30 → 완료 22:21:53 (2 cells, wall clock 1h 58min, GPU 1)
SuperNode 학습 GPU 0 와 병렬, 충돌 없음.

### 신규 Extractor 구현

- **MSTExtractor**: `seed_mode ∈ {topk, threshold}` + `score_threshold=0.1` 추가
- **MSTKruskalExtractor** (신규): `score_threshold=0.1` induced subgraph 위 Kruskal MST, Steiner point 없음
- **MSTPCSTUnionExtractor** (신규): MSTKruskal ∪ PCSTExtractor (Basic) 합집합

### Selector params 공통

- `EnsembleSelector(alpha=0.5, top_k=20, weight_path=best_gat_model.pt, encoder_type=plm)` (Plain encoder)

### Stage 별

- **+ Extractor (no_filter)**: extractor 선택 + `filter.name=None` + `auto_join_keys=true`
- **+ Filter (Final, GLM)**: 동일 extractor + `XiYanFilter(provider=glm, model=zai-org/glm-4.7, max_iteration=1, temperature=0.0)`

### 6-cell summary (R / P / F1, 4자리)

| Extractor | + Extractor (no_filter) | + Filter (Final, GLM) | Filter ΔF1 |
|---|---|---|---|
| Steiner Tree threshold seed (`MSTExtractor seed_mode=threshold`) | 0.9914 / 0.1223 / **0.2177** | 0.8720 / 0.8538 / **0.8628** | +0.6451 |
| **MST Kruskal (진짜 MST)** ★ | 0.9914 / 0.1222 / **0.2176** | **0.8724 / 0.8561 / 0.8642** | +0.6466 |
| **MST ∪ PCST union** 🆕 | **0.9914 / 0.1222 / 0.2176** | **0.8787 / 0.8560 / 0.8672** | **+0.6496 (max)** |

### 시나리오 판정 + anchor 결정

- **anchor 갱신 (옵션 C 4 cells, 직전)**: `qcond_gat_basic_glm` F1=0.8383 → **`plain_ens_a05_mst_kruskal_glm` F1=0.8642** (ΔF1=+0.0259, 임계 +0.005 의 5배 초과)
- **Union 시나리오 B (anchor 유지)**: ΔF1 = 0.8672 - 0.8642 = +0.0030 (plateau 임계 +0.005 미달). MST Kruskal anchor 유지, union 동률 후보 표기.

### 핵심 발견

1. **🚀 anchor 갱신** — MST Kruskal F1=0.8642 (vs 직전 anchor 0.8383, ΔF1=+0.0259). 사용자 의문 정확 해소
2. **MST Kruskal R 상한 도달** — no_filter R=0.9914 모든 변형 동일, score>0.1 노드 안 gold 회수율 자연 상한
3. **Union 미세 final F1 +0.0030 향상** — ΔR=+0.0063, ΔP=-0.0001. Filter 가 추가 엣지 정보로 정답 식별 미세 향상 (paper insight 보조)
4. **Algorithm 차이 무시 가능** — MST Kruskal vs Steiner Tree threshold ΔF1=+0.0014, no_filter R 동일
5. **Filter ΔF1 위계**: union (+0.6496) > MST Kruskal (+0.6466) > Steiner Tree threshold (+0.6451) > Basic PCST (+0.6131)
6. **명명 정정**: 기존 `MSTExtractor` = Steiner 2-approx (Kou-Markowsky-Berman 1981). post-deadline rename → `SteinerTreeExtractor` (alias 유지)

### 산출물

- Configs (6): `s04_ablation/extractor/{no_filter/,}plain_ens_a05_{steiner_threshold, mst_kruskal, mst_pcst_union}_{no_filter,glm}.yaml`
- Scripts: `scripts/run_mst_variants.sh`, inline launch
- 신규 Extractor 코드: `src/modules/extractors/mst.py` (seed_mode), `mst_kruskal.py`, `mst_pcst_union.py`

---

## Paper Main Pipeline (옵션 A2, 2026-04-28, 2 cells) — End-to-End Co-Design with Modular LLM Filter

발사: 2026-04-28 00:10:20 → 완료 01:12:39 (wall clock 1h 02min, GPU 1 only, SuperNode 학습 동시 진행 충돌 없음).

### 4 module 구성 (방향 F' paper main pipeline)

| Module | 결정 | 사유 |
|---|---|---|
| Builder | EnrichedHeteroGraphBuilder | Description-aware (CSV + tables.json 자연어명) |
| Encoder | LocalPLMEncoder (MiniLM-L6-v2) | 경량 PLM, GAT 입력 384-dim |
| Selector | EnsembleSelector α=0.5 + best_gat_qcond_nl3.pt + query_conditioned=true | Query-Conditioned GAT (Concat), neutral ensemble |
| Extractor | MSTKruskalExtractor 또는 MSTPCSTUnionExtractor | Score-threshold seed pool widening |
| Filter | XiYanFilter (provider=glm, model=zai-org/glm-4.7) | First-class fourth stage |

### 신규 폴더: `s04_ablation/pipeline/`

paper main pipeline 측정 전용 카테고리. 학술적 narrative 우선 표기 + 모듈 종합 측정.

### 등재된 cells (2 신규)

| 신규 ID | Extractor | R | P | F1 |
|---------|-----------|---|---|---|
| **`s04_pipeline_enriched_qcond_a05_mst_kruskal_glm`** ★ | MSTKruskalExtractor | 0.8741 | 0.8606 | **0.8673** (paper main) |
| `s04_pipeline_enriched_qcond_a05_mst_pcst_union_glm` | MSTPCSTUnionExtractor | 0.8772 | 0.8564 | **0.8667** (Union 변형) |

### 시나리오 판정 + anchor 결정

- **시나리오 A 부분 채택**: 둘 다 Plain anchor (F1=0.8642) 보다 미세 우세 (+0.0031, +0.0025)
- 갱신 임계 +0.005 미달 (LLM noise 범위 ±0.003~0.005)
- **paper anchor 권장**: `s04_pipeline_enriched_qcond_a05_mst_kruskal_glm` F1=0.8673 (학술적 narrative 우선)
- 측정 anchor (`plain_ens_a05_mst_kruskal_glm` F1=0.8642) 와 plateau 동등

### 핵심 발견

1. **🚀 End-to-End Co-Design 통합 효과 = Plain anchor 동등 F1** (+0.0031, plateau 내) — 학술적으로 강한 narrative
2. **MST Kruskal > Union (paper main)** ΔF1=-0.0006 — 사용자 의도 main pipeline = MST Kruskal stack
3. **Description + QCond + MST Kruskal + XiYan GLM 통합 효과 검증** — F1 손실 없음, narrative 정당성 확보

### 산출물

- Configs (2): `s04_ablation/pipeline/enriched_qcond_a05_{mst_kruskal, mst_pcst_union}_glm.yaml`
- Script: `scripts/run_paper_main_pipeline.sh`

---

## SuperNode QCond GAT 학습 완료 (옵션 A, 2026-04-28)

학습 완료 entry. paper main pipeline 의 H6 future work (Selector Concat vs SuperNode 결정) 의 base ckpt.

### 학습 config

- 파일: `configs/training/train_gat_query_supernode_qcond.yaml`
- experiment_name: `gat_query_supernode_qcond`
- model: `query_conditioned=true, query_supernode=true, in_channels=384` (effective_in=in_channels*2=768 자동)
- training: epochs=300, lr=1e-4, batch=8, infonce_lambda=0.5, num_hard_negatives=15
- 발사: 2026-04-27 18:32 → 완료 2026-04-28 03:35 (wall clock 9h 8min)

### Best ckpt

- **ckpt**: `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_query_supernode_qcond.pt` (symlink → `outputs/checkpoints/best_gat_query_supernode_qcond.pt`, 220 MB)
- **best epoch**: 228 / 300 (plateau 도달, 이후 72 epoch 추가 학습 무익)
- **val recall@15**: **0.5737**
- **lin_dict.column.weight shape**: (256, **768**) ← effective_in=768 정상 (query_conditioned=True 활성)
- **query_node lin_dict 존재**: query_supernode=True 활성

### Smoke test (load_state_dict 검증)

- Config: `configs/experiments/s04_ablation/stagewise/selector_only/supernode_qcond_a0_smoke.yaml`
- 결과: **load_state_dict + forward pass 정상 PASS** (700+ preds 산출, size mismatch 없음)
- 이전 supernode ckpt (input dim 384) 와 분리 — 새 ckpt 는 input dim 768 (사용자 framing 정의 매칭)

### 활용 (post-deadline)

- **H6 Selector 결정**: Concat (`best_gat_qcond_nl3.pt`, paper main pipeline 현재) vs SuperNode (`best_gat_query_supernode_qcond.pt`, 신규) 비교 측정
- **paper_research_direction.md §1 Selector 결정**: 측정 결과 기반 paper main pipeline 의 final Selector 선택
- **Wave 4 a05_filter_agentic**: SuperNode stack 도 base anchor 후보

---

## SuperNode 9-cell Matrix (Ablation 2 SuperNode, 2026-04-29, 9 cells)

발사: 2026-04-29 19:45:46 → 완료 21:38:51 (wall clock 1h 53min, GPU 0/1 split). 사용자 요청: SuperNode 의 α∈{0, 0.5, 1} × {Selector_only, +Basic PCST, +XiYan GLM} 단계별 점수.

### Selector params 공통

- `EnsembleSelector(weight_path=best_gat_query_supernode_qcond.pt, query_conditioned=true, query_supernode=true, encoder_type=plm, top_k=20)`
- α: 0.0 / 0.5 / 1.0
- 코드 fix 적용: `ensemble_selector.py:241-243` (SuperNode 분기 query_emb 전달)

### Stage 별

- **Selector_only**: extractor=None, filter=None
- **+Basic PCST (no_filter)**: PCSTExtractor (node_threshold=0.1) + filter=None
- **+XiYan GLM (Final)**: PCSTExtractor + XiYanFilter (provider=glm, model=zai-org/glm-4.7)

### 9-cell summary (R / P / F1, 4-decimal)

| α / Stage | Selector only | + Basic PCST (no_filter) | + XiYan(GLM) Final |
|---|---|---|---|
| **α=0** (GAT only) | 0.6035 / 0.2534 / **0.3569** | 0.5539 / 0.2809 / **0.3728** | 0.4738 / 0.6487 / **0.5476** |
| **α=0.5** (neutral) | 0.7276 / 0.2787 / **0.4030** | 0.9564 / 0.1396 / **0.2436** | **0.8353 / 0.8330 / 0.8341** |
| **α=1** (Cosine only) | 0.7693 / 0.2549 / **0.3829** | 0.9662 / 0.1302 / **0.2295** | **0.8441 / 0.8296 / 0.8368** |

### Filter ΔF1 by α (no_filter → Final)

| α | Filter ΔF1 |
|---|---|
| α=0 | +0.1748 (small) |
| α=0.5 | +0.5905 (large) |
| α=1 | +0.6073 (max) |

→ **α=0 SuperNode 의 small Filter Δ** = Filter 가 GAT-only signal noise prune 부담 ↑

### vs QCond Concat 비교 (Final GLM)

| α | Concat F1 | SuperNode F1 | ΔF1 (SN − Concat) |
|---|---|---|---|
| α=0 | 0.7211 | 0.5476 | **-0.1735 ⚠️** |
| α=0.5 | 0.8306 | 0.8341 | +0.0035 (noise) |
| α=1 | 0.8424 | 0.8368 | -0.0056 (noise) |

### H6 결정

- **Concat 채택, SuperNode 보류**
- paper main pipeline anchor 유지: `s04_pipeline_enriched_qcond_a05_mst_kruskal_glm` F1=0.8673 (Concat)
- α=0 SuperNode 손실 mechanism = paper future work (paper limitation 강화)

### 핵심 발견

1. **🚨 α=0 (GAT only) SuperNode 큰 손실 (-0.1735 vs Concat)** — Concat 의 direct query concat vs SuperNode 의 indirect message passing 차이
2. **α=0.5/1 plateau 동등** (Cosine 우세 영역 dilute)
3. **No-filter R: SuperNode α=0.5/1 매우 높음 (0.9564/0.9662)** — Concat 과 거의 동일 selector signal
4. **Filter ΔF1 위계**: α=1 (+0.6073) > α=0.5 (+0.5905) >> α=0 (+0.1748) — Filter 효율도 α 에 강 의존

### 산출물

- Configs (9): `s04_ablation/stagewise/{selector_only/, no_filter/, }supernode_qcond_a{0,05,1}_{selector_only,no_filter,glm}.yaml`
- Script: `scripts/run_supernode_qcond_9cells.sh`
- ckpt: `best_gat_query_supernode_qcond.pt` (val recall@15=0.5737)
- 코드 fix: `src/modules/selectors/ensemble_selector.py:241-243`

---

## SuperNode + Enriched Paper Main Pipeline (2026-04-29, 2 cells)

발사: 2026-04-29 22:57:30 → 완료 23:58 (wall clock ~62min, GPU 0/1 split). 사용자 요청: "Enriched + QCond-SuperNode + MST + PCST + XiYan Filter".

### Stack 구성

| 모듈 | 결정 | 비고 |
|---|---|---|
| Builder | EnrichedHeteroGraphBuilder | description-aware (CSV + tables.json) |
| Encoder | LocalPLMEncoder (MiniLM-L6-v2) | |
| Selector | EnsembleSelector α=0.5 + best_gat_query_supernode_qcond.pt + query_conditioned=true + query_supernode=true | SuperNode QCond 통합 stack |
| Extractor | MSTKruskalExtractor 또는 MSTPCSTUnionExtractor | score_threshold=0.1 |
| Filter | XiYanFilter (provider=glm, model=zai-org/glm-4.7) | |

### 등재 cells (2 신규, all 측정 완료)

| 신규 ID | Extractor | R | P | F1 |
|---------|-----------|---|---|---|
| `s04_pipeline_enriched_supernode_a05_mst_kruskal_glm` | MSTKruskalExtractor | 0.8706 | 0.8591 | **0.8648** |
| **`s04_pipeline_enriched_supernode_a05_mst_pcst_union_glm`** ★ | MSTPCSTUnionExtractor | 0.8742 | 0.8597 | **0.8669** |

### Concat vs SuperNode Paper Main Pipeline 4-cell plateau

| Selector | Extractor | F1 |
|---|---|---|
| Concat | MST Kruskal | **0.8673** (paper main anchor) |
| Concat | Union | 0.8667 |
| SuperNode | MST Kruskal | 0.8648 |
| SuperNode | Union | 0.8669 |

→ **4-cell plateau (F1=0.8648~0.8673, ΔF1 ±0.0025 모두 noise 임계 미달)** — anchor 갱신 X

### 핵심 발견

1. **🚀 Enriched 효과 SuperNode 에서도 발현** (Plain SuperNode α=0.5 0.8341 → Enriched 0.8648/0.8669, ΔF1 +0.03)
2. **SuperNode ≈ Concat plateau 동등** (paper main anchor 유지, Concat + MST Kruskal F1=0.8673)
3. **SuperNode + Union (0.8669) 가 SuperNode 변형 중 best** — Concat + Union (0.8667) 와 거의 동률
4. **사용자 의도 stack F1=0.8669** — paper main pipeline 동률 후보 narrative 가능

### 산출물

- Configs (2): `s04_ablation/pipeline/enriched_supernode_a05_{mst_kruskal, mst_pcst_union}_glm.yaml`

---

## H-A + H-D Ablation (2026-05-04, 13 cells)

발사: 2026-05-04 15:35:08 → 완료 18:45 (wall clock ~3h 10min, GPU 0/1 split). 사용자 결정 narrative resolution 검증 — H-A (Distribution shift) + H-D (Score normalization).

### H-A 11 cells 등재 (best_gat_enriched.pt + α∈{0.0~1.0})

Stack: Enriched Builder + best_gat_enriched.pt (Enriched features 학습, query_conditioned=False) + α + MSTPCSTUnion + XiYan(GLM, num_examples=3) + LLMSQLGenerator(GLM)

| α | Cell ID | F1 | EX |
|---|---|---|---|
| 0.0 | s04_pipeline_t00_enriched_ckpt_alpha_00 | 0.7195 | 0.2177 |
| 0.1 | s04_pipeline_t00_enriched_ckpt_alpha_01 | 0.7820 | 0.2432 |
| 0.2 | s04_pipeline_t00_enriched_ckpt_alpha_02 | 0.8566 | 0.3188 |
| 0.3 | s04_pipeline_t00_enriched_ckpt_alpha_03 | 0.8634 | 0.3292 |
| 0.4 | s04_pipeline_t00_enriched_ckpt_alpha_04 | 0.8648 | 0.3331 |
| 0.5 | s04_pipeline_t00_enriched_ckpt_alpha_05 | 0.8637 | 0.3403 |
| 0.6 | s04_pipeline_t00_enriched_ckpt_alpha_06 | 0.8632 | 0.3403 |
| 0.7 | s04_pipeline_t00_enriched_ckpt_alpha_07 | 0.8625 | 0.3396 |
| 0.8 | s04_pipeline_t00_enriched_ckpt_alpha_08 | 0.8634 | **0.3429** |
| 0.9 | s04_pipeline_t00_enriched_ckpt_alpha_09 | 0.8642 | 0.3383 |
| 1.0 | s04_pipeline_t00_enriched_ckpt_alpha_10 | **0.8651** | 0.3390 |

→ **F1 plateau α∈[0.2,1.0] 유지** (9/10 cells within 0.01 of best F1=0.8651)
→ **GAT contribution 회복 X** — Distribution shift 해소가 plateau 변경 못 함

### H-D 2 cells 등재 (norm 변형, t_00 base)

Stack: t_00 base (qcond_nl3 + α=0.5 + ...) + score_normalization 만 변경
코드 fix: `src/modules/selectors/ensemble_selector.py` `score_normalization` 파라미터

| Variant | Cell ID | F1 | EX |
|---|---|---|---|
| t_00 (minmax, default) | s04_pipeline_enriched_qcond_a05_mst_pcst_union_glm_sql | **0.8657** | **0.3377** |
| norm_none | s04_pipeline_t00_norm_none | 0.8553 | 0.3214 |
| norm_zscore | s04_pipeline_t00_norm_zscore | 0.8325 | 0.2881 |

→ minmax > none > zscore — minmax 가 best, 다른 norm 으로 변경 시 손실

### 시나리오 ② 채택 — 옵션 1 + 옵션 4 통합 narrative

- 기존 "QCondGAT main contribution" → 신규 "4 module Co-Design + Filter dominance"
- Selector = "GAT-floor (α=0 baseline robustness) + Cosine-ceiling (α≥0.2 plateau)"
- Filter = first-class stage, F1 driver (P 회복 +0.64), EX marginal (+0.01)

### 산출물

- Configs (13): `s04_ablation/pipeline/t00_enriched_ckpt_alpha_*.yaml` (11) + `t00_norm_{none, zscore}.yaml` (2)
- Scripts: `scripts/run_h_a_enriched_ckpt_alpha_sweep.sh` + `scripts/run_h_d_norm_variants.sh`
- 코드 수정: `src/modules/selectors/ensemble_selector.py` — score_normalization 파라미터

---

## Wave 4 Filter Ablation (2026-05-04 → 05, 14 cells GLM, 🚀 신규 최고 F1=0.8809)

발사: 2026-05-04 19:08 → 완료 2026-05-05 03:06 (wall clock 7h 58min, GPU 0/1 split). 사용자 결정 옵션 B GLM 통일 — Filter 모듈 14 변형 효과 정량.

### 등재된 cells (14 신규)

Stack: paper main pipeline (Enriched + QCond α=0.5 + qcond_nl3 + MSTPCSTUnion + LLMSQLGenerator(GLM)) + Filter 만 변경

| Cell ID | Filter Variant | F1 | EX |
|---|---|---|---|
| s04_pipeline_wave4_a05_08_tiered_verifier_stack | StackedFilter (Tiered → Verifier) | **0.8809** ★ | 0.3351 |
| s04_pipeline_wave4_a05_22_symverify_reflection_verifier_stacked | SymVerify+Reflection+Verifier 3-stack | 0.8759 | 0.3364 |
| s04_pipeline_wave4_a05_05_tiered_no_tools | TieredBidirectionalAgent (no_tools) | 0.8695 | 0.3429 |
| s04_pipeline_wave4_a05_09_tiered_retry | TieredBidirectionalAgent + ExtractionRetry | 0.8684 | 0.3377 |
| s04_pipeline_wave4_a05_06_tiered_full_tools | TieredBidirectionalAgent (full_tools) | 0.8678 | 0.3422 |
| s04_pipeline_wave4_a05_04_verifier | VerifierFilter (CHESS-style) | 0.8662 | 0.3383 |
| s04_pipeline_wave4_a05_19_symverify_xiyan_repair | SymbolicVerifier + XiYan repair | 0.8650 | 0.3409 |
| s04_pipeline_wave4_a05_21_symverify_xiyan_detect | SymbolicVerifier + XiYan detect | 0.8645 | 0.3370 |
| s04_pipeline_wave4_a05_07_adaptive_depth | AdaptiveDepthFilter (uncertainty-gated) | 0.8633 | **0.3501** ★ |
| s04_pipeline_wave4_a05_02_reflection_1iter | ReflectionFilter (1 iter) | 0.8631 | 0.3429 |
| s04_pipeline_wave4_a05_10_adaptive_retry | AdaptiveDepth + ExtractionRetry | 0.8623 | 0.3422 |
| s04_pipeline_wave4_a05_20_symverify_reflection_repair | SymVerify + Reflection repair | 0.8620 | 0.3396 |
| s04_pipeline_wave4_a05_03_reflection_3iter | ReflectionFilter (3 iter) | 0.8594 | 0.3344 |
| s04_pipeline_wave4_a05_01_adaptive_multi_agent | AdaptiveMultiAgent (Sem+Struct+Skeptic) | 0.8070 ⚠️ | 0.3279 |

→ **F1 최고**: a05_08 Stacked Tiered+Verifier (R=0.8880, P=0.8739, F1=0.8809) — t_00 base F1=0.8657 대비 +0.0152
→ **EX 최고**: a05_07 AdaptiveDepth (EX=0.3501) — 14 cells 중 유일하게 EX > 0.35 (t_00 0.3377 대비 +0.0124)
→ **R 최고**: a05_04 Verifier (R=0.9155) — but P trade-off (0.8220) 로 F1 plateau 내 (0.8662)

### 핵심 발견

1. **StackedFilter sweet spot** — 단일 agent 변형 모두 능가, F1 ceiling 0.8809 갱신
2. **F1 ↔ EX decoupling 일관** — Filter 변형으로 F1 +0.015 가능 but EX 대부분 plateau, a05_07 만 +0.0124
3. **VerifierFilter R↑ trade-off** — recall 회복 약 50% (Δ=+0.0421) but P↓ (-0.0361) 으로 F1 중간
4. **AdaptiveMultiAgent 실패** — Skeptic conservative voting 으로 R 급락 (-0.10), 14 cells 중 유일한 outlier
5. **Reflection iter↑ 역효과** — 1iter > 3iter (P drift), over-correction 으로 false negative 증가

### Config 주의사항

- weight_path: `best_gat_qcond_nl3.pt` (t_00 default)
- query_conditioned: true, alpha: 0.5, top_k: 20
- connectivity_extractor: MSTPCSTUnionExtractor (score_threshold=0.1)
- sql_generator: LLMSQLGenerator (provider=glm, llm_model=zai-org/glm-4.7, temperature=0.0)
- 모든 Filter agent provider=glm + temperature=0.0

### Paper Narrative 함의

- **§3.5 mechanism 갱신 candidate**: Filter design variation (ΔF1=0.0739) 가 Filter on/off (ΔF1=0.63) 의 12% scale — Filter "first-class stage" narrative 한층 보강
- **paper main anchor**: 옵션 A 권장 (t_00 anchor 유지 + a05_08 을 §3.5 evidence)
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Wave 4 Filter Ablation (2026-05-04 → 05)](EXPERIMENT_HISTORY.md).

### 산출물

- Configs (14): `s04_ablation/pipeline/wave4/t00_a05_*.yaml`
- Script: `scripts/run_wave4_filter_ablation_glm.sh`
- Plan: `planning/templates/vivid-sprouting-sunbeam.md`

---

## F-1 Alpha Sweep + H-G Adaptive PCST F-1 (2026-05-05, 17 cells, 🔥 Stage 2 Filter dominance 결정적 evidence)

발사: 2026-05-05 11:16 → 완료 12:57 (wall clock 1h 41min, GPU 0/1 split). 사용자 결정 (DECISIONS 2026-05-04 옵션 A) — F-1 (paper main minus Filter+SQL) 10 cells + H-G (Adaptive Extractor 교체) 7 cells.

### 등재된 cells (17 신규)

**F-1 MSTPCSTUnion (10 신규 + α=0.5 baseline 기존)** — Stack: Enriched Builder + qcond_nl3 ckpt + α + MSTPCSTUnion(score_threshold=0.1) + No Filter + No SQL gen

| α | Cell ID | R | P | F1 |
|---|---|---|---|---|
| 0.0 | s04_pipeline_t00_f1_alpha_00 | 0.7585 | 0.2047 | 0.3224 |
| 0.1 | s04_pipeline_t00_f1_alpha_01 | 0.8535 | 0.2137 | **0.3418** |
| 0.2 | s04_pipeline_t00_f1_alpha_02 | 0.9645 | 0.1728 | 0.2931 |
| 0.3 | s04_pipeline_t00_f1_alpha_03 | 0.9845 | 0.1438 | 0.2509 |
| 0.4 | s04_pipeline_t00_f1_alpha_04 | 0.9905 | 0.1320 | 0.2330 |
| 0.5 | s04_pipeline_enriched_qcond_a05_mst_pcst_union_no_filter (기존) | 0.9927 | 0.1268 | 0.2249 |
| 0.6 | s04_pipeline_t00_f1_alpha_06 | 0.9939 | 0.1240 | 0.2205 |
| 0.7 | s04_pipeline_t00_f1_alpha_07 | 0.9940 | 0.1224 | 0.2180 |
| 0.8 | s04_pipeline_t00_f1_alpha_08 | 0.9943 | 0.1212 | 0.2161 |
| 0.9 | s04_pipeline_t00_f1_alpha_09 | 0.9945 | 0.1208 | 0.2154 |
| 1.0 | s04_pipeline_t00_f1_alpha_10 | **0.9947** | 0.1207 | 0.2153 |

**H-G AdaptivePCST (7 신규)** — Stack: Enriched + qcond_nl3 + α + AdaptivePCST(per-q P80, top-K=20) + No Filter + No SQL gen

| α | Cell ID | R | P | F1 |
|---|---|---|---|---|
| 0.0 | s04_pipeline_t00_hg_adaptive_f1_alpha_00 | 0.5074 | 0.2566 | 0.3408 |
| 0.2 | s04_pipeline_t00_hg_adaptive_f1_alpha_02 | 0.6480 | 0.3142 | 0.4232 |
| 0.4 | s04_pipeline_t00_hg_adaptive_f1_alpha_04 | 0.7017 | 0.3268 | 0.4459 |
| 0.5 | s04_pipeline_t00_hg_adaptive_f1_alpha_05 | 0.7260 | 0.3315 | 0.4552 |
| 0.6 | s04_pipeline_t00_hg_adaptive_f1_alpha_06 | 0.7500 | 0.3392 | 0.4671 |
| 0.8 | s04_pipeline_t00_hg_adaptive_f1_alpha_08 | **0.7834** | 0.3511 | **0.4849** |
| 1.0 | s04_pipeline_t00_hg_adaptive_f1_alpha_10 | 0.7778 | 0.3428 | 0.4759 |

### Spread 정량 + 결과 분기

| Stack | R spread | F1 spread | 분기 |
|-------|----------|-----------|------|
| F-1 MSTPCSTUnion | **0.2362** | **0.1265** | ✅ 분기 1 (>0.05 의 4-5배) |
| H-G AdaptivePCST | **0.2760** | **0.1441** | ✅ 분기 1 (>0.05 의 5-6배) |

→ DECISIONS **분기 1 확정**: Stage 2 Filter precision absorption 결정적 evidence

### Config 주의사항

- weight_path: `best_gat_qcond_nl3.pt` (paper main t_00 default)
- query_conditioned: true, top_k: 20
- score_normalization: minmax (default)
- F-1: connectivity_extractor=MSTPCSTUnionExtractor (score_threshold=0.1)
- H-G: connectivity_extractor=AdaptivePCSTExtractor (base_cost=0.05, fk_cost=0.05, percentile=80.0, top_k=25)
- filter: NoneFilter (LLM-free)
- sql_generator: enabled=false

### 결론 — Filter dominance (단일-stage main mechanism, Stack-dependent Stage 1)

- **🚨 Stage 1 부정**: Extractor MST set saturation 가설 — basic PCST 한정 (H-C partial 결과). paper main 의 MSTPCSTUnion 은 plateau 부재 (R 0.7585 → 0.9947), AdaptivePCST 도 plateau 부재 (R 0.5074 → 0.7834)
- **✅ Stage 2 결정적 evidence**: Filter plateau-region (α∈[0.2,1.0]) F1 spread 6× 압축 (F-1 0.0778 → With-Filter 0.0129), P 균일 elevate (0.12-0.21 → 0.83-0.86)
- **§3.5 narrative 정정**: "2-stage absorption" → **"Filter dominance" single-stage main + Extractor stack-dependent**
- 세부 실행 이력: [EXPERIMENT_HISTORY.md F-1 Alpha Sweep + H-G Adaptive PCST F-1 (2026-05-05)](EXPERIMENT_HISTORY.md).

### 산출물

- Configs (17): `s04_ablation/pipeline/t00_f1_alpha_0[0~10].yaml` (10) + `s04_ablation/pipeline/t00_hg_adaptive_f1_alpha_*.yaml` (7)
- Scripts: `scripts/run_f1_full_alpha_sweep.sh` + `scripts/run_hg_adaptive_f1_sweep.sh`
- 비용: ₩0, wall 1h 41min

---

## Directed Top-K SuperNode GAT 학습 (V-3-ext 단계 2, 2026-05-06, 3 변형)

발사: 2026-05-06 00:20 → 완료 10:37 (wall ~10h 17min, GPU 0/1/2 split). 학위 논문 Part III V-3-ext 단계 2 (사용자 결정 옵션 A epochs=300).

### 등재된 ckpt (3 신규)

Stack (학습): Enriched Builder + DualTowerProjector + DirectedSuperNode (supernode_edge_direction=directed_from_sn) + threshold mode 변형 + 300 epochs

| 변형 | mode | value | best val recall@15 | NAS ckpt path |
|---|---|---|---:|---|
| **PRIMARY p80** | percentile | 80.0 | **0.6097** | `/SSL_NAS/.../best_gat_directed_supernode_p80.pt` |
| **BASELINE topk20** | top_k | 20 | 0.5839 | `/SSL_NAS/.../best_gat_directed_supernode_topk20.pt` |
| **OPTIONAL abstau07** | abs_tau | 0.7 | 0.5805 | `/SSL_NAS/.../best_gat_directed_supernode_abstau07.pt` |

→ 3 변형 모두 epoch 100~150 부터 saturation, 추가 학습 효과 거의 없음
→ p80 가 raw R (0.6133) 거의 회복, topk20 는 raw R (0.6865) 보다 -0.10 underperform, abstau07 는 raw R (0.4857) 능가 +0.10

### Config 주의사항

- **p80**: `supernode_threshold_mode: percentile, supernode_threshold_value: 80.0, supernode_score_normalization: minmax`
- **topk20**: `supernode_threshold_mode: top_k, supernode_topk: 20`
- **abstau07**: `supernode_threshold_mode: abs_tau, supernode_threshold_value: 0.7`
- 공통: `query_supernode: true, supernode_edge_direction: directed_from_sn, in_channels: 384, hidden_channels: 256, num_layers: 3, heads: 4, epochs: 300, batch_size: 8, pos_weight: 100.0`

### 결론 — 시나리오 A 잠정 (Filter Dominance 5번째 축)

- GAT 학습이 selector R 한계 (raw R 0.69 → 학습 0.61) 회복 못 함 — paper §3.5 Filter Dominance narrative 일관
- 단계 3 alpha sweep (paper main stack + 신규 ckpt × α∈{0.0~1.0}) 결과로 시나리오 A/B/C 확정
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Directed Top-K SuperNode GAT 학습 (V-3-ext 단계 2, 2026-05-06)](EXPERIMENT_HISTORY.md).

### 산출물

- Configs (3): `configs/training/train_gat_directed_supernode_{p80, topk20, abstau07}.yaml` (epochs=300)
- Script: `scripts/run_directed_supernode_training.sh`
- 학습 logs (NAS symlink): `logs/train/gat_directed_supernode_*_20260506_*.log`

---

## DSN Phase 1 Alpha Sweep (V-3-ext 단계 3, 2026-05-06, 9 cells, 🎯 시나리오 A 확정)

발사: 2026-05-06 11:11 → 완료 13:36 (wall ~2h 25min, GPU 0/1 9 cells 병렬). paper main t_00 stack + DirectedTopKSuperNodeSelector × 3 신규 ckpt × α∈{0.0, 0.5, 1.0}.

### 등재된 cells (9 신규)

Stack: paper main t_00 (Enriched + MSTPCSTUnion + XiYan GLM + LLMSQLGenerator GLM) + DirectedTopKSuperNodeSelector

| 순위 | Cell ID | R | P | F1 | EX |
|---|---|---|---|---|---|
| 1 | s04_pipeline_dsn_phase1_topk20_alpha_10 | 0.8776 | 0.8547 | **0.8660** | **0.3396** |
| 1 | s04_pipeline_dsn_phase1_abstau07_alpha_10 | 0.8787 | 0.8536 | **0.8660** | 0.3377 |
| 3 | s04_pipeline_dsn_phase1_p80_alpha_10 | 0.8766 | 0.8534 | 0.8648 | **0.3396** |
| 3 | s04_pipeline_dsn_phase1_abstau07_alpha_05 | 0.8753 | 0.8546 | 0.8648 | 0.3364 |
| 5 | s04_pipeline_dsn_phase1_topk20_alpha_05 | 0.8742 | 0.8551 | 0.8645 | 0.3318 |
| 6 | s04_pipeline_dsn_phase1_p80_alpha_05 | 0.8738 | 0.8546 | 0.8641 | 0.3331 |
| 7 | s04_pipeline_dsn_phase1_p80_alpha_00 | 0.7415 | 0.7877 | 0.7639 | 0.2288 |
| 8 | s04_pipeline_dsn_phase1_abstau07_alpha_00 | 0.7315 | 0.7792 | 0.7546 | 0.2484 |
| 9 | s04_pipeline_dsn_phase1_topk20_alpha_00 | 0.6932 | 0.7656 | 0.7276 | 0.2269 |

→ **Best F1 = 0.8660** (topk20_α=1.0 + abstau07_α=1.0 동률) — t_00 base F1=0.8657 대비 +0.0003
→ **Best EX = 0.3396** (p80_α=1.0 + topk20_α=1.0 동률) — t_00 base 0.3377 대비 +0.0019
→ α∈{0.5, 1.0} 6 cells F1 plateau **[0.8641, 0.8660]** spread = 0.0019 (직전 qcond_nl3 plateau 와 동일 패턴)

### 시나리오 A 확정 + Filter Dominance topology-invariant 5번째 축

- **F1 ≤ 0.870** (best 0.8660) → 시나리오 A 확정 (Filter Dominance 5축 격상 candidate)
- graph topology (Concat → directed_from_sn) + selector threshold (top_k vs percentile vs abs_tau) 변경에도 plateau 동일 → Filter mechanism 의 **6번째 evidence (topology-invariant)**
- 3 ckpt 학습 차이 (best val recall 0.5805~0.6097, Δ 0.0292) 가 With-Filter F1 에서 ~26× 압축됨 (Δ 0.0012)

### Config 주의사항

- weight_path: 신규 3 ckpt (`outputs/checkpoints/best_gat_directed_supernode_{p80,topk20,abstau07}.pt`)
- selector: `DirectedTopKSuperNodeSelector` (V-3-ext)
- threshold_mode + threshold_value: ckpt 학습 시 모드 일치 (percentile=80 / top_k=20 / abs_tau=0.7)
- supernode_edge_direction: directed_from_sn (학습 일치)
- score_normalization: minmax
- Extractor: MSTPCSTUnionExtractor(score_threshold=0.1)
- Filter: XiYanFilter(provider=glm, max_iteration=1)
- SQL gen: LLMSQLGenerator(provider=glm, llm_model=zai-org/glm-4.7)

### 결론

- 시나리오 A 확정 → paper §3.5 Filter Dominance topology-invariant 5번째 축 추가
- 학위 논문 Part III V-3-ext 단계 3 Phase 1 완료
- 세부 실행 이력: [EXPERIMENT_HISTORY.md DSN Phase 1 Alpha Sweep (V-3-ext 단계 3, 2026-05-06)](EXPERIMENT_HISTORY.md).

### 산출물

- Configs (9): `s04_ablation/pipeline/dsn_phase1/t00_dsn_*.yaml`
- Script: `scripts/run_dsn_alpha_sweep_phase1.sh`
- 비용: ~₩18-36K, wall 2h 25min

---

## Baseline Correction — qcond_nl3 best val recall@15 = 0.6061 (2026-05-06, analyzer 부산물)

### 정정 record

| Ckpt | best val recall@15 | best epoch | Final R@15 | 학습 일자 |
|---|---:|---:|---:|---|
| **best_gat_qcond_nl3.pt** | **0.6061** | **59** | 0.5958 | 2026-04-23 (추정) |

직전 entries 의 "main baseline qcond_nl3" cross-references 에서 best val recall@15 명시 부재 → 본 entry 가 단일 출처. 인접 ckpt (best_gat_query_supernode_direct 0.5548, best_gat_query_supernode_qcond 0.5737) 와의 유사성 추정 ~0.55 부정확 — 실측 0.6061.

### 함의

- DSN p80 (0.6097) ≈ qcond_nl3 baseline (0.6061), Δ=+0.0036 — 학습 saturation 동등
- DSN topk20 (0.5839) / abstau07 (0.5805) 는 baseline underperform — 이전 narrative 에서 baseline 우월이라는 일부 표현 정정 필요
- BCE-Recall divergence ep23~38 (4 ckpt 모두) — 학습 saturation 결정적 evidence

### 근거

- Analyzer 산출: [notebooks/analysis_results/dsn_oversmoothing_analysis.md §1.1](../notebooks/analysis_results/dsn_oversmoothing_analysis.md)
- 세부 정정 이력: [EXPERIMENT_HISTORY.md Baseline Correction](EXPERIMENT_HISTORY.md)

---

## DSN Phase 2 + Phase 3 4-trial Mitigation Sweep (V-3-ext 단계 5, 2026-05-06 → 05-07, 🎯 시나리오 P3-A 결정적 confirm)

발사: 2026-05-06 17:45 → 완료 2026-05-07 15:29 (병렬 wall ~46h, 학습 합산 ~39h). Mitigation 적용에도 raw R 한계 갱신 X — Filter Dominance 6번째 축 (training-pathology-invariant) 결정적 evidence.

### 등재된 ckpts (3 신규)

Stack: V-3-ext (DSN p80 directed_from_sn + percentile=80) + B5 mitigation 변형

| Ckpt | 학습 entry | AC target | LR config | Best R@15 | Best Epoch | NAS path |
|---|---|---|---|---|---|---|
| best_gat_directed_supernode_p80_b5_mitigation.pt | train_gat_s06.py | fusion | base 1e-4 | 0.6018 | ep157 | /SSL_NAS/.../best_gat_directed_supernode_p80_b5_mitigation.pt (113MB) |
| best_gat_directed_supernode_p80_b5_phase3_directAC.pt | train_gat_s06.py | **gat_out_L_last** | base 1e-4 | 0.5927 | ep51 | /SSL_NAS/.../best_gat_directed_supernode_p80_b5_phase3_directAC.pt (113MB) |
| best_gat_directed_supernode_p80_b5_phase3_layerwiseLR.pt | train_gat_s06.py | fusion | **gat 5e-4 / other 1e-4** | 0.5935 | ep172 | /SSL_NAS/.../best_gat_directed_supernode_p80_b5_phase3_layerwiseLR.pt (113MB) |

### 4-trial mitigation 결과 표 (decreasing R@15)

| 순위 | Variant | Best R@15 | Best Epoch | Δ vs Phase 1 |
|---|---|---:|---|---:|
| **1** | **Phase 1 P80 (no mit)** | **0.6097** | ep91 | (baseline) |
| 2 | Phase 3 #4 (LR x5) | 0.5935 | ep172 | -0.0162 |
| 3 | Phase 3 #3 (Direct AC gat_out_L_last) | 0.5927 | ep51 | -0.0170 |
| 4 | Phase 2 b8 (mit fusion) | 0.6018 | ep157 | -0.0079 |

→ **모든 mitigation variants 가 baseline 보다 lower** — graph topology + B5 mitigation + Direct AC + Layer-wise LR 모두에도 raw R 한계 갱신 X
→ Phase 2 가 mitigation variants 중 best, Phase 3 #3/#4 (더 적극적) 가 오히려 underperform → mitigation 강도 ↑ 가 학습 saturation 더 일찍 induce

### Config 주의사항

- weight_path: V-3-ext 단계 2 학습 ckpt 인 `best_gat_directed_supernode_p80.pt` 와는 별개 (각자 from-scratch 학습)
- 학습 entry: `train_gat_s06.py` (root 가 V-3-ext options forward 추가 2026-05-06)
- 공통: query_supernode=true, supernode_edge_direction='directed_from_sn', supernode_threshold_mode='percentile', supernode_threshold_value=80.0, score_normalization=minmax
- B5 mitigation 공통: pairnorm=pairnorm, initial_residual_alpha=0.2, jumping_knowledge=concat, dual_stream=true, num_layers=2, anti_collapse_weight=0.1, loss_type=listnet
- 변경 차원:
  - Phase 2 b8: anti_collapse_target='fusion' (default)
  - Phase 3 #3: anti_collapse_target='gat_out_L_last'
  - Phase 3 #4: optimizer_layer_wise_lr=true, gat_lr_multiplier=5.0

### 결론 — 시나리오 P3-A 절대 confirm + Filter Dominance 6번째 축

- 4-trial mitigation null effect → paper §3.5 narrative 6번째 축 (training-pathology-invariant) 결정적 evidence
- AC loss 0.62 일관 유지 (Phase 3 #3) → main GAT path 가 collapse 압박 처리 못함의 정량 evidence
- 학위 논문 Part III main contribution 후보 — mechanism deep dive analyzer 위임
- Alpha sweep skip (사용자 명시) → val recall@15 evidence only
- 세부 실행 이력: [EXPERIMENT_HISTORY.md DSN Phase 2 + Phase 3 4-trial Mitigation Sweep (V-3-ext 단계 5, 2026-05-06 → 05-07)](EXPERIMENT_HISTORY.md).

### 산출물

- Configs (3): `train_gat_directed_supernode_p80_b5_{mitigation, phase3_directAC, phase3_layerwiseLR}.yaml`
- 학습 entry 확장: `src/train_gat_s06.py`
- 비용: ₩0, 학습 wall 합산 ~39h, 병렬 wall ~46h

---

## DSN Mitigation v2 3-trial Sweep (V-3-ext 단계 6, 2026-05-07 → 05-08, 🎯 시나리오 V2-A 확정)

발사: 2026-05-07 16:35 → 완료 2026-05-08 13:54 (병렬 wall ~21h, 3개 동시 GPU 0). 사용자 결정 옵션 A. 7-trial mitigation 통합 결과: Filter Dominance 6번째 축 결정적 evidence.

### 등재된 ckpts (3 신규)

Stack: V-3-ext (DSN p80) + B5 mitigation + Mitigation v2 변형

| Ckpt | Mitigation v2 옵션 | Best R@15 | Best Epoch | NAS path |
|---|---|---:|---|---|
| best_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.pt | drop_message_p=0.2 | 0.5974 | ep157 | /SSL_NAS/.../v2_drop_message.pt (113MB) |
| best_gat_directed_supernode_p80_b5_mitigation_v2_layernorm.pt | use_layernorm_pre_softmax=true | **0.6011** ★ | ep289 | /SSL_NAS/.../v2_layernorm.pt (113MB) |
| best_gat_directed_supernode_p80_b5_mitigation_v2_sum_aggr.pt | aggregation_type='sum' | 0.5761 | ep194 | /SSL_NAS/.../v2_sum_aggr.pt (113MB) |

### 7-trial 누적 결과 (decreasing R@15)

| 순위 | Variant | Best R@15 | Δ vs Phase 1 |
|---|---|---:|---:|
| **1** | **Phase 1 P80 (no mit)** | **0.6097** | (baseline) |
| 2 | Phase 2 b8 (mit fusion) | 0.6018 | -0.0079 |
| 3 | **v2 #3 LayerNorm pre-softmax** | **0.6011** ★ | -0.0086 |
| 4 | v2 #1 DropMessage | 0.5974 | -0.0123 |
| 5 | Phase 3 #4 (LR x5) | 0.5935 | -0.0162 |
| 6 | Phase 3 #3 (Direct AC) | 0.5927 | -0.0170 |
| 7 | v2 #2 Sum Aggregation | 0.5761 | -0.0336 |

→ **모든 mitigation variants 가 baseline 보다 lower** — graph topology + B5 mitigation + Direct AC + LR x5 + DropMessage + LayerNorm + Sum Aggr 모두에도 raw R 한계 갱신 X
→ **v2 #3 LayerNorm 가 mitigation variants 중 best** — Phase 2 (0.6018) 와 거의 동등
→ **시나리오 V2-A 절대 confirm**: Filter Dominance 6번째 축 (training-pathology-invariant) 7-trial evidence 결정적

### Config 주의사항 (training)

- 학습 entry: `src/train_gat_s06.py` (Mitigation v2 옵션 forward 추가됨)
- 공통 V-3-ext (DSN p80): query_supernode=true, supernode_edge_direction='directed_from_sn', percentile=80, score_normalization=minmax
- 공통 B5 mitigation: pairnorm=pairnorm, IR α=0.2, JK=concat, dual_stream=true, L=2, AC=0.1, ListNet, AC target=fusion
- 변경 차원 (Mitigation v2):
  - v2 #1: drop_message_p=0.2 (model param)
  - v2 #3: use_layernorm_pre_softmax=true (model param)
  - v2 #2: aggregation_type='sum' (model param)

### 결론 — 시나리오 V2-A 확정 + Filter Dominance 6번째 축 7-trial 강화

- 7-trial × 4 mitigation 카테고리 (graph topology / B5 mitigation / loss-level / model-level) 모두 raw R 한계 갱신 X
- v2 #3 LayerNorm partial recovery (mech(ii) edge softmax over-concentration) 신호 — but baseline 미달
- v2 #2 Sum Aggregation 압도적 underperform (-0.0336) — mech(i) Aggregation collapse 직접 evidence
- 학위 논문 Part III main contribution 7-trial mechanism finding base — analyzer 위임
- 세부 실행 이력: [EXPERIMENT_HISTORY.md DSN Mitigation v2 3-trial Sweep (V-3-ext 단계 6, 2026-05-07 → 05-08)](EXPERIMENT_HISTORY.md).

### 산출물

- Configs (3): `train_gat_directed_supernode_p80_b5_mitigation_v2_{drop_message, layernorm, sum_aggr}.yaml`
- Scripts: `scripts/run_mitigation_v2_sweep.sh` + `run_mitigation_v2_layernorm_resume.sh`
- 학습 entry 확장: `src/train_gat_s06.py` + `src/models/gat_network_v2.py`
- 비용: ₩0, 병렬 wall ~21h

---

## Filter Module Confirmation Sweep v2 — 9-cell with Evidence Forward (GLM 4.7 + EX) (2026-05-13, 🎯 Filter-Invariant F1 + EX 양쪽 확정)

발사: 2026-05-13 01:33:31 KST → 종료: 2026-05-13 08:56:01 KST (wall **7h22min30s**, 9-cell PARALLEL). v1 (5/12, no evidence) anchor EX gap 21.91%p vs Baseline B1' 의 dominant 원인 (LLMSQLGenerator 의 evidence 미사용) 진단 + fix 후 v2 재측정. v1 결과 archive.

### 등재된 9 cell — v2 최종 R/P/F1/EX (F1 정렬)

| 순위 | Cell | Filter | R | P | **F1** | **EX** |
|---|---|---|---:|---:|---:|---:|
| **1** | **c4_stacked_glm** | Stacked (Refl→Verif) | 0.8781 | 0.8629 | **0.8704** ⭐best F1 | 0.5267 |
| 2 | c7_bidirectional_glm | TieredBidirectional | 0.8923 | 0.8433 | 0.8671 | **0.5287** ⭐best EX |
| 3 | **c0_xiyan_glm_sql** (anchor) | XiYan | 0.8706 | 0.8596 | **0.8651** | 0.5202 |
| 4 | c1_reflection_glm | Reflection (1 iter) | 0.8907 | 0.8407 | 0.8650 | 0.5222 |
| 4 | c5_symverify_glm | SymbolicVerifier | 0.8717 | 0.8585 | 0.8650 | 0.5222 |
| 6 | c2_verifier_glm | Verifier | **0.9163** ⭐best R | 0.8161 | 0.8633 | 0.5267 (biggest ΔEX +0.1916) |
| 7 | c6_adaptive_depth_glm | AdaptiveDepth | 0.8786 | 0.8484 | 0.8632 | 0.5248 |
| 8 | c3_adaptive_multi_agent_glm | AdaptiveMultiAgent ⚠️outlier | 0.7734 | 0.8373 | **0.8041** | 0.5189 |
| 9 | c8_no_filter (baseline) | None | **0.9927** | 0.1269 | 0.2250 | 0.5156 |

### v1 → v2 핵심 변화

- **anchor EX**: 0.3396 → **0.5202** (+0.1806, gap 의 82% 회수)
- **best EX**: 0.3468 → **0.5287** (C1 → C7, +0.1819)
- F1 변화: sub-noise (±0.0031 max) — evidence 가 schema linking F1 무영향 (의도)
- C3 outlier F1=0.8041 무변동 — multi-agent vote pathology 확정 outlier
- Baseline B1' Full (55.87%) 대비 anchor v2 gap: **-3.85%p** (v1 -21.91%p 에서 회수)

### Stack 공통 (C0~C8)

- Builder: EnrichedHeteroGraphBuilder
- Selector: EnsembleSelector + qcond_nl3.pt + α=0.5 (QCond Concat)
- Extractor: MSTKruskalExtractor (score_threshold=0.1)
- LLM (C0~C7): GLM 4.7 (zai-org/glm-4.7, provider="glm", T=0.0)
- SQL Generator (C0~C8 모두 활성): LLMSQLGenerator + GLM 4.7
- C8: filter.name="None" (LLM-free filter, sql_gen 만)

### 🎯 시나리오 재확정 — Filter-Invariant (F1 + EX 양쪽)

- **7 LLM filter (C3 outlier 제외) F1 spread = 0.8704 − 0.8632 = 0.0072** → **Filter-Invariant** ★
- **7 LLM filter EX spread = 0.5287 − 0.5202 = 0.0085** → **Filter-Invariant** ★ (양쪽 sub-noise)
- v1 의 F1 spread 0.0116 (Filter-Modest) → v2 0.0072 (Filter-Invariant) 더 좁아짐
- v1 narrative "EX 측 effect ≈ zero" 폐기 — v2 의 +18.06%p EX 회복으로 Filter Dominance 의 EX axis 도 결정적 evidence 확보

### 핵심 발견 (5 — HISTORY 자세한 narrative)

1. **Evidence fix +18.06%p EX**: anchor EX 0.3396 → 0.5202 (Baseline B1' 의 93% 도달)
2. **C7 best EX=0.5287** (anchor +0.0085) — TieredBidirectional 의 evidence-aware restoration
3. **C4 best F1=0.8704** (anchor +0.0053, cost 6× — 비용 대비 marginal lift)
4. **C3 outlier 확정** — F1=0.8041 evidence 와 무관 (multi-agent vote pathology)
5. **C8 no_filter EX=0.5156 ≈ anchor v2 EX=0.5202** — F1=0.2250 (over-include) 가 GLM 4.7 SQL gen 으로 흡수, Filter 의 EX 측 lift 는 +0.0046 marginal (단 F1 측 lift 0.6401 결정적)

### 비용 / 운영 (v2)

- 학습 wall: **7h22min30s** (v1 9h26min 대비 2h 단축)
- 비용: ~₩수십만 (총 token in=82.0M, out=2.4M, v1 과 유사)
- GPU 0 max memory: 7.7GB (9 cell × ~850MB, 24GB 의 32%)
- 모든 9 cell 정상 완료, fail 0
- v1 archive: `outputs/experiments/s04_ablation/pipeline/filter_sweep_v1_no_evidence/`

### 산출물

- Configs (9): `configs/experiments/s04_ablation/pipeline/filter_sweep/c{0..8}_*.yaml`
- Sweep script: `scripts/run_filter_sweep_glm.sh` (PARALLEL 9-cell wait-based)
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Filter Module Confirmation Sweep](EXPERIMENT_HISTORY.md)

### 후속 (즉시 trigger 가능)

- **Analyzer 위임**: `notebooks/analysis_results/filter_sweep_glm_9cell.md` 신규 — 9-cell × R/P/F1/EX 매트릭스 + anchor↔best contrast + EX-F1 decoupling 발견 + Filter Dominance 7번째 축 narrative
- **Planner 위임 (analyzer 후)**: paper §3.5 통합 + anchor F1=0.8663 ranking + main pipeline anchor 유지/변경 결정 + DECISIONS prepend

---

## SGBE Phase 3-5 — Score-Gated Batch Extractive Filter (2026-05-12, 🚀 Phase 3 launch active)

근거: DECISIONS 2026-05-12 SGBE Chain Phase 3 Launch Trigger + SGBE Filter 채택. Module:filters 가 step_mode 3-mode + score_collapse_threshold (Option A default 0.05) 옵션 추가 (16/16 smoke PASSED). Phase 3 launch active (21:44 KST, PID 1241364).

### Anchor stack (paper main mirror)

- **Builder**: `EnrichedHeteroGraphBuilder` (include_views=False, run_leiden_clustering=True)
- **NLQ Encoder**: `LocalPLMEncoder` — `sentence-transformers/all-MiniLM-L6-v2`
- **Projection**: enabled=False
- **Seed Selector**: `EnsembleSelector` — `weight_path=outputs/checkpoints/best_gat_qcond_nl3.pt`, `alpha=0.5`, `top_k=20`, `query_conditioned=true`, `encoder_type=plm`
- **Connectivity Extractor**: `MSTKruskalExtractor` — `score_threshold=0.1`
- **Filter**: `ScoreGatedBatchExtractiveFilter` (SGBE) — `provider=glm`, `model_name=zai-org/glm-4.7`, `temperature=0.0`, `fk_pk_hardcode=true`, `num_examples=3`
- **Post-processing**: `auto_join_keys=true`

### 3 master config + sweep matrix

| Phase | Master config | Sweep dimension | Status |
|---|---|---|---|
| **Phase 3 θ calibration** (9-cell, LLM 없음) | `configs/experiments/s04_ablation/pipeline/sgbe/sgbe_calibration_base.yaml` | θ_keep ∈ {0.50, 0.55, 0.60} × θ_drop ∈ {0.20, 0.25, 0.30} + `step_mode="step_0+1"` | 🚀 active (21:44 KST) |
| **Phase 4 Final SGBE** (best θ, GLM 4.7) | `configs/experiments/s04_ablation/pipeline/sgbe/sgbe_final.yaml` | Phase 3 best θ + SQL generator + `step_mode="step_0+1+2"` | ⏸ Phase 3 후 |
| **Phase 5 Step Ablation** (3 cells) | `configs/experiments/s04_ablation/pipeline/sgbe/sgbe_step_ablation_base.yaml` | `step_mode` ∈ {`step_0`, `step_0+1`, `step_0+1+2`} | ⏸ Phase 4 후 |

### Sweep scripts

- `scripts/run_sgbe_calibration.sh` — Phase 3 9-cell sweep (base yaml + sed override → temp yaml). Pre-check: SGBE step_mode option 존재 검증.
- `scripts/run_sgbe_final_ablation.sh` — Phase 4 (Final) + Phase 5 (3 step ablation) 통합. STEP_MODES 정식 값 ('step_0', 'step_0+1', 'step_0+1+2').
- `scripts/run_sgbe_ablation.sh` (5/12 신규) — Phase 5 only 분리, 3 cell 순차.

### 후속

- **module:filters**: SGBE 의 `skip_llm` + `step_mode` option 추가
- **root (option 추가 후)**: Phase 3 launch → analyzer best θ 결정 → Phase 4+5 launch
- **analyzer**: `notebooks/analysis_results/sgbe_filter_results.md` 신규
- **planner**: Filter Dominance 7번째 axis (Filter-invariance) + 8번째 axis (Score-Gated Hybrid) candidate

---

## DSN Mitigation V5 Tier 1+2 4-Direction (V5-A GATE / V5-B GCNII L=2/4/6 / V5-C Full AERO) (V-3-ext 단계 9, 2026-05-12, 🚧 코드 준비 + Launch 보류)

근거: DECISIONS 2026-05-12 (V5 Mitigation Plan — Tier 1+2 4 Direction). 본 entry 는 Launch 보류 placeholder — module 세션 review 후 launch 결정.

### 등재 예정 ckpt (5 신규, Launch 보류)

| Ckpt | gat_layer_type / 옵션 | num_layers | NAS path (예정) |
|---|---|---:|---|
| **V5-A GATE** | `gat_layer_type='gate'` (att + att_self 분리) | 2 | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_v5a_gate.pt` |
| **V5-B GCNII L=2** | `gat_layer_type='gcnii'`, `gcnii_beta_lambda=0.5` (Identity Mapping) | 2 | `best_gat_directed_supernode_p80_v5b_gcnii_L2.pt` |
| **V5-B GCNII L=4** | gcnii + num_layers=4 | 4 | `best_gat_directed_supernode_p80_v5b_gcnii_L4.pt` |
| **V5-B GCNII L=6** | gcnii + num_layers=6 | 6 | `best_gat_directed_supernode_p80_v5b_gcnii_L6.pt` |
| **V5-C Full AERO** | `gat_layer_type='aero_full'`, `aero_hop_attention=true`, JK='none' | 2 | `best_gat_directed_supernode_p80_v5c_aero_full.pt` |

### Stack

- V5-A/B (L=2/4/6): V-3-ext (DSN p80 directed_from_sn + percentile=80) + B5 mitigation (PN+IR α=0.2+JK=concat+Dual-Stream+AC=0.1+ListNet) + V5 architectural intervention
- V5-C: 위와 동일 단 `jumping_knowledge='none'` (Hop Attention 가 cross-layer aggregation 대체)

### 차별점 (V4 와)

- V5-A: row-stochasticity **유지** (softmax 그대로) — Conservation Law 만 수정. Wu 2023 JSR<1 가정 violation 가 아닌 task-irrelevant aggregation switch-off 표적.
- V5-B: row-stochasticity **유지** + Identity Mapping (β_l = log(λ/l+1)) 으로 gradient flow upper bound 안정화. Initial Residual (α=0.2) + Identity Mapping 동시 적용 — Chen 2020 GCNII 의 핵심 두 component.
- V5-C: V4-B (Softplus + Sym-Norm) **+ Hop Attention** (V4-B 의 H10.1c 직접 표적). cumulative attention residual 차단 → AERO Theorem 3 의 full architecture 실현.

### 11-trial mitigation 매트릭스 (V5 결과 합산 후 갱신 예정)

(현재 10-trial — V4 결과 까지. V5 5 ckpt 종료 시 15-trial 매트릭스로 확장.)

### 후속

- **module:selectors (또는 신규 module:models)**: V5-A/B/C 클래스 + Hop Attention forward code review → launch 결정. Root 영역 위반으로 본 root 가 직접 launch 불가.
- **analyzer (launch 완료 후)**: 15-trial 매트릭스 + Layer 1/2/3 evidence 재정량 + dsn_mitigation_v5_4dir.md 신규
- **analyzer (V5-D-1, 별도 chain)**: PLM Lower Bound 진단 (Plain vs Enriched c_L0/c_L3) → outputs/analysis/v5_d1_plm_lower_bound_diagnostic/
- **planner (analyzer 후)**: narrative pivot 결정 + 5 over-smoothing planning 문서 갱신 + paper §3.5

---

## DSN Mitigation V4 Architectural Intervention (V4-A LN+GIN Combo + V4-B AERO Softplus) 학습 (V-3-ext 단계 8, 2026-05-11 → 05-12, 🎯 시나리오 V4-Combo-Null 확정)

발사: 2026-05-11 23:23 KST → 종료: V4-B 09:05 (wall 9h 38min) + V4-A 10:14 (wall 10h 47min). DECISIONS 2026-05-11 §V4 채택 + `planning/oversmoothing_solution_methodology_2026-05-11_apa.md` §C-1 + §C-2 — combo 가설 (mech(ii-b) softmax-weighted-mean DOMINANT) 의 정량 검증. **결과: 둘 다 baseline 0.6097 미달 → mech(ii-b) DOMINANT 5/5 absolute confirm.**

### 등재된 ckpt (2 신규)

| Ckpt | 옵션 | Best R@15 | Best Epoch | NAS path |
|---|---|---:|---:|---|
| **V4-A LN+GIN Combo** | gat_layer_type='lngin' (Pre-softmax LN + GIN MLP) | **0.5929** | **ep259** | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_v4a_lngin_combo.pt` (**257MB**) |
| **V4-B AERO Softplus** | gat_layer_type='softplus' + softplus_symmetric_norm=true | **0.5951** | **ep58** | `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_v4b_aero.pt` (**113MB**) |

### Stack

V-3-ext (DSN p80 directed_from_sn + percentile=80) + B5 mitigation (PN+IR α=0.2+JK=concat+Dual-Stream+L=2+AC=0.1+ListNet) + V4 architectural intervention.

### 10-trial mitigation 통합 매트릭스 (최종, V4 결과 반영)

| 순위 | Variant | Best R@15 | Best Epoch | Δ vs Phase 1 |
|------|---------|-----------|------------|--------------|
| **1** | **Phase 1 P80 (no mit)** | **0.6097** | ep91 | (baseline) |
| 2 | Phase 2 b8 (B5 mit fusion) | 0.6018 | ep157 | -0.0079 |
| 3 | v2 #3 LayerNorm pre-softmax | 0.6011 | ep289 | -0.0086 |
| 4 | v2 #1 DropMessage | 0.5974 | ep157 | -0.0123 |
| 5 | v3 #1 GIN-style aggregation | 0.5954 | ep246 | -0.0143 |
| **6** | **🆕 V4-B AERO Softplus** | **0.5951** | **ep58** | **-0.0146** |
| 7 | Phase 3 #4 (LR x5) | 0.5935 | ep172 | -0.0162 |
| 8 | Phase 3 #3 (Direct AC) | 0.5927 | ep51 | -0.0170 |
| **9** | **🆕 V4-A LN+GIN Combo** | **0.5929** | **ep259** | **-0.0168** |
| 10 | v2 #2 Sum Aggregation | 0.5761 | ep194 | -0.0336 |

### 🎯 시나리오 V4-Combo-Null 확정

- ❌ V4-Combo-Win (≥0.6097): 미충족
- **✅ V4-Combo-Null** (둘 다 <0.6097): **확정** — mech(ii-b) DOMINANT 4/5 → **5/5 absolute confirm** + Filter Dominance 6번째 축 narrative 결정적 강화
- ❌ V4-Mixed: 미충족

핵심 발견:
1. V4-A combo Δ=-0.0168 — partial mit 합산이 새 회복 만들지 못함 (destructive interference 가능성)
2. V4-B row-stochasticity 파괴 Δ=-0.0146 — Wu et al. 2023 의 over-smoothing 회피 이론이 본 schema linking stack 에서는 실증 안 됨
3. Best epoch 분포 (V4-A ep259 / V4-B ep58) 모두 직전 8-trial 의 ceiling 흡수 패턴과 일관

### 비용 / 운영

- 학습 wall: V4-B 9h 38min + V4-A 10h 47min (병렬, max=10h 47min)
- 비용 ₩0 (LLM-free)
- ckpt NAS: V4-A **257MB** + V4-B **113MB**
- 자동 후속: sweep script 가 `dsn_oversmoothing_analysis.py --max_queries 50 --skip_step1 --skip_step2` 자동 호출 (10:15 KST start) — attention/cosine OK, ⚠️ grad_flow 일부 미호환

### 산출물

- Configs (2): `configs/training/dsn/train_dsn_p80_v4{a_lngin_combo,b_aero}.yaml`
- 모델 확장: `src/models/gat_network_v2.py` (`LNGINGATv2Conv` + `SoftplusGATv2Conv` + `GAT_LAYER_TYPES` + V4 forwarding/validation)
- 학습 entry: `src/train_gat_s06.py` (V4 kwargs forwarding line ~243-258)
- 분석 확장: `src/analysis/dsn_oversmoothing_analysis.py` (CKPTS v4a/v4b 등록 + V4 ckpt 분기)
- Sweep script: `scripts/run_v4_mitigation_sweep.sh`
- 세부 실행 이력: [EXPERIMENT_HISTORY.md DSN Mitigation V4 Architectural Intervention 학습 (V-3-ext 단계 8, 2026-05-11)](EXPERIMENT_HISTORY.md)

### 후속

- **Analyzer 위임 (즉시 trigger 가능)**: `notebooks/analysis_results/dsn_mitigation_v4_combo.md` 신규 (V4-A=0.5929 / V4-B=0.5951 정량 + 10-trial matrix + mech(ii-b) 5/5 absolute confirm)
- **Planner 위임 (analyzer 후)**: advisor briefing + root cause report + paper §V.5.4 main finding narrative integration

---

## DSN Mitigation v3 #1 GIN-style aggregation 학습 (V-3-ext 단계 7, 2026-05-08 → 05-09, 🎯 시나리오 V3-A 1차 confirm)

발사: 2026-05-08 17:12 → 완료 2026-05-09 04:51 (wall ~11h 39min, GPU 0). Phase 1 deep dive (A1+A2+A3) 후 Phase 2 GIN 구현 + Phase 3 학습. 8-trial mitigation 통합 결과: 모든 variants baseline 미달.

### 등재된 ckpt (1 신규)

| Ckpt | 옵션 | Best R@15 | Best Epoch | NAS path |
|---|---|---:|---|---|
| best_gat_directed_supernode_p80_b5_mitigation_v3_gin.pt | aggregation_type='gin' | **0.5954** | ep246 | /SSL_NAS/.../v3_gin.pt (140MB) |

### 8-trial 통합 누적 (decreasing R@15)

| 순위 | Variant | Best R@15 | Δ vs Phase 1 |
|---|---|---:|---:|
| **1** | **Phase 1 P80 (no mit)** | **0.6097** | (baseline) |
| 2 | Phase 2 b8 (mit fusion) | 0.6018 | -0.0079 |
| 3 | v2 #3 LayerNorm pre-softmax | 0.6011 | -0.0086 |
| 4 | v2 #1 DropMessage | 0.5974 | -0.0123 |
| **5** | **🆕 v3 #1 GIN-style aggregation** | **0.5954** | **-0.0143** |
| 6 | Phase 3 #4 (LR x5) | 0.5935 | -0.0162 |
| 7 | Phase 3 #3 (Direct AC) | 0.5927 | -0.0170 |
| 8 | v2 #2 Sum Aggregation | 0.5761 | -0.0336 |

### Config 주의사항

- 학습 entry: `src/train_gat_s06.py` (aggregation_type='gin' forward)
- 공통 V-3-ext (DSN p80) + B5 mitigation: Phase 2 b8 와 동일
- 변경 차원: aggregation_type='gin' — `_make_gin_conv` factory + `HeteroConv aggr='mean'` fix + 18 inner GINConvs
- attention 자체 부재 (mech(ii-a) 측정 X), mech(ii-b) propagation pathology 직접 검증

### 결론 — 시나리오 V3-A 1차 confirm + mech(ii-b) DOMINANT 후보

- 8-trial × 5 mitigation 카테고리 (graph topology / B5 mitigation / loss-level / model-level / aggregation-level) 모두 baseline 미달
- GIN 가 mit variants 5위 — mech(ii-a) 부재해도 ceiling 유사 → mech(ii-b) aggregation family limitation 강화
- v2 #3 LayerNorm (mech(ii-a) partial mitigation) > GIN (mech(ii-b) 직접) → mech(ii-a) 우위 잠정
- analyzer Phase 4 deep dive 후 mech(ii-a)/(ii-b) sub-mechanism 정식 확정
- 세부 실행 이력: [EXPERIMENT_HISTORY.md DSN Mitigation v3 #1 GIN-style aggregation 학습 (V-3-ext 단계 7, 2026-05-08 → 05-09)](EXPERIMENT_HISTORY.md).

### 산출물

- Config: `train_gat_directed_supernode_p80_b5_mitigation_v3_gin.yaml`
- 모델 확장: `src/models/gat_network_v2.py` (_make_gin_conv + AGGREGATION_TYPES)
- Smoke test: `src/modules/selectors/tests/test_mitigation_v3.py` (7/7 통과)
- 비용: ₩0, wall ~11h 39min

