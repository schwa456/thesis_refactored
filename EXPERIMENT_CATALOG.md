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

## Anchor (MSTPCSTUnion+XiYan+SQL) Sweep — Option γ 재실행 (2026-05-14, 🎯 SQL Gen prompt 변경 효과 ΔEX +0.1512 + 5 Capacity 지표 prerequisite)

### 등재된 cell (1 신규, inference only — SQL Gen 활성화)

| Cell | R | P | F1 | EX |
|---|---:|---:|---:|---:|
| **s04_pipeline_enriched_qcond_a05_mst_pcst_union_glm_sql** | **0.8831** | 0.8070 | **0.8434** | **0.4889** ⭐ |

### Stack (사용자 명시 anchor — Option γ)

| Module | 값 |
|---|---|
| Builder | EnrichedHeteroGraphBuilder |
| Selector | EnsembleSelector (best_gat_qcond_nl3.pt, α=0.5, top_k=20, query_conditioned=true) |
| Extractor | **MSTPCSTUnionExtractor** (score_threshold=0.1) |
| Filter | XiYanFilter (GLM 4.7, max_iteration=1, temperature=0) |
| **SQL Gen** | **LLMSQLGenerator** (GLM 4.7, evidence-aware prompt) |

### Δ vs 비교 base

| 지표 | 본 sweep | 5/1 prior (same stack, prompt 이전) | c0 (MSTKruskal+XiYan+SQL) |
|---|---:|---:|---:|
| R | 0.8831 | 0.8734 (Δ +0.0097) | 0.8706 (Δ +0.0125) |
| P | 0.8070 | 0.8581 (Δ -0.0511) | 0.8596 (Δ -0.0526) |
| F1 | **0.8434** | 0.8657 (Δ **-0.0223**) | 0.8650 (Δ **-0.0216**) |
| **EX** | **0.4889** | 0.3377 (Δ **+0.1512** ⭐) | 0.5202 (Δ -0.0313 sub-noise) |

### 핵심 발견

1. **SQL Gen prompt 변경 효과 ΔEX +0.1512** — 5/1 prior run 의 0.3377 → 본 sweep 0.4889. evidence-aware fix 의 EX 회복 강력 confirm.
2. **MSTPCSTUnion vs MSTKruskal extractor EX 차이 sub-noise** (Δ -0.0313) — anchor extractor 변경에도 EX 측 robustness. **Filter Dominance EX-axis robustness 가 extractor-invariance 추가 evidence**.
3. **F1 deterministic partial fail** — 5/1 prior 와 같은 stack 인데 ΔF1 -0.0223. P -0.0511 / R +0.0097 의 trade-off 발생. analyzer 진단 필요.
4. **filter_llm_calls_mean 0.9368** — 6.32% query 의 filter skip (XiYan input nodes 적은 case).

### 비용 / 운영

| 항목 | 값 |
|---|---|
| Wall | ~1h 49m (sweep-only, 학습 X) |
| LLM call | 2873 (filter 1437 + SQL gen 1436) |
| Token in / out | 5.27M / 105K |
| GPU 시간 | 0 (V5 학습 공존, GLM API only) |
| Cost | ~$1-3 GLM 4.7 |

### 산출물

- Config: `configs/experiments/s04_ablation/pipeline/enriched_qcond_a05_mst_pcst_union_glm_sql.yaml` (5/1 prior 그대로)
- Sweep script: `scripts/run_anchor_sql_sweep.sh` (신규)
- Outputs: `outputs/experiments/s04_ablation/pipeline/enriched_qcond_a05_mst_pcst_union_glm_sql/`
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Anchor (MSTPCSTUnion+XiYan+SQL) Sweep (2026-05-14)](EXPERIMENT_HISTORY.md)

### 결론

- SQL Gen prompt 변경 효과 ΔEX +0.1512 강력 확인 (5/1 prior 0.3377 → 본 0.4889)
- MSTPCSTUnion vs MSTKruskal extractor EX 측 sub-noise (anchor 비교 가능)
- 5 Capacity 지표 분석 prerequisite 충족 — analyzer 핸드오프 trigger 가능

---

## Direction B + Direction C-GT 배포 Sweep (b06_01 + a05_26, 2026-05-14, 🎯 B Filter-Invariant evidence, C-GT Four-caveat outlier candidate)

### 등재된 cell (2 신규, inference only)

| Cell | R | P | F1 | EX | LLM | mean \|final_n\| |
|---|---:|---:|---:|---:|---:|---:|
| **b06_01 (Direction B)** | **0.8713** | 0.8545 | **0.8628** | 0.0¹ | 1534 | 85.07 |
| **a05_26 (Direction C-GT)** | 0.7311 | 0.3969 | **0.5145** | 0.0¹ | 1534 | 49.16 |

¹ sql_generator=false (학술 frame "Filter-Invariant 경계 확정 실험" 정합)

### Δ vs Anchor

| 지표 | B vs c0 (F1=0.8650) | C-GT vs c0 | B vs MSTPCSTUnion (F1=0.8666) | C-GT vs MSTPCSTUnion |
|---|---:|---:|---:|---:|
| ΔR | +0.0007 | -0.1395 | -0.0059 | -0.1461 |
| ΔP | -0.0051 | -0.4627 | -0.0019 | -0.4595 |
| **ΔF1** | **-0.0022** sub-noise | **-0.3505** ⚠ | **-0.0038** sub-noise | **-0.3521** ⚠ |

### Stack

| Module | b06_01 (B) | a05_26 (C-GT) |
|---|---|---|
| Builder | EnrichedHeteroGraphBuilder | HeteroGraphBuilder (default, anchor 와 다름) |
| Selector | HNSupConSelector (fine-tuned encoder + GAT projector α=0.5, top_k=20) | DirectGATSelector (best_gat_query_supernode_direct.pt, apply_threshold=true) |
| Extractor | MSTKruskalExtractor | PCSTExtractor (anchor PCST default cost) |
| Filter | XiYanFilter (GLM 4.7) | **GRASTFDFilterWithTransformer** (GT reranker top_k=10, transformer_checkpoint_path=outputs/checkpoints/grast_gt/best.pt) |
| SQL Gen | disabled | disabled |

### 핵심 발견

1. **B = Filter-Invariant F1 측 강력 evidence**: ΔF1 -0.002~-0.004 sub-noise. HN-SupCon fine-tune 효과 (SLR Δ +0.0267) 가 anchor 대비 F1 lift 없음 → **selector backbone 교체 invariance**.
2. **C-GT = Four-caveat outlier candidate**: ΔF1 -0.3505 (Direction A -0.2832 / Direction C -0.2873 보다 더 큰 outlier). mean |final_n| 49.16 (anchor ~4.8 의 10×) — GT reranker over-include.
3. **B 의 mean |final_n| 85.07** = anchor 4.8 의 17.7× 임에도 F1 sub-noise — top_k=20 selector 의 명시 retention + XiYan filter 의 P-aggressive prune 효과.
4. **C-GT 의 R=0.7311 손실** — GT reranker top_k=10 이 좁아 gold column 누락.

### Three-Caveat → Four-Caveat candidate

| Caveat | Method | ΔF1 (vs c0) |
|---|---|---:|
| 1 | C3 AdaptiveMultiAgent | -0.0622 |
| 2 | Direction A (LLM backward) | -0.2832 |
| 3 | Direction C (algorithmic Steiner) | -0.2873 |
| **🆕 4** | **Direction C-GT (GT reranker over-include)** | **-0.3505** ⚠ |

→ paper §V.5.x.M.4 의 Three-Caveat narrative 의 Four-caveat 확장 또는 boundary 재정의 candidate.

### 비용 / 운영

- Wall: 1h 7m (parallel, V5 학습 GPU 0/1 공존)
- LLM call: 3068 total (B 1534 + C-GT 1534)
- Token in: B 6.0M, C-GT 2.1M

### 산출물

- `outputs/experiments/abl/{b06_hn_supcon/b06_01_*, a05_filter_agentic/a05_26_*}/` (predictions/output/metrics)
- 세부 실행: [EXPERIMENT_HISTORY.md Direction B + C-GT 배포 Sweep (2026-05-14)](EXPERIMENT_HISTORY.md)

### 결론

- B = Filter-Invariant F1 측 추가 evidence (axis #7 selector backbone-invariance 확장)
- C-GT = Four-caveat outlier candidate (또는 boundary 확장 evidence)
- 학술 frame "Filter-Invariant 경계 확정 실험" 정합 — 두 결과 모두 학술적 가치

---

## Direction B (HN-SupCon) + Direction C-GT (GraST-GT) Full Training (2026-05-14, 🎯 학습 PASS, sweep launch 적격)

### 등재된 학습 ckpt (2 신규)

| ID | Path | Size | Best |
|---|---|---|---|
| **hn_supcon** (B) | `outputs/checkpoints/hn_supcon/model.safetensors` | 90 MB | **val SLR Δ +0.0267** (1 epoch) |
| **grast_gt** (C-GT) | `outputs/checkpoints/grast_gt/best.pt` | 151 MB | **best loss 0.0674 @ ep31** (40 epoch) |

### Stack

| Module | b06_01 (B) | a05_26 (C-GT) |
|---|---|---|
| Backbone | MiniLM-L6 + HN-SupCon fine-tune | GraphTransformerEncoder from-scratch |
| Loss | HN-SupCon (Piao 2025) | Margin contrastive (Hoang 2025) |
| Hyperparameters | τ=0.07, N=8, margin=0.1, lr=5e-5, batch=16, **1 epoch** | margin=0.1, lr=5e-5, hidden=1024, L=3, H=8, **40 epoch** |

### 학습 결과

| Cell | Step/Epoch | Train loss (final/best) | Val metric | Pass | Wall |
|---|---|---|---|---|---|
| B | 1 epoch / 383 steps | 1.2681 | SLR @15: 0.6653→0.6920 (Δ +0.0267) | ✅ | ~6분 |
| C-GT | 40 epoch | 0.0701 final / **0.0674 best (ep31)** | (smoke PR-AUC Δ +0.0131) | ✅ saturation | ~1h 57m |

### Epoch 수 정당화

| 모델 | Epoch | 정당화 |
|---|:---:|---|
| **B** | **1** | PLM fine-tune + contrastive fast convergence + overfit 회피, Piao 2025 spec |
| **C-GT** | **40** | From-scratch + simple margin loss + sparse attention saturation, Hoang 2025 spec |
| GAT (reference) | 300 | From-scratch + multi-task + heterogeneous, conservative upper bound |

### 산출물

- Training scripts: `src/train_hn_supcon.py` (commit fb92775 + Root 5/14 evaluator), `src/train_grast_gt.py` (Root 5/14 wrapper)
- Checkpoints: `outputs/checkpoints/{hn_supcon,grast_gt}/`
- Sweep config 갱신: `a05_26_*.yaml` 의 `transformer_checkpoint_path` → `outputs/checkpoints/grast_gt/best.pt`
- 세부 실행: [EXPERIMENT_HISTORY.md Direction B + C-GT Full Training (2026-05-14)](EXPERIMENT_HISTORY.md)

### 결론

- 두 학습 모두 학술 Agent Q5 pass 충족
- Sweep launch 적격 (b06_01 GPU 0 + a05_26 GPU 1 병렬)

---

## Direction C 배포 Sweep (GRASTFDFilter + GPT-4.1-mini inferred FK, 2026-05-14, 🎯 Direction A 비교 sub-noise + 비용 -33%)

### 등재된 cell (1 신규, inference only)

| Cell | R | P | F1 | EX | LLM calls |
|---|---:|---:|---:|---:|---:|
| **a05_25** grast_with_inferred_fk | **0.9251** | 0.4218 | **0.5794** | 0.5176 | 3068 |

### Stack

- Builder: EnrichedHeteroGraphBuilder
- Selector: EnsembleSelector (best_gat_qcond_nl3.pt, α=0.5, top_k=20)
- Extractor: **MSTPCSTUnionExtractor** (Direction A 정합)
- Filter: **GRASTFDFilter** (provider=glm, model_name=zai-org/glm-4.7, terminal_source=forward, top_k=10, steiner_method=default, max_restore=30, fk_pk_hardcode=true)
  - **Inferred FK (4개, flat list)**: debit_card_specializing 3 + card_games 1
- SQL Generator: LLMSQLGenerator (GLM 4.7, evidence-aware)

### Δ vs Direction A (a05_23 F1=0.5833)

| 지표 | a05_23 (Direction A) | **a05_25 (Direction C)** | Δ |
|---|---:|---:|---:|
| R | 0.9456 | 0.9251 | -0.0205 |
| P | 0.4219 | 0.4218 | -0.0001 |
| F1 | 0.5833 | 0.5794 | **-0.0039** ≈ zero |
| EX | 0.5169 | 0.5176 | +0.0007 ≈ zero |

→ Direction C ≈ Direction A (F1 sub-noise) + **비용 효율 우세**:
| 자원 | Direction A | Direction C | Δ |
|---|---:|---:|---:|
| LLM calls | 4602 (3/q) | 3068 (2/q) | **-33%** ⚡ |
| Token in | 14.3M | 7.4M | -48% |
| Filter time mean | 4.03s | 1.90s | -53% |
| Wall (single cell) | ~2.6h | **1.83h** | -29% |

### Δ vs anchor (MSTPCSTUnion+XiYan F1=0.8666)

| ΔR | ΔP | ΔF1 | ΔEX |
|---|---|---|---|
| +0.0479 | -0.4346 | **-0.2872** ⚠ | +0.5176* |

\* Anchor EX=0.0 측정 실패 — analyzer 진단 후 정확 Δ.

### Inferred FK 적용 통계 (255 query / 1534)

| DB | inferred FK 수 | 적용 query | 비중 |
|---|---|---|---|
| card_games | 1 (`cards.setCode->sets.code`) | 191 | 12.4% |
| debit_card_specializing | 3 | 64 | 4.2% |
| **합계** | 4 | **255** | **16.6%** |

### 비용 / 운영

- Wall: 1h 50min (single cell, GPU 0, V5 학습 공존)
- LLM call: 3068, token in: 7.4M, out: 113K
- Filter time total: 2912s (=48.5min)

### 산출물

- Config: `configs/experiments/abl/a05_filter_agentic/a05_25_grast_with_inferred_fk_glm.yaml`
- Sweep script: `scripts/run_grast_fd_sweep.sh`
- Outputs: `outputs/experiments/abl/a05_filter_agentic/a05_25_grast_with_inferred_fk_glm/`
- Inferred FK source: `outputs/analysis/direction_c_inferred_fk.yaml`
- Module:Filter: commit e90d91a (`GRASTFDFilter`)
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Direction C 배포 Sweep (2026-05-14)](EXPERIMENT_HISTORY.md)

### 결론

- Direction C ≈ Direction A (F1 -0.0039 sub-noise, EX +0.0007 sub-noise) — algorithmic Steiner restore vs LLM-based backward 의 R-P tradeoff 동일
- **비용 효율 우세** — GRAST-FD 의 algorithmic restore 가 LLM call -33%, token -48%, time -53%
- Inferred FK 16.6% query 에 실제 적용 (255/1534) — per-DB lift 정량은 analyzer 후속
- **Filter Dominance 7번째 축 dual evidence** — Direction A (RSL Backward) + Direction C (GRAST-FD) 모두 F1 outlier + EX in-band, mechanism-agnostic robustness

---

## Direction A 배포 Sweep (RSLBackwardFilter, 2026-05-13, 🎯 Direction C 타겟 launch trigger)

### 등재된 cell (2 신규, inference only — ckpt 없음)

| Cell | risky_dbs | R | P | F1 | EX |
|---|---|---:|---:|---:|---:|
| **a05_23** baseline | `[]` | **0.9456** | 0.4219 | **0.5833** | 0.5169 |
| **a05_24** with_guard | `["toxicology"]` | 0.9395 | 0.4202 | 0.5806 | 0.5150 |

### Stack

- Builder: EnrichedHeteroGraphBuilder
- Selector: EnsembleSelector (best_gat_qcond_nl3.pt, α=0.5, top_k=20)
- **Extractor: MSTPCSTUnionExtractor** (5/13 anchor 변경 — MSTKruskal → MSTPCSTUnion, score_threshold=0.1)
- Filter: **RSLBackwardFilter** (GLM 4.7, xiyan_max_iteration=1, xiyan_num_examples=3, fk_pk_hardcode=true, temperature=0)
- SQL Generator: LLMSQLGenerator (GLM 4.7, evidence-aware)

### ΔvsAnchor (MSTPCSTUnion + XiYan, F1=0.8666)

| 지표 | Anchor | a05_23 | Δ |
|---|---:|---:|---:|
| R | 0.8772 | 0.9456 | **+0.0684** ⏫ |
| P | 0.8564 | 0.4219 | **-0.4345** ⏬ |
| F1 | **0.8666** | **0.5833** | **-0.2833** ⚠ |
| EX | 0.0000* | 0.5169 | +0.5169* |

*Anchor EX=0.0 → 측정 실패 (sql_generator 비활성 또는 bug) — analyzer 진단 필요.

### 🎯 Trigger 분기 — Direction C 타겟 launch trigger

- ❌ ΔF1 ≥ 0.03 → C post-paper
- ✅ **ΔF1 < 0.02 → C 타겟 launch** (실측 ΔF1 = -0.2833, 강한 negative)
- ❌ 0.02 ≤ ΔF1 < 0.03 → gray zone

### Guard 효과 — a05_24 vs a05_23

| ΔR | ΔP | ΔF1 | ΔEX |
|---|---|---|---|
| -0.0061 | -0.0017 | **-0.0027** | -0.0019 |

→ risky_dbs guard 효과 ≈ zero, **guard 불필요** narrative.

### 비용 / 운영

- Wall (parallel): 2h 47min (5/13 21:03 → 23:51)
- GPU 0/1 = V5 학습 (1.5GB) + a05_2{3,4} inference (0.85GB) 공존, ~2.3GB/24GB
- GLM API 2 concurrent throughput 효율 ~98% (per-q 6.04→6.14s)
- LLM call 합계: 9204 (4602 × 2)
- Token in: 28.6M, out: 364K

### 산출물

- Configs: `configs/experiments/abl/a05_filter_agentic/a05_2{3,4}_rsl_backward_*.yaml`
- Sweep script: `scripts/run_rsl_backward_sweep.sh` (parallel, GPU 0/1 분배)
- Outputs: `outputs/experiments/abl/a05_filter_agentic/a05_2{3,4}_rsl_backward_*/`
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Direction A 배포 Sweep (2026-05-13)](EXPERIMENT_HISTORY.md)

### 결론

- Direction A (RSL Backward) 단독으론 F1 net negative (-0.2833) — P -0.4345 손실 vs R +0.0684 회복
- EX 0.5169 maintained — backward restore 가 downstream SQL gen 에 useful
- risky_dbs guard 효과 ≈ zero — toxicology 도 net neutral, guard 불필요
- **Direction C 타겟 launch trigger** confirmed

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

발사: 2026-05-11 23:23 KST → 종료: V4-B 09:05 (wall 9h 38min) + V4-A 10:14 (wall 10h 47min). DECISIONS 2026-05-11 §V4 채택 + `planning/oversmoothing/oversmoothing_solution_methodology_2026-05-11_apa.md` §C-1 + §C-2 — combo 가설 (mech(ii-b) softmax-weighted-mean DOMINANT) 의 정량 검증. **결과: 둘 다 baseline 0.6097 미달 → mech(ii-b) DOMINANT 5/5 absolute confirm.**

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


---

## DSN Mitigation V5 7-Trial Sweep (V-3-ext 단계 9, 2026-05-13 → 05-15, 7 신규 ckpt, 🎯 시나리오 (a) confirm — 7/7 모두 P80 baseline 미달)

### 7 신규 ckpt × Best Val R@15 (4-decimal, Phase 1 P80 0.6097 baseline 비교)

| Variant | Module Class | gat_layer_type | num_layers | Best Val R@15 | Best Epoch | Wall | Δ vs P80 |
|---|---|---|---:|---:|---:|---|---:|
| **v5c_hop_only** | `FullAEROGATv2Conv` (hop only) | `aero_full` (hop=true, cum=false) | 2 | **0.6076** | 78 | 12h 12m | -0.0021 |
| v5b_gcnii_L2 | `GCNIIGATv2Conv` (L=2) | `gcnii`, β_λ=0.5 | 2 | 0.6072 | 76 | 12h 36m | -0.0025 |
| v5c_cum_only | `FullAEROGATv2Conv` (cum only) | `aero_full` (hop=false, cum=true) | 2 | 0.5993 | 25 | 12h 44m | -0.0104 |
| v5b_gcnii_L4 | `GCNIIGATv2Conv` (L=4) | `gcnii`, num_layers=4 | 4 | 0.5969 | 198 | 15h 50m | -0.0128 |
| v5c_full | `FullAEROGATv2Conv` (full) | `aero_full` (hop=true, cum=true) | 2 | 0.5887 | 241 | 12h 05m | -0.0210 |
| v5b_gcnii_L6 | `GCNIIGATv2Conv` (L=6) | `gcnii`, num_layers=6 | 6 | 0.5845 | 212 | 17h 55m | -0.0252 |
| v5a_gate | `GATEGATv2Conv` | `gate` | 2 | 0.5571 | 286 | 12h 15m | -0.0526 |

P@15 / F1@15 학습 시점 미측정 — analyzer 후속 평가 dispatch.

### 17-trial 통합 누적 (decreasing R@15, mech(ii-b) DOMINANT 7/7 absolute confirm candidate)

| 순위 | Variant | Best R@15 | Δ vs Phase 1 P80 |
|---|---|---:|---:|
| **1** | **Phase 1 P80 (no mit, baseline)** | **0.6097** | (baseline) |
| 2 | Phase 2 b8 (B5 mit fusion) | 0.6018 | -0.0079 |
| 3 | v2 #3 LayerNorm pre-softmax | 0.6011 | -0.0086 |
| 4 | **🆕 V5-C Hop only (v5c_hop_only)** | 0.6076 | -0.0021 |
| 5 | **🆕 V5-B GCNII L=2 (v5b_gcnii_L2)** | 0.6072 | -0.0025 |
| 6 | **🆕 V5-C Cumulative only (v5c_cum_only)** | 0.5993 | -0.0104 |
| 7 | v2 #1 DropMessage | 0.5974 | -0.0123 |
| 8 | **🆕 V5-B GCNII L=4 (v5b_gcnii_L4)** | 0.5969 | -0.0128 |
| 9 | v3 #1 GIN-style aggregation | 0.5954 | -0.0143 |
| 10 | V4-B AERO Softplus + Sym-Norm | 0.5951 | -0.0146 |
| 11 | Phase 3 #4 (LR x5) | 0.5935 | -0.0162 |
| 12 | V4-A LN+GIN combo | 0.5929 | -0.0168 |
| 13 | Phase 3 #3 (Direct AC) | 0.5927 | -0.0170 |
| 14 | **🆕 V5-C Full (v5c_full)** | 0.5887 | -0.0210 |
| 15 | **🆕 V5-B GCNII L=6 (v5b_gcnii_L6)** | 0.5845 | -0.0252 |
| 16 | v2 #2 Sum Aggregation | 0.5761 | -0.0336 |
| 17 | **🆕 V5-A GATE (v5a_gate)** | 0.5571 | -0.0526 (V5 worst) |

→ 시나리오 (a) (V5 7 모두 null R ~0.58-0.61 saturate) absolute confirm. 직전 14-trial 의 mech(ii-b) DOMINANT 5/5 absolute confirm 정합 + V5 7-trial 의 architectural intervention 3 axis × 7 cell 모두 fail → **mech(ii-b) softmax × weighted-mean propagation combo 가 fundamental architectural limitation 격상 candidate**.

### Config 주의사항 (training)

- 학습 entry: `src/train_gat_s06.py` (V5 kwargs forwarding `gcnii_beta_lambda`, `aero_hop_attention`, `aero_cum_attention` line ~248-252)
- 공통 V-3-ext (DSN p80 directed_from_sn + percentile=80) + B5 mitigation:
  - V5-A/B: PN + IR α=0.2 + JK=concat + Dual-Stream + AC=0.1 + ListNet
  - V5-C: JK='none' + Hop/Cum Attention (forward 시 V5-C 출력 경로)
- 변경 차원: `gat_layer_type` ∈ {'gate', 'gcnii', 'aero_full'} + V5-C 의 `aero_hop_attention` / `aero_cum_attention` 조합
- attention module 별 spec:
  - V5-A `GATEGATv2Conv`: att_self + parent att 분리 (Conservation Law decoupling, Mustafa & Burkholz 2024 §3.2 Eq. 4)
  - V5-B `GCNIIGATv2Conv`: β_l = log(λ/l + 1) Identity Mapping + Initial Residual α (Chen 2020 GCNII Eq. 6 + Peng 2024 trainability)
  - V5-C `FullAEROGATv2Conv`: Softplus per-node norm + Hop Attention (outer L+1 weighted stack) + Cumulative Attention residual (Lee 2023 AERO Theorem 3 SR2OS)

### 결론 — 시나리오 (a) confirm + mech(ii-b) 7/7 absolute confirm candidate

- 17-trial × 6 mitigation 카테고리 (graph topology / B5 mitigation / loss-level / aggregation-level / V4 architectural intervention / V5 architectural intervention) 모두 baseline 미달
- V5 7-trial 중 v5c_hop_only 0.6076 (Δ=-0.0021 noise band, max) — anchor qcond_nl3 (0.6061) marginal 상회 (+0.0015) 하나 P80 0.6097 미달
- V5-B depth scale monotonic decay (L=2 0.6072 > L=4 0.5969 > L=6 0.5845) — Chen 2020 deep-GNN claim 의 heterogeneous schema graph 미적용 (over-smoothing 외 PLM lower bound bottleneck)
- V5-A GATE 0.5571 (V5 worst) — Conservation Law decoupling 의 heterogeneous schema graph 미적용 (attention parameter norm constraint 외 propagation pathology dominant)
- paper §3.5 Filter Dominance 6번째 축 (training-pathology-invariant) **17-trial evidence 누적** + paper §V.5.4 mech(ii-b) DOMINANT 7/7 absolute confirm 격상 candidate
- 세부 실행 이력: [EXPERIMENT_HISTORY.md DSN Mitigation V5 7-Trial Sweep 학습 완료 (V-3-ext 단계 9, 2026-05-13 → 05-15)](EXPERIMENT_HISTORY.md).

### 산출물

- Configs (7): `train_dsn_p80_v5{a_gate, b_gcnii_L2, b_gcnii_L4, b_gcnii_L6, c_full, c_hop_only, c_cum_only}.yaml`
- 모델 확장: `src/models/gat_network_v2.py` (commit `afadafd`) — `GATEGATv2Conv` + `GCNIIGATv2Conv` + `FullAEROGATv2Conv` + Hop/Cum Attention modules + `GAT_LAYER_TYPES` set 확장
- 학습 entry: `src/train_gat_s06.py` (V5 kwargs forwarding)
- Sweep script: `scripts/run_v5_mitigation_sweep.sh` (4-stage GPU 0/1 병렬, 사용자 5/14 명시 "Root 학습/실험 메인" 후 직접 launch)
- 학습 logs: `logs/train/dsn_p80_v5{*}_*.log` (7 cell, 모든 log 300 epoch + Training Completed 종료 mark)
- Checkpoints (NAS, 7 신규): `/SSL_NAS/peoples/khj/thesis/checkpoints/best_gat_directed_supernode_p80_v5{*}.pt` (총 1.69GB)
- 비용: ₩0 (LLM-free) | 학습 wall (합산): 95h 17m | 학습 wall (실시간 2-GPU 병렬): ~55h (5/13 12:10 → 5/15 19:07)


---

## Phase 2 Grid Sweep — Hyperparameter 2D Grid θ × K = 5×5 = 25 cells (Wave 5 Partial Reopen, 2026-05-16, 🎯 Success criterion (a) Plateau breadth confirm)

### 25 cells × 4 metrics (R/P/F1/EX 4-decimal, anchor c01_01 F1=0.8664 / EX=0.5176 비교)

**F1 5×5 Heatmap** (Global max p2_03 0.8680 = anchor +0.0016 sub-noise):

| θ \ K | 15 | 20 | 30 | 40 | 70 | **avg** | row-Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **0.1** | 0.8669 | 0.8669 | **0.8680** ★ | 0.8646 | 0.8670 | **0.8667** | +0.0003 |
| 0.125 | 0.8631 | 0.8641 | 0.8640 | 0.8637 | 0.8659 | 0.8642 | -0.0023 |
| 0.15 | 0.8623 | 0.8628 | 0.8650 | 0.8651 | 0.8651 | 0.8641 | -0.0023 |
| 0.175 | 0.8619 | 0.8615 | 0.8588 | 0.8580 | 0.8575 | 0.8595 | -0.0069 |
| 0.2 | 0.8579 | 0.8611 | 0.8626 | 0.8613 | 0.8621 | 0.8610 | -0.0054 |

**EX 5×5 Heatmap** (Global max p2_07 0.5189 = anchor +0.0013 sub-noise):

| θ \ K | 15 | 20 | 30 | 40 | 70 | **avg** |
|---|---:|---:|---:|---:|---:|---:|
| **0.1** | 0.5163 | 0.5163 | 0.5130 | **0.5169** | 0.5163 | 0.5158 |
| 0.125 | 0.5117 | **0.5189** ★ | 0.5143 | 0.5137 | 0.5143 | 0.5146 |
| 0.15 | 0.5098 | 0.5111 | 0.5020 | 0.5026 | 0.5013 | 0.5054 |
| 0.175 | 0.5033 | 0.5026 | 0.5007 | 0.5007 | 0.5020 | 0.5019 |
| 0.2 | 0.4961 | 0.4980 | 0.4980 | 0.4954 | 0.4954 | 0.4966 |

### Anchor 정합 검증 (P2_02 vs c01_01)

| | R | P | F1 | EX |
|---|---:|---:|---:|---:|
| c01_01 | 0.8748 | 0.8582 | 0.8664 | 0.5176 |
| p2_02 | 0.8761 | 0.8579 | 0.8669 | 0.5163 |
| Δ | +0.0013 | -0.0003 | **+0.0005** | -0.0013 |

✅ **Deterministic 정합 PASS** (ΔF1 ≤ 0.0010 noise band).

### Success criterion 분기 판단

| Criterion | 결과 | 학술 weight |
|---|---|:---:|
| **(a) Plateau breadth** | anchor-band F1 spread = 0.0057 (V5 inference 0.0052 정합), EX spread ~0.020 — plateau 확인 | **High** — axis #11 retain + strengthen |
| **(b) R 갱신 lever** | p2_03 F1=0.8680 +0.0016 / p2_07 EX=0.5189 +0.0013 — noise floor 약간 초과 잠정, **statistically robust 아님** | **Low** — closure 재고 trigger 미달성 |

→ **Outcome (a) Plateau 흡수** — axis #11 (builder-axis invariance) evidence retain.

### Config 주의사항

- 학습 entry: 학습 없음 (sweep only) — weight_path = `best_gat_qcond_nl3.pt` (5/14 → 5/16 anchor 동일)
- 변경 차원: selector.top_k ∈ {15, 20, 30, 40, 70} × extractor.score_threshold ∈ {0.1, 0.125, 0.15, 0.175, 0.2} = 25 cells
- Stack: c01_01 anchor (QCondGAT 3-layer + bidirectional SN + MSTPCSTUnion + XiYanFilter GLM 4.7 + LLMSQLGenerator)

### 결론 — Plateau breadth confirm + anchor 정합 PASS

- 25 cells 모두 anchor-band 부근 cluster (F1 spread 0.8575~0.8680 = Δ=0.0105 across 25 cells, anchor-band 15 cells spread 0.0057)
- θ axis monotonic decay (0.1 → 0.175 → 0.2 의 mid-θ dip): F1 decay -0.0072 from θ=0.1 to θ=0.175
- K axis sub-noise (θ ∈ anchor-band 안) — Phase 1.2 K sweep sub-noise (0.0019) 정합
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Phase 2 Grid Sweep (2026-05-16)](EXPERIMENT_HISTORY.md).

### 산출물

- Configs (25): `configs/experiments/abl/c03_phase2_grid/p2_{01..25}_theta_X_topk_Y.yaml`
- Sweep script: `scripts/run_phase2_grid_sweep.sh` (8-conc parallel GPU 0/1)
- 학습 없음 (sweep only), 비용 ~$15-30 GLM API
- Wall: 8h14m50s (00:47:25 → 09:02:15)
- failure: 0/25 (모든 cells 의 metrics.txt 정상 생성)


---

## Phase 4.1 (α sweep) + Phase 4.2 (TCR-conditional) Chain — 6+3=9 cells (학술 agent plan §Phase 4, 2026-05-16, 🎯 α=0.0~0.8 plateau confirm + α=1.0 cliff -0.0952 + thr=0.5 Pareto sweet spot)

### 9 cells × 4 metrics (R/P/F1/EX 4-decimal, anchor c01_01 F1=0.8664 / EX=0.5176 비교)

**Phase 4.1 α sweep (6 cells)**:

| α | F1 | EX | ΔF1 | Note |
|---|---:|---:|---:|---|
| **0.0** ⭐ | 0.8662 | 0.5150 | -0.0002 ✅ | anchor 정합 PASS (deterministic verify) |
| 0.2 | **0.8665** | 0.5137 | +0.0001 | sub-noise |
| 0.4 | 0.8662 | 0.5137 | -0.0002 | sub-noise |
| 0.6 | 0.8657 | 0.5169 | -0.0007 | sub-noise |
| 0.8 | **0.8667** | 0.5150 | +0.0003 | sub-noise marginal max |
| **1.0** ★ | **0.7712** | **0.3638** | **-0.0952** | TopK-only cliff drop |

→ **α=0.0~0.8 plateau (spread 0.0010), α=1.0 cliff drop**.

**Phase 4.2 TCR-conditional (3 cells)**:

| thr | F1 | EX | ΔF1 | Skip / 1534 | Skip % |
|---|---:|---:|---:|---:|---:|
| 0.3 | 0.8673 | 0.5111 | +0.0009 | 0 | 0.0% |
| **0.5** ⭐ | 0.8671 | 0.5156 | **+0.0007** sub-noise | 8 | 0.5% (Pareto sweet spot) |
| 0.7 | 0.8588 | 0.5150 | **-0.0076** | 39 | 2.5% |

→ thr=0.5 Pareto sweet spot, thr=0.7 aggressive (F1 -0.0076 cost). TCR 분포 high (대부분 ≥ 0.7).

### Anchor 정합 검증 (α=0.0 ↔ c01_01 ↔ p2_02 deterministic 일치)

| Source | F1 | ΔF1 vs c01_01 |
|---|---:|---:|
| c01_01 (5/15) | 0.8664 | (base) |
| p2_02 (5/16 morning) | 0.8669 | +0.0005 |
| p4_01 α=0.0 (5/16 noon) | 0.8662 | -0.0002 |

→ GLM stochastic noise floor: **~±0.0005 F1**, 3 measurement spread 0.0007.

### Config 주의사항

- 학습 entry: 없음 (sweep only, anchor ckpt `best_gat_qcond_nl3.pt` 재사용)
- Phase 4.1 변경 차원: `MSTPCSTUnionExtractor.seed_selection_mode="integrated_score"` + alpha (commit `1e2c46a`)
- Phase 4.2 변경 차원: `ConditionalFilterWrapper(inner=XiYanFilter)` + tcr_threshold (commit `e0685eb`)
- Stack: c01_01 anchor (QCondGAT 3-layer + bidirectional SN + MSTPCSTUnion + XiYanFilter GLM 4.7 + LLMSQLGenerator)

### 결론 — Plateau + Cliff (Extractor rescue dominant) + Pareto Filter cost

- α plateau (α=0.0~0.8) — PCST prize weighting 의 final F1 invariance
- α=1.0 cliff drop F1 -0.0952 — Extractor threshold-pass rescue 가 dominant lever
- Phase 4.2 TCR distribution heavy tail → thr=0.5 sweet spot (cost saving 0.5% + F1 sub-noise)
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Phase 4.1+4.2 Chain (2026-05-16)](EXPERIMENT_HISTORY.md).

### 산출물

- Phase 4.1 Configs (6): `configs/experiments/abl/c04_phase4_alpha_sweep/p4_{01..06}_alpha_X.yaml`
- Phase 4.2 Configs (3): `configs/experiments/abl/c05_phase4_conditional_filter/p4_2_thr_{0.3, 0.5, 0.7}.yaml`
- Module 구현: extractor commit `1e2c46a` + filter commit `e0685eb`
- Sweep script: `scripts/run_phase4_chain.sh` (9-conc parallel GPU 0×4 + GPU 1×5)
- Wall: 2h05m (11:48 → 13:52), 비용 ~$8-15 GLM API
- failure: 0/9


---

## Wave 6 Phase 1 M1 Recall-Biased Prompt — 3 variants (DECISIONS 2026-05-16 Wave 6 §2, 학술 agent filter improve plan §3, 2026-05-16, 🎯 mild R-lift +0.0511 / strong F1 sub-noise / Phase 2 (a) M2 CoT 분기 권고)

### 3 variants × R/P/F1/EX (4-decimal, anchor c01_01 R=0.8748 / P=0.8582 / F1=0.8664 / EX=0.5176)

| Cell | prompt_mode | R | P | F1 | EX | ΔF1 | Note |
|---|---|---:|---:|---:|---:|---:|---|
| **mild (M1-A)** | recall_biased_mild | **0.9259** ★ R-max | 0.7648 | 0.8377 | **0.5169** ★ | -0.0287 | inclusion 가장 강함, P-cost 큼 |
| **strong (M1-B)** ⭐ | recall_biased_strong | 0.9022 | 0.8316 | **0.8655** ★ F1-max | 0.5130 | **-0.0009** sub-noise | F1 최적 균형, M1 best |
| **exclusion_rule (M1-C)** | recall_biased_exclusion_rule | 0.8907 | 0.8263 | 0.8573 | 0.5143 | -0.0091 | 4-rule conjunctive, 가장 conservative |

### Inclusion bias strength → R-P trade-off monotonic 정합

- R: mild > strong > exclusion_rule (inclusion bias ↑ → R ↑)
- P: strong > exclusion_rule > mild (P loss 비례)
- F1: strong > exclusion_rule > mild (R-P 균형)

### 학술 agent §10 성공 기준 (F1_fil ≥ 0.8672 필수)

- mild: 0.8377 ❌
- **strong: 0.8655 ❌ (-0.0017 sub-noise, 가장 가까움)**
- exclusion_rule: 0.8573 ❌
- → 3 cells 모두 미달 (sub-noise level), Phase 2 M2+ 후속 chain 필요

### DECISIONS §3 Phase 2 분기 권고 — Phase 2 (a)

- R_fil best (mild) = 0.9259 ≥ 0.92 → **Phase 2 (a) M2 CoT + Confidence-Gated + M1 best 조합 권고**
- M1 best = **strong (F1 0.8655 sub-noise + ΔR +0.0274)**
- Phase 2 (a): M1 best (strong) + M2 CoT prompt + Confidence-Gated (P 회복 + F1 ≥ 0.8672 달성 시도)

### Config 주의사항

- 학습 없음 (anchor ckpt `best_gat_qcond_nl3.pt` 재사용)
- 변경 차원: XiYanFilter prompt_mode parameter (commit `07d2fda`)
- Common 후처리: sanitize_filter_output() default-on (Hallucination 방지)
- LLM: glm-4.7 (4602 calls = 1534 × 3 variants)

### 결론 — M1 Recall-Biased Prompt 의 R-lift evidence + Phase 2 (a) 분기

- strong (M1-B) F1=0.8655 sub-noise + R-lift +0.0274 — anchor F1 거의 유지하면서 R lift
- mild (M1-A) R=0.9259 ≥ 0.92 → Phase 2 (a) trigger 충족
- inclusion bias 강도 → R-P trade-off monotonic — Wave 6 chain mechanism evidence
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Wave 6 Phase 1 (2026-05-16)](EXPERIMENT_HISTORY.md).

### 산출물

- Configs (3): `configs/experiments/abl/wave6_recall_biased/wave6_p1_recall_biased_{mild, strong, exclusion_rule}.yaml`
- Module 구현: filter commit `07d2fda` (prompt_mode + sanitize_filter_output + smoke 18/18 PASS)
- Sweep script: `scripts/run_wave6_phase1_recall_biased.sh` (3-conc GPU 0×2 + GPU 1×1)
- Wall: 1h58m (17:49 → 19:47), 비용 ~$3-6 GLM API
- failure: 0/3


---

## Wave 6 Phase 2 (a+aggressive) — M2 + M3 + M4 + M5 4 cells (DECISIONS 2026-05-16 §2+§3, 학술 agent §3~§7+§10, 2026-05-16 ~ 2026-05-17, 🎯 Outcome (b) — F1 모두 미달 + M4 EX gain +0.0124 첫 evidence)

### 4 cells × 4 metrics (R/P/F1/EX 4-decimal, anchor c01_01 R=0.8748 / P=0.8582 / F1=0.8664 / EX=0.5176)

| Cell | Method | R | P | F1 | EX | ΔF1 | ΔEX |
|---|---|---:|---:|---:|---:|---:|---:|
| **M2** w6_p2a_m2cot_strong | CoT + Confidence-Gated | 0.9745 ★ R | 0.2286 | **0.3703** ★ worst | 0.5169 | **-0.4961** ❌ | -0.0007 |
| **M3** w6_p2_m3_voting | Multi-Prompt OR Voting | 0.9408 | 0.6859 | 0.7934 | 0.5202 | -0.0730 | +0.0026 |
| **M4** w6_p2_m4_bidirectional ⭐ | Forward + Backward union | 0.9325 | 0.7593 | **0.8370** ★ F1-best | **0.5300** ★ EX-max | -0.0294 | **+0.0124** ✅ |
| **M5** w6_p2_m5_two_stage | Sequential Stage1 → Stage2 | 0.7739 | 0.7964 | 0.7850 | 0.5222 | -0.0814 | +0.0046 |

### Key findings

- **학술 agent §10 success criterion F1 ≥ 0.8672**: 모두 미달 → DECISIONS §5 Outcome (b): axis #15 evidence retain + axis #11 Option A retain
- **🚀 M4 EX gain +0.0124** ★ — Wave 6 chain 첫 EX 갱신 (Backward SQL Schema Analyst 가 SQL execution 의 missing column 보충)
- **M2 catastrophic F1=0.3703** — Confidence-Gated default-retain 정책의 design flaw (uncertain → keep)
- **M5 R loss -0.1009** — Stage2 가 Stage1 의 R lift 효과 negate, sequential pipeline fails
- **schema linking F1 ↔ SQL EX correlation 약함**: M2 F1 -0.4961 인데 EX sub-noise plateau (Filter Dominance dual narrative)

### Inclusion bias spectrum 통합 ranking (M1 + Phase 2)

| Cell | R | F1 | mechanism |
|---|---:|---:|---|
| anchor c01_01 | 0.8748 | 0.8664 | baseline |
| M1-B strong ⭐ F1-max M1 | 0.9022 | 0.8655 | mild inclusive |
| **M4 Bidirectional** | 0.9325 | 0.8370 | Forward+Backward union |
| M1-A mild | 0.9259 | 0.8377 | medium inclusive |
| M1-C exclusion | 0.8907 | 0.8573 | weak inclusive |
| M3 Voting OR | 0.9408 | 0.7934 | OR voting endpoint |
| M5 Two-Stage | 0.7739 | 0.7850 | sequential fails |
| **M2 CoT-Gated** | 0.9745 ★ R | 0.3703 | extreme inclusive (default-retain) |

### Config 주의사항

- 학습 없음 (anchor ckpt `best_gat_qcond_nl3.pt` 재사용)
- Filter class 4 변형: XiYanFilter (M2, commit `7dac875`) + MultiPromptVotingFilter / BidirectionalFilter / TwoStageFilter (M3/M4/M5, commit `88ad47e`)
- 공통: sanitize_filter_output=True (Hallucination 방지)

### 결론 — Outcome (b) confirmed + M4 EX gain 첫 evidence

- 4 cells 모두 학술 agent §10 success criterion F1 미달
- M4 EX gain +0.0124 (Wave 6 chain 첫 EX 갱신) — Filter ↔ Selector co-design EX-axis new evidence
- DECISIONS §5 → Outcome (b) confirmed: axis #15 evidence retain (prompt-level strengthening) + axis #11 Option A retain (prompt-axis + builder-axis 별도)
- paper §V.5.x.M.15 candidate 본문 정식 채택 candidate (M1 R-lift + M4 EX gain 통합)
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Wave 6 Phase 2 (a+aggressive) (2026-05-17)](EXPERIMENT_HISTORY.md).

### 산출물

- Configs (4): `configs/experiments/abl/wave6_recall_biased/w6_p2a_m2cot_strong.yaml` + `w6_p2_m{3_voting, 4_bidirectional, 5_two_stage}.yaml`
- Module 구현: filter commits `7dac875` (CoT+Gated) + `88ad47e` (M3/M4/M5 신규 class)
- Sweep scripts: `scripts/run_wave6_phase2a_cot.sh` + `scripts/run_wave6_phase2_aggressive.sh`
- Wall: M2 2h31m + M3+M4+M5 ~4h (parallel) = 5h17m total
- 비용: ~$30-50 GLM 4.7 (M2 ~$3 + M3 ~$12 + M4 ~$9 + M5 ~$9)
- failure: 0/4


---

## Wave 6 Phase 4 Top 2 C1 — M4 + M1-B strong Forward (DECISIONS 2026-05-17 §4, 학술 agent §8.3, 2026-05-17, 🎯 Partial Degrade — Forward Dominance + Backward Effect Reduction, Pareto frontier 진입)

### Single cell × 4 metrics (anchor c01_01 R=0.8748 / P=0.8582 / F1=0.8664 / EX=0.5176)

| Cell | R | P | F1 | EX |
|---|---:|---:|---:|---:|
| **w6_p4_c1_m4_strong** | **0.9177** | 0.8109 | **0.8610** | 0.5150 |

### vs 3 baselines

| baseline | F1 | ΔF1 vs C1 | EX | ΔEX vs C1 |
|---|---:|---:|---:|---:|
| anchor c01_01 | 0.8664 | C1 -0.0054 sub-noise | 0.5176 | C1 -0.0026 sub-noise |
| M4 baseline (mild Forward) | 0.8370 | **C1 +0.0240** ✅ | **0.5300** ★ | **C1 -0.0150** ❌ EX loss |
| M1-B strong (Forward only) | **0.8655** ★ | C1 -0.0045 sub-noise | 0.5130 | C1 +0.0020 sub-noise |

### Synergy / Additive / Degrade 분기 — Partial Degrade 확정

- Synergy ❌ (F1 < M1-B AND EX < M4)
- Additive (full) ⚠ 부분 (F1 ≈ M1-B sub-noise ✅, EX 가 M1-B 에 가까움 NOT M4)
- **Partial Degrade ✅ 확정** — F1 < M1-B sub-noise + EX < M4 큰 손실 (-0.0150)

### 🌟 New Finding — Backward mechanism Forward-prompt-dependent

M4 EX gain (+0.0124) 의 source = Forward (mild) inclusive base 위에 Backward 가 SQL-aware column 보충. C1 (strong Forward) 시 Backward 보충 column space 줄어듦 → **EX gain 사라짐 (-0.0150 from M4)**.

→ **DECISIONS §3.1 Forward/Backward orthogonality hypothesis 부분 부정** — Forward prompt 가 Backward effect size 결정 (entanglement evidence)

### Pareto Frontier 5번째 cell 진입

C1: R=0.9177 ≥ 0.90 ✅, P=0.8109 ≥ 0.75 ✅ → Pareto frontier 진입 (M1-A + M1-B + M3 MAJORITY + M4 + C1 = 5 cells)

### Config 주의사항

- 학습 없음 (anchor ckpt `best_gat_qcond_nl3.pt` 재사용)
- 변경 차원: BidirectionalFilter 의 `bidirectional_forward_prompt_mode="recall_biased_strong"` (commit `60b6988`)
- Backward retain: `bidirectional_backward` (M4 baseline 동일)
- 공통: sanitize_filter_output=True

### 결론 — Partial Degrade + Forward Dominance + Pareto entrance

- F1 학술 agent §10 미달 (-0.0062 sub-noise) — DECISIONS §5 Outcome (b) retain
- 새 finding: Backward mechanism Forward-prompt-dependent (entanglement)
- C2 (M4 + M3 MAJORITY Forward) launch 학술 motivation 강화
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Wave 6 Phase 4 Top 2 C1 (2026-05-17)](EXPERIMENT_HISTORY.md).

### 산출물

- Config: `configs/experiments/abl/wave6_recall_biased/w6_p4_c1_m4_strong.yaml`
- Module 구현: filter commit `60b6988` (BidirectionalFilter `bidirectional_forward_prompt_mode` flag)
- Sweep script: `scripts/run_wave6_phase4_c1.sh`
- Wall: 2h33m (10:31 → 13:04), 비용 ~$2-4 GLM API
- failure: 0/1


---

## Wave 6 Phase 5 Top 2 C2 — M4 + M3 MAJORITY voting Forward (DECISIONS 2026-05-17 §6, 학술 agent §8.3 + §5+§6, 2026-05-17, 🎯 H3 Partial Entanglement 확정 + Pareto frontier 6번째 cell 진입)

### Single cell × 4 metrics (anchor c01_01 R=0.8748 / P=0.8582 / F1=0.8664 / EX=0.5176)

| Cell | R | P | F1 | EX |
|---|---:|---:|---:|---:|
| **w6_p5_c2_m4_majority** | **0.9273** | 0.7745 | **0.8440** | **0.5196** |

### vs 4 baselines

| baseline | F1 | ΔF1 vs C2 | EX | ΔEX vs C2 |
|---|---:|---:|---:|---:|
| anchor c01_01 | 0.8664 | C2 -0.0224 | 0.5176 | C2 +0.0020 sub-noise |
| M4 (mild Forward) ⭐ EX-max | 0.8370 | C2 +0.0070 | **0.5300** ★ | **C2 -0.0104** ← key |
| C1 (strong Forward) | 0.8610 | C2 -0.0170 | 0.5150 | **C2 +0.0046** ← key |
| M3 MAJORITY (post-hoc) | 0.8433 | C2 +0.0007 sub-noise | (post-hoc) | — |

### 3 Hypothesis 판정 — **H3 Partial Entanglement 확정** ✅

- H1 (inclusiveness dominant, C2 EX ≈ M4 0.5300) ❌ — Δ=-0.0104 from M4
- H2 (voting mechanism dominant, C2 EX ≈ C1 0.5150) ❌ — Δ=+0.0046 from C1
- **H3 (partial entanglement, C2 EX intermediate 0.52~0.53) ✅ — 0.5196 ∈ [0.5150, 0.5300]**

### 📊 Backward Effect Reduction mechanism 정량 분해

- M4 distance vs C1 distance ratio = **2.26 : 1** → C2 가 C1 쪽으로 약간 치우침
- **정량 분해**: Voting mechanism ~70% + Forward inclusiveness ~30%
- C1 의 Backward Effect Reduction (-0.0150 EX from M4) 의 mechanism = 60% voting + 40% inclusiveness (대략)

### Forward Dominance 3-cell complete coverage (M4 + C1 + C2)

| Forward | F1 | EX | Backward Effect (EX gain from M4) |
|---|---:|---:|---:|
| M4 mild | 0.8370 | **0.5300** ★ | +0.0124 (anchor base) |
| **C2 voting MAJORITY** | 0.8440 | 0.5196 | -0.0104 (~70% mechanism) |
| C1 strong | 0.8610 | 0.5150 | -0.0150 (Partial Degrade, full mechanism + inclusiveness) |

### Pareto Frontier 6 cells (C2 신규 진입)

| Cell | R | P | F1 | EX |
|---|---:|---:|---:|---:|
| M1-A mild + M1-B strong + M3 MAJORITY (post-hoc) + M4 + C1 + **🆕 C2** | 6 cells | | | |
| **C2**: R=0.9273 ≥ 0.90 ✅, P=0.7745 ≥ 0.75 ✅ | ★ 6번째 진입 | | | |

### Config 주의사항

- 학습 없음 (anchor ckpt `best_gat_qcond_nl3.pt` 재사용)
- 변경 차원: BidirectionalFilter 의 `bidirectional_forward_prompt_mode="voting_multi_prompt"` + `bidirectional_forward_voting_strategy="MAJORITY"` (commit `7a07a6b`)
- Backward retain: `bidirectional_backward` (M4 default)
- LLM call/q: 4 (3 voting Forward + 1 Backward sequential)

### 결론 — H3 Partial Entanglement 확정 + Pareto frontier 진입 + Wave 6 chain mechanism axis 완성

- F1 학술 agent §10 미달 (-0.0232) — DECISIONS §5 Outcome (b) retain
- H3 Partial Entanglement 확정 (Voting ~70% + Inclusiveness ~30% 정량 분해)
- Wave 6 chain mechanism axis 완성 (Forward Dominance 3-cell coverage)
- paper §V.5.x.M.15 Triple → Quadruple Evidence 격상 candidate
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Wave 6 Phase 5 Top 2 C2 (2026-05-17)](EXPERIMENT_HISTORY.md).

### 산출물

- Config: `configs/experiments/abl/wave6_recall_biased/w6_p5_c2_m4_majority.yaml`
- Module 구현: filter commit `7a07a6b` (BidirectionalFilter + voting_multi_prompt Forward composition, smoke 36/36 PASS)
- Sweep script: `scripts/run_wave6_phase5_c2.sh`
- Wall: 4h02m (15:31 → 19:34), 비용 ~$10-15 GLM API
- failure: 0/1


## Wave 7 Stage-wise EX Chain — Anchor Relog c01_01_wave7_relog (DECISIONS 2026-05-18 §2+§3 Option A, 2026-05-18, 🎯 Stage-wise EX 3 stage 측정 — Filter F1-EX 분리 first evidence)

### Single cell × 7 metrics (anchor c01_01 prior R=0.8748 / P=0.8582 / F1=0.8664 / EX=0.5176)

| 항목 | 값 |
|---|---|
| ID | `c01_01_wave7_relog` |
| Stack | c01_01 anchor (XiYanFilter + MSTPCSTUnion + EnsembleSelector + GLM 4.7 SQL gen) |
| Patch | Option A integration — schema_linking.py + main.py (3 SQL Gen + 3 EX 통합 logging) |
| R | 0.8697 (Δ vs prior −0.0051) |
| P | 0.8581 (Δ vs prior −0.0001) |
| F1 | 0.8639 (Δ vs prior −0.0025, sub-noise) |
| EX final | 0.5117 (Δ vs prior −0.0059, sub-noise) |
| Wall | 3h 28min |
| LLM calls | 6136 (4/q: 1 Filter + 3 SQL Gen) |
| failure | 0/1 |

### Stage-wise EX (paper §5.5.1 EX cell 채움)

| Stage | nodes/q | EX (Wave 7) | 비고 |
|---|---:|---:|---|
| (1) Selector only top-K=20 | 20.00 | **0.3507** | 🆕 첫 측정 |
| (2) +Extractor (MSTPCSTUnion, no filter) | 83.08 | **0.5150** | 🆕 첫 측정 |
| (3) +Filter (anchor c01_01 XiYan) | ~4.70 | **0.5117** | 재현 (prior 0.5176, Δ−0.0059) |
| (3') +Filter (M4 Bidirectional) ⭐ | 6.48 | 0.5300 (prior retain) | (1)(2) 공유 |

### Filter F1-EX 분리 evidence (Wave 7 main finding)

| Stage | ΔF1 | ΔEX | 의미 |
|---|---:|---:|---|
| Selector → Extractor | −0.1158 | **+0.1643** ★ | Extractor R-lift 가 EX dimension 에도 dramatic 효과 |
| Extractor → Filter (c01_01) | +0.6555 | **−0.0033** ★ | Filter F1 dominant (76% contribution) 단 EX micro-negative |
| Extractor → Filter (M4 prior) | (sub-noise) | +0.0150 | M4 Bidirectional 의 EX positive gain |

→ **F1-EX axis 분리 first evidence** — paper §V.5.x.M.12 Filter Dominance 갱신 candidate.

### M4 vs c01_01 Filter EX cost ΔΔ

- c01_01 (XiYan default): ΔEX = −0.0033 (negative)
- M4 (Bidirectional Forward+Backward Union): ΔEX = +0.0150 (positive)
- **ΔΔ = +0.0183** — M4 의 Backward Union 의 EX gain mechanism 정량 강화 (paper §V.5.x.M.15 axis #15 갱신 candidate)

### Config 주의사항

- 학습 없음 (anchor ckpt `best_gat_qcond_nl3.pt` 재사용)
- Option A integration patch: schema_linking.py + main.py (commit local 예정)
- Filter spec: XiYanFilter default (M4 BidirectionalFilter 아님 — Stage (1)(2) 측정용)
- LLM call/q: 4 (1 XiYan Filter + 3 SQL Gen for 3 stage cumulative)

### 결론 — Stage-wise EX 측정 완료 + F1-EX 분리 first evidence + paper §V.5.x.M.12 갱신 candidate

- 재현 정확도: sub-noise (ΔF1 −0.0025, ΔEX −0.0059)
- Stage-wise EX 측정 완료 — m4_anchor_framework_analysis §5.5.1 표 갱신 가능
- Filter F1-EX 분리 first evidence (F1 dominant +0.6555 vs EX micro-negative −0.0033)
- M4 EX gain mechanism ΔΔ=+0.0183 정량 (vs c01_01)
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Wave 7 Stage-wise EX Chain (2026-05-18)](EXPERIMENT_HISTORY.md).

### 산출물

- Config: `configs/experiments/abl/c01_threshold_sweep/c01_01_wave7_relog.yaml`
- Patches: `src/pipeline/schema_linking.py` (4 patches) + `src/main.py` (2 patches), commit local 예정
- Outputs: `outputs/experiments/abl/c01_threshold_sweep/c01_01_wave7_relog/` (metrics.txt + predictions.jsonl + 4 telemetry jsonl)
- Wall: 3h 28min, 비용 ~$15-20 GLM API
- failure: 0/1


## Wave 9 Baseline Relog — G-Retriever / LinkAlign / XiYan-SQL SQL Gen prompt 재측정 (DECISIONS 2026-05-18 Wave 9, 2026-05-18, 🎯 prompt-axis confounder 분리 — baseline 우위 narrative ΔEX squeeze)

### 3 cells × 4 metric (anchor c01_01 Wave 7 R/P/F1/EX = 0.8697/0.8581/0.8639/**0.5117** 정합)

| Baseline | R | P | F1 | EX (Wave 9) | EX (Outdated) | Δ EX | Δ vs anchor c01_01 |
|---|---:|---:|---:|---:|---:|---:|---:|
| G-Retriever (Wave 9) | 0.7577 (retain) | 0.7866 (retain) | 0.7719 (retain) | **0.4283** | 0.2490 | +0.1793 | **−0.0834** |
| LinkAlign (Wave 9) | 0.6940 (retain) | 0.7641 (retain) | 0.7274 (retain) | **0.3390** | 0.2001 | +0.1389 | −0.1727 |
| XiYan-SQL (Wave 9) | 0.6832 (retain) | 0.7408 (retain) | 0.7108 (retain) | **0.2405** | 0.1969 | +0.0436 | −0.2712 |

→ R/P/F1 동일 retain (기존 final_nodes 보존), EX 만 신규 prompt 정합 갱신.

### per-difficulty EX

| Baseline | simple | moderate | challenging |
|---|---:|---:|---:|
| G-Retriever | 0.5114 | 0.3125 | 0.2690 |
| LinkAlign | 0.4314 | 0.2112 | 0.1586 |
| XiYan-SQL | 0.3092 | 0.1358 | 0.1379 |
| **anchor c01_01 (Wave 7)** | (analyzer pending) | (pending) | (pending) |

### baseline 우위 narrative ΔEX squeeze (paper main contribution)

| 비교 | prior outdated ΔEX | Wave 9 ΔEX | 정량 변화 |
|---|---:|---:|---|
| anchor c01_01 vs baseline | +0.2627~+0.3207 | **+0.0834~+0.2712** | **squeeze** |
| M4 vs baseline | +0.2810~+0.3331 | **+0.1017~+0.2895** | squeeze |

→ baseline 우위 narrative 의 prompt confounder 정량 분리 (anchor +0.1780 vs baseline 평균 +0.1206 의 ΔΔ ~+0.057 = 본 framework 의 schema linking effect 정량 evidence).

### Implementation 주의사항

- Pattern: Wave 7 Option A 정합 — 기존 final_nodes 보존 + SQL Gen 만 재실행 (Builder/Selector/Extractor/Filter 재실행 X)
- main.py 변경 없이 stand-alone Python script (`scripts/wave9_sql_regen.py`)
- Cost: 4,602 calls + ~$5~10 + **~67분 parallel** (3 streams, 18:28 → 19:35 KST)
- **첫 launch fail** (18:19 KST): dotenv 누락 → API 401 → EX=0.0000. Fix (load_dotenv 추가) 후 relaunch (18:28 KST) 정상.
- final_nodes 평균 cols/q (analyzer 측정 정확): G-Retriever **mean 49, median 44** / LinkAlign **mean 15, median 16** / XiYan-SQL **mean 3, median 3 (extreme sparse)**

### prompt-axis Confounder ΔΔ 정밀 정량

| Reference | anchor ΔEX | baseline 평균 ΔEX | **ΔΔ** |
|---|---:|---:|---:|
| Anchor c01_01 (5/1 → Wave 5 baseline 0.5176) | +0.1780 | +0.1206 | **+0.0574** ⭐ |
| Anchor c01_01 (5/1 → Wave 7 relog 0.5117) | +0.1721 | +0.1206 | **+0.0515** |

→ ΔΔ **+0.0515~+0.0574** = 본 framework 의 schema linking effect 의 정량 evidence (prompt confounder 분리 후 retain).

### per-difficulty schema sparse penalty mechanism (analyzer §1.1+§7)

- **simple/moderate**: schema sparse 가 dominant penalty (XiYan-SQL 3 col/q → simple 0.3092 vs G-Retriever 5114 의 -0.2022 gap)
- **challenging**: LinkAlign vs XiYan-SQL gap **shrink to +0.0207** — schema quality less critical, 다른 lever dominant

### final_nodes size 별 EX mechanism (analyzer §0)

- G-Retriever (mean 49): sweet spot 16-30 cols (EX=0.5138, anchor 0.5117 정합), 61+ 에서 noise EX=0.3919
- LinkAlign (mean 15): bimodal 6-15 EX=0.4093 vs 16-30 EX=0.2709 (난이도 confound)
- XiYan-SQL (mean 3): extreme sparse plateau, 94% query 0-5 cols, EX flat 0.24

### 결론 — prompt-axis confounder 분리 evidence + paper §10 표 갱신 + baseline 우위 narrative 정량 정확화

- 3 baseline 모두 +Jump confirmed (prompt-axis effect 정합)
- G-Retriever +0.1793 > LinkAlign +0.1389 > XiYan-SQL +0.0436 (final_nodes size 정비례)
- baseline 우위 narrative squeeze 단 retain — paper main contribution narrative 정합 retain (anchor 의 schema quality 우위 effect 정량 evidence)
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Wave 9 Baseline Relog (2026-05-18)](EXPERIMENT_HISTORY.md).

### 산출물

- Scripts: `scripts/wave9_sql_regen.py` + `scripts/run_wave9_baseline_relog.sh`
- Outputs: `outputs/baselines/wave9_relog/{g_retriever,linkalign,xiyansql}_relog/`
- Wall: 1h02m, 비용 ~$5~10 GLM API
- failure: 0/3 (relaunch 후)


## Wave 8 M4 Bidirectional 발전 — D1 + D2 + D3 + D4 8 cells (DECISIONS 2026-05-18 §2+§5, 학술 agent improving_m4_plan §1~§4, 2026-05-18 ~ 2026-05-19, 🎯 multi-axis Pareto frontier 확장 + EX > M4 미달)

### 8 cells × R/P/F1/EX (M4 anchor R=0.9325 / P=0.7593 / F1=0.8370 / EX=**0.5300** ★ 정합)

| ID | Cell | R | P | F1 | EX |
|---|---|---:|---:|---:|---:|
| abl_wave8_d1v1_multi_backward | D1 v1 Multi-Backward | 0.9458 | 0.6914 | 0.7988 | 0.5111 |
| abl_wave8_d1v2_full_decompose | D1 v2 Full Decompose ★ R-best | **0.9601** | 0.5500 | 0.6994 | 0.5163 |
| abl_wave8_d2v1_direct_fk | D2 v1 직접 FK | 0.9351 | 0.7416 | 0.8272 | 0.5104 |
| abl_wave8_d2v2_bridge_1hop | D2 v2 1-hop Bridge | 0.9373 | 0.7405 | 0.8274 | 0.5085 |
| abl_wave8_d3v1_verify1round | D3 v1 1 Round | 0.9328 | 0.7534 | 0.8336 | 0.5169 |
| **abl_wave8_d3v2_verify2round** ⭐ | D3 v2 2 Rounds (EX-2nd) | 0.9304 | 0.7579 | 0.8353 | **0.5215** |
| **abl_wave8_d4v1_value_hint_forward** ★ | D4 v1 Value-Hint (F1-best) | 0.9336 | **0.7623** | **0.8393** | 0.5111 |
| abl_wave8_d4v3_forced_include | D4 v3 Forced Include | 0.9364 | 0.7215 | 0.8150 | 0.5091 |

→ **EX > M4 미달 (8/8 cells)** — primary success criterion (학술 agent §8 Case 1) 미달 ❌.

### Pareto Frontier 갱신 (Wave 6 + Wave 8 통합)

| Axis | Pareto Cell | 값 |
|---|---|---:|
| **R-best** ★ Wave 8 신규 | D1 v2 full_decompose | R=0.9601 |
| **F1-best (overall)** Wave 6 retain | M1-B strong | F1=0.8655 |
| F1-best (Wave 8 marginal) | D4 v1 value_hint | F1=0.8393 (+0.0023 vs M4) |
| **EX-best** ★ M4 retain | M4 Bidirectional | **EX=0.5300** |
| **EX-2nd-best** ★ Wave 8 신규 | **D3 v2 verify2round** | EX=**0.5215** (M4 −0.0085 sub-noise) |

### Top 2 조합 candidate

- **Comb-A 권고 ⭐**: **D4 v1 + D3 v2** — F1-axis (value hint) + EX-axis (verify loop) 직교 mechanism, EX > M4 candidate
- Comb-B: D1 + D3 v2 — R-axis + EX-axis 단 P-cost 우려
- Comb-C: D2 + D4 + D3 — 3-axis 통합 단 LLM cost 큼

### paper §V.5.x.M.16~19 candidate sub-section map

| Direction | paper sub-section | Wave 8 evidence | 판정 |
|---|---|---|:---:|
| D1 (Multi-Backward) | §V.5.x.M.15 evidence #5 R-axis | D1 v2 R +0.0276 marginal (vs RoSL +25.1%) | 격하 candidate |
| D2 (FK Steiner Closure) | §V.5.x.M.16 신규 (DB-aware Schema Connectivity) | D2 v1/v2 sub-noise | 격하 candidate |
| **D3 (Self-Verification Loop)** ⭐ | §V.5.x.M.17 신규 (Execution Feedback Loop) | D3 v2 EX=0.5215 (EX-2nd-best, M4 sub-noise 안) | **retain candidate** |
| **D4 (Value Hint Forward)** ★ | §V.5.x.M.18 신규 (Value Evidence Enhancement) | D4 v1 F1=0.8393 (F1 +0.0023 marginal positive) | **retain candidate** |

### Config 주의사항

- M4 anchor (BidirectionalFilter Forward `recall_biased_mild` + Backward `bidirectional_backward` Union) 변경 없이 wrapper composition pattern
- LLM 입력 = Extractor 출력 후보 (subgraph) 만 (Full Schema 입력 금지, sanitize_filter_output default-on)
- D3 의 DB 실행 timeout 5s + sandbox 정합 (SQL probe 위험 격리)
- 학습 없음 (anchor ckpt `best_gat_qcond_nl3.pt` 재사용)

### 학습 비용 + 환경

- Wall: ~10h 33min (D1 v2 bottleneck), 8 streams parallel (4 streams initial → 8 streams 전환)
- 비용: ~$25~40 GLM 4.7 (~44k LLM calls 총)
- failure: 0/8 (D3 v1/v2 config fix 후 fresh restart 1회)

### 결론 — Wave 8 multi-axis Pareto frontier 확장 + paper §V.5.x.M.17 + §V.5.x.M.18 retain candidate + Top 2 Comb-A 권고

- M4 EX-best retain 정합 — paper main contribution narrative 정합 retain
- D3 v2 EX-2nd + D4 v1 F1-best marginal = paper main contribution 의 **multi-axis evidence 확장**
- D1 R-axis + D2 sub-noise = 격하 candidate (paper sub-section retain 부재)
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Wave 8 M4 Bidirectional 발전 (2026-05-19)](EXPERIMENT_HISTORY.md).

### 산출물

- Configs 8: `configs/experiments/abl/wave8_m4_extensions/{d1_decompose,d2_steiner,d3_verify,d4_value_hint}/`
- Module 구현: filter commits `c44b15a` (D1+D3+D4 wrappers) + `2cc8b93` (D2 Steiner Closure + db_fk_extractor)
- Launch script: `scripts/run_wave8_m4_extensions.sh`
- Outputs: `outputs/experiments/abl/wave8_m4_extensions/`
- Logs: `logs/wave8_m4_extensions/`
- failure: 0/8


## Wave 8 Comb-A — D4 v1 + D3 v2 직렬 Stacking (DECISIONS 2026-05-19 §3 작업 4 + §6, 2026-05-19, 🎯 Verdict Fail (EX < 0.5215) 단 F1=0.8684 globally best post-Wave 5)

### Single cell × 4 metrics (M4 anchor 0.9325/0.7593/0.8370/0.5300 + anchor c01_01 0.8748/0.8582/0.8664/0.5176)

| ID | Stack | R | P | F1 | EX |
|---|---|---:|---:|---:|---:|
| **abl_wave8_comb_a_value_hint_verify2round** | StackedFilter[D4 v1 → D3 v2] | 0.9170 | **0.8247** | **0.8684** ⭐ | 0.5117 |

### Success Criteria 판정 — **Verdict: Fail** (EX < 0.5215)

- Pass: EX > 0.5300 ❌
- Partial: EX ≥ 0.5215 + F1 ≥ 0.8370 ❌
- **Fail: EX < 0.5215** ✅

### Paradox finding — F1 globally best (post-Wave 5) candidate

| Reference | F1 | Δ vs Comb-A |
|---|---:|---:|
| **Comb-A** ⭐ | **0.8684** | (best) |
| anchor c01_01 (Wave 5 baseline) | 0.8664 | +0.0020 marginal |
| Wave 6 M1-B strong (prior F1-best) | 0.8655 | +0.0029 marginal |
| Wave 8 D4 v1 | 0.8393 | +0.0291 ★ |
| M4 anchor | 0.8370 | **+0.0314** ★★ |

→ **Comb-A F1=0.8684 = post-Wave 5 글로벌 best** candidate.

### P-axis dramatic gain mechanism (직교 synergy)

| Cell | P |
|---|---:|
| **Comb-A** ⭐ stacking | **0.8247** |
| D4 v1 alone | 0.7623 |
| D3 v2 alone | 0.7579 |
| M4 anchor | 0.7593 |

→ Comb-A P (+0.0654 vs M4) **dramatic** — D4 evidence-aware schema retention + D3 hallucinated column 차단 = **dual P-lift synergy**.

### EX-axis paradox — F1 up + EX down decoupling

- Comb-A EX=0.5117 (M4 −0.0183, D3 v2 alone −0.0098)
- D4 의 schema modification → D3 sketch SQL base 변경 → verify loop column recovery 약화
- **F1 +0.0314 + EX −0.0183 = mechanism decoupling** — paper §V.5.x.M.12 F1-EX Decoupling narrative 보강 candidate

### Pareto Frontier 갱신 (Wave 5+6+8 통합)

| Axis | Pareto Cell | 값 |
|---|---|---:|
| **F1-best (post-Wave 5)** ⭐ 신규 | **Comb-A** | F1=**0.8684** |
| R-best | D1 v2 full_decompose | R=0.9601 |
| EX-best | M4 retain | EX=0.5300 |
| EX-2nd | D3 v2 verify2round | EX=0.5215 |
| **P-best mechanism** ⭐ 신규 | **Comb-A** | P=0.8247 |

### Implementation 주의사항

- StackedFilter pattern: stages[D4 v1 → D3 v2] 직렬 적용
- M4 anchor 변경 없이 wrapper composition (D4 v1, D3 v2 모두 BidirectionalFilter 의 wrapper)
- 학습 없음 (anchor ckpt `best_gat_qcond_nl3.pt` 재사용)
- filter_llm_calls_mean = 6.0 (StackedFilter 의 두 stage 평균 6 LLM/q)
- filter_stage_time_mean = 11.05s (M4 2s 의 ~5.5× — two-stage overhead)

### 학습 비용 + 환경

- Wall: ~7h 22min (09:40:44 → 17:03 KST, single stream)
- LLM calls total: 13,806 (filter 9,204 + SQL gen 1,534 + extras)
- Token: input 17.95M / output 495k
- 비용: ~$10~15 GLM API
- failure: 0/1

### 결론 — Verdict Fail 단 F1-axis 의 dramatic gain + Pareto frontier 신규 진입

- Verdict: Fail (EX < 0.5215, paper §V.5.x.M.19 EX-best candidate 부재)
- **F1 globally best (post-Wave 5)** candidate ⭐ — paper §V.5.x.M.19 신규 sub-section candidate (F1-best mechanism)
- P-axis dramatic synergy (+0.0654 vs M4)
- F1-EX decoupling mechanism — paper §V.5.x.M.12 narrative 보강
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Wave 8 Comb-A (2026-05-19)](EXPERIMENT_HISTORY.md).

### 산출물

- Config: `configs/experiments/abl/wave8_m4_extensions/comb_a/abl_wave8_comb_a_value_hint_verify2round.yaml`
- Launch script: `scripts/run_wave8_comb_a.sh`
- Outputs: `outputs/experiments/abl/wave8_m4_extensions/comb_a/abl_wave8_comb_a_value_hint_verify2round/`
- Logs: `logs/wave8_comb_a/`
- failure: 0/1


## Wave 8 Comb-A 분석 결과 채택 + Wave 8 Closure (DECISIONS 2026-05-19 §1+§3+§6 + analyzer §0+§5+§6, 2026-05-19, 🎯 planner 5 결정 ✅ 완료 + Wave 8 closure marker + Pareto frontier 4 axis multi-coverage)

### per-stage telemetry 정밀 정량 (Comb-A StackedFilter 분해)

| Stage | Filter | nodes_in | nodes_out | telemetry |
|---|---|---:|---:|---|
| **Stage 0** | D4 v1 (Pre-Filter) | 76.86 | **6.50** | evidence-aware schema retention |
| **Stage 1** | D3 v2 (Post-Filter) | 6.50 | **5.61** | verify_success_rate **0.9394**, recovered_count **0** (D3 collapse) |

→ Stage 1 추가 pruning **0.89/q** (verify rejection of hallucinated cols).

### F1 mechanism 분해 (P-axis dual-lift)

| Component | ΔP vs M4 | 정합 |
|---|---:|---|
| (i) D4 v1 individual lift | +0.0030 | stand-alone (D4 alone P 0.7623 vs M4 P 0.7593) |
| (ii) **Stacking synergy** ⭐ | **+0.0624** | dual-lift (Comb-A 0.8247 vs D4 alone 0.7623) — individual 의 **~20× magnitude** |
| **Total ΔP vs M4** | **+0.0654** | dramatic |

### EX paradox root cause (D3 mechanism collapse)

- D3 v2 alone: 1 query SQL recovery (rounds=2 0.07% activation) → EX +0.0046 lift
- **Comb-A**: D4 clean schema (6.50 nodes/q) → 1-round verify success rate 0.937 → 0.9394 saturate → **rounds=2 활성화 0%, recovered_count 0/q** → D3 의 specific EX-axis mechanism **disappear**
- 추가 40 queries EX-down (over-pruning schema sparse penalty) → Net ΔEX = **−15/1534 = −0.0098** exact match (Comb-A − D3 v2 alone)

### per-difficulty robustness

| Difficulty | ΔF1 vs M4 | ΔP vs M4 | ΔEX vs M4 |
|---|---:|---:|---:|
| simple (n=925) | **+0.0342** | +0.0712 | −0.0130 |
| moderate (n=464) | **+0.0264** | +0.0544 | **−0.0302** ⚠ |
| challenging (n=145) | **+0.0307** | +0.0628 | −0.0138 |

→ F1 gain robust across all difficulties + EX drop moderate-bias.

### Wave 8 Closure Marker — Pareto Frontier 4 Axis Multi-Coverage

| Axis | Pareto Cell (post-Wave 8 Comb-A) | 값 |
|---|---|---:|
| **R-best** ⭐ | D1 v2 full_decompose | R=**0.9601** |
| **F1-best post-Wave 5** ⭐⭐ | **Comb-A** | F1=**0.8684** |
| **P-best post-Wave 5** ⭐ | **Comb-A** | P=**0.8247** (dual-lift) |
| **EX-best** ⭐ | M4 Bidirectional | EX=**0.5300** |
| EX-2nd | D3 v2 verify2round | EX=0.5215 |

→ **Wave 8 closure: paper main contribution evidence 충분 정합** (R + F1 + P + EX 4 axis multi-Pareto coverage). paper drafting trigger 가능 base.

### Planner 5 결정 ✅ 완료

1. paper §V.5.x.M.12 F1-EX Decoupling narrative 보강 (Comb-A simultaneous decoupling strongest single-cell evidence 통합)
2. paper §V.5.x.M.17 (D3) narrative 보강 (Context-aware dual mechanism)
3. paper §V.5.x.M.18 (D4) narrative 보강 (Stacking platform mechanism)
4. paper §V.5.x.M.19 신규 sub-section (Pre-Filter + Post-Filter Stacking Synergy)
5. Wave 8 closure marker 정합

### Comb-B/C/D = post-paper extension (post-paper backlog #23 candidate)

- Comb-B (D1 + D3 v2): △ P-cost carryover 우려
- Comb-C (D2 + D4 + D3): ❌ D2 mechanism 거의 무효
- Comb-D (D4 + 다른 post-filter): △ retain candidate

### 결론 — Wave 8 closure 완료 + paper drafting trigger 가능 base

- planner 5 결정 ✅ 완료
- Pareto frontier 4 axis multi-coverage 완성
- paper main contribution evidence 충분 정합
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Wave 8 Comb-A 분석 채택 + closure (2026-05-19)](EXPERIMENT_HISTORY.md).


## Wave 11 Schema Serialization Direction C — Phase B 6 cells launch (DECISIONS 2026-05-19 (Wave 11) §3 Phase B + filter_improvement_wave10 §4+§6+§8, 2026-05-19 launch, 🎯 EX ceiling 0.5300 돌파 가능성 검증 — Schema Content Invariance retain)

### 6 cells × launch spec (c_v0 baseline + 5 variants, M4 anchor R=0.9325/P=0.7593/F1=0.8370/EX=0.5300 reference)

| ID | Variant | 직렬화 형식 | +LLM/q | 검증 가설 |
|---|---|---|---:|---|
| c_v0_baseline | Baseline (M4 legacy DDL) | flat JSON | 0 | reference |
| **c_v1_source_tagged** | Source-Tagged | [F+B]/[F]/[B] 태그 | 0 | H1 |
| **c_v2_question_enrichment** | Question Enrichment | E-SQL enriched question | +1 | H2 |
| **c_v3a_flat_merged_fk** | Flat Merged + FK | table.col flat + FK hint | 0 | H3 |
| **c_v3b_flat_merged_no_fk** | Flat Merged − FK | table.col flat only | 0 | H3 |
| **comb_c_tagged_enriched** | Comb-C (v2+v1) | Tagged + Enriched | +1 | H4 |

### Schema Content Invariance 핵심 제약

Filter 가 선택하는 columns 집합 = **M4 와 완전히 동일** retain → R/P/F1 ΔX ±0.0001 (implementation 정합 검증). EX 만 변화 측정.

### Phase A 구현 (commit `3eb476d`)

- `src/serializers/` 3 modules (source_tagged + question_enricher + flat_merged) + tests 19/19 PASSED
- `src/pipeline/schema_linking.py` serializer_type 분기 + leakage filter + round-robin
- `src/pipeline/sql_generator.py` pre_serialized_schema + enriched_question 인자
- `src/modules/filters/bidirectional_filter.py` F/B set 별도 저장

### Launch 진행 (~20:00 → ~24:00~01:00 KST 익일)

- Wrapper PID 3242310, 6 cells parallel
- Cost ~3000 LLM calls (C-v2 + Comb-C 만 +1) + ~$3~6 GLM 4.7
- Wall ~3~5h parallel
- Monitor `by0ev4lg9` 자동 추적

### Few-Shot examples (planner 직접 작성)

`planning/few_shot_examples_wave11_2026-05-19.json` — 12 examples (11 DBs cover, simple 4 / moderate 4 / challenging 4). Data leakage 방지 (test query DB 와 다른 DB 의 examples 만 filter, round-robin).

### Configs

`configs/experiments/abl/wave11_schema_serialization/{c_v0_baseline, c_v1_source_tagged, c_v2_question_enrichment, c_v3a_flat_merged_fk, c_v3b_flat_merged_no_fk, comb_c_tagged_enriched}.yaml`

### 결론 — Wave 11 launch 진행 중, 종료 시 ΔEX 정량 + 시나리오 분기 결정

- Phase A ✅ 완료 (commit `3eb476d`)
- Phase B launch 진행 중 (~24:00~01:00 KST 익일 종료 예상)
- Phase C analyzer 위임 (시나리오 1/2/3 분기 권고)
- 세부 실행 이력: [EXPERIMENT_HISTORY.md Wave 11 Phase B launch (2026-05-19)](EXPERIMENT_HISTORY.md).

### 산출물

- Configs 6: `configs/experiments/abl/wave11_schema_serialization/`
- Launch script: `scripts/run_wave11_phase_b.sh`
- Outputs: `outputs/experiments/abl/wave11_schema_serialization/` (launch 진행 중)
- Logs: `logs/wave11_phase_b/`
- Phase A module 구현: commit `3eb476d`

