# Selector 모듈 실험 계획 — 9 아키텍처 제안 중 Selector 관련 축

> **⚑ 먼저 루트 계획을 읽을 것**: [/home/hyeonjin/thesis_refactored/EXPERIMENT_PLAN.md](/home/hyeonjin/thesis_refactored/EXPERIMENT_PLAN.md) — 전 모듈 통합 로드맵, Cross-Module Dependency, 통합 실험(int_01~08), 우선순위 Phase A~E, 논문 매핑이 거기에 있다. **루트 PLAN은 수정하지 않는다** — 수정이 필요하면 루트 세션에 요청.
> **이 파일의 역할**: 루트 PLAN에서 Selector에 할당된 5축(S-I ~ S-V)의 **모듈 내부 구현 상세**만 담는다.
>
> **현재 진입점**: `EnsembleSelector` (α=0.85, baseline), `DirectGATSelector` (a03_17 anchor F1=0.6940).
> **Selector의 실질 기여**: score quality 자체 — top-k는 PCST 게이트가 아님을 반복 강조.
> **선결 의존성**: S-II는 Builder B-I, S-III는 Builder B-II, S-V는 Builder B-III에 의존. 루트 PLAN Phase A 완료 전에는 S-V/S-III/S-II 블록.

---

## 이 모듈이 받아야 할 5가지 제안

| # | 이름 | Selector의 역할 | 우선순위 |
|---|------|----------------|---------|
| S-I | **RL Schema Linker (Schema-R1 / GRPO)** (원안 #1) | Selector을 policy network으로 RL 학습 — precision/recall 보상 | 중~상 |
| S-II | **Relational Foundation Model Zero-Shot Transfer Encoder** (원안 #2) | PLM을 RFM으로 교체, 쿼리·스키마 공동 encoder | 상 |
| S-III | **Edge Hypergraph Attention Network (EHGAT)** (원안 #3) | node-centric GAT → edge-centric line graph GAT | 중 |
| S-IV | **Extractive Decoder-only LLM Schema Selector** (원안 #7) | LLM을 token-level span extraction으로 사용해 score 계산 | 중 |
| S-V | **Symbolic-Neural Layer 1** (원안 #9 Layer 1) | FK reachability prior + GAT score 결합 (Builder가 제공한 matrix 사용) | 상 |

---

## S-I. RL Schema Linker (Schema-R1 / GRPO)

### 동기
- 현재 GAT은 supervised(BCE + InfoNCE). BIRD의 noisy label(gold 집합의 불완전성)을 보상 기반으로 보정.
- GRPO(Group Relative Policy Optimization)은 샘플링된 group 내 reward baseline을 사용 — PPO critic 없이 학습 가능.

### 설계 요소
- **Policy**: 현재 `EnsembleSelector` 의 score 분포 → top-k 샘플링 (temperature sampling).
- **Reward**: predicted set vs gold set F1 (or Recall with λ·Precision).
- **Group sampling**: 쿼리당 N=8 샘플, 그룹 내 reward z-score.
- **Warm-start**: 기존 `best_gat_model.pt` 에서 시작.

### 인터페이스
```python
class RLSchemaLinker(EnsembleSelector):
    def select(self, scores, candidates, question, graph_data, metadata, **kwargs):
        # train: sample with temperature; inference: greedy
        return seeds
    def compute_reward(self, selected_set, gold_set) -> float
    def grpo_step(self, group_of_selections, group_of_rewards)
```

### 의존성 / 주의
- Gold 집합이 noisy — reward가 실제 파이프라인 F1 (Extractor+Filter 통과 후) 이어야 더 정확하나, sampling cost 폭발.
- 초기 proxy reward: **Selector 단계 P80-covered gold recall**.
- 학습 불안정성 — KL constraint (참조 policy = supervised GAT) 필수.

### 예상 실험
| 실험 ID | 학습 variant | Reward |
|---------|-------------|--------|
| `abl_s06_rl_01` | GRPO from T4 | Selector-level Recall@P80 |
| `abl_s06_rl_02` | GRPO from T7 (SuperNode) | Selector-level F1 proxy |
| `abl_s06_rl_03` | 전체 pipeline reward | End-to-end F1 (느림, 소수 epoch) |

### 검증
- Validation R/P/F1 곡선이 supervised 체크포인트보다 상승.
- Policy entropy가 0으로 수렴하지 않는지 (reward hacking 방지).

---

## S-II. Relational Foundation Model (RFM) Zero-Shot Encoder

### 동기
- 현재 PLM은 MiniLM-L6 (384-dim, general-purpose). **DB schema 특화 pre-training이 아님**.
- 최근 Schema-Llama, TableLlama, TaPEx 계열 RFM들은 schema 토큰/구조를 직접 입력받아 훨씬 강한 zero-shot transfer.
- Builder가 serialize한 input을 받아 **쿼리·스키마 공동 임베딩** 생성.

### 설계 요소
- Backbone 후보: TableLlama / Schema-Llama / BGE-M3 (large variant).
- Encoder API: `encode_schema(serialized_str) -> (node_embeddings, mask)` — serialized 토큰 span을 각 node에 역매핑.
- Query-conditioning: concat `[Q] question [SEP] [SCHEMA] ...` 또는 cross-attention.
- Fine-tune 가능하나, **zero-shot 먼저 평가** — 이게 RFM의 세일즈 포인트.

### 인터페이스
```python
class RFMEncoder(BaseEncoder):
    def encode(self, texts) -> torch.Tensor  # 기존 계약 유지
    def encode_with_schema_context(self, question, serialized_schema) -> Dict[str, Tensor]

class RFMSelector(EnsembleSelector):
    def select(...):  # RFM의 query-conditioned node embedding 사용
```

### 의존성 / 주의
- Builder의 `RFMCompatibleBuilder` 필수 (EXPERIMENT_PLAN_builders B-I).
- GPU 메모리 — 30B 체크포인트면 PLM 캐시 생성만 수 시간. **사전 계산 후 저장** 전략 필수.
- 기존 HeteroData 캐시와 분리 (`_rfm` suffix).

### 예상 실험
| 실험 ID | Encoder | 구조 |
|---------|---------|------|
| `abl_s06_rfm_01` | BGE-M3 large (zero-shot) | Ensemble baseline 대체 |
| `abl_s06_rfm_02` | TableLlama-7B | schema-aware encoder |
| `abl_s06_rfm_03` | Fine-tuned RFM (small LoRA) | 1 epoch BIRD fine-tune |

### 검증
- ROC-AUC / PR-AUC vs MiniLM 기준선(0.776 / 0.317).
- Unseen DB(dev)에 대한 transfer 격차 관찰.

---

## S-III. Edge Hypergraph Attention (EHGAT)

### 동기
- 현재 GAT은 node(table/column/fk_node)에 attention — **FK가 노드임에도 edge-centric structural pattern은 간접학습**.
- Line graph 변환(Builder B-II) 후 edge → node로 승격하면 "FK path가 score의 주체"가 됨.

### 설계 요소
- Input: Builder B-II의 `LineGraphData`.
- HeteroGATv2 대신 **HomogeneousGATv2 on line graph** (단순화 가능) 또는 기존 heterograph 위에 edge-attention layer 추가.
- Label: edge_node의 gold 여부 (양 끝 모두 gold when available).
- 최종 score → 원 노드로 역매핑: `node_score = mean(edge_scores of incident edges)`.

### 인터페이스
```python
class EHGATSelector(BaseSelector):
    def __init__(self, line_graph_model_ckpt, edge_to_node_aggregation='mean'):
        ...
    def select(self, scores, candidates, question, graph_data, metadata, **kwargs):
        line_data = LineGraphBuilder.build(...)
        edge_scores = self.model(line_data)
        node_scores = aggregate_to_nodes(edge_scores, metadata["edge_node_to_orig"])
        self.latest_scores = node_scores
        return candidates
```

### 의존성 / 주의
- Builder B-II (LineGraphBuilder) 선결.
- 학습 데이터 재생성 필요 — 기존 `_graphs.pt` 재사용 불가.
- FK 노드 성능 집중 검증 (3-table JOIN gold가 많은 쿼리).

### 예상 실험
| 실험 ID | 구조 | 비고 |
|---------|------|-----|
| `abl_s06_ehgat_01` | LineGraph + homogeneous GAT | EHGAT pilot |
| `abl_s06_ehgat_02` | Node+Edge 앙상블 (score fusion) | 기존 GAT과의 상호보완 |

### 검증
- 특히 **FK 노드 recall** 에서 개선 확인.
- Bridge table 인식율 (3-table JOIN 쿼리만 필터링해 측정).

---

## S-IV. Extractive Decoder-only LLM Schema Selector

### 동기
- Selector에 LLM을 token-level span classifier로 활용 — full DB schema를 프롬프트로 넣고 "이 쿼리에 필요한 컬럼들을 span으로 추출".
- GAT score와 직교한 signal: LLM은 **semantic selectivity에 강함**.

### 설계 요소
- Input prompt: `f"Query: {q}\nSchema:\n[TAB_1] col_1.1, col_1.2...\n[TAB_2] ..."`.
- Output: tokens 혹은 json `{"selected": ["t1.c1", "t1.c2"]}`.
- **Logit-level scoring**: selected column의 **생성 log-prob** 또는 classifier head attach.

### 인터페이스
```python
class ExtractiveLLMSelector(BaseSelector):
    def __init__(self, model_name, api_client):
        ...
    def select(self, scores, candidates, question, graph_data, metadata, **kwargs):
        schema_str = metadata.get("serialized_schema", ...)
        selection = self.llm.extract(question, schema_str)
        node_scores = map_selection_to_scores(selection, metadata)
        self.latest_scores = node_scores
        return candidates
```

### 의존성 / 주의
- **Filter(F-IV)** 와 동일 LLM 재사용 가능 — 단, Selector는 학습/prior 제공, Filter는 refinement 역할 구분 필요.
- Latency — 쿼리당 1 LLM call (>=500ms).
- Token 폭발 위험: BIRD DB 중 50+ 컬럼 DB는 컨텍스트 초과 가능 → **FK reachability prior로 선절단**.

### 예상 실험
| 실험 ID | LLM | 비고 |
|---------|-----|-----|
| `abl_s06_xllm_01` | Qwen3-Coder-30B | GAT와 병렬, α·GAT + (1−α)·LLM |
| `abl_s06_xllm_02` | GPT-4o-mini | Backbone 민감도 |

### 검증
- LLM-only selector recall/precision vs GAT.
- 결합 시(α 튜닝) 순 기여.

---

## S-V. Symbolic-Neural Layer 1 (Neurosymbolic)

### 동기
- 제안 #9 Layer 1 = **결정론적 FK reachability prior** 를 neural score와 결합.
- Builder B-III가 이미 `fk_reachability`, `fk_components` metadata 제공 → Selector가 이를 score rescaling에 활용.

### 설계 요소
- 쿼리에 등장하는 entity(NE / string match)와 각 테이블의 매칭 정도 → "anchor tables" 집합.
- Anchor tables 간 FK-reachable 여부로 "plausible schema backbone" 식별.
- GAT score에 symbolic boost: `score_v = GAT(v) + λ · I[v ∈ reachable_from_anchors]`.

### 인터페이스
```python
class NeurosymbolicLayer1Selector(EnsembleSelector):
    def select(self, scores, candidates, question, graph_data, metadata, **kwargs):
        anchors = identify_anchors(question, metadata)  # NER + string match
        reach_mask = build_reach_mask(anchors, metadata["fk_reachability"])
        boosted = scores + self.lambda_sym * reach_mask
        self.latest_scores = boosted
        return candidates
```

### 의존성 / 주의
- Builder B-III 선결 (FK reachability precomputed).
- Anchor identification 품질이 결정 — spaCy NER + column name substring match로 시작.
- λ 튜닝: 작은 값(0.05~0.2)부터.

### 예상 실험
| 실험 ID | 구조 | 비고 |
|---------|------|-----|
| `abl_s06_ns1_01` | Ensemble + FK-reach boost (λ=0.1) | Layer 1 pilot |
| `abl_s06_ns1_02` | Direct + FK-reach boost | SuperNode-Direct 조합 |

### 검증
- Gold SQL의 JOIN 대상 테이블이 anchor → reachable 영역에 포함되는 비율.
- 3-table JOIN 쿼리 F1 (이 쿼리군에 특화 효과 기대).

---

## QCondGAT 계열 Ablation Track (2026-04-21 지도교수 피드백)

> **출처**: [planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md](/home/hyeonjin/thesis_refactored/planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md) §4 의견 1/3/4, §7 (approved), §8 Proposal A/D/E.
> **스토리라인 우선순위 (§8, 2026-04-28 발표 15~20분, 중요 지점만)**: **A > D > E** (A 는 core, D/E 는 시간 여유 시).
> **루트 PLAN 파급**: int_05 전제 "SuperNode v2 (directed SN→node + top-k selective edge, Raw Score 기준 선별)" 로 재정의 (§7.1). §9 리스크에 SN v2 over-smoothing 재등장 가능성 추가 (§7.4).
> **Anchor baseline**: `s03_a02_03_xiyan_filter` (Ensemble + AdaptivePCST + XiYan, F1=0.4936) — 모든 cell 이 이 anchor 대비 비교.

---

### A. Raw Score Ablation 축 (의견 1)

#### 동기
- **BCE 의 기여를 분리** 하기 위해 Raw Score (α=0, GAT 미사용) vs GAT-blend (α>0) 를 Ensemble / QCond 양쪽에서 측정.
- Selector / Extractor / Filter **3 단계 cumulative R/P/F1** 로 분해 (§10 Q3 A3 = cumulative).
- 기존 s04_04 (QCond α=0), s04_05 (SuperNode α=0) 재활용 + Ensemble α=0 셀이 공백 → **신규 실험 1 개** 로 매트릭스 완성.

#### 실험 매트릭스 (5 행 × 3 단계 cumulative)

| 모델 | α | Extractor | Filter | 실험 ID | 비고 |
|------|---|-----------|--------|---------|-----|
| Baseline (VectorOnly) | — | AdaptivePCST | XiYan | 기존 `s03_a02_01_vector_xiyan_filter` | reference |
| **Ensemble Raw** | **0** | AdaptivePCST | XiYan | **`abl_sel_raw_ens_01`** (신규) | Ensemble α=0 공백 채움 |
| Ensemble GAT-blend | 0.85 | AdaptivePCST | XiYan | 기존 `s03_a02_03_xiyan_filter` | best baseline |
| QCond Raw | 0 | AdaptivePCST | XiYan | 기존 `s04_04_qcond_a0_xiyan` | 재라벨 |
| QCond GAT-blend | 0.85 | AdaptivePCST | XiYan | 기존 `s04_01_qcond_a085_xiyan` | — |

- **단계별 cumulative**: Selector top-k=15 R/P/F1 → PCST 통과 후 R/P/F1 → XiYan 통과 후 R/P/F1. 3 stage × 5 row = 15 cell 표.
- 원자료: 기존 4 실험 로그 재집계 + 1 개 신규 실험.

#### 인터페이스 / 구현
- `EnsembleSelector(alpha=0.0)` 이미 지원 — config 만 분리.
- **신규 config**: `configs/experiments/abl/sel/rawscore/abl_sel_raw_ens_01.yaml`
  - Selector: Ensemble, α=0.0 (=cosine-only with ensemble wrapper)
  - Extractor: AdaptivePCST (p80), Filter: XiYanFilter (Qwen3-Coder-30B)
- **단계별 분해 분석**: Analyzer 세션 담당 (§9 프롬프트 3). Output: `notebooks/analysis_results/stagewise_qcond_ablation.md`.

#### 검증
- 각 stage 에서 Raw vs GAT-blend **F1 차이** → BCE(GAT) 의 단계별 순 기여 분리.
- 가설: (a) Selector stage 에서 GAT-blend +P, Raw +R 경향, (b) Filter 통과 후엔 gap 축소.

---

### D. SuperNode Directed Edge (의견 3)

#### 동기
- 현재 SuperNode: schema node ↔ SN 양방향 → SN 이 GAT 메시지 전파로 **희석**.
- 단방향 (SN → schema node) 으로 바꾸면 SN 은 encoder(question) 임베딩 그대로 유지, schema node 만 그 신호를 받음.
- Over-smoothing 대응 — SN 이 쿼리 정보 anchor 역할을 유지.

#### 설계 요소
- **구현 위치**: `src/models/gat_network.py` (SuperNode 구현) / `src/models/gat_network_v2.py` (v2 SuperNode 지원 시).
- **edge_index_dict 변경**:
  - 기존 양방향: `('query', 'to', 'table')` + `('table', 'to', 'query')` 양쪽
  - 신규 directed: `('query', 'to', 'table')` / `('query', 'to', 'column')` 만. 역방향 제거.
- 학습 시 SN 임베딩은 encoder 출력 고정 — GAT layer 내에서 schema 메시지 수용 안 함.
- 재학습 필수 — 기존 `best_gat_query_supernode.pt` (T7), `*_direct.pt` (T9) 모두 양방향 전제로 학습됨.

#### 인터페이스
```python
class SchemaHeteroGATv2(...):
    def __init__(self, ..., supernode_directed: bool = False):
        self.supernode_directed = supernode_directed

    def _build_supernode_edges(self) -> List[Tuple[str, str, str]]:
        fwd = [('query', 'to_t', 'table'), ('query', 'to_c', 'column')]
        if self.supernode_directed:
            return fwd
        return fwd + [('table', 't_to', 'query'), ('column', 'c_to', 'query')]
```

- 빌더 측 변경 없음 — 기존 SuperNode builder 의 edge 구성에서 역방향을 옵션으로 생략.

#### 예상 실험 ID
| ID | Variant | Checkpoint | 비고 |
|----|---------|-----------|-----|
| `abl_sel_sn_directed_bce` | Direct (BCE only) | `best_gat_sn_directed_direct.pt` (신규) | T9 대체 후보 |
| `abl_sel_sn_directed_proj` | Projector (BCE + InfoNCE) | `best_gat_sn_directed_proj.pt` (신규) | T7 대체 후보 |

#### 검증
- **SN embedding drift**: 학습 epoch 별 SN 노드 embedding L2 norm 추이 — 양방향은 감소 (희석), 단방향은 유지 예상.
- **Distant-node signal**: `node_distance_to_supernode × recall` 분포 — distant node 가 SN 신호를 받는지 (§7.4 리스크: 단방향에서 distant 연결성 저하 가능).
- **Val R@15 vs T7/T9**: directed 가 상승 시 over-smoothing 완화 근거.

---

### E. SuperNode Top-k Selective Connection (의견 4)

#### 동기
- SN 을 모든 schema node 에 연결 시 edge 수백 개 → attention 분산으로 SN 신호 희석 가속.
- **Top-k selective connection**: 사전 선별된 k 개 node 에만 SN→node edge 를 그려 attention 집중.
- 의견 3 (directed) 와 **조합 시 시너지** — 단방향 + 희소 연결로 SN anchor 성격 극대화.

#### 선별 기준 (§10 Q2 A2)
- **Phase 1 권장 1 개 = Raw Score** (cosine similarity of encoder output). 이유:
  1. 의견 1 Raw Score ablation 축과 인프라 공유 — 별도 전처리기 불필요.
  2. 기존 Ensemble Selector 의 cosine 경로 재사용.
  3. BCE / CE 는 현재 bottleneck 분석 대상이라 기준축으로 부적절.
- **Phase 2 (성능 양호 시)**: CE (classifier logit), Cosine (non-ensemble) 추가.

#### 설계 요소
- **k 스윕**: Phase 1 에서 `k ∈ {3, 5, 10, 20}` 4 값.
- **전처리기**: Selector 진입 시 `cos(Q, schema_node)` 으로 node 별 raw_score 계산 → top-k mask 생성.
- **Edge 구성**: `query → node` edge_index 를 top-k 인덱스만 포함하도록 제한.
- **Directed 조합**: Phase 1 전부 `supernode_directed=True` 전제 (의견 3 과 연결).

#### 인터페이스
```python
class SchemaHeteroGATv2(...):
    def __init__(self, ...,
                 supernode_top_k: Optional[int] = None,
                 supernode_topk_metric: str = "raw"):  # "raw" | "cosine" | "ce"
        self.supernode_top_k = supernode_top_k
        self.supernode_topk_metric = supernode_topk_metric

    def _select_topk_nodes(self, query_emb: Tensor,
                           node_emb_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """반환: {"table": idx_tensor, "column": idx_tensor} — 각 node_type 내 top-k global indices."""

    def _build_supernode_edges_topk(self, topk_dict: Dict[str, Tensor]) -> EdgeIndexDict:
        """top-k subset 으로 query→node edge_index 제한, directed 조합 시 역방향 생략."""
```

- Builder 측: 기존 SuperNode builder 에 `top_k` / `topk_metric` 파라미터 추가. 전처리는 encoder 호출 직후 수행 (학습마다 재계산, forward overhead 수 ms).

#### 예상 실험 ID — Phase 1 (Raw Score × k 스윕, 권장)

| ID | k | metric | directed | 비고 |
|----|---|--------|----------|-----|
| `abl_sel_sn_topk_raw_k3_01` | 3 | raw | ✅ | over-smoothing 최소, gold 누락 리스크 ↑ |
| `abl_sel_sn_topk_raw_k5_01` | 5 | raw | ✅ | |
| `abl_sel_sn_topk_raw_k10_01` | 10 | raw | ✅ | 중간 점 — int_05 전제 default 후보 |
| `abl_sel_sn_topk_raw_k20_01` | 20 | raw | ✅ | conservative |

#### Phase 2 (Phase 1 최적 k 확정 후, optional)

| ID | k | metric | 비고 |
|----|---|--------|-----|
| `abl_sel_sn_topk_cos_kX_01` | best | cosine (non-ensemble) | |
| `abl_sel_sn_topk_ce_kX_01` | best | ce (classifier logit) | 학습된 head 필요 |

#### 검증
- **Gold coverage within top-k**: k 별로 dev set gold schema node 가 top-k mask 에 포함되는 비율. k=3 에서 급락 예상 — 이 지점이 "희소화의 상한".
- **Recall 회복 curve**: k ∈ {3,5,10,20} 의 Selector-stage / pipeline-final recall 비교.
- **Over-smoothing 재등장 (§7.4 리스크)**: directed + 작은 k 조합에서 distant node 가 SN 신호 단절. intra-table cosine trajectory (s06 bottleneck v2 스크립트 재활용) 관찰.

---

### 조합 — "SuperNode v2" (의견 3 + 4, int_05 전제)

- **default**: `supernode_directed=True` + `supernode_top_k=10` + `supernode_topk_metric="raw"` (Phase 1 스윕 후 k 확정).
- **루트 PLAN int_05 전제 재정의 (§7.1 approved)**: `int_05_direct_ns` = Enriched + B-III + **SuperNode v2** + DirectGAT head + S-V (Neurosymbolic L1) + E-III + FL-III + XiYan.
- S-V (§S-V) 와의 연계:
  - SuperNode v2 의 top-k selective edge 는 **확률적 soft routing 의 구조적 변형** — S-V 의 `boosted = GAT + λ·reach_mask` 와 철학적으로 동형.
  - SuperNode v2 체크포인트가 확보되면 S-V Enriched variant 와 결합 가능: `abl_sel_ns_l1_sn_v2_*`.

### 학습 비용 / 일정

| Track | 실험 수 | 학습 필요 | 예상 시간/실험 | 총 GPU 시간 |
|-------|---------|----------|---------------|------------|
| A | 1 (신규) | 없음 (Selector 교체만) | 추론만, ~1h | 1h |
| D | 2 | ✅ from scratch | ~9h (B5 batched 기준) | ~18h |
| E Phase 1 | 4 | ✅ from scratch | ~9h (k 변경만) | ~36h |
| 합계 | **7** | | | **~55h** |

- 직렬 실행 시 ~2.3 일. 2026-04-28 발표 시점에 A 는 확정, D/E 는 Phase 1 중 최소 2 cell 확보 목표.
- A/F/C 는 2026-04-28 발표 core, D/E 는 여유 시.

### 리스크 / 주의

- **SuperNode v2 over-smoothing 재등장** (§7.4): directed + 작은 k 조합에서 distant node 의 SN 신호 단절. Diameter 기반 num_layers 튜닝 (루트 Proposal C) 과 병행 시 완화 가능.
- **재학습 필요**: 기존 T7/T9 체크포인트 모두 양방향 + 전체 연결 전제 → D/E 실험은 from scratch (9h/실험). batched dual_stream 이 s06_a01_07 에서 3.1× 가속 (9h 14m) 실증 — 동일 전략 적용.
- **k 하한선**: gold coverage within top-k 가 90% 미만으로 떨어지면 Selector-stage recall 이 구조적으로 상한 설정됨. k=3 은 진단용, 실전 default 은 k=10 전후 예상.

### 변경될 파일 (본 track 한정)

| 파일 | 변경 |
|------|------|
| `src/models/gat_network.py` / `gat_network_v2.py` | `supernode_directed` / `supernode_top_k` / `supernode_topk_metric` 파라미터 추가, edge 구성 로직 분기 |
| `src/modules/builders/graph_builder.py` (SuperNode builder 분기) | top-k mask 기반 edge_index 생성 옵션 |
| `configs/experiments/abl/sel/rawscore/abl_sel_raw_ens_01.yaml` | 신규 (A track) |
| `configs/experiments/abl/sel/sn_v2/abl_sel_sn_directed_{bce,proj}.yaml` | 신규 (D track, 2 개) |
| `configs/experiments/abl/sel/sn_v2/abl_sel_sn_topk_raw_k{3,5,10,20}_01.yaml` | 신규 (E track Phase 1, 4 개) |
| `src/train_gat.py` or `train_gat_s06.py` | SuperNode v2 학습 entrypoint 확장 (기존 config 파서 재활용) |

---

## 통합 실험 로드맵 (Selector 관점)

| Phase | 실험 | 의존 | 비고 |
|-------|------|------|-----|
| **QC-A** | `abl_sel_raw_ens_01` + 기존 s04_04/05 재집계 | 없음 (Selector 단독) | **2026-04-21 지도교수 의견 1 대응, core** |
| **QC-D** | `abl_sel_sn_directed_{bce,proj}` | 재학습 | **의견 3, directed SN** |
| **QC-E** | `abl_sel_sn_topk_raw_k{3,5,10,20}_01` | 재학습 + Q2 Raw 선별 | **의견 4, Phase 1 raw 만** |
| S1 | `abl_s06_ns1_*` | Builder B-III | 가장 저비용, 먼저 실행 |
| S2 | `abl_s06_rfm_01` (BGE-M3) | Builder B-I | Encoder 교체 단독 효과 |
| S3 | `abl_s06_ehgat_*` | Builder B-II | Line graph 학습 재생성 필요 |
| S4 | `abl_s06_xllm_*` | Filter와 조율 | LLM 공유 자원 |
| S5 | `abl_s06_rl_*` | Warm-start ckpt | 가장 복잡, 마지막 |

**발표 우선순위 (2026-04-28 15~20분)**: **QC-A** (core) → F (SteinerBackbone, 루트) → C (Diameter, 루트) → **QC-D** → **QC-E** → B (T2T, Builder). QC-D/QC-E 는 시간 여유 시만.

## 변경될 파일

| 파일 | 변경 |
|------|------|
| [ensemble_selector.py](ensemble_selector.py) | λ·reach mask hook |
| [neurosymbolic_l1_selector.py](neurosymbolic_l1_selector.py) | 신규 — S-V |
| [rfm_selector.py](rfm_selector.py) | 신규 — S-II |
| [ehgat_selector.py](ehgat_selector.py) | 신규 — S-III |
| [extractive_llm_selector.py](extractive_llm_selector.py) | 신규 — S-IV |
| [rl_selector.py](rl_selector.py) | 신규 — S-I |
| `src/train_gat.py` → `src/train_selector.py` | 통합 training entrypoint |

## 인터페이스 계약 (유지)
- `select(scores, candidates, question, graph_data, metadata, **kwargs)` → `List[int]` (seeds)
- `self.latest_scores` 에 전체 노드 score 저장 — 모든 신규 selector 준수.

## 검증 방법 (모듈 내)
- **Score 품질**: ROC-AUC/PR-AUC vs baseline (0.776 / 0.317).
- **Gold recovery**: GAT rescued/hurt 분석 (현재 +2.1% 기준).
- **End-to-end**: AdaptivePCST + XiYan 고정, Selector만 교체로 F1 비교.
