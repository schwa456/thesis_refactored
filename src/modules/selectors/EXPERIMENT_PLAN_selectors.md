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

## QCondGAT v2 계열 — 구현 스펙 (2026-04-21 의견 2/3/4 + Q1/Q2 수렴)

> **출처 (proposals)**:
> - [planning/proposals/abl_sel_diameter_layers.md](/home/hyeonjin/thesis_refactored/planning/proposals/abl_sel_diameter_layers.md) §4 (의견 2, Q1 답변 = per-DB D_max)
> - [planning/proposals/abl_sel_supernode_directed.md](/home/hyeonjin/thesis_refactored/planning/proposals/abl_sel_supernode_directed.md) §4 (의견 3)
> - [planning/proposals/abl_sel_supernode_topk.md](/home/hyeonjin/thesis_refactored/planning/proposals/abl_sel_supernode_topk.md) §4 (의견 4, Q2 답변 = Raw Score 우선)
> - [planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md](/home/hyeonjin/thesis_refactored/planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md) §8 Proposal C/D/E
>
> **앞 섹션과의 관계**: "QCondGAT 계열 Ablation Track" (§A/D/E) 은 **일정·비용·매트릭스** 중심. 본 섹션은 **코드·Config·인터페이스 정본**. Config key 명명을 proposal 기준으로 canonical 화:
> - `supernode_directed` (bool, 앞 §D) → **`supernode_edge_direction ∈ {bidirectional, directed_from_sn}`** (enum) 으로 대체
> - `supernode_top_k` (앞 §E) → **`supernode_topk`**
> - `supernode_topk_metric` (앞 §E) → **`supernode_topk_criterion`**
> - V-1 (per-DB dynamic num_layers) 는 앞 섹션에 없던 **신규 축**.
>
> **공통 학습 전제**:
> 1. 세 항목 모두 edge topology 또는 depth 변경 → T5/T7/T9 체크포인트 재사용 불가. GAT **재학습 필수** (Enriched builder 기반 v2).
> 2. **체크포인트 명명**: `best_gat_enriched_<variant>.pt` 패턴. 예: `best_gat_enriched_sn_directed.pt`, `best_gat_enriched_sn_topk_raw_k10.pt`, `best_gat_enriched_nl_dmax.pt`.
> 3. **저장 위치 (루트 CLAUDE.md NAS 규칙)**: 실제 파일은 `/SSL_NAS/peoples/khj/thesis/checkpoints/`, 로컬 `outputs/checkpoints/<name>.pt` 는 symlink. 학습 중인 체크포인트는 로컬에 두다 종료 직후 NAS 이관 + symlink 교체.

---

### V-1. Per-DB Dynamic `num_layers` (의견 2, Q1: D_max)

#### 동기
- 현행 고정 `num_layers=3` 은 작은 DB (`card_games`, V≈20) 에서 over-smoothing 과잉, 큰 DB (`european_football_2`, `formula_1`) 에서 under-reaching.
- Proposal C: per-DB schema heterograph 의 **최대 shortest-path (D_max)** 에 맞춰 layer 수 설정.

#### Config 스펙
```yaml
selector:
  num_layers_mode: D_max            # {fixed, D_max, D_max_plus1}
  num_layers_fallback: 3            # D_max 조회 실패 시 int
  diameter_path: data/processed/train_diameter.pt   # per-DB D_max dict
```
- `fixed` 모드: 기존 `model.num_layers` 그대로 사용 (back-compat).
- `D_max`: `diameter_dict[db_name]` 로 조회.
- `D_max_plus1`: `D_max + 1`.
- 조회 실패 (dict 키 부재 or dict 파일 없음) 시 `num_layers_fallback` 으로 fallback + warning 로그.

#### 구현
- **선결 (Builder 세션 Phase A, proposal §4.1)**: `data/processed/<split>_diameter.pt` 사전 생성. Per-DB schema-only subgraph → all-pairs shortest path (BFS, V≤80 → μs 단위) → max. B-III FK reachability 계산과 1-패스 공유.
- **Loader / Trainer 로직** (`src/data/bird_dataset.py` 또는 graph loader):
  ```python
  diameter_dict = torch.load(cfg.selector.diameter_path) if cfg.selector.num_layers_mode != "fixed" else None

  def resolve_num_layers(db_name: str) -> int:
      if cfg.selector.num_layers_mode == "fixed":
          return cfg.model.num_layers
      d = diameter_dict.get(db_name)
      if d is None:
          logger.warning(f"diameter missing for {db_name}; fallback {cfg.selector.num_layers_fallback}")
          return cfg.selector.num_layers_fallback
      return d if cfg.selector.num_layers_mode == "D_max" else d + 1
  ```
- **배치 이질성 처리**: batch 내 sample 간 `num_layers` 가 다르면 (a) DB-grouped batch sampler 로 동질 batch 만 만들거나 (b) 모델을 `max(num_layers_in_batch)` 로 빌드하고 layer 별 halt mask. **초기 구현은 (a)** — 단순, 디버그 용이.
- **모델 변경**: `src/models/gat_network_v2.py` 의 `num_layers` 는 이미 파라미터화됨. 래퍼에서 per-sample resolve 해서 전달만 추가.

#### 예상 실험 ID
| ID | mode | fallback | 비고 |
|----|------|----------|-----|
| `abl_sel_diameter_layers_nl_fixed3` | fixed | — | 기존 baseline 재라벨 |
| `abl_sel_diameter_layers_nl_dmax` | D_max | 3 | peak 후보 |
| `abl_sel_diameter_layers_nl_dmax_plus1` | D_max+1 | 3 | over-depth 리스크 관찰 |

#### 체크포인트 (NAS + symlink)
- `best_gat_enriched_nl_fixed3.pt` (T5 재라벨 가능 — 동일 값)
- `best_gat_enriched_nl_dmax.pt` (신규)
- `best_gat_enriched_nl_dmax_plus1.pt` (신규)

#### 검증
- F1 peak (Δ > 0.5% vs fixed=3) 존재 여부 → H1.
- D_max 가 큰 DB (예: `european_football_2`) 에서 `D_max` cell 이 오히려 하락하면 H2 (depth 상한 실재) 증거.

#### H2 subtask — inference-only per-DB dynamic (2026-04-22 추가)

> **Status: closed (2026-04-26).** Selector impl truncate forward 2 cell 실측 결과 partial neutral + nl=7 truncate training mismatch 확인. 2026-04-25 H2 원래 가설 (naive resolve(db)=D_max) 기각 결정 유지, truncate mechanism 도 anchor 갱신 임계 미달. 본 selector 세션이 구축한 인프라 (EnsembleSelector v2 분기 + db_name threading + truncate forward) 는 **H3 (schema feature → optimal depth predictor, future work)** 의 inference 단에 그대로 재활용 가능. 결과 표·H3 가이드는 본 H2 블록 마지막 §"2026-04-26 closure" 참조.

**동기**: Wave 2 Phase 1 에서 `L ∈ {1, 2, 3, 6, 7}` 전역 고정 깊이 5-cell sweep 을 재학습. 동일 체크포인트를 재사용하여 재학습 없이 H2 ("극단 D_max DB 에서 per-DB 맞춤 깊이가 전역 고정보다 높은가") 를 측정하기 위한 **inference-time early-exit** 경로를 구현.

**구현 (2026-04-22)**:
- `src/models/gat_network.py` `SchemaHeteroGAT.forward(..., active_num_layers: Optional[int] = None)` 파라미터 추가.
  - `None` 이면 기존처럼 `self.num_layers` 전부 사용 (back-compat).
  - 정수가 주어지면 `clamp(1, self.num_layers)` 후 `self.convs[:L']` 까지만 메시지 패싱 수행. ModuleList slicing 방식 — 체크포인트 key 변경 없음.
- `src/modules/selectors/ensemble_selector.py`:
  - `num_layers` (int) 파라미터를 GAT 생성자에 전달 (기존 누락 버그 동시 수정 — 이전엔 config 와 무관하게 `num_layers=3` 디폴트 사용).
  - `num_layers_mode ∈ {fixed, per_db_dynamic, D_max, D_max_plus1}`, `diameter_cache_path`, `num_layers_fallback` 신규 인자.
  - `_resolve_active_depth(metadata)` helper: `metadata['db_id']` → `diameter_dict` → mode 에 따라 `active_num_layers` 산출. 누락/unknown DB 는 fallback (+경고 로그).
  - `_compute_gat_scores` 내 세 갈래 (`query_supernode` / `query_conditioned` / 기본) 모두 `active_num_layers=depth` 를 forward 로 전달.
- `src/pipeline/schema_linking.py`: `self.builder.build(...)` 직후 `metadata.setdefault("db_id", db_id)` 로 DB id 를 metadata 에 주입. Extractor/Filter 도 동일 키로 접근 가능.
- 모드 값 중 `per_db_dynamic` 는 `D_max` 의 별칭 (장래 mix policy 확장 여지).

**체크포인트 재사용 가능성 (dev_diameter.pt 분포)**:
- 11 DB, D_max ∈ [3, 6], median=5, max=6, p95=6. D_max=6 DB: `european_football_2`, `formula_1`, `student_club`, `superhero`. D_min=3: `debit_card_specializing`.
- `D_max` 모드 최대 필요 깊이 = 6 → **Wave 2 Phase 1 L=6 ckpt 그대로 재사용 가능** (재학습 불필요).
- `D_max_plus1` 모드 최대 필요 깊이 = 7 → **Wave 2 Phase 1 L=7 ckpt 그대로 재사용 가능**.
- 따라서 H2 측정은 추가 학습 0회. inference config 2개만 새로 정의.

**H2 전용 실험 ID (재학습 없음, Wave 2 Phase 1 ckpt 재사용)**:
| ID | 체크포인트 | num_layers_mode | 비교 대상 |
|----|-----------|-----------------|----------|
| `abl_sel_diameter_layers_nl_dmax_infer_L6ckpt` | `best_gat_enriched_nl_fixed6.pt` (Wave 2 Phase 1 L=6) | `D_max` (fallback=6) | 동일 ckpt + mode=fixed |
| `abl_sel_diameter_layers_nl_dmax_plus1_infer_L7ckpt` | `best_gat_enriched_nl_fixed7.pt` (Wave 2 Phase 1 L=7) | `D_max_plus1` (fallback=7) | 동일 ckpt + mode=fixed |

두 쌍(mode=fixed vs 해당 dynamic) per-DB R/P/F1 비교로 **D_max=6 DB 에서 dynamic cell 이 fixed 대비 peak shift 를 보이는지** 직접 검증. Wave 2 Phase 1 학습 완료 후 Phase 2 inference 배치에 병합.

**Smoke test**: `scripts/smoke_test_per_db_dynamic.py` — (1) diameter 분포 로그, (2) forward early-exit/clamp 동작, (3) `_resolve_active_depth` 정책표 (체크포인트 로드 없이 stub 으로). **Pass 확인 2026-04-22**.

**Wave 2.5 mini-sweep 과의 관계**: 본 subtask 는 inference 경로 (ckpt 재사용). Phase 2 결과에 따라 필요 시 per-DB dynamic 전용 재학습(batch sampler + per-sample depth resolve, 본 V-1 §구현) 으로 승격. 현 시점엔 그 비용을 치르지 않고 H2 가설만 먼저 측정.

**2026-04-24 추가 (Phase 전환 §결정 (c) 에스컬레이션 수행)**:
- `EnsembleSelector` 에 `gat_version: str = "v1"` 스위치 + `gat_v2_kwargs: dict | None` 주입 경로 추가. `v2` 선택 시 `SchemaHeteroGATv2` 를 instantiate 하며 기본값은 `num_layers_mode="fixed"` 로 두어 selector 쪽 `_resolve_active_depth` 가 단일 depth 소스로 유지됨 (model 내부 lookup 비활성).
- **체크포인트 호환 확인**: v1 과 v2 의 state_dict 는 default option (`pairnorm='none'`, `jumping_knowledge='none'`, `dual_stream=False`, `initial_residual_alpha=0.0`) 하에서 **bit-wise 동일 key 198개**. 즉 `best_gat_qcond_nl{1,2,3,6,7}.pt` 5개 ckpt 를 v2 branch 에 그대로 로드 가능 → **신규 학습 0**.
- `train_gat_s06.py` 에 `num_layers_mode`, `num_layers_fallback`, `diameter_path`, `diameter_dict` 4개 flag forward. 학습 기본은 `fixed` (global num_layers 로 훈련), 재학습이 필요하면 per-DB curriculum 도 가능하도록 hook 만 열어둠.
- 새 smoke test: `src/modules/selectors/tests/test_h2_per_db_dynamic.py` — v1/v2 양쪽 branch 에서 실제 `best_gat_qcond_nl6.pt` 로드 후 `D_max` 모드로 3개 DB (D_max=3/5/6) 를 통과시켜 `last_resolved_depth` 가 **서로 다른 int** 로 분리되는지 + forward pass 가 에러 없이 완료되는지 검증. Fixed/Unknown-DB fallback branch 도 동시 커버. Pass 확인 2026-04-25.
- H2 inference config 신규 2개 (root 세션 실행 대기):
  - [layers_Ldbmax_glm.yaml](/home/hyeonjin/thesis_refactored/configs/experiments/s04_ablation/diameter_layers/layers_Ldbmax_glm.yaml) — `D_max` + nl=6 ckpt (primary H2 cell)
  - [layers_Ldbmax_plus1_glm.yaml](/home/hyeonjin/thesis_refactored/configs/experiments/s04_ablation/diameter_layers/layers_Ldbmax_plus1_glm.yaml) — `D_max_plus1` + nl=7 ckpt (over-capacity-per-DB 회복 probe)

**2026-04-26 closure — H2 작업 종료 + H3 future work 인계**:

근거: planning/DECISIONS.md 2026-04-26 (후속) 엔트리 §결정 (a)~(c).

실측 결과 (root 세션 2026-04-25 01:36~02:33, scripts/run_h2_truncate.sh):

| Cell | R | P | F1 | ΔF1 vs L6_glm | ΔF1 vs analyzer recon | 분기 판정 |
|------|---|---|---|---|---|------|
| L6_glm (anchor, global fixed nl=6) | 0.5018 | 0.6939 | 0.5824 | — | +0.0019 | — |
| analyzer recon (sweep 5-cell 재조합) | — | — | 0.5805 | -0.0019 | — | (보고용) |
| **Ldbmax_glm** (D_max truncate, nl=6 ckpt) | 0.5036 | 0.7031 | **0.5869** | **+0.0045** | **+0.0064** | partial neutral (anchor 갱신 임계 미달) |
| **Ldbmax_plus1_glm** (D_max+1 truncate, nl=7 ckpt) | 0.4778 | 0.6776 | **0.5604** | **-0.0220** | -0.0201 | 기각 확고 (training mismatch) |

해석:
- Ldbmax_glm: D_max=3/4/5 DB (944 q, 61.5%) 에서 selector impl truncate 가 analyzer recon (ckpt 부재 fallback) 대비 +0.0064 partial positive — H2 spirit 의 mechanism 자체는 fallback 보다 약간 낫지만, **anchor 갱신 임계 +0.005 미달** → 실용적 개선 한계.
- Ldbmax_plus1_glm: 동일 truncate mechanism 인데 ckpt 만 nl=7 로 바꿨더니 sign 반전 (-0.0220). nl=7 ckpt 자체 over-smoothing 영향 (sweep 에서 nl=6 대비 ΔF1=-0.0062) 을 빼도 truncate mismatch 순효과 ~-0.0158 추정. **Over-smoothing 영향권 ckpt 의 truncate 는 추가 손실** — H3 ckpt 선정 가이드 근거.
- 두 cell 모두 2026-04-25 H2 기각 결정 유지 (변경 없음).

**H3 future work — schema feature → optimal depth predictor**:
- 동기: H1 (global fixed) sweep peak = nl=D_max global=6, H2 (per-DB D_max) 는 partial neutral. 학술적 다음 step 은 학습된 predictor 가 schema feature (V/E/D_max/연결 패턴) 로부터 optimal depth 를 예측 → 단순 D_max 휴리스틱 대비 추가 이득 가능.
- **본 H2 인프라 재활용 경로**:
  - `EnsembleSelector` 의 `gat_version`/`gat_v2_kwargs` 분기 + `_resolve_active_depth(metadata)` hook → predictor 출력값을 `metadata['active_num_layers']` 또는 `metadata['db_id']` → 학습된 lookup 으로 주입하는 단계만 추가.
  - `metadata['db_id']` threading (`pipeline/schema_linking.py:82`) 은 이미 `db_id` 이상 임의 schema feature 로 확장 가능 (e.g., `metadata['schema_features']`).
  - Truncate forward mechanism (v1·v2 의 `active_num_layers` 인자) 은 이미 검증됨 (smoke test pass 2026-04-25).
  - 즉 H3 의 inference 단은 **추가 인프라 0**, predictor 학습 (별도 head 또는 외부 MLP) 만 신규 작업.
- **H3 ckpt 선정 가이드 (2026-04-26 nl=7 truncate 결과 근거)**:
  - **Over-smoothing 영향권 ckpt (nl > D_max global = 6) 회피**. nl=7 ckpt 의 truncate mismatch 순효과 ~-0.0158 (over-smoothing 효과 분리 후) 가 직접 증거.
  - 권장: nl=D_max global (BIRD dev: nl=6) 학습 ckpt 를 backbone 으로 두고, predictor 출력으로 `active_num_layers ∈ [1, D_max global]` 만 truncate.
  - 학습 시 per-DB depth curriculum 은 본 selector 세션이 `train_gat_s06.py` 에 forward 한 `num_layers_mode` flag 로 선택지 제공됨 (현 시점엔 미사용).
- 우선순위: H3 는 발표 후 작업 (post-2026-04-28). 본 PLAN §V-1 의 retraining 경로 (batch sampler 등) 와 통합하여 별도 mini-wave (Wave 2.5 또는 Wave 5) 로 승격 시 planner 에스컬레이션.

---

### V-2. SuperNode Edge Direction Flag (의견 3)

#### 동기
- 현행 bidirectional SN ↔ schema: GAT 메시지 전파로 SN embedding 이 schema 정보에 희석 (dilution).
- Proposal D: 단방향 (SN → schema only) → SN 을 query 신호 anchor 로 보존, schema node 만 수신.

#### Config 스펙
```yaml
selector:
  supernode_edge_direction: directed_from_sn   # {bidirectional, directed_from_sn}
  # directed_from_sn 의미:
  #   - ('query', '*', 'table'|'column') edge 유지
  #   - ('table'|'column', '*', 'query') edge 제거 (schema→SN 차단)
  #   - ('query', 'self', 'query') self-loop 유지 (SN query feature 보존)
```

#### 구현
- **파일**: `src/models/gat_network.py` (+ `gat_network_v2.py` v2 SN 지원 시).
- **edge_index_dict 빌드 분기**:
  ```python
  class SchemaHeteroGATv2(...):
      def __init__(self, ..., supernode_edge_direction: str = "bidirectional"):
          assert supernode_edge_direction in {"bidirectional", "directed_from_sn"}
          self.supernode_edge_direction = supernode_edge_direction

      def _build_supernode_edges(self) -> List[Tuple[str, str, str]]:
          fwd       = [('query', 'to_t', 'table'), ('query', 'to_c', 'column')]
          rev       = [('table', 't_to', 'query'), ('column', 'c_to', 'query')]
          self_loop = [('query', 'self', 'query')]
          if self.supernode_edge_direction == "directed_from_sn":
              return fwd + self_loop                      # schema→SN 제거
          return fwd + rev + self_loop                    # bidirectional (기존)
  ```
- **Builder 측**: 기존 SuperNode builder 의 edge 구성에서 역방향을 flag 로 생략 (`supernode_edge_direction` forward propagation).
- **학습 재필요**: T7 (`best_gat_query_supernode.pt`) · T9 (`best_gat_query_supernode_direct.pt`) 모두 양방향 전제 → directed 계열 from scratch.

#### 예상 실험 ID
| ID | variant | checkpoint |
|----|---------|-----------|
| `abl_sel_supernode_bidir_a0` (재라벨) | bidirectional | T7 (재사용) |
| `abl_sel_supernode_directed_proj` | directed + Projector (BCE+InfoNCE) | `best_gat_enriched_sn_directed_proj.pt` |
| `abl_sel_supernode_directed_bce`  | directed + Direct (BCE only)       | `best_gat_enriched_sn_directed_bce.pt`  |

#### 검증
- **SN embedding drift**: 학습 epoch 별 SN 노드 embedding L2 norm 추이 (bidir 감소 vs directed 유지 예상).
- **Distant-node signal**: `node_distance_to_SN × recall` 분포 — directed 에서 distant node 신호 단절 여부 (§7.4 리스크, **V-1 D_max 와 교호** — Proposal D §2 H2).
- **Val R@15 vs T7/T9**: directed 가 상승 시 over-smoothing 완화 근거.

---

### V-3. SuperNode Top-k Selective Connection (의견 4, Q2: Raw Score)

#### 동기
- SN 을 모든 schema node 에 연결 → attention 분산. **pre-GAT raw score 상위 k 개 에만** SN edge → attention 집중 + noisy node gradient 희석 감소.
- Phase 1 `criterion=raw` 만 — Ensemble selector 의 cosine 경로 재사용, BCE/CE 는 현재 bottleneck 분석 대상이라 기준축 부적절.

#### Config 스펙
```yaml
selector:
  supernode_topk: 10                          # {null, 3, 5, 10, 20}; null = 기존 all-node
  supernode_topk_criterion: raw               # {raw, ce, cosine}; Phase 1 = raw
  supernode_edge_direction: directed_from_sn  # 권장 default — V-2 와 조합
```

#### 구현
- **Pre-GAT raw score 추출**:
  - `EnsembleSelector` 의 기존 cosine 경로 재사용 → `get_raw_scores(query_emb, node_emb)` utility 분리/노출.
  - `raw_score(v) = cos(e_Q, e_v)` (encoder output, GAT 미통과).
- **Top-k 인덱스 추출**: `torch.topk(raw_scores, k=supernode_topk, dim=-1)` — per node_type 별 (table/column 따로) 또는 global mix — 초기 구현은 **node_type 내부 top-k 후 합집합** (type 별 최소 대표권 보장).
- **Edge 구성** (`src/models/gat_network.py`):
  ```python
  class SchemaHeteroGATv2(...):
      def __init__(self, ...,
                   supernode_topk: Optional[int] = None,
                   supernode_topk_criterion: str = "raw"):
          self.supernode_topk = supernode_topk
          assert supernode_topk_criterion in {"raw", "ce", "cosine"}
          self.supernode_topk_criterion = supernode_topk_criterion

      def _compute_topk_nodes(self, query_emb, node_emb_dict):
          """criterion 분기:
          - raw   : cos(query_emb, node_emb) via EnsembleSelector.get_raw_scores
          - cosine: 동일 수식이나 Ensemble α blending 경로 제외 (non-Ensemble)
          - ce    : pre-GAT classifier head logit (학습된 head 필요 — Phase 2)
          반환: {"table": idx_tensor, "column": idx_tensor} (node_type 내 global indices)
          """

      def _build_supernode_edges_topk(self, topk_dict):
          # query→{table,column} edge_index 를 top-k idx subset 으로만 구성.
          # supernode_edge_direction=directed_from_sn 조합 시 역방향 edge 는 이미 제외.
  ```
- **Phase 1 default combo**: `supernode_edge_direction=directed_from_sn` + `supernode_topk ∈ {3,5,10,20}` + `supernode_topk_criterion=raw`.
- **Phase 2 트리거 (조건부)**: Phase 1 peak F1 > D baseline +0.5% → `criterion ∈ {ce, cosine}` × best k 확장.

#### 예상 실험 ID — Phase 1 (Raw × k)
| ID | k | criterion | direction | checkpoint |
|----|---|-----------|-----------|-----------|
| `abl_sel_supernode_topk_raw_k3`  | 3  | raw | directed_from_sn | `best_gat_enriched_sn_topk_raw_k3.pt`  |
| `abl_sel_supernode_topk_raw_k5`  | 5  | raw | directed_from_sn | `best_gat_enriched_sn_topk_raw_k5.pt`  |
| `abl_sel_supernode_topk_raw_k10` | 10 | raw | directed_from_sn | `best_gat_enriched_sn_topk_raw_k10.pt` |
| `abl_sel_supernode_topk_raw_k20` | 20 | raw | directed_from_sn | `best_gat_enriched_sn_topk_raw_k20.pt` |

#### Phase 2 (조건부, 별도 승인 필요)
| ID | k | criterion | 비고 |
|----|---|-----------|-----|
| `abl_sel_supernode_topk_cos_kbest` | best (P1) | cosine (non-Ensemble) | α blending 제외 |
| `abl_sel_supernode_topk_ce_kbest`  | best (P1) | ce | 학습된 classifier head 필요 |

#### 검증
- **Gold coverage within top-k**: dev gold schema node 가 top-k mask 에 포함되는 비율 — k=3 에서 급락 예상 (희소화 상한 진단).
- **Recall curve over k**: k ∈ {3,5,10,20} 의 Selector-stage / pipeline-final recall.
- **Over-smoothing 재등장**: directed + 작은 k 조합에서 distant node SN 신호 단절 — intra-table cosine trajectory (s06 bottleneck v2 재활용).

---

### V-3-ext. Directed Top-K SuperNode (학위 논문 Part III, 2026-05-05) — 단계 1 구현 완료

> **Status: 단계 1 구현 완료 (2026-05-09 ~ 2026-05-11 가속, 5/5 완료).** advisor (지도교수) 제안 — "Graph 를 Directed Edge + Raw Score top-K SuperNode + GAT 학습" — 학위 논문 Part III. V-3 (top-K=raw, 단일값) 의 **threshold 일반화 + 학습 변형 3 종 + selector inference path** 를 추가.
>
> 근거: planning/DECISIONS.md 2026-05-05 (Q1/Q2/Q3 confirm + threshold P80 primary 채택), notebooks/analysis_results/raw_score_distribution_for_directed_topk.md.

#### 동기 — V-3 와의 차이
- V-3: `supernode_topk: int` 단일값 (top-K=20 default).
- V-3-ext: **threshold mode 3 종**으로 일반화 — `top_k` (V-3 기존), `percentile` (per-query Pn), `abs_tau` (per-query min-max norm 후 절대 cutoff). DB 별 schema 크기 variability (european_football 237 vs toxicology 20) 자동 보정 위한 query-aware threshold (P80 primary).

#### Config 스펙 (학습)
```yaml
model:
  query_supernode: true
  supernode_edge_direction: directed_from_sn
  # V-3-ext threshold mode dispatch
  supernode_threshold_mode: percentile   # {top_k, percentile, abs_tau}
  supernode_threshold_value: 80.0        # P80 cutoff (or top-K count, or abs cutoff)
  supernode_score_normalization: minmax  # {minmax, none}; analyzer 정의와 동일
  # backward-compat (top_k mode 시): supernode_topk 그대로 사용 가능
```

#### 인터페이스 / 구현 (selector + GAT)
- **Selector (inference)**: `src/modules/selectors/directed_topk_supernode_selector.py` 신규.
  - `DirectedTopKSuperNodeSelector(EnsembleSelector)` 상속. SuperNode 분기 override:
    - query_node x 주입 + threshold mask 산출 (per-query min-max norm cosine 기반)
    - `attends_to_*` directed edge 만 (필터링된 schema 만). `attended_by_*` 비등록/0-len.
    - GAT 모델 측 자동 self_loop (directed_from_sn) 사용.
  - 등록: `__init__.py` (selector registry).
- **GAT model 측**: `gat_network.py` / `gat_network_v2.py` 양쪽에 `_compute_supernode_mask` dispatch:
  - `top_k` (기존 V-3 동치, supernode_topk 사용)
  - `percentile` (torch.quantile cutoff)
  - `abs_tau` (>= cutoff)
  - `_compute_topk_mask` 는 backward-compat alias.
- **train_gat.py**: `supernode_threshold_mode` / `supernode_threshold_value` / `supernode_score_normalization` 옵션 forward.
- **Smoke test**: `tests/test_directed_topk_supernode.py` 7 케이스 통과 (P80 ~22% 선택, top_k=20 정확, abs_tau=0.7 선택적, directed edge 구조 검증, baseline SuperNode 31 vs Directed 7 edge, v1/v2 dispatch).

#### 학습 변형 3 종 + checkpoint
| 변형 | mode | value | \|sel\| mean (raw) | Raw R | Raw F1 | config |
|---|---|---|---:|---:|---:|---|
| **#1 (PRIMARY)** | percentile | 80.0 | **18.9** ± 5.5 | 0.6133 | 0.3466 | `train_gat_directed_supernode_p80.yaml` |
| #2 (BASELINE) | top_k | 20.0 | 20.0 ± 0.0 | 0.6865 | 0.3640 | `train_gat_directed_supernode_topk20.yaml` |
| #3 (OPTIONAL) | abs_tau | 0.7 | 10.2 ± 8.9 | 0.4857 | **0.3942** ★ raw F1 max | `train_gat_directed_supernode_abstau07.yaml` |

신규 ckpt 명: `best_gat_directed_supernode_{p80, topk20, abstau07}.pt` (NAS + symlink 권장).

#### 시나리오 분기 (DECISIONS 2026-05-05 §1(d) 인용)
- **시나리오 A** (F1 ≤ 0.870, plateau 흡수): GAT 학습이 raw R 0.69 → 0.85 영역 회복 + Filter 가 plateau 안 흡수 → Filter Dominance 5 축 격상 (🆕 topology-invariant).
- **시나리오 B** (F1 > 0.870): 학위 논문 main contribution 5 항목 격상.
- **시나리오 C** (F1 < 0.85): paper §V.5.3 negative result + advisor 제안 mechanism deep dive.
- **확률 추정**: 시나리오 A 가장 가능성 高 (Filter 가 raw R 차이 흡수 mechanism 의 직전 narrative 와 일관) — 단 GAT 학습 결과 의존.

#### 변경된 파일 (단계 1 산출물)
| 파일 | 변경 |
|------|------|
| `src/models/gat_network.py` | `supernode_threshold_mode/value/score_normalization` 파라미터 + `_compute_supernode_mask` dispatch + alias |
| `src/models/gat_network_v2.py` | 동일 (v2 분기 호환) |
| `src/train_gat.py` | 신규 옵션 forward to GAT model |
| `src/modules/selectors/directed_topk_supernode_selector.py` | 신규 selector 클래스 (EnsembleSelector 상속, SuperNode 분기 override) |
| `src/modules/selectors/ensemble_selector.py` | `supernode_edge_direction` 옵션 노출 (GAT 모델 측 forward) |
| `src/modules/selectors/__init__.py` | `DirectedTopKSuperNodeSelector` 등록 |
| `src/modules/selectors/tests/test_directed_topk_supernode.py` | smoke test 7 케이스 |
| `configs/training/train_gat_directed_supernode_{p80, topk20, abstau07}.yaml` | 학습 config 3 종 |

#### 다음 단계 (root + planner 핸드오프)
- **단계 2 (5/12~5/13, root)**: 신규 GAT 학습 3 변형 × ~9h GPU. CUDA_VISIBLE_DEVICES=0,1 만 사용 (memory rule). 신규 ckpt NAS 저장 + symlink 자동화.
- **단계 3 (5/13~5/15, root)**: paper main stack (Enriched + 신규 ckpt + α=0.5 + MSTPCSTUnion + XiYan GLM + LLM SQL Gen GLM) 위 alpha sweep subset (α∈{0.0, 0.5, 1.0} 최소 + α∈{0.3, 0.7} 권장) 또는 full 11 cells. 비용 ~₩8-10K.
- **단계 4 (5/16~5/22, planner + 사용자)**: 시나리오 A/B/C 분기 처리, paper §3.5 Filter Dominance narrative 갱신 또는 §V.5.3 negative result.

#### 단계 4-bis. Attention 호환성 보강 — `extract_layerwise_attention_v2` (2026-05-06 구현 완료)

> **Status: 구현 + 6 smoke 통과 + Phase 1 4 ckpt 호환 (p80 / topk20 / abstau07 / qcond_nl3) — 통합 dsn_oversmoothing_analysis.py 에 wired.**
>
> 동기: V-3-ext 단계 4 진단에서 v1 `extract_layerwise_attention` 가 `directed_from_sn` self-loop + supernode threshold filter 를 manual conv 호출로 재현 못함 (matrix shape mismatch). attention entropy 미측정 한계 → forward hook 기반 v2 로 보강. 학위 논문 Part III mechanism deep dive evidence 의 base.
>
> 근거: planning/DECISIONS.md 2026-05-06 §1(B) (2)-A Attention 호환성 selector 위임 + notebooks/analysis_results/dsn_oversmoothing_analysis.md §7 Caveat.

**구현 파일**:
- [src/analysis/extract_layerwise_attention_v2.py](/home/hyeonjin/thesis_refactored/src/analysis/extract_layerwise_attention_v2.py) — `AttentionCapture` (monkey-patch wrap GATv2Conv.forward) + `extract_layerwise_attention_v2()` + `aggregate_attention_metrics()` + heatmap helpers.
- [src/modules/selectors/tests/test_attention_extract_v2.py](/home/hyeonjin/thesis_refactored/src/modules/selectors/tests/test_attention_extract_v2.py) — 6 케이스: capture/restore, directed_from_sn no-reverse, value sanity, aggregate, Phase 1 ckpt 호환 (4 ckpt × forward), qcond_nl3 (no SuperNode) 검증.

**메커니즘**:
- 각 GATv2Conv 의 forward 를 monkey-patch wrap → 호출 시 `return_attention_weights=True` 강제 + alpha tensor capture. HeteroConv 자체는 정상 forward → V-3-ext 의 `_compute_supernode_mask` / `_inject_sn_self_loop` / threshold edge filter 모두 그대로 적용.
- v2 가 v1 대비 우월한 점: (a) directed_from_sn 의 self-loop 자동 처리, (b) supernode threshold filter 후의 edge 만 capture (학습 시점과 동일 graph topology), (c) `__exit__` 시 instance attr 깔끔히 제거 (re-entrant safe).

**Metric 산출**:
- **Attention entropy** `H(α) = -Σ p_i log p_i` (per-edge-type, dst-node 별 평균)
  - 균일 분포 (entropy 高) vs 집중 분포 (entropy 低) 정량
- **Top-K=5 concentration** `(top-5 alpha sum) / (total alpha sum)` (dst-node 별 평균, [0,1])
- 둘 다 layer × edge_type 별 분리. `aggregate_attention_metrics` 으로 다중 query mean/std 환원.

**출력 파일** (max_queries=2 sanity 검증, 2026-05-06):
- `outputs/analysis/dsn_attention/<ckpt>/attention_metrics.json` — per-layer × per-edge-type entropy + top-5 concentration (mean/std over queries)
- `outputs/analysis/dsn_attention/<ckpt>/attention_entropy_layerwise.png` — heatmap
- `outputs/analysis/dsn_attention/<ckpt>/attention_topk5_concentration.png` — heatmap
- `outputs/analysis/dsn_attention/comparison_4ckpt.png` — 4 ckpt cross-model layer-wise 비교 (entropy + top-5 concentration 2 panel)

**초기 sanity 결과 (n=2 query)**:
- DSN 3 ckpt (p80/topk20/abstau07): L1~L3 ent ≈ 0.51~0.52, top-5 conc ≈ 0.91 — directed_from_sn + threshold 의 attention 이 매우 집중적 (top-5 가 ~91% 흡수)
- qcond_nl3 baseline: ent ≈ 0.83, top-5 conc ≈ 0.85 — 더 균일한 분포 (5 edge types only, no SuperNode)
- 의의: DSN 3 ckpt 의 over-smoothing 이 발생함에도 attention 자체는 집중적 → over-smoothing 의 root cause 가 **attention dispersion 이 아닌 다른 factor** (학위 논문 mechanism evidence)

**다음 (Phase 2 학습 후)**:
- root 신규 ckpt (DSN p80 + s06 B5 mitigation) 학습 완료 (5/12) 후 본 도구 재호출 → mitigation 변형의 attention 정량 (PN / IR / AC / Dual-Stream / L=2 의 attention 영향)
- analyzer 세션 (5/15~5/16) — full 50 queries 재측정 + 시나리오 분기 evidence

**변경된 파일 (단계 4-bis 산출물)**:
| 파일 | 변경 |
|------|------|
| `src/analysis/extract_layerwise_attention_v2.py` | 신규 — AttentionCapture + extract_layerwise_attention_v2 + aggregate + heatmap |
| `src/analysis/dsn_oversmoothing_analysis.py` | Step 3 attention extract 를 v1 → v2 교체 + JSON dump + per-ckpt heatmap + cross-model comparison |
| `src/modules/selectors/tests/test_attention_extract_v2.py` | 신규 smoke test 6 케이스 (Phase 1 4 ckpt 호환 검증) |

#### 단계 5. Phase 3 Skip Mitigation (Direct AC + Layer-wise LR, 2026-05-06 구현 완료)

> **Status: 구현 + 7 smoke 통과. 학위 본 심사 (5/22~6/19) 전 학습 진행 — DECISIONS 2026-05-06 §1(C)/(H)/(I).**
>
> 동기 (Phase 2 mitigation null mechanism finding): Skip Dependence Pathology DOMINANT — main GAT path gradient 1/10 축소 + 학습이 fusion_head + query_encoder path 로 우회. AC loss 가 model output (skip + fusion 후) 에 적용되어 skip path 가 흡수 가능. → **AC loss target / optimizer LR 직접 변경으로 GAT path 회복 시도**.
>
> 근거: planning/DECISIONS.md 2026-05-06 §1(C)/(H)/(I), notebooks/analysis_results/dsn_phase2_mitigation_null_mechanism.md §8.1 #3/#4.

##### Phase 2 baseline 의 AC loss 위치 확인 (사전 작업, 코드 trace)

`train_gat_s06.py:391-395` (수정 전):
```python
if anti_collapse_weight > 0.0 and "column" in node_embs:
    if COL_TO_TAB_EDGE in batch.edge_index_dict:
        col_embs = node_embs["column"]   # ← model.forward 결과
        cb_edge = batch.edge_index_dict[COL_TO_TAB_EDGE]
        step_loss_ac = anti_collapse_fn(col_embs, cb_edge)
```

- `node_embs` 는 `gat_model(...)` 의 반환값 = v2 model.forward 결과.
- `dual_stream=True` 시 `forward` 의 마지막 단계가 `fusion_head[nt](concat([h, z_q, h*z_q]))` (gat_network_v2.py L390-405). 즉 **AC loss 가 fusion 후 결과에 적용** — skip path (skip_dict + fusion_head) 가 우회 가능 — **null mechanism finding 의 정확한 root cause 확인**.

##### Phase 3 #3 — Direct AC on GAT output (PRIMARY)

**메커니즘**: AC loss target 을 `'gat_out_L_last'` 로 변경. forward hook 으로 마지막 `HeteroConv` (= `gat_model.convs[-1]`) 의 column 출력을 capture → AC loss 그 위에 적용 → skip + fusion 우회 차단, main GAT path gradient 회복.

**구현** (`train_gat_s06.py`):
- `anti_collapse_target ∈ {'fusion' (default, Phase 2), 'gat_out_L_last' (Phase 3 #3)}` 옵션 추가.
- `'gat_out_L_last'` 시 학습 시작 전 `gat_model.convs[-1].register_forward_hook(...)` 등록 → 매 forward 마다 column output capture.
- step loop 의 `col_embs` 를 hook capture 로 교체 (default 시 `node_embs["column"]`).

**Config**: [`configs/training/train_gat_directed_supernode_p80_b5_phase3_directAC.yaml`](/home/hyeonjin/thesis_refactored/configs/training/train_gat_directed_supernode_p80_b5_phase3_directAC.yaml)
- Base: Phase 2 b5_mitigation.yaml 그대로
- 변경: `training.anti_collapse_target: "gat_out_L_last"`

**Smoke 검증**:
- hook capture tensor (raw GAT, [N, hidden×heads]) ≠ fusion output ([N, out_channels]) — shape 분리 확인 (24×128 vs 24×64).
- AC loss on hook capture 의 backward 가 last conv 의 inner GATv2Conv params 에 grad 전달 (6/54).

##### Phase 3 #4 — Layer-wise LR (SECONDARY)

**메커니즘**: PyTorch optimizer 의 `param_groups` 활용 — `convs.*` (HeteroConv ModuleList) 와 `*.convs.*` (inner GATv2Conv) 산하 파라미터만 `base_lr × multiplier` (= 5e-4 = 5×). 그 외 (lin_dict / out_lin_dict / skip_dict / pairnorms / fusion_head / query_encoder / classifier_heads) 는 base_lr (1e-4) 그대로 → main GAT path 가 우회 path 대비 5× 빠른 학습.

**구현** (`train_gat_s06.py`):
- `optimizer_layer_wise_lr: bool` + `gat_lr_multiplier: float` 옵션 추가.
- True 시 3 param groups (`gat_convs` / `gat_other` / `classifier_heads`) — `name.startswith("convs.") or ".convs." in name` filter 로 분리.
- False (default) 시 기존 단일 LR optimizer (Phase 2 backward compat).

**Config**: [`configs/training/train_gat_directed_supernode_p80_b5_phase3_layerwiseLR.yaml`](/home/hyeonjin/thesis_refactored/configs/training/train_gat_directed_supernode_p80_b5_phase3_layerwiseLR.yaml)
- Base: Phase 2 b5_mitigation.yaml
- 변경: `training.optimizer_layer_wise_lr: true`, `training.gat_lr_multiplier: 5.0`
- `anti_collapse_target` 미설정 (= 'fusion' default) — #3 와 분리 측정.

**Smoke 검증**:
- filter 정확성: gat-path 108 params / other 52 params (synthetic, num_layers=2). lin_dict / out_lin_dict / skip_dict / fusion_head / query_encoder / pairnorms 모두 other 로 분류.
- LR assignment: gat_convs=5e-4, gat_other=1e-4, classifier_heads=1e-4.
- backward compat: `layer_wise_lr=False` 시 1 group + lr=base_lr (Phase 2 동일).

##### Smoke test (7 케이스 통과)

[`src/modules/selectors/tests/test_phase3_mitigations.py`](/home/hyeonjin/thesis_refactored/src/modules/selectors/tests/test_phase3_mitigations.py):
- `test_p3_3_hook_captures_last_conv_output` — fusion vs raw GAT shape 분리
- `test_p3_3_hook_backward_graph_intact` — AC loss → last conv params grad 전달
- `test_p3_4_param_group_filter_correctness` — 'convs' filter 정확
- `test_p3_4_optimizer_lr_assignment` — 5× LR 적용
- `test_p3_4_backward_compat_baseline` — Phase 2 단일 LR 보존
- `test_phase3_config_parsing` — 두 신규 config 정상
- `test_phase2_baseline_unchanged` — Phase 2 baseline regression 보존

##### 다음 단계 (root + analyzer 핸드오프)

- **5/13~5/16 (root)**: `train_gat_s06.py --config configs/training/train_gat_directed_supernode_p80_b5_phase3_directAC.yaml` 학습 (~12-13h, GPU 0 또는 1). 신규 ckpt: `best_gat_directed_supernode_p80_b5_phase3_directAC.pt`. NAS 저장 + symlink. STEP 3 alpha sweep subset 5 cells (~₩3.8K).
- **5/16~5/19 (root)**: `train_gat_s06.py --config configs/training/train_gat_directed_supernode_p80_b5_phase3_layerwiseLR.yaml` 학습 동일. 신규 ckpt: `best_gat_directed_supernode_p80_b5_phase3_layerwiseLR.pt`.
- **5/19+ (analyzer)**: 두 ckpt 의 (a) val recall ceiling 회복 정도, (b) Skip dependence ratio (gradient flow 재측정), (c) AC target 별 main GAT path gradient 회복 정도 분석. extract_layerwise_attention_v2 재호출.
- **5/19~5/22 (사용자 + planner)**: 학위 논문 Part III chapter — mechanism finding + 4-trial mitigation 시도 narrative. 시나리오 P3-A/B/C 분기 확정.

##### 시나리오 분기 (DECISIONS §1(F))

- **P3-A** (가능성 高): #3 + #4 둘 다 null effect → Filter Dominance 6번째 축 절대 evidence 강화 (3-stage × 4-trial mitigation 모두 ineffective).
- **P3-B** (가능성 中): #3 또는 #4 가 raw R 0.65~0.75 회복 + final F1 plateau 갱신 mid → Skip Dependence pathology partial mitigation 발견 (mechanism + 시도 모두 contribution).
- **P3-C** (낮음): raw R 0.85+ + F1 plateau 결정적 갱신 → main 5 항목 격상 후보.

##### 변경된 파일 (단계 5 산출물)

| 파일 | 변경 |
|------|------|
| `src/train_gat_s06.py` | `anti_collapse_target` (fusion / gat_out_L_last) + `optimizer_layer_wise_lr` + `gat_lr_multiplier` 옵션. AC loss 의 `col_embs` source 분기 + forward hook capture. 단일 LR optimizer → 3 param groups (layer-wise 시) |
| `configs/training/train_gat_directed_supernode_p80_b5_phase3_directAC.yaml` | 신규 — Phase 3 #3 학습 config |
| `configs/training/train_gat_directed_supernode_p80_b5_phase3_layerwiseLR.yaml` | 신규 — Phase 3 #4 학습 config |
| `src/modules/selectors/tests/test_phase3_mitigations.py` | 신규 smoke test 7 케이스 |

#### 단계 6. Mitigation v2 — mech(ii) edge softmax mitigation 3 candidate (2026-05-07 구현 완료)

> **Status: 구현 + 12 smoke 통과. 학위 본 심사 (5/22~6/19) 전 #1+#2+#3 병렬/순차 학습 — DECISIONS 2026-05-07 §1(C)/(D).**
>
> 동기 (4-trial dominance 진단 후속): mech(ii) edge softmax over-concentration **DOMINANT**. attention 이 top-1~5 노드에 sharp peak → 학습이 sharp neighbor selection 에 over-fit. Phase 3 #3/#4 (skip path) 와 별개 root cause. → **edge-softmax level 직접 mitigation 3 candidate 시도**.
>
> 근거: planning/DECISIONS.md 2026-05-07 §1(C)/(D) (사용자 (3)A+B+C 병렬 결정), notebooks/analysis_results/dsn_phase3_mitigation_results.md §8.1.

##### 구현 — gat_network_v2.py 의 GATv2Conv subclass 변형 + HeteroConv aggr

| Candidate | Module / Mechanism | 활성화 옵션 |
|---|---|---|
| **#1 PRIMARY** — DropMessage | `DropMessageGATv2Conv(GATv2Conv)` — `message(x_j, alpha)` 출력 (= x_j × α) 에 `F.dropout(p=drop_message_p, training=training)` 적용. attention α 는 그대로 유지하되 attended-to neighbor 의 feature contribution 분산 | `drop_message_p: 0.2` |
| **#3 SECONDARY** — LayerNorm pre-softmax | `LayerNormGATv2Conv(GATv2Conv)` — `edge_update` 의 raw alpha 산출 후 softmax 직전에 `nn.LayerNorm(heads)` 삽입. softmax sharp peaking 완화 | `use_layernorm_pre_softmax: true` |
| **#2 TERTIARY** — Sum aggregation | HeteroConv `aggr` 인자 변경 (mean → sum / max). cross-edge-type aggregation level 의 inductive bias 변경 (edge softmax 와 별개 layer) | `aggregation_type: "sum"` (또는 `"max"`) |

`_make_gatv2_conv()` factory 가 옵션 조합 dispatch — `#1+#3` combo 는 multiple-inheritance 방식의 `_LayerNormDropMessageGATv2Conv` 동적 클래스 생성. 모든 옵션 default 시 기존 `GATv2Conv` (Phase 2 b8 backward compat).

`SchemaHeteroGATv2.__init__` signature 에 3 옵션 추가:
```python
drop_message_p: float = 0.0,
use_layernorm_pre_softmax: bool = False,
aggregation_type: str = "mean",   # ∈ {mean, sum, max, min, mul}
```

`train_gat_s06.py` 에서 cfg["model"] 의 3 키를 v2 model 로 forward — Phase 2 b8 backward compat (옵션 미설정 시 default OFF, 동일 동작 검증됨).

##### Configs (3 신규)

- [`train_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.yaml`](/home/hyeonjin/thesis_refactored/configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.yaml)  
  Base = Phase 2 b8 mitigation + `drop_message_p: 0.2`
- [`train_gat_directed_supernode_p80_b5_mitigation_v2_layernorm.yaml`](/home/hyeonjin/thesis_refactored/configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_layernorm.yaml)  
  Base = Phase 2 b8 mitigation + `use_layernorm_pre_softmax: true`
- [`train_gat_directed_supernode_p80_b5_mitigation_v2_sum_aggr.yaml`](/home/hyeonjin/thesis_refactored/configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_sum_aggr.yaml)  
  Base = Phase 2 b8 mitigation + `aggregation_type: "sum"`

##### Smoke test (12 케이스 통과)

[`src/modules/selectors/tests/test_mitigation_v2.py`](/home/hyeonjin/thesis_refactored/src/modules/selectors/tests/test_mitigation_v2.py):
- `#1` DropMessage subclass — train ≠ rerun (random), eval deterministic
- `#1` `drop_message_p=0.0` backward compat (super class 동일 결과, atol=1e-6)
- `#1` SchemaHeteroGATv2 + DropMessage forward shape 정합성
- `#3` LayerNormGATv2Conv subclass — `alpha_layernorm` 모듈 (heads,) shape 등록
- `#3` SchemaHeteroGATv2 + LayerNorm — 18 inner convs 모두 alpha_layernorm (9 edge types × 2 layers)
- `#2` `aggregation_type='sum'` forward + HeteroConv.aggr 검증
- `#2` `aggregation_type='max'` forward (사용자 spec sum/max 양쪽)
- Combo `#1+#3` — 18 inner convs 모두 LayerNorm + DropMessage 결합 클래스 적용
- Backward compat — default vs explicit-OFF identical (params=859,008, state_dict keys=160)
- LayerNorm 추가 overhead — +72 params (heads=2 × 2 (γ,β) × 18 LN modules)
- 3 신규 config 파싱 + 옵션 정확
- Phase 2 baseline regression — 신규 옵션 미설정 보존

##### 다음 단계 (root + analyzer 핸드오프)

- **5/9 launch (root)**: GPU 0 = `train_gat_s06.py --config configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.yaml`, GPU 1 = `..._v2_layernorm.yaml` **병렬 학습** (~10h, ETA 5/10 04:00 KST)
- **5/10 launch (root)**: GPU 0 sequential = `..._v2_sum_aggr.yaml` (~10h, ETA 5/11 KST)
- 신규 ckpt: `best_gat_directed_supernode_p80_b5_mitigation_v2_{drop_message, layernorm, sum_aggr}.pt` (NAS path + symlink)
- **5/10, 5/11 (analyzer)**: protocol 재실행 (3 candidate 추가, 7 ckpt × 5 step). mech(ii) attention concentration 회복 정량 (top-5 conc / entropy / L1_GAT cosine).
- **5/12~5/14 (planner)**: 통합 dominance scoring 갱신 (4-trial → 7-trial). §V.5.4 narrative 정식 채택 결정 (시나리오 V2-A/B/C).

##### 시나리오 분기 (DECISIONS §1(F))

- **V2-A** (가능성 高): 3 모두 fail / null effect → §V.5.4 정식 채택 + Filter Dominance 6번째 축 절대 evidence 강화 (7-trial null effect = robustness 결정적).
- **V2-B** (가능성 中): 1-2 partial recovery (val R@15 0.62-0.70) → §V.5.4 narrative 미세 수정 + "Skip Dep null but mech(ii) partial mitigation 발견" contribution.
- **V2-C** (가능성 낮음): 3 모두 ceiling 갱신 (R 0.85+) → §V.5.4 큰 수정 + paper main contribution 재평가 후보.

##### 변경된 파일 (단계 6 산출물)

| 파일 | 변경 |
|------|------|
| `src/models/gat_network_v2.py` | `DropMessageGATv2Conv` + `LayerNormGATv2Conv` subclass + `_make_gatv2_conv` factory + `HETEROCONV_AGGR_TYPES` constant. SchemaHeteroGATv2 __init__ 에 3 옵션 (`drop_message_p` / `use_layernorm_pre_softmax` / `aggregation_type`) + 검증 + state. HeteroConv 인스턴스화 시 factory 사용 + aggr 인자 동적 |
| `src/train_gat_s06.py` | v2 model 인스턴스화에 3 신규 옵션 forward (default OFF backward compat) + log line 보강 |
| `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_drop_message.yaml` | 신규 #1 학습 config |
| `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_layernorm.yaml` | 신규 #3 학습 config |
| `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v2_sum_aggr.yaml` | 신규 #2 학습 config |
| `src/modules/selectors/tests/test_mitigation_v2.py` | 신규 smoke test 12 케이스 |

#### 단계 7. Mitigation v3 #1 — GIN-style aggregation (2026-05-08 구현 완료)

> **Status: 구현 + 7 smoke 통과. 학위 본 심사 (5/22) 전 학습 — DECISIONS 2026-05-08 §1(c) 사용자 결정 A+B 통합 (#1 GIN-style 만 학위 본 심사 전, #2/#3/#4 post-paper).**
>
> 동기 (Mitigation v2 7-trial null effect 후속): mech(ii) edge softmax over-concentration mitigation 3 candidate 모두 fail. softmax 내부 변경 (DropMessage / LayerNorm pre-softmax) 또는 cross-edge aggregation 변경 (Sum/Max) 모두 ineffective → **aggregation function family 자체 변경** (softmax + weighted-mean → MLP + sum) 으로 mech(ii) 가 softmax 한정인지 / family limitation 인지 분기 검증.
>
> 근거: planning/DECISIONS.md 2026-05-08 §1(c)/(e), Xu et al. ICLR 2019 (GIN invariance theorem — sum + injective MLP = WL test 동치).

##### 구현 — gat_network_v2.py

**`AGGREGATION_TYPES` 확장**:
```python
HETEROCONV_AGGR_TYPES = {"mean", "sum", "max", "min", "mul"}     # for HeteroConv aggr
AGGREGATION_TYPES = HETEROCONV_AGGR_TYPES | {"gin"}              # full set
```

**`_make_gin_conv(in_channels, hidden_channels, heads)` factory**:
- PyG `GINConv(mlp, eps=0.0, train_eps=False)` 인스턴스 반환
- MLP = `Sequential(LazyLinear(out_dim), LeakyReLU(0.1), Linear(out_dim, out_dim))` — out_dim = hidden×heads (기존 GATv2Conv 와 동일 차원 → PairNorm/JK/skip path 호환)
- LazyLinear 로 in_channels=-1 동치 (기존 GATv2Conv(-1, ...) 패턴 일관)

**SchemaHeteroGATv2 분기**:
```python
if aggregation_type == "gin":
    conv_dict = {et: _make_gin_conv(-1, hidden, heads=heads) for et in all_edge_types}
    self.convs.append(HeteroConv(conv_dict, aggr="mean"))   # cross-type aggr fixed
else:
    conv_dict = {et: _make_gatv2_conv(...) for et in all_edge_types}
    self.convs.append(HeteroConv(conv_dict, aggr=self.aggregation_type))
```

**GIN incompat 검증**: `aggregation_type='gin'` + `drop_message_p>0` 또는 `use_layernorm_pre_softmax=True` 시 `ValueError` raise — GINConv 는 attention/softmax 자체가 없어 v2 #1/#3 결합 무의미.

##### Config (1 신규)

[`train_gat_directed_supernode_p80_b5_mitigation_v3_gin.yaml`](/home/hyeonjin/thesis_refactored/configs/training/train_gat_directed_supernode_p80_b5_mitigation_v3_gin.yaml)
- Base = Phase 2 b8 mitigation 그대로 (PN+IR+JK+DS+L=2+AC fusion+ListNet)
- 변경: `aggregation_type: "gin"`

##### Smoke test (7 케이스 통과)

[`src/modules/selectors/tests/test_mitigation_v3.py`](/home/hyeonjin/thesis_refactored/src/modules/selectors/tests/test_mitigation_v3.py):
- `test_gin_factory_and_homograph_forward` — GINConv 인스턴스 + homo forward shape (8, 64)
- `test_gin_factory_bipartite_forward` — bipartite (x_src, x_dst) 호환 (HeteroConv 호출 패턴) shape (7, 16)
- `test_full_model_gin_forward` — 18 inner GINConvs (9 edge_types × 2 layers), HeteroConv aggr='mean' fix
- `test_backward_compat_default_mean` — default 시 18 GATv2Convs (no GINConv) regression
- `test_gin_incompatible_with_v2_options` — GIN + #1/#3 ValueError raise
- `test_gin_config_parsing` — 신규 v3 config 정상 + Phase 2 baseline 영향 X
- `test_gin_forward_backward_path` — 11 GINConvs received gradient (column dst path)

##### 다음 단계 (root + analyzer 핸드오프)

- **5/11~5/12 (root)**: `train_gat_s06.py --config configs/training/train_gat_directed_supernode_p80_b5_mitigation_v3_gin.yaml` 학습 (GPU 0, ~10h). 신규 ckpt: `best_gat_directed_supernode_p80_b5_mitigation_v3_gin.pt` (NAS path + symlink).
- **5/12~5/14 (analyzer)**: protocol 재실행 — 7-trial → 8-trial dominance scoring. mech(ii) DOMINANT 가 softmax 한정인지 / aggregation family 한정인지 분기 정량.
  - V3-A (가능성 中, GIN fail): mech(ii) DOMINANT 5/5 절대 강화 (8-trial null = aggregation family 자체 limitation)
  - V3-B (가능성 中, GIN partial): mech(ii) softmax 한정 부분 부정 + GIN sum effect 발견
  - V3-C (가능성 낮음, R 0.85+ ceiling 갱신): mech(ii) 부정 + paper main contribution 재평가
- **5/14~5/22 (사용자)**: 학위 논문 Part III chapter draft 작성 — analyzer Phase 1 후속 3 deep dive (A1/A2/A3) + Mitigation v3 #1 결과 통합 narrative.

##### 변경된 파일 (단계 7 산출물)

| 파일 | 변경 |
|------|------|
| `src/models/gat_network_v2.py` | `AGGREGATION_TYPES = HETEROCONV_AGGR_TYPES ∪ {"gin"}` constant + `_make_gin_conv` factory (PyG GINConv + LazyLinear MLP). SchemaHeteroGATv2 의 aggregation_type 검증 확장 (gin 추가) + GIN incompat 검증 (drop_message_p / use_layernorm_pre_softmax). HeteroConv 인스턴스화 시 GIN/GAT 분기 |
| `configs/training/train_gat_directed_supernode_p80_b5_mitigation_v3_gin.yaml` | 신규 — Mitigation v3 #1 GIN 학습 config |
| `src/modules/selectors/tests/test_mitigation_v3.py` | 신규 smoke test 7 케이스 |

---

#### 단계 8. Mitigation V5 — Tier 1+2 4-Direction (2026-05-13 selector module 정식 ownership)

> **Status: V5-A/V5-B/V5-C 구현 완료. 각 variant smoke 통과 (test_v5_a_gate.py: 5/5, test_v5_b_gcnii.py: 5/5, test_v5_c_aero_full.py: 6/6). 학습 ⏸ root chain 위임 (DECISIONS 2026-05-13 V5 Sweep Launch 재시도).**

> 근거:
> - planning/DECISIONS.md 2026-05-13 (V5 Sweep Launch 재시도) — module:selector 가 처음부터 V5-A/B/C 코드 구현
> - planning/DECISIONS.md 2026-05-12 (V5 Mitigation Plan) — Tier 1+2 4 Direction 병렬 결정
> - planning/oversmoothing_v5_plan.md §4.1/§4.2/§4.3 (학술 Agent 의 mechanism reference)
> - V4 era 의 LN+GIN combo + AERO Softplus + Symmetric Norm 이중 fail → mech(ii-b) 5/5 absolute confirm → V5 architectural intervention 의 4 direction 후속

##### V5-A `GATEGATv2Conv` (alias `GATEConv`) — Conservation Law 수정

Reference: Mustafa, N., & Burkholz, R. (2024). GATE: How to Keep Out Intrusive Neighbors. **NeurIPS 2024**. arXiv:2406.00418.

Mathematical form (paper §3.2 Eq. 4):
$$e_{ij} = \mathbf{a}_s^\top \text{LeakyReLU}(\mathbf{W}\mathbf{h}_i) + \mathbf{a}_t^\top \text{LeakyReLU}(\mathbf{W}\mathbf{h}_j)$$

- 단일 attention vector `a` → `a_s` (self) + `a_t` (neighbor) 분리
- Conservation Law 수정 (paper §3.1 Theorem 1): two-parameter budget exchange 로 small norm 만으로 task-irrelevant neighbor switch-off 가능
- W 공유 (parent GATv2Conv `lin_l/lin_r`) — paper minimal form
- row-stochasticity 유지 (softmax 그대로) — V4-B / V5-C 와 다른 axis

##### V5-B `GCNIIGATv2Conv` — Trainability (Initial Residual + Identity Mapping)

Reference: Chen, M., et al. (2020). Simple and Deep Graph Convolutional Networks (GCNII). **ICML 2020**. arXiv:2007.02133.<br>
Peng, J., Lei, R., & Wei, Z. (2024). Beyond Over-smoothing: Uncovering the Trainability Challenges in Deep Graph Neural Networks. **CIKM 2024**. DOI:10.1145/3627673.3679776.

Mathematical form (Chen 2020 §3.2 Eq. 6):
$$\mathbf{h}^{(l+1)} = \sigma\Big( \big((1-\alpha) \mathbf{P}\mathbf{h}^{(l)} + \alpha \mathbf{h}^{(0)}\big) \big((1-\beta_l) \mathbf{I} + \beta_l \mathbf{W}^{(l)}\big) \Big), \quad \beta_l = \log(\lambda/l + 1)$$

- α (Initial Residual): outer `SchemaHeteroGATv2.initial_residual_alpha` 처리
- β_l: 본 conv 의 `_beta()` — 1-indexed `gcnii_layer_idx` forwarding
- `gcnii_w = Linear(out_dim, out_dim, bias=False)`, `nn.init.eye_` 초기화 (Chen 2020 §3.3 핵심)
- L=2/4/6 sweep — Peng 2024 §4 gradient flow upper bound 분석 검증
- Paradox 2 (ρ_skip ≈ 3) 의 trainability 해석 표적

##### V5-C `FullAEROGATv2Conv` (alias `FullAEROGATConv`) — Full AERO (V4-B + Hop + Cumulative)

Reference: Lee, S. Y., Bu, F., Yoo, J., & Shin, K. (2023). Towards Deep Attention in Graph Neural Networks: Problems and Remedies (AERO-GNN). **ICML 2023**. arXiv:2306.02376.

V4-B (Softplus + Symmetric Norm) + **(b) Cumulative Attention** + **(c) Node-Adaptive Hop Attention** = AERO Theorem 3 SR2OS guarantee 의 full form.

Three components:
- **(a) Softplus + Symmetric Norm** — parent `SoftplusGATv2Conv` (V4-B 상속)
- **(b) Cumulative Attention** (V5-C **신규**): paper §3.2 `α^(l) = α^(l-1) + softplus(e^(l))`. 본 framework 의 layer-별 별도 conv 구조에서 conv-level α 누적 불가 → outer SchemaHeteroGATv2.forward 의 layer loop 에서 **hidden-state level residual** `H^(l) ← H^(l) + λ_cum · H^(l-1)` 로 simulate. `aero_cumulative_attention=True` + `aero_cumulative_decay ∈ [0, 2]`.
- **(c) Node-Adaptive Hop Attention** (Theorem 4): per-node weighted sum `H^out_v = Σ_l ω_v^(l) · H^(l)_v` (L+1 hops 포함 h_0). `aero_hop_attention=True`.

AERO Theorem 3 SR2OS guarantee 의 본 도메인 transfer 검증 — V4-B H10.1c (Hop Attention 부재 가설) 직접 표적.

##### Smoke test 결과 (2026-05-13)

| 파일 | 케이스 | 핵심 검증 |
|---|---:|---|
| `tests/test_v5_a_gate.py` | 5 | alias + att_self Xavier + V-3-ext + att/att_self gradient decoupling + row-stochasticity 유지 |
| `tests/test_v5_b_gcnii.py` | 5 | eye_init + β_l layer-monotone + L=2/4/6 sweep + α/β 분리 + ValueError raise |
| `tests/test_v5_c_aero_full.py` | 6 | alias + SoftplusGATv2Conv 상속 + Theorem 3 full + Hop only + Cumulative only + symmetric_norm 상속 + raise |

##### 시나리오 분기 (DECISIONS 2026-05-13)

| 시나리오 | Trigger | Narrative 영향 |
|---|---|---|
| V5-A 또는 V5-C 단독 R 갱신 | mech(ii-b) 5/5 부분 부정 | architectural intervention 일부 path 효과 |
| V5-B (L=2/4/6) 중 1+ R 갱신 | trainability 가설 (Peng 2024) 부분 confirm | Paradox 2 trainability 해석 confirm |
| V5 4 Direction 모두 fail | mech(ii-b) 5/5 결정적 강화 (current working hypothesis) | 14-trial null + Filter Dominance 6번째 axis 강화 |

##### 변경된 파일 (단계 8 산출물)

| 파일 | 변경 |
|---|---|
| `src/models/gat_network_v2.py` | V5-A `GATEGATv2Conv` + V5-B `GCNIIGATv2Conv` + V5-C `FullAEROGATv2Conv` + alias `GATEConv` / `FullAEROGATConv` + `GAT_LAYER_TYPES` 확장 + `_make_gatv2_conv` dispatch + SchemaHeteroGATv2 V5 ctor (`gcnii_beta_lambda`, `aero_hop_attention`, **`aero_cumulative_attention`**, `aero_cumulative_decay`) + forward V5-C Cumulative path + V5 validation (5 raises) |
| `src/train_gat_s06.py` | V5 kwargs forwarding (gat_layer_type / gcnii_beta_lambda / aero_hop_attention / aero_cumulative_attention / aero_cumulative_decay) |
| `src/modules/selectors/tests/test_v5_a_gate.py` | 신규 — V5-A smoke 5 |
| `src/modules/selectors/tests/test_v5_b_gcnii.py` | 신규 — V5-B smoke 5 |
| `src/modules/selectors/tests/test_v5_c_aero_full.py` | 신규 — V5-C smoke 6 |

##### 다음 단계 (Root chain 위임)

- **5/13~5/16 (Root)**: configs 5 신규 (V5-A / V5-B L=2/4/6 / V5-C) + `scripts/run_v5_mitigation_sweep.sh` + nohup launch (GPU 0/1 병렬, ~30-40h)
- **5/16~5/18 (Root + Analyzer)**: HISTORY/CATALOG/ID_MIGRATION + 14-trial 보고서 `notebooks/analysis_results/dsn_mitigation_v5_4dir.md`
- **5/18~5/22 (Planner)**: narrative pivot 결정 + paper §V.5.4 final integration

---

### V-1/V-2/V-3 통합 default ("SuperNode v2")

- **int_05 전제 default combo**:
  - `selector.num_layers_mode = D_max` (V-1)
  - `selector.supernode_edge_direction = directed_from_sn` (V-2)
  - `selector.supernode_topk = 10` + `supernode_topk_criterion = raw` (V-3, Phase 1 peak 확정 후)
- **Cross-effect**: V-2 단방향은 distant node 가 SN 신호를 1-hop 만 받으므로 V-1 `D_max`(+1) 과 교호 (Proposal D §2 H2). V-3 top-k 는 attention 집중을 강화하지만 과소 k 에서 V-1 depth 가 더 필요해질 수 있음.
- **S-V (Neurosymbolic L1) 와 호환**: V-2/V-3 의 "구조적 희소 라우팅" ≡ S-V 의 "symbolic mask 가산 boost" 와 동형 — 조합 가능 (후속 `abl_sel_ns_l1_sn_v2_*`).

### 변경될 파일 (V-track 한정)

| 파일 | 변경 |
|------|------|
| `src/data/bird_dataset.py` (or graph loader) | `diameter_path` 로드, per-DB `num_layers` resolve, DB-grouped batch sampler (V-1) |
| `src/models/gat_network.py` / `gat_network_v2.py` | `supernode_edge_direction`, `supernode_topk`, `supernode_topk_criterion` 파라미터 + edge 구성 분기 (V-2, V-3) |
| `src/modules/selectors/ensemble_selector.py` | `get_raw_scores(query_emb, node_emb)` utility 노출 — V-3 가 재사용 |
| `src/train_gat_s06.py` (or 후속 trainer) | 신규 flag 노출 + 체크포인트 저장 path 를 NAS 로 라우팅 (symlink step 포함) |
| `configs/experiments/abl/sel/diameter/abl_sel_diameter_layers_nl_{fixed3,dmax,dmax_plus1}.yaml` | 신규 3 개 (V-1) |
| `configs/experiments/abl/sel/sn_v2/abl_sel_supernode_directed_{proj,bce}.yaml` | 신규 2 개 (V-2) |
| `configs/experiments/abl/sel/sn_v2/abl_sel_supernode_topk_raw_k{3,5,10,20}.yaml` | 신규 4 개 (V-3 Phase 1) |
| `data/processed/train_diameter.pt`, `data/processed/dev_diameter.pt` | Builder B-III 산출 (V-1 선결, 1-패스 공유) |

### 재학습 비용 / 일정

| Track | 실험 수 (Phase 1) | 학습 | per-exp ≈ | 총 GPU |
|-------|------------------|------|-----------|--------|
| V-1 | 3 | 3× from scratch | ~9h (batched dual_stream) | ~27h |
| V-2 | 2 | 2× from scratch | ~9h | ~18h |
| V-3 | 4 | 4× from scratch | ~9h | ~36h |
| **합계** | **9** | | | **~81h** (직렬 ~3.4일) |

- 순서: **V-2 → V-3** (V-3 가 V-2 directed 결과 의존). **V-1 은 독립** (병렬 실행 가능, GPU 여유 시).
- 2026-04-28 발표: V-1 + V-2 최소 2 cell 확보 목표 (minimum narrative).

### 리스크

- **V-1 batch 이질성**: per-DB `num_layers` 가 서로 다르면 단일 배치 내 동질성 깨짐 → DB-grouped sampler 필수. 누락 시 성능 측정 신뢰도 급락.
- **V-2 distant-node 단절** (§7.4): 단방향 SN 은 distant schema node 를 1-hop 으로만 도달. **V-1 (`D_max`) 와 조합 권장** — 단독 V-2 는 깊이 부족으로 실패 가능.
- **V-3 gold coverage 상한**: k=3 에서 gold 누락 급증. k=3 은 **진단용**, 실전 default 은 k=10 전후.
- **공통**: 9 개 체크포인트 × ~300 MB → NAS ~3 GB 추가 점유 (여유 1.1 TB 영향 無). 학습 script 에 **NAS 저장 + symlink 스텝 명시** 필수 — 누락 시 로컬 디스크 터짐 위험.

---

## 통합 실험 로드맵 (Selector 관점)

| Phase | 실험 | 의존 | 비고 |
|-------|------|------|-----|
| **QC-A** | `abl_sel_raw_ens_01` + 기존 s04_04/05 재집계 | 없음 (Selector 단독) | **2026-04-21 지도교수 의견 1 대응, core** |
| **V-1** | `abl_sel_diameter_layers_nl_{fixed3,dmax,dmax_plus1}` | **Builder Diameter precompute** + 재학습 | **의견 2 (Q1: D_max), per-DB dynamic `num_layers`** |
| **V-2** | `abl_sel_supernode_directed_{proj,bce}` | 재학습 (canonical `supernode_edge_direction=directed_from_sn`) | **의견 3 — 앞 QC-D rename 수렴** |
| **V-3** | `abl_sel_supernode_topk_raw_k{3,5,10,20}` | V-2 directed 의존 + 재학습 | **의견 4 (Q2: Raw), 앞 QC-E rename 수렴** |
| S1 | `abl_s06_ns1_*` | Builder B-III | 가장 저비용, 먼저 실행 |
| S2 | `abl_s06_rfm_01` (BGE-M3) | Builder B-I | Encoder 교체 단독 효과 |
| S3 | `abl_s06_ehgat_*` | Builder B-II | Line graph 학습 재생성 필요 |
| S4 | `abl_s06_xllm_*` | Filter와 조율 | LLM 공유 자원 |
| S5 | `abl_s06_rl_*` | Warm-start ckpt | 가장 복잡, 마지막 |

**발표 우선순위 (2026-04-28 15~20분)**: **QC-A** (core) → F (SteinerBackbone, 루트) → **V-1** (Diameter, Selector/Builder 공동) → **V-2** (directed SN) → **V-3** (top-k) → B (T2T, Builder). V-2/V-3 는 시간 여유 시, V-1 은 Builder Diameter precompute 완료 조건부. (명명 rename: 앞 QC-D→V-2, QC-E→V-3)

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
