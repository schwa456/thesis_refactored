# Builder 모듈 실험 계획 — 9 아키텍처 제안 중 Builder 관련 축

> **⚑ 먼저 루트 계획을 읽을 것**: [/home/hyeonjin/thesis_refactored/EXPERIMENT_PLAN.md](/home/hyeonjin/thesis_refactored/EXPERIMENT_PLAN.md) — 전 모듈 통합 로드맵, Cross-Module Dependency, 통합 실험(int_01~08), 우선순위 Phase A~E, 논문 매핑이 거기에 있다. **루트 PLAN은 수정하지 않는다** — 수정이 필요하면 루트 세션에 요청.
> **이 파일의 역할**: 루트 PLAN에서 Builder에 할당된 3축(B-I/B-II/B-III)의 **모듈 내부 구현 상세**만 담는다.
>
> **현재 진입점**: `HeteroGraphBuilder` (baseline), `EnrichedHeteroGraphBuilder` (E1, F1 0.7327), `TripletGraphBuilder` (E2, F1 0.7424).
> **주의**: B-III는 Selector S-V / Extractor E-III / Filter FL-III가 의존하는 **critical path 병목** — 루트 PLAN Phase A에서 최우선.

---

## 이 모듈이 받아야 할 제안

| # | 이름 | Builder의 역할 | 우선순위 |
|---|------|---------------|---------|
| B-I | **Relational Foundation Model 호환 Schema Tokenization** (원안 #2) | Zero-shot transfer encoder 입력용 schema serialization & unified token scheme | 중 |
| B-II | **Edge Hypergraph / Line Graph 구성** (원안 #3 EHGAT) | 기존 node 중심 heterograph → edge 중심 line graph 변환 모듈 | 상 |
| B-II.b | **Base heterograph T2T edge toggle** (advisor 2026-04-21 의견 2) | base 단계에서 `(table, table_to_table, table)` edge on/off 스위치 (line-graph 스위치와 직교) | 중 |
| B-III | **FK Reachability / Symbolic Layer 1 전처리** (원안 #9 Neurosymbolic Layer 1) | FK graph 상 reachability matrix, transitive closure, join path 사전 계산 | 상 |
| B-III.b | **Schema Graph Diameter precompute** (advisor 2026-04-21 의견 2) | 각 DB heterograph 의 D_max + node eccentricity. B-III 와 1 패스 공유. Selector num_layers 자동 튜닝 근거 | 상 |

---

## B-I. Relational Foundation Model (RFM) 호환 Schema Tokenization

### 동기
- 제안 #2는 DB-agnostic zero-shot transfer encoder(e.g., Schema-Llama, TableLlama 계열)를 Selector로 채택하는 방향.
- Builder는 **이 encoder가 기대하는 입력 포맷으로 schema를 serialize** 해주어야 한다.
- 현재 `EnrichedHeteroGraphBuilder`의 텍스트는 MiniLM-L6용 문장형 (`"Column: {name} in table {table}..."`) — RFM에는 **구조적 구분 토큰**이 필요.

### 설계 요소
- **Unified schema tokenizer**: `[TAB] table_name [COL] col1 [TYPE] int [DESC] ... [VAL] ... [FK→] tab2.col` 등 special token 체계.
- **DB 단위 serialization API**: `serialize_db(db_id) -> str` — FK 관계까지 연결된 flat 텍스트.
- Builder는 동일 DB에 대해 (a) 기존 heterograph 경로 (b) flat-serialized 경로를 모두 반환 가능해야 함.

### 인터페이스
```python
class RFMCompatibleBuilder(EnrichedHeteroGraphBuilder):
    def build(self, db_id, db_dir) -> Tuple[HeteroData, metadata_dict]
    def serialize(self, db_id, db_dir, include_values=True, max_values=3) -> str
    # metadata에 "rfm_tokens": List[str] 추가 (토큰 단위 alignment 저장)
```

### 의존성 / 주의
- 실제 RFM 체크포인트는 Selector 세션이 선정. Builder는 **다양한 포맷을 유연히 지원** 하기만 하면 됨.
- 토큰 길이 폭발 가능 — per-DB 평균 토큰 수 프로파일링 필수.
- 기존 캐시 구조와 분리: `_rfm` suffix.

### 검증
- 동일 DB(예: california_schools)에 대해 serialize 결과 수동 확인 (누락 FK, 타입 정보 체크).
- RFM encoder 추론 1건 작동 확인 (mock model로라도).

### 학술 기여
- "Our Builder provides a zero-shot-ready schema serialization pipeline that is decoupled from downstream encoder choice" — 재현성 축.

---

## B-II. Edge Hypergraph / Line Graph 구성 (EHGAT 지원)

### 동기
- 제안 #3 EHGAT은 **edge를 노드로 승격한 line graph** 에서 attention을 수행 — 기존 node 중심 GAT이 놓치는 multi-hop join pattern 학습이 목적.
- Builder는 **기존 HeteroData → LineGraph 변환기** 를 제공해야 Selector/Encoder가 EHGAT을 학습 가능.

### 설계 요소
- `has_column`, `belongs_to`, `is_source_of`, `points_to`, `table_to_table` 등 5종 edge를 노드로 승격.
- 두 edge가 **공유 노드를 가질 때** line-graph에 edge 연결.
- Edge 자체의 feature는 다음으로 구성:
  - edge type one-hot
  - 양 끝 노드 임베딩 평균 / 차이
  - FK의 경우 triplet embedding (TripletGraphBuilder와 합류 가능)

### 인터페이스
```python
class LineGraphBuilder:
    def __init__(self, base_builder: HeteroGraphBuilder):
        self.base = base_builder
    def build(self, db_id, db_dir) -> Tuple[LineGraphData, metadata_dict]
    # metadata["edge_node_to_orig"]: new_idx → (src, dst, edge_type)
    # metadata["orig_node_to_edges"]: node_idx → List[edge_node_idx]
```

### 의존성 / 주의
- `torch_geometric.transforms.LineGraph` 는 homogeneous만 지원 → heterograph 수동 전개 필요.
- Label 전파 규칙 정의 필요: **edge_node 가 gold 인가?** (예: 양 끝 노드 모두 gold면 1.0)
- PCST 인터페이스 계약과 별도 저장소: EHGAT은 별도 실험 경로 (`abl_ehgat_*`).

### 검증
- 작은 DB(예: california_schools 단일)에 대해 line graph 변환 smoke test.
- 노드/에지 수 sanity check (line graph 노드 수 ≈ 원 그래프 edge 수).
- 레이블 분포가 extreme하게 skew되지 않는지 확인.

### 예상 효과 (원안 인용)
- Multi-hop join 경로 학습으로 FK 노드 recall 개선 기대.
- 특히 3-table JOIN 쿼리(현재 문제 지점)에서 bridge table 인식 강화.

---

## B-II.b. Base Heterograph T2T Edge Toggle (advisor 2026-04-21 의견 2)

### 동기
- 현재 `LineGraphBuilder.skip_macro_edges` 는 **line-graph 변환 단계**에서만 macro edge 를 제거 → base heterograph 의 `(table, table_to_table, table)` edge 는 여전히 GAT/PCST 메시지 패싱에 흐른다.
- 지도교수 의견 2: **GAT Over-smoothing** 진단 후 처방 — T2T edge 가 over-smoothing 가속 요인일 가능성. base 단계에서 끄고 비교해야 함.
- B-II 의 line-graph 스위치와 **직교** (base T2T on/off × line-graph T2T on/off → 4 조합 가능).

### 설계 요소
- **`HeteroGraphBuilder` 생성자 인자**: `include_table_to_table: bool = True` (default True 로 backward compat).
  - False 시: `(table, table_to_table, table)` edge 와 메타데이터의 `edge_types == "table_to_table"` 항목을 모두 제외.
  - `EnrichedHeteroGraphBuilder` / `TripletGraphBuilder` / `RFMCompatibleBuilder` 모두 super 호출로 자동 전파.
- **PCST flat indexing 영향 없음**: T2T 는 macro edge 영역만 차지 → table/column/fk_node 노드 인덱스는 동일.
- **Cache suffix**: `_no_t2t` (Enriched off → `train_enriched_no_t2t_graphs.pt`). `bird_dataset.py` 에서 builder param 검사 후 분기.

### 인터페이스
```python
class HeteroGraphBuilder:
    def __init__(self, ..., include_table_to_table: bool = True):
        self.include_table_to_table = include_table_to_table
    def _build_macro_edges(self, ...):
        if not self.include_table_to_table:
            return [], []  # skip both edges and edge_types
        ...
```

### 의존성 / 주의
- Triplet builder의 edge embedding 계산에서 T2T 가 차지하는 비중 확인 (있다면 dim 변동 없도록 type-only zero embed 유지).
- `LineGraphBuilder` 의 `EDGE_TYPE_ORDER` 는 4 종류 고정 — base 에 T2T 가 없을 때 line-graph 가 해당 idx 를 빈 채로 두는지 확인. (현재 spec: 비어 있어도 OK, type_oh 가 0-vector 가 됨)
- FK reachability 메타데이터 (B-III) 는 T2T 와 무관 (FK adjacency 만 사용) → 영향 없음.

### 검증
- california_schools 에서 `include_table_to_table=False` 시 macro edge 0 개 확인.
- Smoke test: 기존 `scripts/smoke_test_b3_fk_reach.py` 를 `include_table_to_table=False` 로 한 번 더 실행해 reachability/components 가 변하지 않는지.
- (실험 abl_build_05_no_t2t — §통합 실험 로드맵 참조)

### 학술 기여
- "We expose a base-level T2T edge toggle to disentangle macro-edge contribution from line-graph reformulation" — over-smoothing 분석의 control variable 제공.

---

## B-III. FK Reachability / Symbolic Layer 1 (Neurosymbolic 지원)

### 동기
- 제안 #9의 Layer 1(**Symbolic**)은 FK 그래프를 **결정론적으로** 활용 — reachability, 최단 join path, transitive closure.
- 현재 Builder는 FK 노드를 graph에 주입하지만 **FK-only subgraph의 알고리즘적 성질**(connected components, shortest path matrix)은 계산하지 않음.
- Selector/Extractor/Filter가 symbolic 힌트를 쓸 수 있게 metadata에 포함시켜야 함.

### 설계 요소
1. **FK-only adjacency matrix** (`fk_adj[table_i][table_j] = 1 if FK exists`)
2. **Reachability matrix**: transitive closure (Floyd-Warshall).
3. **All-pairs shortest FK path**: distance + actual path (FK edges list).
4. **Connected components**: disconnected table cluster 식별.

### 인터페이스
```python
class HeteroGraphBuilder:  # 또는 Enriched/Triplet
    def build(self, db_id, db_dir):
        data, meta = ...
        meta["fk_reachability"] = compute_reachability(meta)   # numpy (T, T)
        meta["fk_shortest_paths"] = compute_all_pairs_paths(meta)  # Dict[(i,j)] → List[edge_ids]
        meta["fk_components"] = compute_components(meta)       # Dict[table_idx] → comp_id
        return data, meta
```

### 의존성 / 주의
- 테이블 수가 크지 않음 (BIRD dev 평균 10개 미만) → Floyd-Warshall O(T³) 허용.
- **Selector(#9 Layer 1)** 는 이 matrix를 활용해 "gold-likely seed 간 FK reachable 여부"로 GAT prior 보강.
- **Extractor(#9 Layer 2, PCST ensemble)** 는 이 matrix로 bridge table 강제 포함.
- **Filter(#9 Layer 3 Verifier)** 는 최종 선택 테이블 집합이 FK 그래프에서 disconnected인지 검증.

### 검증
- 4~5 DB에 대해 수동으로 FK 그래프 그려 비교.
- Gold SQL의 JOIN이 실제 FK reachability 안에 있는지 (covered ratio).
- 캐시 용량 프로파일링: matrix 저장 시 metadata 크기 증가 확인.

### 학술 기여
- "We precompute FK-reachability at the Builder stage, enabling all downstream modules to consult a unified symbolic prior without redundant computation."
- Neurosymbolic framing의 **인프라적 기여** 축.

---

## B-III.b. Schema Graph Diameter Precompute (advisor 2026-04-21 의견 2)

### 동기
- 지도교수 의견 2: GAT 의 `num_layers` 를 **schema graph 의 diameter** 에서 결정하자는 처방.
- 너무 작은 num_layers → distant gold node 가 NLQ 신호를 못 받음. 너무 큰 num_layers → over-smoothing.
- D_max (전체 그래프 최대 shortest-path) 를 metadata 로 노출 → Selector 가 `num_layers ∈ {1, 2, 3, D_max, D_max+1}` 자동 스윕 가능.
- **B-III FK reachability 와 1 패스 공유** — 동일 BFS/Floyd-Warshall 인프라 활용. 추가 비용 microsecond 단위.

### 설계 요소 (이번 라운드 — full hetero 만)
- **계산 대상**: **Full hetero schema graph** (table + column + fk_node 모두 포함, undirected) → `schema_diameter`
  - GAT/QCondGAT 의 메시지 패싱이 hetero 그래프 위에서 일어나므로 num_layers 결정에 직결.
  - Undirected 처리 (메시지 패싱은 양방향). disconnected 컴포넌트가 있을 때는 각 컴포넌트별 diameter 계산 후 max.
- **Eccentricity**: 각 노드별 max finite shortest-path → `schema_eccentricity` (Dict[flat_idx, int])

### 별도 트랙 (이번 라운드 미포함)
- **Table-only FK subgraph diameter** (`schema_diameter_table_only`): B-III 의 `fk_adjacency_undirected` 위 diameter. **별도 sub-task 로 분리** — full hetero 와 의미하는 바가 다르므로 (table-only 는 join path 기반 receptive field, full hetero 는 GAT depth) 사용처도 다름. 필요 시점에 추가.

### 인터페이스
```python
class HeteroGraphBuilder:
    def _compute_schema_diameter(self, table_to_id, col_to_id, fk_to_id, pcst_edges) -> Dict[str, Any]:
        # 전체 hetero graph 의 무방향 인접 → BFS all-pairs
        # disconnected components 각각 diameter 계산 후 max
        return {
            "schema_diameter": int,                  # full hetero D_max
            "schema_eccentricity": Dict[int, int],   # flat_idx → ecc
        }
```

별도 함수로 두고 build() 끝에서 `_compute_fk_reachability` 와 함께 호출. PCST flat indexing (table → column → fk_node concat) 의 `pcst_edges` 를 재활용해 sparse adjacency 를 단발 구성.

### 메타데이터 추가 키
| 키 | 타입 | 설명 |
|----|------|------|
| `schema_diameter` | int | 전체 hetero graph 무방향 D_max (disconnected 일 시 component별 max 의 최대값) |
| `schema_eccentricity` | `Dict[flat_idx, int]` | hetero 그래프 노드별 ecc |

### 의존성 / 주의
- BIRD-Dev 평균 schema 노드 수 < 100 → BFS 비용 무시 가능.
- Disconnected 컴포넌트가 있을 때: 각 component diameter max 를 `schema_diameter` 로 정의 (cross-component 거리 = inf 는 제외).
- `LineGraphBuilder` 는 base 의 schema_diameter 를 forward 만 (line-graph 자체 diameter 는 별도 키로 둘 가치 낮음 — Selector 가 base graph num_layers 결정용).
- 캐시: `data/processed/{json_basename}_diameter.pt` 에 per-DB 저장 (B-III reachability 와 같은 캐시 파일에 통합 가능).

### 검증
- california_schools (T=3, 1 component): D_max ∈ {2, 3} 예상. `frpm`-`schools`-`satscores` 형태.
- BIRD-Dev 11 DB 전체에 대해 D_max 분포 프로파일링 (hist + median). 큰 DB (e.g., european_football_2 T~24) 의 D_max 가 현재 GAT default 3-layer 대비 너무 큰지 체크.
- `num_layers` 스윕 실험 (`abl_sel_diameter_layers`, advisor proposal C) 결과와 비교 — diameter가 최적 num_layers 와 상관 있는지.

### 학술 기여
- "Schema-aware adaptive depth: we expose per-DB graph diameter at build time so encoders can adapt receptive field to schema topology without manual tuning."

### 의존성 그래프 (이 모듈 내)
```
B-III FK reachability  ──┐
                          ├── 1-pass precompute @ build() end
B-III.b Diameter (이 절) ──┘
                                    ↓
                  metadata{fk_*, schema_diameter*}
                                    ↓
              Selector S-V (FK 게이트) / QCondGAT (num_layers tuning)
```

---

## 통합 실험 로드맵 (Builder 관점)

모든 실험은 기존 cache 구조와 충돌하지 않도록 **builder suffix**로 분리. E1/E2와 호환 유지.

| Phase | 실험 ID | Builder | 연동 하류 | 학습 포인트 |
|-------|---------|---------|----------|------------|
| B1 | `abl_build_01_fk_reach` | `HeteroGraphBuilder` + FK reachability | 기존 Selector/Extractor 그대로 | Layer 1 infra 검증 |
| B2 | `abl_build_02_linegraph` | `LineGraphBuilder(base=Enriched)` | EHGAT Selector (S-III) | Edge-centric 학습 pilot |
| B3 | `abl_build_03_rfm_tokens` | `RFMCompatibleBuilder` | RFM Selector (S-II) | Zero-shot transfer readiness |
| B4 | `abl_build_04_enriched_triplet` | Enriched + Triplet 결합 | EdgePrize PCST | 두 최고점(E1 × E2) 시너지 |
| **B5** | `abl_build_05_no_t2t` | `EnrichedHeteroGraphBuilder(include_table_to_table=False)` | 기존 GAT/QCondGAT | **B-II.b** — base T2T off → over-smoothing 변화 |
| **B6** | `abl_build_06_diameter_meta` | 모든 빌더 (metadata 키 추가만) | Selector QCondGAT (`num_layers ∈ {1,2,3,D_max,D_max+1}`) | **B-III.b** — diameter precompute, 인프라 검증. anchor E1 와 noise 일치 |

E1/E2의 precision 상한(0.81) 유지가 선결 조건. 새 builder가 기존 베이스라인 대비 precision 하락이 크면(>−3%p) rollback.

---

## 변경될 파일

| 파일 | 변경 |
|------|------|
| [graph_builder.py](graph_builder.py) | `RFMCompatibleBuilder` 신규 + `_compute_fk_reachability` 함수 + **`include_table_to_table` 생성자 인자 (B-II.b)** + **`_compute_schema_diameter` 함수 (B-III.b)** |
| [line_graph_builder.py](line_graph_builder.py) | 신규 — EHGAT용 line graph 변환. **B-II.b 상황에서 `EDGE_TYPE_ORDER` 의 빈 T2T idx 처리 검증** |
| [cached_builder.py](cached_builder.py) | 새 builder suffix 캐시 정책 추가 (`_enriched`, `_no_t2t` 조합) |
| `src/data/bird_dataset.py` | RFM / LineGraph builder 캐시 경로 처리 + **`include_table_to_table=False` 시 `_no_t2t` suffix 분기 (B-II.b)** |
| `configs/experiments/abl/build/no_t2t/abl_build_05_no_t2t.yaml` | 신규 (B-II.b) |
| `configs/experiments/abl/build/diameter_meta/abl_build_06_diameter_meta.yaml` | 신규 (B-III.b — anchor E1 noise 일치 확인용) |
| `scripts/smoke_test_b2b_no_t2t.py` | 신규 — B-II.b smoke + reachability invariance 검증 |
| `scripts/smoke_test_b3b_diameter.py` | 신규 — B-III.b smoke + 11 DB D_max 프로파일링 |

## 하류 모듈에 대한 계약 (유지)
- 반환 포맷 `(HeteroData, metadata_dict)` 는 **절대 깨지 않음**.
- 새 정보는 모두 `metadata_dict` 의 추가 키로만 노출 — 기존 Selector/Extractor/Filter가 무시해도 동작.
- LineGraph는 별도 타입 (`LineGraphData`) 을 반환하며, 이를 기대하는 Selector만 소비.

## 검증 방법 (모듈 내)
- **단위 smoke test**: `pytest tests/test_builders.py` — 각 builder에 대해 california_schools build 성공 + 필수 metadata 키 존재.
- **정합성**: FK reachability가 gold SQL의 JOIN 관계를 포함하는 비율(coverage) > 95%.
- **캐시 분리**: suffix 충돌 없는지 dry run (실제 저장 전 경로 출력만).
