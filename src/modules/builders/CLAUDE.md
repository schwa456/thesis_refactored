# Graph Builder 모듈

> **루트 CLAUDE.md 참조 (읽기 전용)**: 실험 실행, 디렉토리 구조, 문서화 규칙 등 프로젝트 전역 규칙은 [/home/hyeonjin/thesis_refactored/CLAUDE.md](/home/hyeonjin/thesis_refactored/CLAUDE.md)를 반드시 먼저 읽고 따른다. 단, 루트 CLAUDE.md는 수정하지 않는다 — 수정이 필요하면 루트 세션에 요청한다.

## 이 세션의 집중 주제
**Schema → HeteroGraph 변환**. 노드 features, edge 설계, description/메타정보 enrichment.
다른 모듈(Selector, Extractor, Filter)의 내부 구현에 대한 의견은 가급적 보류하고,
**"Builder가 넘겨주는 그래프가 하류 모듈 입력으로 적절한가"** 관점에서만 언급.

## 현재 Builder 구현
- **HeteroGraphBuilder** ([graph_builder.py](graph_builder.py)) — 베이스라인 (default)
  - Table text: `"Table: {name}"`
  - Column text: `"Column: {name} in table {table}, type {type}. Example values: {samples}"`
  - FK text: `"Foreign key relationship connecting ..."`
- **EnrichedHeteroGraphBuilder** (같은 파일) — 완료 (E1 best precision)
  - `database_description/*.csv`의 `column_description`, `value_description` 추가
  - `tables.json`의 자연어 테이블명/컬럼명 추가
  - 캐시: `data/processed/train_enriched_graphs.pt`
  - Checkpoint: `best_gat_enriched.pt`
- **TripletGraphBuilder** — 완료 (E2)
  - `data/processed/triplet_relations.json` 기반 triplet edge embedding
  - `EdgePrizePCSTExtractor`와 결합하여 edge prize 계산
- **RFMCompatibleBuilder** ([graph_builder.py](graph_builder.py)) — 완료 (B-I, 2026-04-20)
  - Enriched 그래프 + RFM 직렬화 텍스트/토큰을 metadata에 부착
  - Special tokens: `[DB] [TAB] [/TAB] [COL] [TYPE] [PK] [DESC] [VAL] [FKS] [FK→]`
  - `metadata['rfm_text' / 'rfm_tokens' / 'rfm_special_tokens']`
  - 별도 API: `serialize(db_id, db_dir)` (그래프 없이 텍스트만)
  - 현 GAT/PCST/XiYan stack은 무시 (behavioral identical to Enriched). S-II RFM encoder wired 후 효과 측정.
- **LineGraphBuilder** ([line_graph_builder.py](line_graph_builder.py)) — 완료 (B-II, 2026-04-20)
  - base builder (Enriched/Triplet/Default) 결과를 line graph로 변환
  - 노드 = 원본 edge (`edge_node`), edge = 노드 공유 (`shares_node`)
  - feat_dim 772 (Enriched base) 또는 1156 (Triplet base, +384 triplet emb)
  - Selector S-III (EHGAT) 가 `edge_node`를 소비할 수 있어야 end-to-end 동작

## 노드 타입 & Edge 구조
노드 타입: `table`, `column`, `fk_node` (3종)
Edge 타입 (heterogeneous):
- `(table, has_column, column)` / `(column, belongs_to, table)`
- `(column, is_source_of, fk_node)` / `(fk_node, points_to, column)`
- `(table, table_to_table, table)` — macro edge

PCST용 flat 인덱싱: `table` 블록 → `column` 블록 → `fk_node` 블록 순으로 concat

## 임베딩
- PLM: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- 학습시 Table/Column/FK 별도 텍스트 리스트로 batch encode

## Builder별 대표 성능 (HISTORY §6-7, §7)
| Builder | Full Pipeline | Recall | Precision | F1 |
|---------|--------------|--------|-----------|----|
| Default | Ensemble + AdaptivePCST + XiYan (B4b) | 0.6244 | 0.7930 | 0.6987 |
| **Enriched** | + `best_gat_enriched.pt` (E1) | 0.6658 | **0.8147** | 0.7327 |
| **Triplet** | + EdgePrizePCST (E2) | 0.6823 | 0.8139 | **0.7424** |

**Insight**: 풍부한 node/edge feature가 전체 파이프라인의 precision 상한을 결정. Enriched와 Triplet 모두 P≈0.81로 최상위.

## 검증 방식 (빠른 smoke test)
```python
from modules.builders.graph_builder import EnrichedHeteroGraphBuilder
builder = EnrichedHeteroGraphBuilder(tables_json_path='data/raw/BIRD_dev/dev_tables.json')
data, meta = builder.build('california_schools', 'data/raw/BIRD_dev/dev_databases')
# 확인: data['table'].x.shape, data['column'].x.shape, len(meta['edges'])
```

## 추후 고려할 축
- Primary Key 여부를 column text에 반영
- 테이블 row count, 컬럼 cardinality 등 통계 정보
- 다른 인코더 (e.g., BGE-m3) 적용 가능성
- 다국어 DB에 대한 대응
- Enriched + Triplet 결합 (서로 직교한 정보원)

## 하류 모듈 인터페이스 계약
Builder는 `(HeteroData, metadata_dict)` 반환. metadata_dict는:
- `table_to_id`, `col_to_id`, `fk_to_id`
- `node_metadata`: flat_idx → text_name
- `edges`: List[(src, dst)] — flat 인덱스 기준
- `edge_types`: List[str] — 위와 같은 길이

**이 계약을 깨면 Selector/Extractor가 모두 깨진다.**

## 생성자 옵션 (B-II.b, 2026-04-21)

| 옵션 | 기본 | 설명 |
|------|------|------|
| `add_t2t_edges` | `True` | False 시 base graph 와 PCST flat 표현 모두에서 `(table, table_to_table, table)` macro edges 제거. `LineGraphBuilder.skip_macro_edges` 와 직교 (base on/off × line-graph on/off → 4 조합). Cache suffix `_no_t2t` 자동. |

`metadata['add_t2t_edges']` 에 사용 중인 값이 그대로 노출되므로 하류에서 토글 상태 검사 가능.

## metadata 추가 키 (B-III, 2026-04-20 / B-III.b, 2026-04-21 — 모든 빌더 공통)

`HeteroGraphBuilder._compute_fk_reachability()` + `_compute_schema_diameter()` 가 build() 종료 직전 자동 주입. 옵션 플래그 없이 항상 활성. T<20 기준 microsec 수준.

### FK reachability (B-III)
| 키 | 타입 | 설명 |
|----|------|------|
| `fk_adjacency` | `np.ndarray[T,T] int8` | 방향성 FK 인접 |
| `fk_adjacency_undirected` | `np.ndarray[T,T] int8` | 조인 경로용 무방향 |
| `fk_reachability` | `np.ndarray[T,T] bool` | transitive closure (undirected) |
| `fk_distance` | `np.ndarray[T,T] float32` | BFS hop, 비도달 = `inf` |
| `fk_shortest_paths` | `Dict[(i,j), {distance, table_path, edge_path, fk_edge_ids}]` | (i≠j 만) |
| `fk_components` | `Dict[table_idx, comp_id]` | undirected weakly connected |
| `fk_num_components` | int | |
| `fk_edge_lookup` | `Dict[(src_tbl, dst_tbl), List[fk_edge_id]]` | 멀티-FK 처리 |

### Schema diameter (B-III.b — full hetero, advisor 2026-04-21 의견 2)
| 키 | 타입 | 설명 |
|----|------|------|
| `schema_diameter` | int | 전체 hetero graph 무방향 D_max (disconnected 시 component max). T2T toggle 영향 받음 (예: california_schools D=4(on)→8(off)) |
| `schema_eccentricity` | `Dict[flat_idx, int]` | 노드별 max finite shortest-path |

**활용**: Selector QCondGAT 의 `num_layers ∈ {1, 2, 3, D_max, D_max+1}` 자동 스윕 (advisor proposal C). Table-only FK subgraph diameter 는 별도 sub-task 로 분리.

**Selector 편의 캐시** ([scripts/build_diameter_cache.py](../../../scripts/build_diameter_cache.py)): `data/processed/<split>_diameter.pt` (`{db_id: D_max}` dict). Enriched/triplet cache 와 동일 패턴 — NAS `/SSL_NAS/peoples/khj/thesis_refactored_offload/processed/` 에 실파일, 로컬은 symlink. Selector 가 graph cache 를 로드하지 않고 D_max 만 읽을 수 있도록 분리.

**활용 시나리오** (하류 세션에서 구현):
- Selector S-V — fk_reachability를 게이트로 사용 (gold join path 가산점)
- Extractor E-III — fk_shortest_paths로 Steiner backbone 보강
- Filter FL-III — fk_components 기반 cross-component column 차감

**커버리지 (BIRD-Dev 11 DB)**: pair 0.9353 / query 0.9445. 미스는 declared FK가 아닌 shared-column join (e.g., `cards.setCode = set_translations.setCode`). 보강 옵션은 implicit FK 추론.

`LineGraphBuilder` / `RFMCompatibleBuilder` 는 위 키들을 그대로 forward + 자체 키 추가 (line graph는 `edge_node_to_orig` 등, RFM은 `rfm_text` 등).

## Wave 8 D2 선결 utility (db_fk_extractor, 2026-05-18)

filters 모듈 Wave 8 Direction 2 (FK/PK Connectivity Steiner Closure) 가 소비하는 DB DDL FK/PK 메타데이터 추출기. builders 영역 책임으로 분리 (DB DDL 파싱 = builder), filters 모듈은 algorithm 만 담당.

- **파일**: [db_fk_extractor.py](db_fk_extractor.py) — SQLite PRAGMA 기반 추출 + `load_db_fk_metadata()` loader
- **출력**: `data/processed/db_fk_metadata.json` (BIRD-Dev 11 DBs / 75 tables / 105 FK constraints, ~40KB)
- **실행**: `PYTHONPATH=src python src/modules/builders/db_fk_extractor.py` (idempotent, `--force` 로 재빌드)
- **형식**: `{db_id: {"tables": {t: {fk_cols, pk_cols, referenced}}, "fk_constraints": [...]}}` — DECISIONS §2 D2 spec + 학술 agent §2.1 flat list 동시 제공
- **하류**: filters 의 `D2SteinerFilter.refine()` 가 lazy load 후 per-DB 사용. LLM 0× (algorithm only).

[EXPERIMENT_PLAN_builders.md](EXPERIMENT_PLAN_builders.md) 의 "Wave 8 D2 선결 인프라" 섹션 + [DECISIONS 2026-05-18 §2 D2](../../../planning/DECISIONS.md) 참조.
