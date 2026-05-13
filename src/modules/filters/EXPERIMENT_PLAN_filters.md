# Filter 모듈 실험 계획 — 9 아키텍처 제안 중 Filter 관련 축

> **⚑ 먼저 루트 계획을 읽을 것**: [/home/hyeonjin/thesis_refactored/EXPERIMENT_PLAN.md](/home/hyeonjin/thesis_refactored/EXPERIMENT_PLAN.md) — 전 모듈 통합 로드맵, Cross-Module Dependency, 통합 실험(int_01~08), 우선순위 Phase A~E, 논문 매핑이 거기에 있다. **루트 PLAN은 수정하지 않는다** — 수정이 필요하면 루트 세션에 요청.
> **이 파일의 역할**: 루트 PLAN에서 Filter에 할당된 3축(FL-I/FL-II/FL-III)의 **모듈 내부 구현 상세**만 담는다.
>
> **현재 진입점**: `XiYanFilter` (anchor, F1=0.6940 on a03_17), `ReflectionFilter` (F1=0.7068 신기록 on a05_02).
> **이미 존재하는 a05 agentic 라인**과 통합하여 중복 실험을 피한다.
> **선결 의존성**: FL-III는 Builder B-III의 FK reachability matrix에 의존. 루트 PLAN Phase A 완료 전에는 FL-III 블록.
> **연관 계획**: [/home/hyeonjin/.claude/plans/vivid-sprouting-sunbeam.md](/home/hyeonjin/.claude/plans/vivid-sprouting-sunbeam.md) (F1~F5, a05 시리즈).

---

## 이 모듈이 받아야 할 5가지 제안

| # | 이름 | Filter의 역할 | 우선순위 | a05와의 관계 |
|---|------|---------------|---------|-------------|
| FL-I | **Autonomous Schema Exploration Agent (AutoLink-style)** (원안 #6) | Iterative ReAct agent + graph-native tools로 탐색적 refinement | 상 | a05 F3 (Tiered Bidirectional) 의 확장 — full exploration variant |
| FL-II | **Extractive Decoder-only LLM — Filter mode** (원안 #7) | LLM span extraction + logit score로 token-level pruning | 중 | 새 축 (a05 외) |
| FL-III | **Symbolic-Neural Layer 3 — Verifier** (원안 #9 Layer 3) | FK graph 상 connectivity 검증 + disconnected 결과 reject | 상 | a05 F2 (VerifierFilter) 의 강화형 |
| **SGBE** | **Score-Gated Batch Extractive Filter** (학술 Agent 2026-05-12) | GAT score 분포 기반 column-level 3-way routing + extractive binary LLM | **최상** | XiYan 대체 candidate. FL-I/II/III 와 직교 — 결합 가능 |
| **RSL-A** | **RSL-SQL Backward Filter (Cao 2024)** (학술 Agent Phase 3, 2026-05-13) | XiYan forward + preliminary SQL backward → S_restore union (forward 가 놓친 column 회복) | **최상** | XiYan 의 recall 보강 path. Direction A GO 확정 (Phase 2 PASS 3/3). |

---

## RSL-A. RSL-SQL Backward Filter (★ 최상, Direction A GO 확정 2026-05-13)

### 동기
- Cao 2024 RSL-SQL 의 **backward path**: forward filter (예: XiYan) 가 PCST subgraph 위에서 prune-only 로 동작 → recall 손실. backward 는 **full schema** 위에서 preliminary SQL 을 생성해 거기 등장하는 column 들을 forward 결과 위에 합쳐 (union) recall 을 보강.
- 학술 Agent Phase 2 (fix 후, 2026-05-13) 측정:
  - mean(`S_restore_precision`) = **0.6434** (threshold ≥ 0.60, margin 1.07×) ✅
  - mean(Δrecall_union vs fwd) = **+0.0771** (threshold ≥ +0.05, margin 1.54×) ✅
  - mean(`recall_gained_by_restore`) = **0.5709** — "forward 가 놓친 gold column 의 **57% 를 backward 가 회복**" (학위 논문 §V.5.x 핵심 인용)
- 학술 Agent Phase 3 (2026-05-13) **Direction A GO 확정** + B Hold + C 재결정 ΔF1(A) trigger 분기.

### 설계 — 4-step pipeline (2 LLM calls per query)
```
Step 1  XiYan forward (의존성)
        S_fwd = XiYanFilter.refine(query, subgraph, db_id).final_nodes

Step 2  Preliminary SQL backward (GLM 4.7, full schema input)
        prelim_sql = client.generate_text(prompt=load("rsl_backward_preliminary_sql", schema_str=full, ...))
        L_bwd = sqlglot.extract_columns(prelim_sql, col-only-distinct)
        ^ Phase 2 bug fix 후 normalization 정합 (alias-distinct → col-only)
        ^ SQL keyword 검증 (SELECT/WITH/...) 후 parse, parse 실패 시 빈 set (recall-safe)

Step 3  S_restore + DB-level guard (조건부)
        S_restore_col = L_bwd - col_only(S_fwd)
        if db_id ∈ risky_dbs:                         ← Phase 3 margin caveat 1.07× → guard
            S_restore = ∅
        else:
            S_restore = expand_to_full_paths(S_restore_col, full_schema)
                        ^ col_name 이 여러 table 에 있으면 모두 후보
                          (Cao 2024 RSL-SQL 정합)

Step 4  S_struct FK/PK hardcode (CHESS Talaei 2024)

Output: final_nodes = S_fwd ∪ S_restore ∪ S_struct
```

### 인터페이스 (계약 유지)
```python
@register("filter", "RSLBackwardFilter")
class RSLBackwardFilter(BaseFilter):
    def __init__(self,
                 model_name="zai-org/glm-4.7", temperature=0.0,
                 xiyan_max_iteration=1, xiyan_model_name=None,
                 xiyan_num_examples=3,
                 db_dir="./data/raw/BIRD_dev/dev_databases", num_examples=3,
                 fk_pk_hardcode=True,
                 risky_dbs=None,                    # ["toxicology", ...] 명시
                 provider="glm", api_key=None, base_url=None, **kwargs): ...

    def refine(self, query, subgraph, db_id=None,
               tier2_pool=None, gat_scores=None,    # SGBE 와 동일 시그니처
               metadata=None, evidence=None, **kwargs) -> Dict:
        # 반환: {"status", "final_nodes", "reasoning",
        #        "stats": {"fwd_nodes", "bwd_col_names", "restore_col_diff",
        #                  "restore_expanded", "struct", "final",
        #                  "db_guard_active", "sql_parse_ok", "restore_is_empty"},
        #        "preliminary_sql": <str>,
        #        "filter_info": {...}}
```

### DB-level Guard (Phase 3 margin caveat)
- Phase 1 margin 1.20× → Phase 2 fix 후 **1.07× 좁아짐**. toxicology 외 추가 low-precision DB 발견 시 `risky_dbs` 갱신 권장.
- Implementation 결정: yaml configurable `risky_dbs: List[str]` (default 빈 list — 전체 적용). guard 동작 시 `stats["db_guard_active"]=True` + reasoning 에 명시.
- 학술 Agent Q3 implementation detail 위임 — 본 모듈에서는 simple skip-list 채택 (옵션 a). query-level estimate (옵션 b) 는 future work.

### 비용
- LLM calls per query: **2** (XiYan + preliminary SQL)
- Token cost: anchor 대비 **~+100%** (preliminary SQL 의 full schema input)
- sqlglot parse: 무시 가능 cost (CPU 수십 ms)

### 한계 / Caveat
- **margin 1.07× 좁음** — toxicology 외 추가 low-precision DB 시 risky_dbs 갱신 필요 (Phase 3).
- **full schema input** → DB 가 큰 경우 (debit_card_specializing 100+ table 등) prompt 길이 증가. max_tokens 조정 또는 schema trimming candidate (future work).
- **col_name 중복** — backward SQL 의 wrong-table-prefix 가 schema 의 모든 후보 table 에 expand → noise 증가 가능. 학술 Agent Phase 2 측정에서 precision 0.6434 로 PASS, 단 future work 으로 sqlglot qualify 기반 정확 table resolve 가능.

### 산출물 (본 모듈 책임, 2026-05-13)
- [`rsl_backward_filter.py`](rsl_backward_filter.py) — `RSLBackwardFilter` 신규 구현. XiYan forward composition + sqlglot col-only-distinct extraction + risky_dbs guard + FK/PK hardcode.
- [`tests/test_rsl_backward.py`](tests/test_rsl_backward.py) — 15-scenario smoke test (**PASSED 15/15**). 핵심 시나리오: clean SQL restore / S_restore=∅ (54.50% Phase 1 정합) / risky_db guard / FK hardcode / SQL parse fail recall-safe / metadata fallback.
- [`/home/hyeonjin/thesis_refactored/src/prompts/filter.md`](/home/hyeonjin/thesis_refactored/src/prompts/filter.md) — `rsl_backward_preliminary_sql` section 추가.
- [`__init__.py`](__init__.py) — `RSLBackwardFilter` export.

### 다음 단계 (Root 책임)
- Direction A pipeline config 작성 (`configs/experiments/abl/.../rsl_backward_*.yaml`)
- anchor (XiYan) + Backward sweep launch → ΔF1 / ΔEX 정량
- Analyzer 보고: per-DB breakdown + ΔF1 trigger 분기 (≥ 0.03 → Direction C post-paper / < 0.02 → C 타겟 launch)

---

## SGBE. Score-Gated Batch Extractive Filter (★ 최상)

### 동기
- XiYan (anchor F1=0.6940) 의 prune-only recall 손실 ~0.15 의 mechanism 이 진단됨 (Yuan 2025):
  - TP (gold+kept) mean GAT score **0.7108**
  - Filter✗ (wrong-pruned gold) mean **0.6394**
  - TN (non-gold+dropped) mean **~0.40**
- 세 group 의 score 분포가 **이미 구간으로 분리**되어 있다는 사실이 핵심. LLM 이 전체 subgraph 를 한 번에 보면 Filter✗ 그룹을 잘못 판단 → recall 손실. **Column-level routing** 으로 LLM 의 판단 범위를 mid-confidence 구간으로 좁히면 recall+precision+속도가 동시에 개선됨.
- 학술 Agent 2026-05-12 ([planning/filtering_suggestion_by_scholar_agent_2026-05-12.md](/home/hyeonjin/thesis_refactored/planning/filtering_suggestion_by_scholar_agent_2026-05-12.md)) 가 5 references (Glass 2025 / Hoang 2025 / Talaei 2024 / Maamari 2024 / Yuan 2025) 로 합성한 hybrid 설계.

### 설계 — 3-step routing
```
Step 0  Structural Hard Keep        0 LLM calls
        S_struct = FK/PK columns in S_pcst      ← 무조건 keep (CHESS hardcode rule, Talaei 2024)

Step 1  Score-Gate                  0 LLM calls, O(n)
        θ_keep = 0.65 (TP mean 0.7108 기반)
        θ_drop = 0.40 (TN mean ~0.40 기반)
        S_keep_hard  = {v | s_v ≥ θ_keep}       → 즉시 keep
        S_drop_hard  = {v | s_v < θ_drop}       → 즉시 drop
        S_uncertain  = {v | θ_drop ≤ s_v < θ_keep}  → LLM 대상

Step 2  Extractive LLM              1 LLM call, S_uncertain 만
        per-column binary 판단 ("yes/no + one-line reason") with value samples
        S_lm_keep ⊆ S_uncertain

Output: final_nodes = S_keep_hard ∪ S_lm_keep ∪ S_struct
```

### 세 조건 충족 mechanism
- **Recall 보호**: TP mean 0.7108 → θ_keep=0.65 로 대부분 TP 가 Step 1 에서 즉시 keep. LLM 이 TP 그룹에 접근 불가 → wrong-prune 이 구조적으로 불가능.
- **Precision 향상**: TN mean ~0.40 → θ_drop=0.40 으로 명확한 noise 가 LLM 없이 즉시 제거. Step 2 의 extractive binary 판단은 generative list 보다 column 간 독립.
- **빠른 추론**: LLM input token **60~80% 감소** (S_uncertain ≈ 20~40% 의 전체).

### 인터페이스 (계약 유지)
```python
@register("filter", "ScoreGatedBatchExtractiveFilter")
class ScoreGatedBatchExtractiveFilter(BaseFilter):
    def __init__(self,
                 model_name="zai-org/glm-4.7",
                 theta_keep=0.65, theta_drop=0.40, temperature=0.0,
                 db_dir="./data/raw/BIRD_dev/dev_databases",
                 num_examples=3, fk_pk_hardcode=True,
                 step_mode="step_0+1+2",              # 신규 (2026-05-12 follow-up)
                 score_collapse_threshold=0.05,        # 신규 (2026-05-12 follow-up)
                 provider="glm", api_key=None, base_url=None, **kwargs): ...

    def refine(self, query, subgraph, db_id=None,
               tier2_pool=None, gat_scores=None, metadata=None, **kwargs) -> Dict:
        # 반환: {"status", "final_nodes", "reasoning",
        #        "stats": {"step_mode", "keep_hard", "drop_hard", "uncertain",
        #                  "lm_keep", "struct", "score_collapse_detected"},
        #        "filter_info": {...}}
```

### Option 1 — `step_mode` (Phase 3/5 분리 평가용, 2026-05-12 follow-up)
| step_mode | 흐름 | LLM call | 용도 |
|-----------|------|----------|------|
| `"step_0"` | FK/PK Hardcode 만 | 0 | Phase 5 ablation 의 Step 0 only baseline |
| `"step_0+1"` | + Score-Gate (S_uncertain 전부 drop) | 0 | Phase 3 calibration sweep 의 "LLM call 없는 Step 0+1 평가" — θ_keep × θ_drop grid 빠른 탐색 |
| `"step_0+1+2"` (default) | Full SGBE | 1 | Phase 4 final SGBE 평가 |

- Backward compat: 미명시 시 default `"step_0+1+2"` → 기존 검증 시나리오 그대로 통과.
- `stats["step_mode"]` 가 결과에 동봉되어 analyzer 가 step 별 contribution 을 직접 집계 가능.

### Option 2 — `score_collapse_threshold` (학술 Agent §"한계" 보강)
- candidate score 들의 std 가 threshold 미만이면 score 분포가 collapse 한 것으로 간주, 모두 S_uncertain 으로 라우팅하여 LLM 판단에 위임 (XiYan-equivalent recall-safe fallback).
- 근거: V4-era over-smoothing 시 score 분포가 균일해져 θ_keep / θ_drop 이 무의미 (Maamari 2024).
- Default 0.05. `None` 설정 시 감지 비활성화 — anchor stack 처럼 score 분포가 분리된 정상 era 에서는 항상 정상 score-gate.
- `stats["score_collapse_detected"]` + `filter_info["filter_score_std"]` 가 결과에 기록되어 analyzer 가 era 별 collapse 빈도 측정 가능.

### 의존성
- **Selector 의 raw GAT score 가 filter 단까지 전달**되어야 함 — 별도 module session (selector Phase 2 SGBE-A) 책임.
- `gat_scores=None` 시 graceful fallback: 모든 candidates 를 S_uncertain 로 → XiYan-equivalent 동작 (LLM 1 call, recall-safe).
- FK column 추출: `metadata["fk_to_id"]` 키 (SymbolicVerifierFilter 와 동일 패턴) — 추가 의존 없음.
- PK column 추출: 우선 `metadata["primary_keys"]` 시도, 없으면 SQLite PRAGMA `table_info` 직접 조회 (best-effort).

### 한계 / Caveat
- **Score collapse era 무력화**: over-smoothing 이 심한 V4-era 결과처럼 score 분포가 균일해지면 θ_keep / θ_drop 이 무의미. 단 Step 0 (FK/PK hardcode) 와 token 감소 효과는 항상 유효 (Maamari 2024).
- **GAT score column-level calibration 전제**: anchor stack 의 score 분포가 TP/Filter✗/TN 별 분리됨을 (selector module session) 별도 진단.
- **JSON parsing 실패 fallback**: S_uncertain 전부 keep (recall-safe). a05_01 의 Unanswerable fallback recall 파괴 교훈을 따름.

### 예상 효과 (학술 Agent 정량 — Yuan 2025 분포 기반)
| Filter | LLM Input | Recall | Precision | 속도 | Backbone 민감도 |
|---|---|---|---|---|---|
| XiYan (anchor) | 전체 subgraph | 0.6761 | 0.7128 | 1× | -0.032 |
| Reflection 1iter | 전체 × 2 | 0.7320 | 0.6833 | ~0.5× | -0.035 |
| Verifier | 전체 + unit test | 0.7093 | 0.6676 | ~0.6× | -0.017 |
| **SGBE (제안)** | **S_uncertain (20-40%)** | **≥0.73** | **≥0.70** | **1.5-2×** | **~-0.015** |

### 예상 실험 (Root chain Phase 3-5, [planning/DECISIONS.md 2026-05-12 SGBE entry](/home/hyeonjin/thesis_refactored/planning/DECISIONS.md))
| Phase | 실험 ID prefix | 셀 수 | 비고 |
|-------|---------------|------|-----|
| 3 (θ calibration) | `s04_ablation/pipeline/sgbe/calib_*` | 9 (3 × 3 grid) | Step 0+1 only, LLM 없음. fast (~2-3h). |
| 4 (final SGBE) | `s04_ablation/pipeline/sgbe/final_glm` | 1 | Optimal θ × GLM 4.7 backbone (~5-9h LLM API) |
| 5 (ablation chain) | `s04_ablation/pipeline/sgbe/{step0_only,step01_only,full}` | 3 + anchor XiYan | Step contribution decomposition |

### 학술 기여
- **Filter Dominance 8번째 axis (candidate)**: "Score-Gated Hybrid 가 prune-only recall 손실의 mechanism-level cure" — 6 axis + 9-cell sweep 의 Filter-invariance 와 결합.
- **Open Question #9.4 / #9.5 직접 답변** (학술 Agent §9): prune-only recall mechanism 과 GNN selector role 재정의.
- **Layer 분리 narrative 보강**: Layer 1 (selector) score 분포가 Layer 3 (filter) routing 의 input 으로 직접 활용 — 두 Layer 간 정보 흐름의 구체적 instance.

### 산출물 (본 모듈 책임)
- [`score_gated_batch_extractive_filter.py`](score_gated_batch_extractive_filter.py) — `ScoreGatedBatchExtractiveFilter` 신규 구현. step_mode 3-mode + score_collapse_threshold 옵션 추가 (2026-05-12 follow-up).
- [`tests/test_sgbe.py`](tests/test_sgbe.py) — 16-scenario smoke test (**PASSED 16/16**). 신규: step_0 / step_0+1 / step_0+1+2 explicit / score collapse detect / collapse-disabled / invalid step_mode.
- [`/home/hyeonjin/thesis_refactored/src/prompts/filter.md`](/home/hyeonjin/thesis_refactored/src/prompts/filter.md) — `sgbe_extractive` section 추가
- [`__init__.py`](__init__.py) — registry export 추가

### 다음 단계 (selector module session 의존)
- selector EnsembleSelector / DirectGATSelector 의 raw GAT score 가 main pipeline 의 `gat_scores=...` 인자로 filter 단에 전달되도록 interface 보강 → 본 모듈은 정상 routing
- 통합 smoke test 는 selector Phase 2 완료 후 root chain Phase 3 에서

---

## FL-I. Autonomous Schema Exploration Agent (AutoLink-style)

### 동기
- AutoLink (AAAI'26, BIRD-dev strict recall 97.4%)은 **full schema를 미입력**, 필요 시 tool로 탐색 확장.
- 우리 기존 `TieredBidirectionalAgentFilter` (a05 F3)는 Tier-1/2로 **pre-filtered pool 탐색** — AutoLink와의 중간 변형.
- 제안 #6의 full autonomous variant: **Tier 제약 없이 DB 전체 스키마에 tool로 접근 허용** — precision 위험은 크나 hard query에서 상한 확장.

### 설계 요소
- **ReAct loop** (max 5 steps):
  - Thought: 현재 선택으로 쿼리 답변 가능?
  - Action: `get_neighbors(node)`, `get_fk_path(t1, t2)`, `get_all_tables(db)`, `get_column_values(col)`, `get_similar_columns_by_name(q_keyword)`
  - Observation → Thought 재진입
- **Termination condition**: "Answerable" verdict + confidence > τ, 또는 max_steps.
- **State tracking**: 방문한 노드, tool call log (reproducibility).

### 인터페이스 (기존 유지)
```python
class AutonomousAgentFilter(BaseFilter):
    def __init__(self, llm_client, tools: List[Tool], max_steps=5, db_full_schema_access=True):
        ...
    def refine(self, query, subgraph, db_id, tier2_pool=None, gat_scores=None, metadata=None, **kwargs):
        # subgraph = initial seed from PCST
        # tier2_pool = optional Tier-2
        # metadata includes DB-wide schema snapshot
        result = self.react_loop(query, subgraph, metadata)
        return {"final_nodes": ..., "status": ..., "reasoning": ...}
```

### 의존성 / 주의
- **`tools/graph_tools.py`** 확장 필요: `get_all_tables(db)`, `get_similar_columns_by_name(q)` 추가.
- **Token budget**: 쿼리당 5회 × ~1.5k tokens = 7.5k. Qwen 30B에서 latency 3~5s/query.
- **JSON parsing failure fallback**: 반드시 XiYan 결과 유지 (a05_01의 교훈).
- **Full schema access는 token 제한 쿼리에 대해 FK reachability로 선절단** (Builder B-III 의존).

### 예상 실험
| 실험 ID | 구조 | Backbone | a05 anchor |
|---------|------|----------|-----------|
| `abl_a05_13_autolink_qwen` | AutoLink full-exploration | Qwen-30B | 대비 a05_06 (F3 Tier) |
| `abl_a05_14_autolink_gpt4omini` | 동일 | GPT-4o-mini | Backbone 민감도 |
| `abl_a05_15_autolink_bounded` | Tier-2까지만 허용 (중간) | Qwen | F3와 full AutoLink 사이 |

### 예상 효과
- Hard query R +5~10%p. Easy query에서 precision 손해 가능 → F4 uncertainty gating 결합 필수.
- AutoLink 원문 수치는 EX 기반 — 우리는 R/P/F1로 재평가.

### 학술 기여
- "Graph-prior bounded autonomous agent" vs AutoLink의 cold-start full exploration — token 효율 비교 축.
- Tier 유무 ablation이 핵심 기여.

---

## FL-II. Extractive Decoder-only LLM — Filter Mode

### 동기
- 제안 #7은 LLM을 **span extractor** 로 활용 — Selector(S-IV)에서도 활용 가능하나, Filter mode에서는 **기존 PCST subgraph + Tier-2 pool** 위에서 span 단위로 선별.
- **차별점**: XiYanFilter의 "list of table.column" JSON 출력 대신, **token logit 기반 soft score** 를 뽑아 **선택 보류/강행** 판단에 활용 가능.

### 설계 요소
- Prompt: `f"Query: {q}\nCandidate schema (pre-filtered):\n{subgraph_str}\nExtract the minimal set of columns needed:"`
- **Logit extraction**: selected column의 첫 토큰 확률 → confidence.
- **Soft thresholding**: logit > τ만 최종 포함 (Filter가 자기 필터링).
- XiYan은 hard pruning, ExtractiveLLM는 soft probabilistic — 결합 가능 (stacked).

### 인터페이스
```python
class ExtractiveLLMFilter(BaseFilter):
    def __init__(self, llm_client, logit_threshold=0.5, return_logits=True):
        ...
    def refine(self, query, subgraph, db_id, **kwargs):
        prompt = build_extractive_prompt(query, subgraph)
        output, logits = self.llm.generate_with_logits(prompt)
        selected = parse_with_logit_filter(output, logits, self.logit_threshold)
        return {"final_nodes": selected, "status": "ok", "reasoning": ..., "logits": logits}
```

### 의존성 / 주의
- vLLM의 `logprobs` 옵션 활용 (현재 `APIClient`에 추가 필요).
- GPT-4o-mini는 logprobs를 full으로 제공하지 않음 — Qwen backbone 제한.
- Selector S-IV와 **LLM call 공유** 가능 (동일 prompt, 단 Selector는 score용, Filter는 final selection).

### 예상 실험
| 실험 ID | 구조 | 비고 |
|---------|------|-----|
| `abl_a05_16_xllm_filter` | ExtractiveLLMFilter 단독 | vs XiYan |
| `abl_a05_17_xllm_stacked` | XiYan → ExtractiveLLMFilter | Cascaded pruning |
| `abl_a05_18_xllm_reflection` | Reflection → ExtractiveLLM | 최고 R → 최종 soft filter |

### 검증
- Logit-based pruning이 deterministic JSON pruning 대비 우월한가.
- False positive 감소 비율 (Filter✗ false positive 집중 분석).

---

## FL-III. Symbolic-Neural Layer 3 — Verifier (★ 고우선)

### 동기
- 제안 #9 Layer 3 = **결정론적 정합성 검증기**.
- LLM Filter가 선택한 `{table.column}` 집합이 **FK graph 상 connected subgraph를 이루는지** 검증.
- 쿼리가 JOIN을 암시하는데 선택 테이블들이 FK disconnected → 명확한 오류 → **자동 복구** (추가 bridge table 삽입).
- 기존 `VerifierFilter` (a05 F2)는 NL unit test — **symbolic Verifier는 graph topology 검증** — 직교 축.

### 설계 요소
- **Verification checks**:
  1. 선택 테이블 집합이 FK 상 connected component를 이루는가?
  2. 선택 컬럼의 테이블이 모두 표함되어 있는가? (orphan column 방지)
  3. 쿼리에 `JOIN/ON` 신호 있을 때 bridge FK 노드가 포함되어 있는가?
- **Failure action**:
  - Disconnected → Builder B-III의 `fk_shortest_paths` 로 최단 bridge 삽입.
  - Missing bridge FK → FK 노드 강제 포함.
  - Orphan column → 자동 삭제 or 테이블 추가 (configurable).

### 인터페이스
```python
class SymbolicVerifierFilter(BaseFilter):
    def __init__(self, base_filter: BaseFilter, auto_repair=True):
        self.base = base_filter  # XiYan or Reflection
    def refine(self, query, subgraph, db_id, **kwargs):
        base_result = self.base.refine(query, subgraph, db_id, **kwargs)
        metadata = kwargs["metadata"]
        is_valid, issues = verify_connectivity(base_result["final_nodes"], metadata)
        if not is_valid and self.auto_repair:
            repaired = repair_by_fk_paths(base_result["final_nodes"], issues, metadata)
            return {"final_nodes": repaired, "status": "repaired", "repair_log": issues}
        return base_result
```

### 의존성 / 주의
- **Builder B-III 필수** (FK reachability, shortest paths).
- Repair가 recall 복원에 효과적이나 precision 하락 가능 — configurable `auto_repair`.
- 기존 `VerifierFilter` (NL unit test) 와 stackable.

### 예상 실험
| 실험 ID | base_filter | auto_repair | 비고 |
|---------|-------------|-------------|-----|
| `abl_a05_19_symverify_xiyan` | XiYan | True | Baseline anchor 강화 |
| `abl_a05_20_symverify_reflection` | Reflection | True | 현 최고점 강화 |
| `abl_a05_21_symverify_detect` | XiYan | False (detect only) | Error rate 측정만 |
| `abl_a05_22_symverify_stacked` | Reflection + VerifierFilter | True | 3-layer stacking 상한 |

### 검증
- Disconnected 선택 비율 (현재 Filter 출력 중 %).
- Repair 성공률 (disconnected 중 connectivity 복원).
- Recall 상승 폭 vs precision 하락 폭.

### 학술 기여
- "Deterministic graph-topology verification as post-hoc guardrail on LLM filter output."
- 논문 argument: LLM은 semantic에 강하나 topology 검증에는 약함 → symbolic Verifier가 보완.

---

## 통합 실험 로드맵 (Filter 관점)

a05 라인 연장선. **anchor는 a03_17 (Direct 최고) + abl_ens_basic_xiyan (Ensemble 최고) 양쪽**.

| Phase | 실험 | 의존 | 비고 |
|-------|------|------|-----|
| FL1 | `abl_a05_19~22 (SymVerify)` | Builder B-III | 가장 저위험, 우선 |
| FL2 | `abl_a05_13~15 (AutoLink)` | Tools 확장 | 고위험 고보상 |
| FL3 | `abl_a05_16~18 (ExtractiveLLM)` | vLLM logprobs | Selector S-IV와 LLM 공유 |

**기존 a05_03, 05~12 (진행/대기)** 와 ID 충돌 방지: 이 계획은 a05_13부터 시작.

## 변경될 파일

| 파일 | 변경 |
|------|------|
| [bidirectional_agent_filter.py](bidirectional_agent_filter.py) | AutoLink full-exploration mode 추가 |
| [symbolic_verifier_filter.py](symbolic_verifier_filter.py) | 신규 — FL-III |
| [extractive_llm_filter.py](extractive_llm_filter.py) | 신규 — FL-II |
| [score_gated_batch_extractive_filter.py](score_gated_batch_extractive_filter.py) | 신규 — SGBE (완료 2026-05-12) |
| [tests/test_sgbe.py](tests/test_sgbe.py) | 신규 — SGBE 16-scenario smoke (PASSED) |
| [rsl_backward_filter.py](rsl_backward_filter.py) | 신규 — RSL Backward (완료 2026-05-13) |
| [tests/test_rsl_backward.py](tests/test_rsl_backward.py) | 신규 — RSL Backward 15-scenario smoke (PASSED) |
| [tools/graph_tools.py](tools/graph_tools.py) | `get_all_tables`, `get_similar_columns_by_name` 등 추가 |
| `src/prompts/filter.md` | `sgbe_extractive` section 추가 (완료) |
| `src/llm_client/api_handler.py` | vLLM `logprobs` 지원 |
| `configs/experiments/abl/a05_filter_agentic/` | a05_13 ~ a05_22 yaml |
| `configs/experiments/s04_ablation/pipeline/sgbe/` | SGBE θ calibration + final + ablation (root chain Phase 3-5) |

## 인터페이스 계약 (유지)
- `refine(query, subgraph, db_id, tier2_pool=None, gat_scores=None, metadata=None, **kwargs)` → `Dict`
- 신규 필터도 동일 signature 준수.
- JSON parsing 실패 fallback은 **XiYan 결과 유지**, 절대 Unanswerable 아님.

## 검증 방법 (모듈 내)
- **R/P/F1 (4소수점)** vs a03_17 anchor (F1=0.6940), a05_02 (F1=0.7068).
- **Connectivity check pass rate**: SymVerify에서만 측정.
- **Token cost**: AutoLink의 쿼리당 평균 tool call 수, 총 tokens.
- **Reasoning quality**: 5~10 케이스 수동 review.

## a05 기존 결과와 묶어 보기
- a05_02 (Reflection) = 현 최고. FL-III Repair 결합으로 상한 재탐색.
- a05_01 (AdaptiveMultiAgent 실패, −22.3%p) = JSON parsing 실패 원인 — 신규 필터 설계 시 반드시 parsing robust하게.
- a05_04 (Verifier, −0.6%p) = NL unit test 한계 — FL-III symbolic Verifier와 stacked로 보완 기대.
