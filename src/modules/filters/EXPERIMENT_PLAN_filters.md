# Filter 모듈 실험 계획 — 9 아키텍처 제안 중 Filter 관련 축

> **⚑ 먼저 루트 계획을 읽을 것**: [/home/hyeonjin/thesis_refactored/EXPERIMENT_PLAN.md](/home/hyeonjin/thesis_refactored/EXPERIMENT_PLAN.md) — 전 모듈 통합 로드맵, Cross-Module Dependency, 통합 실험(int_01~08), 우선순위 Phase A~E, 논문 매핑이 거기에 있다. **루트 PLAN은 수정하지 않는다** — 수정이 필요하면 루트 세션에 요청.
> **이 파일의 역할**: 루트 PLAN에서 Filter에 할당된 3축(FL-I/FL-II/FL-III)의 **모듈 내부 구현 상세**만 담는다.
>
> **현재 진입점**: `XiYanFilter` (anchor, F1=0.6940 on a03_17), `ReflectionFilter` (F1=0.7068 신기록 on a05_02).
> **이미 존재하는 a05 agentic 라인**과 통합하여 중복 실험을 피한다.
> **선결 의존성**: FL-III는 Builder B-III의 FK reachability matrix에 의존. 루트 PLAN Phase A 완료 전에는 FL-III 블록.
> **연관 계획**: [/home/hyeonjin/.claude/plans/vivid-sprouting-sunbeam.md](/home/hyeonjin/.claude/plans/vivid-sprouting-sunbeam.md) (F1~F5, a05 시리즈).

---

## 이 모듈이 받아야 할 3가지 제안

| # | 이름 | Filter의 역할 | 우선순위 | a05와의 관계 |
|---|------|---------------|---------|-------------|
| FL-I | **Autonomous Schema Exploration Agent (AutoLink-style)** (원안 #6) | Iterative ReAct agent + graph-native tools로 탐색적 refinement | 상 | a05 F3 (Tiered Bidirectional) 의 확장 — full exploration variant |
| FL-II | **Extractive Decoder-only LLM — Filter mode** (원안 #7) | LLM span extraction + logit score로 token-level pruning | 중 | 새 축 (a05 외) |
| FL-III | **Symbolic-Neural Layer 3 — Verifier** (원안 #9 Layer 3) | FK graph 상 connectivity 검증 + disconnected 결과 reject | 상 | a05 F2 (VerifierFilter) 의 강화형 |

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
| [tools/graph_tools.py](tools/graph_tools.py) | `get_all_tables`, `get_similar_columns_by_name` 등 추가 |
| `src/llm_client/api_handler.py` | vLLM `logprobs` 지원 |
| `configs/experiments/abl/a05_filter_agentic/` | a05_13 ~ a05_22 yaml |

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
