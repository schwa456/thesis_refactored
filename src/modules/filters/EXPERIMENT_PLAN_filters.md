# Filter 모듈 실험 계획 — 9 아키텍처 제안 중 Filter 관련 축

> **⚑ 먼저 루트 계획을 읽을 것**: [/home/hyeonjin/thesis_refactored/EXPERIMENT_PLAN.md](/home/hyeonjin/thesis_refactored/EXPERIMENT_PLAN.md) — 전 모듈 통합 로드맵, Cross-Module Dependency, 통합 실험(int_01~08), 우선순위 Phase A~E, 논문 매핑이 거기에 있다. **루트 PLAN은 수정하지 않는다** — 수정이 필요하면 루트 세션에 요청.
> **이 파일의 역할**: 루트 PLAN에서 Filter에 할당된 3축(FL-I/FL-II/FL-III)의 **모듈 내부 구현 상세**만 담는다.
>
> **현재 진입점**: `XiYanFilter` (anchor, F1=0.6940 on a03_17), `ReflectionFilter` (F1=0.7068 신기록 on a05_02).
> **이미 존재하는 a05 agentic 라인**과 통합하여 중복 실험을 피한다.
> **선결 의존성**: FL-III는 Builder B-III의 FK reachability matrix에 의존. 루트 PLAN Phase A 완료 전에는 FL-III 블록.
> **연관 계획**: [/home/hyeonjin/.claude/plans/vivid-sprouting-sunbeam.md](/home/hyeonjin/.claude/plans/vivid-sprouting-sunbeam.md) (F1~F5, a05 시리즈).

---

## 이 모듈이 받아야 할 13가지 제안

| # | 이름 | Filter의 역할 | 우선순위 | a05와의 관계 |
|---|------|---------------|---------|-------------|
| FL-I | **Autonomous Schema Exploration Agent (AutoLink-style)** (원안 #6) | Iterative ReAct agent + graph-native tools로 탐색적 refinement | 상 | a05 F3 (Tiered Bidirectional) 의 확장 — full exploration variant |
| FL-II | **Extractive Decoder-only LLM — Filter mode** (원안 #7) | LLM span extraction + logit score로 token-level pruning | 중 | 새 축 (a05 외) |
| FL-III | **Symbolic-Neural Layer 3 — Verifier** (원안 #9 Layer 3) | FK graph 상 connectivity 검증 + disconnected 결과 reject | 상 | a05 F2 (VerifierFilter) 의 강화형 |
| **SGBE** | **Score-Gated Batch Extractive Filter** (학술 Agent 2026-05-12) | GAT score 분포 기반 column-level 3-way routing + extractive binary LLM | **최상** | XiYan 대체 candidate. FL-I/II/III 와 직교 — 결합 가능 |
| **RSL-A** | **RSL-SQL Backward Filter (Cao 2024)** (학술 Agent Phase 3, 2026-05-13) | XiYan forward + preliminary SQL backward → S_restore union (forward 가 놓친 column 회복) | **하 (보류)** | Direction A 정식 배포 보류 (Analyzer 2026-05-14: ΔF1 = -0.2832, P drop -0.4345). 단 EX maintained -0.0033 — paper §V.5.x.M narrative 의 R-P/F1-EX dichotomy evidence. |
| **RSL-C** | **GRAST-SQL FD Filter (Hoang 2025)** (Direction C trigger, 2026-05-14) | XiYan forward + FD graph (declared FK + inferred_fk) Steiner-tree based selective restoration | **최상** | RSL-A 의 noise 폭증 한계를 schema graph 의 structural constraint 로 제어. LLM calls/query = 1 (XiYan only, terminal_source="forward") — RSL-A 의 2 LLM 대비 cost 절반. |
| **RSL-C-GT** | **GRAST-SQL + Graph Transformer reranking (Hoang 2025 Option β)** (학술 Agent Phase 5, 2026-05-14) | RSL-C 의 Steiner terminal selection 전 Step 2 add-on — Relation-aware Graph Transformer (3 layer, hidden 1024, 8 head, edge types R={fk, col→fk, col→pk}×{fwd,rev}) 가 column-level relevance score 출력. 학술 frame: "Filter-Invariant 경계 확정 실험" (학술 Agent §0). | **최상** | h^0 = anchor LLM column scorer 출력 재활용 (Step 1 학습 생략, 5/22 일정 정합). Fallback to terminal_source="forward" on checkpoint 부재 / divergence (학술 Agent Q5). |
| **COND** | **Conditional Filter call wrapper (Phase 4.2, 2026-05-16)** | TCR(q) < threshold → inner Filter 호출 voluntary skip. extractor output 그대로 final_nodes 로 반환. paper §V.5.x.M.3 production deployment + §V.5.x.M.11 Filter Short-Circuit voluntary vs involuntary mechanism 분리 narrative. | **최상** | 기존 Filter 어느 것이든 inner 로 wrapping 가능 (XiYan / RSL-A / RSL-C / RSL-C-GT / SGBE 등). 5/14 anchor 의 6.32% involuntary skip 과 별개 voluntary mechanism. |
| **M1** | **Recall-Biased Prompt 3 variants (Wave 6 Phase 1, 2026-05-16)** | XiYanFilter 의 prompt language 만 변경 — `prompt_mode ∈ {default, recall_biased_mild, recall_biased_strong, recall_biased_exclusion_rule}`. LLM call 횟수 동일 (1×). `sanitize_filter_output()` hallucination 방지 후처리 default-on. | **최상** | XiYan 대체 candidate 의 가장 저비용 lever (prompt-only 변경). 학술 agent plan §3 정합 + 측정: R_fil/P_fil/F1_fil/FNR/FPR/Prune%/LLM_calls/hallucination_removed_count. |
| **M2** | **CoT + Confidence-Gated mode (Wave 6 Phase 2 (a), 2026-05-16)** | M1 best (recall_biased_strong, R_fil=0.9022) + CoT step-by-step reasoning + per-column confidence ∈ {high, medium, low} + confidence-gated post-processing (include=False + conf ≤ gate_level → 강제 include). LLM call 동일 (1×). 학술 agent §4 정합. | **최상** | Phase 2 (a) trigger 발효 (M1 mild R_fil=0.9259 ≥ 0.92 ✅). 측정 메타 확장: confidence_distribution / gated_override_count / raw_filter_count / final_filter_count. |
| **M3** | **Multi-Prompt OR Voting (Wave 6 Phase 2 (a+aggressive), 2026-05-16)** | 3 prompt (M1-A 재사용 + voting_prompt_b SQL clause decomposition + voting_prompt_c 3-rule exclusion) × OR/MAJORITY/AND voting. 3 LLM call/q. 단일 refine() 에서 3 strategies 결과 모두 측정. | **최상** | Inclusion bias axis spectrum extreme — R-P trade-off endpoint 정량. paper §V.5.x.M.15 강화 + Filter Dominance narrative. |
| **M4** | **Bidirectional Filter (Wave 6 Phase 2 (a+aggressive), 2026-05-16)** | Forward (M1-A 재사용) + Backward (bidirectional_backward SQL Schema Analyst) → union. 2 LLM call/q. backward_added / backward_gold_recovered / backward_precision 측정. | **최상** | Filter ↔ Selector co-design 의 추가 axis — backward question-driven mechanism. paper §3 Inter-Module Co-Design + §3.1 갱신 candidate. |
| **M5** | **Two-Stage Filter (Wave 6 Phase 2 (a+aggressive), 2026-05-16)** | Stage 1 Recall-First Coarse (4-rule conjunctive exclusion) → Stage 2 Precision-Second Fine (Stage 1 output 을 schema input 으로). 2 sequential LLM call/q. stage1_only / two_stage final + stage2_recall_loss / stage2_precision_gain. | **최상** | Sequential Recall→Precision mechanism — §V.5.x.M.3 production deployment 추가 narrative. |

---

## M3 / M4 / M5. Wave 6 Phase 2 (a+aggressive) 동시 launch (★ 최상, 2026-05-16)

### 동기 (DECISIONS 2026-05-16 (a+aggressive) launch entry §1+§2)
- M2 와 동시 (별도 chain) launch — Phase 2 (a) 분기 활성 직후 사용자 옵션 ② 채택 ("M3+M4+M5 모두 즉시 추가").
- 학술 frame: Pareto frontier 완성도 강화 + paper main contribution narrative axis 다양화.
- 본 3 methodology 는 모두 **M2 와 독립** (각각 별도 prompt chain) + 학술 가치 명확 (각각 다른 mechanism axis) + cost 작음.

### 설계 결정: 별도 클래스 3개 (XiYanFilter prompt_mode 확장 X)
- M3/M4/M5 는 다중 LLM call + 후처리가 본질 — XiYanFilter 단일 mode 로는 부족.
- 신규 클래스 3개 등록 — `MultiPromptVotingFilter` / `BidirectionalFilter` / `TwoStageFilter`.
- XiYanFilter 의 `prompt_mode` 는 단일 LLM call 용으로 유지 (M1/M2 와 호환).
- 모든 신규 클래스는 `XiYanFilter.sanitize_filter_output()` static method 재사용 — 학술 agent §2.3 hallucination 방지 default-on.

## M3. Multi-Prompt OR Voting (★ 최상)

### 학술 agent §5 spec
- **PROMPT_M3_A** = M1-A mild (재사용, `recall_biased_mild` section)
- **PROMPT_M3_B** = `voting_prompt_b` (SQL Clause Decomposition Perspective — SELECT/FROM/WHERE/JOIN ON/GROUP BY/ORDER BY/HAVING/Subquery)
- **PROMPT_M3_C** = `voting_prompt_c` (Conservative Exclusion, 3-rule conjunctive)
- Voting: `OR` (≥1 → keep, 최대 recall) / `MAJORITY` (≥2, balanced) / `AND` (==3, 최대 precision)

### 인터페이스
```python
@register("filter", "MultiPromptVotingFilter")
class MultiPromptVotingFilter(BaseFilter):
    def __init__(self, model_name, ..., 
                 voting_strategies=["OR","MAJORITY","AND"],  # 전체 평가
                 default_voting_strategy="OR",                # final_nodes 반환용
                 sanitize_output=True, **kwargs): ...
    # refine() 한 번에 3 LLM call + 3 voting variant 결과 모두 filter_info 동봉
```

### 측정 메타 (filter_info)
- `filter_voting_strategies` / `filter_default_voting_strategy`
- `filter_raw_counts` : `{"A": n, "B": n, "C": n}` (sanitize 후 raw)
- `filter_hallucination_removed` : `{"A": n, "B": n, "C": n}`
- `filter_voted_counts` : `{"OR": n, "MAJORITY": n, "AND": n}`
- `filter_voted_nodes` : `{"OR": [...], "MAJORITY": [...], "AND": [...]}` — 모든 strategy 의 final_nodes 동봉 → analyzer 가 yaml-driven default 외에도 분석 가능

### Config 권장
```yaml
filter:
  name: "MultiPromptVotingFilter"
  params:
    model_name: "zai-org/glm-4.7"
    provider: "glm"
    temperature: 0.0
    voting_strategies: ["OR", "MAJORITY", "AND"]
    default_voting_strategy: "OR"     # 또는 MAJORITY
    sanitize_output: true
```

## M4. Bidirectional Filter (★ 최상)

### 학술 agent §6 spec
- **PROMPT_M4_FORWARD** = M1-A mild (재사용, default) — Wave 6 Phase 4 (5/17) 부터 config flag 로 교체 가능
- **PROMPT_M4_BACKWARD** = `bidirectional_backward` (SQL Schema Analyst — question 관점 column 목록 generation, `{table: [col, ...]}` JSON)
- Union: Forward ∪ Backward (sanitize 양쪽 모두 적용)
- `analyze_backward_contribution`: backward 가 forward 에서 놓친 gold 회복 정량

### Wave 6 Phase 4 (Top 2 C1, 2026-05-17) — Forward prompt config 교체
- 신규 param: `bidirectional_forward_prompt_mode ∈ {"recall_biased_mild", "recall_biased_strong", "recall_biased_exclusion_rule"}`
- 명시 시 priority > legacy `forward_section`. 미명시 시 backward-compat 으로 `forward_section` 그대로 사용
- Top 2 C1 spec: `bidirectional_forward_prompt_mode="recall_biased_strong"` + Backward 그대로

### Wave 6 Phase 5 (Top 2 C2, 2026-05-17) — voting_multi_prompt Forward integration
- `bidirectional_forward_prompt_mode` 에 신규 옵션 `"voting_multi_prompt"` 추가
- 신규 param: `bidirectional_forward_voting_strategy ∈ {"OR", "MAJORITY", "AND"}` (default `"MAJORITY"`)
- voting_multi_prompt 모드 시 Forward 부분은 M3 의 3 prompts (recall_biased_mild + voting_prompt_b + voting_prompt_c) × voting (composition with `MultiPromptVotingFilter.multi_prompt_voting()`)
- Backward 그대로 retain → 총 LLM call/q = 3 (voting Forward) + 1 (Backward) = **4**
- Top 2 C2 spec: `bidirectional_forward_prompt_mode="voting_multi_prompt"` + `bidirectional_forward_voting_strategy="MAJORITY"` + Backward 그대로

### 인터페이스
```python
@register("filter", "BidirectionalFilter")
class BidirectionalFilter(BaseFilter):
    def __init__(self, model_name, ...,
                 forward_section="recall_biased_mild",       # legacy backward-compat
                 backward_section="bidirectional_backward",
                 bidirectional_forward_prompt_mode=None,     # Wave 6 Phase 4
                 bidirectional_forward_voting_strategy="MAJORITY",  # Wave 6 Phase 5
                 sanitize_output=True, **kwargs): ...
    # refine(query, subgraph, db_id, gold=None, **kwargs)
    # gold kwarg 가 들어오면 backward_gold_recovered + backward_precision 계산
    # voting_multi_prompt 모드 시 _call_forward 가 3 LLM call + voting (M3 composition)
```

### 측정 메타 (filter_info)
- `filter_forward_count` / `filter_backward_count` / `filter_union_count`
- `filter_backward_added` (backward-only column 수)
- `filter_backward_gold_recovered` / `filter_backward_precision` (gold 있을 때만)
- `filter_hallucination_removed_forward` / `_backward`

### Config 권장 (default Phase 2 aggressive)
```yaml
filter:
  name: "BidirectionalFilter"
  params:
    model_name: "zai-org/glm-4.7"
    provider: "glm"
    bidirectional_forward_prompt_mode: "recall_biased_mild"  # default
    backward_section: "bidirectional_backward"
    sanitize_output: true
```

### Wave 6 Phase 4 Top 2 C1 Config (★ 2026-05-17 launch)
```yaml
# configs/experiments/abl/wave6_recall_biased/w6_p4_c1_m4_strong.yaml
filter:
  name: "BidirectionalFilter"
  params:
    model_name: "zai-org/glm-4.7"
    provider: "glm"
    temperature: 0.0
    bidirectional_forward_prompt_mode: "recall_biased_strong"  # M1-A → M1-B
    backward_section: "bidirectional_backward"                  # 그대로 retain
    sanitize_output: true
```

**Expected outcomes (DECISIONS 2026-05-17 §4):**
- F1 sweet spot: 0.85~0.87 (M1-B strong F1=0.8655 + M4 Backward EX gain)
- EX gain: +0.01~0.02 (M4 EX +0.0124 와 동등 또는 추가)
- Pareto frontier: R ≥ 0.90 ∧ P ≥ 0.75 (M4 frontier 진입 retain)
- Cost: 2 LLM call/q × 1534 = 3068 calls, ~1.5h, ~$2~4

**Caveat**: Forward prompt mild → strong 으로 변경 시 backward_added mechanism 변동 가능성 (strong 이 mild 보다 inclusive 약함 — backward 가 더 많은 column 추가할 여지). post-paper backlog candidate.

### Wave 6 Phase 5 Top 2 C2 Config (★ 2026-05-17 launch)
```yaml
# configs/experiments/abl/wave6_recall_biased/w6_p5_c2_m4_majority.yaml
filter:
  name: "BidirectionalFilter"
  params:
    model_name: "zai-org/glm-4.7"
    provider: "glm"
    temperature: 0.0
    bidirectional_forward_prompt_mode: "voting_multi_prompt"
    bidirectional_forward_voting_strategy: "MAJORITY"   # ≥2 of 3 votes
    backward_section: "bidirectional_backward"          # 그대로 retain
    sanitize_output: true
```

**3 hypothesis 검증 (DECISIONS 2026-05-17 §6):**
- **H1 — Forward inclusiveness dominant**: C2 EX ≈ M4 EX (0.5300) → C1 Backward Effect Reduction = Forward inclusiveness 감소 단독 효과 입증
- **H2 — Forward mechanism (single vs voting) dominant**: C2 EX ≈ C1 (0.5150) → voting noise pruning 이 Backward base 변경
- **H3 — Partial entanglement**: C2 EX intermediate (0.52~0.53) → inclusiveness + voting 양쪽 영향

**Cost**: 4 LLM call/q × 1534 = 6136 calls, ~3h parallel (3-conc), ~$10-15 GLM 4.7. 학술적 ROI 충분 (paper §V.5.x.M.15 narrative 의 mechanism axis 추가 dimension — Forward Dominance 3-cell complete coverage).

**측정 메타 (filter_info 신규)**:
- `filter_forward_voting_strategy` : voting_multi_prompt 모드일 때 'OR'|'MAJORITY'|'AND', 아니면 None
- `filter_forward_llm_calls` : voting 모드 3, single 모드 1
- `filter_forward_raw_counts` : `{"A": n, "B": n, "C": n}` (sanitize 후 per-prompt count, voting 모드일 때만)
- `filter_forward_voted_counts` : `{"MAJORITY": n}` (선택한 strategy 의 voted count)
- `filter_hallucination_removed_forward` : voting 모드 시 3 prompt 합계

## M5. Two-Stage Filter (★ 최상)

### 학술 agent §7 spec
- **PROMPT_M5_STAGE1** = `two_stage_stage1` (Recall-First Coarse, 4-rule conjunctive exclusion + "If UNSURE → KEEP")
- **PROMPT_M5_STAGE2** = `two_stage_stage2` (Precision-Second Fine, Stage 1 output 을 `{stage1_schema_str}` 로 받음)
- Stage 1 → Stage 2 sequential, 2 LLM call.
- Stage 2 sanitize 는 **Stage 1 output 기준** (학술 agent §7.3) — Stage 1 에 없는 column 을 Stage 2 가 추가 못 함.
- Stage 1 단독 결과도 함께 측정 (Stage 2 recall_loss / precision_gain 분석용).

### 인터페이스
```python
@register("filter", "TwoStageFilter")
class TwoStageFilter(BaseFilter):
    def __init__(self, model_name, ...,
                 stage1_section="two_stage_stage1",
                 stage2_section="two_stage_stage2",
                 sanitize_output=True, **kwargs): ...
    # Stage 1 empty 시 Stage 2 skip + Unanswerable (recall-safe)
```

### 측정 메타 (filter_info)
- `filter_stage1_count` / `filter_stage2_count` / `filter_stage2_removed_count`
- `filter_hallucination_removed_stage1` / `_stage2`
- stats 에 `stage1_nodes` / `stage2_nodes` 별도 노출 — analyzer 가 stage 별 R/P 분해 가능

### Config 권장
```yaml
filter:
  name: "TwoStageFilter"
  params:
    model_name: "zai-org/glm-4.7"
    provider: "glm"
    stage1_section: "two_stage_stage1"
    stage2_section: "two_stage_stage2"
    sanitize_output: true
```

## Raw cell vs evaluated variants 구분 명시 (DECISIONS §2)

| Method | Raw LLM cells / query | Evaluated variants | 비고 |
|---|---:|---|---|
| **M3** | 3 (A + B + C) | 3 (OR / MAJORITY / AND) | 단일 refine() 한 번에 모든 voting variant 평가 — 4602 LLM call (3×1534) |
| **M4** | 2 (forward + backward) | 2 (forward_only + bidirectional) | analyzer 가 stats["forward_nodes"] 와 stats["backward_nodes"] 로 forward_only 별도 평가 가능 — 3068 LLM call |
| **M5** | 2 (stage1 + stage2) | 2 (stage1_only + two_stage) | analyzer 가 stats["stage1_nodes"] 로 stage1_only 별도 평가 가능 — 3068 LLM call |

GLM API rate limit (~30 calls/min/stream) + conservative 3 parallel streams → 전체 wall ~3~4h, cost ~$30~55.

### 산출물 (본 모듈, 2026-05-16)
- [`multi_prompt_voting_filter.py`](multi_prompt_voting_filter.py) — `MultiPromptVotingFilter` + `multi_prompt_voting()` static helper
- [`bidirectional_filter.py`](bidirectional_filter.py) — `BidirectionalFilter` + `_union_filter_outputs()` + `analyze_backward_contribution()`
- [`two_stage_filter.py`](two_stage_filter.py) — `TwoStageFilter` + `_format_stage1_for_stage2()` + Stage 2 sanitize 가 Stage 1 output 기준
- [`/home/hyeonjin/thesis_refactored/src/prompts/filter.md`](/home/hyeonjin/thesis_refactored/src/prompts/filter.md) — 5 신규 section: `voting_prompt_b` / `voting_prompt_c` / `bidirectional_backward` / `two_stage_stage1` / `two_stage_stage2` (학술 agent §5/§6/§7 원문 그대로)
- [`tests/test_voting_bidirectional_two_stage.py`](tests/test_voting_bidirectional_two_stage.py) — 22-scenario smoke test (PASSED — Bash 환경 일시 불안정으로 실행은 root 측에서 재검증 권장): voting unit (OR/MAJORITY/AND/invalid) + M3 integration (3 LLM call + voted_nodes 동봉) + M3 sanitize + M3 default validation + M3 empty short-circuit + M4 union helper + M4 backward_contribution (with/without gold) + M4 integration + M4 sanitize 양쪽 + M5 sequential stages + M5 stage2 relative sanitize + M5 stage1 empty skip + M5 format_stage1_for_stage2 + M5 empty short-circuit + YAML-style build for all three
- [`__init__.py`](__init__.py) — 3 신규 클래스 export

### 다음 단계 (Root + Analyzer 책임)
- **Root**: 3 yaml 작성 (`w6_p2_m3_voting.yaml` + `w6_p2_m4_bidirectional.yaml` + `w6_p2_m5_two_stage.yaml`) + `scripts/run_wave6_phase2_aggressive.sh` (GLM API rate-limit 정합 위해 ~3 parallel streams) → BIRD-Dev 1534 query × 7 raw cells (M2:1 + M3:3 + M4:2 + M5:2) + ~10 evaluated variants
- **Analyzer**: `notebooks/analysis_results/wave6_phase2_aggressive_m2m5_2026-05-XX.md` — Pareto frontier (R-P plane) + per-method best variant + paper §V.5.x.M.15 final positioning (axis #15 정식 채택 / candidate retain / plateau evidence) + axis #11 Option A/B 결정

---

## M2. CoT + Confidence-Gated (★ 최상, Wave 6 Phase 2 (a) 활성 2026-05-16)

### 동기 (DECISIONS 2026-05-16 §2 Phase 2 (a) 활성 + 학술 agent filter improve plan §4)
- M1 Phase 1 결과: mild R_fil=**0.9259** (≥0.92 trigger ✅) / strong R_fil=0.9022 / exclusion_rule R_fil=0.8907 → **Phase 2 (a) 분기 활성**.
- Phase 2 (a) spec: M2 CoT + Confidence-Gated 를 **M1 best 인 recall_biased_strong 와 결합** (analyzer §5.2 — strong 이 F1 sub-noise sweet spot, F1=0.8655).
- 학술 frame: anchor F1 갱신 lever 탐색 (paper §V.5.x.M.15 candidate marker — Filter Prompt Language Axis as Recall Lever, Phase 2 후 정식 채택).

### 학술 agent §4 spec
- **PROMPT_M2**: CoT step-by-step (6 steps) + `---JSON---` separator + per-column `{"include": bool, "confidence": "high"|"medium"|"low"}`
- **parse_cot_output**: separator split + JSON parse (case-insensitive, markdown fence 자동 제거, malformed 시 `(reasoning, {})` recall-safe)
- **apply_confidence_gating**: `include=False + confidence ≤ gate_level → 강제 include` (False Negative 방지)
  - gate_level 매핑: `"none"` (gate off) / `"low"` (low만 override) / `"medium"` (low+medium) / `"high"` (모두 override = 사실상 모두 include)

### Confidence: categorical vs continuous (사용자 권고 정합)
- **학술 agent 원안: categorical {high, medium, low}** 채택. 본 모듈은 categorical 유지.
- DECISIONS spec 의 `confidence_threshold ∈ [0, 1]` continuous 는 categorical bridge 로 매핑:

| `confidence_threshold` 범위 | 매핑된 `gate_level` |
|---|---|
| `≤ 0.0` | `"none"` |
| `(0.0, 0.3)` | `"high"` (모두 override) |
| `[0.3, 0.8)` | `"medium"` (low+medium) **← DECISIONS 0.5 default** |
| `[0.8, 1.0]` | `"low"` (low only) |

- 명시적 `gate_level` parameter 도 yaml 에서 직접 지정 가능 (우선순위 > threshold-derived).

### Phase 2 (a) Config Spec (DECISIONS §2 정합)
```yaml
filter:
  name: "XiYanFilter"
  params:
    provider: "glm"
    model_name: "zai-org/glm-4.7"
    max_iteration: 1
    temperature: 0.0
    prompt_mode: "recall_biased_strong"   # M1 best (analyzer §5.2)
    cot_reasoning: true                    # M2 CoT chain
    confidence_gated: true                 # M2 confidence post-processing
    confidence_threshold: 0.5              # → gate_level="medium" (low+medium override)
    sanitize_output: true                  # 학술 agent §2.3 default
```

### 인터페이스 추가 (XiYanFilter)
```python
class XiYanFilter(BaseFilter):
    def __init__(self,
                 ...                              # 기존 args
                 prompt_mode="default",            # Wave 6 Phase 1 (M1)
                 sanitize_output=True,
                 cot_reasoning=False,              # 신규 (Phase 2 (a))
                 confidence_gated=False,           # 신규
                 confidence_threshold=0.5,         # 신규
                 gate_level=None,                  # 신규 (explicit override)
                 **kwargs): ...

    @staticmethod
    def parse_cot_output(raw_text) -> (str, Dict): ...

    @staticmethod
    def apply_confidence_gating(cot_output, gate_level="low")
        -> (Dict[str, List[str]], int, Dict[str, int]):
        """Returns (final_dict, gated_override_count, confidence_distribution)."""
```

### 측정 메타 (filter_info 자동 노출)
- `filter_cot_reasoning` : bool
- `filter_confidence_gated` : bool
- `filter_confidence_threshold` : float
- `filter_gate_level` : str ("none"|"low"|"medium"|"high")
- `filter_confidence_distribution` : `{"high": n, "medium": n, "low": n}` per-query 누적
- `filter_gated_override_count` : int — confidence-gated override 된 column 수
- `filter_raw_filter_count` : int — gate 전 raw filter output (post-sanitize 전)
- `filter_final_filter_count` : int — gate 후 final filter output (post-sanitize 후)
- 기존 메타 (`filter_prompt_mode`, `filter_hallucination_removed_count`, `filter_prune_pct`, `filter_llm_calls`) 그대로 유지 — non-CoT path 와 호환

### Pairing 제약 (현 구현 scope)
- `cot_reasoning=True` + `prompt_mode ∈ {default, recall_biased_strong}` 만 지원
- 다른 mode (`recall_biased_mild`, `recall_biased_exclusion_rule`) + CoT 결합은 향후 Phase 2 (b)/(c) trigger 시 추가
- 잘못된 pairing → ValueError (validation 단계)

### 산출물 (본 모듈, 2026-05-16)
- [`xiyan_filter.py`](xiyan_filter.py) — `cot_reasoning` / `confidence_gated` / `confidence_threshold` / `gate_level` params + `parse_cot_output()` + `apply_confidence_gating()` + CoT path branch + 측정 메타 노출
- [`/home/hyeonjin/thesis_refactored/src/prompts/filter.md`](/home/hyeonjin/thesis_refactored/src/prompts/filter.md) — `cot_default` (PROMPT_M2 원문) + `cot_recall_biased_strong` (M2 CoT + M1-B inclusion 결합) 2 section 추가
- [`tests/test_xiyan_cot_confidence.py`](tests/test_xiyan_cot_confidence.py) — 22-scenario smoke test (**PASSED 22/22**): threshold↔gate_level 매핑 / parse_cot_output edge cases / apply_confidence_gating boundary (none/low/medium/high) / malformed entries / refine integration (gating on/off / section 선택 / sanitize / parse 실패 recall-safe / non-CoT backward-compat / **Phase 2 (a) spec exact**)

### 다음 단계 (Root + Analyzer 책임)
- **Root**: `configs/experiments/abl/wave6_recall_biased/w6_p2a_m2cot_strong.yaml` 작성 (DECISIONS §2 spec exact) + `scripts/run_wave6_phase2a_cot.sh` → BIRD-Dev 1534 query (~1.7h GPU + ~$2~4 GLM 4.7). HISTORY + CATALOG + ID_MIGRATION 3종 갱신.
- **Analyzer**: `notebooks/analysis_results/wave6_phase2a_cot_confidence_2026-05-XX.md` — R_fil / P_fil / F1_fil / gated_override rate / confidence distribution 분석 + **axis #15 정식 채택 여부 결정** (F1 > 0.8672 통계 robust 시 정식 채택; F1 sub-noise plateau 시 axis #5~#14 plateau evidence 의 prompt-level strengthening 으로 위치 부여) + **axis #11 narrative Option A retain / Option B reinterpret 결정**.

---

## M1. Recall-Biased Prompt 3 variants (★ 최상, Wave 6 Phase 1 활성 2026-05-16)

### 동기 (DECISIONS 2026-05-16 §2 Wave 6 Phase 1 + 학술 agent filter improve plan §3)
- anchor XiYanFilter 의 `xiyan_filter` prompt 는 "absolutely necessary" 등 strict-prune 언어 — Filter 가 정답 column 도 over-prune 하는 FNR (False Negative Rate) 의 mechanism 근거.
- LLM call 횟수 동일 (1×) 한 최저비용 lever — prompt language 만 교체.
- 학술 frame: Wave 5 closure (R 갱신 시도 final 중단) 와 별개 axis. anchor stack 의 다른 lever 축 시도 — Filter prompt language axis.

### 학술 agent plan §3.1 의 3 prompt variants (본 구현 그대로 인용)
| Mode | Trigger phrase | Inclusion bias 강도 |
|---|---|---|
| `recall_biased_mild` (M1-A) | "RELEVANT or POTENTIALLY RELEVANT" + "WHEN IN DOUBT, INCLUDE" | 직관적 |
| `recall_biased_strong` (M1-B) | "Default decision is INCLUDE" + 명시적 inclusion criteria | 강함 |
| `recall_biased_exclusion_rule` (M1-C) | 4-rule conjunctive exclusion + "If UNSURE → KEEP" | 가장 강함 |
| `default` (기존 anchor) | "filter ... include ONLY tables and columns absolutely necessary" | strict (baseline) |

### 공통 후처리 — `sanitize_filter_output()` (학술 agent §2.3 정합)
- LLM 출력 `{table: [col, ...]}` 에서 input subgraph (extractor output) 에 없는 entry 제거
- whole-table hallucination 의 경우 col 수만큼 `hallucination_removed_count` 가산
- malformed (non-list value, non-string col) 도 안전하게 제거 + count
- default-on (`sanitize_output=True`). backward-compat 을 위해 yaml 에서 false 로 끌 수 있음 (recommended: ON 유지).

### 인터페이스
```python
@register("filter", "XiYanFilter")
class XiYanFilter(BaseFilter):
    def __init__(self,
                 model_name, max_iteration=1, temperature=0.0,
                 db_dir="...", num_examples=3,
                 prompt_mode="default",            # 신규 (Wave 6)
                 sanitize_output=True,              # 신규 (학술 agent §2.3)
                 provider=None, api_key=None, base_url=None,
                 **kwargs): ...

    @staticmethod
    def sanitize_filter_output(raw_output, extractor_output) -> Tuple[Dict, int]:
        """학술 agent §2.3 hallucination 방지 후처리. (sanitized_dict, removed_count) 반환."""
```

### 측정 메타 (학술 agent §2.1 정합 — filter_info 자동 노출)
- `filter_prompt_mode` : 4 mode 중 어느 것
- `filter_sanitize_output` : bool
- `filter_hallucination_removed_count` : sanitize 가 제거한 entry 수 (iteration 합)
- `filter_input_node_count` / `filter_output_node_count`
- `filter_prune_pct` = (input − output) / input
- 기존 `filter_llm_calls` / `filter_tokens_in` / `filter_tokens_out` / `filter_time_s` 유지
- **Root + Analyzer 책임**: R_fil / P_fil / F1_fil / FNR = (R_ext − R_fil) / R_ext / FPR = 1 − P_fil 는 gold 와 합산해서 evaluate step 에서 계산

### Config (Wave 6 Phase 1 — anchor c01_01 stack 그대로, prompt_mode 만 차이)
```yaml
filter:
  name: "XiYanFilter"
  params:
    provider: "glm"
    model_name: "zai-org/glm-4.7"
    max_iteration: 1
    temperature: 0.0
    prompt_mode: "recall_biased_mild"   # 또는 strong / exclusion_rule
    sanitize_output: true                # 학술 agent §2.3, default-on
```

3 cells: `wave6_p1_recall_biased_{mild, strong, exclusion_rule}.yaml`. Root 책임.

### 산출물 (본 모듈, 2026-05-16)
- [`xiyan_filter.py`](xiyan_filter.py) — `prompt_mode` parameter + `sanitize_filter_output()` static method + 측정 메타 노출
- [`/home/hyeonjin/thesis_refactored/src/prompts/filter.md`](/home/hyeonjin/thesis_refactored/src/prompts/filter.md) — `recall_biased_mild` / `recall_biased_strong` / `recall_biased_exclusion_rule` 3 section 추가 (학술 agent §3.1 PROMPT_M1_A/B/C 원문 그대로)
- [`tests/test_xiyan_recall_biased.py`](tests/test_xiyan_recall_biased.py) — 18-scenario smoke test (**PASSED 18/18**): sanitize unit (table/col hallucination 제거, dedup, non-list, non-dict) + prompt_mode validation + 각 mode 의 prompt section 검증 (signature phrase 확인) + 통합 sanitize (LLM hallucination 자동 제거) + 측정 메타 (mode/prune_pct/halluc_count) + backward-compat

### 다음 단계 (Root + Analyzer 책임)
- **Root**: `scripts/run_wave6_phase1_recall_biased.sh` 작성 → 3 yaml parallel launch (GPU 0+1, ~1.5h, ~17K LLM call 의 일부). HISTORY + CATALOG + ID_MIGRATION 3종 갱신.
- **Analyzer**: `notebooks/analysis_results/wave6_phase1_recall_biased_2026-05-XX.md` — 3 variants × 1534q × 7 metrics 매트릭스 + R_gain / P_loss / ΔF1 trajectory + hallucination rate per variant + **Phase 2 분기 결정 권고** (DECISIONS §3 분기 spec: R_fil ≥ 0.92 → M2 CoT 결합 / R_fil 0.88~0.92 → M3 OR Voting / R_fil < 0.88 → M4 Bidirectional 우선).

---

## COND. Conditional Filter call wrapper (★ 최상, Phase 4.2 활성 2026-05-16)

### 동기 (DECISIONS 2026-05-16 §3 Phase 4.2 + 학술 Agent Improving Plan §Phase 4.2)
- 5/14 anchor sweep 결과 anchor-band Prune% **92~94%** (Phase 1+2 grid evidence) — extractor 가 schema 의 대부분을 trim 한 query 에서 추가 LLM Filter call 의 marginal value 미미.
- 직전 5/14 anchor 의 6.32% **involuntary** skip (filter 자체가 빈 결과 반환 등) 과 별개 **voluntary** cost-effective skip mechanism — production deployment 정량.
- 학술 frame: paper §V.5.x.M.3 production deployment narrative + §V.5.x.M.11 Filter Short-Circuit voluntary/involuntary mechanism 분리 evidence.

### 설계 — TCR(q) gated voluntary skip
```
TCR(q) = |filter input subgraph columns| / |full schema columns|
         (작을수록 extractor 가 schema 를 잘 trim 한 query — Filter 추가 호출 marginal)

if TCR(q) < tcr_threshold:                     ← voluntary skip
    final_nodes = subgraph 의 모든 column 그대로
    inner Filter NOT called → LLM call cost 0
else:
    final_nodes = inner.refine(query, subgraph, ...).final_nodes   ← 정상 호출
```

- TCR 우선순위 (compute_tcr): kwargs `tcr` override > metadata['col_to_id'] 자체 계산 > None (caller 가 safe path = inner-call 결정)
- skip 시: status="Answerable" if final_nodes else "Unanswerable" — extractor output 자체가 비어있으면 그대로 전파

### 인터페이스 (계약 유지)
```python
@register("filter", "ConditionalFilterWrapper")
class ConditionalFilterWrapper(BaseFilter):
    def __init__(self,
                 inner_filter: Dict,                   # any registered filter
                 call_mode: str = "conditional",        # "conditional" | "always"
                 tcr_threshold: float = 0.5,
                 **kwargs): ...

    def refine(self, query, subgraph, db_id=None,
               tier2_pool=None, gat_scores=None,
               metadata=None, tcr=None,                # kwargs override 가능
               **kwargs) -> Dict:
        # stats: call_mode, tcr_threshold, tcr_value, voluntary_skipped,
        #        inner_called, inner_filter_name, n_input_columns,
        #        n_full_schema_columns, n_final_nodes
        # filter_info: filter_call_mode / filter_tcr_threshold / filter_tcr_value /
        #              filter_tcr_source ("override"|"computed"|"unavailable") /
        #              filter_voluntary_skipped / filter_inner_called /
        #              filter_inner_filter_name / filter_inner_status /
        #              (inner_*: inner filter 의 진단 일체 carry over)
```

### 측정 메타 (output 자동 노출)
- **Filter 호출 비율 (cumulative)**: aggregate `filter_voluntary_skipped` 의 (1 − rate)
- **Filter skip 시 F1 손실 (per-query)**: skip 한 query 의 final F1 vs always-call baseline F1 비교 (Root + analyzer 분담)
- **LLM call 절감 % (cost 정량)**: aggregate `filter_inner_called=False` 비율 × inner filter 의 평균 LLM call 수 — paper §V.5.x.M.3 production deployment 핵심 정량

### Config (Phase 4.2 3 cells)
- [`configs/experiments/abl/c05_phase4_conditional_filter/p4_2_thr_0.3.yaml`](../../../configs/experiments/abl/c05_phase4_conditional_filter/p4_2_thr_0.3.yaml) — 보수적
- [`configs/.../p4_2_thr_0.5.yaml`](../../../configs/experiments/abl/c05_phase4_conditional_filter/p4_2_thr_0.5.yaml) — 학술 agent default
- [`configs/.../p4_2_thr_0.7.yaml`](../../../configs/experiments/abl/c05_phase4_conditional_filter/p4_2_thr_0.7.yaml) — 공격적

전부 anchor c01_01 (θ=0.1, K=20) stack 그대로 + Filter 만 `ConditionalFilterWrapper(inner=XiYanFilter GLM 4.7)`.

### 한계 / Caveat
- **TCR 정의 의존성**: 본 구현은 `|subgraph cols| / |full schema cols|` 단순 ratio. 학술 agent doc §4.2 의 더 풍부한 정의 (`Confidence(q) = R_sel(q) + 1/TCR(q)`) 는 Root pipeline 측에서 `tcr` kwargs 로 override 주입 시 그대로 동작 (본 wrapper 가 override 우선).
- **skip 시 precision 위험**: extractor 가 noise 컬럼을 통과시킨 경우 그대로 final 에 반영 → P drop 가능. paper §V.5.x.M.3 의 voluntary skip cost-effective trade-off narrative.
- **threshold sensitivity**: 0.3 (skip ↓ → safe, marginal cost saving) vs 0.7 (skip ↑ → high cost saving, F1 drop risk) — Phase 4.2 3-cell sweep 의 frontier 정량.

### 산출물 (본 모듈 책임, 2026-05-16)
- [`conditional_filter_wrapper.py`](conditional_filter_wrapper.py) — `ConditionalFilterWrapper` 신규 클래스
- [`tests/test_conditional_filter_wrapper.py`](tests/test_conditional_filter_wrapper.py) — 16-scenario smoke test (**PASSED 16/16**): TCR 계산 / skip vs call / always mode / override / unknown TCR safe path / 빈 subgraph / invalid arg / yaml-build
- [`__init__.py`](__init__.py) — `ConditionalFilterWrapper` export
- 3 sweep configs (위 §"Config")

### 다음 단계 (Root + Analyzer 책임)
- **Root**: `scripts/run_phase4_2_conditional_filter.sh` (3 yaml 순차/병렬 실행 — GLM API 단일 endpoint 부하 고려) → 5/14 anchor a05_xxx 와 함께 ablation chain. ETA ~ 5~9h × 3 cells (or skip rate 에 따라 단축).
- **Analyzer**: `notebooks/analysis_results/phase4_2_conditional_filter_2026-05-XX.md` — TCR 분포 / Filter 호출 비율 / skip 시 F1 손실 per-difficulty / cost 절감 % / threshold frontier 정량 / paper §V.5.x.M.3 + §V.5.x.M.11 narrative evidence 직접 매핑.

---

## RSL-C-GT. GRAST-SQL + Graph Transformer reranking (★ 최상, 학술 Agent Phase 5 Option β, 2026-05-14)

### 동기
- Direction A (RSL-A) + Direction C (RSL-C) 둘 다 **F1 -0.28 붕괴 + EX sub-noise ±0.003** — 학술 Agent Phase 5 §0 의 학술 frame 재정의: "Filter-Invariant 경계 확정 실험".
- Graph Transformer 의 query-aware encoding 이 Steiner tree 의 query-무관 selection 한계 mitigation 의 **유일 candidate** (학술 Agent Q4(b)). 단 근본적 R-P 긴장 해소 사전 보증 X.
- positive (R-P trade-off mitigation) / null (Filter-Invariance 경계 추가 evidence) 모두 학술적 가치 — paper §V.5.x.M.6 "Mechanism-Agnostic R-P Limit" narrative 강화.

### 학술 Agent Phase 5 Q1+Q2 확정 spec
| 항목 | 원문 / 학술 Agent 권고 | 본 구현 |
|---|---|---|
| Integration 위치 | **Option β** (Module:Filter GRASTFDFilter Step 2 add-on) | `GRASTFDFilterWithTransformer(GRASTFDFilter)` 상속 |
| Step 1 (LLM-Reranker) | **Skipped** — 현 anchor LLM column scorer 출력 h^0 재활용 (5/22 일정) | `_build_h0`: XiYan selected bit + GAT score + FK/PK flag concat → in_dim |
| Encoder | 3 layers, hidden 2048 (ROC-AUC) / **1024 (PR-AUC, default)**, heads 미명시 (default 8) | `GraphTransformerEncoder(3, 1024, 8)` |
| Edge types R | `{fk, col→fk, col→pk}` directed + reverse = **6 channels** | 6 distinct `edge_type` enum |
| PE | Relation-specific attention coefficient ψ^(ℓ)(i,j) — 표준 RPE 아님 | per-layer + per-head + per-edge-type learnable scalar bias |
| Belongs_to | **별도 채널 없음** — node feature (학술 Agent §1.1) | h^0 에 table membership 인코딩, edge type 에 미포함 |
| Loss | margin-based contrastive (gold > non-gold), lr 5e-5, **40 epochs**, batch 32 | `margin_contrastive_loss` |
| GPU 시간 | Step 2 only ~1~3h (Step 1 생략) | training script 외부 |

### Pipeline (per query)
```
Step 1   XiYan forward                 (anchor 정합)
         S_fwd, h^0 = anchor LLM column scorer output 재활용

Step 2   Relation-aware Graph Transformer
         input  : h^0 [N, in_dim], edge_index [2, E], edge_type [E]
         output : refined node repr [N, 1024], column scores [N]

Step 3   Steiner Tree (기존 GRASTFDFilter)
         terminal_source="graph_transformer":
           - relevance = sigmoid(GT_score)
           - top-K (default 10) or threshold filter → terminal columns
           - ∪ S_fwd (connectivity 확보 위한 forward retention)
         steiner_tree(FD_graph, terminals) → restore columns

Step 4   FK/PK hardcode (기존)

Output  final_nodes = S_fwd ∪ S_steiner_restore ∪ S_struct
```

### Fallback (학술 Agent Q5 fallback plan)
- **checkpoint 부재 / load 실패**: `transformer=None` 으로 두고 `terminal_source` 자동 fallback to `"forward"` (recall-safe, RSL-C base behavior 와 동일)
- **GT forward divergence (NaN/Inf, exception)**: forward 호출 try/except → fallback to `"forward"` + `diag["terminal_fallback"]` 기록
- **40 epoch 학습 divergence**: `smoke_train_protocol()` 의 plateau detector → early stop + 보고서에 "Step 2 없이 Step 1 단독 결과" Caveat 3 variant 로 학술 위치 부여 (학술 Agent §6.3)

### 인터페이스
```python
@register("filter", "GRASTFDFilterWithTransformer")
class GRASTFDFilterWithTransformer(GRASTFDFilter):
    def __init__(self,
                 transformer_checkpoint_path=None,
                 transformer_in_dim=16, transformer_hidden_dim=1024,
                 transformer_num_layers=3, transformer_num_heads=8,
                 transformer_dropout=0.1,
                 transformer_score_top_k=10,
                 transformer_score_threshold=None,
                 transformer_device="cpu",
                 terminal_source="graph_transformer",
                 # GRASTFDFilter args 모두 그대로 (inferred_fk, fk_pk_hardcode, ...)
                 **kwargs): ...

    # GRASTFDFilter.refine 의 _resolve_terminals override —
    # terminal_source=="graph_transformer" 일 때 GT forward + top-K/threshold 선택
```

### Smoke Test (학술 Agent Q5 protocol — 본 모듈 구현)
- `smoke_train_protocol(model, batches, val_batches, num_epochs=5, ...)`:
  - train margin loss < 0.3 + val PR-AUC Δ ≥ +0.01 → pass
  - 2 epoch 연속 loss 개선 없음 → plateau → early stop
  - NaN/Inf loss → divergence → early stop + fallback flag
- 본 chain 의 unit test (`test_grast_fd_transformer.py`):
  - GT architecture forward shape / gradient flow / edge type bias distinct
  - margin loss boundary (pos < neg / pos > neg / no pos)
  - filter integration (checkpoint 부재 fallback / random-init GT 활성 / top-K / threshold / FK metadata 경유)
  - invalid arg ValueError

### 한계 / Caveat (학술 Agent Phase 5 §5.2 + §6.1)
- **P/R ratio 9.07× 개선 보증 안 됨** — GT query-aware 가 mitigation candidate 일 뿐, R-P 긴장 해소 사전 보증 X
- **Risk High** (GAT 7-trial null 재현 가능성 + NaN divergence) — Step 2 only ~1~3h 학습 시간 단 학습 결과 null 가능
- **EX 개선 기대 낮음** — Filter-Invariant boundary 확정 frame 정합. positive/null 모두 학술적 가치
- **h^0 quality 의존** — Step 1 fine-tune 생략으로 anchor LLM column scorer 의 representational power 가 GT 의 ceiling 결정

### 산출물 (본 모듈 책임, 2026-05-14)
- [`grast_fd_transformer.py`](grast_fd_transformer.py) — `GraphTransformerEncoder` (3 layer, hidden=1024, 8 head, edge type bias) + `RelationAwareGTLayer` (sparse relation-aware attention with index_add aggregation) + `margin_contrastive_loss` + `smoke_train_protocol` (학술 Agent Q5).
- [`grast_fd_filter_with_transformer.py`](grast_fd_filter_with_transformer.py) — `GRASTFDFilterWithTransformer(GRASTFDFilter)` — Step 2 add-on, h^0 builder, GT forward, top-K/threshold terminal selection, checkpoint load + fallback.
- [`tests/test_grast_fd_transformer.py`](tests/test_grast_fd_transformer.py) — 16-scenario smoke test (**PASSED 16/16**): GT architecture + training loss + filter integration + fallback paths.
- [`configs/.../a05_26_grast_with_transformer_glm.yaml`](../../../configs/experiments/abl/a05_filter_agentic/a05_26_grast_with_transformer_glm.yaml) — Direction C-GT sweep config (checkpoint path placeholder, 학습 완료 후 갱신).
- [`__init__.py`](__init__.py) — `GRASTFDFilterWithTransformer` export.

### 다음 단계 (Root 책임)
- **학습 launch** (학술 Agent §6.4 5/15~5/22 일정): BIRD-Train 으로 Step 2 GT 40 epoch 학습 — `smoke_train_protocol` 로 5 epoch (12.5%) 사전 smoke + plateau detect → 정식 학습 진행 or fallback.
- **Checkpoint 저장**: `outputs/checkpoints/grast_fd_transformer_*.pt` + a05_26 yaml 의 `transformer_checkpoint_path` 갱신.
- **Sweep launch**: a05_26 BIRD-Dev 1534 query.
- **Analyzer 보고**: `notebooks/analysis_results/direction_c_gt_sweep.md` — RSL-A / RSL-C / RSL-C-GT 3-way 비교 + Filter-Invariant boundary 확정 narrative + Mechanism-Agnostic R-P Limit (paper §V.5.x.M.6) 정량 evidence 확장.

---

## RSL-C. GRAST-SQL FD Filter (★ 최상, Direction C trigger 발효 2026-05-14)

### 동기
- Analyzer Direction A sweep 결과 (2026-05-14, `notebooks/analysis_results/direction_a_rsl_backward_sweep.md`):
  - **ΔF1 = -0.2832** (학술 Agent threshold +0.02 의 강한 negative), P drop **-0.4345**, R gain **+0.0684** → R-P trade-off ratio = -6.4× P loss per R gain
  - EX maintained **-0.0033** (sub-noise) — F1-EX dichotomy
- 학술 Agent Phase 3 trigger: **ΔF1(A) < +0.02 → Direction C 타겟 launch**. C-1 feasibility 1.46× (mean fk_coverage_rate = 0.7312), C-2 mid-priority (mean is_join_complete = 0.8624, multi-table 13.76% miss).
- **핵심 가설** (DECISIONS 2026-05-14 §2.2): Phase 2 C-2 의 multi-table miss 9~13% 는 FK declaration 부족 → join col miss. Steiner-tree 가 query mentioned cols 를 terminal 로 한 connectivity 회복 cols 만 restore → backward union 처럼 noise 폭증 없음.

### 설계 — 4-step pipeline (terminal_source="forward" 시 1 LLM call/query)
```
Step 1  XiYan forward (anchor 정합)
        S_fwd = XiYanFilter.refine(query, subgraph, db_id).final_nodes

Step 2  FD Graph 구성 (algorithm-only, networkx)
        nodes: "table.col" + "table"
        edges:
          (i) belongs_to  : column -- table         (intra-table grouping)
          (ii) FK         : src.col -- dst.col       (metadata fk_to_id)
          (iii) inferred  : src.col -- dst.col       (yaml `inferred_fk`,
                            Analyzer 후속 GPT-4.1-mini 보완)

Step 3  Steiner Tree Restore (networkx steiner_tree)
        terminals ← terminal_source policy:
          - "forward" (default, no LLM): S_fwd 의 column 노드
          - "gat_topk": gat_scores 의 top-K + S_fwd (fallback to forward)
          - "prelim_sql": RSL-A 의 prelim SQL prompt 재사용 (+1 LLM call)
        steiner = nx.approximation.steiner_tree(FD_graph, terminals,
                                                method=steiner_method)
        S_steiner_restore = {n ∈ steiner.nodes() | "." in n} − S_fwd
        # disconnected component 별로 따로 계산. single-terminal component skip.
        # max_restore cap 으로 over-restoration 차단.

Step 4  S_struct FK/PK hardcode (anchor 정합)

Output: final_nodes = S_fwd ∪ S_steiner_restore ∪ S_struct
```

### 인터페이스
```python
@register("filter", "GRASTFDFilter")
class GRASTFDFilter(BaseFilter):
    def __init__(self,
                 model_name="zai-org/glm-4.7", temperature=0.0,
                 xiyan_max_iteration=1, xiyan_model_name=None,
                 xiyan_num_examples=3,
                 db_dir="./data/raw/BIRD_dev/dev_databases", num_examples=3,
                 inferred_fk=None,            # ["src.col->dst.col", ...]
                 include_belongs_to=True,
                 terminal_source="forward",   # "forward" | "gat_topk" | "prelim_sql"
                 top_k=10,
                 steiner_method="default",    # "default" | "mehlhorn" | "kou"
                 max_restore=30,
                 fk_pk_hardcode=True,
                 provider="glm", api_key=None, base_url=None, **kwargs): ...

    def refine(self, query, subgraph, db_id=None,
               tier2_pool=None, gat_scores=None, metadata=None,
               evidence=None, **kwargs) -> Dict:
        # stats: fwd_nodes, terminal_count, steiner_restore, struct, final,
        #        graph_nodes, graph_edges, declared_fk_count, inferred_fk_count,
        #        terminal_source_used, restore_is_empty, restore_capped_from,
        #        steiner_skipped
```

### LLM Cost 비교
| Filter | LLM calls/query | Token cost vs anchor |
|---|---:|---:|
| XiYan anchor | 1 | 1× (baseline) |
| RSL-A (Direction A) | 2 | ~+100% (preliminary SQL full schema input) |
| **RSL-C (terminal_source="forward")** | **1** | **+0% (algorithm-only)** ⭐ |
| RSL-C (terminal_source="prelim_sql") | 2 | ~+100% (RSL-A 와 동일 prompt 재사용) |

### inferred_fk (Analyzer 후속 prerequisite)
- C-1 의 outlier DB: debit_card_specializing (fk_coverage=0.20) / card_games (0.5714)
- 학술 Agent Phase 3 권고: GPT-4.1-mini 로 두 DB 의 missing FK 예측. Analyzer 후속 chain 책임.
- 본 모듈은 yaml `inferred_fk: List[str]` (default empty) 만 받음. 형식 `"src_tbl.src_col->dst_tbl.dst_col"`.

### 한계 / Caveat
- **disconnected component**: Steiner tree 는 단일 connected graph 필요. 본 구현은 component 별로 계산 후 union — terminal 이 component 내 1 개뿐이면 그 component 의 restore 는 skip.
- **inferred_fk 의 GPT 보완 필요**: 본 chain 의 prerequisite. 미보완 시 fk_coverage 낮은 DB (debit_card, card_games) 에서 Steiner tree 효과 제한.
- **column name 중복**: Steiner tree 는 node id 가 "table.col" 이므로 RSL-A 의 col_name expansion 같은 중복 candidate 폭증 없음 (precision 보호).

### 산출물 (본 모듈 책임, 2026-05-14)
- [`grast_fd_filter.py`](grast_fd_filter.py) — `GRASTFDFilter` 신규 구현. FD graph + Steiner tree + 3-mode terminal_source + max_restore cap.
- [`tests/test_grast_fd.py`](tests/test_grast_fd.py) — 17-scenario smoke test (**PASSED 17/17**). 핵심: FK 경유 join col restore / disconnected component partial restore / inferred_fk bridge / 3 terminal_source mode / max_restore cap / FK hardcode rescue / metadata fallback.
- [`__init__.py`](__init__.py) — `GRASTFDFilter` export.

### 다음 단계 (Root + Analyzer 책임)
- **Analyzer**: debit_card_specializing + card_games 의 GPT-4.1-mini inferred_fk 보완 (Phase 3 prerequisite). 출력: yaml-ingestible `inferred_fk: List[str]` snippet.
- **Root**: Direction C pipeline config 작성 (terminal_source="forward" 가 cost 최소). a05 sweep 에 셀 추가 → ΔF1 / ΔEX 정량.

---

## RSL-A. RSL-SQL Backward Filter (직전 axis, Direction A 정식 배포 보류 2026-05-14)

> ⚠️ **Status 변경 (2026-05-14)**: Analyzer Direction A sweep 결과 ΔF1 = -0.2832 (net negative) → **정식 배포 보류**. 단 EX maintained -0.0033 + paper §V.5.x.M 의 R-P/F1-EX dichotomy narrative evidence 로 유지. Direction C (위 RSL-C) 가 우선 launch.

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
- 학술 Agent 2026-05-12 ([planning/filter/filtering_suggestion_by_scholar_agent_2026-05-12.md](/home/hyeonjin/thesis_refactored/planning/filter/filtering_suggestion_by_scholar_agent_2026-05-12.md)) 가 5 references (Glass 2025 / Hoang 2025 / Talaei 2024 / Maamari 2024 / Yuan 2025) 로 합성한 hybrid 설계.

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
| [rsl_backward_filter.py](rsl_backward_filter.py) | 신규 — RSL Backward (완료 2026-05-13, 정식 배포 보류 2026-05-14) |
| [tests/test_rsl_backward.py](tests/test_rsl_backward.py) | 신규 — RSL Backward 15-scenario smoke (PASSED) |
| [grast_fd_filter.py](grast_fd_filter.py) | 신규 — GRAST-FD Direction C (완료 2026-05-14) |
| [tests/test_grast_fd.py](tests/test_grast_fd.py) | 신규 — GRAST-FD 17-scenario smoke (PASSED) |
| [grast_fd_transformer.py](grast_fd_transformer.py) | 신규 — Relation-aware Graph Transformer (Hoang 2025 §3.3 Option β) + training utility (2026-05-14) |
| [grast_fd_filter_with_transformer.py](grast_fd_filter_with_transformer.py) | 신규 — Direction C-GT Filter (Step 2 add-on) |
| [tests/test_grast_fd_transformer.py](tests/test_grast_fd_transformer.py) | 신규 — GT 16-scenario smoke (PASSED) |
| `configs/.../a05_26_grast_with_transformer_glm.yaml` | 신규 — Direction C-GT sweep config (checkpoint placeholder) |
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
