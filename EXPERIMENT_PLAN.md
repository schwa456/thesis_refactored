# Full Framework Experiment Plan — 9 아키텍처 제안 통합 로드맵

> **이 문서의 역할**: 루트(오케스트레이터) 세션에서 관리하는 **전 모듈 통합 실험 계획**.
> 각 모듈 세션의 상세 계획은 하위 EXPERIMENT_PLAN을 참조 — 이 문서는 **의존성, 우선순위, 통합 실험, 논문 매핑** 에만 집중.
>
> - [src/modules/builders/EXPERIMENT_PLAN_builders.md](src/modules/builders/EXPERIMENT_PLAN_builders.md)
> - [src/modules/selectors/EXPERIMENT_PLAN_selectors.md](src/modules/selectors/EXPERIMENT_PLAN_selectors.md)
> - [src/modules/extractors/EXPERIMENT_PLAN_extractors.md](src/modules/extractors/EXPERIMENT_PLAN_extractors.md)
> - [src/modules/filters/EXPERIMENT_PLAN_filters.md](src/modules/filters/EXPERIMENT_PLAN_filters.md)
> - a05 filter agentic 세부: [~/.claude/plans/vivid-sprouting-sunbeam.md](~/.claude/plans/vivid-sprouting-sunbeam.md)

---

## 0. 현재 베이스라인 (모든 실험의 출발점)

| 범주 | Anchor | R | P | F1 |
|------|--------|---|---|-----|
| Ensemble 최고 | `abl_ens_enriched_xiyan` (E1, Enriched Builder) | 0.6658 | 0.8147 | 0.7327 |
| Ensemble + Triplet | `abl_ens_triplet_xiyan` (E2) | 0.6823 | 0.8139 | **0.7424** |
| Direct variant 최고 | `abl_a03_17` (SuperNode-Direct + Fixed PCST + XiYan) | 0.6761 | 0.7128 | 0.6940 |
| Filter 최고 | `abl_a05_02` (Reflection, anchor=a03_17) | **0.7320** | 0.6833 | 0.7068 |

**핵심 관찰**:
- **Precision 상한(≈0.81)은 Builder가 결정** (Enriched/Triplet)
- **Recall 상한은 Filter의 restore path가 결정** (Reflection이 prune-only 한계 돌파)
- 두 상한의 **교차 결합이 아직 미검증** (E1 × Reflection, E2 × Reflection)

---

## 1. 9 제안 전역 배분 (Cross-Module Matrix)

| # | 제안 | Builder | Selector | Extractor | Filter | Pipeline |
|---|------|---------|----------|-----------|--------|----------|
| 1 | RL Schema Linker (GRPO) | | ★ S-I | | | |
| 2 | Relational Foundation Model | B-I | ★ S-II | | | |
| 3 | EHGAT (Edge Hypergraph) | B-II | ★ S-III | | | |
| 4 | Louvain Community PCST | | | ★ E-I | | |
| 5 | Pathfinding + PCST Ensemble | | | ★ E-II | | |
| 6 | Autonomous Agent (AutoLink) | | | | ★ FL-I | |
| 7 | Extractive Decoder-only LLM | | S-IV | | FL-II | |
| 8 | Multi-agent RL Pipeline (MARS-SQL) | | | | | ★ PL-I |
| 9 | Neurosymbolic 3-layer Cooperation | B-III (L1) | S-V (L1) | E-III (L2) | FL-III (L3) | ★ PL-II |

★ = 해당 제안의 **주 구현 지점**.

---

## 2. Cross-Module Dependency Graph (Critical Path)

```
Builder B-III (FK reachability precompute)
    ├─→ Selector S-V (Neurosymbolic L1 score boost)
    ├─→ Extractor E-III (Neurosymbolic L2 cost adjustment)
    └─→ Filter FL-III (Neurosymbolic L3 connectivity verifier)

Builder B-II (Line graph)
    └─→ Selector S-III (EHGAT)

Builder B-I (RFM tokenization)
    └─→ Selector S-II (RFM encoder)

Selector S-IV / Filter FL-II (LLM logprobs)
    └─→ src/llm_client/api_handler.py 확장 (vLLM logprobs 지원)

Pipeline PL-II (Neurosymbolic 3-layer 루프)
    ← B-III + S-V + E-III + FL-III 모두 선결
```

**병목**: **Builder B-III (FK reachability)** — 3 모듈이 의존하는 infra. **가장 먼저 구현**.

---

## 3. 통합 실험 — 2×2×2 확장 (신규 상한 탐색)

기존 2×2×2는 (Seed × PCST × Filter). **신규 축 추가** 로 의미있는 상한을 탐색:

### 3.1 Cross-Module Synergy Grid (최고점 조합)
| 실험 ID | Builder | Selector | Extractor | Filter | 목적 |
|---------|---------|----------|-----------|--------|------|
| `int_01_e1_refl` | Enriched | Ensemble | AdaptivePCST | Reflection | E1 × a05_02 최초 결합 |
| `int_02_e2_refl` | Triplet | Ensemble | EdgePrize | Reflection | E2 × a05_02 |
| `int_03_e2_path_refl` | Triplet | Ensemble | **E-II Pathfinding** | Reflection | Extractor 신규 × Filter 신규 |
| `int_04_ns_full` | Enriched + **B-III** | Ensemble + **S-V** | **E-III** | **FL-III** + Reflection | **Neurosymbolic 3-layer 전체** |
| `int_05_direct_ns` | Enriched + B-III | DirectGAT-SuperNode + S-V | E-III | FL-III + XiYan | Direct variant에서 NS 효과 |
| `int_06_best_x_autolink` | Triplet | Ensemble | EdgePrize | **FL-I AutoLink** | 최고 × autonomous agent |

**int_04가 논문의 주력 결과 후보** — 모든 기여가 수렴하는 설정.

### 3.2 Backbone 민감도 (논문 섹션 보강)
| 실험 ID | 변경축 | 비교 |
|---------|--------|------|
| `int_07_int04_gpt4o` | FL-III+Reflection backbone을 GPT-4o-mini로 | vs int_04 |
| `int_08_rfm_s_v` | Selector를 RFM (S-II) + S-V 보강 | vs int_04 |

---

## 4. 우선순위 & 타임라인 (작업 순서)

### Phase A — Infrastructure (선결)
- [ ] **Builder B-III** — FK reachability metadata precompute (가장 중요)
- [ ] Builder B-II — LineGraph 변환기
- [ ] Builder B-I — RFM serialize API
- [ ] `src/llm_client/api_handler.py` — vLLM logprobs 지원

### Phase B — Low-risk Quick Wins
- [ ] **Selector S-V** (Neurosymbolic L1, λ 튜닝)
- [ ] **Extractor E-III** (FK prior cost adjustment)
- [ ] **Filter FL-III** (Symbolic Verifier, detect-only 먼저)
- [ ] **Extractor E-II** (Pathfinding ensemble, MSTExtractor 재활용)

### Phase C — Medium Effort
- [ ] **int_01~03** (Cross-module synergy — 기존 모듈 재조합)
- [ ] **Extractor E-I** (Louvain community)
- [ ] **Filter FL-I** (AutoLink full-exploration, a05_13~15)
- [ ] **Filter FL-II** (Extractive LLM, a05_16~18)
- [ ] **Selector S-III** (EHGAT)

### Phase D — High Effort (후순위)
- [ ] **Selector S-II** (RFM encoder, GPU/latency 높음)
- [ ] **Selector S-I** (RL/GRPO, 학습 불안정성)
- [ ] **Pipeline PL-I** (MARS-SQL — 제안 #8, multi-agent RL)

### Phase E — Integration
- [ ] **int_04** (Neurosymbolic 3-layer 전체) — 논문 주력
- [ ] **int_05~08** (Direct variant, backbone 민감도)

---

## 5. 논문 매핑 (한국지능정보시스템학회 2026 춘계)

**현재 초안**: [notebooks/analysis_results/paper_draft_abstract_intro.md](notebooks/analysis_results/paper_draft_abstract_intro.md)

| 논문 섹션 | 주력 기여 | 근거 실험 |
|-----------|----------|-----------|
| II. Related Works | 9 제안의 선행 연구 조망 | 각 서브세션 plan의 "학술 기여" |
| III. Methodology | Neurosymbolic 3-layer framework 중심 | B-III + S-V + E-III + FL-III |
| IV. Experiments | int_04 주력 + ablation 단계별 | int_01~04, phase별 축별 ablation |
| V. Conclusion | 그래프 prior와 LLM agent의 결합 축 | 전체 |

**핵심 argument 후보**:
1. "Graph-prior-conditioned agentic refinement" (Filter FL-I + S-IV 결합)
2. "Deterministic symbolic verification as guardrail on neural pipeline" (FL-III + E-III)
3. "Tiered schema linking: from GAT score to PCST membership to LLM verdict, each tier as explicit evidence" (pipeline 전반)

---

## 6. 닫힌 주제 (재탐색 금지)
- **방안 A (Score-driven PCST cost)**: `ScoreDrivenPCSTExtractor` BO2 tuned. 완료.
- **방안 B (Bayesian Optimization)**: 완료, F1=0.6751.
- **Idea 2 (Product Cost) / Idea 4 (Component-Aware)**: I24b F1=0.7063 완료.
- 위 변형은 기본값으로만 인용, **신규 실험에서 재변형 금지** (memory rule).

---

## 7. 실험 문서화 규칙 (재확인)
모든 신규 실험 실행 후 3개 파일 동기 업데이트 (memory rule):
- [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md)
- [EXPERIMENT_CATALOG.md](EXPERIMENT_CATALOG.md)
- [EXPERIMENT_ID_MIGRATION.md](EXPERIMENT_ID_MIGRATION.md)

**메트릭 표기**: R, P, F1 순서 / 소수점 4자리.

**ID 명명**:
- 단일 모듈 신규: `abl_{module}_{feature}_{nn}` (예: `abl_ext_path_01`)
- 통합 실험: `int_{nn}_{short_desc}` (예: `int_04_ns_full`)
- a05 라인 연장: `abl_a05_13 ~ a05_22` (filter 신규 3축)

---

## 8. 디렉토리 구조 (신규)

```
configs/experiments/abl/
├── a05_filter_agentic/       # 기존 (a05_01~12) + 신규 (a05_13~22)
├── build/                    # 신규 — B-I/B-II/B-III pilot
│   ├── fk_reach/
│   ├── linegraph/
│   └── rfm_tokens/
├── sel/                      # 신규 — S-I ~ S-V
│   ├── ns_l1/
│   ├── rfm/
│   ├── ehgat/
│   ├── xllm/
│   └── rl/
├── ext/                      # 신규 — E-I ~ E-III
│   ├── louvain/
│   ├── path_ensemble/
│   └── fk_prior/
└── integration/              # 신규 — int_01 ~ int_08
```

3개 디렉토리(configs/outputs/logs) **동일 하위 구조** 유지 (CLAUDE.md rule).

---

## 9. 검증 & 리스크 관리

### 검증 공통
- 모든 실험은 **Recall/Precision/F1 (4소수)** + **쿼리당 처리시간** 리포트.
- 진행중 실험은 ETA 함께 보고 (memory rule).
- End-to-end SQL execution accuracy는 이 단계에서 미포함 (schema linking 단위 평가 유지).

### 리스크 맵

| 리스크 | 영향 | 대응 |
|--------|------|------|
| Builder B-III metadata 크기 폭발 | 캐시 용량 | matrix sparsification, 큰 DB만 옵션 적용 |
| vLLM logprobs 미지원 | S-IV/FL-II 블록 | GPT-4o-mini logprobs proxy or fallback to text parsing |
| RL(GRPO) 학습 불안정 | S-I 블록 | KL constraint + supervised ckpt warm-start |
| Neurosymbolic 3-layer combined regression | int_04 실패 | 각 Layer 독립 ablation 먼저 검증 (int_01~03) |
| LLM 비용 증가 (Filter FL-I 5-step ReAct) | Latency/budget | F4 uncertainty gating으로 hard query만 |
| FK reachability가 DB에 없는 경우 | Layer 전체 비활성 | Gracefully fallback to baseline |

---

## 10. 세션 워크플로우 (루트 ↔ 서브세션)

### 서브세션 진입 시 체크리스트
각 모듈 세션(builders/selectors/extractors/filters)은 시작 시 **순서대로** 읽는다:
1. `/home/hyeonjin/thesis_refactored/CLAUDE.md` — 전역 규칙 (읽기 전용)
2. `/home/hyeonjin/thesis_refactored/EXPERIMENT_PLAN.md` — **본 문서** (루트 로드맵, 읽기 전용)
3. `src/modules/{module}/CLAUDE.md` — 모듈 맥락 (수정 가능)
4. `src/modules/{module}/EXPERIMENT_PLAN_{module}.md` — 모듈 내부 구현 상세 (주 작업 스펙)

### 수정 책임 분할
| 문서 | 수정 책임 |
|------|----------|
| `CLAUDE.md` (루트) | **루트 세션만** |
| `EXPERIMENT_PLAN.md` (루트, 본 문서) | **루트 세션만** |
| `src/modules/{module}/CLAUDE.md` | 해당 모듈 세션 |
| `src/modules/{module}/EXPERIMENT_PLAN_{module}.md` | 해당 모듈 세션 |
| `EXPERIMENT_HISTORY.md` / `CATALOG.md` / `ID_MIGRATION.md` | **실험 실행 주체가 모두 갱신** (memory rule) |

### 루트 세션의 책임
1. 본 문서를 기준으로 각 서브세션에 작업 분배 (subsession 호출 시 루트 PLAN + 해당 모듈 PLAN 참조 명시).
2. Phase A (infrastructure) 완료 시까지 다른 phase 블록.
3. **int_04가 "논문 주력 결과"** 임을 각 서브세션에 공유 — 모든 모듈이 "이 실험에 들어갈 품질"로 구현.
4. 루트 PLAN 변경 시 각 서브세션 PLAN과의 정합성 확인 (Cross-Module Matrix §1).
5. 통합 실험(int_01~08)은 **루트 세션이 직접 실행** — 여러 모듈이 동시 개입하므로.

### 각 서브세션의 책임
- 담당 plan 파일(`EXPERIMENT_PLAN_{subsession}.md`) 을 그대로 작업 스펙으로 사용.
- 인터페이스 계약(`build`, `select`, `extract`, `refine` signature)은 **절대 깨지 않도록** 재확인 — 하류 모듈 파손 방지.
- 신규 구현마다 HISTORY/CATALOG/ID_MIGRATION 3파일 동기 갱신.
- 루트 PLAN의 변경이 필요한 판단(예: 제안 재분류, 우선순위 변경, 새로운 의존성 발견)은 **루트 세션에 에스컬레이션**.
