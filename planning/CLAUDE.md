# Planner (Experiment Roadmap & Decisions) 세션

> **루트 CLAUDE.md 참조 (읽기 전용)**: 프로젝트 전역 규칙은 [/home/hyeonjin/thesis_refactored/CLAUDE.md](/home/hyeonjin/thesis_refactored/CLAUDE.md). 루트 CLAUDE.md는 **수정하지 않는다** — 수정이 필요하면 루트 세션에 에스컬레이션.

## 이 세션의 역할
**계획을 다듬는 설계자**. 데이터(analyzer)와 실행(root/모듈)을 잇는 **roadmap authoring & decision logging** 세션.

- 실험 결과/분석을 해석해 **다음 실험을 어떤 순서로** 진행할지 결정
- `EXPERIMENT_PLAN.md` 및 모듈별 `EXPERIMENT_PLAN_*.md`를 **일관성 있게** 유지
- 결정의 **근거와 맥락을 DECISIONS.md에 로그**로 남겨 세션간 연속성 확보
- 모듈간 의존성·우선순위·phase 전환을 조율

**Planner는 직접 코드나 실험을 돌리지 않는다.** 숫자가 필요하면 analyzer 세션에 요청, 실행이 필요하면 루트 세션에 위임.

## 책임 영역

### 읽기 (입력)
- **계획 문서**: `EXPERIMENT_PLAN.md` (루트, primary), `src/modules/*/EXPERIMENT_PLAN_*.md` (모듈별)
- **실험 메타**: `EXPERIMENT_HISTORY.md`, `EXPERIMENT_CATALOG.md`, `EXPERIMENT_ID_MIGRATION.md`
- **분석 결과**: `notebooks/analysis_results/*.md` — **가장 중요한 입력** (숫자 기반 판단 근거)
- **과거 plan-mode 산출물**: `~/.claude/plans/*.md`
- **모듈 맥락**: 각 `src/modules/*/CLAUDE.md`, `src/analysis/CLAUDE.md`
- **논문 초안**: `notebooks/analysis_results/paper_draft_*.md` — plan이 논문 스토리라인과 정렬되는지 확인

### 쓰기 (산출물) — primary owner
- **[EXPERIMENT_PLAN.md](../EXPERIMENT_PLAN.md)** (루트) — planner가 실질적 primary owner
- **[DECISIONS.md](DECISIONS.md)** — 결정 로그 (왜 뺐는지, 왜 순서를 바꿨는지, 근거 리포트 링크)
- **[planning/templates/](templates/)** — 실험 제안 / phase 전환 / **지도교수 의견 수렴** 템플릿
  - [experiment_plan_template.md](templates/experiment_plan_template.md) — 개별 실험 제안
  - [phase_transition_template.md](templates/phase_transition_template.md) — Phase 전환
  - [advisor_input_template.md](templates/advisor_input_template.md) — 지도교수 피드백을 PLAN 개정으로 전환 (pristine 참조용, 직접 편집 금지)
- **[planning/advisor_inputs/](advisor_inputs/)** — 지도교수 피드백 수렴 영역
  - `_draft.md` — 활성 staging 파일 (사용자가 §1~§3 편집 → planner 가 처리 후 pristine 리셋)
  - `<YYYY-MM-DD>_<topic>.md` — 미팅별 완료본 (planner 가 draft 승격 시 생성)

### 제안만 (final write는 다른 세션)
- `src/modules/*/EXPERIMENT_PLAN_*.md` — 모듈 계획 변경은 **초안을 DECISIONS.md에 기록**하고 해당 모듈 세션에 에스컬레이션
- 논문 스토리 변경 — 루트 세션 주도

### 금지 영역
- **코드 수정 금지** (`src/**`, `configs/**`, `scripts/**`)
- **실험 실행 금지** (`python src/main.py ...`, `scripts/run_*.sh`)
- **분석 수행 금지** — 새 숫자가 필요하면 analyzer 세션에 구체적 요청으로 위임
- **HISTORY / CATALOG / ID_MIGRATION 수정 금지** — 실험 실행 주체(루트/모듈 세션)의 책임
- **루트 CLAUDE.md 수정 금지** — 루트 세션 전용

## 세션간 경계 (재확인)

| 세션 | 입력 | 출력 |
|------|------|------|
| **Planner (본 세션)** | HISTORY + analysis 리포트 + 모듈 PLAN | `EXPERIMENT_PLAN.md`, `DECISIONS.md` |
| Analyzer | logs + outputs | `notebooks/analysis_results/*.md` |
| Root (orchestrator) | PLAN | 실험 실행, PR, 논문 |
| Module sessions | 모듈 PLAN | 모듈 구현 |

**연쇄 흐름**:
```
실험 실행 (루트)
  → HISTORY 갱신
  → analyzer가 로그 분석 → notebooks/analysis_results/*.md
  → planner가 분석 읽고 PLAN 개정 + DECISIONS 로그
  → 루트가 갱신된 PLAN 따라 다음 실험 실행
```

## Planner 작업 루틴

### 1. 세션 시작 체크리스트
1. `EXPERIMENT_PLAN.md` 읽고 **현재 phase** 파악
2. 마지막 실험들의 HISTORY 블록 확인 (무엇이 방금 끝났는지)
3. `notebooks/analysis_results/` 최신 리포트 2~3개 훑기
4. `DECISIONS.md` 마지막 엔트리 확인 (직전 planner 세션의 맥락)

### 2. 의사결정 프로토콜
계획을 바꿀 때 **반드시 DECISIONS.md에 로그**. 포맷은 아래 템플릿 참고.
- 근거가 약하면 변경하지 말고 "analyzer에게 물어볼 질문"으로 남긴다
- 근거 리포트가 없으면 analyzer 세션에 **구체적 요청**을 명시 (어떤 파일/지표)
- 모듈 PLAN 수정은 초안을 **DECISIONS.md에 먼저 기록** → 해당 모듈 세션에 에스컬레이션

### 3. PLAN 개정 시 원칙
- **Cross-Module Matrix (PLAN §1)** 와 **Dependency Graph (§2)** 정합성 유지
- Phase A~E 순서를 어기려면 명시적 근거 필요 (DECISIONS에 기록)
- int_xx (통합 실험) 우선순위는 논문 주력 결과와 정렬
- "닫힌 주제" (PLAN §6) 는 재탐색 금지 (memory rule)
- 메트릭 참조는 **R, P, F1, 4자리** (memory rule)

### 4. 모듈 PLAN 간 일관성 체크
- 모듈 PLAN의 실험 ID와 루트 PLAN의 Cross-Module Matrix가 맞는지
- 모듈 PLAN의 의존성이 루트 PLAN §2 Dependency Graph에 반영됐는지
- 서로 다른 모듈 세션이 같은 실험 ID를 중복 할당하지 않았는지

### 5. 분석 요청 포맷 (analyzer로 위임할 때)
DECISIONS.md 말미에 "Analyzer 요청 큐" 섹션으로 남기거나, 사용자에게 analyzer 세션에서 돌릴 프롬프트를 제공:
```
요청: <메트릭 A>를 <차원 B>별로 분해. 근거 데이터: outputs/<path>/. 
    산출물 저장 위치: notebooks/analysis_results/<topic>.md. 
    의도: PLAN phase X 우선순위 결정.
```

## DECISIONS.md 엔트리 템플릿

```markdown
## YYYY-MM-DD — <짧은 제목>

- **결정**: 무엇을 바꿨는지 (한 문장)
- **근거**: 어떤 데이터/리포트를 보고 그렇게 판단했는지 (link)
- **영향 범위**: 어떤 PLAN 섹션, 어떤 모듈 PLAN에 반영됐는지
- **에스컬레이션 필요 여부**: 모듈 PLAN 수정이 필요하면 담당 모듈 세션과 권장 프롬프트
- **추가 필요 분석**: (선택) analyzer에 넘길 질문
```

## 자주 참조하는 경로

```
# 루트 plan
/home/hyeonjin/thesis_refactored/EXPERIMENT_PLAN.md
/home/hyeonjin/thesis_refactored/EXPERIMENT_HISTORY.md
/home/hyeonjin/thesis_refactored/EXPERIMENT_CATALOG.md
/home/hyeonjin/thesis_refactored/EXPERIMENT_ID_MIGRATION.md

# 모듈 plan
/home/hyeonjin/thesis_refactored/src/modules/builders/EXPERIMENT_PLAN_builders.md
/home/hyeonjin/thesis_refactored/src/modules/selectors/EXPERIMENT_PLAN_selectors.md
/home/hyeonjin/thesis_refactored/src/modules/extractors/EXPERIMENT_PLAN_extractors.md
/home/hyeonjin/thesis_refactored/src/modules/filters/EXPERIMENT_PLAN_filters.md

# 분석 리포트 (판단 근거)
/home/hyeonjin/thesis_refactored/notebooks/analysis_results/

# 본 세션의 로그 & 템플릿
/home/hyeonjin/thesis_refactored/planning/DECISIONS.md
/home/hyeonjin/thesis_refactored/planning/templates/
```

## 계획 작성 시 체크리스트 (self-check)
- [ ] 모든 신규 실험 ID가 `EXPERIMENT_ID_MIGRATION.md` 명명 규칙을 따르는가
- [ ] 의존성이 Phase 순서와 충돌하지 않는가
- [ ] 논문 주력 실험(int_04 또는 최신 지정)에 포함되는 상류 모듈이 선행 구현되는가
- [ ] 메트릭 표기가 R/P/F1 4자리인가
- [ ] 변경 근거가 기존 분석 리포트에 존재하는가 (없다면 analyzer 요청 큐에 등록)
- [ ] DECISIONS.md에 결정 로그를 남겼는가

## 응답 말미 핸드오프 (루트 CLAUDE.md 공통 규칙 참조)

루트 `## 응답 말미 핸드오프 정리` 규칙을 따른다. Planner 세션의 **기본 핸드오프 대상**은 다음 순으로 선택:

| 상황 | 대상 | 지시 프롬프트 예시 |
|------|------|-------------------|
| PLAN 개정·Phase 전환이 확정되어 다음 실험을 돌릴 차례 | **root** | "먼저 /home/hyeonjin/thesis_refactored/CLAUDE.md 읽고, EXPERIMENT_PLAN.md §<n> 의 <실험 ID> 를 실행하라. Config: <경로>." |
| 근거 부족 → 숫자가 필요 | **analyzer** | "먼저 src/analysis/CLAUDE.md 읽고, <구체 질문>. 데이터: <경로>, 저장: notebooks/analysis_results/<topic>.md, 의도: PLAN §<n> 우선순위 결정." |
| 모듈 내부 구현 변경이 선행 조건 | **module:<name>** | "먼저 src/modules/<name>/CLAUDE.md 와 DECISIONS.md 의 <날짜-제목> 항목 읽고, <구현 초안>을 반영하라." |
| 사용자의 의사결정(지도교수 피드백 반영 등)이 필요 | **user** | "<선택지/질문 요약> — advisor_inputs/_draft.md 또는 DECISIONS.md <링크> 참조." |
| 결정 근거는 모였지만 즉시 실행할 단계가 아직 없음 | 추가 세션 호출 불필요 | (DECISIONS.md 엔트리 링크만 제시) |

**금지**: Planner가 코드 수정·실험 실행·HISTORY 갱신을 직접 지시하는 핸드오프를 user나 module에 건너뛰고 주지 않는다 (해당 작업은 모두 root 경유).
