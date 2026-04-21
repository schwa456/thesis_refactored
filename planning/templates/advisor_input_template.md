# 지도교수 의견 수렴 — <YYYY-MM-DD> <짧은 주제>

> Planner 세션이 **지도교수님의 피드백**을 받아 PLAN 개정으로 전환할 때 사용하는 템플릿.
>
> ## 핵심 전제 (2026-04-21 기준)
> **지도교수님께는 루트 `EXPERIMENT_PLAN.md` 전체가 공유되어 있지 않다.**
> 현재 인지 범위는 **Query-Conditioned GAT + BCE-Loss 학습 + Over-smoothing 진단 까지** — 
> 2026-04-10 5 아이디어 + 2026-04-21 미팅까지 누적된 브리핑 ledger.
>
> 따라서 피드백은 **브리핑된 범위에 한정**되어 들어온다:
> - **§4 브리핑 내 직접 영향**: 공유된 아티팩트(코드/실험/문서)에 대한 교수님의 요구
> - **§5 Scope gap (unbriefed PLAN 파급)**: planner가 해석하는 PLAN 전반 영향
>
> 이 두 층을 분리하는 것이 본 템플릿의 핵심.
>
> ## 사용 흐름 (Option B — draft 기반)
> **본 템플릿 파일은 pristine reference — 직접 편집 금지**. 편집은 항상 draft 에서.
>
> 1. **사용자**: `planning/advisor_inputs/_draft.md` 를 열어 **§1~§3만** 편집 (원문 그대로 기록, 해석 금지).
> 2. **사용자 → planner**: "피드백 수렴해줘" 같은 신호.
> 3. **Planner (자동)**:
>    - a. `_draft.md` 읽기
>    - b. `planning/advisor_inputs/<YYYY-MM-DD>_<topic>.md` 신규 생성 (§1~§3 그대로 + §4~§14 채움)
>    - c. `_draft.md` 를 **본 템플릿 기준으로 pristine 리셋** (다음 미팅용)
>    - d. `planning/DECISIONS.md` 엔트리 추가
>    - e. 필요 시 모듈/루트 세션 에스컬레이션 프롬프트 제시
>
> ## Planner 유지 책임
> - 브리핑 ledger(§1.2) 의 "공유된 범위" 를 누적 갱신 (**template 의 default 도 함께 개정**) — 새 미팅에서 공유된 내용이 있으면 다음 _draft 리셋 시 반영.
> - 요점은 `EXPERIMENT_PLAN.md` 에 반영(루트 세션 위임), 결정 요약은 `DECISIONS.md` 에 로그.
>
> ## 레퍼런스
> - 이전 미팅 분석(analyzer 관점): [advisor_meeting_ideas_analysis.md](../../notebooks/analysis_results/advisor_meeting_ideas_analysis.md) (2026-04-10, 5 아이디어)
> - Query-Conditioned GAT 구현 기록: [query_conditioned_training.md](../../notebooks/analysis_results/query_conditioned_training.md) (아이디어 5)
> - 관련 템플릿: [experiment_plan_template.md](experiment_plan_template.md), [phase_transition_template.md](phase_transition_template.md)

---

## 1. 메타 & 브리핑 ledger (사용자/planner 기입)

### 1.1 미팅 메타
- **날짜**: YYYY-MM-DD
- **형식**: (정기 미팅 / 이메일 / 구두 / 서면 피드백 / 논문 리뷰 / 발표 / 기타)
- **세션 길이**: (분 단위, 선택)
- **주제 한 줄 요약**:
- **관련 이벤트**: (학회 발표 / 예심 후속 / 중간 보고 / 그 외)
- **현재 PLAN phase (internal)**: (A/B/C/D/E — `EXPERIMENT_PLAN.md §4`)

### 1.2 지도교수 인지 범위 (브리핑 ledger) — **중요**
> 피드백이 어떤 맥락 위에서 나왔는지 명시. **PLAN 전체가 공유된 것이 아님**.
> 이 섹션이 §4 ↔ §5 분리의 근거.

**이번 미팅 직전까지 공유된 범위** (default — 필요시 수정):
- 2026-04-10 예심 후 5 아이디어 제시 (prior 분석: `advisor_meeting_ideas_analysis.md`)
- 아이디어 5 (Query-Conditioned GAT) 구현: 방안 A concat / 방안 B super node
- 관련 실험 결과 (구두/수치 공유 수준): concat / super node 방안의 R/P/F1 vs baseline
- Query-Conditioned GAT 기반 BCE Loss 학습 결과 (2026-04-21 공유)
- GAT 모델 Over-smoothing 진단 보고 (2026-04-21 공유)
- 논문 초안 공유 여부: (yes/no, 어느 섹션까지)

**아직 공유되지 않은 PLAN 영역** (default — 필요시 수정):
- 9 제안 통합 로드맵 (PLAN §1 Cross-Module Matrix) 전체
- Neurosymbolic 3-layer 프레임 (B-III / S-V / E-III / FL-III)
- int_04 논문 주력 결과 후보 지정
- 최신 2×2×2 재측정 결론 (#6 E+Basic+X, R=0.8149/P=0.7597/F1=0.7863 우세)
- SteinerBackbone 정의 및 기존 `abl_a03_15/18` 결과
- s06 over-smoothing 처방 ablation (PairNorm 등 b0~b5)
- Phase A~E 우선순위 체계
- 그 외:

**이번 미팅에서 새로 공유한 내용**: (있으면 기록 — 다음 미팅부터 "공유된 범위"로 승격)
-

### 1.3 피드백 대상
- **이번 피드백의 1차 대상**: (예: "Query-Conditioned GAT 방안 A 구현 결과", "방안 B Super Node 설계", "그 외")
- **관련 실험 ID**: (HISTORY §n / abl_xxx / int_xx)
- **관련 코드**: (파일 경로 — 예: `src/models/gat_network.py`, `src/train_gat.py`)
- **관련 문서**: (notebooks/analysis_results/… / paper_draft_…)

---

## 2. 교수님 의견 원문 (raw capture) — 사용자 기입

> **원칙**: 의역 금지, 가능한 한 원문(또는 near-verbatim). 해석은 §3 이후에서 planner가 수행.
> 원문이 모호해도 그대로 두고 §10에 재확인 질문으로 남긴다.

### 의견 1
> "..." (직접 인용 또는 near-verbatim)

- **교수님이 제시한 맥락/근거**: (과거 사례, 참고 논문, 반례 등)
- **함께 제기된 질문**: (있으면)

### 의견 2
> "..."

- **맥락/근거**:
- **질문**:

*(의견 개수만큼 반복)*

### 그 외 논의
- 부차적 언급 / 잡담에서 나온 미결 주제 / 사담 중 연구 힌트 등

---

## 3. 의견 분류 (planner 기입)

| 의견 # | 라벨 | 한 줄 해석 |
|--------|------|-----------|
| 1 | directive / suggestion / question / warning / story | |
| 2 | | |

- **directive**: 명시적 지시 — 원칙적으로 반영 (단 §6 충돌 체크 후)
- **suggestion**: 권고 — 근거 데이터와 대조해 채택/보류 결정
- **question**: 물음 — 답변/데이터 필요. §9 에스컬레이션 or §10 재확인
- **warning**: 피해야 할 방향 — `PLAN §6 닫힌 주제` 에 추가 후보
- **story**: 논문 서사 (방법론 강조점, framing, 관련 연구) — 논문 섹션 매핑

---

## 4. 브리핑 범위 내 직접 영향 (planner 기입)

> 피드백이 **§1.2 에서 이미 공유된 아티팩트** (Query-Conditioned GAT 구현 등) 에 대해 요구하는 변경.
> PLAN 범위 아님 — 교수님이 직접 볼 수 있는 레벨의 변경.

| 의견 # | 대상 아티팩트 | 요구 변경 | 강도 (대/중/소) |
|--------|--------------|----------|---------------|
| 1 | (예: `src/models/gat_network.py` 방안 A concat) | (예: "query를 마지막 layer에도 주입하라") | |
| 2 | | | |

**이 섹션에서 나온 항목은 대개 `src/modules/selectors/EXPERIMENT_PLAN_selectors.md` 또는 해당 모듈 PLAN 에 반영됨 → §9 에스컬레이션으로 넘김**.

---

## 5. Scope gap — Unbriefed PLAN 파급 영향 (planner 기입) — **본 템플릿의 핵심**

> 지도교수님은 알지 못하시지만, 이 피드백이 **루트 PLAN 의 unbriefed 영역**에 어떻게 파급되는지.
> Query-Conditioned GAT 피드백이 Neurosymbolic 3-layer / int_04 / Phase 우선순위 등 PLAN 내부에 파급되는 연결고리를 planner가 해석.

| 의견 # | 파급되는 PLAN 요소 (unbriefed) | 파급 이유 | 반영 방향 |
|--------|-------------------------------|----------|----------|
| 1 | (예: Selector S-V Neurosymbolic L1 / int_04 주력 / §4 Phase B 우선순위) | (예: "query-conditioned가 GAT 기여를 확대하면 S-V λ 튜닝 가치 상승") | (예: Phase B 상단으로 이동 / 보류 / 추가 실험 제안) |
| 2 | | | |

**판단 원칙**
- 피드백이 unbriefed 실험의 전제를 **강화** → 우선순위 상향
- **약화/반증** → 해당 실험 보류 or 재검토 (DECISIONS 에 근거 기록)
- 피드백이 unbriefed 영역과 **직접 연관 없음** → 이 섹션 비워두거나 "영향 없음" 명시

**주의**: 여기서 planner가 상상한 파급은 교수님의 의사가 아님. 
강한 파급일수록 §10 재확인 질문 또는 §11 다음 브리핑 후보로 연결하여 다음 미팅에서 검증.

---

## 6. 기존 PLAN과의 충돌 / 정합성 체크 (planner 기입) — **필수**

- [ ] **PLAN §6 닫힌 주제** 와 충돌? 
  - 방안 A (Score-driven PCST cost) / 방안 B (BO) / Idea 2 (Product Cost) / Idea 4 (Component-Aware) 재탐색 요구?
  - 충돌 시: 신규 근거 여부 확인 → 없으면 §10 재확인 질문
- [ ] **PLAN §2 Dependency Graph** 와 충돌? (선결 미달 상류 건너뛰기)
- [ ] **논문 주력 (int_04)** 와 정렬? 서사 변경 필요?
- [ ] **Phase 순서** 를 어기는 요구? 어기면 DECISIONS에 명시적 근거 필수
- [ ] **메트릭 표기 rule (R/P/F1, 4자리)** 충돌? (보통 없음)
- [ ] **최신 2×2×2 결론 (#6 E+Basic+X 우세, F1=0.7863)** 과 충돌? 
  - 교수님이 이전 결과(예: Adaptive PCST 우세)를 전제로 피드백 주셨다면 재브리핑 필요 → §11
- [ ] **실행 중/큐잉된 실험 (예: a05_05~10 Qwen 큐, FK-Steiner hold)** 에 영향?
- [ ] **모듈 PLAN 중복**? 이미 `src/modules/*/EXPERIMENT_PLAN_*.md` 에 반영된 내용인지 확인

---

## 7. PLAN 개정 초안 (proposed diff) — planner 기입

> 실제 수정할 섹션별 before / after. 수정이 없는 섹션은 블록 자체 삭제.
> 이 diff는 planner 제안 — 실제 `EXPERIMENT_PLAN.md` 쓰기는 루트 세션이 수행.

### 7.1 §1 Cross-Module Matrix
- **Before**: …
- **After**: …

### 7.2 §3.1 Synergy Grid (신규 int_xx 실험)
- **Before**: …
- **After**: …

### 7.3 §4 Phase 우선순위
- **Before**: …
- **After**: …

### 7.4 §5 논문 매핑
- **Before**: …
- **After**: …

### 7.5 §6 닫힌 주제 (재개 허용 / 새로 닫기)
- **Before**: …
- **After**: …

### 7.6 §9 리스크 맵
- **Before**: …
- **After**: …

---

## 8. 신규 실험 제안 (있을 경우) — planner 기입

> 피드백이 새 실험을 요구하면 [experiment_plan_template.md](experiment_plan_template.md) 를 별도로 채워 
> `planning/proposals/<id>.md` 에 보관, 여기에는 링크만.

- [ ] 제안서 작성 완료? → `planning/proposals/<id>.md`
- **실험 ID**: (명명: `EXPERIMENT_ID_MIGRATION.md`)
- **Anchor**:
- **주요 의존성**:
- **예상 phase**:
- **예상 LLM/GPU cost**:

---

## 9. 에스컬레이션 필요 항목 — planner 기입

Planner 혼자 닫지 못하는 항목. 대상 세션과 권장 프롬프트 초안을 함께.

| 대상 세션 | 요청 내용 | 권장 프롬프트 (copy-paste용) |
|----------|----------|---------------------------|
| Root (orchestrator) | 실험 실행 / 논문 초안 수정 / PLAN 실제 쓰기 | "먼저 /home/hyeonjin/thesis_refactored/CLAUDE.md 를 읽고, `<config_name>` 실행. HISTORY/CATALOG/ID_MIGRATION 갱신." |
| Selector 세션 (가장 흔함 — QC-GAT 관련) | `EXPERIMENT_PLAN_selectors.md` 갱신 | "먼저 /home/hyeonjin/thesis_refactored/src/modules/selectors/CLAUDE.md 를 읽고, `EXPERIMENT_PLAN_selectors.md` 의 <섹션>을 …로 개정하라." |
| Builder / Extractor / Filter | 모듈 PLAN 반영 | (유사 포맷) |
| Analyzer 세션 | 근거 리포트 | "outputs/<path>/ 에서 <메트릭>을 <차원>별 분해 → notebooks/analysis_results/<topic>.md. 의도: PLAN phase X 결정." |

---

## 10. 교수님께 재확인 필요한 사항 — planner 기입

> 원문이 모호하거나 기존 PLAN과 충돌해 즉시 반영 불가한 항목. 다음 미팅/이메일 전까지 **보류**.

- 질문 1: "..." (왜 필요한지: ...)
- 질문 2: "..."

**재확인 전까지의 처리**: §7 diff 에서 `pending` 표기, `adopted` 금지.

---

## 11. 다음 브리핑 후보 (planner 기입) — **신규**

> 이번 피드백 덕분에 다음 미팅에서 **새로 공유하면 좋을 PLAN 영역**.
> 사용자가 교수님의 인지 범위를 확장할 때 판단 근거가 됨.

- [ ] 후보 1: (예: "Neurosymbolic 3-layer 개요" — 근거: 이번 피드백 §5 에서 파급 확인됨)
- [ ] 후보 2: (예: "2×2×2 재측정 결과 Basic+XiYan 우세" — 근거: 메트릭 framing 변화 설명 필요)
- [ ] 후보 3:

**공유 시점 권장**: (다음 정기 미팅 / 논문 초고 공유 시점 / 중간 보고 / 그 외)
**공유 형태**: (구두 / 슬라이드 / 노트북 공유 / 논문 초안 reviewer mode)
**공유 시 주의점**: (예: "int_04는 아직 실행 전 → 계획 단계로만 언급")

> 실제 공유가 이뤄지면 **다음 advisor_input 문서의 §1.2 "공유된 범위"** 에 반영 — ledger 누적.

---

## 12. 결정 요약 (DECISIONS.md 엔트리 초안) — planner 기입

아래 블록을 그대로 `planning/DECISIONS.md` 최상단에 붙여넣기. 의견이 여러 개면 의견별로 엔트리 분리 가능.

```markdown
## YYYY-MM-DD — <제목> (지도교수 의견 반영)

- **결정**: (PLAN 또는 브리핑 아티팩트의 어느 부분을 어떻게 바꿨는지 한 문장)
- **근거**: 지도교수 의견 — `planning/advisor_inputs/<YYYY-MM-DD>_<topic>.md` §2.<의견 #>
  + 브리핑 범위: §1.2 참조 (교수님 인지: Query-Conditioned GAT 수준)
  + 지지 데이터: `notebooks/analysis_results/<...>.md` (있으면)
- **영향 범위 (브리핑 내 직접)**: §4 / 모듈 PLAN (`selectors` 등) / 코드 변경
- **영향 범위 (Scope gap — PLAN 파급)**: §5 / 루트 PLAN §X 갱신 필요
- **에스컬레이션**: (대상 세션 + 권장 프롬프트 §9 참조)
- **추가 필요 분석**: (analyzer 요청)
- **다음 브리핑 후보**: (§11 참조)
- **교수님께 후속 질문**: (§10 참조)
```

---

## 13. 수용 상태 (planner 기입)

의견이 여러 개면 의견별로 별도 체크 행.

- [ ] **adopted**: 전체 그대로 반영 완료
- [ ] **partially-adopted**: 일부만 반영 (범위 명시: …)
- [ ] **pending-clarification**: §10 재확인 후 반영 — 보류
- [ ] **deferred**: 시점 연기 (언제까지: Phase X / <date>)
- [ ] **rejected-with-reason**: 반영 불가 (근거 필수: §6 충돌 / 데이터 부재 / 닫힌 주제)

**결정 시각**: YYYY-MM-DD

---

## 14. 문서화 체크리스트 (planner 기입)

- [ ] 본 원본 파일: `planning/advisor_inputs/<YYYY-MM-DD>_<topic>.md` 저장 완료
- [ ] `EXPERIMENT_PLAN.md` 해당 섹션 갱신 요청 (루트 세션 위임 — §9 에스컬레이션)
- [ ] `planning/DECISIONS.md` 엔트리 추가 완료
- [ ] 모듈 PLAN 수정 있으면 해당 모듈 세션에 에스컬레이션 (§9)
- [ ] 신규 실험이면 `planning/proposals/<id>.md` 작성 (§8)
- [ ] 실행 주체(루트)에 `HISTORY/CATALOG/ID_MIGRATION` 3종 갱신 규칙 환기
- [ ] 논문 초안(`notebooks/analysis_results/paper_draft_*.md`) 수정 필요 여부 루트 세션 공유
- [ ] Analyzer 요청 있으면 `DECISIONS.md` 말미 "Analyzer 요청 큐" 등록
- [ ] **다음 advisor_input 문서의 §1.2 에 "이번 미팅에서 새로 공유한 내용" 승격**

---

## 부록: "지도교수 인지 범위" 갱신 원칙

Planner 세션은 이 템플릿의 §1.2 를 누적적으로 관리한다.

1. **이번 미팅에서 새로 공유한 내용 (§1.2 말미)** 은 다음 미팅 전까지 "공유된 범위"로 승격.
2. 사용자가 미팅 외 경로(이메일/슬랙)로 PLAN 일부를 공유했다면, 그 사실도 다음 advisor_input §1.2 에 기록.
3. "공유된 범위" 가 누적되면서 PLAN 전체가 공개될 때까지 Scope gap(§5) 의 역할은 점점 줄어든다.
4. 반대로 **아직 공유되지 않은 영역이 많을수록 §5 의 planner 해석이 결정적** — 이 해석의 정합성을 DECISIONS.md 로 남겨 세션간 연속성 확보.
