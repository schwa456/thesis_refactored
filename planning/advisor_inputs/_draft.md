<!-- ══════════════════════════════════════════════════════════════════════════
     ACTIVE DRAFT — advisor input staging file
     ──────────────────────────────────────────────────────────────────────
     ▸ 사용자: §1~§3 만 편집하세요 (아래 placeholder 위에 그대로 덮어쓰기).
     ▸ 완료 후 planner에 "피드백 수렴해줘" 같은 신호만 주시면:
         1. 내용을 `planning/advisor_inputs/<YYYY-MM-DD>_<topic>.md` 로 이관
         2. §4~§14 해석/diff/에스컬레이션/DECISIONS 초안 채우기
         3. 이 파일을 pristine 상태로 복원 (다음 미팅용)
     ▸ 참조 원본: planning/templates/advisor_input_template.md
     ▸ 이 파일 자체를 rename 하거나 다른 경로로 옮기지 마세요 — path 고정.
     ▸ Last reset: 2026-04-21 (이전 승격: `2026-04-21_qcondgat_detailed_analysis.md`)
════════════════════════════════════════════════════════════════════════ -->

# 지도교수 의견 수렴 — <YYYY-MM-DD> <짧은 주제>

> Planner 세션이 **지도교수님의 피드백**을 받아 PLAN 개정으로 전환할 때 사용하는 DRAFT.
>
> ## 핵심 전제 (2026-04-21 기준)
> **지도교수님께는 루트 `EXPERIMENT_PLAN.md` 전체가 공유되어 있지 않다.**
> 현재 인지 범위는 **Query-Conditioned GAT + BCE-Loss 학습 + Over-smoothing 진단 까지** 
> (2026-04-10 5 아이디어 + 2026-04-21 미팅까지 누적).
>
> 따라서 피드백은 **브리핑된 범위에 한정**되어 들어온다:
> - **§4 브리핑 내 직접 영향**: 공유된 아티팩트(코드/실험/문서)에 대한 교수님의 요구
> - **§5 Scope gap (unbriefed PLAN 파급)**: planner가 해석하는 PLAN 전반 영향
>
> 이 두 층을 분리하는 것이 본 템플릿의 핵심.
>
> ## 레퍼런스
> - 이전 미팅 분석: [advisor_meeting_ideas_analysis.md](../../notebooks/analysis_results/advisor_meeting_ideas_analysis.md) (2026-04-10, 5 아이디어)
> - QC-GAT 구현 기록: [query_conditioned_training.md](../../notebooks/analysis_results/query_conditioned_training.md)
> - 직전 수렴본: [2026-04-21_qcondgat_detailed_analysis.md](2026-04-21_qcondgat_detailed_analysis.md)
> - 참조 템플릿: [../templates/advisor_input_template.md](../templates/advisor_input_template.md)

---

## 1. 메타 & 브리핑 ledger (사용자 기입)

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
- **Query-Conditioned GAT 기반 BCE Loss 학습 결과** (2026-04-21 공유, 승격)
- **GAT 모델 Over-smoothing 진단 보고** (2026-04-21 공유, 승격)
- 논문 초안 공유 여부: no

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

<!-- ═══════════ 아래부터는 planner (Claude) 가 자동으로 채웁니다 ═══════════ -->
<!-- 사용자께서는 §1~§3 편집 완료 후 아무 신호(예: "수렴해줘") 만 주시면 됩니다 -->

## 3. 의견 분류 (planner 기입)

| 의견 # | 라벨 | 한 줄 해석 |
|--------|------|-----------|
| 1 | directive / suggestion / question / warning / story | |
| 2 | | |

---

## 4. 브리핑 범위 내 직접 영향 (planner 기입)

| 의견 # | 대상 아티팩트 | 요구 변경 | 강도 (대/중/소) |
|--------|--------------|----------|---------------|
| 1 | | | |

---

## 5. Scope gap — Unbriefed PLAN 파급 영향 (planner 기입) — **핵심**

| 의견 # | 파급되는 PLAN 요소 (unbriefed) | 파급 이유 | 반영 방향 |
|--------|-------------------------------|----------|----------|
| 1 | | | |

---

## 6. 기존 PLAN과의 충돌 / 정합성 체크 (planner 기입)

- [ ] PLAN §6 닫힌 주제 와 충돌?
- [ ] PLAN §2 Dependency Graph 와 충돌?
- [ ] 논문 주력 (int_04) 와 정렬?
- [ ] Phase 순서 어김?
- [ ] 메트릭 표기 rule 충돌?
- [ ] 최신 2×2×2 결론 (#6 E+Basic+X 우세) 과 충돌?
- [ ] 실행 중/큐잉 실험 영향?
- [ ] 모듈 PLAN 중복?

---

## 7. PLAN 개정 초안 (proposed diff) — planner 기입

*(해당 없는 섹션은 삭제)*

### 7.1 §1 Cross-Module Matrix
### 7.2 §3.1 Synergy Grid
### 7.3 §4 Phase 우선순위
### 7.4 §5 논문 매핑
### 7.5 §6 닫힌 주제
### 7.6 §9 리스크 맵

---

## 8. 신규 실험 제안 (planner 기입)

---

## 9. 에스컬레이션 필요 항목 (planner 기입)

| 대상 세션 | 요청 내용 | 권장 프롬프트 |
|----------|----------|-------------|

---

## 10. 교수님께 재확인 필요한 사항 (planner 기입)

---

## 11. 다음 브리핑 후보 (planner 기입)

---

## 12. 결정 요약 (DECISIONS.md 엔트리 초안)

---

## 13. 수용 상태

- [ ] adopted / partially-adopted / pending-clarification / deferred / rejected-with-reason

---

## 14. 문서화 체크리스트

- [ ] 본 원본: `planning/advisor_inputs/<YYYY-MM-DD>_<topic>.md` 저장
- [ ] `EXPERIMENT_PLAN.md` 갱신 요청 (루트 세션)
- [ ] `planning/DECISIONS.md` 엔트리 추가
- [ ] 모듈 PLAN 에스컬레이션
- [ ] 신규 실험 제안서 (`planning/proposals/<id>.md`)
- [ ] HISTORY/CATALOG/ID_MIGRATION 3종 환기
- [ ] 논문 초안 수정 여부 루트 공유
- [ ] Analyzer 요청 큐 등록
- [ ] **template `§1.2 공유된 범위` 갱신**: 이번 미팅에서 새로 공유한 내용을 승격
