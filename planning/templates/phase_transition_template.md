# Phase Transition — Phase <X> → Phase <Y>

> 특정 phase의 실험들이 완료되어 다음 phase로 이동할 때 Planner가 작성.

## 1. 완료된 Phase 요약
- **Phase**: (예: Phase A — Infrastructure)
- **포함 실험/작업**: (bullet)
- **완료 기준**: (각 작업이 어느 조건 만족했는지)

## 2. 메트릭 & 관찰
| 작업 | 지표 | 값 | 비고 |
|------|------|----|----|
| ... | R/P/F1 | 0.xxxx / 0.xxxx / 0.xxxx | ... |

- **핵심 관찰** (2~4개 bullet)

## 3. 다음 Phase 진입 조건 점검
- [ ] Dependency Graph의 상류 조건이 모두 충족됐는가
- [ ] 논문 주력 실험(int_04 등)에 필요한 인프라가 준비됐는가
- [ ] Closed topic 위반 여부 재확인

## 4. 다음 Phase 실행 순서 확정
- **최우선**: (실험 ID)
- **병렬 가능**: (GPU/리소스 충돌 없는 묶음)
- **후순위**: (조건부)

## 5. 리스크 체크
- 이번 phase에서 발견된 새 리스크
- PLAN §9 리스크 맵 갱신 필요 여부

## 6. 논문 서사 영향
- 방법론 스토리에 어떤 변화가 생겼는가
- 논문 초안에 즉시 반영 필요한 포인트

## 7. Planner 메모
- 다음 Planner 세션이 참고할 맥락
- DECISIONS.md 에 로그할 결정 요약
