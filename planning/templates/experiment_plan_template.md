# <실험 ID> — <짧은 이름>

> 모듈/통합 실험 제안 템플릿. Planner가 신규 실험을 PLAN에 추가하기 전 근거를 정리하는 용도.
>
> 채운 뒤 `EXPERIMENT_PLAN.md` 해당 섹션에 요약만 반영, 이 원본은 옵션으로 `planning/proposals/<id>.md`에 보관 가능.

## 1. 실험 개요
- **ID**: `abl_<module>_<feature>_<nn>` 또는 `int_<nn>_<short_desc>`
- **카테고리**: (builder / selector / extractor / filter / integration)
- **Anchor**: 어떤 베이스 설정 위에서 수행하는가 (예: `abl_a03_17` or `abl_ens_triplet_xiyan`)
- **변경 축 1개**: 한 번에 하나의 축만 바꾸는 원칙. 위배 시 근거 필수.

## 2. 가설
- **H1 (primary)**: 이 실험이 증명/반증하려는 주장 한 문장
- **예상 효과**: R / P / F1 중 어디에 얼마나 (+/-) 영향을 주리라 예상하는지

## 3. 근거
- 선행 리포트: `notebooks/analysis_results/<...>.md`
- 선행 실험: `abl_xxx` HISTORY §n
- 학술적 근거 (있으면): 논문 링크

## 4. 설계
- 구현 모듈 변경 요구: yes/no (있으면 어느 파일)
- Config 파일: `configs/experiments/.../<id>.yaml` 경로
- 예상 LLM 호출 수 / 토큰: (cost 추정)
- 예상 실행 시간 / 쿼리당 avg

## 5. Dependency
- 선결 필요한 실험 / 인프라
- 동시에 돌릴 수 없는 실험 (GPU/port 충돌)
- Cross-Module Matrix 갱신 필요 여부

## 6. 성공 기준
- 어떤 메트릭이 어느 수준을 넘으면 성공으로 간주
- 실패 시 다음 단계 (롤백? 파라미터 스위프?)

## 7. 문서화 체크리스트
- [ ] `EXPERIMENT_HISTORY.md` 엔트리 추가 (실행 후)
- [ ] `EXPERIMENT_CATALOG.md` 엔트리 추가
- [ ] `EXPERIMENT_ID_MIGRATION.md` 엔트리 추가 (명명 규칙 준수)
- [ ] `EXPERIMENT_PLAN.md` (루트) 관련 섹션 갱신
- [ ] 해당 모듈 `EXPERIMENT_PLAN_<module>.md` 갱신
- [ ] `planning/DECISIONS.md` 에 결정 근거 로그
