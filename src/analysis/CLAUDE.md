# Analyzer (Logs & Results Analysis) 세션

> **루트 CLAUDE.md 참조 (읽기 전용)**: 프로젝트 전역 규칙(실험 실행 방법, 디렉토리 구조, 문서화 규칙)은 [/home/hyeonjin/thesis_refactored/CLAUDE.md](/home/hyeonjin/thesis_refactored/CLAUDE.md)를 먼저 읽고 따른다. 루트 CLAUDE.md와 루트 EXPERIMENT_PLAN.md는 **수정하지 않는다** — 변경이 필요하면 루트 세션에 에스컬레이션.

## 이 세션의 위치
이 세션은 **사후 분석 전용 (post-hoc analyzer)**. 실험 산출물(logs/outputs)과 모델 체크포인트를 읽어 **메트릭·통계·시각화·리포트**를 만드는 역할만 맡는다. 파이프라인/모듈 구현, 실험 실행, 설정 파일 변경은 범위 밖이다.

## 책임 영역 (이 세션이 수행하는 일)

### 입력 (읽기 전용)
- `logs/<config_name>/` — per-stage JSONL, filter/extractor 프로파일링, train_step.jsonl (GAT 학습 로그)
- `outputs/<config_name>/` — predictions.jsonl, score_analysis_*.jsonl, output_*.jsonl, metrics.txt, profiling_*.jsonl
- `outputs/summary_all.csv` — 모든 실험 누적 메트릭 (실험간 비교에 우선 참조)
- `outputs/checkpoints/` (NAS symlink) — 학습된 GAT weights, frozen layer outputs
- `data/processed/` — graph caches (분석용 그래프 통계)
- `EXPERIMENT_HISTORY.md`, `EXPERIMENT_CATALOG.md`, `EXPERIMENT_ID_MIGRATION.md` — 실험 메타데이터 조회

### 코드 작성 위치
- `src/analysis/` — **재사용 가능한 분석 스크립트** (이 세션의 주 작업 디렉토리)
  - 기존 자산: `per_stage_failure_analysis*.py`, `gat_bottleneck_analysis*.py`, `selector_score_analysis.py`, `fk_steiner_percentile_sweep.py`, `visualize_graph_app.py`, `b4b_dashboard.py`, `phase{1,2,3}_analysis.py` 등
  - 신규: `analyze_<topic>.py` 명명 규칙. 큰 스크립트는 한 파일, 반복 유틸은 공유 모듈로.
- `notebooks/` — 탐색적 .ipynb (대화형 분석). 정적 리포트는 `.md`로 `notebooks/analysis_results/`에.

### 산출물 저장 위치
- `notebooks/analysis_results/*.md` — **정적 리포트의 표준 위치**. 메트릭 표기 = Recall, Precision, F1 순, 소수점 4자리 (memory rule).
- `notebooks/analysis_results/*.csv` — 중간 집계 (plot 재작성용)
- 기존 리포트 예시:
  - `full_ablation_2x2x2.md` (축별 ablation)
  - `per_stage_failure_analysis.md` (Selector/PCST/Filter 단계별 실패)
  - `selector_analysis.md` (ROC-AUC / GAT 기여도)
  - `difficulty_stratified_ablation.md`
  - `s06_bottleneck_*.md` (GAT residual stream 병목)
  - `fk_steiner_percentile_sweep.md`

### 금지 영역 (다른 세션으로 에스컬레이션)
- **모듈 구현 변경**: `src/modules/**`, `src/pipeline/**`, `src/train_gat*.py`, `src/main.py` → 해당 모듈 세션 또는 루트 세션
- **실험 실행**: `python src/main.py --config ...`, `scripts/run_*.sh` → 루트 세션이 담당
- **설정 변경**: `configs/` 아래 yaml 추가/수정 → 루트 or 해당 모듈 세션
- **EXPERIMENT_HISTORY / CATALOG / ID_MIGRATION 갱신**: 실험 실행 주체가 담당 (analyzer는 **참조만**)
- **루트 CLAUDE.md / EXPERIMENT_PLAN.md 수정**: 루트 세션 전용

예외: 분석 결과가 기존 메트릭을 정정해야 하는 경우(예: 집계 버그 발견), 문제를 발견한 데이터와 함께 **루트 세션에 보고**하고 직접 고치지 않는다.

## 주요 데이터 포맷 (표준화됨)

### Filter 로깅 (`filter_info`, 최근 표준화)
모든 filter가 `AgentUtils.build_filter_info()`를 통해 prefixed key로 기록:
- `filter_type`, `filter_time_s` (모든 filter에 존재)
- `filter_llm_calls`, `filter_tokens_in`, `filter_tokens_out`, `filter_cached_tokens`
- `filter_route` (AdaptiveDepth: `xiyan|reflection|bidirectional`)
- `filter_repair_attempted`, `filter_connectivity_valid`, `filter_pipeline_retry_attempts`
- StackedFilter: `stage_infos[]` (stage별 nested info), `stage_time_total_s`, `short_circuited_at`
- AdaptiveDepth: `inner_time_s`, route 분기 근거 (uncertainty signal)
- `src/main.py`의 `_aggregate_stage_telemetry()`가 이 키들을 집계 — `route_distribution`, `repair_successes`, `pipeline_retry_attempts_mean` 등으로 변환

### GAT training (`logs/<exp>/train/train_step.jsonl`)
Step별 payload (P1 로깅 구현):
- per-node-type BCE/InfoNCE loss, num_nodes, num_pos
- `gat_grad_norm`, `projector_grad_norm`, lr, step_time, `mem_alloc_gb`, `mem_peak_gb`
- `layer_stats.<layer_idx>.<node_type>.{norm_mean, norm_std, abs_mean, p50, p95, dead_ratio, delta_norm_mean}` (flattened)
- Skip vs out_lin norm ratio per layer, forward_time_s

### Per-stage telemetry (`output_*.jsonl`)
한 레코드 = 한 쿼리. 각 stage (builder/selector/extractor/filter) 정보 포함. Filter에는 위 `filter_info` block이 들어감.

## 표준 분석 축 (사전 정의된 관점)

분석 요청이 들어오면 **기존 축에 매핑**해서 확장하거나 새 축을 제안하기 전에 기존 리포트 재사용부터 검토한다.

1. **2×2×2 ablation 확장** — Seed(Cos/Ens) × PCST(Basic/Adaptive) × Filter(None/XiYan) + 추가 축(Reflection, Tiered, AdaptiveDepth, Retry)
2. **Per-stage failure attribution** — Gold 노드 기준, 어느 단계에서 드롭됐는지 분해 (Selector✗/PCST✗/Filter✗)
3. **Filter routing & cost** — AdaptiveDepth의 route distribution, latency/token별 F1 Pareto
4. **GAT 학습 diagnostic** — layer norm 추세, dead_ratio, gradient 건강성, loss 수렴
5. **Difficulty stratified** — 난이도(easy/medium/hard)별 메트릭 분해
6. **Cross-experiment synergy** — `summary_all.csv` 기반 Builder×Filter 조합 시너지 탐색
7. **Selector score distribution** — GAT sigmoid, top-k gap, entropy → uncertainty proxy 품질

## 워크플로우 (권장 루틴)

1. **요청 파싱** → 기존 리포트에서 답 가능 여부 먼저 확인 (`notebooks/analysis_results/`)
2. **데이터 출처 확인** → `outputs/<exp>/` 구조, `summary_all.csv`, `logs/<exp>/` 읽기
3. **재사용 가능한 분석 스크립트가 있나?** → `src/analysis/` 검색
4. **신규 분석** → `src/analysis/analyze_<topic>.py` 작성, 산출물은 `notebooks/analysis_results/<topic>.md`
5. **리포트 포맷** — 메트릭 테이블 + 해석 + "어떤 실험 설정이 필요한지" 제안으로 종료
6. **메트릭 표기는 R, P, F1, 소수점 4자리** (절대 깨지 말 것)

## 자주 쓰이는 참조 경로

```
# 실험 메트릭 조회
outputs/summary_all.csv
EXPERIMENT_HISTORY.md

# 실험별 로그
logs/experiments/<path>/<exp>/           # per-stage JSONL, train_step.jsonl
outputs/experiments/<path>/<exp>/        # predictions, score_analysis, profiling, metrics.txt

# 분석 리포트
notebooks/analysis_results/*.md          # 정적 리포트
notebooks/analysis_results/*.csv         # 중간 집계

# NAS
/SSL_NAS/peoples/khj/thesis/checkpoints/ # 체크포인트 원본
/SSL_NAS/peoples/khj/thesis/{train,dev}/ # 원본 BIRD 데이터
```

## 협업 규칙 요약

- 읽기 전용 구간(모듈/pipeline 코드)은 **절대 수정하지 않는다**
- 분석 결과로 실험 설계가 바뀌어야 한다면 **루트 세션에 리포트 링크와 함께 제안**
- 실험 문서 3종 (HISTORY/CATALOG/ID_MIGRATION)은 **실험 실행자 책임** — analyzer는 읽기만
- 메모리 규칙 재확인: 메트릭 R/P/F1 4자리, ETA 보고, kill 전 확인
