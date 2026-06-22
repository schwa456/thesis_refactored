# Thesis: Graph-RAG Schema Linking — Full Framework

## 역할 (이 세션의 위치)
이 세션은 프로젝트 **루트**에서 실행되는 **Full Framework / 오케스트레이터** 세션이다.
개별 모듈(Builder/Selector/Extractor/Filter) 세션은 각 폴더에서 별도로 실행되며, 이 세션은:
- End-to-end 파이프라인 통합 및 실험 (전체 2×2×2 ablation)
- 논문 작성 (초록, 서론, 방법론, 실험, 결론)
- Cross-module 분석 및 시너지 해석
- EXPERIMENT_HISTORY / CATALOG / ID_MIGRATION 관리
- 모듈 간 인터페이스가 걸리는 이슈 조율

개별 모듈의 딥 구현은 해당 모듈 폴더에서 별도 세션을 열어 진행한다.

## 파이프라인 전체 구조
```
GraphBuilder → LocalPLMEncoder → EnsembleSeedSelector → AdaptivePCSTExtractor → AutoJoinKeys → XiYanFilter → (SQLGenerator)
```

## 실행 방법

### 1. GAT 학습
```bash
cd /home/hyeonjin/thesis_refactored
conda run -n base python src/train_gat.py --config configs/training/train_gat_config.yaml
```
- **Enriched features 버전**: `configs/training/train_gat_enriched_config.yaml`
  - Builder가 `database_description/*.csv`와 `tables.json`의 자연어명까지 포함해 노드 텍스트 생성
  - 캐시 파일명이 `_enriched` suffix로 분리됨 (`data/processed/train_enriched_graphs.pt`)
  - 체크포인트: `best_gat_enriched.pt` (config의 `checkpoint_name` 필드로 제어)
- **기본 체크포인트 경로**: `outputs/checkpoints/best_gat_model.pt`
- **Graph 캐시**: 최초 실행 시 `data/processed/{json_basename}_graphs.pt` 생성, 이후 재사용

### 2. 전체 파이프라인 실행 (실험)
```bash
conda run -n base python src/main.py --config <config_name>
```
- `<config_name>`은 `configs/` 하위 경로에서 `.yaml` 제외한 이름. 예:
  - `python src/main.py --config experiments/s03_gat_ensemble/a02_adaptive_pcst/s03_a02_03_xiyan_filter`
  - `python src/main.py --config experiments/abl/a05_filter_agentic/a05_01_adaptive_multi_agent`
- **Base config + experiment config** 병합 로드 (`configs/base_config.yaml` → 실험 config)
- **출력 경로** — configs 디렉토리 구조를 그대로 미러링:
  - Logs: `logs/<config_name>/` (예: `logs/experiments/s03_gat_ensemble/a02_adaptive_pcst/s03_a02_03_xiyan_filter/`)
  - Outputs: `outputs/<config_name>/` (predictions.jsonl, score_analysis_*.jsonl, output_*.jsonl, metrics.txt)
  - Summary 누적: `outputs/summary_all.csv`
- **3개 디렉토리가 동일 구조**: `configs/`, `outputs/`, `logs/`는 모두 같은 하위 경로 구조를 따른다

### 2-1. 복수 실험 일괄 실행 (중요)
**여러 실험을 순차/병렬로 돌려야 할 때는 반드시 `scripts/` 폴더에 실행 스크립트(.sh)를 만들어 한 번의 권한 승인으로 전체가 돌아가게 한다.** 매 실험마다 사용자의 허용을 받는 방식은 금지 — 사용자가 자리를 비운 사이에도 실험 체인이 멈추지 않아야 한다.

- 스크립트 위치: `scripts/run_<purpose>.sh` (예: `scripts/run_fk_steiner_sweep.sh`)
- **모든 스크립트는 `scripts/` 폴더에 위치** — 프로젝트 루트에 .sh 파일을 두지 않는다
- 스크립트 첫 줄에 `cd "$(dirname "$0")/.."` 로 프로젝트 루트로 이동
- 스크립트 내부에서 여러 config를 순차 실행, 실패해도 다음 실험으로 진행 (`|| true` 또는 에러 로깅)
- 백그라운드 실행 가능하도록 단일 명령으로 호출할 수 있는 형태 유지
- 실험 종료 후 3개 문서(HISTORY/CATALOG/ID_MIGRATION) 업데이트까지 스크립트에 포함하거나 명시적 가이드로 남길 것

### 3. 시각화 / 분석
- Streamlit 그래프 시각화: `bash scripts/run_visualizer.sh [port]` (default 8501)
- 8개 ablation 모델 비교 모드 + 전체 실험 단일 모드 지원 (카테고리 필터 포함)

### 주요 환경 제약
- **numpy < 2 필수** (pcst_fast 바이너리 호환성 — numpy 2.x에서 all-zeros 반환 버그)
- conda base env에서 실행
- pyvis, torch-geometric, sentence_transformers 의존

### 저장소 규칙 (NAS) — 중요
로컬 디스크는 포화에 가깝다. **체크포인트와 대용량 데이터는 반드시 NAS(`/SSL_NAS/peoples/khj/thesis/`)에 저장**한다.

- **체크포인트**: `/SSL_NAS/peoples/khj/thesis/checkpoints/`
  - 로컬 `outputs/checkpoints/`는 NAS 파일을 가리키는 **symlink 디렉토리**로 운영 중
  - 새 체크포인트 저장 시: 실제 파일은 NAS에 쓰고, 필요하면 `outputs/checkpoints/<name>.pt → NAS 경로` symlink를 걸어 기존 코드 경로 호환 유지
  - 예외: 학습이 **실시간으로 쓰고 있는** 체크포인트는 로컬에 두고, 학습 종료 후 NAS로 이동 + symlink 교체
- **데이터**:
  - 원본 BIRD 데이터: `/SSL_NAS/peoples/khj/thesis/train/`, `/SSL_NAS/peoples/khj/thesis/dev/`
  - Processed graph cache: enriched는 이미 NAS에 저장됨 (train.json과 같은 디렉토리). 기본 builder cache는 `data/processed/`에 있으나, **신규로 생성되는 대용량 캐시는 NAS로 보낼 것**
  - **BIRD dev 만 예외** — 로컬 SSD (`data/raw/BIRD_dev/`) 에 유지. NAS 포화/지연 상태에서 XiYan 필터의 DB 값 조회 (`sqlite_schema`) 가 `folio_wait_bit_common` 커널 스톨을 만드는 이슈가 2026-04-22 관측됨. dev 는 크지 않으므로 (~수 GB) 로컬 유지.
- **새 경로 추가 시**: 하드코딩된 `outputs/checkpoints/`나 `data/processed/`가 1GB 이상을 쓸 가능성이 있으면, NAS 경로로 작성하거나 symlink를 미리 건다
- **NAS 공간 확인**: 현재 여유 ~1.1T / 총 26T. 대용량 쓰기 전에 `df -h /SSL_NAS` 체크

## 디렉토리 구조 규칙 (중요)

`configs/`, `outputs/`, `logs/` 3개 디렉토리는 **동일한 하위 경로 구조**를 유지한다.

```
configs/experiments/s03_gat_ensemble/a09_topology_cost/s03_a09_01_topology_no_filter.yaml
outputs/experiments/s03_gat_ensemble/a09_topology_cost/s03_a09_01_topology_no_filter/
  logs/experiments/s03_gat_ensemble/a09_topology_cost/s03_a09_01_topology_no_filter/
```

- `config_parser.py`가 `--config` 인자의 **전체 경로를 보존**하여 outputs/logs 하위에 동일 구조로 디렉토리를 생성한다
- **outputs/ 루트에 실험 폴더를 직접 만들지 않는다** — 반드시 experiments/ 또는 baselines/ 하위에 위치
- 새 실험 추가 시: configs에 yaml 생성 → main.py --config로 실행하면 outputs/logs가 자동으로 같은 구조에 생성됨

## 실험 체계 (2×2×2 Ablation)
- 축 1: Seed Scoring — Cosine vs Ensemble (α·cos + (1−α)·GAT, α=0.85)
- 축 2: PCST — Basic (fixed θ=0.1) vs Adaptive (per-query P80)
- 축 3: Filter — None vs XiYan (LLM backbone: vLLM era `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` / **GLM era (2026-04-24~)** `zai-org/glm-4.7` via Elice ML API, OpenAI-compatible)
- 8 cells 중 vLLM era 최고: **#6 E+Basic+X `abl_ens_basic_xiyan` (R=0.8149, P=0.7597, F1=0.7863)** — Basic PCST가 넓게 포함 → XiYan이 정밀 pruning. Adaptive PCST+XiYan(#8)은 F1=0.6987로 낮음.
- **GLM era 전체 최고** (2026-04-24): `s04_stagewise_qcond_gat_basic_glm` (R=0.8438, P=0.8329, F1=0.8383) — vLLM era best 대비 ΔF1=+0.0506. [EXPERIMENT_HISTORY.md Wave 2 Proposal C GLM era kickoff](EXPERIMENT_HISTORY.md) 참조.

ID 체계: 2026-04-14 재정리 (b0/s01-s05/abl 접두어) + 2026-04-24 `_glm` suffix (GLM era backbone 전환). [EXPERIMENT_ID_MIGRATION.md](EXPERIMENT_ID_MIGRATION.md) 참조.

## 논문
- **학회**: 한국지능정보시스템학회 2026 춘계 학술대회 (Extended Abstract, cover + 3p)
- **초안**: [outputs/analysis/paper/paper_draft_abstract_intro.md](outputs/analysis/paper/paper_draft_abstract_intro.md)
- **II장**: Related Works / **III장**: Methodology / **IV장**: Experiments / **V장**: Conclusion
- 초록/서론에서 실험 수치는 간접적으로만 언급 (실험이 아직 최종이 아님)

## 실험 문서화 규칙
실험 진행 후 반드시 3개 파일을 함께 업데이트:
- [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md)
- [EXPERIMENT_CATALOG.md](EXPERIMENT_CATALOG.md)
- [EXPERIMENT_ID_MIGRATION.md](EXPERIMENT_ID_MIGRATION.md)

메트릭 표기: **Recall, Precision, F1 순서, 소수점 4자리**

### 단계별 분해 보고 규범 (2026-04-21 지도교수 G2)
실험 결과를 보고/문서화할 때 가능하면 **Selector / Extractor / Filter 3단계 cumulative R/P/F1** 표를 함께 제시. 근거: `planning/advisor_inputs/2026-04-21_qcondgat_detailed_analysis.md` §4 G2 + §10 Q3.
- **Cumulative 정의**: 각 행 = 파이프라인을 해당 stage까지 실행한 뒤의 최종 노드셋 대 gold로 측정.
- 단계 컬럼 예시: `Selector only` / `+ Extractor (no filter)` / `+ Filter (final)`
- 기존 anchor 실험(a03_09, a03_11, a03_13, a03_14, a03_17 등)이 같은 Selector/Extractor 세팅을 공유하면 cumulative 셀의 출처로 활용 가능.
- 해당 셀이 직접 측정되지 않았으면 **"pending (analyzer reconstruction)"** 표기 + `output_*.jsonl`에서 재집계 요청을 analyzer 큐에 등록.
- 예시: [notebooks/analysis_results/03_extractor_pcst_steiner/steiner_backbone_stagewise_report.md](notebooks/analysis_results/03_extractor_pcst_steiner/steiner_backbone_stagewise_report.md)
- **예외**: 2×2×2 전체 매트릭스 등 셀 수가 많아 가독성이 떨어지는 경우는 final-stage 표 + 별도 stage-wise 부록으로 분리.

## 진행중 실험 현황 보고
- 쿼리당 평균 처리시간과 ETA를 함께 제시

## 핵심 분석 산출물 (루트 참조)
- [outputs/analysis/ablation/full_ablation_2x2x2.md](outputs/analysis/ablation/full_ablation_2x2x2.md) — 전체 매트릭스 비교
- [outputs/analysis/component_analysis/per_stage_failure_analysis.md](outputs/analysis/component_analysis/per_stage_failure_analysis.md) — PCST✗/Filter✗ 단계별 실패 분석
- [outputs/analysis/component_analysis/selector_analysis.md](outputs/analysis/component_analysis/selector_analysis.md) — Selector ROC-AUC / GAT 기여도
- [outputs/analysis/component_analysis/ensemble_contribution_analysis.md](outputs/analysis/component_analysis/ensemble_contribution_analysis.md)
- [outputs/analysis/ablation/difficulty_stratified_ablation.md](outputs/analysis/ablation/difficulty_stratified_ablation.md)

## 세션 분할 구조 (Orchestrator + Subsessions)
루트 세션 외에 전용 subsession이 있다. 각 세션은 해당 폴더의 `CLAUDE.md` 를 entry로 인식한다 (Claude Code가 cwd + 상위 경로를 자동 로드).

| 세션 | 위치 | 역할 |
|------|------|------|
| **Root (본 세션)** | `/home/hyeonjin/thesis_refactored/` | 실험 실행 · 모듈 조율 · HISTORY/CATALOG/ID_MIGRATION 관리 · 논문 |
| **Planner** | [planning/](planning/CLAUDE.md) | `EXPERIMENT_PLAN.md` 개정, phase 전환 결정, `DECISIONS.md` 로그. 코드/실험/분석 수행 금지 |
| **Analyzer** | [src/analysis/](src/analysis/CLAUDE.md) | logs/outputs 분석, 리포트 생성 (`notebooks/analysis_results/`). 계획/실행 금지 |
| **Module (Builder/Selector/Extractor/Filter)** | `src/modules/<module>/` | 모듈별 구현과 모듈 PLAN 반영 |

### 역할 분담 원칙
- **Planner**: 다음 실험을 결정하고 PLAN을 고친다. 숫자 필요시 → analyzer 요청, 실행 필요시 → 루트 요청.
- **Analyzer**: 데이터 읽고 리포트 작성. PLAN 변경 제안은 루트/planner로 에스컬레이션.
- **Root**: 실험 실행, 통합 ablation, HISTORY 3종 갱신, 크로스모듈 이슈 조율.
- **Module**: 모듈 내부 구현 + 모듈 PLAN 반영.

### 서브세션 호출이 필요할 때
서브에이전트로 위임 시 해당 세션의 CLAUDE.md를 읽도록 프롬프트에 명시:
```
먼저 /home/hyeonjin/thesis_refactored/<path>/CLAUDE.md 를 읽고 그 맥락을 지킨 채로 작업하라.
```
- 모듈 작업: `src/modules/<module>/CLAUDE.md`
- 분석 작업: `src/analysis/CLAUDE.md`
- 계획 개정: `planning/CLAUDE.md`

### 응답 말미 핸드오프 정리 (모든 세션 공통)
모든 세션(루트 · planner · analyzer · module)은 **한 턴의 응답을 마칠 때 "다음 핸드오프" 블록을 포함**한다. 목적: 멀티세션 체인에서 다음 행동 주체와 구체 지시가 한눈에 드러나도록.

**형식** (응답 말미에 위치, 헤더 `## 다음 핸드오프` 사용):
```
## 다음 핸드오프
- **대상 세션**: <root | planner | analyzer | module:<name> | user>
- **지시 프롬프트**: <그 세션의 cwd에서 `claude` 열고 그대로 붙여넣을 수 있는 문장>
- **근거/링크**: <이번 응답의 어떤 산출물/리포트/결정에서 파생됐는지>
```

**규칙**:
- 핸드오프가 불필요하면 `## 다음 핸드오프: 추가 세션 호출 불필요` 한 줄로 명시. 생략 금지 — 사용자가 상태 파악에 추가 질문을 하지 않도록.
- 여러 경로가 병렬로 필요하면 복수 블록으로 나열(analyzer 요청 + 루트 실행 등).
- 지시 프롬프트에는 **파일 경로·config 이름·데이터 출처를 명시** — "분석 해줘" 같은 모호한 문장 금지.
- 서브세션이 다른 서브세션에 직접 핸드오프할 수 있으나, **루트 CLAUDE.md / 실험 실행 / HISTORY 갱신이 필요하면 반드시 root 경유**.
- 세션별 기본 핸드오프 패턴은 각 CLAUDE.md 에 정의됨 ([planning/CLAUDE.md](planning/CLAUDE.md), [src/analysis/CLAUDE.md](src/analysis/CLAUDE.md)).

## 크로스모듈 이슈 해결 시 주의
- Selector의 top-k는 PCST에 게이트로 쓰이지 않음 (모든 노드 → PCST로 전달)
- PCST의 Prize는 `score − threshold`, Cost는 `edge_type`별 고정값 → 스케일 불일치 있음
- Filter✗ 실패 노드는 TP와 score 차이가 크지 않음 (제거되는 gold가 많음)
