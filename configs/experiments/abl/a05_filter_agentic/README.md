# a05 — Filter Module Agentic Refinement

Filter 모듈을 agentic하게 고도화하기 위한 12개 실험. Anchor: `a03_17`
(SuperNode Direct + Fixed PCST + XiYan, F1=0.6940). 각 실험은 anchor의 필터만
교체하거나 pipeline에 F5 retry loop를 추가한다.

## 실험 리스트

| ID | Filter | 비고 |
|----|--------|------|
| a05_01 | AdaptiveMultiAgentFilter (기존 scaffold) | Semantic+Structural+Skeptic |
| a05_02 | ReflectionFilter (F1, 1 iter) | Self-refine baseline |
| a05_03 | ReflectionFilter (F1, 3 iter) | Iteration depth |
| a05_04 | VerifierFilter (F2) | XiYan + NL unit tests |
| a05_05 | TieredBidirectionalAgent (F3, no tools) | Tier-1/2 prompt만 |
| a05_06 | TieredBidirectionalAgent (F3, full tools) | ★ 핵심 기여 |
| a05_07 | AdaptiveDepthFilter (F4) | 신뢰도 기반 분기 |
| a05_08 | StackedFilter (F3→F2) | 상한 |
| a05_09 | F3 + 파이프라인 F5 retry (K=2) | Extractor 역피드백 |
| a05_10 | F4 + F5 retry | Hard query 선택적 재시도 |
| a05_11 | F3 + GPT-4o-mini | Backbone 민감도 vs a05_06 |
| a05_12 | F4+F5 + GPT-4o-mini | Backbone 민감도 vs a05_10 |

## 실행 전 준비

1. **vLLM 서버 기동** (Qwen 실험용):
   ```bash
   # Qwen3-Coder-30B-A3B-Instruct-FP8 on localhost:8000
   # 프로젝트 기존 서버 기동 절차 그대로
   ```

2. **GPT-4o-mini 실험 (a05_11, a05_12)**: `.env`에 `OPENAI_API_KEY` 설정.
   `api_key: null`인 config는 APIClient가 `os.getenv("OPENAI_API_KEY")`로
   자동 폴백한다.

3. **GAT 체크포인트**: a03_17과 공유 → `outputs/checkpoints/best_gat_query_supernode_direct.pt`

## 실행

전부 실행:
```bash
bash run_a05_filter_agentic.sh
```

일부만 실행:
```bash
bash run_a05_filter_agentic.sh a05_02 a05_06 a05_09
```

## 결과 수집 후 필수 작업 (memory rule)

각 실험 종료 후 다음 문서를 **한 번에** 갱신:
- `EXPERIMENT_HISTORY.md` — 결과 요약
- `EXPERIMENT_CATALOG.md` — 하이퍼파라미터 블록
- `EXPERIMENT_ID_MIGRATION.md` — ID 매핑 테이블

지표 표기: Recall / Precision / F1 (모두 소수점 4자리).

## 구현 참조

- 필터: [src/modules/filters/](../../../../src/modules/filters/)
  - `reflection_filter.py` — F1
  - `verifier_filter.py` — F2
  - `bidirectional_agent_filter.py` — F3
  - `adaptive_depth_filter.py` — F4
  - `stacked_filter.py` — a05_08용 체이닝
- 프롬프트: [src/prompts/filter.md](../../../../src/prompts/filter.md)
  - `reflection_critique`, `reflection_revise`
  - `verifier_unit_tests`, `verifier_check`
  - `restore_agent`, `extraction_retry_hint`
- 파이프라인:
  - [src/pipeline/schema_linking.py](../../../../src/pipeline/schema_linking.py)
  - F3용 `(tier2_pool, gat_scores, metadata)` 추가 전달
  - F5용 `_apply_retry_strategy`, `_restore_extractor_params` + retry loop
- 그래프 도구: [src/modules/filters/tools/graph_tools.py](../../../../src/modules/filters/tools/graph_tools.py)
