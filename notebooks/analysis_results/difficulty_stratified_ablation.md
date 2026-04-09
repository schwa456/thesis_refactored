# Difficulty-Stratified Ablation Analysis (BIRD Dev)

**생성일**: 2026-04-07
**목적**: BIRD Dev의 difficulty (`simple`/`moderate`/`challenging`)별로 본 연구 및 baseline의 Recall/Precision/F1을 분해하여, 난이도에 따른 구조적 기여와 병목을 분석한다.

## 0. 분석 방법

- **데이터 조인**: 각 실험 output의 `question_id`를 `data/raw/BIRD_dev/dev.json`의 `difficulty` 필드와 매칭
- **평가 지표**: per-query Recall / Precision → 난이도 그룹 내 산술 평균, F1은 `2PR/(P+R)`로 계산
- **평가 대상**: BIRD Dev 1,534 쿼리 (simple 925 / moderate 464 / challenging 145)
- **대조군**: 3 baselines (G-Retriever, LinkAlign, XiYanSQL) + 6 ablation configs

## 1. 난이도 분포

| Difficulty | N | 비율 |
|---|---|---|
| simple | 925 | 60.3% |
| moderate | 464 | 30.2% |
| challenging | 145 | 9.5% |
| **Total** | **1,534** | 100% |

## 2. 전체 결과 표

### 2.1 Baselines

| Method | Difficulty | Recall | Precision | **F1** |
|---|---|---|---|---|
| **G-Retriever** | simple | 0.7698 | 0.8064 | **0.7877** |
| | moderate | 0.7441 | 0.7605 | **0.7522** |
| | challenging | 0.7246 | 0.7437 | **0.7340** |
| **LinkAlign** | simple | 0.7288 | 0.7960 | 0.7609 |
| | moderate | 0.6448 | 0.7192 | 0.6800 |
| | challenging | 0.6296 | 0.7045 | 0.6649 |
| **XiYanSQL** | simple | 0.7192 | 0.7800 | 0.7484 |
| | moderate | 0.6275 | 0.6827 | 0.6540 |
| | challenging | 0.6318 | 0.6759 | 0.6531 |

### 2.2 본 연구 Progressive Ablation

| Config | Components | Difficulty | Recall | Precision | F1 |
|---|---|---|---|---|---|
| **b0_raw_pcst_baseline** | raw cos + Fixed PCST (θ=0.1) | simple | 0.9514 | 0.1504 | 0.2598 |
| | | moderate | 0.9469 | 0.1574 | 0.2699 |
| | | challenging | 0.9398 | 0.1979 | 0.3270 |
| **b1_adaptive_pcst** | raw cos + Adaptive PCST | simple | 0.7148 | 0.3677 | 0.4856 |
| | | moderate | 0.6360 | 0.3776 | 0.4739 |
| | | challenging | 0.5129 | 0.4078 | 0.4544 |
| **b2_ensemble** | Ensemble + Fixed PCST (θ=0.1) | simple | 0.9674 | 0.1165 | 0.2079 |
| | | moderate | 0.9692 | 0.1354 | 0.2377 |
| | | challenging | 0.9668 | 0.1917 | 0.3199 |
| **b_combined** | Ensemble + Adaptive PCST | simple | 0.7596 | 0.3367 | 0.4666 |
| | | moderate | 0.6853 | 0.3539 | 0.4668 |
| | | challenging | 0.5887 | 0.3918 | 0.4705 |
| **b4_single_filter** | Ensemble + Adaptive + SingleAgent | simple | 0.6172 | 0.7989 | 0.6964 |
| | | moderate | 0.5199 | 0.7602 | 0.6175 |
| | | challenging | 0.4500 | 0.7177 | 0.5532 |
| **b4_xiyan_filter (Full)** | Ensemble + Adaptive + XiYan | simple | 0.6563 | 0.8016 | **0.7217** |
| | | moderate | 0.5916 | 0.7842 | **0.6744** |
| | | challenging | 0.5261 | 0.7667 | **0.6240** |

## 3. 핵심 관찰

### 관찰 1. Precision 우위는 moderate/challenging에서 나타남

| Difficulty | Ours P | G-R P | Δ | 해석 |
|---|---|---|---|---|
| simple | 0.8016 | 0.8064 | −0.0048 | 동률 |
| moderate | **0.7842** | 0.7605 | **+0.0237** | 본 연구 우위 |
| challenging | **0.7667** | 0.7437 | **+0.0230** | 본 연구 우위 |

→ **핵심 메시지**: 본 연구의 구조적 기여(Adaptive PCST + XiYan Filter)는 **어려운 쿼리에서 두드러진다**. Simple 쿼리는 baseline도 이미 잘 처리하므로 마진이 없음.

### 관찰 2. Recall 열세는 난이도가 올라갈수록 확대

| Difficulty | Ours R | G-R R | Δ |
|---|---|---|---|
| simple | 0.6563 | 0.7698 | −0.1135 |
| moderate | 0.5916 | 0.7441 | −0.1525 |
| challenging | 0.5261 | 0.7246 | **−0.1985** |

→ Recall 격차가 **simple(−11%p) → challenging(−20%p)로 확대**. 복잡한 쿼리일수록 Filter의 보수성이 심화됨.

### 관찰 3. F1 열세도 challenging에서 최대

| Difficulty | Ours F1 | G-R F1 | Δ |
|---|---|---|---|
| simple | 0.7217 | 0.7877 | −0.066 |
| moderate | 0.6744 | 0.7522 | −0.078 |
| challenging | 0.6240 | 0.7340 | **−0.110** |

→ 정직하게: 모든 난이도에서 F1은 G-Retriever가 우위. challenging에서 격차 가장 큼.

### 관찰 4. Adaptive PCST의 Recall 손실이 challenging에 집중

**b2 → b_combined** (Fixed θ=0.1 → Adaptive P80) Recall 감소폭:

| Difficulty | b2 R | b_combined R | ΔR |
|---|---|---|---|
| simple | 0.9674 | 0.7596 | −0.2078 |
| moderate | 0.9692 | 0.6853 | −0.2839 |
| challenging | 0.9668 | 0.5887 | **−0.3781** |

→ Adaptive PCST의 P80 + max_prize_nodes=25 clamp가 **challenging 쿼리에서 가장 큰 Recall 손실**을 유발. 복잡한 쿼리는 많은 gold 컬럼을 요구하는데 25개 상한이 이를 억제.

### 관찰 5. Filter의 keep-rate는 난이도 의존성이 낮음

**b_combined → b4_xiyan_filter** Filter 단계 keep-rate (= Filter 후 Recall / Filter 전 Recall):

| Difficulty | Before (b_combined) | After (b4_xiyan) | Keep-rate |
|---|---|---|---|
| simple | 0.7596 | 0.6563 | 86.4% |
| moderate | 0.6853 | 0.5916 | 86.3% |
| challenging | 0.5887 | 0.5261 | **89.4%** |

→ 흥미롭게도 **Filter는 challenging에서 오히려 더 보존적**. 이미 PCST에서 많이 잃어 남은 후보가 적어졌기 때문으로 추정.

### 관찰 6. 병목은 PCST (특히 challenging)

각 난이도별 Recall 손실 분해 (100% 기준):

| Difficulty | Scoring → PCST | PCST → Filter | Final Recall |
|---|---|---|---|
| simple | **−20.8%p** | −10.3%p | 65.6% |
| moderate | **−28.4%p** | −9.4%p | 59.2% |
| challenging | **−37.8%p** | −6.3%p | 52.6% |

→ **challenging에서 PCST 손실이 Filter 손실의 6배**. 난이도별 개선의 핵심은 PCST 튜닝.

### 관찰 7. Baseline 간 난이도 민감도

F1의 simple → challenging 낙폭:

| Method | simple F1 | challenging F1 | 낙폭 |
|---|---|---|---|
| **G-Retriever** | 0.7877 | 0.7340 | **−0.054** |
| LinkAlign | 0.7609 | 0.6649 | −0.096 |
| XiYanSQL | 0.7484 | 0.6531 | −0.095 |
| **Ours (b4_xiyan)** | 0.7217 | 0.6240 | **−0.098** |

→ G-Retriever는 난이도에 가장 강건. LinkAlign/XiYanSQL/Ours는 비슷한 수준의 낙폭 (약 −0.09~−0.10). **본 연구는 난이도 민감도가 상대적으로 높다.**

## 4. 해석 및 발표 서사

### 4.1 긍정적 서사 — "구조적 기여는 어려운 쿼리에서 발현"
- Simple 쿼리는 baseline 수준으로도 충분 → 본 연구의 추가 가치가 적음
- Moderate/Challenging에서 **Precision +2.3%p 우위** → Adaptive PCST + XiYan Filter의 구조적 설계가 복잡한 쿼리에서 의미 있음
- S17 DB별 분석의 "thrombosis_prediction 등 복잡 스키마 DB 우위"와 **이중 triangulation**

### 4.2 정직한 한계 — "Recall 병목은 PCST에 집중"
- Adaptive PCST의 P80 + max_25 clamp가 challenging에서 −37.8%p 손실 유발
- Filter는 오히려 균일하게 동작
- **→ PCST 파라미터 재조정 또는 Recall-first 설계가 Future Works**

### 4.3 Future Works 구체화
- **(c) PCST 재조정**: percentile을 80 → 50/30 하향 또는 max_prize_nodes=50/80 상향하여 난이도별 Recall 균형 회복
- **Challenging-aware adaptive**: 쿼리 복잡도 추정치를 받아 PCST threshold를 동적으로 완화하는 extension

## 5. 발표자료 반영 제안

### Main 슬라이드 (권장)
- **Slide**: "Difficulty-Stratified Analysis"
  - Grouped bar chart 3개 (simple/moderate/challenging), x축 method, y축 F1 (또는 Recall/Precision 2개)
  - 하단 메시지: "Precision 우위는 moderate/challenging 구간에서 발현 (+2.3%p)"

### Appendix 슬라이드
- **App-?**: 전체 Method × Difficulty × Metric 표
- **App-?**: 난이도별 Recall 손실 waterfall (Scoring → PCST → Filter → Final)

## 6. 원본 데이터

- 평가 스크립트: `notebooks/analysis_results/difficulty_stratified_ablation.md` 상단 Python 블록 참조
- Output 파일 경로:
  - G-Retriever: `outputs/baselines/baseline_g_retriever/output_baseline_g_retriever.jsonl`
  - LinkAlign: `outputs/baselines/baseline_linkalign/output_baseline_linkalign.jsonl`
  - XiYanSQL: `outputs/baselines/baseline_xiyansql/output_baseline_xiyansql.jsonl`
  - b0~b4: `outputs/experiments/experiment_*/output_*.jsonl`
- 난이도 매핑 원천: `data/raw/BIRD_dev/dev.json` (question_id → difficulty)

## 7. 재현 코드 (Python)

```python
import json
from collections import defaultdict

with open('data/raw/BIRD_dev/dev.json') as f:
    dev = json.load(f)
qid_to_diff = {d['question_id']: d['difficulty'] for d in dev}

def load_metrics(path):
    per_diff = defaultdict(lambda: {'R':[], 'P':[]})
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            diff = qid_to_diff.get(o['question_id'], '?')
            per_diff[diff]['R'].append(o['recall'])
            per_diff[diff]['P'].append(o['precision'])
    result = {}
    for diff, m in per_diff.items():
        r = sum(m['R'])/len(m['R'])
        p = sum(m['P'])/len(m['P'])
        f1 = 2*r*p/(r+p) if (r+p)>0 else 0
        result[diff] = (r, p, f1, len(m['R']))
    return result
```
