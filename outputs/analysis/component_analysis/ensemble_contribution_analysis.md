# Ensemble Seed Selection — Contribution Analysis

**생성일**: 2026-04-07
**목적**: 본 연구의 핵심 구성요소 중 하나인 **Score Ensemble** (α·cosine + (1−α)·GAT, α=0.85)의 기여도를 정량화하고, **어떤 조건에서 효과적인지**를 규명한다.

---

## 0. 분석 방법

### 비교 페어
Ensemble의 순수 효과를 측정하기 위해 **다른 모든 설정을 고정한 채 Seed Scoring만 교체**한 쌍을 비교:

| Pair | Seed Scoring | PCST | Filter |
|---|---|---|---|
| **Pair-A** (Fixed PCST 하) | b0: raw cos → b2: Ensemble | Fixed PCST (θ=0.1) | None |
| **Pair-B** (Adaptive PCST 하) | b1: raw cos → b_combined: Ensemble | Adaptive PCST (P80) | None |
| **Pair-C** (Full pipeline) | ??? → b4_xiyan: Ensemble | Adaptive PCST | XiYan | ⚠ **미실행** |

> **⚠ 결정적 공백**: `raw cos + Adaptive + XiYan` run이 없어 **Full pipeline에서의 Ensemble 효과를 직접 측정할 수 없음**. 현재 결론은 Pair-A, B에 한정된 관찰이며 Filter 단계의 상호작용은 추정.

---

## 1. 전체 평균 결과

### Pair-A: Fixed PCST 하에서의 Ensemble 효과

| Subset | N | ΔRecall | ΔPrecision | **ΔF1** |
|---|---|---|---|---|
| ALL | 1534 | **+0.019** | −0.028 | **−0.041** |
| simple | 925 | +0.016 | −0.034 | −0.052 |
| moderate | 464 | +0.022 | −0.022 | −0.032 |
| challenging | 145 | +0.027 | −0.006 | −0.007 |

**관찰**: Fixed PCST 환경에서 Ensemble은 **Recall 미세 상승, Precision 감소, F1 순손실**.
→ Fixed threshold가 noise에 관대한 상태에서는 Ensemble의 GAT score가 오히려 false positive를 추가하는 방향으로 작용.

### Pair-B: Adaptive PCST 하에서의 Ensemble 효과

| Subset | N | ΔRecall | ΔPrecision | **ΔF1** |
|---|---|---|---|---|
| ALL | 1534 | **+0.049** | −0.027 | −0.012 |
| simple | 925 | +0.045 | −0.031 | −0.019 |
| moderate | 464 | +0.049 | −0.024 | −0.007 |
| **challenging** | **145** | **+0.076** | −0.016 | **+0.016** |

**관찰**:
- Ensemble의 **Recall 기여가 Pair-A의 2.5배** (+0.019 → +0.049)
- F1 손실이 Pair-A 대비 **1/3 수준**으로 축소 (−0.041 → −0.012)
- **Challenging 난이도에서 F1이 최초로 양수 (+0.016)**
- Recall 기여가 **난이도에 비례하여 증가**: +0.045 → +0.049 → +0.076

**→ Ensemble은 Adaptive PCST + Challenging 조합에서 진짜 효과가 발현됨.**

---

## 2. 난이도별 Ensemble 순효과 (요약 표)

| Difficulty | Pair-A ΔF1 (Fixed) | Pair-B ΔF1 (Adaptive) |
|---|---|---|
| simple | −0.052 | −0.019 |
| moderate | −0.032 | −0.007 |
| challenging | −0.007 | **+0.016** ← 유일한 양수 |

**패턴**: 
- **PCST가 정교할수록** (Fixed → Adaptive) Ensemble의 F1 손실이 축소
- **쿼리가 어려울수록** Ensemble의 효과가 긍정적으로 전환

→ Ensemble은 **"Adaptive PCST와 결합해야, 그리고 복잡한 쿼리에서" 빛을 발하는 컴포넌트**.

---

## 3. DB별 Ensemble 효과 (Adaptive PCST 기준)

| DB | N | ΔRecall | ΔPrecision | ΔF1 | 해석 |
|---|---|---|---|---|---|
| **toxicology** | 145 | +0.091 | +0.051 | **+0.091** | ★ 유일하게 Recall/Precision/F1 모두 동반 상승 |
| student_club | 158 | +0.084 | +0.004 | +0.024 | 양쪽 기여 |
| financial | 106 | +0.077 | −0.011 | +0.012 | Recall 주도 양수 |
| card_games | 191 | +0.067 | +0.005 | +0.011 | 소폭 양수 |
| thrombosis_prediction | 163 | +0.046 | −0.017 | −0.013 | Recall ↑, F1 미세 음수 |
| california_schools | 89 | −0.003 | −0.010 | −0.013 | 무효 |
| debit_card_specializing | 64 | +0.012 | −0.066 | −0.011 | P 손실 |
| formula_1 | 174 | +0.046 | −0.041 | −0.035 | P 손실 주도 |
| superhero | 129 | +0.018 | −0.075 | −0.044 | P 큰 손실 |
| codebase_community | 186 | +0.018 | −0.054 | −0.058 | P 큰 손실 |
| **european_football_2** | 129 | +0.048 | **−0.116** | **−0.134** | ★ 가장 큰 F1 손실 |

### 관찰
- **Winner DBs (4개)**: toxicology, student_club, financial, card_games — F1 동반 상승
- **Neutral (1개)**: california_schools
- **Loser DBs (6개)**: europen_football_2, codebase_community, superhero, formula_1, debit_card_specializing, thrombosis_prediction

→ **Ensemble 효과는 DB에 따라 극적으로 갈림**. 특히 `european_football_2`는 Precision이 11.6%p 급락.

### 가설: Ensemble이 잘 작동하는 DB의 특징
- **toxicology**: 화학 용어 중심, 컬럼명이 도메인 특화 → cosine alone이 약한 영역에서 GAT의 구조 정보가 보완
- **student_club**: 정규화된 클럽/멤버 관계 → 구조적 근접성이 중요
- **financial**: 다중 테이블 조인 필요 → FK 기반 구조 점수 유용

### 가설: Ensemble이 실패하는 DB의 특징
- **european_football_2**: 플레이어/경기/클럽 등 고도로 정규화된 다대다 관계 → GAT가 과도하게 확장된 후보를 제시
- **codebase_community, superhero**: 범용 용어 (user, hero 등) → GAT가 관련 없는 컬럼까지 구조적으로 연결

---

## 4. Query-level 개선/악화 분석 (Adaptive PCST)

| 결과 | 쿼리 수 | 비율 |
|---|---|---|
| Ensemble이 Recall 개선 | 328 | 21.4% |
| Ensemble이 Recall 유지 | 1,114 | 72.6% |
| Ensemble이 Recall 악화 | 92 | 6.0% |
| **Net improvement** | **+236** | **+15.4%p** |

→ **21.4% 쿼리에서 실질적 Recall 개선**. 72.6%는 무영향. 악화는 6%에 불과.
→ **쿼리 단위로 보면 명확한 양의 효과**가 있으나, 평균 지표에서는 개선 폭이 악화 폭에 상쇄되어 약하게 보임.

---

## 5. 결론 — Ensemble은 "조건부 contribution"

### 5.1 정직한 평가

| 질문 | 답 |
|---|---|
| Ensemble이 F1 메인 지표를 개선하는가? | ❌ 평균적으로는 **−0.012 ~ −0.041** (조건에 따라) |
| Ensemble이 Recall을 개선하는가? | ✅ **일관되게 +0.019 ~ +0.049**, 난이도 올라갈수록 확대 |
| Ensemble이 어떤 조건에서 F1 양수를 내는가? | ✅ **Adaptive PCST + Challenging 난이도** (+0.016) |
| Ensemble이 어떤 조건에서 명확히 효과적인가? | ✅ **toxicology, student_club, financial 등 4개 DB** (+0.01~0.09 F1) |
| Full pipeline(+XiYan Filter)에서도 효과적인가? | ❓ **미확인** — `raw cos + Adaptive + XiYan` run 필요 |

### 5.2 본 연구의 기여도 재정렬

현재까지의 증거만 놓고 보면, 본 연구의 contribution ranking은 다음과 같이 재해석 필요:

| 순위 | Component | F1 기여도 | 근거 |
|---|---|---|---|
| 1 | **XiYan Filter** | **+0.230** | b_combined → b4_xiyan |
| 2 | **Adaptive PCST** | **+0.240** (no-filter context) | b2 → b_combined |
| 3 | **Ensemble Seed Selection** | 조건부 **+0.016 ~ +0.091** | 특정 DB/난이도 한정 |

→ **"Ensemble이 가장 큰 contribution"** 이라는 가설은 현재 데이터로 **지지되지 않습니다**. 다만 이는 **Filter 유무의 상호작용을 관찰할 수 없었다는 한계** 때문이며, 다음 실험이 있어야 확정 가능:

### 5.3 Contribution 증명에 필요한 추가 실험

| ID | Config | 목적 |
|---|---|---|
| **EXP-ENS-1** | raw cos + Adaptive PCST + XiYan Filter | Ensemble의 Full pipeline 기여도 직접 측정 |
| **EXP-ENS-2** | 동일 config, difficulty별 재분해 | Full pipeline에서도 "challenging에서 우세" 패턴이 유지되는지 |
| **EXP-ENS-3** | α sweep (0.0, 0.3, 0.5, 0.7, 0.85, 1.0) + Full pipeline | α=0.85가 최적인지 확인 |

**예상 시나리오**:
- 낙관: Filter가 Ensemble의 추가 후보를 잘 활용해 Recall이 3-5%p 추가 개선 → Full pipeline F1 +0.02 이상
- 비관: Filter가 Ensemble의 false positive를 제거하며 효과 상쇄 → F1 변화 미미

### 5.4 발표 서사 제안

Ensemble을 **"Ablation의 숨은 주역"**으로 포지셔닝하는 대신, 다음 두 가지 중 선택:

#### (A) 정직한 하향 조정 (추천)
- S13/S14 Architecture 슬라이드에서 Ensemble을 "Adaptive PCST의 Recall 보완 장치"로 설명
- S21 Future Works에 **"DB/난이도에 적응하는 α"** 명시
- Main Results에서는 Adaptive PCST + XiYan Filter를 핵심 기여로 강조

#### (B) Targeted 강조
- Ensemble 전용 슬라이드 1장: **"Ensemble은 challenging 난이도에서만 F1 양수 기여"**
- 이는 "복잡한 쿼리일수록 구조 정보의 가치가 높다"는 구조적 해석과 정합
- Winner DB 4개 + Challenging 난이도 데이터로 좁은 범위의 strong claim

---

## 6. 재현 코드

```python
import json
from collections import defaultdict

with open('data/raw/BIRD_dev/dev.json') as f:
    dev = json.load(f)
qid_to_diff = {d['question_id']: d['difficulty'] for d in dev}
qid_to_db = {d['question_id']: d['db_id'] for d in dev}

def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            out[o['question_id']] = (o['recall'], o['precision'])
    return out

# Ensemble 효과 = b2/b_combined - b0/b1 (others fixed)
b0 = load('outputs/experiments/experiment_b0_raw_pcst_baseline/output_b0_raw_pcst_baseline.jsonl')
b2 = load('outputs/experiments/experiment_b2_ensemble/output_b2_ensemble.jsonl')
b1 = load('outputs/experiments/experiment_b1_adaptive_pcst/output_b1_adaptive_pcst.jsonl')
bc = load('outputs/experiments/experiment_b_combined/output_b_combined.jsonl')

def agg(d, filter_fn=None):
    qids = [q for q in d if filter_fn is None or filter_fn(q)]
    R = sum(d[q][0] for q in qids)/len(qids)
    P = sum(d[q][1] for q in qids)/len(qids)
    F = 2*R*P/(R+P) if (R+P)>0 else 0
    return R, P, F, len(qids)
```

---

## 7. 파일 링크

- 이 분석: `notebooks/analysis_results/ensemble_contribution_analysis.md`
- 난이도별 전체 Ablation: `notebooks/analysis_results/difficulty_stratified_ablation.md`
- B-4b 성능 리포트: `notebooks/analysis_results/preliminary_defense_b4b.md`
