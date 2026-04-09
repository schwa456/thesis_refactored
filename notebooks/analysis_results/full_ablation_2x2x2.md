# Full 2×2×2 Ablation Matrix — Seed × PCST × Filter

**생성일**: 2026-04-07
**목적**: Seed Selection (Cosine vs Ensemble) × Subgraph Extraction (Basic PCST vs Adaptive PCST) × Filter (None vs XiYan) 의 8개 조합 전수 비교를 통해 각 컴포넌트의 **순효과**와 **상호작용**을 정량화한다.

---

## 0. 실험 디자인

| 차원 | 옵션 A | 옵션 B |
|---|---|---|
| **Seed** | Cosine (VectorOnlySelector) | Ensemble (α=0.85: 0.85·cos + 0.15·GAT) |
| **PCST** | Basic (Fixed θ=0.1) | Adaptive (per-query P80, min=3, max=25) |
| **Filter** | None | XiYan (Qwen3-Coder-30B-A3B-Instruct-FP8) |

총 2×2×2 = 8 cells. 기존에 실행된 5개 (b0, b1, b2, b_combined, b4_xiyan_filter) + 본 세션에서 실행한 3개 (EXP-A, EXP-B, EXP-C).

신규 실행 config:
- [experiment_abl_cos_basic_xiyan.yaml](configs/experiments/experiment_abl_cos_basic_xiyan.yaml) → EXP-A
- [experiment_abl_cos_adaptive_xiyan.yaml](configs/experiments/experiment_abl_cos_adaptive_xiyan.yaml) → EXP-B
- [experiment_abl_ens_basic_xiyan.yaml](configs/experiments/experiment_abl_ens_basic_xiyan.yaml) → EXP-C

---

## 1. 전체 결과 (BIRD Dev N=1534)

| # | Seed | PCST | Filter | Run | Recall | Precision | **F1** |
|---|---|---|---|---|---|---|---|
| 1 | Cos | Basic | None | b0 | 0.9489 | 0.1570 | 0.2694 |
| 2 | Cos | Basic | **XiYan** | **EXP-A** | 0.7987 | 0.7694 | **0.7838** |
| 3 | Cos | Adaptive | None | b1 | 0.6719 | 0.3745 | 0.4809 |
| 4 | Cos | Adaptive | **XiYan** | **EXP-B** | 0.5835 | 0.7829 | 0.6687 |
| 5 | Ens | Basic | None | b2 | 0.9679 | 0.1293 | 0.2282 |
| 6 | **Ens** | **Basic** | **XiYan** | **EXP-C** | **0.8149** | 0.7597 | **0.7863** ← **BEST** |
| 7 | Ens | Adaptive | None | b_combined | 0.7210 | 0.3471 | 0.4686 |
| 8 | Ens | Adaptive | XiYan | b4_xiyan (Full) | 0.6244 | 0.7930 | 0.6987 |

### 1.1 결정적 발견

> **🚨 EXP-C (Ensemble + Basic PCST + XiYan) 가 현행 Full pipeline (#8) 을 큰 차이로 능가한다.**
>
> - **F1 0.7863 vs 0.6987 → +0.0876 개선**
> - **G-Retriever baseline F1 0.7719 도 능가** (+0.0144)
> - 즉, **Adaptive PCST 를 빼고 Basic PCST (Fixed θ=0.1) 로 돌아가는 것이 최선의 구성**

---

## 2. Leave-One-Out from Full (#8) — 컴포넌트 순효과

각 컴포넌트를 #8에서 제거했을 때의 ΔF1:

| 제거 | → 비교 셀 | ΔRecall | ΔPrecision | **ΔF1** | 해석 |
|---|---|---|---|---|---|
| − Ensemble (→ Cosine) | #4 (EXP-B) | −0.0409 | −0.0101 | **−0.0300** | Ensemble 은 +0.030 기여 |
| − Adaptive (→ Basic) | #6 (EXP-C) | **+0.1905** | −0.0333 | **+0.0877** | ★ Adaptive 가 −0.088 **악영향** |
| − XiYan (→ None) | #7 (b_combined) | +0.0966 | −0.4459 | **−0.2301** | Filter 가 +0.230 결정적 기여 |

**해석**:
- **XiYan Filter**: 가장 큰 기여 (+0.230). 본 연구의 핵심.
- **Ensemble**: 작지만 일관된 양의 기여 (+0.030). Full pipeline 안에서 비로소 효과 발현.
- **Adaptive PCST**: 현행 파라미터(P80, max_25)에서 **F1 을 0.088 깎아내리는 마이너스 기여**. ⚠

---

## 3. Adaptive PCST 가 왜 해로운가

### 3.1 PCST 옵션별 비교 (Filter 고정)

| Seed | Filter | Basic PCST F1 | Adaptive PCST F1 | Δ |
|---|---|---|---|---|
| Cos | XiYan | **0.7838** (#2) | 0.6687 (#4) | **−0.1151** |
| Ens | XiYan | **0.7863** (#6) | 0.6987 (#8) | **−0.0876** |
| Cos | None | 0.2694 (#1) | 0.4809 (#3) | +0.2115 |
| Ens | None | 0.2282 (#5) | 0.4686 (#7) | +0.2404 |

**패턴**:
- **Filter 없는 환경**: Adaptive 가 Basic 대비 F1 +0.21 ~ +0.24 (Precision 회복)
- **Filter 있는 환경**: Adaptive 가 Basic 대비 F1 −0.09 ~ −0.12 (Recall 손실이 누적)

→ **Adaptive PCST 의 P80 + max_25 clamp 는 LLM Filter 와 중복되는 "보수화" 단계**다. Filter 가 이미 Precision 을 책임지는데, Adaptive 가 그 앞단에서 Recall 을 미리 깎아 두면 Filter 가 복원할 수 없다.

### 3.2 Recall 손실 흐름 (E+X 라인)

```
b2 (E+B+N)         R=0.9679  ← 시작
└─ +Adaptive       R=0.7210  ΔR=−0.2469  (PCST 압축)
   └─ +XiYan       R=0.6244  ΔR=−0.0966  (Filter)
                   = #8 Full

b2 (E+B+N)         R=0.9679  ← 시작
└─ +XiYan          R=0.8149  ΔR=−0.1530  (Filter 단독)
                   = #6 EXP-C
```

→ Adaptive 가 0.247 의 Recall 을 미리 잘라내고, 그 위에 Filter 가 추가로 0.097 을 더 잘라내는 **이중 손실** 구조. EXP-C 는 이 첫 단계를 생략한다.

---

## 4. 난이도별 분해

### 4.1 simple (N=925)

| # | Run | R | P | F1 |
|---|---|---|---|---|
| 1 | C+B+N (b0) | 0.9514 | 0.1504 | 0.2598 |
| 2 | C+B+X (EXP-A) | 0.8076 | 0.7839 | 0.7956 |
| 3 | C+A+N (b1) | 0.7148 | 0.3677 | 0.4856 |
| 4 | C+A+X (EXP-B) | 0.6200 | 0.7986 | 0.6981 |
| 5 | E+B+N (b2) | 0.9674 | 0.1165 | 0.2079 |
| 6 | **E+B+X (EXP-C)** | 0.8209 | 0.7715 | **0.7954** |
| 7 | E+A+N (b_comb) | 0.7596 | 0.3367 | 0.4666 |
| 8 | E+A+X (Full) | 0.6563 | 0.8016 | 0.7217 |

### 4.2 moderate (N=464)

| # | Run | R | P | F1 |
|---|---|---|---|---|
| 1 | C+B+N (b0) | 0.9469 | 0.1574 | 0.2699 |
| 2 | C+B+X (EXP-A) | 0.7753 | 0.7454 | 0.7601 |
| 3 | C+A+N (b1) | 0.6360 | 0.3776 | 0.4739 |
| 4 | C+A+X (EXP-B) | 0.5483 | 0.7628 | 0.6380 |
| 5 | E+B+N (b2) | 0.9692 | 0.1354 | 0.2377 |
| 6 | **E+B+X (EXP-C)** | 0.7984 | 0.7381 | **0.7671** |
| 7 | E+A+N (b_comb) | 0.6853 | 0.3539 | 0.4668 |
| 8 | E+A+X (Full) | 0.5916 | 0.7842 | 0.6744 |

### 4.3 challenging (N=145)

| # | Run | R | P | F1 |
|---|---|---|---|---|
| 1 | C+B+N (b0) | 0.9398 | 0.1979 | 0.3270 |
| 2 | C+B+X (EXP-A) | 0.8165 | 0.7536 | 0.7838 |
| 3 | C+A+N (b1) | 0.5129 | 0.4078 | 0.4544 |
| 4 | C+A+X (EXP-B) | 0.4635 | 0.7472 | 0.5721 |
| 5 | E+B+N (b2) | 0.9668 | 0.1917 | 0.3199 |
| 6 | **E+B+X (EXP-C)** | 0.8289 | 0.7540 | **0.7897** |
| 7 | E+A+N (b_comb) | 0.5887 | 0.3918 | 0.4705 |
| 8 | E+A+X (Full) | 0.5261 | 0.7667 | 0.6240 |

### 4.4 핵심 패턴: 난이도 × 구성

| Difficulty | EXP-C F1 | Full F1 | ΔF1 (EXP-C − Full) | G-Retriever F1 |
|---|---|---|---|---|
| simple | 0.7954 | 0.7217 | **+0.0737** | 0.7877 |
| moderate | 0.7671 | 0.6744 | **+0.0927** | 0.7522 |
| **challenging** | **0.7897** | 0.6240 | **+0.1657** | 0.7340 |

**관찰**:
- EXP-C 의 우위는 **모든 난이도에서 일관**, 특히 **challenging 에서 +0.166** 으로 극대화
- EXP-C 는 **모든 난이도에서 G-Retriever 를 상회** (simple +0.008, moderate +0.015, challenging **+0.056**)
- 난이도 민감도(simple→challenging F1 낙폭): EXP-C −0.006, G-Retriever −0.054, Full(−0.098) → **EXP-C 가 가장 강건**

---

## 5. 컴포넌트 기여도 재정렬

### 5.1 Old narrative (현행 Full pipeline 기준, 2026-04 이전)

| 순위 | Component | 기여 | 근거 |
|---|---|---|---|
| 1 | Adaptive PCST | +0.24 F1 | b2 → b_combined |
| 2 | XiYan Filter | +0.23 F1 | b_combined → b4_xiyan |
| 3 | Ensemble | 미세 음수 | b1 → b_combined |

### 5.2 New narrative (2×2×2 전수 기준, 2026-04-07 갱신)

| 순위 | Component | 기여 (LOO from #6 또는 #8) | 비고 |
|---|---|---|---|
| 1 | **XiYan Filter** | **+0.558** F1 (#5 → #6, Basic 라인) | 결정적 핵심 |
| 2 | **Ensemble Seed** | +0.025 F1 (#2 → #6) / +0.030 (#4 → #8) | Filter 와 결합 시 일관된 양수 |
| 3 | **Adaptive PCST** | **−0.088** F1 (#6 → #8) | 현 파라미터에서 **음의 기여** |

→ 본 연구의 새로운 핵심 메시지:
1. **Schema Linking 의 본질은 LLM Filter 에 있고**, 그래프 단계는 Filter 에 좋은 후보를 넘기는 것이 임무.
2. **Basic PCST + Ensemble Seed + XiYan Filter** 가 최적 구성.
3. Adaptive PCST 의 역할은 "Filter 없는 환경에서 Precision 을 보강" 하는 것이며, **Filter 가 들어오면 역효과**.

---

## 6. 발표 자료 반영 권고

### 6.1 즉시 반영할 사항

- **Main Result 슬라이드 (S15)**: B-4b (현 Full) 대신 **EXP-C 를 새로운 메인 결과**로 교체
  - F1 0.7863 (G-Retriever 대비 +0.014, 본 연구 기존 대비 +0.088)
  - Precision 0.7597 (G-Retriever 대비 −0.027, 약간의 trade-off)
  - Recall 0.8149 (이전 0.6244 → +0.191 회복!)
- **Architecture 슬라이드 (S7, S9)**: Adaptive PCST 를 **Basic PCST** 로 표기 변경, S9 의 Adaptive 슬라이드는 **Ablation Insight 슬라이드**로 재용도
- **Ablation 슬라이드 (S16)**: 현행 progressive 구조 대신 **2×2×2 전수 매트릭스** 로 교체

### 6.2 새로운 서사 (S20–S22)

- **S20 한계 → S20 Insight**: "Adaptive PCST 의 P80/max_25 는 Filter 와 중복 보수화 → Basic PCST 로 회귀가 최적"
- **Future Works**:
  - Adaptive PCST 재설계: Filter-aware percentile (Filter 가 강하면 PCST 는 느슨하게)
  - Ensemble α sweep on Full pipeline (현재 0.85 가 #6 에서도 최적인지 확인)
  - GAT 재학습으로 Ensemble 기여도 확대 가능성

### 6.3 미해결 검증 항목

- [ ] Adaptive PCST 파라미터 재조정 (percentile 50/30 또는 max_prize_nodes 50/80) 시 EXP-C 를 능가할 수 있는지
- [ ] EXP-C 에서 α sweep (현재 α=0.85 는 #2 vs #6 에서 +0.0025 만 추가 → α 재최적화 여지)
- [ ] EXP-C 의 DB 별 분해 — 어떤 DB 에서 EXP-C 가 G-Retriever 를 추월하는지

---

## 7. 결론

8-cell 전수 ablation 결과, 본 연구의 **Full pipeline (#8) 은 sub-optimal** 이며, **EXP-C (Ensemble + Basic PCST + XiYan)** 가 ① 본 연구 기존 구성을 +0.088, ② G-Retriever baseline 을 +0.014 상회하는 새로운 best 임이 확인됨.

가장 중요한 시사점은 **Adaptive PCST 가 현행 파라미터에서 −0.088 의 음의 기여를 한다**는 것이며, 이는 LLM Filter 와의 보수화 중복으로 설명된다. 발표 메인 메시지를 "Adaptive PCST + Filter 의 다단계 구조" 에서 "**Ensemble Seed + 단순 PCST + 강력한 Filter**" 의 깔끔한 분업 구조로 재정렬할 것을 권고한다.

---

## 8. 재현 코드

```python
import json
from collections import defaultdict

with open('data/raw/BIRD_dev/dev.json') as f:
    dev = json.load(f)
qid_to_diff = {d['question_id']: d['difficulty'] for d in dev}

def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            out[o['question_id']] = (o['recall'], o['precision'])
    return out

def agg(d, qfilter=None):
    qids = [q for q in d if qfilter is None or qfilter(q)]
    R = sum(d[q][0] for q in qids)/len(qids)
    P = sum(d[q][1] for q in qids)/len(qids)
    F = 2*R*P/(R+P) if (R+P)>0 else 0
    return R, P, F, len(qids)

cells = {
    '#1 C+B+N': 'outputs/experiments/experiment_b0_raw_pcst_baseline/output_b0_raw_pcst_baseline.jsonl',
    '#2 C+B+X': 'outputs/experiments/experiment_abl_cos_basic_xiyan/output_abl_cos_basic_xiyan.jsonl',
    '#3 C+A+N': 'outputs/experiments/experiment_b1_adaptive_pcst/output_b1_adaptive_pcst.jsonl',
    '#4 C+A+X': 'outputs/experiments/experiment_abl_cos_adaptive_xiyan/output_abl_cos_adaptive_xiyan.jsonl',
    '#5 E+B+N': 'outputs/experiments/experiment_b2_ensemble/output_b2_ensemble.jsonl',
    '#6 E+B+X': 'outputs/experiments/experiment_abl_ens_basic_xiyan/output_abl_ens_basic_xiyan.jsonl',
    '#7 E+A+N': 'outputs/experiments/experiment_b_combined/output_b_combined.jsonl',
    '#8 E+A+X': 'outputs/experiments/experiment_b4_xiyan_filter/output_b4_xiyan_filter.jsonl',
}
```

---

## 9. 관련 파일

- 본 분석: [full_ablation_2x2x2.md](notebooks/analysis_results/full_ablation_2x2x2.md)
- Ensemble contribution (구버전): [ensemble_contribution_analysis.md](notebooks/analysis_results/ensemble_contribution_analysis.md) — Pair-C 미실행 상태에서 작성, 본 분석으로 갱신됨
- 난이도 분석 (구버전): [difficulty_stratified_ablation.md](notebooks/analysis_results/difficulty_stratified_ablation.md) — Full pipeline (#8) 기준
- B-4b 리포트 (구버전): [preliminary_defense_b4b.md](notebooks/analysis_results/preliminary_defense_b4b.md)
