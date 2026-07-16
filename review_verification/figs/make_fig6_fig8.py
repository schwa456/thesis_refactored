"""
그림 6 (Recall/Precision vs EX 2패널) + 그림 8 (Filter before/after) 재생성.
LLM 0, matplotlib only. 스타일 = scripts/plot_ex_recall_regression.py 답습
(COL_OURS/COL_BASE, 별표 s=160 / 원형 s=110, '--' 회귀선, annotate offset (7,5) fs9).

실행: python review_verification/figs/make_fig6_fig8.py
출력: review_verification/figs/fig6_recall_precision_ex.{pdf,png}, fig8_filter_before_after.{pdf,png}
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "review_verification/figs"
OUT.mkdir(parents=True, exist_ok=True)

# 스타일 (기존 스크립트 답습)
COL_OURS, COL_BASE = "#c0392b", "#2c3e50"
COL_P, COL_EX, COL_BAR = "#c0392b", "#27ae60", "#7f8c8d"  # fig8: precision 빨강 / EX 초록 / bar 회색

# ── 확정 좌표 (element-level, stage3_unified_metrics.csv 대조 검증됨) ──
DATA = [
    # method, Recall, Precision, EX(bprime for Proposed, baseline EX)
    ("Proposed",    0.9539, 0.7950, 0.6089),
    ("XiYanSQL",    0.7233, 0.8348, 0.4081),
    ("G-Retriever", 0.9481, 0.1789, 0.5469),
    ("LinkAlign",   0.8462, 0.3442, 0.4276),
]
# CSV 대조 검증
_csv = {x["method"]: x for x in csv.DictReader(open(ROOT / "review_verification/stage3_unified_metrics.csv"))}
for m, R, P, EX in DATA:
    x = _csv[m]
    ex_csv = float(x["EX_bprime"]) if x.get("EX_bprime") else float(x["EX_asrun_or_baseline"])
    assert abs(float(x["R_element"]) - R) < 1e-3 and abs(float(x["P_element"]) - P) < 1e-3 and abs(ex_csv - EX) < 1e-3, \
        f"FLAG 좌표 불일치: {m}"
print("좌표 CSV 대조 검증 OK")

methods = [d[0] for d in DATA]
R = np.array([d[1] for d in DATA]); P = np.array([d[2] for d in DATA]); EX = np.array([d[3] for d in DATA])
is_ours = np.array([m == "Proposed" for m in methods])

def annotate(a, xs, ys, names, dx=7, dy=5):
    for x, y, n in zip(xs, ys, names):
        a.annotate(n, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=9)

# ══ 그림 6 ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.0))

for i, (X, xlabel) in enumerate([(P, "Schema-linking Precision"), (R, "Schema-linking Recall")]):
    a = ax[i]
    a.scatter(X[~is_ours], EX[~is_ours], s=110, c=COL_BASE, zorder=3, label="baseline")
    a.scatter(X[is_ours], EX[is_ours], s=160, c=COL_OURS, marker="*", zorder=4, label="Proposed")
    annotate(a, X, EX, methods)
    # 4점 선형회귀
    lr = linregress(X, EX)
    xx = np.linspace(X.min() - 0.03, X.max() + 0.03, 50)
    a.plot(xx, lr.slope * xx + lr.intercept, "--", c=COL_BASE, lw=1.6,
           label=f"linear fit (4 points), r={lr.rvalue:.2f}")
    a.set_xlabel(xlabel, fontsize=11)
    a.set_ylabel("Execution Accuracy (EX)", fontsize=11)
    a.legend(loc="best", fontsize=8.5); a.grid(alpha=.3)
    a.set_ylim(0.38, 0.64)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig6_recall_precision_ex.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved fig6 → {OUT}/fig6_recall_precision_ex.pdf/.png")

# ══ 그림 8: Filter before/after ═════════════════════════════════════
# before = extractor-fed, after = filter-fed(BiFilter)
P_before, P_after = 0.1488, 0.7950
EX_before, EX_after = 0.6004, 0.6089   # 이제 소폭 상승 (수평 아님)
NODE_before, NODE_after = 83.1, 7.4
xlab = ["before (extractor out)", "after (BiFilter)"]
xpos = [0, 1]

fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.8))

# 왼쪽: precision(빨강) + EX(초록) 선
a = ax[0]
a.plot(xpos, [P_before, P_after], "-o", c=COL_P, lw=2.2, ms=9, label="Precision")
a.plot(xpos, [EX_before, EX_after], "-s", c=COL_EX, lw=2.2, ms=8, label="EX")
# 값 주석
a.annotate(f"{P_before:.4f}", (0, P_before), textcoords="offset points", xytext=(-8, -16), fontsize=9, c=COL_P)
a.annotate(f"{P_after:.4f}", (1, P_after), textcoords="offset points", xytext=(-20, 8), fontsize=9, c=COL_P)
a.annotate(f"{EX_before:.4f}", (0, EX_before), textcoords="offset points", xytext=(2, 8), fontsize=9, c=COL_EX)
a.annotate(f"{EX_after:.4f}", (1, EX_after), textcoords="offset points", xytext=(-38, 8), fontsize=9, c=COL_EX)
# EX 소폭 상승 강조: Δ 텍스트 (선 아래 여백에 배치)
a.annotate(f"ΔEX = +{EX_after-EX_before:.4f} (not flat)", (0.5, 0.72),
           ha="center", fontsize=8.5, c=COL_EX, style="italic")
a.set_xticks(xpos); a.set_xticklabels(xlab, fontsize=10)
a.set_ylabel("Precision / EX", fontsize=11)
a.set_ylim(0.0, 1.0); a.legend(loc="center right", fontsize=9.5); a.grid(alpha=.3)
a.set_title("Precision & EX", fontsize=11)

# 오른쪽: 평균 후보 노드 막대
a = ax[1]
bars = a.bar(xpos, [NODE_before, NODE_after], color=COL_BAR, width=0.55, zorder=3)
for x, v in zip(xpos, [NODE_before, NODE_after]):
    a.annotate(f"{v}", (x, v), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=10, fontweight="bold")
a.set_xticks(xpos); a.set_xticklabels(xlab, fontsize=10)
a.set_ylabel("Avg. candidate nodes", fontsize=11)
a.set_ylim(0, 92); a.grid(alpha=.3, axis="y")
a.set_title("Candidate schema size", fontsize=11)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig8_filter_before_after.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"saved fig8 → {OUT}/fig8_filter_before_after.pdf/.png")
