#!/usr/bin/env python
"""EX vs Precision 산점도 + EX~Precision 선형회귀 기울기.
EX~Recall (plot_ex_recall_regression.py) 의 Precision 짝. 헤드라인 정합:
  baseline 횡단 = P↔EX 음의 상관(P는 EX lever 아님=예산축) / STE = P는 recall 의
  passenger(k↑ 시 R·P 동반 상승)이라 + 기울기는 비인과(주의 표기).
출력: docs/figs/fig_ex_precision_regression.png + 콘솔 회귀 요약
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs",
                      "figs", "fig_ex_precision_regression.png"))

# ── baseline 4-method (docs 02 §10) ──
methods = ["M4 (ours)", "G-Retriever", "LinkAlign", "XiYan-SQL"]
P  = np.array([0.7950, 0.1548, 0.3081, 0.8052])
EX = np.array([0.6030, 0.5469, 0.4276, 0.4081])
is_ours = np.array([True, False, False, False])

# ── STE-topk k-sweep (docs 02 §V.z) ──
ste_k  = np.array([15, 20, 25, 30, 40, 50, 60, 70, 80, 90])
ste_P  = np.array([0.6921, 0.7089, 0.7238, 0.7377, 0.7623,
                   0.7756, 0.7777, 0.7882, 0.7908, 0.7922])
ste_EX = np.array([0.3103, 0.3631, 0.3853, 0.4192, 0.4857,
                   0.5287, 0.5593, 0.5613, 0.5789, 0.5900])


def fit(x, y, label):
    lr = stats.linregress(x, y)
    print(f"\n[{label}]  n={len(x)}")
    print(f"  EX = {lr.slope:.4f} * Precision + ({lr.intercept:.4f})")
    print(f"  slope dEX/dPrecision = {lr.slope:.4f}  (95% CI ±{1.96*lr.stderr:.4f})")
    print(f"  Pearson r = {lr.rvalue:.4f}   R^2 = {lr.rvalue**2:.4f}   p = {lr.pvalue:.4g}")
    return lr


print("=" * 64)
print("EX ~ Precision 선형회귀 (기울기 = dEX/dPrecision)")
print("=" * 64)
lr_b4 = fit(P, EX, "baseline 4-method (M4 + 3 baseline)")
lr_b3 = fit(P[~is_ours], EX[~is_ours], "baseline 3 only (M4 제외)")
lr_ste = fit(ste_P, ste_EX, "STE k-sweep 10-cell (within-method, P=recall passenger)")

fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.2))
COL_OURS, COL_BASE, COL_STE = "#c0392b", "#2c3e50", "#2980b9"

# Panel (a): EX vs Precision — baseline (anti-corr 예상)
ax[0].scatter(P[~is_ours], EX[~is_ours], s=110, c=COL_BASE, zorder=3, label="baseline")
ax[0].scatter(P[is_ours], EX[is_ours], s=170, c=COL_OURS, marker="*", zorder=4,
              label="M4 (ours)")
for x, y, n in zip(P, EX, methods):
    ax[0].annotate(n, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=9)
xx = np.linspace(P.min() - 0.04, P.max() + 0.04, 50)
ax[0].plot(xx, lr_b4.slope * xx + lr_b4.intercept, "-", c=COL_OURS, lw=1.8,
           label=f"fit(4): slope={lr_b4.slope:+.3f}")
ax[0].plot(xx, lr_b3.slope * xx + lr_b3.intercept, "--", c=COL_BASE, lw=1.5,
           label=f"fit(3): slope={lr_b3.slope:+.3f}")
ax[0].set_xlabel("Schema-linking Precision"); ax[0].set_ylabel("Execution Accuracy (EX)")
ax[0].set_title(f"(a) EX vs Precision — baseline\nanti-corr: slope={lr_b3.slope:+.3f}, "
                f"r={lr_b3.rvalue:+.3f} (3 baseline)")
ax[0].legend(loc="upper center", fontsize=8.5); ax[0].grid(alpha=.3)

# Panel (b): EX vs Precision — STE k-sweep (P = recall passenger, 비인과)
sc = ax[1].scatter(ste_P, ste_EX, s=90, c=ste_k, cmap="viridis", zorder=3)
xx = np.linspace(ste_P.min() - 0.005, ste_P.max() + 0.005, 50)
ax[1].plot(xx, lr_ste.slope * xx + lr_ste.intercept, "-", c=COL_STE, lw=1.8)
for x, y, k in zip(ste_P, ste_EX, ste_k):
    ax[1].annotate(f"k{k}", (x, y), textcoords="offset points", xytext=(5, -3),
                   fontsize=7.5)
ax[1].set_xlabel("Schema-linking Precision"); ax[1].set_ylabel("Execution Accuracy (EX)")
ax[1].set_title(f"(b) EX vs Precision — STE k-sweep (within-method, n=10)\n"
                f"slope={lr_ste.slope:+.3f}, r={lr_ste.rvalue:+.3f} "
                f"(P = recall passenger, non-causal)")
cb = fig.colorbar(sc, ax=ax[1], fraction=0.046, pad=0.04); cb.set_label("STE top-k")
ax[1].grid(alpha=.3)

fig.suptitle("Precision vs EX: cross-method flat/negative (P is not the EX lever; "
             "budget axis) — contrast with positive EX vs Recall", fontsize=12, y=1.02)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
