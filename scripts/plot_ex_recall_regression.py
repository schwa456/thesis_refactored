#!/usr/bin/env python
"""EX vs F1 / EX vs Recall 산점도 + EX~Recall 선형회귀 기울기.

데이터:
  (1) Sonnet baseline 비교 (docs/02 §10 / 99 §4.2.1): M4 anchor + 3 baseline
  (2) STE-topk k-sweep (docs/02 §V.z, 10 cell, within-method) — robust 기울기

출력: docs/figs/fig_ex_recall_regression.png  +  콘솔 회귀 요약
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figs",
                   "fig_ex_recall_regression.png")
OUT = os.path.abspath(OUT)

# ── (1) Sonnet baseline 4-method (EXPERIMENT_HISTORY Sonnet era / docs 02 §10) ──
methods = ["M4 (ours)", "G-Retriever", "LinkAlign", "XiYan-SQL"]
R   = np.array([0.9539, 0.9527, 0.8167, 0.6906])
F1  = np.array([0.8537, 0.2663, 0.4474, 0.7435])
EX  = np.array([0.6030, 0.5469, 0.4276, 0.4081])
is_ours = np.array([True, False, False, False])

# ── (2) STE-topk k-sweep (docs 02 §V.z, full 1534q, BiFilter Sonnet) ──
ste_k  = np.array([15, 20, 25, 30, 40, 50, 60, 70, 80, 90])
ste_R  = np.array([0.7222, 0.7719, 0.8084, 0.8331, 0.8765,
                   0.9027, 0.9194, 0.9327, 0.9426, 0.9471])
ste_EX = np.array([0.3103, 0.3631, 0.3853, 0.4192, 0.4857,
                   0.5287, 0.5593, 0.5613, 0.5789, 0.5900])


def fit(x, y, label):
    lr = stats.linregress(x, y)
    print(f"\n[{label}]  n={len(x)}")
    print(f"  EX = {lr.slope:.4f} * Recall + ({lr.intercept:.4f})")
    print(f"  slope dEX/dRecall = {lr.slope:.4f}  (95% CI ±{1.96*lr.stderr:.4f})")
    print(f"  Pearson r = {lr.rvalue:.4f}   R^2 = {lr.rvalue**2:.4f}   p = {lr.pvalue:.4g}")
    return lr


print("=" * 64)
print("EX ~ Recall 선형회귀 (기울기 = dEX/dRecall)")
print("=" * 64)
lr_b4 = fit(R, EX, "baseline 4-method (M4 + 3 baseline)")
lr_b3 = fit(R[~is_ours], EX[~is_ours], "baseline 3 only (M4 제외)")
lr_ste = fit(ste_R, ste_EX, "STE k-sweep 10-cell (within-method, robust)")

# 대조: EX ~ F1 (baseline 4) — anti-correlation 확인
lr_f1 = stats.linregress(F1, EX)
print(f"\n[대조 EX~F1 baseline 4-method]  slope={lr_f1.slope:.4f}  "
      f"Pearson r={lr_f1.rvalue:.4f}  R^2={lr_f1.rvalue**2:.4f}")
lr_f1b3 = stats.linregress(F1[~is_ours], EX[~is_ours])
print(f"[대조 EX~F1 baseline 3 only]    slope={lr_f1b3.slope:.4f}  "
      f"Pearson r={lr_f1b3.rvalue:.4f}  R^2={lr_f1b3.rvalue**2:.4f}")

# ──────────────────────── 플롯 (3 panel) ────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
COL_OURS, COL_BASE, COL_STE = "#c0392b", "#2c3e50", "#2980b9"


def annotate(a, xs, ys, names):
    for x, y, n in zip(xs, ys, names):
        a.annotate(n, (x, y), textcoords="offset points", xytext=(7, 5),
                   fontsize=9)


# Panel 1: EX vs F1 (baseline) — anti-correlation
ax[0].scatter(F1[~is_ours], EX[~is_ours], s=110, c=COL_BASE, zorder=3, label="baseline")
ax[0].scatter(F1[is_ours], EX[is_ours], s=160, c=COL_OURS, marker="*",
              zorder=4, label="M4 (ours)")
annotate(ax[0], F1, EX, methods)
xx = np.linspace(F1.min() - 0.03, F1.max() + 0.03, 50)
ax[0].plot(xx, lr_f1b3.slope * xx + lr_f1b3.intercept, "--", c=COL_BASE, lw=1.6)
ax[0].set_xlabel("Schema-linking F1"); ax[0].set_ylabel("Execution Accuracy (EX)")
ax[0].set_title(f"(a) EX vs F1 — baseline\nanti-corr: slope={lr_f1b3.slope:+.3f}, "
                f"r={lr_f1b3.rvalue:+.3f} (3 baseline)")
ax[0].legend(loc="upper right", fontsize=9); ax[0].grid(alpha=.3)

# Panel 2: EX vs Recall (baseline) — positive
ax[1].scatter(R[~is_ours], EX[~is_ours], s=110, c=COL_BASE, zorder=3, label="baseline")
ax[1].scatter(R[is_ours], EX[is_ours], s=160, c=COL_OURS, marker="*",
              zorder=4, label="M4 (ours)")
annotate(ax[1], R, EX, methods)
xx = np.linspace(R.min() - 0.03, R.max() + 0.03, 50)
ax[1].plot(xx, lr_b4.slope * xx + lr_b4.intercept, "-", c=COL_OURS, lw=1.8,
           label=f"fit(4): slope={lr_b4.slope:+.3f}")
ax[1].plot(xx, lr_b3.slope * xx + lr_b3.intercept, "--", c=COL_BASE, lw=1.5,
           label=f"fit(3): slope={lr_b3.slope:+.3f}")
ax[1].set_xlabel("Schema-linking Recall"); ax[1].set_ylabel("Execution Accuracy (EX)")
ax[1].set_title(f"(b) EX vs Recall — baseline\nslope={lr_b4.slope:+.3f} "
                f"(4pt, r={lr_b4.rvalue:+.3f}) / {lr_b3.slope:+.3f} (3pt)")
ax[1].legend(loc="upper left", fontsize=8.5); ax[1].grid(alpha=.3)

# Panel 3: EX vs Recall (STE k-sweep) — robust within-method slope
sc = ax[2].scatter(ste_R, ste_EX, s=90, c=ste_k, cmap="viridis", zorder=3)
xx = np.linspace(ste_R.min() - 0.01, ste_R.max() + 0.01, 50)
ax[2].plot(xx, lr_ste.slope * xx + lr_ste.intercept, "-", c=COL_STE, lw=1.8)
for x, y, k in zip(ste_R, ste_EX, ste_k):
    ax[2].annotate(f"k{k}", (x, y), textcoords="offset points", xytext=(5, -3),
                   fontsize=7.5)
ax[2].set_xlabel("Schema-linking Recall"); ax[2].set_ylabel("Execution Accuracy (EX)")
ax[2].set_title(f"(c) EX vs Recall — STE k-sweep (within-method, n=10)\n"
                f"slope={lr_ste.slope:+.3f}, r={lr_ste.rvalue:+.3f}, "
                f"R²={lr_ste.rvalue**2:.3f}")
cb = fig.colorbar(sc, ax=ax[2], fraction=0.046, pad=0.04); cb.set_label("STE top-k")
ax[2].grid(alpha=.3)

fig.suptitle("Recall-bounded EX: EX∝Recall (positive) vs EX∝F1 (cross-method anti-corr)",
             fontsize=12, y=1.02)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
