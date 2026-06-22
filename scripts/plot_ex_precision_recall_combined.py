#!/usr/bin/env python
"""EX vs Precision (baseline) + EX vs Recall (baseline) 를 한 PNG 로 결합.
- 점: 3 baseline + Ours(별)
- 선: fit(4-point) 만 (fit3 제거)
- 범례: 두 패널 공통 → 오른쪽에 하나만 배치
- 'M4 (ours)' → 'Ours'
출력: docs/figs/fig_ex_precision_recall_combined.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs",
                      "figs", "fig_ex_precision_recall_combined.png"))

# ── baseline 4-method (docs 02 §10) ──
methods = ["Ours", "G-Retriever", "LinkAlign", "XiYan-SQL"]
P  = np.array([0.7950, 0.1548, 0.3081, 0.8052])
R  = np.array([0.9539, 0.9527, 0.8167, 0.6906])
EX = np.array([0.6030, 0.5469, 0.4276, 0.4081])
is_ours = np.array([True, False, False, False])

C_BASE, C_OURS, C_FIT = "#2c3e50", "#c0392b", "#c0392b"

def panel(ax, x, xlabel, title_letter):
    lr = stats.linregress(x, EX)
    ax.scatter(x[~is_ours], EX[~is_ours], s=110, c=C_BASE, zorder=3)
    ax.scatter(x[is_ours], EX[is_ours], s=200, marker="*", c=C_OURS, zorder=4,
               edgecolors="k", linewidths=0.4)
    for xi, yi, n in zip(x, EX, methods):
        ax.annotate(n, (xi, yi), textcoords="offset points", xytext=(7, 6), fontsize=9)
    xx = np.linspace(x.min() - 0.04, x.max() + 0.04, 50)
    ax.plot(xx, lr.slope * xx + lr.intercept, "-", c=C_FIT, lw=1.9, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_title(f"({title_letter}) EX vs {xlabel.split()[-1]}\n"
                 f"linear fit: slope={lr.slope:+.3f}, r={lr.rvalue:+.3f}")
    ax.grid(alpha=.3)
    return lr

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.2))
panel(axA, P, "Schema-linking Precision", "a")
panel(axB, R, "Schema-linking Recall", "b")
axA.set_ylabel("Execution Accuracy (EX)")
# y축 통일
ylo = min(EX.min(), 0.40) - 0.02; yhi = EX.max() + 0.03
for ax in (axA, axB):
    ax.set_ylim(ylo, yhi)

# ── 공유 범례 (오른쪽 바깥에 하나) ──
handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=C_BASE,
           markersize=10, label="baseline"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor=C_OURS,
           markeredgecolor="k", markersize=15, label="Ours"),
    Line2D([0], [0], color=C_FIT, lw=1.9, label="linear fit (4 points)"),
]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.84, 0.5),
           fontsize=10, frameon=True, framealpha=.95)

fig.suptitle("EX vs Precision (flat) vs EX vs Recall (positive) — Recall is the EX lever",
             fontsize=12.5, y=1.01)
fig.tight_layout(rect=[0, 0, 0.84, 1.0])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"저장: {OUT}")
