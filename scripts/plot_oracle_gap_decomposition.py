#!/usr/bin/env python
"""Oracle 상한 + 갭 분해 (where EX comes from) — 단순화판.

전체 질의(100%) = anchor 달성 EX 0.6030 + schema-linking recoverable 0.1160
                  + SQL generator floor 0.2810 (완벽 SL로도 실패).
perfect-SL ceiling = 0.7190 = anchor + recoverable.
근거: oracle_gap_decomp_sonnet.md + ideal envelope sweep (2026-06-13).
출력: docs/figs/fig_oracle_gap_decomposition.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs",
                      "figs", "fig_oracle_gap_decomposition.png"))

seg = [
    ("anchor EX (ours)",          0.6030, "#27ae60"),
    ("schema-linking recoverable", 0.1160, "#e67e22"),
    ("SQL generator floor",        0.2810, "#95a5a6"),
]
CEIL = 0.7190   # perfect-SL ceiling

fig, ax = plt.subplots(figsize=(12.5, 3.0))
left = 0.0
for name, val, col in seg:
    ax.barh(0, val, left=left, color=col, edgecolor="white", height=0.6, label=name)
    txt_col = "white" if col in ("#27ae60", "#95a5a6") else "black"
    ax.text(left + val/2, 0, f"{val:.4f}\n({val*100:.1f}%)", ha="center", va="center",
            fontsize=10.5, color=txt_col, fontweight="bold")
    left += val

# 참조선: anchor(0.6030) / perfect-SL ceiling(0.7190)
ax.axvline(0.6030, ymin=0.08, ymax=0.95, ls="--", c="#1e8449", lw=1.6)
ax.text(0.598, 0.33, "ours 0.6030\n(84% of ceiling)", ha="right", va="bottom",
        fontsize=9.5, color="#1e8449", fontweight="bold")
ax.axvline(0.7190, ymin=0.08, ymax=0.95, ls="--", c="#b9550f", lw=1.6)
ax.text(0.721, 0.33, "perfect-SL\nceiling 0.7190", ha="left", va="bottom",
        fontsize=9.5, color="#b9550f", fontweight="bold")

ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, 0.7)
ax.set_yticks([])
ax.set_xlabel("Fraction of queries (BIRD-Dev 1534)")
ax.set_title("Where EX comes from")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=3, fontsize=9.5,
          frameon=False)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"저장: {OUT}")
print(f"sum check = {sum(v for _, v, _ in seg):.4f} (=1.0) | ceiling {CEIL} = anchor + recoverable")
