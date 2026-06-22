#!/usr/bin/env python
"""EX Ruled by Candidate Width — 이론적 상한 포락선(P=1 / R=1) + 실제 측정 overlay.

framing (postdoc 코멘트):
  - P=1 (perfect precision, noise=0): 후보 ⊆ gold → k ≤ g 에서만 성립. k↑ → recall↑ → EX↑ → k=g 에서 oracle 천장.
  - R=1 (perfect recall, gold 누락 0): 모든 gold 포함 → k ≥ g 에서만 성립. k↑ → noise만 ↑ → EX 완만히 ↓.
  - 두 곡선이 k=g 에서 만나는 dome 포락선. 실제(STE+M4)는 그 아래.

데이터 상태:
  - R=1 = 측정 (oracle B1/B2/B3, docs 02 §V.5.x.M): EX 0.6330 / 0.6584 / 0.7190
  - actual = 측정 (STE k-sweep docs 02 §V.z + M4 anchor)
  - P=1 (k<g) = 미측정 → 이론선(점선). gold-subset oracle 로 측정 가능(실험 제안).
  - g(gold size), B1/B2 width = 근사값(라벨 표기). B3 width=g.

출력: docs/figs/fig_ex_width_ideal_envelope.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs",
                      "figs", "fig_ex_width_ideal_envelope.png"))

ORACLE_CEIL = 0.7190          # B3 perfect-SL (R=1,P=1)
G = 7.0                        # gold size 근사 (M4 final ~7.4 nodes @ R0.954/P0.795 → gold ~6)

# ── R=1 measured (oracle), width 근사 ──
r1_w  = np.array([G, 22.0, 80.0])         # B3(=g), B2(gold-table cols), B1(full) — width 근사
r1_ex = np.array([0.7190, 0.6584, 0.6330])
r1_lab = ["B3 gold-col\n(R=1,P=1)", "B2 gold-table", "B1 full-schema"]

# ── actual measured: STE k-sweep (ext_nodes vs e2e EX) + M4 anchor ──
ste_w  = np.array([15, 19.81, 24.34, 28.70, 37.34, 45.47, 53.54, 60.94, 67.45, 73.85])
ste_ex = np.array([0.3103, 0.3631, 0.3853, 0.4192, 0.4857,
                   0.5287, 0.5593, 0.5613, 0.5789, 0.5900])
m4_w, m4_ex = 83.0, 0.6030

# ── P=1 theoretical (k<g, 미측정): recall buildup, 점선 (shape illustrative) ──
p1_w = np.linspace(0.4, G, 40)
p1_ex = ORACLE_CEIL * (p1_w / G)          # linear recall-buildup (illustrative)

fig, ax = plt.subplots(figsize=(10.5, 6.4))

# P=1 left arm (theoretical)
ax.plot(p1_w, p1_ex, "--", c="#27ae60", lw=2.2,
        label="Precision=1.0 ideal  (k≤g, THEORETICAL - gold-subset oracle not measured)")
# R=1 right arm (measured) — connect oracle points, extend flat-ish
ax.plot(r1_w, r1_ex, "-o", c="#8e44ad", lw=2.2, ms=9,
        label="Recall=1.0 ideal  (k≥g, MEASURED oracle B1/B2/B3)")
for w, e, l in zip(r1_w, r1_ex, r1_lab):
    ax.annotate(l, (w, e), textcoords="offset points", xytext=(6, 8), fontsize=8.5,
                color="#6c3483")
# apex
ax.scatter([G], [ORACLE_CEIL], s=240, marker="*", c="#d35400", zorder=6,
           edgecolors="k", linewidths=0.6, label=f"apex = oracle ceiling {ORACLE_CEIL:.4f} (k=g)")

# actual measured
ax.plot(ste_w, ste_ex, "-s", c="#2980b9", lw=1.8, ms=6,
        label="actual (MEASURED): STE k-sweep")
ax.scatter([m4_w], [m4_ex], s=170, marker="D", c="#c0392b", zorder=6,
           edgecolors="k", linewidths=0.6, label=f"actual M4 anchor (EX {m4_ex:.4f})")

# k=g 경계선
ax.axvline(G, ls=":", c="gray", lw=1.3)
ax.text(G + 0.6, 0.13, "k = g (gold size)\nP=1 / R=1 boundary", fontsize=8.5, color="gray")

# imperfection gap (actual ↔ R=1 envelope) 표시 — STE k50 예시
import numpy as _np
r1_at = _np.interp(45.47, r1_w, r1_ex)   # R=1 envelope @ width 45
ax.annotate("", xy=(45.47, r1_at), xytext=(45.47, 0.5287),
            arrowprops=dict(arrowstyle="<->", color="#e67e22", lw=1.6))
ax.text(46.5, (r1_at + 0.5287) / 2,
        "imperfection gap\n(recall-loss 0.0544\n + precision-loss 0.0616)",
        fontsize=8, color="#b9550f", va="center")

# 영역 라벨
ax.text(3.3, 0.66, "<- recall buildup\n(P=1: k up -> recall up -> EX up)", fontsize=8.5,
        color="#1e8449", ha="center")
ax.text(60, 0.695, "noise tolerance (R=1: k up -> EX nearly flat = free-to-prune) ->",
        fontsize=8.5, color="#6c3483", ha="center")

ax.set_xlabel("Candidate width  k  (nodes; B1/B2 width approx)")
ax.set_ylabel("Execution Accuracy (EX)")
ax.set_title("EX Ruled by Candidate Width — ideal envelope (P=1 left / R=1 right) vs actual\n"
             "dome peaks at k=g (oracle 0.7190); actual sits below, rises with width toward R=1 envelope")
ax.set_xlim(0, 92); ax.set_ylim(0, 0.78)
ax.legend(loc="lower right", fontsize=8.3, framealpha=.95)
ax.grid(alpha=.3)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"저장: {OUT}")
print(f"\napex(oracle 천장) = {ORACLE_CEIL} @ k=g≈{G}")
print(f"R=1 measured: B3 {r1_ex[0]} / B2 {r1_ex[1]} / B1 {r1_ex[2]}  (width approx {list(r1_w)})")
print(f"actual STE: EX {ste_ex.min():.4f}→{ste_ex.max():.4f} (width {ste_w.min():.0f}→{ste_w.max():.0f})")
print(f"P=1 left arm = THEORETICAL (linear illustrative) — gold-subset oracle 로 측정 필요")
