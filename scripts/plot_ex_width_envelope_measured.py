#!/usr/bin/env python
"""EX Ruled by Candidate Width — 실측 dome envelope (P=1 / R=1 oracle sweep).

x = SQL 생성기에 투입되는 subschema width (nodes = tables + columns).
  P=1 arm (MEASURED): gold-subset sweep P1_20/40/60/80 → apex B3   (precision=1.0)
  R=1 arm (MEASURED): apex B3 → gold+noise sweep R1_n2/5/10/20/40  (recall=1.0)
  actual: M4 anchor post-filter (~7.4 nodes) + (-Filter) candidate (~83 nodes, R≈1)

데이터: outputs/.../oracle_ideal_envelope/ideal_envelope_report.json (Sonnet, 1534q, $11.90)
출력: docs/figs/fig_ex_width_ideal_envelope_measured.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REP = os.path.join(ROOT, "outputs/experiments/sonnet_rebaseline_2026_06_10",
                   "oracle_ideal_envelope/ideal_envelope_report.json")
OUT = os.path.join(ROOT, "docs/figs/fig_ex_width_ideal_envelope_measured.png")

cells = json.load(open(REP))["cells"]
def we(c): return cells[c]["avg_width"], cells[c]["EX"]

# R=1 우측 팔 — fraction sweep (10%~100%, apex→full schema) 신규 실측
RFRAC = os.path.join(ROOT, "outputs/experiments/sonnet_rebaseline_2026_06_10",
                     "oracle_r1_fraction/r1_fraction_report.json")
fcells = json.load(open(RFRAC))["cells"]
def fwe(c): return fcells[c]["avg_width"], fcells[c]["EX"]

# P=1 left arm (measured) — up to apex
p1 = ["P1_20", "P1_40", "P1_60", "P1_80", "B3_gold_column"]
p1_w = np.array([we(c)[0] for c in p1]); p1_ex = np.array([we(c)[1] for c in p1])
# R=1 right arm (measured) — apex(B3) + fraction f10..f100 (full schema)
apex_w, apex_ex = we("B3_gold_column")
rf = [f"R1_f{p}" for p in range(10, 101, 10)]
r1_w = np.array([apex_w] + [fwe(c)[0] for c in rf])
r1_ex = np.array([apex_ex] + [fwe(c)[1] for c in rf])
# 저해상 cross-check: 이전 절대-노이즈 셀 (n2..n40) — 같은 곡선 위 정합 확인용
nz = ["R1_n2", "R1_n5", "R1_n10", "R1_n20", "R1_n40"]
nz_w = np.array([we(c)[0] for c in nz]); nz_ex = np.array([we(c)[1] for c in nz])

# STE k-sweep (actual e2e pipeline) — docs 02 §V.z. ★ x = extractor output(=filter-input) width,
# oracle subschema width 와는 다른 stage (STE EX 는 filter+sqlgen 거친 e2e 값).
ste_w  = np.array([15, 19.81, 24.34, 28.70, 37.34, 45.47, 53.54, 60.94, 67.45, 73.85])
ste_ex = np.array([0.3103, 0.3631, 0.3853, 0.4192, 0.4857,
                   0.5287, 0.5593, 0.5613, 0.5789, 0.5900])

# actual (subschema fed to SQLgen 축): M4 post-filter ~7.4 / -Filter candidate ~83
m4_w, m4_ex = 7.4, 0.6030
nf_w, nf_ex = 83.0, 0.6004           # -Filter (no filter, R 0.9964 ~ R=1)
# prior oracle 참고 (R=1, 더 큰 width): B2 gold-table 0.6584 / B1 full 0.6330
b2_ex, b1_ex = 0.6584, 0.6330

fig, ax = plt.subplots(figsize=(11, 6.6))
C_P1, C_R1, C_AP, C_ACT = "#27ae60", "#8e44ad", "#d35400", "#c0392b"

ax.plot(p1_w, p1_ex, "--o", c=C_P1, lw=2.3, ms=8,
        label="Precision = 1.0 ideal")
ax.plot(r1_w, r1_ex, "-o", c=C_R1, lw=2.3, ms=8,
        label="Recall = 1.0 ideal")
ax.scatter([apex_w], [apex_ex], s=260, marker="*", c=C_AP, zorder=6,
           edgecolors="k", linewidths=0.6,
           label="Gold Schema (Column)")

# STE k-sweep (actual e2e; x = extractor output / filter-input width)
ax.plot(ste_w, ste_ex, "-s", c="#2980b9", lw=1.8, ms=5.5, zorder=5,
        label="Extraction Sweep")

# saturation knee (STE k060: ext~54 = 93% of M4 ceiling 0.6030)
KNEE_W, KNEE_EX = 53.54, 0.5593
ax.axvline(KNEE_W, ls=":", c="gray", lw=1.2, zorder=1)
ax.scatter([KNEE_W], [KNEE_EX], s=200, marker="o", facecolors="none",
           edgecolors="#16a085", linewidths=2.4, zorder=8,
           label="saturation knee")

# actual (ours)
ax.scatter([m4_w], [m4_ex], s=85, marker="D", c=C_ACT, zorder=7, edgecolors="k",
           linewidths=0.6, label="Ours")
ax.scatter([nf_w], [nf_ex], s=120, marker="s", c=C_ACT, zorder=7, edgecolors="k",
           linewidths=0.6, label="Ours − Filter")

ax.set_xlabel("Candidate width  k  =  subschema nodes fed to SQL generator (tables + columns)")
ax.set_ylabel("Execution Accuracy (EX)")
ax.set_title("EX Ruled by Candidate Width — measured ideal envelope (oracle sweep, Sonnet 1534q)")
ax.set_xlim(0, 96); ax.set_ylim(0.15, 0.78)
ax.legend(loc="lower right", fontsize=8.2, framealpha=.95)
ax.grid(alpha=.3)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"저장: {OUT}")
print(f"P=1 arm: width {list(p1_w)} EX {list(p1_ex)}")
print(f"R=1 arm: width {list(r1_w)} EX {list(r1_ex)}")
print(f"apex {apex_ex} @ {apex_w} | M4 gap to R=1 envelope = {gap_top-m4_ex:.4f}")
