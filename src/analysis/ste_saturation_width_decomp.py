#!/usr/bin/env python3
"""STE 10-cell EX 곡선의 포화-폭 분해 (paper §III filter=saturation-width precision lever 근거).
EX vs ext_nodes 의 marginal-gain + 포화 knee 추정 + figure.
"""
import os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/hyeonjin/thesis_refactored"
FIG = os.path.join(ROOT, "notebooks/analysis_results/figs")
os.makedirs(FIG, exist_ok=True)

k = [15, 20, 25, 30, 40, 50, 60, 70, 80, 90]
ext = [15.00, 19.81, 24.34, 28.70, 37.34, 45.47, 53.54, 60.94, 67.45, 73.85]
EX = [0.3103, 0.3631, 0.3853, 0.4192, 0.4857, 0.5287, 0.5593, 0.5613, 0.5789, 0.5900]
CEIL = 0.6030  # M4 anchor (canonical Sonnet) 경험적 EX ceiling

# Michaelis-Menten 적합 (참조용 — Emax 외삽 overshoot)
xs = [1/x for x in ext]; ys = [1/y for y in EX]; n = len(xs)
mx = sum(xs)/n; my = sum(ys)/n
b = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))/sum((xs[i]-mx)**2 for i in range(n))
a = my - b*mx
Emax = 1/a; K = b*Emax

# figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
ax = axes[0]
ax.plot(ext, EX, "-o", color="tab:blue", ms=6, label="STE-topk Sonnet EX")
ax.axhline(CEIL, color="tab:red", ls="--", lw=1.2, label=f"M4 anchor ceiling = {CEIL}")
ax.axhline(0.93*CEIL, color="gray", ls=":", lw=1)
ax.axvline(53.5, color="tab:green", ls=":", lw=1.4, label="saturation knee ≈ ext 54 (k060)")
for i, kk in enumerate(k):
    ax.annotate(f"k{kk}", (ext[i], EX[i]), fontsize=7, xytext=(0, -11), textcoords="offset points", ha="center")
ax.annotate("93% ceiling", (16, 0.93*CEIL+0.004), fontsize=8, color="gray")
ax.set_xlabel("ext_nodes (extractor candidate width = filter input)")
ax.set_ylabel("EX")
ax.set_title("EX vs candidate width — saturation curve")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax2 = axes[1]
mg = [(EX[i]-EX[i-1])/(ext[i]-ext[i-1]) for i in range(1, len(k))]
midx = [(ext[i]+ext[i-1])/2 for i in range(1, len(k))]
ax2.plot(midx, mg, "-s", color="tab:purple", ms=5)
ax2.axhline(mg[0]*0.5, color="orange", ls=":", lw=1, label=f"50% of initial ({mg[0]*0.5:.4f})")
ax2.axhline(mg[0]*0.25, color="brown", ls=":", lw=1, label=f"25% of initial ({mg[0]*0.25:.4f})")
ax2.axvline(53.5, color="tab:green", ls=":", lw=1.4)
ax2.set_xlabel("ext_nodes (midpoint of segment)")
ax2.set_ylabel("marginal EX gain per added node")
ax2.set_title("Marginal EX/node — diminishing returns")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
plt.suptitle("STE 10-cell (k015~090) — filter=saturation-width precision lever 근거", fontsize=12)
plt.tight_layout()
f = os.path.join(FIG, "ste_saturation_width_2026-06-11.png")
plt.savefig(f, dpi=120); plt.close()
print("Wrote", f)
print(f"MM fit: Emax={Emax:.4f} (외삽 overshoot, 경험적 ceiling 0.6030 사용 권장) K(half-sat)={K:.1f}")
print("포화 onset: marginal 50% at ext~34 (k040), 25% at ext~58 (k060)")
print("knee: ext~54 (k060) = 93% of M4 ceiling; ext 54→74 (+37%폭) 은 +5%ceiling 만")
