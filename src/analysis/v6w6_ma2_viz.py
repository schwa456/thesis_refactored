#!/usr/bin/env python3
"""V6-W6/MA-2 분석 결과 종합 시각화 (기존 JSON 위)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/hyeonjin/thesis_refactored"
J = os.path.join(ROOT, "outputs/analysis/v6w6_ma2_theta_sweep_score_dist_2026-06-08.json")
FIG = os.path.join(ROOT, "notebooks/analysis_results/figs")
os.makedirs(FIG, exist_ok=True)
d = json.load(open(J))
TH = d["thetas"]
dist, sweep = d["dist"], d["sweep_secondary"]

NEW = ["ma2_a_p50", "v6w6_a_p50", "ma2_a", "ma2_b_p50", "ma2_b", "v6w6_a"]
BASE = ["M4_anchor", "w3_c", "w2_sum", "w5_b", "w2_phase1"]
COL = {"ma2_a_p50": "tab:red", "v6w6_a_p50": "tab:orange", "ma2_a": "gold", "ma2_b_p50": "tab:brown",
       "ma2_b": "gray", "v6w6_a": "olive", "M4_anchor": "tab:blue", "w3_c": "tab:green",
       "w2_sum": "tab:purple", "w5_b": "tab:cyan", "w2_phase1": "black"}

# ── Figure A: separation 지표 bar (gap / Cohen d / ROC) ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
cells = NEW + BASE
labs = [c for c in cells]
for ax, (key, title) in zip(axes, [("p50_gap", "p50 gap (gold−nongold)"), ("cohen_d", "Cohen's d"), ("roc_auc", "ROC-AUC")]):
    vals = [dist[c][key] for c in cells]
    bars = ax.bar(range(len(cells)), vals, color=[COL[c] for c in cells])
    ax.set_xticks(range(len(cells))); ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11); ax.axhline(0, color="k", lw=0.6)
    ax.axvline(5.5, color="k", ls=":", lw=0.8)  # new|baseline divider
    for i, v in enumerate(vals):
        ax.text(i, v + (0.01 if v >= 0 else -0.03), f"{v:.2f}", ha="center", fontsize=7)
axes[0].text(2.5, axes[0].get_ylim()[1]*0.95, "신규 MA cells", ha="center", fontsize=9, color="dimgray")
axes[0].text(8, axes[0].get_ylim()[1]*0.95, "baselines", ha="center", fontsize=9, color="dimgray")
plt.suptitle("분석2 — cell별 gold/non-gold 분리도 (column 노드)", fontsize=13)
plt.tight_layout()
fa = os.path.join(FIG, "v6w6_ma2_separation_bars_2026-06-08.png"); plt.savefig(fa, dpi=110); plt.close()
print("Wrote", fa)

# ── Figure B: θ-sweep curves (recall / nong_pass / precision) ──
keyc = ["ma2_a_p50", "v6w6_a_p50", "w3_c", "w2_sum", "M4_anchor", "ma2_b_p50"]
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for c in keyc:
    rec = [sweep[c][str(t)]["recall"] for t in TH]
    npa = [sweep[c][str(t)]["nong_pass"] for t in TH]
    pre = [sweep[c][str(t)]["precision"] for t in TH]
    axes[0].plot(TH, rec, "-o", color=COL[c], label=c, ms=4)
    axes[1].plot(TH, npa, "-o", color=COL[c], label=c, ms=4)
    axes[2].plot(TH, pre, "-o", color=COL[c], label=c, ms=4)
axes[0].set_title("gold recall@θ (retention)"); axes[0].set_xlabel("θ"); axes[0].set_ylabel("recall")
axes[1].set_title("non-gold pass@θ (= Filter input proxy)"); axes[1].set_xlabel("θ"); axes[1].set_ylabel("nong nodes")
axes[2].set_title("precision@θ"); axes[2].set_xlabel("θ"); axes[2].set_ylabel("precision")
for ax in axes:
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle("분석1 secondary — θ-raise sweep (pure score≥θ): raise-θ는 recall을 input보다 빨리 깎음", fontsize=12)
plt.tight_layout()
fb = os.path.join(FIG, "v6w6_ma2_theta_sweep_2026-06-08.png"); plt.savefig(fb, dpi=110); plt.close()
print("Wrote", fb)

# ── Figure C: recall ↔ input(nong_pass) trade-off (efficiency frontier) ──
fig, ax = plt.subplots(figsize=(9, 7))
for c in keyc:
    rec = [sweep[c][str(t)]["recall"] for t in TH]
    npa = [sweep[c][str(t)]["nong_pass"] for t in TH]
    ax.plot(npa, rec, "-o", color=COL[c], label=c, ms=5)
    ax.annotate("θ.1", (npa[0], rec[0]), fontsize=7); ax.annotate("θ.9", (npa[-1], rec[-1]), fontsize=7)
ax.set_xlabel("non-gold pass (Filter input proxy) ← 작을수록 효율↑")
ax.set_ylabel("gold recall ↑")
ax.set_title("recall ↔ Filter-input trade-off (좌상단=이상적, clean 운영점 부재)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
fc = os.path.join(FIG, "v6w6_ma2_recall_input_tradeoff_2026-06-08.png"); plt.savefig(fc, dpi=110); plt.close()
print("Wrote", fc)
