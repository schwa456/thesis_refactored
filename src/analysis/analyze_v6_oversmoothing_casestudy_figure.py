#!/usr/bin/env python3
"""학위논문 §4.4 over-smoothing 사례연구 figure (3-panel).

(a) intra-table hi-deg column MAD trajectory L0(PLM)→L3, 개입 전/후 4 series
    — column self-loop + per-layer residual 가 L1 collapse(0.0136)를 0.28~0.35로
      20~25× 회복 (L0 PLM 0.4201 의 67~82%).
(b) MAD 회복(L1 hi-deg) ↔ 종단 EX 직교 산점도 (V6-W5 3 cells, Spearman −0.50)
    + 현 Sonnet anchor 의 −Scorer corroboration (EX 0.6030→0.6089, GAT off 가
      오히려 +0.0059) — selector 표현 품질 ⊥ e2e.
(c) Dirichlet energy depth (5.16→17.49→56.31, 3.4×/layer monotonic GROWTH)
    — 고전 over-smoothing(energy→0 collapse) 아님을 명시.

데이터 출처 (모두 측정 완료, canonical):
  - notebooks/analysis_results/01_v6_oversmoothing_chain/v6_w5_l1_mad_disconnect_2026-06-07.md §2.1/§4
  - notebooks/analysis_results/01_v6_oversmoothing_chain/v6_phase0_diagnostics_2026-06-01.md §(Dirichlet)
  - notebooks/analysis_results/08_paper_narratives/paper_sonnet_results_2026-06-11.md (−Scorer 0.6089)
  - docs/thesis_ablation_inventory_2026-06-16.md A.2

honesty scoping: architecture-specific(single-shared-source) · fixable · e2e-disconnect.
보편 over-smoothing 법칙 주장 금지 (캡션 명시).

산출: notebooks/analysis_results/01_v6_oversmoothing_chain/
      fig_v6_oversmoothing_casestudy_2026-06-19.{png,pdf}
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

OUT_DIR = (
    "/home/hyeonjin/thesis_refactored/notebooks/analysis_results/"
    "01_v6_oversmoothing_chain"
)
STAMP = "2026-06-19"

# ─────────────────────────────────────────────────────────────
# Canonical data (4-decimal, fixed)
# ─────────────────────────────────────────────────────────────
# (a) intra-table hi-deg MAD trajectory  — v6_w5 §2.1 (selector-stage, backbone 무관)
LAYERS = ["L0\n(PLM in)", "L1", "L2", "L3"]
LX = [0, 1, 2, 3]
TRAJ = {
    "baseline (no fix)": [0.4201, 0.0136, 0.0092, 0.0046],
    "(a) column self-loop":         [0.4201, 0.2813, 0.1897, 0.1398],
    "(b) per-layer residual":       [0.4201, 0.3416, 0.1540, 0.0539],
    "(c) self-loop + residual":     [0.4201, 0.3463, 0.2648, 0.1240],
}
TRAJ_COLORS = {
    "baseline (no fix)": "#b0182b",
    "(a) column self-loop":         "#2e86c1",
    "(b) per-layer residual":       "#1e8449",
    "(c) self-loop + residual":     "#7d3c98",
}

# (b) MAD recovery ↔ e2e EX  — v6_w5 §4 (3 W5 cells, GLM-era e2e)
W5 = {
    "a": {"mad": 0.2813, "ex": 0.3168},
    "b": {"mad": 0.3416, "ex": 0.3201},
    "c": {"mad": 0.3463, "ex": 0.2934},
}
# Sonnet anchor corroboration (paper_sonnet_results §, ablation_inventory A.2)
SONNET = {
    "anchor (α=0.5, GAT on)":  0.6030,
    "−Scorer (α=1.0, GAT off)": 0.6089,  # ΔEX +0.0059 — over-smoothing-bearing GAT 제거가 오히려 ↑
}

# (c) Dirichlet energy depth  — v6_phase0 §
DE_LAYERS = ["L1", "L2", "L3"]
DE_ENERGY = [5.156, 17.489, 56.309]          # geometric ~3.4×/layer (GROWTH, not →0)
DE_PER_EDGE = [0.0295, 0.112, 0.364]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.55))
    axA, axB, axC = axes

    # ── Panel (a): intra-table MAD trajectory ────────────────────
    for name, ys in TRAJ.items():
        c = TRAJ_COLORS[name]
        lw = 2.6 if name.startswith("baseline") else 2.0
        ls = "-" if not name.startswith("baseline") else "--"
        axA.plot(LX, ys, marker="o", ms=6, lw=lw, ls=ls, color=c,
                 label=name, zorder=3)
    axA.axhspan(0.0, 0.02, color="#b0182b", alpha=0.07, zorder=0)
    # L1 회복 화살표 (0.0136 → 0.3463, 25.5×)
    axA.annotate(
        "", xy=(1.0, 0.3463), xytext=(1.0, 0.0136),
        arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.6, shrinkA=2, shrinkB=2),
        zorder=4,
    )
    axA.text(1.13, 0.185, "L1: 0.0136 → 0.3463\n(25.5×, ≈82% of L0)",
             fontsize=8.7, color="#333", va="center")
    axA.text(0.02, 0.013, "collapse band", fontsize=7.5, color="#b0182b",
             style="italic", transform=axA.get_yaxis_transform())
    axA.set_xticks(LX)
    axA.set_xticklabels(LAYERS)
    axA.set_ylabel("intra-table hi-deg column MAD")
    axA.set_xlabel("GAT depth")
    axA.set_ylim(-0.02, 0.46)
    axA.set_title("(a) Intra-table column collapse & repair\n(per-layer, selector-stage · backbone-independent)",
                  fontsize=10.5)
    axA.legend(fontsize=7.8, loc="upper right", framealpha=0.92, handlelength=1.8)
    axA.grid(axis="y", ls=":", alpha=0.4)

    # ── Panel (b): MAD recovery ⊥ e2e EX ─────────────────────────
    mad_v = np.array([W5[k]["mad"] for k in ("a", "b", "c")])
    ex_v = np.array([W5[k]["ex"] for k in ("a", "b", "c")])
    rho, _ = stats.spearmanr(mad_v, ex_v)  # = -0.5
    # 점 라벨 = 개입 종류 (내부 cell ID 미사용), 색은 panel (a) 곡선과 일치
    W5_LABEL = {"a": "self-loop", "b": "residual", "c": "self-loop\n+ residual"}
    W5_COLOR = {"a": "#2e86c1", "b": "#1e8449", "c": "#7d3c98"}
    W5_ANN = {"a": dict(xytext=(10, -3), ha="left"),
              "b": dict(xytext=(0, 11), ha="center"),
              "c": dict(xytext=(0, -26), ha="center")}
    for k in ("a", "b", "c"):
        axB.scatter(W5[k]["mad"], W5[k]["ex"], s=130, color=W5_COLOR[k],
                    edgecolor="k", lw=0.8, zorder=3)
        axB.annotate(W5_LABEL[k], (W5[k]["mad"], W5[k]["ex"]),
                     textcoords="offset points", fontsize=8.3, **W5_ANN[k])
    # 추세선 (downward — 직교/약한 역상관 시각화)
    z = np.polyfit(mad_v, ex_v, 1)
    xs = np.linspace(mad_v.min() - 0.01, mad_v.max() + 0.01, 20)
    axB.plot(xs, np.polyval(z, xs), ls="--", color="#7d3c98", alpha=0.6, lw=1.5)
    axB.text(0.36, 0.93,
             f"Spearman ρ = {rho:+.2f}  (best MAD recovery → lowest EX)",
             transform=axB.transAxes, fontsize=8.6, color="#5b2c6f", ha="center")
    axB.set_xlabel("L1 hi-deg MAD recovery  (mechanism fix)")
    axB.set_ylabel("end-to-end EX  (secondary backbone)")
    axB.set_ylim(0.284, 0.330)
    axB.set_xlim(0.270, 0.372)
    axB.set_title("(b) Selector quality ⊥ pipeline EX\n(MAD repaired 20–25×, e2e unchanged)",
                  fontsize=10.5)
    axB.grid(ls=":", alpha=0.4)
    # Sonnet anchor corroboration box (현 backbone 정합 — 별도 축, 좌하단 텍스트 박스)
    corr = (
        "Current Sonnet pipeline (corroboration):\n"
        "  GAT on  (α=0.5) : EX 0.6030\n"
        "  GAT off (α=1.0) : EX 0.6089\n"
        "  ΔEX = +0.0059  → removing the\n"
        "  over-smoothing-bearing GAT does\n"
        "  NOT lower e2e EX (slightly ↑)."
    )
    axB.text(0.015, 0.03, corr, transform=axB.transAxes, fontsize=7.5,
             va="bottom", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.45", fc="#f4ecf7", ec="#a569bd", lw=1.0))

    # ── Panel (c): Dirichlet energy depth (monotonic GROWTH) ─────
    bars = axC.bar(DE_LAYERS, DE_ENERGY, color="#2874a6", alpha=0.85,
                   edgecolor="k", lw=0.7, width=0.55, zorder=3)
    for b, e in zip(bars, DE_ENERGY):
        axC.text(b.get_x() + b.get_width() / 2, e + 1.5, f"{e:.2f}",
                 ha="center", fontsize=9, fontweight="bold")
    # growth ratio 화살표
    axC.annotate("×3.4 / layer", xy=(1.5, 35), fontsize=9,
                 color="#1a5276", ha="center", fontweight="bold")
    axC.annotate("", xy=(2.0, 50), xytext=(0.0, 8),
                 arrowprops=dict(arrowstyle="-|>", color="#1a5276", lw=1.8,
                                 connectionstyle="arc3,rad=-0.18"), zorder=2)
    axC.set_ylabel("Dirichlet energy")
    axC.set_xlabel("GAT depth")
    axC.set_ylim(0, 66)
    axC.set_title("(c) NOT classic over-smoothing\n(energy GROWS, does not collapse→0)",
                  fontsize=10.5)
    axC.grid(axis="y", ls=":", alpha=0.4)
    # per-edge normalized 보조 표기
    pe = " · ".join(f"L{i+1}={v:.3f}" for i, v in enumerate(DE_PER_EDGE))
    axC.text(0.5, -0.235, f"per-edge: {pe}", transform=axC.transAxes,
             fontsize=7.8, ha="center", color="#555")

    fig.suptitle(
        "Over-smoothing case study: intra-table column collapse is architecture-specific, "
        "fixable (20–25×), yet orthogonal to end-to-end EX",
        fontsize=12.5, y=1.015, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.99))

    png = os.path.join(OUT_DIR, f"fig_v6_oversmoothing_casestudy_{STAMP}.png")
    pdf = os.path.join(OUT_DIR, f"fig_v6_oversmoothing_casestudy_{STAMP}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"saved: {png}")
    print(f"saved: {pdf}")
    print(f"panel(b) Spearman(mad_recovery, e2e_EX) over 3 W5 cells = {rho:+.4f}")
    print(f"panel(a) L1 recovery factor c = {0.3463/0.0136:.1f}x, a = {0.2813/0.0136:.1f}x")


if __name__ == "__main__":
    main()
