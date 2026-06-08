#!/usr/bin/env python3
"""V7 Extractor chain 6축 분석: Pareto / Δvs M4 / STE-vs-FKP / k-trajectory / axis / narrative.
Input: outputs/analysis/v7_extractor_chain_2026-06-05.json (measure 단계 산출)."""
import json, os

ROOT = "/home/hyeonjin/thesis_refactored"
JP = os.path.join(ROOT, "outputs/analysis/v7_extractor_chain_2026-06-05.json")
with open(JP) as f:
    D = json.load(f)
M4 = D["m4_anchor"]
cells = {c["cell"]: c for c in D["cells"]}

# F1 ranking (extractor cells only, exclude baseline seeds dup → 1 baseline)
ext = [c for c in D["cells"] if c["wave"] in ("W2", "W3")]
print("=== F1 RANKING (top 8) ===")
for c in sorted(ext, key=lambda x: -x["ext_F1"])[:8]:
    print(f"{c['cell']:16s} R={c['ext_R']:.4f} P={c['ext_P']:.4f} F1={c['ext_F1']:.4f} n={c['ext_n_avg']:.2f}")

# vs M4 anchor delta + count F1 >= M4
print("\n=== vs M4 anchor (F1=%.4f) ===" % M4["F1"])
ge = [c for c in ext if c["ext_F1"] >= M4["F1"]]
print(f"cells with F1 >= M4 anchor: {len(ge)}/{len(ext)} = {100*len(ge)/len(ext):.1f}%")
for c in sorted(ext, key=lambda x: -x["ext_F1"])[:5]:
    print(f"  {c['cell']:16s} dF1={c['ext_F1']-M4['F1']:+.4f} dP={c['ext_P']-M4['P']:+.4f} "
          f"dR={c['ext_R']-M4['R']:+.4f} compact={M4['n_avg']/c['ext_n_avg']:.1f}x")

# Pareto frontier (maximize R and P)
def pareto(points):
    # points: list of (cell, R, P)
    front = []
    for a in points:
        dominated = False
        for b in points:
            if b is a:
                continue
            if b[1] >= a[1] and b[2] >= a[2] and (b[1] > a[1] or b[2] > a[2]):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda x: -x[1])

pts = [(c["cell"], c["ext_R"], c["ext_P"]) for c in ext]
pts.append(("M4_anchor", M4["R"], M4["P"]))
front = pareto(pts)
print("\n=== PARETO FRONTIER (R-P, V7 cells + M4) ===")
for cell, R, P in front:
    print(f"  {cell:16s} R={R:.4f} P={P:.4f}")
m4_on = any(c == "M4_anchor" for c, _, _ in front)
print(f"M4 anchor on Pareto front: {m4_on}")

# STE vs FKP @ same k
print("\n=== STE vs FKP @ same k (FK path bridge 효과) ===")
print(f"{'k':>4} | {'STE_R':>7} {'STE_P':>7} | {'FKP_R':>7} {'FKP_P':>7} | {'dR(FKP-STE)':>11} {'dP':>8}")
for k in ["005", "010", "015", "020", "030", "050", "100"]:
    s = cells.get(f"ste_k{k}"); fp = cells.get(f"fkp_k{k}")
    if s and fp:
        print(f"{int(k):>4} | {s['ext_R']:.4f}  {s['ext_P']:.4f} | {fp['ext_R']:.4f}  {fp['ext_P']:.4f} "
              f"| {fp['ext_R']-s['ext_R']:+.4f}     {fp['ext_P']-s['ext_P']:+.4f}")

# k trajectory
print("\n=== k TRAJECTORY ===")
for fam in ["ste", "fkp"]:
    print(f"[{fam.upper()}]")
    for k in ["005", "010", "015", "020", "030", "050", "100"]:
        c = cells.get(f"{fam}_k{k}")
        if c:
            print(f"  k={int(k):>3} R={c['ext_R']:.4f} P={c['ext_P']:.4f} F1={c['ext_F1']:.4f} n={c['ext_n_avg']:.2f}")

# axis variants
print("\n=== AXIS VARIANTS ===")
print("[STE] (k=20 base = ste_k020)")
base = cells["ste_k020"]
print(f"  ste_k020(topk,col+tbl,cap)  R={base['ext_R']:.4f} P={base['ext_P']:.4f} F1={base['ext_F1']:.4f} n={base['ext_n_avg']:.2f}")
for cell, lab in [("ste_ax_thr05", "threshold0.5"), ("ste_ax_thr03", "threshold0.3"),
                  ("ste_ax_colonly", "column-only terminal"), ("ste_ax_nocap", "cap_to_k=False")]:
    c = cells[cell]
    print(f"  {cell:16s}({lab})  R={c['ext_R']:.4f} P={c['ext_P']:.4f} F1={c['ext_F1']:.4f} "
          f"n={c['ext_n_avg']:.2f}  dF1={c['ext_F1']-base['ext_F1']:+.4f}")
print("[FKP] (k=20 base = fkp_k020, terminal=column)")
base = cells["fkp_k020"]
print(f"  fkp_k020(topk,col,fk=T)     R={base['ext_R']:.4f} P={base['ext_P']:.4f} F1={base['ext_F1']:.4f} n={base['ext_n_avg']:.2f}")
for cell, lab in [("fkp_ax_coltbl", "col+table terminal"), ("fkp_ax_thr05", "threshold0.5"),
                  ("fkp_ax_nofk", "use_fk_paths=False")]:
    c = cells[cell]
    print(f"  {cell:16s}({lab})  R={c['ext_R']:.4f} P={c['ext_P']:.4f} F1={c['ext_F1']:.4f} "
          f"n={c['ext_n_avg']:.2f}  dF1={c['ext_F1']-base['ext_F1']:+.4f}")
# FK path 순기여 (fkp_k020 use_fk=T vs fkp_ax_nofk use_fk=F)
fk_on = cells["fkp_k020"]; fk_off = cells["fkp_ax_nofk"]
print(f"\n  FK path 순기여 (fkp_k020 - fkp_ax_nofk): dR={fk_on['ext_R']-fk_off['ext_R']:+.4f} "
      f"dP={fk_on['ext_P']-fk_off['ext_P']:+.4f} dF1={fk_on['ext_F1']-fk_off['ext_F1']:+.4f} "
      f"dn={fk_on['ext_n_avg']-fk_off['ext_n_avg']:+.2f}")

# R gate pass cells
print("\n=== R>=0.90 GATE PASS ===")
for c in sorted(ext, key=lambda x: -x["ext_R"]):
    if c["ext_R"] >= 0.90:
        ste_p = c["ext_P"] >= 0.30; fkp_p = c["ext_P"] >= 0.25
        gate = "STE-gate✓" if (c["wave"] == "W3" and ste_p) else ("FKP-gate✓" if (c["wave"] == "W2" and fkp_p) else "P-gate✗")
        print(f"  {c['cell']:16s} R={c['ext_R']:.4f} P={c['ext_P']:.4f} [{gate}]")
