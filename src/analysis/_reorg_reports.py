#!/usr/bin/env python3
"""analysis_results/ 92 .md 를 연구 주제별 폴더로 정리 + 상호링크/figs 경로 자동 보정 + INDEX 생성.
dry-run: python _reorg_reports.py  (매핑만 출력)
execute: python _reorg_reports.py --go
"""
import os, re, sys, subprocess

ROOT = "/home/hyeonjin/thesis_refactored/notebooks/analysis_results"

# 카테고리: (폴더명, 설명, [파일 prefix/정확명 패턴])
CATS = [
    ("01_v6_oversmoothing_chain", "V6 chain (W0~W6) + MA calibration + disconnect/Filter-Dominance",
     ["v6_phase0", "v6_phase1", "v6_phase2", "v6_phase3", "v6_intra_table", "v6_w5", "v6w2_",
      "v6w6_", "v1_v5_retrospective", "alpha_gat_contribution", "ma_raise_theta",
      "selector_monitor_inference", "multiseed_robustness"]),
    ("02_v1_v5_dsn_mitigation", "V1~V5 DSN over-smoothing 14-trial null + mech(ii-b)",
     ["dsn_", "mechanism_final", "query_conditioned_training", "selector_gold_score_discrimination",
      "raw_score_distribution_for_directed_topk", "diameter_layers_sweep"]),
    ("03_extractor_pcst_steiner", "Extractor / PCST / Steiner / FK pathfinding",
     ["v7_extractor", "fk_steiner", "steiner_backbone", "direction_c_grast", "direction_c_gt",
      "direction_c_inferred", "selector_extractor_math"]),
    ("04_filter", "Filter (XiYan/GLM/AdaptiveDepth/proposal)",
     ["filter_proposal", "filter_sweep_glm", "sgbe_filter", "phase4_2_conditional_filter"]),
    ("05_ablation_waves", "Wave6~16 / Phase1~4 ablation matrices + capacity/builder axis",
     ["wave6", "wave7", "wave8", "wave9", "wave16", "phase1_sensitivity", "phase2_grid",
      "phase3_shapley", "phase4_1_integrated", "anchor_capacity", "builder_axis",
      "three_caveat_recalibration", "stagewise_qcond"]),
    ("06_selector_encoder_bottleneck", "Selector score / encoder backbone / GAT bottleneck / alpha-plateau",
     ["s06_bottleneck", "alpha_plateau", "v5_d1_plm", "v5_inference_phase1"]),
    ("07_baselines_audits", "Baselines + 측정 audit/정정 + evaluator/denominator 검증",
     ["glm_baseline", "oracle_baseline", "measurement_framework_audit", "m2_r_inconsistency",
      "evaluator_alias_fix", "recall_gained_denominator", "diagnostic_state",
      "direction_a_full_schema", "direction_a_rsl", "direction_b_hn"]),
    ("08_paper_narratives", "Paper 초안 / outline / mechanism narrative",
     ["paper_"]),
    ("09_generalization", "Cross-dataset generalization (Spider 2.0)",
     ["g_s2_", "spider2"]),
    ("10_misc_planning", "Advisor ideas / improvement opportunities 등 기타",
     ["advisor_meeting", "improvement_opportunities"]),
]


def categorize(fn):
    for folder, _desc, pats in CATS:
        for p in pats:
            if fn.startswith(p) or p in fn:
                return folder
    return "10_misc_planning"  # fallback


def main():
    go = "--go" in sys.argv
    mds = sorted(f for f in os.listdir(ROOT) if f.endswith(".md") and f != "_INDEX.md")
    mapping = {fn: categorize(fn) for fn in mds}
    # 파일명 → 새 폴더 (링크 보정용)
    file2folder = dict(mapping)

    # 카테고리별 출력
    from collections import defaultdict
    bycat = defaultdict(list)
    for fn, cat in mapping.items():
        bycat[cat].append(fn)
    for folder, desc, _ in CATS:
        print(f"\n## {folder}  ({len(bycat[folder])}개) — {desc}")
        for fn in sorted(bycat[folder]):
            print(f"   {fn}")
    print(f"\n총 {len(mds)}개 → {len([c for c in bycat if bycat[c]])} 폴더")
    if not go:
        print("\n[dry-run] 실행하려면 --go")
        return

    # 1) 폴더 생성 + git mv
    for folder, _, _ in CATS:
        os.makedirs(os.path.join(ROOT, folder), exist_ok=True)
    for fn, cat in mapping.items():
        src = os.path.join(ROOT, fn); dst = os.path.join(ROOT, cat, fn)
        subprocess.run(["git", "mv", src, dst], cwd=ROOT, check=False)

    # 2) 링크 보정: 각 이동된 md 위 ](X.md) 와 ](figs/Y) 를 새 상대경로로
    link_md = re.compile(r"\]\(([a-zA-Z0-9_./-]+\.md)\)")
    link_fig = re.compile(r"\]\((\.\./)*figs/")
    for fn, cat in mapping.items():
        path = os.path.join(ROOT, cat, fn)
        if not os.path.exists(path):
            continue
        txt = open(path).read()
        # figs 는 top-level 유지 → 한 단계 위
        txt = link_fig.sub("](../figs/", txt)

        def fix_md(m):
            target = os.path.basename(m.group(1))
            if target in file2folder:
                tcat = file2folder[target]
                rel = target if tcat == cat else f"../{tcat}/{target}"
                return f"]({rel})"
            return m.group(0)
        txt = link_md.sub(fix_md, txt)
        open(path, "w").write(txt)

    # 3) INDEX 생성
    idx = ["# 분석 리포트 인덱스 (notebooks/analysis_results/)\n",
           "> 연구 주제별 폴더 정리 (2026-06-10). figs/ 는 top-level 유지.\n"]
    for folder, desc, _ in CATS:
        files = sorted(bycat[folder])
        if not files:
            continue
        idx.append(f"\n## {folder} — {desc} ({len(files)})\n")
        for fn in files:
            idx.append(f"- [{fn}]({folder}/{fn})")
    open(os.path.join(ROOT, "_INDEX.md"), "w").write("\n".join(idx) + "\n")
    print("\n✅ 이동 + 링크보정 + _INDEX.md 완료")


if __name__ == "__main__":
    main()
