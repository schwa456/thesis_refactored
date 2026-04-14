#!/bin/bash
# ============================================================
# 지도교수 면담 아이디어 Ablation Study Runner
# (신규 ID 체계 반영 — EXPERIMENT_ID_MIGRATION.md 참조)
# ============================================================
# 사용법: bash run_idea_ablation.sh [실행할 그룹]
#   그룹: all | pcst | alpha | combined | baseline
# ============================================================

set -e
cd "$(dirname "$0")"

GROUP="${1:-all}"

BASELINE_CONFIGS=(
    "s03_gat_ensemble/a02_adaptive_pcst/s03_a02_01_combined"
)

ALPHA_CONFIGS=(
    "abl/a02_alpha_sweep/abl_a02_02_alpha075"
    "abl/a02_alpha_sweep/abl_a02_03_alpha070"
)

PCST_CONFIGS=(
    "s03_gat_ensemble/a03_product_cost/s03_a03_01_product_cost"
    "s03_gat_ensemble/a03_product_cost/s03_a03_02_product_cost_xiyan"
    "s03_gat_ensemble/a05_component_aware/s03_a05_01_component_aware"
    "s03_gat_ensemble/a06_component_product/s03_a06_01_product_component"
    "s03_gat_ensemble/a06_component_product/s03_a06_02_product_component_xiyan"
    "s03_gat_ensemble/a04_steiner_backbone/s03_a04_01_steiner"
)

COMBINED_CONFIGS=(
    "s03_gat_ensemble/a06_component_product/s03_a06_03_idea124_combined"
    "s03_gat_ensemble/a06_component_product/s03_a06_04_idea124_combined_xiyan"
)

run_experiments() {
    local configs=("$@")
    for config in "${configs[@]}"; do
        name=$(basename "$config")
        echo ""
        echo "============================================================"
        echo "  Running: ${name}"
        echo "============================================================"
        python src/main.py --config "experiments/${config}" 2>&1 | tee "logs/${name}.log"
        echo "  Completed: ${name}"
    done
}

echo "Advisor Meeting Ideas - Ablation Study [Group: ${GROUP}]"

case $GROUP in
    baseline) run_experiments "${BASELINE_CONFIGS[@]}" ;;
    alpha)    run_experiments "${ALPHA_CONFIGS[@]}" ;;
    pcst)     run_experiments "${PCST_CONFIGS[@]}" ;;
    combined) run_experiments "${COMBINED_CONFIGS[@]}" ;;
    all)
        run_experiments "${BASELINE_CONFIGS[@]}"
        run_experiments "${ALPHA_CONFIGS[@]}"
        run_experiments "${PCST_CONFIGS[@]}"
        run_experiments "${COMBINED_CONFIGS[@]}"
        ;;
    *) echo "Unknown group: ${GROUP}"; exit 1 ;;
esac
