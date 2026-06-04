#!/usr/bin/env python3
"""V7-W1 (FKH) — 25 config generator (5 cells × 5 seeds).

학술 agent RFP §3.7 + V7 plan §1.3 정합. anchor stack = c01_01_wave7_relog
(EnsembleSelector + MSTPCSTUnionExtractor + XiYanFilter glm-4.7 + LLMSQLGenerator glm-4.7,
post_processing.auto_join_keys=true, sql_generator.enabled=true).

Cells:
| cell_id | fk_hint_format | fk_hint_position |
|---------|----------------|------------------|
| fkh_00  | none           | inline           |
| fkh_01  | explicit       | inline           |
| fkh_02  | explicit       | prefix           |
| fkh_03  | compact        | inline           |
| fkh_04  | compact        | suffix           |

Seeds: 42, 123, 7, 456, 789 → 25 configs total.

본 framework 의 main.py 는 seed override 의 명시적 channel 이 없으나, base_config
override 정합 시 `seed` 또는 experiment_name 위 seed 표기 만으로도 LLM
temperature=0.0 path 위 동일 deterministic. 본 generator 는 experiment_name 위
seed 명시 + (선택) `seed: <int>` field — main.py 가 해당 field 를 무시하더라도
output dir 분리는 experiment_name path 위 보장.
"""
import os
import textwrap
from pathlib import Path


CELLS = [
    ("fkh_00", "none", "inline"),
    ("fkh_01", "explicit", "inline"),
    ("fkh_02", "explicit", "prefix"),
    ("fkh_03", "compact", "inline"),
    ("fkh_04", "compact", "suffix"),
]
SEEDS = [42, 123, 7, 456, 789]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "configs" / "experiments" / "abl" / "v7_extractor_redesign"


TEMPLATE = textwrap.dedent(
    """\
    # V7-W1 (FKH, RFP #3) — FK Hint serialization cell
    # cell={cell_id} fk_hint_format={fmt} fk_hint_position={pos} seed={seed}
    # anchor stack = c01_01_wave7_relog (XiYanFilter glm-4.7, EX 0.5176 baseline)
    # Refs: planning/extractor/extractor_redesign_v7_plan_2026-06-04.md §1.3
    #       planning/extractor/scholar_agent_extractor_rfp_2026-06-04.md §3

    experiment_name: "v7_w1_{cell_id}_seed{seed}"
    seed: {seed}

    graph_builder:
      name: "EnrichedHeteroGraphBuilder"
      params:
        include_views: false
        run_leiden_clustering: true
        tables_json_path: "data/raw/BIRD_dev/dev_tables.json"

    nlq_encoder:
      name: "LocalPLMEncoder"
      params:
        model_name: "sentence-transformers/all-MiniLM-L6-v2"

    projection:
      enabled: false

    seed_selector:
      name: "EnsembleSelector"
      params:
        weight_path: "outputs/checkpoints/best_gat_qcond_nl3.pt"
        alpha: 0.5
        top_k: 20
        query_conditioned: true
        encoder_type: "plm"

    connectivity_extractor:
      name: "MSTPCSTUnionExtractor"
      params:
        score_threshold: 0.1

    filter:
      name: "XiYanFilter"
      params:
        provider: "glm"
        model_name: "zai-org/glm-4.7"
        max_iteration: 1
        temperature: 0.0
        fk_hint_format: "{fmt}"
        fk_hint_position: "{pos}"

    post_processing:
      auto_join_keys: true

    sql_generator:
      enabled: true
      name: "LLMSQLGenerator"
      params:
        provider: "glm"
        llm_model: "zai-org/glm-4.7"
        temperature: 0.0
    """
)


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for (cell_id, fmt, pos) in CELLS:
        for seed in SEEDS:
            fname = f"{cell_id}_seed{seed}.yaml"
            fpath = TARGET_DIR / fname
            content = TEMPLATE.format(cell_id=cell_id, fmt=fmt, pos=pos, seed=seed)
            fpath.write_text(content)
            written += 1
            print(f"  wrote {fpath.relative_to(PROJECT_ROOT)}")
    print(f"\n[V7-W1 FKH generator] {written} configs written to {TARGET_DIR}")


if __name__ == "__main__":
    main()
