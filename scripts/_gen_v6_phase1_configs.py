"""V6-W1 Phase 1 config generator.

산출: configs/training/v6_phase1/{p1a_*, p1b_*, p1c_*, p1d_combo}.yaml (9 main)
      configs/training/v6_phase1/sweep_loss/{temp_*, hn_*, bce_infonce_*}.yaml (8 parallel sweep)

근거:
  - planning/oversmoothing/oversmoothing_v6_plan_2026-06-01.md §1 Phase 1 + §4 Ablation Matrix
  - planning/DECISIONS.md 2026-06-01 §V6-W1 활성 launch
  - planning/oversmoothing/oversmoothing_rfp_2026-06-01.md §5 (병행 sweep) + §6

Anchor:
  - configs/training/diameter_layers/train_qcond_nl3.yaml (M4 anchor QCond NL3 architecture)
  - in/hidden/out=384/256/256, num_layers=3, heads=4, query_conditioned=True

Trainer:
  - Main matrix (P1a/P1b/P1c/P1d): src/train_gat_s06.py + V6-W1 classes (SchemaHeteroGATv2)
    → DirectClassifierHead, BCE loss (anti_collapse=0)
  - Parallel sweep (loss-side): src/train_gat.py (DualTowerProjector + BCE+InfoNCE)
    → infonce_lambda + temperature + num_hard_negatives 스윕
"""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "configs" / "training" / "v6_phase1"
SWEEP_DIR = OUT_DIR / "sweep_loss"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_DIR.mkdir(parents=True, exist_ok=True)


def _common_paths():
    return {
        "train_json": "/SSL_NAS/peoples/khj/thesis/train/train.json",
        "train_db_dir": "/SSL_NAS/peoples/khj/thesis/train/train_databases",
        "test_json": "/home/hyeonjin/thesis_refactored/data/raw/BIRD_dev/dev.json",
        "test_db_dir": "/home/hyeonjin/thesis_refactored/data/raw/BIRD_dev/dev_databases",
        "checkpoint_dir": "./outputs/checkpoints",
        "cache_dir": "./data/processed",
    }


# =============================================================================
# Main matrix — V6-W1 isolation cells (train_gat_s06.py + DirectClassifierHead)
# =============================================================================
def _main_base():
    """V6-W1 main matrix base: QCond NL3 + pure isolation (no other mitigation)."""
    return {
        "project_name": "Text-to-SQL-Alignment",
        "builder": {
            "type": "EnrichedHeteroGraphBuilder",
            "tables_json_path": "/SSL_NAS/peoples/khj/thesis/train/train_tables.json",
        },
        "paths": _common_paths(),
        "model": {
            # NL3 (M4 anchor)
            "in_channels": 384,
            "hidden_channels": 256,
            "out_channels": 256,
            "num_layers": 3,
            "heads": 4,
            "dropout": 0.1,
            "classifier_hidden": 256,
            # QCond (M4 anchor)
            "query_conditioned": True,
            "query_supernode": False,
            "dual_stream": False,
            # 모든 mitigation OFF (각 cell 위 하나만 ON)
            "gat_layer_type": "standard",      # V4/V5/V6 dispatch
            "pairnorm_mode": "none",
            "pairnorm_scale": 1.0,
            "initial_residual_alpha": 0.0,
            "jumping_knowledge": "none",
            "v6w1_pairnorm_scale": 1.0,
            "v6w1_jk_mode": "concat",
            "gcnii_beta_lambda": 0.5,
            "capture_layerwise_outputs": False,
            "drop_message_p": 0.0,
            "use_layernorm_pre_softmax": False,
            "aggregation_type": "mean",
            "aero_hop_attention": False,
            "aero_cumulative_attention": False,
            "aero_cumulative_decay": 1.0,
            "softplus_symmetric_norm": True,
            "num_layers_mode": "fixed",
        },
        "training": {
            "epochs": 300,
            "batch_size": 8,
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "pos_weight": 100.0,
            "val_split": 0.1,
            "recall_k": 15,
            "loss_type": "bce",
            "anti_collapse_weight": 0.0,
        },
    }


def write_main_matrix():
    """9 unique configs (P1a×3 + P1b×3 + P1c×2 + P1d×1)."""
    # P1a — PairNorm 단독 (scale 스윕 {0.5, 1.0, 2.0})
    for scale in [0.5, 1.0, 2.0]:
        cfg = _main_base()
        tag = f"p1a_pairnorm_scale{str(scale).replace('.', '_')}"
        cfg["experiment_name"] = f"v6w1_{tag}"
        cfg["checkpoint_name"] = f"best_gat_v6w1_{tag}.pt"
        cfg["model"]["gat_layer_type"] = "pairnorm"
        cfg["model"]["v6w1_pairnorm_scale"] = float(scale)
        cfg["model"]["capture_layerwise_outputs"] = True
        path = OUT_DIR / f"{tag}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"  wrote {path.relative_to(ROOT)}")

    # P1b — GCNII Initial Residual 단독 (α 스윕 {0.05, 0.1, 0.2})
    for alpha in [0.05, 0.1, 0.2]:
        cfg = _main_base()
        tag = f"p1b_gcnii_alpha{str(alpha).replace('.', '_')}"
        cfg["experiment_name"] = f"v6w1_{tag}"
        cfg["checkpoint_name"] = f"best_gat_v6w1_{tag}.pt"
        cfg["model"]["gat_layer_type"] = "gcnii_ir"
        cfg["model"]["initial_residual_alpha"] = float(alpha)
        cfg["model"]["gcnii_beta_lambda"] = 0.5
        cfg["model"]["capture_layerwise_outputs"] = True
        path = OUT_DIR / f"{tag}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"  wrote {path.relative_to(ROOT)}")

    # P1c — JK 단독 ({concat, max})
    for mode in ["concat", "max"]:
        cfg = _main_base()
        tag = f"p1c_jk_{mode}"
        cfg["experiment_name"] = f"v6w1_{tag}"
        cfg["checkpoint_name"] = f"best_gat_v6w1_{tag}.pt"
        cfg["model"]["gat_layer_type"] = "jk"
        cfg["model"]["v6w1_jk_mode"] = mode
        cfg["model"]["jumping_knowledge"] = mode
        cfg["model"]["capture_layerwise_outputs"] = True
        path = OUT_DIR / f"{tag}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"  wrote {path.relative_to(ROOT)}")

    # P1d — P1a + P1b + P1c 조합 (middle defaults: pairnorm=1.0, alpha=0.1, jk=concat)
    cfg = _main_base()
    tag = "p1d_combo"
    cfg["experiment_name"] = f"v6w1_{tag}"
    cfg["checkpoint_name"] = f"best_gat_v6w1_{tag}.pt"
    cfg["model"]["gat_layer_type"] = "dropin_combo"
    cfg["model"]["v6w1_pairnorm_scale"] = 1.0
    cfg["model"]["initial_residual_alpha"] = 0.1
    cfg["model"]["v6w1_jk_mode"] = "concat"
    cfg["model"]["jumping_knowledge"] = "concat"
    cfg["model"]["gcnii_beta_lambda"] = 0.5
    cfg["model"]["capture_layerwise_outputs"] = True
    path = OUT_DIR / f"{tag}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"  wrote {path.relative_to(ROOT)}")


# =============================================================================
# Parallel sweep — Loss-side ablation (train_gat.py + DualTowerProjector)
# =============================================================================
def _sweep_base():
    """Loss-side sweep base: M4 anchor QCond NL3 + DualTowerProjector (train_gat.py)."""
    return {
        "project_name": "Text-to-SQL-Alignment",
        "paths": _common_paths(),
        "model": {
            "in_channels": 384,
            "hidden_channels": 256,
            "out_channels": 256,
            "num_layers": 3,
            "heads": 4,
            "dropout": 0.1,
            "query_conditioned": True,
        },
        "training": {
            "epochs": 300,
            "batch_size": 8,
            "learning_rate": 0.0001,
            "weight_decay": 0.00001,
            "pos_weight": 100.0,
            "val_split": 0.1,
            "recall_k": 15,
            "infonce_lambda": 0.5,
            "temperature": 0.07,
            "num_hard_negatives": 15,
        },
    }


def write_sweep_loss():
    """8 unique configs: T×3 + HN×2 + BCE:InfoNCE ratio×3."""
    # Temperature sweep {0.05, 0.1, 0.2}
    for T in [0.05, 0.1, 0.2]:
        cfg = _sweep_base()
        tag = f"sweep_temp{str(T).replace('.', '_')}"
        cfg["experiment_name"] = f"v6w1_{tag}"
        cfg["checkpoint_name"] = f"best_gat_v6w1_{tag}.pt"
        cfg["training"]["temperature"] = float(T)
        path = SWEEP_DIR / f"{tag}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"  wrote {path.relative_to(ROOT)}")

    # Hard negative on/off
    for hn_label, n_hn in [("on", 15), ("off", 999999)]:
        cfg = _sweep_base()
        tag = f"sweep_hn_{hn_label}"
        cfg["experiment_name"] = f"v6w1_{tag}"
        cfg["checkpoint_name"] = f"best_gat_v6w1_{tag}.pt"
        cfg["training"]["num_hard_negatives"] = int(n_hn)
        path = SWEEP_DIR / f"{tag}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"  wrote {path.relative_to(ROOT)}")

    # BCE:InfoNCE ratio {0.5:0.5, 0.7:0.3, 0.3:0.7} → infonce_lambda = bce_ratio/infonce_ratio
    # 0.5:0.5 → λ = 0.5/0.5 = 1.0
    # 0.7:0.3 → λ = 0.3/0.7 ≈ 0.43
    # 0.3:0.7 → λ = 0.7/0.3 ≈ 2.33
    for bce, info, lam in [(0.5, 0.5, 1.0), (0.7, 0.3, 0.43), (0.3, 0.7, 2.33)]:
        cfg = _sweep_base()
        tag = f"sweep_bceinfo_{str(bce).replace('.', '_')}__{str(info).replace('.', '_')}"
        cfg["experiment_name"] = f"v6w1_{tag}"
        cfg["checkpoint_name"] = f"best_gat_v6w1_{tag}.pt"
        cfg["training"]["infonce_lambda"] = float(lam)
        path = SWEEP_DIR / f"{tag}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"  wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    print("=== V6-W1 main matrix (9 configs) ===")
    write_main_matrix()
    print("=== V6-W1 parallel sweep loss-side (8 configs) ===")
    write_sweep_loss()
    print("✅ Done.")
