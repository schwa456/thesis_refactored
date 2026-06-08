"""V6 unified train wrapper — seed control + experiment/checkpoint suffix override.

사용:
    PYTHONPATH=src python scripts/_v6_train_with_seed.py <SEED> <TRAINER_MODULE> <CONFIG_PATH>

예시:
    python scripts/_v6_train_with_seed.py 11 train_gat_s06 configs/training/v6_phase1/p1a_pairnorm_scale1_0.yaml
    python scripts/_v6_train_with_seed.py 22 train_gat   configs/training/v6_phase1/sweep_loss/sweep_temp0_05.yaml

동작:
    1. torch / numpy / random / PYTHONHASHSEED 를 seed 로 set
    2. config yaml 를 /tmp 위에 복사 + experiment_name / checkpoint_name 에 `_s{SEED}` suffix 박음
    3. <TRAINER_MODULE>.run_train(tmp_cfg) 호출 (train_gat 또는 train_gat_s06)

근거: planning/oversmoothing/oversmoothing_v6_plan_2026-06-01.md §1 Phase 1 + DECISIONS 2026-06-01 §V6-W1.
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def main():
    if len(sys.argv) < 4:
        print("Usage: _v6_train_with_seed.py <SEED> <TRAINER_MODULE> <CONFIG_PATH>",
              file=sys.stderr)
        sys.exit(2)

    seed = int(sys.argv[1])
    trainer = sys.argv[2]
    base_yaml = sys.argv[3]

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with open(base_yaml, 'r') as f:
        cfg = yaml.safe_load(f)

    base_stem = Path(base_yaml).stem
    suffix = f"s{seed}"
    cfg['experiment_name'] = f"{cfg.get('experiment_name', base_stem)}_{suffix}"
    base_ckpt = cfg.get('checkpoint_name', f"best_gat_{base_stem}.pt")
    ckpt_stem = Path(base_ckpt).stem
    cfg['checkpoint_name'] = f"{ckpt_stem}_{suffix}.pt"
    cfg.setdefault('training', {})['seed'] = seed

    tmp_dir = Path('/tmp')
    tmp_cfg = tmp_dir / f"v6_{cfg['experiment_name']}.yaml"
    with open(tmp_cfg, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"[v6_seed] seed={seed} trainer={trainer} cfg={tmp_cfg} "
          f"exp={cfg['experiment_name']} ckpt={cfg['checkpoint_name']}",
          flush=True)

    sys.path.insert(0, 'src')
    mod = __import__(trainer)
    mod.run_train(str(tmp_cfg))


if __name__ == "__main__":
    main()
