"""V6-W0 baseline seed wrapper.

사용:
    PYTHONPATH=src python scripts/_v6_w0_train_with_seed.py <SEED> <BASE_YAML>

동작:
    1. SEED 를 torch / numpy / random / PYTHONHASHSEED 위 설정
    2. BASE_YAML 을 로드 후 experiment_name + checkpoint_name 을 seed suffix 로 override
    3. /tmp 위 임시 yaml 작성 후 src.train_gat.run_train 호출

근거: planning/oversmoothing/oversmoothing_v6_plan_2026-06-01.md §1 Phase 0
       — 베이스라인 시드 3개 다중 실행 (seed 변동 폭 측정)
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def main():
    if len(sys.argv) < 3:
        print("Usage: _v6_w0_train_with_seed.py <SEED> <BASE_YAML>", file=sys.stderr)
        sys.exit(2)
    seed = int(sys.argv[1])
    base_yaml = sys.argv[2]

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with open(base_yaml, 'r') as f:
        cfg = yaml.safe_load(f)

    tag = f"v6w0_s{seed}"
    cfg['experiment_name'] = f"gat_qcond_nl3_{tag}"
    cfg['checkpoint_name'] = f"best_gat_qcond_nl3_{tag}.pt"
    # 시드 정보를 cfg 에도 박아둠 (downstream 추적 위)
    cfg.setdefault('training', {})['seed'] = seed

    tmp_dir = Path('/tmp')
    tmp_cfg = tmp_dir / f"v6w0_s{seed}_train.yaml"
    with open(tmp_cfg, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[v6_w0_seed] seed={seed} tmp_cfg={tmp_cfg} "
          f"experiment_name={cfg['experiment_name']} "
          f"checkpoint_name={cfg['checkpoint_name']}", flush=True)

    sys.path.insert(0, 'src')
    from train_gat import run_train
    run_train(str(tmp_cfg))


if __name__ == "__main__":
    main()
