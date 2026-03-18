import os
import yaml
import argparse
from pathlib import Path
from .logger import get_logger

logger = get_logger(__name__)

def load_and_merge_config(config_name: str) -> dict:
    """Base Config와 Experiment Config를 병합하고, 실험 전용 디렉토리를 생성합니다."""

    # 1. Base Config 로드
    base_dir = Path(__file__).resolve().parents[2]
    config_name = config_name.replace(".yaml", "")

    base_config_path = base_dir / "configs" / "base_config.yaml"
    if not base_config_path.exists():
        raise FileNotFoundError(f"🚨 Base config not found at: {base_config_path}")
        
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    # 3. Experiment Config 로드 및 병합
    exp_config_path = base_dir / "configs" / "experiments" / f"{config_name}.yaml"
    
    if exp_config_path.exists():
        with open(exp_config_path, 'r', encoding='utf-8') as f:
            exp_config = yaml.safe_load(f) or {}

        for key, value in exp_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        logger.info(f"Merged experiment config from {exp_config_path}")
    else:
        logger.warning(f"Experiment config '{exp_config_path}' not found. Using base config only.")

    # 💡 4. 버그 수정: exp_config가 아니라, '반환될 config 객체'에 이름을 새깁니다!
    config['experiment_name'] = config_name

    # 5. 디렉토리 자동 생성
    dirs_to_create = {
        'log_dir': base_dir / "logs" / config_name,
        'output_dir': base_dir / "outputs" / config_name,
        'checkpoint_dir': base_dir / "checkpoints" / config_name
    }

    config['paths'] = config.get('paths', {})
    for dir_key, dir_path in dirs_to_create.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        config['paths'][dir_key] = str(dir_path)
    
    return config

def get_args_and_config():
    parser = argparse.ArgumentParser(description="Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Name of the experiment")
    args = parser.parse_args()

    config = load_and_merge_config(args.config)
    return args, config