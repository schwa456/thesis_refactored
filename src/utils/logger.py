# src/utils/logger.py

import logging
import os
import sys
from datetime import datetime

def setup_logger(log_dir: str, exp_name: str = "experiment", sub_dir: str = "") -> logging.Logger:
    """
    지정된 sub_dir(train 또는 eval) 아래에 로그 파일을 생성합니다.
    """
    # 최종 로그 저장 경로 설정 (예: ./logs/train)
    final_log_dir = os.path.join(log_dir, sub_dir) if sub_dir else log_dir
    os.makedirs(final_log_dir, exist_ok=True)

    logger = logging.getLogger("ThesisRefactored")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. 콘솔 출력
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. 파일 출력 (타임스탬프 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{exp_name}_{timestamp}.log"
    log_file_path = os.path.join(final_log_dir, log_filename)
    
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"📂 Log directory: {final_log_dir}")
    logger.info(f"💾 Log file: {log_filename}")
    
    return logger

def get_logger(module_name: str):
    return logging.getLogger("ThesisRefactored").getChild(module_name)