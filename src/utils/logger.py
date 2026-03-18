import logging
import os
import sys

def setup_logger(log_dir: str, exp_name: str = "experiment") -> logging.Logger:
    """
    실험 디렉토리에 run.log 파일을 생성하고, 콘솔과 파일에 동시에 로그를 출력하도록 설정합니다.
    """

    # 최상위 로거 이름 지정
    logger = logging.getLogger("ThesisRefactored")
    logger.setLevel(logging.INFO)

    # 중복 핸들러 방지
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 로그 포맷 정의
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. 콘솔 출력 핸들러(터미널 용)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. 파일 출력 핸들러 (logs 폴더 저장)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, 'run.log')
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"{'='*50}")
    logger.info(f"🚀 Logger Initialized for Experiment: {exp_name}")
    logger.info(f"{'='*50}")

    return logger

def get_logger(module_name: str):
    """
    각 파이썬 파일에서 호출하여 사용할 자식(Child) 로거를 반환합니다.
    사용법: logger = get_logger(__name__)
    """

    return logging.getLogger("ThesisRefactored").getChild(module_name)