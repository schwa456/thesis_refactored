import subprocess
import pandas as pd
import os
import sys
import signal
import time

from utils.logger import setup_logger, get_logger

benchmark_log_dir = "./logs/benchmark"
os.makedirs(benchmark_log_dir, exist_ok=True)
benchmark_exp_name = "benchmark_run"

setup_logger(log_dir=benchmark_log_dir, exp_name=benchmark_exp_name)
logger = get_logger(__name__)

def sigterm_handler(signum, frame):
    logger.warning("🛑 [System Signal] Received SIGTERM (kill command). Initiating graceful shutdown...")
    raise KeyboardInterrupt

def run_benchmarks():
    signal.signal(signal.SIGTERM, sigterm_handler)

    experiments = [
        "baselines/preliminary_vector_only.yaml",
        "baselines/preliminary_graph_expansion.yaml",
        "baselines/preliminary_graph_and_agent.yaml",
        "baselines/baseline_xiyansql.yaml",
        "baselines/baseline_g_retriever.yaml",
        "baselines/baseline_linkalign.yaml",
        "experiments/experiment_gat_classifier.yaml",
        "experiments/experiment_gat_multi_agent.yaml",
        "experiments/experiment_gat_classifier_multi_agent.yaml",
    ]

    logger.info(f"📚 Starting total {len(experiments)} experiments...")

    try:
        for cfg in experiments:
            start_time = time.perf_counter()
            cfg_path = f"configs/{cfg}"
            if not os.path.exists(cfg_path):
                logger.error(f"[Error] Config Missing: {cfg_path}")
                continue
            logger.info(f"{'=' * 80}")
            logger.info(f"Running: {cfg}")

            try:
                subprocess.run(["python", "src/main.py", "--config", cfg], check=True)
                logger.info(f"Finished: {cfg}")
            except Exception as e:
                logger.error(f"[Error] Failed: {cfg} with exit code {e.returncode}")

            elapsed_exp_time = time.perf_counter() - start_time
            logger.info(f"Experiment [{cfg}] took {elapsed_exp_time:.4f} seconds")
            logger.info(f"{'=' * 80}")
    except KeyboardInterrupt:
        # 💡 [핵심 방어 로직] 터미널에서 Ctrl+C가 입력된 경우
        print("\n🛑 [Benchmark Stopped] Ctrl+C detected! Terminating the benchmark loop...")
        logger.warning("Benchmark execution interrupted by the user.")
        # 루프를 즉시 종료하고 아래의 Summary 출력 단계로 넘어갑니다.
    except Exception as e:
        logger.error(f"[Critical Error] Unexpected failure in benchmark loop: {e}")
        
    
    summary_path = "./outputs/summary_all.csv"
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        print("\n" + "=" * 80)
        print("🏆 FINAL COMPARISON TABLE FOR THESIS")
        print("="*80)
        # 가장 최근에 수행된 실험들이 아래에 쌓이므로, 마지막 N개를 보여줌
        print(summary_df.tail(len(experiments)).to_string(index=False))
        print("="*80)

if __name__ == "__main__":
    run_benchmarks()