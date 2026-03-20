import subprocess
import pandas as pd
import os
from utils.logger import get_logger

logger = get_logger(__name__)

def run_benchmarks():
    experiments = [
        "baseline/preliminary_vector_only.yaml",
        "baseline/preliminary_graph_expansion.yaml",
        "baseline/preliminary_graph_and_agent.yaml",
        "baseline/baseline_xiyansql.yaml",
        "baseline/baseline_g_retriever.yaml",
        "baseline/baseline_linkalign.yaml",
    ]

    logger.info(f"📚 Starting total {len(experiments)} experiments...")

    for cfg in experiments:
        cfg_path = f"./configs/experiments/{cfg}"
        if not os.path.exists(cfg_path):
            logger.error(f"[Error] Config Missing: {cfg_path}")
            continue
        
        logger.info(f"Running: {cfg}")

        try:
            subprocess.run(["python", "src/main.py", "--config", cfg_path], check=True)
            logger.info(f"Finished: {cfg}")
        except Exception as e:
            logger.error(f"[Error] Failed: {cfg} with error {e}")
        
    
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