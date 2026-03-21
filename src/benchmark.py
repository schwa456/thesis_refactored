import subprocess
import pandas as pd
import os
from utils.logger import get_logger

logger = get_logger(__name__)

def run_benchmarks():
    experiments = [
        "baselines/preliminary_vector_only.yaml",
        "baselines/preliminary_graph_expansion.yaml",
        "baselines/preliminary_graph_and_agent.yaml",
        "baselines/baseline_xiyansql.yaml",
        "baselines/baseline_g_retriever.yaml",
        "baselines/baseline_linkalign.yaml",
        "experiments/proposed_gat_multi_agent.yaml"
    ]

    logger.info(f"📚 Starting total {len(experiments)} experiments...")

    for cfg in experiments:
        cfg_path = f"configs/{cfg}"
        if not os.path.exists(cfg_path):
            logger.error(f"[Error] Config Missing: {cfg_path}")
            continue
        
        logger.info(f"Running: {cfg}")

        try:
            subprocess.run(["python", "src/main.py", "--config", cfg], check=True)
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