"""B5 t-SNE용: gold column이 충분한 query 몇 개 찾기."""
import yaml
from pathlib import Path
from torch_geometric.loader import DataLoader
from analysis.gat_bottleneck_analysis_v2 import _build_dataset

CFG = "configs/experiments/s06_gat_bottleneck_fix/a01_additive_ablation/s06_a01_06_b5_dual_stream.yaml"

with open(CFG) as f:
    cfg = yaml.safe_load(f)
dataset = _build_dataset(cfg)

candidates = []
for idx in range(min(200, len(dataset))):
    b = dataset[idx]
    y = b["column"].y
    n_col = len(y)
    n_gold = int(y.sum().item())
    if n_col >= 15 and 2 <= n_gold <= min(8, n_col // 2):
        candidates.append((idx, n_col, n_gold))
    if len(candidates) >= 20:
        break

print("idx  n_col  n_gold")
for c in candidates[:15]:
    print(f"{c[0]:4d}  {c[1]:4d}   {c[2]:3d}")
