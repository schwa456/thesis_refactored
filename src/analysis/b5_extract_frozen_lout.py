"""B5 frozen GAT forward → (column L_out, y, query_idx) 캐시 저장.

Retrain 실험용: head만 재학습할 수 있도록 L_out을 미리 한 번만 계산.
split ∈ {train, dev}: 각각 train.json / dev.json 기반.
"""
import argparse
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

from data.bird_dataset import BIRDGraphDataset, BIRDSuperNodeDataset
from modules.builders.graph_builder import HeteroGraphBuilder, EnrichedHeteroGraphBuilder
from modules.encoders.local_encoder import LocalPLMEncoder
from analysis.gat_bottleneck_analysis_v2 import (
    _build_model_v2,
    extract_layerwise_embeddings_v2,
)


def build_split_dataset(cfg: dict, split: str):
    builder_type = cfg.get("builder", {}).get("type", "HeteroGraphBuilder")
    if builder_type == "EnrichedHeteroGraphBuilder":
        tables_json = cfg["builder"].get("tables_json_path",
                                          "data/raw/BIRD_dev/dev_tables.json")
        builder = EnrichedHeteroGraphBuilder(tables_json_path=tables_json)
    else:
        builder = HeteroGraphBuilder()
    encoder = LocalPLMEncoder()
    if split == "train":
        json_path = cfg["paths"]["train_json"]
        db_dir = cfg["paths"]["train_db_dir"]
    else:
        json_path = cfg["paths"]["test_json"]
        db_dir = cfg["paths"]["test_db_dir"]
    dataset = BIRDGraphDataset(json_path=json_path, db_dir=db_dir,
                                builder=builder, encoder=encoder)
    if cfg["model"].get("query_supernode", False):
        dataset = BIRDSuperNodeDataset(dataset)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_queries", type=int, default=None)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract] device={device}, split={args.split}")

    dataset = build_split_dataset(cfg, args.split)
    model = _build_model_v2(cfg, Path(args.checkpoint), device)
    model.eval()

    n = len(dataset) if args.max_queries is None else min(args.max_queries, len(dataset))
    print(f"[extract] processing {n}/{len(dataset)} queries...")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    records = []
    skipped = 0
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= n:
                break
            batch = batch.to(device)
            q_emb = batch["query"] if "query" in batch else None
            # dual_stream validate 흐름과 동일: 2D → 1D (per-batch mean)
            if q_emb is not None and q_emb.dim() == 2:
                q_emb_in = q_emb.mean(dim=0)
            else:
                q_emb_in = q_emb
            try:
                layer_embs = extract_layerwise_embeddings_v2(model, batch, q_emb_in)
            except Exception as e:
                skipped += 1
                continue
            if "column" not in layer_embs[-1]:
                skipped += 1
                continue
            lout_col = layer_embs[-1]["column"].detach().cpu()
            y = batch["column"].y.detach().cpu().float()
            if y.size(0) != lout_col.size(0):
                skipped += 1
                continue
            records.append({
                "query_idx": idx,
                "l_out": lout_col,
                "y": y,
            })
            if (idx + 1) % 500 == 0:
                print(f"  {idx+1}/{n}  (saved={len(records)}, skipped={skipped})")

    package = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "split": args.split,
        "records": records,
        "n_queries": len(records),
        "n_skipped": skipped,
        "embedding_dim": int(records[0]["l_out"].size(1)) if records else 0,
    }
    torch.save(package, out_path)
    total_nodes = sum(r["l_out"].size(0) for r in records)
    total_gold = sum(int(r["y"].sum().item()) for r in records)
    print(f"[extract] saved → {out_path}")
    print(f"  records={len(records)}, skipped={skipped}")
    print(f"  total column nodes={total_nodes}, gold={total_gold} "
          f"({100*total_gold/max(1,total_nodes):.2f}%)")


if __name__ == "__main__":
    main()
