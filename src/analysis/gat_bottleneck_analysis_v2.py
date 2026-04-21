"""
GAT Bottleneck Analysis v2 — s06 SchemaHeteroGATv2 adapted (B0~B5 batch).

v1 대비 차이점:
  - SchemaHeteroGATv2 지원 (PairNorm / Initial Residual / JK / Dual-Stream)
  - EPOCH_PATTERN: Main:/AC: 필드 포함
  - 2-layer (B5) / 3-layer (B0~B4) 동적 layer_names
  - run_batch: 6개 (config, checkpoint) 쌍을 한번에 처리
"""
import os
import re
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

from data.bird_dataset import BIRDGraphDataset, BIRDSuperNodeDataset
from modules.builders.graph_builder import HeteroGraphBuilder, EnrichedHeteroGraphBuilder
from modules.encoders.local_encoder import LocalPLMEncoder
from models.gat_network_v2 import SchemaHeteroGATv2
from utils.logger import get_logger

# v1 재사용
from analysis.gat_bottleneck_analysis import (
    plot_multi_model_overlay,
    plot_oversmoothing_trajectory,
    plot_tsne_per_layer,
    intra_table_sims,
    plot_attention_entropy,
    plot_gradient_flow,
    COL_TO_TAB_EDGE,
)

logger = get_logger(__name__)


EPOCH_PATTERN_V2 = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s*\|\s*"
    r"Loss:\s*(?P<loss>[\d.]+)\s*\|\s*"
    r"(?:BCE:\s*(?P<bce>[\d.]+)\s*\|\s*)?"
    r"(?:InfoNCE:\s*(?P<infonce>[\d.]+)\s*\|\s*)?"
    r"(?:Main:\s*(?P<main>[\d.]+)\s*\|\s*)?"
    r"(?:AC:\s*(?P<ac>[\d.]+)\s*\|\s*)?"
    r"Val Recall@15:\s*(?P<recall>[\d.]+)"
)


def parse_training_log_v2(log_path: Path) -> dict:
    records = {"epoch": [], "loss": [], "bce": [], "infonce": [],
               "main": [], "ac": [], "recall": []}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = EPOCH_PATTERN_V2.search(line)
            if not m:
                continue
            records["epoch"].append(int(m.group("epoch")))
            records["loss"].append(float(m.group("loss")))
            records["bce"].append(float(m.group("bce")) if m.group("bce") else np.nan)
            records["infonce"].append(float(m.group("infonce")) if m.group("infonce") else np.nan)
            records["main"].append(float(m.group("main")) if m.group("main") else np.nan)
            records["ac"].append(float(m.group("ac")) if m.group("ac") else np.nan)
            records["recall"].append(float(m.group("recall")))
    return {k: np.array(v) for k, v in records.items()}


def plot_objective_mismatch_v2(records: dict, model_name: str, output_dir: Path) -> Path:
    epochs = records["epoch"]
    main = records["main"]
    ac = records["ac"]
    loss = records["loss"]
    recall = records["recall"]
    has_main = not np.isnan(main).all()
    has_ac = not np.isnan(ac).all() and np.nansum(ac) > 1e-9

    fig, ax1 = plt.subplots(figsize=(10, 5))
    y_primary = main if has_main else loss
    label_primary = "Main Loss" if has_main else "Total Loss"
    ax1.plot(epochs, y_primary, color="tab:red", marker="o", markersize=3, label=label_primary)
    if has_ac:
        ax1.plot(epochs, ac, color="tab:orange", marker="^", markersize=3, alpha=0.7, label="Anti-Collapse")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_xlabel("Epoch")
    ax1.legend(loc="upper left", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(epochs, recall, color="tab:blue", marker="s", markersize=3, label="Val Recall@15")
    ax2.set_ylabel("Val Recall@15", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title(f"Loss vs Recall — {model_name}")
    fig.tight_layout()
    plot_path = output_dir / f"objective_mismatch_{model_name}.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def summarize_step1_v2(records: dict) -> dict:
    recall = records["recall"]
    loss = records["loss"]
    main = records["main"]
    ac = records["ac"]
    best_idx = int(np.argmax(recall))
    return {
        "num_epochs": int(len(records["epoch"])),
        "best_recall": float(recall[best_idx]),
        "best_recall_epoch": int(records["epoch"][best_idx]),
        "final_recall": float(recall[-1]),
        "final_loss": float(loss[-1]),
        "min_loss": float(np.nanmin(loss)),
        "min_loss_epoch": int(records["epoch"][int(np.nanargmin(loss))]),
        "final_main": float(main[-1]) if not np.isnan(main).all() else None,
        "final_ac": float(ac[-1]) if not np.isnan(ac).all() else None,
    }


def run_step1_batch(log_paths: Dict[str, Path], output_dir: Path) -> Dict[str, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Step 1: Objective trajectory (B0-B5)")
    logger.info("=" * 60)

    all_records = {}
    summaries = {}
    for name, log_path in log_paths.items():
        if not log_path.exists():
            logger.warning(f"Log not found: {log_path}")
            continue
        rec = parse_training_log_v2(log_path)
        if len(rec["epoch"]) == 0:
            logger.warning(f"No epoch lines: {log_path}")
            continue
        plot_objective_mismatch_v2(rec, name, output_dir)
        summary = summarize_step1_v2(rec)
        all_records[name] = rec
        summaries[name] = summary
        logger.info(f"[{name}] best Recall@15={summary['best_recall']:.4f} @ep{summary['best_recall_epoch']}, "
                    f"epochs={summary['num_epochs']}, final_loss={summary['final_loss']:.4f}")

    if len(all_records) >= 2:
        overlay_path = plot_multi_model_overlay(all_records, output_dir)
        logger.info(f"Overlay plot → {overlay_path}")

    logger.info("Step 1 done.")
    return summaries


def _build_dataset(cfg: dict) -> BIRDGraphDataset:
    builder_type = cfg.get("builder", {}).get("type", "HeteroGraphBuilder")
    if builder_type == "EnrichedHeteroGraphBuilder":
        dev_tables = "data/raw/BIRD_dev/dev_tables.json"
        builder = EnrichedHeteroGraphBuilder(tables_json_path=dev_tables)
    else:
        builder = HeteroGraphBuilder()
    encoder = LocalPLMEncoder()
    dev_json = cfg["paths"].get("test_json", "data/raw/BIRD_dev/dev.json")
    dev_db_dir = cfg["paths"].get("test_db_dir", "data/raw/BIRD_dev/dev_databases")
    dataset = BIRDGraphDataset(json_path=dev_json, db_dir=dev_db_dir,
                                builder=builder, encoder=encoder)
    if cfg["model"].get("query_supernode", False):
        dataset = BIRDSuperNodeDataset(dataset)
    return dataset


def _build_model_v2(cfg: dict, checkpoint_path: Path, device: torch.device) -> SchemaHeteroGATv2:
    m = cfg["model"]
    model = SchemaHeteroGATv2(
        in_channels=m["in_channels"],
        hidden_channels=m["hidden_channels"],
        out_channels=m["out_channels"],
        num_layers=m["num_layers"],
        heads=m["heads"],
        query_conditioned=m.get("query_conditioned", False),
        query_supernode=m.get("query_supernode", False),
        pairnorm_mode=m.get("pairnorm_mode", "none"),
        pairnorm_scale=m.get("pairnorm_scale", 1.0),
        initial_residual_alpha=m.get("initial_residual_alpha", 0.0),
        jumping_knowledge=m.get("jumping_knowledge", "none"),
        dual_stream=m.get("dual_stream", False),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("gat_state_dict", "model_state_dict", "state_dict"):
            if key in ckpt:
                state = ckpt[key]
                break
        else:
            state = ckpt
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded: {checkpoint_path.name} (L={m['num_layers']}, "
                f"PN={m.get('pairnorm_mode','none')}, α={m.get('initial_residual_alpha',0)}, "
                f"JK={m.get('jumping_knowledge','none')}, DS={m.get('dual_stream',False)})")
    return model


def extract_layerwise_embeddings_v2(model: SchemaHeteroGATv2, batch,
                                     query_emb: Optional[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    """v2 forward 재생: L0_PLM, L1~LN GAT 출력(after PairNorm+IniRes), L_final."""
    x_dict = {nt: x for nt, x in batch.x_dict.items()}
    edge_index_dict = batch.edge_index_dict

    embeddings = [{nt: x.detach().clone() for nt, x in x_dict.items()}]

    if model.query_conditioned and query_emb is not None:
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        x_dict = {nt: torch.cat([x, query_emb.expand(x.size(0), -1)], dim=-1)
                  for nt, x in x_dict.items()}

    h0_dict = {nt: F.leaky_relu(model.lin_dict[nt](x)) for nt, x in x_dict.items()}
    out_dict = h0_dict

    layer_outputs = []
    for i in range(model.num_layers):
        conv_out = model.convs[i](out_dict, edge_index_dict)
        conv_out = {nt: F.elu(x) for nt, x in conv_out.items()}
        if model.res_proj is not None:
            for nt in model.node_types:
                if nt not in conv_out and nt in h0_dict:
                    conv_out[nt] = model.res_proj[nt](h0_dict[nt])
            mixed = {}
            for nt, x in conv_out.items():
                h0_proj = model.res_proj[nt](h0_dict[nt])
                mixed[nt] = (1.0 - model.initial_residual_alpha) * x + \
                            model.initial_residual_alpha * h0_proj
            conv_out = mixed
        if model.pairnorms is not None:
            conv_out = {nt: model.pairnorms[i][nt](x) for nt, x in conv_out.items()}
        out_dict = conv_out
        layer_outputs.append(out_dict)
        embeddings.append({nt: x.detach().clone() for nt, x in out_dict.items()})

    # 최종 출력
    if model.jumping_knowledge == "concat":
        final = {}
        for nt in model.node_types:
            if nt not in h0_dict:
                continue
            parts = [h0_dict[nt]]
            for l in range(model.num_layers):
                if nt in layer_outputs[l]:
                    parts.append(layer_outputs[l][nt])
            concat = torch.cat(parts, dim=-1)
            final[nt] = model.jk_lin[nt](concat) + model.skip_dict[nt](x_dict[nt])
    elif model.jumping_knowledge == "max":
        final = {}
        for nt in model.node_types:
            layers_nt = [layer_outputs[l][nt] for l in range(model.num_layers)
                         if nt in layer_outputs[l]]
            if not layers_nt:
                continue
            pooled = torch.stack(layers_nt, dim=0).max(dim=0).values
            final[nt] = model.jk_lin[nt](pooled) + model.skip_dict[nt](x_dict[nt])
    else:
        final = {}
        for nt, x in out_dict.items():
            if nt not in x_dict:
                continue
            final[nt] = model.out_lin_dict[nt](x) + model.skip_dict[nt](x_dict[nt])

    # Dual-stream fusion
    if model.dual_stream and query_emb is not None:
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        z_q = model.query_encoder(query_emb)
        fused = {}
        for nt in model.node_types:
            if nt not in final:
                continue
            h = final[nt]
            z_q_exp = z_q.expand(h.size(0), -1)
            concat = torch.cat([h, z_q_exp, h * z_q_exp], dim=-1)
            fused[nt] = model.fusion_head[nt](concat)
        final = fused

    embeddings.append({nt: x.detach().clone() for nt, x in final.items()})
    return embeddings


def _layer_names(num_layers: int) -> List[str]:
    names = ["L0_PLM"]
    for i in range(num_layers):
        names.append(f"L{i+1}_GAT")
    names.append("L_out")
    return names


def run_step2_single(name: str, config_path: Path, checkpoint_path: Path,
                     output_dir: Path, max_queries: Optional[int],
                     tsne_sample_idx: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    logger.info("-" * 60)
    logger.info(f"Step 2 [{name}]")
    logger.info("-" * 60)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = _build_dataset(cfg)
    model = _build_model_v2(cfg, checkpoint_path, device)

    n = len(dataset) if max_queries is None else min(max_queries, len(dataset))
    num_layers = model.num_layers
    layer_names = _layer_names(num_layers)
    all_sims = [[] for _ in layer_names]

    sample_embeddings = None
    sample_col_to_table = None

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= n:
                break
            batch = batch.to(device)
            q_emb = batch["query"] if "query" in batch else None
            layer_embs = extract_layerwise_embeddings_v2(model, batch, q_emb)
            if COL_TO_TAB_EDGE not in batch.edge_index_dict:
                continue
            cb_edge = batch.edge_index_dict[COL_TO_TAB_EDGE]
            for l, ed in enumerate(layer_embs):
                if "column" in ed:
                    all_sims[l].extend(intra_table_sims(ed["column"], cb_edge))
            if idx == tsne_sample_idx:
                sample_embeddings = [d["column"] for d in layer_embs if "column" in d]
                col_ids = cb_edge[0].tolist()
                tab_ids = cb_edge[1].tolist()
                sample_col_to_table = {c: t for c, t in zip(col_ids, tab_ids)}
            if (idx + 1) % 200 == 0:
                logger.info(f"  [{name}] {idx+1}/{n}")

    # per-model plot
    out_sub = output_dir / name
    out_sub.mkdir(parents=True, exist_ok=True)
    plot_oversmoothing_trajectory(all_sims, layer_names, out_sub)
    if sample_embeddings and sample_col_to_table:
        try:
            plot_tsne_per_layer(sample_embeddings, sample_col_to_table, layer_names,
                                out_sub, tsne_sample_idx)
        except Exception as e:
            logger.warning(f"t-SNE fail [{name}]: {e}")

    stats = {layer_names[l]: {"mean": float(np.mean(s)) if s else float("nan"),
                               "std": float(np.std(s)) if s else float("nan"),
                               "median": float(np.median(s)) if s else float("nan"),
                               "n": len(s)}
             for l, s in enumerate(all_sims)}
    logger.info(f"[{name}] over-smoothing: "
                f"L0={stats['L0_PLM']['mean']:.4f} → "
                f"L_out={stats['L_out']['mean']:.4f}")
    return {"stats": stats, "all_sims": all_sims, "layer_names": layer_names}


def compute_gradient_flow_v2(model: SchemaHeteroGATv2, batch, query_emb) -> Dict[str, float]:
    """v2 forward로 proxy BCE loss → 그룹별 gradient L2 norm."""
    model.train()
    model.zero_grad(set_to_none=True)

    out_dict = model(batch.x_dict, batch.edge_index_dict, query_emb)
    if query_emb is None:
        return {}
    q = query_emb if query_emb.dim() == 2 else query_emb.unsqueeze(0)

    total_loss = torch.tensor(0.0)
    n_terms = 0
    for nt in ("table", "column"):
        if nt not in out_dict:
            continue
        node_emb = out_dict[nt]
        if node_emb.size(0) == 0:
            continue
        if not hasattr(batch[nt], "y") or batch[nt].y is None:
            continue
        y = batch[nt].y.float()
        if y.size(0) != node_emb.size(0):
            continue
        q_proj = q[:, : node_emb.size(1)] if q.size(1) >= node_emb.size(1) else \
                 F.pad(q, (0, node_emb.size(1) - q.size(1)))
        logits = F.normalize(node_emb, dim=-1) @ F.normalize(q_proj, dim=-1).T
        logits = logits.squeeze(-1) * 10.0
        total_loss = total_loss + F.binary_cross_entropy_with_logits(logits, y)
        n_terms += 1
    if n_terms == 0:
        return {}
    total_loss = total_loss / n_terms
    total_loss.backward()

    def _gnorm(named_params) -> float:
        sq = 0.0
        for _, p in named_params:
            if p.grad is not None:
                sq += float(p.grad.detach().pow(2).sum().item())
        return sq ** 0.5

    groups: Dict[str, float] = {}
    groups["lin_dict"] = _gnorm(model.lin_dict.named_parameters())
    for i in range(model.num_layers):
        groups[f"conv_L{i+1}"] = _gnorm(model.convs[i].named_parameters())
    if model.out_lin_dict is not None:
        groups["out_lin_dict"] = _gnorm(model.out_lin_dict.named_parameters())
    if model.jk_lin is not None:
        groups["jk_lin"] = _gnorm(model.jk_lin.named_parameters())
    if model.res_proj is not None:
        groups["res_proj"] = _gnorm(model.res_proj.named_parameters())
    groups["skip_dict"] = _gnorm(model.skip_dict.named_parameters())
    if model.query_encoder is not None:
        groups["query_encoder"] = _gnorm(model.query_encoder.named_parameters())
    if model.fusion_head is not None:
        groups["fusion_head"] = _gnorm(model.fusion_head.named_parameters())

    model.eval()
    return groups


def extract_layerwise_attention_v2(model: SchemaHeteroGATv2, batch, query_emb) -> List[Dict[tuple, float]]:
    """v2 forward 재생 + 각 GAT layer에서 edge-type별 attention entropy."""
    x_dict = {nt: x for nt, x in batch.x_dict.items()}
    edge_index_dict = batch.edge_index_dict

    if model.query_conditioned and query_emb is not None:
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        x_dict = {nt: torch.cat([x, query_emb.expand(x.size(0), -1)], dim=-1)
                  for nt, x in x_dict.items()}

    h0_dict = {nt: F.leaky_relu(model.lin_dict[nt](x)) for nt, x in x_dict.items()}
    out_dict = h0_dict

    per_layer_entropy: List[Dict[tuple, float]] = []
    for i in range(model.num_layers):
        hetero_conv = model.convs[i]
        inner_convs = hetero_conv.convs
        layer_entropy: Dict[tuple, float] = {}

        next_parts = defaultdict(list)
        for edge_type in edge_index_dict.keys():
            src, _, dst = edge_type
            if src not in out_dict or dst not in out_dict:
                continue
            if edge_type not in inner_convs:
                continue
            ei = edge_index_dict[edge_type]
            if ei.numel() == 0:
                continue
            conv = inner_convs[edge_type]
            out, (att_ei, alpha) = conv(
                (out_dict[src], out_dict[dst]), ei, return_attention_weights=True
            )
            next_parts[dst].append(out)
            alpha_mean = alpha.mean(dim=-1) if alpha.dim() > 1 else alpha
            dst_idx = att_ei[1]
            ent_sum = torch.zeros(out_dict[dst].size(0))
            for e in range(alpha_mean.size(0)):
                p = alpha_mean[e].item()
                if p > 1e-12:
                    ent_sum[dst_idx[e]] -= p * np.log(p + 1e-12)
            has_in = torch.zeros(out_dict[dst].size(0), dtype=torch.bool)
            has_in[dst_idx] = True
            if has_in.any():
                layer_entropy[edge_type] = ent_sum[has_in].mean().item()

        # merge + fill + IniRes + PairNorm (다음 layer용)
        merged: Dict[str, torch.Tensor] = {}
        for dst, parts in next_parts.items():
            merged[dst] = torch.stack(parts, dim=0).mean(dim=0)
        new_out = {nt: F.elu(merged[nt]) for nt in merged}
        if model.res_proj is not None:
            for nt in model.node_types:
                if nt not in new_out and nt in h0_dict:
                    new_out[nt] = model.res_proj[nt](h0_dict[nt])
            mixed = {}
            for nt, x in new_out.items():
                h0_proj = model.res_proj[nt](h0_dict[nt])
                mixed[nt] = (1.0 - model.initial_residual_alpha) * x + \
                            model.initial_residual_alpha * h0_proj
            new_out = mixed
        if model.pairnorms is not None:
            new_out = {nt: model.pairnorms[i][nt](x) for nt, x in new_out.items()}
        out_dict = new_out

        per_layer_entropy.append(layer_entropy)

    return per_layer_entropy


def run_step3_single(name: str, config_path: Path, checkpoint_path: Path,
                     output_dir: Path, max_queries: Optional[int]) -> dict:
    device = torch.device("cpu")
    logger.info("-" * 60)
    logger.info(f"Step 3 [{name}]")
    logger.info("-" * 60)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    dataset = _build_dataset(cfg)
    model = _build_model_v2(cfg, checkpoint_path, device)

    n = len(dataset) if max_queries is None else min(max_queries, len(dataset))
    num_layers = model.num_layers
    entropy_accum: List[Dict[tuple, List[float]]] = [defaultdict(list) for _ in range(num_layers)]
    grad_accum: Dict[str, List[float]] = defaultdict(list)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for idx, batch in enumerate(loader):
        if idx >= n:
            break
        batch = batch.to(device)
        q_emb = batch["query"] if "query" in batch else None
        try:
            with torch.no_grad():
                per_layer = extract_layerwise_attention_v2(model, batch, q_emb)
            for l, ld in enumerate(per_layer):
                for et, ent in ld.items():
                    entropy_accum[l][et].append(ent)
        except Exception as e:
            logger.warning(f"  [att:{name}] q{idx} skip: {e}")
        try:
            grads = compute_gradient_flow_v2(model, batch, q_emb)
            for k, v in grads.items():
                grad_accum[k].append(v)
        except Exception as e:
            logger.warning(f"  [grad:{name}] q{idx} skip: {e}")
        if (idx + 1) % 200 == 0:
            logger.info(f"  [{name}] {idx+1}/{n}")

    out_sub = output_dir / name
    out_sub.mkdir(parents=True, exist_ok=True)
    plot_attention_entropy(entropy_accum, out_sub)
    if grad_accum:
        plot_gradient_flow(dict(grad_accum), out_sub)

    # summary
    grad_summary = {k: float(np.mean(v)) for k, v in grad_accum.items() if v}
    conv_keys = sorted([k for k in grad_summary if k.startswith("conv_L")])
    grad_ratio = None
    if len(conv_keys) >= 2:
        first = grad_summary[conv_keys[0]]
        last = grad_summary[conv_keys[-1]]
        grad_ratio = last / first if first > 0 else float("nan")
        logger.info(f"[{name}] grad ratio (last/first conv) = {grad_ratio:.4f}")

    entropy_summary = {}
    for l, ld in enumerate(entropy_accum):
        entropy_summary[f"L{l+1}"] = {
            "→".join(et): float(np.mean(v)) if v else float("nan")
            for et, v in ld.items()
        }
    return {"grad_mean": grad_summary, "grad_ratio": grad_ratio,
            "entropy": entropy_summary, "num_layers": num_layers}


def plot_cross_model_oversmoothing(step2_results: Dict[str, dict], output_dir: Path) -> Path:
    """각 모델의 L0/deepest_GAT/L_out 세 지점을 묶어 비교."""
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(step2_results.keys())
    x = np.arange(len(names))
    l0 = [step2_results[n]["stats"]["L0_PLM"]["mean"] for n in names]
    lfinal_gat = []
    lout = []
    for n in names:
        ln = step2_results[n]["layer_names"]
        gat_layers = [k for k in ln if k.startswith("L") and "GAT" in k]
        lfinal_gat.append(step2_results[n]["stats"][gat_layers[-1]]["mean"])
        lout.append(step2_results[n]["stats"]["L_out"]["mean"])
    width = 0.27
    ax.bar(x - width, l0, width, label="L0_PLM", color="tab:gray")
    ax.bar(x, lfinal_gat, width, label="deepest_GAT", color="tab:purple")
    ax.bar(x + width, lout, width, label="L_out", color="tab:green")
    ax.axhline(0.85, color="red", linestyle="--", alpha=0.5, label="critical 0.85")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("Intra-Table Cosine (mean)")
    ax.set_title("Over-smoothing at Key Depths (B0-B5)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    p = output_dir / "cross_model_oversmoothing.png"
    plt.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_cross_model_grad_ratio(step3_results: Dict[str, dict], output_dir: Path) -> Path:
    names = list(step3_results.keys())
    ratios = [step3_results[n]["grad_ratio"] if step3_results[n]["grad_ratio"] is not None
              else float("nan") for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(names, ratios, color="tab:orange", alpha=0.85)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="balanced")
    ax.axhline(0.1, color="red", linestyle="--", alpha=0.5, label="vanish threshold")
    ax.set_ylabel("grad(last conv) / grad(first conv)")
    ax.set_title("Gradient Flow Ratio (B0-B5)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    p = output_dir / "cross_model_grad_ratio.png"
    plt.savefig(p, dpi=120)
    plt.close(fig)
    return p


def run_batch(configs: Dict[str, Path], checkpoints: Dict[str, Path],
              logs: Dict[str, Path], output_dir: Path,
              max_queries: Optional[int], tsne_sample_idx: int,
              skip_step1: bool, skip_step2: bool, skip_step3: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, dict] = {}

    if not skip_step1:
        results["step1"] = run_step1_batch(logs, output_dir)

    if not skip_step2:
        step2_all = {}
        for name in configs:
            step2_all[name] = run_step2_single(name, configs[name], checkpoints[name],
                                                output_dir, max_queries, tsne_sample_idx)
        plot_cross_model_oversmoothing(step2_all, output_dir)
        results["step2"] = {n: {"stats": r["stats"], "layer_names": r["layer_names"]}
                             for n, r in step2_all.items()}

    if not skip_step3:
        step3_all = {}
        for name in configs:
            step3_all[name] = run_step3_single(name, configs[name], checkpoints[name],
                                                output_dir, max_queries)
        plot_cross_model_grad_ratio(step3_all, output_dir)
        results["step3"] = step3_all

    import json
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Batch summary → {summary_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="s06 GAT Bottleneck Analysis (B0-B5)")
    parser.add_argument("--configs_dir", type=str,
                        default="configs/experiments/s06_gat_bottleneck_fix/a01_additive_ablation")
    parser.add_argument("--ckpt_dir", type=str,
                        default="/SSL_NAS/peoples/khj/thesis/checkpoints/s06_gat_bottleneck_fix")
    parser.add_argument("--logs_dir", type=str, default="outputs/logs/s06")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis/s06_bottleneck")
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--tsne_sample_idx", type=int, default=0)
    parser.add_argument("--skip_step1", action="store_true")
    parser.add_argument("--skip_step2", action="store_true")
    parser.add_argument("--skip_step3", action="store_true")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of model IDs to process (e.g. B5E). Default: all.")
    args = parser.parse_args()

    cd = Path(args.configs_dir)
    kd = Path(args.ckpt_dir)
    ld = Path(args.logs_dir)
    od = Path(args.output_dir)

    pairs = [
        ("B0",  "s06_a01_01_b0_baseline.yaml",             "best_gat_s06_a01_01_b0.pt",         "s06_a01_01_b0_baseline.log"),
        ("B1",  "s06_a01_02_b1_pairnorm.yaml",             "best_gat_s06_a01_02_b1.pt",         "s06_a01_02_b1_pairnorm.log"),
        ("B2",  "s06_a01_03_b2_initial_residual.yaml",     "best_gat_s06_a01_03_b2.pt",         "s06_a01_03_b2_initial_residual.log"),
        ("B3",  "s06_a01_04_b3_listnet.yaml",              "best_gat_s06_a01_04_b3.pt",         "s06_a01_04_b3_listnet.log"),
        ("B4",  "s06_a01_05_b4_anti_collapse.yaml",        "best_gat_s06_a01_05_b4.pt",         "s06_a01_05_b4_anti_collapse.log"),
        ("B5",  "s06_a01_06_b5_dual_stream.yaml",          "best_gat_s06_a01_06_b5.pt",         "s06_a01_06_b5_dual_stream.log"),
        ("B5E", "s06_a01_07_b5_enriched_dual_stream.yaml", "best_gat_s06_a01_07_b5_enriched.pt","s06_a01_07_b5_enriched.log"),
    ]
    if args.models:
        allowed = set(args.models)
        pairs = [p for p in pairs if p[0] in allowed]
        if not pairs:
            raise SystemExit(f"No pairs match --models {args.models}")
    configs = {n: cd / c for n, c, _, _ in pairs}
    checkpoints = {n: kd / c for n, _, c, _ in pairs}
    logs = {n: ld / l for n, _, _, l in pairs}

    run_batch(configs, checkpoints, logs, od,
              args.max_queries, args.tsne_sample_idx,
              args.skip_step1, args.skip_step2, args.skip_step3)


if __name__ == "__main__":
    main()
