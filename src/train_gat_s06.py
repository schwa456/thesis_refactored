"""
s06 ablation — GAT Bottleneck Fix 전용 학습 스크립트.

기존 train_gat_direct.py 에서 확장된 요소:
  - SchemaHeteroGATv2 사용 (PairNorm / Initial residual / JK / Dual-stream)
  - ListNet loss 옵션
  - AntiCollapseRegularizer 옵션 (column 임베딩에 대해 intra-table cosine 제약)

config yaml 의 s06 전용 키 (model):
  pairnorm_mode: 'none' | 'pairnorm'
  pairnorm_scale: float (default 1.0)
  initial_residual_alpha: float (default 0.0)
  jumping_knowledge: 'none' | 'concat' | 'max'
  dual_stream: bool

config yaml 의 s06 전용 키 (training):
  loss_type: 'bce' | 'listnet' | 'bce_listnet'
  anti_collapse_weight: float (default 0.0)
  anti_collapse_tau_max: float (default 0.85)

  # Phase 3 #3 (Direct AC on GAT output, 2026-05-06):
  #   AC loss 가 model output (skip + fusion 후) 에 적용되면 skip path 가 우회 가능.
  #   target='gat_out_L_last' 시 forward hook 으로 마지막 conv layer column output 을
  #   capture → AC loss 그 위에 적용. main GAT path gradient 회복 의도.
  anti_collapse_target: 'fusion' | 'gat_out_L_last'   # default 'fusion' (Phase 2 동치)

  # Phase 3 #4 (Layer-wise LR, 2026-05-06):
  #   GAT path (HeteroConv) 의 LR 을 base_lr × gat_lr_multiplier 로 별도 설정. main GAT path
  #   gradient 가 1/10 로 축소된 mechanism null finding 의 직접 대응.
  optimizer_layer_wise_lr: bool (default false)
  gat_lr_multiplier: float (default 1.0)
"""
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass
os.environ.setdefault("WANDB_DIR", str(Path(__file__).resolve().parents[1]))
import yaml
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
import wandb
import argparse

from data.bird_dataset import BIRDGraphDataset, BIRDSuperNodeDataset
from modules.builders.graph_builder import HeteroGraphBuilder, EnrichedHeteroGraphBuilder
from modules.encoders.local_encoder import LocalPLMEncoder
from models.gat_network_v2 import SchemaHeteroGATv2
from models.direct_classifier import DirectClassifierHead
from models.losses import ListNetLoss, AntiCollapseRegularizer, combined_loss
from utils.logger import setup_logger, get_logger


PATHS = {
    "checkpoint_dir": "./outputs/checkpoints",
    "cache_dir": "./data/processed",
}

COL_TO_TAB_EDGE = ("column", "belongs_to", "table")


def calculate_recall_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int = 15) -> float:
    if labels.sum() == 0:
        return 0.0
    k_actual = min(k, logits.size(0))
    _, top_k_indices = torch.topk(logits, k_actual)
    hits = labels[top_k_indices].sum().item()
    return hits / labels.sum().item()


def compute_gold_calibration_metrics(logits: torch.Tensor, labels: torch.Tensor,
                                     theta: float = 0.1):
    """MA-0/MA-1 (DECISIONS 2026-06-07 #4) — gold 노드의 **절대 score calibration** 지표.

    inference extractor 는 score≥θ (절대 cutoff) 기반이므로 rank (R@15) 가 아니라 calibration 이
    inference recall 을 결정 (Spearman: gold recall@θ +0.9273 / gold p50 +0.9545 vs R@15 −0.19).

    score = sigmoid(logit) (pipeline 정합). 반환:
      - gold_recall_at_theta: gold 노드 중 score≥θ 비율 (gold 없으면 None → 집계 제외)
      - gold_p50: gold 노드 score 중앙값 (None if no gold)
    """
    gold = labels == 1
    if gold.sum() == 0:
        return None, None
    scores = torch.sigmoid(logits[gold].detach())
    gold_recall = (scores >= theta).float().mean().item()
    gold_p50 = scores.median().item()
    return gold_recall, gold_p50


def gold_margin_loss(logits: torch.Tensor, labels: torch.Tensor,
                     theta_target: float = 0.15) -> torch.Tensor:
    """MA-2 (DECISIONS 2026-06-07 #4) — gold score margin loss.

    gold 노드 score(=sigmoid(logit)) 가 θ_target 이상이도록 강제 (phase1-류 압축으로 gold 가
    θ 미통과하는 calibration deficit 해소). L = mean(max(0, θ_target − sigmoid(gold_logit))).
    gold 없으면 0 (grad 0). theta_target 은 보통 θ(0.1) + ε(0.05) = 0.15.
    """
    gold = labels == 1
    if gold.sum() == 0:
        return logits.new_tensor(0.0)
    scores = torch.sigmoid(logits[gold])
    return torch.clamp(theta_target - scores, min=0.0).mean()


def per_table_normalize_logits(col_logits: torch.Tensor, belongs_to_edge: torch.Tensor,
                               num_cols: int) -> torch.Tensor:
    """MA-2 (DECISIONS 2026-06-07 #4) — per-table column score normalization.

    각 table 내 column logit 분포를 zero-mean / unit-std 로 normalize (PairNorm-류 압축 해소).
    belongs_to_edge: [2, E] (row0=column local idx, row1=table local idx) — batch 전역 인덱스라
    table id 가 (graph, table) 를 자연 인코딩. column 1개 이하 table 은 원본 유지.
    """
    col2tab = torch.full((num_cols,), -1, dtype=torch.long, device=col_logits.device)
    if belongs_to_edge.numel() > 0:
        col2tab[belongs_to_edge[0]] = belongs_to_edge[1]
    out = col_logits.clone()
    for t in torch.unique(col2tab):
        if int(t) < 0:
            continue
        m = col2tab == t
        if int(m.sum()) < 2:
            continue
        grp = col_logits[m]
        out = out.masked_scatter(m, (grp - grp.mean()) / (grp.std() + 1e-5))
    return out


def validate(gat_model, classifier_heads, loader, device, k=15,
             query_conditioned=False, query_supernode=False, dual_stream=False,
             theta=0.1):
    """MA-1 (DECISIONS 2026-06-07 #4): dict 반환 — {recall_at_15, gold_recall_at_theta, gold_p50}.

    best-epoch/early-stop 은 monitor_metric (default gold_recall_at_theta) 기준. recall_at_15 는
    보조 지표로 retain (inference recall 예측력 약함 ρ=−0.19).
    """
    gat_model.eval()
    for head in classifier_heads.values():
        head.eval()
    total_recall = 0.0
    count = 0
    # MA-1 calibration 지표 누적 (gold 있는 graph 만).
    sum_gold_recall = 0.0
    sum_gold_p50 = 0.0
    calib_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            q_emb = batch["query"]

            # query pooling (3D → 2D)
            if q_emb.dim() == 3:
                q_pooled = q_emb.mean(dim=1)
            elif q_emb.dim() == 2:
                q_pooled = q_emb
            else:
                q_pooled = q_emb.unsqueeze(0)

            # forward
            node_batch_dict = {
                nt: batch[nt].batch
                for nt in batch.node_types
                if hasattr(batch[nt], "batch")
            }
            if query_conditioned and not dual_stream:
                augmented_x = {}
                for n_type, x in batch.x_dict.items():
                    node_batch_idx = batch[n_type].batch
                    q_per_node = q_pooled[node_batch_idx]
                    augmented_x[n_type] = torch.cat([x, q_per_node], dim=-1)
                node_embs_dict = gat_model(augmented_x, batch.edge_index_dict)
            elif dual_stream:
                # dual_stream: query_emb [B, d_q] 를 node_batch_dict 로 per-graph fusion
                node_embs_dict = gat_model(
                    batch.x_dict, batch.edge_index_dict,
                    query_emb=q_pooled,
                    node_batch_dict=node_batch_dict,
                )
            else:
                node_embs_dict = gat_model(batch.x_dict, batch.edge_index_dict)

            for i in range(batch.num_graphs):
                logits_list, labels_list = [], []
                for n_type in ["table", "column"]:
                    if n_type not in classifier_heads:
                        continue
                    mask = (batch[n_type].batch == i)
                    if not mask.any():
                        continue
                    node_emb = node_embs_dict[n_type][mask]
                    score = classifier_heads[n_type](node_emb)
                    logits_list.append(score)
                    labels_list.append(batch[n_type].y[mask])

                if logits_list:
                    all_logits = torch.cat(logits_list)
                    all_labels = torch.cat(labels_list)
                    total_recall += calculate_recall_at_k(all_logits, all_labels, k=k)
                    count += 1
                    # MA-1: gold calibration 지표 (gold 있는 graph 만 집계).
                    gr, p50 = compute_gold_calibration_metrics(all_logits, all_labels, theta=theta)
                    if gr is not None:
                        sum_gold_recall += gr
                        sum_gold_p50 += p50
                        calib_count += 1
    return {
        "recall_at_15": total_recall / count if count > 0 else 0.0,
        "gold_recall_at_theta": sum_gold_recall / calib_count if calib_count > 0 else 0.0,
        "gold_p50": sum_gold_p50 / calib_count if calib_count > 0 else 0.0,
    }


def run_train(config_path: str, resume: bool = False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resume-from-checkpoint (2026-06-09, 디스크풀 crash multiseed E100 손실 재발 방지).
    # RESUME=1 환경변수 또는 --resume 플래그 → 시작 시 latest(우선)/best checkpoint 로드.
    resume_enabled = bool(resume) or os.getenv("RESUME", "0") == "1"

    cfg_paths = cfg.get("paths", {})
    if "checkpoint_dir" in cfg_paths:
        PATHS["checkpoint_dir"] = cfg_paths["checkpoint_dir"]
    if "cache_dir" in cfg_paths:
        PATHS["cache_dir"] = cfg_paths["cache_dir"]

    wandb.init(
        project=cfg.get("project_name", os.getenv("WANDB_PROJECT", "Text-to-SQL-Alignment")),
        name=cfg["experiment_name"],
        config=cfg,
    )

    setup_logger(log_dir="./logs/", exp_name=cfg["experiment_name"], sub_dir="train")
    logger = get_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(PATHS["checkpoint_dir"], exist_ok=True)
    os.makedirs(PATHS["cache_dir"], exist_ok=True)
    logger.info(f"[s06] checkpoint_dir={PATHS['checkpoint_dir']}")

    # Builder / Encoder
    builder_cfg = cfg.get("builder", {})
    builder_type = builder_cfg.get("type", "HeteroGraphBuilder")
    tables_json = builder_cfg.get("tables_json_path", "")
    if builder_type == "EnrichedHeteroGraphBuilder":
        builder = EnrichedHeteroGraphBuilder(tables_json_path=tables_json)
        logger.info(f"Using EnrichedHeteroGraphBuilder (tables_json={tables_json})")
    elif builder_type == "V6W3VirtualSummaryBuilder":
        # V6-W3 variant A — table_summary virtual node (DECISIONS 2026-06-06).
        from modules.builders.v6w3_builders import V6W3VirtualSummaryBuilder
        builder = V6W3VirtualSummaryBuilder(tables_json_path=tables_json)
        logger.info(f"Using V6W3VirtualSummaryBuilder (A, tables_json={tables_json})")
    elif builder_type == "V6W3ColumnPoolingBuilder":
        # V6-W3 variant B — table.x = column pooling override (구조 무변경).
        from modules.builders.v6w3_builders import V6W3ColumnPoolingBuilder
        builder = V6W3ColumnPoolingBuilder(
            tables_json_path=tables_json,
            pool_mode=builder_cfg.get("pool_mode", "uniform"),
        )
        logger.info(
            f"Using V6W3ColumnPoolingBuilder (B, pool_mode={builder_cfg.get('pool_mode', 'uniform')})")
    elif builder_type == "V6W3HubLocalVNBuilder":
        # V6-W3 variant C — hub-only Local VN_G (local_vn node).
        from modules.builders.v6w3_builders import V6W3HubLocalVNBuilder
        builder = V6W3HubLocalVNBuilder(
            tables_json_path=tables_json,
            hub_strategy=builder_cfg.get("hub_strategy", "median"),
            hub_min_columns=builder_cfg.get("hub_min_columns", 0),
        )
        logger.info(
            f"Using V6W3HubLocalVNBuilder (C, hub_strategy={builder_cfg.get('hub_strategy', 'median')}, "
            f"hub_min_columns={builder_cfg.get('hub_min_columns', 0)})")
    else:
        builder = HeteroGraphBuilder()
    encoder = LocalPLMEncoder()

    logger.info("Loading Training Dataset...")
    full_train_dataset = BIRDGraphDataset(
        json_path=cfg["paths"]["train_json"],
        db_dir=cfg["paths"]["train_db_dir"],
        builder=builder,
        encoder=encoder,
    )

    query_conditioned = cfg["model"].get("query_conditioned", False)
    query_supernode = cfg["model"].get("query_supernode", False)
    dual_stream = cfg["model"].get("dual_stream", False)

    if query_supernode:
        logger.info("Wrapping with BIRDSuperNodeDataset (query_supernode=True)")
        full_train_dataset = BIRDSuperNodeDataset(full_train_dataset)

    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_ds, val_ds = random_split(full_train_dataset, [train_size, val_size])

    # dual_stream 도 node_batch_dict 를 통해 batched forward 지원 (config 에서 지정)
    batch_size = cfg["training"].get("batch_size", 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    # Proposal C H2 (V-1 per-DB dynamic num_layers) config forward.
    # 기본값 'fixed' 에서는 v1 SchemaHeteroGAT 과 bit-wise 동일 경로로 학습. 'D_max' /
    # 'D_max_plus1' 은 diameter_dict 를 model 에 전달하여 runtime resolve. 학습 시에는 일반적으로
    # 'fixed' (= global num_layers) 로 두고, inference 단계에서 selector/config 가 per-DB depth 를
    # 해제하는 경로 (EnsembleSelector.num_layers_mode) 를 쓰는 것을 권장. 필요 시 학습 시에도
    # per-DB depth curriculum 을 돌릴 수 있도록 forward 만 해 둔다.
    diameter_path_cfg = cfg["model"].get("diameter_path")
    diameter_dict_cfg = cfg["model"].get("diameter_dict")
    gat_model = SchemaHeteroGATv2(
        in_channels=cfg["model"]["in_channels"],
        hidden_channels=cfg["model"]["hidden_channels"],
        out_channels=cfg["model"]["out_channels"],
        num_layers=cfg["model"]["num_layers"],
        heads=cfg["model"]["heads"],
        query_conditioned=query_conditioned,
        query_supernode=query_supernode,
        pairnorm_mode=cfg["model"].get("pairnorm_mode", "none"),
        pairnorm_scale=cfg["model"].get("pairnorm_scale", 1.0),
        initial_residual_alpha=cfg["model"].get("initial_residual_alpha", 0.0),
        jumping_knowledge=cfg["model"].get("jumping_knowledge", "none"),
        dual_stream=dual_stream,
        num_layers_mode=cfg["model"].get("num_layers_mode", "fixed"),
        num_layers_fallback=cfg["model"].get(
            "num_layers_fallback", cfg["model"]["num_layers"]
        ),
        diameter_path=diameter_path_cfg,
        diameter_dict=diameter_dict_cfg,
        supernode_edge_direction=cfg["model"].get(
            "supernode_edge_direction", "bidirectional"
        ),
        supernode_topk=cfg["model"].get("supernode_topk", None),
        supernode_topk_criterion=cfg["model"].get("supernode_topk_criterion", "raw"),
        supernode_threshold_mode=cfg["model"].get(
            "supernode_threshold_mode", "top_k"
        ),
        supernode_threshold_value=cfg["model"].get(
            "supernode_threshold_value", None
        ),
        supernode_score_normalization=cfg["model"].get(
            "supernode_score_normalization", "minmax"
        ),
        # Mitigation v2 (DECISIONS 2026-05-07 §1(C)/(D)) — backward compat default OFF.
        drop_message_p=cfg["model"].get("drop_message_p", 0.0),
        use_layernorm_pre_softmax=cfg["model"].get("use_layernorm_pre_softmax", False),
        aggregation_type=cfg["model"].get("aggregation_type", "mean"),
        # Mitigation V4 (DECISIONS 2026-05-11) — architectural intervention.
        gat_layer_type=cfg["model"].get("gat_layer_type", "standard"),
        softplus_symmetric_norm=cfg["model"].get("softplus_symmetric_norm", True),
        # Mitigation V5 (DECISIONS 2026-05-12 + 2026-05-13) — Tier 1+2 4-direction.
        gcnii_beta_lambda=cfg["model"].get("gcnii_beta_lambda", 0.5),
        aero_hop_attention=cfg["model"].get("aero_hop_attention", False),
        aero_cumulative_attention=cfg["model"].get("aero_cumulative_attention", False),
        aero_cumulative_decay=cfg["model"].get("aero_cumulative_decay", 1.0),
        # V6-W1 (DECISIONS 2026-06-01 §V6-W1 활성 launch) — Drop-in 4 classes 위.
        v6w1_pairnorm_scale=cfg["model"].get("v6w1_pairnorm_scale", 1.0),
        v6w1_jk_mode=cfg["model"].get("v6w1_jk_mode", "concat"),
        capture_layerwise_outputs=cfg["model"].get("capture_layerwise_outputs", False),
        # V6-W2 (DECISIONS 2026-06-04 §V6-W2 활성 launch) — edge-type 분리 (HeteroConv).
        # edge_type_split_key_relations 는 Builder per-column PK/FK mask 의존 → NotImplementedError.
        edge_type_split=cfg["model"].get("edge_type_split", False),
        edge_type_split_self_loops=cfg["model"].get("edge_type_split_self_loops", True),
        edge_type_split_aggr=cfg["model"].get("edge_type_split_aggr", "mean"),
        # V6-W3 (DECISIONS 2026-06-06) — hub 차수 축소 builder 정합 node/edge type 등록.
        # A=table_summary / B=column pooling (구조 무변경) / C=local_vn. None=backward compat.
        v6w3_variant=cfg["model"].get("v6w3_variant", None),
        # V6-W5 (DECISIONS 2026-06-07) — GAT layer-level intervention (single-shared-source fix).
        # a=column self-loop / b=per-layer residual / c=a+b. builder 변경 없음 (enriched).
        v6w5_variant=cfg["model"].get("v6w5_variant", None),
        # V6-W6 (DECISIONS 2026-06-07 #3) — directed SuperNode + V6-W5 self-loop 조합.
        # a=directed SN + self-loop / b=+ per-layer residual. query_supernode=True 결합 (config).
        v6w6_variant=cfg["model"].get("v6w6_variant", None),
    ).to(device)

    logger.info(
        f"[s06] model: QC={query_conditioned}, SN={query_supernode}, DS={dual_stream}, "
        f"PN={cfg['model'].get('pairnorm_mode', 'none')}, "
        f"α={cfg['model'].get('initial_residual_alpha', 0.0)}, "
        f"JK={cfg['model'].get('jumping_knowledge', 'none')}, "
        f"nl_mode={cfg['model'].get('num_layers_mode', 'fixed')}, "
        f"|diameter_dict|={len(gat_model.diameter_dict)} | "
        f"mitV2: drop_msg_p={cfg['model'].get('drop_message_p', 0.0)}, "
        f"layernorm_pre_softmax={cfg['model'].get('use_layernorm_pre_softmax', False)}, "
        f"aggr={cfg['model'].get('aggregation_type', 'mean')} | "
        f"mitV4: gat_layer_type={cfg['model'].get('gat_layer_type', 'standard')}, "
        f"softplus_sym_norm={cfg['model'].get('softplus_symmetric_norm', True)} | "
        f"mitV5: gcnii_beta_lambda={cfg['model'].get('gcnii_beta_lambda', 0.5)}, "
        f"aero_hop_attention={cfg['model'].get('aero_hop_attention', False)}, "
        f"aero_cumulative_attention={cfg['model'].get('aero_cumulative_attention', False)}, "
        f"aero_cumulative_decay={cfg['model'].get('aero_cumulative_decay', 1.0)} | "
        f"V6W2: edge_type_split={cfg['model'].get('edge_type_split', False)}, "
        f"self_loops={cfg['model'].get('edge_type_split_self_loops', True)}, "
        f"aggr={cfg['model'].get('edge_type_split_aggr', 'mean')}"
    )

    classifier_types = ["table", "column", "fk_node"]
    classifier_heads = nn.ModuleDict(
        {
            nt: DirectClassifierHead(
                in_dim=cfg["model"]["out_channels"],
                hidden_dim=cfg["model"].get("classifier_hidden", 256),
                dropout=cfg["model"].get("dropout", 0.1),
            ).to(device)
            for nt in classifier_types
        }
    )
    logger.info(f"DirectClassifierHead initialized for: {list(classifier_heads.keys())}")

    # Lazy-init
    logger.info("Initializing model parameters with a dummy batch...")
    gat_model.train()
    dummy_batch = full_train_dataset[0].clone().to(device)
    with torch.no_grad():
        if query_conditioned and not dual_stream:
            dummy_q = dummy_batch["query"]
            if dummy_q.dim() >= 2:
                dummy_q = dummy_q.mean(dim=0, keepdim=True)
            else:
                dummy_q = dummy_q.unsqueeze(0)
            augmented_x = {}
            for nt, x in dummy_batch.x_dict.items():
                q_exp = dummy_q.expand(x.size(0), -1)
                augmented_x[nt] = torch.cat([x, q_exp], dim=-1)
            _ = gat_model(augmented_x, dummy_batch.edge_index_dict)
        elif dual_stream:
            dummy_q = dummy_batch["query"]
            if dummy_q.dim() >= 2:
                dummy_q = dummy_q.mean(dim=0)
            _ = gat_model(dummy_batch.x_dict, dummy_batch.edge_index_dict, query_emb=dummy_q)
        else:
            _ = gat_model(dummy_batch.x_dict, dummy_batch.edge_index_dict)

    wandb.watch(gat_model, log="all")

    # Phase 3 #4 (Layer-wise LR): GAT path (HeteroConv) 만 base_lr × multiplier.
    base_lr = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])
    use_layerwise = bool(cfg["training"].get("optimizer_layer_wise_lr", False))
    gat_lr_multiplier = float(cfg["training"].get("gat_lr_multiplier", 1.0))

    if use_layerwise and gat_lr_multiplier != 1.0:
        # 'convs' 모듈 (HeteroConv 의 ModuleList) 산하 파라미터만 GAT path. 나머지 (lin_dict /
        # out_lin_dict / skip_dict / pairnorms / fusion_head / query_encoder / classifier_heads)
        # 는 base_lr 그대로.
        gat_params, other_gat_params = [], []
        for name, param in gat_model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("convs.") or ".convs." in name:
                gat_params.append(param)
            else:
                other_gat_params.append(param)
        cls_params = [p for p in classifier_heads.parameters() if p.requires_grad]
        param_groups = [
            {"params": gat_params, "lr": base_lr * gat_lr_multiplier,
             "name": "gat_convs"},
            {"params": other_gat_params, "lr": base_lr,
             "name": "gat_other"},
            {"params": cls_params, "lr": base_lr, "name": "classifier_heads"},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        logger.info(
            f"[s06] optimizer (layer-wise LR): gat_convs={len(gat_params)} "
            f"params @ lr={base_lr * gat_lr_multiplier:.2e} (×{gat_lr_multiplier}), "
            f"gat_other={len(other_gat_params)} params @ lr={base_lr:.2e}, "
            f"classifier_heads={len(cls_params)} params @ lr={base_lr:.2e}"
        )
    else:
        optimizer = torch.optim.AdamW(
            list(gat_model.parameters()) + list(classifier_heads.parameters()),
            lr=base_lr,
            weight_decay=weight_decay,
        )

    # ── Resume-from-checkpoint 로드 (lazy-init + optimizer 생성 이후라야 안전) ──
    # latest(<name>.latest.pt, 정확한 resume) 우선, 없으면 best(<name>) fallback.
    # load_state_dict 는 in-place 라 optimizer 의 param 참조 유지 (위에서 이미 생성됨).
    resume_start_epoch = 0
    resume_best_monitor = -1.0
    if resume_enabled:
        _ckpt_name = cfg.get("checkpoint_name", "best_gat_s06.pt")
        _best_path = os.path.join(PATHS["checkpoint_dir"], _ckpt_name)
        _latest_path = _best_path + ".latest.pt"
        _resume_path = _latest_path if os.path.exists(_latest_path) else (
            _best_path if os.path.exists(_best_path) else None)
        if _resume_path is None:
            logger.info(
                f"[s06][resume] 요청됐으나 checkpoint 없음 ({_latest_path} / {_best_path}) "
                f"— 처음부터 학습 시작.")
        else:
            _rk = torch.load(_resume_path, map_location=device)
            gat_model.load_state_dict(_rk["gat_state_dict"])
            classifier_heads.load_state_dict(_rk["classifier_state_dict"])
            if _rk.get("optimizer_state_dict") is not None:
                try:
                    optimizer.load_state_dict(_rk["optimizer_state_dict"])
                except Exception as _oe:
                    logger.warning(f"[s06][resume] optimizer state 복원 실패 ({_oe}) — optimizer 초기화 유지.")
            else:
                logger.warning("[s06][resume] optimizer_state_dict 없음 (구버전 ckpt) — optimizer 초기화 유지.")
            # epoch 은 '완료된 epoch 수(1-indexed)' = 다음 실행할 0-indexed epoch.
            resume_start_epoch = int(_rk.get("epoch", 0))
            # 진행 중 running-best (latest) 우선, 없으면 best ckpt 의 monitor_value.
            resume_best_monitor = float(
                _rk.get("best_monitor", _rk.get("monitor_value", -1.0)))
            logger.info(
                f"[s06][resume] {os.path.basename(_resume_path)} 로드 — "
                f"start_epoch={resume_start_epoch}, best_monitor={resume_best_monitor:.4f} "
                f"(latest={'Y' if _resume_path == _latest_path else 'N'})")

    # Loss 설정
    loss_type = cfg["training"].get("loss_type", "bce")
    pos_weight = torch.tensor([cfg["training"]["pos_weight"]]).to(device)
    listnet_fn = ListNetLoss().to(device)

    # MA-1 monitor (DECISIONS 2026-06-07 #4) — best-epoch/early-stop 지표.
    # default = recall_at_15 (backward compat). MA configs 는 gold_recall_at_theta.
    monitor_metric = str(cfg["training"].get("monitor_metric", "recall_at_15"))
    _valid_monitors = ("recall_at_15", "gold_recall_at_theta", "gold_p50")
    if monitor_metric not in _valid_monitors:
        raise ValueError(f"monitor_metric must be in {_valid_monitors}, got {monitor_metric!r}")
    theta_for_monitor = float(cfg["training"].get("theta_for_monitor", 0.1))

    # MA-2 calibration loss (DECISIONS 2026-06-07 #4) — gold score margin + per-table norm.
    # 게이트 = inference recall (EX 아님, V6 disconnect caveat). default OFF (backward compat).
    calib_margin_weight = float(cfg["training"].get("calibration_margin_weight", 0.0))
    calib_theta_target = float(cfg["training"].get("calibration_theta_target", 0.15))
    per_table_norm = bool(cfg["training"].get("per_table_score_norm", False))
    logger.info(
        f"[s06] MA-1 monitor_metric={monitor_metric} (θ={theta_for_monitor}) | "
        f"MA-2 calib_margin_weight={calib_margin_weight} (θ_target={calib_theta_target}), "
        f"per_table_score_norm={per_table_norm}"
    )
    anti_collapse_weight = float(cfg["training"].get("anti_collapse_weight", 0.0))
    anti_collapse_fn = AntiCollapseRegularizer(
        tau_max=float(cfg["training"].get("anti_collapse_tau_max", 0.85))
    ).to(device)

    # Phase 3 #3 (Direct AC on GAT output): AC loss target 선택.
    #   'fusion'         (default) — node_embs (model.forward output, skip + fusion 후)
    #   'gat_out_L_last' — 마지막 HeteroConv layer 의 column 출력 (skip + fusion 전 raw GAT)
    anti_collapse_target = str(cfg["training"].get("anti_collapse_target", "fusion"))
    if anti_collapse_target not in ("fusion", "gat_out_L_last"):
        raise ValueError(
            f"anti_collapse_target must be 'fusion' or 'gat_out_L_last', got {anti_collapse_target!r}"
        )
    # forward hook capture container — 매 forward 마다 갱신. backward graph 에 detach 안 함.
    _gat_hook_capture: dict = {"column": None}

    def _make_gat_hook(capture):
        def _hook(module, inputs, output):
            if isinstance(output, dict) and "column" in output:
                capture["column"] = output["column"]
        return _hook

    _gat_hook_handle = None
    if anti_collapse_weight > 0.0 and anti_collapse_target == "gat_out_L_last":
        if len(gat_model.convs) == 0:
            raise RuntimeError("anti_collapse_target='gat_out_L_last' requires gat_model.convs non-empty")
        _gat_hook_handle = gat_model.convs[-1].register_forward_hook(
            _make_gat_hook(_gat_hook_capture)
        )

    logger.info(
        f"[s06] loss: type={loss_type}, pos_weight={pos_weight.item()}, "
        f"anti_collapse_weight={anti_collapse_weight}, "
        f"anti_collapse_target={anti_collapse_target}"
    )

    # MA-1: best-epoch 는 monitor_metric 값 기준 (recall 변수명 retain 단 의미는 monitor).
    # resume 시 복원된 running-best / start_epoch 사용 (없으면 -1.0 / 0).
    best_monitor = resume_best_monitor
    start_epoch = resume_start_epoch
    epochs = cfg["training"]["epochs"]
    # latest checkpoint 저장 주기 (crash 복원용). default 매 epoch (최대 안전).
    save_latest_every = int(cfg["training"].get("save_latest_every_n_epochs", 1))
    if start_epoch >= epochs:
        logger.info(f"[s06][resume] start_epoch({start_epoch}) ≥ epochs({epochs}) — 이미 완료, 학습 skip.")

    for epoch in range(start_epoch, epochs):
        gat_model.train()
        for head in classifier_heads.values():
            head.train()
        epoch_loss = 0.0
        epoch_loss_main = 0.0
        epoch_loss_ac = 0.0
        epoch_loss_calib = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            batch = batch.to(device)
            optimizer.zero_grad()

            q_emb = batch["query"]
            if q_emb.dim() == 3:
                q_pooled = q_emb.mean(dim=1)
            elif q_emb.dim() == 2:
                q_pooled = q_emb
            else:
                q_pooled = q_emb.unsqueeze(0)

            node_batch_dict = {
                nt: batch[nt].batch
                for nt in batch.node_types
                if hasattr(batch[nt], "batch")
            }
            if query_conditioned and not dual_stream:
                augmented_x = {}
                for n_type, x in batch.x_dict.items():
                    node_batch_idx = batch[n_type].batch
                    q_per_node = q_pooled[node_batch_idx]
                    augmented_x[n_type] = torch.cat([x, q_per_node], dim=-1)
                node_embs = gat_model(augmented_x, batch.edge_index_dict)
            elif dual_stream:
                # batched dual_stream: q_pooled [B, d_q] + node_batch_dict 로 per-graph fusion
                node_embs = gat_model(
                    batch.x_dict, batch.edge_index_dict,
                    query_emb=q_pooled,
                    node_batch_dict=node_batch_dict,
                )
            else:
                node_embs = gat_model(batch.x_dict, batch.edge_index_dict)

            # Main loss (per-query listwise or BCE)
            step_loss_main = 0.0
            n_terms = 0
            # MA-2 calibration: gold margin loss 누적 (default weight 0 → no-op).
            step_loss_calib = torch.tensor(0.0, device=device)
            if loss_type in ("listnet", "bce_listnet"):
                # per-query 로 쪼개서 계산
                for i in range(batch.num_graphs):
                    logits_q, labels_q = [], []
                    for n_type in classifier_types:
                        if n_type not in node_embs or not hasattr(batch[n_type], "y"):
                            continue
                        mask = (batch[n_type].batch == i)
                        if not mask.any():
                            continue
                        logits = classifier_heads[n_type](node_embs[n_type][mask])
                        logits_q.append(logits)
                        labels_q.append(batch[n_type].y[mask])
                    if logits_q:
                        all_logits = torch.cat(logits_q)
                        all_labels = torch.cat(labels_q)
                        step_loss_main = step_loss_main + combined_loss(
                            all_logits, all_labels, loss_type=loss_type,
                            pos_weight=pos_weight, listnet_fn=listnet_fn,
                        )
                        n_terms += 1
                if n_terms > 0:
                    step_loss_main = step_loss_main / n_terms
                else:
                    continue
            else:  # 'bce' — per-node BCE, batch-wide
                for n_type in classifier_types:
                    if n_type not in node_embs or not hasattr(batch[n_type], "y"):
                        continue
                    if batch[n_type].num_nodes == 0:
                        continue
                    logits = classifier_heads[n_type](node_embs[n_type])
                    labels_nt = batch[n_type].y
                    # MA-2 per-table score normalization (column 한정) — PairNorm-류 압축 해소.
                    if per_table_norm and n_type == "column" and COL_TO_TAB_EDGE in batch.edge_index_dict:
                        logits = per_table_normalize_logits(
                            logits, batch.edge_index_dict[COL_TO_TAB_EDGE], logits.size(0))
                    step_loss_main = step_loss_main + combined_loss(
                        logits, labels_nt, loss_type="bce",
                        pos_weight=pos_weight, listnet_fn=listnet_fn,
                    )
                    # MA-2 gold score margin loss (gold 를 θ_target 위로).
                    if calib_margin_weight > 0.0:
                        step_loss_calib = step_loss_calib + gold_margin_loss(
                            logits, labels_nt, theta_target=calib_theta_target)
                    n_terms += 1
                if not torch.is_tensor(step_loss_main):
                    continue

            # Anti-collapse regularization — column embedding 기준
            # Phase 3 #3 dispatch:
            #   target='fusion'         (Phase 2 default) — model.forward 결과 (skip + fusion 후)
            #   target='gat_out_L_last' (Phase 3 #3)      — 마지막 HeteroConv 직후 column 출력
            #                                              (skip 결합 전 raw GAT, hook capture)
            step_loss_ac = torch.tensor(0.0, device=device)
            if anti_collapse_weight > 0.0 and COL_TO_TAB_EDGE in batch.edge_index_dict:
                if anti_collapse_target == "gat_out_L_last":
                    col_embs = _gat_hook_capture.get("column")
                    if col_embs is None:
                        # 첫 step 에서 hook 미동작 (예: 'column' 이 hetero output 에 없음). skip.
                        col_embs = node_embs.get("column")
                else:
                    col_embs = node_embs.get("column")
                if col_embs is not None:
                    cb_edge = batch.edge_index_dict[COL_TO_TAB_EDGE]
                    step_loss_ac = anti_collapse_fn(col_embs, cb_edge)

            step_loss = (step_loss_main
                         + anti_collapse_weight * step_loss_ac
                         + calib_margin_weight * step_loss_calib)

            step_loss.backward()
            optimizer.step()

            epoch_loss += step_loss.item()
            epoch_loss_main += float(step_loss_main.item() if torch.is_tensor(step_loss_main) else step_loss_main)
            epoch_loss_ac += float(step_loss_ac.item() if torch.is_tensor(step_loss_ac) else step_loss_ac)
            epoch_loss_calib += float(step_loss_calib.item() if torch.is_tensor(step_loss_calib) else step_loss_calib)

            if step % 10 == 0:
                wandb.log({
                    "train/loss_total": step_loss.item(),
                    "train/loss_main": float(step_loss_main.item() if torch.is_tensor(step_loss_main) else step_loss_main),
                    "train/loss_anti_collapse": float(step_loss_ac.item() if torch.is_tensor(step_loss_ac) else step_loss_ac),
                    "train/loss_calibration": float(step_loss_calib.item() if torch.is_tensor(step_loss_calib) else step_loss_calib),
                    "epoch": epoch + 1,
                })
            pbar.set_postfix({"loss": f"{step_loss.item():.4f}"})

        val_metrics = validate(
            gat_model, classifier_heads, val_loader, device, k=15,
            query_conditioned=query_conditioned,
            query_supernode=query_supernode,
            dual_stream=dual_stream,
            theta=theta_for_monitor,
        )
        val_recall = val_metrics["recall_at_15"]
        val_gold_recall = val_metrics["gold_recall_at_theta"]
        val_gold_p50 = val_metrics["gold_p50"]
        # MA-1: best-epoch/early-stop 선택 지표 (default gold_recall_at_theta for MA configs).
        monitor_value = val_metrics[monitor_metric]

        # Per-epoch Dirichlet energy + MAD (oversmoothing/energy, oversmoothing/mad)
        os_log = {}
        os_mean_e = 0.0
        os_mean_m = 0.0
        try:
            from utils.oversmoothing_metrics import compute_metrics_v2_gat, metrics_to_wandb_log
            gat_model.eval()
            with torch.no_grad():
                sample_batch = next(iter(val_loader)).to(device)
                q_emb_s = None
                try:
                    q_emb_s = sample_batch['query']
                except Exception:
                    q_emb_s = None
                if q_emb_s is not None and q_emb_s.dim() == 2 and q_emb_s.size(0) > 1:
                    q_emb_s = q_emb_s.mean(dim=0, keepdim=True)
                elif q_emb_s is not None and q_emb_s.dim() == 3:
                    q_emb_s = q_emb_s.mean(dim=1).mean(dim=0, keepdim=True)
                os_metrics = compute_metrics_v2_gat(
                    gat_model, sample_batch.x_dict, sample_batch.edge_index_dict,
                    query_emb=q_emb_s if query_conditioned else None,
                    device=str(device))
                os_log = metrics_to_wandb_log(os_metrics)
                os_mean_e = float(os_metrics['energy'].get('mean', 0.0) or 0.0)
                os_mean_m = float(os_metrics['mad'].get('mean', 0.0) or 0.0)
        except Exception as _e:
            logger.warning(f"Oversmoothing metric capture failed: {_e}")

        n_batches = max(len(train_loader), 1)
        wandb.log({
            "train/epoch_loss_total": epoch_loss / n_batches,
            "train/epoch_loss_main": epoch_loss_main / n_batches,
            "train/epoch_loss_anti_collapse": epoch_loss_ac / n_batches,
            "train/epoch_loss_calibration": epoch_loss_calib / n_batches,
            "val/recall_at_15": val_recall,
            # MA-1 calibration 모니터 (gold_recall_at_theta = monitor default, gold_p50 = secondary).
            "val/gold_recall_at_theta": val_gold_recall,
            "val/gold_p50": val_gold_p50,
            "val/monitor_metric_value": monitor_value,
            "epoch": epoch + 1,
            **os_log,
        })

        logger.info(
            f"Epoch {epoch+1} | Loss: {epoch_loss/n_batches:.4f} "
            f"| Main: {epoch_loss_main/n_batches:.4f} "
            f"| AC: {epoch_loss_ac/n_batches:.4f} "
            f"| Calib: {epoch_loss_calib/n_batches:.4f} "
            f"| Val R@15: {val_recall:.4f} "
            f"| gold_recall@θ: {val_gold_recall:.4f} "
            f"| gold_p50: {val_gold_p50:.4f} "
            f"| [monitor={monitor_metric}={monitor_value:.4f}] "
            f"| os/mad: {os_mean_m:.4f}"
        )

        ckpt_name = cfg.get("checkpoint_name", "best_gat_s06.pt")
        if monitor_value > best_monitor:
            best_monitor = monitor_value
            save_path = os.path.join(PATHS["checkpoint_dir"], ckpt_name)
            torch.save({
                "epoch": epoch + 1,
                "gat_state_dict": gat_model.state_dict(),
                "classifier_state_dict": classifier_heads.state_dict(),
                # 정확한 resume 위 optimizer state 동봉 (best 에서 resume 시에도 momentum 복원).
                "optimizer_state_dict": optimizer.state_dict(),
                # recall = R@15 retain (보조). monitor_metric/monitor_value 로 실제 선택 기준 기록.
                "recall": val_recall,
                "gold_recall_at_theta": val_gold_recall,
                "gold_p50": val_gold_p50,
                "monitor_metric": monitor_metric,
                "monitor_value": monitor_value,
                "config": cfg,
            }, save_path)
            wandb.run.summary["best_monitor_value"] = best_monitor
            wandb.run.summary["best_monitor_metric"] = monitor_metric
            logger.info(
                f"New Best Model Saved! {monitor_metric}={best_monitor:.4f} "
                f"(R@15={val_recall:.4f}, gold_p50={val_gold_p50:.4f})")

        # ── Latest checkpoint (crash-resume 용, save_latest_every epoch 마다 overwrite) ──
        # best-only 저장은 best epoch 이후 crash 시 진행분 손실 → latest 로 정확 복원.
        # epoch=완료 epoch 수(1-indexed)=resume 시 다음 0-indexed epoch. best_monitor=running-best.
        if save_latest_every > 0 and ((epoch + 1) % save_latest_every == 0 or epoch + 1 == epochs):
            latest_path = os.path.join(PATHS["checkpoint_dir"], ckpt_name + ".latest.pt")
            torch.save({
                "epoch": epoch + 1,
                "gat_state_dict": gat_model.state_dict(),
                "classifier_state_dict": classifier_heads.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "recall": val_recall,
                "gold_recall_at_theta": val_gold_recall,
                "gold_p50": val_gold_p50,
                "monitor_metric": monitor_metric,
                "monitor_value": monitor_value,
                "best_monitor": best_monitor,  # resume 시 running-best 복원용
                "config": cfg,
            }, latest_path)

    logger.info(f"Training Completed. Best {monitor_metric}={best_monitor:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="checkpoint_dir/<name>.latest.pt(우선)/<name> 에서 resume "
                             "(RESUME=1 환경변수로도 활성화).")
    args = parser.parse_args()
    run_train(args.config, resume=args.resume)
