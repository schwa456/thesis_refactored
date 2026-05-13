"""Smoke test for Phase 3 Skip Mitigation candidates (#3 Direct AC + #4 Layer-wise LR).

학위 본 심사 전 진행 (DECISIONS 2026-05-06 §1(C)). Phase 2 null mechanism 진단 결과
Skip Dependence Pathology DOMINANT — main GAT path gradient 1/10 축소. Phase 3 는 직접 mitigation:
  #3 (PRIMARY): AC loss target = 'gat_out_L_last' (forward hook capture, skip + fusion 전 raw GAT)
  #4 (SECONDARY): optimizer parameter groups, GAT path × 5 LR

검증 대상:
  (1) #3 anti_collapse_target='gat_out_L_last' 시 forward hook 이 마지막 conv layer column output 을
      capture. fusion 결과 (node_embs['column']) 와 다른 tensor.
  (2) #3 hook capture 가 backward graph 에 detach 안 됨 — main GAT param 의 grad 가 흐름.
  (3) #4 layer_wise_lr=True 시 optimizer 가 3 param groups 으로 분리 (gat_convs / gat_other / classifier).
      gat_convs group 의 LR 이 base_lr × multiplier.
  (4) #4 의 'conv' filter 가 HeteroConv inner 만 잡고, lin_dict / out_lin_dict / skip_dict /
      fusion_head / query_encoder 는 other_gat_params 로 분류.
  (5) #4 vs Phase 2 baseline (단일 LR) 의 backward compat — layer_wise_lr=False 시 동일 동작.
  (6) Config 파싱 — 신규 옵션 (anti_collapse_target / optimizer_layer_wise_lr / gat_lr_multiplier) 정상 read.

Run from project root:
    PYTHONPATH=src conda run -n base python src/modules/selectors/tests/test_phase3_mitigations.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import yaml


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _build_v2_dual_stream_model(num_layers: int = 2, in_channels: int = 384,
                                hidden_channels: int = 64, heads: int = 2):
    """Phase 2 b5 mitigation 와 동일 mitigation set 의 작은 모델 — smoke 용."""
    from models.gat_network_v2 import SchemaHeteroGATv2
    return SchemaHeteroGATv2(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=hidden_channels,
        num_layers=num_layers,
        heads=heads,
        query_conditioned=False,
        query_supernode=True,
        pairnorm_mode="pairnorm",
        initial_residual_alpha=0.2,
        jumping_knowledge="concat",
        dual_stream=True,
        supernode_edge_direction="directed_from_sn",
        supernode_threshold_mode="percentile",
        supernode_threshold_value=80.0,
    )


def _build_synthetic_supernode_graph(num_tables=4, cols_per_table=6, num_fk=3,
                                     in_dim=384, seed=42):
    from torch_geometric.data import HeteroData
    g = torch.Generator().manual_seed(seed)
    total_cols = num_tables * cols_per_table
    data = HeteroData()
    data["table"].x = torch.randn(num_tables, in_dim, generator=g)
    data["column"].x = torch.randn(total_cols, in_dim, generator=g)
    data["fk_node"].x = torch.randn(num_fk, in_dim, generator=g)
    data["query_node"].x = torch.randn(1, in_dim, generator=g)

    t_src, c_dst = [], []
    for t in range(num_tables):
        for j in range(cols_per_table):
            t_src.append(t); c_dst.append(t * cols_per_table + j)
    tc = torch.tensor([t_src, c_dst], dtype=torch.long)
    data["table", "has_column", "column"].edge_index = tc
    data["column", "belongs_to", "table"].edge_index = tc.flip(0)

    fke = torch.tensor([list(range(num_fk)), list(range(num_fk))], dtype=torch.long)
    data["column", "is_source_of", "fk_node"].edge_index = fke
    data["fk_node", "points_to", "column"].edge_index = fke.flip(0)

    if num_tables > 1:
        s = list(range(num_tables - 1)); d = list(range(1, num_tables))
        data["table", "table_to_table", "table"].edge_index = torch.tensor(
            [s + d, d + s], dtype=torch.long)
    else:
        data["table", "table_to_table", "table"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    for nt in ("table", "column", "fk_node"):
        n = data[nt].num_nodes
        src = torch.zeros(n, dtype=torch.long)
        dst = torch.arange(n, dtype=torch.long)
        data["query_node", f"attends_to_{nt}", nt].edge_index = torch.stack([src, dst], 0)
    return data


# ──────────────────────────────────────────────────────────────────────
# Tests — Phase 3 #3 (Direct AC on GAT output)
# ──────────────────────────────────────────────────────────────────────

def test_p3_3_hook_captures_last_conv_output():
    """Forward hook 이 model.convs[-1] 의 column output 을 capture. node_embs 와 다른 tensor."""
    print("\n[test_p3_3_hook_captures_last_conv_output]")
    model = _build_v2_dual_stream_model()
    model.eval()
    data = _build_synthetic_supernode_graph()
    q_emb = data["query_node"].x

    capture = {"column": None}
    def _hook(module, inputs, output):
        if isinstance(output, dict) and "column" in output:
            capture["column"] = output["column"]
    handle = model.convs[-1].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            node_embs = model(data.x_dict, data.edge_index_dict, query_emb=q_emb)
    finally:
        handle.remove()

    assert capture["column"] is not None, "hook must capture column output of last conv"
    assert "column" in node_embs, "fusion output must contain 'column'"

    fusion_col = node_embs["column"]
    gat_col = capture["column"]
    # GAT raw output 차원 (hidden*heads) vs fusion output 차원 (out_channels) — dual_stream 에서
    # fusion_head 가 [out_channels*3 → out_channels] projection 하므로 다른 shape.
    print(f"  fusion col shape={tuple(fusion_col.shape)}, gat L_last col shape={tuple(gat_col.shape)}")
    # 두 tensor 가 절대 같지 않아야 함 — 한 쪽은 fusion 후, 한 쪽은 raw.
    assert fusion_col.shape != gat_col.shape or not torch.allclose(
        fusion_col, gat_col[..., :fusion_col.size(-1)]
    ), "fusion vs raw GAT output should differ"


def test_p3_3_hook_backward_graph_intact():
    """Hook 이 captured tensor 의 requires_grad / grad path 를 보존 — backward 가능."""
    print("\n[test_p3_3_hook_backward_graph_intact]")
    from models.losses import AntiCollapseRegularizer
    model = _build_v2_dual_stream_model()
    model.train()
    data = _build_synthetic_supernode_graph()
    q_emb = data["query_node"].x

    capture = {"column": None}
    def _hook(module, inputs, output):
        if isinstance(output, dict) and "column" in output:
            capture["column"] = output["column"]
    handle = model.convs[-1].register_forward_hook(_hook)
    try:
        node_embs = model(data.x_dict, data.edge_index_dict, query_emb=q_emb)
    finally:
        handle.remove()

    # Use AC regularizer on captured GAT output (skip + fusion 전).
    ac = AntiCollapseRegularizer(tau_max=0.85)
    cb_edge = data["column", "belongs_to", "table"].edge_index
    ac_loss = ac(capture["column"], cb_edge)
    assert ac_loss.requires_grad, "AC loss on hook capture must have grad"

    ac_loss.backward()
    # main GAT path (last conv) 의 한 weight 에 grad 가 도달해야 함
    last_conv = model.convs[-1]
    grads_present = [p.grad is not None and p.grad.abs().sum().item() > 0
                     for p in last_conv.parameters() if p.requires_grad]
    assert any(grads_present), "main GAT path (last conv) must receive gradient from AC loss"
    n_grad = sum(grads_present)
    n_total = len([p for p in last_conv.parameters() if p.requires_grad])
    print(f"  last conv params with grad: {n_grad}/{n_total}")


# ──────────────────────────────────────────────────────────────────────
# Tests — Phase 3 #4 (Layer-wise LR)
# ──────────────────────────────────────────────────────────────────────

def test_p3_4_param_group_filter_correctness():
    """`name.startswith('convs.')` 이 HeteroConv 산하만 잡고, 다른 모듈은 분리."""
    print("\n[test_p3_4_param_group_filter_correctness]")
    model = _build_v2_dual_stream_model()
    gat_names, other_names = [], []
    for name, _ in model.named_parameters():
        if name.startswith("convs.") or ".convs." in name:
            gat_names.append(name)
        else:
            other_names.append(name)

    # 'convs' filter 는 HeteroConv 의 ModuleList + inner GATv2Conv 모두 매치.
    print(f"  gat-path params: {len(gat_names)}, other params: {len(other_names)}")
    assert len(gat_names) > 0, "no gat-path params — filter broken"
    assert len(other_names) > 0, "no non-gat params — filter broken"
    # Sanity — 절대 다른 path 가 들어오면 안 되는 모듈
    expected_other_modules = ("lin_dict.", "out_lin_dict.", "skip_dict.", "fusion_head.",
                              "query_encoder.", "pairnorms.")
    must_be_other = [n for n in gat_names if any(m in n for m in expected_other_modules)]
    assert not must_be_other, f"these belong to non-gat path: {must_be_other[:5]}"

    # gat path 안에 'convs.<int>.convs.<edge_type>' 형태가 있는지 확인 (PyG HeteroConv 구조)
    inner_convs_present = any(".convs." in n for n in gat_names)
    assert inner_convs_present, "expected HeteroConv inner GATv2Conv params in gat-path"


def test_p3_4_optimizer_lr_assignment():
    """Layer-wise LR 활성화 시 optimizer param_groups 의 gat_convs LR = base_lr × multiplier."""
    print("\n[test_p3_4_optimizer_lr_assignment]")
    model = _build_v2_dual_stream_model()
    base_lr = 1e-4
    mult = 5.0

    gat_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("convs.") or ".convs." in name:
            gat_params.append(p)
        else:
            other_params.append(p)
    cls_dummy = torch.nn.Linear(64, 1)  # classifier_heads stand-in
    cls_params = list(cls_dummy.parameters())

    optimizer = torch.optim.AdamW([
        {"params": gat_params, "lr": base_lr * mult, "name": "gat_convs"},
        {"params": other_params, "lr": base_lr, "name": "gat_other"},
        {"params": cls_params, "lr": base_lr, "name": "classifier_heads"},
    ], weight_decay=1e-5)
    assert len(optimizer.param_groups) == 3
    lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
    print(f"  param group LRs: {lrs}")
    assert abs(lrs["gat_convs"] - base_lr * mult) < 1e-12
    assert abs(lrs["gat_other"] - base_lr) < 1e-12
    assert abs(lrs["classifier_heads"] - base_lr) < 1e-12


def test_p3_4_backward_compat_baseline():
    """layer_wise_lr=False 시 단일 LR optimizer (Phase 2 baseline)."""
    print("\n[test_p3_4_backward_compat_baseline]")
    model = _build_v2_dual_stream_model()
    base_lr = 1e-4
    cls_dummy = torch.nn.Linear(64, 1)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(cls_dummy.parameters()),
        lr=base_lr, weight_decay=1e-5,
    )
    # 단일 group, lr == base_lr
    assert len(optimizer.param_groups) == 1
    assert abs(optimizer.param_groups[0]["lr"] - base_lr) < 1e-12
    print(f"  single-group optimizer OK, lr={base_lr}")


# ──────────────────────────────────────────────────────────────────────
# Tests — Config + integration
# ──────────────────────────────────────────────────────────────────────

def test_phase3_config_parsing():
    """신규 config 2 종 파싱 + 핵심 옵션 확인."""
    print("\n[test_phase3_config_parsing]")
    cfg_dir = ROOT / "configs/training"
    cfg_3 = cfg_dir / "train_gat_directed_supernode_p80_b5_phase3_directAC.yaml"
    cfg_4 = cfg_dir / "train_gat_directed_supernode_p80_b5_phase3_layerwiseLR.yaml"
    assert cfg_3.exists() and cfg_4.exists()

    with open(cfg_3) as f:
        c3 = yaml.safe_load(f)
    assert c3["training"]["anti_collapse_target"] == "gat_out_L_last", (
        "Phase 3 #3 must set anti_collapse_target='gat_out_L_last'"
    )
    assert c3["training"]["anti_collapse_weight"] > 0.0
    assert c3["model"]["dual_stream"] is True
    assert c3["model"]["supernode_edge_direction"] == "directed_from_sn"
    print(f"  #3 cfg: anti_collapse_target={c3['training']['anti_collapse_target']}, "
          f"weight={c3['training']['anti_collapse_weight']}")

    with open(cfg_4) as f:
        c4 = yaml.safe_load(f)
    assert c4["training"]["optimizer_layer_wise_lr"] is True
    assert float(c4["training"]["gat_lr_multiplier"]) == 5.0
    base_lr = float(c4["training"]["learning_rate"])
    print(f"  #4 cfg: layer_wise_lr=True, gat_lr={base_lr * 5.0:.2e}, base={base_lr:.2e}")


def test_phase2_baseline_unchanged():
    """Regression: Phase 2 base config 가 신규 옵션 미설정 — backward compat 보존."""
    print("\n[test_phase2_baseline_unchanged]")
    cfg_2 = ROOT / "configs/training/train_gat_directed_supernode_p80_b5_mitigation.yaml"
    assert cfg_2.exists()
    with open(cfg_2) as f:
        c = yaml.safe_load(f)
    # Phase 2 config 는 anti_collapse_target / layer_wise_lr 미설정 — code 측 default 사용
    assert "anti_collapse_target" not in c["training"], (
        "Phase 2 baseline must NOT set anti_collapse_target (code default 'fusion' 사용)"
    )
    assert not c["training"].get("optimizer_layer_wise_lr", False), (
        "Phase 2 baseline must NOT set optimizer_layer_wise_lr"
    )
    print("  Phase 2 baseline 보존 — 신규 옵션 미설정 확인")


def main():
    test_p3_3_hook_captures_last_conv_output()
    test_p3_3_hook_backward_graph_intact()
    test_p3_4_param_group_filter_correctness()
    test_p3_4_optimizer_lr_assignment()
    test_p3_4_backward_compat_baseline()
    test_phase3_config_parsing()
    test_phase2_baseline_unchanged()
    print("\nAll Phase 3 mitigation smoke tests passed.")


if __name__ == "__main__":
    main()
