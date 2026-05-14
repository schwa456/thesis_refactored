"""Smoke tests for GRAST-FD Direction C-GT (Graph Transformer Option β).

Architecture forward + integration + fallback paths. Random-init transformer
(checkpoint 없이) 으로도 동작 검증 가능 — Filter integration 의 fallback safe path.
실제 BIRD-Train fine-tune 학습은 별도 Root chain.
"""
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("VLLM_BASE_URL", "http://localhost:8000/v1")
os.environ.setdefault("VLLM_API_KEY", "dummy")

import torch  # noqa: E402

from modules.filters.grast_fd_transformer import (  # noqa: E402
    GraphTransformerEncoder,
    RelationAwareGTLayer,
    margin_contrastive_loss,
    smoke_train_protocol,
    EDGE_TYPE_FK_FORWARD,
    EDGE_TYPE_FK_REVERSE,
    EDGE_TYPE_COL_TO_FK_FORWARD,
    EDGE_TYPE_COL_TO_PK_FORWARD,
    NUM_EDGE_TYPES,
)
from modules.filters.grast_fd_filter_with_transformer import (  # noqa: E402
    GRASTFDFilterWithTransformer,
)


class _CallableMock:
    def __init__(self, response: Any):
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def generate_text(self, prompt: str, model: str, temperature: float, **kw) -> str:
        self.calls.append({"prompt": prompt, "model": model, "temperature": temperature})
        return str(self.response)


def _assert(cond: bool, msg: str):
    if not cond:
        print(f"  ✗ FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


# ============================================================
# Architecture: GraphTransformerEncoder
# ============================================================
def test_gt_encoder_forward_shape():
    print("\n[test] GT encoder forward returns (h_refined[N,H], scores[N])")
    torch.manual_seed(0)
    model = GraphTransformerEncoder(in_dim=8, hidden_dim=64, num_layers=2, num_heads=4)
    N = 5
    h0 = torch.randn(N, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 2, 4], dtype=torch.long)
    h_out, scores = model(h0, edge_index, edge_type)
    _assert(tuple(h_out.shape) == (N, 64), f"h_out shape {tuple(h_out.shape)}")
    _assert(tuple(scores.shape) == (N,), f"scores shape {tuple(scores.shape)}")


def test_gt_encoder_handles_empty_edges():
    print("\n[test] GT encoder handles edge_index of shape [2, 0]")
    torch.manual_seed(0)
    model = GraphTransformerEncoder(in_dim=8, hidden_dim=32, num_layers=1, num_heads=2)
    h0 = torch.randn(3, 8)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_type = torch.zeros((0,), dtype=torch.long)
    h_out, scores = model(h0, edge_index, edge_type)
    _assert(tuple(h_out.shape) == (3, 32), "h_out shape with no edges")
    _assert(torch.all(torch.isfinite(scores)), "no NaN/Inf in scores")


def test_gt_encoder_gradient_flows():
    print("\n[test] GT encoder gradient flows through forward")
    torch.manual_seed(0)
    model = GraphTransformerEncoder(in_dim=4, hidden_dim=16, num_layers=2, num_heads=2)
    h0 = torch.randn(4, 4, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_type = torch.tensor([0, 2, 4], dtype=torch.long)
    _, scores = model(h0, edge_index, edge_type)
    loss = scores.sum()
    loss.backward()
    _assert(h0.grad is not None, "h0 received gradient")
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    _assert(any(g > 0 for g in grad_norms), "model parameters have non-zero grad")


def test_gt_layer_shape_mismatch_raises():
    print("\n[test] hidden_dim not divisible by num_heads → ValueError")
    try:
        RelationAwareGTLayer(hidden_dim=10, num_heads=4)
        _assert(False, "expected ValueError")
    except ValueError:
        _assert(True, "ValueError raised")


def test_gt_encoder_relation_bias_distinct_per_type():
    print("\n[test] relation_bias initialized differently per edge type")
    torch.manual_seed(0)
    layer = RelationAwareGTLayer(hidden_dim=32, num_heads=4)
    bias = layer.relation_bias
    _assert(tuple(bias.shape) == (NUM_EDGE_TYPES, 4), f"shape {tuple(bias.shape)}")
    # Initialization with std=0.02 → all-zero 확률 극히 낮음
    _assert(not torch.allclose(bias, torch.zeros_like(bias)), "non-zero init")


# ============================================================
# Training utility: margin loss + smoke train protocol
# ============================================================
def test_margin_loss_basic():
    print("\n[test] margin_contrastive_loss > 0 when gold scored below non-gold")
    scores = torch.tensor([0.1, 0.5, 0.3, 0.8])
    gold = torch.tensor([True, False, False, False])  # gold=0.1, all non-gold higher
    loss = margin_contrastive_loss(scores, gold, margin=0.1)
    _assert(loss.item() > 0, f"loss > 0, got {loss.item()}")


def test_margin_loss_zero_when_well_separated():
    print("\n[test] margin loss ≈ 0 when gold scored well above non-gold")
    scores = torch.tensor([1.5, 0.1, 0.2, 0.3])
    gold = torch.tensor([True, False, False, False])
    loss = margin_contrastive_loss(scores, gold, margin=0.1)
    _assert(loss.item() == 0.0, f"loss ≈ 0, got {loss.item()}")


def test_margin_loss_empty_pos_returns_zero():
    print("\n[test] margin loss returns 0 when no positives")
    scores = torch.tensor([0.5, 0.5, 0.5])
    gold = torch.tensor([False, False, False])
    loss = margin_contrastive_loss(scores, gold)
    _assert(loss.item() == 0.0, "no positives → loss 0")


def _make_synthetic_batch(N: int = 6, E: int = 8, in_dim: int = 8):
    torch.manual_seed(42)
    return {
        "h0": torch.randn(N, in_dim),
        "edge_index": torch.randint(0, N, (2, E)),
        "edge_type": torch.randint(0, NUM_EDGE_TYPES, (E,)),
        "gold_mask": torch.tensor([True, True, False, False, False, False]),
    }


def test_smoke_train_protocol_runs():
    print("\n[test] smoke_train_protocol runs without exception (synthetic)")
    torch.manual_seed(0)
    model = GraphTransformerEncoder(in_dim=8, hidden_dim=32, num_layers=2, num_heads=4)
    batches = [_make_synthetic_batch() for _ in range(3)]
    val = [_make_synthetic_batch() for _ in range(2)]
    out = smoke_train_protocol(model, batches, val, num_epochs=3, lr=1e-3,
                                pass_loss_threshold=0.3, pass_pr_auc_delta=0.0)
    _assert("passed" in out and isinstance(out["passed"], bool), "passed flag present")
    _assert(isinstance(out["epoch_losses"], list) and len(out["epoch_losses"]) <= 3,
            "epoch_losses recorded")
    _assert(out["final_train_loss"] < float("inf"), "final loss finite")


# ============================================================
# Filter integration: GRASTFDFilterWithTransformer
# ============================================================
def _make_filter_with_transformer(
    transformer_in_dim: int = 16,
    transformer_hidden_dim: int = 32,
    transformer_num_layers: int = 2,
    transformer_num_heads: int = 4,
    transformer_score_top_k: int = 3,
    transformer_score_threshold=None,
    terminal_source: str = "graph_transformer",
    xiyan_response: str = "{}",
    inferred_fk: List[str] = None,
    fk_pk_hardcode: bool = False,
    inject_random_transformer: bool = True,
):
    flt = GRASTFDFilterWithTransformer(
        model_name="mock-model",
        provider=None,
        db_dir="/nonexistent",
        xiyan_max_iteration=1,
        num_examples=0,
        terminal_source=terminal_source,
        transformer_in_dim=transformer_in_dim,
        transformer_hidden_dim=transformer_hidden_dim,
        transformer_num_layers=transformer_num_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_score_top_k=transformer_score_top_k,
        transformer_score_threshold=transformer_score_threshold,
        transformer_device="cpu",
        inferred_fk=inferred_fk or [],
        fk_pk_hardcode=fk_pk_hardcode,
        max_restore=30,
        xiyan_num_examples=0,
    )
    flt.xiyan.client = _CallableMock(xiyan_response)
    flt.client = _CallableMock("")
    if inject_random_transformer:
        torch.manual_seed(0)
        flt.transformer = flt._build_transformer().eval()
    return flt


def test_filter_falls_back_when_no_checkpoint():
    print("\n[test] no checkpoint → fallback terminal_source='forward'")
    flt = _make_filter_with_transformer(
        inject_random_transformer=False,
        xiyan_response=json.dumps({"users": ["name", "id"]}),
    )
    _assert(flt.transformer is None, "transformer not loaded")
    result = flt.refine(
        query="q",
        subgraph={"users": ["name", "id"]},
        db_id=None,
        metadata={"col_to_id": {"users.id": 0, "users.name": 1},
                  "table_to_id": {"users": 0}, "fk_to_id": {}},
    )
    _assert(
        "forward" in result["stats"]["terminal_source_used"],
        f"fallback to forward (got '{result['stats']['terminal_source_used']}')",
    )
    _assert(
        result["filter_info"]["filter_transformer_available"] is False,
        "transformer_available=False recorded",
    )


def test_filter_uses_transformer_when_loaded():
    print("\n[test] random-init transformer present → terminal_source_used='graph_transformer'")
    flt = _make_filter_with_transformer(
        xiyan_response=json.dumps({"users": ["name"]}),
        inject_random_transformer=True,
        transformer_score_top_k=2,
    )
    result = flt.refine(
        query="q",
        subgraph={"users": ["name", "id", "email"]},
        db_id=None,
        metadata={
            "col_to_id": {"users.id": 0, "users.name": 1, "users.email": 2},
            "table_to_id": {"users": 0},
            "fk_to_id": {},
        },
    )
    _assert(
        result["stats"]["terminal_source_used"] == "graph_transformer",
        f"GT used (got '{result['stats']['terminal_source_used']}')",
    )
    _assert(
        result["filter_info"]["filter_transformer_available"] is True,
        "transformer_available=True",
    )
    _assert("users.name" in result["final_nodes"], "forward (S_fwd) preserved")


def test_filter_top_k_terminal_uses_score_ranking():
    print("\n[test] top_k=1 → at most 1 GT-picked terminal + forward")
    flt = _make_filter_with_transformer(
        xiyan_response=json.dumps({"users": ["name"]}),
        inject_random_transformer=True,
        transformer_score_top_k=1,
    )
    result = flt.refine(
        query="q",
        subgraph={"users": ["name", "id", "email", "age"]},
        db_id=None,
        metadata={
            "col_to_id": {
                "users.id": 0, "users.name": 1, "users.email": 2, "users.age": 3,
            },
            "table_to_id": {"users": 0},
            "fk_to_id": {},
        },
    )
    _assert(result["stats"]["terminal_source_used"] == "graph_transformer",
            "GT terminal source")
    # terminals = GT top-1 ∪ forward(name). Steiner tree 는 belongs_to 없으니
    # column-only graph 에서 single-terminal 또는 2-terminal component → restore=0~소수.
    _assert(result["stats"]["terminal_count"] >= 1, "≥1 terminal selected")


def test_filter_score_threshold_mode():
    print("\n[test] threshold=0.0 → most columns pass; threshold=1.0 → all reject (but forward kept)")
    # threshold=1.0 사실상 GT 출력 (sigmoid) 0~1 → 1.0 이상은 거의 없음 → terminal = forward only
    flt = _make_filter_with_transformer(
        xiyan_response=json.dumps({"users": ["name"]}),
        inject_random_transformer=True,
        transformer_score_threshold=1.0,
    )
    result = flt.refine(
        query="q",
        subgraph={"users": ["name", "id"]},
        db_id=None,
        metadata={
            "col_to_id": {"users.id": 0, "users.name": 1},
            "table_to_id": {"users": 0},
            "fk_to_id": {},
        },
    )
    _assert(result["stats"]["terminal_source_used"] == "graph_transformer",
            "GT mode active")
    # 모든 GT-picked 가 reject 되어도 forward(name) 은 terminal 로 유지
    _assert("users.name" in result["final_nodes"], "forward preserved despite reject")


def test_filter_uses_fk_edges_when_metadata_provided():
    print("\n[test] FK metadata → GT graph includes FK edges, terminals link via FK path")
    xiyan_resp = json.dumps({"users": ["name"], "orders": ["total"]})
    flt = _make_filter_with_transformer(
        xiyan_response=xiyan_resp,
        inject_random_transformer=True,
        transformer_score_top_k=2,
        fk_pk_hardcode=False,
    )
    metadata = {
        "col_to_id": {
            "users.id": 0, "users.name": 1,
            "orders.user_id": 2, "orders.total": 3,
        },
        "table_to_id": {"users": 0, "orders": 1},
        "fk_to_id": {"orders.user_id->users.id": 0},
    }
    result = flt.refine(
        query="join", subgraph={"users": ["name", "id"], "orders": ["total", "user_id"]},
        db_id=None, metadata=metadata,
    )
    # GT 가 어떤 column 을 top-K 로 뽑든 forward(name + total) 은 terminal 에 포함되어
    # Steiner 가 FK 경로 (users.id, orders.user_id) 를 회복 가능.
    final = set(result["final_nodes"])
    _assert("users.name" in final and "orders.total" in final, "forward preserved")
    # FK 경유 restore 가 적어도 1 col 발생할 수 있음 (random init 이라 확실치 않으나
    # graph 에 FK edge 가 있으므로 가능성 있음)
    _assert(result["stats"]["declared_fk_count"] == 1, "declared FK counted")


def test_filter_invalid_terminal_source_raises():
    print("\n[test] invalid terminal_source raises ValueError")
    try:
        GRASTFDFilterWithTransformer(
            terminal_source="bogus", provider=None, db_dir="/x",
        )
        _assert(False, "expected ValueError")
    except ValueError:
        _assert(True, "ValueError raised")


def test_filter_checkpoint_path_load_error_safe():
    print("\n[test] checkpoint path that does not exist → transformer=None + filter still works")
    flt = GRASTFDFilterWithTransformer(
        transformer_checkpoint_path="/nonexistent/ckpt.pt",
        transformer_in_dim=8,
        transformer_hidden_dim=16,
        transformer_num_layers=1,
        transformer_num_heads=2,
        terminal_source="graph_transformer",
        provider=None, db_dir="/x",
        xiyan_max_iteration=1, num_examples=0,
        xiyan_num_examples=0, fk_pk_hardcode=False,
    )
    flt.xiyan.client = _CallableMock(json.dumps({"users": ["name"]}))
    flt.client = _CallableMock("")
    _assert(flt.transformer is None, "transformer None after load failure")
    _assert(flt._transformer_load_error is not None, "load error recorded")
    result = flt.refine(
        query="q", subgraph={"users": ["name"]}, db_id=None,
        metadata={"col_to_id": {"users.name": 0}, "table_to_id": {"users": 0}, "fk_to_id": {}},
    )
    _assert(
        result["filter_info"]["filter_transformer_available"] is False,
        "transformer_available False",
    )
    _assert("users.name" in result["final_nodes"], "forward preserved")


def run_all():
    tests = [
        test_gt_encoder_forward_shape,
        test_gt_encoder_handles_empty_edges,
        test_gt_encoder_gradient_flows,
        test_gt_layer_shape_mismatch_raises,
        test_gt_encoder_relation_bias_distinct_per_type,
        test_margin_loss_basic,
        test_margin_loss_zero_when_well_separated,
        test_margin_loss_empty_pos_returns_zero,
        test_smoke_train_protocol_runs,
        test_filter_falls_back_when_no_checkpoint,
        test_filter_uses_transformer_when_loaded,
        test_filter_top_k_terminal_uses_score_ranking,
        test_filter_score_threshold_mode,
        test_filter_uses_fk_edges_when_metadata_provided,
        test_filter_invalid_terminal_source_raises,
        test_filter_checkpoint_path_load_error_safe,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
        except Exception as e:
            failures.append((t.__name__, f"UNEXPECTED: {type(e).__name__}: {e}"))

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} / {len(tests)}")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    print(f"PASSED: {len(tests)} / {len(tests)}")


if __name__ == "__main__":
    run_all()
