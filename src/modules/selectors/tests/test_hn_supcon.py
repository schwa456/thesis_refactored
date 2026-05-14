"""HN-SupCon (Direction B) smoke test — toy data 로 margin mask + NT-Xent + selector wiring 검증.

학술 Agent Phase 5 Q3 + Q5 의 hyperparameter 및 form 정합성 확인.
Pipeline-level smoke (10% data / 0.1 epoch) 은 root chain (train_hn_supcon.py launch) 가 처리.

Run from project root:
    PYTHONPATH=src conda run -n base python src/modules/selectors/tests/test_hn_supcon.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F


def test_margin_mask_form():
    """학술 Agent Phase 5 §4.1: m_ij = 1[s(q,n) > s(q,p) - margin]."""
    print("\n[test_margin_mask_form]")
    from modules.selectors.hn_supcon_selector import build_hard_negative_mask

    # B=1, single positive @0.8, three negatives @0.85 / 0.75 / 0.5, margin=0.1
    sim_qp = torch.tensor([[0.8]])
    sim_qn = torch.tensor([[0.85, 0.75, 0.5]])
    mask = build_hard_negative_mask(sim_qp, sim_qn, margin=0.1)
    # p_max=0.8, threshold = 0.8 - 0.1 = 0.7
    # 0.85 > 0.7 → True / 0.75 > 0.7 → True / 0.5 > 0.7 → False
    expected = torch.tensor([[True, True, False]])
    assert torch.equal(mask, expected), f"mask mismatch: {mask.tolist()} vs {expected.tolist()}"
    print(f"  OK margin=0.1 mask = {mask.tolist()} (p_max=0.8, threshold=0.7)")


def test_margin_mask_multi_positive():
    """Multi-positive 시 p_max = max(positives), threshold = p_max - margin."""
    print("\n[test_margin_mask_multi_positive]")
    from modules.selectors.hn_supcon_selector import build_hard_negative_mask

    # B=2 queries. Q1: 2 positives @ {0.7, 0.85}. Q2: 1 positive @ 0.6.
    sim_qp = torch.tensor([[0.7, 0.85], [0.6, float("-inf")]])
    sim_qn = torch.tensor([[0.8, 0.5, 0.9], [0.55, 0.4, 0.65]])
    mask = build_hard_negative_mask(sim_qp, sim_qn, margin=0.1)
    # Q1: p_max=0.85, thr=0.75 → [True (0.8>0.75), False (0.5≤0.75), True (0.9>0.75)]
    # Q2: p_max=0.6,  thr=0.5  → [True (0.55>0.5), False (0.4≤0.5), True (0.65>0.5)]
    expected = torch.tensor([[True, False, True], [True, False, True]])
    assert torch.equal(mask, expected), f"mask multi-pos mismatch: {mask.tolist()}"
    print(f"  OK multi-positive p_max correctly = max(positives), masked by threshold")


def test_margin_ablation_grid():
    """학술 Agent §4.3: margin ∈ {0, 0.1, 0.2} ablation — default 0.1."""
    print("\n[test_margin_ablation_grid]")
    from modules.selectors.hn_supcon_selector import build_hard_negative_mask

    sim_qp = torch.tensor([[0.8]])
    sim_qn = torch.tensor([[0.85, 0.75, 0.5, 0.65]])
    # margin=0 → strict (> 0.8): [True, False, False, False]
    m0 = build_hard_negative_mask(sim_qp, sim_qn, margin=0.0)
    # margin=0.1 → > 0.7: [True, True, False, False]
    m1 = build_hard_negative_mask(sim_qp, sim_qn, margin=0.1)
    # margin=0.2 → > 0.6: [True, True, False, True]
    m2 = build_hard_negative_mask(sim_qp, sim_qn, margin=0.2)
    assert m0.tolist() == [[True, False, False, False]]
    assert m1.tolist() == [[True, True, False, False]]
    assert m2.tolist() == [[True, True, False, True]]
    n0 = m0.sum().item(); n1 = m1.sum().item(); n2 = m2.sum().item()
    print(f"  OK margin sweep: |m@0|={n0}, |m@0.1|={n1}, |m@0.2|={n2} (monotone ↑)")


def test_ntxent_loss_form():
    """학술 Agent §4.2: L = -log[exp(s_p/τ) / (exp(s_p/τ) + Σ m·exp(s_n/τ))]."""
    print("\n[test_ntxent_loss_form]")
    from modules.selectors.hn_supcon_selector import hn_supcon_loss

    # B=1, 1 positive @0.9, 2 hard negatives @0.8 + 0.7, τ=0.07
    sim_qp = torch.tensor([[0.9]])
    sim_qn = torch.tensor([[0.8, 0.7]])
    pos_mask = torch.tensor([[True]])
    hn_mask = torch.tensor([[True, True]])
    tau = 0.07

    loss = hn_supcon_loss(sim_qp, sim_qn, pos_mask, hn_mask, tau=tau, n_per_query=8)
    # Manual: numerator = exp(0.9/τ), denom = numerator + exp(0.8/τ) + exp(0.7/τ)
    num = math.exp(0.9 / tau)
    denom = num + math.exp(0.8 / tau) + math.exp(0.7 / tau)
    expected = -math.log(num / denom)
    assert abs(loss.item() - expected) < 1e-4, f"loss {loss.item()} vs expected {expected}"
    print(f"  OK loss={loss.item():.4f} matches analytic NT-Xent form (τ=0.07)")


def test_ntxent_n_per_query_truncation():
    """학술 Agent §4.3 N_i=8 — top-N hard negatives 만 사용."""
    print("\n[test_ntxent_n_per_query_truncation]")
    from modules.selectors.hn_supcon_selector import hn_supcon_loss

    # 10 hard negatives — top-3 만 사용해야 함
    sim_qp = torch.tensor([[0.9]])
    sim_qn = torch.tensor([[0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]])
    pos_mask = torch.tensor([[True]])
    hn_mask = torch.ones(1, 10, dtype=torch.bool)

    loss_top3 = hn_supcon_loss(sim_qp, sim_qn, pos_mask, hn_mask, tau=0.07, n_per_query=3)
    loss_top10 = hn_supcon_loss(sim_qp, sim_qn, pos_mask, hn_mask, tau=0.07, n_per_query=10)
    # top10 should have larger denominator (more negatives) → larger loss
    assert loss_top10 > loss_top3, f"top10 loss {loss_top10.item()} should > top3 {loss_top3.item()}"
    print(f"  OK n_per_query truncation: loss_top3={loss_top3.item():.4f} "
          f"< loss_top10={loss_top10.item():.4f} (denominator scaling)")


def test_hn_supcon_selector_inherits_ensemble():
    """HNSupConSelector 가 EnsembleSelector subclass — anchor logic (α blend, top_k, raw_*_scores) 유지."""
    print("\n[test_hn_supcon_selector_inherits_ensemble]")
    from modules.selectors.hn_supcon_selector import HNSupConSelector
    from modules.selectors.ensemble_selector import EnsembleSelector
    assert issubclass(HNSupConSelector, EnsembleSelector), \
        "HNSupConSelector must subclass EnsembleSelector"
    # Class-level signature: ctor 가 hn_supcon_ckpt_path 받음
    import inspect
    sig = inspect.signature(HNSupConSelector.__init__)
    assert "hn_supcon_ckpt_path" in sig.parameters, "HNSupConSelector ctor missing hn_supcon_ckpt_path"
    print(f"  OK HNSupConSelector inherits EnsembleSelector + accepts hn_supcon_ckpt_path")


def test_registry_registration():
    """`HNSupConSelector` 가 registry 에 등록됨 (config 에서 build() 가능)."""
    print("\n[test_registry_registration]")
    from modules.selectors import HNSupConSelector  # noqa: F401
    # Registry 등록 — @register("selector", "HNSupConSelector") 가 decorator 로 등록
    from modules.registry import REGISTRY  # 경로 확인
    sel_registry = REGISTRY.get("selector", {})
    assert "HNSupConSelector" in sel_registry, \
        f"HNSupConSelector not in selector registry: {list(sel_registry.keys())[:5]}..."
    print(f"  OK HNSupConSelector registered in selector registry")


def test_paper_hyperparams_defaults():
    """학술 Agent §4.3 default τ=0.07, N=8, margin=0.1, lr=5e-5, batch=16, 1 epoch."""
    print("\n[test_paper_hyperparams_defaults]")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_hn_supcon", str(ROOT.parent / "src/train_hn_supcon.py"))
    if spec is None or spec.loader is None:
        # ROOT here is .../selectors/tests/.. so adjust
        spec = importlib.util.spec_from_file_location(
            "train_hn_supcon", str(SRC / "train_hn_supcon.py"))
    src = (SRC / "train_hn_supcon.py").read_text()
    # check defaults via static src grep — argparse defaults
    assert 'default=5e-5' in src or 'default=5e-05' in src, "lr default mismatch"
    assert 'default=0.07' in src, "tau default mismatch"
    assert 'default=8' in src, "n_per_query default mismatch"
    assert 'default=0.1' in src, "margin default mismatch"
    assert 'default=16' in src, "batch_size default mismatch"
    assert 'default=1)' in src or '"--epochs", type=int, default=1' in src, "epochs default mismatch"
    assert '"sentence-transformers/all-MiniLM-L6-v2"' in src, "backbone default mismatch"
    print(f"  OK train_hn_supcon.py 의 argparse defaults: τ=0.07 N=8 margin=0.1 lr=5e-5 batch=16 1ep anchor backbone")


def main():
    test_margin_mask_form()
    test_margin_mask_multi_positive()
    test_margin_ablation_grid()
    test_ntxent_loss_form()
    test_ntxent_n_per_query_truncation()
    test_hn_supcon_selector_inherits_ensemble()
    test_registry_registration()
    test_paper_hyperparams_defaults()
    print("\nAll HN-SupCon Direction B smoke tests passed.")


if __name__ == "__main__":
    main()
