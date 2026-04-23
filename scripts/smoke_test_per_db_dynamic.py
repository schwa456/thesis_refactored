"""Smoke test — Proposal C H2 infrastructure (per-DB dynamic num_layers, inference-only).

목적:
  1. `SchemaHeteroGAT.forward(..., active_num_layers=L')` early-exit 이 다른 L' 에서
     (a) crash 없이 동작하고 (b) L' 에 따라 출력이 달라지는지 확인.
  2. `EnsembleSelector._resolve_active_depth` 가 diameter dict 와 mode 조합을 정확히
     해석하는지 확인 (체크포인트 로드 없는 단위 테스트).
  3. `data/processed/dev_diameter.pt` 의 11 개 DB 분포를 로그로 출력.

실행:
  conda run -n base python scripts/smoke_test_per_db_dynamic.py

본 스크립트는 체크포인트를 불러오지 않으며, 데이터셋 IO 없이 합성 HeteroData 로 GAT
forward 만 검증한다. Wave 2 Phase 1 L=6/L=7 ckpt 실측은 별도 inference experiment 에서.
"""
from __future__ import annotations

import os
import sys
import statistics
from typing import Dict

import torch
from torch_geometric.data import HeteroData

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from models.gat_network import SchemaHeteroGAT  # noqa: E402


DIAM_PATH = os.path.join(_ROOT, "data/processed/dev_diameter.pt")


def _banner(msg: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n {msg}\n{line}")


def build_dummy_hetero(num_t: int = 6, num_c: int = 30, num_fk: int = 5,
                      in_dim: int = 384) -> HeteroData:
    """Construct a minimal HeteroData with required edge types."""
    torch.manual_seed(0)
    data = HeteroData()
    data["table"].x = torch.randn(num_t, in_dim)
    data["column"].x = torch.randn(num_c, in_dim)
    data["fk_node"].x = torch.randn(num_fk, in_dim)

    # table -> column (each column belongs to a table)
    col_to_tbl = torch.randint(0, num_t, (num_c,))
    has_col = torch.stack([col_to_tbl, torch.arange(num_c)], dim=0)
    belongs = torch.stack([torch.arange(num_c), col_to_tbl], dim=0)
    data["table", "has_column", "column"].edge_index = has_col
    data["column", "belongs_to", "table"].edge_index = belongs

    # column <-> fk_node
    fk_src_col = torch.randint(0, num_c, (num_fk,))
    fk_dst_col = torch.randint(0, num_c, (num_fk,))
    data["column", "is_source_of", "fk_node"].edge_index = \
        torch.stack([fk_src_col, torch.arange(num_fk)], dim=0)
    data["fk_node", "points_to", "column"].edge_index = \
        torch.stack([torch.arange(num_fk), fk_dst_col], dim=0)

    # table <-> table
    t2t_src = torch.randint(0, num_t, (num_t * 2,))
    t2t_dst = torch.randint(0, num_t, (num_t * 2,))
    data["table", "table_to_table", "table"].edge_index = \
        torch.stack([t2t_src, t2t_dst], dim=0)
    return data


def test_diameter_cache_distribution() -> Dict[str, int]:
    _banner("1) dev_diameter.pt distribution")
    assert os.path.exists(DIAM_PATH), f"diameter cache not found: {DIAM_PATH}"
    d = torch.load(DIAM_PATH, map_location="cpu")
    if isinstance(d, dict) and "diameters" in d:
        d = d["diameters"]
    assert isinstance(d, dict) and all(isinstance(v, int) for v in d.values())
    values = sorted(d.values())
    p95 = values[max(0, int(len(values) * 0.95) - 1)]
    print(f"  #DB        : {len(d)}")
    print(f"  min        : {min(values)}")
    print(f"  median     : {int(statistics.median(values))}")
    print(f"  max        : {max(values)}")
    print(f"  p95        : {p95}")
    extreme = sorted([(k, v) for k, v in d.items() if v == max(values)])
    minimal = sorted([(k, v) for k, v in d.items() if v == min(values)])
    print(f"  D_max DBs  : {[k for k, _ in extreme]}")
    print(f"  D_min DBs  : {[k for k, _ in minimal]}")
    return d


def test_active_num_layers_early_exit() -> None:
    _banner("2) SchemaHeteroGAT.forward(active_num_layers=L') early-exit")
    model = SchemaHeteroGAT(
        in_channels=384, hidden_channels=64, out_channels=64,
        num_layers=6, heads=2,
    )
    model.eval()
    data = build_dummy_hetero()

    # Initialize lazy parameters with one full-depth pass.
    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)

    outs: Dict[str, torch.Tensor] = {}
    for L in [None, 1, 3, 6, 10]:
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict, active_num_layers=L)
        key = "full" if L is None else f"L={L}"
        table_emb = out["table"]
        assert table_emb.shape == (data["table"].x.size(0), 64), \
            f"output shape mismatch for {key}: {table_emb.shape}"
        outs[key] = table_emb
        print(f"  active_num_layers={str(L):>5} → table[0,:4]={table_emb[0, :4].tolist()}")

    diff_1_vs_3 = (outs["L=1"] - outs["L=3"]).abs().mean().item()
    diff_3_vs_6 = (outs["L=3"] - outs["L=6"]).abs().mean().item()
    diff_6_vs_full = (outs["L=6"] - outs["full"]).abs().mean().item()
    diff_6_vs_10 = (outs["L=6"] - outs["L=10"]).abs().mean().item()
    print(f"  |L1 - L3| mean      = {diff_1_vs_3:.6f}")
    print(f"  |L3 - L6| mean      = {diff_3_vs_6:.6f}")
    print(f"  |L6 - full| mean    = {diff_6_vs_full:.6e}  (expect ~0)")
    print(f"  |L6 - L10| mean     = {diff_6_vs_10:.6e}   (expect ~0, clamped to num_layers=6)")

    assert diff_1_vs_3 > 1e-6, "L=1 and L=3 outputs should differ"
    assert diff_6_vs_full < 1e-6, "L=6 should equal full (num_layers=6)"
    assert diff_6_vs_10 < 1e-6, "L=10 must clamp to num_layers=6"
    print("  ✓ early-exit works, clamp works")


def test_resolve_depth_logic(diameter_dict: Dict[str, int]) -> None:
    """_resolve_active_depth 로직만 단위 테스트 (체크포인트 로드 없이)."""
    _banner("3) _resolve_active_depth policy table")

    class _Stub:
        def __init__(self, mode: str, fallback: int = 3, max_depth: int = 6,
                     diam: Dict[str, int] = None):
            from modules.selectors.ensemble_selector import EnsembleSelector
            self.num_layers_mode = mode
            self.num_layers_fallback = fallback
            self.diameter_dict = {k: int(v) for k, v in (diam or {}).items()}

            class _G:
                pass
            self.gat_model = _G()
            self.gat_model.num_layers = max_depth
            self._resolve_active_depth = EnsembleSelector._resolve_active_depth.__get__(self)

    rows = []
    for db, dm in sorted(diameter_dict.items()):
        stub_dm = _Stub("D_max", fallback=3, max_depth=6, diam=diameter_dict)
        stub_dmp1 = _Stub("D_max_plus1", fallback=3, max_depth=6, diam=diameter_dict)
        stub_fixed = _Stub("fixed", fallback=3, max_depth=6, diam=diameter_dict)
        d_fixed = stub_fixed._resolve_active_depth({"db_id": db})
        d_dmax = stub_dm._resolve_active_depth({"db_id": db})
        d_dmp1 = stub_dmp1._resolve_active_depth({"db_id": db})
        rows.append((db, dm, d_fixed, d_dmax, d_dmp1))

    print(f"  {'db':<26} {'D_max':>6} {'fixed':>6} {'D_max':>6} {'D_max+1':>8}")
    for r in rows:
        db, dm, df, dd, dp = r
        print(f"  {db:<26} {dm:>6} {str(df):>6} {dd:>6} {dp:>8}")

    for db, dm, _, dd, dp in rows:
        assert dd == min(dm, 6)
        assert dp == min(dm + 1, 6)

    stub_miss = _Stub("D_max", fallback=3, max_depth=6, diam=diameter_dict)
    assert stub_miss._resolve_active_depth({"db_id": "unknown_db"}) == 3
    assert stub_miss._resolve_active_depth({}) == 3
    assert stub_miss._resolve_active_depth(None) == 3
    assert stub_fixed._resolve_active_depth({"db_id": "european_football_2"}) is None
    print("  ✓ D_max / D_max_plus1 / fixed / missing-db / None metadata all handled")


def main() -> int:
    diam = test_diameter_cache_distribution()
    test_active_num_layers_early_exit()
    test_resolve_depth_logic(diam)
    _banner("ALL SMOKE TESTS PASSED")
    print("Next: run an actual end-to-end with Wave 2 Phase 1 L=6/L=7 ckpt + "
          "num_layers_mode=D_max (no retraining needed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
