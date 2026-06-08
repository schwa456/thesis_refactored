#!/usr/bin/env python3
"""V6-W3 Phase 3 — L1 mechanism gate: hi-deg/lo-deg intra-table MAD L3 측정.

V6-W3 3 cells (A VirtualSummary / B ColumnPooling / C HubLocalVN) checkpoint forward +
layer-3 column embedding capture → intra-table MAD hi-deg(>30col)/lo-deg split (per-DB).

baseline (Phase 0, v6_phase0_diagnostics_2026-06-01.md §5.2):
  M4 anchor intra-MAD L3: overall 0.0355, hi-deg 0.0046, lo-deg 0.0407 (hub severity 0.113)
  european_football_2 hi-deg L3 = 0.0000 (115-col complete collapse) — natural test bed

게이트 통과: V6-W3 hi-deg intra-MAD L3 가 0.0046 → lo-deg 수준 (~0.04) 회복 방향.

reuse: v1_v5_retrospective_v6_metrics (model load + layer hook + col-table index)
GPU: CUDA_VISIBLE_DEVICES=0,1 (memory rule)
"""
import os, sys, json, time, argparse
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F

ROOT = Path("/home/hyeonjin/thesis_refactored")
SRC = ROOT / "src"
for p in (str(SRC), str(SRC / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import v1_v5_retrospective_v6_metrics as R  # reuse infra
from v1_v5_retrospective_v6_metrics import (
    load_model_for_cell, extract_layerwise_via_hook, load_dev,
    _build_col_table_index, DEVICE, DEV_DB_DIR, TABLES_JSON, ENCODER_MODEL_NAME,
    COL_TO_TAB_EDGE,
)
from modules.encoders.local_encoder import LocalPLMEncoder
from modules.builders.v6w3_builders import (
    V6W3VirtualSummaryBuilder, V6W3ColumnPoolingBuilder, V6W3HubLocalVNBuilder,
)

# V6-W3 model 은 v6w3_variant 로 추가 node/edge type (table_summary 등) 등록 → whitelist 확장
R.V2_KW_WHITELIST |= {"v6w3_variant", "capture_layerwise_outputs"}

OUT_DIR = ROOT / "outputs/analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HI_THR = 30

# V6-W3 cells: model_class v2 (standard DirectGATv2), builder per variant
CELLS = [
    dict(tag="v6w3_a", label="VirtualSummary (table_summary)", variant="A",
         ckpt_path="outputs/checkpoints/best_gat_v6w3_a_s11.pt", model_class="v2",
         builder_cls=V6W3VirtualSummaryBuilder, builder_kwargs={}, r15=0.5672),
    dict(tag="v6w3_b", label="ColumnPooling (uniform)", variant="B",
         ckpt_path="outputs/checkpoints/best_gat_v6w3_b_s11.pt", model_class="v2",
         builder_cls=V6W3ColumnPoolingBuilder, builder_kwargs=dict(pool_mode="uniform"), r15=0.5637),
    dict(tag="v6w3_c", label="HubLocalVN (median)", variant="C",
         ckpt_path="outputs/checkpoints/best_gat_v6w3_c_s11.pt", model_class="v2",
         builder_cls=V6W3HubLocalVNBuilder, builder_kwargs=dict(hub_strategy="median", hub_min_columns=0), r15=0.5633),
]
# baseline = M4 anchor, Phase 0 stored (outputs/v6_w0/diagnostics_s0_reference/per_query.jsonl) L3
# 검증: overall lo-deg 0.0407 / hi-deg 0.0046 = §5.2 정합. per-DB hi-deg from Phase0 stored.
BASELINE = dict(overall=0.0355, hi_deg=0.0046, lo_deg=0.0407,
                per_db_hi={"european_football_2": 0.0000, "thrombosis_prediction": 0.0000,
                           "card_games": 0.0082, "california_schools": 0.0117},
                per_db_lo={"european_football_2": 0.0100})


def intra_mad_hilo(col_emb, cb_edge, hi_thr=HI_THR):
    """per-table intra-column MAD, hi-deg(>hi_thr)/lo-deg split.
    Returns dict with intra_all/hi/lo means + pair counts + per-table list."""
    _, table_to_cols = _build_col_table_index(cb_edge)
    normed = F.normalize(col_emb.float(), dim=-1)
    hi_d, lo_d, all_d = [], [], []
    n_hi_tab = n_lo_tab = 0
    for t, cols in table_to_cols.items():
        if len(cols) < 2:
            continue
        ishi = len(cols) > hi_thr
        emb = normed[torch.tensor(cols, dtype=torch.long)]
        sim = emb @ emb.T
        n = len(cols)
        iu = torch.triu_indices(n, n, offset=1)
        d = (1.0 - sim)[iu[0], iu[1]]
        all_d.append(d)
        if ishi:
            hi_d.append(d); n_hi_tab += 1
        else:
            lo_d.append(d); n_lo_tab += 1

    def m(lst):
        if not lst:
            return None, 0
        cat = torch.cat(lst)
        return float(cat.mean().item()), int(cat.numel())
    a, na = m(all_d); h, nh = m(hi_d); lo, nl = m(lo_d)
    return dict(intra_all=a, intra_hi=h, intra_lo=lo,
                n_all_pairs=na, n_hi_pairs=nh, n_lo_pairs=nl,
                n_hi_tab=n_hi_tab, n_lo_tab=n_lo_tab)


def analyze_cell(cell, dev, qids, encoder, db_cache):
    print("=" * 80)
    print(f"[{cell['tag']}] {cell['label']} (variant {cell['variant']}, R@15={cell['r15']:.4f})")
    model = load_model_for_cell(cell)
    n_layers = len(model.convs)
    L3_idx = n_layers  # layer_embs = [L0_PLM, L1, L2, L3, L_out]; L3_GAT at index n_layers (=3)
    print(f"  convs={n_layers}, capturing L{n_layers}_GAT (index {L3_idx})")

    builder = cell["builder_cls"](
        plm_model_name=ENCODER_MODEL_NAME, tables_json_path=str(TABLES_JSON),
        **cell["builder_kwargs"],
    )
    # per-query L3 intra-MAD, grouped by db
    by_db = defaultdict(lambda: dict(hi=[], lo=[], all=[]))
    overall = dict(hi=[], lo=[], all=[])
    n_skip = 0
    t0 = time.time()
    for i, qid in enumerate(qids):
        item = dev[qid]; db_id = item["db_id"]
        if db_id not in db_cache:
            try:
                gd, md = builder.build(db_id=db_id, db_dir=str(DEV_DB_DIR))
                db_cache[db_id] = (gd, md)
            except Exception as e:
                print(f"  [qid={qid}] build fail: {e}"); n_skip += 1; continue
        gd, md = db_cache[db_id]
        data = gd.clone()
        try:
            enc = encoder.encode([item["question"]])
            q_emb = enc[0] if isinstance(enc, tuple) else enc
            if q_emb.dim() == 3:
                q_emb = q_emb.mean(dim=1)
            if q_emb.dim() == 1:
                q_emb = q_emb.unsqueeze(0)
            q_emb = q_emb.to(DEVICE)
            x_dict = {nt: x.to(DEVICE) for nt, x in data.x_dict.items()}
            eidx = {et: ei.to(DEVICE) for et, ei in data.edge_index_dict.items()}
            cb_edge = eidx.get(COL_TO_TAB_EDGE)
            if cb_edge is None:
                n_skip += 1; continue
            cb_cpu = cb_edge.cpu()
            layer_embs, n_active = extract_layerwise_via_hook(model, x_dict, eidx, query_emb=q_emb)
        except Exception as e:
            print(f"  [qid={qid}] forward fail: {e}")
            import traceback; traceback.print_exc(); n_skip += 1; continue
        # L3 = index min(L3_idx, len-2) — the last GAT layer before L_out
        idx = len(layer_embs) - 2  # L_out is last; last GAT layer is -2
        col_emb = layer_embs[idx].get("column")
        if col_emb is None:
            n_skip += 1; continue
        r = intra_mad_hilo(col_emb, cb_cpu)
        if r["intra_hi"] is not None:
            by_db[db_id]["hi"].append(r["intra_hi"]); overall["hi"].append(r["intra_hi"])
        if r["intra_lo"] is not None:
            by_db[db_id]["lo"].append(r["intra_lo"]); overall["lo"].append(r["intra_lo"])
        if r["intra_all"] is not None:
            by_db[db_id]["all"].append(r["intra_all"]); overall["all"].append(r["intra_all"])
        if (i + 1) % 300 == 0:
            print(f"    {i+1}/{len(qids)} ({time.time()-t0:.0f}s)")

    def mean(lst):
        return sum(lst) / len(lst) if lst else None
    res = dict(tag=cell["tag"], variant=cell["variant"], r15=cell["r15"], n_skip=n_skip,
               intra_hi_L3=mean(overall["hi"]), intra_lo_L3=mean(overall["lo"]),
               intra_all_L3=mean(overall["all"]),
               per_db={db: dict(hi=mean(v["hi"]), lo=mean(v["lo"]), all=mean(v["all"]),
                                n=len(v["all"])) for db, v in by_db.items()})
    hi = res["intra_hi_L3"]; lo = res["intra_lo_L3"]
    res["hub_severity_L3"] = (hi / lo) if (hi is not None and lo) else None
    print(f"  → intra-MAD L3: hi-deg={hi:.4f} lo-deg={lo:.4f} all={res['intra_all_L3']:.4f} "
          f"hub_sev={res['hub_severity_L3']:.4f} (skip={n_skip}, {time.time()-t0:.0f}s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0=all dev queries; else first N")
    args = ap.parse_args()
    print(f"Device: {DEVICE} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    dev = load_dev()
    qids = list(range(len(dev)))
    if args.limit > 0:
        qids = qids[:args.limit]
    print(f"Queries: {len(qids)}")
    encoder = LocalPLMEncoder(model_name=ENCODER_MODEL_NAME)
    results = []
    for cell in CELLS:
        db_cache = {}  # per-cell (builder differs)
        results.append(analyze_cell(cell, dev, qids, encoder, db_cache))

    out = dict(baseline=BASELINE, hi_thr=HI_THR, n_queries=len(qids), cells=results)
    suffix = f"_limit{args.limit}" if args.limit else ""
    jp = OUT_DIR / f"v6_phase3_l1_hub_mad_2026-06-06{suffix}.json"
    with jp.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== SUMMARY (baseline hi-deg L3=0.0046, lo-deg=0.0407) ===")
    print(f"{'cell':10s} {'hi-deg L3':>10} {'lo-deg L3':>10} {'all L3':>9} {'hub_sev':>8} {'Δhi vs base':>11}")
    for r in results:
        print(f"{r['tag']:10s} {r['intra_hi_L3']:>10.4f} {r['intra_lo_L3']:>10.4f} {r['intra_all_L3']:>9.4f} "
              f"{r['hub_severity_L3']:>8.4f} {r['intra_hi_L3']-0.0046:>+11.4f}")
    print(f"\n  european_football_2 hi-deg L3 (baseline 0.0000):")
    for r in results:
        ef = r["per_db"].get("european_football_2", {})
        print(f"    {r['tag']:10s} hi={ef.get('hi')} lo={ef.get('lo')} (n={ef.get('n')})")
    print(f"\nWrote {jp}")


if __name__ == "__main__":
    main()
