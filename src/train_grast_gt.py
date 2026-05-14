"""Direction C-GT (GraST-GT) — training entry (Hoang 2025 Option β).

학술 Agent Phase 5 §1.3 + Q5:
  - GraphTransformerEncoder (3-layer relation-aware GT, hidden 1024, 8 heads)
  - margin-based contrastive loss (margin=0.1)
  - smoke mode (--smoke): 5 epoch / loss<0.3 / PR-AUC Δ≥+0.01 (학술 Agent Q5)
  - full mode (default): max-epochs=40, AdamW lr=5e-5, batch=32

h^0 input (학술 Agent Q2 권장 — anchor scorer 출력 재활용):
  학습 wrapper 에서는 anchor LLM scorer 호출 없이 simplified h^0 사용
  (filter._build_h0 가 s_fwd_set=∅, gat_scores=None 시 fk/pk binary flag 만 활성)

Run from project root:
    conda run -n base python src/train_grast_gt.py \\
        --train-json /SSL_NAS/peoples/khj/thesis/train/train.json \\
        --train-tables /SSL_NAS/peoples/khj/thesis/train/train_tables.json \\
        --output-dir outputs/checkpoints/grast_gt \\
        --max-epochs 40 --margin 0.1 --lr 5e-5

Smoke:
    + --smoke --max-epochs 5 --epoch-fraction 0.125
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.filters.grast_fd_transformer import (  # noqa: E402
    GraphTransformerEncoder,
    NUM_EDGE_TYPES,
    EDGE_TYPE_FK_FORWARD, EDGE_TYPE_FK_REVERSE,
    EDGE_TYPE_COL_TO_FK_FORWARD, EDGE_TYPE_COL_TO_FK_REVERSE,
    EDGE_TYPE_COL_TO_PK_FORWARD, EDGE_TYPE_COL_TO_PK_REVERSE,
    margin_contrastive_loss,
    smoke_train_protocol,
)


# ──────────────────────────────────────────────────────────────
# Schema + metadata loader (train_tables.json)
# ──────────────────────────────────────────────────────────────

def build_db_schema_map(train_tables_path: Path) -> Tuple[Dict, Dict]:
    """{db_id: {table_lower: [col_lower, ...]}} + {db_id: metadata}.

    metadata 는 GRASTFDFilter._build_fd_graph 호환 형식:
      {fk_to_id: {"src.col->dst.col": 1, ...}, pk_set: Set[str], col_to_id: Dict}
    """
    with open(train_tables_path) as f:
        tables_data = json.load(f)
    db_schema_map: Dict[str, Dict[str, List[str]]] = {}
    db_metadata: Dict[str, Dict[str, Any]] = {}
    for entry in tables_data:
        db = entry["db_id"]
        table_names = entry["table_names_original"]
        full_schema: Dict[str, List[str]] = {t.lower(): [] for t in table_names}
        col_to_id: Dict[str, int] = {}
        for col_idx, (tbl_idx, col_name) in enumerate(entry["column_names_original"]):
            if tbl_idx < 0:
                continue
            t = table_names[tbl_idx].lower()
            c = col_name.lower()
            full_schema[t].append(c)
            col_to_id[f"{t}.{c}"] = col_idx
        # primary keys
        pk_set: Set[str] = set()
        for pk in entry.get("primary_keys", []) or []:
            keys = pk if isinstance(pk, list) else [pk]
            for pi in keys:
                if not isinstance(pi, int) or not (0 <= pi < len(entry["column_names_original"])):
                    continue
                tbl_idx, col_name = entry["column_names_original"][pi]
                if tbl_idx >= 0:
                    pk_set.add(f"{table_names[tbl_idx].lower()}.{col_name.lower()}")
        # foreign keys → "src.col->dst.col" form
        fk_to_id: Dict[str, int] = {}
        for fk_pair in entry.get("foreign_keys", []) or []:
            if not (isinstance(fk_pair, (list, tuple)) and len(fk_pair) == 2):
                continue
            src_idx, dst_idx = fk_pair
            if not (0 <= src_idx < len(entry["column_names_original"])
                    and 0 <= dst_idx < len(entry["column_names_original"])):
                continue
            src_t_idx, src_col = entry["column_names_original"][src_idx]
            dst_t_idx, dst_col = entry["column_names_original"][dst_idx]
            if src_t_idx < 0 or dst_t_idx < 0:
                continue
            fk_key = (
                f"{table_names[src_t_idx].lower()}.{src_col.lower()}"
                f"->{table_names[dst_t_idx].lower()}.{dst_col.lower()}"
            )
            fk_to_id[fk_key] = 1
        db_schema_map[db] = full_schema
        db_metadata[db] = {"fk_to_id": fk_to_id, "col_to_id": col_to_id, "pk_set": pk_set}
    return db_schema_map, db_metadata


def load_bird_train_records(
    train_json: Path,
    db_schema_map: Dict,
    max_queries: Optional[int],
) -> List[Dict]:
    with open(train_json) as f:
        train_data = json.load(f)
    records: List[Dict] = []
    for row in train_data:
        if "question" not in row or "db_id" not in row:
            continue
        db = row["db_id"]
        if db not in db_schema_map:
            continue
        sql = (row.get("SQL") or row.get("query") or "").lower()
        gold_cols: List[str] = []
        for tbl, cols in db_schema_map[db].items():
            for c in cols:
                # column-name substring 매칭 (train_hn_supcon 패턴)
                if c and c in sql:
                    gold_cols.append(f"{tbl}.{c}")
        if not gold_cols:
            continue
        records.append({"question": row["question"], "db_id": db, "gold_cols": gold_cols})
        if max_queries is not None and len(records) >= max_queries:
            break
    return records


# ──────────────────────────────────────────────────────────────
# Batch construction (inline copy from Module:Filter helpers — 학습 wrapper scope)
# ──────────────────────────────────────────────────────────────

def _parse_fk_key(fk_key: str) -> Optional[Tuple[str, str]]:
    if not isinstance(fk_key, str) or "->" not in fk_key:
        return None
    left, right = fk_key.split("->", 1)
    left, right = left.strip(), right.strip()
    if "." not in left or "." not in right:
        return None
    return left, right


def encode_graph_for_transformer(
    full_schema: Dict[str, List[str]],
    metadata: Dict[str, Any],
    fk_pk_columns: Set[str],
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """Inline copy of GRASTFDFilterWithTransformer._encode_graph_for_transformer."""
    col_nodes = [f"{t}.{c}" for t, cols in full_schema.items() for c in (cols or [])]
    node_idx = {n: i for i, n in enumerate(col_nodes)}
    fk_to_id = metadata.get("fk_to_id", {}) or {}

    src_list: List[int] = []
    dst_list: List[int] = []
    et_list: List[int] = []

    def _add(s: str, d: str, et_fwd: int, et_rev: int):
        if s not in node_idx or d not in node_idx:
            return
        i, j = node_idx[s], node_idx[d]
        src_list.append(i); dst_list.append(j); et_list.append(et_fwd)
        src_list.append(j); dst_list.append(i); et_list.append(et_rev)

    # FK edges (declared)
    for fk_key in list(fk_to_id.keys()):
        parsed = _parse_fk_key(fk_key)
        if parsed is None:
            continue
        s, d = parsed
        _add(s, d, EDGE_TYPE_FK_FORWARD, EDGE_TYPE_FK_REVERSE)

    # col → FK / col → PK
    fk_endpoints: Set[str] = set()
    for fk_key in list(fk_to_id.keys()):
        parsed = _parse_fk_key(fk_key)
        if parsed is None:
            continue
        fk_endpoints.update(parsed)

    for tbl, cols in full_schema.items():
        full_cols = [f"{tbl}.{c}" for c in (cols or [])]
        for src in full_cols:
            for dst in full_cols:
                if src == dst:
                    continue
                if dst in fk_endpoints:
                    _add(src, dst,
                         EDGE_TYPE_COL_TO_FK_FORWARD, EDGE_TYPE_COL_TO_FK_REVERSE)
                if dst in fk_pk_columns and dst not in fk_endpoints:
                    _add(src, dst,
                         EDGE_TYPE_COL_TO_PK_FORWARD, EDGE_TYPE_COL_TO_PK_REVERSE)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(et_list, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
    return col_nodes, edge_index, edge_type


def build_h0_simplified(
    col_nodes: List[str],
    fk_pk_columns: Set[str],
    in_dim: int = 16,
) -> torch.Tensor:
    """Simplified h^0 — fk/pk binary flag only (anchor scorer 호출 회피).

    feat[i, 0] = 0  (xiyan_selected — 학습 시 X)
    feat[i, 1] = 0  (gat_score — 학습 시 X)
    feat[i, 2] = is_fk_pk binary
    feat[i, 3..] = 0 padding
    """
    N = len(col_nodes)
    if N == 0:
        return torch.zeros((0, in_dim), dtype=torch.float32)
    feat = torch.zeros((N, in_dim), dtype=torch.float32)
    for i, node in enumerate(col_nodes):
        if node in fk_pk_columns:
            feat[i, 2] = 1.0
    return feat


def build_batch_from_record(
    record: Dict,
    db_schema_map: Dict,
    db_metadata: Dict,
    in_dim: int,
) -> Optional[Dict[str, torch.Tensor]]:
    db = record["db_id"]
    full_schema = db_schema_map[db]
    metadata = db_metadata[db]
    pk_set: Set[str] = metadata.get("pk_set", set())
    # fk endpoints from declared FK 도 fk/pk columns 에 포함
    fk_endpoints: Set[str] = set()
    for fk_key in metadata.get("fk_to_id", {}) or {}:
        parsed = _parse_fk_key(fk_key)
        if parsed:
            fk_endpoints.update(parsed)
    fk_pk_columns = pk_set | fk_endpoints

    col_nodes, edge_index, edge_type = encode_graph_for_transformer(
        full_schema=full_schema, metadata=metadata, fk_pk_columns=fk_pk_columns,
    )
    if len(col_nodes) == 0:
        return None

    h0 = build_h0_simplified(col_nodes, fk_pk_columns, in_dim=in_dim)
    gold_set = set(c.lower() for c in record["gold_cols"])
    gold_mask = torch.tensor([n.lower() in gold_set for n in col_nodes], dtype=torch.bool)
    if gold_mask.sum() == 0 or (~gold_mask).sum() == 0:
        return None
    return {"h0": h0, "edge_index": edge_index, "edge_type": edge_type, "gold_mask": gold_mask}


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-json", required=True, type=str)
    ap.add_argument("--train-tables", required=True, type=str)
    ap.add_argument("--output-dir", required=True, type=str)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--epoch-fraction", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--in-dim", type=int, default=16)
    ap.add_argument("--hidden-dim", type=int, default=1024)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="학술 Agent Q5 smoke mode (loss<0.3 + PR-AUC Δ≥+0.01)")
    ap.add_argument("--smoke-loss-threshold", type=float, default=0.3)
    ap.add_argument("--smoke-pr-auc-delta", type=float, default=0.01)
    ap.add_argument("--smoke-patience", type=int, default=2)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[train_grast_gt] device={device}")

    print(f"[train_grast_gt] loading data: {args.train_tables}")
    db_schema_map, db_metadata = build_db_schema_map(Path(args.train_tables))
    print(f"[train_grast_gt] loaded {len(db_schema_map)} DBs")

    print(f"[train_grast_gt] loading train records: {args.train_json}")
    records = load_bird_train_records(Path(args.train_json), db_schema_map, args.max_queries)
    print(f"[train_grast_gt] loaded {len(records)} records (gold non-empty)")

    if args.epoch_fraction < 1.0:
        n_use = max(1, int(len(records) * args.epoch_fraction))
        records = records[:n_use]
        print(f"[train_grast_gt] epoch_fraction={args.epoch_fraction} — using {n_use} records")

    print("[train_grast_gt] building batches...")
    batches: List[Dict[str, torch.Tensor]] = []
    skipped = 0
    for r in records:
        b = build_batch_from_record(r, db_schema_map, db_metadata, in_dim=args.in_dim)
        if b is None:
            skipped += 1
            continue
        b = {k: v.to(device) for k, v in b.items()}
        batches.append(b)
    print(f"[train_grast_gt] built {len(batches)} batches (skipped {skipped})")

    if not batches:
        print("[train_grast_gt] ❌ no usable batches — abort")
        sys.exit(1)

    n_val = max(1, int(len(batches) * args.val_fraction))
    val_batches = batches[:n_val]
    train_batches = batches[n_val:]
    print(f"[train_grast_gt] train={len(train_batches)} / val={len(val_batches)}")

    model = GraphTransformerEncoder(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_edge_types=NUM_EDGE_TYPES,
        dropout=args.dropout,
    ).to(device)
    print(f"[train_grast_gt] model: GraphTransformerEncoder "
          f"(in_dim={args.in_dim}, hidden={args.hidden_dim}, "
          f"L={args.num_layers}, H={args.num_heads})")

    if args.smoke:
        print(f"[train_grast_gt] smoke_train_protocol — "
              f"epochs={args.max_epochs} margin={args.margin} lr={args.lr}")
        result = smoke_train_protocol(
            model=model,
            batches=train_batches,
            val_batches=val_batches,
            num_epochs=args.max_epochs,
            lr=args.lr,
            margin=args.margin,
            pass_loss_threshold=args.smoke_loss_threshold,
            pass_pr_auc_delta=args.smoke_pr_auc_delta,
            plateau_patience=args.smoke_patience,
        )
        # Convert non-JSON-serializable (e.g. torch values)
        result_save = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in result.items()}
        # epoch_losses can be list of floats — keep
        print(f"[train_grast_gt] smoke result: {result_save}")
        with open(output_dir / "smoke_result.json", "w") as f:
            json.dump(result_save, f, indent=2, default=str)
    else:
        # Full training loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
        epoch_losses: List[float] = []
        for epoch in range(args.max_epochs):
            model.train()
            epoch_loss = 0.0
            n_step = 0
            for batch in train_batches:
                optimizer.zero_grad()
                _, scores = model(batch["h0"], batch["edge_index"], batch["edge_type"])
                loss = margin_contrastive_loss(scores, batch["gold_mask"], margin=args.margin)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[train_grast_gt] ⚠ NaN/Inf at epoch {epoch} — abort")
                    break
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                n_step += 1
            avg = epoch_loss / max(1, n_step)
            epoch_losses.append(avg)
            print(f"[train_grast_gt] epoch {epoch + 1}/{args.max_epochs} loss={avg:.4f}")
        with open(output_dir / "train_log.json", "w") as f:
            json.dump({"epoch_losses": epoch_losses, "config": vars(args)}, f, indent=2)

    ckpt_path = output_dir / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_dim": args.in_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "num_edge_types": NUM_EDGE_TYPES,
            "dropout": args.dropout,
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"[train_grast_gt] ✅ saved {ckpt_path}")


if __name__ == "__main__":
    main()
