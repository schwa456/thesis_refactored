"""Direction B HN-SupCon — training entry (학술 Agent Phase 5 §4).

Piao 2025 LitE-SQL 의 Hard Negative Supervised Contrastive 목적함수를 anchor encoder
backbone (`sentence-transformers/all-MiniLM-L6-v2`) 위에 적용.

Hyperparameters (학술 Agent Phase 5 §4.3 원문 확정):
  - Temperature τ = 0.07
  - Negative count N_i = 8 (3~8 ablation, 8 최적)
  - Margin = 0.1 ({0, 0.1, 0.2} ablation, 0.1 최적)
  - Backbone: 현 anchor embedding (sentence-transformers/all-MiniLM-L6-v2) fine-tune
  - 1 epoch, AdamW, lr 5e-5, batch 16 (BIRD)

Smoke (학술 Agent Phase 5 §6.2):
  10% data (700 queries), 0.1 epoch → loss curve + val SLR Δ ≥ +1.0%p
  Pass 미달 시 lr 5e-5 → 1e-4

Fallback (학술 Agent Phase 5 §6.3):
  smoke fail → backbone 교체 (Qwen3-0.6B-Embedding)

Run from project root:
    conda run -n base python src/train_hn_supcon.py \\
        --train-json /SSL_NAS/peoples/khj/thesis/train/train.json \\
        --train-tables /SSL_NAS/peoples/khj/thesis/train/train_tables.json \\
        --output-dir outputs/checkpoints/hn_supcon \\
        --backbone sentence-transformers/all-MiniLM-L6-v2 \\
        --epochs 1 --batch-size 16 --lr 5e-5 \\
        --tau 0.07 --n-per-query 8 --margin 0.1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modules.selectors.hn_supcon_selector import build_hard_negative_mask, hn_supcon_loss  # noqa: E402


def load_bird_train(
    train_json: Path,
    train_tables_json: Path,
    max_queries: Optional[int] = None,
) -> List[Dict]:
    """BIRD-Train 의 query/db_id/SQL → (question, gold_columns_normalized, schema_columns)."""
    with open(train_json) as f:
        train_data = json.load(f)
    with open(train_tables_json) as f:
        tables_data = json.load(f)

    db_schema: Dict[str, List[str]] = {}
    for entry in tables_data:
        db = entry["db_id"]
        cols = []
        for tbl_idx, col_name in entry.get("column_names_original", [])[1:]:
            tbl_name = entry["table_names_original"][tbl_idx].lower()
            cols.append(f"{tbl_name}.{col_name.lower()}")
        db_schema[db] = cols

    records = []
    for row in train_data:
        if "question" not in row or "db_id" not in row:
            continue
        db = row["db_id"]
        if db not in db_schema:
            continue
        sql = (row.get("SQL") or row.get("query") or "").lower()
        gold = []
        for col in db_schema[db]:
            col_name = col.split(".", 1)[1] if "." in col else col
            if col_name and col_name in sql:
                gold.append(col)
        if not gold:
            continue
        records.append({
            "question": row["question"],
            "db_id": db,
            "gold_cols": gold,
            "schema_cols": db_schema[db],
        })
        if max_queries is not None and len(records) >= max_queries:
            break
    return records


def precompute_hard_negatives(
    records: List[Dict],
    encoder,
    margin: float = 0.1,
    n_per_query: int = 8,
    device: str = "cuda",
) -> List[Dict]:
    """Static hard negative pre-computation — margin mask + top-N selection."""
    out = []
    for rec in records:
        q_emb = encoder.encode([rec["question"]], convert_to_tensor=True, device=device)
        col_embs = encoder.encode(rec["schema_cols"], convert_to_tensor=True, device=device)
        sim = F.cosine_similarity(q_emb, col_embs, dim=-1)
        gold_set = set(rec["gold_cols"])
        pos_indices = [i for i, c in enumerate(rec["schema_cols"]) if c in gold_set]
        neg_indices = [i for i, c in enumerate(rec["schema_cols"]) if c not in gold_set]
        if not pos_indices or not neg_indices:
            continue
        p_max = sim[pos_indices].max().item()
        hard_neg_indices = [i for i in neg_indices if sim[i].item() > p_max - margin]
        if not hard_neg_indices:
            continue
        if len(hard_neg_indices) > n_per_query:
            scored = [(i, sim[i].item()) for i in hard_neg_indices]
            scored.sort(key=lambda x: x[1], reverse=True)
            hard_neg_indices = [i for i, _ in scored[:n_per_query]]
        rec_out = dict(rec)
        rec_out["hard_negatives"] = [rec["schema_cols"][i] for i in hard_neg_indices]
        out.append(rec_out)
    return out


class HNSupConDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx) -> Dict:
        rec = self.records[idx]
        return {
            "question": rec["question"],
            "positives": rec["gold_cols"],
            "hard_negatives": rec["hard_negatives"],
        }


def collate_hn_supcon(batch: List[Dict]) -> Dict:
    return {
        "questions": [b["question"] for b in batch],
        "positives": [b["positives"] for b in batch],
        "hard_negatives": [b["hard_negatives"] for b in batch],
    }


def train_hn_supcon(args):
    from sentence_transformers import SentenceTransformer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[train_hn_supcon] device={device} backbone={args.backbone}")

    encoder = SentenceTransformer(args.backbone, device=device)
    encoder.train()
    for p in encoder.parameters():
        p.requires_grad = True

    print(f"[train_hn_supcon] loading BIRD-Train: {args.train_json}")
    records = load_bird_train(
        Path(args.train_json), Path(args.train_tables), max_queries=args.max_queries
    )
    print(f"[train_hn_supcon] loaded {len(records)} records")

    print(f"[train_hn_supcon] pre-computing hard negatives (margin={args.margin}, N={args.n_per_query})")
    encoder.eval()
    with torch.no_grad():
        records = precompute_hard_negatives(
            records, encoder, margin=args.margin, n_per_query=args.n_per_query, device=device
        )
    encoder.train()
    print(f"[train_hn_supcon] {len(records)} records after hard-negative filtering")
    if len(records) == 0:
        print("[ERROR] No records — abort", file=sys.stderr)
        return

    if args.epoch_fraction < 1.0:
        n_steps = int(len(records) * args.epoch_fraction)
        records = records[:n_steps]
        print(f"[train_hn_supcon] epoch_fraction={args.epoch_fraction} → {n_steps} records")

    ds = HNSupConDataset(records)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_hn_supcon)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=0.01)

    encoder.train()
    losses: List[float] = []
    global_step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            questions = batch["questions"]
            B = len(questions)
            P_max = max(len(p) for p in batch["positives"])
            N_pool = max(len(n) for n in batch["hard_negatives"])

            all_pos: List[str] = []
            pos_offsets: List[Tuple[int, int]] = []
            for ps in batch["positives"]:
                pos_offsets.append((len(all_pos), len(all_pos) + len(ps)))
                all_pos.extend(ps)
            all_neg: List[str] = []
            neg_offsets: List[Tuple[int, int]] = []
            for ns in batch["hard_negatives"]:
                neg_offsets.append((len(all_neg), len(all_neg) + len(ns)))
                all_neg.extend(ns)

            q_features = encoder.tokenize(questions)
            q_features = {k: v.to(device) for k, v in q_features.items()}
            q_out = encoder(q_features)["sentence_embedding"]
            q_norm = F.normalize(q_out, dim=-1)

            if not all_pos or not all_neg:
                continue
            p_features = encoder.tokenize(all_pos)
            p_features = {k: v.to(device) for k, v in p_features.items()}
            p_out = encoder(p_features)["sentence_embedding"]
            p_norm = F.normalize(p_out, dim=-1)

            n_features = encoder.tokenize(all_neg)
            n_features = {k: v.to(device) for k, v in n_features.items()}
            n_out = encoder(n_features)["sentence_embedding"]
            n_norm = F.normalize(n_out, dim=-1)

            sim_qp = q_norm.new_full((B, P_max), float("-inf"))
            pos_mask = torch.zeros(B, P_max, dtype=torch.bool, device=device)
            for i, (s, e) in enumerate(pos_offsets):
                if e > s:
                    sims = (q_norm[i:i + 1] @ p_norm[s:e].T).squeeze(0)
                    sim_qp[i, : e - s] = sims
                    pos_mask[i, : e - s] = True
            sim_qn = q_norm.new_full((B, N_pool), float("-inf"))
            for i, (s, e) in enumerate(neg_offsets):
                if e > s:
                    sims = (q_norm[i:i + 1] @ n_norm[s:e].T).squeeze(0)
                    sim_qn[i, : e - s] = sims

            hn_mask = build_hard_negative_mask(sim_qp, sim_qn, margin=args.margin)
            valid_neg = (sim_qn > float("-inf"))
            hn_mask = hn_mask & valid_neg

            loss = hn_supcon_loss(
                sim_qp=sim_qp, sim_qn=sim_qn,
                pos_mask=pos_mask, hard_neg_mask=hn_mask,
                tau=args.tau, n_per_query=args.n_per_query,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())
            global_step += 1
            if global_step % args.log_every == 0:
                avg = sum(losses[-args.log_every:]) / min(args.log_every, len(losses))
                print(f"[step {global_step:5d}] loss={loss.item():.4f} avg_recent={avg:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder.save(str(output_dir))
    with open(output_dir / "loss_curve.json", "w") as f:
        json.dump({"losses": losses, "config": vars(args)}, f, indent=2)
    final = losses[-1] if losses else None
    print(f"[train_hn_supcon] saved encoder to {output_dir} ({len(losses)} steps, final loss={final})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-json", required=True, type=str)
    ap.add_argument("--train-tables", required=True, type=str)
    ap.add_argument("--output-dir", required=True, type=str)
    ap.add_argument("--backbone", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--epoch-fraction", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--n-per-query", type=int, default=8)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train_hn_supcon(args)


if __name__ == "__main__":
    main()
