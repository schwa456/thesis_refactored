"""B5 frozen L_out 위에 column classifier head 재학습.

Options:
  --head      : linear | mlp
  --loss      : bce | listnet
  --normalize : none | zscore  (per-query z-score normalization on L_out)

Matrix (head=mlp):
  Exp B: bce    + none
  Exp C: listnet + none
  Exp D: bce    + zscore
  Exp E: listnet + zscore
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from models.direct_classifier import DirectClassifierHead
from models.losses import ListNetLoss


class LinearHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


def calculate_recall_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int = 15) -> float:
    if labels.sum() == 0:
        return 0.0
    k_actual = min(k, logits.size(0))
    _, top_k = torch.topk(logits, k_actual)
    hits = labels[top_k].sum().item()
    return hits / labels.sum().item()


def normalize_records(records: list, method: str) -> list:
    """Per-query L_out normalization (on CPU — lazy transfer to device later)."""
    if method == "none":
        return records
    out = []
    for r in records:
        x = r["l_out"]
        if method == "zscore":
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True).clamp(min=1e-6)
            x_norm = (x - mean) / std
        else:
            raise ValueError(f"unknown normalize method: {method}")
        out.append({"query_idx": r["query_idx"], "l_out": x_norm, "y": r["y"]})
    return out


def evaluate_head(head: nn.Module, records: list, device: torch.device) -> dict:
    head.eval()
    logits_all, labels_all, recalls = [], [], []
    with torch.no_grad():
        for r in records:
            X = r["l_out"].to(device)
            y = r["y"]
            logits = head(X).cpu()
            logits_all.append(logits.numpy())
            labels_all.append(y.numpy().astype(int))
            if y.sum() > 0:
                recalls.append(calculate_recall_at_k(logits, y, k=15))
    logits_all = np.concatenate(logits_all)
    labels_all = np.concatenate(labels_all)
    auc = float(roc_auc_score(labels_all, logits_all)) if labels_all.sum() > 0 else 0.0
    return {"auc": auc, "recall@15": float(np.mean(recalls)) if recalls else 0.0,
            "n_queries": len(records), "n_with_gold": len(recalls),
            "n_nodes": int(len(labels_all)), "n_gold": int(labels_all.sum())}


def train_epoch_bce(head, X_tr, y_tr, opt, bce_fn) -> float:
    head.train()
    opt.zero_grad()
    logits = head(X_tr)
    loss = bce_fn(logits, y_tr)
    loss.backward()
    opt.step()
    return float(loss.item())


def train_epoch_listnet(head, tr_recs, opt, listnet_fn, device, batch_q: int = 32,
                        seed: int = 0) -> float:
    head.train()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tr_recs))
    total_loss = 0.0
    n_steps = 0
    for start in range(0, len(order), batch_q):
        opt.zero_grad()
        step_loss = None
        n_valid = 0
        for qi in order[start:start + batch_q]:
            r = tr_recs[qi]
            y = r["y"]
            if y.sum() <= 0:
                continue
            X = r["l_out"].to(device)
            y_d = y.to(device)
            logits = head(X)
            li = listnet_fn(logits, y_d)
            step_loss = li if step_loss is None else step_loss + li
            n_valid += 1
        if step_loss is None or n_valid == 0:
            continue
        step_loss = step_loss / n_valid
        step_loss.backward()
        opt.step()
        total_loss += float(step_loss.item())
        n_steps += 1
    return total_loss / max(1, n_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cache", type=str, required=True)
    parser.add_argument("--dev_cache", type=str, required=True)
    parser.add_argument("--head", choices=["linear", "mlp"], required=True)
    parser.add_argument("--loss", choices=["bce", "listnet"], default="bce")
    parser.add_argument("--normalize", choices=["none", "zscore"], default="none")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--pos_weight", type=float, default=100.0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--listnet_batch_q", type=int, default=32)
    parser.add_argument("--ldbo_frac", type=float, default=0.0,
                        help="if >0, split by DB instead of query (fraction of DBs to hold out)")
    parser.add_argument("--train_json", type=str, default=None,
                        help="BIRD train.json for db_id lookup (required for LDBO)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"[retrain] head={args.head}, loss={args.loss}, normalize={args.normalize}, "
          f"device={device}, seed={args.seed}")

    # -------- Load caches --------
    print("[retrain] loading caches ...")
    train_pkg = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    dev_pkg = torch.load(args.dev_cache, map_location="cpu", weights_only=False)
    train_records = train_pkg["records"]
    dev_records = dev_pkg["records"]
    in_dim = train_pkg["embedding_dim"]
    print(f"  train queries={len(train_records)}, dev queries={len(dev_records)}, dim={in_dim}")

    # -------- Normalize --------
    print(f"[retrain] normalizing (method={args.normalize}) ...")
    train_records = normalize_records(train_records, args.normalize)
    dev_records = normalize_records(dev_records, args.normalize)

    # -------- Split: query-random (default) OR LDBO (by DB) --------
    ldbo_info = None
    if args.ldbo_frac > 0:
        assert args.train_json is not None, "--train_json required when --ldbo_frac > 0"
        train_data = json.load(open(args.train_json))
        db_ids_per_record = [train_data[r["query_idx"]]["db_id"] for r in train_records]
        unique_dbs = sorted(set(db_ids_per_record))
        rng_ldbo = np.random.default_rng(args.seed)
        n_val_dbs = max(1, int(round(args.ldbo_frac * len(unique_dbs))))
        val_dbs = sorted(rng_ldbo.choice(unique_dbs, size=n_val_dbs, replace=False).tolist())
        val_dbs_set = set(val_dbs)
        tr_recs = [r for r, db in zip(train_records, db_ids_per_record) if db not in val_dbs_set]
        val_recs = [r for r, db in zip(train_records, db_ids_per_record) if db in val_dbs_set]
        ldbo_info = {"n_total_dbs": len(unique_dbs), "n_val_dbs": n_val_dbs,
                     "val_dbs": val_dbs}
        print(f"  [LDBO] {n_val_dbs}/{len(unique_dbs)} held-out DBs: {val_dbs}")
        print(f"  tr/val queries: {len(tr_recs)} / {len(val_recs)}")
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(train_records))
        n_val = int(args.val_split * len(train_records))
        val_q_idx = set(perm[:n_val].tolist())
        tr_recs = [r for i, r in enumerate(train_records) if i not in val_q_idx]
        val_recs = [r for i, r in enumerate(train_records) if i in val_q_idx]
        print(f"  tr/val queries: {len(tr_recs)} / {len(val_recs)}")

    # -------- Build head --------
    if args.head == "linear":
        head = LinearHead(in_dim).to(device)
    else:
        head = DirectClassifierHead(in_dim=in_dim, hidden_dim=256, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"  head params = {n_params}")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # -------- Loss setup --------
    if args.loss == "bce":
        # BCE: flatten training set for full-batch
        X_tr = torch.cat([r["l_out"] for r in tr_recs], dim=0).to(device)
        y_tr = torch.cat([r["y"] for r in tr_recs], dim=0).to(device)
        pos_weight = torch.tensor([args.pos_weight], device=device)
        bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"  train nodes={X_tr.size(0)}, gold={int(y_tr.sum().item())} "
              f"({100*y_tr.mean().item():.2f}%)")
    else:
        listnet_fn = ListNetLoss().to(device)

    history = []
    best_val_r15 = -1.0
    best_epoch = -1
    best_state = None

    print("\n[retrain] ===== training =====")
    for epoch in range(1, args.epochs + 1):
        if args.loss == "bce":
            loss_val = train_epoch_bce(head, X_tr, y_tr, opt, bce_fn)
        else:
            loss_val = train_epoch_listnet(head, tr_recs, opt, listnet_fn, device,
                                            batch_q=args.listnet_batch_q,
                                            seed=args.seed + epoch)

        val_m = evaluate_head(head, val_recs, device)
        dev_m = evaluate_head(head, dev_records, device)
        history.append({
            "epoch": epoch,
            "loss": float(loss_val),
            "val_auc": val_m["auc"], "val_r15": val_m["recall@15"],
            "dev_auc": dev_m["auc"], "dev_r15": dev_m["recall@15"],
        })

        marker = ""
        if val_m["recall@15"] > best_val_r15:
            best_val_r15 = val_m["recall@15"]
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            marker = "  ★"

        if epoch == 1 or epoch % 2 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d} | Loss {loss_val:.4f} | "
                  f"Val AUC {val_m['auc']:.4f} R@15 {val_m['recall@15']:.4f} | "
                  f"Dev AUC {dev_m['auc']:.4f} R@15 {dev_m['recall@15']:.4f}{marker}")

    # -------- Load best & final eval --------
    if best_state is not None:
        head.load_state_dict(best_state)
    final_val = evaluate_head(head, val_recs, device)
    final_dev = evaluate_head(head, dev_records, device)

    # -------- Also find best-dev-R15 epoch (for proper early stopping analysis) --------
    best_dev_r15 = max(h["dev_r15"] for h in history)
    best_dev_auc = max(h["dev_auc"] for h in history)
    best_dev_r15_epoch = [h["epoch"] for h in history if h["dev_r15"] == best_dev_r15][0]
    best_dev_auc_epoch = [h["epoch"] for h in history if h["dev_auc"] == best_dev_auc][0]

    summary = {
        "head": args.head, "loss": args.loss, "normalize": args.normalize,
        "n_params": int(n_params),
        "epochs": args.epochs, "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pos_weight": args.pos_weight,
        "val_split": args.val_split, "seed": args.seed,
        "listnet_batch_q": args.listnet_batch_q if args.loss == "listnet" else None,
        "best_epoch": int(best_epoch),
        "best_val_r15": float(best_val_r15),
        "best_dev_r15": float(best_dev_r15),
        "best_dev_r15_epoch": int(best_dev_r15_epoch),
        "best_dev_auc": float(best_dev_auc),
        "best_dev_auc_epoch": int(best_dev_auc_epoch),
        "final_val": final_val,
        "final_dev": final_dev,
        "history": history,
        "ldbo": ldbo_info,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(head.state_dict(), out_dir / "head.pt")

    print("\n[retrain] ===== done =====")
    print(f"  Best (val-ES) epoch={best_epoch}, Val R@15={best_val_r15:.4f}")
    print(f"  Best (dev-ES) : Dev R@15={best_dev_r15:.4f} @ epoch {best_dev_r15_epoch}, "
          f"Dev AUC={best_dev_auc:.4f} @ epoch {best_dev_auc_epoch}")
    print(f"  Final VAL: AUC={final_val['auc']:.4f}  R@15={final_val['recall@15']:.4f}")
    print(f"  Final DEV: AUC={final_dev['auc']:.4f}  R@15={final_dev['recall@15']:.4f}")


if __name__ == "__main__":
    main()
