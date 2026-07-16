"""
Stage 0 — 훈련 신호 검증: gradient smoke test (읽기 전용, 원본 미수정).

목적: z_q, z_n 이 BCE/InfoNCE 로 backward 될 때 gradient 가
      (a) HGAT(SchemaHeteroGAT) 파라미터,
      (b) DualTowerProjector(text/graph tower) 파라미터
      각각에 실제로 흐르는지 확인한다.

실행: PYTHONPATH=src python review_verification/stage0_grad_smoke.py
CPU 로 동작 (합성 그래프 1개).
"""
import os
import sys
import json

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

SEED = 42
torch.manual_seed(SEED)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.gat_network import SchemaHeteroGAT           # noqa: E402
from modules.projectors.dual_tower import DualTowerProjector  # noqa: E402
# 학습 스크립트의 실제 손실 함수를 그대로 import (재구현 아님)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "train_gat_mod", os.path.join(os.path.dirname(__file__), "..", "src", "train_gat.py"))
# train_gat.py 는 import 시 부작용(wandb 등)이 없다 — 상단은 순수 def. 하지만
# 안전하게 함수만 뽑기 위해 소스에서 compute_batched_infonce_loss 를 직접 재사용한다.
# (train_gat.py 를 통째 exec 하면 dotenv/wandb import 만 실행되고 run_train 은 호출 안 됨)
import train_gat  # noqa: E402
compute_batched_infonce_loss = train_gat.compute_batched_infonce_loss

# ----------------------------------------------------------------
# 1. 합성 이종 그래프 1개 (query_conditioned: node feat = 384 + query 384 = 768)
# ----------------------------------------------------------------
IN = 384
HID = 256
OUT = 256

N_TAB, N_COL, N_FK = 3, 6, 2

def make_graph():
    d = HeteroData()
    # query_conditioned 학습에서 gat_model 에 넘어가는 augmented_x 는 이미 768 차원
    d['table'].x = torch.randn(N_TAB, IN * 2)
    d['column'].x = torch.randn(N_COL, IN * 2)
    d['fk_node'].x = torch.randn(N_FK, IN * 2)
    # 라벨 (타입별 최소 1 pos / 1 neg — InfoNCE 유효 그래프 조건)
    d['table'].y = torch.tensor([1., 0., 0.])
    d['column'].y = torch.tensor([1., 1., 0., 0., 0., 0.])
    d['fk_node'].y = torch.tensor([1., 0.])
    d['table', 'has_column', 'column'].edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [0, 1, 2, 3, 4, 5]])
    d['column', 'belongs_to', 'table'].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 2]])
    d['column', 'is_source_of', 'fk_node'].edge_index = torch.tensor([[0, 3], [0, 1]])
    d['fk_node', 'points_to', 'column'].edge_index = torch.tensor([[0, 1], [1, 4]])
    d['table', 'table_to_table', 'table'].edge_index = torch.tensor([[0, 1], [1, 2]])
    return d

loader = DataLoader([make_graph()], batch_size=1)
batch = next(iter(loader))

# query 임베딩 (frozen PLM E_q 를 흉내 — 384 차원, 그래프 1개)
q_emb = torch.randn(1, IN)

device = torch.device("cpu")

# ----------------------------------------------------------------
# 2. 모델 (학습 스크립트와 동일 하이퍼파라미터)
# ----------------------------------------------------------------
gat = SchemaHeteroGAT(in_channels=IN, hidden_channels=HID, out_channels=OUT,
                      num_layers=3, heads=4, query_conditioned=True).to(device)
projector = DualTowerProjector(text_dim=IN, graph_dim=HID, joint_dim=HID).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100.0]))
infonce_lambda = 0.5
temperature = 0.07
num_hard_negatives = 15

gat.train(); projector.train()

# ----------------------------------------------------------------
# 3. Forward — 학습 루프(train_gat.py 359-407)와 동일 경로
# ----------------------------------------------------------------
# query_conditioned: augmented_x 는 이미 768 차원이므로 query_emb 없이 호출
node_embs = gat(batch.x_dict, batch.edge_index_dict)

total_loss = 0.0
for n_type in ['table', 'column', 'fk_node']:
    z_q, z_n = projector(q_emb, node_embs[n_type], batch_index=batch[n_type].batch)
    logits = projector.compute_similarity(z_q, z_n)
    bce = criterion(logits, batch[n_type].y)
    infonce = compute_batched_infonce_loss(
        z_q, z_n, batch[n_type].y, batch[n_type].batch,
        temperature=temperature, num_hard_negatives=num_hard_negatives)
    total_loss = total_loss + bce + infonce_lambda * infonce

print(f"[forward] total_loss = {float(total_loss):.6f}")

# requires_grad 계보 확인
print(f"[lineage] node_embs['column'].requires_grad = {node_embs['column'].requires_grad}")
print(f"[lineage] z_n (graph tower out).requires_grad tested via total_loss.backward()")

# ----------------------------------------------------------------
# 4. Backward + grad norm 집계
# ----------------------------------------------------------------
total_loss.backward()

def grad_report(named_params, tag):
    rows = []
    flowed = 0
    total = 0
    for n, p in named_params:
        if not p.requires_grad:
            continue
        total += 1
        g = 0.0 if p.grad is None else float(p.grad.norm().item())
        ok = g > 0.0
        flowed += int(ok)
        rows.append((n, g, "흐름O" if ok else "흐름X"))
    print(f"\n=== {tag}: {flowed}/{total} 파라미터에 grad 흐름 ===")
    for n, g, verdict in rows:
        print(f"  {verdict}  grad_norm={g:.6e}  {n}")
    return flowed, total, rows

gat_flow, gat_total, gat_rows = grad_report(gat.named_parameters(), "HGAT (SchemaHeteroGAT)")
proj_flow, proj_total, proj_rows = grad_report(projector.named_parameters(), "DualTowerProjector")

# ----------------------------------------------------------------
# 5. 요약 판정 + JSON 덤프 (리포트 재사용)
# ----------------------------------------------------------------
summary = {
    "seed": SEED,
    "total_loss": float(total_loss),
    "node_embs_requires_grad": bool(node_embs['column'].requires_grad),
    "gat": {"flowed": gat_flow, "total": gat_total,
            "all_flowed": gat_flow == gat_total},
    "projector": {"flowed": proj_flow, "total": proj_total,
                  "all_flowed": proj_flow == proj_total},
    "gat_grad_norms": {n: g for n, g, _ in gat_rows},
    "projector_grad_norms": {n: g for n, g, _ in proj_rows},
    "logit_scale_grad": next((g for n, g, _ in proj_rows if "logit_scale" in n), None),
}
out_path = os.path.join(os.path.dirname(__file__), "stage0_grad_smoke_result.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n================ VERDICT ================")
print(f"HGAT gradient: {gat_flow}/{gat_total} 파라미터 흐름 "
      f"→ {'전부 흐름 (HGAT 학습됨)' if gat_flow == gat_total else '일부/전부 미흐름 (재확인 필요)'}")
print(f"Projector gradient: {proj_flow}/{proj_total} 파라미터 흐름")
print(f"결과 JSON: {out_path}")
