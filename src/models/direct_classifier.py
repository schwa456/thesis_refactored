import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectClassifierHead(nn.Module):
    """
    Query-Conditioned GAT 출력 노드 임베딩을 직접 binary logit으로 매핑하는 MLP.

    DualTowerProjector와 달리, query 임베딩을 재입력하지 않는다.
    GAT 내부에서 query-aware attention이 이미 일어났다고 가정하고,
    각 노드 임베딩의 '정답 여부' 확률만을 예측한다.

    이는 query 정보가 학습 loss에서 중복되는 문제를 제거한다:
        - Concat/SuperNode GAT가 query-aware node embedding 생산
        - MLP classifier가 node embedding -> gold label 직접 학습
    """
    def __init__(self, in_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        """
        node_emb: (N, in_dim) - GAT 출력 노드 임베딩
        returns: (N,) - sigmoid 이전 raw logit
        """
        return self.mlp(node_emb).squeeze(-1)
