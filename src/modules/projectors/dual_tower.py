import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from modules.registry import register
from modules.base import BaseProjector
from utils.logger import get_logger

logger = get_logger(__name__)

@register("projector", "DualTowerProjector")
class DualTowerProjector(BaseProjector):
    """
    Symmetric Dual-Tower 구조를 통해 텍스트(NLQ)와 그래프(Node) 임베딩을
    동일한 공유 공간(Shared Latent Space)으로 투영합니다.
    """
    def __init__(self, text_dim: int = 384, graph_dim: int = 256, joint_dim: int = 256, dropout_rate: float = 0.1, **kwargs):
        super(DualTowerProjector, self).__init__()
        
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        self.joint_dim = joint_dim
        
        logger.info(f"Initializing DualTowerAlignment Projector (Text: {text_dim} -> {joint_dim}, Graph: {graph_dim} -> {joint_dim})")
        
        # 1. Text Projection Head
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.joint_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.joint_dim, self.joint_dim)
        )
        
        # 2. Graph Projection Head
        self.graph_proj = nn.Sequential(
            nn.Linear(self.graph_dim, self.joint_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.joint_dim, self.joint_dim)
        )
        
        # 대조 학습을 위한 Temperature 파라미터 (초기값 0.07 기반)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, text_embs: torch.Tensor, graph_embs: torch.Tensor, batch_index: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        batch_index를 사용하여 질의 임베딩을 노드별로 매핑합니다.

        text_embs: (Batch_Size, Text_Dim)
        graph_embs: (Total_Nodes, Graph_Dim)
        batch_index: (Total_Nodes,) - 각 노드가 속한 그래프 인덱스
        """
        # 차원 검증
        if text_embs.shape[-1] != self.text_dim:
            logger.warning(f"Text embedding dimension mismatch! Expected {self.text_dim}, got {text_embs.shape[-1]}")
        if graph_embs.shape[-1] != self.graph_dim:
            logger.warning(f"Graph embedding dimension mismatch! Expected {self.graph_dim}, got {graph_embs.shape[-1]}")

        # 투영 및 L2 정규화
        z_text = F.normalize(self.text_proj(text_embs), p=2, dim=-1)
        z_graph = F.normalize(self.graph_proj(graph_embs), p=2, dim=-1)

        if batch_index is not None:
            # batch_index가 [0, 0, 1, 1, 1]이면, 0번 질의 벡터 2개와 1번 질의 벡터 3개로 확장
            z_text = z_text[batch_index] # (Total_Nodes, Joint_Dim)
            
        return z_text, z_graph

    def compute_similarity(self, z_query: torch.Tensor, z_nodes: torch.Tensor) -> torch.Tensor:
        """
        투영된 두 텐서 간의 코사인 유사도 점수(Logits)를 계산합니다.
        z_query: (N, D), z_nodes: (N, D)
        """
        # 행 단위 내적 (Row-wise Dot Product)
        # 이미 L2 정규화가 되어 있으므로 내적은 코사인 유사도와 동일함
        logits = (z_query * z_nodes).sum(dim=-1) # (N,)
        
        # 온도(Temperature) 적용
        return logits * self.logit_scale.exp()

    def compute_contrastive_loss(self, z_text: torch.Tensor, z_graph: torch.Tensor) -> torch.Tensor:
        """
        학습(Training) 단계에서 InfoNCE 손실을 계산하는 유틸리티 메서드입니다.
        (z_text와 z_graph는 배치 내에서 1:1 Positive Pair로 구성되어야 함)
        """
        logit_scale = self.logit_scale.exp()
        sim_matrix = logit_scale * torch.matmul(z_text, z_graph.t())
        
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device, dtype=torch.long)
        
        loss_t2g = F.cross_entropy(sim_matrix, labels)
        loss_g2t = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_t2g + loss_g2t) / 2