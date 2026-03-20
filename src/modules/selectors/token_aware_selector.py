import torch
from typing import List, Dict, Any, Tuple
from modules.registry import register
from modules.base import BaseSelector
from utils.logger import get_logger

logger = get_logger(__name__)

@register("selector", "TokenAwareSelector")
class TokenAwareSelector(BaseSelector):
    """
    NLQ의 각 토큰(불용어 제외)과 그래프 노드 간의 유사도를 비교하여,
    각 토큰별로 가장 연관성이 높은 노드를 1개씩 추출하는 Selector입니다.
    """
    def __init__(self, top_k_per_token: int = 1, **kwargs):
        self.top_k_per_token = top_k_per_token
        logger.info(f"Initialized TokenAwareSelector (Top-{top_k_per_token} per token)")

    def select(self, token_embs: torch.Tensor, node_embs: torch.Tensor, mask: torch.Tensor, candidates: List[Any], **kwargs) -> List[Any]:
        """
        token_embs: (Seq_Len, Dim)
        node_embs: (Total_Nodes, Dim)
        mask: (Seq_Len,) - True if valid token
        """
        # 1. 유효한 토큰만 필터링
        valid_token_embs = token_embs[mask] # (Valid_Seq_Len, Dim)
        
        # 2. 유사도 행렬 계산 (Cosine Similarity)
        # (Valid_Seq_Len, Dim) @ (Dim, Total_Nodes) -> (Valid_Seq_Len, Total_Nodes)
        z_tokens = torch.nn.functional.normalize(valid_token_embs, p=2, dim=-1)
        z_nodes = torch.nn.functional.normalize(node_embs, p=2, dim=-1)
        sim_matrix = torch.matmul(z_tokens, z_nodes.t())
        
        # 3. 각 토큰별로 가장 유사한 노드 추출
        # 각 행(토큰)에서 최대값의 인덱스를 찾음
        _, top_node_indices = torch.topk(sim_matrix, k=self.top_k_per_token, dim=-1)
        
        # 4. 중복 제거 및 최종 Seed 리스트 구성
        unique_indices = torch.unique(top_node_indices).tolist()
        selected_seeds = [candidates[idx] for idx in unique_indices]
        
        logger.debug(f"[TokenAware] Selected {len(selected_seeds)} unique seeds from {len(valid_token_embs)} valid tokens.")
        return selected_seeds