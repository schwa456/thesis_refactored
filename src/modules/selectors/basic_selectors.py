import torch
from typing import List, Any, Union

from modules.registry import register
from modules.base import BaseSelector
from utils.logger import get_logger

logger = get_logger(__name__)

@register("selector", "FixedTopKSelector")
class FixedTopKSelector(BaseSelector):
    """ 단순 Top-k 선택기 """
    def __init__(self, k: int = 3, **kwargs):
        self.k = k
        logger.info(f"Initialized FixedTopKSelector (k={self.k})")
    
    def select(self, scores: torch.Tensor, candidates: List[Any], **kwargs) -> List[Any]:
        k_actual = min(self.k, len(candidates))
        top_k_indices = torch.topk(scores, k=k_actual).indices
        
        selected = [candidates[i] for i in top_k_indices]
        logger.debug(f"Selected Top-{k_actual} seeds.")
        return selected

@register("selector", "AdaptiveSelector")
class AdaptiveSelector(BaseSelector):
    """ 임계값 기반 적응형 선택기 """
    def __init__(self, alpha: float = 0.8, min_k: int = 2, max_k: int = 5, **kwargs):
        self.alpha = alpha
        self.min_k = min_k
        self.max_k = max_k
        logger.info(f"Initialized AdaptiveSelector (alpha={self.alpha}, k={self.min_k}~{self.max_k})")

    def select(self, scores: torch.Tensor, candidates: List[Any], **kwargs) -> List[Any]:
        if not candidates: 
            return []
        
        sorted_indices = torch.argsort(scores, descending=True)
        top_score = scores[sorted_indices[0]].item()
        
        seeds = []
        for idx in sorted_indices:
            score = scores[idx].item()
            # 조건: (Top1 점수의 alpha% 이상) OR (최소 개수 미달 시)
            if (score >= top_score * self.alpha) or (len(seeds) < self.min_k):
                seeds.append(candidates[idx])
            else:
                break
            
            if len(seeds) >= self.max_k:
                break
                
        logger.debug(f"Adaptively selected {len(seeds)} seeds.")
        return seeds
    
@register("selector", "VectorOnlySelector")
class VectorOnlySelector(BaseSelector):
    """
    미리 계산된 NLQ와 Node 간의 유사도 점수(node_scores)를 바탕으로,
    단순히 상위 K개의 노드 인덱스(Seed)를 반환하는 가장 기본적인 Selector입니다.
    """
    def __init__(self, top_k: int = 15, **kwargs):
        self.top_k = top_k
        logger.info(f"Initialized VectorOnlySelector (top_k={self.top_k})")

    def select(self, scores: Union[torch.Tensor, List[float]], candidates: List[Any], **kwargs) -> List[Any]:
        # 1. Type 검사 및 Tensor 변환
        if isinstance(scores, list):
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
        else:
            scores_tensor = scores.squeeze()

        total_candidates = len(candidates)
        
        # 2. 방어적 K값 설정 (G-Retriever를 위한 -1 처리 및 Out-of-bounds 방지)
        if self.top_k <= 0 or self.top_k > total_candidates:
            k_actual = total_candidates
        else:
            k_actual = self.top_k

        # 3. Top-K 추출
        top_scores, top_indices = torch.topk(scores_tensor, k=k_actual)

        # 4. Pipeline에서 넘겨준 후보군(보통 정수 인덱스 리스트)으로 매핑하여 반환
        selected_seeds = [candidates[idx.item()] for idx in top_indices]

        logger.debug(f"[VectorOnly] Selected Top-{k_actual} seeds purely based on Vector Similarity.")
        
        return selected_seeds