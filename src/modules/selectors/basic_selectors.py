import torch
from typing import List, Any

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