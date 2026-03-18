import torch
from typing import List
from sentence_transformers import SentenceTransformer

from modules.registry import register
from modules.base import BaseEncoder
from utils.logger import get_logger

logger = get_logger(__name__)

@register("encoder", "LocalPLMEncoder")
class LocalPLMEncoder(BaseEncoder):
    """
    HuggingFace SentenceTransformers를 사용하여 로컬 GPU/CPU에서 직접 임베딩을 뽑는 인코더입니다.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        super().__init__()
        self.model_name = model_name
        logger.info(f"Loading local PLM to memory: [{self.model_name}]...")
        
        # 로컬 메모리에 모델 직접 로드
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]) -> torch.Tensor:
        # convert_to_tensor=True 옵션으로 바로 PyTorch Tensor 반환
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings