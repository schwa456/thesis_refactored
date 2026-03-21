import torch
from typing import List

from modules.registry import register
from modules.base import BaseEncoder
from llm_client.api_handler import APIClient
from utils.logger import get_logger

logger = get_logger(__name__)

@register("encoder", "APIEncoder")
class APIEncoder(BaseEncoder):
    """
    외부 API(OpenAI 호환)를 호출하여 텍스트를 Dense Vector로 변환하는 인코더입니다.
    """
    # 💡 Config의 params에 적힌 값들이 여기로 자동으로 들어옵니다!
    def __init__(self, model_name: str, base_url: str = None, **kwargs):
        super().__init__()
        self.model_name = model_name
        
        # 앞서 만든 공통 API 클라이언트 초기화
        self.client = APIClient(api_key="vllm", base_url=base_url)
        logger.info(f"Initialized APIEncoder with model: [{self.model_name}]")

    def encode(self, texts: List[str]) -> torch.Tensor:
        logger.debug(f"Requesting embeddings for {len(texts)} texts...")
        
        # API Client에게 Config에서 받은 모델명을 전달하여 임베딩 요청
        embeddings_list = self.client.get_embeddings(texts, model=self.model_name)
        
        # PyTorch Tensor로 변환하여 반환 (BaseEncoder 규격 준수)
        return torch.tensor(embeddings_list, dtype=torch.float)