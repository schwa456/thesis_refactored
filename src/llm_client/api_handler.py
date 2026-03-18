import os
from openai import OpenAI
from typing import List, Optional, Union
from utils.logger import get_logger

logger = get_logger(__name__)

class APIClient:
    """
    LLM (텍스트 생성) 및 PLM (텍스트 임베딩) 호출을 전담하는 통신 클라이언트입니다.
    OpenAI 표준 규격을 따르므로 vLLM, Ollama, OpenAI, DeepSeek API 모두에 호환됩니다.
    """
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        # 환경 변수를 우선적으로 사용하고, 없으면 파라미터로 받습니다.
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY") 
        # base_url이 있으면 로컬 vLLM/Ollama 서버 등을 찌릅니다.
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        logger.info(f"Initializing API Client... (Base URL: {self.base_url if self.base_url else 'Default OpenAI'})")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def get_embeddings(self, texts: Union[str, List[str]], model: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[List[float]]:
        """
        PLM(임베딩 모델) 서버를 호출하여 Dense Vector를 받아옵니다.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        logger.debug(f"Calling Embedding API (Model: {model}, Batch Size: {len(texts)})...")
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=model
            )
            # 결과물에서 임베딩 벡터 리스트만 추출하여 반환
            return [data.embedding for data in response.data]
            
        except Exception as e:
            logger.error(f"🚨 Embedding API 호출 실패: {e}")
            raise

    def generate_text(self, prompt: str, model: str = "Meta/Llama-3.1-8B-Instruct", temperature: float = 0.0) -> str:
        """
        LLM 서버를 호출하여 텍스트(또는 JSON/SQL)를 생성합니다. (에이전트 필터링용)
        """
        logger.debug(f"Calling LLM API (Model: {model}, Temp: {temperature})...")
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful database expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"🚨 LLM API 호출 실패: {e}")
            raise