import os
from openai import OpenAI
from typing import List, Optional, Union, Dict, Any
from utils.logger import get_logger
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

def _empty_usage() -> Dict[str, int]:
    return {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "calls": 0}


class APIClient:
    """
    LLM (텍스트 생성) 및 PLM (텍스트 임베딩) 호출을 전담하는 통신 클라이언트입니다.
    OpenAI 표준 규격을 따르므로 vLLM, Ollama, OpenAI, DeepSeek API 모두에 호환됩니다.
    """

    # 모든 인스턴스에서 공유되는 누적 토큰 사용량 (파이프라인 종료 시 요약용)
    TOKEN_USAGE: Dict[str, int] = _empty_usage()
    USAGE_BY_MODEL: Dict[str, Dict[str, int]] = {}

    @classmethod
    def get_usage_summary(cls) -> Dict[str, Any]:
        return {
            "total": dict(cls.TOKEN_USAGE),
            "by_model": {m: dict(v) for m, v in cls.USAGE_BY_MODEL.items()},
        }

    @classmethod
    def reset_usage(cls) -> None:
        cls.TOKEN_USAGE = _empty_usage()
        cls.USAGE_BY_MODEL = {}

    @classmethod
    def _record_usage(cls, model: str, usage_obj: Any) -> None:
        if usage_obj is None:
            return
        prompt = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        details = getattr(usage_obj, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0

        cls.TOKEN_USAGE["input_tokens"] += prompt
        cls.TOKEN_USAGE["cached_input_tokens"] += cached
        cls.TOKEN_USAGE["output_tokens"] += completion
        cls.TOKEN_USAGE["calls"] += 1

        bucket = cls.USAGE_BY_MODEL.setdefault(model, _empty_usage())
        bucket["input_tokens"] += prompt
        bucket["cached_input_tokens"] += cached
        bucket["output_tokens"] += completion
        bucket["calls"] += 1

        logger.debug(
            f"[LLM usage] model={model} input={prompt} cached={cached} output={completion}"
        )

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        # If caller passes an explicit base_url, we're talking to a non-default
        # endpoint — don't mix in VLLM credentials (local setup convenience).
        # Otherwise VLLM env takes priority for backward-compatible local runs.
        if base_url is not None:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "sk-missing"
            self.base_url = base_url
        else:
            self.api_key = (
                api_key
                or os.getenv("VLLM_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or "vllm"
            )
            self.base_url = (
                os.getenv("VLLM_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
            )
        
        logger.info(f"Initializing API Client... (Base URL: {self.base_url if self.base_url else 'Default OpenAI'})")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.local_encoder = None

    def get_embeddings(self, texts: Union[str, List[str]], model: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[List[float]]:
        """
        PLM(임베딩 모델) 서버를 호출하여 Dense Vector를 받아옵니다.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        logger.debug(f"Calling Embedding API (Model: {model}, Batch Size: {len(texts)})...")
        
        try:
            if self.local_encoder is None:
                logger.info(f"Loading Local Sentence Transformer Model: {model}")
                self.local_encoder = SentenceTransformer(model)

            embeddings = self.local_encoder.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"🚨 Local Embedding 추출 실패: {e}")
            raise

    def generate_text(self, prompt: str, model: str, temperature: float) -> str:
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
                timeout=60.0,
                max_tokens=300
            )
            APIClient._record_usage(model, getattr(response, "usage", None))
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"🚨 LLM API 호출 실패 (OOM, Timeout, or Connection Error): {e}")
            return "SELECT 'API ERROR'"