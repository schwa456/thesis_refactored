import os
import json
import time
import torch
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from modules.registry import register
from modules.base import BaseEncoder
from utils.logger import get_logger

logger = get_logger(__name__)


@register("encoder", "LocalPLMEncoder")
class LocalPLMEncoder(BaseEncoder):
    """
    HuggingFace SentenceTransformers를 사용하여 로컬 GPU/CPU에서 직접 임베딩을 뽑는 인코더입니다.

    Wave 16 (2026-05-21): encoding profile logging 추가 (DECISIONS §7.1).
      - LOCAL_ENCODER_PROFILE_PATH 환경변수가 지정되면 per-call 의 wall time, peak GPU,
        sample/token throughput, embed dim 을 jsonl 로 append.
      - LOCAL_ENCODER_PROFILE_TAG 환경변수로 row tag 부착 (예: db_id).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs):
        super().__init__()
        self.model_name = model_name
        logger.info(f"Loading local PLM to memory: [{self.model_name}]...")
        self.model = SentenceTransformer(self.model_name)
        self._profile_path: Optional[str] = os.environ.get("LOCAL_ENCODER_PROFILE_PATH")
        if self._profile_path:
            os.makedirs(os.path.dirname(self._profile_path), exist_ok=True)
            logger.info(f"Encoder profile logging enabled: {self._profile_path}")

    def encode(self, texts: List[str]) -> torch.Tensor:
        if self._profile_path:
            return self._encode_with_profile(texts)
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings

    def _encode_with_profile(self, texts: List[str]) -> torch.Tensor:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        if cuda_available:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_samples = len(texts)
        gpu_peak_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if cuda_available else 0.0
        token_count = sum(len(t.split()) for t in texts)
        embed_dim = int(embeddings.shape[-1]) if embeddings.dim() >= 1 else 0
        row = {
            "tag": os.environ.get("LOCAL_ENCODER_PROFILE_TAG", ""),
            "model_name": self.model_name,
            "n_samples": n_samples,
            "n_tokens_approx": token_count,
            "encoding_time_s": round(elapsed, 6),
            "throughput_samples_per_s": round(n_samples / elapsed, 4) if elapsed > 0 else 0.0,
            "throughput_tokens_per_s": round(token_count / elapsed, 4) if elapsed > 0 else 0.0,
            "gpu_peak_mb": round(gpu_peak_mb, 2),
            "embed_dim": embed_dim,
        }
        try:
            with open(self._profile_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to append encoder profile row: {e}")
        return embeddings
