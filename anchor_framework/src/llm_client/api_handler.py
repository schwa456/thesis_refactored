import os
import time
from openai import OpenAI
from typing import List, Optional, Union, Dict, Any
from utils.logger import get_logger
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

def _empty_usage() -> Dict[str, int]:
    return {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "calls": 0}


# Provider → (BASE_URL env var, API_KEY env var)
# base_url env 가 None 이면 OpenAI SDK 기본 endpoint 사용 (OpenAI 공식 API).
_PROVIDER_ENV_MAP: Dict[str, tuple] = {
    "vllm":   ("VLLM_BASE_URL",  "VLLM_API_KEY"),
    "openai": (None,             "OPENAI_API_KEY"),
    "glm":    ("GLM_BASE_URL",   "GLM_API_KEY"),
    "zhipu":  ("GLM_BASE_URL",   "GLM_API_KEY"),  # alias
}

# Anthropic (Claude) native SDK 를 쓰는 provider 별칭 (OpenAI-compatible 아님).
# DECISIONS 2026-06-10 #5: GLM-4.7 → claude-sonnet-4-6 backbone 전환.
_ANTHROPIC_PROVIDERS = {"anthropic", "sonnet", "claude"}


class APIClient:
    """
    LLM (텍스트 생성) 및 PLM (텍스트 임베딩) 호출을 전담하는 통신 클라이언트입니다.
    OpenAI 표준 규격을 따르므로 vLLM, Ollama, OpenAI, Zhipu GLM 등에 호환됩니다.
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

    @classmethod
    def _record_usage_anthropic(cls, model: str, usage_obj: Any) -> None:
        """Anthropic messages usage (input_tokens / output_tokens / cache_read_input_tokens)."""
        if usage_obj is None:
            return
        prompt = int(getattr(usage_obj, "input_tokens", 0) or 0)
        completion = int(getattr(usage_obj, "output_tokens", 0) or 0)
        cached = int(getattr(usage_obj, "cache_read_input_tokens", 0) or 0)
        cache_creation = int(getattr(usage_obj, "cache_creation_input_tokens", 0) or 0)
        # input_tokens 는 Anthropic 에서 cache-read 를 별도 집계 → 총 입력 = input + cache_read + cache_creation
        cls.TOKEN_USAGE["input_tokens"] += prompt + cache_creation
        cls.TOKEN_USAGE["cached_input_tokens"] += cached
        cls.TOKEN_USAGE["output_tokens"] += completion
        cls.TOKEN_USAGE["calls"] += 1

        bucket = cls.USAGE_BY_MODEL.setdefault(model, _empty_usage())
        bucket["input_tokens"] += prompt + cache_creation
        bucket["cached_input_tokens"] += cached
        bucket["output_tokens"] += completion
        bucket["calls"] += 1

        logger.debug(
            f"[LLM usage] model={model} input={prompt} cache_read={cached} "
            f"cache_create={cache_creation} output={completion}"
        )

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, provider: Optional[str] = None):
        # 우선순위:
        #   1. provider 가 지정되면 _PROVIDER_ENV_MAP 에서 env pair 를 로드
        #      (caller 가 api_key/base_url 을 같이 넘기면 그것이 override)
        #   2. base_url 만 지정되면 외부 endpoint 직접 지정 — OPENAI_API_KEY 로 붙음
        #   3. 그 외엔 기존 fallback: VLLM env 우선 → OPENAI env
        # ── Anthropic (Claude Sonnet 4.6) native path ─────────────────────
        self.is_anthropic = provider is not None and provider.lower() in _ANTHROPIC_PROVIDERS
        if self.is_anthropic:
            import anthropic
            self.api_key = (
                api_key
                or os.getenv("CLAUDE_API_KEY")
                or os.getenv("ANTHROPIC_API_KEY")
                or "sk-missing"
            )
            self.base_url = base_url  # 보통 None (Anthropic 기본 endpoint)
            self.anthropic_client = anthropic.Anthropic(
                api_key=self.api_key,
                **({"base_url": base_url} if base_url else {}),
            )
            self.client = None  # OpenAI client 미사용
            self.local_encoder = None
            logger.info(
                f"Initializing API Client... (Provider: {provider} [Anthropic native], "
                f"model defaults to caller-specified, e.g. claude-sonnet-4-6)"
            )
            return

        if provider is not None:
            key = provider.lower()
            if key not in _PROVIDER_ENV_MAP:
                raise ValueError(
                    f"Unknown LLM provider '{provider}'. Valid: {list(_PROVIDER_ENV_MAP) + sorted(_ANTHROPIC_PROVIDERS)}"
                )
            url_env, key_env = _PROVIDER_ENV_MAP[key]
            resolved_url = os.getenv(url_env) if url_env else None
            resolved_key = os.getenv(key_env)
            self.base_url = base_url or resolved_url
            self.api_key = api_key or resolved_key or "sk-missing"
        elif base_url is not None:
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

        logger.info(
            f"Initializing API Client... (Provider: {provider or 'auto'}, "
            f"Base URL: {self.base_url if self.base_url else 'Default OpenAI'})"
        )
        
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
        Rate limit (429) 위 exponential backoff (1, 2, 4, 8, 16, 32s) 6회 retry. (2026-06-08 GLM-4.7 concurrency=2)
        provider 가 Anthropic 계열이면 native messages API 경로로 분기 (DECISIONS 2026-06-10 #5).
        """
        if getattr(self, "is_anthropic", False):
            return self._generate_text_anthropic(prompt, model, temperature)

        logger.debug(f"Calling LLM API (Model: {model}, Temp: {temperature})...")

        max_retries = 6
        for attempt in range(max_retries + 1):
            try:
                # GLM-4.7 위 thinking={"type":"disabled"} 위 reasoning chain 비활성
                # (default enabled → output 1500+ tokens, latency 30+s/call). 2026-06-08 Z.AI param doc.
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful database expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    timeout=60.0,
                    max_tokens=300,
                    extra_body={"thinking": {"type": "disabled"}}
                )
                APIClient._record_usage(model, getattr(response, "usage", None))
                return response.choices[0].message.content

            except Exception as e:
                err_str = str(e)
                is_rate_limit = '429' in err_str or 'rate limit' in err_str.lower() or 'Rate limit' in err_str
                if is_rate_limit and attempt < max_retries:
                    wait = min(1.0 * (2 ** attempt), 32.0)
                    logger.warning(f"⏳ Rate limit hit (429), retry {attempt+1}/{max_retries} after {wait}s...")
                    time.sleep(wait)
                    continue
                logger.error(f"🚨 LLM API 호출 실패 (OOM, Timeout, or Connection Error): {e}")
                return "SELECT 'API ERROR'"

    # ── Anthropic (Claude Sonnet 4.6) native generation ───────────────────
    # DECISIONS 2026-06-10 #5 제약:
    #   - assistant prefill 금지 (Sonnet 4.6 에서 400). system+user 메시지만.
    #   - temperature OR top_p 택1 (여기선 temperature 만).
    #   - budget_tokens deprecated. 기본 standard mode (extended thinking off) =
    #     GLM thinking-disabled 와 동등한 cost-conscious 설정 (probe 비용 통제).
    #     extended thinking 필요 시 ANTHROPIC_THINKING env 로 활성 (probe 후 결정).
    #   - JSON 강제는 호출측 parser 가 담당 (output_config.format 은 Phase 3 재검증).
    #   - cache_control ephemeral 을 system 에 부여 (실제 절감은 Phase 4 schema-prefix
    #     breakpoint 에서; 여기선 breakpoint 위치만 선점, 무해).
    def _generate_text_anthropic(self, prompt: str, model: str, temperature: float) -> str:
        logger.debug(f"Calling Anthropic API (Model: {model}, Temp: {temperature})...")
        max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024"))
        thinking_mode = os.getenv("ANTHROPIC_THINKING", "").strip().lower()  # "", "adaptive", ...
        extra = {}
        if thinking_mode in {"adaptive", "high", "max"}:
            extra["thinking"] = {"type": "adaptive"}

        system_blocks = [{
            "type": "text",
            "text": "You are a helpful database expert.",
            "cache_control": {"type": "ephemeral"},
        }]

        max_retries = 6
        for attempt in range(max_retries + 1):
            try:
                resp = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_blocks,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=120.0,
                    **extra,
                )
                APIClient._record_usage_anthropic(model, getattr(resp, "usage", None))
                # text content block 만 추출 (thinking block 은 제외)
                parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
                return "".join(parts).strip() if parts else ""
            except Exception as e:
                err_str = str(e)
                is_rate_limit = (
                    '429' in err_str or '529' in err_str
                    or 'rate limit' in err_str.lower() or 'overloaded' in err_str.lower()
                )
                if is_rate_limit and attempt < max_retries:
                    wait = min(1.0 * (2 ** attempt), 32.0)
                    logger.warning(f"⏳ Anthropic rate/overload, retry {attempt+1}/{max_retries} after {wait}s...")
                    time.sleep(wait)
                    continue
                logger.error(f"🚨 Anthropic API 호출 실패: {e}")
                return "SELECT 'API ERROR'"