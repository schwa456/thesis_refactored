"""C-v2 Question Enrichment (Wave 11, 학술 agent §3.3 + §5.3 정합).

M4 Filter 출력 schema 만으로 LLM 이 question 을 enrich (E-SQL 정합 + No Full
Schema 제약). enriched question 이 SQL Generator 의 query 자리를 대체.

핵심 제약 (학술 agent §9 + DECISIONS 2026-05-19 §3 Phase A):
  - 입력 schema 는 반드시 M4 Filter 출력만 (전체 DB schema 포함 금지)
  - Enrichment 결과 캐싱 (동일 question 의 enrichment 한 번만)

LLM call/q: +1 (per query, cache miss). cache hit 시 0.
"""
import hashlib
import json
import time
from typing import Any, Dict, List, Optional

# 학술 agent §5.3 — system prompt
ENRICHMENT_SYSTEM_PROMPT = """
You are a database expert who links natural language questions to database schemas.
Given a question and a filtered schema, rewrite the question to:
1. Explicitly reference relevant table.column names where applicable
2. State the JOIN conditions needed between tables
3. Include any filter conditions (WHERE clauses) with specific column references
4. Preserve the original intent of the question completely

Output only the enriched question as a single paragraph.
Do NOT generate SQL. Do NOT add extra commentary.
""".strip()

# 학술 agent §5.3 — few-shot template
ENRICHMENT_FEW_SHOT_TEMPLATE = """
--- Example {idx} ---
Original Question: {question}
Filtered Schema:
{schema}
Enriched Question: {enriched}
""".strip()

ENRICHMENT_QUERY_TEMPLATE = """
--- Your Task ---
Original Question: {question}
Filtered Schema:
{schema}
Enriched Question:""".strip()


def format_schema_for_enrichment(m4_schema: Dict[str, List[str]]) -> str:
    """M4 schema → enrichment prompt 용 text. No Full Schema 제약 — M4 출력만."""
    lines: List[str] = []
    for table, cols in (m4_schema or {}).items():
        lines.append(f"Table: {table}")
        for col in (cols or []):
            lines.append(f"  - {col}")
    return "\n".join(lines) if lines else "(empty schema)"


class EnrichmentCache:
    """동일 (question, schema) pair 의 enrichment 결과 재사용 (학술 agent §9 정합).

    Process-level memoization — same question + same M4 schema → same enriched
    question. 학술 agent §9 의 "반복 실험 시 재사용" 정합.
    """

    def __init__(self):
        self._store: Dict[str, str] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(question: str, schema_str: str) -> str:
        h = hashlib.sha1()
        h.update((question or "").encode("utf-8", errors="replace"))
        h.update(b"\x1f")
        h.update((schema_str or "").encode("utf-8", errors="replace"))
        return h.hexdigest()

    def get(self, question: str, schema_str: str) -> Optional[str]:
        k = self._key(question, schema_str)
        if k in self._store:
            self.hits += 1
            return self._store[k]
        self.misses += 1
        return None

    def set(self, question: str, schema_str: str, value: str) -> None:
        k = self._key(question, schema_str)
        self._store[k] = value

    def size(self) -> int:
        return len(self._store)

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "size": len(self._store)}


# Module-level singleton cache (process-wide). 별도 caller 가 override 가능.
_DEFAULT_CACHE = EnrichmentCache()


def enrich_question(
    question: str,
    m4_schema: Dict[str, List[str]],
    few_shot_examples: List[Dict[str, Any]],
    llm_client,
    model_name: str,
    temperature: float = 0.0,
    cache: Optional[EnrichmentCache] = None,
    fallback_on_error: bool = True,
) -> str:
    """학술 agent §5.3 정합. Enrichment 결과 cache lookup 우선.

    Args:
        question: 원 자연어 query
        m4_schema: {table: [col, ...]} — M4 Filter 출력 (No Full Schema)
        few_shot_examples: List of {"question": str, "schema": dict, "enriched_question": str}
                           — 학술 agent §5.4 정합 (BIRD-dev simple 4 / moderate 4 /
                             challenging 4, data leakage 방지: test query DB 와 다른 DB)
        llm_client: APIClient instance with generate_text(prompt, model, temperature)
        model_name: LLM model
        temperature: LLM temperature (학술 agent 0.0 권장)
        cache: EnrichmentCache instance. None 이면 module-level singleton 사용.
        fallback_on_error: LLM call 실패 시 원 question 반환 (default True)

    Returns:
        Enriched question string. Cache hit 시 prior result 그대로.
    """
    cache = cache if cache is not None else _DEFAULT_CACHE
    schema_str = format_schema_for_enrichment(m4_schema)

    cached = cache.get(question, schema_str)
    if cached is not None:
        return cached

    # Build prompt: few-shots + query
    if few_shot_examples:
        few_shot_str = "\n\n".join(
            ENRICHMENT_FEW_SHOT_TEMPLATE.format(
                idx=i + 1,
                question=ex.get("question", ""),
                schema=format_schema_for_enrichment(ex.get("schema") or {}),
                enriched=ex.get("enriched_question", ""),
            )
            for i, ex in enumerate(few_shot_examples)
        )
    else:
        few_shot_str = ""
    query_str = ENRICHMENT_QUERY_TEMPLATE.format(
        question=question, schema=schema_str,
    )
    # System prompt 은 prompt body 의 prefix 로 inject (current APIClient 가 system
    # 인자 별도 미지원 — generate_text(prompt, model, temperature)).
    full_prompt = (
        ENRICHMENT_SYSTEM_PROMPT
        + "\n\n" + few_shot_str
        + ("\n\n" if few_shot_str else "")
        + query_str
    )
    try:
        response = llm_client.generate_text(
            prompt=full_prompt, model=model_name, temperature=temperature,
        )
    except Exception:
        if fallback_on_error:
            cache.set(question, schema_str, question)
            return question
        raise
    if not isinstance(response, str) or not response.strip():
        enriched = question if fallback_on_error else ""
    else:
        # Strip leading/trailing whitespace + 가능한 LLM 의 markdown 잔재
        enriched = response.strip()
        # 한 줄 paragraph 가 spec — 만약 LLM 이 줄바꿈을 많이 넣으면 첫 paragraph 만
        # 채택 (학술 agent §5.3 "Output only the enriched question as a single
        # paragraph").
        if "\n\n" in enriched:
            enriched = enriched.split("\n\n", 1)[0].strip()
    cache.set(question, schema_str, enriched)
    return enriched
