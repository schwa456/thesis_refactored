"""Wave 11 Schema Serialization Direction C — serializer modules (2026-05-19).

학술 frame: Filter 출력의 직렬화(Serialization) 방식 변경만으로 EX ceiling 0.5300
돌파 가능성 검증. Wave 8 F1-EX Decoupling 의 인과 검증 trigger. Schema Content
Invariance retain — 같은 M4 column 집합 → 직렬화만 다르게.

5 cells (DECISIONS 2026-05-19 §3 Phase A):
  - C-v1 Source-Tagged Schema       (source_tagged_serializer)
  - C-v2 Question Enrichment        (question_enricher, +1 LLM)
  - C-v3a Flat Merged (FK 포함)     (flat_merged_serializer with fk_relations)
  - C-v3b Flat Merged (FK 없음)     (flat_merged_serializer without fk_relations)
  - Comb-C C-v2 + C-v1              (question_enricher + source_tagged_serializer)
"""
from .source_tagged_serializer import (
    tag_columns,
    format_tagged_schema,
)
from .flat_merged_serializer import (
    format_flat_schema,
)
from .question_enricher import (
    enrich_question,
    EnrichmentCache,
    ENRICHMENT_SYSTEM_PROMPT,
    ENRICHMENT_FEW_SHOT_TEMPLATE,
    ENRICHMENT_QUERY_TEMPLATE,
    format_schema_for_enrichment,
)

__all__ = [
    "tag_columns",
    "format_tagged_schema",
    "format_flat_schema",
    "enrich_question",
    "EnrichmentCache",
    "ENRICHMENT_SYSTEM_PROMPT",
    "ENRICHMENT_FEW_SHOT_TEMPLATE",
    "ENRICHMENT_QUERY_TEMPLATE",
    "format_schema_for_enrichment",
]
