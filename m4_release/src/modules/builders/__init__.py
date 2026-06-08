# M4 anchor: EnrichedHeteroGraphBuilder (base HeteroGraphBuilder 포함).
# RFMCompatible / Triplet 변형은 graph_builder.py 에 동거하므로 함께 노출하되,
# M4 config 가 채택하는 것은 EnrichedHeteroGraphBuilder 이다.
from .graph_builder import (
    HeteroGraphBuilder,
    EnrichedHeteroGraphBuilder,
)

__all__ = [
    "HeteroGraphBuilder",
    "EnrichedHeteroGraphBuilder",
]
