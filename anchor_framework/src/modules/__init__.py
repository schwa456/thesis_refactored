# Anchor framework — 레지스트리 트리거. anchor 파이프라인 모듈만 import.
import modules.builders
import modules.encoders
import modules.projectors
import modules.selectors
import modules.extractors
import modules.filters
import modules.generators

from .registry import build

__all__ = ["build"]
