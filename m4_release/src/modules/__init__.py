# M4 reproducibility package — trimmed module registry bootstrap.
# 각 카테고리 패키지를 import 하면 @register 데코레이터가 실행되어
# 레지스트리에 M4 anchor 모듈이 등록된다. (full src/ tree 의 ablation 변형은 제외)
import modules.builders
import modules.encoders
import modules.projectors
import modules.selectors
import modules.extractors
import modules.filters
import modules.generators

from .registry import build

__all__ = ["build"]
