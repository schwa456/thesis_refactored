# anchor: LocalPLMEncoder (encoder_type=plm). TokenEncoder 는 ensemble_selector 에서 lazy import (spacy optional).
from .local_encoder import LocalPLMEncoder

__all__ = ["LocalPLMEncoder"]
