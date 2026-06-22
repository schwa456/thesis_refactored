"""Smoke test: XiYanSelector / LinkAlignSelector backbone provider 스위치.

배경: Sonnet baseline 측정을 위해 두 LLM-기반 selector 가 config 의
`provider`/`model_name` 을 APIClient 로 전달해야 함 (LLMSQLGenerator/BidirectionalFilter 패턴).

검증 항목:
  (1) provider="sonnet" + model_name="claude-sonnet-4-6" → selector.client 가
      Anthropic native 경로 (is_anthropic=True, anthropic_client 생성, OpenAI client=None).
  (2) 1-call: messages.create 를 stub 으로 대체해 generate_text 가 실제로 Anthropic
      native 경로(_generate_text_anthropic)를 타고 text 블록을 추출하는지 (실 API 키 불요).
  (3) model_name 이 selector.model_name 으로 반영 → generate_text(model=...) 에 그대로 전달.
  (4) provider 미지정(None) → 기존 OpenAI-compatible(vLLM/OPENAI) 경로 유지 (무회귀).

실 Anthropic 키 없이 동작 (messages.create stub). CPU 강제 — GPU 미사용.

Run from project root:
    conda run -n base python src/modules/selectors/tests/test_selector_provider_backbone.py
"""
import os
import sys

# GPU 미사용 (reserved GPU 보호 + 임베더 CPU 로드). torch import 이전에 설정.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from modules.selectors.xiyan_selector import XiYanSelector
from modules.selectors.linkalign_selector import LinkAlignSelector

EMB = "sentence-transformers/all-MiniLM-L6-v2"
MODEL = "claude-sonnet-4-6"


# --- Anthropic messages.create stub (실 API 키/네트워크 불요) -----------------
class _TextBlock:
    type = "text"
    text = "id, name, created_at"


class _Usage:
    input_tokens = 7
    output_tokens = 4
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0


class _Resp:
    content = [_TextBlock()]
    usage = _Usage()


def _install_stub(client):
    """anthropic_client.messages.create 를 stub 으로 교체. (call 카운트/모델 캡처)"""
    state = {"calls": 0, "model": None, "temperature": None}

    def _stub(**kw):
        state["calls"] += 1
        state["model"] = kw.get("model")
        state["temperature"] = kw.get("temperature")
        return _Resp()

    client.anthropic_client.messages.create = _stub
    return state


def _check_anthropic_wiring(sel, name):
    c = sel.client
    assert getattr(c, "is_anthropic", False) is True, \
        f"{name}: provider='sonnet' 인데 is_anthropic != True"
    assert c.anthropic_client is not None, f"{name}: anthropic_client 미생성"
    assert c.client is None, f"{name}: Anthropic 경로에서 OpenAI client 는 None 이어야 함"
    assert sel.model_name == MODEL, f"{name}: model_name 미반영 ({sel.model_name})"

    state = _install_stub(c)
    out = c.generate_text(prompt="Extract keywords: how many users?",
                          model=MODEL, temperature=0.0)
    assert state["calls"] == 1, f"{name}: messages.create 정확히 1회여야 함 (got {state['calls']})"
    assert state["model"] == MODEL, f"{name}: stub 에 전달된 model 불일치 ({state['model']})"
    assert out == "id, name, created_at", f"{name}: Anthropic text 블록 추출 실패 ({out!r})"
    print(f"  [{name}] is_anthropic=True, 1-call Anthropic 경로 OK, "
          f"model='{state['model']}', temp={state['temperature']} → '{out}'")


def _check_default_no_regression(sel, name):
    c = sel.client
    assert getattr(c, "is_anthropic", False) is False, \
        f"{name}: provider=None 인데 Anthropic 경로로 빠짐"
    assert c.client is not None, \
        f"{name}: provider=None 이면 OpenAI-compatible client 유지되어야 함 (기존 경로 회귀)"
    print(f"  [{name}] provider=None → OpenAI-compatible 경로 유지 (무회귀)")


def main():
    print("=== (1)~(3) provider='sonnet' Anthropic native 경로 ===")
    xs = XiYanSelector(model_name=MODEL, top_k=15, embedding_model=EMB,
                       db_dir="/tmp/_nonexistent_smoke", provider="sonnet")
    _check_anthropic_wiring(xs, "XiYanSelector")

    ls = LinkAlignSelector(model_name=MODEL, top_k=15, embedding_model=EMB,
                           provider="sonnet")
    _check_anthropic_wiring(ls, "LinkAlignSelector")

    print("=== (4) provider=None 무회귀 (기존 OpenAI-compatible 경로) ===")
    xs2 = XiYanSelector(model_name="qwen-or-glm", top_k=15, embedding_model=EMB,
                        db_dir="/tmp/_nonexistent_smoke")
    _check_default_no_regression(xs2, "XiYanSelector")

    ls2 = LinkAlignSelector(model_name="qwen-or-glm", top_k=15, embedding_model=EMB)
    _check_default_no_regression(ls2, "LinkAlignSelector")

    print("\n[PASS] selector backbone provider 스위치 smoke 전체 통과")


if __name__ == "__main__":
    main()
