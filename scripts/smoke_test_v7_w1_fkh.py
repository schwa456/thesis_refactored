#!/usr/bin/env python3
"""V7-W1 (FKH, RFP #3) — smoke test (5 scenarios).

CPU only · LLM 0× — 직렬화 함수만 호출. extracted_nodes / extracted_edges 자체
변경 없음 검증 (R/P/F1 동일 보장).

5 scenarios:
1. fkh_00 (none/inline) → baseline 직렬화 결과 == _build_mschema_with_values 그대로
2. fkh_01 (explicit/inline) → 각 table 의 Columns 줄 뒤 "FK:" line 존재
3. fkh_02 (explicit/prefix) → schema block 앞 "# Foreign Key Relationships" + "FK:" line
4. fkh_03 (compact/inline) → 각 table 의 Columns 줄 뒤 "[FK ...=...]" token
5. fkh_04 (compact/suffix) → schema block 뒤 "# Foreign Key Relationships" + "[FK ...=...]"

usage:
    PYTHONPATH=src conda run -n base python scripts/smoke_test_v7_w1_fkh.py
"""
import os
import sys
import traceback
from pathlib import Path

# Ensure src/ on path (script invocation 시 fallback)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Synthetic fixture — 본 framework 의 builder metadata 정합 형식
# ---------------------------------------------------------------------------
SUBGRAPH = {
    "schools": ["CDSCode", "School", "City"],
    "frpm": ["CDSCode", "Enrollment_K12", "FRPM_Count"],
    "satscores": ["cds", "NumTstTakr", "AvgScrRead"],
}

# metadata['fk_to_id'] 형식: "src_t.src_c->dst_t.dst_c" → int (builder
# `_generate_fk_descriptions()` 정합)
METADATA = {
    "fk_to_id": {
        "frpm.CDSCode->schools.CDSCode": 0,
        "satscores.cds->schools.CDSCode": 1,
        # 다음 FK 는 extracted_subgraph 의 column 이 양쪽 모두 포함되지 않음 → hint 위 제외
        "schools.City->cities.name": 2,
    },
    "node_metadata": {},
}


# ---------------------------------------------------------------------------
# Filter helper — XiYanFilter._serialize_schema 직접 호출 (LLM 0×)
# ---------------------------------------------------------------------------
def make_filter(fk_hint_format: str, fk_hint_position: str):
    """LLM client 미초기화 path 위 XiYanFilter instance 구축 (smoke 전용)."""
    from modules.filters.xiyan_filter import XiYanFilter

    f = XiYanFilter.__new__(XiYanFilter)
    # _serialize_schema / _build_mschema_with_values 가 사용하는 attr 만 set
    f.model_name = "smoke"
    f.max_iteration = 1
    f.temperature = 0.0
    f.db_dir = "./data/raw/BIRD_dev/dev_databases"
    f.provider = None
    f.num_examples = 0  # DB lookup skip — smoke 위 sqlite 없어도 무영향
    f.prompt_mode = "default"
    f.cot_reasoning = False
    f.confidence_gated = False
    f.confidence_threshold = 0.5
    f.gate_level = "none"
    f._prompt_section = "xiyan_filter"
    f.sanitize_output = True
    f.fk_hint_format = fk_hint_format
    f.fk_hint_position = fk_hint_position
    return f


def assert_eq(label: str, actual, expected) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def scenario_fkh_00():
    """baseline — fk_hint_format='none' 시 _build_mschema_with_values 결과 그대로."""
    f = make_filter("none", "inline")
    base = f._build_mschema_with_values(SUBGRAPH, "")
    out = f._serialize_schema(SUBGRAPH, "", metadata=METADATA)
    assert out == base, "fkh_00 baseline 직렬화 결과가 base mschema 와 달라짐"
    # FK 관련 token 가 등장하지 않아야 함
    assert "FK:" not in out, "baseline 위 'FK:' token 존재"
    assert "[FK " not in out, "baseline 위 '[FK ' token 존재"
    assert "Foreign Key Relationships" not in out, "baseline 위 FK header 존재"
    return out


def scenario_fkh_01():
    """explicit / inline — 각 src table 의 Columns 줄 뒤 'FK: ...' line."""
    f = make_filter("explicit", "inline")
    out = f._serialize_schema(SUBGRAPH, "", metadata=METADATA)
    # 직접 inscope FK 2개 (frpm.CDSCode->schools.CDSCode, satscores.cds->schools.CDSCode)
    # → 양쪽 column 모두 extracted_subgraph 위 존재. 'schools.City->cities.name' 는
    # cities table 가 subgraph 위 없으므로 hint 위 제외.
    assert "FK: frpm.CDSCode -> schools.CDSCode" in out, "fkh_01 explicit FK line 누락"
    assert "FK: satscores.cds -> schools.CDSCode" in out, "fkh_01 explicit FK line 누락"
    # out-of-scope FK 는 포함되지 않아야 함 (schools.City -> cities.name)
    assert "cities" not in out, "fkh_01 out-of-scope FK 가 hint 위 포함됨"
    # inline 위치 — Columns 줄 뒤 hint line 가 직접 따라옴 확인 (가장 가까운 일치)
    lines = out.split("\n")
    found_inline = False
    for i, line in enumerate(lines):
        if line.strip().startswith("Columns:") and i + 1 < len(lines):
            if lines[i + 1].strip().startswith("FK:"):
                found_inline = True
                break
    assert found_inline, "fkh_01 inline 위치 — Columns 줄 직후 FK line 미발견"
    return out


def scenario_fkh_02():
    """explicit / prefix — schema block 앞 FK list."""
    f = make_filter("explicit", "prefix")
    out = f._serialize_schema(SUBGRAPH, "", metadata=METADATA)
    assert out.startswith("# Foreign Key Relationships"), (
        "fkh_02 prefix — header 가 schema 시작 위 없음"
    )
    assert "FK: frpm.CDSCode -> schools.CDSCode" in out
    assert "FK: satscores.cds -> schools.CDSCode" in out
    # prefix 이후 base schema 가 따라옴
    assert "# Table: schools" in out
    # inline 으로 들어가지 않음 — Columns 줄 직후 'FK:' line 가 없어야 함
    lines = out.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("Columns:") and i + 1 < len(lines):
            assert not lines[i + 1].strip().startswith("FK:"), (
                f"fkh_02 prefix 모드 위 Columns 줄 직후 FK line 발견 (line {i})"
            )
    return out


def scenario_fkh_03():
    """compact / inline — 각 src table 의 Columns 줄 뒤 '[FK ...=...]'."""
    f = make_filter("compact", "inline")
    out = f._serialize_schema(SUBGRAPH, "", metadata=METADATA)
    assert "[FK frpm.CDSCode=schools.CDSCode]" in out, "fkh_03 compact FK token 누락"
    assert "[FK satscores.cds=schools.CDSCode]" in out, "fkh_03 compact FK token 누락"
    # explicit 'FK:' token 은 등장하지 않아야 함 (format 분리 검증)
    assert "FK: " not in out, "fkh_03 compact 모드 위 explicit 'FK:' token 존재"
    # inline 위치 — Columns 줄 직후
    lines = out.split("\n")
    found_inline = False
    for i, line in enumerate(lines):
        if line.strip().startswith("Columns:") and i + 1 < len(lines):
            if lines[i + 1].strip().startswith("[FK "):
                found_inline = True
                break
    assert found_inline, "fkh_03 inline 위치 — Columns 줄 직후 [FK token 미발견"
    return out


def scenario_fkh_04():
    """compact / suffix — schema block 뒤 FK list."""
    f = make_filter("compact", "suffix")
    out = f._serialize_schema(SUBGRAPH, "", metadata=METADATA)
    assert out.endswith("]") or "[FK " in out.split("\n")[-1], (
        "fkh_04 suffix — 마지막 line 위 [FK token 미발견"
    )
    # FK header 가 schema 끝 위 존재
    last_section = out.rsplit("# Foreign Key Relationships", 1)
    assert len(last_section) == 2, "fkh_04 suffix — '# Foreign Key Relationships' header 누락"
    suffix_body = last_section[1]
    assert "[FK frpm.CDSCode=schools.CDSCode]" in suffix_body
    assert "[FK satscores.cds=schools.CDSCode]" in suffix_body
    # base schema 가 header 앞 retain
    assert "# Table: schools" in last_section[0], "fkh_04 suffix — base schema 가 사라짐"
    return out


# ---------------------------------------------------------------------------
# Invariance check — extracted_subgraph 자체 mutate 되지 않음
# ---------------------------------------------------------------------------
def check_invariance() -> None:
    """모든 cell 위 input subgraph 가 mutate 되지 않음 검증.

    extracted_nodes / extracted_edges 자체 변경 없음 → R/P/F1 동일 보장의
    structural 근거.
    """
    import copy

    snapshot = copy.deepcopy(SUBGRAPH)
    for (fmt, pos) in [
        ("none", "inline"),
        ("explicit", "inline"),
        ("explicit", "prefix"),
        ("compact", "inline"),
        ("compact", "suffix"),
    ]:
        f = make_filter(fmt, pos)
        _ = f._serialize_schema(SUBGRAPH, "", metadata=METADATA)
        if SUBGRAPH != snapshot:
            raise AssertionError(
                f"[invariance] subgraph mutated by ({fmt}, {pos}): "
                f"before={snapshot}, after={SUBGRAPH}"
            )


def main() -> int:
    scenarios = [
        ("fkh_00 (none/inline)", scenario_fkh_00),
        ("fkh_01 (explicit/inline)", scenario_fkh_01),
        ("fkh_02 (explicit/prefix)", scenario_fkh_02),
        ("fkh_03 (compact/inline)", scenario_fkh_03),
        ("fkh_04 (compact/suffix)", scenario_fkh_04),
    ]
    pass_count = 0
    fail_count = 0
    for label, fn in scenarios:
        try:
            out = fn()
            print(f"[PASS] {label}")
            pass_count += 1
        except Exception as e:
            print(f"[FAIL] {label}: {e}")
            traceback.print_exc()
            fail_count += 1

    # Invariance (R/P/F1 동일 보장 검증)
    try:
        check_invariance()
        print("[PASS] invariance — subgraph not mutated")
        pass_count += 1
    except Exception as e:
        print(f"[FAIL] invariance: {e}")
        traceback.print_exc()
        fail_count += 1

    print(f"\n[V7-W1 FKH smoke] {pass_count} pass, {fail_count} fail")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
