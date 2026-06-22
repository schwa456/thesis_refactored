#!/usr/bin/env python3
"""HISTORY 전체에서 메트릭 표 행을 (섹션 헤더 + 표 헤더 + 행) 구조로 덤프.
era/stage 추론 보조용. 출력: outputs/analysis/_ablation_rows_dump.txt
"""
import re, os

H = "/home/hyeonjin/thesis_refactored/EXPERIMENT_HISTORY.md"
OUT = "/home/hyeonjin/thesis_refactored/outputs/analysis/_ablation_rows_dump.txt"

lines = open(H).readlines()
sec2 = sec3 = ""          # 최근 ## / ### 헤더
tbl_header = ""           # 최근 표 헤더 행
num_re = re.compile(r"0\.[0-9]{3,4}")
out = []

for i, ln in enumerate(lines):
    s = ln.rstrip("\n")
    if s.startswith("## ") and not s.startswith("###"):
        sec2 = s[3:].strip(); sec3 = ""
    elif s.startswith("### "):
        sec3 = s[4:].strip()
    elif s.startswith("####"):
        sec3 = s.lstrip("#").strip()
    # 표 행 판정
    if s.startswith("|") and s.count("|") >= 3:
        cells = [c.strip() for c in s.strip("|").split("|")]
        # 헤더/구분선 감지
        if set("".join(cells)) <= set("-: "):
            continue  # separator
        nnum = len(num_re.findall(s))
        # 헤더 행: 숫자 거의 없고 R/P/F1/EX/recall 등 포함
        low = s.lower()
        if nnum <= 1 and any(k in low for k in ["r |", "recall", "precision", "f1", " ex", "| p ", "δ", "cell", "wave", "metric", "아이템", "baseline", "config"]):
            tbl_header = " | ".join(cells)
            continue
        # 데이터 행: 숫자 2+ 개
        if nnum >= 2:
            out.append(f"[L{i+1}] §2={sec2[:60]} | §3={sec3[:55]}\n   HDR: {tbl_header[:120]}\n   ROW: {' | '.join(cells)}\n")

open(OUT, "w").write("\n".join(out))
print(f"rows={len(out)} → {OUT}")
