"""
LLM을 사용하여 DB 스키마의 각 edge에 대해 KG-style camelCase relation label을 생성하고 캐싱합니다.

사용법: python scripts/generate_triplet_relations.py
출력: data/processed/triplet_relations.json
"""
import os
import sys
import json
import sqlite3
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from openai import OpenAI

CLIENT = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
DEV_DIR = "data/raw/BIRD_dev"
TABLES_JSON = os.path.join(DEV_DIR, "dev_tables.json")
OUTPUT_PATH = "data/processed/triplet_relations.json"


def get_schema_info(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall() if r[0] != 'sqlite_sequence']

    columns = {}
    fks = []
    for t in tables:
        safe = t.replace("'", "''")
        cursor.execute(f"PRAGMA table_info('{safe}');")
        columns[t] = [row[1] for row in cursor.fetchall()]
        cursor.execute(f"PRAGMA foreign_key_list('{safe}');")
        for row in cursor.fetchall():
            fks.append({
                "from_table": t, "from_column": row[3],
                "to_table": row[2], "to_column": row[4]
            })
    conn.close()
    return tables, columns, fks


def load_table_descriptions(db_id):
    """dev_tables.json에서 NL 테이블명을 가져옵니다."""
    if not os.path.exists(TABLES_JSON):
        return {}
    with open(TABLES_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for db in data:
        if db['db_id'] == db_id:
            originals = db.get('table_names_original', [])
            nl_names = db.get('table_names', [])
            return {o: n for o, n in zip(originals, nl_names) if o.lower() != n.lower()}
    return {}


def load_column_descriptions(db_path):
    """database_description/*.csv에서 컬럼 설명을 로드합니다."""
    desc_dir = os.path.join(os.path.dirname(db_path), "database_description")
    if not os.path.isdir(desc_dir):
        return {}
    result = {}
    for csv_file in os.listdir(desc_dir):
        if not csv_file.endswith('.csv'):
            continue
        table_name = csv_file[:-4]
        csv_path = os.path.join(desc_dir, csv_file)
        try:
            with open(csv_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    orig = row.get('original_column_name', '').strip()
                    desc = row.get('column_description', '').strip()
                    if orig and desc:
                        result[f"{table_name}.{orig}"] = desc
        except Exception:
            pass
    return result


def build_prompt(db_id, tables, columns, fks, table_nl, col_descs):
    """DB 전체에 대한 relation 생성 프롬프트를 구성합니다."""

    # Table descriptions
    table_lines = []
    for t in tables:
        nl = table_nl.get(t, "")
        if nl:
            table_lines.append(f"- {t}: {nl}")
        else:
            table_lines.append(f"- {t}")

    # Build edge list
    edges = []
    idx = 1

    # 1) Column-Table edges (belongs_to direction: table -> column)
    for t in tables:
        for c in columns[t]:
            full = f"{t}.{c}"
            desc = col_descs.get(full, "")
            desc_hint = f" (desc: {desc})" if desc else ""
            edges.append(f"{idx}. ({t}, ???, {t}.{c}){desc_hint}")
            idx += 1

    # 2) FK edges
    for fk in fks:
        src = f"{fk['from_table']}.{fk['from_column']}"
        dst = f"{fk['to_table']}.{fk['to_column']}"
        edges.append(f"{idx}. ({src}, ???, {dst}) [FK relationship]")
        idx += 1

    # 3) Table-Table edges (unique pairs from FKs)
    seen_pairs = set()
    for fk in fks:
        pair = tuple(sorted([fk['from_table'], fk['to_table']]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            edges.append(f"{idx}. ({pair[0]}, ???, {pair[1]}) [table join relationship]")
            idx += 1

    edge_block = "\n".join(edges)

    prompt = f"""You are a knowledge graph relation labeler for database schemas.

Given a database schema, generate a camelCase relation label for each edge.
The relation should capture the SEMANTIC meaning of the relationship, not just structural (avoid generic labels like hasColumn, belongsTo, isPartOf).

Rules:
- Output format: (subject, relation, object) per line, numbered to match input
- Relation must be a single camelCase token (e.g., identifiesSchoolIn, recordsTestScoreOf)
- Focus on WHAT the column means in the context of the table
- For FK relationships, describe what the join means semantically
- For table-to-table, describe what shared information connects them
- Do NOT output anything other than the numbered triplets

Database: {db_id}
Tables:
{chr(10).join(table_lines)}

Edges to label:
{edge_block}

Output only the numbered triplets:"""

    return prompt, idx - 1


def parse_response(response_text, expected_count):
    """LLM 응답에서 (subject, relation, object) triplet을 파싱합니다."""
    triplets = {}
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        # "1. (subject, relation, object)" 형태 파싱
        try:
            # 번호 제거
            dot_idx = line.index('.')
            content = line[dot_idx + 1:].strip()
            # 괄호 제거
            content = content.strip('()')
            parts = [p.strip() for p in content.split(',', 2)]
            if len(parts) == 3:
                idx = int(line[:dot_idx])
                triplets[idx] = {
                    "subject": parts[0],
                    "relation": parts[1],
                    "object": parts[2].rstrip(')')
                }
        except (ValueError, IndexError):
            continue
    return triplets


def _call_llm(prompt):
    """LLM 호출. 입력 토큰에 맞춰 max_tokens를 자동 조정합니다."""
    # Rough estimate: 1 token ≈ 4 chars
    est_input_tokens = len(prompt) // 3
    max_tokens = min(4096, max(1024, 8192 - est_input_tokens - 200))

    response = CLIENT.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def _build_batch_prompt(db_id, table_lines, edge_batch, start_idx):
    """edge 일부에 대한 프롬프트를 구성합니다."""
    numbered = []
    for i, edge_text in enumerate(edge_batch):
        numbered.append(f"{start_idx + i}. {edge_text}")
    edge_block = "\n".join(numbered)

    prompt = f"""You are a knowledge graph relation labeler for database schemas.

Given a database schema, generate a camelCase relation label for each edge.
The relation should capture the SEMANTIC meaning of the relationship, not just structural (avoid generic labels like hasColumn, belongsTo, isPartOf).

Rules:
- Output format: (subject, relation, object) per line, numbered to match input
- Relation must be a single camelCase token (e.g., identifiesSchoolIn, recordsTestScoreOf)
- Focus on WHAT the column means in the context of the table
- For FK relationships, describe what the join means semantically
- For table-to-table, describe what shared information connects them
- Do NOT output anything other than the numbered triplets

Database: {db_id}
Tables:
{chr(10).join(table_lines)}

Edges to label:
{edge_block}

Output only the numbered triplets:"""
    return prompt


def generate_for_db(db_id):
    db_path = os.path.join(DEV_DIR, "dev_databases", db_id, f"{db_id}.sqlite")
    tables, columns, fks = get_schema_info(db_path)
    table_nl = load_table_descriptions(db_id)
    col_descs = load_column_descriptions(db_path)

    # Table description lines (shared across batches)
    table_lines = []
    for t in tables:
        nl = table_nl.get(t, "")
        table_lines.append(f"- {t}: {nl}" if nl else f"- {t}")

    # Build all edge texts
    all_edge_texts = []
    edge_metas = []  # track type and data for each edge

    for t in tables:
        for c in columns[t]:
            full = f"{t}.{c}"
            desc = col_descs.get(full, "")
            desc_hint = f" (desc: {desc})" if desc else ""
            all_edge_texts.append(f"({t}, ???, {t}.{c}){desc_hint}")
            edge_metas.append(("column_table", t, f"{t}.{c}"))

    for fk in fks:
        src = f"{fk['from_table']}.{fk['from_column']}"
        dst = f"{fk['to_table']}.{fk['to_column']}"
        all_edge_texts.append(f"({src}, ???, {dst}) [FK relationship]")
        edge_metas.append(("fk", src, dst))

    seen_pairs = set()
    for fk in fks:
        pair = tuple(sorted([fk['from_table'], fk['to_table']]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            all_edge_texts.append(f"({pair[0]}, ???, {pair[1]}) [table join relationship]")
            edge_metas.append(("table_table", pair[0], pair[1]))

    total = len(all_edge_texts)
    print(f"[{db_id}] Generating relations for {total} edges...")

    # Batch: max 80 edges per call to stay within context limit
    BATCH_SIZE = 80
    all_triplets = {}

    for batch_start in range(0, total, BATCH_SIZE):
        batch = all_edge_texts[batch_start:batch_start + BATCH_SIZE]
        start_idx = batch_start + 1
        prompt = _build_batch_prompt(db_id, table_lines, batch, start_idx)
        text = _call_llm(prompt)
        parsed = parse_response(text, len(batch))

        # Re-map: LLM might restart numbering from 1 in each batch
        # Detect if parsed keys start from 1 instead of start_idx
        if parsed and min(parsed.keys()) == 1 and start_idx > 1:
            remapped = {k + batch_start: v for k, v in parsed.items()}
            parsed = remapped

        all_triplets.update(parsed)

        if total > BATCH_SIZE:
            print(f"  batch {batch_start//BATCH_SIZE + 1}: parsed {len(parsed)}/{len(batch)}")

    # Build structured result
    result = {"db_id": db_id, "edges": []}
    defaults = {"column_table": "hasAttribute", "fk": "references", "table_table": "relatesTo"}

    for i, (etype, subj, obj) in enumerate(edge_metas):
        tri = all_triplets.get(i + 1, {})
        result["edges"].append({
            "type": etype,
            "subject": subj,
            "object": obj,
            "relation": tri.get("relation", defaults[etype]),
        })

    parsed_count = sum(1 for e in result["edges"] if e["relation"] not in defaults.values())
    print(f"  → Parsed {parsed_count}/{total} relations from LLM")

    return result


def main():
    # Get all dev DB ids
    with open(os.path.join(DEV_DIR, 'dev.json')) as f:
        dev = json.load(f)
    db_ids = sorted(set(d['db_id'] for d in dev))

    all_results = {}
    for db_id in db_ids:
        result = generate_for_db(db_id)
        all_results[db_id] = result

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    total_edges = sum(len(r["edges"]) for r in all_results.values())
    print(f"\nDone! {total_edges} triplet relations saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
