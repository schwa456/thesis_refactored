import pathlib
import json
import traceback
import torch
from typing import List, Dict, Any

import numpy as np
from sqlalchemy import text
from langchain_community.utilities import SQLDatabase
from langchain_huggingface import HuggingFacePipeline
from sentence_transformers import SentenceTransformer, util
from thefuzz import process

import sys
sys.path.append('/home/sql/people/hyeonjin/M-Schema/')
from schema_engine import SchemaEngine

from config import *
from generator import get_db_id, get_out_of_box_llm

# --- 1. 유틸리티 및 모델 로딩 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(EMBEDDING_LLM_ID, device=device)

def get_db_id(db: SQLDatabase) -> str:
    path = db._engine.url.database
    path_obj = pathlib.Path(path)
    db_id = path_obj.stem
    return db_id

# --- 2. Multi-path Retrieval 상세 구현 ---
class SchemaFilter:
    def __init__(self, 
                 question: str, 
                 evidence: str, 
                 db: SQLDatabase, 
                 llm: HuggingFacePipeline, 
                 iteration: int, 
                 num_cols: int, 
                 num_vals: int
                ):
        
        self.question = question
        self.evidence = evidence
        self.db = db
        self.llm = llm
        self.iteration = iteration
        self.num_cols = num_cols
        self.num_vals = num_vals
    
    def _extract_keywords_from_llm(self) -> List[str]:
        print(">> Schema Selector: 키워드 추출 중...")

        llm = self.llm
        question = self.question
        evidence = self.evidence

        prompt = f"""
        Extract critical keywords from the following question and evidence for database schema retrieval. 
        Give me the keywords in comma separated format.
        
        Question: 
        {question}
        
        Evidence: 
        {evidence}
        """

        response = llm.invoke(prompt)
        kwyeords_str = response if isinstance(response, str) else response.content

        if ":" in keywords_str:
            keywords_str = keywords_str.split(":")[:-1]
        
        keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]

        print(f">> 추출된 키워드: {keywords}")

        return keywords


def _extract_keywords_from_llm(llm: HuggingFacePipeline, question: str, evidence: str) -> List[str]:
    print(">> Schema Selector: 키워드 추출 중...")

    prompt = f"""
    Extract critical keywords from the following question and evidence for database schema retrieval. 
    Give me the keywords in comma separated format.
    
    Question: 
    {question}
    
    Evidence: 
    {evidence}
    """

    response = llm.invoke(prompt)
    keywords_str = response if isinstance(response, str) else response.content

    if ":" in keywords_str:
        keywords_str = keywords_str.split(":")[-1]

    keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
    
    print(f">> 추출된 키워드: {keywords}")

    return keywords

def _multi_path_retrieval(db: SQLDatabase, keywords: List[str], question: str, evidence: str, top_k: int) -> Dict[str, Any]:
    print(">> Schema Selector: Multi-path Retrieval 실행 중...")
    inspector = db._inspector
    all_tables = db.get_usable_table_names()

    # --- 데이터 및 임베딩 준비 ---
    q_and_e_text = question + " " + evidence
    q_and_e_emb = embedding_model.encode(q_and_e_text, convert_to_tensor=True)
    keyword_embs = embedding_model.encode(keywords, convert_to_tensor=True)

    table_meta = {tbl: f"Table {tbl}" for tbl in all_tables}
    table_embs = {tbl: embedding_model.encode(meta, convert_to_tensor=True) for tbl, meta in table_meta.items()}

    # --- 컬럼 스코어링 (논문 수식 1) ---
    column_meta = {}
    for tbl in all_tables:
        column_meta[tbl] = [f"Column {col['name']} in table {tbl}" for col in inspector.get_columns(tbl)]
    column_embs = {tbl: embedding_model.encode(metas, convert_to_tensor=True) for tbl, metas in column_meta.items()}

    column_scores = {}
    for tbl_idx, tbl in enumerate(all_tables):
        # 1. <V_{Q||E}, V_{Tab(c_j)}> 계산
        score_qe_table = util.cos_sim(q_and_e_emb, table_embs[tbl]).item()

        for col_idx, col_name in enumerate([c['name'] for c in inspector.get_columns(tbl)]):
            # 2. <V_{k_i}, V_{c_j}> 계산
            col_emb = column_embs[tbl][col_idx]
            scores_kw_col = util.cos_sim(keyword_embs, col_emb)
            max_score_kw_col = torch.max(scores_kw_col).item()

            # 3. 최종 점수
            final_score = score_qe_table * max_score_kw_col
            column_scores[f"{tbl}.{col_name}"] = final_score
    
    # --- 상위 K개 컬럼 선택 ---
    sorted_columns = sorted(column_scores.items(), key=lambda item: item[1], reverse=True)
    top_columns = [item[0] for item in sorted_columns[:top_k]]

    retrieved_schema = {}
    for col_fullname in top_columns:
        tbl, col = col_fullname.split('.')
        if tbl not in retrieved_schema:
            retrieved_schema[tbl] = []
        retrieved_schema[tbl].append(col)
    
    # --- 값 검색 (Value Retrieval) - Fuzzy-matching으로 간소화 ---
    for tbl, cols in retrieved_schema.items():
        for col in cols:
            try:
                # DB에서 실제 값 샘플 가져오기
                with db._engine.connect() as connection:
                    result = connection.execute(text(f"SELECT DISTINCT `{col}` FROM `{tbl}` LIMIT 100"))
                    values = [str(row[0]) for row in result if row[0] is not None]
                
                # 키워드와 가장 유사한 값 찾기
                for keyword in keywords:
                    matches = process.extractBetas(keyword, values, score_cutoff=80, limit=2)
                    if matches:
                        print(f">> 값 검색: Keyword '{keyword}' -> Table '{tbl}.{col}'에서 유사 값 발견: {[m[0] for m in matches]}")
            except Exception:
                continue
    print(f">> Schema Selector: 1차 필터링된 스키마(S^rtrv) 생성 완료: {retrieved_schema}")
    return retrieved_schema

# --- 3. Iterative Column Selection 상세 구현 ---

def _select_columns_with_llm(llm: HuggingFacePipeline, schema_to_select_from: Dict, question: str, evidence: str) -> Dict:
    print(">> Schema Selector: LLM으로 컬럼 선택 중...")
    example_tables = list(schema_to_select_from.keys())[:2] # 최대 2개 테이블로 예시 구성
    example_json_obj = {}
    if len(example_tables) > 0:
        example_json_obj[example_tables[0]] = schema_to_select_from[example_tables[0]][:2] # 첫 테이블의 2개 컬럼
    if len(example_tables) > 1:
        example_json_obj[example_tables[1]] = schema_to_select_from[example_tables[1]][:1] # 두 번째 테이블의 1개 컬럼
        
    example_json_str = json.dumps(example_json_obj)

    # 2. 스키마 문자열 생성 및 프롬프트 수정
    schema_str = "\n".join([f"Table {tbl}: {', '.join(cols)}" for tbl, cols in schema_to_select_from.items()])
    
    prompt = f"""Given the database schema below:
            ---
            {schema_str}
            ---
            Select the most relevant tables and columns to answer the question: '{question}'. Evidence: '{evidence}'.

            IMPORTANT: 
            1. Your output MUST be a single valid JSON object.
            2. ONLY use the table and column names provided in the schema above. Do not invent new names.

            For example, your output should look like this:
            {example_json_str}
            """

    response = llm.invoke(prompt)
    response_str = response if isinstance(response, str) else response.content

    try:
        json_part = response_str[response_str.find('{'):response_str.find('}')+1]
        selected_columns = json.loads(json_part)
        return selected_columns
    except (json.JSONDecodeError, IndexError):
        print("!! LLM 컬럼 선택 결과 파싱 실패 !!")
        return {}
    
def _identify_and_add_keys(db: SQLDatabase, schema_dict: Dict) -> Dict:
    inspector = db._inspector
    tables_in_schema = list(schema_dict.keys())

    for table in tables_in_schema:
        pk_info = inspector.get_pk_constraint(table).get('constrained_columns', [])
        for pk_col in pk_info:
            if pk_col not in schema_dict[table]:
                schema_dict[table].append(pk_col)
        
        fks = inspector.get_foreign_keys(table)
        for fk in fks:
            if fk['referred_table'] in tables_in_schema:
                for col in fk['constrained_columns']:
                    if col not in schema_dict[table]:
                        schema_dict[table].append(col)
                
                for col in fk['referred_columns']:
                    if col not in schema_dict[fk['referred_table']]:
                        schema_dict[fk['referred_table']].append(col)
    
    return schema_dict
    
def _iterative_column_selection(db: SQLDatabase, llm: HuggingFacePipeline, retrieved_schema: Dict, question: str, evidence: str, max_iteration: int) -> List[Dict]:
    print(">> Schema Selector: Iterative Column Selection 시작...")
    final_schemas = []
    schema_for_iteration = retrieved_schema.copy()
    inspector = db._inspector
    
    for i in range(max_iteration):
        print(f">> Schema Selector: 반복 {i+1}/{max_iteration}")
        if not any(schema_for_iteration.values()):
            print("!! 선택할 스키마가 더 이상 없어 반복을 중단합니다. !!")
            break

        selected_cols_this_iter = _select_columns_with_llm(llm, schema_for_iteration, question, evidence)
        if not selected_cols_this_iter:
            continue

        schema_with_keys = _identify_and_add_keys(db, selected_cols_this_iter)
        final_schemas.append(schema_with_keys)
        print(f">> Schema Selector: 스키마 후보 {i+1} 생성 완료: {schema_with_keys}")

        # 다음 반복을 위해 선택된 컬럼 제거
        for table, cols in selected_cols_this_iter.items():
            if table in schema_for_iteration:
                # PK는 다음 선택을 위해 남겨둘 수 있음
                pk_cols = set(inspector.get_pk_constraint(table).get('constrained_columns', []))
                cols_to_remove = set(cols) - pk_cols
                schema_for_iteration[table] = [c for c in schema_for_iteration[table] if c not in cols_to_remove]
                if not schema_for_iteration[table]:
                    del schema_for_iteration[table]
    
    return final_schemas

# --- 4. 최종 M-Schema 포맷팅 및 메인 함수 ---

# src/schema_selector.py 내부에 위치

def _format_to_mschema(db: SQLDatabase, schema_dict: Dict) -> str:
    """
    최종 선택된 스키마 딕셔너리를 M-Schema 형식의 문자열로 변환합니다.
    """
    try:
        db_id = get_db_id(db)
        inspector = db._inspector
        output = [f"【DB_ID】 {db_id}", "【Schema】"]
        all_fks = []

        # schema_dict에서 테이블 이름을 가져옴
        for table_name, selected_columns in schema_dict.items():
            # --- 1. 테이블 코멘트 조회 (NotImplementedError 처리) ---
            table_comment = ""
            try:
                table_comment_dict = inspector.get_table_comment(table_name)
                table_comment = table_comment_dict.get('text', '')
            except NotImplementedError:
                pass

            table_header = f"# Table: {table_name}"
            if table_comment:
                table_header += f", {table_comment}"
            
            table_output = [table_header]
            field_lines = []
            
            # --- 2. 전체 컬럼 정보 중 '선택된' 컬럼만 처리 ---
            all_table_columns_info = inspector.get_columns(table_name)
            pk_info = inspector.get_pk_constraint(table_name).get('constrained_columns', [])

            # 전체 컬럼 정보에서, schema_dict에 있는 컬럼만 골라서 처리
            for col_info in all_table_columns_info:
                col_name = col_info['name']
                if col_name not in selected_columns:
                    continue

                col_type = str(col_info['type'])
                field_line_parts = [f"{col_name}:{col_type.upper()}"]

                col_comment = col_info.get('comment', '')
                if col_comment:
                    field_line_parts.append(col_comment)
                
                if col_name in pk_info:
                    field_line_parts.append("Primary Key")

                # DB에서 값 예시 가져오기
                try:
                    with db._engine.connect() as connection:
                        query = text(f"SELECT DISTINCT `{col_name}` FROM `{table_name}` WHERE `{col_name}` IS NOT NULL LIMIT 3")
                        result = connection.execute(query)
                        examples = [str(row[0]) for row in result if row[0] is not None]
                        if examples:
                            field_line_parts.append(f"Examples: [{', '.join(examples)}]")
                except Exception:
                    pass

                field_lines.append("(" + ", ".join(field_line_parts) + ")")
            
            table_output.append("[\n" + ",\n".join(field_lines) + "\n]")
            output.append("\n".join(table_output))

            # --- 3. 관련된 FK만 추가 ---
            fks = inspector.get_foreign_keys(table_name)
            for fk in fks:
                constrained_col = fk['constrained_columns'][0]
                referred_table = fk['referred_table']
                referred_col = fk['referred_columns'][0]

                # FK의 양쪽 컬럼이 모두 현재 스키마에 포함되어 있을 때만 추가
                if (referred_table in schema_dict and
                    constrained_col in selected_columns and
                    referred_col in schema_dict[referred_table]):
                    fk_str = f"{table_name}.{constrained_col}={referred_table}.{referred_col}"
                    all_fks.append(fk_str)
        
        if all_fks:
            output.append("【Foreign keys】")
            for fk_str in sorted(list(set(all_fks))):
                output.append(fk_str)

        return "\n".join(output)

    except Exception as e:
        # <<< 4. 예외 발생 시, 타입에 맞는 문자열 반환 >>>
        tb_str = traceback.format_exc()
        print(f"\n!!! M-SCHEMA 생성 중 진짜 오류 발생 !!!\n{tb_str}")
        # 오류 발생 시 빈 스키마 정보 또는 에러 메시지를 '문자열'로 반환
        return f"# ERROR: M-Schema 생성 중 오류 발생 - {e}"

    

def schema_selection(db: SQLDatabase, question: str, evidence: str) -> List[str]:
    llm = get_out_of_box_llm()
    
    # 1. Multi-path Retrieval
    keywords = _extract_keywords_from_llm(llm, question, evidence)
    retrieved_schema = _multi_path_retrieval(db, keywords, question, evidence, top_k=20)

    # 2. Iterative Column Selection
    list_of_schema_dicts = _iterative_column_selection(db, llm, retrieved_schema, question, evidence, max_iteration=2)

    # 3. M-Schema Formatting
    final_schema_strings = [_format_to_mschema(db, schema_dict) for schema_dict in list_of_schema_dicts]

    if not final_schema_strings:
        print("!! 최종 스키마 생성 실패. 전체 스키마를 사용합니다. (Fallback) !!")
        return [db.get_table_info()]
    
    return final_schema_strings

if __name__ == "__main__":
    question = "전체 문의 현황을 상태별로 푸른 테마에서 범례를 왼쪽에 두고, 반원형 파이 차트로 시각화해줘"
    evidence = ""
    llm = get_out_of_box_llm()
    iteration = 2
    num_cols = 10
    num_vals = 10

    schema_filter = SchemaFilter(
        question=question,
        evidence=evidence,
        llm=llm,
        iteration=iteration,
        num_cols=num_cols,
        num_vals=num_vals
    )

    keywords = schema_filter._extract_keywords_from_llm()
    print(keywords)