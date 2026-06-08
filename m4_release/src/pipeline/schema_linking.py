import time
import torch
from typing import Dict, Any, List
from modules import build  # 마스터 스위치 (registry 기반 모듈 조립)
from utils.logger import get_logger

logger = get_logger(__name__)


class SchemaLinkingPipeline:
    """M4 anchor end-to-end Graph-RAG Schema Linking 파이프라인.

    Stages:
        [1] Builder      EnrichedHeteroGraphBuilder  — 스키마 → 이종 그래프
        [2] Encoder      LocalPLMEncoder             — NLQ 임베딩 (all-MiniLM-L6-v2)
        [3] Cosine       q ↔ node 코사인 유사도       — selector ensemble 의 cos 항
        [4] Selector     EnsembleSelector            — s_ens = α·cos + (1-α)·GAT, top_k
        [5] Extractor    MSTPCSTUnionExtractor       — MST(Kruskal) ∪ Basic PCST
        [6] Filter       BidirectionalFilter         — LLM 정제 (Forward + Backward)
        [7] Generator    LLMSQLGenerator             — 최종 스키마 → SQL (default-DDL)

    NOTE (M4 reproducibility 패키지):
        전체 연구 저장소 버전에는 projection / Wave 11 serializer / F5 extraction-retry /
        Wave 7 stage-wise EX / SGBE raw-score plumbing 등 비-M4 ablation 분기가 있으나,
        M4 anchor config 는 이들을 사용하지 않으므로 본 패키지에서는 제거했다.
    """

    def __init__(self, config: Dict[str, Any]):
        logger.info("🚀 Assembling the M4 Schema Linking Pipeline from Config...")

        self.db_dir = config['paths']['data_dir']
        self.auto_join_keys = config.get('post_processing', {}).get('auto_join_keys', False)

        self.builder = build("builder", config['graph_builder'])
        self.encoder = build("encoder", config['nlq_encoder'])
        self.selector = build("selector", config['seed_selector'])
        self.extractor = build("extractor", config['connectivity_extractor'])
        self.filter = build("filter", config['filter'])

        if config['sql_generator']['enabled']:
            self.generator = build("generator", config['sql_generator'])
        else:
            self.generator = None

        logger.info("✅ Pipeline assembly completed successfully.")

    def run(self, db_id: str, query: str, evidence: str = "") -> Dict[str, Any]:
        """단일 질의(Query) 처리.

        evidence: BIRD-dev `external_knowledge` 필드. LLMSQLGenerator 의 SQL gen prompt
        [External Knowledge] 섹션에 삽입된다.
        """
        logger.debug(f"[{db_id}] Processing Query: '{query}'")
        execution_times: Dict[str, float] = {}

        # --- Stage 1: Build / Load Graph ---
        t_start = time.perf_counter()
        graph_data, metadata = self.builder.build(db_id=db_id, db_dir=self.db_dir)
        # Selector/Extractor 가 DB 별 정책을 적용할 수 있도록 db_id 주입.
        if isinstance(metadata, dict):
            metadata.setdefault("db_id", db_id)
        execution_times["graph_build"] = time.perf_counter() - t_start

        # --- Stage 2: Encode NLQ ---
        t_start = time.perf_counter()
        encoded_output = self.encoder.encode([query])
        q_embs = encoded_output[0] if isinstance(encoded_output, tuple) else encoded_output
        logger.debug(f"q_embs: {q_embs.shape}")
        execution_times["encoding_nlq"] = time.perf_counter() - t_start

        # --- Stage 3: Cosine similarity (raw cosine → EnsembleSelector 의 cos 항) ---
        t_start = time.perf_counter()
        table_embs = graph_data['table'].x
        col_embs = graph_data['column'].x
        embs_list = [table_embs, col_embs]
        try:
            fk_embs = graph_data['fk_node'].x
            if fk_embs.size(0) > 0:
                embs_list.append(fk_embs)
        except (KeyError, AttributeError):
            pass
        node_embs = torch.cat(embs_list, dim=0).to('cpu')
        try:
            node_scores = torch.nn.functional.cosine_similarity(q_embs.to('cpu'), node_embs)
        except RuntimeError:
            logger.debug("Cosine similarity skipped (dimension mismatch).")
            node_scores = None
        execution_times["cosine"] = time.perf_counter() - t_start

        # --- Stage 4: Seed Selection (EnsembleSelector) ---
        t_start = time.perf_counter()
        if node_scores is not None:
            candidates_idx = list(range(len(node_scores)))
        else:
            candidates_idx = list(range(len(metadata.get('node_metadata', {}))))

        seeds = self.selector.select(
            scores=node_scores,
            candidates=candidates_idx,
            question=query,
            graph_data=graph_data,
            metadata=metadata,
        )

        # PCST 로 넘길 node score: selector 가 노출한 ensemble score 우선.
        if hasattr(self.selector, 'latest_scores') and self.selector.latest_scores:
            scores_list = self.selector.latest_scores
        elif node_scores is not None:
            scores_list = node_scores.squeeze().tolist()
        else:
            scores_list = [1.0] * len(candidates_idx)
        execution_times["seed_selection"] = time.perf_counter() - t_start

        # --- Stage 5: Subgraph Extraction (MST ∪ PCST) ---
        t_start = time.perf_counter()
        # edge-prize extractor 를 위한 query embedding 전달 (해당 extractor 만 사용).
        if 'edge_embeddings' in metadata and q_embs is not None:
            metadata['query_embedding'] = q_embs.squeeze(0).cpu()

        selected_nodes_idx, selected_edges = self.extractor.extract(
            graph_data=metadata,
            node_scores=scores_list,
            seed_nodes=seeds,
        )

        # index → "table.column" 번역
        subgraph_dict: Dict[str, List[str]] = {}
        for n_id in selected_nodes_idx:
            n_id_key = int(n_id) if isinstance(n_id, (int, float)) or (
                isinstance(n_id, str) and n_id.isdigit()) else n_id
            name = metadata['node_metadata'].get(n_id_key, str(n_id_key))
            if "." in name:
                tbl, col = name.split(".", 1)
                subgraph_dict.setdefault(tbl, []).append(col)
            else:
                subgraph_dict.setdefault(name, [])

        # JOIN key 자동 포함: 2개 이상 테이블 선택 시 FK 컬럼 보강.
        if self.auto_join_keys and len(subgraph_dict) >= 2:
            node_meta = metadata.get('node_metadata', {})
            added_keys = []
            for idx, name in node_meta.items():
                if '->' not in str(name):
                    continue
                parts = str(name).split('->')
                if len(parts) != 2:
                    continue
                src, dst = parts[0].strip(), parts[1].strip()
                src_tbl = src.split('.')[0] if '.' in src else src
                dst_tbl = dst.split('.')[0] if '.' in dst else dst
                src_col = src.split('.', 1)[1] if '.' in src else None
                dst_col = dst.split('.', 1)[1] if '.' in dst else None
                if src_tbl in subgraph_dict and dst_tbl in subgraph_dict:
                    if src_col and src_col not in subgraph_dict.get(src_tbl, []):
                        subgraph_dict[src_tbl].append(src_col)
                        added_keys.append(src)
                    if dst_col and dst_col not in subgraph_dict.get(dst_tbl, []):
                        subgraph_dict[dst_tbl].append(dst_col)
                        added_keys.append(dst)
            if added_keys:
                logger.debug(f"[AutoJoinKeys] Added {len(added_keys)} FK columns: {added_keys}")
        execution_times["subgraph_extraction"] = time.perf_counter() - t_start

        # --- Stage 6: Filtering (BidirectionalFilter) ---
        t_start = time.perf_counter()
        final_result = self.filter.refine(
            query=query,
            subgraph=subgraph_dict,
            db_id=db_id,
            metadata=metadata,
        )
        execution_times["filtering"] = time.perf_counter() - t_start
        logger.info(
            f"[Filter] status={final_result.get('status')} "
            f"nodes={len(final_result.get('final_nodes', []))} "
            f"time={execution_times['filtering']:.2f}s | "
            f"reasoning={(final_result.get('reasoning') or '')[:400]}"
        )

        # --- Stage 7: SQL Generation (default-DDL) ---
        generated_sql = ""
        t_start = time.perf_counter()
        if self.generator is not None:
            generated_sql = self.generator.generate(
                query=query, subgraph=subgraph_dict, evidence=evidence,
            )
            logger.debug(f"Generated SQL: {generated_sql}")
        execution_times["sql_generation"] = time.perf_counter() - t_start
        final_result["generated_sql"] = generated_sql

        # --- Assemble result ---
        final_result["execution_time"] = execution_times
        final_result["raw_scores"] = scores_list
        final_result["node_names"] = [
            metadata['node_metadata'].get(i, str(i)) for i in range(len(scores_list))
        ]

        builder_info = getattr(self.builder, 'last_info', None) or metadata.get('builder_info')
        if builder_info:
            final_result["builder_info"] = dict(builder_info)
        selector_info = getattr(self.selector, 'last_info', None)
        if selector_info:
            final_result["selector_info"] = dict(selector_info)
        extractor_info = getattr(self.extractor, 'last_info', None)
        if extractor_info:
            final_result["extractor_info"] = dict(extractor_info)
        filter_info = getattr(self.filter, 'last_info', None) or final_result.get("filter_info")
        if filter_info:
            final_result["filter_info"] = dict(filter_info)

        return final_result
