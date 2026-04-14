import time
import torch
from typing import Dict, Any, List
from modules import build  # 마스터 스위치!
from utils.logger import get_logger

logger = get_logger(__name__)

class SchemaLinkingPipeline:
    """
    E2E Graph RAG 기반 Schema Linking 파이프라인.
    Config에 명시된 6개의 모듈을 조립하여 데이터를 순차적으로 통과시킵니다.
    """
    def __init__(self, config: Dict[str, Any]):
        logger.info("🚀 Assembling the Schema Linking Pipeline from Config...")

        self.db_dir = config['paths']['data_dir']
        self.auto_join_keys = config.get('post_processing', {}).get('auto_join_keys', False)

        self.builder = build("builder", config['graph_builder'])
        self.encoder = build("encoder", config['nlq_encoder'])
        
        self.use_projection = config.get('projection', {}).get('enabled', False)
        if self.use_projection:
            self.projector = build("projector", config['projection'])
            
        self.selector = build("selector", config['seed_selector'])
        self.extractor = build("extractor", config['connectivity_extractor'])
        self.filter = build("filter", config['filter'])

        # [F5] Extraction retry — re-run Extractor with relaxed params when
        # the Filter returns "Unanswerable" (or too few nodes). Max K retries.
        retry_cfg = config.get('extraction_retry', {}) or {}
        self.retry_enabled = bool(retry_cfg.get('enabled', False))
        self.retry_max = int(retry_cfg.get('max_retries', 2))
        self.retry_min_nodes = int(retry_cfg.get('min_nodes', 2))
        self.retry_strategies = retry_cfg.get(
            'strategies', ['widen', 'steiner']
        )

        if config['sql_generator']['enabled']:
            self.generator = build("generator", config['sql_generator'])
        else:
            self.generator = None
        
        logger.info("✅ Pipeline assembly completed successfully.")

    def _apply_retry_strategy(self, strategy: str) -> Dict[str, Any]:
        """Relax Extractor parameters in-place. Returns snapshot for restore."""
        ext = self.extractor
        snapshot = {}
        if strategy == "widen":
            for attr in ("base_cost", "belongs_to_cost", "fk_cost"):
                if hasattr(ext, attr):
                    snapshot[attr] = getattr(ext, attr)
                    setattr(ext, attr, max(0.0, getattr(ext, attr) * 0.5))
        elif strategy == "steiner":
            if hasattr(ext, "backbone_bonus"):
                snapshot["backbone_bonus"] = getattr(ext, "backbone_bonus")
                setattr(ext, "backbone_bonus", getattr(ext, "backbone_bonus") + 0.2)
            if hasattr(ext, "base_cost"):
                snapshot["base_cost"] = getattr(ext, "base_cost")
                setattr(ext, "base_cost", max(0.0, getattr(ext, "base_cost") * 0.7))
        return snapshot

    def _restore_extractor_params(self, snapshot: Dict[str, Any]) -> None:
        for k, v in snapshot.items():
            setattr(self.extractor, k, v)

    def run(self, db_id: str, query: str) -> Dict[str, Any]:
        """단일 질의(Query) 처리 파이프라인"""
        logger.debug(f"[{db_id}] Processing Query: '{query}'")

        execution_times = {}

        # Stage 1: Build / Load Graph
        logger.debug(f"Building Graph")
        t_start = time.perf_counter()
        graph_data, metadata = self.builder.build(db_id=db_id, db_dir=self.db_dir)
        logger.debug(f"Graph Build Completed.")
        logger.debug(f"graph_data: {graph_data}")
        logger.debug(f"metadata: {metadata}")
        execution_times["graph_build"] = time.perf_counter() - t_start

        # Stage 2: Encode NLQ
        logger.debug("Encoding NLQ")
        t_start = time.perf_counter()
        encoded_output = self.encoder.encode([query]) 

        if isinstance(encoded_output, tuple):
            q_embs = encoded_output[0]
        else:
            q_embs = encoded_output
        logger.debug("Encoding NLQ Completed.")
        logger.debug(f"q_embs: {q_embs.shape}")
        execution_times["encoding_nlq"] = time.perf_counter() - t_start

        # Stage 3: Projection & Similarity 계산
        t_start = time.perf_counter()
        if self.use_projection:
            logger.debug("Using Projection")
            # 💡 [신규] Projector가 GAT처럼 Graph 구조를 요구하는 경우 (is_graph_aware 플래그 활용)
            if getattr(self.projector, 'is_graph_aware', False):
                node_scores = self.projector.compute_scores(q_embs, graph_data)
            else:
                # 기존 일반 Projector 처리 로직
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
                
                z_q, z_nodes = self.projector(q_embs.to('cpu'), node_embs)
                node_scores = self.projector.compute_similarity(z_q, z_nodes)

        else:
            logger.debug("Not Using Projection")
            # Vector Only (Projection 없음)
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
            except RuntimeError as e:
                logger.debug(f"Cosine similarity skipped due to dimension mismatch (likely End-to-End Selector in use).")
                node_scores = None

            if node_scores is not None:
                logger.debug(f"🚨 강제 확인용 - node_scores 길이: {len(node_scores)}")
            else:
                logger.debug(f"🚨 강제 확인용 - node_scores is None")

        logger.debug(f"node_scores: {node_scores}")
        execution_times["projection"] = time.perf_counter() - t_start
            
        # Stage 4: Seed Selection
        logger.debug("Selecting Seed Nodes")
        t_start = time.perf_counter()
        
        if node_scores is not None:
            candidates_idx = list(range(len(node_scores)))
        else:
            candidates_idx = list(range(len(metadata.get('node_metadata', {}))))
        
        seeds = self.selector.select(
            scores=node_scores if 'node_scores' in locals() else None, 
            candidates=candidates_idx, 
            question=query,
            graph_data=graph_data,
            metadata=metadata
        )

        if hasattr(self.selector, 'latest_scores') and self.selector.latest_scores:
            scores_list = self.selector.latest_scores
        elif node_scores is not None:
            scores_list = node_scores.squeeze().tolist()
        else:
            scores_list = [1.0] * len(candidates_idx)

        logger.debug("Seed Nodes Selected")
        logger.debug(f"seeds: {seeds}")
        execution_times["seed_selection"] = time.perf_counter() - t_start
        
        # Stage 5: Subgraph Extraction
        logger.debug("Subgraph Extracting")
        t_start = time.perf_counter()
            
        # Pass query embedding for edge-prize extractors
        if 'edge_embeddings' in metadata and q_embs is not None:
            metadata['query_embedding'] = q_embs.squeeze(0).cpu()

        selected_nodes_idx, selected_edges = self.extractor.extract(
            graph_data=metadata,
            node_scores=scores_list,
            seed_nodes=seeds
        )
        
        # [Helper] Index to Text 번역
        subgraph_dict = {}
        for n_id in selected_nodes_idx:
            # 안전한 형변환 (Seed 노드가 이미 텍스트로 넘어온 경우 방어)
            n_id_key = int(n_id) if isinstance(n_id, (int, float)) or (isinstance(n_id, str) and n_id.isdigit()) else n_id
            name = metadata['node_metadata'].get(n_id_key, str(n_id_key))
            
            if "." in name:
                tbl, col = name.split(".", 1)
                subgraph_dict.setdefault(tbl, []).append(col)
            else:
                subgraph_dict.setdefault(name, [])

        # [Phase B-3] JOIN key 자동 포함: 2개 이상 테이블이 선택된 경우 FK 컬럼 보강
        if self.auto_join_keys and len(subgraph_dict) >= 2:
            fk_descriptions = metadata.get('fk_descriptions', [])
            node_meta = metadata.get('node_metadata', {})
            added_keys = []
            for idx, name in node_meta.items():
                if '->' in str(name):
                    # FK node format: "table1.col1->table2.col2"
                    parts = str(name).split('->')
                    if len(parts) == 2:
                        src = parts[0].strip()
                        dst = parts[1].strip()
                        src_tbl = src.split('.')[0] if '.' in src else src
                        dst_tbl = dst.split('.')[0] if '.' in dst else dst
                        src_col = src.split('.', 1)[1] if '.' in src else None
                        dst_col = dst.split('.', 1)[1] if '.' in dst else None
                        # 두 테이블 모두 선택된 경우에만 FK 컬럼 추가
                        if src_tbl in subgraph_dict and dst_tbl in subgraph_dict:
                            if src_col and src_col not in subgraph_dict.get(src_tbl, []):
                                subgraph_dict[src_tbl].append(src_col)
                                added_keys.append(src)
                            if dst_col and dst_col not in subgraph_dict.get(dst_tbl, []):
                                subgraph_dict[dst_tbl].append(dst_col)
                                added_keys.append(dst)
            if added_keys:
                logger.debug(f"[AutoJoinKeys] Added {len(added_keys)} FK columns: {added_keys}")

        logger.debug("Subgraph Extracted")
        logger.debug(f"subgraph_dict: {subgraph_dict}")
        execution_times["subgraph_extraction"] = time.perf_counter() - t_start

        # Stage 6: Agent Filtering
        logger.debug("Filtering")
        t_start = time.perf_counter()

        # [F3] Tier-2 pool: Selector-positive but PCST-rejected nodes.
        # A node is "Selector-positive" when its score passes a threshold
        # (default 0.5). Filters that don't consume these kwargs simply ignore
        # them (BaseFilter.refine accepts **kwargs).
        node_meta = metadata.get('node_metadata', {}) or {}
        tier1_indices = set()
        for n_id in selected_nodes_idx:
            n_id_key = int(n_id) if isinstance(n_id, (int, float)) or (
                isinstance(n_id, str) and str(n_id).isdigit()) else n_id
            tier1_indices.add(n_id_key)

        tier2_threshold = 0.5
        tier2_pool: List[str] = []
        gat_scores: Dict[str, float] = {}
        for idx, score in enumerate(scores_list):
            name = node_meta.get(idx, str(idx))
            name_str = str(name)
            if "." not in name_str:
                continue
            if score is None:
                continue
            try:
                s = float(score)
            except (TypeError, ValueError):
                continue
            gat_scores[name_str] = s
            if s >= tier2_threshold and idx not in tier1_indices:
                tier2_pool.append(name_str)

        final_result = self.filter.refine(
            query=query,
            subgraph=subgraph_dict,
            db_id=db_id,
            tier2_pool=tier2_pool,
            gat_scores=gat_scores,
            metadata=metadata,
        )

        logger.debug("Filtered")
        execution_times["filtering"] = time.perf_counter() - t_start

        # [F5] Extraction retry loop — triggered when filter verdict is
        # Unanswerable or selection is too sparse.
        retry_trace: List[str] = []
        if self.retry_enabled:
            attempts = 0
            while attempts < self.retry_max:
                status = final_result.get("status", "")
                n_final = len(final_result.get("final_nodes", []))
                needs_retry = (status == "Unanswerable") or (n_final < self.retry_min_nodes)
                if not needs_retry:
                    break
                strategy = self.retry_strategies[min(attempts, len(self.retry_strategies) - 1)]
                logger.info(f"[F5 retry {attempts+1}/{self.retry_max}] strategy={strategy}")
                snapshot = self._apply_retry_strategy(strategy)
                try:
                    selected_nodes_idx, selected_edges = self.extractor.extract(
                        graph_data=metadata, node_scores=scores_list, seed_nodes=seeds
                    )
                    subgraph_dict = {}
                    for n_id in selected_nodes_idx:
                        n_id_key = int(n_id) if isinstance(n_id, (int, float)) or (
                            isinstance(n_id, str) and str(n_id).isdigit()) else n_id
                        name = metadata['node_metadata'].get(n_id_key, str(n_id_key))
                        if "." in name:
                            tbl, col = name.split(".", 1)
                            subgraph_dict.setdefault(tbl, []).append(col)
                        else:
                            subgraph_dict.setdefault(name, [])
                    tier1_indices = {
                        (int(x) if isinstance(x, (int, float)) or (isinstance(x, str) and str(x).isdigit()) else x)
                        for x in selected_nodes_idx
                    }
                    tier2_pool = [
                        str(node_meta.get(idx, idx))
                        for idx, sc in enumerate(scores_list)
                        if sc is not None
                        and "." in str(node_meta.get(idx, ""))
                        and float(sc) >= tier2_threshold
                        and idx not in tier1_indices
                    ]
                    final_result = self.filter.refine(
                        query=query, subgraph=subgraph_dict, db_id=db_id,
                        tier2_pool=tier2_pool, gat_scores=gat_scores, metadata=metadata,
                    )
                    retry_trace.append(
                        f"retry{attempts+1}:{strategy}->{len(final_result.get('final_nodes', []))}nodes"
                    )
                finally:
                    self._restore_extractor_params(snapshot)
                attempts += 1
            if retry_trace:
                final_result["reasoning"] = (
                    final_result.get("reasoning", "") + f" | F5[{'; '.join(retry_trace)}]"
                )
                final_result["retry_attempts"] = attempts

        logger.debug(f"✅ Final Decision: {final_result.get('status', 'Unknown')} | Nodes: {len(final_result.get('final_nodes', []))}")
        logger.debug(f"Final Nodes: {final_result.get('final_nodes')}")

        generated_sql = ""
        t_start = time.perf_counter()
        if self.generator is not None:
            logger.debug("SQL Generation")
            # Stage 7: SQL Generation
            generated_sql = self.generator.generate(query=query, subgraph=subgraph_dict)
            logger.debug(f"Generated SQL: {generated_sql}")
        execution_times["sql_generation"] = time.perf_counter() - t_start

        final_result["generated_sql"] = generated_sql
        final_result["execution_time"] = execution_times

        final_result["raw_scores"] = scores_list
        final_result["node_names"] = [metadata['node_metadata'].get(i, str(i)) for i in range(len(scores_list))]
        
        return final_result