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
        
        self.builder = build("builder", config['graph_builder'])
        self.encoder = build("encoder", config['nlq_encoder'])
        
        self.use_projection = config.get('projection', {}).get('enabled', False)
        if self.use_projection:
            self.projector = build("projector", config['projection'])
            
        self.selector = build("selector", config['seed_selector'])
        self.extractor = build("extractor", config['connectivity_extractor'])
        self.filter = build("filter", config['filter'])

        if config['sql_generator']['enabled']:
            self.generator = build("generator", config['sql_generator'])
        else:
            self.generator = None
        
        logger.info("✅ Pipeline assembly completed successfully.")

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

        logger.debug("Subgraph Extracted")
        logger.debug(f"subgraph_dict: {subgraph_dict}")
        execution_times["subgraph_extraction"] = time.perf_counter() - t_start

        # Stage 6: Agent Filtering
        logger.debug("Agent Filtering")
        t_start = time.perf_counter()
        # 💡 [수정됨] db_id를 Filter로 전달하여 Value Retrieval 및 Example 조회를 허용
        final_result = self.filter.refine(
            query=query, 
            subgraph=subgraph_dict,
            db_id=db_id              # <-- 신규 추가: 필터링 프롬프트에 DB Value를 포함시키기 위함
        )

        logger.debug("Agent Filtered")
        execution_times["agent_filtering"] = time.perf_counter() - t_start

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

        if node_scores is not None:
            raw_scores_list = node_scores.squeeze().tolist()
            node_names = [metadata['node_metadata'].get(i, str(i)) for i in range(len(raw_scores_list))]
        else:
            raw_scores_list = []
            node_names = []

        final_result["raw_scores"] = scores_list
        final_result["node_names"] = node_names
        
        return final_result