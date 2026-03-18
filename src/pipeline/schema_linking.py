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
        
        self.builder = build("builder", config['graph_builder'])
        self.encoder = build("encoder", config['nlq_encoder'])
        
        self.use_projection = config.get('projection', {}).get('enabled', False)
        if self.use_projection:
            self.projector = build("projector", config['projection'])
            
        self.selector = build("selector", config['seed_selector'])
        self.extractor = build("extractor", config['connectivity_extractor'])
        self.filter = build("filter", config['filter'])

        if 'sql_generator' in config:
            self.generator = build("generator", config['sql_generator'])
        else:
            self.generator = None
        
        logger.info("✅ Pipeline assembly completed successfully.")

    def run(self, db_id: str, query: str) -> Dict[str, Any]:
        """단일 질의(Query) 처리 파이프라인"""
        logger.info(f"[{db_id}] Processing Query: '{query}'")

        # Stage 1: Build / Load Graph
        graph_data, metadata = self.builder.build(db_id=db_id)
        
        # Stage 2: Encode NLQ
        q_embs = self.encoder.encode([query]) 
        
        # Stage 3: Projection & Similarity (선택)
        # (구현에 따라 FAISS 등에서 노드 임베딩을 가져오는 로직)
        node_embs = torch.tensor(graph_data.reconstruct_n(0, graph_data.ntotal))
        
        if self.use_projection:
            z_q, z_nodes = self.projector(q_embs, node_embs)
            node_scores = self.projector.compute_similarity(z_q, z_nodes)
        else:
            node_scores = torch.nn.functional.cosine_similarity(q_embs, node_embs)
            
        # Stage 4: Seed Selection
        candidates_idx = list(range(len(node_scores)))
        seeds = self.selector.select(scores=node_scores, candidates=candidates_idx, question=query)
        
        # Stage 5: Subgraph Extraction
        scores_list = node_scores.squeeze().tolist() 
        selected_nodes_idx, selected_edges = self.extractor.extract(
            graph_data=metadata, 
            node_scores=scores_list, 
            seed_nodes=seeds
        )
        
        # [Helper] Index to Text 번역
        subgraph_dict = {}
        for n_id in selected_nodes_idx:
            name = metadata['node_metadata'].get(n_id, str(n_id))
            if "." in name:
                tbl, col = name.split(".")
                subgraph_dict.setdefault(tbl, []).append(col)
            else:
                subgraph_dict.setdefault(name, [])

        # Stage 6: Agent Filtering
        final_result = self.filter.refine(query=query, subgraph=subgraph_dict)

        logger.info(f"✅ Final Decision: {final_result['status']} | Nodes: {len(final_result.get('final_nodes', []))}")

        generated_sql = ""
        if self.generator is not None:
            # Stage 7: SQL Generation
            generated_sql = self.generator.generate(query=query, subgraph=subgraph_dict)
        
        final_result["generated_sql"] = generated_sql
        return final_result