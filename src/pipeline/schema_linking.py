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
        
        # Stage 3: Projection & Similarity 계산
        if self.use_projection:
            # 💡 [신규] Projector가 GAT처럼 Graph 구조를 요구하는 경우 (is_graph_aware 플래그 활용)
            if getattr(self.projector, 'is_graph_aware', False):
                node_scores = self.projector.compute_scores(q_embs, graph_data)
            else:
                # 기존 일반 Projector 처리 로직
                table_embs = graph_data['table'].x
                col_embs = graph_data['column'].x
                fk_embs = graph_data.get('fk_node', {}).get('x', torch.empty(0, table_embs.size(-1)))
                node_embs = torch.cat([table_embs, col_embs, fk_embs], dim=0).to('cpu')
                
                z_q, z_nodes = self.projector(q_embs.to('cpu'), node_embs)
                node_scores = self.projector.compute_similarity(z_q, z_nodes)
        else:
            # Vector Only (Projection 없음)
            table_embs = graph_data['table'].x
            col_embs = graph_data['column'].x
            fk_embs = graph_data.get('fk_node', {}).get('x', torch.empty(0, table_embs.size(-1)))
            node_embs = torch.cat([table_embs, col_embs, fk_embs], dim=0).to('cpu')
            
            node_scores = torch.nn.functional.cosine_similarity(q_embs.to('cpu'), node_embs)
            
        # Stage 4: Seed Selection
        candidates_idx = list(range(len(node_scores)))
        
        # 💡 [수정됨] db_id와 metadata를 Selector로 전달하여 DB 접근 및 Index-Text 변환 허용
        seeds = self.selector.select(
            scores=node_scores, 
            candidates=candidates_idx, 
            question=query,
            db_id=db_id,             # <-- 신규 추가: Value Retrieval을 위한 DB 경로 탐색용
            metadata=metadata        # <-- 신규 추가: 정수 Index를 실제 Schema Text로 변환하기 위함
        )
        
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
            # 안전한 형변환 (Seed 노드가 이미 텍스트로 넘어온 경우 방어)
            n_id_key = int(n_id) if isinstance(n_id, (int, float)) or (isinstance(n_id, str) and n_id.isdigit()) else n_id
            name = metadata['node_metadata'].get(n_id_key, str(n_id_key))
            
            if "." in name:
                tbl, col = name.split(".", 1)
                subgraph_dict.setdefault(tbl, []).append(col)
            else:
                subgraph_dict.setdefault(name, [])

        # Stage 6: Agent Filtering
        # 💡 [수정됨] db_id를 Filter로 전달하여 Value Retrieval 및 Example 조회를 허용
        final_result = self.filter.refine(
            query=query, 
            subgraph=subgraph_dict,
            db_id=db_id              # <-- 신규 추가: 필터링 프롬프트에 DB Value를 포함시키기 위함
        )

        logger.info(f"✅ Final Decision: {final_result.get('status', 'Unknown')} | Nodes: {len(final_result.get('final_nodes', []))}")

        generated_sql = ""
        if self.generator is not None:
            # Stage 7: SQL Generation
            generated_sql = self.generator.generate(query=query, subgraph=subgraph_dict)
        
        final_result["generated_sql"] = generated_sql
        return final_result