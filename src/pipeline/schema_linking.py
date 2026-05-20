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

        # Wave 11 Schema Serialization Direction C (DECISIONS 2026-05-19) —
        # serializer 분기 + question enrichment. config['sql_generator']['serializer']
        # ∈ {None, "source_tagged", "flat_merged_fk", "flat_merged_no_fk",
        #    "question_enrichment", "tagged_enriched"}
        # config['sql_generator']['serializer_config']: dict (type-specific)
        sql_gen_cfg = config.get('sql_generator', {}) or {}
        self.serializer_type = sql_gen_cfg.get('serializer')  # None = legacy DDL path
        self.serializer_config = sql_gen_cfg.get('serializer_config', {}) or {}
        self._enrichment_cache = None
        self._enrichment_client = None
        self._enrichment_few_shots = None
        if self.serializer_type in ("question_enrichment", "tagged_enriched"):
            # Enrichment cache + LLM client lazy init (재사용)
            from serializers.question_enricher import EnrichmentCache
            from llm_client.api_handler import APIClient
            self._enrichment_cache = EnrichmentCache()
            self._enrichment_client = APIClient(
                provider=self.serializer_config.get('enricher_provider', 'glm'),
            )
            # Few-shot examples 로드 (학술 agent §5.4 — 사용자 직접 작성 12 examples).
            # 파일 형식: {"_meta": {...}, "examples": [{"db_id", "difficulty",
            #                       "original_question", "filtered_schema",
            #                       "enriched_question"}, ...]} (planner 5/19 정합).
            few_shot_path = self.serializer_config.get('few_shot_path')
            raw_few_shots: List[Dict[str, Any]] = []
            if few_shot_path:
                import os as _os, json as _json
                if _os.path.exists(few_shot_path):
                    with open(few_shot_path, "r", encoding="utf-8") as _f:
                        loaded = _json.load(_f)
                    if isinstance(loaded, dict) and "examples" in loaded:
                        raw_few_shots = loaded.get("examples", []) or []
                    elif isinstance(loaded, list):
                        raw_few_shots = loaded
                else:
                    logger.warning(
                        f"[Wave 11 Enrichment] few_shot_path '{few_shot_path}' missing — "
                        "enrichment will run with 0 examples (suboptimal quality)."
                    )
            # JSON 키 정규화 → enricher 가 요구하는 {question, schema,
            # enriched_question, db_id} 형식으로 변환 (data leakage filter 위한
            # db_id retain). question/schema 키는 본 file format (original_question/
            # filtered_schema) 와 enricher key (question/schema) 둘 다 호환.
            normalized: List[Dict[str, Any]] = []
            for ex in raw_few_shots:
                if not isinstance(ex, dict):
                    continue
                normalized.append({
                    "question": ex.get("original_question") or ex.get("question") or "",
                    "schema": ex.get("filtered_schema") or ex.get("schema") or {},
                    "enriched_question": ex.get("enriched_question", ""),
                    "db_id": ex.get("db_id"),
                    "difficulty": ex.get("difficulty"),
                })
            self._enrichment_few_shots = normalized
            logger.info(
                f"[Wave 11] Enrichment initialized — few_shots="
                f"{len(self._enrichment_few_shots)} normalized "
                f"(from {len(raw_few_shots)} raw), "
                f"model={self.serializer_config.get('enricher_model_name', 'zai-org/glm-4.7')}"
            )
        if self.serializer_type:
            logger.info(
                f"[Wave 11 Serializer] type='{self.serializer_type}' active "
                f"(schema_str + query 자리 직렬화 교체)"
            )

        logger.info("✅ Pipeline assembly completed successfully.")

    def _select_few_shots_for_query(
        self, test_db_id: str = None, max_examples: int = 8,
    ) -> List[Dict[str, Any]]:
        """Data leakage 방지 — test query 의 DB 와 다른 DB 의 examples 만 사용.

        학술 agent §5.4 + 사용자 정합 (5/19): "각 test query의 DB와 다른 DB의 예시만 사용".
        difficulty-stratified sampling 시도 (가능하면 simple/moderate/challenging 균등).
        """
        if not self._enrichment_few_shots:
            return []
        filtered = [
            ex for ex in self._enrichment_few_shots
            if not test_db_id or ex.get("db_id") != test_db_id
        ]
        # Difficulty 별 grouping → round-robin 으로 max_examples 까지 선택
        by_diff: Dict[str, List[Dict[str, Any]]] = {"simple": [], "moderate": [], "challenging": []}
        others: List[Dict[str, Any]] = []
        for ex in filtered:
            d = ex.get("difficulty")
            if d in by_diff:
                by_diff[d].append(ex)
            else:
                others.append(ex)
        ordered: List[Dict[str, Any]] = []
        # Round-robin
        idx = 0
        while len(ordered) < max_examples and any(by_diff.values()):
            d_key = ["simple", "moderate", "challenging"][idx % 3]
            if by_diff[d_key]:
                ordered.append(by_diff[d_key].pop(0))
            idx += 1
            if idx > max_examples * 3:
                break
        if len(ordered) < max_examples and others:
            ordered.extend(others[: max_examples - len(ordered)])
        return ordered

    def _apply_wave11_serializer(
        self,
        final_nodes: List[str],
        filter_info: Dict[str, Any],
        query: str,
        metadata: Dict[str, Any] = None,
        db_id: str = None,
    ):
        """Wave 11 serializer 적용. Returns (pre_serialized_schema, enriched_question).

        **Immutability contract (Wave 11 Debug 2026-05-20, commit 3eb476d 후속)**:
        본 함수는 final_nodes 와 filter_info 를 read-only 사용. 새 m4_output dict
        (new local) 를 생성해 serializer 에 전달. 어떤 serializer 도 m4_output 을
        mutate 하지 않음 (단순 string format). 즉 본 path 는 final_result["final_nodes"]
        를 절대 변경 안 함. Wave 11 Phase B 의 c_v3a/v3b Schema Content Invariance
        violation 의 root cause 는 LLM stochastic (temperature=0.0 라도 GLM 4.7 API
        의 internal sampling) — 본 코드 path 의 implementation bug 아님 (regression
        test 로 검증, src/serializers/tests/test_wave11_serializers.py).
        """
        if not self.serializer_type:
            return None, None
        # M4 output (final_nodes → {table: [col, ...]}). list/dict 모두 new instance
        # (defensive — final_nodes 가 외부에서 mutation 돼도 m4_output 만 영향).
        m4_output: Dict[str, List[str]] = {}
        for n in list(final_nodes or []):
            if not isinstance(n, str):
                continue
            if "." in n:
                t, c = n.split(".", 1)
                if c not in m4_output.setdefault(t, []):
                    m4_output[t].append(c)
            else:
                m4_output.setdefault(n, [])

        pre_serialized = None
        enriched = None

        # Source-Tagged 또는 Comb-C 의 schema 부분
        if self.serializer_type in ("source_tagged", "tagged_enriched"):
            from serializers.source_tagged_serializer import format_tagged_schema
            fwd = (filter_info or {}).get("filter_forward_set") or []
            bwd = (filter_info or {}).get("filter_backward_set") or []
            pre_serialized = format_tagged_schema(m4_output, fwd, bwd)

        # Flat Merged (C-v3a / C-v3b)
        elif self.serializer_type in ("flat_merged_fk", "flat_merged_no_fk"):
            from serializers.flat_merged_serializer import format_flat_schema
            fk_relations = None
            if self.serializer_type == "flat_merged_fk" and metadata:
                fk_relations = self._extract_fk_relations_from_metadata(metadata)
            pre_serialized = format_flat_schema(m4_output, fk_relations=fk_relations)

        # Question Enrichment 단독 (C-v2). Schema 는 default DDL.
        elif self.serializer_type == "question_enrichment":
            pre_serialized = None  # SQL gen 의 default DDL path

        # Comb-C 의 enrichment 부분 + tagged_enriched 모두 enrich
        if self.serializer_type in ("question_enrichment", "tagged_enriched"):
            from serializers.question_enricher import enrich_question
            # data leakage 방지 — test DB 와 다른 DB 의 examples 만 (사용자 5/19 정합)
            max_examples = int(self.serializer_config.get(
                'enricher_max_few_shots', 8,
            ))
            selected_few_shots = self._select_few_shots_for_query(
                test_db_id=db_id, max_examples=max_examples,
            )
            try:
                enriched = enrich_question(
                    question=query,
                    m4_schema=m4_output,
                    few_shot_examples=selected_few_shots,
                    llm_client=self._enrichment_client,
                    model_name=self.serializer_config.get(
                        'enricher_model_name', 'zai-org/glm-4.7',
                    ),
                    temperature=float(self.serializer_config.get(
                        'enricher_temperature', 0.0,
                    )),
                    cache=self._enrichment_cache,
                )
            except Exception as e:
                logger.warning(f"[Wave 11 Enrichment] failed — fallback original query: {e}")
                enriched = query
        return pre_serialized, enriched

    @staticmethod
    def _extract_fk_relations_from_metadata(metadata: Dict[str, Any]):
        """metadata['fk_to_id'] keys → list of (pt, pc, ct, cc) tuples."""
        out = []
        fk_to_id = (metadata or {}).get("fk_to_id") or {}
        for fk_key in fk_to_id.keys():
            if not isinstance(fk_key, str) or "->" not in fk_key:
                continue
            left, right = fk_key.split("->", 1)
            left, right = left.strip(), right.strip()
            if "." not in left or "." not in right:
                continue
            pt, pc = left.split(".", 1)
            ct, cc = right.split(".", 1)
            out.append((pt, pc, ct, cc))
        return out

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

    def run(self, db_id: str, query: str, evidence: str = "") -> Dict[str, Any]:
        """단일 질의(Query) 처리 파이프라인.

        evidence: BIRD-dev `external_knowledge` 필드. LLMSQLGenerator 에 전달되어 SQL gen prompt 의
        [External Knowledge] 섹션에 삽입. 직전 (2026-05-12 Filter sweep) 까지 미사용 → 본 fix 로
        evidence forward — Baseline 대비 anchor EX 21.91%p gap 의 dominant 원인 회수.
        """
        logger.debug(f"[{db_id}] Processing Query: '{query}'")

        execution_times = {}

        # Stage 1: Build / Load Graph
        logger.debug(f"Building Graph")
        t_start = time.perf_counter()
        graph_data, metadata = self.builder.build(db_id=db_id, db_dir=self.db_dir)
        # Selector/Extractor 가 DB 별 정책을 적용할 수 있도록 db_id 주입 (Proposal C H2 등).
        if isinstance(metadata, dict):
            metadata.setdefault("db_id", db_id)
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

        # Wave 7 patch (DECISIONS 2026-05-18 §3): selector top-K snapshot
        # (named table.col) for downstream Selector-only EX cell (w7_s1).
        selector_top_k_nodes_named: List[str] = []
        for _s_id in (seeds or []):
            _s_key = (
                int(_s_id)
                if isinstance(_s_id, (int, float))
                or (isinstance(_s_id, str) and str(_s_id).isdigit())
                else _s_id
            )
            _s_name = metadata['node_metadata'].get(_s_key, str(_s_key))
            selector_top_k_nodes_named.append(_s_name)

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

        # Wave 7 patch (DECISIONS 2026-05-18 §3): extractor output snapshot
        # (before filter) for downstream Extractor-only EX cell (w7_s2).
        extractor_output_snapshot: Dict[str, List[str]] = {
            k: list(v) for k, v in subgraph_dict.items()
        }
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

        # SGBE Phase 2 (DECISIONS 2026-05-12) — selector 가 노출한 raw GAT score (sigmoid 후, normalize 전)
        # 와 raw cosine score 를 column name → score dict 로 변환해 filter 에 전달.
        # 기존 `gat_scores` (blended + minmax-normalize) 는 backward compat 으로 유지.
        # SGBE filter 만 raw_gat_scores 의 column-level calibration 을 사용 (θ_keep/θ_drop gating).
        #
        # 키 형식: 'table.column' 우선 + 'column' 단독 fallback (SGBE _lookup_score 가 둘 다 지원).
        # fk_node ('table.col->table.col') 는 제외. column 단독 키는 첫 등장 score 유지 (setdefault).
        raw_gat_scores: Dict[str, float] = {}
        raw_cos_scores: Dict[str, float] = {}
        selector_raw_gat = getattr(self.selector, "latest_raw_gat_scores", None) or {}
        selector_raw_cos = getattr(self.selector, "latest_raw_cos_scores", None) or {}
        for idx, name in node_meta.items():
            name_str = str(name)
            if "." not in name_str or "->" in name_str:
                continue
            col_only = name_str.split(".", 1)[1] if "." in name_str else name_str
            if idx in selector_raw_gat:
                score_val = float(selector_raw_gat[idx])
                raw_gat_scores[name_str] = score_val
                raw_gat_scores.setdefault(col_only, score_val)
            if idx in selector_raw_cos:
                cos_val = float(selector_raw_cos[idx])
                raw_cos_scores[name_str] = cos_val
                raw_cos_scores.setdefault(col_only, cos_val)

        final_result = self.filter.refine(
            query=query,
            subgraph=subgraph_dict,
            db_id=db_id,
            tier2_pool=tier2_pool,
            gat_scores=gat_scores,
            raw_gat_scores=raw_gat_scores,
            raw_cos_scores=raw_cos_scores,
            metadata=metadata,
        )

        execution_times["filtering"] = time.perf_counter() - t_start
        _reason_preview = (final_result.get("reasoning") or "")[:400]
        _extra = []
        if final_result.get("adaptive_route"):
            _extra.append(
                f"route={final_result['adaptive_route']}"
                f"({final_result.get('adaptive_confidence', 0):.3f})"
            )
        for _k in ("tier1_dropped_count", "tier2_pool_count", "restored_count", "promoted_count"):
            if _k in final_result:
                _extra.append(f"{_k}={final_result[_k]}")
        logger.info(
            f"[Filter] status={final_result.get('status')} "
            f"nodes={len(final_result.get('final_nodes', []))} "
            f"time={execution_times['filtering']:.2f}s "
            f"{' '.join(_extra)} | reasoning={_reason_preview}"
        )

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
                        tier2_pool=tier2_pool, gat_scores=gat_scores,
                        raw_gat_scores=raw_gat_scores, raw_cos_scores=raw_cos_scores,
                        metadata=metadata,
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
            # Wave 11 Serializer 적용 (None 이면 legacy DDL + original query)
            pre_serialized, enriched = self._apply_wave11_serializer(
                final_nodes=final_result.get("final_nodes") or [],
                filter_info=final_result.get("filter_info") or {},
                query=query,
                metadata=metadata,
                db_id=db_id,
            )
            # Stage 7: SQL Generation
            generated_sql = self.generator.generate(
                query=query, subgraph=subgraph_dict, evidence=evidence,
                pre_serialized_schema=pre_serialized,
                enriched_question=enriched,
            )
            logger.debug(f"Generated SQL: {generated_sql}")
            # filter_info 에 Wave 11 진단 노출 (analyzer 가 출력 분석 시)
            if self.serializer_type:
                fi = final_result.get("filter_info") or {}
                fi["wave11_serializer_type"] = self.serializer_type
                fi["wave11_pre_serialized_used"] = pre_serialized is not None
                fi["wave11_enriched_question_used"] = enriched is not None and enriched != query
                if self._enrichment_cache is not None:
                    fi["wave11_enrichment_cache_stats"] = self._enrichment_cache.stats()
                final_result["filter_info"] = fi
        execution_times["sql_generation"] = time.perf_counter() - t_start

        final_result["generated_sql"] = generated_sql

        # Wave 7 Option A patch (DECISIONS 2026-05-18 §3): additional SQL Gen for
        # Selector-only + Extractor-only EX cells (1 SQL Gen / q × 2 stages).
        if self.generator is not None:
            sel_subgraph: Dict[str, List[str]] = {}
            for _name in selector_top_k_nodes_named:
                if "." in _name:
                    _tbl, _col = _name.split(".", 1)
                    sel_subgraph.setdefault(_tbl, []).append(_col)
                else:
                    sel_subgraph.setdefault(_name, [])
            final_result["generated_sql_selector_only"] = (
                self.generator.generate(query=query, subgraph=sel_subgraph, evidence=evidence)
                if sel_subgraph else ""
            )
            final_result["generated_sql_extractor_only"] = (
                self.generator.generate(query=query, subgraph=extractor_output_snapshot, evidence=evidence)
                if extractor_output_snapshot else ""
            )
        else:
            final_result["generated_sql_selector_only"] = ""
            final_result["generated_sql_extractor_only"] = ""

        final_result["execution_time"] = execution_times

        final_result["raw_scores"] = scores_list
        final_result["node_names"] = [metadata['node_metadata'].get(i, str(i)) for i in range(len(scores_list))]

        builder_info = getattr(self.builder, 'last_info', None) or metadata.get('builder_info')
        if builder_info:
            final_result["builder_info"] = dict(builder_info)
        selector_info = getattr(self.selector, 'last_info', None)
        if selector_info:
            final_result["selector_info"] = dict(selector_info)
        extractor_info = getattr(self.extractor, 'last_info', None)
        if extractor_info:
            final_result["extractor_info"] = dict(extractor_info)

        # Wave 7 patch (DECISIONS 2026-05-18 §3): add stage-wise node lists for
        # downstream SQL-only EX cells (w7_s1 Selector only / w7_s2 Extractor only).
        if "selector_info" not in final_result:
            final_result["selector_info"] = {}
        final_result["selector_info"]["selected_nodes_top_k"] = selector_top_k_nodes_named
        if "extractor_info" not in final_result:
            final_result["extractor_info"] = {}
        final_result["extractor_info"]["extractor_selected_nodes"] = extractor_output_snapshot

        filter_info = getattr(self.filter, 'last_info', None) or final_result.get("filter_info")
        if filter_info:
            filter_info = dict(filter_info)
            filter_info["pipeline_tier2_pool_size"] = len(tier2_pool)
            filter_info["pipeline_gat_scores_available"] = len(gat_scores)
            filter_info["pipeline_retry_enabled"] = bool(self.retry_enabled)
            filter_info["pipeline_retry_attempts"] = int(final_result.get("retry_attempts", 0))
            if retry_trace:
                filter_info["pipeline_retry_trace"] = list(retry_trace)
            final_result["filter_info"] = filter_info

        return final_result