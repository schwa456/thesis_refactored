import os
import pickle
import torch
from pyvis.network import Network

def visualize_hetero_graph(pkl_path: str, output_html: str = "schema_graph.html"):
    print(f"🔍 그래프 로딩 중: {pkl_path}")
    
    if not os.path.exists(pkl_path):
        print(f"❌ 파일을 찾을 수 없습니다: {pkl_path}")
        return

    # 1. PKL 파일 로드 (graph_data: HeteroData, metadata_mapping: dict)
    with open(pkl_path, 'rb') as f:
        graph_data, metadata_mapping = pickle.load(f)

    # 역방향 매핑 (ID -> Name)
    id_to_table = {v: k for k, v in metadata_mapping['table_to_id'].items()}
    id_to_col = {v: k for k, v in metadata_mapping['col_to_id'].items()}

    # 2. PyVis 네트워크 초기화 (다크 모드, 물리 엔진 적용)
    net = Network(height="900px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    
    # 3. 노드(Nodes) 추가
    print(" ➔ 노드 추가 중...")
    
    # 3-1. Table 노드 (파란색 박스)
    for t_id, t_name in id_to_table.items():
        node_id = f"table_{t_id}"
        net.add_node(node_id, label=f"[{t_name}]", title=f"Table: {t_name}", color="#1f78b4", shape="box", size=25)

    # 3-2. Column 노드 (녹색 타원)
    for c_id, c_name in id_to_col.items():
        node_id = f"column_{c_id}"
        # 이름이 너무 길면 테이블명은 떼고 컬럼명만 보여줌
        short_name = c_name.split('.')[-1] if '.' in c_name else c_name
        net.add_node(node_id, label=short_name, title=f"Column: {c_name}", color="#33a02c", shape="ellipse", size=15)

    # 3-3. FK (외래키) 노드 (빨간색 다이아몬드)
    # GAT 코드 상 'fk_node'가 존재합니다.
    if 'fk_node' in graph_data.node_types:
        num_fks = graph_data['fk_node'].num_nodes
        for fk_id in range(num_fks):
            node_id = f"fk_node_{fk_id}"
            net.add_node(node_id, label="FK", title="Foreign Key Relation", color="#e31a1c", shape="diamond", size=15)

    # 4. 엣지(Edges) 추가
    print(" ➔ 엣지 추가 중...")
    
    edge_types = graph_data.edge_index_dict.keys()
    
    for edge_type in edge_types:
        src_type, relation, dst_type = edge_type
        
        # 시각적 깔끔함을 위해 역방향 엣지는 그리지 않음 (방향성이 있으면 충분히 파악 가능)
        if relation in ['belongs_to', 'points_to']: 
            continue 
            
        edge_index = graph_data.edge_index_dict[edge_type]
        src_nodes = edge_index[0].tolist()
        dst_nodes = edge_index[1].tolist()

        for src, dst in zip(src_nodes, dst_nodes):
            src_id = f"{src_type}_{src}"
            dst_id = f"{dst_type}_{dst}"
            
            # 관계에 따라 선 색상과 두께 조정
            if relation == 'has_column':
                net.add_edge(src_id, dst_id, color="#888888", width=1)
            elif relation == 'is_source_of': # Column -> FK
                net.add_edge(src_id, dst_id, color="#ff7f0e", width=2, dashes=True)
            elif relation == 'points_to': # FK -> Column (현재 스킵 처리했지만, 필요시 추가)
                pass

    # 5. FK 노드에서 타겟 컬럼으로 향하는 엣지는 별도로 추가 (시각화 명확성을 위해)
    if ('fk_node', 'points_to', 'column') in edge_types:
        edge_index = graph_data.edge_index_dict[('fk_node', 'points_to', 'column')]
        src_nodes = edge_index[0].tolist()
        dst_nodes = edge_index[1].tolist()
        for src, dst in zip(src_nodes, dst_nodes):
            src_id = f"fk_node_{src}"
            dst_id = f"column_{dst}"
            net.add_edge(src_id, dst_id, color="#d62728", width=2, arrows="to")

    # 6. 물리 엔진 설정 (노드들이 겹치지 않게 잘 퍼지도록)
    net.repulsion(node_distance=120, spring_length=150)
    
    # 7. HTML 저장
    net.write_html(output_html)
    print(f"✅ 그래프가 저장되었습니다: {output_html}")
    print("👉 생성된 HTML 파일을 다운로드하여 웹 브라우저(크롬 등)로 열어보세요!")

if __name__ == "__main__":

    db_ids = [
        'california_schools',
        'card_games',
        'codebase_community',
        'debit_card_specializing',
        'european_football_2',
        'financial',
        'formula_1',
        'student_club',
        'superhero',
        'thrombosis_prediction',
        'toxicology'
    ]
    
    for db_id in db_ids:
        sample_pkl = f"{db_id}_graph.pkl"
    
        if os.path.exists(sample_pkl):
            visualize_hetero_graph(sample_pkl, f"{db_id}_schema.html")
        else:
            # 파일이 없다면 폴더 내 첫 번째 pkl 파일을 자동으로 찾아서 시각화합니다.
            import glob
            pkl_files = glob.glob("./data/processed/*.pkl")
            if pkl_files:
                visualize_hetero_graph(pkl_files[0], "sample_schema.html")
            else:
                print("❌ ./data/processed/ 폴더에 .pkl 파일이 없습니다.")