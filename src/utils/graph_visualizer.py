import networkx as nx
from pyvis.network import Network
import os
from typing import List, Any
from utils.logger import get_logger

logger = get_logger(__name__)

class GraphVisualizer:
    """
    스키마 그래프와 Selector/Filter의 결과를 인터랙티브 HTML로 시각화하는 도구.
    """
    def __init__(self, output_dir: str = "logs/visualization"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize(self,
                    graph: nx.Graph,
                    question: str="",
                    seeds: List[Any] = None,
                    final_nodes: List[Any]= None,
                    gold_nodes: List[Any] = None,
                    file_name: str = "schema_graph.html"
                    ):
        """
        graph: DB 스키마가 담긴 NetworkX 그래프 객체
        seeds: Selector가 선택한 1차 후보 노드 리스트
        final_nodes: Filter까지 통과한 최종 노드 리스트
        gold_nodes: 실제 정답(Gold) 노드 리스트 (비교용, 옵션)
        """
        # 화면 크기 100%, 물리 엔진(Physics) 활성화
        net = Network(height='100vh', width='100%', bgcolor='#111827', font_color='white', directed=True)

        # 노드 집합 초기화
        seeds_set = {str(x).strip() for x in seeds} if seeds else set()
        final_set = {str(x).strip() for x in final_nodes} if final_nodes else set()
        gold_set = {str(x).strip() for x in gold_nodes} if gold_nodes else set()

        # 1. 노드 추가 및 색상 / 모양 결정
        for node_id, node_data in graph.nodes(data=True):
            safe_node_id = str(node_id).strip()
            node_name = node_data.get('name', safe_node_id)
            node_type = node_data.get('type', 'column')

            shape = "box" if node_type == 'table' else "dot"
            size = 30 if node_type == 'table' else 15

            if safe_node_id in final_set and safe_node_id in gold_set:
                color = "#10B981"  # 초록색 (TP: 정답)
                shape = "star"     # 별 모양 강조!
                size = 50          # 크기 폭발
            elif safe_node_id in final_set:
                color = "#EF4444"  # 빨간색 (FP: 틀린 노드)
                size = 15          # 작게
            elif safe_node_id in gold_set:
                color = "#3B82F6"  # 파란색 (FN: 놓친 정답)
                shape = "diamond"  # 다이아몬드 강조!
                size = 40
            elif safe_node_id in seeds_set:
                color = "#F59E0B"  # 노란색 (TN: 필터가 잘 걸러냄)
            else:
                color = "#4B5563"  # 짙은 회색 (선택 안 됨)

            title_text = f"Name: {node_name}\n"
            title_text += f"Type: {node_type.upper()}\n"
            title_text += "-" * 30 + "\n" # 구분선 역할
            
            for k, v in node_data.items():
                if k not in ['name', 'type', 'label', 'shape', 'color', 'size', 'title']:
                    # 텍스트가 너무 긴 경우를 대비해 문자열 처리
                    val_str = str(v)
                    title_text += f"{k}: {val_str}\n"

            net.add_node(
                node_id,
                label=node_name,
                title=title_text,
                color=color,
                shape=shape,
                size=size
            )

        # 2. 엣지 관계 추가
        for u, v, edge_data in graph.edges(data=True):
            edge_type = edge_data.get('type', 'relation')
            safe_u = str(u).strip()
            safe_v = str(v).strip()

            title_text = f"Edge: {safe_u} -> {safe_v}\n"
            title_text += "-" * 30 + "\n"
            for k, val in edge_data.items():
                if k not in ['title', 'color']:
                    title_text += f"{k}: {str(val)}\n"

            # 엣지 색상: 두 노드가 모두 선택된 경우 하이라이트
            if safe_u in final_set and safe_v in final_set:
                edge_color = "#10B981" # 초록색 연결선
            elif safe_u in seeds_set and safe_v in seeds_set:
                edge_color = "#F59E0B" # 노란색 연결선
            else:
                edge_color = "#374151" # 어두운 회색 (기본)

            net.add_edge(u, v, title=title_text, color=edge_color)

        # 물리 엔진 설정 추가 (그래프가 예쁘게 퍼지도록)
        net.set_options("""
        var options = {
        "physics": {
            "barnesHut": {
            "gravitationalConstant": -30000,
            "centralGravity": 0.3,
            "springLength": 150
            }
        }
        }
        """)

        # 상단에 질문 표시 (Question Node)
        if question:
            import textwrap
            # 질문이 너무 길면 60자 단위로 예쁘게 줄바꿈
            wrapped_q = "\n".join(textwrap.wrap(question, width=60))
            
            # 💡 [수정] 우주로 날아가지 않게 x, y 좌표를 고정하고 물리 엔진(physics) 끄기!
            net.add_node(
                "QUESTION_NODE", 
                label=f"Q: {wrapped_q}", 
                shape="text", 
                font={"size": 400, "color": "white", "align": "center"},
                x=0,         # 화면 가운데
                y=-1000,      # 화면 위쪽
                physics=False, # 물리 엔진 적용 안 함 (튕겨 나가지 않음)
                fixed=True     # 고정시킴
            )
        # HTML 파일로 저장
        out_path = os.path.join(self.output_dir, file_name)
        net.save_graph(out_path)

        with open(out_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        custom_css = """
        <style>
        .vis-tooltip {
            white-space: pre-wrap !important; /* \n을 줄바꿈으로 인식 */
            max-width: 400px !important;      /* 가로 폭 제한 */
            word-wrap: break-word !important; /* 긴 텍스트 자동 쪼개기 */
            padding: 10px !important;
            font-family: Arial, sans-serif !important;
            font-size: 13px !important;
            background-color: #f8fafc !important; /* 깔끔한 배경색 */
            color: #1e293b !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1) !important;
        }
        </style>
        </head>
        """

        html_content = html_content.replace("</head>", custom_css)

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Graph Visualization saved to {out_path}")

        return out_path