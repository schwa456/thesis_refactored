import torch
import torch.nn as nn
import torch.nn.functional as F

class SchemaNodeClassifier(nn.Module):
    """
    GAT의 구조적 임베딩(Node)과 PLM의 시맨틱 임베딩(Query)을 Concat하여
    각 노드가 질문과 관련이 있는지(Relevance) 판단하는 이진 분류기입니다.
    """
    def __init__(self, node_dim, query_dim, hidden_dim=256, dropout=0.2):
        super(SchemaNodeClassifier, self).__init__()
        
        # 입력: [Node_Emb + Query_Emb] (예: 256 + 384 = 640)
        self.classifier = nn.Sequential(
            nn.Linear(node_dim + query_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1) 
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, node_embs, query_embs):
        """
        Args:
            node_embs: [N, Node_Dim] (GAT 출력 노드들)
            query_embs: [B, Query_Dim] (PLM Query CLS/Mean 벡터)
        Returns:
            logits: [B, N] (Sigmoid 전)
            probs: [B, N] (Sigmoid 후)
        """
        B = query_embs.size(0)
        N = node_embs.size(0)
        
        # 1. 브로드캐스팅을 위한 차원 확장
        # query_embs: [B, Query_Dim] -> [B, 1, Query_Dim] -> [B, N, Query_Dim]
        q_expanded = query_embs.unsqueeze(1).expand(-1, N, -1)
        # node_embs: [N, Node_Dim] -> [1, N, Node_Dim] -> [B, N, Node_Dim]
        n_expanded = node_embs.unsqueeze(0).expand(B, -1, -1)
        
        # 2. Concat (Feature Fusion)
        combined = torch.cat([n_expanded, q_expanded], dim=-1) # [B, N, Node_Dim + Query_Dim]
        
        # 3. Classification
        logits = self.classifier(combined).squeeze(-1) # [B, N]
        probs = self.sigmoid(logits)
        
        return logits, probs