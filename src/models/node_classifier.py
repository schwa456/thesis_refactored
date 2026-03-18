import torch
import torch.nn as nn
from gat_network import SchemaHeteroGAT
from plm_encoder import PLMEncoder

class SimpleNodeLinker(nn.Module):
    def __init__(self, hidden_dim, plm_name: str=""):
        super().__init__()

        #1. Encoders
        self.plm_encoder = PLMEncoder(plm_name)
        self.gat = SchemaHeteroGAT(hidden_channels=hidden_dim, out_channels=hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, batch_data, queries):
        node_embeddings_dict = self.gat(batch_data.x_dict, batch_data.edge_index_dict)

        query_embeddings = self.plm_encoder(queries)

        out_probs = {}

        for node_type in ['table', 'column']:
            if node_type not in node_embeddings_dict:
                continue
                
            node_feats = node_embeddings_dict[node_type]

            if hasattr(batch_data[node_type], 'batch'):
                batch_idx = batch_data[node_type].batch
            else:
                batch_idx = torch.zeros(node_feats.size(0), dtype=torch.long, device=node_feats.device)

            expanded_queries = query_embeddings[batch_idx]
            fused_feats = torch.cat([node_feats, expanded_queries], dim=-1)

            probs = self.classifier(fused_feats).squeeze(-1)
            out_probs[node_type] = probs
        
        return out_probs