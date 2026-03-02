import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class CoxformerNet(nn.Module):
    """
    Improved GraphSAGE model with support for edge features (correlation) as input
    """
    def __init__(self, input_dim, hidden_dims, edge_dim=1, dropout=0.2, use_edge_features=True):
        super(CoxformerNet, self).__init__()
        
        self.num_layers = len(hidden_dims)
        self.use_edge_features = use_edge_features
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.convs.append(SAGEConv(hidden_dims[i-1], hidden_dims[i]))
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(dim) for dim in hidden_dims])
        
        # Edge weight prediction head - includes edge features
        if use_edge_features:
            edge_input_dim = hidden_dims[-1] * 2 + edge_dim
        else:
            edge_input_dim = hidden_dims[-1] * 2
            
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        
        # Edge feature preprocessing layer
        if use_edge_features:
            self.edge_feature_transform = nn.Sequential(
                nn.Linear(edge_dim, edge_dim * 2),
                nn.ReLU(),
                nn.Linear(edge_dim * 2, edge_dim),
                nn.Tanh()
            )
    
    def forward(self, x, edge_index, num_sampled_nodes_per_hop=None):
        """
        Forward propagation with sampling support
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if x.shape[0] > 1:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x
    
    def predict_edges(self, node_embeddings, edge_index, edge_attr=None):
        """
        Predict edge weights using edge features
        """
        row, col = edge_index
        
        if self.use_edge_features and edge_attr is not None:
            edge_features_processed = self.edge_feature_transform(edge_attr)
            edge_embeddings = torch.cat([
                node_embeddings[row], 
                node_embeddings[col], 
                edge_features_processed
            ], dim=1)
        else:
            edge_embeddings = torch.cat([node_embeddings[row], node_embeddings[col]], dim=1)
        
        edge_weights = self.edge_mlp(edge_embeddings)
        return edge_weights.squeeze()
