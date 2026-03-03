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



# Define the autoencoder with efficient attention
class CoxformerAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, seq_length=512, embedding_dim=64, original_input_dim=None):
        super(CoxformerAE, self).__init__()
        self.input_dim = input_dim
        self.original_input_dim = original_input_dim
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim

        # Ensure input_dim is divisible by seq_length
        assert input_dim % seq_length == 0, "input_dim must be divisible by seq_length"

        # Reshape input into (batch_size, seq_length, embedding_dim)
        self.chunk_size = input_dim // seq_length

        # Define projection layers if necessary
        if self.chunk_size != self.embedding_dim:
            self.input_projection = nn.Linear(self.chunk_size, self.embedding_dim)
            self.output_projection = nn.Linear(self.embedding_dim, self.chunk_size)
        else:
            self.input_projection = None
            self.output_projection = None

        # Positional encoding (optional)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, embedding_dim))

        # Encoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=8, dim_feedforward=2048, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

        # Final linear layer to get latent representation
        self.encoder_linear = nn.Linear(seq_length * embedding_dim, hidden_dim)

        # Decoder layers
        self.decoder_linear = nn.Linear(hidden_dim, seq_length * embedding_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim, nhead=8, dim_feedforward=2048, activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2)

    def forward(self, x):
        batch_size = x.size(0)

        # Reshape input to (batch_size, seq_length, chunk_size)
        x = x.view(batch_size, self.seq_length, self.chunk_size)

        # Project chunk_size to embedding_dim if necessary
        if self.input_projection is not None:
            x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding

        # Permute for transformer (seq_length, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)

        # Encoder
        encoded = self.transformer_encoder(x)

        # Flatten and get latent representation
        encoded_flat = encoded.permute(1, 0, 2).contiguous().view(batch_size, -1)
        latent = self.encoder_linear(encoded_flat)

        # Decoder
        decoded = self.decoder_linear(latent)
        decoded = decoded.view(batch_size, self.seq_length, self.embedding_dim).permute(1, 0, 2)
        decoded = self.transformer_decoder(decoded, encoded)

        # Output layer
        decoded = decoded.permute(1, 0, 2)
        if self.output_projection is not None:
            decoded = self.output_projection(decoded)

        # Reshape to original input shape
        decoded = decoded.contiguous().view(batch_size, -1)
        return decoded, latent