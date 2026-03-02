import pandas as pd
import numpy as np
import pickle
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset


# Now create the dataset and dataloader
class CoexpressDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]

# Define the autoencoder with efficient attention
class EfficientAttentionAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, seq_length=512, embedding_dim=64, original_input_dim=None):
        super(EfficientAttentionAutoencoder, self).__init__()
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

emb_list = ['coexpression','correlation']
for emb_name in emb_list:
    file_path = f'/home/wangshu/scratch/COPT/Embeddings/{emb_name}.pkl'
    with open(file_path , 'rb') as file:
        df = pickle.load(file)
    
    # Extract the Coexpress embeddings and convert them to a NumPy array
    coexpress_embeddings = np.array(df['Embedding'].tolist())
    
    # Ensure the embeddings are of type float32
    coexpress_embeddings = coexpress_embeddings.astype(np.float32)
    
    # Check for NaNs or infinite values
    if np.isnan(coexpress_embeddings).any() or np.isinf(coexpress_embeddings).any():
        print("Data contains NaNs or infinite values. Please clean your data.")
        exit()
    
    # Get input_dim and save original_input_dim
    input_dim = coexpress_embeddings.shape[1]
    original_input_dim = input_dim  # Save original input dimension
    print(f"Original input_dim: {input_dim}")
    
    # Set seq_length and embedding_dim
    seq_length = 512
    embedding_dim = 64
    
    # Ensure input_dim is divisible by seq_length
    if input_dim % seq_length != 0:
        # Adjust input_dim by padding
        padding_size = seq_length - (input_dim % seq_length)
        coexpress_embeddings = np.pad(coexpress_embeddings, ((0, 0), (0, padding_size)), mode='constant')
        input_dim = coexpress_embeddings.shape[1]
        print(f"Adjusted input_dim after padding: {input_dim}")
    
    # Now compute chunk_size
    chunk_size = input_dim // seq_length
    print(f"Chunk size: {chunk_size}")
     
    dataset = CoexpressDataset(coexpress_embeddings)
    batch_size = 32  # Adjust if necessary
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    # Initialize the model
    hidden_dim = 512
    model = EfficientAttentionAutoencoder(
        input_dim=input_dim,
        original_input_dim=original_input_dim,
        hidden_dim=hidden_dim,
        seq_length=seq_length,
        embedding_dim=embedding_dim
    )
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use DataParallel to utilize multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,     
        eta_min=1e-6   
    )
    
    # Training loop
    num_epochs = 200  
    patience = 50
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in tqdm(range(1, num_epochs+1), desc="Epochs", ncols=100):
        model.train()
        running_loss = 0.0
        for data in dataloader:
            data = data.to(device, non_blocking=True)
            # Forward pass
            outputs, _ = model(data)
            loss = criterion(outputs, data)  # Now 'outputs' and 'data' have the same size
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        scheduler.step()
        # —— EarlyStopping 判断 —— #
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no improvement in the last {patience} epochs).")
                break
    # Extract embeddings from the encoder
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for data in DataLoader(dataset, batch_size=batch_size):
            data = data.to(device)
            outputs, encoded = model(data)
            # Remove padding from 'encoded' if necessary
            all_embeddings.append(encoded.cpu().numpy())
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    # Add the new embeddings to the DataFrame
    df['Embedding'] = list(all_embeddings)
    
    # Save the updated DataFrame
    output_file = f'/home/wangshu/scratch/COPT/Embeddings/{emb_name}_rd.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(df, f)
    
    print(f"Embeddings reduced and saved to {output_file}")