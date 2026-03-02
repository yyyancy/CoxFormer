# ====== Basic Python Utilities ======
import math   

# ====== Numerical Computing & Data Processing ======
import numpy as np        

# ====== Deep Learning (PyTorch) ======
import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_embeddings_to_divisible(x_embeddings, nhead: int):
    if isinstance(x_embeddings, np.ndarray):
        x = torch.from_numpy(x_embeddings)
    else:
        x = x_embeddings

    dim = int(x.shape[1])
    new_dim = int(math.ceil(dim / nhead) * nhead)
    pad_dim = new_dim - dim
    if pad_dim == 0:
        return x_embeddings, 0

    x_pad = F.pad(x, (0, pad_dim), mode="constant", value=0.0)

    if isinstance(x_embeddings, np.ndarray):
        return x_pad.numpy(), pad_dim
    else:
        return x_pad, pad_dim


def weighted_huber_loss(pred, target, mask_non_zero,  weight_zero=0.5, weight_non_zero=0.5):
    abs_error = torch.abs(pred - target)
    residuals = pred - target
    delta = torch.std(residuals, dim=1, unbiased=False).unsqueeze(1)
    small_error_loss = 0.5 * abs_error ** 2
    large_error_loss = delta * abs_error - 0.5 * delta ** 2
    loss = torch.where(abs_error <= delta, small_error_loss, large_error_loss)
    loss_non_zero = loss * mask_non_zero * weight_non_zero 
    loss_zero = loss * (1 - mask_non_zero) * weight_zero
    return loss_non_zero.mean() + loss_zero.mean()


class ELU(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta
        

class ImageEncoder(nn.Module):
    def __init__(self, condition_dim, output_dim = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class FourierCoordEncoder(nn.Module):
    def __init__(self, condition_dim=2, output_dim=512, num_frequencies=32):
        super().__init__()
        self.freqs = 2 ** torch.arange(num_frequencies).float() * torch.pi
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim + 2 * condition_dim * num_frequencies, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, coords):
        freqs = self.freqs.to(coords.device)
        coords_expanded = coords.unsqueeze(-1) * freqs  # (N, 2, F)
        sincos = torch.cat([torch.sin(coords_expanded), torch.cos(coords_expanded)],dim=-1)
        pe = sincos.view(coords.shape[0], -1)  # (N, 2 * F * 2)
        return self.mlp(torch.cat([coords, pe], dim=-1))

class TriEncoder(nn.Module):
    def __init__(self, condition_dim, output_dim=512):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(condition_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, coords, image, learn) -> torch.Tensor:
        combine = torch.cat([coords, image, learn], dim=-1)
        return self.regressor(combine)


class TransformerDecoderWithSpatialQuery(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim=256, 
                  nhead=8, num_layers=1, dim_feedforward=1024, dropout=0.1, Modality='location', Pattern='spot'):
        super().__init__()
        self.Modality = Modality
        self.Pattern = Pattern
        self.input_dim = input_dim
        if self.Modality == 'location':
            parts = np.array_split(np.arange(input_dim), 2)
            parts_num = [len(p) for p in parts]
            self.spatial_encoder = FourierCoordEncoder(condition_dim=condition_dim['location'], output_dim=input_dim)
            
        if self.Modality == 'image':
            parts = np.array_split(np.arange(input_dim), 2)
            parts_num = [len(p) for p in parts] 
            self.image_encoder = ImageEncoder(condition_dim=condition_dim['image'], output_dim=input_dim)
        
        if self.Modality == 'combine':
            parts = np.array_split(np.arange(input_dim), 3)
            parts_num = [len(p) for p in parts] 
            self.image_encoder = ImageEncoder(condition_dim=condition_dim['image'], output_dim=parts_num[0])
            self.spatial_encoder = FourierCoordEncoder(condition_dim=condition_dim['location'], output_dim=parts_num[1])
            self.learnable_query = nn.Parameter(torch.randn(condition_dim['none'], parts_num[2]))
            nn.init.xavier_uniform_(self.learnable_query)
            self.combine_encoder = TriEncoder(condition_dim=input_dim, output_dim=input_dim)
        
        if self.Modality == 'none':
            self.learnable_query = nn.Parameter(torch.randn(condition_dim['none'], input_dim))
            nn.init.xavier_uniform_(self.learnable_query)
 
        decoder_layer = nn.TransformerDecoderLayer(
                d_model=input_dim, nhead=nhead, 
                dim_feedforward=dim_feedforward, dropout=dropout, 
                activation='relu', batch_first=True
            )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        

        self.output_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Linear(256, 1),
            ELU(alpha=0.01, beta=0.01),
        )

    def forward(self, x, condition):
        B = x.size(0)
        x = x.float()
        if "pixel" in self.Pattern:
            condition = condition.float()
            query = self.image_encoder(condition) # (B, N, hidden_dim)
                
        elif self.Pattern == "spot":
            if self.Modality  == 'location':
                condition= condition['location'].squeeze(0)
                query = self.spatial_encoder(condition)  # (N, hidden_dim)
                query = query.unsqueeze(0).repeat(B, 1, 1)  # (B, N, hidden_dim)
            
            elif self.Modality  == 'image':
                condition= condition['image'].squeeze(0)
                query = self.image_encoder(condition) # (N, hidden_dim)
                query = query.unsqueeze(0).repeat(B, 1, 1)  # (B, N, hidden_dim)
    
            elif self.Modality  == 'combine':
                cords = condition['location'].squeeze(0)
                image = condition['image'].squeeze(0)
                cords = self.spatial_encoder(cords)
                image = self.image_encoder(image) 
                query = self.combine_encoder(cords, image, self.learnable_query) # (N, hidden_dim)
                query = query.unsqueeze(0).repeat(B, 1, 1)  # (B, N, hidden_dim)

            elif self.Modality == 'none':
                query = self.learnable_query.unsqueeze(0).repeat(B, 1, 1)
    
        kv = x.unsqueeze(1)
        out = self.transformer_decoder(tgt=query, memory=kv, tgt_mask=None)  # (B, N, hidden_dim)        
        reg = self.output_proj(out).squeeze(-1)  # (B, N)  
        return reg    


