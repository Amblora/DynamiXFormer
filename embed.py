import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class RelativeEventEmbedding(nn.Module):
    def __init__(self, d_model, feature_dim, initial_distance_scales=5, initial_alpha=0.5):
        super(RelativeEventEmbedding, self).__init__()

        self.relative_positional_encoding = nn.Linear(3, d_model)  
        self.event_encoding_projection = nn.Linear(feature_dim, d_model)  
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.distance_scales = nn.Parameter(torch.tensor(float(initial_distance_scales)))  
        self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))  
        self.sigma_weights = nn.Parameter(torch.ones(1))  
        
        self.event_attention = nn.Linear(feature_dim, feature_dim)  
        
        self.event_query = nn.Linear(feature_dim, feature_dim)
        self.event_key = nn.Linear(feature_dim, feature_dim)
        
        self.scale_weights = nn.Parameter(F.softplus(torch.randn(int(initial_distance_scales))))  
    
    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        
        distances = x[:, :, 0]  
        energies = x[:, :, -1]  

        delta_distances_list, delta_energies_list, weighted_distances_list = [], [], []
        distance_scales = torch.clamp(self.distance_scales, min=3, max=15).round().int()

        for scale in range(1, distance_scales + 1):  
            if scale < seq_len:
                delta_distance = distances[:, :-scale] - distances[:, scale:]  
                delta_energy = energies[:, :-scale] - energies[:, scale:]  
                weighted_distance = self.scale_weights[scale - 1] * torch.exp(
                    -torch.abs(delta_distance) / (2 ** scale)
                )  

                padding_shape = (batch_size, scale)  
                delta_distance = torch.cat([torch.zeros(padding_shape, device=x.device), delta_distance], dim=1)
                delta_energy = torch.cat([torch.zeros(padding_shape, device=x.device), delta_energy], dim=1)
                weighted_distance = torch.cat([torch.zeros(padding_shape, device=x.device), weighted_distance], dim=1)
                delta_distances_list.append(delta_distance.unsqueeze(-1))  
                delta_energies_list.append(delta_energy.unsqueeze(-1))
                weighted_distances_list.append(weighted_distance.unsqueeze(-1))

        delta_distances = torch.cat(delta_distances_list, dim=-1)  
        delta_energies = torch.cat(delta_energies_list, dim=-1)
        weighted_distances = torch.cat(weighted_distances_list, dim=-1)

        relative_positions = torch.stack(
            [delta_distances, delta_energies, weighted_distances], dim=-1
        )  

        relative_positions_encoded = self.relative_positional_encoding(relative_positions).sum(dim=2)  
        relative_positions_encoded = self.layer_norm1(relative_positions_encoded)  # 新增归一化

        x_norm = F.normalize(x, p=2, dim=-1)
        cos_sim = torch.matmul(x_norm, x_norm.transpose(1, 2))
        cos_sim = (cos_sim + 1) / 2  
        euclidean_dist = torch.norm(x.unsqueeze(2) - x.unsqueeze(1), p=2, dim=-1)
    
        mean_dist = euclidean_dist.mean()
        std_dist = euclidean_dist.std()
        euclidean_dist = torch.where(
            torch.abs(euclidean_dist - mean_dist) > 2 * std_dist,
            torch.sign(euclidean_dist - mean_dist) * (2 * std_dist) + mean_dist,
            euclidean_dist
        )
        euclidean_dist = (euclidean_dist - euclidean_dist.min()) / (euclidean_dist.max() - euclidean_dist.min() + 1e-6)

        alpha = torch.sigmoid(self.alpha)
        hybrid_similarity = alpha * cos_sim + (1 - alpha) * (1 - euclidean_dist)

        sigma = F.softplus(self.sigma_weights) + 1e-3  

        queries = self.event_query(x)
        keys = self.event_key(x)
        event_attention_scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(feature_dim)
        event_weights = F.softmax(event_attention_scores, dim=-1)

        E = event_weights * torch.exp(-hybrid_similarity / (2 * sigma**2))  

        PE_event = torch.matmul(E, x)  

        PE_event_projected = self.event_encoding_projection(PE_event)  
        PE_event_projected = self.layer_norm2(PE_event_projected)  # 新增归一化

        combined_encoding = relative_positions_encoded + PE_event_projected  
        return combined_encoding

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=1, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, use_event_embeding=True, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.event_embedding = RelativeEventEmbedding(d_model=d_model, feature_dim=c_in)
        self.dropout = nn.Dropout(p=dropout)
        self.use_event_embeding = use_event_embeding

    def forward(self, x):
        if self.use_event_embeding: x = self.value_embedding(x) + self.position_embedding(x) + self.event_embedding(x)
        else: x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)