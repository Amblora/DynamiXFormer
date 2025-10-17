import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelativeEventEmbedding(nn.Module):
    def __init__(self, d_model, feature_dim, distance_idx=0, energy_idx=-1, max_scales=15, initial_alpha=0.5):
        super(RelativeEventEmbedding, self).__init__()

        self.d_model = d_model
        self.feature_dim = feature_dim
        
        # Feature Indices
        # Explicitly define indices for key features to avoid hardcoding in the forward pass.
        self.distance_idx = distance_idx
        self.energy_idx = energy_idx
        
        self.max_scales = max_scales

        # Projection and Normalization Layers
        # Projects the 3-dimensional relative features (Δd, Δe, weighted_d) to the model dimension.
        self.relative_positional_encoding = nn.Linear(3, d_model)
        # Projects the event-aware context vector to the model dimension.
        self.event_encoding_projection = nn.Linear(feature_dim, d_model)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # Learnable Parameters for Adaptive Control
        # A learnable parameter to balance cosine similarity and Euclidean distance in the hybrid metric.
        self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))
        # A learnable parameter that controls the width (scale) of the Gaussian kernel.
        self.sigma_weights = nn.Parameter(torch.ones(1))
        
        # A learnable parameter to dynamically determine the number of scales to consider.
        self.learnable_num_scales = nn.Parameter(torch.tensor(float(max_scales / 2)))
        
        # Learnable weights for each scale in the weighted distance calculation.
        self.scale_weights = nn.Parameter(F.softplus(torch.randn(self.max_scales)))  
        
        # Standard Layers for Event Attention
        self.event_query = nn.Linear(feature_dim, feature_dim)
        self.event_key = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Multi-scale Relative Positional Encoding
        # Extract specific features (distance and energy) based on the provided indices.
        distances = x[:, :, self.distance_idx]
        energies  = x[:, :, self.energy_idx]

        delta_distances_list, delta_energies_list, weighted_distances_list = [], [], []
        
        # Loop through a predefined maximum number of scales to compute relative features.
        for scale in range(1, self.max_scales + 1):  
            if scale < seq_len:
                # Calculate the difference in distance and energy between time steps t and t-scale.
                delta_distance = distances[:, :-scale] - distances[:, scale:]  
                delta_energy = energies[:, :-scale] - energies[:, scale:]
                
                # Calculate a weighted distance that decays exponentially with the distance difference.
                weighted_distance = self.scale_weights[scale - 1] * torch.exp(-torch.abs(delta_distance) / (2 ** scale))  
                
                # Pad the results to maintain the original sequence length.
                delta_distance = F.pad(delta_distance, (scale, 0))
                delta_energy = F.pad(delta_energy, (scale, 0))
                weighted_distance = F.pad(weighted_distance, (scale, 0))

                delta_distances_list.append(delta_distance.unsqueeze(-1))  
                delta_energies_list.append(delta_energy.unsqueeze(-1))
                weighted_distances_list.append(weighted_distance.unsqueeze(-1))

        # Concatenate the features from all scales.
        delta_distances = torch.cat(delta_distances_list, dim=-1)  
        delta_energies = torch.cat(delta_energies_list, dim=-1)
        weighted_distances = torch.cat(weighted_distances_list, dim=-1)
        
        # Stack the three types of relative features together.
        relative_positions = torch.stack([delta_distances, delta_energies, weighted_distances], dim=-1)  
        
        # Project the features into the model's embedding space.
        encoded_positions = self.relative_positional_encoding(relative_positions)

        # Dynamically Select Scales
        # Determine the number of scales to use based on a learnable parameter.
        current_num_scales = torch.clamp(self.learnable_num_scales, min=1, max=self.max_scales).round().int()
        # Create a mask to zero out contributions from unused scales.
        mask = torch.arange(self.max_scales, device=x.device) < current_num_scales
        encoded_positions = encoded_positions * mask[None, None, :, None] 
        # Sum the contributions from the selected scales and apply layer normalization.
        relative_positions_encoded = encoded_positions.sum(dim=2)  
        relative_positions_encoded = self.layer_norm1(relative_positions_encoded)

        # Event-Driven Similarity Encoding
        # Hybrid Similarity Calculation 
        # Calculate cosine similarity to capture trend/directional similarity.
        x_norm = F.normalize(x, p=2, dim=-1)
        cos_sim = torch.matmul(x_norm, x_norm.transpose(1, 2))
        cos_sim = (cos_sim + 1) / 2 # Normalize to [0, 1]

        # Calculate robust Euclidean distance to capture magnitude similarity.
        euclidean_dist_raw = torch.cdist(x, x, p=2)
        # Use quantiles for robust normalization, making it less sensitive to outliers.
        v_min = torch.quantile(euclidean_dist_raw, 0.05, dim=-1, keepdim=True)
        v_max = torch.quantile(euclidean_dist_raw, 0.95, dim=-1, keepdim=True)
        euclidean_dist = torch.clamp((euclidean_dist_raw - v_min) / (v_max - v_min + 1e-6), 0, 1)

        # Combine the two similarity metrics using a learnable weight 'alpha'.
        alpha = torch.sigmoid(self.alpha)
        hybrid_similarity = alpha * cos_sim + (1 - alpha) * (1 - euclidean_dist)
        
        # Event Attention Calculation
        # Compute standard attention scores.
        queries = self.event_query(x)
        keys = self.event_key(x)
        event_attention_scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(self.feature_dim)
        event_weights = F.softmax(event_attention_scores, dim=-1)
        
        # Modulate the attention weights with the hybrid similarity using a Gaussian kernel.
        sigma = F.softplus(self.sigma_weights) + 1e-3
        E = event_weights * torch.exp(-hybrid_similarity / (2 * sigma**2))
        
        # Compute the final event-aware embedding and project it.
        PE_event = torch.matmul(E, x)
        PE_event_projected = self.event_encoding_projection(PE_event)
        PE_event_projected = self.layer_norm2(PE_event_projected)
        
        # Final Combined Embedding
        # Combine the relative positional encoding and the event-driven encoding.
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
