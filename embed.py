import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelativeEventEmbedding(nn.Module):
    """
    Relative Event Embedding

    This module aims to replace traditional, data-agnostic positional encodings (like Sinusoidal
    Positional Encoding). It learns temporal relationships directly from the data itself in two
    complementary ways to generate a rich, data-driven embedding.

    1.  Multi-scale Relative Positional Encoding:
          Calculates the relative changes in features (e.g., distance, energy) across different time spans (scales).
          This approach captures both local and global dynamic properties of the sequence.

    2.  Event Contextual Encoding:
          Constructs a "hybrid similarity" matrix by combining cosine similarity (for angular similarity) and Euclidean distance (for magnitude similarity).
          Based on this similarity, an attention mechanism computes a contextual representation for each time point (event).
          This allows the model to understand which events are similar in feature space and aggregate information accordingly.
          
    The final embedding is the sum of these two encodings, providing the model with dual information about "how events change relative to each other" and "which events are similar in their features."
    """
    def __init__(self, d_model, feature_dim, initial_distance_scales=5, initial_alpha=0.5):
        """
        Initialization.

        param d_model: The target embedding dimension of the model.
        param feature_dim: The feature dimension of the input sequence.
        param initial_distance_scales: The initial value for the maximum time span (scale) used to calculate relative features. This is a learnable parameter.
        param initial_alpha: The initial value for balancing the weights of cosine similarity and Euclidean distance in the hybrid similarity metric. Also learnable.
        """
        super(RelativeEventEmbedding, self).__init__()

        # A linear layer to project the multi-scale relative features (3 dims: rel_dist, rel_energy, weighted_dist) to d_model.
        self.relative_positional_encoding = nn.Linear(3, d_model)  
        # A linear layer to project the event contextual encoding (feature_dim) to d_model.
        self.event_encoding_projection = nn.Linear(feature_dim, d_model)  
        
        # Layer Normalization to stabilize the training process.
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # Learnable Parameters 
        # Dynamically determines the maximum time span (scale) for relative feature calculation.
        self.distance_scales = nn.Parameter(torch.tensor(float(initial_distance_scales)))  
        
        # Dynamically balances the weights of cosine similarity and Euclidean distance.
        self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))  
        
        # Dynamically controls the bandwidth (sigma) of the Gaussian kernel function.
        self.sigma_weights = nn.Parameter(torch.ones(1))  
        
        # Query and Key projection layers for calculating inter-event attention scores.
        self.event_query = nn.Linear(feature_dim, feature_dim)
        self.event_key = nn.Linear(feature_dim, feature_dim)
        
        # A learnable weight for the relative distance at each time span (scale).
        self.scale_weights = nn.Parameter(F.softplus(torch.randn(int(initial_distance_scales))))  
    
    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        
        # 1: Calculate Multi-scale Relative Positional Encoding 
        
        delta_distances_list, delta_energies_list, weighted_distances_list = [], [], []
        # Clamp the learnable distance_scales parameter to a reasonable range [3, 15] for stability.
        distance_scales = torch.clamp(self.distance_scales, min=3, max=15).round().int()

        # Iterate over different time spans (scales).
        for scale in range(1, distance_scales + 1):  
            if scale < seq_len:
                # Calculate feature differences between time steps that are `scale` apart.
                delta_distance = distances[:, :-scale] - distances[:, scale:]  
                delta_energy = energies[:, :-scale] - energies[:, scale:]  
                
                # Calculate weighted distance: use a learnable weight and an exponential decay function.
                # Intuition: The impact of a relative distance decreases exponentially as the distance itself grows.
                weighted_distance = self.scale_weights[scale - 1] * torch.exp(
                    -torch.abs(delta_distance) / (2 ** scale)
                )  

                # Since differencing shortens the sequence, pad at the beginning to restore original length.
                padding_shape = (batch_size, scale)  
                delta_distance = torch.cat([torch.zeros(padding_shape, device=x.device), delta_distance], dim=1)
                delta_energy = torch.cat([torch.zeros(padding_shape, device=x.device), delta_energy], dim=1)
                weighted_distance = torch.cat([torch.zeros(padding_shape, device=x.device), weighted_distance], dim=1)
                
                # Collect results for each scale.
                delta_distances_list.append(delta_distance.unsqueeze(-1))  
                delta_energies_list.append(delta_energy.unsqueeze(-1))
                weighted_distances_list.append(weighted_distance.unsqueeze(-1))

        # Concatenate the results from all scales.
        delta_distances = torch.cat(delta_distances_list, dim=-1)  
        delta_energies = torch.cat(delta_energies_list, dim=-1)
        weighted_distances = torch.cat(weighted_distances_list, dim=-1)

        # Stack the three types of relative features, creating a tensor of shape (B, L, num_scales, 3).
        relative_positions = torch.stack(
            [delta_distances, delta_energies, weighted_distances], dim=-1
        )  

        # Project the 3 relative features to d_model, then sum over the scale dimension to aggregate multi-scale info.
        relative_positions_encoded = self.relative_positional_encoding(relative_positions).sum(dim=2)  
        relative_positions_encoded = self.layer_norm1(relative_positions_encoded)  # Normalize.

        # 2: Calculate Event Contextual Encoding 

        # Calculate pairwise cosine similarity.
        x_norm = F.normalize(x, p=2, dim=-1)
        cos_sim = torch.matmul(x_norm, x_norm.transpose(1, 2))
        cos_sim = (cos_sim + 1) / 2  # Normalize to [0, 1].
        
        # Calculate pairwise Euclidean distance.
        euclidean_dist = torch.norm(x.unsqueeze(2) - x.unsqueeze(1), p=2, dim=-1)
    
        # Clip outliers in the Euclidean distance (Winsorizing) for robustness.
        mean_dist = euclidean_dist.mean()
        std_dist = euclidean_dist.std()
        euclidean_dist = torch.where(
            torch.abs(euclidean_dist - mean_dist) > 2 * std_dist,
            torch.sign(euclidean_dist - mean_dist) * (2 * std_dist) + mean_dist,
            euclidean_dist
        )
        # Normalize the clipped Euclidean distance to [0, 1].
        euclidean_dist = (euclidean_dist - euclidean_dist.min()) / (euclidean_dist.max() - euclidean_dist.min() + 1e-6)

        # Calculate hybrid similarity: a weighted average of cosine similarity and (1 - normalized Euclidean distance).
        # We use (1 - dist) because smaller distance means higher similarity.
        alpha = torch.sigmoid(self.alpha) # Ensure alpha is between (0, 1).
        hybrid_similarity = alpha * cos_sim + (1 - alpha) * (1 - euclidean_dist)

        # Calculate the bandwidth sigma for the Gaussian kernel; softplus ensures it's positive.
        sigma = F.softplus(self.sigma_weights) + 1e-3  

        # Calculate standard content-based attention scores.
        queries = self.event_query(x)
        keys = self.event_key(x)
        event_attention_scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(feature_dim)
        event_weights = F.softmax(event_attention_scores, dim=-1)

        # The final attention matrix E is an element-wise product of content attention (event_weights)
        # and similarity-based attention (Gaussian kernel). This means the association between two events
        # is strong only if they are both relevant in content (high Q-K score) and similar in features (high Gaussian kernel value).
        E = event_weights * torch.exp(-hybrid_similarity / (2 * sigma**2))  

        # Use matrix E to perform a weighted aggregation of the original input x, yielding a contextual representation for each event.
        PE_event = torch.matmul(E, x)  

        # Project the contextual representation to d_model dimension and normalize.
        PE_event_projected = self.event_encoding_projection(PE_event)  
        PE_event_projected = self.layer_norm2(PE_event_projected)  

        # Add the two types of encodings to get the final embedding.
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
