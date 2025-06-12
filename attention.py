import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask
    


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class DynamicSparseAttention(nn.Module):
    """
    Dynamic Sparse Attention Mechanism.

    This module replaces the standard full attention in a Transformer, aiming to improve
    efficiency and performance by dynamically generating a sparse attention pattern.
    It combines several strategies to determine which Keys each Query should attend to:
    1. Dynamic Local Window: Sets a dynamically sized local attention window for each query
       based on local volatility.
    2. Dynamic Future Window: Sets a dynamically sized future-looking window for each query
       based on the sequence's trend changes.
    3. Keypoint Detection: Identifies abrupt change points in the sequence and forces all
       queries to attend to these keypoints, while keypoint queries can attend to all others.
    4. Stratified Global Sampling: Samples global keypoints from different segments of the
       sequence based on an importance score to capture long-range dependencies.
    5. Random Connection Augmentation: If the generated connections are too sparse, random
       connections are added to ensure sufficient information flow.
    """
    def __init__(self, key_dim, num_heads, local_window, future_window, threshold=0.5, dropout=0.05, mask=False):
        """
        Initializes the DynamicSparseAttention module.

        param key_dim: The dimension of the input features (d_model).
        param num_heads: The number of attention heads.
        param local_window: The base size for the dynamic local window.
        param future_window: The base size for the dynamic future window.
        param threshold: The standard deviation multiplier for the keypoint detection threshold.
        param dropout: The dropout rate.
        param mask: Whether to apply an upper-triangular causal mask (for decoder self-attention).
        """
        super(DynamicSparseAttention, self).__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.local_base = local_window   # Base size for the local window
        self.future_base = future_window # Base size for the future window
        self.threshold = threshold       # Threshold for keypoint detection
        self.mask = mask                 # Flag for applying causal mask

        assert key_dim % num_heads == 0, "Key dimension must be divisible by the number of heads"
        self.head_dim = key_dim // num_heads

        # Linear projection layers for Query, Key, and Value
        self.q_proj = nn.Linear(key_dim, key_dim)
        self.k_proj = nn.Linear(key_dim, key_dim)
        self.v_proj = nn.Linear(key_dim, key_dim)
        # Output linear projection layer
        self.out_proj = nn.Linear(key_dim, key_dim)
        
        # Layer normalization for Q, K, V to stabilize training
        self.norm_q = nn.LayerNorm(key_dim)
        self.norm_k = nn.LayerNorm(key_dim)
        self.norm_v = nn.LayerNorm(key_dim)
        
        # Gating mechanism to adaptively modulate the attention output
        self.gate = nn.Sequential(
            nn.Linear(key_dim, key_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights using Xavier uniform distribution
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _calculate_dynamic_windows(self, sequence):
        """
        Calculates dynamic local and future window sizes for each time step based on the
        sequence's local volatility and trend changes.

        param sequence: Input sequence tensor of shape (batch_size, seq_len, feature_dim).
        return: A tuple of (local_window, future_window), both of shape (B, L).
        """
        batch_size, seq_len, feature_dim = sequence.shape

        # 1. Calculate local volatility to determine local window size
        # Calculate differences at multiple scales as a proxy for volatility
        deltas_short = torch.abs(sequence[:, 1:] - sequence[:, :-1])
        deltas_short_padded = F.pad(deltas_short, (0, 0, 0, 1), "constant", 0) # Pad to maintain length
        
        if seq_len >= 3:
            deltas_mid = torch.abs(sequence[:, 2:] - sequence[:, :-2]) / 2
            deltas_mid_padded = F.pad(deltas_mid, (0, 0, 0, 2), "constant", 0)
        else:
            deltas_mid_padded = torch.zeros_like(deltas_short_padded)
            
        if seq_len >= 5:
            deltas_long = torch.abs(sequence[:, 4:] - sequence[:, :-4]) / 4
            deltas_long_padded = F.pad(deltas_long, (0, 0, 0, 4), "constant", 0)
        else:
            deltas_long_padded = torch.zeros_like(deltas_short_padded)
        
        # Ensure all padded delta tensors have the same shape
        assert deltas_short_padded.shape == deltas_mid_padded.shape == deltas_long_padded.shape
        
        # Combine volatilities from different scales with weights
        deltas_combined = deltas_short_padded * 0.5 + deltas_mid_padded * 0.3 + deltas_long_padded * 0.2
        
        # Average over the feature dimension to get an importance score for each time step
        importance = deltas_combined.mean(dim=-1)
        
        # Normalize importance scores to the [0, 1] range
        importance = (importance - importance.min(dim=1, keepdim=True)[0]) / (
            importance.max(dim=1, keepdim=True)[0] - importance.min(dim=1, keepdim=True)[0] + 1e-6
        )
        
        # Calculate dynamic local window size based on importance scores
        # The window size varies dynamically between [0.5 * base, 1.0 * base]
        local_window = torch.round(
            self.local_base * (0.5 + 0.5 * importance)
        ).long()
        
        # Clamp the window size to a reasonable range
        local_window = torch.clamp(local_window, min=2, max=min(seq_len, self.local_base * 2))

        # 2. Calculate trend changes to determine future window size
        # Use cumulative sum to approximate the trend
        trend = torch.cumsum(sequence - sequence.mean(dim=1, keepdim=True), dim=1)
        # Calculate the rate of change of the trend
        trend_change = torch.abs(trend[:, 1:] - trend[:, :-1])
        trend_change_padded = F.pad(trend_change, (0, 0, 0, 1), "constant", 0)
        
        # Normalize trend importance
        trend_importance = trend_change_padded.mean(dim=-1)
        trend_importance = (trend_importance - trend_importance.min(dim=1, keepdim=True)[0]) / (
            trend_importance.max(dim=1, keepdim=True)[0] - trend_importance.min(dim=1, keepdim=True)[0] + 1e-6
        )
        
        # Calculate dynamic future window size based on trend importance
        future_window = torch.round(
            self.future_base * (0.5 + 0.5 * trend_importance)
        ).long()
        
        # Clamp the future window size
        future_window = torch.clamp(future_window, min=1, max=min(seq_len//2, self.future_base))

        return local_window, future_window

    def _detect_keypoints(self, sequence):
        """
        Detects keypoints (e.g., abrupt changes, peaks) in the sequence.
        Keypoints can be attended to by all other time steps, and they can attend to all others.
        
        param sequence: Input sequence tensor of shape (batch_size, seq_len, feature_dim).
        return: keypoint_mask (B, L, L), a boolean mask marking connections related to keypoints.
        """
        batch_size, seq_len, feature_dim = sequence.shape

        # Calculate rates of change at multiple scales
        deltas = []
        for scale in [1, 2, 3, 5]:
            if seq_len > scale:
                delta = torch.abs(sequence[:, scale:] - sequence[:, :-scale]) / scale
                delta = F.pad(delta, (0, 0, 0, scale), "constant", 0)
                deltas.append(delta)
            else:
                deltas.append(torch.zeros_like(sequence))

        # Weighted combination of multi-scale rates of change
        weights = [0.4, 0.3, 0.2, 0.1] 
        combined_delta = sum(w * d for w, d in zip(weights, deltas))

        # Average over the feature dimension
        delta_mean = combined_delta.mean(dim=-1)

        keypoints = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=sequence.device)
        
        # Detect keypoints for each sample in the batch independently
        for b in range(batch_size):
            # Adaptive threshold: mean + N * std
            threshold = delta_mean[b].mean() + self.threshold * delta_mean[b].std()

            # Find points that are local peaks and exceed the threshold
            for i in range(1, seq_len-1):
                if (delta_mean[b, i] > delta_mean[b, i-1] and 
                    delta_mean[b, i] > delta_mean[b, i+1] and
                    delta_mean[b, i] > threshold):
                    keypoints[b, i] = True

            # Check the first and last points of the sequence
            if delta_mean[b, 0] > threshold:
                keypoints[b, 0] = True
            if delta_mean[b, -1] > threshold:
                keypoints[b, -1] = True

        # Generate the attention mask based on detected keypoints
        keypoint_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=sequence.device)
        
        for b in range(batch_size):
            key_indices = torch.where(keypoints[b])[0]
            
            if len(key_indices) > 0:
                # Set the rows and columns corresponding to keypoints to True.
                # This means: 1. Any query can attend to these keypoints.
                #             2. Queries at keypoint positions can attend to any other point.
                keypoint_mask[b, :, key_indices] = True
                keypoint_mask[b, key_indices, :] = True
            else:
                # If no keypoints are detected, sample a few points uniformly as a fallback
                # to ensure connectivity.
                sampled_indices = torch.linspace(0, seq_len-1, min(5, seq_len)).long()
                keypoint_mask[b, :, sampled_indices] = True
        
        return keypoint_mask

    def _get_global_random_indices(self, input_tensor, num_layers=4, samples_per_layer=4):
        """
        Performs stratified sampling of globally important points to capture long-range dependencies.
        
        param input_tensor: Input tensor of shape (B, L, C).
        param num_layers: The number of strata (segments).
        param samples_per_layer: The number of samples to draw from each stratum.
        return: global_indices (B, K), indices of the globally sampled points.
        """
        batch_size, seq_len, feature_dim = input_tensor.shape

        num_global_points = min(seq_len, num_layers * samples_per_layer)
        global_indices = torch.zeros((batch_size, num_global_points), dtype=torch.long, device=input_tensor.device)
        
        for b in range(batch_size):
            seq_features = input_tensor[b]
            # Calculate an importance score for each time step
            # 1. Magnitude: L2 norm of the feature vector.
            magnitude = torch.norm(seq_features, dim=-1)
            # 2. Rate of change: Mean absolute difference with the previous time step.
            if seq_len > 1:
                changes = torch.abs(seq_features[1:] - seq_features[:-1]).mean(dim=-1)
                changes = F.pad(changes, (0, 1), "constant", 0)
            else:
                changes = torch.zeros_like(magnitude)
            # 3. Frequency score: Comparison of local frequency to global frequency.
            if seq_len >= 4:
                diffs = seq_features[1:] - seq_features[:-1]
                if diffs.size(0) > 1:
                    freq = torch.var(diffs, dim=0).mean() # Proxy for global frequency
                    freq_score = torch.zeros_like(magnitude)
                    for i in range(1, seq_len-1):
                        local_var = torch.var(seq_features[max(0, i-2):min(seq_len, i+3)], dim=0).mean() # Proxy for local frequency
                        freq_score[i] = local_var / (freq + 1e-6)
                else:
                    freq_score = torch.zeros_like(magnitude)
            else:
                freq_score = torch.zeros_like(magnitude)
            
            # Combine the three scores
            importance = magnitude * 0.3 + changes * 0.4 + freq_score * 0.3
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-6)
            
            if seq_len <= num_global_points:
                indices = torch.arange(seq_len, device=input_tensor.device)
            else:
                # Stratified Sampling 
                indices = []
                segment_size = seq_len // num_layers
                
                for l in range(num_layers):
                    start_idx = l * segment_size
                    end_idx = min(seq_len, (l + 1) * segment_size)

                    segment_importance = importance[start_idx:end_idx]

                    if len(segment_importance) <= samples_per_layer:
                        segment_indices = torch.arange(start_idx, end_idx, device=input_tensor.device)
                    else:
                        # Perform multinomial sampling based on importance scores
                        probs = segment_importance / segment_importance.sum()
                        segment_indices = torch.multinomial(probs, samples_per_layer, replacement=False)
                        segment_indices += start_idx
                    
                    indices.append(segment_indices)
                
                indices = torch.cat(indices)
            global_indices[b, :len(indices)] = indices
        
        return global_indices

    def _get_sparse_indices(self, seq_len_q, seq_len_kv, queries, sparsity_ratio=0.3):
        """
        Integrates all strategies to generate the final sparse attention index matrix.
        
        param seq_len_q: Length of the query sequence.
        param seq_len_kv: Length of the key/value sequence.
        param queries: The query tensor (B, L, C).
        param sparsity_ratio: The base target sparsity for random augmentation.
        return: sparse_indices (B, L_q, L_kv), the final boolean attention mask.
        """
        batch_size = queries.size(0)

        # 1. Apply dynamic local and future windows
        local_window, future_window = self._calculate_dynamic_windows(queries)
        sparse_indices = torch.zeros((batch_size, seq_len_q, seq_len_kv), dtype=torch.bool, device=queries.device)

        for b in range(batch_size):
            for q in range(seq_len_q):
                # Local window
                local_size = local_window[b, q]
                start_idx = max(0, q - local_size)
                end_idx = min(seq_len_kv, q + local_size + 1)
                sparse_indices[b, q, start_idx:end_idx] = True
            
            for q in range(seq_len_q):
                if q < seq_len_kv: 
                    # Future window
                    future_size = future_window[b, q]
                    end_idx = min(seq_len_kv, q + future_size + 1)
                    sparse_indices[b, q, q:end_idx] = True
        
        # 2. Apply keypoint mask
        keypoint_mask = self._detect_keypoints(queries)
        sparse_indices = sparse_indices | keypoint_mask  # Combine using logical OR

        # 3. Apply globally sampled points
        global_indices = self._get_global_random_indices(queries)
        for b in range(batch_size):
            valid_indices = global_indices[b][global_indices[b] < seq_len_kv]
            if len(valid_indices) > 0:
                # All queries can attend to these global points
                sparse_indices[b, :, valid_indices] = True
        
        # 4. Augment with random connections to meet an adaptive sparsity target
        # Calculate sequence complexity; more complex sequences are allowed denser connections.
        complexity = queries.var(dim=-1).mean(dim=1)
        normalized_complexity = (complexity - complexity.min()) / (complexity.max() - complexity.min() + 1e-6)
        
        for b in range(batch_size):
            # Adaptive sparsity ratio
            adaptive_ratio = sparsity_ratio * (0.5 + normalized_complexity[b])
            adaptive_ratio = min(0.5, max(0.1, adaptive_ratio)) # Clamp between [0.1, 0.5]
            current_density = sparse_indices[b].float().mean().item()
            
            # If current density is below the target, add random connections
            if current_density < adaptive_ratio:
                needed_connections = int((adaptive_ratio - current_density) * seq_len_q * seq_len_kv)
                unconnected = ~sparse_indices[b]
                unconnected_indices = torch.nonzero(unconnected, as_tuple=True)
                
                if len(unconnected_indices[0]) > 0:
                    random_indices = torch.randperm(len(unconnected_indices[0]), device=queries.device)
                    random_indices = random_indices[:min(needed_connections, len(random_indices))]

                    q_indices = unconnected_indices[0][random_indices]
                    kv_indices = unconnected_indices[1][random_indices]
                    sparse_indices[b, q_indices, kv_indices] = True
        
        # 5. Ensure every query attends to at least one key to prevent NaNs after softmax
        for b in range(batch_size):
            for q in range(seq_len_q):
                if not sparse_indices[b, q].any():
                    if q < seq_len_kv:
                        sparse_indices[b, q, q] = True # Attend to self
                    else:
                        sparse_indices[b, q, seq_len_kv-1] = True # Attend to the last key
        
        return sparse_indices

    def forward(self, queries, keys, values, attn_mask=None):
        batch_size, seq_len_q, _ = queries.size()
        seq_len_kv = keys.size(1)

        # Apply layer normalization to inputs first
        queries = self.norm_q(queries)
        keys = self.norm_k(keys)
        values = self.norm_v(values)

        # Linearly project and reshape for multi-head attention
        Q = self.q_proj(queries).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(keys).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(values).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Get the dynamic sparse attention mask
        sparse_indices = self._get_sparse_indices(seq_len_q, seq_len_kv, queries)
        # Expand mask to match the multi-head dimension (B, H, L, S)
        sparse_indices = sparse_indices.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        scale = 1.0 / math.sqrt(self.head_dim)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Apply masks
        # Create a mask that is the inverse of the sparse indices (True for positions to be masked)
        attention_mask = ~sparse_indices
        # Fill masked positions with a large negative value, which becomes 0 after softmax
        scores.masked_fill_(attention_mask, -1e9)

        # If required, apply the causal mask
        if self.mask:
            causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_kv, device=queries.device), diagonal=1).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
            scores.masked_fill_(causal_mask, -1e9)

        # Apply any external mask passed to the function (e.g., padding mask)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)

        # Calculate attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to V
        output = torch.matmul(attn_weights, V)
        
        # Reshape the output tensor
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.key_dim)
        
        # Apply the gating mechanism
        gate_values = self.gate(queries)
        output = output * gate_values
        
        # Final linear projection
        output = self.out_proj(output)
        
        return output, attn_weights


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, dpa=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.dim = d_model
        self.dpa = dpa

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.permute(0, 2, 1, 3).contiguous()  # (B, H, L, d_values)
        out = out.view(B, L, -1)  # (B, L, H * d_values)
        return self.out_projection(out), attn
    
