import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt

class DynamicSparseAttention(nn.Module):
    def __init__(self, key_dim, num_heads, local_window, threshold=0.5, dropout=0.05, mask=False):
        super(DynamicSparseAttention, self).__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads
        # Base window size for local attention.
        self.local_base = local_window
        # Threshold for keypoint detection sensitivity.
        self.threshold = threshold
        # Flag to apply a causal mask for auto-regressive decoding.
        self.mask = mask
        assert key_dim % num_heads == 0, "Key dimension must be divisible by num_heads"
        self.head_dim = key_dim // num_heads

        # Standard Attention Projections and Normalization
        self.q_proj = nn.Linear(key_dim, key_dim)
        self.k_proj = nn.Linear(key_dim, key_dim)
        self.v_proj = nn.Linear(key_dim, key_dim)
        self.out_proj = nn.Linear(key_dim, key_dim)
        self.norm_q = nn.LayerNorm(key_dim)
        self.norm_k = nn.LayerNorm(key_dim)
        self.norm_v = nn.LayerNorm(key_dim)
        
        # Dynamic Window Calculation Network
        # A small MLP to learn an adjustment factor for the local window size.
        self.window_adjust_proj = nn.Sequential(
            nn.Linear(key_dim, key_dim // 4), nn.ReLU(),
            nn.Linear(key_dim // 4, 1), nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes projection layer weights with Xavier uniform for better training stability."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None: nn.init.constant_(proj.bias, 0)

    def _calculate_dynamic_context_window(self, sequence: torch.Tensor) -> torch.Tensor:
        """Calculates an adaptive local attention window size for each query."""
        # The MLP outputs an adjustment factor between 0.5 and 1.5.
        adjust_factor = 0.5 + self.window_adjust_proj(sequence).squeeze(-1)
        # Scale the base window size by the learned factor.
        context_window = torch.round(self.local_base * adjust_factor).long()
        seq_len = sequence.size(1)
        # Clamp the window size to prevent it from becoming too small or too large.
        max_win = min(seq_len, self.local_base * 2)
        context_window = torch.clamp(context_window, min=1, max=max_win)
        return context_window

    def _detect_keypoints(self, sequence: torch.Tensor) -> torch.Tensor:
        """Identifies critical points (keypoints) in the sequence based on rate of change."""
        batch_size, seq_len, _ = sequence.shape
        device = sequence.device
        deltas = []
        scales = [1, 2, 3, 5] # Define different time scales for differencing.
        
        # Multi-scale Change Calculation
        # Calculate the rate of change at different scales.
        for scale in scales:
            if seq_len > scale:
                delta = torch.abs(sequence[:, scale:] - sequence[:, :-scale]) / scale
                deltas.append(F.pad(delta, (0, 0, 0, scale), "constant", 0))
            else:
                deltas.append(torch.zeros_like(sequence))
        
        # Compute a weighted average of the changes across scales.
        weights = torch.tensor([0.4, 0.3, 0.2, 0.1], device=device).view(1, -1, 1, 1)
        combined_delta = torch.sum(torch.stack(deltas, dim=1) * weights, dim=1)
        delta_mean = combined_delta.mean(dim=-1)

        # Dynamic Thresholding and Peak Detection
        # Calculate a dynamic threshold based on the sequence's own statistics.
        threshold_val = delta_mean.mean(dim=1, keepdim=True) + self.threshold * delta_mean.std(dim=1, keepdim=True)
        # Find local peaks in the rate of change.
        padded_delta = F.pad(delta_mean, (1, 1), 'constant', -1)
        is_peak = (padded_delta[:, 1:-1] > padded_delta[:, :-2]) & (padded_delta[:, 1:-1] > padded_delta[:, 2:])
        # A point is a keypoint if it's a peak and exceeds the dynamic threshold.
        keypoints = (delta_mean > threshold_val) & is_peak
        # Ensure first and last points can be keypoints if they are significant.
        keypoints[:, 0] |= (delta_mean[:, 0] > threshold_val[:, 0])
        keypoints[:, -1] |= (delta_mean[:, -1] > threshold_val[:, 0])

        # Mask Generation
        # Create a mask where keypoints can attend to all other points (and vice versa).
        keypoint_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=device)
        has_keypoints = keypoints.any(dim=1)
        
        # Apply the keypoint mask for samples that have keypoints.
        if has_keypoints.any():
            b_indices_with_kps = torch.where(has_keypoints)[0]
            keypoint_mask[b_indices_with_kps, :, :] = keypoints[b_indices_with_kps].unsqueeze(1)
            keypoint_mask[b_indices_with_kps, :, :] |= keypoints[b_indices_with_kps].unsqueeze(2)

        # Fallback Mechanism
        # If a sequence has no keypoints, attend to a few uniformly sampled points to ensure information flow.
        if (~has_keypoints).any():
            b_indices_no_kps = torch.where(~has_keypoints)[0]
            num_samples = min(5, seq_len)
            sampled_indices = torch.linspace(0, seq_len - 1, num_samples, device=device).long()
            if b_indices_no_kps.numel() > 0:
                keypoint_mask[b_indices_no_kps[:, None], :, sampled_indices] = True
                keypoint_mask[b_indices_no_kps[:, None], sampled_indices, :] = True
                
        return keypoint_mask

    def _get_global_random_indices(self, input_tensor: torch.Tensor, num_layers: int = 4, samples_per_layer: int = 4) -> torch.Tensor:
        """Performs stratified sampling to select globally important indices."""
        batch_size, seq_len, _ = input_tensor.shape
        device = input_tensor.device
        
        # Importance Score Calculation
        # Importance is a combination of the point's magnitude and its local rate of change.
        magnitude = torch.norm(input_tensor, p=2, dim=-1)
        if seq_len > 1:
            changes = torch.abs(input_tensor[:, 1:] - input_tensor[:, :-1]).mean(dim=-1)
            changes = F.pad(changes, (0, 1), "constant", 0)
        else:
            changes = torch.zeros_like(magnitude)
        importance = magnitude * 0.5 + changes * 0.5
        
        # Normalize importance scores to be used as sampling probabilities.
        importance_min = importance.min(dim=1, keepdim=True)[0]
        importance_max = importance.max(dim=1, keepdim=True)[0]
        importance_norm = (importance - importance_min) / (importance_max - importance_min + 1e-6)
        
        num_global_samples = min(seq_len, num_layers * samples_per_layer)
        if seq_len <= num_global_samples:
            return torch.arange(seq_len, device=device).expand(batch_size, -1)
        else:
            # Stratified Sampling
            # Divide the sequence into segments (strata).
            segment_size = seq_len // num_layers
            padded_importance = F.pad(importance_norm, (0, num_layers * segment_size - seq_len))
            reshaped_importance = padded_importance.view(batch_size * num_layers, segment_size)
            # Sample from each segment based on the importance scores.
            probs = reshaped_importance / (reshaped_importance.sum(dim=-1, keepdim=True) + 1e-6)
            sampled_in_segment = torch.multinomial(probs, samples_per_layer, replacement=False)
            # Reconstruct the original indices from the sampled local indices.
            layer_offsets = torch.arange(num_layers, device=device, dtype=torch.long) * segment_size
            layer_offsets = layer_offsets.view(1, num_layers, 1)
            indices = (sampled_in_segment.view(batch_size, num_layers, samples_per_layer) + layer_offsets).view(batch_size, -1)
            return indices

    def _get_sparse_indices(self, seq_len_q: int, seq_len_kv: int, queries: torch.Tensor, sparsity_ratio: float = 0.3) -> torch.Tensor:
        """Combines all sparse strategies to generate the final attention mask."""
        batch_size = queries.size(0)
        device = queries.device
        
        # Combine Main Sparse Strategies
        # Start with the dynamic local attention mask.
        q_indices = torch.arange(seq_len_q, device=device).view(1, seq_len_q, 1)
        kv_indices = torch.arange(seq_len_kv, device=device).view(1, 1, seq_len_kv)
        context_window = self._calculate_dynamic_context_window(queries)
        context_mask = (kv_indices >= (q_indices - context_window.unsqueeze(2))) & \
                       (kv_indices <= (q_indices + context_window.unsqueeze(2)))
        sparse_indices = context_mask
        # Add the keypoint attention mask.
        sparse_indices |= self._detect_keypoints(queries)
        # Add the global attention mask.
        global_indices = self._get_global_random_indices(queries)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, global_indices.size(1))
        sparse_indices[batch_indices, :, global_indices] = True
        
        # Adaptive Random Connection Augmentation
        # Calculate the complexity of the sequence.
        complexity = queries.var(dim=-1).mean(dim=1)
        normalized_complexity = (complexity - complexity.min()) / (complexity.max() - complexity.min() + 1e-6)
        # Determine the target attention density based on complexity.
        adaptive_ratio = torch.clamp(sparsity_ratio * (0.5 + normalized_complexity), min=0.1, max=0.5)
        current_density = sparse_indices.float().mean(dim=(1, 2))
        # Calculate how many random connections are needed to reach the target density.
        needed_connections = torch.ceil((adaptive_ratio - current_density) * (seq_len_q * seq_len_kv)).long()
        needed_connections = torch.clamp(needed_connections, min=0)
        
        # Add the required number of random connections.
        needs_update_mask = needed_connections > 0
        if needs_update_mask.any():
            b_indices_to_update = torch.where(needs_update_mask)[0]
            for b_idx in b_indices_to_update:
                num_needed = needed_connections[b_idx].item()
                if num_needed > 0:
                    q_rand = torch.randint(0, seq_len_q, (num_needed,), device=device)
                    kv_rand = torch.randint(0, seq_len_kv, (num_needed,), device=device)
                    sparse_indices[b_idx, q_rand, kv_rand] = True

        # Final Sanity Check
        # Ensure that every query attends to at least one key to prevent NaN outputs.
        no_attention_rows = ~sparse_indices.any(dim=-1)
        if no_attention_rows.any():
            fix_indices_b, fix_indices_q = torch.where(no_attention_rows)
            # Make it attend to itself.
            fix_indices_kv = torch.clamp(fix_indices_q, max=seq_len_kv - 1)
            sparse_indices[fix_indices_b, fix_indices_q, fix_indices_kv] = True

        return sparse_indices

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attn_mask: torch.Tensor = None):
        batch_size, seq_len_q, _ = queries.size()
        seq_len_kv = keys.size(1)
        
        # Standard Transformer Attention Flow
        # Apply layer normalization and project queries, keys, and values.
        queries_norm = self.norm_q(queries)
        keys_norm = self.norm_k(keys)
        values_norm = self.norm_v(values)
        Q = self.q_proj(queries_norm).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(keys_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(values_norm).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Mask Application 
        # Generate the dynamic sparse mask.
        sparse_mask = self._get_sparse_indices(seq_len_q, seq_len_kv, queries_norm)
        # Invert the mask because PyTorch's attention function expects `True` for positions to be masked out.
        final_mask = ~sparse_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # If it's a decoder self-attention, apply an additional causal mask.
        if self.mask:
            causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_kv, device=queries.device, dtype=torch.bool), diagonal=1)
            final_mask |= causal_mask.unsqueeze(0).unsqueeze(0)

        # Incorporate any other external masks (e.g., padding mask).
        if attn_mask is not None:
            final_mask |= attn_mask
        
        # Scaled Dot-Product Attention
        # Use PyTorch's efficient fused attention implementation.
        output = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=final_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        # Output and Residual Connection
        # Reshape the output and apply the final projection.
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.key_dim)
        output = self.out_proj(output) + queries # Add residual connection.
        return output, None   


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

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

