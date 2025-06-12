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
    def __init__(self, key_dim, num_heads, local_window, future_window, threshold=0.5, dropout=0.05, mask=False):
        super(DynamicSparseAttention, self).__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.local_base = local_window
        self.future_base = future_window
        self.threshold = threshold
        self.mask = mask

        assert key_dim % num_heads == 0, "Key dimension must be divisible by the number of heads"
        self.head_dim = key_dim // num_heads

        self.q_proj = nn.Linear(key_dim, key_dim)
        self.k_proj = nn.Linear(key_dim, key_dim)
        self.v_proj = nn.Linear(key_dim, key_dim)
        self.out_proj = nn.Linear(key_dim, key_dim)
        
        self.norm_q = nn.LayerNorm(key_dim)
        self.norm_k = nn.LayerNorm(key_dim)
        self.norm_v = nn.LayerNorm(key_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(key_dim, key_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _calculate_dynamic_windows(self, sequence):
        batch_size, seq_len, feature_dim = sequence.shape

        deltas_short = torch.abs(sequence[:, 1:] - sequence[:, :-1])
        
        deltas_short_padded = F.pad(deltas_short, (0, 0, 0, 1), "constant", 0)
        
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
        
        assert deltas_short_padded.shape == deltas_mid_padded.shape == deltas_long_padded.shape, \
            f"Shape mismatch: {deltas_short_padded.shape}, {deltas_mid_padded.shape}, {deltas_long_padded.shape}"
        
        deltas_combined = deltas_short_padded * 0.5 + deltas_mid_padded * 0.3 + deltas_long_padded * 0.2
        
        importance = deltas_combined.mean(dim=-1)
        
        importance = (importance - importance.min(dim=1, keepdim=True)[0]) / (
            importance.max(dim=1, keepdim=True)[0] - importance.min(dim=1, keepdim=True)[0] + 1e-6
        )
        
        local_window = torch.round(
            self.local_base * (0.5 + 0.5 * importance)
        ).long()
        
        local_window = torch.clamp(local_window, min=2, max=min(seq_len, self.local_base * 2))

        trend = torch.cumsum(sequence - sequence.mean(dim=1, keepdim=True), dim=1)
        trend_change = torch.abs(trend[:, 1:] - trend[:, :-1])
        trend_change_padded = F.pad(trend_change, (0, 0, 0, 1), "constant", 0)
        
        trend_importance = trend_change_padded.mean(dim=-1)
        trend_importance = (trend_importance - trend_importance.min(dim=1, keepdim=True)[0]) / (
            trend_importance.max(dim=1, keepdim=True)[0] - trend_importance.min(dim=1, keepdim=True)[0] + 1e-6
        )
        
        future_window = torch.round(
            self.future_base * (0.5 + 0.5 * trend_importance)
        ).long()
        
        future_window = torch.clamp(future_window, min=1, max=min(seq_len//2, self.future_base))

        return local_window, future_window

    def _detect_keypoints(self, sequence):
        batch_size, seq_len, feature_dim = sequence.shape

        deltas = []
        for scale in [1, 2, 3, 5]:
            if seq_len > scale:
                delta = torch.abs(sequence[:, scale:] - sequence[:, :-scale]) / scale
                delta = F.pad(delta, (0, 0, 0, scale), "constant", 0)
                deltas.append(delta)
            else:
                deltas.append(torch.zeros_like(sequence))

        weights = [0.4, 0.3, 0.2, 0.1] 
        combined_delta = sum(w * d for w, d in zip(weights, deltas))

        delta_mean = combined_delta.mean(dim=-1)

        keypoints = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=sequence.device)
        
        for b in range(batch_size):
            threshold = delta_mean[b].mean() + self.threshold * delta_mean[b].std()

            for i in range(1, seq_len-1):
                if (delta_mean[b, i] > delta_mean[b, i-1] and 
                    delta_mean[b, i] > delta_mean[b, i+1] and
                    delta_mean[b, i] > threshold):
                    keypoints[b, i] = True

            if delta_mean[b, 0] > threshold:
                keypoints[b, 0] = True
            if delta_mean[b, -1] > threshold:
                keypoints[b, -1] = True

        keypoint_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=sequence.device)
        
        for b in range(batch_size):
            key_indices = torch.where(keypoints[b])[0]
            
            if len(key_indices) > 0:
                keypoint_mask[b, :, key_indices] = True
                keypoint_mask[b, key_indices, :] = True
            else:
                sampled_indices = torch.linspace(0, seq_len-1, min(5, seq_len)).long()
                keypoint_mask[b, :, sampled_indices] = True
        
        return keypoint_mask

    def _get_global_random_indices(self, input_tensor, num_layers=4, samples_per_layer=4):
        batch_size, seq_len, feature_dim = input_tensor.shape

        global_indices = torch.zeros((batch_size, min(seq_len, num_layers * samples_per_layer)), 
                                    dtype=torch.long, device=input_tensor.device)
        
        for b in range(batch_size):
            seq_features = input_tensor[b]
            magnitude = torch.norm(seq_features, dim=-1)
            if seq_len > 1:
                changes = torch.abs(seq_features[1:] - seq_features[:-1]).mean(dim=-1)
                changes = F.pad(changes, (0, 1), "constant", 0)
            else:
                changes = torch.zeros_like(magnitude)
            if seq_len >= 4:
                diffs = seq_features[1:] - seq_features[:-1]
                if diffs.size(0) > 1:
                    freq = torch.var(diffs, dim=0).mean()
                    freq_score = torch.zeros_like(magnitude)
                    for i in range(1, seq_len-1):
                        local_var = torch.var(seq_features[max(0, i-2):min(seq_len, i+3)], dim=0).mean()
                        freq_score[i] = local_var / (freq + 1e-6)
                else:
                    freq_score = torch.zeros_like(magnitude)
            else:
                freq_score = torch.zeros_like(magnitude)
            importance = magnitude * 0.3 + changes * 0.4 + freq_score * 0.3
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-6)
            
            if seq_len <= num_layers * samples_per_layer:
                indices = torch.arange(seq_len, device=input_tensor.device)
            else:
                indices = []
                segment_size = seq_len // num_layers
                
                for l in range(num_layers):
                    start_idx = l * segment_size
                    end_idx = min(seq_len, (l + 1) * segment_size)

                    segment_importance = importance[start_idx:end_idx]

                    if len(segment_importance) <= samples_per_layer:
                        segment_indices = torch.arange(start_idx, end_idx, device=input_tensor.device)
                    else:
                        probs = segment_importance / segment_importance.sum()
                        segment_indices = torch.multinomial(probs, samples_per_layer, replacement=False)
                        segment_indices += start_idx
                    
                    indices.append(segment_indices)
                
                indices = torch.cat(indices)
            global_indices[b, :len(indices)] = indices
        
        return global_indices

    def _get_sparse_indices(self, seq_len_q, seq_len_kv, queries, sparsity_ratio=0.3):
        batch_size = queries.size(0)

        range_q = torch.arange(seq_len_q, device=queries.device).view(1, seq_len_q, 1)
        range_kv = torch.arange(seq_len_kv, device=queries.device).view(1, 1, seq_len_kv)

        local_window, future_window = self._calculate_dynamic_windows(queries)

        sparse_indices = torch.zeros((batch_size, seq_len_q, seq_len_kv), dtype=torch.bool, device=queries.device)

        for b in range(batch_size):
            for q in range(seq_len_q):
                local_size = local_window[b, q]
                start_idx = max(0, q - local_size)
                end_idx = min(seq_len_kv, q + local_size + 1)
                sparse_indices[b, q, start_idx:end_idx] = True
            
            for q in range(seq_len_q):
                if q < seq_len_kv: 
                    future_size = future_window[b, q]
                    end_idx = min(seq_len_kv, q + future_size + 1)
                    sparse_indices[b, q, q:end_idx] = True
        
        keypoint_mask = self._detect_keypoints(queries)
        sparse_indices = sparse_indices | keypoint_mask

        global_indices = self._get_global_random_indices(queries)
        
        for b in range(batch_size):
            valid_indices = global_indices[b][global_indices[b] < seq_len_kv]
            if len(valid_indices) > 0:
                sparse_indices[b, :, valid_indices] = True
        complexity = queries.var(dim=-1).mean(dim=1)
        normalized_complexity = (complexity - complexity.min()) / (complexity.max() - complexity.min() + 1e-6)
        
        for b in range(batch_size):
            adaptive_ratio = sparsity_ratio * (0.5 + normalized_complexity[b])
            adaptive_ratio = min(0.5, max(0.1, adaptive_ratio))
            current_density = sparse_indices[b].float().mean().item()
            
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
        
        for b in range(batch_size):
            for q in range(seq_len_q):
                if not sparse_indices[b, q].any():
                    if q < seq_len_kv:
                        sparse_indices[b, q, q] = True
                    else:
                        sparse_indices[b, q, seq_len_kv-1] = True
        
        return sparse_indices

    def forward(self, queries, keys, values, attn_mask=None):
        batch_size, seq_len_q, _ = queries.size()
        seq_len_kv = keys.size(1)

        queries = self.norm_q(queries)
        keys = self.norm_k(keys)
        values = self.norm_v(values)

        Q = self.q_proj(queries).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(keys).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(values).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

        sparse_indices = self._get_sparse_indices(seq_len_q, seq_len_kv, queries)
        sparse_indices = sparse_indices.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        scale = 1.0 / math.sqrt(self.head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        attention_mask = ~sparse_indices

        if self.mask:
            causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_kv, device=queries.device), diagonal=1).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
            scores.masked_fill_(causal_mask, -1e9)

        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.key_dim)
        gate_values = self.gate(queries)
        output = output * gate_values
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
    
