import torch
import torch.nn as nn
from torch_dct import dct, idct 


import torch
import torch.nn as nn
from torch_dct import dct, idct

class AdaptiveFreqDenoiseBlock(nn.Module):
    def __init__(self, seq_len, dim, max_scale_levels=5, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.max_scale_levels = max_scale_levels

        # Learnable Parameters for Multi-scale Filtering
        # Learnable weights for high-pass filtering at different scales.
        self.weights_high = nn.ParameterList([
            nn.Parameter(torch.randn(seq_len) * 0.02) for _ in range(max_scale_levels)
        ])
        # Learnable weights for low-pass filtering at different scales.
        self.weights_low = nn.ParameterList([
            nn.Parameter(torch.randn(seq_len) * 0.02) for _ in range(max_scale_levels)
        ])
        
        # Learnable Parameters for Adaptive Control
        # Learnable logits to determine the importance of each frequency scale via softmax.
        self.scale_importance_logits = nn.Parameter(torch.zeros(max_scale_levels))
        
        # A learnable parameter to balance between the denoised signal and the detail-enhanced signal.
        self.channel_balance = nn.Parameter(torch.tensor([0.5])) 
        
        # Learnable threshold for creating the adaptive mask for denoising.
        self.denoise_threshold = nn.Parameter(torch.tensor([0.3]))
        
        # Learnable threshold for creating the adaptive mask for detail enhancement.
        self.detail_threshold = nn.Parameter(torch.tensor([0.5]))

        # Standard Layers
        self.dropout = nn.Dropout(dropout)
        self.input_projection = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)

    def vectorized_multi_scale_op(self, x, weights, scale_attention_weights):
        """Performs multi-scale filtering in a vectorized way for efficiency."""
        # Stack the list of weight parameters into a single tensor. Shape: (max_scale_levels, seq_len)
        stacked_weights = torch.stack(list(weights), dim=0)

        # Apply sigmoid to the weights to create filter gates for each scale.
        gates = torch.sigmoid(stacked_weights).unsqueeze(1)

        # Apply the gates to the input tensor 'x' for each scale using broadcasting.
        x_per_scale = x.unsqueeze(0) * gates.unsqueeze(1)
        
        # Reshape the scale attention weights for broadcasting.
        reshaped_attention = scale_attention_weights.view(-1, 1, 1, 1)
        
        # Apply the attention weights to the filtered results.
        x_weighted = x_per_scale * reshaped_attention

        # Sum the results across all scales to get the final output.
        x_scaled = x_weighted.sum(dim=0)
        
        return x_scaled

    def create_adaptive_mask(self, x_dct, threshold_param):
        """Dynamically generates a mask based on the energy of the frequency components."""
        # Calculate the energy by squaring the DCT coefficients.
        energy = x_dct.pow(2)
        # Compute the median energy for stable normalization.
        median_energy = torch.quantile(energy + 1e-8, 0.5, dim=2, keepdim=True)
        # Normalize the energy to make the thresholding scale-invariant.
        normalized_energy = energy / (median_energy + 1e-6)
        # Clamp the learnable threshold parameter for stability.
        threshold = torch.clamp(threshold_param, 0.1, 0.9)
        # Create a smooth, differentiable mask using a sigmoid function.
        adaptive_mask = torch.sigmoid(1.0 * (normalized_energy - threshold))
        return adaptive_mask

    def forward(self, x_in):
        # Store the original input for the final residual connection.
        x_residual = x_in
        # Project the input and permute dimensions for frequency domain processing. (B, N, C) -> (B, C, N)
        x = self.input_projection(x_in).permute(0, 2, 1) 

        # Frequency Domain Processing 
        # Calculate the attention weights for each scale.
        scale_attention = torch.softmax(self.scale_importance_logits, dim=0)
        
        # Transform the signal into the frequency domain using Discrete Cosine Transform.
        x_dct = dct(x, norm='ortho')
        
        # Separate the signal into high and low-frequency components using multi-scale filtering.
        x_high = self.vectorized_multi_scale_op(x_dct, self.weights_high, scale_attention)
        x_low = self.vectorized_multi_scale_op(x_dct, self.weights_low, scale_attention)

        # Adaptive High-Frequency Refinement
        # Create a mask to suppress noise in the high-frequency component.
        denoise_mask = self.create_adaptive_mask(x_high, self.denoise_threshold)
        x_high_denoised = x_high * denoise_mask
        
        # Create a mask to enhance important details in the high-frequency component.
        detail_mask = self.create_adaptive_mask(x_high, self.detail_threshold)
        x_high_detailed = x_high * detail_mask
        
        # Create a gate to balance the two processed high-frequency signals.
        balance_gate = torch.sigmoid(self.channel_balance)
        # Combine the denoised and detail-enhanced signals using the learned balance.
        x_high_combined = x_high_denoised * balance_gate + x_high_detailed * (1 - balance_gate)
        
        # Signal Recombination and Output
        # Recombine the processed high-frequency component with the original low-frequency component.
        x_dct_recombined = x_high_combined + x_low
        
        # Transform the signal back to the time domain using inverse DCT.
        x_filtered = idct(x_dct_recombined, norm='ortho').permute(0, 2, 1) # (B, C, N) -> (B, N, C)
        
        # Project the filtered signal back to the original dimension.
        x_denoised = self.output_projection(x_filtered)
        # Add the residual connection and apply dropout for the final output.
        output = x_residual + self.dropout(x_denoised)
        return output
