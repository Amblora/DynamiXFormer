import torch
import torch.nn as nn
from torch_dct import dct, idct 


class AdaptiveFreqDenoiseBlock(nn.Module):
    """
    Adaptive Frequency Denoise Block

    The core idea of this module is to process time-series data in the frequency domain,
    achieved via the Discrete Cosine Transform (DCT). It aims to accomplish several goals:
    
    1.  Frequency Separation: Decomposes the signal into low-frequency components (which typically represent the main trend and structure) and high-frequency components (which often contain details and noise).
    2.  Adaptive Filtering: Instead of using a fixed filter, it dynamically generates a "soft mask" based on the energy of each frequency component. This allows it to
        preserve important signal details while suppressing noise.
    3.  Multi-task Handling: The module is specifically designed to handle multiple downstream tasks (e.g., regression and classification) that might require different features. 
        It generates distinct filtered high-frequency results for two task-specific paths and merges them based on a learnable parameter.
    4.  Multi-scale Analysis: Through learnable weights, the model can adaptively decide which frequency ranges should be considered "high-frequency" or "low-frequency," rather than using a hard-coded cutoff frequency.
    """
    def __init__(self, dim, scale_levels=3, use_noise_reduction=True):
        """
        Initializer.

        param dim: The input feature dimension.
        param scale_levels: The number of levels for multi-scale analysis.
        param use_noise_reduction: Whether to use an additional convolutional layer for noise reduction in the regression task path.
        """
        super().__init__()
        self.dim = dim
        self.scale_levels = scale_levels
        self.use_noise_reduction = use_noise_reduction
        
        # Multi-scale weights for generating the high-pass filter.
        self.weights_high = nn.ParameterList([
            nn.Parameter(torch.randn(dim) * 0.02) for _ in range(scale_levels)
        ])
        # Multi-scale weights for generating the low-pass filter.
        self.weights_low = nn.ParameterList([
            nn.Parameter(torch.randn(dim) * 0.02) for _ in range(scale_levels)
        ])
        
        # A learnable parameter to balance the outputs of the two task paths (regression and classification).
        self.task_balance = nn.Parameter(torch.tensor([0.5]))
        
        # Adaptive mask threshold for the regression task path.
        self.reg_threshold = nn.Parameter(torch.tensor([0.3])) 
        # Adaptive mask threshold for the classification task path.
        self.cls_threshold = nn.Parameter(torch.tensor([0.5]))  

        if self.use_noise_reduction:
            # An optional noise reduction module for the regression path, using depthwise
            # separable convolutions to further smooth the signal.
            self.noise_reduction = nn.Sequential(
                nn.Conv1d(dim, dim*2, kernel_size=5, padding=2, groups=dim), # Depthwise Conv
                nn.GELU(),
                nn.Conv1d(dim*2, dim, kernel_size=3, padding=1, groups=dim)  # Depthwise Conv
            )
        else:
            self.noise_reduction = None
            
        # A module for the classification path, designed to preserve or sharpen
        # high-frequency features that are important for the classification task
        self.high_freq_preserve = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim), # Depthwise Conv
            nn.GELU()
        )

    def create_adaptive_mask(self, x_dct, threshold_param):
        """
        Creates an adaptive soft mask based on the energy of frequency components.
        
        param x_dct: Frequency coefficients after DCT (B, N, C).
        param threshold_param: A learnable parameter for generating the threshold.
        return: A soft mask (values between 0 and 1) with the same shape as x_dct.
        """
        # Calculate the energy (squared magnitude) of each frequency component.
        energy = x_dct.pow(2).sum(dim=-1, keepdim=True)
        
        # Use quantile (median) as a robust baseline for energy, as it's less sensitive to outliers than the mean.
        median_energy = torch.quantile(energy + 1e-8, 0.5, dim=1, keepdim=True)
        # Normalize the energy with respect to the median.
        normalized_energy = energy / (median_energy + 1e-6)
        
        # Clamp the learnable threshold parameter to a reasonable range.
        threshold = torch.clamp(threshold_param, 0.1, 0.9)
        # Generate a soft mask using a sigmoid function. Components with energy above the threshold
        # will have mask values close to 1, and vice versa. The multiplier (10) sharpens the transition.
        adaptive_mask = torch.sigmoid(10 * (normalized_energy - threshold))
        return adaptive_mask

    def multi_scale_dct(self, x, weights):
        """
        Performs a multi-scale weighted sum on the DCT coefficients to form a "soft" band-pass filter.
        
        param x: DCT coefficients (B, N, C).
        param weights: A ParameterList containing learnable weights for each scale.
        return: The weighted DCT coefficients.
        """
        x_scale = 0
        for i in range(self.scale_levels):
            # Map the weights to the (0,1) range using sigmoid to act as filter weights for the scale.
            weight = torch.sigmoid(weights[i])  
            # The weights are applied channel-wise.
            x_scale += x * weight.view(1, 1, -1) 
        return x_scale

    def forward(self, x_in, adaptive_filter=True):
        B, N, C = x_in.shape
        
        # Time to Frequency Domain: Transform the input signal to the frequency domain using DCT.
        # The output shape of dct is the same as the input (B, N, C).
        x_dct = dct(x_in, norm='ortho')
        
        # Frequency Separation: Separate DCT coefficients into high and low frequency components
        # using learnable multi-scale weights. Note: 'high' and 'low' are learned, not based on a fixed cutoff.
        x_high = self.multi_scale_dct(x_dct, self.weights_high)
        x_low = self.multi_scale_dct(x_dct, self.weights_low)
        
        # Adaptive High-Frequency Processing (if enabled).
        if adaptive_filter:
            # Create an adaptive mask for the regression task, typically using a lower threshold to preserve more structural information.
            reg_mask = self.create_adaptive_mask(x_high, self.reg_threshold)
            # Apply the mask to filter out low-energy high-frequency components (likely noise).
            x_high_reg = x_high * reg_mask 
            
            if self.use_noise_reduction:
                # Optional depthwise convolutional layers for further smoothing and denoising.
                x_high_reg = self.noise_reduction(x_high_reg.transpose(1, 2)).transpose(1, 2)
            
            # Create an adaptive mask for the classification task, typically using a higher threshold to focus on the most salient features.
            cls_mask = self.create_adaptive_mask(x_high, self.cls_threshold)
            # Apply the mask.
            x_high_cls = x_high * cls_mask 
            
            # Process with depthwise convolution, potentially to sharpen or preserve sharp features important for classification.
            x_high_cls = self.high_freq_preserve(x_high_cls.transpose(1, 2)).transpose(1, 2)
            
            # Task Fusion: Fuse the processed high-frequency components from the two task paths using a learnable balance parameter.
            balance = torch.sigmoid(self.task_balance)
            x_high = x_high_reg * balance + x_high_cls * (1 - balance)
        
        # Frequency Recombination: Recombine the processed high-frequency components with the original low-frequency components.
        x_combined = x_high + x_low
        
        # Frequency to Time Domain: Transform the signal back to the time domain using Inverse DCT (IDCT).
        x_recon = idct(x_combined, norm='ortho')

        # Numerical Stability Check: Prevent NaN values in the output.
        if torch.isnan(x_recon).any():
            x_recon = torch.nan_to_num(x_recon)
            
        return x_recon



class fourier_decomp(nn.Module):
    def __init__(self, frequency_threshold):
        super(fourier_decomp, self).__init__()
        self.frequency_threshold = frequency_threshold

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        fft_result = torch.fft.fft(x, dim=1) 
        fft_freqs = torch.fft.fftfreq(seq_len, d=1.0).to(x.device)  
        low_freq_mask = torch.abs(fft_freqs) <= self.frequency_threshold

        low_freq_fft = fft_result.clone()
        low_freq_fft[:, ~low_freq_mask, :] = 0 
        trend = torch.fft.ifft(low_freq_fft, dim=1).real  


        high_freq_fft = fft_result.clone()
        high_freq_fft[:, low_freq_mask, :] = 0  
        res = torch.fft.ifft(high_freq_fft, dim=1).real 

        trend = torch.nan_to_num(trend)
        res = torch.nan_to_num(res)

        return res, trend

class my_Layernorm(nn.Module):
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias
