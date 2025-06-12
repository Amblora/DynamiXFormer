import torch
import torch.nn as nn
from torch_dct import dct, idct 

class AdaptiveFreqDenoiseBlock(nn.Module):
    def __init__(self, dim, scale_levels=3, use_noise_reduction=True):
        super().__init__()
        self.dim = dim
        self.scale_levels = scale_levels
        self.use_noise_reduction = use_noise_reduction
        
        self.weights_high = nn.ParameterList([
            nn.Parameter(torch.randn(dim) * 0.02) for _ in range(scale_levels)
        ])
        self.weights_low = nn.ParameterList([
            nn.Parameter(torch.randn(dim) * 0.02) for _ in range(scale_levels)
        ])
        
        self.task_balance = nn.Parameter(torch.tensor([0.5]))
        
        self.reg_threshold = nn.Parameter(torch.tensor([0.3])) 
        self.cls_threshold = nn.Parameter(torch.tensor([0.5]))  # 分类通道保留更多高频特征
        
        if self.use_noise_reduction:
            self.noise_reduction = nn.Sequential(
                nn.Conv1d(dim, dim*2, kernel_size=5, padding=2, groups=dim),
                nn.GELU(),
                nn.Conv1d(dim*2, dim, kernel_size=3, padding=1, groups=dim)
            )
        else:
            self.noise_reduction = None
            
        self.high_freq_preserve = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU()
        )

    def create_adaptive_mask(self, x_dct, threshold_param):
        energy = x_dct.pow(2).sum(dim=-1, keepdim=True)
        
        median_energy = torch.quantile(energy + 1e-8, 0.5, dim=1, keepdim=True)
        normalized_energy = energy / (median_energy + 1e-6)
        
        threshold = torch.clamp(threshold_param, 0.1, 0.9)
        adaptive_mask = torch.sigmoid(10 * (normalized_energy - threshold))
        return adaptive_mask

    def multi_scale_dct(self, x, weights):
        x_scale = 0
        for i in range(self.scale_levels):
            weight = torch.sigmoid(weights[i])  
            x_scale += x * weight.view(1, 1, -1)
        return x_scale

    def forward(self, x_in, adaptive_filter=True):
        B, N, C = x_in.shape
        
        x_dct = dct(x_in, norm='ortho')
        
        x_high = self.multi_scale_dct(x_dct, self.weights_high)
        x_low = self.multi_scale_dct(x_dct, self.weights_low)
        
        if adaptive_filter:
            reg_mask = self.create_adaptive_mask(x_high, self.reg_threshold)
            x_high_reg = x_high * reg_mask
            
            if self.use_noise_reduction:
                x_high_reg = self.noise_reduction(x_high_reg.transpose(1, 2)).transpose(1, 2)
            
            cls_mask = self.create_adaptive_mask(x_high, self.cls_threshold)
            x_high_cls = x_high * cls_mask
            
            x_high_cls = self.high_freq_preserve(x_high_cls.transpose(1, 2)).transpose(1, 2)
            
            balance = torch.sigmoid(self.task_balance)
            x_high = x_high_reg * balance + x_high_cls * (1 - balance)
        
        x_combined = x_high + x_low
        x_recon = idct(x_combined, norm='ortho')

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