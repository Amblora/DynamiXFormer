import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, seq_len, d_ff=None, dropout=0.1, activation="relu", use_apdc=True):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu # 推荐使用GELU

        self.use_apdc = use_apdc
        if self.use_apdc:
            self.apdc = AdaptiveFreqDenoiseBlock(dim=d_model, seq_len=seq_len)
            self.norm_apdc = nn.LayerNorm(d_model) 

        self.norm1 = nn.LayerNorm(d_model)  
        self.norm2 = nn.LayerNorm(d_model)  

    def forward(self, x, attn_mask=None):
        if self.use_apdc:
            x = self.norm_apdc(x + self.apdc(x))

        attn_output, attn = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attn_mask)
        x = x + self.dropout(attn_output)

        y = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = x + y
        return x, attn 
    

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer  
 
    def forward(self, x, attn_mask=None):

        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
