import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers import *

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, c_out, label_len, pred_len, d_ff=None,
                 dropout=0.1, activation="relu", use_apdc=True):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

        self.use_apdc = use_apdc
        if self.use_apdc:
            self.apdc = AdaptiveFreqDenoiseBlock(dim=d_model, seq_len=label_len + pred_len)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=x_mask)[0])
        x = x + self.dropout(self.cross_attention(self.norm2(x), cross, cross, attn_mask=cross_mask)[0])
        y = self.norm3(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = x + y
        
        if self.use_apdc:
            x = self.apdc(x)
            
        return x

 
class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection 

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            
        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
            
        return x
