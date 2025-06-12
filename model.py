import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import *
from embed import *
from encoder import *
from decoder import *
from attention import *

class DynamiXFormer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 d_model, n_heads, e_layers, d_layers, d_ff,
                 dropout, activation, series_decomp,
                 output_attention=False, encoder_apdc=True, decoder_apdc=True,use_event_embeding_enc=True,use_event_embeding_dec=True,
                 device=torch.device('cpu')):
        
        super(DynamiXFormer, self).__init__()

        self.pred_len = pred_len
        self.output_attention = output_attention
        self.device = device
        self.label_len = label_len

        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout=dropout, use_event_embeding=use_event_embeding_enc)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout=dropout, use_event_embeding=use_event_embeding_dec)
        self.decomp = fourier_decomp(series_decomp)

        self.encoder = Encoder(
            [EncoderLayer(
                DynamicSparseAttention(key_dim=d_model, num_heads=n_heads, mask=False, local_window=pred_len, future_window=seq_len),
                series_decomp = series_decomp,
                d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation,
                use_apdc=encoder_apdc) for _ in range(e_layers)], 
                norm_layer=my_Layernorm(d_model)
                )

        self.decoder = Decoder(
            [DecoderLayer(
                DynamicSparseAttention(key_dim=d_model, num_heads=n_heads, mask=True,local_window=pred_len, future_window=seq_len),
                AttentionLayer(FullAttention(False, attention_dropout=dropout, output_attention=False),d_model, n_heads),
                series_decomp = series_decomp,
                d_model=d_model,d_ff=d_ff,dropout=dropout,activation=activation, c_out=c_out,
                use_apdc=decoder_apdc) for _ in range(d_layers)], 
                norm_layer=my_Layernorm(d_model),
                projection = nn.Linear(d_model, c_out, bias=True)
            )
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        mean = torch.mean(x_enc[:, :, -1:], dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc[:, :, -1:])
        
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        
        enc_enmbed_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_enmbed_out, attn_mask=enc_self_mask)

        dec_embed_out = self.dec_embedding(seasonal_init)
        seasonal_part, trend_part = self.decoder(dec_embed_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        pred_out = seasonal_part + trend_part
        
        if self.output_attention: return pred_out[:, -self.pred_len:, ], attns
        else: return pred_out[:, -self.pred_len:, ]
