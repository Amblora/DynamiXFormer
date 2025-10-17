import torch
import torch.nn as nn
import torch.nn.functional as F
from Layers import *
from embed import *
from encoder import *
from decoder import *
from attention import *

class DynamiXFormer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 d_model, n_heads, e_layers, d_layers, d_ff,
                 dropout, activation, 
                 output_attention=False, encoder_apdc=True, decoder_apdc=True,use_event_embeding_enc=True,use_event_embeding_dec=True,
                 device=torch.device('cpu')):
        
        super(AyakaFormer, self).__init__()
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.device = device
        self.label_len = label_len
        self.seq_len = seq_len
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout=dropout, use_event_embeding=use_event_embeding_enc)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout=dropout, use_event_embeding=use_event_embeding_dec)

        self.encoder = Encoder(
            [EncoderLayer(
                DynamicSparseAttention(key_dim=d_model, num_heads=n_heads, mask=False, local_window=pred_len),
                d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation,
                use_apdc=encoder_apdc, seq_len=self.seq_len) for _ in range(e_layers)], 
                norm_layer=my_Layernorm(d_model),
                )

        self.decoder = Decoder(
            [DecoderLayer(
                DynamicSparseAttention(key_dim=d_model, num_heads=n_heads, mask=True,local_window=pred_len),
                AttentionLayer(FullAttention(False, attention_dropout=dropout, output_attention=False),d_model, n_heads),
                d_model=d_model,d_ff=d_ff,dropout=dropout,activation=activation, c_out=c_out,
                use_apdc=decoder_apdc, label_len=self.label_len, pred_len=self.pred_len) for _ in range(d_layers)], 
                norm_layer=my_Layernorm(d_model),
                projection = nn.Linear(d_model, c_out, bias=True),
                
            )
        

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        enc_embed_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_embed_out, attn_mask=enc_self_mask)
        dec_input_context = x_enc[:, -self.label_len:, :]
        batch_size = x_enc.shape[0]
        num_features = x_enc.shape[-1]
        dec_input_placeholder = torch.zeros(
            batch_size, self.pred_len, num_features, device=x_enc.device).float()
        dec_input = torch.cat([dec_input_context, dec_input_placeholder], dim=1)
        dec_embed_out = self.dec_embedding(dec_input)
        pred_out = self.decoder(dec_embed_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        if self.output_attention:
            return pred_out[:, -self.pred_len:, :], attns
        else:
            return pred_out[:, -self.pred_len:, :]
        
