import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs['pred_len']
        self.seq_len = 252
        self.label_len = 48
        
        enc_in = 6
        d_model = configs['d_model']
        
        self.enc_embedding = DataEmbedding(enc_in, d_model, 'timeF', 'h', configs['dropout'])
        self.dec_embedding = DataEmbedding(enc_in, d_model, 'timeF', 'h', configs['dropout'])
        
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(FullAttention(False, configs.get('factor', 1), attention_dropout=configs['dropout'], output_attention=False), d_model, configs['n_heads']),
                d_model, configs['d_ff'], dropout=configs['dropout'], activation=configs['activation']
            ) for l in range(configs['e_layers'])],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        self.decoder = Decoder(
            [DecoderLayer(
                AttentionLayer(FullAttention(True, configs.get('factor', 1), attention_dropout=configs['dropout'], output_attention=False), d_model, configs['n_heads']),
                AttentionLayer(FullAttention(False, configs.get('factor', 1), attention_dropout=configs['dropout'], output_attention=False), d_model, configs['n_heads']),
                d_model, configs['d_ff'], dropout=configs['dropout'], activation=configs['activation'],
            ) for l in range(configs['d_layers'])],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, enc_in, bias=True)
        )

    def forward(self, x_enc, x_mark_enc=None):
        # We need a dummy decoder input for Vanilla Transformer
        B = x_enc.shape[0]
        x_dec = torch.zeros([B, self.pred_len, x_enc.shape[2]], device=x_enc.device)
        x_dec = torch.cat([x_enc[:, -self.label_len:, :], x_dec], dim=1)
        
        # We will just pass None for x_mark_enc and x_mark_dec here since we aren't loading time features
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, None)
        dec_out, _ = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        
        return dec_out[:, -self.pred_len:, 3] # Close price
