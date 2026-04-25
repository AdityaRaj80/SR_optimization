import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.get('context_len', 504)
        self.pred_len = configs['pred_len']
        d_model = configs['d_model']
        enc_in = 6
        
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, d_model, 'timeF', 'h', configs['dropout'])
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=configs['dropout'], output_attention=False), 
                        d_model, configs['n_heads']
                    ),
                    d_model, configs['d_ff'], dropout=configs['dropout'], activation=configs['activation']
                ) for l in range(configs['e_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        self.projection = nn.Linear(d_model, self.pred_len, bias=True)

    def forward(self, x_enc, x_mark_enc=None):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out[:, :, 3] # Extract the Close prediction
