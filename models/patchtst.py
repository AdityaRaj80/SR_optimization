import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs['context_len'] if 'context_len' in configs else 252
        self.pred_len = configs['pred_len']
        
        patch_len = configs['patch_len']
        stride = configs['stride']
        padding = stride
        d_model = configs['d_model']
        enc_in = 6 # features
        
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, padding, configs['dropout'])

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=configs['dropout'], output_attention=False), 
                        d_model, configs['n_heads']
                    ),
                    d_model, configs['d_ff'], dropout=configs['dropout'], activation='gelu'
                ) for l in range(configs['e_layers'])
            ],
            norm_layer=nn.Sequential(nn.BatchNorm1d(d_model))
        )

        self.head_nf = d_model * int((self.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(enc_in, self.head_nf, self.pred_len, head_dropout=configs['head_dropout'])
        self.proj = nn.Linear(enc_in, 1) # to map back to univariate Close predictor if needed

    def forward(self, x_enc, x_mark_enc=None):
        # x_enc: [B, seq_len, n_vars]
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc = x_enc.permute(0, 2, 1) # [B, n_vars, seq_len]
        enc_out, n_vars = self.patch_embedding(x_enc)

        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2) # [bs, n_vars, d_model, patch_num]

        dec_out = self.head(enc_out) # [bs, n_vars, target_window]
        dec_out = dec_out.permute(0, 2, 1) # [bs, target_window, n_vars]

        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        # We only want the Close prediction (which is idx 3 in features: Open, High, Low, Close, Vol, Sent)
        return dec_out[:, :, 3]

