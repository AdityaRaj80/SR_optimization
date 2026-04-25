import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.get('context_len', 504)
        self.pred_len = configs['pred_len']
        d_model = configs['d_model']
        self.quantiles = configs['quantiles']
        
        # simplified TFT to just rely on projection and attention
        # Since standard DataEmbedding from TSLib handles temporal features natively
        from layers.Embed import DataEmbedding
        enc_in = 6
        self.enc_embedding = DataEmbedding(enc_in, d_model, 'timeF', 'h', configs['dropout'])
        self.dec_embedding = DataEmbedding(enc_in, d_model, 'timeF', 'h', configs['dropout'])
        
        # Temporal Fusion layers (simplified with MultiHead Attention)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=configs['dropout'], output_attention=False), 
                        d_model, configs['n_heads']
                    ),
                    d_model, configs['d_ff'], dropout=configs['dropout'], activation='gelu'
                ) for l in range(configs.get('lstm_layers', 2)) # Using multiple layers for "fusion"
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Output quantile projections
        self.quantile_proj = nn.ModuleList([nn.Linear(d_model, enc_in) for _ in self.quantiles])

    def forward(self, x_enc, x_mark_enc=None):
        # We dummy-encode dec_in since our benchmark doesn't natively provide decoder input 
        # (similar to Vanilla Transformer)
        B = x_enc.shape[0]
        x_dec = torch.zeros([B, self.pred_len, x_enc.shape[2]], device=x_enc.device)
        x_dec = torch.cat([x_enc[:, -48:, :], x_dec], dim=1) # label len = 48
        
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, None)
        dec_out_embed = self.dec_embedding(x_dec, None)
        
        # Pass through the encoder
        fus_out, attns = self.encoder(enc_out)
        
        # For a simplified TFT, we just project out the quantile results from the temporal fusion output
        # In a full seq2seq, we use dec_out. Here we just take the last 'pred_len' states of the fused sequence or a projection
        # Let's project fus_out to pred_len directly from the last step or use a projection on sequence
        
        pass_out = self.quantile_proj[1](fus_out) # index 1 is 0.5 (median)
        
        # Since fus_out is [B, 252, enc_in], we take the last pred_len steps... 
        # wait, if pred_len > seq_len, we need a spatial projection.
        # Let's add a seq_len -> pred_len projection
        pass_out = pass_out.permute(0, 2, 1) # [B, C, S]
        if not hasattr(self, 'temporal_proj'):
            self.temporal_proj = nn.Linear(self.seq_len, self.pred_len).to(pass_out.device)
        
        pass_out = self.temporal_proj(pass_out).permute(0, 2, 1)
        
        pass_out = pass_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        pass_out = pass_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return pass_out[:, :, 3] # Extract the Close prediction
