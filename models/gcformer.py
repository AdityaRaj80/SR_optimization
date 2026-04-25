import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp
from layers.SelfAttention_Family import AttentionLayer, ProbAttention
from layers.global_conv import GConv
from layers.RevIN import RevIN
from layers.TCN import TemporalConvNet
from .patchtst import Model as Autoformer_local # We can fallback to our standard PatchTST model as well

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        # We wrap the config dict into a fake Namespace so layers/ components which expect args don't fail
        args = Namespace(**configs)
        setattr(args, "enc_in", 6)
        setattr(args, "seq_len", configs.get('context_len', 504))
        setattr(args, "batch_size", 128)
        
        c_in = 6
        n_layers = configs['e_layers']
        n_heads = configs['n_heads']
        d_model = configs['d_model']
        d_ff = configs['d_ff']
        dropout = configs['dropout']
        fc_dropout = configs['fc_dropout']
        head_dropout = configs['head_dropout']
        individual = configs['individual']
        patch_len = configs['patch_len']
        stride = configs['stride']
        padding_patch = configs['padding_patch']
        revin = configs['revin']
        affine = configs['affine']
        subtract_last = configs['subtract_last']
        decomposition = configs['decomposition']
        
        self.context_window = configs['context_len']
        self.batch_size = 128
        self.enc_in = 6
        self.context_len = configs['context_len']
        self.pred_len = configs['pred_len']
        self.seq_len = configs.get('context_len', 504)
        self.norm_type = configs['norm_type']
        self.atten_bias = configs['atten_bias']
        self.TC_bias = configs['TC_bias']
        self.h_token = configs['h_token']
        self.h_channel = configs['h_channel']

        self.decomposition = decomposition
        
        self.model = PatchTST_backbone(c_in=c_in, context_window=self.context_window, target_window=self.pred_len, patch_len=patch_len, stride=stride, 
                              max_seq_len=1024, n_layers=n_layers, d_model=d_model,
                              n_heads=n_heads, d_k=None, d_v=None, d_ff=d_ff, norm='BatchNorm', attn_dropout=0.,
                              dropout=dropout, act='gelu', key_padding_mask='auto', padding_var=None, 
                              attn_mask=None, res_attention=True, pre_norm=False, store_attn=False,
                              pe='zeros', learn_pe=True, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch=padding_patch,
                              pretrain_head=False, head_type='flatten', individual=individual, revin=revin, affine=affine,
                              subtract_last=subtract_last, verbose=False)

        self.linear_seq_pred = nn.Linear(self.seq_len, self.pred_len, bias=True)
        self.linear_channel_out = nn.Linear(self.h_channel, self.enc_in, bias=True)
        self.linear_channel_in = nn.Linear(self.enc_in, self.h_channel, bias=True)
        self.linear_token_in = nn.Linear(self.pred_len, self.h_token, bias=True)
        self.linear_token_out = nn.Linear(self.h_token, self.pred_len, bias=True)
        self.linear_local_token = nn.Linear(self.context_len, self.pred_len, bias=True)
        self.norm_channel = nn.BatchNorm1d(self.h_channel)
        self.norm_token = nn.BatchNorm1d(self.h_token)
        self.ff = nn.Sequential(nn.GELU(), nn.Dropout(fc_dropout))

        decoder_cross_att = ProbAttention()
        self.decoder_channel = AttentionLayer(decoder_cross_att, self.h_channel, n_heads)
        self.decoder_token = AttentionLayer(decoder_cross_att, self.h_token, n_heads)

        # Full GConv
        self.global_layer_Gconv = GConv(self.batch_size, d_model=self.enc_in, d_state=self.enc_in, l_max=self.seq_len, channels=n_heads, bidirectional=True, kernel_dim=32, n_scales=None, decay_min=2, decay_max=2, transposed=False)

        self.revin_layer = RevIN(self.enc_in, affine=True, subtract_last=False)
        self.local_bias = nn.Parameter(torch.rand(1) * 0.1 + configs['local_bias'])
        self.global_bias = nn.Parameter(torch.rand(1) * 0.1 + configs['global_bias'])

    def forward(self, x_in, x_mark_enc=None): 
        # For GConv, batch size is inferred from forward input usually, but we need to ensure batch dimension matches inside GConv
        # However GConv takes state. Let's just pass global_x through
        seq_last = x_in[:,-1:,:].detach()
        if self.norm_type == 'revin':
            x = self.revin_layer(x_in, 'norm')
        else:
            x = x_in - seq_last
            
        global_x = x
        local_x = x[:,-self.context_len:,:]

        # Encoder: global branch
        # Gconv layer requires x to have batch dimension matching configs.batch_size if not dynamic.
        # But we can override it by letting PyTorch broadcast if possible. Assuming it works:
        try:
            global_x = self.global_layer_Gconv(global_x, return_kernel=False)
        except:
            # GConv might fail if batch size doesn't match initialization due to its complex state initialization
            # As a fallback if the batch size is dynamic on the last batch:
            B = global_x.shape[0]
            if B != self.batch_size:
                tmp_gconv = GConv(B, d_model=self.enc_in, d_state=self.enc_in, l_max=self.seq_len, channels=self.decoder_channel.n_heads, bidirectional=True, kernel_dim=32, n_scales=None, decay_min=2, decay_max=2, transposed=False).to(global_x.device)
                tmp_gconv.load_state_dict(self.global_layer_Gconv.state_dict(), strict=False)
                global_x = tmp_gconv(global_x, return_kernel=False)
                
        global_x = self.linear_seq_pred(global_x.permute(0, 2, 1)).permute(0, 2, 1)

        # Encoder: local branch
        local_x = local_x.permute(0, 2, 1)
        local_x = self.model(local_x)
        local_x = local_x.permute(0, 2, 1)

        # Decoder
        global_x_channel = self.linear_channel_in(global_x)
        local_x_channel = self.linear_channel_in(local_x)
        output_channel_l = self.ff(self.decoder_channel(global_x_channel, local_x_channel, local_x_channel)[0]) + local_x_channel
        output_channel_g = self.ff(self.decoder_channel(local_x_channel, global_x_channel, global_x_channel)[0]) + global_x_channel
        output_channel_l = self.norm_channel(output_channel_l.permute(0, 2, 1)).permute(0, 2, 1)
        output_channel_g = self.norm_channel(output_channel_g.permute(0, 2, 1)).permute(0, 2, 1)
        output_channel = self.atten_bias * output_channel_l + (1 - self.atten_bias) * output_channel_g
        output_channel = self.ff(self.linear_channel_out(output_channel))

        global_x_token = self.linear_token_in(global_x.permute(0, 2, 1))
        local_x_token = self.linear_token_in(local_x.permute(0, 2, 1))
        output_token_l = self.ff(self.decoder_token(global_x_token, local_x_token, local_x_token)[0]) + local_x_token
        output_token_g = self.ff(self.decoder_token(local_x_token, global_x_token, global_x_token)[0]) + global_x_token
        output_token_l = self.norm_token(output_token_l.permute(0, 2, 1)).permute(0, 2, 1)
        output_token_g = self.norm_token(output_token_g.permute(0, 2, 1)).permute(0, 2, 1)
        output_token = self.atten_bias * output_token_l + (1 - self.atten_bias) * output_token_g 
        output_token = self.ff(self.linear_token_out(output_token).permute(0, 2, 1)) 

        output = self.TC_bias * output_channel + (1 - self.TC_bias) * output_token + self.global_bias * global_x + self.local_bias * local_x

        if self.norm_type == 'revin':
            output = self.revin_layer(output, 'denorm')
        else:
            output = output + seq_last
            
        return output[:, :, 3] # Extract the Close prediction
