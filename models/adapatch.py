import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        c_in = 6
        self.seq_len = 252
        self.pred_len = configs['pred_len']
        self.patch_len = configs['slice_len']
        self.middle_dim = configs['middle_len']
        self.hidden_dim = configs['hidden_len']
        self.patch_stride = configs['slice_stride']
        self.encoder_dropout = configs['encoder_dropout']

        self.projection = nn.Linear(c_in, 1)

        self.encoder = nn.Sequential(
            nn.Linear(self.patch_len, self.middle_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.encoder_dropout),
            nn.Linear(self.middle_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.middle_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.encoder_dropout),
            nn.Linear(self.middle_dim, self.patch_len),
        )

        self.num_patches = self.seq_len // self.patch_len
        self.num_pred_patches = self.pred_len // self.patch_len
        self.short_horizon = self.num_pred_patches == 0  # pred_len < patch_len

        if self.short_horizon:
            # Direct projection to pred_len when pred_len < patch_len
            self.fc_predictor = nn.Sequential(
                nn.Linear(self.hidden_dim * self.num_patches, configs['d_ff']),
                nn.LeakyReLU(),
                nn.Dropout(self.encoder_dropout),
                nn.Linear(configs['d_ff'], self.pred_len),
            )
        else:
            self.fc_predictor = nn.Sequential(
                nn.Linear(self.hidden_dim * self.num_patches, configs['d_ff']),
                nn.LeakyReLU(),
                nn.Dropout(self.encoder_dropout),
                nn.Linear(configs['d_ff'], self.hidden_dim * self.num_pred_patches),
            )

    def forward(self, x_in, x_mark_enc=None):
        seq_last = x_in[:, -1:, :].detach()
        x = x_in - seq_last

        x = x.permute(0, 2, 1).contiguous()
        B, C, L = x.shape

        patches_rec = x.unfold(-1, self.patch_len, self.patch_stride)
        B_rec, C_rec, N_p_stride, P_L_rec = patches_rec.shape
        slice_orig_flat = patches_rec.reshape(B_rec, C_rec, -1)
        encoded_rec_vec = self.encoder(patches_rec)
        decoded_rec_vec = self.decoder(encoded_rec_vec)
        decoded_slice_flat = decoded_rec_vec.reshape(B_rec, C_rec, -1)

        L_trunc = self.num_patches * self.patch_len
        patches_pred_in = x[:, :, :L_trunc].reshape(B, C, self.num_patches, self.patch_len)

        encoded_pred_vec = self.encoder(patches_pred_in)
        encoded_pred_flat = encoded_pred_vec.reshape(B, C, -1)
        prediction_latent_flat = self.fc_predictor(encoded_pred_flat)

        if self.short_horizon:
            # [B, C, pred_len] -> [B, pred_len, C]
            y_pred = prediction_latent_flat.permute(0, 2, 1)
        else:
            prediction_latent_vec = prediction_latent_flat.reshape(
                B, C, self.num_pred_patches, self.hidden_dim
            )
            prediction_patches_vec = self.decoder(prediction_latent_vec)
            prediction_flat = prediction_patches_vec.reshape(B, C, -1)
            y_pred = prediction_flat.permute(0, 2, 1)

        y_pred = y_pred + seq_last
        y_pred = y_pred[:, :, 3]  # Extract Close prediction
        return y_pred, slice_orig_flat, decoded_slice_flat
