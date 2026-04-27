"""Temporal Fusion Transformer — faithful re-implementation.

Based on Lim et al. 2021 ("Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting", IJF 37(4)).

Adapted to our experimental setup:
  - 6 input features (Open, High, Low, Close, Volume, scaled_sentiment),
    all observed at every encoder step
  - No static covariates and no future-known covariates
  - Univariate target (Close), single-point prediction (no quantiles)
  - Outer instance normalisation matching the rest of our model suite

The full TFT pipeline (encoder → decoder, with future-known + static signals)
collapses cleanly to this case: the decoder VSN receives zeros (no future-
known inputs), the static enrichment GRN receives no context, and the
output GRN→Linear emits a single value per future step.

All architectural building blocks (GLU, GRN, VSN, IMHA, gated skip
connections, LSTM encoder/decoder, full lower-triangular self-attention)
are present and assembled in the order described in the paper.
"""
import torch
import torch.nn as nn

from layers.TFT_components import (
    GatedLinearUnit,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.get('context_len', 504)
        self.pred_len = configs['pred_len']
        self.n_vars = 6
        self.close_idx = 3

        d_model = configs['d_model']
        n_heads = configs['n_heads']
        d_ff = configs['d_ff']
        dropout = configs['dropout']
        lstm_layers = configs.get('lstm_layers', 2)

        # Per-variable scalar embedding (each scalar feature → d_model)
        # paper §3.3 — for continuous covariates, a per-variable Linear suffices
        self.var_embed = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(self.n_vars)]
        )

        # Variable Selection Networks (separate weights for encoder vs decoder
        # because the variable importance distribution can differ between past
        # observations and future-known inputs — TFT design choice)
        self.vsn_encoder = VariableSelectionNetwork(d_model, self.n_vars, dropout=dropout)
        self.vsn_decoder = VariableSelectionNetwork(d_model, self.n_vars, dropout=dropout)

        # LSTM encoder / decoder. PyTorch applies dropout BETWEEN layers when
        # num_layers > 1; lstm_layers=2 is the paper default.
        self.lstm_encoder = nn.LSTM(
            d_model, d_model, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_decoder = nn.LSTM(
            d_model, d_model, num_layers=lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Gated skip after LSTMs (Eq. 13 of the paper)
        self.lstm_glu = GatedLinearUnit(d_model, d_model, dropout=dropout)
        self.lstm_norm = nn.LayerNorm(d_model)

        # Static enrichment (we have no static covariates → unconditioned GRN)
        self.static_enrichment = GatedResidualNetwork(
            d_model, d_model, d_model, dropout=dropout
        )

        # Self-attention with shared values across heads (interpretable MHA)
        self.attention = InterpretableMultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.attn_glu = GatedLinearUnit(d_model, d_model, dropout=dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        # Position-wise feed-forward via a GRN (Eq. 18)
        self.position_wise_ffn = GatedResidualNetwork(
            d_model, d_ff, d_model, dropout=dropout
        )

        # Final gated skip — connects the FFN output back to the LSTM output
        # (this skip preserves the local-pattern signal from LSTM if attention
        # turns out to be unhelpful for some samples)
        self.out_glu = GatedLinearUnit(d_model, d_model, dropout=dropout)
        self.out_norm = nn.LayerNorm(d_model)

        # Output: univariate Close prediction
        self.output_proj = nn.Linear(d_model, 1)

    def _causal_mask(self, T: int, device) -> torch.Tensor:
        """Lower-triangular mask: position i can attend to positions [0, i].
        Returns shape [1, 1, T, T] with 1 = allowed, 0 = masked, ready to
        broadcast across (B, n_heads, T_q, T_k).
        """
        mask = torch.tril(torch.ones(T, T, dtype=torch.uint8, device=device))
        return mask.view(1, 1, T, T)

    def forward(self, x_enc, x_mark_enc=None):
        # ── Outer instance normalisation (RevIN-lite, matches the rest of our
        # model suite — see EXPERIMENT_DESIGN.md) ──────────────────────────────
        means = x_enc.mean(1, keepdim=True).detach()
        x_n = x_enc - means
        stdev = torch.sqrt(torch.var(x_n, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_n = x_n / stdev
        B = x_n.shape[0]

        # ── Per-variable embedding for the encoder ────────────────────────────
        # x_n: [B, seq_len, n_vars] → [B, seq_len, n_vars, d_model]
        enc_var_emb = torch.stack(
            [self.var_embed[i](x_n[:, :, i:i + 1]) for i in range(self.n_vars)],
            dim=2,
        )

        # Decoder receives zero embeddings (no future-known inputs in our setup)
        dec_var_emb = torch.zeros(
            B, self.pred_len, self.n_vars, enc_var_emb.shape[-1], device=x_enc.device
        )

        # ── Variable Selection ────────────────────────────────────────────────
        enc_features, _ = self.vsn_encoder(enc_var_emb)   # [B, seq_len,  d_model]
        dec_features, _ = self.vsn_decoder(dec_var_emb)   # [B, pred_len, d_model]

        # ── Local processing via LSTM (encoder + decoder) ─────────────────────
        enc_lstm, (h, c) = self.lstm_encoder(enc_features)
        dec_lstm, _      = self.lstm_decoder(dec_features, (h, c))
        lstm_out         = torch.cat([enc_lstm, dec_lstm], dim=1)        # [B, T, d_model]
        in_features      = torch.cat([enc_features, dec_features], dim=1) # [B, T, d_model]

        # Gated skip after LSTMs
        x = self.lstm_norm(in_features + self.lstm_glu(lstm_out))

        # Static enrichment (no static context, so this is just a GRN)
        x = self.static_enrichment(x)

        # ── Interpretable self-attention over the full encoder+decoder span ──
        T = x.shape[1]
        mask = self._causal_mask(T, x.device)
        attn_out, _ = self.attention(x, x, x, mask=mask)
        x = self.attn_norm(x + self.attn_glu(attn_out))

        # ── Position-wise feed-forward ───────────────────────────────────────
        ffn_out = self.position_wise_ffn(x)

        # Final gated skip onto the LSTM output (paper Fig. 2 — preserves
        # local-pattern signal)
        x = self.out_norm(lstm_out + self.out_glu(ffn_out))

        # ── Output: univariate Close prediction at decoder positions ─────────
        x_dec = x[:, -self.pred_len:, :]                # [B, pred_len, d_model]
        out = self.output_proj(x_dec).squeeze(-1)       # [B, pred_len]

        # Denormalise using Close-feature stats
        close_stdev = stdev[:, 0, self.close_idx]       # [B]
        close_mean  = means[:, 0, self.close_idx]       # [B]
        out = out * close_stdev.unsqueeze(-1) + close_mean.unsqueeze(-1)
        return out
