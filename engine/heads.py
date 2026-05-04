"""Risk-aware output heads for Track B retraining.

`RiskAwareHead` wraps an existing backbone model that emits a future-Close
prediction tensor of shape [B, pred_len], and adds:
  * `sigma_head` — predicts log-variance of the H-step ahead RETURN
  * `vol_head`   — predicts the forward (next-window) log realized volatility

The auxiliary heads are small MLPs that operate on the last `lookback_for_aux`
rows of the input window (independent of backbone weights), so any existing
model in `models/__init__.py:model_dict` can be wrapped without modifying its
internals. The wrapper exposes the standard `forward(x_enc, x_mark_enc=None)`
signature used elsewhere in the codebase, but returns a *dict* of named
outputs rather than a single tensor — `engine/trainer.py` recognises this
and dispatches to `CompositeRiskLoss` when the dict is present.

Backward-compat: if a calling site passes the wrapper's dict to a code
path that expects a single tensor, it can simply do `out['mu_close']`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import CLOSE_IDX


class RiskAwareHead(nn.Module):
    """Backbone wrapper that adds heteroscedastic σ + forward-vol heads.

    Args:
        backbone: any nn.Module producing a price prediction tensor of
            shape [B, pred_len] (or a tuple whose first element is that
            tensor — handled identically to existing trainer.py logic).
        n_features: number of input feature channels (FNSPID = 6).
        pred_len: forecast horizon H.
        close_idx: index of the Close column in the input feature order.
        lookback_for_aux: number of trailing input rows the σ/vol heads
            see (independent of backbone's own context_len).
        d_hidden: hidden width of the auxiliary MLPs.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_features: int,
        pred_len: int,
        close_idx: int = CLOSE_IDX,
        lookback_for_aux: int = 20,
        d_hidden: int = 64,
    ):
        super().__init__()
        self.backbone = backbone
        self.pred_len = int(pred_len)
        self.close_idx = int(close_idx)
        self.lookback = int(lookback_for_aux)

        in_dim = self.lookback * int(n_features)

        # Sigma head — outputs log_sigma2 of the H-step return (one scalar / sample).
        self.sigma_head = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

        # Vol head — outputs predicted forward log realized vol (one scalar / sample).
        self.vol_head = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None):
        """Forward pass.

        Args:
            x_enc: [B, seq_len, n_features] input window.
            x_mark_enc: time markers (unused here; passed through to backbone).

        Returns:
            dict with keys:
                mu_close       [B, pred_len]  — full price prediction (back-compat)
                mu_close_H     [B]            — H-step ahead price prediction
                mu_return_H    [B]            — H-step ahead return (derived)
                log_sigma2_H   [B]            — log-variance of H-step return
                log_vol_pred   [B]            — forward log realized vol
                last_close     [B]            — last observed close (for inverse scaling)
        """
        backbone_out = self.backbone(x_enc, x_mark_enc)
        if isinstance(backbone_out, tuple):
            mu_close = backbone_out[0]
        else:
            mu_close = backbone_out

        if mu_close.ndim == 1:
            # Edge case: some legacy backbones output [B] for pred_len=1.
            mu_close = mu_close.unsqueeze(1)
        # Final-step prediction
        mu_close_H = mu_close[:, -1]                                     # [B]

        last_close = x_enc[:, -1, self.close_idx]                        # [B]

        # H-step predicted return.
        # NOTE: features are MinMax-scaled per stock to [0, 1] before training,
        # so `last_close` lives in scaled space, NOT in dollars. For samples
        # near a stock's historical minimum the scaled close approaches 0 and
        # `(mu - last) / |last|` would explode. We floor |last_close| at
        # `_RETURN_DENOM_MIN` (1% of MinMax range) so the scaled-space return is
        # numerically bounded. This is a structural approximation: the
        # downstream Sharpe / NLL / position-sizing terms only need a *signed,
        # scale-stable* return signal, which this provides. Semantic
        # dollar-space returns require threading close_min/close_max through
        # the loss path — left for a follow-up if the floor proves too coarse.
        _RETURN_DENOM_MIN = 1e-2
        denom = last_close.abs().clamp(min=_RETURN_DENOM_MIN) + 1e-9
        mu_return_H = (mu_close_H - last_close) / denom                  # [B]

        # Auxiliary heads on last `lookback` rows
        last_window = x_enc[:, -self.lookback:, :]                       # [B, L, F]
        flat = last_window.reshape(last_window.shape[0], -1)             # [B, L*F]
        log_sigma2_H = self.sigma_head(flat).squeeze(-1)                 # [B]
        log_vol_pred = self.vol_head(flat).squeeze(-1)                   # [B]

        return {
            "mu_close": mu_close,
            "mu_close_H": mu_close_H,
            "mu_return_H": mu_return_H,
            "log_sigma2_H": log_sigma2_H,
            "log_vol_pred": log_vol_pred,
            "last_close": last_close,
        }
