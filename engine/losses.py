"""Composite Sharpe-aware loss for Track B retraining.

Implements the 5-term composite specified in `reports/design_rethinked.md` §4:

    L = α · L_SR_gated     ← gated differentiable Sharpe (training objective)
      + β · L_NLL          ← heteroscedastic NLL on H-step return (calibrate σ)
      + γ · L_MSE_R        ← return-MSE anchor (prevent flat-only collapse)
      + δ · L_VOL          ← MSE on log realized vol target (calibrate vol head)
      + η · L_GATE_BCE     ← BCE: gate vs. realized profitability (gate-vs-P&L supervision)

Two terms from the design (`λ_to·L_TURN`, `λ_dd·L_DD`) are NOT in v1 because:
  * Turnover requires temporally-ordered batches; our DataLoader shuffles. Adding
    turnover during training would require either disabling shuffle (hurts conv)
    or implementing a per-stock batch sampler. v2 work.
  * Drawdown per shuffled batch is meaningless for the same reason.
Both penalties are correctly applied at *evaluation* time inside
`Smoke_test/cross_sectional_smoke.py`'s portfolio-return cost model — the
training-time approximation is simply not necessary for the headline result
provided the SR_gated term already pushes toward stable predictions.

All operations work in **return space** (mean/std of H-step returns) so the
loss is independent of stock price scale.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Numerical bounds for log-variance to keep exp() finite under autocast bf16/fp16.
_LOG_VAR_MIN = -12.0   # σ² >= exp(-12) ≈ 6e-6   (return-space σ ~ 0.0025)
_LOG_VAR_MAX = 4.0     # σ² <= exp(4)   ≈ 55     (return-space σ ~ 7.4)

# Floor on |last_close| in scaled (MinMax) space to keep the return denominator
# bounded. See engine/heads.py for the full rationale — features are scaled per
# stock to [0, 1], so for samples near a stock's historical minimum the scaled
# close approaches 0 and naive `(close_t+H - close_t) / |close_t|` explodes.
_RETURN_DENOM_MIN = 1e-2


class CompositeRiskLoss(nn.Module):
    """Sharpe-aware composite loss.

    Coefficients are owned by this module and updated each epoch via
    `step_epoch(epoch)`. The default schedule mirrors §4.6 of design_rethinked.md.

    Example
    -------
        criterion = CompositeRiskLoss()
        for epoch in range(epochs):
            criterion.step_epoch(epoch)
            for x, y, vol_target in train_loader:
                output = model(x)            # dict from RiskAwareHead
                loss, parts = criterion(output, y, vol_target)
                loss.backward()
                optimizer.step()

    Args
    ----
        beta:    NLL coefficient (constant)
        delta:   Vol-MSE coefficient (constant)
        eta:     Gate-BCE coefficient (constant)
        alpha_pos:    tanh sharpness for Kelly-style position sizing
        eps_sigma:    floor on σ in position denominator
        eps_sharpe:   floor on Sharpe denominator (return-space)
        gate_temp_decay: per-epoch cooling factor for gate temperature
        gate_temp_min:   minimum gate temperature (final near-binary)
        tau_vol, s_vol, tau_sigma, s_sigma:
            gate threshold + slope hyperparameters. Default (0, 1) means
            "kill when log_vol_pred or σ exceed the per-batch median by
            more than ~1 unit". Calibrated post-training in eval pipeline.
    """

    # ───────────────────────────────────────────────── construction ──
    def __init__(
        self,
        beta: float = 0.5,
        delta: float = 0.3,
        eta: float = 0.1,
        alpha_pos: float = 5.0,
        eps_sigma: float = 1e-3,
        eps_sharpe: float = 1e-3,
        gate_temp_init: float = 1.0,
        gate_temp_decay: float = 0.92,
        gate_temp_min: float = 0.13,
        tau_vol: float = 0.0,
        s_vol: float = 1.0,
        tau_sigma: float = 0.0,
        s_sigma: float = 1.0,
        # Schedule boundaries (epoch indices, 0-based)
        phase1_end: int = 5,
        phase2_end: int = 15,
        # B1 (differentiable cross-sectional portfolio layer) toggles.
        # When `use_xs_sharpe=True`, L_SR is computed as the Sharpe of K
        # synthetic-cross-section portfolio returns per batch, where each
        # micro-cross-section is a random partition of the batch. Each
        # micro-portfolio uses the SAME long/short Kelly-tanh × gate
        # weighting + leg normalisation that the inference strategy uses,
        # so the training-time loss now exactly mirrors the inference-time
        # portfolio construction. See reports/track_b_findings.md §B1.
        use_xs_sharpe: bool = False,
        xs_n_subgroups: int = 32,
    ):
        super().__init__()
        # Constant coefficients
        self.beta = float(beta)
        self.delta = float(delta)
        self.eta = float(eta)
        # Scheduled coefficients (set by step_epoch)
        self.alpha = 0.0   # SR_gated
        self.gamma = 1.0   # MSE_R
        # Gate hyperparameters
        self.alpha_pos = float(alpha_pos)
        self.eps_sigma = float(eps_sigma)
        self.eps_sharpe = float(eps_sharpe)
        self.gate_temp = float(gate_temp_init)
        self._gate_temp_init = float(gate_temp_init)
        self._gate_temp_decay = float(gate_temp_decay)
        self._gate_temp_min = float(gate_temp_min)
        self.tau_vol = float(tau_vol)
        self.s_vol = float(s_vol)
        self.tau_sigma = float(tau_sigma)
        self.s_sigma = float(s_sigma)
        # Schedule
        self._phase1_end = int(phase1_end)
        self._phase2_end = int(phase2_end)
        # B1 toggles
        self.use_xs_sharpe = bool(use_xs_sharpe)
        self.xs_n_subgroups = int(xs_n_subgroups)

    # ───────────────────────────────────────────────── schedule ──
    def step_epoch(self, epoch: int) -> None:
        """Update α, γ, gate_temp per the warm-up schedule."""
        if epoch < self._phase1_end:
            self.alpha = 0.0
            self.gamma = 1.0
        elif epoch < self._phase2_end:
            self.alpha = 0.3
            self.gamma = 0.5
        else:
            self.alpha = 0.7
            self.gamma = 0.2

        # Gate temperature: T = max(T_min, T_init * decay^epoch)
        self.gate_temp = max(
            self._gate_temp_min,
            self._gate_temp_init * (self._gate_temp_decay ** int(epoch)),
        )

    # ───────────────────────────────────────────────── forward ──
    def forward(
        self,
        output: Dict[str, torch.Tensor],
        true_close_seq: torch.Tensor,
        log_vol_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute composite loss.

        Args:
            output: dict from `RiskAwareHead.forward`, with keys
                {mu_return_H, log_sigma2_H, log_vol_pred, last_close, ...}.
            true_close_seq: [B, pred_len] ground-truth future close prices.
            log_vol_target: [B] target log realized volatility.

        Returns:
            (scalar loss, dict of named components for logging)
        """
        mu_return_H: torch.Tensor = output["mu_return_H"]            # [B]
        log_sigma2_H: torch.Tensor = output["log_sigma2_H"]          # [B]
        log_vol_pred: torch.Tensor = output["log_vol_pred"]          # [B]
        last_close: torch.Tensor = output["last_close"]              # [B]

        # H-step ahead true return derived from the supplied close sequence.
        # Same scaled-space floor as in RiskAwareHead so the predicted and
        # observed returns share an identical denominator for matched samples.
        true_close_H = true_close_seq[:, -1] if true_close_seq.ndim > 1 else true_close_seq
        denom_true = last_close.abs().clamp(min=_RETURN_DENOM_MIN) + 1e-9
        true_return_H = (true_close_H - last_close) / denom_true

        # Bound log-variance for autocast stability
        log_var = log_sigma2_H.clamp(min=_LOG_VAR_MIN, max=_LOG_VAR_MAX)
        sigma = torch.exp(0.5 * log_var)                             # [B]

        # ─── L_NLL: heteroscedastic Gaussian NLL on returns ───
        nll = 0.5 * (log_var + (true_return_H - mu_return_H) ** 2 / torch.exp(log_var))
        L_NLL = nll.mean()

        # ─── L_MSE_R: anchor on return-MSE ───
        L_MSE_R = ((mu_return_H - true_return_H) ** 2).mean()

        # ─── L_VOL: regress log realized vol ───
        L_VOL = ((log_vol_pred - log_vol_target) ** 2).mean()

        # ─── Position (Kelly-scaled tanh) ───
        # Saturates softly: small μ/σ → near-zero; large → ±1
        position = torch.tanh(self.alpha_pos * mu_return_H / (sigma + self.eps_sigma))

        # ─── Gate: continuous, annealed-temperature ───
        T = self.gate_temp
        # Each sigmoid: 1 if value below threshold (low risk), 0 if above (kill)
        gate_vol = torch.sigmoid((self.tau_vol - log_vol_pred) / (self.s_vol * T))
        gate_sigma = torch.sigmoid((self.tau_sigma - sigma) / (self.s_sigma * T))
        gate = gate_vol * gate_sigma                                 # [B], in [0, 1]

        # ─── L_SR_gated: per-batch differentiable Sharpe of GATED returns ───
        if self.use_xs_sharpe:
            # B1 path: differentiable cross-sectional portfolio layer.
            # Partition the batch into K random "synthetic cross-sections".
            # In each, compute long-short normalized weights from
            # (Kelly-tanh × gate) — same operation as the inference strategy
            # — and form a portfolio return. Sharpe is mean / std across the
            # K portfolio returns.
            B = mu_return_H.shape[0]
            K = max(2, min(self.xs_n_subgroups, B // 2))
            # Randomly assign each sample to a group; trim to be divisible.
            chunk = B // K
            B_used = chunk * K
            perm = torch.randperm(B_used, device=mu_return_H.device)
            mu_p   = mu_return_H[perm].view(K, chunk)
            sig_p  = sigma[perm].view(K, chunk)
            gate_p = gate[perm].view(K, chunk)
            ret_p  = true_return_H[perm].view(K, chunk)

            # Kelly-tanh raw weights, gated.  shape: [K, chunk]
            raw = torch.tanh(self.alpha_pos * mu_p / (sig_p + self.eps_sigma)) * gate_p

            # Long-short leg normalisation (each leg's |weights| sum to 1
            # within each cross-section so portfolios are gross-2x by
            # construction and comparable across batches).
            pos_part =  raw.clamp(min=0.0)
            neg_part = (-raw).clamp(min=0.0)
            pos_sum = pos_part.sum(dim=1, keepdim=True) + 1e-9
            neg_sum = neg_part.sum(dim=1, keepdim=True) + 1e-9
            long_w  = pos_part / pos_sum
            short_w = neg_part / neg_sum
            w = long_w - short_w                                # [K, chunk]
            port_return = (w * ret_p).sum(dim=1)                # [K]

            sr_mean = port_return.mean()
            sr_std  = port_return.std(unbiased=False) + self.eps_sharpe
            L_SR_gated = -(sr_mean / sr_std)
        else:
            # Legacy path (Track-B v1): per-sample Sharpe of `gate × position
            # × realised_return`. Surrogate that correlates with portfolio
            # Sharpe but does not match it exactly.
            strat_return = gate * position * true_return_H           # [B]
            sr_mean = strat_return.mean()
            sr_std = strat_return.std(unbiased=False) + self.eps_sharpe
            L_SR_gated = -(sr_mean / sr_std)

        # ─── L_GATE_BCE: gate should match realized profitability ───
        # We compute BCE manually (mathematically identical to
        # `F.binary_cross_entropy(gate, target)`) because the fused PyTorch
        # kernel is autocast-unsafe under bf16/fp16: it raises
        # "binary_cross_entropy is unsafe to autocast" and refuses to run.
        # The manual form uses torch.log + arithmetic which ARE autocast-safe.
        # Recommendation in the docs is to use BCEWithLogits, but our gate is
        # already a product of two sigmoids and is reused downstream as a
        # multiplicative weight on returns, so we keep the sigmoid form and
        # implement BCE inline.
        profitable = (position * true_return_H > 0).float()          # [B] target
        gate_clamped = gate.clamp(1e-6, 1.0 - 1e-6)
        L_GATE_BCE = -(profitable * torch.log(gate_clamped) +
                       (1.0 - profitable) * torch.log(1.0 - gate_clamped)).mean()

        # ─── Composite ───
        L_total = (
            self.alpha * L_SR_gated
            + self.beta * L_NLL
            + self.gamma * L_MSE_R
            + self.delta * L_VOL
            + self.eta * L_GATE_BCE
        )

        components = {
            "L_total": L_total.detach().item(),
            "L_SR_gated": L_SR_gated.detach().item(),
            "L_NLL": L_NLL.detach().item(),
            "L_MSE_R": L_MSE_R.detach().item(),
            "L_VOL": L_VOL.detach().item(),
            "L_GATE_BCE": L_GATE_BCE.detach().item(),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "gate_temp": self.gate_temp,
            "gate_mean": gate.detach().mean().item(),
            "sigma_mean": sigma.detach().mean().item(),
            "position_mean_abs": position.detach().abs().mean().item(),
        }
        return L_total, components
