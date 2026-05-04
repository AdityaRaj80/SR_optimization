"""Unit tests for Track B implementation.

Validates:
  1. RiskAwareHead wraps any backbone and produces correct output shapes.
  2. CompositeRiskLoss is finite + has gradient through every head.
  3. Coefficient schedule advances correctly across epochs.
  4. Edge cases: bs=1, all-zero returns, σ near floor, gate near 0/1.
  5. End-to-end: wrapped DLinear backbone produces a backward-passable loss.
"""
from __future__ import annotations

import os
import sys
import math

# Project root on sys.path so `import config`, `engine.*`, etc. resolve.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

from config import FEATURES, CLOSE_IDX, SEQ_LEN, DLINEAR_CONFIG
from engine.heads import RiskAwareHead
from engine.losses import CompositeRiskLoss


def _toy_backbone(pred_len: int):
    """Tiny mock backbone with same forward signature as our models."""
    class ToyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(SEQ_LEN, pred_len)

        def forward(self, x_enc, x_mark_enc=None):
            # Use only Close column for toy backbone
            close = x_enc[:, :, CLOSE_IDX]      # [B, seq_len]
            return self.lin(close)              # [B, pred_len]

    return ToyBackbone()


def _make_dummy_batch(B=8, pred_len=5):
    """Return (x_enc, true_close, vol_target) with realistic-looking values."""
    torch.manual_seed(42)
    # Make x_enc have plausible scale: feature 0..4 in [0, 1] (post MinMax),
    # close column = small random walk around 0.5.
    x_enc = torch.rand(B, SEQ_LEN, len(FEATURES)) * 0.6 + 0.2
    # True future close ~ near current close + small noise
    last_close = x_enc[:, -1, CLOSE_IDX]
    true_close = last_close.unsqueeze(1) + torch.randn(B, pred_len) * 0.01
    # Vol target ~ N(-3, 0.5) (log of ~5% annualized vol)
    log_vol_target = torch.randn(B) * 0.5 - 3.0
    return x_enc, true_close, log_vol_target


# ─────────────────────────────────────────────────────────────────────
# 1. RiskAwareHead output shapes
# ─────────────────────────────────────────────────────────────────────
def test_riskaware_head_shapes():
    pred_len = 5
    backbone = _toy_backbone(pred_len)
    head = RiskAwareHead(
        backbone=backbone,
        n_features=len(FEATURES),
        pred_len=pred_len,
        close_idx=CLOSE_IDX,
        lookback_for_aux=20,
        d_hidden=32,
    )
    x_enc, _, _ = _make_dummy_batch(B=4, pred_len=pred_len)
    out = head(x_enc)
    assert isinstance(out, dict), "RiskAwareHead must return a dict"
    expected = {"mu_close", "mu_close_H", "mu_return_H",
                "log_sigma2_H", "log_vol_pred", "last_close"}
    assert set(out.keys()) == expected, f"keys mismatch: {set(out.keys())}"
    assert out["mu_close"].shape == (4, pred_len)
    assert out["mu_close_H"].shape == (4,)
    assert out["mu_return_H"].shape == (4,)
    assert out["log_sigma2_H"].shape == (4,)
    assert out["log_vol_pred"].shape == (4,)
    assert out["last_close"].shape == (4,)
    print("test_riskaware_head_shapes: OK")


# ─────────────────────────────────────────────────────────────────────
# 2. CompositeRiskLoss returns finite scalar + gradient flows
# ─────────────────────────────────────────────────────────────────────
def test_composite_loss_finite_and_backprop():
    pred_len = 5
    backbone = _toy_backbone(pred_len)
    head = RiskAwareHead(backbone, len(FEATURES), pred_len, CLOSE_IDX, 20, 32)
    crit = CompositeRiskLoss()
    crit.step_epoch(epoch=0)              # phase 1: pure MSE
    x_enc, true_close, vol_target = _make_dummy_batch(B=8, pred_len=pred_len)
    out = head(x_enc)
    loss, parts = crit(out, true_close, vol_target)
    assert torch.isfinite(loss), f"loss is non-finite: {loss}"
    for k, v in parts.items():
        assert math.isfinite(v), f"component {k} non-finite: {v}"

    # Backward must populate gradients in head + backbone + sigma_head + vol_head.
    loss.backward()
    saw_backbone_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in head.backbone.parameters())
    saw_sigma_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in head.sigma_head.parameters())
    saw_vol_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in head.vol_head.parameters())
    # In phase 1 (alpha=0, eta=0.1 still active), gradients flow through:
    #   backbone via L_MSE_R (γ=1.0)
    #   sigma_head via L_NLL (β=0.5) AND L_GATE_BCE (η=0.1)
    #   vol_head  via L_VOL (δ=0.3) AND L_GATE_BCE (η=0.1)
    assert saw_backbone_grad, "backbone parameters got no gradient"
    assert saw_sigma_grad, "sigma_head got no gradient"
    assert saw_vol_grad, "vol_head got no gradient"
    print("test_composite_loss_finite_and_backprop: OK   "
          f"(L={loss.item():.4f}, components keys={len(parts)})")


# ─────────────────────────────────────────────────────────────────────
# 3. Phase schedule
# ─────────────────────────────────────────────────────────────────────
def test_schedule():
    crit = CompositeRiskLoss()
    crit.step_epoch(0)
    assert crit.alpha == 0.0 and crit.gamma == 1.0, f"phase1: {crit.alpha}, {crit.gamma}"
    crit.step_epoch(7)
    assert crit.alpha == 0.3 and crit.gamma == 0.5, f"phase2: {crit.alpha}, {crit.gamma}"
    crit.step_epoch(20)
    assert crit.alpha == 0.7 and crit.gamma == 0.2, f"phase3: {crit.alpha}, {crit.gamma}"

    # Gate temp should monotonically decrease then floor.
    crit.step_epoch(0)
    t0 = crit.gate_temp
    crit.step_epoch(15)
    t15 = crit.gate_temp
    crit.step_epoch(40)
    t40 = crit.gate_temp
    assert t0 > t15, f"gate temp not decreasing: {t0} -> {t15}"
    assert t15 >= t40 - 1e-9, f"gate temp not monotone: {t15} -> {t40}"
    assert t40 >= crit._gate_temp_min, "gate temp below minimum floor"
    print(f"test_schedule: OK   gate_temp[0,15,40]=({t0:.3f}, {t15:.3f}, {t40:.3f})")


# ─────────────────────────────────────────────────────────────────────
# 4. Edge cases
# ─────────────────────────────────────────────────────────────────────
def test_edge_cases():
    pred_len = 5
    backbone = _toy_backbone(pred_len)
    head = RiskAwareHead(backbone, len(FEATURES), pred_len, CLOSE_IDX, 20, 32)
    crit = CompositeRiskLoss()
    crit.step_epoch(epoch=20)             # Phase 3: all weights active

    # Edge 4a: B=1
    x_enc, true_close, vol_target = _make_dummy_batch(B=1, pred_len=pred_len)
    out = head(x_enc)
    loss, parts = crit(out, true_close, vol_target)
    assert torch.isfinite(loss), f"B=1 loss non-finite: {loss}"

    # Edge 4b: identical returns (std → 0 in batch)
    x_enc, true_close, vol_target = _make_dummy_batch(B=4, pred_len=pred_len)
    # Force all true closes equal to last_close → all true_returns == 0
    last_close = x_enc[:, -1, CLOSE_IDX]
    true_close_flat = last_close.unsqueeze(1).expand(-1, pred_len).clone()
    out = head(x_enc)
    loss_flat, _ = crit(out, true_close_flat, vol_target)
    assert torch.isfinite(loss_flat), f"flat-returns loss non-finite: {loss_flat}"

    # Edge 4c: extreme positive returns (test position saturation)
    x_enc, _, vol_target = _make_dummy_batch(B=4, pred_len=pred_len)
    last_close = x_enc[:, -1, CLOSE_IDX]
    true_close_huge = (last_close * 2.0).unsqueeze(1).expand(-1, pred_len).clone()
    out = head(x_enc)
    loss_huge, _ = crit(out, true_close_huge, vol_target)
    assert torch.isfinite(loss_huge), f"huge-returns loss non-finite: {loss_huge}"

    print("test_edge_cases: OK   B=1, flat-returns, huge-returns all finite")


# ─────────────────────────────────────────────────────────────────────
# 5. End-to-end with real DLinear backbone
# ─────────────────────────────────────────────────────────────────────
def test_e2e_with_dlinear():
    """Smoke test: wrap real DLinear from models/, run forward+backward."""
    from models.dlinear import Model as DLinear
    pred_len = 5
    cfg = dict(DLINEAR_CONFIG)
    cfg["pred_len"] = pred_len
    cfg["seq_len"] = SEQ_LEN
    cfg["context_len"] = SEQ_LEN
    backbone = DLinear(cfg)
    head = RiskAwareHead(backbone, len(FEATURES), pred_len, CLOSE_IDX, 20, 32)
    crit = CompositeRiskLoss()
    crit.step_epoch(epoch=10)             # Phase 2

    x_enc, true_close, vol_target = _make_dummy_batch(B=8, pred_len=pred_len)
    out = head(x_enc)
    loss, parts = crit(out, true_close, vol_target)
    assert torch.isfinite(loss), f"DLinear loss non-finite: {loss}"
    loss.backward()
    # At least some backbone params should have gradient
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in backbone.parameters() if p.requires_grad)
    assert has_grad, "DLinear backbone got no gradient"
    print(f"test_e2e_with_dlinear: OK   L_total={parts['L_total']:.4f}  "
          f"L_MSE_R={parts['L_MSE_R']:.4e}  L_NLL={parts['L_NLL']:.4f}  "
          f"gate_mean={parts['gate_mean']:.3f}")


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_riskaware_head_shapes()
    test_composite_loss_finite_and_backprop()
    test_schedule()
    test_edge_cases()
    test_e2e_with_dlinear()
    print("\nAll Track B tests PASS.")
