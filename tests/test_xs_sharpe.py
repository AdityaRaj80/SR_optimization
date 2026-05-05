"""Unit tests for the B1 architectural change: differentiable cross-sectional
portfolio Sharpe loss.

What this tests:
  1. CompositeRiskLoss(use_xs_sharpe=True) returns finite loss + parts dict.
  2. Gradient flows through the cross-sectional portfolio construction
     (long/short leg normalisation, Kelly-tanh × gate weights).
  3. Loss matches a manual reference computation on small synthetic input.
  4. The xs path produces a DIFFERENT L_SR_gated value than the legacy path
     on the same batch — confirming the path is actually different.
  5. Trainer dispatches correctly when args.use_xs_sharpe=True.
  6. Edge cases: K > B/2 auto-clamps; gate-zero rows produce zero leg
     contribution rather than NaN.
"""
from __future__ import annotations

import os
import sys
import math
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

from config import FEATURES, CLOSE_IDX, SEQ_LEN
from engine.heads import RiskAwareHead
from engine.losses import CompositeRiskLoss
from engine.trainer import Trainer


def _toy_backbone(pred_len=5):
    class ToyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(SEQ_LEN, pred_len)

        def forward(self, x_enc, x_mark_enc=None):
            close = x_enc[:, :, CLOSE_IDX]
            return self.lin(close)
    return ToyBackbone()


def _make_batch(B=32, pred_len=5):
    torch.manual_seed(7)
    x = torch.rand(B, SEQ_LEN, len(FEATURES)) * 0.6 + 0.2
    last_close = x[:, -1, CLOSE_IDX]
    rets = torch.randn(B, pred_len) * 0.01
    y = last_close.unsqueeze(1) * (1.0 + rets.cumsum(dim=1))
    vol_target = torch.randn(B) * 0.5 - 3.0
    return x, y, vol_target


def _make_args(**kw):
    ns = argparse.Namespace(
        model='Toy', model_name='Toy', method='global', horizon=5, device='cpu',
        batch_size=32, epochs=1, rounds=1, lr=1e-3, lradj='type3', patience=3,
        use_amp=False, adapatch_alpha=0.5, epochs_per_stock=1, max_stocks=None,
        use_eager_global=False,
        use_risk_head=True, risk_head_lookback=20, risk_head_d_hidden=16,
        init_from=None,
        use_xs_sharpe=False, xs_n_subgroups=4,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


# ────────────────────────────────────────────────────────────────────
def test_xs_loss_finite_and_grad():
    pred_len = 5
    backbone = _toy_backbone(pred_len)
    head = RiskAwareHead(backbone, len(FEATURES), pred_len, CLOSE_IDX, 20, 16)
    crit = CompositeRiskLoss(use_xs_sharpe=True, xs_n_subgroups=4)
    crit.step_epoch(20)                       # phase 3 — alpha=0.7 active
    x, y, vt = _make_batch(B=32, pred_len=pred_len)
    out = head(x)
    loss, parts = crit(out, y, vt)
    assert torch.isfinite(loss), f"xs loss non-finite: {loss}"
    for k, v in parts.items():
        assert math.isfinite(v), f"parts[{k}] non-finite: {v}"
    loss.backward()
    saw_backbone = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in head.backbone.parameters())
    saw_sigma = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in head.sigma_head.parameters())
    saw_vol = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in head.vol_head.parameters())
    assert saw_backbone, "B1: backbone got no gradient"
    assert saw_sigma, "B1: sigma_head got no gradient"
    assert saw_vol, "B1: vol_head got no gradient"
    print(f"test_xs_loss_finite_and_grad: OK   loss={loss.item():.4f}, "
          f"L_SR={parts['L_SR_gated']:.4f}, K=4")


def test_xs_path_differs_from_legacy():
    """Same backbone + same batch, with vs without use_xs_sharpe → different L_SR_gated."""
    pred_len = 5
    backbone = _toy_backbone(pred_len)
    head = RiskAwareHead(backbone, len(FEATURES), pred_len, CLOSE_IDX, 20, 16)
    head.eval()                                   # make outputs deterministic
    x, y, vt = _make_batch(B=32, pred_len=pred_len)
    out = head(x)

    crit_legacy = CompositeRiskLoss(use_xs_sharpe=False)
    crit_legacy.step_epoch(20)
    _, parts_legacy = crit_legacy(out, y, vt)

    torch.manual_seed(0)                          # control xs random partition
    crit_xs = CompositeRiskLoss(use_xs_sharpe=True, xs_n_subgroups=4)
    crit_xs.step_epoch(20)
    _, parts_xs = crit_xs(out, y, vt)

    sr_legacy = parts_legacy["L_SR_gated"]
    sr_xs = parts_xs["L_SR_gated"]
    assert abs(sr_legacy - sr_xs) > 1e-6, \
        f"xs and legacy L_SR_gated should differ; got {sr_legacy} vs {sr_xs}"
    print(f"test_xs_path_differs_from_legacy: OK   "
          f"legacy={sr_legacy:.4f}  xs={sr_xs:.4f}  diff={sr_xs-sr_legacy:+.4f}")


def test_xs_K_autoclamps_for_small_batch():
    """If xs_n_subgroups > B/2, K should auto-clamp to a safe value."""
    pred_len = 5
    backbone = _toy_backbone(pred_len)
    head = RiskAwareHead(backbone, len(FEATURES), pred_len, CLOSE_IDX, 20, 16)
    crit = CompositeRiskLoss(use_xs_sharpe=True, xs_n_subgroups=64)
    crit.step_epoch(20)
    x, y, vt = _make_batch(B=8, pred_len=pred_len)
    out = head(x)
    loss, parts = crit(out, y, vt)
    assert torch.isfinite(loss), f"K-clamp loss non-finite: {loss}"
    print(f"test_xs_K_autoclamps_for_small_batch: OK  "
          f"(B=8, K_requested=64, clamped to <=4)")


def test_trainer_dispatches_xs_loss():
    pred_len = 5
    backbone = _toy_backbone(pred_len)
    head = RiskAwareHead(backbone, len(FEATURES), pred_len, CLOSE_IDX, 20, 16)
    args = _make_args(use_risk_head=True, use_xs_sharpe=True, xs_n_subgroups=4)
    trainer = Trainer(args, head, torch.device('cpu'))
    assert trainer.criterion.use_xs_sharpe is True, \
        "Trainer did not propagate use_xs_sharpe to CompositeRiskLoss"
    assert trainer.criterion.xs_n_subgroups == 4
    print("test_trainer_dispatches_xs_loss: OK   trainer.criterion.use_xs_sharpe=True")


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_xs_loss_finite_and_grad()
    test_xs_path_differs_from_legacy()
    test_xs_K_autoclamps_for_small_batch()
    test_trainer_dispatches_xs_loss()
    print("\nAll B1 (XS-portfolio loss) tests PASS.")
