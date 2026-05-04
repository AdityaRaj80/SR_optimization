"""Integration test for Phase D trainer + train.py wiring.

Verifies:
  1. `Trainer(args, model, device)` with `args.use_risk_head=True` builds a
     `CompositeRiskLoss` criterion and dispatches dict outputs through it.
  2. `_compute_log_vol_target` produces a finite [B] tensor for H>=2 and H==1.
  3. `train_epoch` runs end-to-end on a tiny synthetic batch with a
     RiskAwareHead-wrapped backbone, parts dict gets populated.
  4. `evaluate` against a dict-output model extracts mu_close and computes
     finite regression metrics.
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
from torch.utils.data import DataLoader, TensorDataset

from config import FEATURES, CLOSE_IDX, SEQ_LEN
from engine.heads import RiskAwareHead
from engine.losses import CompositeRiskLoss
from engine.trainer import Trainer, _compute_log_vol_target
from engine.evaluator import evaluate


def _make_args(**kw):
    """Build a minimal argparse.Namespace mimicking train.py's args."""
    ns = argparse.Namespace(
        model='Toy',
        model_name='Toy',
        method='global',
        horizon=5,
        device='cpu',
        batch_size=8,
        epochs=2,
        rounds=1,
        lr=1e-3,
        lradj='type3',
        patience=3,
        use_amp=False,
        adapatch_alpha=0.5,
        epochs_per_stock=1,
        max_stocks=None,
        use_eager_global=False,
        use_risk_head=False,
        risk_head_lookback=20,
        risk_head_d_hidden=16,
        init_from=None,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _toy_backbone(pred_len=5):
    class ToyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(SEQ_LEN, pred_len)

        def forward(self, x_enc, x_mark_enc=None):
            close = x_enc[:, :, CLOSE_IDX]
            return self.lin(close)
    return ToyBackbone()


def _make_loader(B=16, pred_len=5, batch_size=8):
    torch.manual_seed(0)
    x = torch.rand(B, SEQ_LEN, len(FEATURES)) * 0.6 + 0.2
    last_close = x[:, -1, CLOSE_IDX]
    # H-step close ≈ last_close * (1 + small_return)
    rets = torch.randn(B, pred_len) * 0.01
    y = last_close.unsqueeze(1) * (1.0 + rets.cumsum(dim=1))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


# ───────────────────────────────────────────────────────────────────
def test_compute_log_vol_target():
    # H >= 2: returns std of intra-window returns
    y = torch.tensor([[1.0, 1.01, 0.99, 1.02, 1.0],
                      [2.0, 2.05, 2.10, 2.05, 2.00]])
    out = _compute_log_vol_target(y)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()

    # H == 1 with last_close fallback
    y1 = torch.tensor([[1.05], [2.10]])
    last = torch.tensor([1.0, 2.0])
    out1 = _compute_log_vol_target(y1, last_close=last)
    assert out1.shape == (2,)
    assert torch.isfinite(out1).all()
    print("test_compute_log_vol_target: OK   "
          f"H=5: {out.tolist()}, H=1: {out1.tolist()}")


def test_trainer_init_dispatch():
    pred_len = 5
    backbone = _toy_backbone(pred_len)
    head = RiskAwareHead(backbone, len(FEATURES), pred_len, CLOSE_IDX, 20, 16)
    args = _make_args(use_risk_head=True)
    trainer = Trainer(args, head, torch.device('cpu'))
    assert isinstance(trainer.criterion, CompositeRiskLoss), \
        f"expected CompositeRiskLoss, got {type(trainer.criterion).__name__}"
    assert trainer.use_risk_head is True
    print("test_trainer_init_dispatch: OK   "
          "use_risk_head=True -> CompositeRiskLoss")


def test_trainer_init_legacy():
    pred_len = 5
    backbone = _toy_backbone(pred_len)
    args = _make_args(use_risk_head=False)
    trainer = Trainer(args, backbone, torch.device('cpu'))
    assert isinstance(trainer.criterion, nn.MSELoss), \
        f"expected MSELoss, got {type(trainer.criterion).__name__}"
    assert trainer.use_risk_head is False
    print("test_trainer_init_legacy: OK   "
          "use_risk_head=False -> nn.MSELoss")


def test_train_epoch_risk_head():
    pred_len = 5
    backbone = _toy_backbone(pred_len)
    head = RiskAwareHead(backbone, len(FEATURES), pred_len, CLOSE_IDX, 20, 16)
    args = _make_args(use_risk_head=True)
    trainer = Trainer(args, head, torch.device('cpu'))
    trainer.criterion.step_epoch(0)        # phase 1
    loader = _make_loader(B=16, pred_len=pred_len, batch_size=8)
    avg_loss, parts = trainer.train_epoch(loader)
    assert math.isfinite(avg_loss), f"avg loss non-finite: {avg_loss}"
    expected = {"L_total", "L_MSE_R", "L_NLL", "L_VOL", "L_SR_gated", "L_GATE_BCE",
                "alpha", "gamma", "gate_temp", "gate_mean", "sigma_mean",
                "position_mean_abs"}
    assert expected.issubset(set(parts.keys())), \
        f"parts missing keys: {expected - set(parts.keys())}"
    for k, v in parts.items():
        assert math.isfinite(v), f"parts[{k}] non-finite: {v}"
    print(f"test_train_epoch_risk_head: OK   loss={avg_loss:.4f}, "
          f"L_MSE_R={parts['L_MSE_R']:.4f}")


def test_train_epoch_legacy():
    pred_len = 5
    backbone = _toy_backbone(pred_len)
    args = _make_args(use_risk_head=False)
    trainer = Trainer(args, backbone, torch.device('cpu'))
    loader = _make_loader(B=16, pred_len=pred_len, batch_size=8)
    avg_loss, parts = trainer.train_epoch(loader)
    assert math.isfinite(avg_loss)
    assert parts == {}, f"legacy mode should return empty parts; got {parts}"
    print(f"test_train_epoch_legacy: OK   loss={avg_loss:.4f}")


def test_evaluate_dict_output():
    pred_len = 5
    backbone = _toy_backbone(pred_len)
    head = RiskAwareHead(backbone, len(FEATURES), pred_len, CLOSE_IDX, 20, 16)
    loader = _make_loader(B=16, pred_len=pred_len, batch_size=8)
    metrics = evaluate(head, loader, torch.device('cpu'), nn.MSELoss())
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["mse"])
    assert math.isfinite(metrics["mae"])
    print(f"test_evaluate_dict_output: OK   "
          f"loss={metrics['loss']:.4f} mse={metrics['mse']:.4e}")


# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_compute_log_vol_target()
    test_trainer_init_dispatch()
    test_trainer_init_legacy()
    test_train_epoch_risk_head()
    test_train_epoch_legacy()
    test_evaluate_dict_output()
    print("\nAll Phase D wiring tests PASS.")
