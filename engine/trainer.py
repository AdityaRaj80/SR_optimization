import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW

from engine.early_stopping import EarlyStopping, adjust_learning_rate
from engine.evaluator import evaluate
from engine.losses import CompositeRiskLoss


def _compute_log_vol_target(batch_y: torch.Tensor,
                            last_close: torch.Tensor | None = None,
                            eps: float = 1e-3) -> torch.Tensor:
    """Realised log-vol target for the L_VOL component of CompositeRiskLoss.

    For each sample, we want a forward-looking volatility number — std of the
    intra-prediction-window returns. Inputs are in *scaled* feature space
    (MinMax-normalised closes), but vol of returns is approximately
    scale-invariant within a single stock so this is a useful supervision
    signal regardless.

    Args
    ----
        batch_y    : [B, H]   future close prices (scaled).
        last_close : [B]      last observed close (scaled). Used only when H==1
                              as a fallback.
        eps        : numerical floor on vol before log.

    Returns
    -------
        log_vol_target : [B]
    """
    if batch_y.shape[1] >= 2:
        prev = batch_y[:, :-1].clamp(min=1e-9)
        ret = batch_y[:, 1:] / prev - 1.0                      # [B, H-1]
        std = ret.std(dim=1, unbiased=False)
        return torch.log(std + eps)
    # H == 1: return-std undefined; use |single-step return| as proxy.
    if last_close is not None:
        prev = last_close.clamp(min=1e-9)
        ret = batch_y[:, 0] / prev - 1.0                       # [B]
        return torch.log(ret.abs() + eps)
    return torch.zeros(batch_y.shape[0], device=batch_y.device)


class Trainer:
    def __init__(self, args, model, device):
        self.args = args
        self.model = model
        self.device = device
        # Risk-aware mode: model is wrapped in RiskAwareHead (returns dict).
        # Detect either via explicit args flag (preferred) or via duck-typing
        # for back-compat with callers that don't set the flag.
        self.use_risk_head = bool(getattr(args, 'use_risk_head', False))
        if self.use_risk_head:
            use_xs = bool(getattr(args, 'use_xs_sharpe', False))
            xs_K = int(getattr(args, 'xs_n_subgroups', 32))
            self.criterion = CompositeRiskLoss(
                use_xs_sharpe=use_xs, xs_n_subgroups=xs_K
            ).to(device)
            xs_msg = f" (xs-portfolio loss, K={xs_K})" if use_xs else " (per-sample Sharpe)"
            print(f"[Trainer] CompositeRiskLoss active{xs_msg}. "
                  f"Schedule: phase1<{self.criterion._phase1_end} (MSE-only) "
                  f"-> phase2<{self.criterion._phase2_end} (alpha=0.3) "
                  f"-> phase3 (alpha=0.7).")
        else:
            self.criterion = nn.MSELoss()
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)

    # ────────────────────────────────────────────────────────────── helpers
    def _forward_loss(self, batch_x: torch.Tensor, batch_y: torch.Tensor):
        """Single forward + loss eval, branching on training mode.

        Returns
        -------
            loss        : scalar tensor (backward-passable)
            parts       : dict of named float components (or {} for plain MSE)
        """
        outputs = self.model(batch_x, None)
        if self.use_risk_head:
            # RiskAwareHead returns a dict; we additionally need a per-batch
            # log-vol target derived from batch_y. last_close is taken from
            # the model's own output dict (matches the head's view of x_enc).
            assert isinstance(outputs, dict), (
                "use_risk_head=True but model output is not a dict. "
                "Wrap the backbone in engine.heads.RiskAwareHead."
            )
            last_close = outputs.get("last_close")
            log_vol_target = _compute_log_vol_target(batch_y, last_close=last_close)
            loss, parts = self.criterion(outputs, batch_y, log_vol_target)
            return loss, parts
        # Legacy path: tensor outputs, single MSE.
        if isinstance(outputs, tuple):
            if self.args.model_name == 'AdaPatch':
                pred, orig, dec = outputs
                loss_pred = self.criterion(pred, batch_y)
                loss_rec = self.criterion(dec, orig)
                loss = (self.args.adapatch_alpha * loss_pred
                        + (1 - self.args.adapatch_alpha) * loss_rec)
            else:
                outputs = outputs[0]
                loss = self.criterion(outputs, batch_y)
        else:
            loss = self.criterion(outputs, batch_y)
        return loss, {}

    def train_epoch(self, train_loader):
        self.model.train()
        train_loss = []
        accum_parts: dict[str, list] = {}

        for batch_x, batch_y in train_loader:
            self.optimizer.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            if self.args.use_amp:
                device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                # Use bf16 on Ampere+ GPUs (compute capability >= 8.0: A100, H100, H200, RTX 30xx+).
                # bf16 has the same dynamic range as fp32 (8-bit exponent), so it does NOT overflow
                # in transformer attention softmax — the failure mode that caused iTransformer
                # sequential training to NaN under fp16 AMP. fp16 fallback is kept for older GPUs.
                if self.device.type == 'cuda' and torch.cuda.get_device_capability(self.device)[0] >= 8:
                    amp_dtype = torch.bfloat16
                else:
                    amp_dtype = torch.float16
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    loss, parts = self._forward_loss(batch_x, batch_y)

                loss.backward()  # Should use GradScaler for full AMP, but simplified here
                self.optimizer.step()
            else:
                loss, parts = self._forward_loss(batch_x, batch_y)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            train_loss.append(loss.item())
            for k, v in parts.items():
                accum_parts.setdefault(k, []).append(v)

        avg_parts = {k: sum(vs) / max(len(vs), 1) for k, vs in accum_parts.items()}
        return sum(train_loss) / max(len(train_loss), 1), avg_parts

    @staticmethod
    def _format_parts(parts: dict) -> str:
        """Compact one-line summary of CompositeRiskLoss components."""
        if not parts:
            return ""
        keys = ["L_MSE_R", "L_NLL", "L_VOL", "L_SR_gated", "L_GATE_BCE",
                "alpha", "gamma", "gate_mean", "sigma_mean", "position_mean_abs"]
        bits = []
        for k in keys:
            if k in parts:
                v = parts[k]
                bits.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        return " | " + " ".join(bits)

    # ────────────────────────────────────────────────────────────── global
    def train_global(self, train_loader, val_loader, test_loader, save_path):
        # Track B (composite Sharpe loss) requires running the full schedule —
        # Phase 1 (MSE warm-up), Phase 2 (alpha=0.3), Phase 3 (alpha=0.7).
        # Early stopping on val MSE is INCORRECT here because the composite
        # objective is *designed* to trade some price-MSE for risk-adjusted
        # P&L: as the Sharpe gradient activates in Phase 2/3 the price head
        # shifts from a point-forecast minimiser to a return-direction-with-
        # uncertainty estimator. Val MSE rising is then a feature, not a bug.
        # We therefore DISABLE early stopping in risk-head mode and save the
        # FINAL-epoch state (post-Phase-3) instead of the best-val-MSE one.
        if self.use_risk_head:
            print(f"[Trainer] Track B mode: early stopping DISABLED. "
                  f"Will run full {self.args.epochs}-epoch schedule and save "
                  f"final-epoch state to {save_path}.")
            early_stopping = None
        else:
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        best_val_path = None
        if self.use_risk_head:
            # Also keep a parallel "best val MSE" checkpoint for diagnostics —
            # not loaded back, but useful for ablation tables comparing
            # final-epoch (Sharpe-trained) vs best-val (MSE-aligned) on the
            # cross-sectional smoke pipeline.
            best_val_path = save_path.replace(".pth", "_bestval.pth")
            best_val_loss = float("inf")

        for epoch in range(self.args.epochs):
            t1 = time.time()
            if self.use_risk_head:
                self.criterion.step_epoch(epoch)
            train_loss, parts = self.train_epoch(train_loader)

            # For val/test we want a *stationary* signal — MSE on mu_close —
            # so we explicitly pass nn.MSELoss() (the evaluator extracts
            # mu_close automatically when output is a dict). This MSE is now
            # only used for logging + diagnostics in Track B mode, not for
            # early-stopping decisions.
            eval_crit = nn.MSELoss() if self.use_risk_head else self.criterion
            val_metrics = evaluate(self.model, val_loader, self.device, eval_crit)
            val_loss = val_metrics["loss"]

            test_metrics = evaluate(self.model, test_loader, self.device, eval_crit)
            test_loss = test_metrics["loss"]

            print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f} "
                  f"Vali Loss: {val_loss:.5f} Test Loss: {test_loss:.5f} | "
                  f"Time: {time.time()-t1:.2f}s"
                  f"{self._format_parts(parts)}")

            if early_stopping is not None:
                early_stopping(val_loss, self.model, save_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            else:
                # Track B: save best-val-MSE for diagnostics (not loaded back).
                if best_val_path is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), best_val_path)
                    print(f"  [diagnostic] new best val MSE {val_loss:.6f} -> "
                          f"{os.path.basename(best_val_path)}")

            adjust_learning_rate(self.optimizer, epoch + 1, self.args)

        if early_stopping is not None:
            # Legacy MSE path: load back the best-val checkpoint that
            # EarlyStopping saved.
            self.model.load_state_dict(torch.load(save_path))
        else:
            # Track B: persist the FINAL-epoch state. This is the
            # Phase-3-converged model, the actual artefact we evaluate.
            torch.save(self.model.state_dict(), save_path)
            print(f"[Trainer] Track B final-epoch state saved -> {save_path}")
        return self.model

    # ────────────────────────────────────────────────────────────── sequential
    def train_sequential(self, data_loader_obj, val_loader, test_loader, save_path):
        """data_loader_obj is the UnifiedDataLoader instance. We call its
        `iter_train_loaders()` method ONCE PER ROUND to get a fresh generator,
        so each round re-iterates the full stock pool with only one stock's
        DataLoader resident in memory at a time. Scientifically equivalent to
        the previous list-based approach (same data, same order, same model
        state propagation) but bounded memory footprint.
        """
        # No early stopping to mimic catastrophic forgetting effect if present
        best_val = float('inf')
        epochs_per_stock = getattr(self.args, 'epochs_per_stock', 10)
        total_stocks = len(data_loader_obj.train_stocks)
        import gc

        for r in range(self.args.rounds):
            t1 = time.time()
            if self.use_risk_head:
                # In sequential mode each "round" is the analogue of one epoch
                # over the stock universe; advance the schedule per round.
                self.criterion.step_epoch(r)
            print(f"Starting Round {r+1}/{self.args.rounds} over ~{total_stocks} stocks "
                  f"({epochs_per_stock} epochs/stock)")

            train_losses = []
            last_parts: dict = {}
            # Fresh generator each round (generators are single-use)
            for idx, stock_loader in enumerate(data_loader_obj.iter_train_loaders()):
                stock_t = time.time()
                stock_losses = []
                for ep in range(epochs_per_stock):
                    loss, parts = self.train_epoch(stock_loader)
                    stock_losses.append(loss)
                    if parts:
                        last_parts = parts
                avg_loss = sum(stock_losses) / len(stock_losses)
                train_losses.append(avg_loss)
                if idx % 50 == 0:
                    print(f"  Stock {idx}/{total_stocks} | "
                          f"Avg Loss: {avg_loss:.5f} | Time: {time.time()-stock_t:.1f}s")
                # Explicit cleanup so the underlying dataset/sequences free
                # their numpy arrays before the next stock is built.
                del stock_loader
                gc.collect()

            train_loss = sum(train_losses) / len(train_losses) if train_losses else float('nan')

            eval_crit = nn.MSELoss() if self.use_risk_head else self.criterion
            val_metrics = evaluate(self.model, val_loader, self.device, eval_crit)
            val_loss = val_metrics["loss"]

            print(f"Round: {r+1} | Train Loss: {train_loss:.5f} Vali Loss: {val_loss:.5f} | "
                  f"Time: {time.time()-t1:.2f}s"
                  f"{self._format_parts(last_parts)}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  Saved best model at round {r+1}")

            adjust_learning_rate(self.optimizer, r + 1, self.args)

        self.model.load_state_dict(torch.load(save_path))
        return self.model
