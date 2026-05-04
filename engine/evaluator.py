import torch
import numpy as np
from utils.metrics import metric

def evaluate(model, test_loader, device, criterion=None,
             close_min=None, close_max=None):
    """Run model over test_loader and compute regression metrics.

    Scaled-space metrics (MSE/MAE/RMSE/R²) are always returned. If both
    ``close_min`` and ``close_max`` arrays are supplied (each of shape [N_samples],
    aligned with the loader's iteration order — only valid when the loader has
    shuffle=False), the function ALSO computes dollar-space metrics by
    inverse-transforming Close predictions/targets per-sample using the
    MinMaxScaler formula:

        x_orig = x_scaled * (close_max - close_min) + close_min

    R² is scale-invariant so it is the same in both spaces; we still report it
    once. Dollar-space MSE/MAE are interpretable as USD² and USD respectively.
    """
    model.eval()
    preds = []
    trues = []
    total_loss = 0

    if criterion is None:
        criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            # Forward pass
            # Note:
            #   * AdaPatch returns a tuple (pred, slice, decode) — first element is the prediction.
            #   * RiskAwareHead returns a dict; "mu_close" is the back-compatible price prediction
            #     tensor of shape [B, pred_len], which is what we score against batch_y.
            outputs = model(batch_x, None)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if isinstance(outputs, dict):
                # Score regression metrics against the price head only — the
                # composite training loss is non-stationary across the warm-up
                # schedule and would not be a meaningful early-stopping signal.
                outputs = outputs["mu_close"]

            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mae, mse, rmse, r2 = metric(preds, trues)

    result = {
        "loss": total_loss / len(test_loader),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

    # Dollar-space metrics if per-sample inverse-scale arrays were supplied
    if close_min is not None and close_max is not None:
        N = preds.shape[0]
        if len(close_min) != N or len(close_max) != N:
            print(f"[evaluator] WARN: close_min/max len mismatch ({len(close_min)} vs N={N}) "
                  f"— skipping dollar-space metrics.")
        else:
            cmin = np.asarray(close_min, dtype=np.float64).reshape(-1, 1)  # [N, 1] broadcasts over pred_len
            cmax = np.asarray(close_max, dtype=np.float64).reshape(-1, 1)
            scale = cmax - cmin
            preds_usd  = preds.astype(np.float64)  * scale + cmin
            trues_usd  = trues.astype(np.float64)  * scale + cmin
            err_usd = preds_usd - trues_usd
            mse_usd  = float(np.mean(err_usd ** 2))
            mae_usd  = float(np.mean(np.abs(err_usd)))
            rmse_usd = float(np.sqrt(mse_usd))
            result["mse_usd"] = mse_usd
            result["mae_usd"] = mae_usd
            result["rmse_usd"] = rmse_usd

    return result
