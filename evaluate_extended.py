"""Extended evaluation: naive baseline + directional accuracy + per-stock MAPE.

Runs over saved checkpoints and writes:
  - results/extended_metrics.csv     (aggregate, model + naive baseline side-by-side)
  - results/per_stock_metrics.csv    (per-stock MAPE breakdown)

Usage (currently hard-coded for DLinear sequential horizons):
    python evaluate_extended.py
"""
import os
import gc
import argparse
import numpy as np
import pandas as pd
import torch

from config import SEQ_LEN, CLOSE_IDX, MODEL_SAVE_DIR, RESULTS_DIR
from data_loader import UnifiedDataLoader
from models import model_dict
from train import get_config_for_model


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def basic_metrics(pred, true):
    """Return MSE / MAE / RMSE / R² over flat-arrays."""
    err = pred - true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def directional_accuracy(pred, true, last_close):
    """Fraction of samples where sign(pred − last) == sign(true − last).
    pred / true: [N, pred_len]   last_close: [N]
    Excludes samples where the actual change is exactly zero (rare).
    """
    pred_dir = np.sign(pred - last_close[:, None])
    true_dir = np.sign(true - last_close[:, None])
    mask = true_dir != 0
    if mask.sum() == 0:
        return float("nan")
    return float((pred_dir[mask] == true_dir[mask]).mean())


def per_stock_mape(pred_usd, true_usd, close_max):
    """Group test samples by their stock (proxied by close_max — each stock
    has a unique val-period max) and report MAPE per stock.
    Returns a DataFrame: stock_id, n_samples, mape_pct.
    """
    # Reduce close_max from per-sample-per-step to per-sample (it's the same
    # value across the pred_len axis since one sample is one stock context).
    if close_max.ndim > 1:
        close_max = close_max[:, 0]

    df = []
    for cm in np.unique(close_max):
        mask = close_max == cm
        if mask.sum() == 0:
            continue
        p = pred_usd[mask]
        t = true_usd[mask]
        nz = t != 0
        if nz.sum() == 0:
            continue
        mape = float(np.mean(np.abs((p[nz] - t[nz]) / t[nz]))) * 100.0
        df.append({
            "stock_close_max": float(cm),
            "n_samples": int(mask.sum()),
            "mape_pct": mape,
        })
    return pd.DataFrame(df).sort_values("mape_pct").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (single horizon)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_one(model_name, method, horizon, device, batch_size=512):
    print(f"\n{'='*70}\n{model_name} {method} H={horizon}\n{'='*70}")

    loader = UnifiedDataLoader(seq_len=SEQ_LEN, horizon=horizon, batch_size=batch_size)
    _, test_loader = loader.get_val_test_loaders()

    # Load checkpoint
    configs = get_config_for_model(model_name, horizon)
    model = model_dict[model_name](configs).to(device)
    ckpt_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_{method}_H{horizon}.pth")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # Run predictions
    preds, trues, last_closes = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float().to(device)
            outputs = model(batch_x, None)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.numpy())
            last_closes.append(batch_x[:, -1, CLOSE_IDX].cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    last_closes = np.concatenate(last_closes)

    # Naive baseline: predict last_close repeated across the full horizon
    naive = np.tile(last_closes[:, None], (1, preds.shape[1]))

    # Inverse-transform to dollar space
    cmin = loader.test_close_min.reshape(-1, 1).astype(np.float64)
    cmax = loader.test_close_max.reshape(-1, 1).astype(np.float64)
    scale = cmax - cmin
    preds_usd = preds.astype(np.float64) * scale + cmin
    trues_usd = trues.astype(np.float64) * scale + cmin
    naive_usd = naive.astype(np.float64) * scale + cmin

    # Aggregate metrics
    m_scaled = basic_metrics(preds, trues)
    m_usd = basic_metrics(preds_usd, trues_usd)
    n_scaled = basic_metrics(naive, trues)
    n_usd = basic_metrics(naive_usd, trues_usd)

    # Directional accuracy (use scaled space since signs are scale-invariant
    # under positive monotone transforms; but for a fair check compute in usd)
    last_usd = last_closes * (cmax[:, 0] - cmin[:, 0]) + cmin[:, 0]
    model_dir = directional_accuracy(preds_usd, trues_usd, last_usd)

    # For naive, sign(pred - last) is always 0 → effectively 0% directional
    # accuracy. The more meaningful baseline is the BEST CONSTANT class
    # predictor: max(pct_up, pct_down).
    actual_change = trues_usd - last_usd[:, None]
    pct_up = float((actual_change > 0).mean())
    pct_down = float((actual_change < 0).mean())
    constant_baseline_dir = max(pct_up, pct_down)

    # Per-stock MAPE
    pstock = per_stock_mape(preds_usd, trues_usd, loader.test_close_max)
    pstock["model"] = model_name
    pstock["method"] = method
    pstock["horizon"] = horizon

    summary = {
        "Model": model_name,
        "Method": method,
        "Horizon": horizon,
        # Model metrics
        "R2":          m_scaled["r2"],
        "MAE":         m_scaled["mae"],
        "RMSE":        m_scaled["rmse"],
        "MAE_USD":     m_usd["mae"],
        "RMSE_USD":    m_usd["rmse"],
        "DirAcc":      model_dir,
        # Naive baseline (last-value persistence)
        "Naive_R2":         n_scaled["r2"],
        "Naive_MAE":        n_scaled["mae"],
        "Naive_RMSE":       n_scaled["rmse"],
        "Naive_MAE_USD":    n_usd["mae"],
        "Naive_RMSE_USD":   n_usd["rmse"],
        # Constant-baseline directional accuracy
        "Const_DirAcc":     constant_baseline_dir,
        # Distribution context
        "Pct_Up":      pct_up,
        "Pct_Down":    pct_down,
        # Per-stock summary stats
        "PerStock_MeanMAPE":   float(pstock["mape_pct"].mean()),
        "PerStock_MedianMAPE": float(pstock["mape_pct"].median()),
        "PerStock_StdMAPE":    float(pstock["mape_pct"].std()),
        "PerStock_MinMAPE":    float(pstock["mape_pct"].min()),
        "PerStock_MaxMAPE":    float(pstock["mape_pct"].max()),
        "N_Stocks":            int(len(pstock)),
        "N_Test_Samples":      int(len(preds)),
    }

    # Cleanup before next iteration
    del model, loader, test_loader, preds, trues, last_closes, naive
    del preds_usd, trues_usd, naive_usd
    torch.cuda.empty_cache()
    gc.collect()

    return summary, pstock


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="DLinear")
    p.add_argument("--method", default="sequential")
    p.add_argument("--horizons", type=int, nargs="+", default=[5, 20, 60, 120, 240])
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    summaries, all_pstock = [], []
    for h in args.horizons:
        s, ps = evaluate_one(args.model, args.method, h, device)
        summaries.append(s)
        all_pstock.append(ps)

    summary_df = pd.DataFrame(summaries)
    pstock_df = pd.concat(all_pstock, ignore_index=True)

    out_summary = os.path.join(RESULTS_DIR, "extended_metrics.csv")
    out_pstock = os.path.join(RESULTS_DIR, "per_stock_metrics.csv")
    summary_df.to_csv(out_summary, index=False)
    pstock_df.to_csv(out_pstock, index=False)

    print(f"\nWrote: {out_summary}")
    print(f"Wrote: {out_pstock}")
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)
    cols_show = ["Model", "Method", "Horizon", "R2", "Naive_R2",
                 "MAE_USD", "Naive_MAE_USD", "DirAcc", "Const_DirAcc",
                 "PerStock_MedianMAPE", "PerStock_StdMAPE"]
    print(summary_df[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
