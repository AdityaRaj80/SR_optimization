"""Empirical verification of Theorem A1.

For each (model, horizon) checkpoint trained with `--use_risk_head`, compute:

  1. Per-stock predicted sigma_i at each test rebalance t.
  2. Cross-sectional CV of sigma at each t: CV_sigma(t) = std_i(sigma_i(t)) / mean_i(sigma_i(t)).
  3. Time-mean of CV_sigma across rebalances: bar_CV_sigma(H).

Then load existing summary_xs_*.json files (produced by cross_sectional_smoke.py)
to extract Sharpe-gap delta per horizon:

  Delta_SR(H) = TB_RA_Sharpe(H) - MSE_simple_Sharpe(H)

Theorem A1 prediction: across horizons, Delta_SR(H) should be monotone-increasing
(or at least positively correlated) with bar_CV_sigma(H), since the squared-Sharpe
gap is a sigma-weighted variance of per-stock Kelly scores (Lagrange-identity form
of Theorem A1, see reports/track_b_theorem_A1.md section 5.2).

Usage:
    python verify_A1.py --model GCFormer --ckpt_suffix _riskhead_xs --tag xs
    python verify_A1.py --model GCFormer --ckpt_suffix _riskhead    --tag v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(os.path.dirname(_HERE), "config.py")):
    SR_OPT_DIR = os.path.dirname(_HERE)
else:
    SR_OPT_DIR = os.path.abspath(os.path.join(_HERE, "..", "SR_optimization"))
sys.path.insert(0, SR_OPT_DIR)

from config import SEQ_LEN, CLOSE_IDX, MODEL_SAVE_DIR, FEATURES   # noqa: E402
from data_loader import UnifiedDataLoader                          # noqa: E402
from models import model_dict                                       # noqa: E402
from train import get_config_for_model                              # noqa: E402
from engine.heads import RiskAwareHead                              # noqa: E402

OUT_DIR = os.path.join(_HERE, "results")


def predict_sigma(model, loader, device):
    """Forward pass extracting only the per-sample sigma from RiskAwareHead."""
    model.eval()
    sigmas, stock_ids_proxy = [], []
    with torch.no_grad():
        for batch_x, _ in loader:
            bx = batch_x.float().to(device)
            out = model(bx, None)
            assert isinstance(out, dict), "Need a RiskAwareHead-wrapped model"
            log_var = out["log_sigma2_H"].clamp(min=-12.0, max=4.0)
            sigma = torch.exp(0.5 * log_var)
            sigmas.append(sigma.cpu().numpy())
    return np.concatenate(sigmas).astype(np.float64)


def compute_cv_panel(sigma_per_sample, stock_id_pivot):
    """For each timestamp t, compute cross-sectional CV of sigma across stocks."""
    unique = np.unique(stock_id_pivot)
    per_stock = {sid: sigma_per_sample[stock_id_pivot == sid] for sid in unique}
    T = min(len(s) for s in per_stock.values())
    sigma_panel = np.full((T, len(unique)), np.nan)
    for j, sid in enumerate(unique):
        sigma_panel[:, j] = per_stock[sid][:T]
    cv_per_t = np.full(T, np.nan)
    for t in range(T):
        row = sigma_panel[t]
        valid = ~np.isnan(row)
        if valid.sum() < 5:
            continue
        m = row[valid].mean()
        s = row[valid].std(ddof=1)
        if m > 1e-9:
            cv_per_t[t] = s / m
    return cv_per_t, sigma_panel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="GCFormer")
    p.add_argument("--method", default="global")
    p.add_argument("--horizons", type=str, default="5,20,60")
    p.add_argument("--ckpt_suffix", default="_riskhead",
                   help="_riskhead for Track B v1, _riskhead_xs for B1.")
    p.add_argument("--tag", default="v1",
                   help="String identifying which Track-B variant we are verifying.")
    args = p.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | model={args.model} | horizons={horizons} | "
          f"ckpt_suffix={args.ckpt_suffix}", flush=True)

    rows = []
    for H in horizons:
        ckpt = os.path.join(
            MODEL_SAVE_DIR,
            f"{args.model}_{args.method}_H{H}{args.ckpt_suffix}.pth")
        if not os.path.exists(ckpt):
            print(f"  [skip] H={H}: ckpt not found at {ckpt}")
            continue

        loader_obj = UnifiedDataLoader(seq_len=SEQ_LEN, horizon=H, batch_size=256)
        _, test_loader = loader_obj.get_val_test_loaders_mmap()

        configs = get_config_for_model(args.model, H)
        backbone = model_dict[args.model](configs)
        model = RiskAwareHead(backbone, len(FEATURES), H, CLOSE_IDX, 20, 64).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        sigma = predict_sigma(model, test_loader, device)
        stock_id = loader_obj.test_close_max.astype(np.float64)
        cv_per_t, sigma_panel = compute_cv_panel(sigma, stock_id)
        cv_clean = cv_per_t[~np.isnan(cv_per_t)]
        bar_cv = float(cv_clean.mean()) if len(cv_clean) else float("nan")

        # Pull the existing Sharpe deltas from the summary blobs
        s_mse = os.path.join(OUT_DIR, f"summary_xs_{args.model}_global_H{H}_long_short_mse.json")
        s_ra  = os.path.join(OUT_DIR, f"summary_xs_{args.model}_global_H{H}_long_short_riskhead{'_xs' if args.ckpt_suffix == '_riskhead_xs' else ''}_RA.json")

        sr_mse = sr_ra = float("nan")
        if os.path.exists(s_mse):
            with open(s_mse) as f:
                sr_mse = json.load(f)["cost_sensitivity"]["10.0"]["net_sharpe"]
        if os.path.exists(s_ra):
            with open(s_ra) as f:
                sr_ra = json.load(f)["cost_sensitivity"]["10.0"]["net_sharpe"]
        delta_sr = sr_ra - sr_mse if not (np.isnan(sr_mse) or np.isnan(sr_ra)) else float("nan")

        rows.append({
            "H": H, "n_test_t": int(len(cv_clean)),
            "mean_sigma": float(np.nanmean(sigma_panel)),
            "bar_CV_sigma": bar_cv,
            "MSE_net_SR_at_10bps": sr_mse,
            "TB_RA_net_SR_at_10bps": sr_ra,
            "Delta_SR": delta_sr,
        })
        print(f"  H={H:>3}: bar_CV_sigma={bar_cv:.4f}  "
              f"MSE_SR={sr_mse:6.3f}  TB_RA_SR={sr_ra:6.3f}  "
              f"Delta_SR={delta_sr:+6.3f}")

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))

    if len(df) >= 3 and df["bar_CV_sigma"].notna().all() and df["Delta_SR"].notna().all():
        rho = df[["bar_CV_sigma", "Delta_SR"]].corr().iloc[0, 1]
        from scipy.stats import spearmanr
        sp, _ = spearmanr(df["bar_CV_sigma"], df["Delta_SR"])
        print()
        print(f"Pearson rho(bar_CV_sigma, Delta_SR)  = {rho:+.4f}")
        print(f"Spearman rho(bar_CV_sigma, Delta_SR) = {sp:+.4f}")
        print()
        if rho > 0:
            print("[A1 verified] Sharpe gap is positively correlated with sigma-heterogeneity,")
            print("              consistent with Theorem A1 prediction A1-1.")
        else:
            print("[A1 NOT verified at this scale] Negative correlation; could indicate")
            print("                                small-N noise or a regime mismatch.")

    out_path = os.path.join(OUT_DIR, f"a1_verification_{args.model}_{args.tag}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
