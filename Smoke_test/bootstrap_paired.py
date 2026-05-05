"""Paired stationary-bootstrap CI on the Sharpe-ratio DIFFERENCE between two
return series produced by `cross_sectional_smoke.py`.

The standard Politis-Romano stationary bootstrap (Politis & Romano, 1994,
*JASA* 89(428): 1303-1313) preserves the serial-correlation structure of
each series by resampling geometrically-distributed contiguous blocks.
For PAIRED inference we draw the SAME block-index sequence from both
series so the per-period covariance between the two strategies is
preserved across each bootstrap replication. This is the standard
construction recommended for Sharpe-ratio difference testing in time
series — see Ledoit & Wolf (2008, *Journal of Empirical Finance* 15:
850-859, "Robust performance hypothesis testing with the Sharpe ratio").

Reports:
  * Point Sharpe of A and B (each on the original series).
  * 95% bootstrap CI of A_Sharpe and B_Sharpe (marginal — sanity check).
  * 95% bootstrap CI of (A_Sharpe - B_Sharpe).
  * One-sided bootstrap p-value for H0: A_Sharpe <= B_Sharpe
    (i.e. fraction of bootstrap reps where A_Sharpe - B_Sharpe <= 0).

Usage:
    python bootstrap_paired.py --csv_a TBriskaware.csv --csv_b MSE.csv \
        --column portfolio_return_nonoverlap --horizon 5 --label_a "TrackB+RA" --label_b "MSE"
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np
import pandas as pd


def stationary_bootstrap_indices(n: int, expected_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap index draw. Each restart probability
    is 1/expected_block, giving geometric block lengths with mean expected_block."""
    indices = np.zeros(n, dtype=int)
    indices[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < 1.0 / expected_block:
            indices[i] = rng.integers(0, n)
        else:
            indices[i] = (indices[i - 1] + 1) % n
    return indices


def annualized_sharpe(returns: np.ndarray, horizon: int) -> float:
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return float("nan")
    return float(mu / sd * np.sqrt(252.0 / horizon))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_a", required=True, help="CSV for arm A (e.g. Track B + risk_aware).")
    p.add_argument("--csv_b", required=True, help="CSV for arm B (e.g. MSE baseline + simple).")
    p.add_argument("--label_a", default="A")
    p.add_argument("--label_b", default="B")
    p.add_argument("--column", default="portfolio_return_nonoverlap")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--expected_block", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    a_df = pd.read_csv(args.csv_a)
    b_df = pd.read_csv(args.csv_b)
    a = a_df[args.column].dropna().to_numpy()
    b = b_df[args.column].dropna().to_numpy()

    # Truncate to the common length so block draws line up. Both arms share
    # the same calendar test window in our pipeline so this is a no-op when
    # the two strategies use the same N stocks; otherwise we truncate to the
    # shorter arm's history so the paired draw is well-defined.
    n = min(len(a), len(b))
    if n < 4:
        raise SystemExit(f"Not enough samples for paired bootstrap (n={n}).")
    a, b = a[:n], b[:n]

    point_a = annualized_sharpe(a, args.horizon)
    point_b = annualized_sharpe(b, args.horizon)
    point_d = point_a - point_b

    boot_a = np.zeros(args.n_boot)
    boot_b = np.zeros(args.n_boot)
    for k in range(args.n_boot):
        idx = stationary_bootstrap_indices(n, args.expected_block, rng)
        boot_a[k] = annualized_sharpe(a[idx], args.horizon)
        boot_b[k] = annualized_sharpe(b[idx], args.horizon)
    boot_d = boot_a - boot_b

    # Drop any NaN reps (zero std on a degenerate resample, very rare).
    valid = ~np.isnan(boot_a) & ~np.isnan(boot_b)
    boot_a, boot_b, boot_d = boot_a[valid], boot_b[valid], boot_d[valid]

    def pct(x): return np.percentile(x, [2.5, 50, 97.5])
    a_lo, a_md, a_hi = pct(boot_a)
    b_lo, b_md, b_hi = pct(boot_b)
    d_lo, d_md, d_hi = pct(boot_d)
    p_one_sided = float((boot_d <= 0).mean())   # H0: A <= B (one-sided)

    out = {
        "label_a": args.label_a, "label_b": args.label_b,
        "csv_a": os.path.basename(args.csv_a),
        "csv_b": os.path.basename(args.csv_b),
        "n_returns": int(n),
        "horizon": args.horizon,
        "n_boot": int(args.n_boot),
        "expected_block": args.expected_block,
        "point_sharpe_a": point_a,
        "point_sharpe_b": point_b,
        "point_sharpe_diff": point_d,
        "ci95_a":    {"lo": float(a_lo), "median": float(a_md), "hi": float(a_hi)},
        "ci95_b":    {"lo": float(b_lo), "median": float(b_md), "hi": float(b_hi)},
        "ci95_diff": {"lo": float(d_lo), "median": float(d_md), "hi": float(d_hi)},
        "p_one_sided_a_le_b": p_one_sided,
    }

    print(f"  {args.label_a:<14} point Sharpe = {point_a:6.3f}   95% CI [{a_lo:6.3f}, {a_hi:6.3f}]")
    print(f"  {args.label_b:<14} point Sharpe = {point_b:6.3f}   95% CI [{b_lo:6.3f}, {b_hi:6.3f}]")
    print(f"  diff (A - B)   point        = {point_d:+6.3f}   95% CI [{d_lo:+6.3f}, {d_hi:+6.3f}]")
    print(f"  p(A <= B)      = {p_one_sided:.4f}   "
          f"({'SIGNIFICANT' if p_one_sided < 0.05 else 'not significant'} at α=0.05)")
    print()

    base_a = os.path.basename(args.csv_a).replace(".csv", "").replace("timeseries_", "")
    base_b = os.path.basename(args.csv_b).replace(".csv", "").replace("timeseries_", "")
    out_dir = os.path.dirname(args.csv_a)
    out_path = os.path.join(out_dir, f"paired_bootstrap_{base_a}_vs_{base_b}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
