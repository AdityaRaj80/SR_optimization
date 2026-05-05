"""Cross-sectional ranking smoke test — ICAIF-compliant evaluation.

Strategy: at each rebalance timestamp t, rank all hold-out stocks by predicted
H-day return. Long top-N, short bottom-N (or long-only top-N), equal-weight
within each leg. Portfolio return per timestamp is the weighted sum of legs,
net of transaction costs from turnover.

Reports the ICAIF-standard metric set:
  - Portfolio annualized Sharpe (gross AND net of 0/5/10/20 bps round-trip)
  - Portfolio MDD
  - Calmar ratio (annual return / MDD)
  - Sortino ratio (downside-only deviation)
  - Cumulative return over test period
  - Hit rate (fraction of profitable rebalances)
  - Average turnover per rebalance
  - Cross-sectional Spearman IC (rank correlation of predicted vs actual)

Sweep top_n on val by val portfolio Sharpe, apply chosen value to test.

Reads SR_optimization checkpoints read-only; writes only to ./results/.
"""
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

# Find the SR_optimization root regardless of layout:
#   * INSIDE the repo:    SR_optimization/Smoke_test/cross_sectional_smoke.py  (parent has config.py)
#   * SIBLING (legacy):   ../SR_optimization/config.py
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(os.path.dirname(_HERE), "config.py")):
    SR_OPT_DIR = os.path.dirname(_HERE)
else:
    SR_OPT_DIR = os.path.abspath(os.path.join(_HERE, "..", "SR_optimization"))
sys.path.insert(0, SR_OPT_DIR)

from config import SEQ_LEN, CLOSE_IDX, MODEL_SAVE_DIR, FEATURES  # noqa: E402
from data_loader import UnifiedDataLoader               # noqa: E402
from models import model_dict                            # noqa: E402
from train import get_config_for_model                   # noqa: E402
from engine.heads import RiskAwareHead                   # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)


# ─── Forward pass + return space ─────────────────────────────────────────────
def predict(model, loader, device):
    """Forward pass returning price predictions (mu_close), targets, last_close.

    Handles three model output forms:
      * tensor               — legacy MSE-trained backbone
      * tuple (pred, ...)    — AdaPatch-style backbones
      * dict with mu_close   — RiskAwareHead-wrapped backbone (Track B)
    """
    model.eval()
    preds, trues, last_close = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            bx = batch_x.float().to(device)
            out = model(bx, None)
            if isinstance(out, dict):
                out = out["mu_close"]
            elif isinstance(out, tuple):
                out = out[0]
            preds.append(out.cpu().numpy())
            trues.append(batch_y.numpy())
            last_close.append(bx[:, -1, CLOSE_IDX].cpu().numpy())
    return (np.concatenate(preds).astype(np.float64),
            np.concatenate(trues).astype(np.float64),
            np.concatenate(last_close).astype(np.float64))


def predict_with_risk(model, loader, device, gate_temp=0.13,
                      tau_vol=0.0, tau_sigma=0.0, s_vol=1.0, s_sigma=1.0):
    """Track-B-aware forward pass.

    Returns the price prediction + the auxiliary heads' outputs that Track B
    was trained to expose. The gate value is reconstructed using the same
    sigmoid-of-thresholded-vol/sigma logic as in `engine/losses.py`, with
    the *minimum* gate temperature (post-anneal) so the gate is at its
    sharpest near-binary form. Defaults match `CompositeRiskLoss` ctor.

    Returns
    -------
        mu_close      : [N, pred_len]  scaled-space close predictions
        sigma_R       : [N]            return-space sigma (sqrt(exp(log_sigma2_H)))
        log_vol_pred  : [N]
        gate          : [N]            in [0, 1]
        trues         : [N, pred_len]
        last_close    : [N]
    """
    model.eval()
    mu, sig, lvp, ga, trues, lc = [], [], [], [], [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            bx = batch_x.float().to(device)
            out = model(bx, None)
            if not isinstance(out, dict):
                raise ValueError(
                    "predict_with_risk requires the model to return a dict "
                    "(RiskAwareHead). For non-risk-head checkpoints use "
                    "`predict` instead.")
            mu_close = out["mu_close"]
            log_var = out["log_sigma2_H"].clamp(min=-12.0, max=4.0)
            sigma = torch.exp(0.5 * log_var)
            log_vol = out["log_vol_pred"]
            gate_v = torch.sigmoid((tau_vol - log_vol) / (s_vol * gate_temp))
            gate_s = torch.sigmoid((tau_sigma - sigma) / (s_sigma * gate_temp))
            gate = gate_v * gate_s
            mu.append(mu_close.cpu().numpy())
            sig.append(sigma.cpu().numpy())
            lvp.append(log_vol.cpu().numpy())
            ga.append(gate.cpu().numpy())
            trues.append(batch_y.numpy())
            lc.append(bx[:, -1, CLOSE_IDX].cpu().numpy())
    return (np.concatenate(mu).astype(np.float64),
            np.concatenate(sig).astype(np.float64),
            np.concatenate(lvp).astype(np.float64),
            np.concatenate(ga).astype(np.float64),
            np.concatenate(trues).astype(np.float64),
            np.concatenate(lc).astype(np.float64))


def to_returns(preds, trues, last_close_scaled, cmin, cmax):
    scale = cmax - cmin
    pred_H_usd = preds[:, -1] * scale + cmin
    actual_H_usd = trues[:, -1] * scale + cmin
    last_usd = last_close_scaled * scale + cmin
    return ((pred_H_usd - last_usd) / np.maximum(last_usd, 1e-9),
            (actual_H_usd - last_usd) / np.maximum(last_usd, 1e-9))


def build_panel(pred_return, actual_return, stock_id, align="intersect"):
    """Pivot per-sample arrays into [T, N_stocks] panels.

    Two alignment strategies:

    align="union" (OLD, unaligned):
        Each stock's column starts at index 0 with its own samples.
        Different stocks may have very different N — sample-index t
        does NOT correspond to the same calendar day across stocks.
        This causes survivorship-bias-by-construction at large t and
        inflates downstream cross-sectional Sharpe. Kept for backward
        compat / debugging.

    align="intersect" (NEW, default):
        Clip every stock to the COMMON length = min(N_stock_i). All
        stocks share the same temporal axis [0..T_min). Some samples
        (the tails of long-history stocks) are dropped, but the
        cross-sectional ranking at any t is now apples-to-apples.

    Note: this assumes within each stock, the LAST samples are the
    most recent — so we keep the FIRST T_min from each stock to align
    on the common starting timestamp. Per the mmap loader contract
    (samples within a stock are in chronological order), this drops
    extra recent data from long-history stocks.
    """
    unique = np.unique(stock_id)
    per_stock = {}
    for sid in unique:
        m = stock_id == sid
        per_stock[sid] = (pred_return[m], actual_return[m], int(m.sum()))

    if align == "intersect":
        T = min(n for (_, _, n) in per_stock.values())
        P = np.full((T, len(unique)), np.nan)
        A = np.full((T, len(unique)), np.nan)
        for j, sid in enumerate(unique):
            pr, ar, _ = per_stock[sid]
            P[:T, j] = pr[:T]
            A[:T, j] = ar[:T]
    elif align == "union":
        T = max(n for (_, _, n) in per_stock.values())
        P = np.full((T, len(unique)), np.nan)
        A = np.full((T, len(unique)), np.nan)
        for j, sid in enumerate(unique):
            pr, ar, n = per_stock[sid]
            P[:n, j] = pr
            A[:n, j] = ar
    else:
        raise ValueError(f"Unknown align: {align!r}")
    return P, A, unique


# ─── Strategy with positions + turnover + costs ──────────────────────────────
def cross_sectional_positions(pred_M, top_n, mode="long_short",
                              weighting="equal", vol_M=None,
                              magnitude_threshold=0.0):
    """Return positions matrix [T, N].

    weighting:
        "equal"   - +-1/top_n per leg (default).
        "inv_vol" - weights inversely proportional to per-stock realized
                    vol_M[t,j], normalised so each leg sums to +-1.

    magnitude_threshold:
        Filter out trades where |pred_M[t,j]| <= magnitude_threshold.
        Reduces low-conviction noise. Set to 0 to disable.
    """
    T, N = pred_M.shape
    pos = np.zeros((T, N))
    for t in range(T):
        pred_t = pred_M[t]
        valid = ~np.isnan(pred_t)
        if valid.sum() < 2 * top_n:
            continue
        # Optional magnitude filter
        if magnitude_threshold > 0:
            valid = valid & (np.abs(pred_t) > magnitude_threshold)
            if valid.sum() < 2 * top_n:
                continue
        valid_idx = np.where(valid)[0]
        order_valid = valid_idx[np.argsort(pred_t[valid_idx])]
        bot_idx = order_valid[:top_n]
        top_idx = order_valid[-top_n:]

        # Compute weights
        if weighting == "equal":
            long_w = np.ones(top_n) / top_n
            short_w = np.ones(top_n) / top_n
        elif weighting == "inv_vol":
            if vol_M is None:
                raise ValueError("inv_vol weighting requires vol_M")
            v_long = np.maximum(vol_M[t, top_idx], 1e-6)
            v_short = np.maximum(vol_M[t, bot_idx], 1e-6)
            iv_long = 1.0 / v_long
            iv_short = 1.0 / v_short
            long_w = iv_long / iv_long.sum()
            short_w = iv_short / iv_short.sum()
        else:
            raise ValueError(f"Unknown weighting: {weighting!r}")

        if mode == "long_short":
            pos[t, top_idx] = long_w
            pos[t, bot_idx] = -short_w
        elif mode == "long_only":
            pos[t, top_idx] = long_w
        elif mode == "short_only":
            pos[t, bot_idx] = -short_w
    return pos


def cross_sectional_positions_risk_aware(
    pred_M, sigma_M, gate_M, top_n, mode="long_short",
    alpha_pos=5.0, gate_threshold=0.0, eps_sigma=1e-3,
):
    """Risk-aware cross-sectional positions using Track B's full output.

    Steps per timestamp t:
        1. (Optional) Hard-filter samples whose gate < `gate_threshold`. This
           defaults to 0 because training-time `gate_mean` typically settles
           in [0.3, 0.5] — a 0.5 threshold would kill every trade. The gate
           was trained as a *continuous multiplicative weight*, not a binary
           kill-switch (see CompositeRiskLoss.forward: `gate * position *
           realised_return` is what's optimised).
        2. Rank survivors by `mu / sigma` (predicted Sharpe).
        3. Assign Kelly-tanh raw weights `|tanh(alpha_pos * mu / sigma)|`.
        4. Multiply weights by `gate[t,j]` — confident samples retain full
           weight, uncertain samples are scaled down. This is the
           training-time semantics applied at inference time.
        5. Normalise per leg so absolute weights sum to 1 (so portfolio
           gross exposure is comparable across t and across strategies).

    Returns
    -------
        positions : [T, N]  with each row summing in absolute value to <= 2
                            (long_short) or <= 1 (long_only / short_only).
    """
    T, N = pred_M.shape
    pos = np.zeros((T, N))
    for t in range(T):
        mu_t = pred_M[t]
        sig_t = sigma_M[t]
        gate_t = gate_M[t]
        valid = (~np.isnan(mu_t) & ~np.isnan(sig_t) & ~np.isnan(gate_t)
                 & (gate_t >= gate_threshold))
        if valid.sum() < 2 * top_n:
            continue
        valid_idx = np.where(valid)[0]
        # Rank by predicted Sharpe = mu / sigma
        sharpe_pred = mu_t[valid_idx] / np.maximum(sig_t[valid_idx], eps_sigma)
        order = valid_idx[np.argsort(sharpe_pred)]
        bot_idx = order[:top_n]
        top_idx = order[-top_n:]

        # Kelly-style raw weights from |tanh(alpha*mu/sigma)|, then modulated
        # by the (continuous) gate.
        kelly_long = np.abs(np.tanh(
            alpha_pos * mu_t[top_idx] / np.maximum(sig_t[top_idx], eps_sigma)))
        kelly_short = np.abs(np.tanh(
            alpha_pos * mu_t[bot_idx] / np.maximum(sig_t[bot_idx], eps_sigma)))
        kelly_long  = kelly_long  * gate_t[top_idx]
        kelly_short = kelly_short * gate_t[bot_idx]

        # Normalise per leg so sum of absolute weights = 1. If the leg got
        # gated-out (sum ~ 0), do NOT trade that leg this t (zero positions),
        # so a gate-induced kill is reflected as a flat allocation rather than
        # silently re-spreading equal weights.
        if kelly_long.sum() > 1e-9:
            long_w = kelly_long / kelly_long.sum()
        else:
            long_w = np.zeros(top_n)
        if kelly_short.sum() > 1e-9:
            short_w = kelly_short / kelly_short.sum()
        else:
            short_w = np.zeros(top_n)

        if mode == "long_short":
            pos[t, top_idx] = long_w
            pos[t, bot_idx] = -short_w
        elif mode == "long_only":
            pos[t, top_idx] = long_w
        elif mode == "short_only":
            pos[t, bot_idx] = -short_w
    return pos


def realized_vol_panel(loader, n_total, W=20):
    """Build a [T, N_stocks] panel of realized vol on the trailing-W
    log-returns of each sample's input window. Returns aligned to the
    same pivot order used in build_panel."""
    rvols = np.zeros(n_total, dtype=np.float64)
    idx = 0
    for batch_x, _ in loader:
        bx = batch_x.numpy()
        closes = bx[:, -W - 1:, CLOSE_IDX]
        log_ret = np.diff(np.log(np.maximum(closes, 1e-6)), axis=1)
        rvols[idx: idx + bx.shape[0]] = np.std(log_ret, axis=1, ddof=1) * np.sqrt(252.0)
        idx += bx.shape[0]
    return rvols[:idx]


def apply_volz_gate_portfolio(gross_returns, vol_panel, stock_id_pivot,
                                trade_mask_per_stock, k=2.0, L=40):
    """Gate the WHOLE portfolio in vol-spike regimes.

    For each timestamp t, average the per-stock vol-z scores of the stocks
    actually held at t. If avg z > k, kill the entire portfolio's return at t.
    """
    # Compute z per stock-column, same as smoke_volzscore_killswitch's logic
    T, N = vol_panel.shape
    z = np.full((T, N), np.nan)
    for j in range(N):
        v = vol_panel[:, j]
        for i in range(L, len(v)):
            window = v[i - L:i]
            if np.isnan(window).all():
                continue
            mu = np.nanmean(window)
            sd = np.nanstd(window, ddof=1) + 1e-12
            z[i, j] = (v[i] - mu) / sd

    # Portfolio-level z = mean of held-stocks' z at each t
    held = trade_mask_per_stock                                 # [T, N]
    z_held = np.where(held, z, np.nan)
    port_z = np.nanmean(z_held, axis=1)
    kill = np.where(np.isnan(port_z), False, port_z > k)
    gated = gross_returns.copy()
    gated[kill] = 0.0
    return gated, kill


def portfolio_returns_with_costs(pos, actu_M, cost_bps=0.0):
    """Compute net portfolio return per t with linear transaction costs.

    Cost model: at rebalance t, turnover_t = sum_j |pos_t[j] - pos_{t-1}[j]|.
    Net cost per t = (cost_bps / 10000) * turnover_t. Subtracted from gross
    portfolio return.

    Returns
    -------
    gross : [T] gross return per t (NaN where invalid)
    net   : [T] net return per t (NaN where invalid)
    turnover : [T] turnover per t (0 at t=0)
    """
    T, N = pos.shape
    gross = np.full(T, np.nan)
    net = np.full(T, np.nan)
    turnover = np.zeros(T)
    prev = np.zeros(N)
    for t in range(T):
        if np.all(pos[t] == 0):
            prev = pos[t].copy()
            continue
        valid = ~np.isnan(actu_M[t])
        # Gross: dot product of position and actual return
        contribs = pos[t][valid] * actu_M[t][valid]
        gross[t] = float(contribs.sum())
        # Turnover: change from previous position
        to = float(np.sum(np.abs(pos[t] - prev)))
        turnover[t] = to
        # Net = gross - cost_rate * turnover
        net[t] = gross[t] - (cost_bps / 10000.0) * to
        prev = pos[t].copy()
    return gross, net, turnover


# ─── ICAIF metric set ────────────────────────────────────────────────────────
def annualized_sharpe(returns, horizon):
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return float("nan")
    return mu / sd * np.sqrt(252.0 / horizon)


def annualized_sortino(returns, horizon):
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    mu = r.mean()
    downside = r[r < 0]
    if len(downside) == 0:
        return float("inf")
    dd = np.sqrt((downside ** 2).mean())
    if dd < 1e-12:
        return float("nan")
    return mu / dd * np.sqrt(252.0 / horizon)


def cumulative_return(returns):
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan")
    r_clip = np.clip(r, -0.99, None)
    return float(np.expm1(np.sum(np.log1p(r_clip))))


def annualized_return(returns, horizon):
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan")
    mu_per = r.mean()
    return float((1 + mu_per) ** (252.0 / horizon) - 1)


def max_drawdown(returns):
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan")
    r = np.clip(r, -0.99, None)
    log_eq = np.cumsum(np.log1p(r))
    log_peak = np.maximum.accumulate(log_eq)
    return float(1.0 - np.exp(-np.max(log_peak - log_eq)))


def calmar_ratio(returns, horizon):
    ar = annualized_return(returns, horizon)
    md = max_drawdown(returns)
    if np.isnan(ar) or np.isnan(md) or md < 1e-9:
        return float("nan")
    return float(ar / md)


def hit_rate_portfolio(returns):
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan")
    return float((r > 0).mean())


def cross_sectional_ic(pred_M, actu_M):
    """Spearman rank correlation between predicted and actual returns
    cross-sectionally per t. Returns [T] series of IC, plus mean/IR."""
    T, N = pred_M.shape
    ics = np.full(T, np.nan)
    for t in range(T):
        valid = ~np.isnan(pred_M[t]) & ~np.isnan(actu_M[t])
        if valid.sum() < 5:
            continue
        rho, _ = spearmanr(pred_M[t][valid], actu_M[t][valid])
        ics[t] = rho
    ic_clean = ics[~np.isnan(ics)]
    if len(ic_clean) == 0:
        return ics, float("nan"), float("nan")
    ic_mean = float(ic_clean.mean())
    ic_std = float(ic_clean.std(ddof=1))
    icir = ic_mean / ic_std * np.sqrt(252.0) if ic_std > 1e-12 else float("nan")
    return ics, ic_mean, icir


# ─── Wrap up: full metric block per (config, cost) ──────────────────────────
def full_metric_block(returns, horizon, label=""):
    return {
        f"{label}sharpe": float(annualized_sharpe(returns, horizon)),
        f"{label}sortino": float(annualized_sortino(returns, horizon)),
        f"{label}calmar": float(calmar_ratio(returns, horizon)),
        f"{label}mdd": float(max_drawdown(returns)),
        f"{label}cumulative_return": float(cumulative_return(returns)),
        f"{label}annualized_return": float(annualized_return(returns, horizon)),
        f"{label}hit_rate": float(hit_rate_portfolio(returns)),
    }


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="PatchTST")
    p.add_argument("--method", default="global", choices=["global", "sequential"])
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--top_n_grid", type=str, default="3,5,7,10,15")
    p.add_argument("--mode", default="long_short", choices=["long_short", "long_only", "short_only"])
    p.add_argument("--cost_bps_grid", type=str, default="0,5,10,20,50",
                   help="Round-trip transaction cost in bps (per unit of turnover).")
    p.add_argument("--weighting", default="equal", choices=["equal", "inv_vol"])
    p.add_argument("--magnitude_threshold", type=float, default=0.0,
                   help="Filter out trades where |pred_return| <= threshold.")
    p.add_argument("--volz_gate_k", type=float, default=0.0,
                   help="Apply portfolio-level vol-z kill-switch with this k. 0 to disable.")
    p.add_argument("--volz_gate_L", type=int, default=40)
    p.add_argument("--use_risk_head", action="store_true",
                   help="Wrap backbone in RiskAwareHead before loading the checkpoint. "
                        "Required for Track B (_riskhead.pth) checkpoints, which contain "
                        "sigma_head + vol_head weights alongside the backbone.")
    p.add_argument("--ckpt_suffix", type=str, default="",
                   help="Suffix appended to default checkpoint name. Use '_riskhead' for "
                        "Track B final-epoch ckpts, '_riskhead_bestval' for the diagnostic "
                        "best-val-MSE ckpts saved alongside.")
    p.add_argument("--out_tag", type=str, default="",
                   help="Tag appended to output CSV filenames so MSE / Track B / bestval "
                        "results don't overwrite each other.")
    p.add_argument("--strategy", default="simple",
                   choices=["simple", "risk_aware"],
                   help="simple: rank by mu only, equal/inv_vol weighting (works for any model). "
                        "risk_aware: rank by mu/sigma, Kelly-tanh sizing, gate kill-switch — "
                        "uses Track B's full output. Requires --use_risk_head.")
    p.add_argument("--gate_threshold", type=float, default=0.0,
                   help="Hard sample-level filter (risk_aware only): drops "
                        "samples with gate < threshold before ranking. "
                        "Defaults to 0 because the gate is trained as a "
                        "continuous weight (training-time gate_mean ~0.35), "
                        "not a binary kill-switch — set to 0.3-0.5 to test "
                        "kill-switch behaviour explicitly.")
    p.add_argument("--alpha_pos", type=float, default=5.0,
                   help="Kelly tanh sharpness parameter (risk_aware only). "
                        "Default matches CompositeRiskLoss.")
    args = p.parse_args()
    if args.strategy == "risk_aware" and not args.use_risk_head:
        p.error("--strategy risk_aware requires --use_risk_head (the Track B "
                "machinery is what risk-aware sizing uses).")

    top_n_grid = [int(x) for x in args.top_n_grid.split(",")]
    cost_grid = [float(x) for x in args.cost_bps_grid.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | model={args.model} | H={args.horizon} | "
          f"mode={args.mode} | top_n_grid={top_n_grid} | cost_bps_grid={cost_grid}",
          flush=True)

    loader_obj = UnifiedDataLoader(seq_len=SEQ_LEN, horizon=args.horizon,
                                   batch_size=args.batch_size)
    val_loader, test_loader = loader_obj.get_val_test_loaders_mmap()

    configs = get_config_for_model(args.model, args.horizon)
    backbone = model_dict[args.model](configs)
    if args.use_risk_head:
        # Match exactly the construction used in train.py so the state_dict
        # keys align: backbone.<name>, sigma_head.<name>, vol_head.<name>.
        model = RiskAwareHead(
            backbone=backbone,
            n_features=len(FEATURES),
            pred_len=args.horizon,
            close_idx=CLOSE_IDX,
            lookback_for_aux=20,
            d_hidden=64,
        ).to(device)
    else:
        model = backbone.to(device)
    ckpt = os.path.join(
        MODEL_SAVE_DIR,
        f"{args.model}_{args.method}_H{args.horizon}{args.ckpt_suffix}.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Loaded {ckpt} (use_risk_head={args.use_risk_head})", flush=True)

    # ─── VAL forward pass ──
    print("\n--- VAL forward pass ---", flush=True)
    if args.strategy == "risk_aware":
        vp, v_sig, _, v_gate, vt, vlc = predict_with_risk(model, val_loader, device)
    else:
        vp, vt, vlc = predict(model, val_loader, device)
    val_pr, val_ar = to_returns(vp, vt, vlc,
                                  loader_obj.val_close_min.astype(np.float64),
                                  loader_obj.val_close_max.astype(np.float64))
    val_pred_M, val_act_M, _ = build_panel(val_pr, val_ar, loader_obj.val_close_max.astype(np.float64))

    # Build val realized-vol panel (for legacy inv_vol weighting / volz gate)
    val_vol = realized_vol_panel(loader_obj.get_val_test_loaders_mmap()[0],
                                  len(val_pr))
    val_vol_pivot = np.full_like(val_pred_M, np.nan)
    val_stock_id = loader_obj.val_close_max.astype(np.float64)
    val_unique = np.unique(val_stock_id)
    T_val = val_pred_M.shape[0]
    for j, sid in enumerate(val_unique):
        m = val_stock_id == sid
        v_stock = val_vol[m]
        n_keep = min(T_val, len(v_stock))
        val_vol_pivot[:n_keep, j] = v_stock[:n_keep]

    # For risk_aware strategy: pivot sigma + gate per-stock to match val_pred_M's [T, N] panel.
    if args.strategy == "risk_aware":
        val_sig_M = np.full_like(val_pred_M, np.nan)
        val_gate_M = np.full_like(val_pred_M, np.nan)
        for j, sid in enumerate(val_unique):
            m = val_stock_id == sid
            n_keep = min(T_val, int(m.sum()))
            val_sig_M[:n_keep, j]  = v_sig[m][:n_keep]
            val_gate_M[:n_keep, j] = v_gate[m][:n_keep]

    # Sweep top_n on VAL using GROSS Sharpe of the chosen strategy.
    print(f"\n--- Sweep top_n on VAL (strategy={args.strategy}, "
          f"weighting={args.weighting}, magthr={args.magnitude_threshold}) ---",
          flush=True)
    sweep_rows = []
    for n in top_n_grid:
        if args.strategy == "risk_aware":
            pos = cross_sectional_positions_risk_aware(
                val_pred_M, val_sig_M, val_gate_M, n, args.mode,
                alpha_pos=args.alpha_pos, gate_threshold=args.gate_threshold)
        else:
            pos = cross_sectional_positions(
                val_pred_M, n, args.mode,
                weighting=args.weighting, vol_M=val_vol_pivot,
                magnitude_threshold=args.magnitude_threshold)
        gross, _, turnover = portfolio_returns_with_costs(pos, val_act_M, cost_bps=0)
        gross_nover = gross[::args.horizon]
        sweep_rows.append({
            "top_n": n,
            "val_sharpe_gross": float(annualized_sharpe(gross_nover, args.horizon)),
            "val_mdd": float(max_drawdown(gross_nover)),
            "val_avg_turnover": float(np.nanmean(turnover[::args.horizon])),
            "n_valid_t": int((~np.isnan(gross_nover)).sum()),
        })
    sweep_df = pd.DataFrame(sweep_rows)
    print(sweep_df.to_string(index=False))
    # Robust to all-NaN val Sharpe (long horizons with tiny val window): fall
    # back to the median grid value rather than crashing on idxmax.
    if sweep_df["val_sharpe_gross"].isna().all():
        best_n = int(top_n_grid[len(top_n_grid) // 2])
        print(f"\n[warn] All val Sharpes are NaN — falling back to top_n={best_n}", flush=True)
    else:
        best_n = int(sweep_df.iloc[sweep_df["val_sharpe_gross"].idxmax()]["top_n"])
        print(f"\nBest top_n on val: {best_n}", flush=True)

    # ─── TEST forward pass ──
    print("\n--- TEST forward pass ---", flush=True)
    if args.strategy == "risk_aware":
        tp, t_sig, _, t_gate, tt, tlc = predict_with_risk(model, test_loader, device)
    else:
        tp, tt, tlc = predict(model, test_loader, device)
    test_pr, test_ar = to_returns(tp, tt, tlc,
                                   loader_obj.test_close_min.astype(np.float64),
                                   loader_obj.test_close_max.astype(np.float64))
    test_pred_M, test_act_M, _ = build_panel(test_pr, test_ar, loader_obj.test_close_max.astype(np.float64))

    # Build test realized-vol panel
    test_vol = realized_vol_panel(loader_obj.get_val_test_loaders_mmap()[1],
                                   len(test_pr))
    test_vol_pivot = np.full_like(test_pred_M, np.nan)
    test_stock_id = loader_obj.test_close_max.astype(np.float64)
    test_unique = np.unique(test_stock_id)
    T_test = test_pred_M.shape[0]
    for j, sid in enumerate(test_unique):
        m = test_stock_id == sid
        v_stock = test_vol[m]
        n_keep = min(T_test, len(v_stock))
        test_vol_pivot[:n_keep, j] = v_stock[:n_keep]

    if args.strategy == "risk_aware":
        test_sig_M  = np.full_like(test_pred_M, np.nan)
        test_gate_M = np.full_like(test_pred_M, np.nan)
        for j, sid in enumerate(test_unique):
            m = test_stock_id == sid
            n_keep = min(T_test, int(m.sum()))
            test_sig_M[:n_keep, j]  = t_sig[m][:n_keep]
            test_gate_M[:n_keep, j] = t_gate[m][:n_keep]

    # Strategy on test
    if args.strategy == "risk_aware":
        pos = cross_sectional_positions_risk_aware(
            test_pred_M, test_sig_M, test_gate_M, best_n, args.mode,
            alpha_pos=args.alpha_pos, gate_threshold=args.gate_threshold)
    else:
        pos = cross_sectional_positions(
            test_pred_M, best_n, args.mode,
            weighting=args.weighting, vol_M=test_vol_pivot,
            magnitude_threshold=args.magnitude_threshold)

    # Optionally apply portfolio-level vol-z gate (kill-switch)
    apply_volz = args.volz_gate_k > 0
    if apply_volz:
        print(f"\n--- Applying portfolio vol-z gate (k={args.volz_gate_k}, "
              f"L={args.volz_gate_L}) ---", flush=True)

    # Compute returns at every cost level
    print("\n--- TEST: cost sensitivity ---", flush=True)
    cost_table = {}
    for c_bps in cost_grid:
        gross, net, turnover = portfolio_returns_with_costs(pos, test_act_M, cost_bps=c_bps)
        if apply_volz:
            held = pos != 0
            net, _ = apply_volz_gate_portfolio(
                net, test_vol_pivot, test_stock_id, held,
                k=args.volz_gate_k, L=args.volz_gate_L)
        net_nover = net[::args.horizon]
        cost_table[c_bps] = {
            "cost_bps_round_trip": c_bps,
            **full_metric_block(net_nover, args.horizon, label="net_"),
            "avg_turnover_per_rebalance": float(np.nanmean(turnover[::args.horizon])),
        }
        print(f"  cost={c_bps:>5.1f} bps : net_Sharpe={cost_table[c_bps]['net_sharpe']:6.3f}  "
              f"net_Calmar={cost_table[c_bps]['net_calmar']:6.3f}  "
              f"net_cumret={cost_table[c_bps]['net_cumulative_return']:6.3f}  "
              f"MDD={cost_table[c_bps]['net_mdd']:.3f}", flush=True)
    cost_df = pd.DataFrame(list(cost_table.values()))

    # Gross (cost=0) is the headline; also report cross-sectional IC
    gross, _, turnover = portfolio_returns_with_costs(pos, test_act_M, cost_bps=0)
    gross_nover = gross[::args.horizon]
    ic_series, ic_mean, icir = cross_sectional_ic(test_pred_M, test_act_M)

    # Reference baselines on test
    naive_eq = np.nanmean(test_act_M, axis=1)
    naive_eq_nover = naive_eq[::args.horizon]

    # Pointwise per-stock median (legacy convention)
    point_sharpes = []
    for j in range(test_pred_M.shape[1]):
        vp = test_pred_M[:, j]; va = test_act_M[:, j]
        valid = ~np.isnan(vp) & ~np.isnan(va)
        if valid.sum() < 30:
            continue
        s = (np.sign(vp[valid]) * va[valid])[::args.horizon]
        sh = annualized_sharpe(s, args.horizon)
        if not np.isnan(sh):
            point_sharpes.append(sh)
    pointwise_median = float(np.median(point_sharpes)) if point_sharpes else float("nan")

    summary = {
        "model": args.model,
        "method": args.method,
        "horizon": args.horizon,
        "mode": args.mode,
        "strategy": args.strategy,
        "use_risk_head": bool(args.use_risk_head),
        "ckpt_suffix": args.ckpt_suffix,
        "gate_threshold": args.gate_threshold if args.strategy == "risk_aware" else None,
        "alpha_pos":     args.alpha_pos       if args.strategy == "risk_aware" else None,
        "best_top_n": best_n,
        "n_test_timestamps": int((~np.isnan(gross_nover)).sum()),
        "gross_metrics": full_metric_block(gross_nover, args.horizon, label="gross_"),
        "cost_sensitivity": cost_table,
        "cross_sectional_IC_mean": float(ic_mean),
        "cross_sectional_ICIR": float(icir),
        "naive_eqw_sharpe": float(annualized_sharpe(naive_eq_nover, args.horizon)),
        "naive_eqw_mdd": float(max_drawdown(naive_eq_nover)),
        "naive_eqw_cumulative_return": float(cumulative_return(naive_eq_nover)),
        "pointwise_per_stock_sharpe_median": pointwise_median,
    }

    print("\n========== ICAIF-COMPLIANT SUMMARY ==========", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    out_tag_suffix = f"_{args.out_tag}" if args.out_tag else ""
    tag = f"xs_{args.model}_{args.method}_H{args.horizon}_{args.mode}{out_tag_suffix}"
    sweep_df.to_csv(os.path.join(OUT_DIR, f"sweep_{tag}.csv"), index=False)
    cost_df.to_csv(os.path.join(OUT_DIR, f"costs_{tag}.csv"), index=False)
    pd.DataFrame({
        "portfolio_return_nonoverlap": gross_nover,
        "naive_eqw_return_nonoverlap": naive_eq_nover,
    }).to_csv(os.path.join(OUT_DIR, f"timeseries_{tag}.csv"), index=False)
    with open(os.path.join(OUT_DIR, f"summary_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
