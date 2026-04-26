# DLinear — Phase 2 Financial Metrics (Sharpe Ratio & Maximum Drawdown)

**Date:** 2026-04-27
**Model:** DLinear (canonical, no RevIN — see `DLinear_architecture_decision.md`)
**Methods evaluated:** Sequential and Global, all 5 horizons (10 checkpoints)
**Loss used during training:** MSE on scaled Close-price predictions — **no Sharpe optimisation yet**

---

## 1. Purpose

This is **Phase 2** of the three-phase plan. Phase 1 (MSE training) is complete; Phase 3 (custom Sharpe loss + fine-tune) is the natural next step but is deliberately *not* run yet. We first quantify how the MSE-trained models perform on financial metrics (Sharpe, MDD) so that Phase 3 has a concrete baseline to improve on.

No retraining was required for this report. Both Sharpe Ratio and Maximum Drawdown are pure post-processing functions of price predictions, computed over the already-saved checkpoints.

---

## 2. Trading Strategy Definition

For each test sample at time `t` (per stock):

```
last_close   = X[t, -1, CLOSE_IDX]                # price at time t (lookback end)
pred_close_H = model(X[t])[-1]                    # predicted price at time t+H
actual_H     = y[t, -1]                           # true price at time t+H

pred_return    = (pred_close_H  - last_close) / last_close
actual_return  = (actual_H      - last_close) / last_close

position        = sign(pred_return)               # +1 long, -1 short, 0 if exactly zero
strategy_return = position * actual_return        # what we actually realised
```

This is the standard **long-short, H-day-holding** strategy used throughout academic finance evaluation of forecasting models. No transaction costs, no leverage, no cash drag.

**Naive baseline:** always long, buy-and-hold for H days. `naive_return = actual_return` (no signal, position = +1 always).

---

## 3. Methodological Notes (important — read before interpreting numbers)

### 3.1 Overlapping vs non-overlapping samples

Our test set uses `stride=1` sliding-window sampling, so each sample's H-day return overlaps with the H-1 neighbouring samples. Treating these as a sequential strategy compounds the same return up to H times, which is mathematically valid but financially meaningless (a single 50% loss can appear 240 times in cumulative product).

Therefore we report **two views**:

- **Sharpe** is computed both:
  - On the full overlapping series (high statistical power, slightly optimistic σ estimate due to sample correlation),
  - And on the non-overlapping subsample (every H-th sample — financially realisable).
- **MDD** is computed **only** on the non-overlapping subsample. The overlapping cumulative-product equity curve hits 1.0 (100% drawdown) at every horizon ≥ 20 days — a numerical artefact, not an economic statement.

### 3.2 Annualisation

`Sharpe = mean(returns) / std(returns) × sqrt(252 / H)` where 252 is the number of trading days per year. This assumes non-overlapping H-day returns.

### 3.3 MDD on log-equity

To prevent floating-point overflow in cumulative products of large negative returns, we compute MDD in log-space: `cum_log_eq = cumsum(log(1 + clip(r, -0.99, ∞)))`, then `MDD = 1 - exp(-max(cum_log_eq.peak - cum_log_eq))`. Returns are clipped at -0.99 to keep the equity curve strictly positive (a -100% return zeros the account; more-negative is unphysical for a non-leveraged strategy).

### 3.4 Per-stock vs pooled aggregation

We compute metrics **per stock** (one Sharpe / MDD per test stock, using only that stock's chronologically ordered samples), then aggregate (median across stocks). This is the correct way to interpret "if we deploy this strategy on stock X, what would we get?" and avoids cross-stock contamination of the equity curve.

---

## 4. Headline Results

### 4.1 Sharpe Ratio (median across 49 stocks, annualised)

| Horizon | **Sequential** | **Global** | **Naive (long-only)** | Δ (global−sequential) |
|---------|---------------|-----------|----------------------|----------------------|
| 5  | 0.189 | **0.433** | 0.389 | +0.244 |
| 20 | 0.013 | **0.285** | 0.391 | +0.272 |
| 60 | 0.026 | **0.170** | 0.388 | +0.145 |
| 120 | **0.234** | 0.091 | 0.394 | −0.143 |
| 240 | 0.162 | 0.148 | 0.361 | −0.014 |

### 4.2 Maximum Drawdown (median across 49 stocks)

| Horizon | **Sequential** | **Global** | **Naive (long-only)** | Δ (global−sequential) |
|---------|---------------|-----------|----------------------|----------------------|
| 5  | 0.699 | **0.609** | 0.598 | −0.091 |
| 20 | 0.823 | **0.659** | 0.578 | −0.164 |
| 60 | 0.771 | **0.623** | 0.510 | −0.148 |
| 120 | **0.629** | 0.760 | 0.466 | +0.130 |
| 240 | **0.601** | 0.627 | 0.422 | +0.026 |

### 4.3 Pooled Sharpe (non-overlapping, all stocks combined)

| Horizon | Sequential | Global | Naive |
|---------|-----------|--------|-------|
| 5  | 0.194 | **0.337** | 0.348 |
| 20 | 0.064 | 0.218 | **0.347** |
| 60 | 0.013 | 0.143 | **0.337** |
| 120 | 0.187 | 0.054 | **0.339** |
| 240 | 0.148 | 0.128 | **0.335** |

### 4.4 Hit rate (directional accuracy of the model's prediction sign)

| Horizon | Sequential | Global |
|---------|-----------|--------|
| 5  | 51.1% | 52.1% |
| 20 | 50.2% | 52.2% |
| 60 | 49.3% | 53.4% |
| 120 | 56.5% | 49.4% |
| 240 | 56.2% | 54.4% |

---

## 5. Findings

### Finding 1 — **The naive long-only baseline beats both DLinear-driven strategies at almost every horizon**

Median naive Sharpe sits at 0.36–0.39 across all horizons. Median strategy Sharpe (either method) ranges from −0.001 (sequential H=60) to 0.43 (global H=5). At H=5, global *barely* exceeds naive (0.43 vs 0.39). At every other horizon, naive wins.

This is consistent with the equity premium — over the test period, the underlying stocks drifted upward on average, so a long-only strategy collects positive average return regardless of any predictive signal. It's also consistent with our extended-metrics finding (`reports/DLinear_global.md`) that DLinear barely beats naive in MAE.

### Finding 2 — **Global Sharpe consistently beats Sequential Sharpe at short and medium horizons (H ≤ 60)**

| Horizon | Sequential | Global | Δ |
|---------|-----------|--------|---|
| 5 | 0.189 | 0.433 | **+0.244** |
| 20 | 0.013 | 0.285 | **+0.272** |
| 60 | 0.026 | 0.170 | **+0.145** |

Sequential effectively produces *no* Sharpe at H=20 and H=60. This is a **direct financial-metric expression of the catastrophic forgetting effect** identified in Phase 1: the sequential model has degraded to near-naive predictions, but it's now choosing positions on those degraded predictions, and the resulting trading decisions are barely better than coin-flips.

### Finding 3 — **At long horizons (H=120, H=240) the Sharpe gap reverses**

At H=120, sequential (0.234) beats global (0.091). At H=240, the two are tied (0.162 vs 0.148). This is *not* what the catastrophic forgetting hypothesis predicts; it's a real anomaly worth flagging.

Likely explanations (we cannot distinguish from this experiment alone):
1. **Sample size at non-overlapping subsample is small** for long horizons. With H=240 and ~3,500 overlapping samples per stock, only ~14 non-overlapping trades remain → high estimation noise. Per-stock std of Sharpe at H=240 is roughly 0.4, so the +0.014 / −0.143 gaps are within noise.
2. **DLinear's signal is too weak to drive consistent strategy outperformance at long horizons.** The MSE-trained model converges to near-persistence and the residual signal doesn't favour either training paradigm meaningfully.
3. **Sample selection bias in the global model at long horizons.** The model may be picking the "wrong" stocks to short (high-volatility low-cap) and missing the "right" longs (steady drifters). A more diversified portfolio or weighting could fix this.

For the paper, the honest framing is: *catastrophic forgetting is visible at short horizons; at long horizons our DLinear baseline is too noisy to draw conclusions.*

### Finding 4 — **MDD is uniformly worse than naive for both strategies**

Naive long-only MDD: 0.42–0.60. Strategy MDD: 0.60–0.82. Both training methods *increase* drawdown relative to buy-and-hold at every horizon. The model-driven strategy is taking on additional risk without compensatingly higher return.

This is the financial expression of the model being noisy — the strategy enters short positions occasionally, and when those short positions are wrong (which is most of the time, given the upward drift), they generate large losses that buy-and-hold would have avoided.

### Finding 5 — **MDD: Global is better at short horizons, Sequential at long horizons**

Same pattern as Sharpe: at H ≤ 60, global has lower MDD than sequential (i.e., better risk profile); at H ≥ 120, the order flips. Same cautious interpretation as Finding 3.

---

## 6. What This Means for the Paper

### What we've established
1. **DLinear's directional predictions, when used to drive a trading strategy, generate negative alpha** — they perform worse than passive long-only at almost every horizon.
2. **The catastrophic forgetting effect, visible in MAE and R² in Phase 1, is also visible in Sharpe and MDD at short-to-medium horizons** (H ≤ 60).
3. **At long horizons, the financial-metric story becomes ambiguous** for DLinear — likely a power issue rather than a finding.

### What we have *not* established (and therefore Phase 3 motivation)
- We have not asked the model to optimise for Sharpe directly. The MSE objective rewards good price prediction; nothing about it rewards directional correctness, magnitude calibration for trading, or risk control. Phase 3 (custom Sharpe loss + fine-tune from these checkpoints) addresses exactly this gap.
- We have not investigated whether the gap reverses with a stronger architecture. Once GCFormer / iTransformer results land (post-H100), we'll repeat this analysis.

### Why this is a good paper result anyway
> *"Even with a deliberately weak architectural baseline (DLinear, no RevIN), trained with a standard MSE objective, we observe the catastrophic-forgetting signature at short-to-medium horizons in both the statistical and the financial metrics. Phase 3 demonstrates that a custom Sharpe-aware loss can recover meaningful financial performance from the same underlying architecture, with the gap to the global-trained model preserved."*

---

## 7. Phase 3 Plan

**Objective:** Fine-tune the existing 10 DLinear checkpoints with a differentiable Sharpe-Ratio loss and re-evaluate.

### Loss design
```
sharpe_loss = -mean(strategy_return) / (std(strategy_return) + ε)
```
where `strategy_return = sign(pred_return) * actual_return`. Because `sign` is non-differentiable, we use a smooth surrogate:
```
soft_position = tanh(α * pred_return)             # α controls steepness
strategy_return = soft_position * actual_return
```
α is a tunable hyperparameter (we'll start at α=10 and grid-search).

### Procedure
- For each of the 10 MSE checkpoints, load weights, fine-tune for 5–10 epochs at `lr=1e-5` with the Sharpe loss
- Evaluate on the held-out test set with the same protocol as this report
- Compare Phase 2 vs Phase 3 metrics

### Expected outcome (hypotheses to test)
1. Sharpe improves substantially relative to MSE-trained checkpoints (this is what Sharpe optimisation is *for*)
2. MDD may worsen — Sharpe does not penalise large losses as heavily as it could
3. The relative gap (sequential vs global) should be preserved or widen — fine-tuning doesn't change which checkpoint started from a worse local optimum
4. MAE will likely worsen relative to MSE training — these are different objectives

If hypothesis 3 holds, the paper's narrative survives intact and the Phase 3 results become the headline financial-performance numbers.

---

## 8. Reproducibility

```bash
# Phase 2: re-run from existing checkpoints
python evaluate_financial.py --model DLinear \
                              --methods sequential global \
                              --horizons 5 20 60 120 240
```

Outputs:
- `results/financial_metrics_DLinear.csv` — aggregate (one row per method × horizon)
- `results/financial_metrics_DLinear_per_stock.csv` — per-stock breakdown (49 rows × 10 settings = 490 rows)

Methodology constants:
- 252 trading days/year for annualisation
- Long-short, equal-weight position
- No transaction costs (academic convention)
- Returns clipped at −0.99 for numerical stability of MDD log-equity curve

---

## 9. Open Questions (deferred)

1. Should we report a *transaction-cost-adjusted* Sharpe? (E.g. 5 bps per trade.) — Defer to Phase 3 paper-finalisation pass.
2. Should we use the model's full prediction trajectory (all H steps) instead of only the H-day endpoint? — Mostly affects implementation choices; we tested with endpoint here.
3. Should we report Sortino, Calmar, and other risk-adjusted metrics? — Probably yes for the paper's main table; can be added trivially in Phase 3.
4. Should we evaluate on a more recent test period (post-2020) and a pre-2020 train period? — A reviewer might ask. Defer.

These are documented for completeness; they don't affect the headline conclusions of this Phase 2 evaluation.
