# Sharpe-Optimized Loss & Regime Kill-Switch — Refined Proposal

**Project:** `SR_optimization` (8-model benchmark)
**Target venue:** **ICAIF (deadline 2 Aug 2026)** — primary; **IEEE Big Data (22 Aug)** as fallback. ICDM (6 Jun) is too tight given scope.
**Last updated:** 2026-05-01
**Author note:** This file is a refinement of `design.md` (in `~/Downloads/`), incorporating the critique that flagged six structural concerns and a scope/timeline mismatch. Material below explicitly addresses each.

---

## 0. What changed vs `design.md`

| Concern in original | Resolution here |
|---|---|
| σ might be calibrated for residuals but uncorrelated with trade quality | Added **auxiliary BCE on gate vs. trade outcome** (§4.4) so gate gradient receives direct profitability signal, not just residual signal |
| Coefficient grid bigger than acknowledged | Explicit per-coefficient policy (§4.6): which are constant across all (model, horizon), which are scheduled, which are horizon-tuned |
| Soft-gate (training) → hard-gate (inference) mismatch | Added **gate-temperature annealing** (§4.5) so the gate becomes near-binary by end of training, closing the train/test gap |
| "Predict returns instead of prices" was a hidden breaking change | **Stay in price space.** Don't redo the data pipeline. All existing 60+ checkpoints remain valid (§5.1) |
| 3-class regime CE throws away signal | Replaced with **regression on `log(realized_vol)`** with MSE loss; gate uses sigmoid threshold on the predicted vol (§5.3) |
| No baseline guardrail | Added **Track A post-hoc validation** (§7) before any retraining, plus an explicit "kill paper" criterion |
| Position sizing buried in §7 | Promoted to main design: `position = tanh(α · μ / σ)` Kelly-style (§4.2) |
| Walk-forward CV optional | **Required** for ICAIF; built into Phase E (§8) |
| Sentiment / deep ensembles / return-space migration | **Cut.** Out of scope for this paper |

---

## 1. Repo state we're building on (unchanged from design.md §1)

Three facts from current code:
1. **Training loss is `nn.MSELoss()`** in `engine/trainer.py`. AdaPatch has a small reconstruction add-on. No financial signal in gradient.
2. **Sharpe is computed strictly post-hoc** in `evaluate_financial.py` with `position = sign(pred_return)`, hard long/short, no flat state.
3. **No regime classifier, uncertainty estimator, or kill-switch anywhere.** Existing DLinear/iT/GC reports show the naive long-only baseline beats model-driven strategies at almost every horizon, with model-driven MDD worse than naive. MSE is winning the wrong race.

So the work below is net-new design.

---

## 2. Headline result we already have (anchor for the new work)

The **ungated baseline** that the new design must improve on:

| Model | H | Method | **Strategy Sharpe** | Naive Sharpe |
|-------|---|--------|---------------------|--------------|
| PatchTST | 5 | global | **1.084** ⭐ | 0.389 |
| GCFormer | 5 | global | **1.021** | 0.389 |
| iTransformer | 5 | global | 0.671 | 0.389 |
| DLinear | 5 | global | 0.433 | 0.389 |

PatchTST and GCFormer at H=5 global are already comfortably above the naive baseline — proving that *some* deep models can extract genuine alpha at short horizon. **Everything else (longer horizons, sequential training) underperforms naive.** The new design's job is to lift Sharpe at H ≥ 20 above naive, primarily by reducing MDD and gating out unprofitable trades.

---

## 3. Design constraints (unchanged from design.md §2)

The composite loss must be:

- **Differentiable end-to-end** — kill decision expressed as a soft gate, not a hard `if`
- **Stable on minibatches** — explicit variance floor (`std + ε`, ε ≈ 1e-3 in return space)
- **Finite under degenerate regimes** — graceful collapse (fall back to MSE anchor) when gate fires for whole batch
- **Aligned with annualized Sharpe** — compute on per-stock subsequences, not concatenated batches
- **Compatible with all 8 architectures** — uniform `(μ, log_σ², log_vol_pred)` head wraps the backbone
- **Anchored against trivial solutions** — MSE and turnover penalty kill the "always flat" attractor

---

## 4. The recommended composite loss (refined)

### 4.1 The full expression

```
L_total =   α · L_SR_gated            ← gated differentiable Sharpe
          + β · L_NLL                 ← heteroscedastic negative log-likelihood
          + γ · L_MSE_R               ← anchor on return-MSE (small magnitude)
          + δ · L_VOL                 ← MSE on log realized vol (regime head)
          + η · L_GATE_BCE            ← BCE: gate vs. realized profitability  ⬅ NEW
          + λ_to · turnover_penalty   ← discourage whipsaw
          + λ_dd · drawdown_penalty   ← only enabled if MDD > target
```

The new term `η · L_GATE_BCE` is what closes the σ ↔ trade-quality gap that was the biggest hole in `design.md`.

### 4.2 The gated trading components

```python
position_i      = tanh(α_pos · μ_pred_i / (σ_pred_i + ε_σ))    # Kelly-like, σ-scaled
gate_i          = sigmoid((τ_vol - log_vol_pred_i) / s_vol)    # high vol → gate→0
                  * sigmoid((τ_σ - σ_pred_i) / s_σ)            # high uncertainty → gate→0
strat_return_i  = gate_i · position_i · true_return_i
L_SR_gated      = - mean(strat_return_per_stock) /
                  ( std(strat_return_per_stock) + ε_sharpe )
```

Two changes from `design.md`:

- **Position is σ-scaled**: `tanh(α_pos · μ / σ)` instead of `tanh(α_pos · μ)`. This is the "free win" mentioned but not adopted in the original — once σ is calibrated, scaling by it yields Kelly-like sizing automatically.
- **Gate uses regressed `log_vol_pred` instead of 3-class softmax**: more signal, smoother gradient, single hyperparameter (`τ_vol`) instead of 3 class boundaries.

### 4.3 The auxiliary supervision terms

```python
L_NLL  = mean( 0.5 · log(σ_pred² + ε) + 0.5 · (true_return - μ_pred)² / (σ_pred² + ε) )

L_MSE_R = mean( (μ_pred - true_return)² )

L_VOL  = mean( (log_vol_pred - log_realized_vol_target)² )
         where log_realized_vol_target = log(std(true_returns over t-W : t) + ε)
                                       computed offline in preprocess
```

### 4.4 The new gate-vs-profitability term (closes the critique gap)

The single biggest concern about `design.md` was that NLL only calibrates σ against `|residual|`, but the kill-switch needs σ to predict **trade unprofitability**, which is a different thing. A high-magnitude residual that goes the same way as the prediction is profitable; same magnitude in the wrong direction is unprofitable.

**Fix:** add a direct supervision signal on the gate.

```python
profitable_i = (position_i · true_return_i > 0).float()         # 1 if trade made money
L_GATE_BCE   = mean( BCE(gate_i, profitable_i) )                # gate should ≈ p(profit)
```

This term creates a direct gradient: when the gate was ~1 on a sample where the trade lost money, gradient pushes the gate toward 0; when the gate was ~0 on a sample where the trade would have made money, gradient pushes it toward 1. Through the gate's structure (`sigmoid((τ_vol - log_vol_pred)/s_vol) · sigmoid((τ_σ - σ_pred)/s_σ)`), this back-propagates into the regime and uncertainty heads, training them on the *right* signal — predicted profitability — not just residual magnitude.

Coefficient `η` should be small (0.1) — this is a regularizer pushing σ and vol-pred toward profitability-relevance, not a primary objective.

### 4.5 Gate temperature annealing (closes train→test gap)

```python
T_epoch = max(0.1, 1.0 · 0.92^epoch)            # cooling schedule
gate_i  = sigmoid((τ_vol - log_vol_pred_i) / (s_vol · T_epoch))
        · sigmoid((τ_σ - σ_pred_i)         / (s_σ   · T_epoch))
```

At epoch 1, T=1.0, gate is smooth (gradient flows through marginal regimes). By epoch 30, T≈0.13, gate is near-binary (matches the inference-time hard rule). Eliminates the soft→hard mismatch flagged in the critique.

### 4.6 Coefficient policy (made explicit per critique)

| Coefficient | Status | Value | Notes |
|-------------|--------|-------|-------|
| α (gated Sharpe) | **Scheduled** | 0.0 → 0.3 → 0.7 | Phase 1 → 2 → 3 |
| β (NLL) | **Constant** | 0.5 | Calibration term, model-invariant |
| γ (MSE return anchor) | **Scheduled** | 1.0 → 0.5 → 0.2 | Anneal as α grows |
| δ (vol regression) | **Constant** | 0.3 | Cheap supervised signal |
| η (gate BCE) | **Constant** | 0.1 | Regularizer on profitability |
| λ_to (turnover) | **Horizon-tuned** | {0, 0.01, 0.05} | Off in Phase 1; values per horizon |
| λ_dd (drawdown) | **Conditional** | 0 unless MDD > 0.4 | Penalty only triggers above target |

Constant coefficients (β, δ, η) do not need re-tuning across (model, horizon). Scheduled coefficients (α, γ) follow the same schedule for every model. Only λ_to (and conditionally λ_dd) need per-horizon values — at most 5 numbers (one per horizon) to tune, not 5 × 8 × 2 = 80.

### 4.7 Why this composite is the right choice

Same process-of-elimination as `design.md` §4.1, but with the σ↔profitability gap closed:

- Pure MSE → no Sharpe / regime / uncertainty signal
- Pure Sharpe → degenerate flat optimum, no calibration
- Sharpe + MSE → no regime / uncertainty
- + NLL → calibrated σ but only against residuals, not profit
- + Vol regression + Gate BCE → profit-aware uncertainty, regime-aware trading
- + Turnover & MDD penalties → realistic deployment metrics

Only the full composite supervises every head with the right gradient.

---

## 5. The shared `RiskAwareHead` (refinement of design.md §5)

### 5.1 Stay in price space — don't break existing checkpoints

`design.md` §7 suggested migrating to log-return space. **Rejected.** That breaks every checkpoint we've trained over the last 4 days (DLinear/iT/GC/PT/AP — ~50 checkpoints, hundreds of GPU-hours).

The new head computes pred_return = (pred_close - last_close) / last_close internally, so the loss sees returns even though the model emits prices. No data pipeline change.

### 5.2 Head architecture (same as design.md, refined)

```python
class RiskAwareHead(nn.Module):
    def __init__(self, d_model, pred_len):
        super().__init__()
        self.mu_head      = nn.Linear(d_model, pred_len)        # price prediction
        self.log_sigma_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, pred_len)
        )
        self.log_vol_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1)                         # one vol estimate per sample
        )

    def forward(self, h, last_close):
        mu_price = self.mu_head(h)
        mu_return = (mu_price - last_close) / (last_close + 1e-6)
        log_sigma2 = self.log_sigma_head(h)
        log_vol = self.log_vol_head(h).squeeze(-1)
        return mu_price, mu_return, log_sigma2, log_vol
```

### 5.3 Vol regression target (replaces 3-class CE from design.md)

```python
# preprocess once, cache as .npy alongside scaled features
def compute_log_vol_target(close_window, W=20):
    log_returns = np.diff(np.log(close_window[-(W+1):]))
    return np.log(np.std(log_returns) * np.sqrt(252) + 1e-6)
```

Continuous target, MSE loss. No bucketing, no within-stock percentiles needed for the supervised signal (though percentiles are still used to set the inference-time threshold τ_vol).

### 5.4 Per-model hookup (unchanged from design.md §5.2)

Each model's terminal linear becomes `RiskAwareHead`. Backbones untouched. AdaPatch keeps reconstruction path; the prediction branch routes through the new head.

---

## 6. Inference-time kill-switch (refined)

### 6.1 Hard rule with annealed-soft-gate alignment

```python
# Inference, after the soft-gate temperature has cooled to ~0.13
trade_i  = (log_vol_pred_i < τ_vol) AND (σ_pred_i < τ_σ)
position_i = tanh(α_pos · μ_pred_i / σ_pred_i) if trade_i else 0
```

Because the gate during training cooled to near-binary, the soft→hard switch is now a small perturbation (cliff at temperature 0 vs. T=0.13), not a model-class change.

### 6.2 Threshold calibration

τ_vol and τ_σ are calibrated **once on validation** to maximize val Sharpe, then frozen. Identical procedure across all 8 models. Typically lands at 70–85th percentile of σ and 80th percentile of log-vol.

### 6.3 Position sizing payoff

Because position uses `μ / σ` (not just μ), confident predictions trade larger and uncertain ones trade smaller. Combined with the kill-switch (which zeros out the most uncertain ~15–30%), this gives a 3-tier exposure policy:

- High vol or high σ → 0 (kill)
- Mid σ, normal regime → small position via tanh saturation
- Low σ, low vol → large position via tanh saturation

This is what makes the design financially defensible.

---

## 7. Track A: post-hoc smoke test (do FIRST, before retraining)

This is the most important addition vs. `design.md`. **Don't retrain anything until this is validated.**

### 7.1 The hypothesis to test

> "Applying a kill-switch at inference time to existing MSE-trained models improves Sharpe over the ungated baseline."

If false, the entire Track B retraining is wasted GPU time. If true, the retraining adds incremental value but the core mechanism is already proven.

### 7.2 What "post-hoc" means

We do **not** retrain the models. We **do** add a kill-switch on top of fixed checkpoints:

1. Load existing `PatchTST_global_H5.pth` (best Sharpe baseline: 1.084)
2. For each test sample:
   - Forward pass → `μ_pred` (existing model output)
   - Compute σ_proxy via **MC-Dropout** (30 forward passes with dropout enabled, take std of predictions)
   - Compute regime via **realized vol on input window** (already in our data)
3. Apply hard kill-switch: `trade = (regime != HIGH) AND (σ_proxy < τ_σ)`
4. Compare gated Sharpe vs. ungated 1.084
5. Repeat on `GCFormer_global_H5.pth` (Sharpe 1.021)

### 7.3 Pass/fail criteria

| Outcome | Implication |
|---------|-------------|
| Gated Sharpe > Ungated by ≥ 0.1 | ✅ Hypothesis validated. Track B retraining will likely improve this further. |
| Gated Sharpe ≈ Ungated (±0.05) | ⚠️ MC-dropout σ isn't predictive of trade quality. Need calibrated σ from retraining (Track B), but with the gate-BCE term to make σ profit-aware. |
| Gated Sharpe < Ungated by ≥ 0.1 | ❌ Kill-switch hypothesis broken. Either σ proxy is uninformative, or the gate is removing profitable trades. **Stop and rethink.** |

### 7.4 Implementation (in `D:\Study\CIKM\Smoke_test\`)

Standalone scripts, do **not** touch the production codebase:
- `smoke_kill_switch.py` — load existing checkpoint, MC-dropout σ, hard-rule gating, report Sharpe
- `run_smoke.sbatch` — H100 sbatch
- 1 dedicated H100 GPU; ~30 min per (model, horizon) at MC-dropout 30 passes

### 7.5 What if Track A passes?

Move to Track B with confidence. Use the smoke-test results as:
- A **baseline** in the paper ("naive kill-switch on MSE-trained model")
- An **upper bound check** on Track B ("retrained gated model should beat this")

### 7.6 What if Track A fails?

Two sub-cases:
- **Gate hurts Sharpe by removing profitable trades** → σ proxy needs replacement. Try: vol-only gate (no σ); or σ from a small auxiliary head trained quickly on residuals over val period.
- **Gate is uniformly noisy** → kill-switch story is harder than it looks. Pivot the paper to the catastrophic-forgetting + Sharpe-loss-only story (drop the kill-switch claim).

---

## 8. Track B: full retraining (only after Track A validates)

### 8.1 Scope: 8 models for forgetting analysis, 8 models global-only for new-loss retraining

Two separate scopes by purpose:

**Scope A — Catastrophic-forgetting evidence (seq + glob baselines for all 8):**
The seq < glob comparison is the paper's first claim. We need every architecture's seq vs. glob MSE/Sharpe to fill the capacity-vs-forgetting curve. Phase 0 closes the 2 gaps.

**Scope B — New-loss retraining (global only, all 8):**
Sequential's role is purely as evidence for the forgetting effect — the new loss doesn't fix forgetting (it's a training-protocol issue, not an objective issue). So Track B retrains **global only**. Halves Track B compute.

| Model | MSE baseline (Scope A) | New-loss retrain (Scope B, global) |
|-------|-------------------------|---------------------------------------|
| DLinear | ✅ seq + ✅ glob | ✅ |
| iTransformer | ✅ seq + ✅ glob | ✅ |
| GCFormer | ✅ seq + ✅ glob | ✅ |
| PatchTST | ✅ seq + ✅ glob | ✅ |
| AdaPatch | ✅ seq + ✅ glob (uses α=0.9 for H=120) | ✅ |
| TFT | 🔄 in flight (4/10 done) | ✅ (after baseline) |
| **TimesNet** | ❌ **needs Phase 0** (10 jobs queued) | ✅ (after baseline) |
| **VanillaTransformer** | ❌ **needs Phase 0** (10 jobs queued) | ✅ (after baseline) |

**Track B job count:** 8 models × 5 horizons × **1 method (global)** = **40 retraining jobs** (was 80). At ~3.5h/job × 3-partition parallelism (~5 concurrent) = **~2 days wall-clock**. Walk-forward CV × 3 windows = **~6 days**.

#### Phase 0 prerequisite — finish missing MSE baselines (BOTH seq + glob)

For Scope A (catastrophic-forgetting figure) we need TimesNet and VanillaTransformer's *full* seq + glob baselines — without sequential, the forgetting curve has 2 missing data points and the central thesis weakens.

- **TimesNet**: previously cancelled at 4h; bf16 FFT cast + batch=128 + stdbuf fixes already pushed. Re-queued 10 jobs (5 seq + 5 glob).
- **VanillaTransformer**: never trained. Submitted 10 jobs (5 seq + 5 glob).

20 jobs total — runs in parallel with Track A smoke. ~3-4 days wall-clock, fits before Phase B begins.

### 8.2 Walk-forward CV (mandatory for ICAIF)

3 rolling windows on the train period:
- Window 1: train 2018-2019, val 2020 H1, test 2020 H2 (covers COVID shock)
- Window 2: train 2018-2020 H1, val 2020 H2-2021 H1, test 2021 H2
- Window 3: train 2018-2021 H1, val 2021 H2-2022 H1, test 2022 H2-2023 (covers 2022 inflation regime)

Report Sharpe with stability across windows. Reviewers will ask. Total: 30 retraining jobs × 3 windows = 90 jobs. ~9 days at our pace.

### 8.3 Phases

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 0 — TimesNet + Vanilla MSE baselines (seq + glob)** | 3-4 days | 20 missing-baseline checkpoints (runs in parallel with Phase A) |
| **Phase A — Smoke tests (Track A): MC-Dropout + vol-z + cross-sectional** | 1 week | Go/no-go on each gating variant. Cross-sectional results validated May 3: PatchTST long-short Sharpe 6.52 (gross). |
| **Phase A.5 — Cross-sectional + cost-sensitivity baseline** | 2 days | All 8 models × {long_short, long_only} × 5 cost levels. This becomes the paper's headline table. |
| **Phase B — `RiskAwareHead` integration on all 8 models** | 4 days | Shared head module; smoke run on PatchTST H=5 only |
| **Phase C — Composite loss implementation** | 5 days | `engine/losses.py` module; smoke train of PatchTST H=5; verify gradient flows to all heads |
| **Phase D — Full retrain, GLOBAL ONLY (8 models × 5 horizons = 40 jobs)** | 2 days | 40 checkpoints + financial metrics |
| **Phase E — Walk-forward CV (40 jobs × 3 windows = 120 jobs)** | 6 days | 3-window stability tables |
| **Phase F — Ablation matrix** (no/vol-gate/uncert-gate/both × 8 models × H=5 = 32 jobs) | 3 days | 4-cell matrix per model |
| **Phase G — Statistical hardening** | 2 days | Diebold-Mariano, Politis-Romano stationary bootstrap CIs on portfolio Sharpe, Hansen SPA test |
| **Phase H — Write-up** | 2 weeks | Paper draft + figures |

**Total: ~6.5 weeks** — fits in 13 weeks to ICAIF deadline with ~6.5 weeks slack.

Notes:
- Cross-sectional Phase A.5 is a new addition, motivated by the May 3 ICAIF-convention research showing portfolio Sharpe is the venue-standard metric. It runs on existing MSE-trained checkpoints with zero retraining (~3 min per (model, mode) on H100).
- Per-stock-Sharpe-median (existing reports) becomes a diagnostic, not headline.
- The seq baseline jobs (Phase 0) are evidence for the forgetting claim, not retraining targets — only global gets the new loss.
- If Phase E walk-forward runs over, scale to 2 windows (covers 2020 COVID + 2022 inflation; drops calmer 2021 window).

### 8.4 Kill-paper criterion

If at end of **Phase D**, the gated model's Sharpe at H=5 is not at least 0.1 above the ungated MSE baseline (PT 1.084, GC 1.021), **abandon the kill-switch claim** and pivot to a paper focused only on:
- Catastrophic forgetting analysis (we have it)
- Sharpe-loss training (subset of composite without gate)

That's still a publishable paper at IEEE Big Data, just not the headline result we wanted.

---

## 9. Concrete code-level changes (refined from design.md §6)

### Phase B (RiskAwareHead) — `feature/risk-aware-head` branch
1. `engine/heads.py` (new) — `RiskAwareHead` class. ~50 LOC.
2. `models/{dlinear,gcformer,patchtst}.py` — replace terminal linear with `RiskAwareHead`. ~5 LOC × 3 = 15 LOC.
3. `engine/trainer.py` — handle 4-tuple output for the 3 models on the branch. Add a `--use_risk_head` flag (default False; existing behavior unchanged when off). ~30 LOC.

### Phase C (Composite loss) — same branch
4. `preprocess_global_cache.py` — add `log_vol_target.npy` cache. ~30 LOC.
5. `engine/losses.py` (new) — `CompositeRiskLoss` with all 7 terms + scheduling. ~200 LOC.
6. `engine/trainer.py` — swap `nn.MSELoss()` for `CompositeRiskLoss(args)` when `--use_risk_head` is True. ~20 LOC.
7. `evaluate_financial.py` — extend to apply hard kill-switch and report gated + ungated Sharpe. ~50 LOC.
8. `config.py` — `LOSS_COEFS`, `KILL_SWITCH_VOL_PCT`, `KILL_SWITCH_SIGMA_PCT`, `GATE_TEMP_DECAY`. ~10 LOC.

**Total new: ~360 LOC. Edits: ~80 LOC. None touches main branch until Phase D verifies smoke train works.**

---

## 10. What's explicitly cut (per critique)

To keep ICAIF timeline realistic:
- ❌ **Migration to log-return prediction space** — would re-invalidate all existing checkpoints
- ❌ **Sentiment dispersion as third gate signal** — too much novelty in one paper
- ❌ **Deep ensembles** for uncertainty — breaks fairness criterion
- ❌ **Sortino as secondary objective** — redundant with drawdown penalty
- ❌ **Stock-stratified Sharpe by sector** — useful but not in critical path
- ❌ **Calibration reliability diagrams** — keep for appendix, not main figures
- ❌ **iT / TimesNet / AdaPatch / TFT / VanillaTransformer retraining** — out of headline; ungated baseline only

These can all be follow-up work for a second paper.

---

## 11. ICAIF-compliant evaluation protocol (NEW)

**Convention research** (8 ICAIF papers 2023-2025 sampled): the dominant Sharpe-reporting protocol at ICAIF is **portfolio-level annualized Sharpe on a single combined return stream from a long-short or top-K long portfolio**, not per-stock-Sharpe-median. Our existing per-stock-median is unusual at this venue and would be flagged.

### 11.1 Headline metric set (mandatory)

| Metric | Definition | Notes |
|--------|------------|-------|
| **Portfolio Sharpe** | `mean(daily_port_ret) / std(...) × √252` | Headline number for the paper |
| **Portfolio MDD** | Max peak-to-trough drawdown of NAV curve | Universal companion |
| **Calmar ratio** | `annualised_return / MDD` | Common in RL-flavored ICAIF papers |
| **Cumulative return** | Total return over test period | Always reported |
| **Sortino ratio** | Sharpe with downside-only deviation | RL/FinRL lineage |
| **Hit rate (portfolio)** | Fraction of profitable rebalances | Sanity check |
| **Avg turnover** | Mean position change per rebalance | Drives transaction-cost story |
| **Cross-sectional IC** | Spearman rank corr (pred vs actual) per t | Factor-style papers; cheap to add |
| **ICIR** | `mean(IC) / std(IC) × √252` | Information-coefficient information-ratio |

### 11.2 Transaction-cost sensitivity (mandatory for ICAIF deployment story)

Report the headline metric set at **5 cost levels**: **0 / 5 / 10 / 20 / 50 bps round-trip**. The 5-10 bps band represents typical institutional execution; 20-50 bps represents retail or thin liquidity. The "breakeven cost" at which net Sharpe matches naive is the killer number a finance reviewer wants.

### 11.3 Strategy specification (long-short cross-sectional)

At each rebalance time t:
1. Compute predicted H-day return for every stock in the universe.
2. Rank stocks ascending by predicted return.
3. **Long** the top-N stocks (highest predicted return), equal-weight: weight = `+1/N`.
4. **Short** the bottom-N stocks (lowest predicted return), equal-weight: weight = `-1/N`.
5. **Cash** on the middle stocks.
6. Hold for H days, then rebalance.

Variants reported: `long_short`, `long_only top-N`, `short_only bottom-N`. Best-N is calibrated on validation by gross Sharpe.

### 11.4 Reporting convention going forward

**Headline tables in every model report**:
- Row 1: Portfolio Sharpe (gross) × Cost sweep
- Row 2: Portfolio MDD × Cost sweep  
- Row 3: Calmar × Cost sweep

**Diagnostic appendix in every model report**:
- Per-stock Sharpe distribution (current data — keeps as diagnostic of cross-sectional consistency)
- Per-stock IC distribution
- Turnover histogram

**Cross-architecture comparison table** in the paper:
- 8 models × 1 row per model × `long_short` portfolio Sharpe at 10 bps cost (the headline number)

### 11.5 Empirical first hit (smoke test, May 3 2026)

Cross-sectional ranking on existing MSE-trained checkpoints (zero retraining, just inference + portfolio construction):

| Model (H=5 global) | Best top-N | Portfolio Sharpe (gross) | Portfolio MDD |
|--------------------|-----------:|-------------------------:|--------------:|
| **PatchTST long-short** | 15 | **6.52** | 0.040 |
| GCFormer long-short | 15 | 6.11 | 0.142 |
| PatchTST long-only top-N | 15 | 5.79 | 0.046 |
| iTransformer long-short | 15 | 3.09 | 0.083 |
| Naive equal-weight long-only | n/a | 0.60 | (TBD) |

*Pending: full cost-sensitivity sweep (job 165711) for net-Sharpe at 5/10/20/50 bps. Initial gross-Sharpe numbers may shrink under realistic costs but retain ordering.*

These are 5-6× the per-stock-median Sharpe figures we previously reported — diversification within the portfolio + ranking-based signal extraction yield dramatically better risk-adjusted returns. The headline becomes ICAIF-grade.

---

## 12. Venue-specific framing

### ICAIF (primary target, deadline 2 Aug)
- Frame as: "Three-step recipe to convert MSE forecasters into deployable trading signals: training paradigm correction, Sharpe-loss objective, regime-aware kill-switch."
- 6-page paper format, 2-page appendix.
- Required: walk-forward CV, transaction cost sweep, ablation matrix.
- Reviewers expect: net Sharpe (after costs), MDD, hit rate, IC, turnover.

### IEEE Big Data (fallback, deadline 22 Aug)
- Same paper, framed slightly differently: emphasize the data scale (302 stocks × 5+ years × 6 features × 8 architectures × 2 training methods + walk-forward = ~5,400 model-evaluations).
- Reviewers accept lighter finance rigor; weight scale and reproducibility instead.

### What we drop if pivoting to IEEE Big Data
- Walk-forward CV could become 2 windows instead of 3
- Transaction cost sweep could be a single point (10 bps) rather than a curve
- Ablation matrix optional

### What we can add for ICAIF if Track A passes early
- Cross-sectional ranking head (`predict per-stock rank within universe`) — bonus contribution; aligned with how quants actually deploy

---

## 12. One-paragraph summary

Replace `nn.MSELoss()` with a composite `α·SharpeGated + β·NLL + γ·MSE_R + δ·VolMSE + η·GateBCE + λ_to·Turnover (+ λ_dd·MDD)`, where the Sharpe term operates on returns multiplied by a differentiable kill-switch gate built from the model's own predicted realized vol and predicted σ. Position is Kelly-scaled by σ. Train with a 3-phase coefficient schedule and gate-temperature annealing so the soft training gate becomes near-binary by epoch 30, eliminating the train→inference gap. Make 3 backbones (DLinear, GCFormer, PatchTST) share an identical `RiskAwareHead` so only the encoder varies; leave the other 5 models as ungated baselines. Stay in price-space (don't migrate to log-returns, would invalidate existing checkpoints). Validate the kill-switch claim **first** with a 1-week post-hoc Track A smoke test using MC-dropout σ on existing checkpoints — if gated Sharpe doesn't beat ungated by ≥0.1, abandon the kill-switch claim. If it does, run full Track B retraining with walk-forward CV, transaction cost sweep, and 4-cell ablation matrix, targeting ICAIF (Aug 2) with IEEE Big Data (Aug 22) as fallback.
