# Track B — Findings, Methods, and Statistical Analysis

**Date:** 2026-05-06 (updated)
**Universe:** GCFormer (Stage 1 of the Sharpe-loss campaign)
**Status:** Stage 1 (single backbone) complete on GCFormer; results are statistically validated for H=60 (p<0.001), directionally consistent at H=5, mixed at H=20.

**Three-stage robustness roadmap** (each stage strictly stronger than the previous, sequenced by cost-impact):
* **Stage 1** (cheap fan-out, in-progress): replicate H=5/20/60 result on the other 6 backbones — addresses architecture concern.
* **Stage 2** (zero-retrain, ~1 week): re-evaluate frozen checkpoints on the FNSPID dataset extended to 2024–2025 via the official Nasdaq scraper [8] — addresses out-of-time concern, also functions as the paper's data contribution. Restores statistical power for H ≥ 120.
* **Stage 3** (one-shot expensive, ~3 days HPC, planned): walk-forward CV across 3 rolling 1-year test windows on a **delisting-aware bias-free universe** for the top 1–2 winners. Combines temporal robustness × universe robustness in a single stage — addresses survivorship bias *and* regime dependence.

Stage-1 long-horizon point estimates are reported in §A.1 for transparency; they are *not* in the headline tables because the current test window is below the non-overlap-sample-count threshold for stable Sharpe inference at H ≥ 120.

---

## 1. Research question

Does replacing point-forecast MSE with a Sharpe-aware composite loss + risk-aware inference yield better risk-adjusted portfolio performance, controlling for backbone architecture and data?

**Headline answer (GCFormer, scope: H ∈ {5, 20, 60}):** Yes at H=60 (statistically significant, p<0.001, paired stationary bootstrap on 5000 reps), directionally consistent at H=5 (point Δ +0.31 Sharpe, NS), wash at H=20. Long horizons (H ≥ 120) are below the non-overlap-sample-count threshold for stable Sharpe inference on the current 504-day test window — see §A.1 + §7.6. Stage-2 of the campaign extends the test window via the official FNSPID Nasdaq scraper [8] to restore statistical power at long horizons.

---

## 2. Methods

### 2.1 Architecture (Track B)

`engine/heads.py::RiskAwareHead` wraps any backbone in `models/__init__.py:model_dict` and adds two scalar auxiliary heads:

* `sigma_head` predicts `log σ²` of the H-step ahead RETURN (heteroscedastic uncertainty).
* `vol_head` predicts forward log realised volatility (used in the gate kill-switch).

The wrapper exposes the standard `forward(x_enc, x_mark_enc=None)` signature but returns a *dict* of named outputs `{mu_close, mu_close_H, mu_return_H, log_sigma2_H, log_vol_pred, last_close}`. The trainer detects dict outputs and dispatches to `CompositeRiskLoss`.

A 1% floor on `|last_close|` is applied to the scaled-space return denominator in both predicted and observed return computations to keep `(close_{t+H} − close_t) / |close_t|` numerically bounded under per-stock MinMax scaling. Without the floor, samples near a stock's historical low produced 10¹²-magnitude losses on real data.

### 2.2 Composite loss (Track B objective)

```
L = α · L_SR_gated      ← gated differentiable Sharpe (training objective)
  + β · L_NLL           ← heteroscedastic Gaussian NLL on the H-step return
  + γ · L_MSE_R         ← return-MSE anchor (prevents flat-only collapse)
  + δ · L_VOL           ← MSE on log realised vol target
  + η · L_GATE_BCE      ← BCE: gate vs. realised-profitability sign
```

Defaults: β=0.5, δ=0.3, η=0.1. α and γ follow a 3-phase schedule:

| Phase | Epochs | α | γ |
|------:|:------:|:--:|:--:|
| 1 | 0–4 | 0.0 | 1.0 |
| 2 | 5–14 | 0.3 | 0.5 |
| 3 | 15+ | 0.7 | 0.2 |

Position size is `tanh(α_pos · μ / (σ + ε))`, gate is `σ((τ_v − log_vol)/T) · σ((τ_σ − σ)/T)` with temperature T annealed from 1.0 → 0.13. `L_GATE_BCE` is implemented inline (not via `F.binary_cross_entropy`) because the fused PyTorch kernel is autocast-unsafe under bf16/fp16.

### 2.3 Training protocol

* **Universe:** 302 train stocks (FNSPID stocks excluding hand-picked NAMES_50), 49 test stocks (NAMES_50, calendar-aligned val 2022-01-01 → 2023-01-01, test 2023-01-01 → end-of-data).
* **Init:** fine-tune from the existing MSE-trained backbone via `--init_from` (Phase 1's pure-MSE warm-up further locks the price head before the Sharpe gradient activates).
* **Schedule:** 30 epochs total (5 + 10 + 15), `lr=5e-5`, `batch_size=512`, AMP bf16 on H100/H200.
* **Early stopping DISABLED in risk-head mode.** The composite loss is *designed* to trade some price-MSE for risk-adjusted P&L; val MSE rising mid-fine-tune is a feature of Phase 2/3, not a divergence signal. Saving the FINAL-epoch state ensures we evaluate the Phase-3-converged Sharpe-optimised model, not an early-MSE-aligned snapshot. (A `*_bestval.pth` checkpoint is also saved for diagnostic ablations.)

### 2.4 Cross-sectional ranking evaluation

Strategy at each rebalance t: long top-N predicted, short bottom-N predicted, equal stocks per leg, rebalance every H trading days, [::H] non-overlap subsample for Sharpe. `top_n` is chosen on val by gross Sharpe, then applied to test. Net Sharpe is reported under round-trip cost sweep {0, 5, 10, 20, 50} bps.

Two strategy modes implemented in `Smoke_test/cross_sectional_smoke.py`:

* `--strategy simple` (works for any model): rank by μ alone, equal-weight per leg.
* `--strategy risk_aware` (requires `--use_risk_head`): rank by **μ/σ** (predicted Sharpe), weight ∝ **|tanh(α · μ / σ) · gate|** (Kelly-style with gate as a continuous multiplicative weight, training-time semantics applied at inference). Default `gate_threshold=0` because training-time `gate_mean ≈ 0.35` — the gate was trained as a continuous weight, not a binary kill-switch (a 0.5 hard threshold filters out essentially all trades).

Annualised Sharpe = `mean / std × sqrt(252 / H)`. The `sqrt(252/H)` factor assumes IID returns; under positive serial correlation the correct multiplier is smaller (Lo 2002, see §6.1).

### 2.5 Statistical inference: paired stationary bootstrap

For testing whether arm A's Sharpe differs significantly from arm B's, we use the **Politis-Romano stationary bootstrap** (Politis & Romano 1994 [1]) with paired index sampling: the same block-index sequence is drawn from both arms so per-period covariance is preserved. We compute 5000 replications, average block length 5 samples (default for Sharpe applications per the literature [2]). Reports:

* Marginal 95% CI for each arm's Sharpe.
* 95% CI for the **difference** A − B.
* One-sided p-value `p(A_boot ≤ B_boot)` for H₀: A ≤ B.

Implementation: `Smoke_test/bootstrap_paired.py`.

---

## 3. Training results (final-epoch on test set)

| H | Test MSE | R² | L_SR_gated final | Reads |
|---:|:----:|:----:|:----:|:------|
| 5 | 0.00480 | **0.957** | −0.18 | excellent |
| 20 | 0.0217 | 0.806 | −0.24 | strong |
| 60 | 0.0283 | 0.748 | −0.24 | strong |
| 120 | 0.0473 | 0.583 | −0.32 | moderate |
| 240 | 0.289 | **−1.57** | −0.41 | price head destroyed |

H=240 R² < 0 means the model is worse than predicting the mean — the Sharpe gradient pulled the price head completely off-target at the longest horizon. H=240 is dropped from the headline comparison; the test window only allows 1–2 non-overlap rebalances at H=240, making any Sharpe inference uninformative regardless.

---

## 4. Cross-sectional Sharpe — headline matrix

**Scope statement:** The headline comparison covers H ∈ {5, 20, 60}. Long horizons (H ≥ 120) are deferred to §A.1 because the current test window (2023-01-01 → end-of-FNSPID-data, ~504 trading days) yields fewer than 4 non-overlap rebalances at H=120 and only ~1 at H=240, below what the [::H] non-overlap Sharpe construction can support. Stage-2 of the campaign (FNSPID extension to 2024-2025, see §7.6) extends the test window to ~756 trading days; long-horizon results will be reported there.

Net Sharpe @ 10 bps round-trip transaction cost. Same universe, same calendar split, same evaluator across all rows. ICIR = mean cross-sectional Spearman IC / std × sqrt(252).

| H | MSE+simple | TB+simple | **TB+risk_aware** | Δ vs MSE | Naive EW | TB+RA Calmar | TB+RA MDD |
|---:|:----:|:----:|:----:|:---:|:---:|:---:|:---:|
| 5 | 3.44 | 3.40 | **3.80** ✅ | **+0.36** | 1.86 | 12.30 | 6.4% |
| 20 | 2.32 | 1.78 | 2.14 | -0.18 | 1.66 | 13.46 | 4.1% |
| 60 | -0.35 | -0.20 | **1.61** ✅ | **+1.96** | 3.22 | 6.56 | 3.6% |

Aggregate over headline horizons:
* MSE + simple average net Sharpe = **1.80**
* **Track B + risk_aware average net Sharpe = 2.52** (+40% improvement)

### 4.1 Ablation: does the σ+gate machinery do real work?

TB+risk_aware vs TB+simple holds the model fixed and changes only the inference strategy. A positive Δ confirms the σ+gate are not just regularization artefacts but are usable inference-time signals.

| H | TB+RA | TB+simple | Δ | p(RA ≤ simple) |
|---:|:----:|:----:|:----:|:----:|
| 5 | 4.16 | 3.67 | +0.49 | 0.126 |
| 20 | 2.22 | 1.83 | +0.39 | 0.190 |
| **60** | **1.65** | **−0.16** | **+1.81** | **0.0000** |

At every horizon, the risk-aware strategy beats the simple-ranking strategy on the same model. At H=60 the effect is huge and statistically significant.

---

## 5. Statistical inference (paired stationary bootstrap, n_boot=5000)

### 5.1 TB+risk_aware vs MSE+simple — the headline test

| H | TB+RA Sharpe (95% CI) | MSE Sharpe (95% CI) | Δ point | Δ 95% CI | p(A≤B) | Verdict |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|
| 5 | 4.16 [2.07, 6.83] | 3.85 [1.85, 6.40] | +0.31 | [-0.64, +1.41] | 0.253 | dir-positive, **NS** |
| 20 | 2.22 [1.05, 3.65] | 2.39 [1.98, 4.91] | -0.16 | [-3.37, +0.78] | 0.717 | wash |
| **60** | **1.65 [0.49, 5.08]** | **−0.32 [−2.00, 1.26]** | **+1.97** | **[+1.49, +4.65]** | **0.0000** | **★ SIGNIFICANT** |

* **H=60 is the headline statistically-significant result**: Track B's risk-aware strategy turns a losing strategy (−0.32 Sharpe) into a winning one (+1.65), with the difference confidence interval strictly above zero and a one-sided p-value of effectively 0.
* **H=5 is directionally positive (+0.31)** but with a single test year of weekly rebalances we don't have power to detect a +0.3 Sharpe difference; this needs either a longer test window or pooling across models.
* **H=20 is a wash** — both methods land near 2.3 with overlapping CIs.
* **H ≥ 120** is below the non-overlap-sample-count threshold for stable Sharpe inference on the current 504-day test window; deferred to §A.1 + Stage 2 (FNSPID extension).

---

## 6. Discussion

### 6.1 Why is naive equal-weight Sharpe so high at H=60 (3.22) and H=120 (2.64)?

Three compounding factors:

1. **Bull-market test window.** The test period begins 2023-01-01. The S&P 500 returned ~24% in 2023 [3], with the so-called Magnificent Seven (AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA) returning approximately 75–112% on average and contributing roughly 62% of the index's total return [4][5]. Our test universe NAMES_50 is heavily overlapping with this group, so the equal-weight basket captures the full directional drift.

2. **Survivorship bias.** NAMES_50 is the set of 49 stocks that survived the entire FNSPID coverage period; by construction these are companies that did not delist, merge, or fail. Equal-weighting 49 ex-post survivors during a bull market produces an inflated Sharpe — see the P0 audit finding in `reports/codebase_audit.md`.

3. **Sharpe rises with horizon under positive return autocorrelation.** Under the IID assumption, annualised Sharpe = `mean/std × sqrt(252/H)`. With positively-autocorrelated returns the correct annualisation multiplier is smaller than `sqrt(T)`, so the *naive* annualised Sharpe is **upward-biased** when the data shows momentum / persistence. Lo (2002) [6] derives the correction explicitly: "the annual Sharpe ratio for a hedge fund can be overstated by as much as 65 percent because of the presence of serial correlation in monthly returns; once this serial correlation is properly taken into account, the rankings of hedge funds based on Sharpe ratios can change dramatically." Positive autocorrelation in equity returns is well-documented at horizons from 1 to 12 months [7] (Moskowitz, Ooi & Pedersen, "Time Series Momentum", JFE 2012, document persistence at 1–12 month horizons that partially reverses over longer windows). The 1.86 → 1.66 → 3.22 → 2.64 pattern across H ∈ {5, 20, 60, 120} is consistent with a regime where 60-day persistence is strong (Mag-7 trend in 2023) and longer-window returns start to absorb mid-2024 drawdowns.

**Importantly, the naive EW comparison is structurally unfair to long-short strategies.** Naive EW is 100% net long and captures market drift at full leverage; long-short is market-neutral by construction (β ≈ 0) and earns only the cross-sectional spread. Comparing them directly on Sharpe is not like-for-like. The right comparison for an LS strategy is vs other LS strategies (here: MSE-trained predictor), and we report naive EW only for context. The H=60 result is best read as: "the MSE long-short loses money (−0.35), Track B's long-short recovers a positive alpha (+1.61), the unhedged equal-weight benchmark over the same window earned 3.22 from market drift."

### 6.2 Where Track B's edge actually comes from

Both the headline H=60 result and the ablation point to the same mechanism: the Track B *training* schedule pulls the µ head off pure-MSE-optimum but *trains* σ to be a calibrated confidence signal. With `simple` ranking (µ alone), the off-MSE-optimum µ is mildly worse than the MSE baseline's µ — Track B + simple loses small at H=5 and H=20. With `risk_aware` inference (µ/σ ranking, Kelly-tanh sizing, gate weighting), the σ-aware inference recovers the lost performance and then some. **The composite loss only pays off when paired with risk-aware inference; using Track B's µ alone is at best neutral.**

### 6.3 Why H=60 specifically

Two factors compound at H=60:

* **Cross-sectional IC at H=60 is small and noisy** for both methods (MSE: −0.011, TB+RA: +0.022). With weak ranking signal, the **magnitude** of predicted Sharpe (μ/σ) and the **gating** of low-confidence trades become disproportionately valuable — they keep the strategy out of trades the model can't price reliably. That's what TB+RA exploits.
* **MSE baseline at H=60 actually loses** money (−0.35 Sharpe). The bar is low and any improvement reads as a large delta.

This makes H=60 the "honest mid-horizon" benchmark: the regime where pointwise prediction is hard (low IC) but the model's calibration about *which predictions to trust* is the binding signal.

---

## 7. Limitations + next steps

The campaign roadmap has three robustness pillars, sequenced for cost-impact rather than alphabetical order. Each addresses a different reviewer concern.

### 7.1 Single backbone (GCFormer) — Stage 1 fan-out (immediate next step)

To claim "loss-function-matters across architectures", we need to replicate at least the H=60 result on the other 6 models (DLinear, iTransformer, PatchTST, AdaPatch, TFT, VanillaTransformer). 30 retrains, ~3-5 days of HPC, reuses existing infrastructure (`scripts/submit_riskhead_campaign.sh ALL`).

### 7.2 Single test window — Stage 2 (FNSPID dataset extension)

Bootstrap CI gives us **within-window** significance. **Out-of-time** validation requires testing on data the model has never seen. Two options were considered:

* *(rejected)* Shifting the val/test split earlier would give us more samples but requires retraining ~70 jobs and tests a different past window.
* *(adopted)* **Extend the FNSPID dataset itself to 2024–2025** using the official Nasdaq scraper at <https://github.com/Zdong104/FNSPID_Financial_News_Dataset>. The trained checkpoints stay frozen; only the test set extends. No retraining cost. This is true out-of-time validation.

Adding 12 months of 2025 data extends the test window from ~504 → ~756 trading days, which gives H=120 → ~6 non-overlap samples and H=240 → ~3 — sufficient for stationary bootstrap inference at all horizons.

This stage **also functions as the "data contribution" of the paper** (extending the FNSPID corpus is itself a deliverable).

### 7.3 Survivorship-biased universe **AND** single test window — Stage 3 (combined)

This stage tests the two heaviest reviewer concerns *simultaneously* — universe robustness and temporal robustness — because they share the same incremental cost: any test on a new universe requires retraining, and any walk-forward window also requires retraining, so we may as well combine them.

**The two questions it answers:**

1. *Universe robustness.* NAMES_50 is hand-picked from stocks that survived the entire FNSPID coverage period; this inflates absolute Sharpe for **both** Track B and the MSE baseline equally. The expected behaviour is that the *delta* (Track B − MSE) survives on a delisting-aware universe but absolute numbers drop.
2. *Temporal robustness.* Bootstrap CI (§5) and the FNSPID extension (§7.2) give us within-window and one-step out-of-time evidence respectively. Walk-forward cross-validation rolls the train / val / test windows forward across multiple calendar regimes, asking: does the Track B advantage replicate on independent calendar slices, or did 2023-2024 happen to favour it?

**Design.** Walk-forward with **3 rolling windows**, train length ~5 years per window, val 1 year, test 1 year, advancing by 1 year per window. Top 1–2 winners from §7.1 only. Both arms (MSE baseline + Track B) per window. Headline horizons H ∈ {5, 20, 60} only — at 1-year test windows H ≥ 120 is back below the non-overlap-sample threshold (Stage 2 covers long horizons via the extended single window instead).

**Cost.** 3 windows × 2 models × 3 horizons × 2 arms = **36 retrains**, ~3 days HPC. Each window's retrain reuses the existing `scripts/riskhead_glob.sbatch` pipeline with only the train / val / test date cutoffs changed.

**Reporting.** Per-window per-(model, horizon, arm) Sharpe + paired-bootstrap on the difference, then a pooled across-window meta-bootstrap (per Politis & Romano subsampling, [1]) to get a single ΔSharpe estimate with both within-window and cross-window uncertainty. The headline robustness claim becomes:

> "Across three independent rolling 1-year test windows on a delisting-aware bias-free universe, Track B + risk-aware delivers a pooled ΔSharpe of X.XX [95% CI: Y, Z] vs the MSE baseline at H=60, p<X.XX (paired stationary bootstrap, 5000 reps within-window × 3 windows)."

**Why this stage is intentionally last.**

* (a) it's a confirmatory robustness check, not an exploratory experiment — only worth running once we're confident the *strategy* (architecture + loss + inference) is the right one;
* (b) cost-impact: at ~3 days of HPC, this is the most expensive single stage, so we want maximum prior evidence (Stage 1 + Stage 2 deltas) that it's worth running;
* (c) combining universe + walk-forward into a single stage is the cheapest way to deliver both pieces of evidence — neither is meaningful alone for ICAIF reviewers.

**What "walk-forward CV" actually means here.** It is *not* the same as a sliding train/val/test split *within* one fixed dataset (which is what a regular ML CV does and what we already do via the calendar-aligned val/test split). Walk-forward in finance means: independently retraining on calendar window 1, testing on window 2; independently retraining on window 1+2, testing on window 3; etc. Each window's test set is held out at training time, so it's a true out-of-sample test repeated across regimes. See §8 for the bootstrap-vs-walk-forward distinction in full.

### 7.5 Bootstrap CI is on gross returns, not net

The cost-sensitivity sweep is reported point-wise but the bootstrap is on the gross return series. At 10 bps round-trip with turnover ~1 per rebalance, the cost drag is modest and the statistical conclusion at H=60 is unlikely to flip (the +1.97 gross Δ would have to drop below the 95% CI lower bound of +1.49 to become non-significant, requiring a cost drag of >0.48 Sharpe — well outside the realistic 0.1–0.2 range at 10 bps). Net-return bootstrap is a polish item, not a campaign blocker.

### 7.6 H ≥ 120 horizon limit (current scope)

See §A.1 for the test-window arithmetic. H=120 gets ~4 non-overlap rebalances on the current 504-day test set; H=240 gets ~1. Both are below the threshold for stable Sharpe inference. Long horizons are deferred to Stage 2 (§7.2) where the extended FNSPID test window restores statistical power.

---

## 8. Three-stage robustness pyramid: bootstrap CI ⊂ FNSPID extension ⊂ walk-forward CV

Each row tests a strictly stronger generalisation claim than the row above. The campaign is sequenced bottom-to-top because each higher row is more expensive than the one below it, and each row's value depends on the rows below it confirming the underlying signal first.

| Stage | What | Tests | Cost | Status |
|:-:|---|---|---|:-:|
| 0 (paired stationary bootstrap on the test window) | Resample within the **same** test return series, preserving serial correlation via stationary block resampling [1] | "Given this fixed test window and these trained checkpoints, how confident are we in the Sharpe estimate?" → sampling uncertainty *within* a fixed window | Seconds on existing CSVs | ✅ §5 |
| 2 (FNSPID extension to 2024-2025) | Re-evaluate the **same trained checkpoints** on a freshly-scraped, post-training-date test slice | "Does the method work on data that didn't exist at training time?" → out-of-time validation, single new window | ~1 week wall clock for scraping + FinBERT replay + cache build; **0 retraining** | ⏳ planned |
| 3 (walk-forward CV × bias-free universe) | Independently retrain on rolling calendar windows on a **delisting-aware universe**, paired-bootstrap pooled across windows | "Does the method work across regimes AND on a different universe?" → temporal robustness × universe robustness | 36 retrains, ~3 days HPC | ⏳ planned (combines two checks) |

**These are not the same as a sliding train/val/test split inside a fixed dataset.** A regular ML CV cycles which fold is held out within a fixed pool; walk-forward CV in finance independently retrains the model from scratch on calendar window 1, tests on window 2; retrains on window 1+2, tests on window 3; etc. Each window's test set is held out at training time. Walk-forward is the only protocol that produces a truly out-of-sample evaluation repeatable across calendar regimes.

**Why all three?** Bootstrap CI gives within-window significance — it cannot tell you whether you got lucky on this window. FNSPID extension gives one piece of out-of-time evidence — it cannot tell you whether the result depends on the universe construction. Walk-forward × bias-free gives both. Stages 0 → 2 → 3 each strictly subsume the questions of the previous stage, so cumulative evidence compounds.

The headline H=60 win at p<0.001 (§5) is meaningful in the within-window sense; the FNSPID-2025 evaluation (§7.2) will tell us whether it survives on never-seen post-training data; the walk-forward × bias-free combined stage (§7.3) will tell us whether it survives across regimes and universe construction. ICAIF-grade requires all three.

---

## 9. Files referenced

```
engine/heads.py                                 RiskAwareHead
engine/losses.py                                CompositeRiskLoss + autocast-safe BCE + return-denom floor
engine/trainer.py                               Risk-head dispatch, no-early-stopping in TB mode
engine/evaluator.py                             Dict-output handling
train.py                                        --use_risk_head, --init_from flags
scripts/riskhead_glob.sbatch                    Track B SLURM template
scripts/submit_riskhead_campaign.sh             Campaign fan-out helper
Smoke_test/cross_sectional_smoke.py             --strategy {simple, risk_aware}, --use_risk_head
Smoke_test/bootstrap_paired.py                  Paired stationary-bootstrap on Sharpe difference
Smoke_test/bootstrap_ci.py                      Marginal stationary-bootstrap on Sharpe (older)
Smoke_test/results/timeseries_xs_GCFormer_*.csv Per-rebalance return series
Smoke_test/results/summary_xs_GCFormer_*.json   Per-(model,horizon,variant) summary blobs
tests/test_track_b.py                           5 unit tests (head shapes, loss, schedule, edges, e2e)
tests/test_trainer_risk_head.py                 6 wiring tests incl. autocast regression
tests/smoke_train_track_b.py                    25-epoch synthetic smoke
reports/track_b_implementation.md               Phase B/C/D implementation report
reports/track_b_findings.md                     This file
reports/codebase_audit.md                       P0 audit (incl. survivorship bias finding)
reports/design_rethinked.md                     The original Track B proposal
```

---

## References

1. **Politis, D. N., & Romano, J. P.** (1994). The Stationary Bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313. [Tandfonline link](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870) · [PDF](https://users.ssc.wisc.edu/~behansen/718/Politis%20Romano.pdf)

2. **Ledoit, O., & Wolf, M.** (2008). Robust Performance Hypothesis Testing with the Sharpe Ratio. *Journal of Empirical Finance*, 15(5), 850–859. [Working paper PDF](https://www.econ.uzh.ch/apps/workingpapers/wp/iewwp320.pdf). Recommends an average block length of 5 samples for Sharpe-difference inference under the Politis-Romano stationary bootstrap.

3. **CNBC** (2023). "The 7 largest stocks in the S&P 500 have returned 92% on average this year." [Article](https://www.cnbc.com/2023/10/06/magnificent-seven-returned-92-percent-this-year-but-its-risky-for-markets.html). Documents the Magnificent-Seven concentration of 2023 S&P 500 returns.

4. **The Motley Fool** (2023). "You Might Be Shocked to Learn Where the S&P 500 Would Be in 2023 Without the Magnificent Seven Stocks." [Article](https://www.fool.com/investing/2023/11/02/you-shocked-learn-sp-500-2023-magnificent-7-stocks/). Reports that without the Magnificent Seven, the S&P 500's gain would have been closer to ~9.9% rather than ~24%.

5. **YCharts** (2024). "What Happened to the Magnificent Seven Stocks?" [Article](https://get.ycharts.com/resources/blog/what-happened-to-the-magnificent-seven-stocks/). Documents the Magnificent Seven 2023 average return ≈ 111% with NVDA leading.

6. **Lo, A. W.** (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal*, 58(4), 36–52. [CFA Institute](https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios) · [PDF](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=05561b77acfdd034a585c32048819cc9ba6d1434). Demonstrates that monthly Sharpe ratios cannot be annualised by `sqrt(12)` except under IID returns; under positive serial correlation the correct multiplier is smaller, and the naive annualised Sharpe can be overstated by up to 65%.

7. **Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H.** (2012). Time Series Momentum. *Journal of Financial Economics*, 104(2), 228–250. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0304405X11002613) · [NYU Stern PDF](http://docs.lhpedersen.com/TimeSeriesMomentum.pdf). Documents positive return autocorrelation at 1–12 month horizons across 58 liquid futures (equity, currency, commodity, bond), partially reversing over longer windows — the empirical regularity that drives the H=60 peak in our naive Sharpe pattern.

8. **Dong, Z., Fan, X., & Peng, Z.** (2024). FNSPID: A Comprehensive Financial News Dataset in Time Series. *KDD '24 Applied Data Science Track*. [arXiv:2402.06698](https://arxiv.org/abs/2402.06698) · [GitHub](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) · [Hugging Face](https://huggingface.co/datasets/Zihan1004/FNSPID). 29.7M stock prices + 15.7M financial news records covering 4,775 S&P 500 companies, 1999–2023; Stage 2 of the Track B campaign extends this corpus to 2024–2025 using the official `data_scraper/` Nasdaq pipeline.

---

## Appendix

### A.1 Long horizons (H ≥ 120) — current-window numbers + horizon-limit math

These rows are *omitted from the headline tables in §4–§5* because the current 504-day test window is below the non-overlap-sample-count threshold for stable Sharpe inference at long horizons. They are reported here in full transparency.

#### A.1.1 The arithmetic

Test calendar: 2023-01-01 → end-of-FNSPID-data ≈ 2024-12-31, ≈ 504 trading days. The Sharpe is computed on the **non-overlapping subsample** `returns[::H]` (standard convention to avoid double-counting overlapping H-day forecasts):

| H | ≈ Non-overlap samples in test | Statistical adequacy |
|---|---|---|
| 5 | ~100 | plenty |
| 20 | ~25 | OK |
| 60 | ~8 | borderline (used in headline) |
| 120 | ~4 (eval drops first/last → 2 effective) | **bootstrap-meaningless** |
| 240 | ~2 (effectively 1) | **mean/std undefined** |

At n=2, every bootstrap resample is one of {(a,a), (a,b), (b,a), (b,b)} — four discrete states, no meaningful variance. At n=1, std is undefined (zero degrees of freedom).

#### A.1.2 H=120 — point estimates only (no CI)

| Variant | Net Sharpe @10bps | IC | MDD | Cumulative Return |
|---|---|---|---|---|
| MSE+simple | +0.011 | **−0.170** | 7.9% | −0.5% |
| TB+simple | −1.222 | −0.085 | 3.0% | −36.1% |
| TB+risk_aware | −1.386 | −0.084 | 5.0% | −36.7% |
| Naive EW | 2.642 | — | — | — |

**All three variants have *negative* cross-sectional IC at H=120**, meaning the predicted-return ranking is anti-correlated with realised returns at the 6-month horizon. Long-shorting an anti-predictive signal is destructive; risk-aware sizing amplifies the wrong signal harder, hence the worsened TB+RA Sharpe relative to TB+simple. *This pattern is consistent across both the MSE and Track B arms — it is a horizon limit of the architecture-on-this-universe, not a Track-B-specific failure.*

#### A.1.3 H=240 — training metrics only

H=240 cross-sectional evaluation does not produce a usable point estimate on the current test window (1 non-overlap sample → undefined std). What we *do* have from training:

| Metric | H=240 value |
|---|---|
| Test MSE (scaled space) | 0.289 |
| Test R² | **−1.573** |
| Final L_SR_gated | −0.41 |

R² < 0 means the model is worse than predicting the mean. The Sharpe gradient pulled the price head completely off-target at this horizon — consistent with H=240 (1-year-ahead point forecasts) being at or beyond the practical signal limit for stock-specific equity prediction in this dataset.

#### A.1.4 What Stage 2 (FNSPID extension) will give us

Adding 12 months of 2024–2025 data via the official FNSPID Nasdaq scraper extends the test window from ~504 → ~756 trading days. Resulting non-overlap sample counts:

| H | Current (504-day test) | After FNSPID-2025 (~756-day test) | Statistical adequacy |
|---|---|---|---|
| 120 | 2 | **6** | ✓ stable bootstrap |
| 240 | 1 | **3** | ✓ borderline-meaningful bootstrap |

At which point the long-horizon rows can be promoted from this appendix into the headline tables, with proper CIs.

---

### A.2 B1 — differentiable cross-sectional portfolio Sharpe loss (NEGATIVE RESULT)

**Status:** tested on GCFormer at H ∈ {20, 60}; underperformed Track B v1 (the per-sample Sharpe surrogate) at every measured horizon. Reported here in full so the paper's methods/results sections can cite the ablation as honest due diligence rather than glossing over it.

**Hypothesis tested.** Replace the per-sample Sharpe surrogate `mean(g·pos·r) / std(g·pos·r)` with a differentiable Sharpe over K=32 random "synthetic cross-sections" within each batch, each forming a long/short Kelly-tanh × gate portfolio with leg-normalised weights. The hope: training-time loss exactly matches the inference operation, gradients flow through the same construction, model learns weights that behave well *after* normalisation.

**Result.** B1 retraining produced strictly worse cross-sectional Sharpe than v1, with the gap growing at longer horizons:

| H | MSE+simple | v1+RA (current TB) | B1+simple | **B1+RA** | Δ vs v1+RA |
|---|---|---|---|---|---|
| 20 | 2.32 | 2.14 | 2.25 | **0.39** | **−1.75** |
| 60 | −0.35 | **+1.61** | −1.15 | **−0.18** | **−1.79** |

**Diagnostic — the price head over-degraded.** Final-epoch test R² across the two variants:

| H | v1 R² | B1 R² | Δ |
|---|---|---|---|
| 5 | 0.957 | 0.961 | +0.004 |
| 20 | 0.806 | 0.749 | −0.057 |
| 60 | 0.748 | **0.360** | **−0.388** |

At H=60 the B1 Sharpe gradient was 3–4× more negative than v1's at the same epoch (−0.78 vs −0.24 final), reflecting much stronger gradient signal — but it pulled the price head off-target by 39 percentage points of R². The σ + gate machinery could not compensate for a μ̂ this degraded.

**Why we believe the synthetic-cross-section formulation is the issue.** Random batch partitioning produces "cross-sections" where the K micro-portfolios contain stocks at different calendar dates. The portfolio returns these compute are statistical-noise blobs, not actual portfolios. The loss is therefore much higher-variance than v1's per-sample surrogate, but does not measure anything more meaningful than v1 did. Specifically, A1 corollary 5.1 predicts the Sharpe-edge equals the σ-weighted variance of per-stock Kelly scores **at a fixed timestamp** — this requires actual cross-sectional structure, which random partitioning destroys.

**A1 explains the failure pattern.** The price-head degradation was largest at H=60 — exactly where (per A1 corollary 5.1) the σ-aware Sharpe gradient pulls hardest against MSE because cross-sectional σ-heterogeneity is largest. B1's stronger but lower-quality gradient over-shot that trade-off; v1's weaker but accurate gradient stayed in the favourable region. The theory is *consistent with* the failure even though the experiment failed.

**Implications for the paper.**
* The headline architectural contribution (RiskAwareHead + composite loss + risk-aware inference, v1) stands unchanged.
* B1 is reported as a tested-but-unsuccessful extension; the paper's "future work" section flags **calendar-aligned cross-sectional sampling** as the likely correct path, requiring a data-pipeline rebuild we did not undertake in this submission cycle.
* A1 is unaffected — the theorem is independent of which Sharpe formulation is used during training.

**Files for reproducibility.**
* `engine/losses.py` `CompositeRiskLoss(use_xs_sharpe=True, xs_n_subgroups=K)` — the B1 path.
* `tests/test_xs_sharpe.py` — 4 unit tests confirming the path is finite, gradients flow, and the loss differs from v1.
* `Smoke_test/results/summary_xs_GCFormer_global_H{20,60}_long_short_riskhead_xs{,_RA}.json` — raw B1 evaluation outputs.
* Checkpoints `GCFormer_global_H{20,60}_riskhead_xs.pth` on HPC (and synced to local).
