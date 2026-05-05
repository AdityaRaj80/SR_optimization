# CIKM 2026 Applied Research Track — Positioning Analysis

**Date:** 2026-05-06
**Deadline:** Abstract May 16, paper May 23 (AoE), per [CIKM 2026 Applied Research Papers](https://cikm2026.diag.uniroma1.it/applied-research-papers/).
**Decision:** CIKM 2026 Applied is the primary submission target. ICAIF (Aug 2 deadline) is the fallback.

---

## 1. Is portfolio-Sharpe relevant to CIKM?

**Yes — clear precedent at CIKM 2024.** [DeepClair: Utilizing Market Forecasts for Effective Portfolio Selection](https://dl.acm.org/doi/10.1145/3627673.3680008) was accepted into the main CIKM 2024 proceedings and is a direct portfolio-Sharpe paper:

* **Architecture.** Transformer-based time-series forecaster (FEDformer) + deep RL portfolio policy + LoRA fine-tuning.
* **Datasets.** Top-26 stocks of Nasdaq and Dow Jones, January 1992 onwards. Forecasting train 1992.01–2004.09, test 2006.06–2008.12; portfolio test extended to 2022.12.
* **Metrics.** ARR, Sharpe, MDD, Calmar, Sortino — exactly the metric suite we use.
* **Baselines.** Cross-Sectional Momentum, Buy-Losers-Sell-Winners, EIIE, DeepTrader (2021), HADAPS (2023), market-index benchmarks.
* **Reported headline numbers.** Nasdaq: ARR 15.22%, Sharpe 0.659, MDD 48.59%. Dow Jones: ARR 10.30%, Sharpe 0.526, MDD 46.77%.
* **Statistical methodology.** 10 random seeds; mean reported; **no bootstrap confidence intervals.**

The DeepClair Sharpe numbers (0.66 / 0.53) are much lower than ours (1.61–3.80 at H=5–60) but **the test windows are not directly comparable**: DeepClair tests over 2007–2022 inclusive of the 2008 crisis and COVID; ours tests 2023-2024 inclusive of the Mag-7 bull market. Different absolute numbers do not contradict each other; we should **avoid claiming we beat DeepClair on Sharpe** since the windows differ. Instead we position as complementary.

Other relevant CIKM-adjacent precedent:
* Multiple FinBERT-LSTM papers (e.g., [arXiv:2407.16150](https://arxiv.org/abs/2407.16150)) show that sentiment-augmented forecasting is an active sub-area; they typically report MAE/MAPE rather than Sharpe.
* CIKM 2023/2024 has accepted financial-time-series papers across the spectrum (DRL-based, supervised, sentiment-augmented). The **distinguishing feature** of accepted work is empirical rigor + a clear methodological contribution that goes beyond existing baselines.

---

## 2. Where our paper stands vs CIKM bar

### 2.1 Strengths (CIKM-positive)

| Strength | Evidence |
|---|---|
| **Loss-function design as the primary contribution** | 5-term composite loss with 3-phase schedule + risk-aware inference. Concrete, deployable, and *not* yet in the CIKM portfolio-DL literature. |
| **Statistical rigor above DeepClair's bar** | Paired Politis-Romano stationary bootstrap (5000 reps, Ledoit-Wolf-recommended block length 5). DeepClair just averages 10 seeds. |
| **Architecture-agnostic claim** | RiskAwareHead bolts on to any backbone in `models/`. Stage 1 fan-out across 7 backbones (DLinear, iTransformer, GCFormer, PatchTST, AdaPatch, TFT, VanillaTransformer) tests this claim directly. |
| **Theoretical grounding (Theorem A1)** | Variance-decomposition identity gives a *closed-form lower bound* on the σ-aware Sharpe edge as a function of cross-sectional σ-heterogeneity. DeepClair has no theoretical result. |
| **Honest negative result (B1)** | Documenting the failed differentiable-cross-sectional extension is itself due-diligence credibility for reviewers. |
| **Full ICAIF/CIKM metric suite** | ARR, Sharpe, Sortino, Calmar, MDD, IC, ICIR, hit rate, turnover, **cost sensitivity sweep at {0, 5, 10, 20, 50} bps** — broader than DeepClair. |

### 2.2 Weaknesses (CIKM-negative)

| Weakness | Mitigation in this submission |
|---|---|
| **Short test window (~2 years)** | Stage 2 FNSPID extension (price data through 2026-05) extends to ~3 years. Documented horizon-limit math in §A.1 of `track_b_findings.md`. |
| **Survivorship-biased universe (NAMES_50)** | Stage 3 (walk-forward × bias-free) deferred to follow-up work due to 17-day deadline; honestly disclosed as limitation. |
| **Single primary dataset (FNSPID)** | Frame as "first comprehensive evaluation on FNSPID with rigorous bootstrap CI"; CIKM Applied accepts single-dataset work if other rigor is high. |
| **Sentiment data unavailable for OOT period** | Extended-period sentiment set to neutral (0.5); FNSPID news scraper extension out-of-scope for this submission cycle (Selenium-based, >180 GPU-hours estimate for the 351-stock universe). Document explicitly. |
| **Headline Sharpe-edge result is at H=60 only** | Stage 1 fan-out gives architecture-fan-out evidence at H ∈ {5, 20, 60}; the H=60 result needs to replicate on at least 4 of 7 architectures for the claim to hold. |

### 2.3 What CIKM Applied reviewers will probably ask

| Reviewer concern | Our preempt |
|---|---|
| *"Did you compare against DRL portfolio methods?"* | We compare against MSE-trained backbones of the same architecture (controls for everything except the loss function). DeepClair-style DRL is a different research thread; we cite it as related work. |
| *"Why these 7 architectures?"* | They span the dominant deep-learning forecasting families (linear, attention, patch, hierarchical). Not exhaustive but representative. |
| *"How does this generalise beyond US large-cap equities?"* | Honestly disclosed limitation; the loss/architecture is universe-agnostic but tested on a single universe in this submission. Future work calls for evaluation on FTSE / Nikkei / emerging markets. |
| *"Is the bootstrap CI on net or gross returns?"* | Currently on gross; analytic argument shows the H=60 conclusion is robust to plausible cost drag (in §7.5 of findings doc). Net-return bootstrap is a polish item. |
| *"What's the practical deployment cost?"* | RiskAwareHead is ~12k extra params (negligible). Composite loss is one extra forward through the auxiliary heads; ~10% training-time overhead. Inference is identical to MSE-trained model + 2 small MLP forwards. |

---

## 3. Realistic acceptance probability

Based on the venue precedent + our current evidence base:

| Scenario | Probability | Story |
|---|---|---|
| Stage 1 fan-out shows positive Δ at p<0.05 on **5+ of 7 architectures** at H=60 | 40-50% | Clean architecture-agnostic claim + theorem + bootstrap CIs + cost sensitivity. Strong CIKM Applied submission. |
| Stage 1 fan-out shows positive Δ on **3-4 architectures** | 30-35% | Mixed result; honest reporting helps but the headline weakens. Reviewers may push back. |
| Stage 1 fan-out shows positive Δ on **≤2 architectures** | 15-20% | Headline becomes "GCFormer-specific result"; CIKM may reject as too narrow. ICAIF fallback becomes critical. |

**Decision gate:** by ~May 7 evening (when Stage 1 fan-out completes) we have a concrete answer.

---

## 4. CIKM-aligned framing changes

Compared to the current ICAIF-leaning framing in `track_b_findings.md`, the CIKM submission needs:

1. **Title shifts emphasis to the design choice.** Current draft framing is "Track B retraining"; CIKM-aligned framing: "*Heteroscedasticity-Aware Composite Loss for Cross-Sectional Stock Ranking: Architecture-Agnostic Sharpe Gains with Theoretical Grounding*" or similar.

2. **Lead with the architectural primitive, not the result.** Section structure:
   - §1 Introduction: loss-function-matters thesis + headline result
   - §2 Related work: DRL portfolio (DeepClair), Sharpe-loss methods (Moody-Wu 2001), sentiment-augmented forecasting (FinBERT-LSTM literature)
   - §3 Method: RiskAwareHead + CompositeRiskLoss + risk-aware inference + Theorem A1
   - §4 Experiments: Stage 1 fan-out (architecture robustness) + paired bootstrap CIs + cost sensitivity
   - §5 Discussion + Limitations
   - §6 Conclusion

3. **Frame Theorem A1 prominently.** CIKM's broader audience will appreciate a closed-form result more than ICAIF's specialized audience. A1 should be in §3 (method), not buried in an appendix.

4. **Cost-sensitivity sweep gets its own subsection.** CIKM Applied loves practical-deployment angles; the {0, 5, 10, 20, 50} bps sweep is exactly that.

5. **Cite DeepClair (CIKM 2024) explicitly.** Position our loss-function contribution as complementary to their RL+forecasting approach. Acknowledge their longer test window without competing on absolute Sharpe.

6. **Honest scope.** State explicitly: "We evaluate on the FNSPID dataset (US large-cap equities, 1999–2026 with our extension). Generalisation to non-US universes and bias-free survivorship-aware constructions are deferred to ongoing work."

---

## 5. Sources verified

* [CIKM 2026 Applied Research Papers track page](https://cikm2026.diag.uniroma1.it/applied-research-papers/) — deadline May 23, 2026 AoE, abstract May 16.
* [CIKM 2024 conference report (Gullo)](https://fgullo.github.io/files/papers/SIGWEBNewsl25.pdf) — Applied-track acceptance rate 33% (103/316 in 2024).
* [DeepClair @ CIKM 2024 (ACM Digital Library)](https://dl.acm.org/doi/10.1145/3627673.3680008) — direct portfolio-Sharpe precedent at CIKM Applied/Full track.
* [DeepClair on arXiv](https://arxiv.org/abs/2407.13427) — full paper + numerical results we used to anchor positioning.
* [FNSPID GitHub repo](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) — reviewed scraper structure; Selenium-based with no date filter, infeasible in 17 days for the 351-stock universe; documented as out-of-scope.

All cited numbers (DeepClair Sharpe 0.659/0.526, MDD 48.59/46.77; FNSPID coverage 1999-2023; CIKM 2024 Applied acceptance 33%) are verified from the linked sources. No fabricated numbers.
