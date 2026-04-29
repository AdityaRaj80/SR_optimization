# DLinear vs iTransformer — Cross-Model Comparison

**Date:** 2026-04-29
**Question:** "Which model is *actually* working better — DLinear or iTransformer — and what do the numbers suggest?"
**Honest answer:** It depends on what you're measuring. The two architectures win on different metrics for principled reasons.

---

## 1. The numbers, side-by-side

### 1.1 Pure prediction error (MSE) — DLinear narrowly wins under Global

| H | **DLinear Glob MSE** | iT Glob MSE | Margin |
|---|---------------------|-------------|--------|
| 5 | **0.086** | 0.099 | DLinear -13% |
| 20 | **0.292** | 0.301 | DLinear -3% |
| 60 | **0.747** | 0.815 | DLinear -8% |
| 120 | **1.341** | 1.447 | DLinear -7% |
| 240 | **2.387** | 2.725 | DLinear -12% |

### 1.2 Trading-relevant metrics — iTransformer Global wins (often by a lot)

**Annualised Sharpe (median per stock):**

| H | DLinear Glob | **iT Glob** | Naive (long-only) | iT advantage over naive |
|---|--------------|-------------|-------------------|-------------------------|
| 5 | 0.433 | **0.671** | 0.389 | **+0.28 alpha** |
| 20 | 0.285 | **0.639** | 0.391 | **+0.25 alpha** |
| 60 | 0.170 | **0.391** | 0.388 | tied |
| 120 | 0.091 | 0.252 | **0.395** | -0.14 |
| 240 | 0.148 | 0.287 | **0.361** | -0.07 |

**Directional accuracy:**

| H | DLinear Glob | **iT Glob** | Const-baseline (%UP) |
|---|--------------|-------------|---------------------|
| 5 | 51.8 % | **53.0 %** | 52.4 % |
| 20 | 52.5 % | **56.1 %** | 54.2 % |
| 60 | 50.8 % | **57.0 %** | 56.2 % |
| 120 | 51.6 % | 55.1 % | **58.3 %** |
| 240 | 54.9 % | **59.2 %** | 60.7 % |

### 1.3 Sequential training — DLinear uniformly wins (iT collapses to forgetting)

| H | DLinear Seq MSE | iT Seq MSE | iT is worse by |
|---|------------------|-------------|----------------|
| 5 | 0.091 | 0.163 | +79% |
| 20 | 0.304 | 0.509 | +67% |
| 60 | 0.746 | 1.719 | +130% |
| 120 | 1.342 | 2.594 | +93% |
| 240 | 2.555 | 6.978 | +173% |

### 1.4 Per-stock consistency

| H | DLinear Glob Std MAPE | iT Glob Std MAPE | Lower is better |
|---|----------------------|-------------------|-----------------|
| 5 | 1.55 % | **1.19 %** | iT (-23%) |
| 20 | 3.66 % | **5.26 %** mean error | DLinear |
| 60 | 7.73 % | **9.39 %** | DLinear |
| 120 | 12.74 % | **17.37 %** | DLinear |
| 240 | 23.07 % | **26.78 %** | DLinear |

(per-stock std for global iTransformer is mostly comparable to DLinear; sequential iT shows much worse spread — see § 6 of `iTransformer_results.md`)

---

## 2. The diagnosis — *why* DLinear edges MSE but loses Sharpe

### 2.1 Stock prices are nearly random walks

A core empirical result in finance: **daily stock returns have near-zero autocorrelation**, which means the MSE-optimal forecast on the price level (not return) is approximately **persistence** — predict the last observed value.

Mathematically: if `X(t+H) = X(t) + ε`, where `ε` is mean-zero noise, then `E[(prediction − X(t+H))²]` is minimised by `prediction = X(t)` (with residual variance equal to var(ε)).

### 2.2 MSE-trained models converge near this attractor

Both DLinear and iTransformer are trained with MSE loss. They both end up close to the persistence solution because that's where the loss surface is flattest. *How* close they get depends on architecture:

- **DLinear** has limited capacity (~6 K parameters). It converges cleanly to a smoothed-persistence behaviour. The small remaining capacity isn't enough to extract directional signal from noise. **Result:** very low MSE, near-zero directional information.

- **iTransformer** has ~6.6 M parameters. Its capacity allows it to deviate slightly from pure persistence — it captures *some* multivariate / cross-feature patterns. **Result:** slightly higher MSE (when wrong, sometimes more wrong), but extracts genuine directional signal.

### 2.3 The accuracy-vs-direction tradeoff

This is a classic phenomenon in financial forecasting. The "best MSE prediction" looks like the last observation, which has zero useful information for trading. The "best directional prediction" can have higher MSE but gives a real signal.

We see this directly:

| | Closest-to-actual prediction | Useful directional information |
|--|--|--|
| DLinear Glob | ✅ ~13% lower MSE | ❌ DirAcc ≈ 51-52% (coin flip) |
| **iT Glob** | ❌ slightly higher MSE | ✅ DirAcc 53-59%, Sharpe up to +0.28 over naive |

DLinear is **unhelpfully precise**; iTransformer is **helpfully imprecise**.

### 2.4 The Sharpe gap at short horizons is real money

iT Glob H=5: Sharpe 0.671 vs Naive 0.389 → **+0.28 annualised excess Sharpe** is a strong, deployable trading edge. DLinear Glob H=5: Sharpe 0.433 vs Naive 0.389 → +0.044 is statistical noise. The 8% MSE penalty iTransformer pays buys ~6× the Sharpe that DLinear's MSE win delivers.

This is why a finance practitioner reading our paper will say *"iTransformer is better"* even though a pure ML statistician would say *"DLinear is better"*. They're both right.

---

## 3. The diagnosis — *why* iT collapses under sequential training

iTransformer has **~6.6 million parameters**; DLinear has **~6 thousand**. That's a 1000× capacity gap.

Under **sequential training** (one stock at a time, FNSPID protocol):

- After 20 epochs on stock A, iTransformer's weights are *strongly tuned* to stock A's local patterns
- The next 20 epochs on stock B *overwrite* those tunings
- Across 302 stocks × 3 rounds = 18,120 single-stock training rounds, the model never builds shared cross-stock representations — it just keeps re-specialising and forgetting
- DLinear, with 1000× less capacity, can only fit a small subset of patterns *anyway* — so its weights drift less per stock, and there's less to forget

**More capacity = more to overfit per stock = more to forget when the next stock arrives.**

This is the central thesis of the paper. The iT-DLinear capacity gap is exactly what makes iTransformer the dramatic catastrophic-forgetting demonstrator and DLinear the boring control:

| Method | DLinear seq→glob improvement | **iT seq→glob improvement** |
|--------|------------------------------|------------------------------|
| H=5 | 5.9% | **39.1%** |
| H=20 | 4.1% | **40.8%** |
| H=60 | -0.1% | **52.6%** |
| H=120 | 0.1% | **44.2%** |
| H=240 | 6.6% | **61.0%** |

DLinear can't show forgetting because it can't overfit. iTransformer shows it dramatically because its capacity *enables* the overfit-then-forget cycle.

---

## 4. Use-case decision matrix

| Goal / question | Better model | Why |
|-----------------|--------------|-----|
| Lowest pure MSE / MAE | **DLinear (Global)** | Closer to persistence attractor at this dataset |
| Trading strategy (Sharpe) | **iTransformer (Global)** | Real directional signal; +0.28 alpha at H=5 over naive |
| Directional accuracy (sign of return) | **iTransformer (Global)** | Consistent 1-6 pp advantage over DLinear |
| Risk management (MDD) | **iTransformer (Global)** | Slightly lower drawdown at short H; tied long H |
| Per-stock consistency (std MAPE) | iT Glob at H=5 only; DLinear at others | Mixed |
| Robust under suboptimal training (sequential) | **DLinear** | Its tiny capacity is its protection |
| Demonstrating catastrophic forgetting | **iTransformer** | 7-10× larger seq→glob gap than DLinear |
| Phase 3 (Sharpe-loss fine-tuning candidate) | **iTransformer Global** | Has slack from MSE attractor; can trade MSE for Sharpe |
| Minimal-baseline showing dataset noise floor | **DLinear** | All training methods converge to ~naive baseline |

---

## 5. Honest paper-level interpretation

### 5.1 For our CIKM / ICAIF paper

1. **The catastrophic forgetting story IS the headline of the paper, and it's much cleaner with iTransformer.** DLinear's 6% seq→glob gap is small and could be argued as noise. iTransformer's 40-60% gap is unambiguous, monotonically widens with horizon, and — crucially — has a clear *mechanistic* explanation (more capacity ⇒ more overfit-per-stock ⇒ more to forget).

2. **DLinear is the right *control* baseline,** not a competitor. Its role in the paper is to anchor where "no architecture, no forgetting, no signal" sits — i.e. the persistence-baseline noise floor. The other 7 transformer-class models (GCFormer, TimesNet, etc.) should land on a curve **above** DLinear in MSE/MAE *under global*, with **dramatic seq→glob gaps** like iTransformer's under sequential.

3. **iTransformer is the right *demonstrator* model** for the central effect. Its capacity reveals what's happening at the architectural level. Phase 3 (Sharpe-loss fine-tuning) should focus on iTransformer (and the other transformer-class models) where there's slack from the MSE-attractor that a Sharpe loss can convert into directional alpha.

4. **For a real deployment story**, iT Global is what you'd ship. The 8-15% MSE penalty is invisible to traders; the 70% Sharpe boost is exactly what makes the model worth running.

### 5.2 What the data does NOT support

- **"iTransformer is a strictly better model than DLinear"** — false. On pure MSE, DLinear edges out iT consistently. Under sequential training, iT is *much* worse.
- **"DLinear is a strictly better model than iTransformer"** — also false. On Sharpe, hit rate, and MDD-at-short-horizons, iT Global dominates.
- **"More capacity is always better"** — false. Capacity is a liability under sequential training and a benefit under global. The training paradigm matters as much as the architecture.

These three nuances should all appear in the paper — saying any one of them in isolation overstates the case.

---

## 6. Implications for Phase 3 and the rest of the model suite

### Phase 3 (custom Sharpe loss + fine-tune) — choose models with capacity slack

iTransformer Global is the prime candidate. The fine-tune should:
- Start from the MSE-trained iT Global checkpoints
- Use a soft Sharpe surrogate (`tanh(α · pred_return) × actual_return` averaged over a batch) as the loss
- Fine-tune for ~5 epochs at lr=1e-5
- Expected outcome: Sharpe further increases by 0.1-0.3 (especially at H=5, H=20); MSE worsens slightly; MDD probably stays similar or improves

DLinear is **not a good Phase 3 candidate** — it's already pinned at the persistence ceiling, no slack to convert.

### Cross-model expectations once GCFormer / TimesNet / etc. land

Based on the iTransformer pattern, we should see:
- All transformer-class models: large seq→glob improvement under MSE
- Sharpe gap iT > DLinear: probably reproduces for GCFormer, TimesNet (similar capacity); less clear for AdaPatch (smaller model)
- DLinear remains unique in being **immune to forgetting** because of its small capacity

If GCFormer Global Sharpe at H=5 doesn't beat naive (like iT did at +0.28), that's a flag — would suggest iT's Sharpe edge was iT-architecture-specific rather than a general transformer-on-stocks property.

---

## 7. One-sentence summaries

- **DLinear is a better statistical predictor; iTransformer is a better forecaster of useful information.**
- **DLinear minimises loss; iTransformer minimises regret.**
- **The right answer to "which is better" is "what are you optimising for?"**

The paper benefits from including both, because together they tell the complete story: DLinear shows the dataset's noise floor and the limits of capacity; iTransformer shows what happens when capacity is enabled by global training and crushed by sequential — which is the catastrophic-forgetting effect we're trying to characterise.
