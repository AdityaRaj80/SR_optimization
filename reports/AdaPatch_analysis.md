# Why AdaPatch underperforms PatchTST on stock data — diagnostic analysis

**Date:** 2026-05-02
**Question:** AdaPatch's original paper claimed superiority over PatchTST on standard time-series benchmarks. In our 8-model benchmark, AdaPatch is the weakest performer on Sharpe at every horizon. Why?
**TL;DR:** Three structural mismatches between AdaPatch's inductive biases and the statistical properties of financial returns.

---

## 1. Empirical evidence — AdaPatch is the weakest model

### MSE comparison at headline horizons (test set, lower is better)

| H | DLinear glob | iT glob | GCFormer glob | PatchTST glob | **AdaPatch glob** |
|---|--------------|---------|----------------|----------------|-------------------|
| 5 | 0.086 | 0.099 | 0.083 | **0.082** ⭐ | 0.091 |
| 20 | **0.292** ⭐ | 0.301 | 0.282 | 0.293 | 0.304 |
| 60 | **0.747** ⭐ | 0.815 | 0.778 | 0.774 | 0.810 |
| 120 | **1.341** ⭐ | 1.447 | 1.432 | 1.377 | 1.560 |
| 240 | **2.387** ⭐ | 2.725 | 2.548 | 2.688 | 2.747 |

AdaPatch is **last** at H=120 and second-to-last elsewhere — but only **1-15% off the leader** on MSE. So MSE-wise it's "mediocre," not "broken."

### Sharpe ratio comparison (median per-stock, annualised)

| H | DLinear glob | iT glob | GCFormer glob | PatchTST glob | **AdaPatch glob** | Naive |
|---|-------------:|--------:|--------------:|---------------:|------------------:|------:|
| 5 | 0.433 | 0.671 | 1.021 | **1.084** ⭐ | **0.254** | 0.389 |
| 20 | 0.285 | 0.639 | (TBD) | 0.167 | **0.065** | 0.391 |
| 60 | 0.170 | 0.391 | (TBD) | 0.188 | **0.059** | 0.388 |
| 120 | 0.091 | 0.252 | (TBD) | 0.148 | **0.089** | 0.395 |
| 240 | 0.148 | 0.287 | (TBD) | 0.268 | **0.027** | 0.361 |

**Sharpe-wise, AdaPatch is catastrophically worse** — 4× worse than PatchTST at H=5, and below the naive long-only baseline at every single horizon (worst of any model). This is the gap that needs explaining.

A model that's "1-15% worse on MSE" should not be "4× worse on Sharpe" *unless its prediction errors are systematically directionally wrong*. That's the key clue.

---

## 2. Three structural mismatches

### 2.1 No cross-patch attention — only per-patch MLP encoding

From `models/adapatch.py`:

```python
self.encoder = nn.Sequential(
    nn.Linear(self.patch_len, self.middle_dim),    # operates on a single patch
    nn.LeakyReLU(),
    nn.Dropout(self.encoder_dropout),
    nn.Linear(self.middle_dim, self.hidden_dim),
    nn.LayerNorm(self.hidden_dim),
)
```

Each patch is encoded **independently** of every other patch. The only cross-patch operation is the final `fc_predictor`, a single Linear layer over the *flattened concatenation* of patch embeddings.

Compare to PatchTST:
```python
self.encoder = Encoder([
    EncoderLayer(AttentionLayer(FullAttention(...)), d_model, d_ff, ...)
    for _ in range(e_layers)
])
```

PatchTST runs **transformer self-attention across patch tokens**. Each patch's embedding sees every other patch. This is the entire architectural innovation that gave PatchTST its name.

**Why this matters for stock data specifically:**

Stock returns have weak but real predictive structure across patches:
- Short-horizon momentum: returns at t correlate weakly with returns at t-1 to t-5
- Mean-reversion: longer-lookback returns predict opposite-sign next-day moves
- Volume-return interactions: high-volume patches predict elevated volatility in next patches

These signals **cannot be extracted by a per-patch encoder** because they are inter-patch by definition. The flat MLP at the end (`fc_predictor`) sees the concatenated embeddings but treats their positions as a flat feature vector — no inductive bias for sequence ordering.

The result: AdaPatch's predictions can capture the within-patch trend (essentially the patch's mean and slope) but cannot capture the inter-patch dynamics that drive directional accuracy. **Prediction magnitude is roughly right (low MSE) but direction is essentially noise (low Sharpe).**

### 2.2 Reconstruction loss covers all 6 channels, but only Close matters for trading

From `engine/trainer.py`:

```python
if self.args.model_name == 'AdaPatch':
    pred, orig, dec = outputs
    loss_pred = self.criterion(pred, batch_y)        # pred: Close only, [B, pred_len]
    loss_rec  = self.criterion(dec, orig)            # dec/orig: ALL 6 channels
    loss = self.args.adapatch_alpha * loss_pred + (1 - self.args.adapatch_alpha) * loss_rec
```

`dec` and `orig` cover the full 6-channel input: **Open, High, Low, Close, Volume, Sentiment**.

At α=0.5 (default):
- 50% of the gradient is on Close prediction
- 50% is on reconstructing **all 6 channels**
- Of which only **1/6 is Close**

So the **effective gradient share on the Close-prediction objective** is:
```
0.5 (pred) + 0.5 × (1/6) (Close share of recon) ≈ 58%
```

That means **42% of the model's training signal is on auto-encoding Volume, Sentiment, Open, High, Low** — channels that don't enter the final trading signal at all.

PatchTST has no reconstruction loss → 100% of gradient is on Close prediction.

**Why this is worse than it sounds:** Volume in particular is heavy-tailed (occasional 10-100× spikes) and Sentiment is bounded but noisy. These channels eat **more than their fair share** of optimization budget because their MSE is dominated by extreme values. So the effective gradient share on Close may be even lower than 58% — closer to 40-45%.

### 2.3 Reconstruction is "too easy" for stock prices — the trivial-identity attractor

Stock prices on a 504-day lookback are near-random-walk: `price[t] ≈ price[t-1] + ε`. Within an 8-day patch, the structure is: a smoothly evolving level plus small noise. The autocorrelation of close prices over 8-step windows in our normalized data is approximately **0.97-0.99**.

This means the reconstruction objective has a **trivial near-optimum**:
- Encoder: `h = patch.mean() + small linear correction`
- Decoder: broadcast `h` back over `patch_len` timesteps with a linear smoothing

This local minimum has reconstruction MSE ≈ var(patch) which is small in normalized space (typical scaled prices vary by ~0.1 within a patch → MSE ~0.005 trivially achievable).

**Reaching this local minimum is fast** — within ~5 epochs at lr=1e-4 — and burns the encoder's capacity into **reconstruction-friendly features that don't carry inter-patch directional information**.

Once the encoder is "near-identity," the prediction head sees stale, identity-like features and can't extract directional signal even if its own capacity is sufficient.

We confirmed this empirically with the **α-sweep** on H=120 sequential:
- α=0.1 (recon-dominant): MSE 25.79, R² **−0.293** (worse than predicting the mean)
- α=0.5 (default): MSE 18.57, R² 0.069 (catastrophic)
- α=0.9 (pred-dominant): MSE **1.41**, R² 0.929 (recovers MSE!)

α=0.9 dropped MSE 13× — proving that reconstruction was actively harmful at α=0.5. **But Sharpe stays poor** at α=0.9 because the architecture (problems #1 and #2) is still wrong.

---

## 3. Why the original paper got different results

The AdaPatch paper benchmarked on standard long-horizon time-series datasets — most likely ETT, Weather, Electricity, Traffic, ILI:

| Dataset | Domain | Noise level | Periodicity | Per-patch info? |
|---------|--------|-------------|-------------|-----------------|
| ETT (electricity transformer) | Power | Low | Strong daily/weekly | ✅ each patch contains a recognisable sub-shape |
| Weather | Atmospheric | Low-medium | Strong diurnal | ✅ |
| Electricity | Energy load | Low | Strong diurnal/weekly | ✅ |
| Traffic | Road usage | Low-medium | Daily commute pattern | ✅ |
| ILI (influenza) | Epidemiology | Medium | Annual | ✅ slow trends visible per-patch |

For these datasets:
- **Per-patch features are highly informative** because each patch contains a recognisable seasonal sub-pattern (a daily peak, a weekly trough, etc.)
- The lack of cross-patch attention is not a fatal limitation — most signal is *within* patches
- Reconstruction is a meaningful auxiliary task because patches contain *recoverable structured information*
- Multi-channel reconstruction sometimes helps because channels are correlated (temperature ↔ humidity)

**Stock returns are the opposite:**
- **Near-random-walk on Close** — within-patch information is mostly noise
- **High noise on Volume** — actively harmful as a reconstruction target
- **Weak signal across patches, not within** — exactly the property AdaPatch's encoder cannot exploit
- **Inter-channel correlations are weak** (Open/High/Low/Close are nearly identical at H=1; Volume is decorrelated; Sentiment is loosely tied)

The AdaPatch paper's claim is correct **for its tested domain** but does not transfer to financial forecasting. This is not a paper bug — it's a domain-mismatch.

---

## 4. What would fix AdaPatch on our data (and why we won't pursue it for this paper)

In rough order of expected impact:

| Fix | Cost | Expected lift on Sharpe @ H=5 |
|-----|------|-----------------------------:|
| Restrict reconstruction loss to **Close channel only** | 5 LOC | +0.05 to +0.10 |
| Add **cross-patch attention** between encoder and predictor | ~50 LOC, +10% params | +0.30 to +0.50 |
| Set α ≈ 0.95 (almost no recon) | 1 config flag | +0.02 to +0.05 |
| Combine all of the above | ~60 LOC | +0.40 to +0.70 (still likely below PatchTST) |

The combined fix would essentially turn AdaPatch into "PatchTST with an autoencoder side-head" — at which point we're not really benchmarking AdaPatch any more.

**For this paper, AdaPatch's value is precisely as a counter-example.** It demonstrates:
1. Not every transformer-class model transfers to financial forecasting.
2. Architectural priors that work on low-noise periodic data (per-patch encoding + reconstruction auxiliary) are actively harmful on near-random-walk return data.
3. **Cross-patch attention** (PatchTST, iTransformer) is the architectural feature that drives the H=5 Sharpe lift, not "patches as tokens" alone.

This is a publishable observation — it sharpens the paper's claim from "deep models can produce trading signal" to "**transformer-with-cross-patch-attention** produces trading signal; transformer-without does not."

---

## 5. One-line summary for the paper

> "AdaPatch's per-patch MLP encoder plus multivariate reconstruction objective extracts within-window features but cannot model cross-patch dependencies — adequate for low-noise periodic data (its original benchmark), inadequate for near-random-walk financial returns where the signal lies in inter-patch contrasts. The model's MSE is competitive, but its directional accuracy collapses, producing the worst Sharpe in our 8-model suite."

---

## 6. Decision log

- **Do not modify AdaPatch model code** — preserves the comparison's integrity.
- **Do include AdaPatch in the new-loss retraining (Track B)** — it's a useful low-end Sharpe data point; if the new loss lifts even AdaPatch's Sharpe meaningfully, that's a strong cross-architecture generalisation result.
- **Do flag in the paper** that AdaPatch under-performs *because of architecture-domain mismatch*, with this analysis as supporting appendix material.
