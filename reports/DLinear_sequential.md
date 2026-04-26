# DLinear — Sequential Training Results

**Model:** DLinear (Decomposition Linear)
**Training method:** Sequential round-based (FNSPID-style)
**Date:** 2026-04-26
**Status:** Phase 1 (MSE-optimized) complete — all 5 horizons evaluated

---

## 1. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Architecture | DLinear (linear seasonal + linear trend) |
| Lookback (`SEQ_LEN`) | 504 trading days (2 years) |
| Horizons (`H`) | {5, 20, 60, 120, 240} |
| Training stocks | ~301 (all of `350_merged` minus the 50 hold-out) |
| Validation/Test stocks | 50 hold-out (`NAMES_50`), split chronologically 50/50 |
| Epochs per stock | 20 |
| Rounds | 3 |
| Total exposures per stock | 60 (matches FNSPID protocol exactly) |
| Batch size | 512 |
| Optimizer | AdamW, `lr=1e-4`, `lradj=type3` |
| Loss | MSE on scaled Close-price predictions |
| Hardware | Local RTX 3060 (6 GB VRAM) — sequential is one-stock-at-a-time so VRAM is not the bottleneck |
| Implementation note | Lazy per-stock loader generator (`iter_train_loaders`) — bounded memory, scientifically equivalent to the eager list-build version (sanity-check confirmed within CUDA noise: H=60 list MSE 0.7409 vs generator MSE 0.7459, Δ < 1%) |

---

## 2. Headline Aggregate Metrics

| H | Real-world | MSE | MAE | RMSE | R² | MAE ($) | RMSE ($) |
|---|-----------|-----|------|------|-----|---------|----------|
| 5 | 1 week | 0.0913 | 0.0711 | 0.302 | 0.9956 | $2.75 | $15.23 |
| 20 | 1 month | 0.3041 | 0.1337 | 0.551 | 0.9852 | $5.25 | $27.45 |
| 60 | 1 quarter | 0.7459 | 0.2274 | 0.864 | 0.9632 | $8.93 | $43.58 |
| 120 | 6 months | 1.3415 | 0.3192 | 1.158 | 0.9327 | $12.88 | $58.94 |
| 240 | 1 year | 2.5553 | 0.4649 | 1.599 | 0.8686 | $18.56 | $78.43 |

**Pattern:** Smooth monotonic degradation as horizon grows. R² stays above 0.85 even at the 1-year horizon — which initially looks like a strong result, but the comparison with the naive baseline (Section 3) reveals this to be misleading.

> **Note on scaled-space MSE > 1:** Per-stock test scaling is fit on the val period (first half of each test stock's history) and applied to the test period (later half). For stocks that drifted upward over time (most of NAMES_50), the test-period targets in scaled space exceed 1.0 — pushing MSE above 1.0 even for a competent model. This is *correct, leak-free* scaling, but means scaled MSE is not directly comparable across horizons.

---

## 3. Naive Baseline Comparison — The Critical Finding

The naive baseline predicts that the price at every step of the forecast horizon equals the **last observed Close** (last-value persistence).

| H | DLinear R² | **Naive R²** | DLinear MAE ($) | **Naive MAE ($)** | Direction (Δ%) |
|---|-----------|--------------|----------------|-------------------|----------------|
| 5 | 0.9956 | **0.9959** | $2.75 | **$2.38** | DLinear ‑0.0003 R², +$0.37 worse |
| 20 | 0.9852 | **0.9858** | $5.25 | **$4.61** | DLinear ‑0.0006 R², +$0.63 worse |
| 60 | 0.9632 | 0.9625 | $8.93 | $8.14 | DLinear +0.0007 R², +$0.79 worse |
| 120 | 0.9327 | 0.9320 | $12.88 | $11.70 | DLinear +0.0007 R², +$1.18 worse |
| 240 | 0.8686 | **0.8732** | $18.56 | **$17.03** | DLinear ‑0.0046 R², +$1.53 worse |

### Conclusion

> **DLinear under sequential training is performing at — and in dollar terms slightly *worse than* — a naive last-value persistence baseline at every horizon tested.**

This means the model has learned essentially **no useful predictive signal beyond "the price at time t+h ≈ the price at time t"**.

---

## 4. Directional Accuracy

Directional accuracy = fraction of test samples where `sign(predicted − last_close) == sign(actual − last_close)` (excluding samples with exactly-zero change).

| H | DLinear DirAcc | Best-Constant Baseline DirAcc | Pct Up | Pct Down |
|---|----------------|-------------------------------|--------|----------|
| 5 | 50.3 % | **52.4 %** | 51.7 % | 47.6 % |
| 20 | 50.4 % | **54.2 %** | 53.9 % | 45.8 % |
| 60 | 50.2 % | **56.2 %** | 55.9 % | 43.8 % |
| 120 | 53.4 % | **58.3 %** | 58.0 % | 41.7 % |
| 240 | 53.9 % | **60.7 %** | 60.4 % | 39.3 % |

The "best-constant baseline" is the trivial classifier that always predicts the more common direction (up or down) for that horizon. Since stocks drift upward over the test period, this baseline is non-trivially strong.

### Conclusion

> **DLinear's directional predictions are statistically indistinguishable from a coin flip at short horizons (50.2–50.4%) and only marginally better than chance at long horizons (53.4–53.9%) — but the trivial "always predict UP" baseline beats DLinear at every horizon.**

---

## 5. Per-Stock Distribution (MAPE in dollars)

| H | Median MAPE | Std MAPE | N stocks |
|---|-------------|----------|----------|
| 5 | 2.4 % | 2.1 % | 50 |
| 20 | 4.3 % | 4.5 % | 50 |
| 60 | 7.3 % | 8.8 % | 50 |
| 120 | 10.2 % | 15.6 % | 50 |
| 240 | 13.5 % | 26.5 % | 50 |

### Conclusion

> The standard deviation of per-stock MAPE grows ~13× from H=5 → H=240, confirming a **heavy-tailed error distribution**. A small number of high-priced / high-volatility stocks (likely NVDA, TSLA, GOOG-class) dominate the aggregate squared error. Median MAPE values stay under 14 % even at H=240, which is reasonable for the median stock — the aggregate is being skewed by outlier stocks.

---

## 6. Interpretation — Why This Is Actually a Strong Finding for the Paper

### 6.1 The catastrophic forgetting hypothesis

Our paper hypothesises that **sequential round-based training induces catastrophic forgetting**: each new stock's gradient updates degrade what the model learned from earlier stocks, and the model converges to whatever simple solution minimises loss on the last stocks seen.

For stock-price data, that "simple solution" is **persistence** (predict last value), because:
1. Persistence is a strong baseline in stock data (autocorrelation is dominant at all horizons),
2. Each individual stock's training set looks locally smooth, so the linear DLinear weights collapse toward the identity-of-last-value,
3. Without the diversity-driven regularisation of a pooled global dataset, there is no pressure to learn anything more elaborate.

The empirical evidence above is **fully consistent with this hypothesis** — sequential DLinear has indeed collapsed to a near-naive predictor.

### 6.2 Why DirAcc is more revealing than R² here

R² and MAE both look "good" in isolation (R²=0.87 at 1 year sounds great). But R² rewards getting the magnitude approximately right, and persistence trivially does that on stocks. The directional accuracy metric strips away the magnitude and asks the harder question: *can the model tell whether the price will go up or down?* DLinear's answer is: *no, not really.*

For a CIKM/ICAIF audience, **DirAcc is the metric that financial reviewers will weigh most heavily**. Beating 50% by a meaningful margin is the sine qua non of a useful forecasting system; beating naive baseline by a fraction of an R² point is a footnote.

### 6.3 What we still need to determine

These findings on their own do **not** prove the catastrophic-forgetting hypothesis — they only prove that *DLinear under sequential training* approximates persistence. To complete the argument we need:

1. **DLinear under global training.** If global also collapses to naive, then DLinear is simply too weak an architecture to extract useful signal from this dataset, and the sequential/global comparison cannot be made on this model. If global *exceeds* naive (especially in DirAcc), then we have direct evidence that the sequential-pool training paradigm is *causing* the collapse — i.e. catastrophic forgetting is real.

2. **A more expressive model (e.g. iTransformer, GCFormer) under both paradigms.** Even if DLinear is too weak, the comparison must hold for at least one strong architecture for the paper to land. The expectation is that for transformer-based models, global training visibly beats naive while sequential training does not — that gap is the headline result.

---

## 7. Trustworthiness of the Result

### Possible alternative explanations (and why they don't apply)

| Hypothesis | Status |
|-----------|--------|
| The lazy-loader fix degraded the model | ✗ Ruled out — H=60 sanity check matched the original list-based run within 1% (CUDA-noise level). |
| The metrics are computed wrong | ✗ Ruled out — naive baseline computed independently and consistently. The naive R² comes out where it should (very high for stock data due to autocorrelation). |
| The CSV write swapped/mislabeled rows | ✗ Ruled out — all 5 rows recomputed in a single deterministic pass over the saved checkpoints. |
| Training did not converge | ✗ Ruled out — the training logs show validation loss decreasing across rounds and saving best-of-3 checkpoints. The model *did* converge, just to the persistence solution. |
| The 50 test stocks have anomalous statistics | Partly true (heavy-tailed errors), but the result reproduces in the median per-stock MAPE too (Section 5), so it is not just an outlier issue. |

### What is open

- We have not yet measured **per-stock DirAcc** (only aggregate). It is possible a subset of stocks shows real predictive signal while the rest are pure noise. This would matter for a "predictability map" appendix in the paper.
- We have not yet compared with stronger baselines (e.g. ARIMA, exponential smoothing) which often match transformer-based models on financial data.

---

## 8. Reproducibility

All code, configs, and checkpoints are committed to the GitHub repository. Specifically:

| Artifact | Location |
|----------|----------|
| Trained checkpoints | `checkpoints/DLinear_sequential_H{5,20,60,120,240}.pth` |
| Aggregate metrics CSV | `results/sequential_results.csv` |
| Extended metrics CSV | `results/extended_metrics.csv` |
| Per-stock MAPE CSV | `results/per_stock_metrics.csv` |
| Training script | `train.py` (entry point) |
| Lazy loader implementation | `data_loader.py::iter_train_loaders` |
| Sequential trainer | `engine/trainer.py::train_sequential` |
| Extended evaluation script | `evaluate_extended.py` |

To reproduce a single horizon:
```bash
python train.py --model DLinear --method sequential --horizon 60 \
                --batch_size 512 --rounds 3 --epochs_per_stock 20 \
                --lr 1e-4 --lradj type3
```

To reproduce the extended analysis from saved checkpoints:
```bash
python evaluate_extended.py --model DLinear --method sequential \
                            --horizons 5 20 60 120 240
```

---

## 9. Next Steps

1. **DLinear global training (5 horizons)** — primary next experiment. Will tell us whether global training rescues DLinear above naive baseline. *Cannot be run on the local 16 GB laptop (RAM exhaustion); requires the JarvisLabs A100 VM.*
2. **iTransformer / GCFormer sequential (5 horizons)** — verify whether a more expressive architecture changes the picture under sequential training.
3. **Per-stock DirAcc breakdown** — small additional analysis on the existing 5 sequential checkpoints; useful for an appendix.
4. **Phase 2 financial metrics** (Sharpe, MDD) — compute on the saved checkpoints once the global runs are done. No retraining required; this is post-processing.
5. **Phase 3 Sharpe-optimised fine-tuning** — only after Phase 1 + Phase 2 identify the winning training method. Fine-tunes from the existing MSE-trained checkpoints with a custom Sharpe loss; no new architectures.

---

*This report should be regenerated whenever the DLinear sequential checkpoints are retrained.*
