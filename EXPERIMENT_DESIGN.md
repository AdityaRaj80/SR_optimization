# Experiment Design Report

**Project:** Benchmarking Time-Series Forecasting Architectures on Financial Data — Global vs. Sequential Training Paradigms
**Target Venue:** CIKM / ICAIF
**Last Updated:** 2026-04-26

---

## 1. Purpose of This Document

This report explains the rationale behind every key experimental hyperparameter chosen for the benchmark study. Each decision is justified with reference to (a) the structure of the underlying data, (b) standard practice in the time-series forecasting literature, and (c) the specific experimental hypothesis we are testing — namely, whether **sequential round-based training** suffers measurable catastrophic forgetting compared to **global-pool training** on financial time-series.

---

## 2. Dataset Characterization

### 2.1 Stock Pool

| Property | Value |
|----------|-------|
| Total stocks | 351 |
| Training pool | ~301 stocks (all except `NAMES_50`) |
| Validation/Test pool | 50 hold-out stocks (`NAMES_50`) split chronologically 50/50 |
| Features | `[Open, High, Low, Close, Volume, scaled_sentiment]` |
| Target | future `Close` price sequence |
| Data end date | 2023-12-28 |

### 2.2 Data Availability

A full audit of the stock pool was performed (`D:/Study/CIKM/DATA/350_merged`) to establish the upper bound on usable history:

| Metric | Trading Days | Years |
|--------|------------|-------|
| Shortest stock (OTIS) | 952 | 3.8 |
| Longest stock (KO – Coca-Cola) | 15,606 | 61.9 |
| **Median** | **9,527** | **37.8** |
| Mean | 8,626 | 34.2 |

**Key finding:** Sentiment data covers the **full price history** for every one of the 351 stocks — the sentiment feature does not constrain the usable training window. We are therefore free to pick any `(SEQ_LEN, horizon)` pair as long as the shortest stock can produce at least one valid `(input, target)` pair.

---

## 3. Lookback Window: `SEQ_LEN = 504` (2 trading years)

### 3.1 Previous setting

`SEQ_LEN = 252` (1 trading year)

### 3.2 New setting

`SEQ_LEN = 504` (2 trading years)

### 3.3 Justification

A foundational rule in time-series forecasting is that the **lookback window must be substantially larger than the forecast horizon**, otherwise the model is being asked to extrapolate further than it can see. The standard rule of thumb in the literature is **lookback ≥ 2× horizon**, with 3–4× preferred for harder long-term forecasts.

With our previous `SEQ_LEN = 252` and the previous longest horizon `H = 240`, the ratio was 1.05:1 — essentially the model was asked to predict almost as far as it could see. This is **not a defensible experimental setting** for a CIKM/ICAIF submission and would draw immediate reviewer pushback.

Doubling the lookback to `SEQ_LEN = 504` achieves three things:

1. **Captures multiple seasonal cycles** — annual (~252 days), quarterly earnings (~60 days), and monthly (~20 days) patterns are all observable within a single lookback window.
2. **Maintains a ≥ 2:1 ratio** for every horizon in the experimental matrix, which is the minimum defensible setting in the long-horizon TS-forecasting literature.
3. **All 351 stocks survive** — even the shortest stock (OTIS, 952 days) supports `SEQ_LEN = 504 + H = 240 = 744 days` of required input.

### 3.4 Why not go even longer (e.g., 756, 1008)?

A 3-year (`SEQ_LEN = 756`) lookback would drop OTIS and a handful of other newer stocks, weakening the training pool. It would also roughly **3× the per-step memory cost** of attention layers (transformer attention is O(L²)), substantially slowing training without a clearly demonstrated benefit on this dataset. `SEQ_LEN = 504` is the **largest defensible value that keeps the full 351-stock pool intact**.

---

## 4. Forecast Horizons: `H ∈ {5, 20, 60, 120, 240}`

### 4.1 Previous setting

`HORIZONS = [3, 10, 40, 120, 240]`

### 4.2 New setting

`HORIZONS = [5, 20, 60, 120, 240]`

### 4.3 Mapping to financial reality

| Horizon | Trading Days | Real-world Period | Why this horizon matters |
|---------|-------------|-------------------|--------------------------|
| H=5 | 5 | ~1 week | Short-term trading, swing entry/exit |
| H=20 | 20 | ~1 month | Most-cited horizon in finance literature; aligns with monthly portfolio rebalancing |
| H=60 | 60 | ~3 months / 1 quarter | Quarterly earnings cycle; matches institutional reporting periods |
| H=120 | 120 | ~6 months | Semi-annual outlook; typical mid-term investment horizon |
| H=240 | 240 | ~1 year | Annual investment outlook; classical finance benchmark |

### 4.4 Why these specific horizons (and not the previous ones)?

- **H=3 → H=5** — A 3-day horizon has no clean financial interpretation, while H=5 maps cleanly to a trading week. This makes the result table easier to discuss in the paper.
- **H=10 → H=20** — H=10 (~2 weeks) is a non-standard horizon; H=20 (~1 month) is the **single most-cited horizon in finance literature** and aligns with monthly portfolio rebalancing cycles.
- **H=40 → H=60** — H=40 (~2 months) again has no clean financial mapping; H=60 (~1 quarter) aligns with the earnings cycle, which is the dominant medium-term driver of stock prices.
- **H=120 unchanged** — already maps cleanly to 6 months (semi-annual outlook).
- **H=240 unchanged** — already maps cleanly to 1 year (annual outlook).

The new horizon ladder maps **1 week → 1 month → 1 quarter → 6 months → 1 year**, which is the canonical decomposition used in finance papers and gives reviewers an immediately interpretable result table.

### 4.5 Lookback-to-horizon ratio for the new setting

| Horizon | Ratio (504 ÷ H) | Defensibility |
|---------|----------------|---------------|
| H=5 | 100.8 : 1 | Trivial |
| H=20 | 25.2 : 1 | Excellent |
| H=60 | 8.4 : 1 | Strong |
| H=120 | 4.2 : 1 | Comfortable |
| H=240 | 2.1 : 1 | Minimum standard |

Every cell in the matrix exceeds the minimum 2:1 ratio threshold from the literature.

---

## 5. Training Epochs

### 5.1 Previous setting

- Global: `--epochs 30` with `patience 10`
- Sequential: `10 epochs/stock × 3 rounds = 30 exposures per stock`

### 5.2 New setting

- Global: `--epochs 60` with `patience 10`
- Sequential: `20 epochs/stock × 3 rounds = 60 exposures per stock`

### 5.3 The fairness principle

For the global-vs-sequential comparison to be scientifically meaningful, each method must be given **equivalent training signal per stock**. The natural metric is **number of times each stock is presented to the model during training**:

- **Global at 60 epochs:** every stock is presented 60 times (once per global epoch).
- **Sequential at 20 epochs/stock × 3 rounds:** every stock is presented 60 times.

Both methods produce approximately the **same total number of gradient update steps (~180k)** when batched at `batch_size=512`, with the same learning rate schedule. The only experimentally-controlled difference is **the order in which those updates occur** — interleaved across all stocks (global) vs. concentrated on one stock at a time (sequential). This is precisely the variable we want to isolate.

### 5.4 Why 60 (not 30, not 100)?

Three considerations:

1. **Convergence headroom for transformers.** Most transformer-based time-series models in the literature use 50–100 max epochs (PatchTST: 100, TFT: 100, Vanilla Transformer: 50–100). With early stopping (patience=10), a model that converges in 20 epochs still stops at 20 — but a model that genuinely needs 45 epochs now has the opportunity to reach it.

2. **FNSPID compatibility.** The FNSPID paper — the most directly comparable prior work using sequential round-based training on stock data — used **20 epochs/stock × 3 rounds = 60 exposures**. Matching this exactly gives our paper a direct, citable baseline:

   > "Sequential training follows the FNSPID protocol (20 epochs/stock × 3 rounds), and global training is matched at 60 max epochs to maintain equivalent training signal per stock."

3. **Early stopping is a safety net.** Increasing `--epochs` from 30 to 60 has near-zero cost for fast-converging models (they still stop at 15–25 with patience=10) but rescues models that legitimately need more epochs at long horizons.

### 5.5 Asymmetric early-stopping behavior (intentional)

- **Global** uses early stopping on validation loss (patience=10).
- **Sequential** does *not* use early stopping — it always runs the full 60 exposures per stock.

This asymmetry is **deliberate and central to the paper's hypothesis**:

- Sequential training continues to update the model long past the point at which a global-trained model would have converged.
- This continued single-stock training is precisely the mechanism that produces catastrophic forgetting: the model overfits to the most recent stock and progressively loses representation of earlier stocks.
- Adding early stopping to sequential would mask the very effect the paper is trying to measure.

---

## 6. Rounds: `--rounds 3` (sequential only)

### 6.1 Setting

3 rounds, unchanged from the previous setting.

### 6.2 Justification

A "round" in sequential training is a single sweep over all 301 training stocks, with the model state carried forward from one stock to the next. The number of rounds controls how many times the model **revisits** each stock.

- **1 round:** single pass — maximum forgetting effect, but unrepresentative of how sequential learning is actually deployed in practice.
- **3 rounds:** standard FNSPID protocol — gives the model two opportunities to "re-learn" earlier stocks, providing a realistic but still forgetting-prone setting.
- **>3 rounds:** approaches the limit of repeated exposure, which would gradually mitigate forgetting and weaken the contrast with global training.

3 rounds is the **canonical setting** in the existing sequential-finance literature and gives our results direct comparability with FNSPID.

---

## 6.5 Implementation: Memory-Mapped Global Dataset

The eager global loader (`get_global_train_loader`) materialises every
training stock's sequences into a single contiguous numpy array in RAM.
At `SEQ_LEN = 504` and ~302 training stocks this is ~18 GB, which exceeds
the 16 GB RAM available on the local development laptop and causes OOM
crashes. The same data fits comfortably on the JarvisLabs A100 (80 GB
host RAM) and on H100 nodes, but we want a single implementation that
runs on any environment without code switches.

### What we did
- Added `preprocess_global_cache.py`: pre-scales each training stock with
  its own `MinMaxScaler` (identical logic to the eager path) and writes
  the result to `.cache/global_scaled/<stock>.npy`. Runs once.
- Added `GlobalMmapDataset` in `data_loader.py`: a `torch.utils.data.Dataset`
  that holds one `np.memmap` per stock plus a compact int32/int64 index
  of every valid `(stock, start_offset)` pair. `__getitem__` reads a
  small slice via mmap; OS pages handle caching transparently.
- Added `UnifiedDataLoader.get_global_train_loader_mmap()` and made it
  the default in `train.py` (legacy eager path is reachable via
  `--use_eager_global`).

### Equivalence guarantee
We ran a strict element-wise sanity check with `--max_stocks 5, H=60`:
all 34,916 sequences from the eager pipeline matched the mmap pipeline
**bit-for-bit** (max diff = 0.0 on both X and y). The two pipelines see
identical data in identical order; the only difference is when bytes are
read from disk vs. RAM.

### Performance characteristics
- Cache build time: ~30 s for 302 stocks (one-time cost).
- Cache size on disk: ~60 MB (float32, scaled).
- RAM during training: bounded by OS page cache (a few hundred MB working
  set) regardless of dataset size.
- Per-epoch time on local RTX 3060 with 30 stocks: ~13 s/epoch (DLinear, H=20).
- On H100 with full 302 stocks, expected utilisation: 95%+ for transformer
  models (compute-bound), ~75–80% for DLinear (data-loading-bound but
  fine in absolute terms).

This fix is the same pattern as the lazy `iter_train_loaders` generator
introduced for sequential training — shared with that the property of
*pure implementation change with bit-for-bit identical data*.

---

## 7. Other Fixed Hyperparameters (unchanged)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `batch_size` | 512 | Matches GPU VRAM utilization on A100 (40GB) and RTX 3060 (6GB, only sequential local). |
| `lr` | 1e-4 | Standard AdamW starting point for transformers; not tuned per-model to keep comparison fair. |
| `lradj` | type3 | Cosine-style schedule, smooth decay. Avoids artificial step changes that could confound the catastrophic forgetting analysis. |
| `patience` | 10 (global only) | Standard early-stopping patience in TS-forecasting literature. |
| `optimizer` | AdamW | Standard for transformers; consistent across all 8 models. |
| `loss` | MSE | Phase 1 objective; Phase 3 will replace with custom Sharpe loss. |

---

## 8. Three-Phase Experimental Plan

### Phase 1 — MSE-optimized training (current)
Train all 8 architectures × 5 horizons × 2 methods (global, sequential) using standard MSE loss on the future Close price. Save checkpoints.

**Deliverable:** Full benchmark table of MSE / MAE / R² across the entire matrix.

### Phase 2 — Financial metric evaluation (post-processing)
Compute Sharpe Ratio and Maximum Drawdown directly from Phase 1 predictions. **No retraining required** — these are pure post-hoc metrics on price predictions.

**Deliverable:** Annualized Sharpe Ratio and MDD tables alongside MSE/MAE/R².

### Phase 3 — Sharpe-optimized fine-tuning (winning method only)
After Phase 2 reveals which training method (global or sequential) performs better, fine-tune the winning checkpoints with a **custom differentiable Sharpe Ratio loss** (with ε-stabilized denominator) and a **soft-max approximation of MDD** for the secondary objective. Warm-starting from Phase 1 checkpoints — full retraining from scratch is unnecessary.

**Deliverable:** Direct comparison of MSE-trained vs. Sharpe-trained models on financial metrics, demonstrating actionable improvement over standard statistical loss.

---

## 9. Summary of Key Decisions

| Decision | Old | New | Why |
|----------|-----|-----|-----|
| Lookback window | 252 (1 yr) | **504 (2 yr)** | Maintain ≥ 2:1 lookback:horizon ratio at H=240 |
| Horizon set | [3, 10, 40, 120, 240] | **[5, 20, 60, 120, 240]** | Map to clean financial periods (1wk, 1mo, 1Q, 6mo, 1yr) |
| Global epochs | 30 | **60** | Convergence headroom + match FNSPID-equivalent exposure count |
| Sequential epochs/stock | 10 | **20** | Match FNSPID exactly + give 60 exposures per stock |
| Sequential rounds | 3 | 3 (unchanged) | FNSPID-standard |
| Total exposures per stock | 30 (both methods) | **60 (both methods)** | Fair comparison principle preserved |

---

## 10. References to Prior Work

- **FNSPID** (Financial News and Sentiment-Powered Industrial Decisions) — provides the 20 × 3 sequential training protocol that our sequential setting mirrors exactly.
- **PatchTST**, **iTransformer**, **TimesNet**, **TFT**, **GCFormer** — original architecture papers; epoch budgets cited above.
- **DLinear** — used as the simple-linear baseline; demonstrates that lookback must be sufficient for even trivial models.
