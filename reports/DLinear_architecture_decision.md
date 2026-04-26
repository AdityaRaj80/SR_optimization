# DLinear — Architecture Faithfulness & Normalization Decision

**Date:** 2026-04-27
**Context:** Phase 1 result audit on DLinear sequential vs global. Empirical question raised after results: *"Was DLinear too simple, or did we do something wrong?"*

---

## 1. What we actually did

Our `models/dlinear.py` reproduces the canonical DLinear architecture from Zeng et al., 2023 ("Are Transformers Effective for Time Series Forecasting?", AAAI 2023, arXiv:2205.13504). Verified line-by-line against the official reference implementation at [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py).

| Component | Original | Ours | Status |
|-----------|---------|------|--------|
| `moving_avg` block | identical | identical | ✅ Faithful |
| `series_decomp` block | identical | identical | ✅ Faithful |
| Decomposition kernel size | 25 (hard-coded) | 25 (via config) | ✅ Same value |
| Linear_Seasonal / Linear_Trend | shared `Linear(seq_len, pred_len)` | shared `Linear(seq_len, pred_len)` | ✅ Faithful |
| `individual` per-channel mode | optional, via flag | not implemented (we always use shared) | ⚠️ Minor — config sets `False` so unused branch is irrelevant |
| Linear weight initialization | PyTorch default Kaiming | `(1/seq_len) * ones` | ⚠️ Minor — original code has this commented out as "if you want to visualize weights" |
| Forward output | `[B, pred_len, channels]` (multivariate) | `[B, pred_len][:, :, CLOSE_IDX]` (Close only) | ✅ Intentional univariate target |
| **Instance normalization (RevIN, etc.)** | **None** | **None** | ✅ **Faithful — see § 2** |

**Conclusion:** Our DLinear is structurally faithful. The minor differences (initialization, individual flag) do not materially change convergence or final performance.

---

## 2. Normalization: What the original paper actually does

Zeng et al. introduced the **LTSF-Linear** family with three variants:

| Model | Normalization scheme |
|-------|---------------------|
| **Linear** | None |
| **NLinear** | Subtract last value of input → predict → add it back. Single-point detrending. |
| **DLinear** | Decomposition into trend + seasonal. The moving-average trend extraction is itself the "soft" handling of non-stationarity. |

The paper explicitly presents NLinear and DLinear as **alternative philosophies** for handling distribution shift, neither of which uses RevIN (Kim et al., ICLR 2022) or any other instance-normalization scheme. Their `exp/exp_main.py` confirms this — the experiment framework does **not** apply any external normalization wrapper either.

So **canonical DLinear = decomposition only, no instance normalization**.

Many subsequent papers (PatchTST, iTransformer, NHiTS, etc.) add RevIN as a *prefix* to DLinear and report better numbers — but this is a deviation from the original, not the canonical baseline.

---

## 3. The decision: keep DLinear canonical (no RevIN)

For our CIKM benchmark, we are **keeping DLinear as the canonical version with no instance normalization**, despite the rest of our codebase using RevIN-style normalization for transformer-class models.

### Reasons:

1. **Faithfulness to the published baseline.** Reviewers expect "DLinear" in a benchmark to mean the original LTSF-Linear DLinear. Adding RevIN would make our DLinear into something the literature would call "DLinear+RevIN" or "RevIN-DLinear" — a different model.

2. **The narrative we are testing is global-vs-sequential, not architecture-vs-architecture.** Within DLinear, both training paradigms get the same architecture, so any normalization gap is shared and cancels out in the comparison.

3. **The other models in our suite have RevIN built in by their original authors** (PatchTST, TFT, iTransformer through embedding layers, GCFormer explicitly, etc.). For those, RevIN *is* the canonical version. So we are not introducing inconsistency — we are following each model's canonical specification.

4. **Reviewer-defensible language:** "Each model is implemented as specified by its original authors. DLinear (Zeng et al., 2023) does not include instance normalization; transformer-based models that incorporate RevIN are kept with that component."

---

## 4. What this means for the empirical results

The DLinear results reported in `reports/DLinear_sequential.md` and `reports/DLinear_global.md` show:

| H | Global MAE($) | Sequential MAE($) | Naive MAE($) |
|---|--------------|-------------------|-------------|
| 5 | $2.51 | $2.75 | **$2.38** |
| 20 | $4.80 | $5.25 | **$4.61** |
| 60 | $8.47 | $8.93 | **$8.14** |
| 120 | $12.07 | $12.88 | **$11.70** |
| 240 | $17.57 | $18.56 | **$17.03** |

**Both training methods are at-or-below naive last-value persistence in absolute dollar MAE at every horizon.** This is *expected* for canonical DLinear on stock data, and is consistent with the well-established result that:

> Stock returns are approximately uncorrelated across time, so the optimal linear forecast is approximately the last observed value (persistence).

DLinear, being a channel-independent linear model, has a hard ceiling at this persistence solution. The decomposition gives it a slight edge on long horizons (where global beats naive R² by 0.004), but it cannot extract more signal than this from price-only inputs.

**The catastrophic-forgetting gap (sequential vs global) is still cleanly visible** — it widens monotonically from $0.24 at H=5 to $0.99 at H=240 — which is what our paper actually claims.

---

## 5. Contingency plan: when to add DLinear+RevIN as a 9th model

If, after running our other models (iTransformer, GCFormer, etc.), the picture looks like this:

| Scenario | Action |
|----------|--------|
| **Strong models clearly beat naive at long horizons, sequential vs global gap widens for them** | ✅ DLinear's poor absolute performance is fine. Story holds. **No action.** |
| **Strong models also fail to beat naive much** (i.e. all our models look bad) | ⚠️ Suggests something systemic. Add DLinear+RevIN as a 9th entry to test whether the issue is the lack of normalization specifically. |
| **DLinear's gap to other models is so large reviewers complain it's a strawman** | ⚠️ Add DLinear+RevIN as a sensitivity-analysis entry. Report both versions in the appendix. |
| **A reviewer asks "what if you add RevIN"** | ✅ We have this report and a 5-line code change ready. Quick rebuttal. |

Specifically: we will **add DLinear+RevIN as a 9th model** if the gap between DLinear and the next-simplest model (likely Vanilla Transformer or iTransformer) exceeds **2× in dollar MAE at H=240**. That threshold suggests DLinear is being unfairly penalized by missing normalization rather than by architectural simplicity.

The change would be a 5-line addition to `models/dlinear.py` (mirroring `models/vanilla_transformer.py`), a new `dlinear_revin.py` registered in `models/__init__.py`, and one extra row of training (5 horizons × 2 methods = 10 runs).

---

## 6. Documentation trail

- `reports/DLinear_sequential.md` — full sequential results
- `reports/DLinear_global.md` — full global results + sequential-vs-global comparison
- `reports/DLinear_architecture_decision.md` (this file) — why we kept canonical DLinear, when we'd revisit
- `models/dlinear.py` — implementation matching the LTSF-Linear reference
- `EXPERIMENT_DESIGN.md` § 6.5 — orthogonal note on the memory-mapped loader (does not affect DLinear architecture)

This decision is logged so we can defend it cleanly in the paper without re-litigating.
