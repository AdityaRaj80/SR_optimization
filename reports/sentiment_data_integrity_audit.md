# FinBERT Sentiment Data Integrity Audit — Critical Finding

**Date:** 2026-05-06
**Status:** MAJOR — affects every prior experimental result's *framing* but does not invalidate the architectural / loss-function findings themselves.
**Discovered by:** Aditya, on inspection of the per-stock CSVs in `D:\Study\CIKM\DATA\350_merged\`.

---

## 1. One-line summary

Across the entire FNSPID-derived `350_merged/` dataset that has fed every model trained in this project, the `scaled_sentiment` feature column is **uniformly equal to 0.5 across every stock, every date, every row** — a constant placeholder, not a real FinBERT output. The upstream `350_sentiment_results/*_finbert_tone.csv` files are also uniformly `(positive_prob=0.333, negative_prob=0.333, neutral_prob=0.333, composite_score=0)`, indicating FinBERT was never actually executed; the sentiment columns were initialised to a uniform prior and never populated.

---

## 2. Verified evidence

The findings below were reproduced on 2026-05-06 by direct inspection of the data files. Quoted numbers come from `pandas` aggregations on the actual CSVs; no values are inferred or estimated.

### 2.1 `350_merged/` per-stock CSVs

Spot-check across 8 stocks (mix of NAMES_50 test universe and broader train universe), totalling 58,732 rows:

| Stock | n_rows | unique values of `scaled_sentiment` | min | max | std |
|---|---|---|---|---|---|
| AAPL | 10,852 | **1** | 0.5 | 0.5 | 0.0000 |
| MSFT | 9,526 | **1** | 0.5 | 0.5 | 0.0000 |
| TSLA | 3,399 | **1** | 0.5 | 0.5 | 0.0000 |
| NVDA | 6,275 | **1** | 0.5 | 0.5 | 0.0000 |
| A (Agilent) | 6,066 | **1** | 0.5 | 0.5 | 0.0000 |
| AMD | 11,040 | **1** | 0.5 | 0.5 | 0.0000 |
| GOOG | 4,874 | **1** | 0.5 | 0.5 | 0.0000 |
| AMZN | 6,700 | **1** | 0.5 | 0.5 | 0.0000 |

Across all 8 spot-checked stocks: **every single row has scaled_sentiment = 0.5**. Standard deviation is exactly zero across all years (1980-present). 2020 specifically — when the original FNSPID news data was densest — also shows sentiment standard deviation = 0.0000.

### 2.2 `350_sentiment_results/*_finbert_tone.csv` raw FinBERT files

Spot-check across 4 stocks (A, AAP, ABT, ACN), totalling 14,864 articles:

| Stock | n_articles | positive_prob: range | composite_score: range | uniform-prior rows (0.333, 0.333, 0.333) |
|---|---|---|---|---|
| A | 2,748 | [0.333, 0.333] | [0.0000, 0.0000] | 2,748 / 2,748 (**100.0%**) |
| AAP | 3,674 | [0.333, 0.333] | [0.0000, 0.0000] | 3,674 / 3,674 (**100.0%**) |
| ABT | 4,055 | [0.333, 0.333] | [0.0000, 0.0000] | 4,055 / 4,055 (**100.0%**) |
| ACN | 4,387 | [0.333, 0.333] | [0.0000, 0.0000] | 4,387 / 4,387 (**100.0%**) |

Std of `positive_prob` across all 14,864 articles = 0.0000. Mathematically impossible if FinBERT had actually been run — real FinBERT inference on real news produces probabilities like (0.85, 0.10, 0.05) for a clearly positive article and (0.15, 0.70, 0.15) for a negative one, with std of order 0.2-0.3 across a corpus.

### 2.3 NAMES_50 (test universe) FinBERT files

Of the 49 NAMES_50 stocks, **0 have a `*_finbert_tone.csv` file in `350_sentiment_results/`**. The broken FinBERT pipeline was never even executed on the test universe — only on (a subset of) the train universe.

### 2.4 The model DID consume the sentiment column as input

`config.py` defines `FEATURES = ["Open", "High", "Low", "Close", "Volume", "scaled_sentiment"]`. The data loader passes 6-channel input to every backbone. Models have therefore been ingesting a 6-feature input where the 6th feature is identically 0.5 across every batch, every step, every epoch. A neural network cannot extract any predictive signal from a constant — `scaled_sentiment` has been computationally a no-op input.

---

## 3. What this means for prior experimental results

| Prior claim | Status |
|---|---|
| Track B (RiskAwareHead + composite Sharpe loss + risk-aware inference) improves portfolio Sharpe at H=60 by +1.97 vs MSE baseline (p<0.001, paired Politis-Romano stationary bootstrap) | **VALID** as a price-based result. The signal comes from OHLCV features, not sentiment. The loss-function and architectural contribution stands. |
| Theorem A1 (variance-decomposition lower bound on the σ-aware Sharpe edge) | **VALID** — the theorem is mathematical, independent of input features. |
| B1 (differentiable cross-sectional portfolio loss) negative result | **VALID** — failure mode independent of sentiment. |
| Stage 1 fan-out across 7 backbones (in progress) | **VALID** as a price-based architecture-fan-out, regardless of sentiment status. |
| Stage 2 FNSPID extension to 2024-2025 (price extension complete; sentiment placeholder = 0.5) | **CONSISTENT** with prior training — sentiment was already 0.5 throughout, so adding 2024-2025 with sentiment=0.5 is no methodological change. |
| Headline title / framing as "FNSPID + FinBERT-augmented" | **MUST BE CORRECTED** — false advertising. Any paper draft must drop "sentiment-augmented" / "FinBERT" claims unless a corrected pipeline is run. |

---

## 4. Root-cause hypothesis

The uniform (0.333, 0.333, 0.333) probabilities in the raw FinBERT files indicate one of:

1. The FinBERT inference function was a stub returning the uniform prior — never actually invoked on the article text.
2. FinBERT was loaded but its prediction call failed silently and a fallback prior was returned.
3. The aggregation step copied the uniform prior into every row regardless of the input article.

We cannot determine which without inspecting the original sentiment-extraction script (which is not in the current repo — it was a separate preprocessing step that produced the `350_sentiment_results/` files). For practical purposes the diagnosis is sufficient: **whatever code produced these files did not perform real FinBERT inference.**

---

## 5. Recovery options (decision matrix)

| Option | Description | Compute Time | Risk | Headline impact |
|---|---|---|---|---|
| A. Drop sentiment from paper | Reframe as price-based forecasting; document this audit as a pre-submission limitation | 0 days | Low | Removes the "FNSPID-augmented" angle; main loss-function contribution stands |
| B. Fix FinBERT, retrain everything | Real FinBERT on `All_external.csv`, rebuild caches, retrain all 7×5 backbones from scratch (MSE + Track B), re-evaluate, re-bootstrap | ~10-12 days HPC + few days writing | High (deadline risk for CIKM 2026) | Adds a real "sentiment vs non-sentiment" ablation as a NEW contribution; paper claims become correct |
| C. Hybrid | Run FinBERT on existing news, retrain only GCFormer (the headline architecture), report sentiment-vs-non-sentiment for one model | ~5-7 days | Medium | Smaller-scope sentiment ablation; some retraining needed |

The choice between A, B, and C trades off submission timeline (CIKM 2026 deadline May 23) against paper completeness and the value of the sentiment-vs-non-sentiment comparison.

---

## 6. Lessons + process change

For any future stage involving external feature pipelines, we must run a one-line variance check:

```python
df.groupby("Stock").scaled_sentiment.std().describe()
```

If `min == 0.000`, the feature is a constant for at least one stock — investigate before training. This audit took ~5 minutes to run and would have caught the issue at week 1 of the project rather than week 4.

---

*This document is committed to provide a permanent, dated, evidence-based record of when the issue was discovered, what was verified, and what the implications were. It is intended to be cited in any subsequent paper submission's limitations section, ensuring full transparency about the data pipeline.*
