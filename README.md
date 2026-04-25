# Financial Time-Series Forecasting with Sentiment Analysis

This repository contains the codebase for our CIKM applied research submission, focusing on benchmarking state-of-the-art (SOTA) time-series forecasting models on financial data. The project assesses the predictive power of various deep learning architectures on stock prices, augmented with FinBERT sentiment scores, across multiple forecasting horizons.

## Overview

The core objective is to conduct a rigorous, large-scale empirical study comparing advanced foundational time-series models (like Transformers, PatchTST, TimesNet, TFT) against standard recurrent baselines (LSTM, GRU, RNN). We emphasize evaluating these models not just on standard statistical loss (MSE/MAE), but on actionable financial metrics like the Sharpe Ratio and Maximum Drawdown.

---

## How We Are Approaching the Problem

We have structured the benchmarking framework to ensure an apple-to-apple comparison, strictly preventing data leakage and ensuring fair evaluation schemas.

### 1. Zero-Leakage Data Strategy
- **Dataset:** 351 stocks uniformly processed, combining historical OHLCV data with sentiment scores extracted using FinBERT. Stocks have a median history of **37.8 years** (min: 3.8 yrs, max: 61.9 yrs); sentiment coverage spans the full price history for all stocks.
- **Data Split:** To ensure zero data leakage, we divide the data into a **~300-stock training pool** and a **50-stock hold-out group**. The hold-out group is further chronologically split 50/50 for validation and out-of-sample testing.
- **Lookback Window:** `SEQ_LEN = 504` trading days (**2 calendar years**).
- **Forecast Horizons:** $H \in \{5, 20, 60, 120, 240\}$ trading days, corresponding to approximately **1 week, 1 month, 3 months, 6 months, and 1 year** ahead. The lookback-to-horizon ratio is ≥ 2:1 for all horizons, ensuring the model always sees at least twice as much history as it is asked to predict.

| Horizon | Trading Days | Real-world Period | Lookback:Horizon Ratio |
|---------|-------------|------------------|----------------------|
| H=5     | 5           | ~1 week          | 100:1                |
| H=20    | 20          | ~1 month         | 25:1                 |
| H=60    | 60          | ~3 months        | 8.4:1                |
| H=120   | 120         | ~6 months        | 4.2:1                |
| H=240   | 240         | ~1 year          | 2.1:1                |

### 2. Training Methodology
- **Global-Pool vs. Sequential:** We benchmark learning paradigms by comparing Global-pool training (training a single model on the joint distribution of all stocks) versus Sequential round-based training.
- **Target Variable:** All models are standardized to predict the future sequence of the `Close` price.

### 3. Model Architectures
We benchmark 8 unique architectures:
- **Foundational / Attention-based:** Vanilla Transformer, TimesNet, PatchTST, Temporal Fusion Transformer (TFT).
- **Recurrent Baselines:** LSTM, GRU, RNN.

---

## What We Have Done

1. **Pipeline Scaling & Sentiment Integration:** 
   - Successfully scaled the financial sentiment extraction pipeline from 50 to 350+ stocks.
   - Built robust data loaders (`data_loader.py`) integrating Kaggle-sourced sentiment data seamlessly with market data.

2. **Model Implementation & Standardization:**
   - Replicated and integrated exact, apple-to-apple architectures of SOTA models (Transformer, TimesNet, PatchTST, TFT) inspired by foundational work like FNSPID.
   - Standardized the model I/O dimensions to ensure output predictions exclusively target the required sequence lengths for the Close price.

3. **Optimization & Infrastructure:**
   - Resolved critical shape mismatch, module import, and device placement errors.
   - **Cross-Platform GPU Optimization:** Implemented a robust device-handling mechanism allowing seamless execution of the training pipeline locally (standard CUDA) and on cloud-based H100 HPC clusters via specific device arguments.
   - Consolidated configurations and training schemas (`config.py`, `train.py`) to streamline execution.

---

## What Remains To Be Done

1. **Full-Scale Execution:** Complete the execution of the training pipeline across the entire matrix of 350 stocks, all 8 models, and all 5 horizon windows sequentially.
2. **Financial Evaluation Metrics:** Finalize and execute the rigorous evaluation calculations (`utils/metrics.py`) focusing on quantitative finance metrics:
   - Annualized Sharpe Ratio
   - Maximum Drawdown
   - Cumulative Returns
3. **Hyperparameter Fine-Tuning:** Analyze initial benchmark runs and iteratively optimize specific parameters (e.g., expanding context windows and patch lengths for PatchTST long-term forecasting).
4. **Result Compilation & Analysis:** Aggregate the results into comprehensive comparative analyses and visualizations.
5. **Finalize CIKM Paper:** Formalize the methodology, visualizations, insights on global vs. sequential training, and architectural ablation studies into the final academic submission.

---

## Repository Structure Overview

*   `config.py`: Centralized configuration handlers (hyperparameters, paths, horizons).
*   `data_loader.py`: Utilities for loading, splitting, and scaling the 350-stock sentiment dataset.
*   `models/`: Core implementations of the 8 forecasting architectures.
*   `layers/`: Custom layers and building blocks for specific models (e.g., Transformer Encoders/Decoders, Embeddings, Full Attention components).
*   `engine/` & `utils/`: Training engines, early stopping handlers, metric calculations (MSE, MAE, Sharpe).
*   `train.py` & `run_all.py`: Main execution loops for single model testing and full cross-model benchmarking.

