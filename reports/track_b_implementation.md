# Track B Implementation Report

**Status:** Phases B + C + D complete and validated end-to-end.
**Date:** 2026-05-04
**Goal:** Replace MSE-only training with a Sharpe-aware composite loss + risk-aware
output head so the optimisation objective directly rewards risk-adjusted profitability,
not point-forecast accuracy. The result enters the paper as the *causal* test of
whether financial losses move the needle on portfolio Sharpe / max drawdown vs.
the dominant industry default (MSE on next-H closing price).

---

## 1. Scope

The Phase B/C/D work delivers, on top of the Track A backbones:

1. `engine/heads.py::RiskAwareHead` — backbone-agnostic wrapper that augments
   any existing model with two auxiliary scalar heads (predictive σ and forward
   realised vol).
2. `engine/losses.py::CompositeRiskLoss` — five-term composite loss with
   epoch-anchored phase schedule.
3. `engine/trainer.py` + `engine/evaluator.py` + `train.py` wiring — selecting
   the new objective is a single `--use_risk_head` flag, with an
   `--init_from <ckpt>` flag for fine-tuning from an existing MSE baseline.
4. `scripts/riskhead_glob.sbatch` + `scripts/submit_riskhead_campaign.sh` —
   HPC submission templates.
5. `tests/test_track_b.py` (5 unit tests) and `tests/test_trainer_risk_head.py`
   (6 integration tests) — fixture-level + trainer-level validation.
6. `tests/smoke_train_track_b.py` — 25-epoch synthetic smoke train asserting
   end-to-end learning behaviour.

The contract is deliberately additive: a model trained with `--use_risk_head`
saves to a checkpoint with `_riskhead.pth` suffix and writes results to a
parallel `<method>_results_riskhead.csv`. Existing MSE checkpoints + the
existing `<method>_results.csv` are never touched, so the apples-to-apples
comparison in the paper compares same-universe, same-split, same-evaluator
results from two CSV files.

---

## 2. RiskAwareHead (Phase B)

```
                   ┌────────────────────┐
   x_enc ──────────┤   backbone (any)   ├─── mu_close [B, H]
   [B, T, F]       └────────────────────┘
                            │
        ┌───────────────────┴────────────────────┐
        │                                        │
   last_window                              last_close
   x_enc[:, -L:, :]                         x_enc[:, -1, CLOSE]
        │
        ├──── sigma_head MLP ──── log_sigma2_H [B]
        │
        └──── vol_head   MLP ──── log_vol_pred [B]
```

**Forward output is a dict** with keys
`{mu_close, mu_close_H, mu_return_H, log_sigma2_H, log_vol_pred, last_close}`.
This is the disambiguator the trainer/evaluator branch on:

* `isinstance(output, dict)` ⇒ Sharpe-loss path.
* `isinstance(output, Tensor|tuple)` ⇒ legacy MSE path (untouched).

Both auxiliary heads operate on the **last `lookback_for_aux` rows** of the
input window only (default 20). They are independent of the backbone's own
internal context length, which means **any** existing backbone in
`models/__init__.py:model_dict` can be wrapped without modifying its internals.
The two MLPs are tiny (`Linear → GELU → Linear`, `d_hidden=64` default) and
add ~12k parameters per head — negligible compared to a 1-7 M-param backbone.

### 2.1 Bounded scaled-space return

```python
denom = last_close.abs().clamp(min=1e-2) + 1e-9
mu_return_H = (mu_close_H - last_close) / denom
```

Features are MinMax-scaled per stock to `[0, 1]` before training. For samples
near a stock's historical price minimum the scaled close approaches zero, and
naive `(close_{t+H} − close_t) / |close_t|` explodes by 10+ orders of
magnitude — see §6 for the diagnostic story. Flooring at 1% of MinMax range
keeps the scaled-space return numerically bounded while preserving signed,
scale-stable semantics for the downstream Sharpe/NLL/position terms. This is
a structural approximation; semantically correct dollar-space returns would
require threading `close_min/close_max` through the loss path. Logged as a
follow-up rather than a blocker — see §8.

---

## 3. CompositeRiskLoss (Phase C)

```
L = α · L_SR_gated      gated differentiable Sharpe (training objective)
  + β · L_NLL           heteroscedastic Gaussian NLL on H-step return
  + γ · L_MSE_R         return-MSE anchor (prevent flat-only collapse)
  + δ · L_VOL           MSE on log realised vol target
  + η · L_GATE_BCE      gate vs. realised profitability supervision
```

Defaults: `β = 0.5`, `δ = 0.3`, `η = 0.1` (constant);
`α`, `γ` follow a 3-phase schedule:

| Phase | Epochs    | α     | γ    | Intent                            |
|------:|:---------:|:-----:|:----:|:----------------------------------|
|     1 | 0..4      | 0.0   | 1.0  | Pure MSE warm-up — anchor the predictor |
|     2 | 5..14     | 0.3   | 0.5  | Mid-Sharpe: introduce SR_gated gradient signal |
|     3 | 15..      | 0.7   | 0.2  | Full Sharpe: optimise risk-adjusted P&L |

The phase boundaries are configurable via `phase1_end` / `phase2_end` ctor
args and update each epoch through `criterion.step_epoch(epoch)`.

### 3.1 Term details

| Term            | Mathematics                                                                 | Role                                            |
|:---------------:|:----------------------------------------------------------------------------|:------------------------------------------------|
| `L_NLL`         | `½ · (log σ² + (r − μ)² / σ²)`                                              | Calibrates σ — predicting **wider** when wrong  |
| `L_MSE_R`       | `mean((μ − r)²)`                                                            | Anchor: prevents collapse to constant μ         |
| `L_VOL`         | `mean((log_vol_pred − log_vol_target)²)`                                    | Trains vol head on future realised vol          |
| Position size   | `tanh(α_pos · μ / (σ + ε))`                                                 | Kelly-style soft saturation                     |
| Gate            | `σ((τ_v − log_vol_pred)/T) · σ((τ_σ − σ)/T)`                                | Continuous kill-switch, T anneals 1.0 → 0.13    |
| `L_SR_gated`    | `−mean(g · pos · r) / (std(g · pos · r) + ε)`                               | Per-batch differentiable Sharpe                 |
| `L_GATE_BCE`    | `BCE(g, 1{pos · r > 0})`                                                    | Gate supervised against realised P&L sign       |

### 3.2 What is NOT in v1

The original `design_rethinked.md` §4 specified two additional terms,
`λ_to · L_TURN` and `λ_dd · L_DD`. Both require **temporally-ordered** batches.
Our train DataLoader shuffles for convergence, so per-batch turnover/drawdown
are not meaningful as a training-time signal. Both penalties are correctly
applied at *evaluation* time inside the cross-sectional smoke pipeline's
portfolio cost model — pushing them into training would require disabling
shuffle (hurts convergence) or building a per-stock batch sampler. This is
v2 work and not on the path to ICAIF 2026.

### 3.3 Numerical guards

Under bf16 / fp16 autocast, `exp(log_var)` can overflow for unbounded log-var
predictions. We clamp `log_var ∈ [-12.0, 4.0]` so `σ ∈ [exp(-6), exp(2)]
≈ [0.0025, 7.4]` — comfortably inside fp16 range. The gate sigmoid is
clamped to `[1e-6, 1 − 1e-6]` before BCE for the same reason.

---

## 4. Wiring (Phase D)

### 4.1 `engine/trainer.py`

`Trainer.__init__` reads `args.use_risk_head` and instantiates the right
criterion. A new `_forward_loss(batch_x, batch_y)` helper centralises the
branch:

* Risk-head mode: derives a per-batch `log_vol_target` on the fly from
  `batch_y` (std of intra-window returns when H ≥ 2; |single-step return|
  fallback for H = 1) and passes the dict output + target to
  `CompositeRiskLoss`.
* Legacy mode: tensor / tuple output → `nn.MSELoss`. AdaPatch's
  reconstruction term is preserved.

`train_global` calls `criterion.step_epoch(epoch)` per epoch; `train_sequential`
calls it per round. Both pass `nn.MSELoss()` (not the composite) to the
evaluator: early stopping needs a **stationary** signal because the composite
loss magnitude changes across phase boundaries by design.

### 4.2 `engine/evaluator.py`

When `model(batch_x)` returns a dict, the evaluator extracts
`output["mu_close"]` for regression metrics. A single-line patch keeps the
same evaluator and same `<method>_results.csv` row schema for both arms.

### 4.3 `train.py`

Four new flags:

```
--use_risk_head             Wrap backbone in RiskAwareHead + CompositeRiskLoss.
--risk_head_lookback INT    Lookback rows for sigma/vol heads (default 20).
--risk_head_d_hidden INT    Hidden width of sigma/vol MLPs (default 64).
--init_from PATH            Load a pre-trained MSE checkpoint into the
                            backbone before wrapping. state_dict keys with a
                            "backbone." prefix are stripped (idempotent).
```

`--init_from` is the bridge to fine-tune mode: load the converged MSE backbone,
randomly initialise the new sigma/vol heads, and fine-tune everything jointly.
This replicates the standard supervised-→-RL warmstart pattern.

Checkpoints save to `<MODEL>_<METHOD>_H<H>_riskhead.pth`; results write to
`<METHOD>_results_riskhead.csv`. Original MSE artefacts are never modified.

---

## 5. Verification ladder

| Layer                                          | Test                                          | Status |
|:-----------------------------------------------|:----------------------------------------------|:------:|
| Head shapes + dict keys                        | `tests/test_track_b.py::test_riskaware_head_shapes` | ✓ |
| Composite loss finite + grad through every head| `tests/test_track_b.py::test_composite_loss_finite_and_backprop` | ✓ |
| Phase schedule transitions                     | `tests/test_track_b.py::test_schedule`        | ✓ |
| B=1, flat-returns, huge-returns edge cases     | `tests/test_track_b.py::test_edge_cases`      | ✓ |
| End-to-end with real DLinear backbone          | `tests/test_track_b.py::test_e2e_with_dlinear`| ✓ |
| `_compute_log_vol_target` H≥2 + H=1 fallback   | `tests/test_trainer_risk_head.py`             | ✓ |
| Trainer init dispatch (risk vs. legacy)        | "                                             | ✓ |
| `train_epoch` populates 12-key parts dict      | "                                             | ✓ |
| `evaluate` extracts `mu_close` from dict       | "                                             | ✓ |
| 25-epoch synthetic smoke (regime signal)       | `tests/smoke_train_track_b.py`                | ✓ |
| Real-data fine-tune `DLinear_global_H5.pth`    | local CUDA, 2 epochs, 302 train stocks        | ✓ |

Real-data fine-tune outputs (post-floor-fix):

| Epoch | Train L | Val MSE | Test MSE | L_MSE_R | L_NLL | L_VOL | gate_mean | sigma_mean | pos_abs |
|:-----:|:-------:|:-------:|:--------:|:-------:|:-----:|:-----:|:---------:|:----------:|:-------:|
|   1   |  2.25   | 0.00027 |  0.0867  |  0.112  | 1.59  | 4.22  |   0.38    |   0.34     |  0.30   |
|   2   |  2.00   | 0.00028 |  0.0877  |  0.112  | 3.23  | 0.66  |   0.47    |   0.11     |  0.47   |

Final test R² = 0.99579 (preserved from MSE baseline 0.99577). σ + gate +
position magnitudes all evolving sensibly through Phase 1 (MSE-only warmup).

---

## 6. The loss-floor diagnostic story

The first real-data fine-tune produced `L_MSE_R = 8.4 × 10¹¹` despite
val/test MSE staying healthy at 0.00027 / 0.087. Root cause: per-stock
MinMax scaling pushes some `last_close` values to ~10⁻⁶ (samples near a
stock's historical low), making `(close_{t+H} − close_t) / |close_t|`
explode by 10¹² on those samples. A handful of outliers dominated the
batch mean.

The fix was a 2-line floor on `|last_close|` at 1% of MinMax range, applied
**identically** in both `engine/heads.py` (predicted return) and
`engine/losses.py` (true return) so matched samples share the same
denominator. Post-fix:

| Component  | Pre-fix       | Post-fix | Improvement       |
|:----------:|:-------------:|:--------:|:------------------|
| Train Loss | 1.47 × 10¹²   | 2.25     | 12 orders         |
| L_MSE_R    | 8.36 × 10¹¹   | 0.112    | 13 orders         |
| L_NLL      | 1.27 × 10¹²   | 1.59     | 12 orders         |
| Val MSE    | 0.00027       | 0.00027  | unchanged         |
| Test R²    | 0.99577       | 0.99579  | unchanged         |

The synthetic smoke missed this because regime features clamped to
`[0.2, 0.8]` (well above the floor); real data has the full `[0, 1]` range.
Lesson noted: synthetic smokes are necessary but not sufficient — they must
be supplemented by a one-batch real-data sanity check before HPC kickoff.

---

## 7. Campaign plan

### 7.1 Stage 1 — Track B retrain (35 jobs)

| Models                                                                                     | Horizons             | Method | Init |
|:-------------------------------------------------------------------------------------------|:---------------------|:------:|:----:|
| DLinear, iTransformer, GCFormer, PatchTST, AdaPatch, TFT, VanillaTransformer (TimesNet OUT)| 5, 20, 60, 120, 240  | global | from existing MSE ckpt via `--init_from` |

* 30 epochs (5 MSE warmup + 10 mid-Sharpe + 15 full-Sharpe).
* `lr = 5e-5` (smaller than 1e-4 baseline because the backbone is converged).
* `batch_size = 512` (256 for transformers if OOM).
* Submit alternating `gpu_h100_4` / `gpu_h200_8` partitions; QOS allows
  2 jobs per pool ⇒ 4 in flight. ~8 h per job ⇒ headline campaign clears in
  ~24 h wall clock.

### 7.2 Stage 2 — Robustness (after Stage 1 lands)

* Pick the top 1-2 models by ΔSharpe (Track B vs MSE).
* Retrain BOTH arms (MSE and Track B Sharpe) on a **delisting-aware
  bias-free universe** (P0 audit finding from `reports/codebase_audit.md`).
* Single Section 5.2 robustness table demonstrating the Sharpe edge persists.

### 7.3 Comparison protocol (the headline table)

For each (model, horizon) pair, both arms are evaluated with the **same**
post-training pipeline (`Smoke_test/cross_sectional_smoke.py`):

* **Same** 49-stock test universe (NAMES_50, calendar-aligned 2023-01-01 onwards).
* **Same** train / val / test calendar split.
* Cross-sectional ranking strategy: long top-N predicted, short bottom-N,
  rebalance every H trading days, [::H] non-overlap subsample for Sharpe.
* Reported metrics: annualised Sharpe (Politis-Romano stationary bootstrap CI),
  MaxDD, Calmar, Sortino, IC, turnover, net Sharpe @ 0/5/10/20 bps cost
  sensitivity sweep.

The hypothesis is **directional** — Sharpe-loss should beat MSE on Sharpe and
should not catastrophically lose on MSE. We pre-register the expectation that
ΔSharpe > 0 for ≥ 5 of 7 models pooled across horizons; if not, the paper
becomes an ablation / honest-negative-result story rather than a headline win.

---

## 8. Known limitations + follow-up backlog

1. **Scaled-space return floor (1e-2)** — semantic returns require dollar-space
   conversion; the current floor is a numerical band-aid. Follow-up: thread
   `close_min/close_max` through `RiskAwareHead.forward` (already in val/test
   loaders, just needs to be in the train mmap dataset).

2. **Per-batch Sharpe** — `L_SR_gated` is computed within a shuffled batch.
   This is not the realised portfolio Sharpe; it is a per-batch surrogate.
   The two converge in the large-batch limit. v2: per-stock batch sampler
   producing temporally-ordered mini-portfolios.

3. **Turnover / drawdown training-time penalties** — deferred to v2 (require
   ordered batches, see §3.2). Both are correctly applied at evaluation time.

4. **Phase boundaries (5/15)** — chosen by inspection of synthetic-smoke
   convergence curves. Should sweep on a single (model, horizon) pair before
   freezing for the campaign — quick ablation, low priority.

5. **Calibrating `tau_vol` / `tau_sigma`** — gate threshold defaults
   (`τ = 0`, `s = 1`) are placeholders. Post-Stage-1 we calibrate these on the
   validation set so gate kills the right tail of σ / vol.

6. **TimesNet excluded** — FFT + Inception cost exceeds 12 h SLURM limit
   (verified across 4 prior jobs that all timed out without producing a Round 1
   checkpoint). Documented in `design_rethinked.md`. Final paper omits TimesNet
   from the comparison tables.

---

## 9. Files shipped

```
engine/
  heads.py                        ← RiskAwareHead (NEW)
  losses.py                       ← CompositeRiskLoss (NEW)
  trainer.py                      ← dict-output dispatch + composite loss (MODIFIED)
  evaluator.py                    ← dict-output handling (MODIFIED)
train.py                          ← --use_risk_head / --init_from flags (MODIFIED)
tests/
  test_track_b.py                 ← 5 unit tests (NEW)
  test_trainer_risk_head.py       ← 6 integration tests (NEW)
  smoke_train_track_b.py          ← 25-epoch synthetic smoke (NEW)
scripts/
  riskhead_glob.sbatch            ← Track B SLURM template (NEW)
  submit_riskhead_campaign.sh     ← Campaign submission helper (NEW)
reports/
  track_b_implementation.md       ← THIS FILE (NEW)
```

---

## 10. Pre-kickoff checklist

* [x] Unit tests pass (`tests/test_track_b.py`).
* [x] Integration tests pass (`tests/test_trainer_risk_head.py`).
* [x] Synthetic smoke train passes 25 epochs.
* [x] Real-data fine-tune of `DLinear_global_H5.pth` clean (no NaN / inf,
      val/test MSE preserved, all components finite).
* [x] HPC checkpoint coverage verified — all 7 models present at all 5
      horizons (1 model needs 4 ckpts synced from local).
* [ ] Push Track B implementation to GitHub.
* [ ] `git pull` on HPC; sync missing DLinear checkpoints; verify a single
      job starts cleanly before mass-submit.
* [ ] Submit GCFormer 5 jobs (smoke the campaign with the most reliable
      backbone before scaling to all 7 models).
