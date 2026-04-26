# Architecture Audit — Remaining 6 Models vs Original Papers

**Date:** 2026-04-27
**Models audited:** PatchTST, iTransformer, TimesNet, TFT, VanillaTransformer, AdaPatch
**Already audited separately:** DLinear (`DLinear_architecture_decision.md`), GCFormer (`GCFormer_architecture_audit.md`)
**Goal:** Confirm our implementations are faithful reproductions of canonical published versions before launching full training matrix on H100.

---

## TL;DR

| # | Model | Status | Risk for paper |
|---|-------|--------|---------------|
| 1 | **PatchTST** | ✅ Faithful (TSlib variant) — one *improvement* we made | None — improvement is safer than canonical |
| 2 | **iTransformer** | ✅ Faithful | None |
| 3 | **TimesNet** | ✅ Faithful | None |
| 4 | **TFT** | ⚠️ **Significantly simplified** — missing LSTM, VSN, GRN, IMHA | **Real risk** — see § 4 |
| 5 | **VanillaTransformer** | ✅ Faithful + 1 improvement (added RevIN) | None |
| 6 | **AdaPatch** | ✅ Faithful + 2 improvements (short-horizon branch, seq-len truncation) | None — improvements are bug-fixes |

**5 of 6 are training-ready as-is. TFT needs a decision** (§ 4) before running.

---

## 1. PatchTST (Nie et al., ICLR 2023)

**Original repo:** [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)
**Closest community reference:** [thuml/Time-Series-Library — PatchTST.py](https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py)
**Our file:** `models/patchtst.py`

The TSlib variant is a re-implementation that uses a simpler embedding layer (`PatchEmbedding`) and adds outer instance normalization. Our implementation closely follows TSlib (not yuqinie98 directly).

| Component | TSlib reference | Ours | Status |
|-----------|----------------|------|--------|
| `PatchEmbedding(d_model, patch_len, stride, padding=stride, dropout)` | identical | identical | ✅ |
| Encoder: stack of `EncoderLayer(AttentionLayer(FullAttention(False, factor, ...)))` | identical | identical | ✅ |
| `head_nf = d_model × ((seq_len − patch_len) / stride + 2)` | identical | identical | ✅ |
| `FlattenHead(n_vars, head_nf, target_window)` | identical | identical | ✅ |
| Outer instance norm (mean/std) | identical | identical | ✅ |
| **Encoder norm_layer** | `Sequential(Transpose, BatchNorm1d(d_model), Transpose)` | `nn.LayerNorm(d_model)` | ⚠️ **Improvement** |
| `enc_in` source | `configs.enc_in` | hardcoded 6 | Cosmetic |
| Output | `[B, pred_len, n_vars]` (multivariate) | `[:, :, 3]` (Close only) | Intentional univariate |

### About the BatchNorm → LayerNorm change

TSlib's PatchTST uses `BatchNorm1d(d_model)` inside the encoder, which earlier in this project caused a runtime error on our setup (`running_mean should contain N elements not d_model`) because the patch dimension was being treated as the channel axis by BatchNorm1d. We replaced it with `LayerNorm(d_model)` — the standard transformer normalization — which is what yuqinie98's original PatchTST code actually uses inside `PatchTST_backbone`. **Our LayerNorm is closer to the original PatchTST paper than TSlib's BatchNorm1d.**

**Verdict:** ✅ Faithful — and our normalization is *more* aligned with the original paper than TSlib's adaptation. Train as-is.

---

## 2. iTransformer (Liu et al., ICLR 2024)

**Original repo:** [thuml/iTransformer](https://github.com/thuml/iTransformer/blob/main/model/iTransformer.py)
**Our file:** `models/itransformer.py`

| Component | Original | Ours | Status |
|-----------|---------|------|--------|
| `DataEmbedding_inverted(seq_len, d_model, embed_type, freq, dropout)` | identical | identical | ✅ |
| Encoder: `EncoderLayer(AttentionLayer(FullAttention(False, factor, ...)))` | identical | identical | ✅ |
| `nn.Linear(d_model, pred_len)` projector | identical | identical | ✅ |
| Outer mean/std normalization | identical | identical | ✅ |
| Inversion logic (B,L,N → B,N,E → B,N,S → B,S,N) | identical | identical | ✅ |
| `enc_in` source | `configs.enc_in` | hardcoded 6 | Cosmetic |
| `factor` arg in attention | `configs.factor` | hardcoded 1 | Cosmetic (factor=1 = no probsparse, full attention) |
| `use_norm` flag | optional | always on | Cosmetic — always-on is more robust for stocks |
| `output_attention` flag | optional | not implemented | Cosmetic — irrelevant for training |
| Output | `[:, -pred_len:, :]` (multivariate) | `[:, :, 3]` (Close only) | Intentional univariate |

**Verdict:** ✅ Faithful. Architecture matches the canonical reference line-by-line. Train as-is.

---

## 3. TimesNet (Wu et al., ICLR 2023)

**Original repo:** [thuml/Time-Series-Library — TimesNet.py](https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py)
**Our file:** `models/timesnet.py`

| Component | Original | Ours | Status |
|-----------|---------|------|--------|
| `FFT_for_Period(x, k)` — top-k frequency selection | identical | identical | ✅ |
| `TimesBlock`: `Inception_Block_V1 → GELU → Inception_Block_V1`, period reshape, period-weighted aggregation, residual | identical | identical | ✅ |
| `predict_linear: Linear(seq_len, pred_len + seq_len)` | identical | identical | ✅ |
| `enc_embedding: DataEmbedding(enc_in, d_model, ...)` | identical | identical | ✅ |
| Outer mean/std normalization | identical | identical | ✅ |
| Stack of `TimesBlock`s (`e_layers`) with `LayerNorm` between | identical | identical | ✅ |
| Final projection | `Linear(d_model, c_out)` (typically c_out=enc_in for forecasting) | `Linear(d_model, enc_in=6)` | ✅ Same — c_out defaults to enc_in for forecasting |
| `task_name` branching | yes (forecast / impute / anomaly / classify) | only forecast path | Cosmetic — we only need forecasting |
| Output | `dec_out[:, -pred_len:, :]` | `dec_out[:, -pred_len:, 3]` | Intentional univariate |

**Verdict:** ✅ Faithful. The TimesBlock — the entire architectural innovation of TimesNet — is reproduced exactly. Train as-is.

---

## 4. TFT (Lim et al., 2021) — **⚠️ SIGNIFICANTLY SIMPLIFIED**

**Paper:** Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting", International Journal of Forecasting 37(4), 2021. ([arXiv:1912.09363](https://arxiv.org/abs/1912.09363))
**Reference implementations:** [PyTorch Forecasting](https://github.com/sktime/pytorch-forecasting/blob/master/pytorch_forecasting/models/temporal_fusion_transformer/_tft.py), [Darts](https://unit8co.github.io/darts/), [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/resources/tft_pyt)
**Our file:** `models/tft.py`

This one needs honest documentation. Our implementation is **not a faithful reproduction of TFT** — even the comments in the file say "simplified TFT to just rely on projection and attention".

### What canonical TFT has that we don't

| Component | Original TFT | Ours |
|-----------|-------------|------|
| **Variable Selection Networks** (VSN) — gates input variables in/out per timestep | ✅ | ❌ |
| **LSTM encoder/decoder** for local sequential processing | ✅ | ❌ (we only have transformer encoder) |
| **Gated Residual Networks** (GRN) — variable processing blocks throughout | ✅ | ❌ |
| **Static covariate encoders** | ✅ | N/A (we have no static covariates) |
| **Interpretable Multi-Head Attention** (IMHA) — shared-value heads | ✅ | ❌ (standard MHA) |
| **Quantile output head** with quantile loss | partial — we have ModuleList of linear layers but use only the median, train with MSE not QuantileLoss | ⚠️ Half-implemented |
| Multi-step decoder | ✅ | ❌ — we use a lazily-created `temporal_proj = Linear(seq_len, pred_len)` |

### What we have

A standard transformer encoder + a single linear projection. Calling this "TFT" is misleading.

### Three options, ranked

**Option A (recommended): Rename to a faithful description.**
Our model is essentially a "Transformer encoder + linear head" baseline. Rename it `Transformer-Encoder` or drop it from the suite entirely (since we already have `VanillaTransformer` which is similarly transformer-based).

**Option B: Replace with a real TFT implementation.**
- Adapt PyTorch Forecasting's `TemporalFusionTransformer` to our config interface, OR
- Pull NVIDIA NGC's reference implementation
- Estimated effort: 1–2 days of integration + verification

**Option C: Leave as-is, label clearly.**
Document in the paper as "TFT-Simplified" or "TFT-Lite". Note that we removed VSN, LSTM, GRN, IMHA. Reviewers will likely complain that this isn't TFT, but the code+docs are at least honest.

### Recommendation

**Option A or B, not C.** Option C makes us look like we don't know what TFT is. Option A is honest and trivially achievable. Option B adds substantial work but gives us the real model.

If we go with **Option A** (rename), the benchmark suite becomes 7 models: DLinear, PatchTST, iTransformer, TimesNet, GCFormer, AdaPatch, VanillaTransformer. That's still a strong set.

If we go with **Option B** (real TFT), we get the 8th model but with non-trivial integration work.

**Default choice if no objection: Option A** (rename to `TransformerEncoder` or drop entirely from the matrix).

---

## 5. VanillaTransformer (Vaswani et al., 2017 + TSlib adaptation)

**Reference implementation:** [thuml/Time-Series-Library — Transformer.py](https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py)
**Our file:** `models/vanilla_transformer.py`

| Component | Original (TSlib) | Ours | Status |
|-----------|-----------------|------|--------|
| `enc_embedding: DataEmbedding(enc_in, d_model, ...)` | identical | identical | ✅ |
| `dec_embedding: DataEmbedding(dec_in, d_model, ...)` | uses `configs.dec_in` | hardcoded `enc_in=6` | Cosmetic — same value |
| Encoder: `EncoderLayer(AttentionLayer(FullAttention(False, factor, ...)))` | identical | identical | ✅ |
| Decoder: 2-attention-layer (self + cross), `nn.Linear(d_model, c_out)` projection | identical | identical (we use `enc_in=6` for c_out) | ✅ |
| `label_len` in decoder input | `configs.label_len` (typically 48 for ETT) | hardcoded 48 | Cosmetic |
| Decoder input: zeros for `pred_len` + last `label_len` of encoder | identical | identical | ✅ |
| **Outer instance normalization (RevIN-style)** | **not in TSlib** | **added by us** | ⚠️ **Improvement** |
| Output | full multivariate `[:, -pred_len:, :]` | `[:, -pred_len:, 3]` (Close only) | Intentional univariate |

### About the added instance normalization

TSlib's `Transformer.py` does NOT include RevIN. Without it, we initially observed catastrophically bad performance on stock data (test loss of ~18 instead of ~0.07 after adding it). Adding RevIN brought VanillaTransformer in line with all other transformer-based models in our suite, every one of which uses some form of instance normalization either internally (PatchTST, iTransformer, TimesNet) or via RevIN (GCFormer).

This is a **principled correction** for the cross-stock scale variation in our dataset (price ranges from $1 to $4000 across stocks), not a deviation from the architecture itself. The architecture is identical; we just normalize the inputs before feeding them in.

**Verdict:** ✅ Faithful + improvement. Train as-is.

---

## 6. AdaPatch (Yan et al., CIKM 2025)

**Original repo:** [iuaku/AdaPatch — models/AdaPatch.py](https://github.com/iuaku/AdaPatch/blob/master/models/AdaPatch.py)
**Paper:** [Yan et al., CIKM 2025](https://dl.acm.org/doi/10.1145/3746252.3761360)
**Our file:** `models/adapatch.py`

| Component | Original | Ours | Status |
|-----------|---------|------|--------|
| Encoder MLP: `Linear(patch_len, mid) → LeakyReLU → Dropout → Linear(mid, hidden) → LayerNorm` | identical | identical | ✅ |
| Decoder MLP: `Linear(hidden, mid) → LeakyReLU → Dropout → Linear(mid, patch_len)` | identical | identical | ✅ |
| `fc_predictor`: `Linear(hidden × num_patches, d_ff) → LeakyReLU → Dropout → Linear(d_ff, hidden × num_pred_patches)` | identical | identical | ✅ |
| Reconstruction path via `unfold(patch_len, patch_stride)` | identical | identical | ✅ |
| Prediction path via `reshape` | uses `reshape(B, C, num_patches, patch_len)` directly | adds `L_trunc = num_patches × patch_len` slicing first | ⚠️ **Bug-fix** |
| **Short-horizon branch** (when `pred_len < patch_len`, i.e. `num_pred_patches = 0`) | not implemented (would crash) | **added by us** with direct `Linear(hidden × num_patches, pred_len)` | ⚠️ **Bug-fix** |
| Forward returns `(y_pred, slice_orig_flat, decoded_slice_flat)` | identical | identical (with Close-slice on `y_pred`) | ✅ |
| Trainer uses reconstruction loss + prediction loss with `α` weighting | not in repo we have, but is the paper's claim | implemented in `engine/trainer.py` | ✅ |

### About the two bug-fixes we added

1. **Short-horizon branch.** Our config uses `slice_len = patch_len = 8` and we have horizons `H ∈ {5, 20, 60, 120, 240}`. For `H = 5`, `num_pred_patches = 5 // 8 = 0`, which makes `Linear(d_ff, hidden × 0) = Linear(d_ff, 0)` — an empty linear layer that crashes at forward time. We added a `short_horizon` branch that bypasses the patch decoder and projects directly to `pred_len`. **The canonical AdaPatch would simply not run for our shortest horizon.**

2. **Sequence-length truncation.** When `seq_len % patch_len != 0`, the `reshape(B, C, num_patches, patch_len)` in the canonical version would fail because `num_patches × patch_len < seq_len`. We added `x[:, :, :num_patches × patch_len]` truncation. With our `SEQ_LEN = 504` and `patch_len = 8`, 504 / 8 = 63 cleanly, so this fix is currently a no-op — but it makes the model robust to future config changes.

Both changes are **strictly safer** than the canonical version. The canonical AdaPatch would crash on our H=5 setup; ours runs correctly.

**Verdict:** ✅ Faithful + 2 bug-fixes. Train as-is.

---

## Decisions needed before launching full training matrix

1. **TFT — pick Option A, B, or C** (§ 4). Default if no objection: **Option A — rename to `TransformerEncoder` or drop, leaving 7 models in the matrix**.
2. (Optional) **Cosmetic cleanup**: replace hardcoded `enc_in = 6` with `configs.enc_in` everywhere. Pure quality-of-life, no behavioural change.
3. **Cleanup of GCFormer's dead `weights_real`/`weights_imag`** — from the GCFormer audit, these allocate ~130 MB of unused parameters. Removing them is harmless but optional.

Items 2 and 3 can be deferred to post-paper. Item 1 is paper-critical.

---

## Summary

We are training-ready for **5 of 6 audited models** (PatchTST, iTransformer, TimesNet, VanillaTransformer, AdaPatch) plus the previously-audited **DLinear** and **GCFormer**. That's **7 models** with no architectural concerns.

The 8th model (TFT) needs your decision before launching its sequential and global runs.
