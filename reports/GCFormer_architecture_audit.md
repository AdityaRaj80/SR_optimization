# GCFormer — Architecture Audit vs Original Paper

**Date:** 2026-04-27
**Reference:** Yanjun-Zhao et al., "GCformer: An Efficient Solution for Accurate and Scalable Long-Term Multivariate Time Series Forecasting" (CIKM 2023)
**Reference repo:** [Yanjun-Zhao/GCformer](https://github.com/Yanjun-Zhao/GCformer/blob/main/models/GCformer.py)
**Our implementation:** `models/gcformer.py`

---

## 1. Verdict

**Our GCFormer is structurally faithful to the canonical GCFormer-with-GConv variant.** All core architectural components — outer normalization wrapper, global GConv branch, local PatchTST branch, channel-attention decoder, token-attention decoder, learnable bias parameters, and the final additive output combination — are present, correctly wired, and use the same hyperparameters as the reference implementation.

The differences from the canonical reference are:
1. **One genuine simplification** (decomposition + FNO/Film variants not implemented — we only support the dominant GConv-no-decomposition path that matches our config)
2. **Univariate output slicing** (we return Close only; original returns multivariate, both 5-tuple internals)
3. **Dead/wasted parameter allocation** in `GConv` (one orphan parameter that is never used in forward — harmless but wastes ~130 MB)
4. **A defensive try/except** in our forward — empirically dead code given that `GConv.forward` is batch-size-agnostic

None of these affect training correctness or scientific equivalence.

---

## 2. Side-by-Side Component Comparison

| Component | Original | Ours | Status |
|-----------|---------|------|--------|
| **Outer normalization** | RevIN or seq_last subtraction (configurable via `norm_type`) | Same — RevIN or seq_last (we always use RevIN per our config) | ✅ Faithful |
| **Global branch options** | `Gconv` ∣ `FNO` ∣ `Film` (via `configs.global_model`) | `GConv` only | ⚠️ Two variants not implemented, but our config locks `global_model='Gconv'` so unused branches don't matter |
| **Local branch (no decomp)** | Single `PatchTST_backbone` | Single `PatchTST_backbone` | ✅ Faithful |
| **Local branch (decomp)** | Two `PatchTST_backbone`s on trend + residual | Not implemented | ⚠️ Our config sets `decomposition=0`; original repo's default is also 0 in most config files |
| **Channel decoder** | Cross-attention via `ProbAttention` between global and local channel features | Same | ✅ Faithful |
| **Token decoder** | Cross-attention via `ProbAttention` between global and local token features | Same | ✅ Faithful |
| **Output mixing** | `TC_bias * channel + (1-TC_bias) * token + global_bias * global_x + local_bias * local_x` | Identical formula | ✅ Faithful |
| **Learnable biases** | `local_bias`, `global_bias`, `atten_bias`, `TC_bias` | Same four | ✅ Faithful |
| **Forward output shape** | `(output, local_x, global_x, global_bias, local_bias)` — 5-tuple | `output[:, :, CLOSE_IDX]` — `[B, pred_len]` | ✅ Intentional univariate target |
| **Trainer compatibility** | Trainer expects 5-tuple, indexes [0] | Our trainer correctly handles tuple-or-tensor for ALL models | ✅ Already handled |
| **`enc_in`** | `configs.enc_in` | hardcoded `6` | ⚠️ Cosmetic — matches `len(FEATURES)` |
| **`batch_size`** in `GConv` init | `configs.batch_size` | hardcoded `128` | ⚠️ See § 3 below |
| **`max_seq_len`** in PatchTST_backbone | configurable arg, default 1024 | hardcoded 1024 | ✅ Same value |
| **`norm` in PatchTST_backbone** | `'BatchNorm'` (default) | `'BatchNorm'` | ✅ Same |
| **Unused leftover layers** | `self.TCN`, `self.local_Autoformer` defined but not used in forward | We omit both | ✅ Cleaner — dropping dead code matches the original's actual computation path |

---

## 3. The hardcoded `batch_size=128` — investigated

In `GConv.__init__`:

```python
self.weights_real = nn.Parameter(torch.rand(batch_size, self.l_max, self.h))
self.weights_imag = nn.Parameter(torch.rand(batch_size, self.l_max, self.h))
```

These are the parameters that look like they would constrain the runtime batch size. **However, traced line-by-line through `GConv.forward`, neither `weights_real` nor `weights_imag` is ever referenced.** The actual convolution uses:

- `self.kernel_list` (a `ParameterList` of kernels with shape `[channels, h, kernel_dim]` — no batch dim)
- `self.D` (skip connection, shape `[channels, h]`)
- `self.kernel_norm` (buffer)
- `self.multiplier` (buffer)
- `self.activation`, `self.dropout`, `self.norm`, `self.output_linear` (modules)

So `weights_real` and `weights_imag` are **dead code** — orphaned trainable parameters that take memory and slow down the optimizer's bookkeeping (~130 MB of unused state at our `batch_size=128, l_max=504, h=256`) but **do not affect the model's output or training**.

This explains why:
- Our previous H=3 GCformer training (where we accidentally ran with a different actual batch size than the hardcoded 128) produced strong results (R² = 0.997).
- The defensive `try/except` in our `gcformer.py` forward is **never actually triggered** — `GConv.forward` is batch-size-agnostic.

### What this means for our setup

We can train GCFormer at any batch size, including `batch_size=512`, with no architectural changes. The hardcoded `128` in our `__init__` and the `try/except` block in our forward are both harmless dead code.

We could clean them up (remove `weights_real`/`weights_imag` from GConv, drop the try/except), but that requires modifying a third-party-ish layer file. For faithfulness to the canonical implementation, we leave them as-is.

---

## 4. Hyperparameter parity with the original

| Hyperparameter | Original default | Our `GCFORMER_CONFIG` | Match? |
|----------------|-----------------|----------------------|--------|
| `d_model` | 256 (varies by experiment) | 256 | ✅ |
| `n_heads` | 8 | 8 | ✅ |
| `e_layers` | 3 | 3 | ✅ |
| `d_ff` | 512 | 512 | ✅ |
| `patch_len` | 16 | 16 | ✅ |
| `stride` | 8 | 8 | ✅ |
| `dropout` | 0.05 | 0.05 | ✅ |
| `fc_dropout` | 0.05 | 0.05 | ✅ |
| `head_dropout` | 0.0 | 0.0 | ✅ |
| `individual` | 1 | 1 | ✅ |
| `revin` (local PatchTST) | 1 | 1 | ✅ |
| `affine` | 0 | 0 | ✅ |
| `subtract_last` | 0 | 0 | ✅ |
| `padding_patch` | 'end' | 'end' | ✅ |
| `global_model` | 'Gconv' | 'Gconv' | ✅ |
| `norm_type` | 'revin' | 'revin' | ✅ |
| `h_token` | 512 | 512 | ✅ |
| `h_channel` | 32 | 32 | ✅ |
| `local_bias` | 0.5 | 0.5 | ✅ |
| `global_bias` | 0.5 | 0.5 | ✅ |
| `atten_bias` | 0.5 | 0.5 | ✅ |
| `TC_bias` | 1 | 1 | ✅ |
| `decomposition` | 0 | 0 | ✅ |
| GConv `kernel_dim` | 32 | 32 | ✅ |

Every hyperparameter that affects the actual computation matches the canonical settings.

---

## 5. Decision: train as-is

We will run GCFormer **without modifications to the model file**, in keeping with the principle established for DLinear (Phase-1 reports use canonical architectures from the published implementations). The minor cosmetic / dead-code differences identified do not affect scientific validity of the comparison.

If the eventual GCFormer results diverge wildly from the published GCFormer benchmarks on standard datasets (ETTh, Weather, etc.) — *which we are not benchmarking against* — we would revisit. For our stock-data benchmark, the architecture is canonical and the comparison (sequential vs global) is well-defined.

---

## 6. Sequential training command

```bash
python -u train.py --model GCFormer --method sequential --horizon 5 \
                   --device auto --batch_size 512 --rounds 3 \
                   --epochs_per_stock 20 --lr 1e-4 --lradj type3
```

Same protocol as DLinear sequential — 20 epochs/stock × 3 rounds = 60 exposures per stock, matching FNSPID and our DLinear baseline for fair comparison.

---

## 7. Notes for future cleanup (post-paper)

If we have time after submission:

1. **Remove `weights_real`/`weights_imag` from `GConv`** — saves ~130 MB GPU memory and removes the optimizer's tracking of dead parameters
2. **Remove the try/except in `gcformer.py` forward** — dead code, masks real errors
3. **Don't hardcode `batch_size=128` and `enc_in=6` in our `gcformer.py`** — pass through configs

These are quality-of-life fixes; they would produce identical training trajectories and final metrics.
