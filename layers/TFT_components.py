"""Reusable building blocks for the Temporal Fusion Transformer.

References
----------
Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021).
"Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting."
International Journal of Forecasting, 37(4), 1748–1764.  arXiv:1912.09363

Components implemented:
- GatedLinearUnit (GLU) — learnable gating
- GatedResidualNetwork (GRN) — primary processing block, with optional context
- VariableSelectionNetwork (VSN) — per-variable processing + softmax-weighted aggregation
- InterpretableMultiHeadAttention (IMHA) — MHA with shared values across heads

These match the equations and module structure described in §3 of the paper.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    """`GLU(x) = sigmoid(W1·x + b1) ⊙ (W2·x + b2)`  (Eq. 1)."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc_gate = nn.Linear(input_dim, output_dim)
        self.fc_value = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.dropout(x)
        return torch.sigmoid(self.fc_gate(x)) * self.fc_value(x)


class GatedResidualNetwork(nn.Module):
    """Equation 2-4 of the paper.

    `GRN(a, c) = LayerNorm(skip(a) + GLU(W1·ELU(W2·a + W3·c)))`

    If `output_dim != input_dim`, a linear projection is used for the skip path.
    Context `c` (e.g. static covariate) is optional; if provided it is added
    via a separate linear projection before the ELU, broadcasting over time
    if `c` has fewer dimensions than `a`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        context_dim: int = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Skip path
        self.skip = (
            nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        )

        # Main path
        self.fc_a = nn.Linear(input_dim, hidden_dim)
        self.fc_c = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc_pre_glu = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, output_dim, dropout=dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, a, c=None):
        skip = self.skip(a)
        x = self.fc_a(a)
        if c is not None and self.fc_c is not None:
            c_proj = self.fc_c(c)
            # Broadcast context over time if needed
            while c_proj.dim() < x.dim():
                c_proj = c_proj.unsqueeze(1)
            x = x + c_proj
        x = F.elu(x)
        x = self.fc_pre_glu(x)
        x = self.glu(x)
        return self.layer_norm(x + skip)


class VariableSelectionNetwork(nn.Module):
    """Per-variable processing + softmax-weighted aggregation. (§3.4)

    Each input variable is first embedded to ``d_model`` *outside* this module
    (e.g. a per-variable Linear). Inside, every variable goes through its own
    GRN; the variable selection weights are produced by a separate GRN that
    operates on the flat concatenation of all variable embeddings, with a
    softmax over the variable axis.

    Input shape : ``[B, T, n_vars, d_model]``
    Output      : ``([B, T, d_model],  attn_weights[B, T, n_vars])``
    """

    def __init__(self, d_model: int, n_vars: int, dropout: float = 0.0,
                 context_dim: int = None):
        super().__init__()
        self.d_model = d_model
        self.n_vars = n_vars

        # GRN producing one weight per variable from the flattened concatenation
        self.flat_grn = GatedResidualNetwork(
            input_dim=d_model * n_vars,
            hidden_dim=d_model,
            output_dim=n_vars,
            dropout=dropout,
            context_dim=context_dim,
        )

        # Per-variable processing GRNs (parameters not shared across variables —
        # this is the part that captures variable-specific transformations)
        self.per_var_grns = nn.ModuleList([
            GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
            for _ in range(n_vars)
        ])

    def forward(self, x, context=None):
        """x: [B, T, n_vars, d_model]"""
        B, T, V, D = x.shape
        assert V == self.n_vars, f"expected {self.n_vars} variables, got {V}"
        flat = x.view(B, T, V * D)
        # weights: [B, T, n_vars] after softmax over the variable axis
        weights = torch.softmax(self.flat_grn(flat, context), dim=-1)

        processed = torch.stack(
            [self.per_var_grns[i](x[:, :, i, :]) for i in range(self.n_vars)],
            dim=2,
        )  # [B, T, n_vars, d_model]

        weighted = (processed * weights.unsqueeze(-1)).sum(dim=2)  # [B, T, d_model]
        return weighted, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Equation 14-16 of the paper.

    Multi-head attention with **values shared across heads**, allowing the
    per-head attention weights to be averaged and interpreted jointly.

    Implementation note: ``W_q`` and ``W_k`` are split per head (standard MHA),
    but ``W_v`` projects to a single ``d_head``-sized vector that all heads
    share. The output is `mean_h(softmax(Q_h K_h^T / √d_h)) · V_shared`,
    then projected back to ``d_model``.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, self.d_head, bias=False)  # shared
        self.W_o = nn.Linear(self.d_head, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, q, k, v, mask=None):
        """q,k,v: [B, T, d_model]   mask: [T_q, T_k] with 1=allowed, 0=masked"""
        B, T_q, _ = q.shape
        T_k = k.shape[1]

        Q = self.W_q(q).view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T_q,d_h]
        K = self.W_k(k).view(B, T_k, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,T_k,d_h]
        V_shared = self.W_v(v)                                                    # [B,T_k,d_h]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)    # [B,H,T_q,T_k]
        if mask is not None:
            # Broadcast mask over batch and heads
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Average attention across heads, then apply to shared V — this is the
        # "interpretable" part: a single attention map per (B, T_q, T_k).
        attn_avg = attn.mean(dim=1)                     # [B, T_q, T_k]
        out = torch.matmul(attn_avg, V_shared)          # [B, T_q, d_h]
        out = self.W_o(out)                             # [B, T_q, d_model]
        out = self.out_dropout(out)
        return out, attn_avg
