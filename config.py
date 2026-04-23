import os

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = r"D:\Study\CIKM\DATA\350_merged"
STOCK_LIST_FILE = r"D:\Study\CIKM\DATA\Stock_list.txt"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data / Feature config
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = ["Open", "High", "Low", "Close", "Volume", "scaled_sentiment"]
CLOSE_IDX = 3  # Index of Close in FEATURES array

# ─────────────────────────────────────────────────────────────────────────────
# Train / Eval stock splits
# ─────────────────────────────────────────────────────────────────────────────
# The 50 stocks for evaluation (Validation and Test)
NAMES_50 = [
    "aal", "AAPL", "ABBV", "AMD", "amgn", "AMZN", "BABA", "bhp", "bidu", "biib", 
    "C", "cat", "cmcsa", "cmg", "cop", "COST", "crm", "CVX", "dal", "DIS", "ebay", 
    "GE", "gild", "gld", "GOOG", "gsk", "INTC", "KO", "mrk", "MSFT", "mu", "nke", 
    "nvda", "orcl", "pep", "pypl", "qcom", "QQQ", "SBUX", "T", "tgt", "tm", "TSLA", 
    "TSM", "uso", "v", "WFC", "WMT", "xlf",
]

# We will treat all stocks *not* in NAMES_50 as the training group (~300 of them).
# ─────────────────────────────────────────────────────────────────────────────
# Common training config
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN = 252           # 1 trading year look-back
HORIZONS = [3, 10, 40, 120, 240]

# ─────────────────────────────────────────────────────────────────────────────
# Pre-defined Hyperparameters for each model
# ─────────────────────────────────────────────────────────────────────────────
# All models will output [B, pred_len] (univariate future Close)

PATCHTST_CONFIG = {
    "d_model": 128,
    "n_heads": 8,
    "e_layers": 6,
    "d_ff": 512,
    "patch_len": 16,
    "stride": 8,
    "dropout": 0.1,
    "fc_dropout": 0.05,
    "head_dropout": 0.0,
    "individual": 1,
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "padding_patch": "end"
}

TFT_CONFIG = {
    "d_model": 128,
    "n_heads": 4,
    "d_ff": 256,
    "lstm_layers": 2,
    "dropout": 0.1,
    "quantiles": [0.1, 0.5, 0.9]
}

ADAPATCH_CONFIG = {
    "slice_len": 8,
    "middle_len": 128,
    "hidden_len": 16,
    "slice_stride": 1,
    "encoder_dropout": 0.1,
    "d_ff": 2048,
    "alpha": 0.5  # Customizable via CLI
}

GCFORMER_CONFIG = {
    "d_model": 128,
    "n_heads": 8,
    "e_layers": 3,
    "global_layers": 1,
    "d_ff": 512,
    "patch_len": 16,
    "stride": 8,
    "dropout": 0.05,
    "fc_dropout": 0.05,
    "head_dropout": 0.0,
    "individual": 1,
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "padding_patch": "end",
    "global_model": "Gconv",
    "norm_type": "revin",
    "h_token": 512,
    "h_channel": 32,
    "local_bias": 0.5,
    "global_bias": 0.5,
    "atten_bias": 0.5,
    "TC_bias": 1,
    "decomposition": 0,
    "context_len": SEQ_LEN,
}

ITRANSFORMER_CONFIG = {
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_ff": 2048,
    "dropout": 0.1,
    "activation": "gelu"
}

VANILLA_TRANSFORMER_CONFIG = {
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 2048,
    "dropout": 0.1,
    "activation": "gelu",
    "factor": 1,
}

TIMESNET_CONFIG = {
    "d_model": 512,
    "e_layers": 2,
    "d_ff": 2048,
    "dropout": 0.1,
    "top_k": 5,
    "num_kernels": 6
}

DLINEAR_CONFIG = {
    "moving_avg": 25,
    "individual": False,
    "enc_in": len(FEATURES)
}
