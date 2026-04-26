"""Pre-scale every training stock's data and write to .npy files for the
memory-mapped global loader.

Replicates the EXACT scaling logic used by `UnifiedDataLoader.get_global_train_loader`:
  1. _load_raw(stock, FEATURES)              # column normalization, NaN drop, sort by date
  2. MinMaxScaler(feature_range=(0,1))       # one scaler per stock, fit on full history
  3. data = scaler.fit_transform(data)       # scaled to [0, 1]
  4. cast to float32, save to .npy

Stocks shorter than the minimum required (SEQ_LEN + max horizon + 1) are skipped
with a warning. The manifest records every cached stock so the runtime loader
can mmap them in a deterministic order — same order as the eager loader uses.

Usage:
    python preprocess_global_cache.py
    python preprocess_global_cache.py --force  # overwrite existing cache
"""
import os
import json
import argparse
import numpy as np
import sys

from sklearn.preprocessing import MinMaxScaler

from config import CACHE_DIR, FEATURES, SEQ_LEN, HORIZONS, NAMES_50
from data_loader import _load_raw, UnifiedDataLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Re-scale even if cache exists")
    p.add_argument("--min_required", type=int, default=None,
                   help="Min length for a stock to be cached (defaults to SEQ_LEN + max(HORIZONS) + 1)")
    args = p.parse_args()

    if args.min_required is None:
        args.min_required = SEQ_LEN + max(HORIZONS) + 1
    print(f"Cache dir       : {CACHE_DIR}")
    print(f"Min required len: {args.min_required} ({SEQ_LEN} lookback + {max(HORIZONS)} max horizon + 1)")
    print(f"Force overwrite : {args.force}")

    # Use the same train/test partition logic as UnifiedDataLoader so that
    # the cached stocks match exactly what the eager loader iterates.
    loader = UnifiedDataLoader(seq_len=SEQ_LEN, horizon=max(HORIZONS), batch_size=1)
    train_stocks = loader.train_stocks
    print(f"Training stocks : {len(train_stocks)} (after excluding NAMES_50)")

    manifest = []
    skipped_short = []
    skipped_error = []

    for i, stock in enumerate(train_stocks):
        out_path = os.path.join(CACHE_DIR, f"{stock}.npy")
        if os.path.exists(out_path) and not args.force:
            try:
                arr = np.load(out_path, mmap_mode="r")
                manifest.append({
                    "stock": stock,
                    "path": out_path,
                    "n_rows": int(arr.shape[0]),
                    "n_features": int(arr.shape[1]),
                })
                continue
            except Exception as e:
                # Corrupt cache → re-scale
                print(f"  [warn] cached file unreadable, regenerating: {stock} ({e})")

        try:
            data = _load_raw(stock, FEATURES)
            if len(data) < args.min_required:
                skipped_short.append((stock, len(data)))
                continue
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(data).astype(np.float32)
            np.save(out_path, scaled)
            manifest.append({
                "stock": stock,
                "path": out_path,
                "n_rows": int(scaled.shape[0]),
                "n_features": int(scaled.shape[1]),
            })
            if (i + 1) % 25 == 0:
                print(f"  cached {i+1}/{len(train_stocks)}: {stock} ({scaled.shape[0]} rows)")
        except Exception as e:
            skipped_error.append((stock, str(e)))

    # Write manifest (sorted by stock name = deterministic order)
    manifest.sort(key=lambda m: m["stock"])
    manifest_path = os.path.join(CACHE_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "n_features": len(FEATURES),
            "feature_names": FEATURES,
            "seq_len_target": SEQ_LEN,
            "horizons_target": HORIZONS,
            "min_required_len": args.min_required,
            "stocks": manifest,
        }, f, indent=2)

    print()
    print(f"Cached         : {len(manifest)} stocks")
    print(f"Skipped (short): {len(skipped_short)}")
    if skipped_short:
        for s, n in skipped_short[:5]:
            print(f"   {s}: {n} rows")
    print(f"Skipped (error): {len(skipped_error)}")
    if skipped_error:
        for s, e in skipped_error[:5]:
            print(f"   {s}: {e}")
    print(f"Manifest       : {manifest_path}")

    # Total cache size
    total_bytes = sum(os.path.getsize(m["path"]) for m in manifest)
    print(f"Total cache    : {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
