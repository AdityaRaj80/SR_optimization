"""Pre-scale every training stock's data (and optionally every val/test
stock's data) and write to .npy files for the memory-mapped loaders.

TRAIN mode (default):
  Replicates the exact scaling logic of `get_global_train_loader`:
    1. _load_raw(stock, FEATURES)              # column normalization, NaN drop, sort by date
    2. MinMaxScaler(feature_range=(0,1))       # one scaler per stock, fit on full history
    3. data = scaler.fit_transform(data)       # scaled to [0, 1]
    4. save to .cache/global_scaled/<stock>.npy

VALTEST mode (--valtest flag):
  Replicates the exact scaling logic of `get_val_test_loaders`:
    1. _load_raw(stock, FEATURES)
    2. half_idx = int(len(data) * 0.5); val=data[:half_idx], test=data[half_idx:]
    3. scaler.fit_transform(val) / scaler.transform(test)
    4. save val_scaled / test_scaled to .cache/valtest_scaled/<stock>__{val,test}.npy
       plus close_min / close_max in the manifest for dollar-space inverse transform

Usage:
    python preprocess_global_cache.py                 # train cache only (default)
    python preprocess_global_cache.py --valtest       # also build val/test cache
    python preprocess_global_cache.py --only-valtest  # only val/test, skip train
    python preprocess_global_cache.py --force         # overwrite existing cache
"""
import os
import json
import argparse
import numpy as np
import sys

from sklearn.preprocessing import MinMaxScaler

from config import (CACHE_DIR, VALTEST_CACHE_DIR, FEATURES, CLOSE_IDX, SEQ_LEN,
                    HORIZONS, NAMES_50, VAL_START_DATE, TEST_START_DATE)
from data_loader import _load_raw, _load_raw_with_dates, calendar_split, UnifiedDataLoader


def build_train_cache(args):
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


def build_valtest_cache(args):
    """Per-stock zero-leakage val/test scaling with CALENDAR-DATE split.

    For each stock in NAMES_50, split by global calendar dates:
      val  = rows with VAL_START_DATE  <= date < TEST_START_DATE
      test = rows with date >= TEST_START_DATE
    Fit scaler on val, apply to both. Save val_scaled.npy and test_scaled.npy
    plus close_min/close_max in the manifest. Stocks lacking enough data in
    either window are skipped (they have a gap during the val/test period).

    This replaces the previous per-stock 50/50 split which produced wildly
    mismatched test calendar windows across stocks (185-7295 sample range).
    Now every cached stock has the SAME val and test calendar windows, which
    enables clean cross-sectional ranking analysis.
    """
    print(f"\n{'='*70}\nVALTEST CACHE\n{'='*70}")
    print(f"Cache dir       : {VALTEST_CACHE_DIR}")
    print(f"Val window      : [{VAL_START_DATE}, {TEST_START_DATE})")
    print(f"Test window     : [{TEST_START_DATE}, end)")
    # Each window must individually fit at least one (seq_len + max_horizon)
    # sample. Use max(HORIZONS) so the cache supports every horizon.
    min_required = SEQ_LEN + max(HORIZONS) + 1
    print(f"Min required len: {min_required} per window (seq_len + max horizon + 1)")
    print(f"Force overwrite : {args.force}")

    test_stocks = [s.lower() for s in NAMES_50]
    print(f"Test stocks     : {len(test_stocks)} (NAMES_50)")

    manifest = []
    skipped_short = []
    skipped_error = []

    for i, stock in enumerate(test_stocks):
        val_path = os.path.join(VALTEST_CACHE_DIR, f"{stock}__val.npy")
        test_path = os.path.join(VALTEST_CACHE_DIR, f"{stock}__test.npy")
        meta_path = os.path.join(VALTEST_CACHE_DIR, f"{stock}__meta.json")

        if (os.path.exists(val_path) and os.path.exists(test_path)
                and os.path.exists(meta_path) and not args.force):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                # Validate that cached split matches current calendar config.
                if meta.get("val_start") == VAL_START_DATE and meta.get("test_start") == TEST_START_DATE:
                    manifest.append(meta)
                    continue
                else:
                    print(f"  [info] {stock}: cached split is stale (different VAL/TEST_START), regenerating")
            except Exception as e:
                print(f"  [warn] cached meta unreadable, regenerating: {stock} ({e})")

        try:
            data, dates = _load_raw_with_dates(stock, FEATURES)
            if dates is None:
                skipped_error.append((stock, "no Date column for calendar split"))
                continue
            # Include `seq_len` lookback rows BEFORE each window so the first
            # predictable sample's target lands at val_start / test_start.
            val_raw, val_first, test_raw, test_first = calendar_split(
                data, dates, VAL_START_DATE, TEST_START_DATE, lookback_rows=SEQ_LEN)
            if len(val_raw) < min_required or len(test_raw) < min_required:
                skipped_short.append((stock, f"val={len(val_raw)}, test={len(test_raw)}"))
                continue
            scaler = MinMaxScaler(feature_range=(0, 1))
            val_scaled = scaler.fit_transform(val_raw).astype(np.float32)
            test_scaled = scaler.transform(test_raw).astype(np.float32)
            np.save(val_path, val_scaled)
            np.save(test_path, test_scaled)
            meta = {
                "stock": stock,
                "val_path": val_path,
                "test_path": test_path,
                "val_n_rows": int(val_scaled.shape[0]),
                "test_n_rows": int(test_scaled.shape[0]),
                "val_first_predict_idx": int(val_first),
                "test_first_predict_idx": int(test_first),
                "n_features": int(val_scaled.shape[1]),
                "close_min": float(scaler.data_min_[CLOSE_IDX]),
                "close_max": float(scaler.data_max_[CLOSE_IDX]),
                "val_start": VAL_START_DATE,
                "test_start": TEST_START_DATE,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            manifest.append(meta)
            if (i + 1) % 10 == 0:
                print(f"  cached {i+1}/{len(test_stocks)}: {stock} "
                      f"(val={val_scaled.shape[0]}, test={test_scaled.shape[0]})")
        except Exception as e:
            skipped_error.append((stock, str(e)))

    manifest.sort(key=lambda m: m["stock"])
    manifest_path = os.path.join(VALTEST_CACHE_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "n_features": len(FEATURES),
            "feature_names": FEATURES,
            "close_idx": CLOSE_IDX,
            "min_required_len": min_required,
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
    total_bytes = sum(os.path.getsize(m["val_path"]) + os.path.getsize(m["test_path"]) for m in manifest)
    print(f"Total cache    : {total_bytes / 1e9:.2f} GB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Re-scale even if cache exists")
    p.add_argument("--min_required", type=int, default=None,
                   help="Min length for a TRAIN stock (defaults to SEQ_LEN + max(HORIZONS) + 1)")
    p.add_argument("--valtest", action="store_true", help="Also build val/test cache (NAMES_50)")
    p.add_argument("--only-valtest", dest="only_valtest", action="store_true",
                   help="Only build val/test cache, skip train")
    args = p.parse_args()

    if args.min_required is None:
        args.min_required = SEQ_LEN + max(HORIZONS) + 1

    if not args.only_valtest:
        build_train_cache(args)
    if args.valtest or args.only_valtest:
        build_valtest_cache(args)


if __name__ == "__main__":
    main()
