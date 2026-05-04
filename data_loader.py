import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from config import (DATA_DIR, FEATURES, CLOSE_IDX, NAMES_50, SEQ_LEN,
                    CACHE_DIR, VALTEST_CACHE_DIR, VAL_START_DATE, TEST_START_DATE)

class TS_Dataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]


class GlobalMmapDataset(Dataset):
    """Memory-mapped Dataset for global training. Each cached stock's full,
    pre-scaled history is mmapped from disk. The Dataset exposes one item per
    valid (stock, start_offset) pair — exactly the same set of samples that
    the eager `build_sequences` + concatenate path produces, in the same
    order. Because mmap does not load the array into RAM until pages are
    actually touched, memory footprint is O(index_size + working_set) instead
    of O(total_dataset_size).

    Args:
        manifest_entries: list of dicts {"stock", "path", "n_rows", ...} —
            one per cached stock, in the desired iteration order.
        seq_len: lookback length.
        horizon: prediction length.
        close_idx: index of Close column in the cached arrays.
    """

    def __init__(self, manifest_entries, seq_len, horizon, close_idx):
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.close_idx = int(close_idx)

        self._mmaps = []           # list of np.memmap, one per stock
        # Compact index arrays (avoid Python list of tuples for memory)
        mmap_idx_list = []
        start_list = []

        min_required = self.seq_len + self.horizon
        for entry in manifest_entries:
            n_rows = entry["n_rows"]
            if n_rows < min_required:
                continue  # match eager loader's filter exactly
            arr = np.load(entry["path"], mmap_mode="r")
            if arr.shape[0] != n_rows:
                # manifest stale → trust the actual file
                n_rows = arr.shape[0]
            n_samples = n_rows - self.seq_len - self.horizon + 1
            if n_samples <= 0:
                continue
            mmap_id = len(self._mmaps)
            self._mmaps.append(arr)
            mmap_idx_list.append(np.full(n_samples, mmap_id, dtype=np.int32))
            # range(0, n_samples) — start offsets, stride=1, matches build_sequences
            start_list.append(np.arange(n_samples, dtype=np.int64))

        if not mmap_idx_list:
            self._mmap_idx = np.zeros(0, dtype=np.int32)
            self._start    = np.zeros(0, dtype=np.int64)
        else:
            self._mmap_idx = np.concatenate(mmap_idx_list)
            self._start    = np.concatenate(start_list)

    def __len__(self):
        return len(self._mmap_idx)

    def __getitem__(self, idx):
        m_id = int(self._mmap_idx[idx])
        s = int(self._start[idx])
        arr = self._mmaps[m_id]
        # Cast away from memmap → contiguous float32 (cheap, just a copy of
        # the slice, which is small relative to the model batch). Equivalent
        # to indexing into the eager all_X / all_y arrays.
        X = np.array(arr[s : s + self.seq_len], dtype=np.float32, copy=True)
        y = np.array(
            arr[s + self.seq_len : s + self.seq_len + self.horizon, self.close_idx],
            dtype=np.float32,
            copy=True,
        )
        return X, y


class ValTestMmapDataset(Dataset):
    """Memory-mapped Dataset for val OR test split. Each cached test stock has
    pre-scaled val and test arrays on disk (`<stock>__val.npy`, `<stock>__test.npy`),
    produced by `preprocess_global_cache.py --valtest`. The split argument
    selects which set to expose. Sample order matches the eager
    `get_val_test_loaders` exactly: stocks in NAMES_50 lower-case order, then
    within each stock by start_offset (stride=1).
    """

    def __init__(self, manifest_entries, seq_len, horizon, close_idx, split):
        assert split in ("val", "test"), split
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.close_idx = int(close_idx)
        self.split = split
        self._mmaps = []
        # Per-sample inverse-scale arrays (one entry per sample) so the
        # evaluator can compute dollar-space metrics. close_min/max come
        # from the val period scaler (same scaler that scaled both val and test).
        close_min_per_sample = []
        close_max_per_sample = []
        mmap_idx_list = []
        start_list = []

        path_key = f"{split}_path"
        rows_key = f"{split}_n_rows"
        # Cache may contain a lookback prefix; only generate samples whose
        # TARGET falls in the actual val/test window (i.e. target_idx >=
        # first_predict_idx). Older caches without this field default to 0.
        first_predict_key = f"{split}_first_predict_idx"
        min_required = self.seq_len + self.horizon
        for entry in manifest_entries:
            n_rows = entry[rows_key]
            if n_rows < min_required:
                continue
            arr = np.load(entry[path_key], mmap_mode="r")
            first_predict_idx = int(entry.get(first_predict_key, 0))
            # First valid sample index i must satisfy i + seq_len >= first_predict_idx
            i_min = max(0, first_predict_idx - self.seq_len)
            n_samples_total = arr.shape[0] - self.seq_len - self.horizon + 1
            if n_samples_total <= i_min:
                continue
            n_samples = n_samples_total - i_min
            mmap_id = len(self._mmaps)
            self._mmaps.append(arr)
            mmap_idx_list.append(np.full(n_samples, mmap_id, dtype=np.int32))
            start_list.append(np.arange(i_min, n_samples_total, dtype=np.int64))
            close_min_per_sample.append(np.full(n_samples, entry["close_min"], dtype=np.float32))
            close_max_per_sample.append(np.full(n_samples, entry["close_max"], dtype=np.float32))

        if not mmap_idx_list:
            self._mmap_idx = np.zeros(0, dtype=np.int32)
            self._start    = np.zeros(0, dtype=np.int64)
            self.close_min = np.zeros(0, dtype=np.float32)
            self.close_max = np.zeros(0, dtype=np.float32)
        else:
            self._mmap_idx = np.concatenate(mmap_idx_list)
            self._start    = np.concatenate(start_list)
            self.close_min = np.concatenate(close_min_per_sample)
            self.close_max = np.concatenate(close_max_per_sample)

    def __len__(self):
        return len(self._mmap_idx)

    def __getitem__(self, idx):
        m_id = int(self._mmap_idx[idx])
        s = int(self._start[idx])
        arr = self._mmaps[m_id]
        X = np.array(arr[s : s + self.seq_len], dtype=np.float32, copy=True)
        y = np.array(
            arr[s + self.seq_len : s + self.seq_len + self.horizon, self.close_idx],
            dtype=np.float32,
            copy=True,
        )
        return X, y


def _find_csv(stock: str) -> str:
    for name in [stock, stock.upper(), stock.lower()]:
        p = os.path.join(DATA_DIR, f"{name}.csv")
        if os.path.exists(p):
            return p
    pattern = os.path.join(DATA_DIR, f"{stock.lower()}*.csv")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No CSV found for '{stock}' in {DATA_DIR}")

def _load_raw(stock: str, feature_cols: list) -> np.ndarray:
    """Load raw features (no dates returned). Backward-compatible signature
    used by the global training cache. For val/test calendar-date splits,
    use `_load_raw_with_dates` instead."""
    data, _ = _load_raw_with_dates(stock, feature_cols)
    return data


def _load_raw_with_dates(stock: str, feature_cols: list):
    """Load raw features AND aligned date array. Both have the same N rows
    after NaN drop, in chronological order. Used by calendar-date split."""
    path = _find_csv(stock)
    df = pd.read_csv(path, low_memory=False)

    df.columns = [c.strip().title().replace(" ", "_") for c in df.columns]

    for old, new in {
        "Adj_Close": "Close", "Adj_close": "Close",
        "Scaled_Sentiment": "scaled_sentiment"
    }.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
        elif old in df.columns and new in df.columns:
            df = df.drop(columns=[old])

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

    result = []
    for col in feature_cols:
        if col in df.columns:
            result.append(df[col].values)
            continue
        match = [c for c in df.columns if c.lower() == col.lower()]
        if match:
            result.append(df[match[0]].values)
            continue
        if "sentiment" in col.lower():
            fill = 0.5 if "scaled" in col.lower() else 0.0
            result.append(np.full(len(df), fill, dtype=float))
        else:
            raise ValueError(f"Column '{col}' not found in dataframe.")

    data = np.column_stack(result).astype(float)
    mask = ~np.any(np.isnan(data), axis=1)
    data = data[mask]

    if "Date" in df.columns:
        dates = df["Date"].values[mask]
    else:
        # No date column → return None; caller must handle (calendar split impossible)
        dates = None
    return data, dates


def calendar_split(data, dates, val_start, test_start, lookback_rows=0):
    """Split (data, dates) into val/test arrays with optional `lookback_rows`
    of historical context PREPENDED to each window. The split semantics are:

      val_data   = up to `lookback_rows` rows immediately preceding val_start
                 + all rows in [val_start, test_start)
      test_data  = up to `lookback_rows` rows immediately preceding test_start
                 + all rows from test_start onwards

    Each returned array also reports `first_predict_idx` — the row offset at
    which the actual val/test window begins (so sample-builders can skip
    samples whose targets fall in the lookback prefix).

    Returns
    -------
    (val_data, val_first_predict_idx, test_data, test_first_predict_idx)
    """
    if dates is None:
        raise ValueError("calendar_split requires dates; CSV missing 'Date' column.")
    val_start_ts  = pd.to_datetime(val_start,  utc=True)
    test_start_ts = pd.to_datetime(test_start, utc=True)

    # Convert dates (numpy datetime64) to pandas timestamps for comparison.
    # pd.to_datetime returns DatetimeIndex; comparison gives ndarray of bool.
    dates_pd = pd.to_datetime(dates, utc=True)
    val_mask_arr = np.asarray(dates_pd >= val_start_ts)
    test_mask_arr = np.asarray(dates_pd >= test_start_ts)
    val_start_idx_full  = int(np.argmax(val_mask_arr))  if val_mask_arr.any()  else len(dates)
    test_start_idx_full = int(np.argmax(test_mask_arr)) if test_mask_arr.any() else len(dates)

    # Val: rows immediately before val_start (lookback) + rows in [val_start, test_start)
    val_lo = max(0, val_start_idx_full - lookback_rows)
    val_hi = test_start_idx_full
    val_data = data[val_lo:val_hi]
    val_first_predict_idx = val_start_idx_full - val_lo  # offset of first actual-val row in val_data

    # Test: rows immediately before test_start (lookback) + rows from test_start to end
    test_lo = max(0, test_start_idx_full - lookback_rows)
    test_data = data[test_lo:]
    test_first_predict_idx = test_start_idx_full - test_lo

    return val_data, int(val_first_predict_idx), test_data, int(test_first_predict_idx)

def build_sequences(data: np.ndarray, seq_len: int, horizon: int, close_idx: int, stride: int = 1):
    X, y = [], []
    for i in range(0, len(data) - seq_len - horizon + 1, stride):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + horizon, close_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class UnifiedDataLoader:
    def __init__(self, seq_len=SEQ_LEN, horizon=10, batch_size=128, max_stocks=None):
        self.seq_len = seq_len
        self.horizon = horizon
        self.batch_size = batch_size
        self.all_stocks = [os.path.basename(f).replace('.csv', '') for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))]

        self.test_stocks = [s.lower() for s in NAMES_50]
        self.train_stocks = [s for s in self.all_stocks if s.lower() not in self.test_stocks]

        if max_stocks is not None:
            self.train_stocks = self.train_stocks[:max_stocks]
            print(f"[max_stocks={max_stocks}] Using {len(self.train_stocks)} training stock(s) for timing test.")

        self.test_stock_scalers = {}

    def get_global_train_loader(self):
        all_X, all_y = [], []
        for stock in self.train_stocks:
            try:
                data = _load_raw(stock, FEATURES)
                if len(data) < self.seq_len + self.horizon:
                    continue
                scaler = MinMaxScaler(feature_range=(0, 1))
                data = scaler.fit_transform(data)
                X, y = build_sequences(data, self.seq_len, self.horizon, CLOSE_IDX)
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
            except Exception as e:
                pass
        
        if len(all_X) == 0:
            raise ValueError("No training data found.")

        final_X = np.concatenate(all_X)
        final_y = np.concatenate(all_y)
        dataset = TS_Dataset(final_X, final_y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def get_sequential_train_loaders(self):
        """Eager list-build (legacy). Holds ALL stock DataLoaders in memory at once.
        On 16 GB RAM with SEQ_LEN=504 + ~300 stocks this exceeds capacity. Prefer
        `iter_train_loaders()` which yields one loader at a time.
        """
        loaders = []
        for stock in self.train_stocks:
            try:
                data = _load_raw(stock, FEATURES)
                if len(data) < self.seq_len + self.horizon:
                    continue
                scaler = MinMaxScaler(feature_range=(0, 1))
                data = scaler.fit_transform(data)
                X, y = build_sequences(data, self.seq_len, self.horizon, CLOSE_IDX)
                if len(X) > 0:
                    dataset = TS_Dataset(X, y)
                    loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
                    loaders.append(loader)
            except:
                pass
        return loaders

    def iter_train_loaders(self):
        """Generator yielding one stock's DataLoader at a time. Scientifically
        equivalent to `get_sequential_train_loaders()` (same data, same order,
        same per-loader RNG behaviour) but with O(1) memory footprint instead
        of O(N_stocks) — only one stock's sequences are in memory at any moment.
        """
        for stock in self.train_stocks:
            try:
                data = _load_raw(stock, FEATURES)
                if len(data) < self.seq_len + self.horizon:
                    continue
                scaler = MinMaxScaler(feature_range=(0, 1))
                data = scaler.fit_transform(data)
                X, y = build_sequences(data, self.seq_len, self.horizon, CLOSE_IDX)
                if len(X) > 0:
                    dataset = TS_Dataset(X, y)
                    loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
                    yield loader
            except:
                pass

    def get_global_train_loader_mmap(self, cache_dir=None, num_workers=0):
        """Memory-mapped global loader. Scientifically equivalent to
        `get_global_train_loader()` (same per-stock scaling, same sample order,
        same DataLoader shuffle behaviour) but with bounded memory because
        each cached stock is held only via numpy.memmap rather than fully
        materialised into RAM.

        Pre-condition: run `python preprocess_global_cache.py` first.

        Args:
            cache_dir: location of pre-scaled .npy files. Defaults to CACHE_DIR.
            num_workers: passed through to DataLoader.

        Returns:
            DataLoader yielding (X, y) batches with shapes
            ([B, seq_len, n_features], [B, horizon]).
        """
        cache_dir = cache_dir or CACHE_DIR
        manifest_path = os.path.join(cache_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"No manifest at {manifest_path}. Run "
                f"`python preprocess_global_cache.py` first."
            )
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Restrict to the train_stocks set (which already excludes NAMES_50
        # and respects --max_stocks).
        train_set = set(self.train_stocks)
        cached_entries = [m for m in manifest["stocks"] if m["stock"] in train_set]
        # Preserve self.train_stocks order (matches eager loader iteration)
        order = {s: i for i, s in enumerate(self.train_stocks)}
        cached_entries.sort(key=lambda m: order[m["stock"]])

        dataset = GlobalMmapDataset(
            cached_entries, self.seq_len, self.horizon, CLOSE_IDX
        )
        if len(dataset) == 0:
            raise ValueError(
                "GlobalMmapDataset is empty. Either no stocks meet the "
                "min-length requirement for this horizon, or the cache is "
                "stale — try `python preprocess_global_cache.py --force`."
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_val_test_loaders_mmap(self, cache_dir=None, num_workers=0):
        """Memory-mapped val/test loaders. Bit-for-bit equivalent to
        `get_val_test_loaders` (same per-stock half/half split, same val-fit
        scaler applied to both halves, same sample order) but with bounded
        memory: only mmap pages, plus a compact per-sample close_min/max
        array, are resident at any time.

        Pre-condition: `python preprocess_global_cache.py --valtest`.

        Sets `self.test_close_min/max` and `self.val_close_min/max` so the
        evaluator can compute dollar-space metrics, identical to the eager
        loader's exposed attributes.
        """
        cache_dir = cache_dir or VALTEST_CACHE_DIR
        manifest_path = os.path.join(cache_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"No val/test manifest at {manifest_path}. Run "
                f"`python preprocess_global_cache.py --only-valtest` first."
            )
        with open(manifest_path) as f:
            manifest = json.load(f)
        entries = manifest["stocks"]

        v_ds = ValTestMmapDataset(entries, self.seq_len, self.horizon, CLOSE_IDX, split="val")
        t_ds = ValTestMmapDataset(entries, self.seq_len, self.horizon, CLOSE_IDX, split="test")

        val_loader = (DataLoader(v_ds, batch_size=self.batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
                      if len(v_ds) > 0 else None)
        test_loader = (DataLoader(t_ds, batch_size=self.batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)
                       if len(t_ds) > 0 else None)

        # Expose inverse-scale arrays — same attribute names as eager path.
        self.val_close_min = v_ds.close_min if len(v_ds) > 0 else None
        self.val_close_max = v_ds.close_max if len(v_ds) > 0 else None
        self.test_close_min = t_ds.close_min if len(t_ds) > 0 else None
        self.test_close_max = t_ds.close_max if len(t_ds) > 0 else None

        return val_loader, test_loader

    def get_val_test_loaders(self):
        val_X,  val_y = [], []
        test_X, test_y = [], []
        # Per-sample Close-feature min/max arrays so we can inverse_transform
        # predictions back to dollars in the evaluator (for the unscaled metrics).
        # Each test sample comes from one specific stock and that stock's scaler
        # determines its close_min/close_max. Preserves order — test_loader has
        # shuffle=False so positions match.
        test_close_min = []
        test_close_max = []
        val_close_min = []
        val_close_max = []
        self.test_stock_scalers = {}

        min_required = self.seq_len + self.horizon
        # Calendar-aware lookback: include `seq_len` trading days of history
        # before each window's start so the FIRST predictable sample's target
        # is exactly at val_start (and test_start).
        lookback_rows = self.seq_len
        for stock in self.test_stocks:
            try:
                data, dates = _load_raw_with_dates(stock, FEATURES)
                if dates is None:
                    # Fall back to legacy 50/50 if no Date column (should not happen on FNSPID)
                    if len(data) < min_required * 2:
                        continue
                    half_idx = int(len(data) * 0.5)
                    val_data = data[:half_idx]
                    test_data = data[half_idx:]
                    val_first = test_first = 0
                else:
                    # Calendar-date split with lookback prefix.
                    val_data, val_first, test_data, test_first = calendar_split(
                        data, dates, VAL_START_DATE, TEST_START_DATE, lookback_rows=lookback_rows)
                    if len(val_data) < min_required or len(test_data) < min_required:
                        continue  # not enough data in either window for this stock

                scaler = MinMaxScaler(feature_range=(0, 1))
                val_data = scaler.fit_transform(val_data)
                test_data = scaler.transform(test_data)
                self.test_stock_scalers[stock] = scaler

                close_min = scaler.data_min_[CLOSE_IDX]
                close_max = scaler.data_max_[CLOSE_IDX]

                X_v, y_v = build_sequences(val_data, self.seq_len, self.horizon, CLOSE_IDX)
                X_t, y_t = build_sequences(test_data, self.seq_len, self.horizon, CLOSE_IDX)

                if len(X_v) > 0:
                    val_X.append(X_v)
                    val_y.append(y_v)
                    val_close_min.append(np.full(len(X_v), close_min, dtype=np.float32))
                    val_close_max.append(np.full(len(X_v), close_max, dtype=np.float32))
                if len(X_t) > 0:
                    test_X.append(X_t)
                    test_y.append(y_t)
                    test_close_min.append(np.full(len(X_t), close_min, dtype=np.float32))
                    test_close_max.append(np.full(len(X_t), close_max, dtype=np.float32))
            except Exception as e:
                print(f"Skipping val/test for {stock}: {e}")

        v_ds = TS_Dataset(np.concatenate(val_X), np.concatenate(val_y)) if len(val_X) > 0 else None
        t_ds = TS_Dataset(np.concatenate(test_X), np.concatenate(test_y)) if len(test_X) > 0 else None

        val_loader = DataLoader(v_ds, batch_size=self.batch_size, shuffle=False) if v_ds else None
        test_loader = DataLoader(t_ds, batch_size=self.batch_size, shuffle=False) if t_ds else None

        # Expose per-sample inverse-scale info for evaluator
        self.test_close_min = np.concatenate(test_close_min) if test_close_min else None
        self.test_close_max = np.concatenate(test_close_max) if test_close_max else None
        self.val_close_min = np.concatenate(val_close_min) if val_close_min else None
        self.val_close_max = np.concatenate(val_close_max) if val_close_max else None

        return val_loader, test_loader
