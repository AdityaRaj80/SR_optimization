import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from config import DATA_DIR, FEATURES, CLOSE_IDX, NAMES_50, SEQ_LEN

class TS_Dataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]

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
    return data

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

    def get_val_test_loaders(self):
        val_X,  val_y = [], []
        test_X, test_y = [], []
        self.test_stock_scalers = {}

        for stock in self.test_stocks:
            try:
                data = _load_raw(stock, FEATURES)
                if len(data) < (self.seq_len + self.horizon) * 2:
                    continue
                
                half_idx = int(len(data) * 0.5)
                val_data = data[:half_idx]
                test_data = data[half_idx:]

                scaler = MinMaxScaler(feature_range=(0, 1))
                val_data = scaler.fit_transform(val_data)
                test_data = scaler.transform(test_data)
                self.test_stock_scalers[stock] = scaler

                X_v, y_v = build_sequences(val_data, self.seq_len, self.horizon, CLOSE_IDX)
                X_t, y_t = build_sequences(test_data, self.seq_len, self.horizon, CLOSE_IDX)

                if len(X_v) > 0:
                    val_X.append(X_v)
                    val_y.append(y_v)
                if len(X_t) > 0:
                    test_X.append(X_t)
                    test_y.append(y_t)
            except Exception as e:
                print(f"Skipping val/test for {stock}: {e}")
        
        v_ds = TS_Dataset(np.concatenate(val_X), np.concatenate(val_y)) if len(val_X) > 0 else None
        t_ds = TS_Dataset(np.concatenate(test_X), np.concatenate(test_y)) if len(test_X) > 0 else None
        
        val_loader = DataLoader(v_ds, batch_size=self.batch_size, shuffle=False) if v_ds else None
        test_loader = DataLoader(t_ds, batch_size=self.batch_size, shuffle=False) if t_ds else None

        return val_loader, test_loader
