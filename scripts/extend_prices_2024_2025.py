"""Extend FNSPID-derived per-stock CSVs in D:\\Study\\CIKM\\DATA\\350_merged
forward from 2023-12-28 (last existing date) to today, using yfinance.

This is Stage 2 of the Track B campaign: out-of-time validation. Trained
checkpoints stay frozen; only the test set extends. The model has never
seen any post-2023-12-28 data, so this is a true out-of-sample test.

Output schema MUST match the existing 350_merged CSVs exactly:
    Date,Open,High,Low,Close,Adj Close,Volume,scaled_sentiment
where Date is 'YYYY-MM-DD HH:MM:SS+00:00' (UTC midnight).

Sentiment for the extension window is currently set to 0.5 (neutral)
because the FNSPID news scraper extension is a separate task. If/when
news data is scraped + scored with FinBERT, this column will be
overwritten with the proper aggregated daily sentiment.

Safety:
  * Read-only on existing CSVs UNLESS --inplace is passed. Default
    is to write to D:\\Study\\CIKM\\DATA\\350_merged_ext\\<stock>.csv
    so we never destroy the existing dataset.
  * Per-stock failures are logged but don't kill the run.
  * Date deduplication on merge: existing rows always win (we never
    overwrite historical data with re-scraped values).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from glob import glob

import pandas as pd
import yfinance as yf

EXISTING_DIR = r"D:\Study\CIKM\DATA\350_merged"
OUT_DIR_DEFAULT = r"D:\Study\CIKM\DATA\350_merged_ext"
SENTIMENT_NEUTRAL = 0.5

EXPECTED_COLS = ["Date", "Open", "High", "Low", "Close", "Adj Close",
                 "Volume", "scaled_sentiment"]


def fetch_one(ticker: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame | None:
    """Download OHLCV for one ticker via yfinance. Returns None on failure."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker, start=start, end=end,
                progress=False, auto_adjust=False, threads=False,
            )
            if df is None or df.empty:
                return None
            # yfinance returns a 2-level column MultiIndex when threads=False
            # for single ticker; flatten.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.reset_index()
            return df
        except Exception as e:
            last_exc = e
            time.sleep(2 ** attempt)   # 1, 2, 4 s back-off
    print(f"  [{ticker}] FAILED after {max_retries} retries: {last_exc}",
          file=sys.stderr, flush=True)
    return None


def merge_and_save(stock: str, existing_path: str, out_path: str,
                   start: str, end: str, inplace: bool) -> dict:
    """Returns a small dict for the manifest:
        {stock, n_existing, n_new, n_total, last_date_existing, last_date_new}"""
    existing = pd.read_csv(existing_path)
    # Sanity check schema
    missing = [c for c in EXPECTED_COLS if c not in existing.columns]
    if missing:
        raise ValueError(f"{stock}: existing CSV missing columns {missing}")
    existing["Date"] = pd.to_datetime(existing["Date"], utc=True)
    last_existing = existing["Date"].max()

    # Skip if existing already covers up to `end`
    end_ts = pd.Timestamp(end).tz_localize("UTC")
    if last_existing >= end_ts - pd.Timedelta(days=1):
        return {
            "stock": stock, "status": "already_current",
            "n_existing": len(existing), "n_new": 0, "n_total": len(existing),
            "last_date_existing": last_existing.isoformat(),
            "last_date_new": last_existing.isoformat(),
        }

    fetch_start = (last_existing + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    new_df = fetch_one(stock.upper(), fetch_start, end)
    if new_df is None or new_df.empty:
        return {"stock": stock, "status": "no_new_data",
                "n_existing": len(existing), "n_new": 0,
                "n_total": len(existing),
                "last_date_existing": last_existing.isoformat(),
                "last_date_new": last_existing.isoformat()}

    # Normalise schema to match existing
    if "Date" in new_df.columns:
        new_df["Date"] = pd.to_datetime(new_df["Date"], utc=True)
    elif "Datetime" in new_df.columns:
        new_df = new_df.rename(columns={"Datetime": "Date"})
        new_df["Date"] = pd.to_datetime(new_df["Date"], utc=True)
    else:
        # yfinance with reset_index puts the date in the first column
        first_col = new_df.columns[0]
        new_df = new_df.rename(columns={first_col: "Date"})
        new_df["Date"] = pd.to_datetime(new_df["Date"], utc=True)

    new_df["scaled_sentiment"] = SENTIMENT_NEUTRAL  # placeholder
    new_df = new_df[EXPECTED_COLS]

    # Strict deduplication: existing rows always win.
    new_df = new_df[new_df["Date"] > last_existing]
    if new_df.empty:
        return {"stock": stock, "status": "no_new_data_after_dedup",
                "n_existing": len(existing), "n_new": 0,
                "n_total": len(existing),
                "last_date_existing": last_existing.isoformat(),
                "last_date_new": last_existing.isoformat()}

    merged = pd.concat([existing, new_df], ignore_index=True)
    merged = merged.sort_values("Date").reset_index(drop=True)

    if inplace:
        target = existing_path
    else:
        target = out_path
    os.makedirs(os.path.dirname(target), exist_ok=True)
    merged.to_csv(target, index=False)

    return {
        "stock": stock, "status": "extended",
        "n_existing": len(existing), "n_new": len(new_df),
        "n_total": len(merged),
        "last_date_existing": last_existing.isoformat(),
        "last_date_new": new_df["Date"].max().isoformat(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--existing_dir", default=EXISTING_DIR)
    p.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    p.add_argument("--end", default=pd.Timestamp.now().strftime("%Y-%m-%d"),
                   help="Inclusive end date for fetching (default: today).")
    p.add_argument("--inplace", action="store_true",
                   help="Overwrite existing CSVs in place. Off by default to "
                        "preserve the original dataset.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N stocks (testing).")
    p.add_argument("--start_marker", type=str, default="2023-12-29",
                   help="Documentation only -- script auto-detects last "
                        "existing date per stock.")
    args = p.parse_args()

    csvs = sorted(glob(os.path.join(args.existing_dir, "*.csv")))
    if args.limit is not None:
        csvs = csvs[: args.limit]

    print(f"existing_dir = {args.existing_dir}")
    print(f"out_dir      = {args.out_dir}")
    print(f"end_date     = {args.end}  (today)")
    print(f"inplace      = {args.inplace}")
    print(f"#stocks      = {len(csvs)}")
    print()

    manifest = []
    t0 = time.time()
    for i, csv_path in enumerate(csvs):
        stock = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = os.path.join(args.out_dir, f"{stock}.csv")
        try:
            r = merge_and_save(stock, csv_path, out_path,
                               start="(auto)", end=args.end,
                               inplace=args.inplace)
            manifest.append(r)
            elapsed = time.time() - t0
            print(f"[{i+1:>3}/{len(csvs)}] {stock:<10} {r['status']:<22} "
                  f"existing={r['n_existing']:>5} new={r['n_new']:>4} "
                  f"total={r['n_total']:>5} "
                  f"last={r['last_date_new'][:10]}  "
                  f"elapsed={elapsed:.0f}s",
                  flush=True)
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            manifest.append({"stock": stock, "status": "ERROR", "error": err_msg})
            print(f"[{i+1:>3}/{len(csvs)}] {stock:<10} ERROR: {err_msg}",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)

    # Write manifest
    manifest_df = pd.DataFrame(manifest)
    manifest_path = os.path.join(args.out_dir, "_extension_manifest.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_df.to_csv(manifest_path, index=False)
    print()
    print(f"Manifest: {manifest_path}")
    print(manifest_df["status"].value_counts())


if __name__ == "__main__":
    main()
