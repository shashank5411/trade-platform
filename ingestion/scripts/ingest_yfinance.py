"""
yfinance market data — incremental ingestion
Dataset: per-ticker watermarks | Frequency: daily

Chunking (OOM fix, 2026-06-19):
  - Backfill range is split into calendar-month chunks; ALL tickers are
    downloaded together within each chunk, but only one month's worth of
    OHLCV data is held in memory at a time. Chunk size stays roughly
    constant regardless of total backfill depth (a 20yr backfill just
    means more chunks of the same size, not bigger chunks).
  - Ticker metadata (sector, market_cap, description, etc.) is fetched
    ONCE per run and written to a separate metadata file — it used to be
    duplicated onto every OHLCV row, which dominated memory usage on a
    multi-year backfill across hundreds of tickers.
"""
import sys
import os

import zipfile

_libs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Locate the extracted --extra-py-files directory in Glue Python Shell
for _entry in os.listdir('/tmp/'):
    if _entry.startswith('glue-python-libs-'):
        _libs_dir = os.path.join('/tmp/', _entry)
        for _f in os.listdir(_libs_dir):
            if _f.endswith('.zip'):
                with zipfile.ZipFile(os.path.join(_libs_dir, _f)) as _z:
                    _z.extractall(_libs_dir)
        sys.path.insert(0, _libs_dir)
        break
import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timezone

import boto3
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

from utils.config import get_default_start, load_source_config
from utils.dates import current_date_str, subtract_days
from utils.watermark import get_watermark, update_watermark

EXCHANGE_NORMALIZE = {
    "NYQ": "NYSE", "NYSEArca": "NYSE",
    "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ",
    "LSE": "LSE",
    "NSI": "NSE",
}


def fetch_ticker_meta(ticker: str) -> dict:
    try:
        t      = yf.Ticker(ticker)
        fast   = t.fast_info
        raw_ex = getattr(fast, "exchange", "NMS")
        info   = t.info
        return {
            "exchange":       EXCHANGE_NORMALIZE.get(raw_ex, "NASDAQ"),
            "currency":       getattr(fast, "currency", "USD"),
            "sector":         info.get("sector", ""),
            "industry":       info.get("industry", ""),
            "company_name":   info.get("longName", ""),
            "market_cap":     info.get("marketCap", None),
            "employees":      info.get("fullTimeEmployees", None),
            "beta":           info.get("beta", None),
            "dividend_yield": info.get("dividendYield", None),
            "pe_ratio":       info.get("trailingPE", None),
            "forward_pe":     info.get("forwardPE", None),
            "week52_high":    info.get("fiftyTwoWeekHigh", None),
            "week52_low":     info.get("fiftyTwoWeekLow", None),
            "avg_volume_10d": info.get("averageVolume", None),
            "avg_volume_3m":  info.get("averageDailyVolume3Month", None),
            "city":           info.get("city", ""),
            "state":          info.get("state", ""),
            "country":        info.get("country", ""),
            "description":    info.get("longBusinessSummary", "")[:500],
        }
    except Exception:
        return {
            "exchange": "NASDAQ", "currency": "USD", "sector": "", "industry": "",
            "company_name": "", "market_cap": None, "employees": None,
            "beta": None, "dividend_yield": None, "pe_ratio": None,
            "forward_pe": None, "week52_high": None, "week52_low": None,
            "avg_volume_10d": None, "avg_volume_3m": None,
            "city": "", "state": "", "country": "", "description": "",
        }


def fetch_all_ticker_metadata(tickers: list) -> list:
    """One metadata record per ticker, fetched ONCE per run (not per chunk) —
    sector/market_cap/etc. don't vary by date range requested."""
    ingested_at = datetime.now(timezone.utc).isoformat()
    metadata = []
    for ticker in tickers:
        meta = fetch_ticker_meta(ticker)
        meta["ticker"]      = ticker
        meta["ingested_at"] = ingested_at
        metadata.append(meta)
    return metadata


def _arg(name: str, default: str = "") -> str:
    for i, a in enumerate(sys.argv):
        if a == f"--{name}" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.getenv(name, default)


SOURCE = "yfinance"

ENV        = _arg("ENVIRONMENT", "dev")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
BUCKET     = f"{ENV}-trade-{SOURCE}-raw-{ACCOUNT_ID}"
S3         = boto3.client("s3", region_name="us-east-2")


def parse_args():
    p = argparse.ArgumentParser(description="Ingest yfinance market data")
    p.add_argument("--start-date", help="Start date YYYY-MM-DD")
    p.add_argument("--end-date",   help="End date   YYYY-MM-DD")
    return p.parse_known_args()[0]


def _resolve_start(tickers: list, config: dict, args) -> str:
    if getattr(args, "start_date", None):
        return args.start_date
    wms = [get_watermark(SOURCE, t) for t in tickers]
    live = [w for w in wms if w]
    if not live:
        return get_default_start(config)
    oldest = min(w["last_ingested_period"] for w in live)
    return subtract_days(oldest, config.get("max_lookback_days", 30))


def _month_chunks(start: str, end: str) -> list:
    """
    Split a date range into calendar-month chunks: [(start, end), ...].

    Chunking by calendar month (not ticker count) keeps chunk SIZE roughly
    constant regardless of total backfill depth — a 20-year backfill just
    means more chunks of the same small size, not bigger chunks. All
    tickers are downloaded together within each chunk; only the date axis
    is chunked.
    """
    start_dt = date.fromisoformat(start)
    end_dt   = date.fromisoformat(end)
    chunks   = []
    cur      = start_dt
    while cur <= end_dt:
        chunk_end = min(
            (cur + relativedelta(months=1)) - relativedelta(days=1),
            end_dt
        )
        chunks.append((cur.isoformat(), chunk_end.isoformat()))
        cur = cur + relativedelta(months=1)
    return chunks


def fetch_market_data(tickers: list, start: str, end: str) -> dict[str, list]:
    """OHLCV only — metadata is fetched separately (see fetch_all_ticker_metadata)."""
    print(f"  Downloading {len(tickers)} tickers {start} → {end}...")
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return {}

    per_ticker: dict[str, list] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                df = raw.xs(ticker, axis=1, level=1).dropna(how="all").copy()
                df.index.name = "date"
                df = df.reset_index()
                df["ticker"] = ticker
                df["date"]   = df["date"].dt.strftime("%Y-%m-%d")
                per_ticker[ticker] = df.to_dict(orient="records")
            except KeyError:
                per_ticker[ticker] = []
    else:
        df = raw.reset_index()
        df["ticker"] = tickers[0]
        df["Date"]   = df["Date"].dt.strftime("%Y-%m-%d")
        per_ticker[tickers[0]] = df.to_dict(orient="records")

    total = sum(len(v) for v in per_ticker.values())
    print(f"  {total} OHLCV rows across {len(tickers)} tickers")
    return per_ticker


def main():
    args    = parse_args()
    config  = load_source_config(SOURCE)
    tickers = config["tickers"]

    start_date = _resolve_start(tickers, config, args)
    end_date   = getattr(args, "end_date", None) or current_date_str()
    source_label = "CLI" if args.start_date else "watermark/first-run"
    print(f"[{SOURCE}] {start_date} → {end_date}  ({source_label})")

    timestamp   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    year, month = timestamp[:4], timestamp[4:6]

    # Metadata: fetched ONCE for the whole run, not per chunk
    print("  Fetching ticker metadata (once per run)...")
    metadata_records = fetch_all_ticker_metadata(tickers)

    if metadata_records:
        meta_key = f"year={year}/month={month}/{SOURCE}_metadata_{timestamp}.json"
        S3.put_object(
            Bucket=BUCKET,
            Key=meta_key,
            Body=json.dumps(metadata_records, default=str, indent=2),
            ContentType="application/json",
        )
        print(f"  Uploaded metadata → s3://{BUCKET}/{meta_key} "
              f"({len(metadata_records)} tickers)")

    chunks = _month_chunks(start_date, end_date)
    print(f"  Backfill split into {len(chunks)} month-chunks")

    ticker_rows_seen = {t: 0 for t in tickers}

    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        print(f"\n  Chunk {i}/{len(chunks)}: {chunk_start} → {chunk_end}")
        per_ticker = fetch_market_data(tickers, chunk_start, chunk_end)

        if not per_ticker:
            print(f"  No data for chunk {chunk_start}→{chunk_end}, skipping")
            continue

        all_records = [r for rows in per_ticker.values() for r in rows]
        chunk_key = (f"year={year}/month={month}/"
                     f"{SOURCE}_{chunk_start}_{chunk_end}_{timestamp}_"
                     f"chunk{i:03d}.json")
        S3.put_object(
            Bucket=BUCKET,
            Key=chunk_key,
            Body=json.dumps(all_records, default=str, indent=2),
            ContentType="application/json",
        )
        print(f"  Uploaded → s3://{BUCKET}/{chunk_key} "
              f"({len(all_records)} records)")

        # Per-ticker watermark update AFTER EACH CHUNK — partial-run
        # recovery: if the job dies on chunk 40/78, tickers already
        # watermarked through chunk 39 don't need to be re-fetched from
        # scratch on retry.
        for ticker, rows in per_ticker.items():
            if rows:
                ticker_rows_seen[ticker] += len(rows)
                update_watermark(SOURCE, ticker, chunk_end, "success", len(rows))

        # Explicitly drop chunk data before next iteration
        del per_ticker, all_records

        if i < len(chunks):  # skip delay after the final chunk
            delay_seconds = 5
            print(f"  Waiting {delay_seconds}s before next chunk "
                  f"(rate-limit cooldown)...")
            time.sleep(delay_seconds)

    print(f"\n[{SOURCE}] Backfill complete. "
          f"{sum(ticker_rows_seen.values())} total rows across "
          f"{len(tickers)} tickers, {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
