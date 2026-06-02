"""
yfinance market data — incremental ingestion
Dataset: per-ticker watermarks | Frequency: daily
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
import pandas as pd
import yfinance as yf

from utils.config import get_default_start, load_source_config
from utils.dates import current_date_str, subtract_days
from utils.watermark import get_watermark, update_watermark


def _arg(name: str, default: str = "") -> str:
    for i, a in enumerate(sys.argv[1:], 1):
        if a == f"--{name}" and i < len(sys.argv):
            return sys.argv[i]
    return os.getenv(name, default)


SOURCE = "yfinance"

ENV        = _arg("ENVIRONMENT", "dev")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
BUCKET     = f"{ENV}-trade-{SOURCE}-raw-{ACCOUNT_ID}"
S3         = boto3.client("s3")


def parse_args():
    p = argparse.ArgumentParser(description="Ingest yfinance market data")
    p.add_argument("--start-date", help="Start date YYYY-MM-DD")
    p.add_argument("--end-date",   help="End date   YYYY-MM-DD")
    return p.parse_args()


def _resolve_start(tickers: list, config: dict, args) -> str:
    if getattr(args, "start_date", None):
        return args.start_date
    wms = [get_watermark(SOURCE, t) for t in tickers]
    live = [w for w in wms if w]
    if not live:
        return get_default_start(config)
    oldest = min(w["last_ingested_period"] for w in live)
    return subtract_days(oldest, config.get("max_lookback_days", 30))


def fetch_market_data(tickers: list, start: str, end: str) -> dict[str, list]:
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


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"Columns:    {list(df.columns)}")
    print(f"Shape:      {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head().to_string()}")
    print(f"\nData types:\n{df.dtypes.to_string()}")
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")
    print("=" * 60)


def main():
    args   = parse_args()
    config = load_source_config(SOURCE)
    tickers = config["tickers"]

    start_date = _resolve_start(tickers, config, args)
    end_date   = getattr(args, "end_date", None) or current_date_str()
    source_label = "CLI" if args.start_date else "watermark/first-run"
    print(f"[{SOURCE}] {start_date} → {end_date}  ({source_label})")

    timestamp  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    per_ticker = fetch_market_data(tickers, start_date, end_date)

    if not per_ticker:
        print("No data returned.")
        return

    all_records = [r for rows in per_ticker.values() for r in rows]
    year, month = timestamp[:4], timestamp[4:6]
    key = f"year={year}/month={month}/{SOURCE}_{start_date}_{end_date}_{timestamp}.json"
    S3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(all_records, default=str, indent=2),
        ContentType="application/json",
    )
    print(f"Uploaded → s3://{BUCKET}/{key}  ({len(all_records)} records)")

    for ticker, rows in per_ticker.items():
        update_watermark(SOURCE, ticker, end_date, "success", len(rows))

    print_summary(pd.DataFrame(all_records))


if __name__ == "__main__":
    main()
