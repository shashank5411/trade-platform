"""
ETL: yfinance raw JSON → market_prices canonical Parquet.

Raw format: bulk file, flat list of OHLCV records
[{"date": "2020-01-02", "Close": 296.88, "High": 297.1,
  "Low": 294.7, "Open": 295.6, "Volume": 59151200, "ticker": "SPY"}]

Notes:
- yfinance returns adjusted close as Close by default
- Capitalized field names need lowercasing
- Exchange/currency/country inferred from ticker config
- All dev tickers are US — extend EXCHANGE_MAP for non-US
"""

import os
import sys
import json
import boto3
import yaml
import pandas as pd
from datetime import date
from io import BytesIO

# ── Path setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/tmp/ingestion")

def _arg(key, default=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--{key}", default=os.environ.get(key.upper(), default))
    args, _ = parser.parse_known_args()
    return getattr(args, key)

ENV     = _arg("env",     "dev")
ACCOUNT = _arg("account", "197411402303")

def _bucket(source, layer):
    return _arg(f"{layer}_bucket",
                f"{ENV}-trade-{source}-{layer}-{ACCOUNT}")

RAW_BUCKET  = _bucket("yfinance", "raw")
PROC_BUCKET = _bucket("yfinance", "processed")
CONFIG_PATH = _arg("config_path",
    os.path.join(os.path.dirname(__file__), "..", "configs", "sources", "yfinance.yaml"))


def _instrument_type(ticker: str) -> str:
    """Classify ticker so ETL handles it correctly."""
    if ticker.startswith("^"):
        return "index"
    elif ticker.endswith("=X"):
        return "fx"
    elif ticker.endswith("=F"):
        return "futures"
    else:
        return "equity"
    
def _safe_partition_value(value: str) -> str:
    """
    Sanitize values for S3 partition paths.
    Hive/Athena breaks on ^, =, special chars in partition paths.
    Real value is preserved in Parquet data column.
    ^GSPC → GSPC, CL=F → CL_F, BRK-B → BRK_B, DX-Y.NYB → DX_Y_NYB
    """
    return (value
            .replace("^", "")
            .replace("=", "_")
            .replace("-", "_")
            .replace(".", "_"))

from utils.transform import (
    now_utc,
    safe_float,
    to_json_str,
    validate_market_price_row,
)

s3 = boto3.client("s3", region_name="us-east-2")


# ── Exchange metadata ──────────────────────────────────────────────────────
EXCHANGE_NORMALIZE = {
    "NYQ": "NYSE", "NYSEArca": "NYSE",
    "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ",
    "LSE": "LSE",
    "NSI": "NSE",
}

EXCHANGE_META = {
    "NYSE":    {"country": "US", "currency": "USD"},
    "NASDAQ":  {"country": "US", "currency": "USD"},
    "INDEX":   {"country": "US", "currency": "USD"},
    "FX":      {"country": "US", "currency": "FX"},
    "FUTURES": {"country": "US", "currency": "USD"},
    "LSE":     {"country": "GB", "currency": "GBp"},
    "NSE":     {"country": "IN", "currency": "INR"},
}


# ── Config ─────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── S3 helpers ─────────────────────────────────────────────────────────────

def read_latest_raw() -> list:
    """Read the latest bulk yfinance raw file from S3."""
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=RAW_BUCKET):
        keys.extend([o["Key"] for o in page.get("Contents", [])])

    keys = [k for k in keys if k.startswith("year=")]
    if not keys:
        raise FileNotFoundError(
            f"No raw files found under year= prefix in s3://{RAW_BUCKET}"
        )

    latest_key = sorted(keys)[-1]
    print(f"  Reading s3://{RAW_BUCKET}/{latest_key}")

    obj     = s3.get_object(Bucket=RAW_BUCKET, Key=latest_key)
    records = json.loads(obj["Body"].read())

    tickers = sorted(set(r["ticker"] for r in records))
    print(f"  Loaded {len(records)} records — "
          f"{len(tickers)} tickers: {tickers}")
    return records


def write_processed(rows: list) -> int:
    """
    Write canonical rows to processed bucket as Parquet.
    Partitioned by year= / exchange= / ticker=
    """
    if not rows:
        print("  No rows to write")
        return 0

    df         = pd.DataFrame(rows)
    df["year"] = df["year"].astype(int)
    total      = 0

    for (year, exchange), group in df.groupby(["year", "exchange"]):
        key = (f"market_prices/year={year}/"
                f"exchange={exchange}/"
                f"data.parquet")
        

        buf = BytesIO()
        # Drop partition columns from path but keep ticker in data
        # ticker in data = original value (^GSPC, EURUSD=X)
        # ticker in path = sanitized (GSPC, EURUSD_X)
        write_df = group.drop(columns=["year", "exchange"])
        # ticker column already has original value from transform — keep it
        write_df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
        buf.seek(0)
        s3.put_object(Bucket=PROC_BUCKET, Key=key, Body=buf.getvalue())
        total += len(group)
        print(f"  Wrote {len(group)} rows → s3://{PROC_BUCKET}/{key}")

    return total


# ── Transform ──────────────────────────────────────────────────────────────

def transform(records: list, config: dict) -> list:
    """
    Transform raw yfinance OHLCV records into canonical market_prices rows.

    Key decisions:
    - yfinance Close is already split+dividend adjusted — mapped to adj_close
    - raw unadjusted close not available in this raw format — both fields
      get the same value, flagged in metadata
    - Capitalized field names lowercased
    - Exchange/country/currency inferred from ticker
    """
    ingested_at    = now_utc()
    rows           = []
    skipped        = 0
    config_tickers = set(config.get("tickers", []))

    for rec in records:
        ticker = rec.get("ticker", "").upper()

        # Filter to config ticker list only
        if config_tickers and ticker not in config_tickers:
            skipped += 1
            continue

        date_str = rec.get("date", "")
        try:
            trade_date = date.fromisoformat(date_str)
        except ValueError:
            print(f"  WARN: unparseable date '{date_str}' "
                  f"for {ticker} — skipping")
            continue

        instrument   = _instrument_type(ticker)
        raw_exchange = rec.get("exchange", "")
        if instrument == "index":
            exchange = "INDEX"
        elif instrument == "fx":
            exchange = "FX"
        elif instrument == "futures":
            exchange = "FUTURES"
        elif raw_exchange:
            exchange = raw_exchange  # already normalized by ingest script
        else:
            exchange = "NASDAQ"  # fallback for old raw files without exchange field

        meta      = EXCHANGE_META.get(exchange, EXCHANGE_META["NASDAQ"])
        currency  = rec.get("currency") or meta["currency"]
        volume    = rec.get("Volume")
        adj_close = safe_float(rec.get("Close"))

        row = {
            "ticker":      ticker,
            "exchange":    exchange,
            "date":        str(trade_date),
            "year":        trade_date.year,
            "country":     "US"      if instrument in ("index", "futures", "fx")
                           else meta["country"],
            "currency":    currency,
            "open":        safe_float(rec.get("Open")),
            "high":        safe_float(rec.get("High")),
            "low":         safe_float(rec.get("Low")),
            "close":       adj_close,
            "adj_close":   adj_close,
            "volume": None if instrument == "fx"
                      else float(volume) if volume is not None else None,
            "source":      "yfinance",
            "metadata":    to_json_str({
                "instrument_type":  instrument,
                "adj_close_note":   "close==adj_close; yfinance returns adjusted only"
            }),
            "ingested_at": ingested_at,
        }

        errors = validate_market_price_row(row)
        if errors:
            print(f"  WARN: skipping {ticker}/{trade_date}: {errors}")
            continue

        rows.append(row)

    if skipped:
        print(f"  Skipped {skipped} records not in config ticker list")

    return rows


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    print(f"ETL yfinance → market_prices | env={ENV}")
    print(f"  raw:       s3://{RAW_BUCKET}")
    print(f"  processed: s3://{PROC_BUCKET}")
    print(f"  config:    {CONFIG_PATH}")

    config = load_config()
    print(f"  Tickers: {config.get('tickers')}")

    records       = read_latest_raw()
    rows          = transform(records, config)
    total_written = write_processed(rows)

    print(f"\n{'─'*50}")
    print(f"Done. {total_written} rows written.")


if __name__ == "__main__":
    main()