"""
ETL: yfinance raw JSON → market_prices canonical Parquet.

Raw format: bulk file, flat list of OHLCV records
[{"date": "2020-01-02", "Close": 296.88, "High": 297.1,
  "Low": 294.7, "Open": 295.6, "Volume": 59151200, "ticker": "SPY"}]

Partition layout (Phase 9+):
  ticker= / year=
  - partition path uses SANITIZED ticker value (^GSPC→GSPC, CL=F→CL_F)
  - 16x cheaper single-ticker Athena queries vs previous year=/exchange= layout
  - exchange kept as data column only (not partition)

Column naming (Phase 9+ fix):
  - Parquet data column is named 'ticker_symbol', NOT 'ticker' — Athena
    throws HIVE_INVALID_METADATA "duplicate columns" if a Parquet column
    has the same name as a partition column. 'ticker' now exists ONLY as
    the partition (sanitized value); 'ticker_symbol' carries the original,
    unsanitized value (^GSPC, EURUSD=X, BRK-B) for display/identification.
  - query/api.py SELECTs 'ticker_symbol AS ticker' to recover the original
    symbol for display, while WHERE clauses filter on the partition
    column 'ticker' using the sanitized value (see _ticker_partition()).

Notes:
- yfinance returns adjusted close as Close by default
- Capitalized field names need lowercasing
- Exchange/currency/country inferred from ticker config
"""
import sys
import os
import zipfile

# Glue places --extra-py-files zip in glue-python-libs-* but does not extract it
# Extract it manually so internal packages like utils/ are importable
_libs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _entry in os.listdir('/tmp/'):
    if _entry.startswith('glue-python-libs-'):
        _libs_dir = os.path.join('/tmp/', _entry)
        for _f in os.listdir(_libs_dir):
            if _f.endswith('.zip'):
                with zipfile.ZipFile(os.path.join(_libs_dir, _f)) as _z:
                    _z.extractall(_libs_dir)
        sys.path.insert(0, _libs_dir)
        break

import json
import boto3
import yaml
import pandas as pd
from datetime import date
from io import BytesIO

sys.path.insert(0, _libs_dir)
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
    os.path.join(_libs_dir, "configs", "sources", "yfinance.yaml"))


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
    Original value is preserved in the 'ticker_symbol' Parquet column.
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
from utils.watermark import get_watermark, update_watermark

SOURCE = "yfinance"
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

def _parse_chunk_filename(key: str):
    """
    Parse a chunked raw filename for its run timestamp and chunk number.

    Filename format (2026-06-19 chunking fix):
      yfinance_{start_date}_{end_date}_{TIMESTAMP}_chunk{NNN}.json
      e.g. yfinance_2023-01-01_2023-01-31_20260619T003428Z_chunk001.json

    Returns (timestamp, chunk_label) or None if the filename doesn't
    match the expected chunk pattern (e.g. the separate metadata file,
    which has no _chunk suffix and is handled by etl_companies.py).

    Sorting/grouping on the full key string is wrong because the filename
    also embeds the requested date RANGE before the timestamp — a narrow
    recent-window chunk can sort after an older backfill chunk purely on
    string comparison. Parsing out just the timestamp segment avoids this.
    """
    filename = key.rsplit("/", 1)[-1]
    if not filename.endswith(".json") or "_chunk" not in filename:
        return None
    stem  = filename[: -len(".json")]
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    chunk_label = parts[-1]   # e.g. "chunk001"
    timestamp   = parts[-2]   # e.g. "20260619T003428Z"
    return timestamp, chunk_label


def find_latest_run_chunk_keys() -> list:
    """
    List S3 keys for ALL chunk files from the most recent ingestion run,
    WITHOUT fetching their contents — just key listing/filtering/grouping.

    A single run now produces multiple chunk files sharing the same
    timestamp suffix (...{timestamp}_chunk{NNN}.json) plus one separate
    metadata file (...{timestamp}.json, no _chunk suffix) — the metadata
    file is EXCLUDED here; it's handled separately by etl_companies.py.
    """
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=RAW_BUCKET):
        keys.extend([o["Key"] for o in page.get("Contents", [])])

    keys = [k for k in keys if k.startswith("year=")]
    parsed = [(k, _parse_chunk_filename(k)) for k in keys]
    parsed = [(k, p) for k, p in parsed if p is not None]
    if not parsed:
        raise FileNotFoundError(
            f"No raw chunk files found under year= prefix in s3://{RAW_BUCKET}"
        )

    # Identify the most recent run's timestamp, then collect ALL chunk
    # files sharing that same timestamp.
    latest_timestamp = max(timestamp for _, (timestamp, _) in parsed)
    run_keys = sorted(k for k, (timestamp, _) in parsed
                       if timestamp == latest_timestamp)

    print(f"  Found {len(run_keys)} chunk files from run {latest_timestamp}")
    return run_keys


def read_one_chunk(key: str) -> list:
    """
    Fetch and parse a SINGLE chunk file. Called once per chunk inside the
    main processing loop — never accumulates raw records across chunks.
    """
    obj = s3.get_object(Bucket=RAW_BUCKET, Key=key)
    return json.loads(obj["Body"].read())


def write_processed(rows: list) -> int:
    """
    Write canonical rows to processed bucket as Parquet.

    Partition layout: ticker= / year=
    - partition path value is sanitized (_safe_partition_value)
    - year and ticker (sanitized) are partition columns — excluded from
      the Parquet file to avoid HIVE_INVALID_METADATA duplicate columns
    - rows carry 'ticker_symbol' (original value) and '_partition_ticker'
      (sanitized, used only for grouping) — '_partition_ticker' is also
      dropped before writing Parquet
    """
    if not rows:
        print("  No rows to write")
        return 0

    df = pd.DataFrame(rows)
    df["year"] = df["year"].astype(int)
    total = 0

    for (partition_ticker, year), group in df.groupby(["_partition_ticker", "year"]):
        key = (
            f"market_prices/"
            f"ticker={partition_ticker}/"
            f"year={year}/"
            f"data.parquet"
        )

        # Merge with existing partition to preserve data across incremental runs.
        # New rows win on date collisions (keep="last" after concat).
        try:
            obj = s3.get_object(Bucket=PROC_BUCKET, Key=key)
            existing_df = pd.read_parquet(BytesIO(obj["Body"].read()))
            write_df = group.drop(columns=["year", "_partition_ticker"])
            write_df = pd.concat([existing_df, write_df], ignore_index=True)
            write_df = write_df.drop_duplicates(subset=["ticker_symbol", "date"], keep="last")
            write_df = write_df.sort_values("date").reset_index(drop=True)
            print(f"  Merging: {len(existing_df)} existing + {len(group)} new → "
                  f"{len(write_df)} rows after dedup")
        except s3.exceptions.NoSuchKey:
            write_df = group.drop(columns=["year", "_partition_ticker"])
        except Exception as e:
            print(f"  ERROR: could not read existing partition "
                  f"s3://{PROC_BUCKET}/{key}: {e}")
            print(f"  SKIPPING ticker={partition_ticker} year={year} — "
                  f"refusing blind overwrite to protect existing data")
            continue

        buf = BytesIO()
        write_df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
        buf.seek(0)
        s3.put_object(Bucket=PROC_BUCKET, Key=key, Body=buf.getvalue())
        total += len(group)
        print(f"  Wrote {len(write_df)} rows → s3://{PROC_BUCKET}/{key}")

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
    - 'ticker_symbol' carries the original value (^GSPC, BRK-B, etc.) as a
      Parquet data column. '_partition_ticker' carries the sanitized value
      used ONLY to build the S3 partition path in write_processed() — it
      is dropped before writing and never appears in the Parquet schema.
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
        sector    = rec.get("sector", "") if instrument == "equity" else ""
        industry  = rec.get("industry", "") if instrument == "equity" else ""
        volume    = rec.get("Volume")
        adj_close = safe_float(rec.get("Close"))

        row = {
            "ticker_symbol":     ticker,                          # data column — original value
            "_partition_ticker": _safe_partition_value(ticker),   # dropped before write — partition path only
            "exchange":    exchange,   # data column only — not a partition
            "date":        str(trade_date),
            "year":        trade_date.year,   # partition column — dropped from Parquet
            "country":     "US"      if instrument in ("index", "futures", "fx")
                           else meta["country"],
            "currency":    currency,
            "sector":      sector,
            "industry":    industry,
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

        # validate_market_price_row() (shared utility, used by all 5 ETL
        # scripts) requires a 'ticker' key — not touching its contract.
        # Pass a shallow copy with 'ticker' aliased back in for validation
        # only; the row actually appended below has no 'ticker' key, just
        # 'ticker_symbol' (data column) and '_partition_ticker' (path only).
        validation_row = dict(row, ticker=ticker)
        errors = validate_market_price_row(validation_row)
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
    print(f"  Tickers: {len(config.get('tickers', []))} configured")

    run_keys = find_latest_run_chunk_keys()

    # No-op guard: skip if this run's timestamp was already ETL-processed.
    run_timestamp = _parse_chunk_filename(run_keys[0])[0]
    sentinel = get_watermark(SOURCE, "_etl_last_processed")
    if sentinel and sentinel.get("last_ingested_period") == run_timestamp:
        print(f"  No new ingestion run since {run_timestamp} was already "
              f"processed — exiting.")
        return

    # Accumulate TRANSFORMED rows only — transform() already filters to
    # the config ticker list and drops invalid rows, so this is meaningfully
    # smaller than holding all chunks' raw records simultaneously. Each
    # chunk's raw records are fetched, transformed, and dropped before the
    # next chunk is fetched — peak memory is bounded to ~one chunk's raw
    # data plus the running transformed-rows total, not the whole run.
    all_rows = []
    for i, key in enumerate(run_keys, 1):
        print(f"\n  Processing chunk {i}/{len(run_keys)}: {key}")
        chunk_records = read_one_chunk(key)
        chunk_rows    = transform(chunk_records, config)
        all_rows.extend(chunk_rows)
        print(f"    {len(chunk_records)} raw records → "
              f"{len(chunk_rows)} transformed rows "
              f"(running total: {len(all_rows)})")
        del chunk_records, chunk_rows

    total_written = write_processed(all_rows)

    if total_written > 0:
        update_watermark(SOURCE, "_etl_last_processed", run_timestamp,
                         "success", total_written)
        print(f"  Sentinel updated: _etl_last_processed = {run_timestamp}")

    print(f"\n{'─'*50}")
    print(f"Done. {total_written} rows written.")


if __name__ == "__main__":
    main()