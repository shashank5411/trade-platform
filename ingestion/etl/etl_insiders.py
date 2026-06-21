import sys
import os
import zipfile

# ── Fix A: Glue zip extraction ────────────────────────────────────────────────
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

import gzip
import json
import argparse
import io
import re

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Fix B: _arg function ──────────────────────────────────────────────────────
def _arg(name: str, default: str = "") -> str:
    for i, a in enumerate(sys.argv):
        if a == f"--{name}" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.getenv(name, default)

ENV     = _arg("ENVIRONMENT", "dev")
REGION  = "us-east-2"
ACCOUNT = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]

RAW_BUCKET  = f"{ENV}-trade-insiders-raw-{ACCOUNT}"
PROC_BUCKET = f"{ENV}-trade-insiders-processed-{ACCOUNT}"

# ── Fix D: S3 client with region ──────────────────────────────────────────────
s3 = boto3.client("s3", region_name=REGION)

# ── Fix C: parse_known_args ───────────────────────────────────────────────────
def _arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ENVIRONMENT", default="dev")
    return p.parse_known_args()[0]

_arg_parser()

# ── Canonical schema ──────────────────────────────────────────────────────────
# Partition columns NOT in Parquet: year=, ticker=
SCHEMA = pa.schema([
    pa.field("filing_id",           pa.string()),
    pa.field("accession",           pa.string()),
    pa.field("filer_name",          pa.string()),
    pa.field("filer_role",          pa.string()),
    pa.field("transaction_date",    pa.string()),
    pa.field("transaction_type",    pa.string()),  # P | S | A | D | F | M | etc.
    pa.field("shares",              pa.float64()),
    pa.field("price_per_share",     pa.float64()),
    pa.field("value_usd",           pa.float64()),
    pa.field("ownership_type",      pa.string()),  # D | I
    pa.field("shares_owned_after",  pa.float64()),
    pa.field("ingested_at",         pa.string()),
])

# ── Read raw transactions ─────────────────────────────────────────────────────
def read_raw_transactions() -> list:
    txns = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=RAW_BUCKET):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json.gz") or key.startswith("tracker/"):
                continue
            try:
                body  = s3.get_object(Bucket=RAW_BUCKET, Key=key)["Body"].read()
                batch = json.loads(gzip.decompress(body).decode("utf-8"))
                # Stamp source key for partition extraction
                for t in batch:
                    t["_source_key"] = key
                txns.extend(batch)
            except Exception as e:
                print(f"WARN: Could not read {key}: {e}")
    return txns

# ── Read existing filing_ids ──────────────────────────────────────────────────
def read_existing_filing_ids() -> set:
    existing = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=PROC_BUCKET, Prefix="insider_trades/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            try:
                body = s3.get_object(Bucket=PROC_BUCKET, Key=key)["Body"].read()
                df   = pq.read_table(io.BytesIO(body)).to_pandas()
                existing.update(df["filing_id"].tolist())
            except Exception as e:
                print(f"WARN: Could not read {key}: {e}")
    return existing

# ── Write Parquet partition ───────────────────────────────────────────────────
def write_partition(rows: list, year: int, ticker: str) -> None:
    """Write a (year, ticker) partition, MERGING with any existing data
    rather than overwriting it. Existing rows are read first, deduplicated
    against new rows by filing_id, and the combined set is written back."""
    if not rows:
        return

    key = f"insider_trades/year={year}/ticker={ticker}/data.parquet"

    # Read existing partition data, if any, before writing
    existing_rows = []
    try:
        obj = s3.get_object(Bucket=PROC_BUCKET, Key=key)
        existing_df = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
        existing_rows = existing_df.to_dict(orient="records")
        print(f"  Merging with {len(existing_rows)} existing rows at {key}")
    except s3.exceptions.NoSuchKey:
        pass
    except Exception as e:
        print(f"  WARN: could not read existing partition {key}: {e} "
              f"— proceeding with new rows only, but this risks "
              f"DROPPING existing data if the file genuinely exists and "
              f"this read failure is not a true NoSuchKey. Investigate "
              f"if this warning appears in practice.")

    # Merge: existing + new, dedup by filing_id (new rows win if somehow
    # the same filing_id appears in both — shouldn't normally happen
    # since main() already filters by existing_ids before calling this,
    # but dedup here too as defense-in-depth against any future caller
    # that doesn't pre-filter)
    combined = existing_rows + rows
    df = pd.DataFrame(combined)
    df = df.drop_duplicates(subset=["filing_id"], keep="last")

    parquet_cols = [f.name for f in SCHEMA]
    for col in parquet_cols:
        if col not in df.columns:
            df[col] = None

    df["shares"]              = pd.to_numeric(df["shares"],              errors="coerce").fillna(0.0)
    df["price_per_share"]     = pd.to_numeric(df["price_per_share"],     errors="coerce").fillna(0.0)
    df["value_usd"]           = pd.to_numeric(df["value_usd"],           errors="coerce").fillna(0.0)
    df["shares_owned_after"]  = pd.to_numeric(df["shares_owned_after"],  errors="coerce").fillna(0.0)

    table = pa.Table.from_pandas(
        df[parquet_cols], schema=SCHEMA, preserve_index=False
    )

    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)

    s3.put_object(Bucket=PROC_BUCKET, Key=key, Body=buf.read())
    print(f"  Written: s3://{PROC_BUCKET}/{key} "
          f"({len(combined)} total rows: {len(existing_rows)} existing + "
          f"{len(rows)} new, after dedup: {len(df)})")

# ── Main ETL ──────────────────────────────────────────────────────────────────
def main():
    print(f"[etl_insiders] env={ENV}, raw={RAW_BUCKET}, processed={PROC_BUCKET}")

    print("Reading raw transactions...")
    txns = read_raw_transactions()
    print(f"  Found {len(txns)} raw transactions")

    if not txns:
        print("No raw transactions found — exiting")
        return

    print("Reading existing filing_ids...")
    existing_ids = read_existing_filing_ids()
    print(f"  Found {len(existing_ids)} already processed")

    # Group by (year, ticker)
    partitions: dict = {}

    for t in txns:
        if t.get("filing_id") in existing_ids:
            continue

        # Extract ticker from S3 key: year=YYYY/ticker=AAPL/...
        ticker_match = re.search(r'ticker=([^/]+)/', t.get("_source_key", ""))
        ticker = ticker_match.group(1) if ticker_match else t.get("ticker", "UNKNOWN")

        try:
            year = int(str(t.get("year", "2026"))[:4])
            if year < 2000 or year > 2030:
                year = int(t.get("transaction_date", "2026")[:4])
        except (ValueError, TypeError):
            year = 2026

        row = {
            "filing_id":          t.get("filing_id", ""),
            "accession":          t.get("accession", ""),
            "filer_name":         t.get("filer_name", ""),
            "filer_role":         t.get("filer_role", ""),
            "transaction_date":   t.get("transaction_date", ""),
            "transaction_type":   t.get("transaction_type", ""),
            "shares":             float(t.get("shares", 0) or 0),
            "price_per_share":    float(t.get("price_per_share", 0) or 0),
            "value_usd":          float(t.get("value_usd", 0) or 0),
            "ownership_type":     t.get("ownership_type", "D"),
            "shares_owned_after": float(t.get("shares_owned_after", 0) or 0),
            "ingested_at":        t.get("ingested_at", ""),
        }

        key = (year, ticker)
        partitions.setdefault(key, []).append(row)

    if not partitions:
        print("No new transactions to process")
        return

    total = 0
    for (year, ticker), rows in sorted(partitions.items()):
        print(f"\n  year={year} ticker={ticker}: {len(rows)} transactions")
        write_partition(rows, year, ticker)
        total += len(rows)

    print(f"\n[etl_insiders] Done — {total} new rows written")

if __name__ == "__main__":
    main()