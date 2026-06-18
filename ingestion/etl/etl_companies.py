"""
ETL: yfinance raw → companies reference table (full-refresh).

Reads latest raw yfinance JSON, deduplicates by ticker (latest ingested_at),
enriches with sp500 flag from sec_sp500.yaml, writes one Parquet file to:
  s3://{ENV}-trade-yfinance-processed-{ACCOUNT}/companies/data.parquet

Full-refresh on every run — deletes and rewrites the single output file.
"""
import sys
import os
import zipfile

# Fix A: extract --extra-py-files zip so utils/ packages are importable
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
from io import BytesIO
from typing import Optional

sys.path.insert(0, _libs_dir)
sys.path.insert(0, "/tmp/ingestion")

from utils.transform import safe_float


# Fix B: _arg() with enumerate(sys.argv) pattern
def _arg(name: str, default: str = "") -> str:
    for i, a in enumerate(sys.argv):
        if a == f"--{name}" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.getenv(name, default)


ENV     = _arg("ENVIRONMENT", "dev")
ACCOUNT = boto3.client("sts").get_caller_identity()["Account"]

# Fix D: region_name="us-east-2"
s3 = boto3.client("s3", region_name="us-east-2")

RAW_BUCKET  = f"{ENV}-trade-yfinance-raw-{ACCOUNT}"
PROC_BUCKET = f"{ENV}-trade-yfinance-processed-{ACCOUNT}"

# Fix F: config path via _libs_dir, not __file__
SP500_CONFIG_PATH = os.path.join(_libs_dir, "configs", "sources", "sec_sp500.yaml")


def load_sp500_tickers() -> set:
    try:
        with open(SP500_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        return set(cfg.get("tickers", []))
    except Exception as exc:
        print(f"  WARN: could not load sp500 config: {exc}")
        return set()


def read_latest_raw() -> list:
    """Read the most-recent bulk yfinance raw JSON from S3 (same pattern as etl_yfinance)."""
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
    print(f"  Loaded {len(records)} raw records")
    return records


def _safe_int(val) -> Optional[int]:
    """Convert to int, returning None on null/invalid."""
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def deduplicate(records: list) -> list:
    """Keep one record per ticker — the one with the latest ingested_at."""
    best = {}
    for rec in records:
        ticker = rec.get("ticker", "").upper()
        if not ticker:
            continue
        ts = rec.get("ingested_at", "")
        if ticker not in best or ts > best[ticker].get("ingested_at", ""):
            best[ticker] = rec
    return list(best.values())


def build_rows(records: list, sp500_set: set) -> list:
    rows = []
    now  = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for rec in records:
        ticker = rec.get("ticker", "").upper()
        rows.append({
            "ticker":         ticker,
            "company_name":   rec.get("company_name", "") or "",
            "sector":         rec.get("sector", "") or "",
            "industry":       rec.get("industry", "") or "",
            "exchange":       rec.get("exchange", "") or "",
            "country":        rec.get("country", "") or "",
            "currency":       rec.get("currency", "") or "",
            "city":           rec.get("city", "") or "",
            "state":          rec.get("state", "") or "",
            "market_cap":     safe_float(rec.get("market_cap")),
            "employees":      _safe_int(rec.get("employees")),
            "beta":           safe_float(rec.get("beta")),
            "dividend_yield": safe_float(rec.get("dividend_yield")),
            "pe_ratio":       safe_float(rec.get("pe_ratio")),
            "forward_pe":     safe_float(rec.get("forward_pe")),
            "week52_high":    safe_float(rec.get("week52_high")),
            "week52_low":     safe_float(rec.get("week52_low")),
            "avg_volume_10d": _safe_int(rec.get("avg_volume_10d")),
            "avg_volume_3m":  _safe_int(rec.get("avg_volume_3m")),
            "description":    rec.get("description", "") or "",
            "sp500":          ticker in sp500_set,
            "ingested_at":    rec.get("ingested_at", now),
        })
    return rows


def write_companies(rows: list) -> None:
    """Full-refresh: delete existing file, write new Parquet."""
    key = "companies/data.parquet"
    try:
        s3.delete_object(Bucket=PROC_BUCKET, Key=key)
    except Exception:
        pass

    df = pd.DataFrame(rows)

    # Fix G: explicit dtypes — no Python 3.10+ syntax
    df["market_cap"]     = df["market_cap"].astype("float64")
    df["beta"]           = df["beta"].astype("float64")
    df["dividend_yield"] = df["dividend_yield"].astype("float64")
    df["pe_ratio"]       = df["pe_ratio"].astype("float64")
    df["forward_pe"]     = df["forward_pe"].astype("float64")
    df["week52_high"]    = df["week52_high"].astype("float64")
    df["week52_low"]     = df["week52_low"].astype("float64")
    df["sp500"]          = df["sp500"].astype("bool")

    buf = BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
    buf.seek(0)
    s3.put_object(Bucket=PROC_BUCKET, Key=key, Body=buf.getvalue())
    print(f"[etl_companies] {len(df)} companies written to {PROC_BUCKET}")


def main():
    print(f"ETL companies | env={ENV}")
    print(f"  raw:       s3://{RAW_BUCKET}")
    print(f"  processed: s3://{PROC_BUCKET}")

    sp500_set = load_sp500_tickers()
    print(f"  sp500 tickers loaded: {len(sp500_set)}")

    records   = read_latest_raw()
    unique    = deduplicate(records)
    print(f"  {len(unique)} unique tickers after dedup")

    rows = build_rows(unique, sp500_set)
    write_companies(rows)


if __name__ == "__main__":
    main()
