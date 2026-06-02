"""
WTO merchandise trade — incremental ingestion
Dataset: merchandise_trade | Granularity: annual

Two modes:
  1. API mode  (set WTO_API_KEY):  automatic fetch via api.wto.org
  2. CSV mode  (--csv-file PATH):  load a manually downloaded CSV from stats.wto.org

Manual CSV download steps (no key needed):
  1. Go to https://stats.wto.org/
  2. Indicator → "Merchandise trade value" → select ITS_MTV_AX + ITS_MTV_AM
  3. Reporter  → World (000)
  4. Year      → 2015 to 2023
  5. Click "Download" → CSV
  6. Run: python ingest_wto.py --csv-file path/to/file.csv
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
import requests

from utils.config import get_default_start, load_source_config
from utils.periods import period_range, resolve_periods
from utils.watermark import get_watermark, update_watermark

SOURCE  = "wto"
DATASET = "merchandise_trade"

ENV        = os.getenv("ENVIRONMENT", "dev")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
BUCKET     = f"{ENV}-trade-wto-raw-{ACCOUNT_ID}"
S3         = boto3.client("s3")

API_BASE   = "https://api.wto.org/timeseries/v1"
API_KEY    = os.getenv("WTO_API_KEY", "")
INDICATORS = "ITS_MTV_AX,ITS_MTV_AM"


def parse_args():
    p = argparse.ArgumentParser(description="Ingest WTO merchandise trade data")
    p.add_argument("--start-period", help="Start year e.g. 2020")
    p.add_argument("--end-period",   help="End year   e.g. 2023")
    p.add_argument(
        "--csv-file",
        help="Path to manually downloaded CSV from stats.wto.org (skips API call)",
    )
    return p.parse_args()


def fetch_via_api(start: str, end: str) -> dict:
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}
    params  = {
        "i":    INDICATORS,
        "r":    "000",
        "p":    "000",
        "ps":   ",".join(period_range(start, end, "year")),
        "fmt":  "json",
        "max":  500,
        "head": "H",
    }
    resp = requests.get(f"{API_BASE}/data", headers=headers, params=params, timeout=30)
    if resp.status_code in (401, 403):
        raise RuntimeError(
            "WTO API key rejected or missing.\n"
            "  → Register at https://developer.wto.org/ and set WTO_API_KEY\n"
            "  → Or download the CSV manually and use --csv-file (see script header)"
        )
    resp.raise_for_status()
    return resp.json()


def load_csv(path: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise column names to lowercase with underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Filter to requested period range if a year-like column exists
    year_col = next((c for c in df.columns if "year" in c or "period" in c), None)
    if year_col:
        df[year_col] = df[year_col].astype(str)
        df = df[df[year_col].between(start, end)]
    return df


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"Columns:    {list(df.columns)}")
    print(f"Row count:  {len(df)}")
    print(f"\nFirst 5 rows:\n{df.head().to_string()}")
    print(f"\nData types:\n{df.dtypes.to_string()}")
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")
    print("=" * 60)


def main():
    args   = parse_args()
    config = load_source_config(SOURCE)
    wm     = get_watermark(SOURCE, DATASET)

    start_period, end_period = resolve_periods(
        args,
        config["incremental_granularity"],
        config.get("max_lookback_periods", 3),
        wm,
        get_default_start(config),
    )
    source_label = "CLI" if args.start_period else ("watermark" if wm else "first-run")
    print(f"[{SOURCE}] {start_period} → {end_period}  ({source_label})")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if args.csv_file:
        # ── CSV mode ──────────────────────────────────────────────────────────
        print(f"  Loading from CSV: {args.csv_file}")
        df = load_csv(args.csv_file, start_period, end_period)
        body     = df.to_csv(index=False).encode()
        ext, ct  = "csv", "text/csv"
        records  = len(df)
    elif API_KEY:
        # ── API mode ──────────────────────────────────────────────────────────
        print("  Using WTO API...")
        raw     = fetch_via_api(start_period, end_period)
        records_list = raw.get("Dataset", [])
        body    = json.dumps(raw, indent=2).encode()
        ext, ct = "json", "application/json"
        records = len(records_list)
        df      = pd.DataFrame(records_list) if records_list else pd.DataFrame([raw])
    else:
        print(
            "\nERROR: No WTO_API_KEY set and no --csv-file provided.\n"
            "Options:\n"
            "  1. Set WTO_API_KEY (register at https://developer.wto.org/)\n"
            "  2. Download CSV manually from https://stats.wto.org/ "
            "and pass --csv-file path/to/file.csv\n"
            "     Steps: Indicator → ITS_MTV_AX + ITS_MTV_AM | "
            "Reporter → World | Year → 2015-2023 | Download CSV"
        )
        return

    key = f"wto/{timestamp}_merchandise_trade_{start_period}_{end_period}.{ext}"
    S3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType=ct)
    print(f"Uploaded → s3://{BUCKET}/{key}")
    update_watermark(SOURCE, DATASET, end_period, "success", records)
    print_summary(df)


if __name__ == "__main__":
    main()
