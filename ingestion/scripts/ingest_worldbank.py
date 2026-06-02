"""
World Bank WDI — incremental ingestion (via wbdata)
Dataset: per-indicator watermarks | Frequency: annual
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
import wbdata

from utils.config import get_default_start, load_source_config
from utils.dates import current_date_str, subtract_days
from utils.watermark import get_watermark, update_watermark


def _arg(name: str, default: str = "") -> str:
    for i, a in enumerate(sys.argv[1:], 1):
        if a == f"--{name}" and i < len(sys.argv):
            return sys.argv[i]
    return os.getenv(name, default)


SOURCE = "worldbank"

ENV        = _arg("ENVIRONMENT", "dev")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
BUCKET     = f"{ENV}-trade-{SOURCE}-raw-{ACCOUNT_ID}"
S3         = boto3.client("s3")


def parse_args():
    p = argparse.ArgumentParser(description="Ingest World Bank WDI data")
    p.add_argument("--start-date", help="Start date YYYY-MM-DD")
    p.add_argument("--end-date",   help="End date   YYYY-MM-DD")
    return p.parse_args()


def _resolve_start(indicator_codes: list, config: dict, args) -> str:
    if getattr(args, "start_date", None):
        return args.start_date
    wms = [get_watermark(SOURCE, code) for code in indicator_codes]
    live = [w for w in wms if w]
    if not live:
        return get_default_start(config)
    oldest = min(w["last_ingested_period"] for w in live)
    return subtract_days(oldest, config.get("max_lookback_days", 730))


def fetch_wdi(indicators: dict, countries: list, start: str, end: str) -> tuple[list, dict]:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    print(f"  Fetching {len(indicators)} indicators × {len(countries)} countries...")
    df = wbdata.get_dataframe(
        indicators,
        country=countries,
        date=(start_dt, end_dt),
    )
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    print(f"  {len(df)} rows")

    # Count rows per indicator (mapped column name) for per-indicator watermarks
    counts = {}
    for code, label in indicators.items():
        col = label.lower().replace(" ", "_")
        if col in df.columns:
            counts[code] = int(df[col].notna().sum())
        else:
            counts[code] = len(df)

    return df.to_dict(orient="records"), counts


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"Columns:    {list(df.columns)}")
    print(f"Shape:      {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head().to_string()}")
    print(f"\nData types:\n{df.dtypes.to_string()}")
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")
    print("=" * 60)


def main():
    args       = parse_args()
    config     = load_source_config(SOURCE)
    indicators = config["indicators"]
    countries  = config["countries"]

    start_date = _resolve_start(list(indicators.keys()), config, args)
    end_date   = getattr(args, "end_date", None) or current_date_str()
    source_label = "CLI" if args.start_date else "watermark/first-run"
    print(f"[{SOURCE}] {start_date} → {end_date}  ({source_label})")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records, counts = fetch_wdi(indicators, countries, start_date, end_date)

    if not records:
        print("No data returned.")
        return

    year, month = timestamp[:4], timestamp[4:6]
    key = f"year={year}/month={month}/{SOURCE}_{start_date}_{end_date}_{timestamp}.json"
    S3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(records, default=str, indent=2),
        ContentType="application/json",
    )
    print(f"Uploaded → s3://{BUCKET}/{key}  ({len(records)} records)")

    for code in indicators:
        update_watermark(SOURCE, code, end_date, "success", counts.get(code, 0))

    print_summary(pd.DataFrame(records))


if __name__ == "__main__":
    main()
