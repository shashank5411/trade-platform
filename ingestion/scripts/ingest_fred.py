"""
FRED economic data — incremental ingestion
Dataset: per-series watermarks | Frequency: monthly/quarterly
Key: FRED_API_KEY  (free at https://fred.stlouisfed.org/docs/api/api_key.html)
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
from datetime import datetime, timezone

import boto3
import pandas as pd
from fredapi import Fred

from utils.config import get_default_start, load_source_config
from utils.dates import current_date_str, subtract_days
from utils.watermark import get_watermark, update_watermark


def _arg(name: str, default: str = "") -> str:
    for i, a in enumerate(sys.argv):
        if a == f"--{name}" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.getenv(name, default)


SOURCE = "fred"

ENV        = _arg("ENVIRONMENT", "dev")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
BUCKET     = f"{ENV}-trade-{SOURCE}-raw-{ACCOUNT_ID}"
S3         = boto3.client("s3", region_name="us-east-2")


def _get_fred_api_key() -> str:
    val = os.getenv("FRED_API_KEY", "")
    if val:
        return val
    secret_name = _arg("FRED_SECRET_NAME", f"trade-platform/{ENV}/fred-api-key")
    try:
        sm = boto3.client("secretsmanager")
        return sm.get_secret_value(SecretId=secret_name)["SecretString"]
    except Exception as e:
        print(f"WARNING: Could not load FRED key from Secrets Manager: {e}")
        return ""


def parse_args():
    p = argparse.ArgumentParser(description="Ingest FRED economic indicators")
    p.add_argument("--start-date", help="Start date YYYY-MM-DD")
    p.add_argument("--end-date",   help="End date   YYYY-MM-DD")
    return p.parse_known_args()[0]


def _resolve_start(series_ids: list, config: dict, args) -> str:
    if getattr(args, "start_date", None):
        return args.start_date
    wms = [get_watermark(SOURCE, sid) for sid in series_ids]
    live = [w for w in wms if w]
    if not live:
        return get_default_start(config)
    oldest = min(w["last_ingested_period"] for w in live)
    return subtract_days(oldest, config.get("max_lookback_days", 90))


def fetch_and_upload(fred: Fred, series: dict, start: str, end: str, timestamp: str) -> list:
    records = []
    for series_id, label in series.items():
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            obs = [
                {
                    "series_id":    series_id,
                    "series_label": label,
                    "date":         dt.strftime("%Y-%m-%d"),
                    "value":        None if pd.isna(val) else float(val),
                }
                for dt, val in s.items()
            ]
            print(f"    {series_id}: {len(obs)} observations")
            records.extend(obs)
            update_watermark(SOURCE, series_id, end, "success", len(obs))
        except Exception as e:
            print(f"    {series_id}: ERROR — {e}")
        time.sleep(0.5)   # FRED rate limit: 120 req/min
    return records


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"Columns:    {list(df.columns)}")
    print(f"Shape:      {df.shape}")
    print(f"\nFirst 5 rows:\n{df.head().to_string()}")
    print(f"\nData types:\n{df.dtypes.to_string()}")
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")
    print("=" * 60)


def main():
    api_key = _get_fred_api_key()
    if not api_key:
        print(
            "ERROR: FRED_API_KEY not set.\n"
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Then add to set_env.ps1:  $env:FRED_API_KEY = 'your_key'"
        )
        return

    args   = parse_args()
    config = load_source_config(SOURCE)
    series = config["series"]

    start_date = _resolve_start(list(series.keys()), config, args)
    end_date   = getattr(args, "end_date", None) or current_date_str()
    source_label = "CLI" if args.start_date else "watermark/first-run"
    print(f"[{SOURCE}] {start_date} → {end_date}  ({source_label})")

    fred      = Fred(api_key=api_key)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print("  Fetching FRED series...")
    records = fetch_and_upload(fred, series, start_date, end_date, timestamp)

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
    print_summary(pd.DataFrame(records))


if __name__ == "__main__":
    main()
