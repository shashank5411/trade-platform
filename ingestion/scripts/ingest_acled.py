"""
ACLED conflict events — incremental ingestion
Dataset: conflict_events | Granularity: monthly
Keys: ACLED_API_KEY + ACLED_EMAIL  (register at https://acleddata.com/register/)
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
from calendar import monthrange
from datetime import datetime, timezone

import boto3
import pandas as pd
import requests

from utils.config import get_default_start, load_source_config
from utils.periods import resolve_periods
from utils.watermark import get_watermark, update_watermark

SOURCE  = "acled"
DATASET = "conflict_events"

ENV        = os.getenv("ENVIRONMENT", "dev")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
BUCKET     = f"{ENV}-trade-acled-raw-{ACCOUNT_ID}"
S3         = boto3.client("s3", region_name="us-east-2")

API_URL = "https://api.acleddata.com/acled/read"
API_KEY = os.getenv("ACLED_API_KEY", "")
EMAIL   = os.getenv("ACLED_EMAIL", "")

COUNTRY_GROUPS = {
    "south_asia": ["India", "Sri Lanka"],
    "red_sea":    ["Yemen", "Somalia", "Eritrea", "Djibouti", "Sudan"],
}


def parse_args():
    p = argparse.ArgumentParser(description="Ingest ACLED conflict event data")
    p.add_argument("--start-period", help="Start month e.g. 2023-01")
    p.add_argument("--end-period",   help="End month   e.g. 2023-12")
    return p.parse_known_args()[0]


def period_to_dates(start_period: str, end_period: str) -> tuple[str, str]:
    sy, sm = int(start_period[:4]), int(start_period[5:7])
    ey, em = int(end_period[:4]),   int(end_period[5:7])
    last_day = monthrange(ey, em)[1]
    return f"{sy}-{sm:02d}-01", f"{ey}-{em:02d}-{last_day:02d}"


def fetch_country(country: str, date_start: str, date_end: str) -> list:
    params = {
        "key":              API_KEY,
        "email":            EMAIL,
        "country":          country,
        "event_date":       f"{date_start}|{date_end}",
        "event_date_where": "BETWEEN",
        "limit":            5000,
    }
    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    if not body.get("success", True):
        raise RuntimeError(f"ACLED error for {country}: {body.get('error', body)}")
    return body.get("data", [])


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"Columns:    {list(df.columns)}")
    print(f"Row count:  {len(df)}")
    print(f"\nFirst 5 rows:\n{df.head().to_string()}")
    print(f"\nData types:\n{df.dtypes.to_string()}")
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")
    print("=" * 60)


def main():
    if not API_KEY or not EMAIL:
        print(
            "ERROR: ACLED_API_KEY and ACLED_EMAIL not set.\n"
            "Register at https://acleddata.com/register/"
        )
        return

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

    date_start, date_end = period_to_dates(start_period, end_period)
    timestamp   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_records = []

    for group, countries in COUNTRY_GROUPS.items():
        for country in countries:
            print(f"  Fetching {country} {date_start} → {date_end}...")
            try:
                records = fetch_country(country, date_start, date_end)
                for r in records:
                    r["country_group"] = group
                all_records.extend(records)
                print(f"    {len(records)} events")
            except Exception as exc:
                print(f"    WARNING: {exc}")

    if not all_records:
        print("No data returned.")
        return

    key = f"acled/{timestamp}_conflict_events_{start_period}_{end_period}.json"
    S3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(all_records, indent=2),
        ContentType="application/json",
    )
    print(f"\nUploaded → s3://{BUCKET}/{key}")
    update_watermark(SOURCE, DATASET, end_period, "success", len(all_records))
    print_summary(pd.DataFrame(all_records))


if __name__ == "__main__":
    main()
