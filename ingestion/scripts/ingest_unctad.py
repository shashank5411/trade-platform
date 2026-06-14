"""
UNCTAD seaborne trade — incremental ingestion
Dataset: seaborne_shipments | Granularity: annual
Note: UNCTAD API returns a full bulk dataset; period args track watermark only.
If DATA_URL returns 404, browse https://unctadstat.unctad.org/datacentre/
and update it with the direct CSV export link for "seaborne shipments".
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
import io
import json
import os
import sys
from datetime import datetime, timezone

import boto3
import pandas as pd
import requests

from utils.config import get_default_start, load_source_config
from utils.periods import resolve_periods
from utils.watermark import get_watermark, update_watermark

SOURCE  = "unctad"
DATASET = "seaborne_shipments"

ENV        = os.getenv("ENVIRONMENT", "dev")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
BUCKET     = f"{ENV}-trade-unctad-raw-{ACCOUNT_ID}"
S3         = boto3.client("s3", region_name="us-east-2")

DATA_URL = (
    "https://unctadstat.unctad.org/api/stats/SummaryExport/en/data/"
    "us_MaritimeSummary_e"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (trade-platform ingestion)",
    "Accept":     "text/csv, application/json, */*",
}


def parse_args():
    p = argparse.ArgumentParser(description="Ingest UNCTAD seaborne trade data")
    p.add_argument("--start-period", help="Start year e.g. 2020  (watermark tracking only)")
    p.add_argument("--end-period",   help="End year   e.g. 2023  (watermark tracking only)")
    return p.parse_known_args()[0]


def fetch_seaborne() -> bytes:
    resp = requests.get(DATA_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.content


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
    print(
        f"[{SOURCE}] tracking {start_period} → {end_period}  ({source_label})\n"
        "  Note: UNCTAD fetches the full dataset regardless of period range."
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_bytes = fetch_seaborne()

    ext, content_type = "csv", "text/csv"
    try:
        json.loads(raw_bytes)
        ext, content_type = "json", "application/json"
    except Exception:
        pass

    key = f"unctad/{timestamp}_seaborne_shipments_{end_period}.{ext}"
    S3.put_object(Bucket=BUCKET, Key=key, Body=raw_bytes, ContentType=content_type)
    print(f"Uploaded → s3://{BUCKET}/{key}")
    update_watermark(SOURCE, DATASET, end_period, "success", -1)

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception:
        df = pd.DataFrame(json.loads(raw_bytes) if ext == "json" else [{}])
    print_summary(df)


if __name__ == "__main__":
    main()
