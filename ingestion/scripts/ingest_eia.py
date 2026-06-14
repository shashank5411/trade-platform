"""
EIA petroleum trade — incremental ingestion
Dataset: crude_oil_trade | Granularity: monthly
Key: EIA_API_KEY  (register free at https://www.eia.gov/opendata/register.php)
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
from datetime import datetime, timezone

import boto3
import pandas as pd
import requests

from utils.config import get_default_start, load_source_config
from utils.periods import resolve_periods
from utils.watermark import get_watermark, update_watermark

SOURCE  = "eia"
DATASET = "crude_oil_trade"

ENV        = os.getenv("ENVIRONMENT", "dev")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
BUCKET     = f"{ENV}-trade-eia-raw-{ACCOUNT_ID}"
S3         = boto3.client("s3", region_name="us-east-2")

BASE_URL   = "https://api.eia.gov/v2"
API_KEY    = os.getenv("EIA_API_KEY", "")
ENDPOINTS  = {
    "imports": "petroleum/move/imp/",
    "exports": "petroleum/move/exp/",
}


def parse_args():
    p = argparse.ArgumentParser(description="Ingest EIA crude oil trade data")
    p.add_argument("--start-period", help="Start month e.g. 2023-01")
    p.add_argument("--end-period",   help="End month   e.g. 2023-12")
    p.add_argument("--debug", action="store_true", help="Test data[0] param against EIA electricity endpoint")
    return p.parse_known_args()[0]


def debug_param_test():
    import urllib.request as _urllib
    import json as _json
    print("\n--- DEBUG: testing data[0] param on EIA electricity endpoint ---")
    url = f"https://api.eia.gov/v2/electricity/retail-sales?api_key={API_KEY}&data[0]=sales&start=2020-01&end=2020-03&length=3"
    req = _urllib.Request(url, headers={"Accept": "application/json"})
    with _urllib.urlopen(req, timeout=30) as r:
        body = _json.loads(r.read().decode())
    data = body.get("response", {}).get("data", [])
    print(f"  data type : {type(data).__name__}")
    print(f"  count     : {len(data) if isinstance(data, list) else 'schema (dict)'}")
    if isinstance(data, list) and data:
        print(f"  first row : {data[0]}")
    elif isinstance(data, dict):
        print(f"  schema keys: {list(data.keys())}")
        print("  → data[0] param NOT working — EIA returning schema instead of rows")
    print("--- END DEBUG ---\n")


def fetch_petroleum(route: str, start: str, end: str) -> list:
    # Use http.client directly — requests/urllib3 2.x encodes brackets (%5B%5D)
    # but EIA v2 API requires literal brackets in the query string
    import http.client
    import json as _json

    import urllib.request as _urllib

    url = (
        f"https://api.eia.gov/v2/{route}"
        f"?api_key={API_KEY}"
        f"&frequency=monthly"
        f"&data[0]=value"
        f"&facets[product][]=EPC0"
        f"&start={start}"
        f"&end={end}"
        f"&sort[0][column]=period"
        f"&sort[0][direction]=desc"
        f"&length=5000"
    )
    req = _urllib.Request(url, headers={"Accept": "application/json"})
    with _urllib.urlopen(req, timeout=30) as r:
        body = _json.loads(r.read().decode())

    response_section = body.get("response", {})
    data = response_section.get("data", []) if isinstance(response_section, dict) else []
    if not isinstance(data, list):
        available_cols = list(data.keys()) if isinstance(data, dict) else data
        avail_facets   = {f["id"]: f.get("options", "?") for f in response_section.get("facets", [])}
        print(f"  Got schema response (data[0] not recognised or column invalid).")
        print(f"  Available data columns : {available_cols}")
        print(f"  Available facets       : {list(avail_facets.keys())}")
        return []
    return data


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"Columns:    {list(df.columns)}")
    print(f"Row count:  {len(df)}")
    print(f"\nFirst 5 rows:\n{df.head().to_string()}")
    print(f"\nData types:\n{df.dtypes.to_string()}")
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")
    print("=" * 60)


def main():
    args = parse_args()

    if not API_KEY:
        print(
            "ERROR: EIA_API_KEY not set.\n"
            "Register at https://www.eia.gov/opendata/register.php"
        )
        return

    if args.debug:
        debug_param_test()
        return
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

    timestamp   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_records = []

    for flow, route in ENDPOINTS.items():
        print(f"  Fetching crude oil {flow} {start_period} → {end_period}...")
        records = fetch_petroleum(route, start_period, end_period)
        for r in records:
            r["flow"] = flow
        all_records.extend(records)
        print(f"    {len(records)} records")

    key = f"eia/{timestamp}_crude_oil_trade_{start_period}_{end_period}.json"
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
