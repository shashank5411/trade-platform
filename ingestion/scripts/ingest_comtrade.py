"""
UN Comtrade — incremental ingestion
Dataset: india_usa_trade | Granularity: annual
Key: COMTRADE_API_KEY  (register at https://comtradeplus.un.org/)
     Without key, falls back to free preview endpoint (500 records max).
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
import os
import sys
from datetime import datetime, timezone

import boto3
import comtradeapicall
import pandas as pd

from utils.config import get_default_start, load_source_config
from utils.periods import period_range, resolve_periods
from utils.watermark import get_watermark, update_watermark

SOURCE  = "comtrade"
DATASET = "india_usa_trade"

ENV        = os.getenv("ENVIRONMENT", "dev")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
BUCKET     = f"{ENV}-trade-comtrade-raw-{ACCOUNT_ID}"
S3         = boto3.client("s3", region_name="us-east-2")

API_KEY  = os.getenv("COMTRADE_API_KEY", "")
REPORTER = "356"   # India
PARTNER  = "842"   # USA


def parse_args():
    p = argparse.ArgumentParser(description="Ingest UN Comtrade India-USA trade data")
    p.add_argument("--start-period", help="Start year e.g. 2020")
    p.add_argument("--end-period",   help="End year   e.g. 2023")
    p.add_argument("--debug", action="store_true", help="Hit the API directly and print raw response")
    return p.parse_known_args()[0]


def debug_raw_request():
    """Try several parameter variations with delays to find what works."""
    import time
    import requests
    print("\n--- DEBUG: probing Comtrade API ---")
    print(f"Key (first 8 chars): {API_KEY[:8]}{'*' * max(0, len(API_KEY)-8)}\n")

    base_url = "https://comtradeapi.un.org/data/v1/get/C/A/HS"
    variants = [
        # Does any India data exist for 2023 at all?
        ("India reporter, no partner filter",
         {"reporterCode": "356", "period": "2023",
          "cmdCode": "TOTAL", "flowCode": "X", "subscription-key": API_KEY}),
        # Try USA (840) as reporter, India as partner
        ("USA reporter (840), India partner",
         {"reporterCode": "840", "period": "2023", "partnerCode": "356",
          "cmdCode": "TOTAL", "flowCode": "X", "subscription-key": API_KEY}),
        # Try earlier year — 2022 data more likely published
        ("India reporter, USA partner 842, year=2022",
         {"reporterCode": "356", "period": "2022", "partnerCode": "842",
          "cmdCode": "TOTAL", "flowCode": "X", "subscription-key": API_KEY}),
        # Try earlier year with USA=840
        ("India reporter, USA partner 840, year=2022",
         {"reporterCode": "356", "period": "2022", "partnerCode": "840",
          "cmdCode": "TOTAL", "flowCode": "X", "subscription-key": API_KEY}),
    ]

    for label, params in variants:
        time.sleep(2)   # respect rate limit
        try:
            resp = requests.get(base_url, params=params, timeout=30)
            body = resp.json()
            count = body.get("count", "?")
            error = body.get("error", "")
            print(f"[{label}]")
            print(f"  status={resp.status_code}  count={count}  error={error!r}\n")
        except Exception as exc:
            print(f"[{label}] FAILED: {exc}\n")
    print("--- END DEBUG ---\n")


def fetch_trade(start: str, end: str) -> pd.DataFrame:
    from datetime import date
    # Comtrade annual data has ~12-18 month lag; cap at 2 years back to be safe
    max_year = str(date.today().year - 2)
    effective_end   = min(end, max_year)
    effective_start = min(start, effective_end)
    period_str = ",".join(period_range(effective_start, effective_end, "year"))
    print(f"  Effective period: {effective_start} → {effective_end}  (Comtrade lag capped)")

    # cmdCode AG2 = all commodities aggregated at 2-digit HS level
    # "TOTAL" is not a valid code in the Comtrade+ API
    frames = []
    for flow in ("X", "M"):
        params = dict(
            typeCode="C", freqCode="A", clCode="HS",
            period=period_str, reporterCode=REPORTER,
            cmdCode="AG2", flowCode=flow, partnerCode=PARTNER,
            partner2Code="0", customsCode="C00", motCode="0",
            maxRecords=500, format_output="JSON",
            aggregateBy=None, breakdownMode="plus",
            countOnly=None, includeDesc=True,
        )
        if API_KEY:
            df = comtradeapicall.getFinalData(subscription_key=API_KEY, **params)
        else:
            print(f"  COMTRADE_API_KEY not set — using free preview for flow={flow}.")
            df = comtradeapicall.previewFinalData(**params)
        if df is not None and not df.empty:
            frames.append(df)
            print(f"  flow={flow}: {len(df)} records")
        else:
            print(f"  flow={flow}: no data returned")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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

    if args.debug:
        debug_raw_request()
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    df = fetch_trade(start_period, end_period)

    if df is None or df.empty:
        print("No data returned — check API key or parameters.")
        return

    key = f"comtrade/{timestamp}_india_usa_trade_{start_period}_{end_period}.json"
    S3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=df.to_json(orient="records", indent=2),
        ContentType="application/json",
    )
    print(f"Uploaded → s3://{BUCKET}/{key}")
    update_watermark(SOURCE, DATASET, end_period, "success", len(df))
    print_summary(df)


if __name__ == "__main__":
    main()
