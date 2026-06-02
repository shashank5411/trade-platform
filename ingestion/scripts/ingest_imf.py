"""
IMF Direction of Trade Statistics (DOTS) — incremental ingestion
Dataset: dots_india_usa | Granularity: annual
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
from utils.periods import resolve_periods
from utils.watermark import get_watermark, update_watermark

SOURCE  = "imf"
DATASET = "dots_india_usa"

ENV        = os.getenv("ENVIRONMENT", "dev")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
BUCKET     = f"{ENV}-trade-imf-raw-{ACCOUNT_ID}"
S3         = boto3.client("s3")

BASE_URL = "https://dataservices.imf.org/REST/SDMX_JSON.svc"


def parse_args():
    p = argparse.ArgumentParser(description="Ingest IMF DOTS bilateral trade data")
    p.add_argument("--start-period", help="Start year e.g. 2020")
    p.add_argument("--end-period",   help="End year   e.g. 2023")
    return p.parse_args()


def fetch_dots(start: str, end: str) -> dict:
    url = (
        f"{BASE_URL}/CompactData/DOT/"
        "A.IN+US.US+IN.TMG_FOB_USD+TXG_FOB_USD"
        f"?startPeriod={start}&endPeriod={end}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def parse_series(raw: dict) -> list:
    records = []
    series = raw["CompactData"]["DataSet"].get("Series", [])
    if isinstance(series, dict):
        series = [series]
    for s in series:
        obs_list = s.get("Obs", [])
        if isinstance(obs_list, dict):
            obs_list = [obs_list]
        for obs in obs_list:
            records.append({
                "reporter":     s["@REF_AREA"],
                "partner":      s["@COUNTERPART_AREA"],
                "indicator":    s["@INDICATOR"],
                "year":         obs["@TIME_PERIOD"],
                "value_usd_mn": obs.get("@OBS_VALUE"),
            })
    return records


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
    raw       = fetch_dots(start_period, end_period)
    records   = parse_series(raw)

    key = f"imf/{timestamp}_dots_india_usa_{start_period}_{end_period}.json"
    S3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(raw, indent=2),
        ContentType="application/json",
    )
    print(f"Uploaded → s3://{BUCKET}/{key}")
    update_watermark(SOURCE, DATASET, end_period, "success", len(records))
    print_summary(pd.DataFrame(records))


if __name__ == "__main__":
    main()
