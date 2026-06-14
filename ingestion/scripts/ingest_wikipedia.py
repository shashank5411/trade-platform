"""
Wikipedia REST API — economic context ingestion
Dataset: per-topic watermarks | Frequency: on-demand/weekly
No API key required.
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
import json
import os
import sys
import time
from datetime import datetime, timezone

import boto3
import pandas as pd
import requests

from utils.config import load_source_config
from utils.dates import current_date_str
from utils.watermark import update_watermark


def _arg(name: str, default: str = "") -> str:
    for i, a in enumerate(sys.argv):
        if a == f"--{name}" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.getenv(name, default)


SOURCE = "wikipedia"

ENV        = _arg("ENVIRONMENT", "dev")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
BUCKET     = f"{ENV}-trade-{SOURCE}-raw-{ACCOUNT_ID}"
S3         = boto3.client("s3", region_name="us-east-2")

BASE_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"
HEADERS  = {"User-Agent": "TradePlatform/1.0 (research@example.com)"}


def fetch_topic(topic: str) -> dict:
    resp = requests.get(f"{BASE_URL}/{topic}", headers=HEADERS, timeout=15)
    if resp.status_code == 404:
        return {"title": topic, "error": "not_found"}
    resp.raise_for_status()
    data = resp.json()
    return {
        "title":         data.get("title", topic),
        "display_title": data.get("displaytitle", topic),
        "description":   data.get("description", ""),
        "extract":       data.get("extract", ""),
        "last_modified": data.get("timestamp", ""),
        "url":           data.get("content_urls", {}).get("desktop", {}).get("page", ""),
    }


def print_summary(records: list) -> None:
    df = pd.DataFrame([{
        "title":       r.get("title", ""),
        "extract_len": len(r.get("extract", "")),
        "error":       r.get("error", None),
    } for r in records])
    print(f"\n{'='*60}")
    print(f"Columns:    {list(df.columns)}")
    print(f"Shape:      {df.shape}")
    print(f"\nAll topics:\n{df.to_string()}")
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")
    print("=" * 60)


def main():
    config    = load_source_config(SOURCE)
    topics    = config["topics"]
    today     = current_date_str()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"[{SOURCE}] fetching {len(topics)} topics  (snapshot {today})")

    records = []
    for topic in topics:
        print(f"  Fetching '{topic}'...")
        try:
            rec = fetch_topic(topic)
            records.append(rec)
            print(f"    {len(rec.get('extract', ''))} chars")
            update_watermark(SOURCE, topic, today, "success", 1)
        except Exception as e:
            print(f"    ERROR: {e}")
            records.append({"title": topic, "error": str(e)})
            update_watermark(SOURCE, topic, today, "error", 0)
        time.sleep(0.2)

    year, month = timestamp[:4], timestamp[4:6]
    key = f"year={year}/month={month}/{SOURCE}_{today}_{timestamp}.json"
    S3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(records, default=str, indent=2),
        ContentType="application/json",
    )
    print(f"Uploaded → s3://{BUCKET}/{key}  ({len(records)} topics)")
    print_summary(records)


if __name__ == "__main__":
    main()
