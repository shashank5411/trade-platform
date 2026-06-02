"""
SEC EDGAR — company filings + financials ingestion
Dataset: per-company watermarks | Frequency: quarterly
Note: SEC FAIR ACCESS policy — max 10 req/sec, User-Agent required.
      Company facts JSON can be 10-50 MB per company.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
import pandas as pd
import requests

from utils.config import get_default_start, load_source_config
from utils.dates import current_date_str, subtract_days
from utils.watermark import get_watermark, update_watermark


def _arg(name: str, default: str = "") -> str:
    for i, a in enumerate(sys.argv[1:], 1):
        if a == f"--{name}" and i < len(sys.argv):
            return sys.argv[i]
    return os.getenv(name, default)


SOURCE = "sec"

ENV        = _arg("ENVIRONMENT", "dev")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
BUCKET     = f"{ENV}-trade-{SOURCE}-raw-{ACCOUNT_ID}"
S3         = boto3.client("s3")

HEADERS   = {"User-Agent": "TradePlatform research@example.com"}
REQ_DELAY = 0.15    # ~6 req/sec — safely under EDGAR 10 req/sec limit


def parse_args():
    p = argparse.ArgumentParser(description="Ingest SEC EDGAR company filings")
    p.add_argument("--start-date", help="Start date YYYY-MM-DD")
    p.add_argument("--end-date",   help="End date   YYYY-MM-DD")
    return p.parse_args()


def _resolve_start(tickers: list, config: dict, args) -> str:
    if getattr(args, "start_date", None):
        return args.start_date
    wms = [get_watermark(SOURCE, t) for t in tickers]
    live = [w for w in wms if w]
    if not live:
        return get_default_start(config)
    oldest = min(w["last_ingested_period"] for w in live)
    return subtract_days(oldest, config.get("max_lookback_days", 90))


def fetch_with_retry(url: str, max_attempts: int = 3) -> dict:
    for attempt in range(max_attempts):
        time.sleep(REQ_DELAY)
        resp = requests.get(url, headers=HEADERS, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"    Rate limited — waiting {wait}s...")
            time.sleep(wait)
        else:
            resp.raise_for_status()
    raise RuntimeError(f"Failed after {max_attempts} attempts: {url}")


def fetch_company(ticker: str, cik: str) -> tuple[dict, dict]:
    base       = "https://data.sec.gov"
    cik_padded = cik.zfill(10)
    submissions = fetch_with_retry(f"{base}/submissions/CIK{cik_padded}.json")
    facts       = fetch_with_retry(f"{base}/api/xbrl/companyfacts/CIK{cik_padded}.json")
    return submissions, facts


def upload_company(ticker: str, data: dict, timestamp: str) -> str:
    year, month = timestamp[:4], timestamp[4:6]
    key = f"year={year}/month={month}/{SOURCE}_{ticker}_{timestamp}.json"
    S3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(data, default=str, indent=2),
        ContentType="application/json",
    )
    return key


def print_summary(summary_rows: list) -> None:
    df = pd.DataFrame(summary_rows)
    print(f"\n{'='*60}")
    print(f"Columns:    {list(df.columns)}")
    print(f"Shape:      {df.shape}")
    print(f"\nAll companies:\n{df.to_string()}")
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")
    print("=" * 60)


def main():
    args      = parse_args()
    config    = load_source_config(SOURCE)
    companies = config["companies"]

    start_date = _resolve_start(list(companies.keys()), config, args)
    end_date   = getattr(args, "end_date", None) or current_date_str()
    source_label = "CLI" if args.start_date else "watermark/first-run"
    print(f"[{SOURCE}] {start_date} → {end_date}  ({source_label})")

    timestamp    = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_rows = []

    for ticker, cik in companies.items():
        print(f"  Fetching {ticker} (CIK {cik})...")
        try:
            submissions, facts = fetch_company(ticker, cik)
            payload = {
                "ticker":      ticker,
                "cik":         cik,
                "submissions": submissions,
                "facts":       facts,
            }
            key = upload_company(ticker, payload, timestamp)
            print(f"    Uploaded → s3://{BUCKET}/{key}")

            recent     = submissions.get("filings", {}).get("recent", {})
            filing_cnt = len(recent.get("form", []))
            fact_ns    = list(facts.keys()) if isinstance(facts, dict) else []
            summary_rows.append({
                "ticker":          ticker,
                "recent_filings":  filing_cnt,
                "fact_namespaces": len(fact_ns),
            })
            update_watermark(SOURCE, ticker, end_date, "success", filing_cnt)
        except Exception as e:
            print(f"    ERROR: {e}")
            summary_rows.append({"ticker": ticker, "recent_filings": None, "fact_namespaces": None})
            update_watermark(SOURCE, ticker, end_date, "error", 0)

    if not summary_rows or all(r["recent_filings"] is None for r in summary_rows):
        print("No data uploaded.")
        return

    print_summary(summary_rows)


if __name__ == "__main__":
    main()
