"""
SEC EDGAR — company filings + financials ingestion
Dataset: per-company watermarks | Frequency: quarterly
Note: SEC FAIR ACCESS policy — max 10 req/sec, User-Agent required.
      Company facts JSON can be 10-50 MB per company.
"""
import sys
import os
import zipfile

# Glue puts --extra-py-files zip in glue-python-libs-* but doesn't extract it
# Extract it ourselves so utils/ is importable
_libs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _entry in os.listdir('/tmp/'):
    if _entry.startswith('glue-python-libs-'):
        _libs_dir = os.path.join('/tmp/', _entry)
        for _f in os.listdir(_libs_dir):
            if _f.endswith('.zip'):
                _zip_path = os.path.join(_libs_dir, _f)
                with zipfile.ZipFile(_zip_path) as _z:
                    _z.extractall(_libs_dir)
                sys.path.insert(0, _libs_dir)
        break
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from utils.watermark import get_watermark, update_watermark, \
    load_sec_tracker, save_sec_tracker, get_new_accessions, mark_accessions_done

import boto3
import gzip
import pandas as pd
import requests
from utils.config import get_default_start, load_source_config
from utils.dates import current_date_str, subtract_days


def _arg(name: str, default: str = "") -> str:
    for i, a in enumerate(sys.argv):
        if a == f"--{name}" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.getenv(name, default)

SOURCE = "sec"

ENV        = _arg("ENVIRONMENT", "dev")
ACCOUNT_ID = boto3.client("sts").get_caller_identity()["Account"]
BUCKET     = f"{ENV}-trade-{SOURCE}-raw-{ACCOUNT_ID}"
S3         = boto3.client("s3", region_name="us-east-2")

# SEC requires a real contact in the User-Agent (FAIR ACCESS policy) —
# set SEC_USER_AGENT to your own contact info before running against EDGAR.
HEADERS   = {"User-Agent": os.environ.get("SEC_USER_AGENT", "TradePlatform research@example.com")}
REQ_DELAY = 0.15    # ~6 req/sec — safely under EDGAR 10 req/sec limit


def parse_args():
    p = argparse.ArgumentParser(description="Ingest SEC EDGAR company filings")
    p.add_argument("--start-date", help="Start date YYYY-MM-DD")
    p.add_argument("--end-date",   help="End date   YYYY-MM-DD")
    return p.parse_known_args()[0]


def _resolve_start(tickers: list, config: dict, args) -> str:
    if getattr(args, "start_date", None):
        return args.start_date
    wms  = [get_watermark(SOURCE, t) for t in tickers]
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


def fetch_submissions(cik_padded: str) -> dict:
    """Fetch all submissions, paginating through continuation files."""
    base_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    data     = fetch_with_retry(base_url)

    filings  = data.get("filings", {})
    recent   = filings.get("recent", {})

    for file_info in filings.get("files", []):
        filename = file_info.get("name", "")
        if not filename:
            continue
        print(f"    Fetching continuation: {filename}")
        cont_data = fetch_with_retry(
            f"https://data.sec.gov/submissions/{filename}"
        )
        for field, values in cont_data.items():
            if isinstance(values, list) and field in recent:
                recent[field].extend(values)

    data["filings"]["recent"] = recent
    return data



def fetch_company(cik: str, ticker: str, tracker: dict) -> tuple:
    """Fetch submissions and facts from EDGAR."""
    cik_padded  = cik.zfill(10)
    submissions = fetch_submissions(cik_padded)
    # Fetch facts with size guard — large banks (JPM, BAC) have 15-37MB facts
    # which exceeds Python Shell memory limits. Skip facts for oversized companies.
    FACTS_SIZE_LIMIT_MB = 12
    facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    try:
        import urllib.request
        req = urllib.request.Request(facts_url, headers=HEADERS, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:
            content_length = int(resp.headers.get("Content-Length", 0))
            size_mb = content_length / 1024 / 1024
            if size_mb > FACTS_SIZE_LIMIT_MB:
                print(f"    Skipping facts for {ticker} — "
                      f"{size_mb:.1f}MB exceeds {FACTS_SIZE_LIMIT_MB}MB limit")
                facts = {}
            else:
                facts = fetch_with_retry(facts_url)
    except Exception:
        # HEAD not supported — fetch normally and handle OOM via Python Shell limits
        facts = fetch_with_retry(facts_url)

    recent     = submissions.get("filings", {}).get("recent", {})
    accessions = recent.get("accessionNumber", [])
    forms      = recent.get("form", [])

    all_target_accns = [
        accn for accn, form in zip(accessions, forms)
        if form in ("10-K", "10-Q")
    ]
    new_accns = get_new_accessions(
        tracker, "fetched_accessions", all_target_accns
    )

    return submissions, facts, new_accns


def upload_company(ticker: str, data: dict,
                   timestamp: str,
                   tracker: dict, new_accns: list) -> str:
    """Upload company data to S3."""
    year, month = timestamp[:4], timestamp[4:6]

    # Main payload (submissions + facts)
    uncompressed = json.dumps(data, default=str).encode()
    body         = gzip.compress(uncompressed)
    print(f"  [{ticker}] uncompressed: {len(uncompressed)/1024/1024:.1f}MB  "
          f"compressed: {len(body)/1024/1024:.1f}MB  "
          f"ratio: {len(uncompressed)/len(body):.1f}x")

    key = f"year={year}/month={month}/{SOURCE}_{ticker}_{timestamp}.json.gz"
    S3.put_object(Bucket=BUCKET, Key=key, Body=body,
                  ContentType="application/json", ContentEncoding="gzip")

    latest_key = f"latest/{SOURCE}_{ticker}_latest.json.gz"
    S3.put_object(Bucket=BUCKET, Key=latest_key, Body=body,
                  ContentType="application/json", ContentEncoding="gzip")

    # Update tracker
    tracker = mark_accessions_done(
        tracker, "fetched_accessions", "total_fetched", new_accns
    )
    tracker["last_ingest"] = datetime.now(timezone.utc).isoformat()
    save_sec_tracker(S3, BUCKET, ticker, tracker)

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

    start_date   = _resolve_start(list(companies.keys()), config, args)
    end_date     = getattr(args, "end_date", None) or current_date_str()
    source_label = "CLI" if args.start_date else "watermark/first-run"
    print(f"[{SOURCE}] {start_date} → {end_date}  ({source_label})")

    timestamp    = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_rows = []

    for ticker, cik in companies.items():
        print(f"  Fetching {ticker} (CIK {cik})...")
        try:
            tracker = load_sec_tracker(S3, BUCKET, ticker)
            print(f"  Tracker: {tracker['total_fetched']} accessions "
                  f"previously fetched")

            submissions, facts, new_accns = fetch_company(
                cik, ticker, tracker
            )
            payload = {
                "ticker":      ticker,
                "cik":         cik,
                "submissions": submissions,
                "facts":       facts,
            }
            key = upload_company(
                ticker, payload, timestamp, tracker, new_accns
            )
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
            summary_rows.append({
                "ticker":          ticker,
                "recent_filings":  None,
                "fact_namespaces": None,
            })
            update_watermark(SOURCE, ticker, end_date, "error", 0)

    if not summary_rows or all(r["recent_filings"] is None
                               for r in summary_rows):
        print("No data uploaded.")
        return

    print_summary(summary_rows)


if __name__ == "__main__":
    main()