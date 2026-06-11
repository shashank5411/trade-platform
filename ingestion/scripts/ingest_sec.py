"""
SEC EDGAR — company filings + financials ingestion
Dataset: per-company watermarks | Frequency: quarterly
Note: SEC FAIR ACCESS policy — max 10 req/sec, User-Agent required.
      Company facts JSON can be 10-50 MB per company.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.watermark import get_watermark, update_watermark, \
    load_sec_tracker, save_sec_tracker, get_new_accessions, mark_accessions_done

import boto3
import gzip
import pandas as pd
import requests
from edgar import Company as EdgarCompany, set_identity
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

# edgartools also talks to EDGAR — use same identity
set_identity("TradePlatform research@example.com")

# 10-K sections: attribute name on TenK object
TENK_SECTIONS = {
    "item_1":   "business",
    "item_1a":  "risk_factors",
    "item_7":   "management_discussion",
}

# 10-Q sections: key into TenQ["..."]
TENQ_SECTIONS = {
    "item_2":   "part_i_item_2",    # MD&A
    "item_1a":  "part_ii_item_1a",  # Risk Factors
}

# Prose retention window
PROSE_CUTOFF = {
    "dev":  {"10-K": "2023-01-01", "10-Q": "2025-01-01"},
    "prod": {"10-K": "2019-01-01", "10-Q": "2023-01-01"},
}


def parse_args():
    p = argparse.ArgumentParser(description="Ingest SEC EDGAR company filings")
    p.add_argument("--start-date", help="Start date YYYY-MM-DD")
    p.add_argument("--end-date",   help="End date   YYYY-MM-DD")
    return p.parse_args()


def _resolve_start(tickers: list, config: dict, args) -> str:
    if getattr(args, "start_date", None):
        return args.start_date
    wms  = [get_watermark(SOURCE, t) for t in tickers]
    live = [w for w in wms if w]
    if not live:
        return get_default_start(config)
    oldest = min(w["last_ingested_period"] for w in live)
    return subtract_days(oldest, config.get("max_lookback_days", 90))


def _is_within_prose_window(filed_date: str, form_type: str) -> bool:
    cutoffs = PROSE_CUTOFF.get(ENV, PROSE_CUTOFF["dev"])
    ft      = "10-K" if "10-K" in form_type else "10-Q"
    cutoff  = cutoffs.get(ft, "2020-01-01")
    return filed_date >= cutoff


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


def fetch_prose_sections(ticker: str, accession_no: str,
                         form_type: str) -> dict:
    """
    Use edgartools to extract clean prose sections from a 10-K or 10-Q.
    Returns dict of {section_name: text} — empty dict on any failure.
    """
    try:
        company = EdgarCompany(ticker)

        # Find the specific filing by accession number
        form    = "10-K" if "10-K" in form_type else "10-Q"
        filings = company.get_filings(form=form)

        filing = None
        for f in filings:
            if f.accession_no == accession_no:
                filing = f
                break

        if filing is None:
            print(f"    WARN: accession {accession_no} not found via edgartools")
            return {}

        doc      = filing.obj()
        sections = {}

        if form == "10-K":
            for section_name, attr in TENK_SECTIONS.items():
                try:
                    text = getattr(doc, attr, None)
                    if text:
                        sections[section_name] = str(text)
                        print(f"    Got {section_name}: "
                              f"{len(str(text))//1024}KB")
                    else:
                        print(f"    WARN: {section_name} ({attr}) is empty")
                except Exception as e:
                    print(f"    WARN: failed to get {section_name}: {e}")

        else:  # 10-Q
            for section_name, key in TENQ_SECTIONS.items():
                try:
                    text = doc[key]
                    if text:
                        sections[section_name] = str(text)
                        print(f"    Got {section_name}: "
                              f"{len(str(text))//1024}KB")
                    else:
                        print(f"    WARN: {section_name} ({key}) is empty")
                except Exception as e:
                    print(f"    WARN: failed to get {section_name}: {e}")

        return sections

    except Exception as e:
        print(f"    WARN: edgartools failed for {accession_no}: {e}")
        return {}


def fetch_company(cik: str, ticker: str, tracker: dict) -> tuple:
    """Fetch submissions, facts, and prose sections via edgartools."""
    cik_padded  = cik.zfill(10)
    submissions = fetch_submissions(cik_padded)
    facts       = fetch_with_retry(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    )

    recent     = submissions.get("filings", {}).get("recent", {})
    accessions = recent.get("accessionNumber", [])
    forms      = recent.get("form", [])
    dates      = recent.get("filingDate", [])

    all_target_accns = [
        accn for accn, form, filed in zip(accessions, forms, dates)
        if form in ("10-K", "10-Q")
        and _is_within_prose_window(filed, form)
    ]
    new_accns = get_new_accessions(
        tracker, "fetched_accessions", all_target_accns
    )

    filing_docs = {}
    for accn, form, filed in zip(accessions, forms, dates):
        if form not in ("10-K", "10-Q"):
            continue
        if not _is_within_prose_window(filed, form):
            continue
        if accn not in new_accns:
            print(f"    Skipping {accn} — already fetched")
            continue

        print(f"    Fetching {form} prose {accn} ({filed})...")
        sections = fetch_prose_sections(ticker, accn, form)

        if sections:
            filing_docs[accn] = {
                "form":     form,
                "filed":    filed,
                "sections": sections,   # dict of {section_name: text}
            }
            print(f"    Got {len(sections)} sections for {accn}")
        else:
            print(f"    WARN: no sections extracted for {accn}")

    return submissions, facts, filing_docs, new_accns


def upload_company(ticker: str, data: dict,
                   filing_docs: dict, timestamp: str,
                   tracker: dict, new_accns: list) -> str:
    """Upload company data and filing documents to S3."""
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

    # Upload prose sections (one file per filing, sections nested inside)
    docs_uploaded = 0
    for accn, doc_info in filing_docs.items():
        doc_body = gzip.compress(
            json.dumps({
                "ticker":    ticker,
                "accession": accn,
                "form":      doc_info["form"],
                "filed":     doc_info["filed"],
                "sections":  doc_info["sections"],  # {section_name: text}
            }, default=str).encode()
        )
        doc_key = (f"year={year}/month={month}/prose/"
                   f"{SOURCE}_{ticker}_{accn}_{timestamp}.json.gz")
        S3.put_object(Bucket=BUCKET, Key=doc_key, Body=doc_body,
                      ContentType="application/json", ContentEncoding="gzip")
        docs_uploaded += 1

    print(f"    Uploaded {docs_uploaded} new prose filings")

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

            submissions, facts, filing_docs, new_accns = fetch_company(
                cik, ticker, tracker
            )
            payload = {
                "ticker":      ticker,
                "cik":         cik,
                "submissions": submissions,
                "facts":       facts,
            }
            key = upload_company(
                ticker, payload, filing_docs, timestamp, tracker, new_accns
            )
            print(f"    Uploaded → s3://{BUCKET}/{key}")

            recent     = submissions.get("filings", {}).get("recent", {})
            filing_cnt = len(recent.get("form", []))
            fact_ns    = list(facts.keys()) if isinstance(facts, dict) else []
            summary_rows.append({
                "ticker":          ticker,
                "recent_filings":  filing_cnt,
                "fact_namespaces": len(fact_ns),
                "prose_docs":      len(filing_docs),
            })
            update_watermark(SOURCE, ticker, end_date, "success", filing_cnt)

        except Exception as e:
            print(f"    ERROR: {e}")
            summary_rows.append({
                "ticker":          ticker,
                "recent_filings":  None,
                "fact_namespaces": None,
                "prose_docs":      None,
            })
            update_watermark(SOURCE, ticker, end_date, "error", 0)

    if not summary_rows or all(r["recent_filings"] is None
                               for r in summary_rows):
        print("No data uploaded.")
        return

    print_summary(summary_rows)


if __name__ == "__main__":
    main()