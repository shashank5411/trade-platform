"""
ETL: SEC EDGAR raw JSON → documents canonical Parquet.

Raw format: per-ticker EDGAR company facts API response
{
  "ticker": "AAPL",
  "cik": "0000320193",
  "submissions": {filings index},
  "facts": {us-gaap XBRL concepts}
}

Strategy:
- Extract 10-K and 10-Q filings from submissions index
- For each filing, build a structured text narrative from
  us-gaap financial facts (revenue, net income, assets etc.)
- Store as documents table rows — one row per filing
- text field = human-readable financial summary for RAG
"""
import sys
import os
import zipfile

# Glue places --extra-py-files zip in glue-python-libs-* but does not extract it
# Extract it manually so internal packages like utils/ are importable
_libs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _entry in os.listdir('/tmp/'):
    if _entry.startswith('glue-python-libs-'):
        _libs_dir = os.path.join('/tmp/', _entry)
        for _f in os.listdir(_libs_dir):
            if _f.endswith('.zip'):
                with zipfile.ZipFile(os.path.join(_libs_dir, _f)) as _z:
                    _z.extractall(_libs_dir)
        sys.path.insert(0, _libs_dir)
        break
import gzip

import os
import sys
import json
import boto3
import yaml
import pandas as pd
from datetime import date
from io import BytesIO
from typing import Optional

# ── Path setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, _libs_dir)
sys.path.insert(0, "/tmp/ingestion")
from utils.watermark import (
    load_sec_tracker, save_sec_tracker,
    get_new_accessions, mark_accessions_done,
)
def _arg(key, default=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--{key}", default=os.environ.get(key.upper(), default))
    args, _ = parser.parse_known_args()
    return getattr(args, key)

ENV     = _arg("env",     "dev")
ACCOUNT = _arg("account", "197411402303")

def _bucket(source, layer):
    return _arg(f"{layer}_bucket",
                f"{ENV}-trade-{source}-{layer}-{ACCOUNT}")

RAW_BUCKET  = _bucket("sec", "raw")
PROC_BUCKET = _bucket("sec", "processed")
CONFIG_PATH = _arg("config_path",
    os.path.join(_libs_dir, "configs", "sources", "sec.yaml"))

from utils.transform import (
    now_utc,
    make_doc_id,
    clean_text,
    to_json_str,
    validate_document_row,
)

s3 = boto3.client("s3", region_name="us-east-2")


# ── Financial concepts to extract ─────────────────────────────────────────
# (concept_name, human_label, unit, scale_divisor, scale_label)
# scale_divisor: divide raw value for readability (1e9 = billions)
KEY_CONCEPTS = [
    ("RevenueFromContractWithCustomerExcludingAssessedTax",
     "Revenue",                    "USD", 1e9,  "billions USD"),
    ("Revenues",
     "Revenue",                    "USD", 1e9,  "billions USD"),
    ("NetIncomeLoss",
     "Net Income",                 "USD", 1e9,  "billions USD"),
    ("GrossProfit",
     "Gross Profit",               "USD", 1e9,  "billions USD"),
    ("OperatingIncomeLoss",
     "Operating Income",           "USD", 1e9,  "billions USD"),
    ("Assets",
     "Total Assets",               "USD", 1e9,  "billions USD"),
    ("Liabilities",
     "Total Liabilities",          "USD", 1e9,  "billions USD"),
    ("StockholdersEquity",
     "Stockholders Equity",        "USD", 1e9,  "billions USD"),
    ("CashAndCashEquivalentsAtCarryingValue",
     "Cash and Equivalents",       "USD", 1e9,  "billions USD"),
    ("EarningsPerShareBasic",
     "EPS Basic",                  "USD", 1,    "USD per share"),
    ("EarningsPerShareDiluted",
     "EPS Diluted",                "USD", 1,    "USD per share"),
    ("CommonStockSharesOutstanding",
     "Shares Outstanding",         "shares", 1e6, "millions shares"),
    ("ResearchAndDevelopmentExpense",
     "R&D Expense",                "USD", 1e9,  "billions USD"),
    ("OperatingCashFlow",
     "Operating Cash Flow",        "USD", 1e9,  "billions USD"),
    ("NetCashProvidedByUsedInOperatingActivities",
     "Operating Cash Flow",        "USD", 1e9,  "billions USD"),
]

# Form types to extract — ignore 4, 8-K, SD etc.
TARGET_FORMS = {"10-K", "10-Q"}


# ── Config ─────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── S3 helpers ─────────────────────────────────────────────────────────────

def list_raw_keys() -> list:
    """Return one latest file per ticker from latest/ prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=RAW_BUCKET, Prefix="latest/"):
        keys.extend([o["Key"] for o in page.get("Contents", [])])
    keys = [k for k in keys
            if k.endswith(".json") or k.endswith(".json.gz")]
    print(f"  Latest files: {[k.split('/')[-1] for k in keys]}")
    return keys


def read_raw_file(key: str) -> dict:
    import gzip
    obj = s3.get_object(Bucket=RAW_BUCKET, Key=key)
    raw = obj["Body"].read()
    compressed_mb = len(raw) / 1024 / 1024
    if key.endswith(".gz"):
        raw = gzip.decompress(raw)
        decompressed_mb = len(raw) / 1024 / 1024
        print(f"  [{key.split('/')[-1]}] "
              f"compressed: {compressed_mb:.1f}MB  "
              f"decompressed: {decompressed_mb:.1f}MB  "
              f"ratio: {decompressed_mb/compressed_mb:.1f}x")
    else:
        print(f"  [{key.split('/')[-1]}] size: {compressed_mb:.1f}MB")
    return json.loads(raw)


def write_processed(rows: list) -> int:
    """Append new rows to existing Parquet, deduplicating on doc_id."""
    if not rows:
        print("  No rows to write")
        return 0

    df         = pd.DataFrame(rows)
    df["year"] = df["year"].astype(int)

    # Dedup within this batch first
    before = len(df)
    df = df.drop_duplicates(subset=["doc_id"], keep="last")
    if len(df) < before:
        print(f"  Deduped {before - len(df)} duplicate rows in batch")

    total = 0

    for (year, doc_type), group in df.groupby(["year", "doc_type"]):
        safe_type = doc_type.replace("-", "").replace("/", "")
        key = (f"documents/source=EDGAR/"
               f"year={year}/"
               f"form_type={safe_type}/"
               f"data.parquet")

        new_df = group.drop(columns=["year", "source", "doc_type"])

        # Read existing Parquet and merge if present
        try:
            obj         = s3.get_object(Bucket=PROC_BUCKET, Key=key)
            existing_df = pd.read_parquet(BytesIO(obj["Body"].read()))
            existing_ids = set(existing_df["doc_id"].tolist())
            new_df = new_df[~new_df["doc_id"].isin(existing_ids)]
            if new_df.empty:
                print(f"  No new rows for year={year} {doc_type} — skipping")
                continue
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"  Appending {len(new_df)} rows to {len(existing_df)} "
                  f"existing → year={year} {doc_type}")
        except s3.exceptions.NoSuchKey:
            combined = new_df
            print(f"  Writing {len(combined)} rows → year={year} {doc_type}")

        buf = BytesIO()
        combined.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
        buf.seek(0)
        s3.put_object(Bucket=PROC_BUCKET, Key=key, Body=buf.getvalue())
        total += len(new_df)

    return total


# ── Facts extraction ───────────────────────────────────────────────────────

def get_concept_value(facts_usgaap: dict, concept: str,
                      accession: str, form: str) -> Optional[float]:
    """
    Get the value for a specific concept from a specific filing.
    Matches by accession number and prefers period-specific (non-cumulative)
    entries — fp=Q1/Q2/Q3/FY rather than cumulative YTD.
    """
    if concept not in facts_usgaap:
        return None

    units = facts_usgaap[concept].get("units", {})
    values = units.get("USD") or units.get("shares") or []

    # Filter to this specific accession
    filing_vals = [v for v in values if v.get("accn") == accession]
    if not filing_vals:
        return None

    # For 10-K prefer FY, for 10-Q prefer quarterly (not YTD)
    if form == "10-K":
        fy_vals = [v for v in filing_vals if v.get("fp") == "FY"]
        if fy_vals:
            return fy_vals[-1]["val"]
    else:
        # For 10-Q prefer single quarter entries (have frame like CY2024Q1)
        q_vals = [v for v in filing_vals if v.get("frame", "").endswith(
            ("Q1", "Q2", "Q3", "Q4"))]
        if q_vals:
            return q_vals[-1]["val"]

    # Fall back to last entry
    return filing_vals[-1]["val"] if filing_vals else None


def build_financial_narrative(company_name: str, ticker: str,
                               form: str, period: str, fy: int,
                               metrics: dict) -> str:
    """
    Build a human-readable financial summary for RAG retrieval.
    Structured prose so LLM can extract specific numbers naturally.
    """
    lines = [
        f"{company_name} ({ticker}) — {form} Filing",
        f"Fiscal Period: {period}, Fiscal Year: {fy}",
        "",
        "Financial Highlights:",
    ]

    for label, value, scale_label in metrics.values():
        if value is not None:
            lines.append(f"  {label}: {value:.2f} {scale_label}")

    return clean_text("\n".join(lines))


# ── Transform ──────────────────────────────────────────────────────────────

def transform_company(raw: dict, tracker: dict) -> tuple:
    """
    Transform one company's EDGAR data into canonical document rows.
    Returns (rows, new_accessions_processed).
    """
    ingested_at  = now_utc()
    ticker       = raw.get("ticker", "").upper()
    cik          = raw.get("cik", "")
    submissions  = raw.get("submissions", {})
    company_name = submissions.get("name", ticker)
    fiscal_ye    = submissions.get("fiscalYearEnd", "")
    sic_code     = submissions.get("sic", "")
    exchange     = (submissions.get("exchanges") or [""])[0]

    facts_usgaap = (raw.get("facts", {})
                       .get("facts", {})
                       .get("us-gaap", {}))

    recent       = submissions.get("filings", {}).get("recent", {})
    accessions   = recent.get("accessionNumber", [])
    forms        = recent.get("form", [])
    dates        = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])

    # Get only accessions not yet transformed
    all_target = [a for a, f in zip(accessions, forms) if f in TARGET_FORMS]
    new_accns  = set(get_new_accessions(
        tracker, "transformed_accessions", all_target
    ))

    rows    = []
    skipped = 0

    for accn, form, filed, reported in zip(
            accessions, forms, dates, report_dates):

        if form not in TARGET_FORMS:
            continue
        if accn not in new_accns:
            continue  # already transformed

        try:
            filed_date = date.fromisoformat(filed)
        except (ValueError, TypeError):
            continue

        metrics = {}
        for concept, label, unit, divisor, scale_label in KEY_CONCEPTS:
            val = get_concept_value(facts_usgaap, concept, accn, form)
            if val is not None and label not in metrics:
                metrics[label] = (label, val / divisor, scale_label)

        fp     = "FY" if form == "10-K" else reported
        fy_val = filed_date.year
        text   = build_financial_narrative(
            company_name, ticker, form, fp, fy_val, metrics
        )
        doc_id = make_doc_id("EDGAR", accn)
        title  = f"{company_name} {form} {filed_date.year}"

        row = {
            "doc_id":      doc_id,
            "source":      "EDGAR",
            "year":        filed_date.year,
            "title":       title,
            "entity":      ticker,
            "doc_type":    form,
            "doc_date":    str(filed_date),
            "text":        text,
            "char_count":  len(text),
            "metadata":    to_json_str({
                "cik":              cik,
                "accession_number": accn,
                "fiscal_year_end":  fiscal_ye,
                "sic_code":         sic_code,
                "company_name":     company_name,
                "exchange":         exchange,
                "reported_date":    reported,
                "metrics_found":    list(metrics.keys()),
            }),
            "ingested_at": ingested_at,
        }

        errors = validate_document_row(row)
        if errors:
            print(f"  WARN: skipping {ticker}/{accn}: {errors}")
            skipped += 1
            continue

        rows.append(row)

    print(f"  {ticker}: {len(rows)} new filings extracted "
          f"({skipped} skipped, {len(all_target) - len(new_accns)} already done)")
    return rows, list(new_accns)


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    print(f"ETL SEC EDGAR → documents | env={ENV}")
    print(f"  raw:       s3://{RAW_BUCKET}")
    print(f"  processed: s3://{PROC_BUCKET}")

    config    = load_config()
    companies = config.get("companies", {})
    print(f"  Companies: {list(companies.keys())}")

    keys = list_raw_keys()
    print(f"  Found {len(keys)} latest files\n")

    all_rows    = []
    all_updates = {}   # ticker → new_accns to mark done after write
    total_files = 0

    for key in sorted(keys):
        filename = key.split("/")[-1]
        try:
            ticker = filename.split("_")[1]
        except IndexError:
            continue

        print(f"Processing {ticker}...")
        try:
            tracker = load_sec_tracker(s3, RAW_BUCKET, ticker)
            print(f"  Tracker: {tracker['total_transformed']} "
                  f"accessions previously transformed")

            raw  = read_raw_file(key)
            rows, new_accns = transform_company(raw, tracker)
            all_rows.extend(rows)
            all_updates[ticker] = (tracker, new_accns)
            total_files += 1
        except Exception as e:
            print(f"  ERROR: {ticker} failed — {e}")
            continue

    total_written = write_processed(all_rows)

    # Update trackers only after successful write
    for ticker, (tracker, new_accns) in all_updates.items():
        if new_accns:
            tracker = mark_accessions_done(
                tracker, "transformed_accessions",
                "total_transformed", new_accns
            )
            tracker["last_etl"] = now_utc()
            save_sec_tracker(s3, RAW_BUCKET, ticker, tracker)

    print(f"\n{'─'*50}")
    print(f"Done. {total_files} companies processed, "
          f"{total_written} filing rows written.")


if __name__ == "__main__":
    main()