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

import os
import sys
import json
import boto3
import yaml
import pandas as pd
from datetime import date
from io import BytesIO

# ── Path setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/tmp/ingestion")

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
    os.path.join(os.path.dirname(__file__), "..", "configs", "sources", "sec.yaml"))

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
    """List all per-ticker raw files under year= prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=RAW_BUCKET):
        keys.extend([o["Key"] for o in page.get("Contents", [])])
    keys = [k for k in keys if k.startswith("year=")]
    return keys


def read_raw_file(key: str) -> dict:
    obj = s3.get_object(Bucket=RAW_BUCKET, Key=key)
    return json.loads(obj["Body"].read())


def write_processed(rows: list) -> int:
    """Write canonical rows partitioned by source= / year= / form_type="""
    if not rows:
        print("  No rows to write")
        return 0

    df         = pd.DataFrame(rows)
    df["year"] = df["year"].astype(int)
    total      = 0

    for (year, doc_type), group in df.groupby(["year", "doc_type"]):
        safe_type = doc_type.replace("-", "").replace("/", "")
        key = (f"documents/source=EDGAR/"
               f"year={year}/"
               f"form_type={safe_type}/"
               f"data.parquet")

        buf = BytesIO()
        group.drop(columns=["year", "source", "doc_type"]).to_parquet(
            buf, index=False, engine="pyarrow", compression="snappy"
        )
        buf.seek(0)
        s3.put_object(Bucket=PROC_BUCKET, Key=key, Body=buf.getvalue())
        total += len(group)
        print(f"  Wrote {len(group)} rows → s3://{PROC_BUCKET}/{key}")

    return total


# ── Facts extraction ───────────────────────────────────────────────────────

def get_concept_value(facts_usgaap: dict, concept: str,
                      accession: str, form: str) -> float | None:
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

def transform_company(raw: dict) -> list:
    """
    Transform one company's EDGAR data into canonical document rows.
    One row per 10-K or 10-Q filing.
    """
    ingested_at  = now_utc()
    ticker       = raw.get("ticker", "").upper()
    cik          = raw.get("cik", "")
    submissions  = raw.get("submissions", {})
    company_name = submissions.get("name", ticker)
    fiscal_ye    = submissions.get("fiscalYearEnd", "")
    sic_code     = submissions.get("sic", "")
    exchange     = (submissions.get("exchanges") or [""])[0]

    # Get us-gaap facts
    facts_usgaap = (raw.get("facts", {})
                       .get("facts", {})
                       .get("us-gaap", {}))

    # Get filing index
    recent     = submissions.get("filings", {}).get("recent", {})
    accessions = recent.get("accessionNumber", [])
    forms      = recent.get("form", [])
    dates      = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])

    rows    = []
    skipped = 0

    for accn, form, filed, reported in zip(
            accessions, forms, dates, report_dates):

        if form not in TARGET_FORMS:
            continue

        try:
            filed_date = date.fromisoformat(filed)
        except (ValueError, TypeError):
            continue

        # Extract key financial metrics for this filing
        metrics = {}
        for concept, label, unit, divisor, scale_label in KEY_CONCEPTS:
            val = get_concept_value(facts_usgaap, concept, accn, form)
            # Only add first match per label (handles concept aliases)
            if val is not None and label not in metrics:
                metrics[label] = (label, val / divisor, scale_label)

        # Get fiscal period info from any matched value
        fp  = "FY" if form == "10-K" else reported
        fy_val = filed_date.year

        # Build narrative text
        text = build_financial_narrative(
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

    print(f"  {ticker}: {len(rows)} filings extracted "
          f"({skipped} skipped)")
    return rows


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    print(f"ETL SEC EDGAR → documents | env={ENV}")
    print(f"  raw:       s3://{RAW_BUCKET}")
    print(f"  processed: s3://{PROC_BUCKET}")

    config  = load_config()
    companies = config.get("companies", {})
    print(f"  Companies: {list(companies.keys())}")
    keys = list_raw_keys()
    print(f"  Found {len(keys)} raw files\n")

    all_rows    = []
    total_files = 0

    for key in sorted(keys):
        # Extract ticker from filename: sec_AAPL_20260531T222040Z.json
        filename = key.split("/")[-1]
        try:
            ticker = filename.split("_")[1]
        except IndexError:
            continue

        print(f"Processing {ticker}...")
        try:
            raw  = read_raw_file(key)
            rows = transform_company(raw)
            all_rows.extend(rows)
            total_files += 1
        except Exception as e:
            print(f"  ERROR: {ticker} failed — {e}")
            continue

    total_written = write_processed(all_rows)

    print(f"\n{'─'*50}")
    print(f"Done. {total_files} companies processed, "
          f"{total_written} filing rows written.")


if __name__ == "__main__":
    main()