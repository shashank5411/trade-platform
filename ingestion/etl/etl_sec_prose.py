"""
ETL: SEC EDGAR prose sections → documents_prose canonical Parquet.

Reads pre-extracted prose sections written by ingest_sec.py (via edgartools).
No LLM calls, no HTML parsing — edgartools already handled extraction cleanly.

Input:  s3://{env}-trade-sec-raw-{account}/year=/month=/prose/*.json.gz
        Each file: {ticker, accession, form, filed, sections: {section_name: text}}

Output: s3://{env}-trade-sec-prose-processed-{account}/documents_prose/
        Parquet partitioned by year= / entity= / form_type=

Output schema (documents_prose table):
  doc_id            STRING    SHA-256 of (source+accession+section)
  parent_doc_id     STRING    links to documents table XBRL row
  source            STRING    EDGAR
  entity            STRING    ticker
  accession         STRING    EDGAR accession number
  form_type         STRING    10K | 10Q (dashes stripped for Parquet compat)
  filed_date        STRING    filing date
  section_name      STRING    item_1 | item_1a | item_7 | item_7a
  section_title     STRING    human readable e.g. "Risk Factors"
  text              STRING    full prose text of section
  char_count        INTEGER
  year              INTEGER   partition key
  extraction_method STRING    always "edgartools"
  ingested_at       STRING
"""

import gzip
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from io import BytesIO

import boto3
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/tmp/ingestion")

from utils.watermark import (
    get_new_accessions,
    load_sec_tracker,
    mark_accessions_done,
    save_sec_tracker,
)


def _arg(key, default=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--{key}",
                        default=os.environ.get(key.upper(), default))
    args, _ = parser.parse_known_args()
    return getattr(args, key)


ENV     = _arg("env",     "dev")
ACCOUNT = _arg("account", "197411402303")
REGION  = "us-east-2"

RAW_BUCKET   = f"{ENV}-trade-sec-raw-{ACCOUNT}"
PROSE_BUCKET = f"{ENV}-trade-sec-prose-processed-{ACCOUNT}"

MAX_SECTION_CHARS = 50_000

# Human-readable titles per section identifier
SECTION_TITLES = {
    "item_1":  "Business",
    "item_1a": "Risk Factors",
    "item_7":  "Management's Discussion and Analysis",
    "item_7a": "Quantitative and Qualitative Disclosures About Market Risk",
}

s3 = boto3.client("s3", region_name=REGION)


# ── helpers ────────────────────────────────────────────────────────────────

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_doc_id(source: str, accession: str, section: str) -> str:
    raw = f"{source}_{accession}_{section}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── S3 I/O ─────────────────────────────────────────────────────────────────

def list_prose_keys() -> list:
    """List all prose JSON files written by ingest_sec.py."""
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=RAW_BUCKET):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if "/prose/" in key and (
                key.endswith(".json") or key.endswith(".json.gz")
            ):
                keys.append(key)
    return keys


def read_prose_doc(key: str) -> dict:
    obj = s3.get_object(Bucket=RAW_BUCKET, Key=key)
    raw = obj["Body"].read()
    if key.endswith(".gz"):
        raw = gzip.decompress(raw)
    return json.loads(raw)


def write_processed(rows: list) -> int:
    """Append new prose rows to existing Parquet, deduplicating on doc_id."""
    if not rows:
        print("  No rows to write")
        return 0

    df         = pd.DataFrame(rows)
    df["year"] = df["year"].astype(int)
    total      = 0

    for (year, entity, form_type), group in df.groupby(
            ["year", "entity", "form_type"]):
        key     = (f"documents_prose/year={year}/entity={entity}/"
                   f"form_type={form_type}/data.parquet")
        new_df  = group.drop(columns=["year", "entity", "form_type"])

        try:
            obj         = s3.get_object(Bucket=PROSE_BUCKET, Key=key)
            existing_df = pd.read_parquet(BytesIO(obj["Body"].read()))
            existing_ids = set(existing_df["doc_id"].tolist())
            new_df       = new_df[~new_df["doc_id"].isin(existing_ids)]
            if new_df.empty:
                print(f"  No new rows for {entity} year={year} "
                      f"{form_type} — skipping")
                continue
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"  Appending {len(new_df)} rows to "
                  f"{len(existing_df)} existing → "
                  f"{entity} year={year} {form_type}")
        except s3.exceptions.NoSuchKey:
            combined = new_df
            print(f"  Writing {len(combined)} rows → "
                  f"{entity} year={year} {form_type}")

        buf = BytesIO()
        combined.to_parquet(
            buf, index=False, engine="pyarrow", compression="snappy"
        )
        buf.seek(0)
        s3.put_object(Bucket=PROSE_BUCKET, Key=key, Body=buf.getvalue())
        total += len(new_df)

    return total


# ── transform ──────────────────────────────────────────────────────────────

def transform_prose_doc(doc: dict) -> list:
    """
    Map one prose doc (from ingest_sec.py) to documents_prose rows.
    sections dict is already clean text — no parsing needed.
    """
    ticker      = doc.get("ticker", "").upper()
    accession   = doc.get("accession", "")
    form_type   = doc.get("form", "10-K")
    filed_date  = doc.get("filed", "")
    sections    = doc.get("sections", {})   # {section_name: text}
    ingested_at = now_utc()

    if not sections:
        print(f"  WARN: {ticker}/{accession} — no sections, skipping")
        return []

    try:
        year = int(filed_date[:4])
    except (ValueError, TypeError):
        year = 2024

    rows = []
    for section_name, section_text in sections.items():
        if not section_text or len(str(section_text)) < 100:
            print(f"  WARN: {ticker}/{accession}/{section_name} "
                  f"too short, skipping")
            continue

        text = str(section_text)[:MAX_SECTION_CHARS]
        rows.append({
            "doc_id":            make_doc_id("EDGAR", accession, section_name),
            "parent_doc_id":     f"EDGAR_{accession}",
            "source":            "EDGAR",
            "entity":            ticker,
            "accession":         accession,
            "form_type":         form_type.replace("-", ""),
            "filed_date":        filed_date,
            "section_name":      section_name,
            "section_title":     SECTION_TITLES.get(section_name, section_name),
            "text":              text,
            "char_count":        len(text),
            "year":              year,
            "extraction_method": "edgartools",
            "ingested_at":       ingested_at,
        })

    print(f"  {ticker} {form_type} {filed_date} → {len(rows)} sections")
    return rows


# ── entry point ────────────────────────────────────────────────────────────

def main():
    print(f"ETL SEC Prose → documents_prose | env={ENV}")
    print(f"  raw:   s3://{RAW_BUCKET}")
    print(f"  prose: s3://{PROSE_BUCKET}\n")

    keys = list_prose_keys()
    print(f"Found {len(keys)} prose filing files\n")

    if not keys:
        print("No prose files found. Run ingest_sec.py first.")
        return

    # Group by ticker for tracker loading
    ticker_keys: dict[str, list] = {}
    for key in sorted(keys):
        filename = key.split("/")[-1]
        try:
            ticker = filename.split("_")[1]
        except IndexError:
            continue
        ticker_keys.setdefault(ticker, []).append(key)

    all_rows    = []
    all_updates = {}

    for ticker, t_keys in sorted(ticker_keys.items()):
        tracker = load_sec_tracker(s3, RAW_BUCKET, ticker)
        print(f"{ticker}: {tracker.get('total_prose', 0)} accessions "
              f"previously prose-extracted")

        # Extract accession numbers from filenames
        all_accns = []
        for key in t_keys:
            parts = key.split("/")[-1].split("_")
            if len(parts) >= 3:
                all_accns.append(parts[2])

        new_accns = set(get_new_accessions(
            tracker, "prose_accessions", all_accns
        ))
        processed_accns = []

        for key in t_keys:
            filename  = key.split("/")[-1]
            parts     = filename.split("_")
            accession = parts[2] if len(parts) >= 3 else ""

            if accession not in new_accns:
                print(f"  Skipping {accession} — already prose-extracted")
                continue

            try:
                doc  = read_prose_doc(key)
                rows = transform_prose_doc(doc)
                all_rows.extend(rows)
                processed_accns.append(accession)
            except Exception as e:
                print(f"  ERROR: {key} — {e}")

        if processed_accns:
            all_updates[ticker] = (tracker, processed_accns)

    total_written = write_processed(all_rows)

    # Update trackers only after successful write
    for ticker, (tracker, new_accns) in all_updates.items():
        tracker = mark_accessions_done(
            tracker, "prose_accessions", "total_prose", new_accns
        )
        tracker["last_prose_etl"] = now_utc()
        save_sec_tracker(s3, RAW_BUCKET, ticker, tracker)

    print(f"\n{'─'*50}")
    print(f"Done.")
    print(f"  Sections extracted: {len(all_rows)}")
    print(f"  Rows written:       {total_written}")


if __name__ == "__main__":
    main()