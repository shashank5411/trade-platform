"""
ETL: Wikipedia raw JSON → documents canonical Parquet.

Raw format: bulk file, list of article summary extracts
[{"title": "Inflation", "description": "...", "extract": "...",
  "last_modified": "2026-05-24T17:58:07Z", "url": "..."}]

Notes:
- extract is already plain text — no Wikitext stripping needed
- one row per article snapshot
- full text stored as-is for Phase 5 chunking
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

import os
import sys
import json
import boto3
import yaml
import pandas as pd
from datetime import date
from io import BytesIO

# ── Path setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, _libs_dir)
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

RAW_BUCKET  = _bucket("wikipedia", "raw")
PROC_BUCKET = _bucket("wikipedia", "processed")
CONFIG_PATH = _arg("config_path",
    os.path.join(_libs_dir, "configs", "sources", "wikipedia.yaml"))

from utils.transform import (
    now_utc,
    make_doc_id,
    clean_text,
    to_json_str,
    validate_document_row,
)

s3 = boto3.client("s3", region_name="us-east-2")


# ── Config ─────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── S3 helpers ─────────────────────────────────────────────────────────────

def read_latest_raw() -> list:
    """Read the latest bulk Wikipedia raw file from S3."""
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=RAW_BUCKET):
        keys.extend([o["Key"] for o in page.get("Contents", [])])

    keys = [k for k in keys if k.startswith("year=")]
    if not keys:
        raise FileNotFoundError(
            f"No raw files found under year= prefix in s3://{RAW_BUCKET}"
        )

    latest_key = sorted(keys)[-1]
    print(f"  Reading s3://{RAW_BUCKET}/{latest_key}")

    obj     = s3.get_object(Bucket=RAW_BUCKET, Key=latest_key)
    records = json.loads(obj["Body"].read())

    print(f"  Loaded {len(records)} articles")
    return records


def write_processed(rows: list) -> int:
    """Write canonical rows partitioned by source= / year="""
    if not rows:
        print("  No rows to write")
        return 0

    df         = pd.DataFrame(rows)
    df["year"] = df["year"].astype(int)
    total      = 0

    for year, group in df.groupby("year"):
        key = (f"documents/source=WIKIPEDIA/"
               f"year={year}/"
               f"data.parquet")

        new_df = group.drop(columns=["year", "source"])

        # Read existing partition and merge — protects against article
        # fetch failures silently dropping rows on incremental runs.
        try:
            obj = s3.get_object(Bucket=PROC_BUCKET, Key=key)
            existing_df = pd.read_parquet(BytesIO(obj["Body"].read()))
            write_df = pd.concat([existing_df, new_df], ignore_index=True)
            write_df = write_df.drop_duplicates(subset=["doc_id"], keep="last")
            print(f"  Merging: {len(existing_df)} existing + {len(new_df)} new → "
                  f"{len(write_df)} rows after dedup")
        except s3.exceptions.NoSuchKey:
            write_df = new_df
        except Exception as e:
            print(f"  ERROR: could not read existing partition "
                  f"s3://{PROC_BUCKET}/{key}: {e}")
            print(f"  SKIPPING year={year} — refusing blind overwrite "
                  f"to protect existing data")
            continue

        buf = BytesIO()
        write_df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
        buf.seek(0)
        s3.put_object(Bucket=PROC_BUCKET, Key=key, Body=buf.getvalue())
        total += len(group)
        print(f"  Wrote {len(write_df)} rows → s3://{PROC_BUCKET}/{key}")

    return total


# ── Transform ──────────────────────────────────────────────────────────────

def transform(records: list) -> list:
    """
    Transform raw Wikipedia extract records into canonical document rows.
    extract field is already plain text — no markup stripping needed.
    """
    ingested_at = now_utc()
    today       = date.today()
    rows        = []
    skipped     = 0

    for rec in records:
        title       = rec.get("title", "")
        extract     = rec.get("extract", "")
        description = rec.get("description", "")
        last_mod    = rec.get("last_modified", "")
        url         = rec.get("url", "")

        if not title or not extract:
            print(f"  WARN: skipping record with no title or extract")
            skipped += 1
            continue

        # Parse snapshot date from last_modified
        try:
            snap_date = date.fromisoformat(last_mod[:10])
        except (ValueError, TypeError):
            snap_date = today

        # Combine description + extract for richer RAG context
        full_text = clean_text(
            f"{description}\n\n{extract}" if description else extract
        )

        doc_id = make_doc_id("WIKIPEDIA", title, str(snap_date))

        row = {
            "doc_id":      doc_id,
            "source":      "WIKIPEDIA",
            "year":        snap_date.year,
            "title":       title,
            "entity":      title.lower().replace(" ", "_"),
            "doc_type":    "wiki_article",
            "doc_date":    str(snap_date),
            "text":        full_text,
            "char_count":  len(full_text),
            "metadata":    to_json_str({
                "url":          url,
                "description":  description,
                "last_modified": last_mod,
            }),
            "ingested_at": ingested_at,
        }

        errors = validate_document_row(row)
        if errors:
            print(f"  WARN: skipping '{title}': {errors}")
            skipped += 1
            continue

        rows.append(row)

    if skipped:
        print(f"  Skipped {skipped} records")

    return rows


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    print(f"ETL Wikipedia → documents | env={ENV}")
    print(f"  raw:       s3://{RAW_BUCKET}")
    print(f"  processed: s3://{PROC_BUCKET}")

    config  = load_config()
    print(f"  Topics: {config.get('topics', [])}")

    records       = read_latest_raw()
    rows          = transform(records)
    total_written = write_processed(rows)

    print(f"\n{'─'*50}")
    print(f"Done. {total_written} rows written.")


if __name__ == "__main__":
    main()