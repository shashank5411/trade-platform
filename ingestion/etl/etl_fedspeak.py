import sys
import os
import zipfile

# ── Fix A: Glue zip extraction ────────────────────────────────────────────────
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

import re
import gzip
import json
import hashlib
import argparse
import datetime

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import io

# ── Fix B: _arg function ──────────────────────────────────────────────────────
def _arg(name: str, default: str = "") -> str:
    for i, a in enumerate(sys.argv):
        if a == f"--{name}" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.getenv(name, default)

ENV        = _arg("ENVIRONMENT", "dev")
ACCOUNT    = boto3.client("sts", region_name="us-east-2").get_caller_identity()["Account"]
RAW_BUCKET  = f"{ENV}-trade-fedspeak-raw-{ACCOUNT}"
PROC_BUCKET = f"{ENV}-trade-fedspeak-processed-{ACCOUNT}"
REGION     = "us-east-2"

# ── Fix D: S3 client with region ──────────────────────────────────────────────
s3 = boto3.client("s3", region_name=REGION)

# ── Fix C: parse_known_args ───────────────────────────────────────────────────
def _arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ENVIRONMENT", default="dev")
    return p.parse_known_args()[0]

_arg_parser()

# ── Canonical schema ──────────────────────────────────────────────────────────
SCHEMA = pa.schema([
    pa.field("doc_id",      pa.string()),
    pa.field("entity",      pa.string()),   # FOMC | Powell | Waller | ...
    pa.field("doc_date",    pa.string()),   # YYYY-MM-DD
    pa.field("title",       pa.string()),
    pa.field("text",        pa.string()),
    pa.field("char_count",  pa.int32()),
    pa.field("url",         pa.string()),
    pa.field("ingested_at", pa.string()),
])

# ── Read all raw docs from S3 ─────────────────────────────────────────────────
def read_raw_docs() -> list:
    """Read all .json.gz files from raw bucket, return list of dicts."""
    docs = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=RAW_BUCKET):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json.gz"):
                continue
            try:
                body = s3.get_object(Bucket=RAW_BUCKET, Key=key)["Body"].read()
                doc  = json.loads(gzip.decompress(body).decode("utf-8"))
                docs.append(doc)
            except Exception as e:
                print(f"WARN: Could not read {key}: {e}")
    return docs

# ── Read existing processed doc_ids ──────────────────────────────────────────
def read_existing_doc_ids() -> set:
    """Collect doc_ids already written to processed bucket to dedup."""
    existing = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=PROC_BUCKET, Prefix="documents/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            try:
                body = s3.get_object(Bucket=PROC_BUCKET, Key=key)["Body"].read()
                df   = pq.read_table(io.BytesIO(body)).to_pandas()
                existing.update(df["doc_id"].tolist())
            except Exception as e:
                print(f"WARN: Could not read existing parquet {key}: {e}")
    return existing

# ── Clean text ────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Basic text normalization."""
    if not text:
        return ""
    # Collapse excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

# ── Write Parquet partition ───────────────────────────────────────────────────
def write_partition(rows: list, year: int, doc_type: str) -> None:
    """Write a list of row dicts to a Parquet file in the processed bucket."""
    if not rows:
        return

    key = f"documents/source=FEDSPEAK/year={year}/doc_type={doc_type}/data.parquet"

    # Read existing partition and merge — prevents incremental runs from
    # discarding previously written rows. Skip (never partial-overwrite)
    # on any read error other than NoSuchKey.
    existing_rows = []
    try:
        obj = s3.get_object(Bucket=PROC_BUCKET, Key=key)
        existing_df = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
        existing_rows = existing_df.to_dict(orient="records")
        print(f"  Merging with {len(existing_rows)} existing rows at {key}")
    except s3.exceptions.NoSuchKey:
        pass
    except Exception as e:
        print(f"  ERROR: could not read existing partition "
              f"s3://{PROC_BUCKET}/{key}: {e}")
        print(f"  SKIPPING year={year} doc_type={doc_type} — "
              f"refusing blind overwrite to protect existing data")
        return

    combined = existing_rows + rows
    df = pd.DataFrame(combined)
    df = df.drop_duplicates(subset=["doc_id"], keep="last")

    # Ensure all schema fields exist
    for field in SCHEMA:
        if field.name not in df.columns:
            df[field.name] = None

    # Cast types
    df["char_count"] = df["char_count"].fillna(0).astype("int32")

    table = pa.Table.from_pandas(df[
        [f.name for f in SCHEMA]
    ], schema=SCHEMA, preserve_index=False)

    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)

    s3.put_object(Bucket=PROC_BUCKET, Key=key, Body=buf.read())
    print(f"  Written: s3://{PROC_BUCKET}/{key} "
          f"({len(existing_rows)} existing + {len(rows)} new → "
          f"{len(df)} rows after dedup)")

# ── Main ETL ──────────────────────────────────────────────────────────────────
def main():
    print(f"[etl_fedspeak] env={ENV}, raw={RAW_BUCKET}, processed={PROC_BUCKET}")

    print("Reading raw documents...")
    docs = read_raw_docs()
    print(f"  Found {len(docs)} raw documents")

    if not docs:
        print("No raw documents found — exiting")
        return

    print("Reading existing processed doc_ids...")
    existing_ids = read_existing_doc_ids()
    print(f"  Found {len(existing_ids)} already processed")

    # Group new docs by (year, doc_type)
    partitions: dict = {}  # (year, doc_type) -> [rows]

    for doc in docs:
        if doc.get("doc_id") in existing_ids:
            continue

        # Parse year from doc_date
        doc_date = doc.get("doc_date", "")
        try:
            year = int(doc_date[:4])
        except (ValueError, TypeError):
            print(f"WARN: Invalid doc_date '{doc_date}', skipping")
            continue

        text = clean_text(doc.get("text", ""))
        if not text or len(text) < 50:
            print(f"WARN: Skipping short/empty doc {doc.get('doc_id', '')[:16]}")
            continue

        row = {
            "doc_id":      doc.get("doc_id", ""),
            "entity":      doc.get("entity", "FOMC"),
            "doc_date":    doc_date,
            "title":       doc.get("title", ""),
            "text":        text,
            "char_count":  len(text),
            "url":         doc.get("url", ""),
            "ingested_at": doc.get("ingested_at", ""),
        }

        key = (year, doc.get("doc_type", "unknown"))
        partitions.setdefault(key, []).append(row)

    if not partitions:
        print("No new documents to process")
        return

    # Write each partition
    total = 0
    for (year, doc_type), rows in sorted(partitions.items()):
        print(f"\n  Processing year={year} doc_type={doc_type} ({len(rows)} docs)")
        write_partition(rows, year, doc_type)
        total += len(rows)

    print(f"\n[etl_fedspeak] Done — {total} new rows written")

if __name__ == "__main__":
    main()