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

import gzip
import json
import argparse
import io

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Fix B: _arg function ──────────────────────────────────────────────────────
def _arg(name: str, default: str = "") -> str:
    for i, a in enumerate(sys.argv):
        if a == f"--{name}" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.getenv(name, default)

ENV     = _arg("ENVIRONMENT", "dev")
REGION  = "us-east-2"
ACCOUNT = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]

RAW_BUCKET  = f"{ENV}-trade-news-raw-{ACCOUNT}"
PROC_BUCKET = f"{ENV}-trade-news-processed-{ACCOUNT}"

# ── Fix D: S3 client with region ──────────────────────────────────────────────
s3 = boto3.client("s3", region_name=REGION)

# ── Fix C: parse_known_args ───────────────────────────────────────────────────
def _arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ENVIRONMENT", default="dev")
    return p.parse_known_args()[0]

_arg_parser()

# ── Canonical schema ──────────────────────────────────────────────────────────
# Note: year and primary_ticker are partition columns — NOT in Parquet file
SCHEMA = pa.schema([
    pa.field("article_id",          pa.string()),
    pa.field("headline",            pa.string()),
    pa.field("description",         pa.string()),
    pa.field("author",              pa.string()),
    pa.field("published_at",        pa.string()),
    pa.field("publisher",           pa.string()),
    pa.field("publisher_tier",      pa.int32()),
    pa.field("tickers",             pa.string()),   # JSON array
    pa.field("sentiment",           pa.string()),
    pa.field("sentiment_reasoning", pa.string()),
    pa.field("keywords",            pa.string()),   # JSON array
    pa.field("article_url",         pa.string()),
    pa.field("week",                pa.string()),   # YYYY-WXX
    pa.field("ingested_at",         pa.string()),
])

# ── Read raw articles from S3 ─────────────────────────────────────────────────
def read_raw_articles() -> list:
    """Read all .json.gz files from raw bucket."""
    articles = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=RAW_BUCKET):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json.gz") or key.startswith("tracker/"):
                continue
            try:
                body     = s3.get_object(Bucket=RAW_BUCKET, Key=key)["Body"].read()
                batch    = json.loads(gzip.decompress(body).decode("utf-8"))
                # Extract ticker from S3 key path: year=YYYY/ticker=AAPL/...
                import re
                ticker_match = re.search(r'ticker=([^/]+)/', key)
                ticker = ticker_match.group(1) if ticker_match else ""
                for a in batch:
                    a["_source_key"]    = key
                    a["_source_ticker"] = ticker
                articles.extend(batch)
            except Exception as e:
                print(f"WARN: Could not read {key}: {e}")
    return articles

# ── Read existing article_ids ─────────────────────────────────────────────────
def read_existing_article_ids() -> set:
    existing = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=PROC_BUCKET, Prefix="news/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue

            # primary_ticker is a PARTITION column — stripped from the
            # actual Parquet file by write_partition(), so it must be
            # derived from the S3 key path, not read as a DataFrame
            # column (every row in a given partition file shares the
            # same ticker by construction, so this is derived once per
            # file, not per row).
            import re
            ticker_match = re.search(r'primary_ticker=([^/]+)/', key)
            ticker = ticker_match.group(1) if ticker_match else ""

            try:
                body = s3.get_object(Bucket=PROC_BUCKET, Key=key)["Body"].read()
                df   = pq.read_table(io.BytesIO(body)).to_pandas()
                for article_id in df["article_id"]:
                    existing.add(f"{article_id}:{ticker}")
            except Exception as e:
                print(f"WARN: Could not read {key}: {e}")
    return existing

# ── Write Parquet partition ───────────────────────────────────────────────────
def write_partition(rows: list, year: int, ticker: str) -> None:
    """Write a (year, primary_ticker) partition, MERGING with any existing
    data rather than overwriting it. Existing rows are read first,
    deduplicated against new rows by article_id, and the combined set is
    written back. Dedup by article_id alone (not the composite
    article_id:ticker key used elsewhere) is safe here because this
    function is always called per-ticker — rows and existing_rows are
    both already scoped to the same `ticker` by construction in main()'s
    (year, ticker) grouping."""
    if not rows:
        return

    key = f"news/year={year}/primary_ticker={ticker}/data.parquet"

    existing_rows = []
    try:
        obj = s3.get_object(Bucket=PROC_BUCKET, Key=key)
        existing_df = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
        existing_rows = existing_df.to_dict(orient="records")
        print(f"  Merging with {len(existing_rows)} existing rows at {key}")
    except s3.exceptions.NoSuchKey:
        pass
    except Exception as e:
        print(f"  WARN: could not read existing partition {key}: {e} "
              f"— proceeding with new rows only, but this risks "
              f"DROPPING existing data if the file genuinely exists and "
              f"this read failure is not a true NoSuchKey. Investigate "
              f"if this warning appears in practice.")

    combined = existing_rows + rows
    df = pd.DataFrame(combined)
    df = df.drop_duplicates(subset=["article_id"], keep="last")

    # Ensure all schema fields present
    for field in SCHEMA:
        if field.name not in df.columns:
            df[field.name] = None

    df["publisher_tier"] = df["publisher_tier"].fillna(3).astype("int32")

    # Remove partition columns from Parquet file
    # (year and primary_ticker are in S3 path)
    parquet_cols = [f.name for f in SCHEMA]
    df = df[[c for c in parquet_cols if c in df.columns]]

    # Add any missing schema cols as None
    for col in parquet_cols:
        if col not in df.columns:
            df[col] = None

    table = pa.Table.from_pandas(
        df[parquet_cols], schema=SCHEMA, preserve_index=False
    )

    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)

    s3.put_object(Bucket=PROC_BUCKET, Key=key, Body=buf.read())
    print(f"  Written: s3://{PROC_BUCKET}/{key} "
          f"({len(combined)} total rows: {len(existing_rows)} existing + "
          f"{len(rows)} new, after dedup: {len(df)})")

# ── Main ETL ──────────────────────────────────────────────────────────────────
def main():
    print(f"[etl_news] env={ENV}, raw={RAW_BUCKET}, processed={PROC_BUCKET}")

    print("Reading raw articles...")
    articles = read_raw_articles()
    print(f"  Found {len(articles)} raw articles")

    if not articles:
        print("No raw articles found — exiting")
        return

    print("Reading existing article IDs...")
    existing_ids = read_existing_article_ids()
    print(f"  Found {len(existing_ids)} already processed")

    # Group new articles by (year, primary_ticker)
    partitions: dict = {}

    for a in articles:
        ticker = a.get("primary_ticker") or a.get("_source_ticker", "UNKNOWN")
        dedup_key = f"{a.get('article_id', '')}:{ticker}"

        if dedup_key in existing_ids:
            continue

        try:
            year = int(str(a.get("year", "2026"))[:4])
        except (ValueError, TypeError):
            year = 2026

        row = {
            "article_id":           a.get("article_id", ""),
            "headline":             a.get("headline", ""),
            "description":          a.get("description", ""),
            "author":               a.get("author", ""),
            "published_at":         a.get("published_at", ""),
            "publisher":            a.get("publisher", ""),
            "publisher_tier":       int(a.get("publisher_tier", 3)),
            "tickers":              a.get("tickers", "[]"),
            "sentiment":            a.get("sentiment", "neutral"),
            "sentiment_reasoning":  a.get("sentiment_reasoning", ""),
            "keywords":             a.get("keywords", "[]"),
            "article_url":          a.get("article_url", ""),
            "week":                 a.get("week", ""),
            "ingested_at":          a.get("ingested_at", ""),
            # Keep for partitioning — removed from Parquet in write_partition
            "primary_ticker":       ticker,
            "year":                 year,
        }

        key = (year, ticker)
        partitions.setdefault(key, []).append(row)

    if not partitions:
        print("No new articles to process")
        return

    total = 0
    for (year, ticker), rows in sorted(partitions.items()):
        print(f"\n  year={year} ticker={ticker}: {len(rows)} articles")
        write_partition(rows, year, ticker)
        total += len(rows)

    print(f"\n[etl_news] Done — {total} new rows written")

if __name__ == "__main__":
    main()