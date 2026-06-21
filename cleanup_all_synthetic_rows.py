"""
One-time cleanup: remove all leftover synthetic test rows injected during
the etl_insiders.py / etl_news.py write_partition() fix verification
(2026-06-19/20 session). Filters out rows whose ID field matches the
known synthetic markers, rewrites the partition file without them, or
deletes the file entirely if removing synthetic rows empties it.

Safe to run multiple times — partitions with no synthetic rows present
are left untouched (reported as "no synthetic rows found", no write).
"""
import boto3
import pyarrow.parquet as pq
import pyarrow as pa
import io

s3 = boto3.client("s3", region_name="us-east-2")

SYNTHETIC_MARKERS = ["SYNTHETIC-TEST", "UNIT-TEST"]

def _is_synthetic(value) -> bool:
    s = str(value)
    return any(marker in s for marker in SYNTHETIC_MARKERS)

def clean_partition(bucket: str, key: str, id_column: str, label: str):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except s3.exceptions.NoSuchKey:
        print(f"  {label}: file not found at {key} — skipping")
        return
    except Exception as e:
        print(f"  {label}: ERROR reading {key}: {e}")
        return

    df = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
    before = len(df)

    if id_column not in df.columns:
        print(f"  {label}: column '{id_column}' not found, columns are "
              f"{list(df.columns)} — skipping")
        return

    mask = df[id_column].apply(_is_synthetic)
    synthetic_count = mask.sum()

    if synthetic_count == 0:
        print(f"  {label}: {before} rows, no synthetic rows found — untouched")
        return

    df_clean = df[~mask]
    after = len(df_clean)
    print(f"  {label}: {before} -> {after} rows (removed {synthetic_count} synthetic)")

    if after == 0:
        s3.delete_object(Bucket=bucket, Key=key)
        print(f"    Deleted {key} entirely (was 100% synthetic)")
    else:
        table = pa.Table.from_pandas(df_clean, preserve_index=False)
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        buf.seek(0)
        s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
        print(f"    Rewrote {key} without synthetic rows")


def main():
    insiders_bucket = "dev-trade-insiders-processed-197411402303"
    news_bucket     = "dev-trade-news-processed-197411402303"

    print("=== Insiders ===")
    clean_partition(insiders_bucket,
                     "insider_trades/year=2019/ticker=JPM/data.parquet",
                     "filing_id", "JPM/2019")
    clean_partition(insiders_bucket,
                     "insider_trades/year=2016/ticker=BAC/data.parquet",
                     "filing_id", "BAC/2016")
    clean_partition(insiders_bucket,
                     "insider_trades/year=1999/ticker=DIS/data.parquet",
                     "filing_id", "DIS/1999")

    print("\n=== News ===")
    clean_partition(news_bucket,
                     "news/year=2026/primary_ticker=BAC/data.parquet",
                     "article_id", "BAC/2026")
    clean_partition(news_bucket,
                     "news/year=1999/primary_ticker=USO/data.parquet",
                     "article_id", "USO/1999")

    print("\nDone. Re-run this script anytime to confirm a clean state "
          "(it's idempotent — partitions with no synthetic rows are "
          "left untouched and reported as such).")


if __name__ == "__main__":
    main()