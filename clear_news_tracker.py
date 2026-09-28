"""
Clear the News (Polygon) ingestion tracker.

Why this is different from clear_fred_watermarks.py / clear_worldbank_watermarks.py:
    ingest_news.py does NOT use DynamoDB watermarks at all. There is no
    _resolve_start()-style shared-min function, and no per-ticker date
    state to clear. The date range fetched each run is always just
    "last BACKFILL_DAYS days from now" (from news.yaml), recalculated
    fresh every time — there's nothing to "clear" about that.

    What ingest_news.py DOES track is a single flat S3 tracker file,
    tracker/ingested_ids.json, holding a set of "{article_id}:{ticker}"
    dedup keys across ALL tickers combined (load_ingested_ids() /
    save_ingested_ids()). This is what actually controls "have we already
    fetched this specific article" — clearing THIS file is the News
    equivalent of clearing a watermark: it forces every article currently
    inside the BACKFILL_DAYS window to be treated as new and re-fetched
    on the next run, regardless of whether it was fetched before.

    Note this only affects de-dup at the ingestion (raw) layer. The
    processed layer has its own separate correctness path — see the
    2026-06-19/20 session notes on the etl_news.py read_existing_article_ids()
    fix, which controls whether the ETL step treats already-processed
    articles as new. Clearing this tracker does NOT touch that — it only
    forces ingest_news.py to re-fetch from Polygon, not etl_news.py to
    reprocess. The two are independent.

Usage:
    python clear_news_tracker.py --dry-run
        Shows the current tracker contents without deleting anything.

    python clear_news_tracker.py
        Deletes tracker/ingested_ids.json from the raw bucket.

    python clear_news_tracker.py --env prod
        Same, against the prod news raw bucket.
"""
import argparse
import os

import boto3

SOURCE = "news"
REGION = "us-east-2"


def _raw_bucket(env: str) -> str:
    account = os.environ.get("ACCOUNT", "<ACCOUNT_ID>")
    return f"{env}-trade-{SOURCE}-raw-{account}"


def main():
    p = argparse.ArgumentParser(
        description="Clear the News (Polygon) ingestion tracker — "
                     "S3-tracker pattern, not DynamoDB watermarks. "
                     "See module docstring for why this differs from "
                     "clear_fred_watermarks.py / clear_worldbank_watermarks.py."
    )
    p.add_argument("--env", default="dev", help="Environment (dev/prod).")
    p.add_argument("--dry-run", action="store_true",
                    help="Show current tracker contents without deleting.")
    args = p.parse_args()

    if args.env == "prod":
        print("WARNING: --env prod selected. Confirm this is intentional "
              "before proceeding.\n")

    bucket = _raw_bucket(args.env)
    key = "tracker/ingested_ids.json"
    s3 = boto3.client("s3", region_name=REGION)

    print(f"Bucket: {bucket}")
    print(f"Key:    {key}")

    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        import json
        ids = json.loads(obj["Body"].read())
        print(f"Tracker currently has {len(ids)} ingested article:ticker keys.")
    except s3.exceptions.NoSuchKey:
        print("No tracker file currently exists — nothing to clear.")
        return
    except Exception as e:
        print(f"ERROR reading tracker: {e}")
        return

    if args.dry_run:
        print("\n[DRY RUN] Tracker was NOT deleted.")
        return

    s3.delete_object(Bucket=bucket, Key=key)
    print(f"\nDeleted {key}.")
    print("Next ingest_news.py run will treat every article inside the "
          "BACKFILL_DAYS window as new and re-fetch it from Polygon, "
          "regardless of whether it was already fetched before.")
    print("Reminder: this does NOT affect etl_news.py's separate "
          "already-processed check at the ETL layer — that's controlled "
          "by read_existing_article_ids() reading the processed Parquet "
          "files directly, independent of this tracker.")


if __name__ == "__main__":
    main()