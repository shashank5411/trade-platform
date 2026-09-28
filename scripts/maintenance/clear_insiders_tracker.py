"""
Clear the Insiders (SEC Form 4) ingestion tracker(s).

Why this is different from clear_fred_watermarks.py / clear_worldbank_watermarks.py:
    ingest_insiders.py does NOT use DynamoDB watermarks. There is no
    _resolve_start()-style shared-min function, and no per-ticker date
    state to clear in DynamoDB. The date range fetched is always
    START_DATE (from insiders.yaml's default_start_date, or a CLI
    override) through today, recalculated fresh every run.

Why this is different from clear_news_tracker.py:
    News keeps ONE combined tracker file (tracker/ingested_ids.json) with
    dedup keys for ALL tickers mixed together. Insiders keeps a SEPARATE
    tracker file PER TICKER (tracker/{ticker}.json — e.g. tracker/AAPL.json,
    tracker/JPM.json), each holding that ticker's own fetched_accessions
    list (load_tracker() / save_tracker() in ingest_insiders.py). This is
    the same per-entity tracker family as SEC's sec_{ticker}_tracker.json
    pattern, not News's single-combined-file pattern.

    Practical effect: you can clear ONE ticker's tracker without touching
    any other ticker's progress — unlike News, where there's only one
    file and no way to selectively clear just one ticker's dedup state.

What clearing does: forces get_new_accessions() to treat ALL of a
ticker's Form 4 filings (within the START_DATE window) as new again on
the next run, causing a full re-fetch of that ticker's insider-trade
history from EDGAR — not just whatever's new since the last run.

Usage:
    python clear_insiders_tracker.py --all-from-yaml --dry-run
        Shows what would be cleared for every ticker in insiders.yaml,
        without deleting anything.

    python clear_insiders_tracker.py --all-from-yaml
        Clears every ticker's tracker — full re-fetch for everyone on
        next run.

    python clear_insiders_tracker.py --ticker JPM
        Clears just JPM's tracker, leaving every other ticker's progress
        untouched.

    python clear_insiders_tracker.py --all-from-yaml --env prod
        Same, against the prod insiders raw bucket.
"""
import argparse
import json
import os
import sys

import boto3
import yaml

SOURCE = "insiders"
REGION = "us-east-2"


def _raw_bucket(env: str) -> str:
    account = os.environ.get("ACCOUNT", "<ACCOUNT_ID>")
    return f"{env}-trade-{SOURCE}-raw-{account}"


def load_tickers_from_yaml(yaml_path: str) -> list:
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    tickers = config.get("tickers", [])
    if not tickers:
        raise ValueError(f"No 'tickers' key found in {yaml_path}")
    return tickers


def clear_one_tracker(s3, bucket: str, ticker: str, dry_run: bool) -> None:
    key = f"tracker/{ticker}.json"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        tracker = json.loads(obj["Body"].read())
        fetched = tracker.get("fetched_accessions", [])
        print(f"  {ticker}: {len(fetched)} accessions currently tracked"
              + (" [DRY RUN — not deleted]" if dry_run else ""))
    except s3.exceptions.NoSuchKey:
        print(f"  {ticker}: no tracker exists — nothing to clear")
        return
    except Exception as e:
        print(f"  {ticker}: ERROR reading tracker: {e}")
        return

    if not dry_run:
        s3.delete_object(Bucket=bucket, Key=key)
        print(f"    Deleted {key}")


def main():
    p = argparse.ArgumentParser(
        description="Clear Insiders (SEC Form 4) ticker tracker(s) — "
                     "per-ticker S3 files, not DynamoDB watermarks. "
                     "See module docstring for how this differs from "
                     "both the FRED/WorldBank watermark scripts and "
                     "clear_news_tracker.py."
    )
    p.add_argument("--all-from-yaml", action="store_true",
                    help="Clear every ticker's tracker, read from "
                         "insiders.yaml's tickers list.")
    p.add_argument("--ticker", default=None,
                    help="Clear a single ticker's tracker only, leaving "
                         "all others untouched.")
    p.add_argument("--yaml-path",
                    default=os.path.join(
                        "ingestion", "configs", "sources", "insiders.yaml"),
                    help="Path to insiders.yaml (used with --all-from-yaml).")
    p.add_argument("--env", default="dev", help="Environment (dev/prod).")
    p.add_argument("--dry-run", action="store_true",
                    help="Show what would be cleared without deleting.")
    args = p.parse_args()

    if not args.all_from_yaml and not args.ticker:
        print("ERROR: specify either --all-from-yaml or --ticker TICKER")
        sys.exit(1)

    if args.env == "prod":
        print("WARNING: --env prod selected. Confirm this is intentional "
              "before proceeding.\n")

    bucket = _raw_bucket(args.env)
    s3 = boto3.client("s3", region_name=REGION)
    print(f"Bucket: {bucket}\n")

    if args.ticker:
        tickers = [args.ticker]
    else:
        if not os.path.exists(args.yaml_path):
            print(f"ERROR: {args.yaml_path} not found. Run from the "
                  f"trade-platform repo root, or pass --yaml-path explicitly.")
            sys.exit(1)
        tickers = load_tickers_from_yaml(args.yaml_path)

    print(f"Tickers ({len(tickers)}): {tickers}\n")

    for ticker in tickers:
        clear_one_tracker(s3, bucket, ticker, args.dry_run)

    if args.dry_run:
        print("\n[DRY RUN] No trackers were deleted.")
    else:
        print(f"\nDone. Next ingest_insiders.py run will treat the "
              f"cleared ticker(s)' entire Form 4 history (within "
              f"START_DATE) as new and re-fetch from EDGAR.")
        print("Reminder: unlike clear_news_tracker.py, this can target a "
              "SINGLE ticker without affecting others, since each ticker "
              "has its own separate tracker file.")


if __name__ == "__main__":
    main()