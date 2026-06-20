"""
Clear DynamoDB watermarks for SEC EDGAR companies.

Why this exists:
    ingest_sec.py's _resolve_start() takes the MINIMUM watermark across
    ALL companies to compute one shared start date for that run. Clearing
    only the newly-added companies' watermarks does nothing useful if the
    existing companies still have recent watermarks — the shared min()
    would resolve to whatever the EXISTING companies' watermarks say,
    not default_start_date, meaning newly-added companies would only get
    a tiny recent slice of filings instead of the intended full backfill
    range.

    Same root cause as the yfinance/FRED/WorldBank watermark bugs. All
    companies must be cleared together for a clean full backfill —
    including ones you're not otherwise touching, if you want everyone
    backfilled from the same default_start_date.

Usage:
    python clear_sec_watermarks.py
        Clears the companies hardcoded in DEFAULT_COMPANIES below.

    python clear_sec_watermarks.py --all-from-yaml
        Reads ingestion/configs/sources/sec.yaml and clears every company
        ticker listed there. Use this one — it can't drift out of sync
        with the live config the way a hardcoded list can.

    python clear_sec_watermarks.py --dry-run
        Shows what would be cleared without touching DynamoDB.

    python clear_sec_watermarks.py --all-from-yaml --env prod
        Same, against the prod watermarks table. Use only when actually
        ready to backfill prod.

Note: this clears the DynamoDB watermark only. SEC also maintains a
SEPARATE per-company S3 tracker (sec_{ticker}_tracker.json in the raw
bucket, tracking fetched_accessions / total_fetched for incremental
filing-level dedup — see utils.watermark.load_sec_tracker /
save_sec_tracker). Clearing the DynamoDB watermark does NOT clear that
tracker. If you want a genuinely fresh full backfill (re-fetching
filings already marked as fetched), you also need to clear each
company's S3 tracker file — see clear_sec_trackers() below, run
separately and explicitly, since this is more aggressive (discards
incremental-fetch state, not just the date-resolution watermark).
"""
import argparse
import os
import sys

import boto3
import yaml

SOURCE = "sec"
REGION = "us-east-2"

# Fallback list — kept in sync with sec.yaml as of 2026-06-19 (13 companies).
# Prefer --all-from-yaml so this never silently drifts from the real config.
DEFAULT_COMPANIES = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "BAC", "XOM",
    "JNJ", "WMT", "CAT", "PG", "KO", "DIS",
]


def _table_name(env: str) -> str:
    return f"trade-platform-{env}-watermarks"


def _raw_bucket(env: str) -> str:
    account = os.environ.get("ACCOUNT", "197411402303")
    return f"{env}-trade-{SOURCE}-raw-{account}"


def load_companies_from_yaml(yaml_path: str) -> list:
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    companies = config.get("companies", {})
    if not companies:
        raise ValueError(f"No 'companies' key found in {yaml_path}")
    return list(companies.keys())


def clear_watermarks(tickers: list, env: str, dry_run: bool = False):
    table_name = _table_name(env)
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(table_name)

    print(f"Table:     {table_name}")
    print(f"Source:    {SOURCE}")
    print(f"Companies: {len(tickers)}")
    for t in tickers:
        print(f"  - {t}")

    if dry_run:
        print("\n[DRY RUN] No watermarks were deleted.")
        return

    cleared = 0
    missing = 0
    for ticker in tickers:
        try:
            resp = table.delete_item(
                Key={"source_name": SOURCE, "dataset_name": ticker},
                ReturnValues="ALL_OLD",
            )
            if "Attributes" in resp:
                cleared += 1
                print(f"  Cleared: {ticker} "
                      f"(was last_ingested_period="
                      f"{resp['Attributes'].get('last_ingested_period', '?')})")
            else:
                missing += 1
                print(f"  No watermark existed for: {ticker}")
        except Exception as e:
            print(f"  ERROR clearing {ticker}: {e}")

    print(f"\nDone. Cleared {cleared}, already-absent {missing}, "
          f"total {len(tickers)}.")
    print("\nAll companies must be cleared together for a clean full "
          "backfill — partial clears will be overridden by the shared "
          "min() watermark across the remaining companies.")
    print("\nNOTE: this clears the DynamoDB watermark only. SEC also "
          "maintains a per-company S3 tracker (fetched_accessions / "
          "total_fetched) for incremental filing-level dedup. If you want "
          "a genuinely fresh re-fetch of filings already marked done, "
          "run with --clear-s3-trackers too — see module docstring.")


def clear_s3_trackers(tickers: list, env: str, dry_run: bool = False):
    """
    Clear each company's S3 tracker file (sec_{ticker}_tracker.json).
    More aggressive than clearing the DynamoDB watermark alone — this
    discards incremental fetched_accessions/total_fetched state, causing
    ingest_sec.py to treat every filing as new on the next run.
    """
    bucket = _raw_bucket(env)
    s3 = boto3.client("s3", region_name=REGION)

    print(f"\nClearing S3 trackers in s3://{bucket}/tracker/")
    cleared = 0
    for ticker in tickers:
        key = f"tracker/sec_{ticker}_tracker.json"
        if dry_run:
            print(f"  [DRY RUN] Would delete: {key}")
            continue
        try:
            s3.delete_object(Bucket=bucket, Key=key)
            print(f"  Deleted: {key}")
            cleared += 1
        except Exception as e:
            print(f"  ERROR deleting {key}: {e}")

    if not dry_run:
        print(f"\nDone. Cleared {cleared} S3 tracker file(s).")


def main():
    p = argparse.ArgumentParser(
        description="Clear SEC watermarks (all companies must be cleared "
                     "together — see module docstring)."
    )
    p.add_argument("--all-from-yaml", action="store_true",
                    help="Read company tickers from sec.yaml instead of "
                         "the hardcoded DEFAULT_COMPANIES list.")
    p.add_argument("--yaml-path",
                    default=os.path.join(
                        "ingestion", "configs", "sources", "sec.yaml"),
                    help="Path to sec.yaml (used with --all-from-yaml).")
    p.add_argument("--env", default="dev", help="Environment (dev/prod).")
    p.add_argument("--dry-run", action="store_true",
                    help="Show what would be cleared without deleting.")
    p.add_argument("--clear-s3-trackers", action="store_true",
                    help="ALSO clear each company's S3 tracker file "
                         "(fetched_accessions state) — more aggressive, "
                         "causes a true re-fetch of every filing, not "
                         "just a date-resolution reset. Off by default.")
    args = p.parse_known_args()[0]

    if args.env == "prod":
        print("WARNING: --env prod selected. Confirm prod readiness "
              "checklist for SEC before proceeding.\n")

    if args.all_from_yaml:
        if not os.path.exists(args.yaml_path):
            print(f"ERROR: {args.yaml_path} not found. "
                  f"Run from the trade-platform repo root, or pass "
                  f"--yaml-path explicitly.")
            sys.exit(1)
        tickers = load_companies_from_yaml(args.yaml_path)
    else:
        tickers = DEFAULT_COMPANIES

    clear_watermarks(tickers, args.env, dry_run=args.dry_run)

    if args.clear_s3_trackers:
        clear_s3_trackers(tickers, args.env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()