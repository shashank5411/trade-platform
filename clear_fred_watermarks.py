"""
Clear DynamoDB watermarks for FRED series.

Why this exists:
    ingest_fred.py's _resolve_start() takes the MINIMUM watermark across
    ALL series to compute one shared start date for that run's fetch loop.
    Clearing only one series' watermark does nothing useful if the other
    series still have recent watermarks — they pull the shared start date
    forward, silently truncating the backfill for every series, including
    the one you "reset."

    Same root cause as the yfinance and WorldBank watermark bugs (see
    clear_yfinance_watermarks.py, clear_worldbank_watermarks.py). All
    series must be cleared together for a clean full backfill.

    Note one difference from WorldBank: ingest_fred.py calls
    update_watermark() per-series, INSIDE the fetch loop, immediately after
    each series' own fetch succeeds (not in one batch at the end). This
    means a partial watermark clear + re-run can produce a run where SOME
    series get backfilled correctly and others don't, all within the same
    job run, with no error — silently inconsistent rather than uniformly
    wrong. All the more reason to always clear the full list.

Usage:
    python clear_fred_watermarks.py
        Clears the series hardcoded in DEFAULT_SERIES below.

    python clear_fred_watermarks.py --all-from-yaml
        Reads ingestion/configs/sources/fred.yaml and clears every series
        code listed there. Use this one — it can't drift out of sync with
        the live config the way a hardcoded list can.

    python clear_fred_watermarks.py --dry-run
        Shows what would be cleared without touching DynamoDB.

    python clear_fred_watermarks.py --all-from-yaml --env prod
        Same, against the prod watermarks table. Use only when actually
        ready to backfill prod — see PROD_CARRYFORWARD.md first.
"""
import argparse
import os
import sys

import boto3
import yaml

SOURCE = "fred"
REGION = "us-east-2"

# Fallback list — kept in sync with fred.yaml as of 2026-06-19 (18 series).
# Prefer --all-from-yaml so this never silently drifts from the real config.
DEFAULT_SERIES = [
    "GDP",
    "CPIAUCSL",
    "FEDFUNDS",
    "UNRATE",
    "DGS10",
    "DGS2",
    "M2SL",
    "UMCSENT",
    "DTWEXBGS",
    "DEXUSEU",
    "DEXJPUS",
    "DEXUSUK",
    "DEXINUS",
    "DEXCHUS",
    "GOLDAMGBD228NLBM",
    "DCOILWTICO",
    "BAMLH0A0HYM2",
    "T10Y2Y",
]


def _table_name(env: str) -> str:
    return f"trade-platform-{env}-watermarks"


def load_series_from_yaml(yaml_path: str) -> list:
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    series = config.get("series", {})
    if not series:
        raise ValueError(f"No 'series' key found in {yaml_path}")
    return list(series.keys())


def clear_watermarks(series_ids: list, env: str, dry_run: bool = False):
    table_name = _table_name(env)
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(table_name)

    print(f"Table:   {table_name}")
    print(f"Source:  {SOURCE}")
    print(f"Series:  {len(series_ids)}")
    for sid in series_ids:
        print(f"  - {sid}")

    if dry_run:
        print("\n[DRY RUN] No watermarks were deleted.")
        return

    cleared = 0
    missing = 0
    for sid in series_ids:
        try:
            resp = table.delete_item(
                Key={"source_name": SOURCE, "dataset_name": sid},
                ReturnValues="ALL_OLD",
            )
            if "Attributes" in resp:
                cleared += 1
                print(f"  Cleared: {sid} "
                      f"(was last_ingested_period="
                      f"{resp['Attributes'].get('last_ingested_period', '?')})")
            else:
                missing += 1
                print(f"  No watermark existed for: {sid}")
        except Exception as e:
            print(f"  ERROR clearing {sid}: {e}")

    print(f"\nDone. Cleared {cleared}, already-absent {missing}, "
          f"total {len(series_ids)}.")
    print("\nAll series must be cleared together for a clean full backfill —"
          " partial clears will be overridden by the shared min() watermark"
          " across the remaining series. update_watermark() also runs"
          " per-series inside the fetch loop, so a partial clear can produce"
          " a run where some series backfill correctly and others silently"
          " don't, within the same job run.")


def main():
    p = argparse.ArgumentParser(
        description="Clear FRED watermarks (all series must be cleared "
                     "together — see module docstring)."
    )
    p.add_argument("--all-from-yaml", action="store_true",
                    help="Read series codes from fred.yaml instead of the "
                         "hardcoded DEFAULT_SERIES list.")
    p.add_argument("--yaml-path",
                    default=os.path.join(
                        "ingestion", "configs", "sources", "fred.yaml"),
                    help="Path to fred.yaml (used with --all-from-yaml).")
    p.add_argument("--env", default="dev", help="Environment (dev/prod).")
    p.add_argument("--dry-run", action="store_true",
                    help="Show what would be cleared without deleting.")
    args = p.parse_known_args()[0]

    if args.env == "prod":
        print("WARNING: --env prod selected. Confirm PROD_CARRYFORWARD.md "
              "items for FRED are all addressed before proceeding.\n")

    if args.all_from_yaml:
        if not os.path.exists(args.yaml_path):
            print(f"ERROR: {args.yaml_path} not found. "
                  f"Run from the trade-platform repo root, or pass "
                  f"--yaml-path explicitly.")
            sys.exit(1)
        series_ids = load_series_from_yaml(args.yaml_path)
    else:
        series_ids = DEFAULT_SERIES

    clear_watermarks(series_ids, args.env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()