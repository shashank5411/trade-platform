"""
Clear DynamoDB watermarks for World Bank indicators.

Why this exists:
    ingest_worldbank.py's _resolve_start() takes the MINIMUM watermark
    across ALL indicators to compute one shared start date for the
    single fetch_wdi() call. Clearing only one indicator's watermark
    does nothing useful if the other 4 still have recent watermarks —
    they pull the shared start date forward, silently truncating the
    backfill for every indicator, including the one you "reset."

    Same root cause as the yfinance watermark bug (see
    clear_yfinance_watermarks.py / SESSION_SUMMARY 2026-06-19):
    per-series watermarking + a shared-min start-date resolver means
    watermark resets are all-or-nothing for the active indicator list.

Usage:
    python clear_worldbank_watermarks.py
        Clears the 5 indicators hardcoded in DEFAULT_INDICATORS below.

    python clear_worldbank_watermarks.py --all-from-yaml
        Reads ingestion/configs/sources/worldbank.yaml and clears
        every indicator code listed there. Use this one — it's the
        version that can't drift out of sync with the live config.

    python clear_worldbank_watermarks.py --dry-run
        Shows what would be cleared without touching DynamoDB.
"""
import argparse
import os
import sys

import boto3
import yaml

SOURCE = "worldbank"
REGION = "us-east-2"

# Fallback list — kept in sync with worldbank.yaml as of 2026-06-18.
# Prefer --all-from-yaml so this never silently drifts from the real config.
DEFAULT_INDICATORS = [
    "NY.GDP.MKTP.CD",
    "NY.GDP.PCAP.CD",
    "FP.CPI.TOTL.ZG",
    "SP.POP.TOTL",
    "NE.TRD.GNFS.ZS",
]


def _table_name(env: str) -> str:
    return f"trade-platform-{env}-watermarks"


def load_indicators_from_yaml(yaml_path: str) -> list:
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    indicators = config.get("indicators", {})
    if not indicators:
        raise ValueError(f"No 'indicators' key found in {yaml_path}")
    return list(indicators.keys())


def clear_watermarks(indicator_codes: list, env: str, dry_run: bool = False):
    table_name = _table_name(env)
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(table_name)

    print(f"Table:      {table_name}")
    print(f"Source:     {SOURCE}")
    print(f"Indicators: {len(indicator_codes)}")
    for code in indicator_codes:
        print(f"  - {code}")

    if dry_run:
        print("\n[DRY RUN] No watermarks were deleted.")
        return

    cleared = 0
    missing = 0
    for code in indicator_codes:
        try:
            resp = table.delete_item(
                Key={"source_name": SOURCE, "dataset_name": code},
                ReturnValues="ALL_OLD",
            )
            if "Attributes" in resp:
                cleared += 1
                print(f"  Cleared: {code} "
                      f"(was last_ingested_period="
                      f"{resp['Attributes'].get('last_ingested_period', '?')})")
            else:
                missing += 1
                print(f"  No watermark existed for: {code}")
        except Exception as e:
            print(f"  ERROR clearing {code}: {e}")

    print(f"\nDone. Cleared {cleared}, already-absent {missing}, "
          f"total {len(indicator_codes)}.")
    print("\nAll indicators must be cleared together for a clean full "
          "backfill — partial clears will be overridden by the shared "
          "min() watermark across the remaining indicators.")


def main():
    p = argparse.ArgumentParser(
        description="Clear World Bank watermarks (all indicators must be "
                     "cleared together — see module docstring)."
    )
    p.add_argument("--all-from-yaml", action="store_true",
                    help="Read indicator codes from worldbank.yaml instead "
                         "of the hardcoded DEFAULT_INDICATORS list.")
    p.add_argument("--yaml-path",
                    default=os.path.join(
                        "ingestion", "configs", "sources", "worldbank.yaml"),
                    help="Path to worldbank.yaml (used with --all-from-yaml).")
    p.add_argument("--env", default="dev", help="Environment (dev/prod).")
    p.add_argument("--dry-run", action="store_true",
                    help="Show what would be cleared without deleting.")
    args = p.parse_known_args()[0]

    if args.all_from_yaml:
        if not os.path.exists(args.yaml_path):
            print(f"ERROR: {args.yaml_path} not found. "
                  f"Run from the trade-platform repo root, or pass "
                  f"--yaml-path explicitly.")
            sys.exit(1)
        indicator_codes = load_indicators_from_yaml(args.yaml_path)
    else:
        indicator_codes = DEFAULT_INDICATORS

    clear_watermarks(indicator_codes, args.env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()