"""
Clear yfinance DynamoDB watermarks for a given ticker list.

Why this exists:
  _resolve_start() in ingest_yfinance.py takes the MINIMUM watermark
  across ALL tickers being ingested in one run. If even one ticker still
  has a live (recent) watermark, it pulls the whole batch's start date
  toward that recent date — silently truncating the backfill for every
  other ticker, even ones with no watermark at all.

  Clearing a subset of tickers (e.g. just AAPL) does NOT fix this if
  other tickers in the same yfinance.yaml config still have live
  watermarks. ALL tickers in the active config must be cleared together
  for a clean full-history backfill.

Usage:
  python clear_yfinance_watermarks.py                  # clears TEST_TICKERS below
  python clear_yfinance_watermarks.py --all-from-yaml   # reads tickers from yfinance.yaml
"""
import argparse
import boto3

REGION = "us-east-2"
TABLE_NAME = "trade-platform-dev-watermarks"
SOURCE = "yfinance"

# Tickers used in the current AAPL + indices/FX/futures test config
TEST_TICKERS = [
    "AAPL",
    "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX",
    "^FTSE", "^GDAXI", "^FCHI", "^STOXX50E",
    "^N225", "^HSI", "^NSEI", "^AXJO", "^KS11",
    "GC=F", "CL=F", "SI=F", "NG=F",
    "DX-Y.NYB", "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDINR=X", "USDCNY=X",
]


def load_tickers_from_yaml(path: str = "ingestion/configs/sources/yfinance.yaml") -> list:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("tickers", [])


def clear_watermarks(tickers: list) -> None:
    ddb = boto3.resource("dynamodb", region_name=REGION)
    table = ddb.Table(TABLE_NAME)

    print(f"Clearing {len(tickers)} watermark(s) for source='{SOURCE}'...")
    cleared = 0
    with table.batch_writer() as batch:
        for t in tickers:
            batch.delete_item(Key={"source_name": SOURCE, "dataset_name": t})
            cleared += 1

    print(f"Done. Cleared {cleared} watermark(s).")
    print("Next ingest run will fall back to default_start_date for ALL these tickers.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-from-yaml",
        action="store_true",
        help="Read the full ticker list from ingestion/configs/sources/yfinance.yaml "
             "instead of the hardcoded TEST_TICKERS list. Use this for the full "
             "S&P 500 run, not the AAPL-only test.",
    )
    parser.add_argument(
        "--yaml-path",
        default="ingestion/configs/sources/yfinance.yaml",
        help="Path to yfinance.yaml when using --all-from-yaml",
    )
    args = parser.parse_args()

    if args.all_from_yaml:
        tickers = load_tickers_from_yaml(args.yaml_path)
        print(f"Loaded {len(tickers)} tickers from {args.yaml_path}")
    else:
        tickers = TEST_TICKERS

    clear_watermarks(tickers)


if __name__ == "__main__":
    main()