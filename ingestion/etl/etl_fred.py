"""
ETL: FRED raw JSON → economic_indicators canonical Parquet.

Raw format: single bulk file, flat list of records
[{"series_id": "GDP", "series_label": "gdp_billions_usd",
  "date": "2020-01-01", "value": 21751.238}, ...]

Revision-only append pattern:
- Reads existing latest value from processed layer
- Only writes a new row when value actually changed
- Preserves all historical vintages
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
import pandas as pd
from datetime import date, datetime, timezone
from io import BytesIO

# ── Path setup — works both locally and in Glue ────────────────────────────
sys.path.insert(0, _libs_dir)
sys.path.insert(0, "/tmp/ingestion")

# ── Arg handling — works locally and in Glue Python Shell ─────────────────
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

RAW_BUCKET  = _bucket("fred", "raw")
PROC_BUCKET = _bucket("fred", "processed")

from utils.transform import (
    now_utc,
    normalize_date_to_period_start,
    safe_float,
    to_json_str,
    validate_indicator_row,
)

s3 = boto3.client("s3", region_name="us-east-2")


# ── Frequency inference ────────────────────────────────────────────────────
# Inferred from series_label since raw file doesn't include frequency.
# Extend this map as you add more series to your YAML config.
FREQ_HINTS = {
    "gdp":          "quarterly",
    "quarterly":    "quarterly",
    "annual":       "annual",
    "yearly":       "annual",
    "unemployment": "monthly",
    "inflation":    "monthly",
    "cpi":          "monthly",
    "rate":         "monthly",
    "DGS10":        "daily",
    "DGS2":         "daily",
    "DGS":          "daily",
}

# Unit lookup by series_id — extend as you add series to YAML config
UNIT_MAP = {
    "GDP":      "billions_usd",
    "CPIAUCSL": "index_1982_84_100",
    "UNRATE":   "percent",
    "FEDFUNDS": "percent",
    "T10Y2Y":   "percent",
    "UMCSENT":  "index_1966_100",
    "HOUST":    "thousands_of_units",
    "INDPRO":   "index_2017_100",
    # Added — dollar / FX
    "DTWEXBGS": "index",
    "DEXUSEU":  "usd_per_eur",
    "DEXJPUS":  "jpy_per_usd",
    "DEXUSUK":  "usd_per_gbp",
    "DEXINUS":  "inr_per_usd",
    "DEXCHUS":  "cny_per_usd",
    # Added — commodities
    "GOLDAMGBD228NLBM": "usd_per_troy_oz",
    "DCOILWTICO":       "usd_per_barrel",
    # Added — credit / spreads
    "BAMLH0A0HYM2": "percent",
}
DEFAULT_UNIT = "units"

# Explicit series_id -> frequency overrides. Always checked first, before
# any fuzzy label-substring matching. Add every new series here when it's
# added to fred.yaml — do not rely on label substrings for known series.
SERIES_FREQUENCY_OVERRIDE = {
    "DGS10": "daily",
    "DGS2": "daily",
    "DTWEXBGS": "daily",
    "DEXUSEU": "daily",
    "DEXJPUS": "daily",
    "DEXUSUK": "daily",
    "DEXINUS": "daily",
    "DEXCHUS": "daily",
    "GOLDAMGBD228NLBM": "daily",
    "DCOILWTICO": "daily",
    "BAMLH0A0HYM2": "daily",
    "T10Y2Y": "daily",
}

def infer_frequency(series_id: str, label: str) -> str:
    if series_id in SERIES_FREQUENCY_OVERRIDE:
        return SERIES_FREQUENCY_OVERRIDE[series_id]
    combined = f"{series_id} {label}".lower()
    for hint, freq in FREQ_HINTS.items():
        if hint in combined:
            return freq
    return "monthly"  # safe default for most FRED series


# ── S3 helpers ─────────────────────────────────────────────────────────────

def read_all_raw_records() -> dict:
    """
    Read the latest bulk FRED raw file from S3.
    Returns dict of {series_id: [records]} grouped for processing.
    Raw path: year={y}/month={m}/fred_*.json
    """
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=RAW_BUCKET):
        keys.extend([o["Key"] for o in page.get("Contents", [])])

    if not keys:
        raise FileNotFoundError(
            f"No raw files found in s3://{RAW_BUCKET}"
        )

    latest_key = sorted(keys)[-1]
    print(f"  Reading s3://{RAW_BUCKET}/{latest_key}")

    obj = s3.get_object(Bucket=RAW_BUCKET, Key=latest_key)
    records = json.loads(obj["Body"].read())

    if not isinstance(records, list):
        raise ValueError(
            f"Expected list of records, got {type(records)}"
        )

    # Group by series_id
    grouped = {}
    for rec in records:
        sid = rec.get("series_id")
        if not sid:
            continue
        grouped.setdefault(sid, []).append(rec)

    print(f"  Loaded {len(grouped)} series, "
          f"{len(records)} total observations")
    return grouped


def get_existing_values_for_series(series_id: str) -> dict:
    """
    Read ALL existing processed values for a series in one pass.
    Returns dict of {date_str: value} for in-memory revision checks.
    Much faster than per-observation S3 lookups.
    """
    prefix = f"economic_indicators/source=FRED/year="
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    try:
        for page in paginator.paginate(Bucket=PROC_BUCKET):
            for obj in page.get("Contents", []):
                if f"indicator_id={series_id}/" in obj["Key"]:
                    keys.append(obj["Key"])

        if not keys:
            return {}  # First run — nothing exists yet

        dfs = []
        for key in keys:
            obj = s3.get_object(Bucket=PROC_BUCKET, Key=key)
            dfs.append(pd.read_parquet(BytesIO(obj["Body"].read())))

        df = pd.concat(dfs)
        # Keep only the latest vintage per date
        df = df.sort_values("vintage_date").groupby("date").last()
        return dict(zip(df.index, df["value"]))

    except Exception:
        return {}


def write_processed(series_id: str, rows: list):
    """Write canonical rows to processed bucket as Parquet, partitioned
    by source= / year= / indicator_id= / vintage date filename."""
    if not rows:
        print(f"  No new/revised rows for {series_id} — skipping write")
        return

    df = pd.DataFrame(rows)
    df["year"]  = df["year"].astype(int)
    df["value"] = df["value"].astype(float)

    for year, group in df.groupby("year"):
        key = (f"economic_indicators/source=FRED/"
               f"year={year}/"
               f"indicator_id={series_id}/"
               f"vintage={date.today().isoformat()}.parquet")

        buf = BytesIO()
        group.drop(columns=["year", "source", "indicator_id"]).to_parquet(
            buf, index=False, engine="pyarrow", compression="snappy"
        )
        buf.seek(0)
        s3.put_object(Bucket=PROC_BUCKET, Key=key, Body=buf.getvalue())
        print(f"  Wrote {len(group)} rows → s3://{PROC_BUCKET}/{key}")


# ── Transform ──────────────────────────────────────────────────────────────
def transform_series(series_id: str, records: list) -> list:
    ingested_at = now_utc()
    today       = date.today()
    rows        = []
    skipped_no_revision = 0

    # ← Load ALL existing values once — replaces per-observation S3 calls
    existing = get_existing_values_for_series(series_id)
    print(f"  Found {len(existing)} existing observations in processed layer")

    for rec in records:
        raw_val = rec.get("value")
        if raw_val is None:
            continue

        date_str = rec.get("date", "")
        try:
            obs_date = date.fromisoformat(date_str)
        except ValueError:
            print(f"  WARN: unparseable date '{date_str}' — skipping")
            continue

        label     = rec.get("series_label", series_id)
        frequency = infer_frequency(series_id, label)
        norm_date = normalize_date_to_period_start(obs_date, frequency)
        value     = safe_float(raw_val)

        # In-memory revision check — no S3 call
        existing_val = existing.get(str(norm_date))
        if existing_val is not None and safe_float(existing_val) == value:
            skipped_no_revision += 1
            continue

        row = {
            "source":         "FRED",
            "indicator_id":   series_id,
            "indicator_name": label,
            "date":           str(norm_date),
            "vintage_date":   str(today),
            "year":           norm_date.year,
            "value":          value,
            "unit":           UNIT_MAP.get(series_id, DEFAULT_UNIT),
            "frequency":      frequency,
            "country":        "US",
            "metadata":       to_json_str({"series_label": label}),
            "ingested_at":    ingested_at,
        }

        errors = validate_indicator_row(row)
        if errors:
            print(f"  WARN: skipping {series_id}/{norm_date}: {errors}")
            continue

        rows.append(row)

    if skipped_no_revision:
        print(f"  Skipped {skipped_no_revision} unchanged observations")

    return rows


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    print(f"ETL FRED → economic_indicators | env={ENV}")
    print(f"  raw:       s3://{RAW_BUCKET}")
    print(f"  processed: s3://{PROC_BUCKET}")

    grouped = read_all_raw_records()

    total_written = 0
    total_skipped = 0

    for series_id, records in grouped.items():
        print(f"\nProcessing {series_id} "
              f"({len(records)} observations)...")
        try:
            rows = transform_series(series_id, records)
            write_processed(series_id, rows)
            total_written += len(rows)
            if not rows:
                total_skipped += 1
        except Exception as e:
            print(f"  ERROR: {series_id} failed — {e}")
            continue

    print(f"\n{'─'*50}")
    print(f"Done. {total_written} rows written, "
          f"{total_skipped} series with no revisions.")


if __name__ == "__main__":
    main()