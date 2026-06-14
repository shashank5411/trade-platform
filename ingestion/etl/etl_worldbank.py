"""
ETL: World Bank raw JSON → economic_indicators canonical Parquet.

Raw format: wide, one row per country+year, one column per indicator
[{"country": "Brazil", "date": "2025",
  "gdp_current_usd": null, "gdp_per_capita_usd": null, ...}]

Transform: melt wide → tall, one row per country+indicator+year.
Overwrites latest value per indicator+country+date (no revision tracking).
Country list and indicators driven by YAML config.
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
import yaml
import pandas as pd
from datetime import date
from io import BytesIO

# ── Path setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, _libs_dir)
sys.path.insert(0, "/tmp/ingestion")

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

RAW_BUCKET  = _bucket("worldbank", "raw")
PROC_BUCKET = _bucket("worldbank", "processed")
CONFIG_PATH = _arg("config_path",
    os.path.join(_libs_dir, "configs", "sources", "worldbank.yaml"))

from utils.transform import (
    now_utc,
    safe_float,
    to_json_str,
    validate_indicator_row,
)

s3 = boto3.client("s3", region_name="us-east-2")


# ── Config ─────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# World Bank API returns full country names, not ISO codes.
# This map covers your 7 YAML countries — extend if you add more.
ISO_TO_NAME = {
    "US": "United States",
    "CN": "China",
    "IN": "India",
    "GB": "United Kingdom",
    "DE": "Germany",
    "JP": "Japan",
    "BR": "Brazil",
    "FR": "France",
    "CA": "Canada",
    "AU": "Australia",
    "KR": "Korea, Rep.",
    "MX": "Mexico",
    "IT": "Italy",
    "RU": "Russian Federation",
    "ZA": "South Africa",
}

# Indicator metadata — human name and unit per WDI code
INDICATOR_META = {
    "NY.GDP.MKTP.CD": ("GDP (current USD)",            "current_usd"),
    "NY.GDP.PCAP.CD": ("GDP per capita (current USD)", "current_usd"),
    "FP.CPI.TOTL.ZG": ("Inflation (annual %)",         "percent"),
    "SP.POP.TOTL":    ("Population (total)",            "persons"),
    "NE.TRD.GNFS.ZS": ("Trade (% of GDP)",             "percent"),
}


# ── S3 helpers ─────────────────────────────────────────────────────────────

def read_latest_raw() -> list:
    """
    Read the latest bulk World Bank raw file.
    Only reads files under year= prefix (rewritten ingestion format).
    Handles NaN tokens written by pandas JSON serialization.
    """
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=RAW_BUCKET):
        keys.extend([o["Key"] for o in page.get("Contents", [])])

    keys = [k for k in keys if k.startswith("year=")]
    if not keys:
        raise FileNotFoundError(
            f"No raw files found under year= prefix in s3://{RAW_BUCKET}"
        )

    latest_key = sorted(keys)[-1]
    print(f"  Reading s3://{RAW_BUCKET}/{latest_key}")

    obj     = s3.get_object(Bucket=RAW_BUCKET, Key=latest_key)
    raw     = obj["Body"].read().decode("utf-8")
    raw     = raw.replace(": NaN", ": null").replace(":NaN", ":null")
    records = json.loads(raw)

    countries = set(r["country"] for r in records)
    years     = sorted(set(r["date"] for r in records))
    print(f"  Loaded {len(records)} records — "
          f"{len(countries)} countries, years: {years}")
    return records


def write_processed(rows: list) -> int:
    """Write canonical rows partitioned by source= / year= / indicator_id="""
    if not rows:
        print("  No rows to write")
        return 0

    df           = pd.DataFrame(rows)
    df["year"]   = df["year"].astype(int)
    total_written = 0

    for (year, indicator_id), group in df.groupby(["year", "indicator_id"]):
        # WDI codes have dots — replace for S3 path safety
        safe_id = indicator_id.replace(".", "_")
        key = (f"economic_indicators/source=WORLDBANK/"
       f"year={year}/"
       f"indicator_id={indicator_id}/"
       f"data.parquet")

        buf = BytesIO()
        group.drop(columns=["year", "source", "indicator_id"]).to_parquet(
            buf, index=False, engine="pyarrow", compression="snappy"
        )
        buf.seek(0)
        s3.put_object(Bucket=PROC_BUCKET, Key=key, Body=buf.getvalue())
        total_written += len(group)
        print(f"  Wrote {len(group)} rows → s3://{PROC_BUCKET}/{key}")

    return total_written


# ── Transform ──────────────────────────────────────────────────────────────

def transform(records: list, config: dict) -> list:
    """
    Melt wide World Bank records into tall canonical rows.

    Wide:  {country, date, gdp_current_usd, inflation_pct, ...}
    Tall:  one row per country + indicator + year

    Null values are kept — World Bank gaps are meaningful.
    Countries filtered to YAML config list only.
    """
    ingested_at = now_utc()
    today       = date.today()
    rows        = []
    skipped_null    = 0
    skipped_country = 0

    # Build name→ISO reverse map from config country list
    iso_countries   = config.get("countries", [])
    name_to_iso     = {ISO_TO_NAME[iso]: iso
                       for iso in iso_countries
                       if iso in ISO_TO_NAME}

    # indicator_id → column name mapping from config
    # YAML: {NY.GDP.MKTP.CD: gdp_current_usd, ...}
    indicators      = config.get("indicators", {})
    # Reverse: column_name → indicator_id
    col_to_id       = {v: k for k, v in indicators.items()}

    indicator_cols  = list(col_to_id.keys())

    for rec in records:
        country_name = rec.get("country", "")
        country_iso  = name_to_iso.get(country_name)

        if not country_iso:
            skipped_country += 1
            continue

        year_str = rec.get("date", "")
        try:
            # Handle both "2024" and "2024-01-01 00:00:00" formats
            year     = int(str(year_str)[:4])
            obs_date = date(year, 1, 1)  # WB is annual — normalize to Jan 1
        except (ValueError, TypeError):
            print(f"  WARN: unparseable date '{year_str}' — skipping")
            continue

        # Melt each indicator column into its own row
        for col in indicator_cols:
            raw_val      = rec.get(col)
            value        = safe_float(raw_val)  # None if null — intentional
            indicator_id = col_to_id[col]
            meta         = INDICATOR_META.get(indicator_id, (col, "units"))
            ind_name, unit = meta

            # Only skip if column is entirely missing from record
            # Keep null values — they're meaningful gaps, not errors
            if col not in rec:
                skipped_null += 1
                continue

            row = {
                "source":         "WORLDBANK",
                "indicator_id":   indicator_id,
                "indicator_name": ind_name,
                "date":           str(obs_date),
                "vintage_date":   str(today),
                "year":           year,
                "value":          value,
                "unit":           unit,
                "frequency":      "annual",
                "country":        country_iso,
                "metadata":       to_json_str({
                    "country_name": country_name,
                    "col_name":     col,
                }),
                "ingested_at":    ingested_at,
            }

            errors = validate_indicator_row(row)
            if errors:
                print(f"  WARN: skipping {indicator_id}/"
                      f"{country_iso}/{obs_date}: {errors}")
                continue

            rows.append(row)

    print(f"  Skipped {skipped_country} records "
          f"(country not in config)")
    return rows


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    print(f"ETL WorldBank → economic_indicators | env={ENV}")
    print(f"  raw:       s3://{RAW_BUCKET}")
    print(f"  processed: s3://{PROC_BUCKET}")
    print(f"  config:    {CONFIG_PATH}")

    config  = load_config()
    print(f"  Countries: {config.get('countries')}")
    print(f"  Indicators: {list(config.get('indicators', {}).keys())}")

    records       = read_latest_raw()
    rows          = transform(records, config)
    total_written = write_processed(rows)

    print(f"\n{'─'*50}")
    print(f"Done. {total_written} rows written "
          f"({len(rows)} canonical rows total).")


if __name__ == "__main__":
    main()