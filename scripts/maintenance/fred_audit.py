import boto3
import pyarrow.parquet as pq
import io

s3 = boto3.client("s3", region_name="us-east-2")
bucket = "dev-trade-fred-processed-<ACCOUNT_ID>"

affected_series = [
    "DTWEXBGS", "DEXUSEU", "DEXJPUS", "DEXUSUK", "DEXINUS", "DEXCHUS",
    "GOLDAMGBD228NLBM", "DCOILWTICO", "BAMLH0A0HYM2", "T10Y2Y", "DGS10", "DGS2",
]

paginator = s3.get_paginator("list_objects_v2")
keys = []
for page in paginator.paginate(Bucket=bucket, Prefix="economic_indicators/source=FRED/"):
    keys.extend([o["Key"] for o in page.get("Contents", [])])

bad_files = []
checked = 0
for key in keys:
    if not any(f"indicator_id={s}/" in key for s in affected_series):
        continue
    checked += 1
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
    freqs = df["frequency"].unique().tolist()
    if "monthly" in freqs or "daily" not in freqs:
        bad_files.append((key, len(df), freqs))

print(f"Checked {checked} files across affected series")
print(f"Found {len(bad_files)} files with wrong/old frequency tag:\n")
for key, rows, freqs in bad_files:
    print(f"  {key}  ({rows} rows, frequency={freqs})")
