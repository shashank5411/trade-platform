import boto3

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

# Exact signature confirmed by audit: vintage=2026-06-14 AND one of the 12 affected series
to_delete = [
    k for k in keys
    if "vintage=2026-06-14.parquet" in k
    and any(f"indicator_id={s}/" in k for s in affected_series)
]

print(f"Files matching delete signature: {len(to_delete)}")
for k in to_delete:
    print(f"  {k}")

with open("fred_cleanup_keys.txt", "w") as f:
    f.write("\n".join(to_delete))
print("\nWritten to fred_cleanup_keys.txt for review before deleting.")
