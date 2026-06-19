import boto3, json, yaml

s3 = boto3.client("s3", region_name="us-east-2")

# Find the chunk3 file from the most recent run
paginator = s3.get_paginator("list_objects_v2")
keys = []
for page in paginator.paginate(Bucket="dev-trade-yfinance-raw-197411402303"):
    keys.extend([o["Key"] for o in page.get("Contents", [])])

chunk3_keys = sorted([k for k in keys if "chunk003" in k])
if not chunk3_keys:
    print("No chunk003 file found - check actual chunk numbering/filenames")
    print("Available chunk-like keys:", [k for k in keys if "_chunk" in k][-10:])
else:
    latest_chunk3 = chunk3_keys[-1]
    print("Using:", latest_chunk3)
    obj = s3.get_object(Bucket="dev-trade-yfinance-raw-197411402303", Key=latest_chunk3)
    records = json.loads(obj["Body"].read())
    tickers_in_chunk3 = set(r["ticker"] for r in records)

    with open("ingestion/configs/sources/yfinance.yaml") as f:
        config = yaml.safe_load(f)
    all_tickers = set(config["tickers"])

    missing = all_tickers - tickers_in_chunk3
    print(f"Tickers present in chunk3: {len(tickers_in_chunk3)}")
    print(f"Tickers MISSING from chunk3: {len(missing)}")
    print(sorted(missing)[:10], "... (showing first 10)")

    with open("chunk3_missing_tickers.txt", "w") as f:
        f.write("\n".join(sorted(missing)))
