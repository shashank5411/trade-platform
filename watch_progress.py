import boto3
s3 = boto3.client("s3", region_name="us-east-2")
paginator = s3.get_paginator("list_objects_v2")
keys = []
for page in paginator.paginate(Bucket="dev-trade-yfinance-raw-197411402303"):
    keys.extend([o["Key"] for o in page.get("Contents", [])])
chunk_keys = sorted([k for k in keys if "_chunk" in k])
meta_keys = [k for k in keys if "_metadata_" in k]
print(f"Chunks written so far: {len(chunk_keys)}")
print(f"Metadata file present: {len(meta_keys) > 0}")
if chunk_keys:
    print("Latest chunk:", chunk_keys[-1])
