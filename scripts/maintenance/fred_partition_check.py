import boto3

s3 = boto3.client("s3", region_name="us-east-2")
bucket = "dev-trade-fred-processed-<ACCOUNT_ID>"

paginator = s3.get_paginator("list_objects_v2")
keys = []
for page in paginator.paginate(Bucket=bucket, Prefix="economic_indicators/source=FRED/"):
    keys.extend([o["Key"] for o in page.get("Contents", [])])

partitions = set()
for k in keys:
    # strip the filename, keep year=/indicator_id=
    parts = k.split("/vintage=")[0]
    partitions.add(parts)

print(f"Distinct (year, indicator_id) partition folders in S3: {len(partitions)}")
print(f"Total files: {len(keys)}")
