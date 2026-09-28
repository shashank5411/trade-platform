import boto3
s3 = boto3.client("s3", region_name="us-east-2")

for bucket, label in [("dev-trade-insiders-raw-<ACCOUNT_ID>", "Insiders"),
                       ("dev-trade-news-raw-<ACCOUNT_ID>", "News")]:
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket):
        keys.extend([o["Key"] for o in page.get("Contents", [])])
    print(f"{label}: {len(keys)} objects in raw bucket")
