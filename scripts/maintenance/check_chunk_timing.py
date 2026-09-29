import boto3
s3 = boto3.client("s3", region_name="us-east-2")
paginator = s3.get_paginator("list_objects_v2")
objs = []
for page in paginator.paginate(Bucket="dev-trade-yfinance-raw-<ACCOUNT_ID>"):
    objs.extend(page.get("Contents", []))
chunks = sorted([o for o in objs if "_chunk" in o["Key"]], key=lambda o: o["Key"])
for i in range(1, len(chunks)):
    gap = (chunks[i]["LastModified"] - chunks[i-1]["LastModified"]).total_seconds()
    print(f"{chunks[i]['Key'].split('/')[-1]}: {gap:.0f}s since previous chunk")
