import boto3, json, gzip

s3 = boto3.client("s3", region_name="us-east-2")
paginator = s3.get_paginator("list_objects_v2")
keys = []
for page in paginator.paginate(Bucket="dev-trade-fedspeak-raw-197411402303"):
    keys.extend([o["Key"] for o in page.get("Contents", [])])

doc_keys = [k for k in keys if k.endswith(".json.gz")]
print(f"Total documents in raw bucket: {len(doc_keys)}")

sample_key = doc_keys[0]
obj = s3.get_object(Bucket="dev-trade-fedspeak-raw-197411402303", Key=sample_key)
doc = json.loads(gzip.decompress(obj["Body"].read()))
print()
print("Sample doc:", sample_key)
print("  Type:", doc["doc_type"], "| Date:", doc["doc_date"], "| Entity:", doc["entity"])
print("  Char count:", doc["char_count"])
print("  Text preview:", doc["text"][:200])
