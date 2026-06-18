import boto3

ddb = boto3.resource("dynamodb", region_name="us-east-2")
table = ddb.Table("trade-platform-dev-watermarks")

# Scan for all yfinance watermarks
response = table.scan(
    FilterExpression="source_name = :s",
    ExpressionAttributeValues={":s": "yfinance"}
)
items = response["Items"]

print(f"Found {len(items)} yfinance watermarks — deleting...")
with table.batch_writer() as batch:
    for item in items:
        batch.delete_item(Key={
            "source_name": item["source_name"],
            "dataset_name": item["dataset_name"],
        })

print("Done.")