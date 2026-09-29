import boto3
dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
table = dynamodb.Table("trade-platform-dev-watermarks")

with open("chunk3_missing_tickers.txt") as f:
    missing = [line.strip() for line in f if line.strip()]

print(f"Checking watermarks for all {len(missing)} chunk3-missing tickers...\n")

no_watermark = []
for ticker in missing:
    resp = table.get_item(Key={"source_name": "yfinance", "dataset_name": ticker})
    item = resp.get("Item")
    if not item:
        no_watermark.append(ticker)

print(f"Tickers with NO watermark at all: {len(no_watermark)}")
print(no_watermark[:20], "..." if len(no_watermark) > 20 else "")
