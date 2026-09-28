import boto3
dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
table = dynamodb.Table("trade-platform-dev-watermarks")
# Check a ticker we know failed in chunk 3 - need the actual failed list,
# but as a stand-in check whether AAPL's watermark reflects 06-17 or 06-19
resp = table.get_item(Key={"source_name": "yfinance", "dataset_name": "AAPL"})
print(resp.get("Item"))
