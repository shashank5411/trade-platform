import boto3
dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
table = dynamodb.Table("trade-platform-dev-watermarks")
# Replace with an actual ticker known to have failed in chunk 3, if you have the list
for ticker in ["<failed_ticker_1>", "<failed_ticker_2>"]:
    resp = table.get_item(Key={"source_name": "yfinance", "dataset_name": ticker})
    print(ticker, "->", resp.get("Item", "NO WATERMARK"))
