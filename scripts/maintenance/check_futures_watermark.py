import boto3
dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
table = dynamodb.Table("trade-platform-dev-watermarks")
for ticker in ["GC=F", "CL=F", "SI=F", "NG=F"]:
    resp = table.get_item(Key={"source_name": "yfinance", "dataset_name": ticker})
    print(ticker, "->", resp.get("Item", "NO WATERMARK -- never attempted"))
