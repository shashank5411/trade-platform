import boto3, time

athena = boto3.client("athena", region_name="us-east-2")

resp = athena.start_query_execution(
    QueryString="MSCK REPAIR TABLE market_prices",
    QueryExecutionContext={"Database": "dev_trade_yfinance_processed"},
    ResultConfiguration={"OutputLocation": "s3://dev-trade-athena-results-<ACCOUNT_ID>/"}
)
qid = resp["QueryExecutionId"]
print("QueryId:", qid)

for _ in range(30):
    time.sleep(2)
    status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
    state  = status["State"]
    print("State:", state)
    if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
        print("Reason:", status.get("StateChangeReason", ""))
        break
