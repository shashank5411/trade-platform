import boto3, time

athena = boto3.client("athena", region_name="us-east-2")

sql = (
    "CREATE EXTERNAL TABLE market_prices ("
    "  ticker      string,"
    "  date        string,"
    "  country     string,"
    "  currency    string,"
    "  open        double,"
    "  high        double,"
    "  low         double,"
    "  close       double,"
    "  adj_close   double,"
    "  volume      double,"
    "  source      string,"
    "  metadata    string,"
    "  ingested_at string"
    ") PARTITIONED BY ("
    "  year        string,"
    "  `exchange`  string"
    ") STORED AS PARQUET"
    " LOCATION 's3://dev-trade-yfinance-processed-<ACCOUNT_ID>/market_prices/'"
    " TBLPROPERTIES ('parquet.compress'='SNAPPY')"
)

resp = athena.start_query_execution(
    QueryString=sql,
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
