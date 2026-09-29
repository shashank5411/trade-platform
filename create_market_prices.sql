CREATE EXTERNAL TABLE market_prices (
  ticker      string,
  date        string,
  country     string,
  currency    string,
  open        double,
  high        double,
  low         double,
  close       double,
  adj_close   double,
  volume      bigint,
  source      string,
  metadata    string,
  ingested_at string
)
PARTITIONED BY (
  year     string,
  exchange string,
  ticker   string
)
STORED AS PARQUET
LOCATION 's3://dev-trade-yfinance-processed-<ACCOUNT_ID>/market_prices/'
TBLPROPERTIES ('parquet.compress'='SNAPPY')
