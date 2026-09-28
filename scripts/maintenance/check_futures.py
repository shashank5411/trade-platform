import sys
sys.path.insert(0, ".")
from query import api
from query.athena import query

for ticker in ["GC=F", "CL=F", "SI=F", "NG=F"]:
    safe = api._safe_partition_value(ticker)
    sql = f"SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as cnt FROM market_prices WHERE ticker = '\''{safe}'\''"
    df = query(sql, api.DB["yfinance"])
    print(ticker, "-> partition:", safe)
    print(df)
    print()
