import sys
sys.path.insert(0, ".")
from query import api
from query.athena import query

df = query("SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as cnt FROM market_prices WHERE ticker_symbol = '"'"'CL=F'"'"'", api.DB["yfinance"])
print(df)
