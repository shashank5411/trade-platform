import sys
sys.path.insert(0, ".")
from query.athena import query
from query import api

db = api.DB["yfinance"]
sql3 = "SELECT COUNT(*) as rows, COUNT(DISTINCT date) as distinct_dates FROM market_prices WHERE ticker = 'AAPL' AND CAST(year AS INTEGER) = 2024"
print(query(sql3, db))
