import sys
sys.path.insert(0, ".")
from query.athena import query
from query import api

db = api.DB["yfinance"]

# 1. Confirm new equities have data
sql1 = "SELECT ticker_symbol, COUNT(*) as rows, MIN(date) as earliest, MAX(date) as latest FROM market_prices WHERE ticker IN ('COR','MRSH','CRWD','TTD','XYZ') GROUP BY ticker_symbol"
print("New/renamed equities:")
print(query(sql1, db))

# 2. Confirm previously-broken futures/FX now have data
sql2 = "SELECT ticker_symbol, COUNT(*) as rows, MIN(date) as earliest, MAX(date) as latest FROM market_prices WHERE ticker IN ('GC_F','CL_F','SI_F','NG_F') GROUP BY ticker_symbol"
print("\nFutures (previously zero rows):")
print(query(sql2, db))

# 3. Spot check AAPL full continuity across year
sql3 = "SELECT COUNT(*) as rows, COUNT(DISTINCT date) as distinct_dates FROM market_prices WHERE ticker = 'AAPL' AND year = 2024"
print("\nAAPL 2024 continuity check:")
print(query(sql3, db))
