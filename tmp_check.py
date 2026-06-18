from query.athena import query
df = query("SELECT MAX(date) as latest, MIN(date) as earliest FROM market_prices WHERE ticker = 'AAPL'", 'dev_trade_yfinance_processed')
print(df.to_string())