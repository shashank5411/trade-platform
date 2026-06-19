import yaml

with open("ingestion/configs/sources/yfinance.yaml") as f:
    config = yaml.safe_load(f)
tickers = set(config["tickers"])

with open("current_sp500_503_fixed.txt") as f:
    fresh = set(line.strip() for line in f if line.strip())

non_equity = {"^GSPC","^DJI","^IXIC","^RUT","^VIX","^FTSE","^GDAXI","^FCHI",
              "^STOXX50E","^N225","^HSI","^NSEI","^AXJO","^KS11",
              "GC=F","CL=F","SI=F","NG=F","EURUSD=X","GBPUSD=X",
              "USDJPY=X","USDINR=X","USDCNY=X"}

equity_in_yaml = tickers - non_equity
print("Equity tickers in yaml:", len(equity_in_yaml))
print("In yaml but not in fresh 503:", sorted(equity_in_yaml - fresh))
print("In fresh 503 but not in yaml:", sorted(fresh - equity_in_yaml))
