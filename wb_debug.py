import wbdata
import pandas as pd
from datetime import datetime

indicators = {
    "NY.GDP.MKTP.CD": "gdp_current_usd",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "FP.CPI.TOTL.ZG": "inflation_pct",
    "SP.POP.TOTL": "population",
    "NE.TRD.GNFS.ZS": "trade_pct_gdp",
}
countries = ["US","CN","IN","GB","DE","JP","BR","FR","RU","MX"]

df = wbdata.get_dataframe(indicators, country=countries)
print("index names:", df.index.names)
print("columns:", df.columns.tolist())
print("shape before filter:", df.shape)

df2 = df.reset_index()
print("columns after reset_index:", df2.columns.tolist())

start_dt = datetime(2015, 1, 1)
end_dt = datetime(2026, 6, 19)

if "date" in df2.columns:
    df2["date"] = pd.to_datetime(df2["date"])
    df3 = df2[(df2["date"] >= pd.Timestamp(start_dt)) & (df2["date"] <= pd.Timestamp(end_dt))]
    print("shape after filter:", df3.shape)
    print("year range after filter:", sorted(df3["date"].dt.year.unique()))
else:
    print("NO date COLUMN FOUND AFTER RESET - filter skipped entirely")
