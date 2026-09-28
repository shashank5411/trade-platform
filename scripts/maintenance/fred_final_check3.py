import sys
sys.path.insert(0, ".")
from query import api

sql = api._indicator_agg_sql("DCOILWTICO", "2023-01-01", "2023-01-31", "daily", "", api.DB["fred"])
print(sql)
print("=" * 60)
from query.athena import query
df = query(sql, api.DB["fred"])
print(f"Rows: {len(df)}")
print(df.to_string(index=False))
