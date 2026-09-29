import sys
sys.path.insert(0, ".")
from query import api

result = api.get_indicator_multi(["FEDFUNDS", "BAMLH0A0HYM2"], "2025-06-19", "2026-06-19")
print(result)
