import sys
sys.path.insert(0, ".")
from query import api

result = api.get_indicator("DCOILWTICO", "2023-01-01", "2023-01-31")
print(result)
