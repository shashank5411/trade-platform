import sys
sys.path.insert(0, ".")
from query import api

result = api.get_indicator("DCOILWTICO", "2023-01-01", "2023-12-31")
print(result[:500])
print("...")
# count rows in output by counting lines that look like data rows
lines = result.split("\n")
print(f"\nTotal output lines: {len(lines)}")
