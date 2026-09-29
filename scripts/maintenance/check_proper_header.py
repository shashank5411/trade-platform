import os
import urllib.request

# SEC requires a real contact in the User-Agent (FAIR ACCESS policy) —
# set SEC_USER_AGENT to your own contact info before running against EDGAR.
HEADERS = {"User-Agent": os.environ.get("SEC_USER_AGENT", "Sample Company Name AdminContact@samplecompany.com")}

url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000200406.json"
req = urllib.request.Request(url, headers=HEADERS, method="HEAD")
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        size_mb = int(resp.headers.get("Content-Length", 0)) / 1024 / 1024
        print(f"JNJ: {size_mb:.1f} MB")
except Exception as e:
    print(f"ERROR with proper header — {e}")
