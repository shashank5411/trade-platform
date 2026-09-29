import os
import urllib.request

# SEC requires a real contact in the User-Agent (FAIR ACCESS policy) —
# set SEC_USER_AGENT to your own contact info before running against EDGAR.
HEADERS = {"User-Agent": os.environ.get("SEC_USER_AGENT", "TradePlatform research@example.com")}

candidates = {
    "JNJ": "0000200406",
    "WMT": "0000104169",
    "CAT": "0000018230",
    "PG":  "0000080424",
    "KO":  "0000021344",
    "DIS": "0001744489",
}

for ticker, cik in candidates.items():
    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    req = urllib.request.Request(url, headers=HEADERS, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            size_mb = int(resp.headers.get("Content-Length", 0)) / 1024 / 1024
            print(f"{ticker}: {size_mb:.1f} MB")
    except Exception as e:
        print(f"{ticker}: ERROR — {e}")
