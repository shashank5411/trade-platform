import urllib.request

HEADERS = {"User-Agent": "Sample Company Name AdminContact@samplecompany.com"}

url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000200406.json"
req = urllib.request.Request(url, headers=HEADERS, method="HEAD")
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        size_mb = int(resp.headers.get("Content-Length", 0)) / 1024 / 1024
        print(f"JNJ: {size_mb:.1f} MB")
except Exception as e:
    print(f"ERROR with proper header — {e}")
