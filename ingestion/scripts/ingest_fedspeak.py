import sys
import os
import zipfile

# ── Fix A: Glue zip extraction ────────────────────────────────────────────────
_libs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _entry in os.listdir('/tmp/'):
    if _entry.startswith('glue-python-libs-'):
        _libs_dir = os.path.join('/tmp/', _entry)
        for _f in os.listdir(_libs_dir):
            if _f.endswith('.zip'):
                with zipfile.ZipFile(os.path.join(_libs_dir, _f)) as _z:
                    _z.extractall(_libs_dir)
        sys.path.insert(0, _libs_dir)
        break

import re
import gzip
import json
import hashlib
import argparse
import datetime
import urllib.request
import urllib.error
import time
import xml.etree.ElementTree as ET

import boto3

# ── Fix B: _arg function ──────────────────────────────────────────────────────
def _arg(name: str, default: str = "") -> str:
    for i, a in enumerate(sys.argv):
        if a == f"--{name}" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.getenv(name, default)

ENV          = _arg("ENVIRONMENT", "dev")
ACCOUNT      = boto3.client("sts", region_name="us-east-2").get_caller_identity()["Account"]
RAW_BUCKET   = f"{ENV}-trade-fedspeak-raw-{ACCOUNT}"
REGION       = "us-east-2"

# ── Fix D: S3 client with region ──────────────────────────────────────────────
s3 = boto3.client("s3", region_name=REGION)

HEADERS = {
    "User-Agent": "trade-platform/1.0 (research; contact@example.com)",
    "Accept-Encoding": "gzip, deflate",
}

CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
SPEECHES_RSS = "https://www.federalreserve.gov/feeds/speeches.xml"
BASE_URL     = "https://www.federalreserve.gov"

# ── Config ────────────────────────────────────────────────────────────────────
def _arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ENVIRONMENT", default="dev")
    p.add_argument("--start-date",  default=None)
    # ── Fix C: parse_known_args ───────────────────────────────────────────────
    return p.parse_known_args()[0]

args = _arg_parser()
START_DATE = args.start_date or ("2020-01-01" if ENV == "dev" else "2000-01-01")
start_dt   = datetime.date.fromisoformat(START_DATE)

# ── HTTP helpers ──────────────────────────────────────────────────────────────
def fetch_url(url: str, retries: int = 3) -> str:
    """Fetch URL, return decoded text. Handles gzip."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                # Handle gzip encoding
                if resp.info().get("Content-Encoding") == "gzip" or raw[:2] == b'\x1f\x8b':
                    import io
                    raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
                return raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return ""
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"WARN: Failed to fetch {url}: {e}")
            return ""
    return ""

def fetch_pdf_text(url: str) -> str:
    """Fetch PDF and extract text using pypdf."""
    try:
        import io
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp:
            pdf_bytes = resp.read()
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            pages  = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages).strip()
        except ImportError:
            # pypdf not available — store raw bytes note
            print("WARN: pypdf not available, storing placeholder")
            return "[PDF text extraction requires pypdf]"
    except Exception as e:
        print(f"WARN: PDF fetch failed for {url}: {e}")
        return ""

# ── S3 upload helper ──────────────────────────────────────────────────────────
def upload_doc(doc: dict, prefix: str) -> None:
    """Upload a document dict as gzipped JSON to S3 raw bucket."""
    key  = f"{prefix}/{doc['doc_id']}.json.gz"
    body = gzip.compress(json.dumps(doc, ensure_ascii=False).encode("utf-8"))
    s3.put_object(Bucket=RAW_BUCKET, Key=key, Body=body,
                  ContentEncoding="gzip", ContentType="application/json")
    print(f"  Uploaded: {key}")

def doc_id(source: str, entity: str, doc_date: str, doc_type: str) -> str:
    return hashlib.sha256(
        f"{source}|{entity}|{doc_date}|{doc_type}".encode()
    ).hexdigest()

# ── Already-ingested tracker ──────────────────────────────────────────────────
def load_ingested_ids() -> set:
    """Load set of already-ingested doc_ids from S3 tracker."""
    key = "tracker/ingested_ids.json"
    try:
        obj = s3.get_object(Bucket=RAW_BUCKET, Key=key)
        return set(json.loads(obj["Body"].read()))
    except s3.exceptions.NoSuchKey:
        return set()
    except Exception:
        return set()

def save_ingested_ids(ids: set) -> None:
    body = json.dumps(list(ids)).encode("utf-8")
    s3.put_object(Bucket=RAW_BUCKET, Key="tracker/ingested_ids.json", Body=body)

# ── FOMC Calendar scraper ─────────────────────────────────────────────────────
def parse_fomc_calendar(html: str) -> list:
    """
    Parse federalreserve.gov/monetarypolicy/fomccalendars.htm
    Returns list of dicts: {meeting_date, doc_type, url, title}
    """
    results = []

    # Find all <a> tags linking to statements, minutes, transcripts
    # Pattern: href contains /monetarypolicy/files/ or /monetarypolicy/fomcminutes
    patterns = [
        # Statements: monetary20240131a1.htm or similar
        (r'href="(/monetarypolicy/files/monetary(\d{8})[^"]*\.htm)"',  "statement"),
        (r'href="(/monetarypolicy/files/monetary(\d{8})[^"]*\.pdf)"',  "statement"),
        # Minutes: fomcminutes20240131.htm
        (r'href="(/monetarypolicy/fomcminutes(\d{8})\.htm)"',          "minutes"),
        (r'href="(/monetarypolicy/files/fomcminutes(\d{8})\.pdf)"',    "minutes"),
        # Transcripts: fomctranscript20190130.pdf (released ~5yr later)
        (r'href="(/monetarypolicy/files/FOMC(\d{8})meeting\.pdf)"',    "transcript"),
        (r'href="(/monetarypolicy/files/fomctranscript(\d{8})\.pdf)"', "transcript"),
    ]

    for pattern, doc_type in patterns:
        for m in re.finditer(pattern, html, re.IGNORECASE):
            path      = m.group(1)
            date_str  = m.group(2)  # YYYYMMDD
            try:
                meeting_date = datetime.date(
                    int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
                )
            except ValueError:
                continue

            if meeting_date < start_dt:
                continue

            url = BASE_URL + path
            results.append({
                "meeting_date": meeting_date.isoformat(),
                "doc_type":     doc_type,
                "url":          url,
                "title":        f"FOMC {doc_type.capitalize()} — {meeting_date.strftime('%B %d, %Y')}",
            })

    # Deduplicate by (date, doc_type) — keep first occurrence
    seen = set()
    unique = []
    for r in results:
        key = (r["meeting_date"], r["doc_type"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique

# ── Speeches RSS parser ───────────────────────────────────────────────────────
def parse_speeches_rss(xml_text: str) -> list:
    """
    Parse Fed speeches RSS feed.
    Returns list of dicts: {date, title, speaker, url}
    """
    results = []
    try:
        root = ET.fromstring(xml_text)
        ns   = {"dc": "http://purl.org/dc/elements/1.1/"}

        for item in root.findall(".//item"):
            title_el = item.find("title")
            link_el  = item.find("link")
            date_el  = item.find("pubDate")
            creator  = item.find("dc:creator", ns)

            if title_el is None or link_el is None or date_el is None:
                continue

            title = (title_el.text or "").strip()
            url   = (link_el.text or "").strip()
            speaker = (creator.text or "Federal Reserve").strip() if creator is not None else "Federal Reserve"

            # Parse pubDate: "Mon, 13 Jan 2025 00:00:00 -0500"
            try:
                from email.utils import parsedate
                parsed = parsedate(date_el.text)
                if parsed is None:
                    continue
                speech_date = datetime.date(parsed[0], parsed[1], parsed[2])
            except Exception:
                continue

            if speech_date < start_dt:
                continue

            results.append({
                "date":    speech_date.isoformat(),
                "title":   title,
                "speaker": speaker,
                "url":     url,
            })
    except ET.ParseError as e:
        print(f"WARN: RSS parse error: {e}")

    return results

# ── Main ingestion ────────────────────────────────────────────────────────────
def main():
    print(f"[ingest_fedspeak] env={ENV}, start={START_DATE}, bucket={RAW_BUCKET}")

    ingested = load_ingested_ids()
    new_ids  = set()
    total    = 0

    # ── 1. FOMC Calendar: statements, minutes, transcripts ───────────────────
    print("\n── FOMC Calendar ──")
    html = fetch_url(CALENDAR_URL)
    if not html:
        print("WARN: Could not fetch FOMC calendar page")
    else:
        items = parse_fomc_calendar(html)
        print(f"  Found {len(items)} calendar documents since {START_DATE}")

        for item in items:
            did = doc_id("FEDSPEAK", "FOMC", item["meeting_date"], item["doc_type"])
            if did in ingested:
                print(f"  SKIP (already ingested): {item['doc_type']} {item['meeting_date']}")
                continue

            print(f"  Fetching {item['doc_type']} {item['meeting_date']}: {item['url']}")
            time.sleep(0.5)  # polite rate limiting

            # Fetch text — HTML or PDF
            if item["url"].endswith(".pdf"):
                text = fetch_pdf_text(item["url"])
            else:
                raw_html = fetch_url(item["url"])
                # Strip HTML tags for clean text
                text = re.sub(r'<[^>]+>', ' ', raw_html)
                text = re.sub(r'\s+', ' ', text).strip()

            if not text or len(text) < 100:
                print(f"  WARN: Empty/short text for {item['url']}, skipping")
                continue

            doc = {
                "doc_id":       did,
                "source":       "FEDSPEAK",
                "entity":       "FOMC",
                "doc_type":     item["doc_type"],
                "doc_date":     item["meeting_date"],
                "title":        item["title"],
                "url":          item["url"],
                "text":         text,
                "char_count":   len(text),
                "ingested_at":  datetime.datetime.utcnow().isoformat() + "Z",
            }

            year = item["meeting_date"][:4]
            upload_doc(doc, prefix=f"year={year}/doc_type={item['doc_type']}")
            new_ids.add(did)
            total += 1

    # ── 2. Speeches RSS ───────────────────────────────────────────────────────
    print("\n── Speeches RSS ──")
    rss_text = fetch_url(SPEECHES_RSS)
    if not rss_text:
        print("WARN: Could not fetch speeches RSS")
    else:
        speeches = parse_speeches_rss(rss_text)
        print(f"  Found {len(speeches)} speeches since {START_DATE}")

        for speech in speeches:
            did = doc_id("FEDSPEAK", speech["speaker"], speech["date"], "speech")
            if did in ingested:
                print(f"  SKIP: {speech['title'][:60]}")
                continue

            print(f"  Fetching speech: {speech['title'][:60]}")
            time.sleep(0.5)

            # Speeches are HTML pages
            if speech["url"].endswith(".pdf"):
                text = fetch_pdf_text(speech["url"])
            else:
                raw_html = fetch_url(speech["url"])
                text     = re.sub(r'<[^>]+>', ' ', raw_html)
                text     = re.sub(r'\s+', ' ', text).strip()

            if not text or len(text) < 100:
                print(f"  WARN: Empty text for {speech['url']}, skipping")
                continue

            doc = {
                "doc_id":       did,
                "source":       "FEDSPEAK",
                "entity":       speech["speaker"],
                "doc_type":     "speech",
                "doc_date":     speech["date"],
                "title":        speech["title"],
                "url":          speech["url"],
                "text":         text,
                "char_count":   len(text),
                "ingested_at":  datetime.datetime.utcnow().isoformat() + "Z",
            }

            year = speech["date"][:4]
            upload_doc(doc, prefix=f"year={year}/doc_type=speech")
            new_ids.add(did)
            total += 1

    # ── Save tracker ──────────────────────────────────────────────────────────
    if new_ids:
        save_ingested_ids(ingested | new_ids)

    print(f"\n[ingest_fedspeak] Done — {total} new documents ingested")

if __name__ == "__main__":
    main()