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

from utils.config import get_default_start, load_source_config

# ── Fix B: _arg function ──────────────────────────────────────────────────────
def _arg(name: str, default: str = "") -> str:
    for i, a in enumerate(sys.argv):
        if a == f"--{name}" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.getenv(name, default)

ENV     = _arg("ENVIRONMENT", "dev")
REGION  = "us-east-2"
ACCOUNT = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]

RAW_BUCKET = f"{ENV}-trade-insiders-raw-{ACCOUNT}"

# ── Fix D: S3 client with region ──────────────────────────────────────────────
s3 = boto3.client("s3", region_name=REGION)

HEADERS = {
    "User-Agent": "trade-platform/1.0 research@example.com",
    "Accept-Encoding": "gzip, deflate",
}
BASE_URL     = "https://data.sec.gov"   # submissions API
ARCHIVES_URL = "https://www.sec.gov"    # filing Archives (data.sec.gov returns 404 here)
DELAY_SEC    = 0.15   # 150ms between EDGAR requests — safely under 10 req/s

# Save tracker every N filings to survive mid-ticker timeouts
CHECKPOINT_EVERY = 50

# Dev companies: ticker → CIK mapping
# CIKs are zero-padded to 10 digits for EDGAR API calls
TICKER_CIKS = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "GOOGL": "0001652044",
    "AMZN":  "0001018724",
    "JPM":   "0000019617",
    "BAC":   "0000070858",
    "XOM":   "0000034088",
    "JNJ":   "0000200406",
    "WMT":   "0000104169",
    "CAT":   "0000018230",
    "PG":    "0000080424",
    "KO":    "0000021344",
    "DIS":   "0001744489",
}

# ── Fix C: parse_known_args ───────────────────────────────────────────────────
def _arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ENVIRONMENT",  default="dev")
    p.add_argument("--start-date",   default=None)
    p.add_argument("--ticker",       default=None,
                   help="Process single ticker only (for testing)")
    return p.parse_known_args()[0]

args        = _arg_parser()
_config     = load_source_config("insiders")
START_DATE  = args.start_date or get_default_start(_config)
start_dt    = datetime.date.fromisoformat(START_DATE)
ONLY_TICKER = args.ticker

# ── HTTP helper ───────────────────────────────────────────────────────────────
def fetch_url(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                ct  = resp.headers.get("Content-Type", "")
                if "gzip" in ct or raw[:2] == b'\x1f\x8b':
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
            print(f"  WARN: fetch failed {url}: {e}")
            return ""
    return ""

# ── Tracker helpers ───────────────────────────────────────────────────────────
def load_tracker(ticker: str) -> dict:
    key = f"tracker/{ticker}.json"
    try:
        obj = s3.get_object(Bucket=RAW_BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return {"fetched_accessions": []}

def save_tracker(ticker: str, tracker: dict) -> None:
    key = f"tracker/{ticker}.json"
    s3.put_object(
        Bucket=RAW_BUCKET,
        Key=key,
        Body=json.dumps(tracker).encode("utf-8")
    )

# ── Fetch Form 4 filings list from submissions ────────────────────────────────
def get_form4_filings(cik: str) -> list:
    """
    Fetch submissions index for a company, return list of Form 4 filings.
    Each entry: {accession, filing_date, form_type}
    """
    url  = f"{BASE_URL}/submissions/CIK{cik}.json"
    text = fetch_url(url)
    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    filings = []
    recent  = data.get("filings", {}).get("recent", {})

    forms      = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates      = recent.get("filingDate", [])

    for form, acc, date in zip(forms, accessions, dates):
        if form != "4":
            continue
        try:
            filing_date = datetime.date.fromisoformat(date)
        except ValueError:
            continue
        if filing_date < start_dt:
            continue
        filings.append({
            "accession":   acc.replace("-", ""),
            "filing_date": date,
            "form_type":   form,
        })

    # Handle pagination — companies with >1000 filings
    for extra_file in data.get("filings", {}).get("files", []):
        time.sleep(DELAY_SEC)
        extra_url  = f"{BASE_URL}/submissions/{extra_file['name']}"
        extra_text = fetch_url(extra_url)
        if not extra_text:
            continue
        try:
            extra_data = json.loads(extra_text)
        except json.JSONDecodeError:
            continue

        extra_forms      = extra_data.get("form", [])
        extra_accessions = extra_data.get("accessionNumber", [])
        extra_dates      = extra_data.get("filingDate", [])

        for form, acc, date in zip(extra_forms, extra_accessions, extra_dates):
            if form != "4":
                continue
            try:
                filing_date = datetime.date.fromisoformat(date)
            except ValueError:
                continue
            if filing_date < start_dt:
                continue
            filings.append({
                "accession":   acc.replace("-", ""),
                "filing_date": date,
                "form_type":   form,
            })

    return filings

# ── Fetch and parse Form 4 XML ────────────────────────────────────────────────
def fetch_form4_xml(cik: str, accession: str) -> str:
    """
    Fetch Form 4 XML by parsing the filing index HTML to discover the actual XML filename.
    EDGAR XML filenames vary per filer (form4.xml, wf-form4.xml, etc.).
    """
    acc_dashed = f"{accession[:10]}-{accession[10:12]}-{accession[12:]}"
    cik_int    = int(cik)
    base_path  = f"{ARCHIVES_URL}/Archives/edgar/data/{cik_int}/{accession}"

    # Step 1: Parse index HTML to find the actual XML filename
    time.sleep(DELAY_SEC)
    index_html = fetch_url(f"{base_path}/{acc_dashed}-index.html")

    if index_html:
        for link in re.findall(r'href="([^"]+\.xml)"', index_html):
            if link.startswith("http"):
                xml_url = link
            elif link.startswith("/"):
                xml_url = f"{ARCHIVES_URL}{link}"
            else:
                xml_url = f"{base_path}/{link}"
            time.sleep(DELAY_SEC)
            xml_text = fetch_url(xml_url)
            if xml_text and "<ownershipDocument" in xml_text:
                return xml_text

    # Step 2: Fallback — try common filenames directly
    for filename in ["form4.xml", "wf-form4.xml", f"{acc_dashed}.xml"]:
        time.sleep(DELAY_SEC)
        xml_text = fetch_url(f"{base_path}/{filename}")
        if xml_text and "<ownershipDocument" in xml_text:
            return xml_text

    return ""

# ── Parse Form 4 XML → transactions ──────────────────────────────────────────
def parse_form4(xml_text: str, ticker: str, accession: str) -> list:
    """
    Parse Form 4 XML, extract non-derivative transactions (Table I only).
    Returns list of transaction dicts.
    """
    transactions = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"  WARN: XML parse error for {accession}: {e}")
        return []

    # Extract filer info
    filer_name = ""
    filer_role = ""

    reporting_owner = root.find(".//reportingOwner")
    if reporting_owner is not None:
        name_el = reporting_owner.find(".//rptOwnerName")
        if name_el is not None:
            filer_name = (name_el.text or "").strip()

        roles   = []
        role_el = reporting_owner.find(".//reportingOwnerRelationship")
        if role_el is not None:
            role_map = {
                "isDirector":        "Director",
                "isOfficer":         None,   # use officerTitle
                "isTenPercentOwner": "10% Owner",
                "isOther":           None,   # use otherText
            }
            for tag, label in role_map.items():
                el = role_el.find(tag)
                if el is not None and el.text and el.text.strip() in ("1", "true"):
                    if label:
                        roles.append(label)
                    elif tag == "isOfficer":
                        title_el = role_el.find("officerTitle")
                        if title_el is not None and title_el.text:
                            roles.append(title_el.text.strip())
                    elif tag == "isOther":
                        other_el = role_el.find("otherText")
                        if other_el is not None and other_el.text:
                            roles.append(other_el.text.strip())

        filer_role = ", ".join(roles) if roles else "Unknown"

    # Non-derivative transactions (Table I)
    for txn in root.findall(".//nonDerivativeTransaction"):
        try:
            def get_text(tag: str) -> str:
                el = txn.find(f".//{tag}")
                return (el.text or "").strip() if el is not None else ""

            transaction_date = get_text("transactionDate/value") or \
                               get_text("transactionDate")
            transaction_code = get_text("transactionCode")
            shares_str       = get_text("transactionShares/value") or \
                               get_text("transactionShares")
            price_str        = get_text("transactionPricePerShare/value") or \
                               get_text("transactionPricePerShare")
            owned_after_str  = get_text("sharesOwnedFollowingTransaction/value") or \
                               get_text("sharesOwnedFollowingTransaction")
            ownership_type   = get_text("directOrIndirectOwnership/value") or \
                               get_text("directOrIndirectOwnership")

            if not transaction_date or not shares_str:
                continue

            shares      = float(shares_str.replace(",", "")) if shares_str else 0.0
            price       = float(price_str.replace(",", ""))  if price_str  else 0.0
            owned_after = float(owned_after_str.replace(",", "")) \
                          if owned_after_str else 0.0
            value_usd   = shares * price if price > 0 else 0.0

            filing_id = hashlib.sha256(
                f"{accession}|{filer_name}|{transaction_date}|{transaction_code}|{shares_str}"
                .encode()
            ).hexdigest()

            transactions.append({
                "filing_id":          filing_id,
                "accession":          accession,
                "ticker":             ticker,
                "filer_name":         filer_name,
                "filer_role":         filer_role,
                "transaction_date":   transaction_date,
                "transaction_type":   transaction_code,
                "shares":             shares,
                "price_per_share":    price,
                "value_usd":          value_usd,
                "ownership_type":     ownership_type or "D",
                "shares_owned_after": owned_after,
                "year":               int(transaction_date[:4]) if transaction_date else 0,
                "ingested_at":        datetime.datetime.utcnow().isoformat() + "Z",
            })

        except (ValueError, AttributeError) as e:
            print(f"  WARN: Could not parse transaction in {accession}: {e}")
            continue

    return transactions

# ── Upload to S3 ──────────────────────────────────────────────────────────────
def upload_filing(ticker: str, accession: str, transactions: list) -> None:
    year = transactions[0]["year"] if transactions else 0
    key  = f"year={year}/ticker={ticker}/{accession}.json.gz"
    body = gzip.compress(
        json.dumps(transactions, ensure_ascii=False).encode("utf-8")
    )
    s3.put_object(
        Bucket=RAW_BUCKET, Key=key, Body=body,
        ContentEncoding="gzip", ContentType="application/json"
    )

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[ingest_insiders] env={ENV}, start={START_DATE}, bucket={RAW_BUCKET}")

    configured_tickers = _config.get("tickers", [])

    if ONLY_TICKER:
        if ONLY_TICKER not in TICKER_CIKS:
            print(f"ERROR: {ONLY_TICKER} has no CIK mapping in "
                  f"TICKER_CIKS — add it before using --ticker")
            return
        tickers = {ONLY_TICKER: TICKER_CIKS[ONLY_TICKER]}
    else:
        tickers = {}
        for t in configured_tickers:
            if t not in TICKER_CIKS:
                print(f"WARN: {t} in insiders.yaml has no CIK mapping "
                      f"in TICKER_CIKS — skipping")
                continue
            tickers[t] = TICKER_CIKS[t]

    print(f"  Resolved tickers ({len(tickers)}): {sorted(tickers.keys())}")

    total_txns    = 0
    total_filings = 0

    for ticker, cik in tickers.items():
        print(f"\n── {ticker} (CIK: {cik}) ──")

        tracker = load_tracker(ticker)
        fetched = set(tracker.get("fetched_accessions", []))

        time.sleep(DELAY_SEC)
        filings = get_form4_filings(cik)
        print(f"  Found {len(filings)} Form 4 filings since {START_DATE}")

        new_filings = [f for f in filings if f["accession"] not in fetched]
        print(f"  {len(new_filings)} new (tracker has {len(fetched)} already fetched)")

        if not new_filings:
            print(f"  Nothing to do for {ticker}")
            continue

        ticker_txns    = 0
        ticker_filings = 0

        for i, filing in enumerate(new_filings):
            acc = filing["accession"]

            # Filing-level log so CloudWatch shows exactly where a timeout cuts off
            print(f"  [{i + 1}/{len(new_filings)}] {acc}  filed={filing['filing_date']}")

            xml_text = fetch_form4_xml(cik, acc)
            if not xml_text:
                print(f"    WARN: Could not fetch XML — marking attempted")
                fetched.add(acc)
            else:
                transactions = parse_form4(xml_text, ticker, acc)
                if not transactions:
                    print(f"    SKIP: No non-derivative transactions")
                    fetched.add(acc)
                else:
                    upload_filing(ticker, acc, transactions)
                    fetched.add(acc)
                    ticker_txns    += len(transactions)
                    ticker_filings += 1
                    print(f"    OK: {len(transactions)} txns uploaded")

            # Checkpoint every N filings — survives mid-ticker timeout on rerun
            if (i + 1) % CHECKPOINT_EVERY == 0:
                tracker["fetched_accessions"] = list(fetched)
                tracker["last_ingest"] = datetime.datetime.utcnow().isoformat() + "Z"
                save_tracker(ticker, tracker)
                print(f"  ── checkpoint at filing {i + 1}/{len(new_filings)} ──")

        total_txns    += ticker_txns
        total_filings += ticker_filings
        print(f"  {ticker} done: {ticker_txns} txns from {ticker_filings} filings")

        # Final save for this ticker
        tracker["fetched_accessions"] = list(fetched)
        tracker["last_ingest"] = datetime.datetime.utcnow().isoformat() + "Z"
        save_tracker(ticker, tracker)

    print(f"\n[ingest_insiders] Done — {total_txns} transactions from {total_filings} filings")

if __name__ == "__main__":
    main()