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
import boto3

# ── Fix B: _arg function ──────────────────────────────────────────────────────
def _arg(name: str, default: str = "") -> str:
    for i, a in enumerate(sys.argv):
        if a == f"--{name}" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return os.getenv(name, default)

ENV     = _arg("ENVIRONMENT", "dev")
REGION  = "us-east-2"
ACCOUNT = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]

RAW_BUCKET = f"{ENV}-trade-news-raw-{ACCOUNT}"

# ── Fix D: S3 + Secrets clients with region ───────────────────────────────────
s3  = boto3.client("s3",              region_name=REGION)
ssm = boto3.client("secretsmanager",  region_name=REGION)

# ── Config ────────────────────────────────────────────────────────────────────
def _arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ENVIRONMENT",  default="dev")
    p.add_argument("--backfill-days", type=int, default=None)
    # ── Fix C: parse_known_args ───────────────────────────────────────────────
    return p.parse_known_args()[0]

args         = _arg_parser()
BACKFILL_DAYS = args.backfill_days or (90 if ENV == "dev" else 90)

# Tickers to track — 7 dev companies + 4 ETF proxies
TICKERS = [
    # Dev companies
    "AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "BAC", "XOM",
    # ETF proxies
    "GLD", "USO", "TLT", "SPY",
]

# Publisher tier mapping
PUBLISHER_TIERS = {
    # Tier 1 — wire services / premium financial press
    "reuters":          1, "associated press": 1, "ap":               1,
    "bloomberg":        1, "financial times":  1, "wall street journal": 1,
    "wsj":              1, "dow jones":        1,
    # Tier 2 — established financial media
    "marketwatch":      2, "benzinga":         2, "barron's":         2,
    "cnbc":             2, "yahoo finance":    2, "investopedia":     2,
    "the street":       2, "thestreet":        2, "zacks":            2,
    "morningstar":      2,
    # Tier 3 — opinion / retail-focused
    "seeking alpha":    3, "the motley fool":  3, "motley fool":      3,
    "fool.com":         3, "gurufocus":        3, "stockanalysis":    3,
}

def get_publisher_tier(publisher_name: str) -> int:
    name = publisher_name.lower().strip()
    for key, tier in PUBLISHER_TIERS.items():
        if key in name:
            return tier
    return 3  # default to tier 3 for unknown publishers

BASE_URL = "https://api.polygon.io"

# ── API key from Secrets Manager ──────────────────────────────────────────────
def get_api_key() -> str:
    secret_name = f"trade-platform/{ENV}/polygon-api-key"
    try:
        resp = ssm.get_secret_value(SecretId=secret_name)
        return resp["SecretString"].strip()
    except Exception as e:
        # Fallback to env var for local testing
        key = os.getenv("POLYGON_API_KEY", "")
        if not key:
            raise RuntimeError(f"Cannot get Polygon API key: {e}")
        return key

# ── HTTP helper ───────────────────────────────────────────────────────────────
def fetch_json(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "trade-platform/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited — wait longer
                wait = 60 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if e.code == 403:
                print(f"  WARN: 403 Forbidden — check API key or plan limits")
                return {}
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"  WARN: fetch failed for {url}: {e}")
            return {}
    return {}

# ── Tracker helpers ───────────────────────────────────────────────────────────
def load_ingested_ids() -> set:
    key = "tracker/ingested_ids.json"
    try:
        obj = s3.get_object(Bucket=RAW_BUCKET, Key=key)
        return set(json.loads(obj["Body"].read()))
    except Exception:
        return set()

def save_ingested_ids(ids: set) -> None:
    s3.put_object(
        Bucket=RAW_BUCKET,
        Key="tracker/ingested_ids.json",
        Body=json.dumps(list(ids)).encode("utf-8")
    )

# ── S3 upload ─────────────────────────────────────────────────────────────────
def upload_articles(articles: list, ticker: str, year: int, week: str) -> None:
    if not articles:
        return
    key  = f"year={year}/ticker={ticker}/week={week}/articles.json.gz"
    body = gzip.compress(
        json.dumps(articles, ensure_ascii=False).encode("utf-8")
    )
    s3.put_object(
        Bucket=RAW_BUCKET, Key=key, Body=body,
        ContentEncoding="gzip", ContentType="application/json"
    )
    print(f"  Uploaded {len(articles)} articles → {key}")

# ── Fetch news for one ticker ─────────────────────────────────────────────────
def fetch_ticker_news(
    ticker: str,
    api_key: str,
    published_gte: str,
    published_lte: str,
) -> list:
    """
    Fetch all news articles for a ticker in the given date range.
    Handles pagination via next_url.
    Rate limit: 5 req/min on free tier — 12s sleep between calls.
    """
    articles = []
    url = (
        f"{BASE_URL}/v2/reference/news"
        f"?ticker={ticker}"
        f"&published_utc.gte={published_gte}"
        f"&published_utc.lte={published_lte}"
        f"&order=desc"
        f"&limit=1000"
        f"&apiKey={api_key}"
    )

    page = 0
    while url:
        page += 1
        print(f"    Page {page}: {url[:80]}...")
        data = fetch_json(url)

        if not data or data.get("status") not in ("OK", "DELAYED"):
            print(f"    WARN: unexpected status {data.get('status')} — stopping")
            break

        results = data.get("results", [])
        articles.extend(results)
        print(f"    Got {len(results)} articles (total: {len(articles)})")

        # Pagination
        next_url = data.get("next_url")
        if next_url:
            url = f"{next_url}&apiKey={api_key}"
            time.sleep(12)  # 5 req/min = 12s between calls on free tier
        else:
            break

    return articles

# ── Process raw article → clean dict ─────────────────────────────────────────
def process_article(raw: dict, primary_ticker: str) -> dict:
    publisher_name = raw.get("publisher", {}).get("name", "Unknown")

    # Extract per-ticker sentiment from insights array
    sentiment         = "neutral"
    sentiment_reasoning = ""
    for insight in raw.get("insights", []):
        if insight.get("ticker") == primary_ticker:
            sentiment           = insight.get("sentiment", "neutral")
            sentiment_reasoning = insight.get("sentiment_reasoning", "")[:500]
            break

    # Parse published date for partitioning
    published_utc = raw.get("published_utc", "")
    try:
        pub_dt = datetime.datetime.fromisoformat(
            published_utc.replace("Z", "+00:00")
        )
        year    = pub_dt.year
        week    = pub_dt.strftime("%Y-W%W")
    except Exception:
        now  = datetime.datetime.utcnow()
        year = now.year
        week = now.strftime("%Y-W%W")

    return {
        "article_id":           raw.get("id", ""),
        "headline":             raw.get("title", ""),
        "description":          raw.get("description", "")[:1000],
        "author":               raw.get("author", ""),
        "published_at":         published_utc,
        "publisher":            publisher_name,
        "publisher_tier":       get_publisher_tier(publisher_name),
        "primary_ticker":       primary_ticker,
        "tickers":              json.dumps(raw.get("tickers", [])),
        "sentiment":            sentiment,
        "sentiment_reasoning":  sentiment_reasoning,
        "keywords":             json.dumps(raw.get("keywords", [])),
        "article_url":          raw.get("article_url", ""),
        "year":                 year,
        "week":                 week,
        "ingested_at":          datetime.datetime.utcnow().isoformat() + "Z",
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[ingest_news] env={ENV}, bucket={RAW_BUCKET}")
    print(f"  Tickers: {TICKERS}")
    print(f"  Backfill days: {BACKFILL_DAYS}")

    api_key = get_api_key()
    ingested = load_ingested_ids()
    new_ids  = set()
    total    = 0

    # Date range — backfill or last 7 days for weekly incremental
    end_dt   = datetime.datetime.utcnow()
    start_dt = end_dt - datetime.timedelta(days=BACKFILL_DAYS)
    published_gte = start_dt.strftime("%Y-%m-%dT00:00:00Z")
    published_lte = end_dt.strftime("%Y-%m-%dT23:59:59Z")

    print(f"  Date range: {published_gte} → {published_lte}")

    for ticker in TICKERS:
        print(f"\n── {ticker} ──")
        time.sleep(12)  # rate limit between tickers

        raw_articles = fetch_ticker_news(
            ticker, api_key, published_gte, published_lte
        )
        print(f"  Fetched {len(raw_articles)} raw articles")

        # Process and dedup
        new_articles = []
        for raw in raw_articles:
            article_id = raw.get("id", "")
            # Dedup key: article_id + primary_ticker (same article can appear
            # for multiple tickers — store once per ticker for sentiment)
            dedup_key = f"{article_id}:{ticker}"
            if dedup_key in ingested:
                continue

            processed = process_article(raw, ticker)
            new_articles.append(processed)
            new_ids.add(dedup_key)

        print(f"  {len(new_articles)} new articles after dedup")

        if new_articles:
            # Group by year+week for S3 partitioning
            partitions: dict = {}
            for a in new_articles:
                key = (a["year"], a["week"])
                partitions.setdefault(key, []).append(a)

            for (year, week), articles in sorted(partitions.items()):
                upload_articles(articles, ticker, year, week)
                total += len(articles)

    # Save tracker
    if new_ids:
        save_ingested_ids(ingested | new_ids)

    print(f"\n[ingest_news] Done — {total} new articles ingested")

if __name__ == "__main__":
    main()