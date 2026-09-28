"""
verify_pipeline.py — End-to-end pipeline verification for any source.

Usage:
    python verify_pipeline.py --source fred
    python verify_pipeline.py --source fred --run-crawler
    python verify_pipeline.py --source fred --reset          # clears S3 + watermarks
    python verify_pipeline.py --source fred --reset --fire   # reset + fire trigger

Flags:
    --run-crawler   Trigger Glue crawler and wait 60s before Athena query
    --reset         Clear raw bucket, processed bucket, watermarks/trackers
    --fire          Fire the manual trigger after reset (use with --reset)

Checks:
    1. Ingestion job — last run state + timestamp
    2. ETL job — last run state + timestamp
    3. S3 raw — recent files written
    4. S3 processed — recent files written
    5. DynamoDB watermarks (non-SEC sources)
    6. SEC tracker (SEC only)
    7. Crawlers — last run state
    8. Athena — simple query to verify data is queryable
"""

import argparse
import boto3
import json
from datetime import datetime, timezone, timedelta

REGION     = "us-east-2"
ACCOUNT_ID = "<ACCOUNT_ID>"
ENV        = "dev"

glue    = boto3.client("glue",     region_name=REGION)
s3      = boto3.client("s3",       region_name=REGION)
athena  = boto3.client("athena",   region_name=REGION)
dynamo  = boto3.client("dynamodb", region_name=REGION)

ATHENA_RESULTS = f"s3://dev-trade-athena-results-{ACCOUNT_ID}/"

# ── Source config ──────────────────────────────────────────────────────────
SOURCE_CONFIG = {
    "fred": {
        "ingest_job":    f"{ENV}-trade-fred-ingestion",
        "etl_job":       f"{ENV}-trade-fred-etl",
        "etl_trigger":   f"{ENV}-trade-fred-etl-trigger",
        "raw_bucket":    f"{ENV}-trade-fred-raw-{ACCOUNT_ID}",
        "proc_bucket":   f"{ENV}-trade-fred-processed-{ACCOUNT_ID}",
        "proc_prefix":   "economic_indicators/source=FRED/",
        "crawler":       f"{ENV}-trade-fred-processed-crawler",
        "athena_db":     f"{ENV}_trade_fred_processed",
        "athena_table":  "economic_indicators",
        "athena_query":  "SELECT indicator_id, date, value FROM economic_indicators LIMIT 5",
        "tracker":       None,  # uses DynamoDB watermarks
        "watermark_source": "fred",
    },
    "worldbank": {
        "ingest_job":    f"{ENV}-trade-worldbank-ingestion",
        "etl_job":       f"{ENV}-trade-worldbank-etl",
        "etl_trigger":   f"{ENV}-trade-worldbank-etl-trigger",
        "raw_bucket":    f"{ENV}-trade-worldbank-raw-{ACCOUNT_ID}",
        "proc_bucket":   f"{ENV}-trade-worldbank-processed-{ACCOUNT_ID}",
        "proc_prefix":   "economic_indicators/source=WORLDBANK/",
        "crawler":       f"{ENV}-trade-worldbank-processed-crawler",
        "athena_db":     f"{ENV}_trade_worldbank_processed",
        "athena_table":  "economic_indicators",
        "athena_query":  "SELECT indicator_id, date, value FROM economic_indicators LIMIT 5",
        "tracker":       None,
        "watermark_source": "worldbank",
    },
    "yfinance": {
        "ingest_job":    f"{ENV}-trade-yfinance-ingestion",
        "etl_job":       f"{ENV}-trade-yfinance-etl",
        "etl_trigger":   f"{ENV}-trade-yfinance-etl-trigger",
        "raw_bucket":    f"{ENV}-trade-yfinance-raw-{ACCOUNT_ID}",
        "proc_bucket":   f"{ENV}-trade-yfinance-processed-{ACCOUNT_ID}",
        "proc_prefix":   "market_prices/",
        "crawler":       f"{ENV}-trade-yfinance-processed-crawler",
        "athena_db":     f"{ENV}_trade_yfinance_processed",
        "athena_table":  "market_prices",
        "athena_query":  "SELECT ticker, date, close FROM market_prices LIMIT 5",
        "tracker":       None,
        "watermark_source": "yfinance",
    },
    "wikipedia": {
        "ingest_job":    f"{ENV}-trade-wikipedia-ingestion",
        "etl_job":       f"{ENV}-trade-wikipedia-etl",
        "etl_trigger":   f"{ENV}-trade-wikipedia-etl-trigger",
        "raw_bucket":    f"{ENV}-trade-wikipedia-raw-{ACCOUNT_ID}",
        "proc_bucket":   f"{ENV}-trade-wikipedia-processed-{ACCOUNT_ID}",
        "proc_prefix":   "documents/source=WIKIPEDIA/",
        "crawler":       f"{ENV}-trade-wikipedia-processed-crawler",
        "athena_db":     f"{ENV}_trade_wikipedia_processed",
        "athena_table":  "documents",
        "athena_query":  "SELECT entity, doc_date, char_count FROM documents LIMIT 5",
        "tracker":       None,
        "watermark_source": "wikipedia",
    },
    "sec": {
        "ingest_job":    f"{ENV}-trade-sec-ingestion",
        "etl_job":       f"{ENV}-trade-sec-etl",
        "etl_trigger":   f"{ENV}-trade-sec-etl-trigger",
        "raw_bucket":    f"{ENV}-trade-sec-raw-{ACCOUNT_ID}",
        "proc_bucket":   f"{ENV}-trade-sec-processed-{ACCOUNT_ID}",
        "proc_prefix":   "documents/source=EDGAR/",
        "crawler":       f"{ENV}-trade-sec-processed-crawler",
        "athena_db":     f"{ENV}_trade_sec_processed",
        "athena_table":  "documents",
        "athena_query":  "SELECT entity, doc_date, char_count FROM documents LIMIT 5",
        "tracker":       f"{ENV}-trade-sec-raw-{ACCOUNT_ID}",
        "watermark_source": None,  # uses S3 tracker
        "extra_jobs": {
            f"{ENV}-trade-sec-prose-etl": "etl_sec_prose",
            f"{ENV}-trade-sec-embed-etl": "etl_embed",
        },
    },
    "fedspeak": {
        "ingest_job":    f"{ENV}-trade-fedspeak-ingestion",
        "etl_job":       f"{ENV}-trade-fedspeak-etl",
        "etl_trigger":   f"{ENV}-trade-fedspeak-etl-trigger",
        "raw_bucket":    f"{ENV}-trade-fedspeak-raw-{ACCOUNT_ID}",
        "proc_bucket":   f"{ENV}-trade-fedspeak-processed-{ACCOUNT_ID}",
        "proc_prefix":   "documents/",
        "crawler":       f"{ENV}-trade-fedspeak-processed-crawler",
        "athena_db":     f"{ENV}_trade_fedspeak_processed",
        "athena_table":  "documents",
        "athena_query":  (
            "SELECT COUNT(*) as cnt, doc_type, year FROM documents"
            " WHERE source='FEDSPEAK'"
            " GROUP BY doc_type, year"
            " ORDER BY year DESC, doc_type"
        ),
        "tracker":       None,
        "watermark_source": None,  # uses S3 tracker/ingested_ids.json
    },
    "news": {
        "ingest_job":    f"{ENV}-trade-news-ingestion",
        "etl_job":       f"{ENV}-trade-news-etl",
        "etl_trigger":   f"{ENV}-trade-news-etl-trigger",
        "raw_bucket":    f"{ENV}-trade-news-raw-{ACCOUNT_ID}",
        "proc_bucket":   f"{ENV}-trade-news-processed-{ACCOUNT_ID}",
        "proc_prefix":   "news/",
        "crawler":       f"{ENV}-trade-news-processed-crawler",
        "athena_db":     f"{ENV}_trade_news_processed",
        "athena_table":  "news",
        "athena_query":  (
            "SELECT primary_ticker, COUNT(*) as cnt,"
            " SUM(CASE WHEN sentiment='positive' THEN 1 ELSE 0 END) as positive,"
            " SUM(CASE WHEN sentiment='negative' THEN 1 ELSE 0 END) as negative,"
            " SUM(CASE WHEN sentiment='neutral' THEN 1 ELSE 0 END) as neutral"
            " FROM news"
            " GROUP BY primary_ticker"
            " ORDER BY cnt DESC"
        ),
        "tracker":       None,
        "watermark_source": None,  # uses S3 tracker/ingested_ids.json
    },
    "insiders": {
        "ingest_job":    f"{ENV}-trade-insiders-ingestion",
        "etl_job":       f"{ENV}-trade-insiders-etl",
        "etl_trigger":   f"{ENV}-trade-insiders-etl-trigger",
        "raw_bucket":    f"{ENV}-trade-insiders-raw-{ACCOUNT_ID}",
        "proc_bucket":   f"{ENV}-trade-insiders-processed-{ACCOUNT_ID}",
        "proc_prefix":   "insider_trades/",
        "crawler":       f"{ENV}-trade-insiders-processed-crawler",
        "athena_db":     f"{ENV}_trade_insiders_processed",
        "athena_table":  "insider_trades",
        "athena_query":  (
            "SELECT ticker, transaction_type,"
            " COUNT(*) as cnt,"
            " SUM(value_usd) as total_value,"
            " COUNT(DISTINCT filer_name) as unique_insiders"
            " FROM insider_trades"
            " GROUP BY ticker, transaction_type"
            " ORDER BY total_value DESC"
            " LIMIT 20"
        ),
        "tracker":          None,
        "watermark_source": None,   # uses per-ticker S3 tracker files
        "ticker_tracker":   True,   # tracker/{ticker}.json, one per ticker
    },
    "companies": {
        "ingest_job":    f"{ENV}-trade-yfinance-ingestion",
        "etl_job":       f"{ENV}-trade-companies-etl",
        "etl_trigger":   f"{ENV}-trade-companies-etl-trigger",
        "raw_bucket":    None,      # derived source — no separate raw bucket
        "proc_bucket":   f"{ENV}-trade-yfinance-processed-{ACCOUNT_ID}",
        "proc_prefix":   "companies/",
        "crawler":       f"{ENV}-trade-yfinance-processed-crawler",
        "athena_db":     f"{ENV}_trade_yfinance_processed",
        "athena_table":  "companies",
        "athena_query":  (
            "SELECT ticker, company_name, sector, market_cap "
            "FROM companies "
            "ORDER BY market_cap DESC NULLS LAST "
            "LIMIT 10"
        ),
        "tracker":          None,
        "watermark_source": None,   # full-refresh — no watermark
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────

def check_mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def get_last_job_run(job_name: str) -> dict:
    try:
        resp = glue.get_job_runs(JobName=job_name, MaxResults=1)
        runs = resp.get("JobRuns", [])
        if not runs:
            return {"State": "NEVER_RUN", "StartedOn": None, "ErrorMessage": None}
        r = runs[0]
        return {
            "State":        r.get("JobRunState"),
            "StartedOn":    r.get("StartedOn"),
            "CompletedOn":  r.get("CompletedOn"),
            "ErrorMessage": r.get("ErrorMessage", ""),
        }
    except Exception as e:
        return {"State": "ERROR", "ErrorMessage": str(e)}


def get_recent_s3_files(bucket: str, prefix: str, hours: int = 24) -> list:
    """List S3 files modified in the last N hours."""
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        paginator = s3.get_paginator("list_objects_v2")
        recent = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["LastModified"] >= cutoff:
                    recent.append({
                        "key":      obj["Key"],
                        "size_kb":  obj["Size"] // 1024,
                        "modified": obj["LastModified"].strftime("%Y-%m-%d %H:%M UTC"),
                    })
        return recent
    except Exception as e:
        return [{"error": str(e)}]


def get_watermarks(source: str) -> list:
    try:
        resp = dynamo.query(
            TableName=f"trade-platform-{ENV}-watermarks",
            KeyConditionExpression="source_name = :s",
            ExpressionAttributeValues={":s": {"S": source}},
        )
        items = resp.get("Items", [])
        return [
            {
                "dataset":    i.get("dataset_name", {}).get("S", ""),
                "last_date":  i.get("last_ingested_period", {}).get("S", ""),
                "status":     i.get("last_run_status", {}).get("S", ""),
                "timestamp":  i.get("last_run_timestamp", {}).get("S", ""),
            }
            for i in items
        ]
    except Exception as e:
        return [{"error": str(e)}]


def get_sec_tracker(bucket: str, ticker: str = "AAPL") -> dict:
    try:
        obj = s3.get_object(
            Bucket=bucket,
            Key=f"tracker/sec_{ticker}_tracker.json"
        )
        t = json.loads(obj["Body"].read())
        return {
            "total_fetched":     t.get("total_fetched", 0),
            "total_transformed": t.get("total_transformed", 0),
            "total_prose":       t.get("total_prose", 0),
            "last_ingest":       t.get("last_ingest", ""),
            "last_etl":          t.get("last_etl", ""),
            "last_prose_etl":    t.get("last_prose_etl", ""),
        }
    except Exception as e:
        return {"error": str(e)}


def get_fedspeak_tracker(bucket: str) -> dict:
    try:
        obj = s3.get_object(Bucket=bucket, Key="tracker/ingested_ids.json")
        data = json.loads(obj["Body"].read())
        count = len(data) if isinstance(data, list) else len(data)
        return {"count": count, "found": True}
    except s3.exceptions.NoSuchKey:
        return {"found": False, "error": "tracker/ingested_ids.json not found"}
    except Exception as e:
        return {"found": False, "error": str(e)}


def get_insiders_tracker(bucket: str) -> dict:
    """Read all per-ticker tracker/{ticker}.json files, sum fetched_accessions counts."""
    try:
        paginator = s3.get_paginator("list_objects_v2")
        total_accessions = 0
        tickers_found    = 0
        for page in paginator.paginate(Bucket=bucket, Prefix="tracker/"):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith(".json"):
                    continue
                body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                data = json.loads(body)
                total_accessions += len(data.get("fetched_accessions", []))
                tickers_found    += 1
        if tickers_found == 0:
            return {"found": False, "error": "No tracker files found under tracker/"}
        return {"found": True, "total": total_accessions, "tickers": tickers_found}
    except Exception as e:
        return {"found": False, "error": str(e)}


def get_crawler_status(crawler_name: str) -> dict:
    try:
        resp = glue.get_crawler(Name=crawler_name)
        c = resp["Crawler"]
        last = c.get("LastCrawl", {})
        return {
            "State":      c.get("State"),
            "LastStatus": last.get("Status"),
            "LastRun":    last.get("StartTime", "never").strftime("%Y-%m-%d %H:%M UTC")
                          if hasattr(last.get("StartTime"), "strftime") else "never",
            "Tables":     last.get("Summary", ""),
        }
    except Exception as e:
        return {"error": str(e)}


def run_crawler(crawler_name: str) -> str:
    try:
        glue.start_crawler(Name=crawler_name)
        return "Started"
    except glue.exceptions.CrawlerRunningException:
        return "Already running"
    except Exception as e:
        return f"Error: {e}"


def run_athena_query(query: str, database: str) -> list:
    try:
        import time
        resp = athena.start_query_execution(
            QueryString=query,
            QueryExecutionContext={"Database": database},
            ResultConfiguration={"OutputLocation": ATHENA_RESULTS},
        )
        exec_id = resp["QueryExecutionId"]

        for _ in range(30):
            status = athena.get_query_execution(
                QueryExecutionId=exec_id
            )["QueryExecution"]["Status"]["State"]
            if status == "SUCCEEDED":
                break
            elif status in ("FAILED", "CANCELLED"):
                return [{"error": f"Query {status}"}]
            time.sleep(2)

        results = athena.get_query_results(QueryExecutionId=exec_id)
        rows = results["ResultSet"]["Rows"]
        if len(rows) <= 1:
            return [{"result": "No rows returned"}]

        headers = [c["VarCharValue"] for c in rows[0]["Data"]]
        return [
            dict(zip(headers, [c.get("VarCharValue", "") for c in row["Data"]]))
            for row in rows[1:]
        ]
    except Exception as e:
        return [{"error": str(e)}]


# ── Reset ─────────────────────────────────────────────────────────────────

def delete_s3_prefix(bucket: str, prefix: str = "") -> int:
    """Delete all objects under a prefix. Returns count deleted."""
    paginator = s3.get_paginator("list_objects_v2")
    deleted = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = page.get("Contents", [])
        if not objects:
            continue
        s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": o["Key"]} for o in objects]},
        )
        deleted += len(objects)
    return deleted


def delete_watermarks(source: str) -> int:
    """Delete all DynamoDB watermarks for a source."""
    resp = dynamo.query(
        TableName=f"trade-platform-{ENV}-watermarks",
        KeyConditionExpression="source_name = :s",
        ExpressionAttributeValues={":s": {"S": source}},
    )
    items = resp.get("Items", [])
    for item in items:
        dynamo.delete_item(
            TableName=f"trade-platform-{ENV}-watermarks",
            Key={
                "source_name":  item["source_name"],
                "dataset_name": item["dataset_name"],
            },
        )
    return len(items)


def delete_sec_trackers(bucket: str) -> int:
    """Delete all SEC tracker files from S3."""
    return delete_s3_prefix(bucket, "tracker/")


def fire_trigger(source: str) -> str:
    """Fire the manual ON_DEMAND trigger for a source."""
    trigger_name = f"{ENV}-trade-{source}-manual-trigger"
    try:
        glue.start_trigger(Name=trigger_name)
        return f"✅ Fired trigger: {trigger_name}"
    except Exception as e:
        return f"❌ Failed to fire trigger: {e}"


def reset(source: str, cfg: dict):
    """Clear raw bucket, processed bucket, and watermarks/trackers."""
    print(f"\n{'='*60}")
    print(f"  RESET — {source.upper()}")
    print(f"{'='*60}\n")

    # Clear raw bucket
    if cfg.get("raw_bucket"):
        print(f"Clearing raw bucket: {cfg['raw_bucket']}")
        # For SEC, preserve the latest/ prefix (needed by ETL) but clear year= and tracker/
        if source == "sec":
            n = delete_s3_prefix(cfg["raw_bucket"], "year=")
            print(f"  Deleted {n} files from year= prefix")
            n = delete_sec_trackers(cfg["raw_bucket"])
            print(f"  Deleted {n} tracker files")
        else:
            n = delete_s3_prefix(cfg["raw_bucket"])
            print(f"  Deleted {n} files")
    else:
        print("   RAW BUCKET: N/A (derived source — no raw bucket)")

    # Clear processed bucket
    print(f"Clearing processed bucket: {cfg['proc_bucket']}")
    n = delete_s3_prefix(cfg["proc_bucket"])
    print(f"  Deleted {n} files")

    # Clear watermarks or SEC trackers
    if cfg.get("watermark_source"):
        print(f"Clearing DynamoDB watermarks for: {cfg['watermark_source']}")
        n = delete_watermarks(cfg["watermark_source"])
        print(f"  Deleted {n} watermark items")
    elif cfg.get("tracker"):
        print(f"Clearing SEC trackers from: {cfg['tracker']}")
        n = delete_sec_trackers(cfg["tracker"])
        print(f"  Deleted {n} tracker files")

    print(f"\n✅ Reset complete for {source.upper()}\n")


# ── Main verification ──────────────────────────────────────────────────────

def verify(source: str, run_crawler_flag: bool = False):
    cfg = SOURCE_CONFIG.get(source)
    if not cfg:
        print(f"Unknown source: {source}. Choose from: {list(SOURCE_CONFIG.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"  PIPELINE VERIFICATION — {source.upper()}")
    print(f"{'='*60}\n")

    # ── 1. Ingestion job ──────────────────────────────────────────────────
    print("1. INGESTION JOB")
    run = get_last_job_run(cfg["ingest_job"])
    ok  = run["State"] == "SUCCEEDED"
    print(f"   {check_mark(ok)} State:    {run['State']}")
    if run.get("StartedOn"):
        print(f"      Started:  {run['StartedOn'].strftime('%Y-%m-%d %H:%M UTC')}")
    if run.get("CompletedOn"):
        print(f"      Completed:{run['CompletedOn'].strftime('%Y-%m-%d %H:%M UTC')}")
    if not ok and run.get("ErrorMessage"):
        print(f"      Error:    {run['ErrorMessage'][:200]}")

    # ── 2. ETL job ────────────────────────────────────────────────────────
    print("\n2. ETL JOB")
    etl = get_last_job_run(cfg["etl_job"])
    ok  = etl["State"] == "SUCCEEDED"
    print(f"   {check_mark(ok)} State:    {etl['State']}")
    if etl.get("StartedOn"):
        print(f"      Started:  {etl['StartedOn'].strftime('%Y-%m-%d %H:%M UTC')}")
    if not ok and etl.get("ErrorMessage"):
        print(f"      Error:    {etl['ErrorMessage'][:200]}")

    # Extra jobs for SEC
    if cfg.get("extra_jobs"):
        for job_name, label in cfg["extra_jobs"].items():
            r = get_last_job_run(job_name)
            ok = r["State"] == "SUCCEEDED"
            print(f"   {check_mark(ok)} {label}: {r['State']}")
            if not ok and r.get("ErrorMessage"):
                print(f"      Error: {r['ErrorMessage'][:200]}")

    # ── 3. S3 Raw — recent files ──────────────────────────────────────────
    print("\n3. S3 RAW — files written in last 24h")
    if cfg.get("raw_bucket"):
        raw_files = get_recent_s3_files(cfg["raw_bucket"], "", hours=24)
        if raw_files and "error" not in raw_files[0]:
            print(f"   {check_mark(len(raw_files) > 0)} {len(raw_files)} new files")
            for f in raw_files[:5]:
                print(f"      {f['modified']}  {f['size_kb']:>8} KB  {f['key']}")
            if len(raw_files) > 5:
                print(f"      ... and {len(raw_files) - 5} more")
        else:
            print(f"   ❌ Error or no recent files: {raw_files}")
    else:
        print("   RAW BUCKET: N/A (derived source — no raw bucket)")

    # ── 4. S3 Processed — recent files ───────────────────────────────────
    print("\n4. S3 PROCESSED — files written in last 24h")
    proc_files = get_recent_s3_files(cfg["proc_bucket"], cfg["proc_prefix"], hours=24)
    if proc_files and "error" not in proc_files[0]:
        print(f"   {check_mark(len(proc_files) > 0)} {len(proc_files)} new files")
        for f in proc_files[:5]:
            print(f"      {f['modified']}  {f['size_kb']:>8} KB  {f['key']}")
        if len(proc_files) > 5:
            print(f"      ... and {len(proc_files) - 5} more")
    else:
        print(f"   ⚠️  No processed files in last 24h (may be dedup skip): {proc_files}")

    # ── 5. Watermarks / Tracker ───────────────────────────────────────────
    if cfg.get("watermark_source"):
        print("\n5. DYNAMODB WATERMARKS")
        wms = get_watermarks(cfg["watermark_source"])
        if wms and "error" not in wms[0]:
            success = [w for w in wms if w.get("status") == "success"]
            print(f"   {check_mark(len(success) > 0)} {len(success)}/{len(wms)} datasets successful")
            for w in wms[:5]:
                status_mark = "✅" if w.get("status") == "success" else "❌"
                print(f"      {status_mark} {w['dataset']:30s} last: {w['last_date']}  {w['status']}")
            if len(wms) > 5:
                print(f"      ... and {len(wms) - 5} more")
        else:
            print(f"   ❌ Error: {wms}")

    elif cfg.get("tracker"):
        print("\n5. SEC TRACKER (AAPL sample)")
        t = get_sec_tracker(cfg["tracker"])
        if "error" not in t:
            print(f"   ✅ fetched: {t['total_fetched']}  "
                  f"transformed: {t['total_transformed']}  "
                  f"prose: {t['total_prose']}")
            print(f"      last_ingest:    {t['last_ingest']}")
            print(f"      last_etl:       {t['last_etl']}")
            print(f"      last_prose_etl: {t['last_prose_etl']}")
        else:
            print(f"   ❌ {t['error']}")

    elif cfg.get("watermark_source") is None and not cfg.get("tracker"):
        if cfg.get("raw_bucket"):
            if cfg.get("ticker_tracker"):
                print("\n5. S3 PER-TICKER TRACKER")
                t = get_insiders_tracker(cfg["raw_bucket"])
                if t["found"]:
                    print(f"   ✅ {t['total']} total accessions across "
                          f"{t['tickers']} ticker tracker file(s)")
                else:
                    print(f"   ❌ {t['error']}")
            else:
                print("\n5. S3 INGESTED-IDS TRACKER")
                t = get_fedspeak_tracker(cfg["raw_bucket"])
                if t["found"]:
                    print(f"   ✅ {t['count']} ingested IDs in tracker/ingested_ids.json")
                else:
                    print(f"   ❌ {t['error']}")
        else:
            print("\n5. WATERMARKS: N/A (full refresh source)")

    # ── 6. Crawler ────────────────────────────────────────────────────────
    print("\n6. GLUE CRAWLER")
    if run_crawler_flag:
        status = run_crawler(cfg["crawler"])
        print(f"   🔄 Triggered crawler: {status}")
        print(f"      Waiting 60s for crawler to complete...")
        import time
        time.sleep(60)

    crawler = get_crawler_status(cfg["crawler"])
    if "error" not in crawler:
        ok = crawler.get("LastStatus") in ("SUCCEEDED", None)
        print(f"   {check_mark(ok)} State:      {crawler['State']}")
        print(f"      Last run:   {crawler['LastRun']}")
        print(f"      Last status:{crawler['LastStatus']}")
    else:
        print(f"   ❌ {crawler['error']}")

    # ── 7. Athena query ───────────────────────────────────────────────────
    print("\n7. ATHENA QUERY")
    print(f"   Running: {cfg['athena_query']}")
    rows = run_athena_query(cfg["athena_query"], cfg["athena_db"])
    if rows and "error" not in rows[0]:
        print(f"   ✅ {len(rows)} rows returned")
        for row in rows:
            print(f"      {row}")
    else:
        print(f"   ❌ {rows}")

    print(f"\n{'='*60}")
    print(f"  VERIFICATION COMPLETE — {source.upper()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify pipeline for a source")
    parser.add_argument("--source",      required=True,
                        choices=list(SOURCE_CONFIG.keys()),
                        help="Source to verify")
    parser.add_argument("--run-crawler", action="store_true",
                        help="Trigger the Glue crawler and wait before Athena query")
    parser.add_argument("--reset",       action="store_true",
                        help="Clear raw bucket, processed bucket, and watermarks/trackers")
    parser.add_argument("--fire",        action="store_true",
                        help="Fire the manual trigger after reset (use with --reset)")
    args = parser.parse_args()

    cfg = SOURCE_CONFIG.get(args.source)
    if not cfg:
        print(f"Unknown source: {args.source}")
        raise SystemExit(1)

    if args.reset:
        reset(args.source, cfg)
        if args.fire:
            print(fire_trigger(args.source))
            print("Run verify again once the pipeline completes.\n")
        else:
            print("Run with --fire to trigger the pipeline, or fire manually.")
    else:
        verify(args.source, run_crawler_flag=args.run_crawler)