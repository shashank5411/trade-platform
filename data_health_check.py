"""
data_health_check.py — High-level data validation across all pipelines.

Separate from agent-level evals: catches data-layer problems (silent gaps,
stale watermarks, under-registered partitions, zero-row tables) that agent
eval can't see, per Production_Eval TODO item #5.

Reuses verify_pipeline.py's SOURCE_CONFIG and helper functions directly —
this script does NOT duplicate bucket/table/job-name config. If you add a
new source to verify_pipeline.py's SOURCE_CONFIG, it's automatically picked
up here too.

Usage:
    python data_health_check.py                  # check all sources, write to DynamoDB
    python data_health_check.py --source fred     # check one source only
    python data_health_check.py --no-write        # print only, skip DynamoDB write
    python data_health_check.py --json            # print raw JSON (for scripting/CI)

Exit code: 0 if all sources "ok", 1 if any source is "fail".
("warn" does not fail the exit code — it's a soft flag for things like
crawler/partition mismatches that need a look but aren't necessarily broken.)

Run from Docker via cron on the EC2 host, e.g.:
    docker exec trade-platform python data_health_check.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone

import boto3

from verify_pipeline import (
    SOURCE_CONFIG,
    REGION,
    ACCOUNT_ID,
    ENV,
    get_last_job_run,
    get_recent_s3_files,
    get_watermarks,
    get_sec_tracker,
    get_fedspeak_tracker,
    get_insiders_tracker,
    get_crawler_status,
    run_athena_query,
)

HEALTH_TABLE = f"trade-platform-{ENV}-data-health"  # NEW table — needs CDK addition, see bottom of file

dynamodb = boto3.resource("dynamodb", region_name=REGION)

# ── Per-source expectations ─────────────────────────────────────────────────
# Cadence-based freshness thresholds (days) — derived from Glue job schedules
# in trade_platform_stack.py.
FRESHNESS_THRESHOLD_DAYS = {
    "yfinance":  4,     # daily (MON-FRI), some slack for weekends/holidays
    "companies": 5,     # conditional on yfinance ETL
    "fred":      40,    # monthly
    "sec":       100,   # quarterly
    "news":      12,    # weekly (Monday)
    "insiders":  100,   # quarterly
    # worldbank, wikipedia, fedspeak intentionally excluded — see
    # NO_FRESHNESS_EXPECTED below.
}

# Sources where "is the latest value recent relative to today" is the wrong
# question entirely, so both the date-coverage staleness check and the
# recent-activity cadence check are informational-only (never fail/warn):
#   - worldbank: known reporting lag baked into the source itself — WorldBank
#     often publishes a given year's data 1-2 years after the fact. Checking
#     "did we ingest the latest AVAILABLE release" would require tracking
#     WorldBank's own release calendar, which this script doesn't do. Row
#     count + job state + watermark checks still apply.
#   - wikipedia: static reference articles, not a recency-driven feed. Low
#     churn is expected and not a signal of anything broken.
#   - fedspeak: event-driven (FOMC calendar), no fixed cadence to compare against.
NO_FRESHNESS_EXPECTED = {"worldbank", "wikipedia", "fedspeak"}

# Date column per athena_table, used for MIN/MAX coverage checks.
# CONFIRMED from athena_query strings in verify_pipeline.py:
#   fred/worldbank (economic_indicators) -> "date"
#   yfinance (market_prices)             -> "date"
#   wikipedia/sec/fedspeak (documents)   -> "doc_date"
# NOT CONFIRMED — news and insider_trades tables have no date column visible
# in their athena_query SELECT lists. Skipping coverage check for these two
# until the real column name (e.g. published_date / transaction_date /
# filing_date) is confirmed against the actual schema — guessing here would
# silently produce a wrong check, which is worse than no check.
DATE_COLUMN = {
    "fred": "date",
    "worldbank": "date",
    "yfinance": "date",
    "companies": None,       # full-refresh, no date semantics
    "wikipedia": "doc_date",
    "sec": "doc_date",
    "fedspeak": "doc_date",
    "news": None,            # TODO: confirm real column name
    "insiders": None,        # TODO: confirm real column name
}


def check_row_count(cfg: dict) -> dict:
    query = f"SELECT COUNT(*) as cnt FROM {cfg['athena_table']}"
    rows = run_athena_query(query, cfg["athena_db"])
    if rows and "error" not in rows[0]:
        return {"row_count": int(rows[0]["cnt"]), "ok": int(rows[0]["cnt"]) > 0}
    return {"row_count": None, "ok": False, "error": rows[0].get("error") if rows else "no result"}


def check_date_coverage(cfg: dict, source_name: str):
    date_col = DATE_COLUMN.get(source_name)
    if date_col is None:
        return None  # not configured — see DATE_COLUMN comment above

    query = f"SELECT MIN({date_col}) as min_date, MAX({date_col}) as max_date FROM {cfg['athena_table']}"
    rows = run_athena_query(query, cfg["athena_db"])
    if not rows or "error" in rows[0]:
        return {"ok": False, "error": rows[0].get("error") if rows else "no result"}

    min_date, max_date = rows[0].get("min_date"), rows[0].get("max_date")
    result = {"min_date": min_date, "max_date": max_date}

    if source_name in NO_FRESHNESS_EXPECTED:
        # Still compute days_stale for visibility in the dashboard, just never
        # score it as ok/not-ok.
        if max_date:
            try:
                max_dt = datetime.strptime(max_date[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                result["days_stale"] = (datetime.now(timezone.utc) - max_dt).days
            except ValueError:
                pass
        result["freshness_ok"] = None
        return result

    threshold = FRESHNESS_THRESHOLD_DAYS.get(source_name)
    if threshold is not None and max_date:
        try:
            max_dt = datetime.strptime(max_date[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days_stale = (datetime.now(timezone.utc) - max_dt).days
            result["days_stale"] = days_stale
            result["freshness_ok"] = days_stale <= threshold
        except ValueError:
            result["freshness_ok"] = None
            result["error"] = f"could not parse max_date: {max_date}"
    else:
        result["freshness_ok"] = None  # no threshold configured — informational only

    return result


def check_watermark_or_tracker(cfg: dict, source_name: str) -> dict:
    """Mirrors the exact branching logic in verify_pipeline.py's verify()."""
    if cfg.get("watermark_source"):
        wms = get_watermarks(cfg["watermark_source"])
        if wms and "error" not in wms[0]:
            success = [w for w in wms if w.get("status") == "success"]
            return {
                "type": "dynamodb_watermark",
                "ok": len(success) > 0,
                "datasets_total": len(wms),
                "datasets_successful": len(success),
            }
        return {"type": "dynamodb_watermark", "ok": False, "error": wms}

    if cfg.get("tracker"):  # SEC-style single S3 tracker
        t = get_sec_tracker(cfg["tracker"])
        if "error" not in t:
            return {"type": "sec_tracker", "ok": True, **t}
        return {"type": "sec_tracker", "ok": False, "error": t["error"]}

    if cfg.get("raw_bucket") and cfg.get("ticker_tracker"):
        t = get_insiders_tracker(cfg["raw_bucket"])
        return {"type": "per_ticker_tracker", "ok": t.get("found", False), **t}

    if cfg.get("raw_bucket") and not cfg.get("watermark_source") and not cfg.get("tracker"):
        t = get_fedspeak_tracker(cfg["raw_bucket"])
        return {"type": "ingested_ids_tracker", "ok": t.get("found", False), **t}

    return {"type": "none", "ok": None, "note": "full-refresh source — no watermark expected"}


def check_recent_activity(cfg: dict, source_name: str):
    """Recent processed-file activity, windowed to the source's own cadence
    instead of a fixed 24h (verify_pipeline.py uses 24h since it's meant to
    run right after a manual trigger; here we want 'did the scheduled job
    run when it should have'). Returns None (skipped, not scored) for sources
    in NO_FRESHNESS_EXPECTED — there's no cadence to check activity against."""
    if source_name in NO_FRESHNESS_EXPECTED:
        return None

    threshold_days = FRESHNESS_THRESHOLD_DAYS.get(source_name, 30)
    hours = threshold_days * 24
    files = get_recent_s3_files(cfg["proc_bucket"], cfg["proc_prefix"], hours=hours)
    if files and "error" in files[0]:
        return {"ok": False, "error": files[0]["error"]}
    return {"ok": len(files) > 0, "file_count": len(files)}


def check_source(source_name: str, cfg: dict) -> dict:
    result = {"source": source_name, "status": "ok", "issues": []}

    def fail(msg):
        result["status"] = "fail"
        result["issues"].append(msg)

    def warn(msg):
        if result["status"] != "fail":
            result["status"] = "warn"
        result["issues"].append(msg)

    # 1. Row count
    row_check = check_row_count(cfg)
    result["row_count"] = row_check
    if not row_check["ok"]:
        fail(f"row count check failed: {row_check.get('error', 'zero rows')}")

    # 2. Job states
    ingest = get_last_job_run(cfg["ingest_job"])
    etl = get_last_job_run(cfg["etl_job"])
    result["ingest_job"] = {"state": ingest["State"]}
    result["etl_job"] = {"state": etl["State"]}
    if ingest["State"] not in ("SUCCEEDED", "RUNNING"):
        warn(f"ingest job last state: {ingest['State']}")
    if etl["State"] not in ("SUCCEEDED", "RUNNING"):
        warn(f"etl job last state: {etl['State']}")

    # 3. Date coverage / freshness (where configured)
    coverage = check_date_coverage(cfg, source_name)
    result["date_coverage"] = coverage
    if coverage is not None:
        if coverage.get("freshness_ok") is False:
            fail(f"stale: {coverage.get('days_stale')}d since max({DATE_COLUMN[source_name]}), "
                 f"threshold {FRESHNESS_THRESHOLD_DAYS.get(source_name)}d")
        elif "error" in coverage:
            warn(f"date coverage check error: {coverage['error']}")

    # 4. Watermark / tracker sanity
    wm = check_watermark_or_tracker(cfg, source_name)
    result["watermark"] = wm
    if wm.get("ok") is False:
        warn(f"watermark/tracker check failed: {wm.get('error', 'not found')}")

    # 5. Recent activity vs own cadence (skipped/None for NO_FRESHNESS_EXPECTED sources)
    activity = check_recent_activity(cfg, source_name)
    result["recent_activity"] = activity
    if activity is not None and not activity.get("ok"):
        warn(f"no processed files within {FRESHNESS_THRESHOLD_DAYS.get(source_name, 30)}-day cadence window")

    # 6. Crawler status
    crawler = get_crawler_status(cfg["crawler"])
    result["crawler"] = crawler
    if "error" not in crawler and crawler.get("LastStatus") not in ("SUCCEEDED", None):
        warn(f"crawler last status: {crawler.get('LastStatus')}")

    return result


def write_results(summary: dict):
    table = dynamodb.Table(HEALTH_TABLE)
    checked_at = summary["checked_at"]
    body = json.dumps(summary, default=str)
    table.put_item(Item={"pk": "latest", "checked_at": checked_at, "body": body})
    table.put_item(Item={"pk": f"run#{checked_at}", "checked_at": checked_at, "body": body})


def main():
    parser = argparse.ArgumentParser(description="High-level data health check across all pipelines")
    parser.add_argument("--source", choices=list(SOURCE_CONFIG.keys()), help="Check a single source only")
    parser.add_argument("--no-write", action="store_true", help="Skip DynamoDB write (print only)")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of formatted output")
    args = parser.parse_args()

    sources = {args.source: SOURCE_CONFIG[args.source]} if args.source else SOURCE_CONFIG

    results = {}
    overall = "ok"
    for name, cfg in sources.items():
        r = check_source(name, cfg)
        results[name] = r
        if r["status"] == "fail":
            overall = "fail"
        elif r["status"] == "warn" and overall != "fail":
            overall = "warn"

    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "sources": results,
    }

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(f"\n{'='*60}")
        print(f"  DATA HEALTH CHECK — overall: {overall.upper()}")
        print(f"{'='*60}\n")
        for name, r in results.items():
            mark = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}[r["status"]]
            rc = r["row_count"].get("row_count")
            print(f"[{mark}] {name:12s} rows={rc}  status={r['status']}")
            for issue in r["issues"]:
                print(f"        - {issue}")
        print()

    if not args.no_write:
        write_results(summary)
        print(f"Written to DynamoDB: {HEALTH_TABLE} (pk='latest' and pk='run#{summary['checked_at']}')")

    sys.exit(1 if overall == "fail" else 0)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# CDK addition needed — new DynamoDB table, add to trade_platform_stack.py:
#
#   data_health_table = dynamodb.Table(
#       self, "DataHealthTable",
#       table_name=f"trade-platform-{env_name}-data-health",
#       partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
#       billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
#       removal_policy=RemovalPolicy.RETAIN,
#   )
#   data_health_table.grant_read_write_data(<role running this script>)
#   data_health_table.grant_read_data(<EC2/admin role>)
#
# Also grant this script's execution role (whatever runs it in Docker,
# presumably the same EC2 instance role query/server.py uses) glue:GetJobRuns,
# glue:GetCrawler, athena:StartQueryExecution/GetQueryResults/GetQueryExecution,
# s3:ListBucket/GetObject on all raw+processed buckets, dynamodb:Query on the
# watermarks table, and dynamodb:PutItem on the new data-health table.
# ─────────────────────────────────────────────────────────────────────────