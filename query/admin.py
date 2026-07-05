"""
query/admin.py — Data layer for the admin dashboard API.

Called by /admin/api/* routes in server.py. All functions are synchronous
(run_in_threadpool wraps them at the route level). No Athena queries here —
fast AWS metadata calls only.
"""

import os
import json
import glob
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import boto3
from boto3.dynamodb.conditions import Key

# ── Config ─────────────────────────────────────────────────────────────────

REGION     = "us-east-2"
ACCOUNT_ID = "197411402303"
ENV        = "dev"

TABLE_CONVERSATIONS = f"trade-platform-{ENV}-conversations"
TABLE_WATERMARKS    = f"trade-platform-{ENV}-watermarks"
LLMOPS_BUCKET       = f"{ENV}-trade-llmops-{ACCOUNT_ID}"
EVAL_RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "evaluations", "results")

_glue   = boto3.client("glue",      region_name=REGION)
_dynamo = boto3.resource("dynamodb", region_name=REGION)
_s3     = boto3.client("s3",        region_name=REGION)

# ── Source map (job names + crawler only — no Athena/S3 details needed) ───

SOURCES = {
    "fred":      {"ingest": f"{ENV}-trade-fred-ingestion",      "etl": f"{ENV}-trade-fred-etl",      "crawler": f"{ENV}-trade-fred-processed-crawler",      "watermark": "fred",      "manual_trigger": f"{ENV}-trade-fred-manual-trigger"},
    "worldbank": {"ingest": f"{ENV}-trade-worldbank-ingestion", "etl": f"{ENV}-trade-worldbank-etl", "crawler": f"{ENV}-trade-worldbank-processed-crawler", "watermark": "worldbank", "manual_trigger": f"{ENV}-trade-worldbank-manual-trigger"},
    "yfinance":  {"ingest": f"{ENV}-trade-yfinance-ingestion",  "etl": f"{ENV}-trade-yfinance-etl",  "crawler": f"{ENV}-trade-yfinance-processed-crawler",  "watermark": "yfinance",  "manual_trigger": f"{ENV}-trade-yfinance-manual-trigger"},
    "sec":       {"ingest": f"{ENV}-trade-sec-ingestion",       "etl": f"{ENV}-trade-sec-etl",       "crawler": f"{ENV}-trade-sec-processed-crawler",       "watermark": None,        "manual_trigger": f"{ENV}-trade-sec-manual-trigger"},
    "fedspeak":  {"ingest": f"{ENV}-trade-fedspeak-ingestion",  "etl": f"{ENV}-trade-fedspeak-etl",  "crawler": f"{ENV}-trade-fedspeak-processed-crawler",  "watermark": None,        "manual_trigger": f"{ENV}-trade-fedspeak-manual-trigger"},
    "news":      {"ingest": f"{ENV}-trade-news-ingestion",      "etl": f"{ENV}-trade-news-etl",      "crawler": f"{ENV}-trade-news-processed-crawler",      "watermark": None,        "manual_trigger": f"{ENV}-trade-news-manual-trigger"},
    "insiders":  {"ingest": f"{ENV}-trade-insiders-ingestion",  "etl": f"{ENV}-trade-insiders-etl",  "crawler": f"{ENV}-trade-insiders-processed-crawler",  "watermark": None,        "manual_trigger": f"{ENV}-trade-insiders-manual-trigger"},
    "wikipedia": {"ingest": f"{ENV}-trade-wikipedia-ingestion", "etl": f"{ENV}-trade-wikipedia-etl", "crawler": f"{ENV}-trade-wikipedia-processed-crawler", "watermark": "wikipedia", "manual_trigger": f"{ENV}-trade-wikipedia-manual-trigger"},
    "companies": {"ingest": f"{ENV}-trade-yfinance-ingestion",  "etl": f"{ENV}-trade-companies-etl", "crawler": f"{ENV}-trade-yfinance-processed-crawler",  "watermark": None},
}

# ── Low-level helpers ──────────────────────────────────────────────────────

def _fmt_dt(dt) -> str | None:
    if dt is None:
        return None
    if hasattr(dt, "isoformat"):
        return dt.astimezone(timezone.utc).isoformat()
    return str(dt)


def _job_status(job_name: str) -> dict:
    try:
        runs = _glue.get_job_runs(JobName=job_name, MaxResults=1).get("JobRuns", [])
        if not runs:
            return {"state": "NEVER_RUN", "started_on": None, "completed_on": None, "error": None}
        r = runs[0]
        return {
            "state":        r.get("JobRunState"),
            "started_on":   _fmt_dt(r.get("StartedOn")),
            "completed_on": _fmt_dt(r.get("CompletedOn")),
            "error":        (r.get("ErrorMessage") or "")[:200] or None,
        }
    except Exception as e:
        return {"state": "ERROR", "started_on": None, "completed_on": None, "error": str(e)}


def _crawler_status(crawler_name: str) -> dict:
    try:
        c    = _glue.get_crawler(Name=crawler_name)["Crawler"]
        last = c.get("LastCrawl", {})
        return {
            "state":       c.get("State"),
            "last_status": last.get("Status"),
            "last_run":    _fmt_dt(last.get("StartTime")),
        }
    except Exception as e:
        return {"state": "ERROR", "last_status": None, "last_run": None, "error": str(e)}


def _watermark_latest(source: str | None) -> str | None:
    """Most recent last_ingested_period across all watermark rows for a source."""
    if not source:
        return None
    try:
        table = _dynamo.Table(TABLE_WATERMARKS)
        resp  = table.query(KeyConditionExpression=Key("source_name").eq(source))
        dates = [
            i.get("last_ingested_period")
            for i in resp.get("Items", [])
            if i.get("last_ingested_period")
        ]
        return max(dates) if dates else None
    except Exception:
        return None


def _source_status(source_name: str, cfg: dict) -> dict:
    ingest  = _job_status(cfg["ingest"])
    etl     = _job_status(cfg["etl"])
    crawler = _crawler_status(cfg["crawler"])
    wm      = _watermark_latest(cfg.get("watermark"))
    return {
        "source":           source_name,
        "ingest":           ingest,
        "etl":              etl,
        "crawler":          crawler,
        "watermark_latest": wm,
    }

# ── Public API ─────────────────────────────────────────────────────────────

def get_pipeline_status() -> list:
    """
    Returns status for all sources, fetched in parallel.
    ~9 sources × 3 AWS calls each = 27 calls, ~2-4s with parallelism.
    """
    with ThreadPoolExecutor(max_workers=9) as ex:
        futures = {
            ex.submit(_source_status, name, cfg): name
            for name, cfg in SOURCES.items()
        }
        results = []
        for future in list(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"source": futures[future], "error": str(e)})
    return sorted(results, key=lambda x: x["source"])


def get_sessions(limit: int = 20) -> list:
    """
    Scan the conversations table and return recent sessions sorted by last_active.
    Groups items by session_id. Fine for dev-scale; add a GSI on timestamp
    (or a separate sessions table) before this hits production volume.
    """
    table    = _dynamo.Table(TABLE_CONVERSATIONS)
    sessions: dict[str, dict] = {}
    scan_kwargs: dict = {}

    while True:
        resp = table.scan(**scan_kwargs)
        for item in resp.get("Items", []):
            sid = item.get("session_id")
            ts  = item.get("timestamp", "")
            if not sid:
                continue

            if sid not in sessions:
                sessions[sid] = {
                    "session_id":   sid,
                    "last_active":  None,
                    "turn_count":   0,
                    "has_summary":  False,
                    "context_note": None,
                }

            s = sessions[sid]

            if ts == "SUMMARY":
                s["has_summary"] = True
                # turn_count is stored directly on the SUMMARY item
                s["turn_count"]  = int(item.get("turn_count", 0))
                # updated_at is the compression timestamp — best proxy for last_active
                if item.get("updated_at"):
                    ua = item["updated_at"]
                    if s["last_active"] is None or ua > s["last_active"]:
                        s["last_active"] = ua
                # Pull CONTEXT_NOTE out of the structured content field
                for line in (item.get("content") or "").split("\n"):
                    if line.strip().startswith("CONTEXT_NOTE"):
                        s["context_note"] = line.split(":", 1)[-1].strip()[:120]
                        break
            else:
                # Raw turn — use its timestamp as a last_active candidate
                if ts and (s["last_active"] is None or ts > s["last_active"]):
                    s["last_active"] = ts
                # Count user messages only so turn_count = number of Q+A pairs
                if item.get("role") == "user":
                    s["turn_count"] += 1

        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    return sorted(
        sessions.values(),
        key=lambda x: x["last_active"] or "",
        reverse=True,
    )[:limit]


def get_evals() -> dict:
    """
    Read run_*/results.json from evaluations/results/ on disk.
    Returns latest run detail + history (last 10 runs).
    """
    if not os.path.isdir(EVAL_RESULTS_DIR):
        return {"error": f"Results dir not found: {EVAL_RESULTS_DIR}", "runs": []}

    run_files = sorted(
        glob.glob(os.path.join(EVAL_RESULTS_DIR, "run_*", "results.json")),
        reverse=True,  # newest first
    )

    if not run_files:
        return {"latest": None, "runs": []}

    runs = []
    for path in run_files[:10]:
        run_id = os.path.basename(os.path.dirname(path)).replace("run_", "")
        try:
            with open(path) as f:
                data = json.load(f)

            # data may be a list of question records or a dict with a "results" key
            records = data if isinstance(data, list) else data.get("results", [])
            total   = len(records)
            passed  = sum(1 for r in records if r.get("passed") or r.get("score") == 1)

            by_category: dict[str, dict] = {}
            for r in records:
                cat = r.get("category", "unknown")
                if cat not in by_category:
                    by_category[cat] = {"passed": 0, "total": 0}
                by_category[cat]["total"] += 1
                if r.get("passed") or r.get("score") == 1:
                    by_category[cat]["passed"] += 1

            runs.append({
                "run_id":      run_id,
                "total":       total,
                "passed":      passed,
                "pass_rate":   round(passed / total, 3) if total else 0,
                "by_category": by_category,
            })
        except Exception as e:
            runs.append({"run_id": run_id, "error": str(e)})

    return {"latest": runs[0] if runs else None, "runs": runs}


def get_telemetry(limit: int = 25) -> list:
    """
    List and parse recent per-agent trace files from the LLMOps S3 bucket.
    Traces are partitioned as traces/year=YYYY/month=MM/AgentName_uuid.json.
    Targets current + previous month only to avoid full-bucket listing (1000+ files).

    One trace = one agent run. A 3-agent query produces 3 trace files.
    """
    from datetime import date

    today = date.today()
    prev_year  = today.year - 1 if today.month == 1 else today.year
    prev_month = 12 if today.month == 1 else today.month - 1
    prefixes = [
        f"traces/year={today.year}/month={today.month:02d}/",
        f"traces/year={prev_year}/month={prev_month:02d}/",
    ]

    all_objects = []
    for prefix in prefixes:
        try:
            paginator = _s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=LLMOPS_BUCKET, Prefix=prefix):
                all_objects.extend(page.get("Contents", []))
        except Exception:
            continue

    if not all_objects:
        return []

    recent  = sorted(all_objects, key=lambda o: o["LastModified"], reverse=True)[:limit]
    records = []

    for obj in recent:
        key = obj["Key"]
        if "test" in key:
            continue
        try:
            body  = _s3.get_object(Bucket=LLMOPS_BUCKET, Key=key)["Body"].read()
            trace = json.loads(body)
            records.append({
                "timestamp":           _fmt_dt(obj["LastModified"]),
                "trace_id":            trace.get("trace_id"),
                "session_id":          trace.get("session_id"),
                "agent":               trace.get("agent"),
                "node_id":             trace.get("node_id"),
                "question":            (trace.get("question") or "")[:120],
                "iterations":          trace.get("iterations"),
                "total_tokens":        trace.get("total_tokens"),
                "input_tokens":        trace.get("input_tokens"),
                "output_tokens":       trace.get("output_tokens"),
                "latency_s":           round(trace["latency_ms"] / 1000, 2) if trace.get("latency_ms") else None,
                "reflexion_triggered": trace.get("reflexion_triggered"),
                "reflexion_passed":    trace.get("reflexion_passed"),
                "hit_max_iter":        trace.get("hit_max_iter"),
                "answer_preview":      (trace.get("answer_preview") or "")[:200],
                "s3_key":              key,
            })
        except Exception:
            continue

    return records


def get_session_detail(session_id: str) -> dict:
    """
    Return full content for a single session: SUMMARY + all raw turns in order.
    Used by the session detail modal in the admin UI.
    """
    table = _dynamo.Table(TABLE_CONVERSATIONS)
    resp  = table.query(KeyConditionExpression=Key("session_id").eq(session_id))
    items = resp.get("Items", [])

    summary = None
    turns   = []

    for item in items:
        ts = item.get("timestamp", "")
        if ts == "SUMMARY":
            summary = {
                "content":    item.get("content"),
                "turn_count": int(item.get("turn_count", 0)),
                "updated_at": item.get("updated_at"),
            }
        else:
            turns.append({
                "role":      item.get("role"),
                "content":   item.get("content"),
                "timestamp": ts,
            })

    turns.sort(key=lambda x: x.get("timestamp") or "")

    return {
        "session_id": session_id,
        "summary":    summary,
        "turns":      turns,
    }


def get_trace_detail(s3_key: str) -> dict:
    """
    Fetch a single full trace from S3 by its key.
    Returns all fields including tools_called and full answer.
    """
    try:
        body  = _s3.get_object(Bucket=LLMOPS_BUCKET, Key=s3_key)["Body"].read()
        trace = json.loads(body)
        return {
            "trace_id":            trace.get("trace_id"),
            "session_id":          trace.get("session_id"),
            "agent":               trace.get("agent"),
            "node_id":             trace.get("node_id"),
            "question":            trace.get("question", ""),
            "iterations":          trace.get("iterations"),
            "total_tokens":        trace.get("total_tokens"),
            "input_tokens":        trace.get("input_tokens"),
            "output_tokens":       trace.get("output_tokens"),
            "latency_s":           round(trace["latency_ms"] / 1000, 2) if trace.get("latency_ms") else None,
            "hit_max_iter":        trace.get("hit_max_iter"),
            "hit_token_budget":    trace.get("hit_token_budget"),
            "reflexion_triggered": trace.get("reflexion_triggered"),
            "reflexion_passed":    trace.get("reflexion_passed"),
            "had_dedup_hits":      trace.get("had_dedup_hits"),
            "tools_called":        trace.get("tools_called", []),
            "answer_preview":      trace.get("answer_preview", ""),
        }
    except Exception as e:
        return {"error": str(e)}


def fire_pipeline(source: str) -> dict:
    """Fire the ON_DEMAND manual trigger for a source. Self-cascades
    through the existing CONDITIONAL trigger chain (ingestion -> ETL ->
    crawler) with no further backend action needed. Returns immediately
    -- does not wait for the pipeline to complete."""
    cfg = SOURCES.get(source)
    if not cfg or not cfg.get("manual_trigger"):
        return {"ok": False, "error": f"No manual trigger available for '{source}'"}
    try:
        _glue.start_trigger(Name=cfg["manual_trigger"])
        return {"ok": True, "source": source, "trigger": cfg["manual_trigger"]}
    except _glue.exceptions.ConcurrentRunsExceededException:
        return {"ok": False, "error": "Trigger already running or a downstream job is mid-run"}
    except Exception as e:
        return {"ok": False, "error": str(e)}