import os
import json
import gzip
from datetime import datetime, timezone
from typing import Optional

import boto3

_ENV = os.getenv("ENVIRONMENT", "dev")
_TABLE_NAME = f"trade-platform-{_ENV}-watermarks"
_table = boto3.resource("dynamodb").Table(_TABLE_NAME)


def get_watermark(source: str, item_id: str) -> Optional[dict]:
    resp = _table.get_item(Key={"source_name": source, "dataset_name": item_id})
    return resp.get("Item")


def update_watermark(
    source: str,
    item_id: str,
    last_date: str,
    status: str,
    count: int,
) -> None:
    _table.put_item(Item={
        "source_name":          source,
        "dataset_name":         item_id,
        "last_ingested_period": last_date,
        "last_run_timestamp":   datetime.now(timezone.utc).isoformat(),
        "last_run_status":      status,
        "records_ingested":     count,
    })


# ── SEC S3 Tracker ─────────────────────────────────────────────────────────
# DynamoDB not used here — accession lists grow too large for a 400KB item.
# Tracker files live at: s3://{raw_bucket}/tracker/sec_{ticker}_tracker.json

_TRACKER_PREFIX = "tracker"


def _tracker_key(ticker: str) -> str:
    return f"{_TRACKER_PREFIX}/sec_{ticker}_tracker.json"


def load_sec_tracker(s3_client, bucket: str, ticker: str) -> dict:
    """
    Load the tracker file for a ticker.
    Returns empty tracker if none exists yet (first run).
    """
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=_tracker_key(ticker))
        return json.loads(obj["Body"].read())
    except s3_client.exceptions.NoSuchKey:
        return {
            "ticker":                   ticker,
            "fetched_accessions":       [],   # populated by ingest_sec.py
            "transformed_accessions":   [],   # populated by etl_sec.py
            "prose_accessions":         [],   # populated by etl_sec_prose.py
            "last_ingest":              None,
            "last_etl":                 None,
            "last_prose_etl":           None,
            "total_fetched":            0,
            "total_transformed":        0,
            "total_prose":              0,
        }


def save_sec_tracker(s3_client, bucket: str, ticker: str, tracker: dict) -> None:
    """Save tracker file back to S3."""
    tracker["updated_at"] = datetime.now(timezone.utc).isoformat()
    s3_client.put_object(
        Bucket=bucket,
        Key=_tracker_key(ticker),
        Body=json.dumps(tracker, indent=2),
        ContentType="application/json",
    )
    print(f"  Tracker saved → s3://{bucket}/{_tracker_key(ticker)}")


def get_new_accessions(tracker: dict, field: str, all_accessions: list) -> list:
    """
    Return only accessions not yet recorded in tracker[field].
    field: 'fetched_accessions' | 'transformed_accessions' | 'prose_accessions'
    """
    already_done = set(tracker.get(field, []))
    new = [a for a in all_accessions if a not in already_done]
    print(f"  [{field}] {len(already_done)} already done, "
          f"{len(new)} new out of {len(all_accessions)} total")
    return new


def mark_accessions_done(
    tracker: dict, field: str, count_field: str, accessions: list
) -> dict:
    """
    Add accessions to tracker[field] and update count.
    Returns updated tracker (caller must still call save_sec_tracker).
    """
    existing = set(tracker.get(field, []))
    existing.update(accessions)
    tracker[field] = sorted(existing)
    tracker[count_field] = len(tracker[field])
    return tracker