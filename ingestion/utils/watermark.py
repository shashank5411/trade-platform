import os
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
