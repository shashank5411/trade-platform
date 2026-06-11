"""
Conversation memory — DynamoDB-backed session storage.
Loads last N turns per session for multi-turn context.
"""

import os
import json
import boto3
from datetime import datetime, timezone, timedelta
from typing import Optional

ENV      = os.environ.get("ENV", "dev")
ACCOUNT  = os.environ.get("ACCOUNT", "197411402303")
TABLE    = f"trade-platform-{ENV}-conversations"
TTL_DAYS  = 30   # auto-expire sessions after 30 days
MAX_TURNS = 10   # default turn window

dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
table    = dynamodb.Table(TABLE)


def save_turn(session_id: str, role: str, content: str) -> None:
    """Save a single turn (user or assistant) to DynamoDB."""
    now = datetime.now(timezone.utc)
    ttl = int((now + timedelta(days=TTL_DAYS)).timestamp())
    table.put_item(Item={
        "session_id": session_id,
        "timestamp":  now.isoformat(),
        "role":       role,
        "content":    content,
        "ttl":        ttl,
    })


def load_turns(session_id: str, max_turns: int = MAX_TURNS) -> list:
    """
    Load last max_turns exchanges for a session.
    Returns list of {role, content} dicts ready for messages array.
    Each exchange = 1 user turn + 1 assistant turn = 2 items.
    So we fetch max_turns * 2 items.
    """
    resp = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key(
            "session_id"
        ).eq(session_id),
        ScanIndexForward=False,  # newest first
        Limit=max_turns * 2,
    )
    items = resp.get("Items", [])
    # Reverse to chronological order
    items = sorted(items, key=lambda x: x["timestamp"])
    return [{"role": i["role"], "content": i["content"]} for i in items]


def list_sessions() -> list:
    """List all active session IDs. Used for CLI --list-sessions."""
    resp = table.scan(
        ProjectionExpression="session_id",
    )
    ids = sorted(set(i["session_id"] for i in resp.get("Items", [])))
    return ids


def clear_session(session_id: str) -> int:
    """Delete all turns for a session. Returns count deleted."""
    resp  = table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key(
            "session_id"
        ).eq(session_id),
    )
    items = resp.get("Items", [])
    with table.batch_writer() as batch:
        for item in items:
            batch.delete_item(Key={
                "session_id": item["session_id"],
                "timestamp":  item["timestamp"],
            })
    return len(items)
