"""
query/memory.py — DynamoDB-backed session memory with rolling summary compression.

DynamoDB schema (trade-platform-{env}-conversations):
  PK = session_id  (string)
  SK = timestamp   (ISO string)  ← raw turns
       "SUMMARY"                 ← compressed rolling summary (one per session)

Raw turn item:  { session_id, timestamp, role, content, ttl }
Summary item:   { session_id, timestamp="SUMMARY", content, turn_count, updated_at, ttl }
"""

import os
import time
import datetime
import boto3
from boto3.dynamodb.conditions import Key

from query.config import get_client  # Anthropic client factory

# ── Config ────────────────────────────────────────────────────────────────────
ENV             = os.environ.get("ENV", "dev")
TABLE_NAME      = f"trade-platform-{ENV}-conversations"
TTL_DAYS        = 30
SUMMARY_SK      = "SUMMARY"

RAW_TURNS_THRESHOLD = 8   # compress when raw turn count exceeds this
COMPRESS_BATCH      = 3   # absorb this many old turns per compression pass
KEEP_RAW            = RAW_TURNS_THRESHOLD - COMPRESS_BATCH  # = 5

HAIKU_MODEL = "claude-haiku-4-5-20251001"

# ── DynamoDB singleton ────────────────────────────────────────────────────────
_dynamo_table = None


def _table():
    global _dynamo_table
    if _dynamo_table is None:
        _dynamo_table = boto3.resource(
            "dynamodb", region_name="us-east-2"
        ).Table(TABLE_NAME)
    return _dynamo_table


def _ttl(days: int = TTL_DAYS) -> int:
    return int(time.time()) + days * 86400


def _now() -> str:
    return datetime.datetime.utcnow().isoformat()


# ── Public API ────────────────────────────────────────────────────────────────

def save_turn(session_id: str, role: str, content: str) -> None:
    """
    Save a single conversation turn.
    Triggers rolling compression if raw turn count exceeds RAW_TURNS_THRESHOLD.
    Called AFTER the answer is returned to the user, so compression latency
    does not affect response time.
    """
    _table().put_item(Item={
        "session_id": session_id,
        "timestamp":  _now(),
        "role":       role,
        "content":    content,
        "ttl":        _ttl(),
    })
    raw_count = _count_raw_turns(session_id)
    if raw_count > RAW_TURNS_THRESHOLD:
        _compress(session_id)


def load_context(session_id: str) -> dict:
    """
    Load full context for a session. Called at the START of each query.

    Returns:
        {
            "summary":      str | None,   # full structured summary (ENTITIES, FACTS, etc.)
            "context_note": str | None,   # extracted 1-2 sentence note, injected into agents
            "recent_turns": list[dict],   # last KEEP_RAW raw turns as {role, content}
        }
    """
    summary_item  = _load_summary(session_id)
    recent_turns  = load_turns(session_id, max_turns=KEEP_RAW)
    summary_text  = summary_item.get("content") if summary_item else None
    context_note  = _extract_context_note(summary_text) if summary_text else None

    return {
        "summary":      summary_text,
        "context_note": context_note,
        "recent_turns": recent_turns,
    }


def load_turns(session_id: str, max_turns: int = 10) -> list:
    """
    Load the most recent raw turns (excludes SUMMARY item).
    Returns list of {role, content} dicts in chronological order.
    """
    resp = _table().query(
        KeyConditionExpression=Key("session_id").eq(session_id),
        ScanIndexForward=False,      # newest first from DynamoDB
        Limit=max_turns + 2,         # small buffer in case SUMMARY item appears
    )
    items = [i for i in resp.get("Items", []) if i.get("timestamp") != SUMMARY_SK]
    items = items[:max_turns]
    items.reverse()                  # back to chronological
    return [{"role": i["role"], "content": i["content"]} for i in items]


def list_sessions() -> list:
    """Return distinct session IDs. CLI utility."""
    resp = _table().scan(ProjectionExpression="session_id")
    return sorted({i["session_id"] for i in resp.get("Items", [])})


def clear_session(session_id: str) -> int:
    """Delete all items for a session including the summary. Returns count deleted."""
    resp = _table().query(
        KeyConditionExpression=Key("session_id").eq(session_id)
    )
    items = resp.get("Items", [])
    with _table().batch_writer() as batch:
        for item in items:
            batch.delete_item(Key={
                "session_id": item["session_id"],
                "timestamp":  item["timestamp"],
            })
    return len(items)


# ── Internal: summary CRUD ────────────────────────────────────────────────────

def _load_summary(session_id: str):
    resp = _table().get_item(
        Key={"session_id": session_id, "timestamp": SUMMARY_SK}
    )
    return resp.get("Item")


def _write_summary(session_id: str, content: str, turn_count: int) -> None:
    _table().put_item(Item={
        "session_id": session_id,
        "timestamp":  SUMMARY_SK,
        "content":    content,
        "turn_count": turn_count,
        "updated_at": _now(),
        "ttl":        _ttl(),   # rolling TTL — reset each compression
    })


def _count_raw_turns(session_id: str) -> int:
    resp  = _table().query(
        KeyConditionExpression=Key("session_id").eq(session_id),
        Select="COUNT",
    )
    total   = resp.get("Count", 0)
    has_sum = _load_summary(session_id) is not None
    return total - (1 if has_sum else 0)


def _load_oldest_raw_turns(session_id: str, n: int) -> list:
    """Return the n oldest raw turns (ascending SK = oldest timestamps first)."""
    resp = _table().query(
        KeyConditionExpression=Key("session_id").eq(session_id),
        ScanIndexForward=True,   # oldest first
        Limit=n + 2,
    )
    items = [i for i in resp.get("Items", []) if i.get("timestamp") != SUMMARY_SK]
    return items[:n]


def _delete_turns(session_id: str, items: list) -> None:
    with _table().batch_writer() as batch:
        for item in items:
            batch.delete_item(Key={
                "session_id": session_id,
                "timestamp":  item["timestamp"],
            })


# ── Internal: compression ─────────────────────────────────────────────────────

def _compress(session_id: str) -> None:
    """
    Absorb the oldest COMPRESS_BATCH raw turns into the rolling summary.
    1. Load existing summary (if any)
    2. Load oldest COMPRESS_BATCH raw turns
    3. Call Haiku to produce updated structured summary
    4. Write new summary, delete absorbed raw turns
    """
    existing    = _load_summary(session_id)
    prior_text  = existing.get("content", "") if existing else ""
    prior_count = existing.get("turn_count", 0) if existing else 0

    oldest = _load_oldest_raw_turns(session_id, COMPRESS_BATCH)
    if not oldest:
        return

    formatted   = _format_turns_for_compression(oldest)
    new_content = _call_haiku_compress(prior_text, formatted)

    _write_summary(session_id, new_content, prior_count + len(oldest))
    _delete_turns(session_id, oldest)


def _format_turns_for_compression(items: list) -> str:
    parts = []
    for idx, item in enumerate(items, 1):
        role    = item.get("role", "unknown").capitalize()
        content = item.get("content", "")[:2000]   # cap per turn
        parts.append(f"[Turn {idx}] {role}: {content}")
    return "\n\n".join(parts)


def _call_haiku_compress(existing_summary: str, formatted_turns: str) -> str:
    """
    Call Haiku to merge existing summary + new turns into an updated summary.
    Returns the raw structured text string.
    """
    system = """You maintain structured memory for a financial analysis AI assistant.
Given an existing summary and new conversation turns to absorb, output ONLY
the updated summary in the exact format below. Do not add any preamble or explanation.

Rules:
- Never invent, infer, or extrapolate. Only preserve what is explicitly stated.
- CONFIRMED_FACTS: include approximate figures and which turns they came from.
- CONTEXT_NOTE: 1-2 sentences max describing what the user is trying to understand.
  If a clear topic shift occurred in the new turns, note it: "Started with X, now focused on Y."
- Remove PENDING items that were answered in the new turns.
- ENTITIES must be comprehensive — include all tickers/topics ever discussed this session.

Output format (use exactly these labels, nothing else):
ENTITIES: [comma-separated tickers, sectors, indicators, or topics]
TIME_SCOPE: [date ranges and fiscal periods explicitly discussed, or "unspecified"]
CONFIRMED_FACTS:
  - [fact with ~figure and turn reference, e.g. "NVDA revenue ~$60B (T1-T3)"]
PENDING: [open questions not yet answered, or "none"]
CONTEXT_NOTE: [1-2 sentences on the user's overall analytical goal]"""

    user_msg = (
        f"Current summary:\n"
        f"{existing_summary if existing_summary else 'None yet — first compression.'}"
        f"\n\nTurns to absorb:\n{formatted_turns}"
    )

    client   = get_client()
    response = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=1200,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text.strip()


# ── Internal: context_note extraction ────────────────────────────────────────

def _extract_context_note(summary_text: str):
    """
    Pull the CONTEXT_NOTE value from a structured summary string.
    Returns a plain string suitable for injecting into agent prompts.
    """
    if not summary_text:
        return None
    lines      = summary_text.splitlines()
    note_lines = []
    in_note    = False
    for line in lines:
        if line.startswith("CONTEXT_NOTE:"):
            in_note = True
            rest    = line[len("CONTEXT_NOTE:"):].strip()
            if rest:
                note_lines.append(rest)
        elif in_note and (line.startswith("  ") or line.startswith("\t")):
            note_lines.append(line.strip())
        elif in_note:
            break
    return " ".join(note_lines) if note_lines else None
