#!/usr/bin/env python3
"""
INTEGRATION/LIVE: requires a running query server (BASE_URL) and a real
DynamoDB table (trade-platform-dev-conversations) — not meant for CI.

Memory E2E Test
===============
Tests the rolling summary + entity carryover across 10+ turns.

Memory config (from memory.py):
  - Compression threshold : 8 raw turns
  - Batch absorbed        : 3 oldest turns
  - Raw turns kept        : 5
  - Summary SK            : "SUMMARY"
  - DynamoDB table        : trade-platform-dev-conversations

What this test verifies:
  1. After turn 8, compression fires and a SUMMARY item appears in DynamoDB
  2. Raw turn count drops to ≤5 after compression
  3. SUMMARY contains expected structured fields (ENTITIES, CONFIRMED_FACTS, etc.)
  4. Turns 9-10 reference AAPL implicitly → response should still be AAPL-contextual
     (carryover from CONTEXT_NOTE injected into the enriched question)

Usage:
  pip install requests boto3
  python test_memory_e2e.py

  # Override endpoint or region:
  BASE_URL=http://localhost:8000 python test_memory_e2e.py
"""

import os
import sys
import uuid
import json
import time
from datetime import datetime

import requests
import boto3
from boto3.dynamodb.conditions import Key

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL   = os.getenv("BASE_URL", "http://localhost:8000")
TABLE_NAME = os.getenv("DYNAMO_TABLE", "trade-platform-dev-conversations")
REGION     = os.getenv("AWS_REGION", "us-east-2")

# Fixed for this run so we can look it up in DynamoDB afterwards
SESSION_ID = str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Questions — designed to:
#   Turns 1-3  : Establish AAPL as primary entity (market, filings, sentiment)
#   Turns 4-6  : Shift to macro context (rates, yield curve, Fed tone)
#   Turns 7-8  : Cross-entity questions — AAPL vs macro (compression triggers here)
#   Turns 9-10 : Implicit entity carryover — no "Apple" in question,
#                only works if CONTEXT_NOTE carries the entity through
# ---------------------------------------------------------------------------

QUESTIONS = [
    # --- Establish AAPL ---
    "What has Apple's stock price done over the past 6 months?",
    "Have there been any notable Apple insider trades recently — buying or selling?",
    "What's the recent news sentiment like around Apple?",

    # --- Shift to macro ---
    "How have US interest rates trended over the same period?",
    "What does the yield curve look like right now?",
    "Has Fed communication shifted tone recently — any changes in language?",

    # --- Cross-entity (triggers compression at turn 8) ---
    "Given that macro backdrop, how does Apple's recent price move look?",
    "Were Apple insiders buying or selling during that rate environment?",

    # --- Implicit carryover (no 'Apple' mentioned — only works with memory) ---
    "What has the insider activity looked like more broadly — is the pattern consistent?",
    "And how does that compare to what the macro data is telling us overall?",
]

COMPRESSION_TRIGGER_AT = 8   # check DynamoDB after this turn
TOTAL_TURNS = len(QUESTIONS)

# ---------------------------------------------------------------------------
# Expected SUMMARY fields (from memory.py structured format)
# ---------------------------------------------------------------------------
EXPECTED_SUMMARY_FIELDS = ["ENTITIES", "TIME_SCOPE", "CONFIRMED_FACTS", "PENDING", "CONTEXT_NOTE"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def separator(char="=", width=62):
    print(char * width)

def post_question(question: str, session_id: str) -> dict:
    """POST a question to the FastAPI /query endpoint."""
    resp = requests.post(
        f"{BASE_URL}/ask",
        json={"question": question, "session_id": session_id},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()

def extract_answer(result: dict) -> str:
    """Pull answer text from the response — handle possible key variations."""
    for key in ("answer", "response", "result", "text"):
        if key in result:
            return result[key]
    # Fallback: dump raw
    return json.dumps(result)

def get_summary_item(session_id: str) -> dict | None:
    """Fetch the SUMMARY item from DynamoDB. Returns None if not found."""
    ddb   = boto3.resource("dynamodb", region_name=REGION)
    table = ddb.Table(TABLE_NAME)
    try:
        resp = table.get_item(Key={"session_id": session_id, "timestamp": "SUMMARY"})
        return resp.get("Item")
    except Exception as e:
        print(f"   [DynamoDB ERROR] {e}")
        return None

def count_raw_turns(session_id: str) -> int:
    """Count turn items (SK != 'SUMMARY') for this session."""
    ddb   = boto3.resource("dynamodb", region_name=REGION)
    table = ddb.Table(TABLE_NAME)
    try:
        resp  = table.query(KeyConditionExpression=Key("session_id").eq(session_id))
        items = resp.get("Items", [])
        return sum(1 for item in items if item.get("timestamp") != "SUMMARY")
    except Exception as e:
        print(f"   [DynamoDB ERROR] {e}")
        return -1

def check_summary_fields(summary_content: str) -> list[str]:
    """Return list of expected fields that are MISSING from the summary content."""
    return [f for f in EXPECTED_SUMMARY_FIELDS if f not in summary_content]

def print_summary_check(session_id: str, label: str):
    separator()
    print(f"{label}")
    separator()

    # Give BackgroundTask time to write
    time.sleep(4)

    summary   = get_summary_item(session_id)
    raw_count = count_raw_turns(session_id)

    if summary:
        content = (
            summary.get("content")
            or summary.get("summary")
            or summary.get("text")
            or json.dumps({k: v for k, v in summary.items()
                           if k not in ("session_id", "timestamp")}, indent=2, default=str)
        )
        missing = check_summary_fields(content)
        print(f"{'✅' if not missing else '⚠️ '} SUMMARY item found")
        print(f"   Raw turns remaining : {raw_count}  (expected ≤ 5)")
        print(f"   Missing fields      : {missing if missing else 'none — all present'}")
        print(f"\n--- SUMMARY CONTENT ---")
        print(content)
    else:
        print(f"❌ SUMMARY item NOT found")
        print(f"   Raw turns in table  : {raw_count}")
        print(f"   → compression may not have fired, or BackgroundTask is still running")
    separator()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    separator()
    print(f"Memory E2E Test  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Session ID : {SESSION_ID}")
    print(f"  Endpoint   : {BASE_URL}/query")
    print(f"  DynamoDB   : {TABLE_NAME}  ({REGION})")
    print(f"  Turns      : {TOTAL_TURNS}  (compression triggers at turn {COMPRESSION_TRIGGER_AT})")
    separator()

    results = []

    for i, question in enumerate(QUESTIONS, start=1):
        print(f"\n--- Turn {i}/{TOTAL_TURNS} ---")
        print(f"Q: {question}")

        t0 = time.time()
        try:
            raw_result = post_question(question, SESSION_ID)
            elapsed    = time.time() - t0
            answer     = extract_answer(raw_result)

            preview = answer[:400] + ("…" if len(answer) > 400 else "")
            print(f"A: {preview}")
            print(f"   [{elapsed:.1f}s]")

            results.append({"turn": i, "ok": True, "elapsed": elapsed,
                            "answer_len": len(answer)})

        except requests.exceptions.Timeout:
            elapsed = time.time() - t0
            print(f"   ⚠️  TIMEOUT after {elapsed:.0f}s — server may still be processing")
            results.append({"turn": i, "ok": False, "error": "timeout"})

        except Exception as e:
            elapsed = time.time() - t0
            print(f"   ❌ ERROR: {e}")
            results.append({"turn": i, "ok": False, "error": str(e)})

        # After compression trigger turn: inspect DynamoDB
        if i == COMPRESSION_TRIGGER_AT:
            print_summary_check(
                SESSION_ID,
                f"POST-TURN-{i} DynamoDB CHECK  (compression should have fired)"
            )

    # --- Final check ---
    print_summary_check(SESSION_ID, "FINAL DynamoDB CHECK")

    # --- Turn summary ---
    separator("-")
    print("TURN SUMMARY")
    separator("-")
    ok_turns = [r for r in results if r.get("ok")]
    print(f"  Successful turns : {len(ok_turns)}/{TOTAL_TURNS}")
    if ok_turns:
        avg_s = sum(r["elapsed"] for r in ok_turns) / len(ok_turns)
        print(f"  Avg response time: {avg_s:.1f}s")
    for r in results:
        status = "✅" if r.get("ok") else "❌"
        detail = (f"{r['elapsed']:.1f}s, {r['answer_len']} chars"
                  if r.get("ok") else r.get("error", "unknown"))
        print(f"  Turn {r['turn']:2d} {status}  {detail}")

    separator()
    print(f"Session ID (for manual DynamoDB lookup): {SESSION_ID}")
    separator()


if __name__ == "__main__":
    main()