#!/usr/bin/env python3
"""
dump_session.py
===============
Prints all DynamoDB items for a given session_id in a readable format.
Sort key attribute is 'timestamp'; SUMMARY item uses value "SUMMARY".

Usage:
  python dump_session.py <session_id>
  python dump_session.py                  # prompts for session_id
"""

import os
import sys
import json
import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal

TABLE_NAME = os.getenv("DYNAMO_TABLE", "trade-platform-dev-conversations")
REGION     = os.getenv("AWS_REGION", "us-east-2")
SK_ATTR    = "timestamp"
SUMMARY_VAL = "SUMMARY"

def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

def sep(char="=", width=62):
    print(char * width)

def fetch_all_items(session_id: str) -> list:
    ddb   = boto3.resource("dynamodb", region_name=REGION)
    table = ddb.Table(TABLE_NAME)
    resp  = table.query(KeyConditionExpression=Key("session_id").eq(session_id))
    return resp.get("Items", [])

def print_item(item: dict, label: str):
    sep("-")
    print(f"  {label}")
    sep("-")
    skip = {"session_id", SK_ATTR}
    for k, v in item.items():
        if k in skip:
            continue
        if isinstance(v, str) and len(v) > 80:
            print(f"  {k}:")
            for line in v.split("\n"):
                print(f"    {line}")
        else:
            print(f"  {k}: {json.dumps(v, default=decimal_default)}")

def main():
    session_id = sys.argv[1] if len(sys.argv) > 1 else input("Session ID: ").strip()
    if not session_id:
        print("No session ID provided.")
        sys.exit(1)

    sep()
    print(f"Session : {session_id}")
    print(f"Table   : {TABLE_NAME}  ({REGION})")
    sep()

    items = fetch_all_items(session_id)
    if not items:
        print("No items found. Check session_id, table name, and region.")
        sys.exit(1)

    summary_items = [i for i in items if i.get(SK_ATTR) == SUMMARY_VAL]
    turn_items    = sorted(
        [i for i in items if i.get(SK_ATTR) != SUMMARY_VAL],
        key=lambda x: x.get(SK_ATTR, "")
    )

    print(f"  Total items   : {len(items)}")
    print(f"  SUMMARY       : {len(summary_items)}")
    print(f"  Raw turns     : {len(turn_items)}  (pairs: {len(turn_items)//2})")
    sep()

    # --- SUMMARY ---
    if summary_items:
        s = summary_items[0]
        tc = s.get("turn_count", "?")
        ua = s.get("updated_at", "?")
        print(f"\nSUMMARY  (turn_count={tc}, updated_at={ua})")
        print_item(s, f"{SK_ATTR} = {SUMMARY_VAL}")
    else:
        print("\nNo SUMMARY item — compression hasn't fired yet.")

    # --- Raw turns ---
    if turn_items:
        print(f"\nRAW TURNS  ({len(turn_items)} messages, {len(turn_items)//2} Q+A pairs remaining)")
        for item in turn_items:
            role = item.get("role", "?")
            ts   = item.get(SK_ATTR, "?")
            print_item(item, f"role={role}  |  {SK_ATTR}={ts}")
    else:
        print("\n(no raw turn items)")

    sep()
    print(f"Done. {len(items)} items total.")
    sep()

if __name__ == "__main__":
    main()