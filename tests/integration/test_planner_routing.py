"""
Direct planner_node tests against a live model — checks the two things
just ported: content-routing trigger phrases (Fed comms -> filings, not
macro) and implicit date resolution ("this quarter", "recently").

INTEGRATION/LIVE: calls the real model via planner_node and runs at
import time (no pytest fixtures) — not meant for CI. Run manually:
python tests/integration/test_planner_routing.py
"""
import os
import sys
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from query_lg.planner import planner_node
from query_lg.state import GraphState

QUESTIONS = [
    # Should route to filings, NOT macro (Fed communications trigger phrase)
    "What has the Fed said recently about inflation risks?",
    # Should route to macro, NOT filings (numeric rate data, not commentary)
    "What was the Fed funds rate in 2022?",
    # Should resolve "this quarter" without asking for clarification
    "How has AAPL performed this quarter?",
    # Should resolve "recently" to a 30-day window, route to sentiment
    "Have insiders been buying AAPL recently?",
]

async def main():
    for q in QUESTIONS:
        state = GraphState(session_id="test", question=q)
        result = await planner_node(state, config=None)
        print(f"Q: {q}")
        print(f"  dag: { {nid: spec.agent for nid, spec in result.get('dag', {}).items()} }")
        for nid, spec in result.get("dag", {}).items():
            print(f"  {nid} reason: {spec.reason}")
        print()

asyncio.run(main())