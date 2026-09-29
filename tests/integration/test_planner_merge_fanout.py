"""
Direct planner_node tests against a live model — checks the DAG SHAPE
(dependency structure), not just agent selection, for the merge/fan-out
patterns just ported.

INTEGRATION/LIVE: calls the real model via planner_node and runs at
import time (no pytest fixtures) — not meant for CI. Run manually:
python tests/integration/test_planner_merge_fanout.py
"""
import os
import sys
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from query_lg.planner import planner_node
from query_lg.state import GraphState

QUESTIONS = [
    # MERGE case: two signals -> market judges magnitude. market_1
    # should depend_on BOTH macro_1 and sentiment_1, not run parallel.
    "Unemployment is rising and JPM insiders are selling — is the size "
    "of JPM's stock price move consistent with the size of those two "
    "signals, or has the price moved more or less than they'd suggest?",

    # FAN-OUT case: one shared context -> two downstream analyses.
    # Both downstream nodes should depend_on the SAME upstream node.
    "Given the Fed's recent rate stance, how did bank stocks react and "
    "what did insiders do?",

    # Should NOT collapse into flat parallel or misroute merge point to filings
    "Does JPM's 10-K explain why insiders were selling right before the "
    "stock dropped 8%?",
]

async def main():
    for q in QUESTIONS:
        state = GraphState(session_id="test", question=q)
        result = await planner_node(state, config=None)
        print(f"Q: {q}")
        for nid, spec in result.get("dag", {}).items():
            print(f"  {nid}: agent={spec.agent} depends_on={spec.depends_on}")
            print(f"    reason: {spec.reason}")
        print()

asyncio.run(main())