"""
Direct planner_node tests against a live model — checks the two things
just ported: content-routing trigger phrases (Fed comms -> filings, not
macro) and implicit date resolution ("this quarter", "recently").
"""
import asyncio
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