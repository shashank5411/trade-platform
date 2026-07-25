"""
Live smoke test — runs the REAL compiled graph against your real
ANTHROPIC_API_KEY, no scripted/fake models (config carries no
model_factory, so models.py's resolve_model() falls through to a real
init_chat_model call).

This confirms the GRAPH MECHANICS work against a real model's actual
output — JSON parsing, tool-calling shape, retry behavior. It does NOT
confirm data accuracy: every tool in tools.py is still a stub returning
a fixed string regardless of input. That's the next milestone once this
passes.

Run from the repo root, with ANTHROPIC_API_KEY set:
    python -m query_lg.test_live_run
"""

import asyncio
from query_lg.graph import compiled_graph
from query_lg.state import build_initial_state


async def main():
    question = "What's AAPL's price on 2026-07-15?"
    state = build_initial_state(question, "live-test-1")
    config = {"configurable": {"thread_id": "live-test-1"}}  # no model_factory -> real API

    print(f"Question: {question}\n")
    result = await compiled_graph.ainvoke(state, config=config)

    print("=== final_answer ===")
    print(result["final_answer"])
    print()

    print("=== dag the REAL planner produced ===")
    print({k: v.agent for k, v in result["dag"].items()} if result.get("dag") else result.get("dag"))
    print()

    print("=== node_results (retry/grounding/critique verdicts) ===")
    for nid, nr in result["node_results"].items():
        print(f"  {nid}: retried={nr.retried}, "
              f"gate.passed={nr.grounding_gate.passed}, "
              f"critique.passed={nr.critique.passed}")
        if nr.grounding_gate.attribution_failures:
            print(f"    attribution_failures: {nr.grounding_gate.attribution_failures}")
    print()

    print("=== injection_check ===")
    print(result.get("injection_check", "(not set — expected on the decline/clarify sentinel paths, "
                                          "which deliberately skip injection_check entirely)"))
    print()

    print("=== REMINDER: tool DATA is still stub content — this validated ===")
    print("    the LLM/graph mechanics, not real market data accuracy.")


if __name__ == "__main__":
    asyncio.run(main())