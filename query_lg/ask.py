"""
ask.py — run any question against the real compiled graph, with
--verbose for per-node detail and --trace for the full whiteboard.
"""

import argparse
import asyncio
import uuid

from query_lg.graph import compiled_graph
from query_lg.state import build_initial_state
from langgraph.types import Command


def summarize_value(v, max_len=200):
    if v is None:
        return "None"
    if isinstance(v, dict):
        if not v:
            return "{}"
        parts = []
        for k, item in v.items():
            item_repr = getattr(item, "answer", None) or str(item)
            parts.append(f"{k}: {str(item_repr)[:80]}")
        return "{ " + " | ".join(parts) + " }"
    s = str(v)
    return s[:max_len] + ("..." if len(s) > max_len else "")


def print_verbose_node(nid, nr):
    print(f"\n  ── {nid} ({nr.agent_type}) ──")
    print(f"     depends_on: {nr.depends_on}")
    print(f"     retried: {nr.retried}")
    print(f"     tool_calls ({len(nr.tool_calls)}):")
    for tc in nr.tool_calls:
        print(f"       • {tc.name}({tc.args})")
        if tc.reasoning_before_call:
            print(f"         reasoning: {tc.reasoning_before_call[:150]}")
        preview = tc.result_full[:300].replace(chr(10), " ")
        print(f"         result: {preview}{'...' if len(tc.result_full) > 300 else ''}")

    gate = nr.grounding_gate
    print(f"     grounding_gate: passed={gate.passed} "
          f"(attribution={gate.attribution_passed}, inversion={gate.inversion_passed})")
    if gate.attribution_failures:
        print(f"       attribution_failures:")
        for f in gate.attribution_failures:
            print(f"         - {f}")
    if gate.inversion_failures:
        print(f"       inversion_failures ({len(gate.inversion_failures)}, non-blocking, first 5):")
        for f in gate.inversion_failures[:5]:
            print(f"         - {f}")

    print(f"     critique: passed={nr.critique.passed}")
    if nr.critique.issues:
        for issue in nr.critique.issues:
            print(f"       issue: {issue}")
    if nr.retried and nr.critique.retry_guidance:
        print(f"       retry_guidance (why this retried): {nr.critique.retry_guidance}")

    print(f"     final answer: {nr.answer}")


async def run_traced(state_or_command, config):
    prev = {}
    step = 0
    final = {}
    pending_updates = []

    async for mode, chunk in compiled_graph.astream(state_or_command, config=config,
                                                       stream_mode=["updates", "values"]):
        if mode == "updates":
            pending_updates.extend(chunk.items())
            continue

        step += 1
        print(f"\n{'=' * 70}")
        if pending_updates:
            node_names = [n for n, _ in pending_updates]
            print(f"STEP {step} — node(s) that just ran: {node_names}")
            for node_name, update in pending_updates:
                print(f"  {node_name} wrote: {summarize_value(update)}")
        else:
            print(f"STEP {step} — initial state (before any node has run)")
        print(f"{'=' * 70}")

        changed = [k for k, v in chunk.items() if prev.get(k) != v]
        if changed:
            print(f"Whiteboard fields now different: {changed}")
            for k in changed:
                print(f"  {k} = {summarize_value(chunk[k])}")

        pending_updates = []
        prev = dict(chunk)
        final = chunk

    return final


async def main():
    parser = argparse.ArgumentParser(description="Ask query_lg's real compiled graph a question.")
    parser.add_argument("question", nargs="?", help="The question to ask")
    parser.add_argument("--model", default=None,
                         help="Provider:model string, e.g. anthropic:claude-haiku-4-5-20251001.")
    parser.add_argument("--thread-id", default=None,
                         help="Reuse a thread-id to resume a paused clarify, or continue a conversation.")
    parser.add_argument("--resume", default=None,
                         help="Answer to a pending clarify question — requires --thread-id.")
    parser.add_argument("--verbose", "-v", action="store_true",
                         help="Print full tool_calls, reasoning, and grounding-gate detail per node.")
    parser.add_argument("--trace", "-t", action="store_true",
                         help="Stream the full whiteboard after every node, labeled with which node ran.")
    args = parser.parse_args()

    thread_id = args.thread_id or str(uuid.uuid4())[:8]
    configurable = {"thread_id": thread_id}
    if args.model:
        configurable["model_name"] = args.model
    config = {"configurable": configurable}

    if args.resume:
        print(f"Resuming thread {thread_id} with: {args.resume}\n")
        payload = Command(resume=args.resume)
    else:
        if not args.question:
            parser.error("a question is required unless using --resume")
        print(f"[thread_id: {thread_id}]  Question: {args.question}\n")
        payload = build_initial_state(args.question, thread_id)

    if args.trace:
        result = await run_traced(payload, config)
        print(f"\n{'=' * 70}\nFINAL STATE REACHED\n{'=' * 70}")
    else:
        result = await compiled_graph.ainvoke(payload, config=config)

    if "__interrupt__" in result:
        print("=== PAUSED — needs clarification ===")
        print(result["__interrupt__"])
        print(f"\nResume with:\n  python -m query_lg.ask --thread-id {thread_id} --resume \"your answer here\"")
        return

    dag = result.get("dag") or {}
    print("\n=== dag ===")
    for nid, spec in dag.items():
        print(f"  {nid}: agent={spec.agent}, depends_on={spec.depends_on}, reason={spec.reason!r}")

    node_results = result.get("node_results") or {}
    if node_results:
        print("\n=== node_results ===")
        for nid, nr in node_results.items():
            if args.verbose:
                print_verbose_node(nid, nr)
            else:
                print(f"  {nid}: retried={nr.retried}, gate.passed={nr.grounding_gate.passed}, "
                      f"critique.passed={nr.critique.passed}, tool_calls={len(nr.tool_calls)}")

    print("\n=== final_answer ===")
    print(result.get("final_answer"))

    crit = result.get("synthesis_critique")
    if crit is not None:
        synthesis_retried = result.get("synthesis_retried")
        print(f"\n=== synthesis reflexion ===")
        print(f"  retried: {synthesis_retried}")
        gate = result.get("synthesis_grounding_gate")
        if gate is not None:
            print(f"  grounding_gate: passed={gate.passed} "
                  f"(attribution={gate.attribution_passed}, inversion={gate.inversion_passed})")
            if args.verbose and gate.attribution_failures:
                print(f"    attribution_failures:")
                for f in gate.attribution_failures:
                    print(f"      - {f}")
        print(f"  critique: passed={crit.passed}")
        if args.verbose and crit.issues:
            for issue in crit.issues:
                print(f"    issue: {issue}")
        if synthesis_retried and crit.retry_guidance:
            print(f"  retry_guidance (why this retried): {crit.retry_guidance}")

    injection = result.get("injection_check")
    if injection is not None:
        print("\n=== injection_check ===")
        print(injection)


if __name__ == "__main__":
    asyncio.run(main())