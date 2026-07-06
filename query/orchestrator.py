"""
Orchestrator — DAG-based multi-agent routing.

Uses LLM planner to produce optimal execution DAG,
then async executor runs agents in dependency order
with maximum parallelism.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from query.planner      import plan
from query.dag_executor import execute


def run(
    question:   str,
    context:    dict = None,
    verbose:    bool = True,
    session_id: str  = None,
) -> tuple:
    """
    Route question through DAG planner and executor.
    Entry point for agent.py and server.py — synchronous wrapper around async execute.

    Returns (final_answer: str, node_tool_calls: dict) where node_tool_calls maps
    node_id -> list of tool-call records (Trace.tools_called structure, including
    result_full) for use by chart_agent.build_charts().

    context dict (from memory.load_context):
        { "summary": str|None, "context_note": str|None, "recent_turns": list }
    Pass context=None for no-memory mode (--no-memory CLI flag).
    """
    if context is None:
        context = {"summary": None, "context_note": None, "recent_turns": []}

    summary      = context.get("summary")
    context_note = context.get("context_note")
    recent_turns = context.get("recent_turns", [])

    # Prepend the 1-2 sentence context note so sub-agents know what the
    # user has been exploring, without bloating their context with the
    # full structured summary.
    enriched_question = question
    if context_note:
        enriched_question = f"[Session context: {context_note}]\n\n{question}"

    dag = plan(enriched_question, history=recent_turns, verbose=verbose,
               summary=summary)

    final_answer, node_tool_calls = asyncio.run(execute(
        question=enriched_question,
        dag=dag,
        history=recent_turns,
        verbose=verbose,
        session_id=session_id,
        summary=summary,
    ))
    return final_answer, node_tool_calls
