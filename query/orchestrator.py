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
    history:    list = None,
    verbose:    bool = True,
    session_id: str  = None,
) -> str:
    """
    Route question through DAG planner and executor.
    Entry point for agent.py — synchronous wrapper around async execute.
    """
    history = history or []

    dag = plan(question, history=history, verbose=verbose)

    return asyncio.run(execute(
        question=question,
        dag=dag,
        history=history,
        verbose=verbose,
        session_id=session_id,
    ))
