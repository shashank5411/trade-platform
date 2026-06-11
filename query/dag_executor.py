"""
DAG Executor — runs agents in dependency order, parallelizing where safe.

Takes a DAG plan from the planner and executes it:
  - Agents with no dependencies run in parallel (Round 1)
  - Agents whose dependencies are complete run in parallel (Round N)
  - Each agent receives original question + outputs from its dependencies
  - Final synthesis combines all agent outputs into one coherent answer
"""

import os
import asyncio
import anthropic

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from query.registry import get_agent

ENV = os.environ.get("ENV", "dev")

SYNTH_MODEL = (
    "claude-sonnet-4-6"
    if ENV == "prod"
    else "claude-haiku-4-5-20251001"
)
import os; get_anthropic_key = lambda: os.environ["ANTHROPIC_API_KEY"]

client = anthropic.Anthropic(api_key=get_anthropic_key())
#client = anthropic.Anthropic()


async def _run_agent_async(
    agent_name: str,
    question:   str,
    history:    list,
    verbose:    bool,
    session_id: str = None,
) -> tuple:
    """Run a single agent in a thread pool (non-blocking)."""
    agent = get_agent(agent_name)
    if not agent:
        return agent_name, f"Agent '{agent_name}' not found in registry."

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: agent.run(
                question,
                history=history,
                verbose=verbose,
                session_id=session_id,
            )
        )
        return agent_name, result
    except Exception as e:
        return agent_name, f"Agent '{agent_name}' failed: {e}"


async def execute(
    question:   str,
    dag:        dict,
    history:    list = None,
    verbose:    bool = True,
    session_id: str  = None,
) -> str:
    """
    Execute a DAG plan and return synthesized answer.

    Args:
        question: Original user question
        dag:      DAG plan from planner {agent: {depends_on, reason}}
        history:  Conversation history for context
        verbose:  Print execution progress
    """
    history   = history or []
    completed = {}
    remaining = set(dag.keys())

    # Single agent — no synthesis needed
    if len(dag) == 1:
        agent_name = list(dag.keys())[0]
        _, answer  = await _run_agent_async(
            agent_name, question, history, verbose, session_id
        )
        return answer

    # Multi-agent — execute in rounds
    round_num = 1
    while remaining:
        ready = [
            name for name in remaining
            if all(dep in completed
                   for dep in dag[name].get("depends_on", []))
        ]

        if not ready:
            if verbose:
                print(f"[Executor] DAG deadlock — forcing remaining: "
                      f"{remaining}")
            ready = list(remaining)

        if verbose:
            print(f"\n[Executor] Round {round_num} — parallel: {ready}")

        tasks = []
        for name in ready:
            dep_answers = [
                f"[{dep.upper()} ANALYSIS]\n{completed[dep]}"
                for dep in dag[name].get("depends_on", [])
                if dep in completed
            ]
            enriched = (
                f"{question}\n\n"
                f"Context from prior analysis:\n"
                + "\n\n".join(dep_answers)
                if dep_answers else question
            )
            tasks.append(_run_agent_async(name, enriched, history, verbose, session_id))

        results = await asyncio.gather(*tasks)
        for agent_name, answer in results:
            completed[agent_name] = answer
            remaining.discard(agent_name)

        round_num += 1

    if verbose:
        print(f"\n[Executor] Synthesizing {len(completed)} agent outputs...")

    agent_outputs = "\n\n".join([
        f"[{name.upper()} ANALYSIS]\n{answer}"
        for name, answer in completed.items()
    ])

    synthesis_prompt = (
        f"Original question: {question}\n\n"
        f"{agent_outputs}\n\n"
        f"Synthesize the above analyses into a single coherent, "
        f"well-structured answer. Integrate the insights naturally — "
        f"do not just concatenate them. Lead with the most important "
        f"finding and support it with data from all relevant analyses.\n\n"
        f"Synthesize ONLY from the agent outputs provided above. "
        f"Do not introduce facts, prices, dates, or causal explanations "
        f"that are not explicitly present in the agent outputs. "
        f"If agents have gaps or disagree, state that explicitly rather "
        f"than filling in from background knowledge."
    )

    response = client.messages.create(
        model=SYNTH_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": synthesis_prompt}],
    )
    return response.content[0].text
