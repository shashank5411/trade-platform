"""
DAG Executor — runs agents in dependency order, parallelizing where safe.

Takes a DAG plan from the planner and executes it:
  - Agents with no dependencies run in parallel (Round 1)
  - Agents whose dependencies are complete run in parallel (Round N)
  - Each agent receives original question + outputs from its dependencies
  - Final synthesis combines all agent outputs into one coherent answer
"""

import os
import re
import asyncio
import anthropic

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from query.registry import get_agent
from query.telemetry import Trace

ENV = os.environ.get("ENV", "dev")

SYNTH_MODEL = (
    "claude-sonnet-4-6"
    if ENV == "prod"
    else "claude-haiku-4-5-20251001"
)
from query.config import get_client
client = get_client()

ROLE_DESCRIPTIONS = {
    "market":    "stock prices, price performance, and market data",
    "macro":     "economic indicators and macro data",
    "filings":   "SEC filings, qualitative documents, and Fed communications",
    "sentiment": "insider trades and news sentiment",
}

SYNTHESIS_SYSTEM = (
    "Synthesize using the data that agents returned. If an agent returned "
    "partial data or flagged missing information, incorporate what is "
    "available and note the gap in one sentence — do not ask the user for "
    "clarification or block the answer on missing data. The user can "
    "follow up if needed.\n\n"
    "Some questions deliberately route to two agents because the SAME "
    "real-world concept has two distinct, equally valid data sources "
    "(e.g. a commodity's futures/contract price vs. its official spot "
    "price index). When this happens, the two numbers being different is "
    "EXPECTED and CORRECT, not an error, inconsistency, or sign of stale "
    "data. Present both clearly labeled by source and what each measures, "
    "without speculating about why they differ or implying one might be "
    "wrong. Only call something a genuine discrepancy if both agents were "
    "asked for the literal same series/ticker and returned different "
    "values for it.\n\n"
    "When two agents return genuinely independent signals being asked "
    "about together (e.g. a macro indicator and an insider/news signal, "
    "or any 'does X relate to Y' question), present each signal factually "
    "side by side and stop there. Do NOT construct a causal or predictive "
    "narrative connecting them — no 'this could translate into', 'this "
    "portends', 'this suggests management is anticipating', or similar "
    "forward-looking inference that isn't explicitly stated in either "
    "agent's output. If the agent outputs themselves don't draw a "
    "connection, synthesis should not invent one. State what each signal "
    "shows, note explicitly that any relationship between them is "
    "speculative if the user wants to draw one, and let the user "
    "interpret — this rule applies even if the connection seems "
    "plausible or well-reasoned to you."
    "This restriction on inventing causal/explanatory narrative applies "
    "to YOUR synthesis too, not just to individual agent outputs. Even "
    "when combining 3+ independent signals, do not construct phrases "
    "like 'this reflects', 'driven by', 'this is not contradiction — it "
    "reflects', or similar explanatory bridges between signals unless "
    "at least one agent's own output explicitly stated that connection. "
    "With 3+ signals the temptation to build a unifying narrative is "
    "stronger — resist it equally regardless of how many signals you "
    "are combining."
)



async def _run_agent_async(
    node_id:    str,
    agent_type: str,
    question:   str,
    history:    list,
    verbose:    bool,
    session_id: str = None,
) -> tuple:
    """Run a single DAG node in a thread pool (non-blocking).

    node_id identifies this specific step in the plan (e.g. "market_2"),
    distinct from agent_type, which is the specialist that executes it
    (e.g. "market"). The same agent_type can run under multiple node_ids
    in one DAG when a question needs the same specialist twice with
    different upstream context (see planner.py's multi-hop chain rules).
    """
    agent = get_agent(agent_type)
    if not agent:
        return node_id, f"Agent type '{agent_type}' not found in registry."

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: agent.run(
                question,
                history=history,
                verbose=verbose,
                session_id=session_id,
                node_id=node_id,
            )
        )
        return node_id, result
    except Exception as e:
        return node_id, f"Node '{node_id}' (agent '{agent_type}') failed: {e}"


_FIGURE_PATTERN = re.compile(r'(\$\s?\d[\d,]*\.?\d*|\d+(?:\.\d+)?%|\b\d+\.\d+\b)')


def _check_unattributed_figures(node_outputs: dict, dag: dict) -> list:
    """
    Soft heuristic: for every node with non-empty depends_on, flag if its
    answer contains a numeric figure (%, $, or decimal number) but no
    '[from prior step:' bracket anywhere in the text. This does not prove
    the node restated upstream data uncited — a node can legitimately have
    its own freshly-fetched figures and nothing borrowed to attribute — so
    treat every entry returned here as a warning to review, not a defect.
    """
    warnings = []
    for node_id, node in dag.items():
        if not node.get("depends_on"):
            continue
        answer = node_outputs.get(node_id, "")
        if not answer or "[from prior step:" in answer:
            continue
        if _FIGURE_PATTERN.search(answer):
            warnings.append(
                f"node '{node_id}' (depends_on {node['depends_on']}) "
                f"contains numeric figures but no '[from prior step:' "
                f"attribution bracket — verify it isn't silently "
                f"restating upstream data as its own."
            )
    return warnings


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
    # Single node — no synthesis needed
    if len(dag) == 1:
        node_id    = list(dag.keys())[0]
        agent_type = dag[node_id].get("agent", node_id)  # fallback for old-style DAGs
        _, answer  = await _run_agent_async(
            node_id, agent_type, question, history, verbose, session_id
        )
        return answer

    # Multi-agent — execute in rounds
    # Multi-agent — execute in rounds
    round_num = 1
    while remaining:
        ready = [
            node_id for node_id in remaining
            if all(dep in completed
                   for dep in dag[node_id].get("depends_on", []))
        ]

        if not ready:
            if verbose:
                print(f"[Executor] DAG deadlock — forcing remaining: "
                      f"{remaining}")
            ready = list(remaining)

        if verbose:
            print(f"\n[Executor] Round {round_num} — parallel: {ready}")

        tasks = []
        for node_id in ready:
            node = dag[node_id]
            agent_type = node.get("agent", node_id)  # fallback for old-style DAGs

            dep_answers = [
                f"[{dag[dep].get('agent', dep).upper()} ANALYSIS ({dep}) — "
                f"VERIFIED BY {dag[dep].get('agent', dep).upper()}, NOT BY YOU]\n"
                f"{completed[dep]}"
                for dep in node.get("depends_on", [])
                if dep in completed
            ]
            if dep_answers:
                # Sequential agent: enrich with prior outputs.
                # IMPORTANT: this prior analysis was verified by a DIFFERENT
                # specialist via ITS OWN tool calls — not by the agent
                # receiving it now. The header makes that boundary explicit
                # so the receiving agent doesn't restate the other agent's
                # figures as if it had fetched them itself.
                enriched = (
                    f"{question}\n\n"
                    f"Context from prior analysis (verified by a different "
                    f"specialist via their own tool calls — you have NOT "
                    f"independently verified these figures yourself):\n"
                    f"MANDATORY CITATION FORMAT: any numeric figure or "
                    f"specific claim you reference from the context below "
                    f"MUST be wrapped exactly as "
                    f"[from prior step: the actual figure or claim], e.g. "
                    f"'crude fell to [from prior step: $58.20/barrel]'. "
                    f"Only figures YOU retrieved via your own tool calls "
                    f"this turn may appear unbracketed. Paraphrasing a "
                    f"prior figure in plain text — even with attribution "
                    f"language like 'per the macro analysis' — does NOT "
                    f"satisfy this requirement; the literal bracket markup "
                    f"is required every time you reference a borrowed "
                    f"figure or claim.\n\n"
                    + "\n\n".join(dep_answers)
                )
            elif len(ready) > 1:
                # Parallel node: inject role-scoping hint so it stays in its lane
                role_desc = ROLE_DESCRIPTIONS.get(agent_type, agent_type)
                enriched = (
                    f"{question}\n\n"
                    f"[Your role in this query: focus on {role_desc} only. "
                    f"Other specialist agents are handling the remaining parts "
                    f"in parallel. Do not ask for clarification about data "
                    f"outside your domain — just answer your part.]"
                )
            else:
                enriched = question
            tasks.append(_run_agent_async(node_id, agent_type, enriched, history, verbose, session_id))

        results = await asyncio.gather(*tasks)
        for node_id, answer in results:
            completed[node_id] = answer
            remaining.discard(node_id)
            if verbose:
                print(f"\n[Executor] --- raw output: {node_id} ---\n{answer}\n"
                      f"[Executor] --- end {node_id} ---")

        round_num += 1

    attribution_warnings = _check_unattributed_figures(completed, dag)
    if attribution_warnings:
        if verbose:
            for w in attribution_warnings:
                print(f"[Executor] ⚠️  ATTRIBUTION WARNING: {w}")
        warn_trace = Trace(
            session_id=session_id or "no-session",
            agent="dag_executor",
            question=question,
            model="n/a",
        )
        warn_trace.record_attribution_warnings(attribution_warnings)
        warn_trace.flush()

    if verbose:
        print(f"\n[Executor] Synthesizing {len(completed)} agent outputs...")

    agent_outputs = "\n\n".join([
        f"[{dag[node_id].get('agent', node_id).upper()} ANALYSIS ({node_id})]\n{answer}"
        for node_id, answer in completed.items()
    ])

    synthesis_prompt = (
        f"Original question: {question}\n\n"
        f"{agent_outputs}\n\n"
        f"Synthesize the above analyses into a single coherent, "
        f"well-structured answer. Lead with the most important finding "
        f"and support it with data from all relevant analyses.\n\n"
        f"IMPORTANT — how to integrate multiple agent outputs: 'integrate "
        f"naturally' does NOT mean inventing a causal or explanatory link "
        f"between independent signals. If the agents' outputs describe "
        f"genuinely separate things (e.g. one macro indicator, one "
        f"sentiment signal, with no stated connection between them), your "
        f"job is clean organization and clear labeling — NOT building a "
        f"narrative that one explains, drives, or matters more than the "
        f"other. Only connect two findings causally if at least one "
        f"agent's own output already stated that connection explicitly — "
        f"you are not permitted to add a new causal or explanatory link "
        f"that isn't already present in the source text.\n\n"
        f"Synthesize ONLY from the agent outputs provided above. "
        f"Do not introduce facts, prices, dates, or causal explanations "
        f"that are not explicitly present in the agent outputs. "
        f"If agents have gaps in coverage, state that explicitly rather "
        f"than filling in from background knowledge. If agents return "
        f"different numbers for the same real-world concept from "
        f"genuinely different sources (e.g. futures price vs. official "
        f"spot price), present both as complementary labeled figures — "
        f"do not call this a disagreement or discrepancy unless the "
        f"sources were supposed to be measuring the identical series."
    )

    response = client.messages.create(
        model=SYNTH_MODEL,
        max_tokens=2000,
        system=SYNTHESIS_SYSTEM,
        messages=[{"role": "user", "content": synthesis_prompt}],
    )
    return response.content[0].text
