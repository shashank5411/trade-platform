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
from query.reflexion import (
    critique_synthesis, CAVEAT, _needs_reflexion_despite_length,
    check_injection_provenance, INJECTION_CAVEAT,
)

ENV = os.environ.get("ENV", "dev")

SYNTH_MODEL = (
    "claude-sonnet-4-6"
    if ENV == "prod"
    else "claude-haiku-4-5-20251001"
)
from query.config import get_client
client = get_client()

# Shared skip-threshold for both synthesis-level reflexion and the
# injection-provenance check below — same value, same reasoning as
# reflexion.py's per-agent REFLEXION_MIN_WORDS (critic overhead not worth
# it for short answers, unless _needs_reflexion_despite_length forces it).
SYNTH_REFLEXION_MIN_WORDS = 200

ROLE_DESCRIPTIONS = {
    "market":    "stock prices, price performance, and market data",
    "macro":     "economic indicators and macro data",
    "filings":   "SEC filings, qualitative documents, and Fed communications",
    "sentiment": "insider trades and news sentiment",
}

# Fixed decline message for planner.py's "decline" sentinel — a question
# with zero financial/market/economic/company-specific component. Not
# templated per-question (unlike agent answers) since there's nothing to
# vary it on — no tool call happened, so there's no data to incorporate.
SCOPE_DECLINE_MESSAGE = (
    "I'm a financial research assistant — I can help with market prices, "
    "economic indicators, SEC filings, Fed communications, insider trades, "
    "and news sentiment, but this question doesn't have a financial, "
    "market, economic, or company-specific component I can address. "
    "Happy to help if you'd like to ask something in those areas instead."
)

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


def _run_injection_check(
    answer: str, question: str, node_outputs: dict,
    verbose: bool, session_id: str, trace: Trace = None,
) -> str:
    """
    Runs reflexion.check_injection_provenance() and appends INJECTION_CAVEAT
    if suspected. Shared by execute()'s single-agent early-return path AND
    its multi-agent synthesis tail — single-agent DAGs return before ever
    reaching the synthesis tail below, but a single agent's answer is
    exactly as exposed to tool-result injection as a synthesized one, so
    this check needs to run on both paths, not just the literal
    "synthesis" case the name might suggest.

    Caller is responsible for the word-count-or-forced skip-gate decision
    (mirrors critique_synthesis()'s call sites — this function is only
    ever called from an already-decided-not-to-skip branch).

    trace: if the caller already has a Trace object for this query (the
    multi-agent tail does — the same one recording synthesis reflexion),
    pass it in so the injection result lands on that SAME record instead
    of a second, redundant one. If None, a fresh dag_executor trace is
    created and flushed here (the single-agent path has no existing trace
    to attach to).
    """
    result = check_injection_provenance(
        answer, question, node_outputs, client, SYNTH_MODEL, verbose
    )
    suspected = result.get("injection_suspected", False)

    owns_trace = trace is None
    if owns_trace:
        trace = Trace(
            session_id=session_id or "no-session",
            agent="dag_executor",
            question=question,
            model=SYNTH_MODEL,
        )
    trace.record_injection_check(suspected)
    if owns_trace:
        trace.flush()

    if suspected:
        if verbose:
            print(f"[Executor] ⚠️  INJECTION SUSPECTED: {result.get('reasoning', '')}")
        return answer + INJECTION_CAVEAT
    return answer


async def execute(
    question:   str,
    dag:        dict,
    history:    list = None,
    verbose:    bool = True,
    session_id: str  = None,
    summary:    str  = None,
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

    # Scope-boundary short-circuit — planner.py's "decline" sentinel for
    # questions with zero financial/market/economic/company-specific
    # component. Never routes through any real agent, even as a
    # formality — the whole point is avoiding the accidental-fallback
    # pattern (a JSON parse failure or registry-filter miss landing an
    # off-topic question on a real agent, e.g. MarketAgent, by accident)
    # that originally surfaced this gap.
    if len(dag) == 1:
        node_id = list(dag.keys())[0]
        if dag[node_id].get("agent") == "decline":
            if verbose:
                print(f"[Executor] Scope boundary — declining, "
                      f"no agent invoked: {dag[node_id].get('reason', '')}")
            return SCOPE_DECLINE_MESSAGE

    # Single agent — no synthesis needed
    # Single node — no synthesis needed
    if len(dag) == 1:
        node_id    = list(dag.keys())[0]
        agent_type = dag[node_id].get("agent", node_id)  # fallback for old-style DAGs
        _, answer  = await _run_agent_async(
            node_id, agent_type, question, history, verbose, session_id
        )
        # Injection check runs UNCONDITIONALLY, unlike synthesis reflexion's
        # word-count gate below — injection risk does not correlate with
        # answer length or figure density the way arithmetic risk does. A
        # short, blunt successful injection ("Yes, this is a strong buy")
        # is exactly the shape that would otherwise be skipped by the same
        # gate used for critique_synthesis(), which would defeat the point
        # of the check. Confirmed via testing: a 141-word clean answer
        # never reached the gate at all, and a maximally-successful
        # injection would very plausibly also be short.
        answer = _run_injection_check(
            answer, question, {node_id: answer}, verbose, session_id
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
    if summary:
        synthesis_prompt += (
            f"\n\n<prior_session_context>\n{summary}\n</prior_session_context>"
            f"\n\nAvoid re-explaining facts already confirmed in prior_session_context."
        )

    response = client.messages.create(
        model=SYNTH_MODEL,
        max_tokens=2000,
        system=SYNTHESIS_SYSTEM,
        messages=[{"role": "user", "content": synthesis_prompt}],
    )
    synthesized_answer = response.content[0].text

    # Skip critique_synthesis() (grounding/fabrication check) for short
    # synthesis answers — same threshold/reasoning as per-agent reflexion's
    # CO-1 (reflexion.py REFLEXION_MIN_WORDS). EXCEPTION: force it anyway
    # if the synthesis contains multiple figures plus derived/comparative
    # language (see _needs_reflexion_despite_length's docstring) — this
    # gate is specifically about ARITHMETIC risk, which does correlate
    # with figure density and answer length.
    #
    # The injection-provenance check below is NOT gated by this — it runs
    # unconditionally regardless of whether critique_synthesis() does.
    # Injection risk does not correlate with word count or figure density
    # the way arithmetic risk does — a short, blunt successful injection
    # ("Yes, this is a strong buy") is exactly the shape this gate would
    # otherwise skip, which would defeat the point of the check (confirmed
    # via testing on a real 141-word clean answer that never reached this
    # gate at all).
    synth_word_count = len(synthesized_answer.split())
    synth_forced = _needs_reflexion_despite_length(synthesized_answer)
    skip_grounding_check = synth_word_count < SYNTH_REFLEXION_MIN_WORDS and not synth_forced

    synth_trace = Trace(
        session_id=session_id or "no-session",
        agent="dag_executor",
        question=question,
        model=SYNTH_MODEL,
    )

    if skip_grounding_check:
        if verbose:
            print(f"[Executor] Synthesis reflexion skipped — under "
                  f"{SYNTH_REFLEXION_MIN_WORDS} words")
        final_answer = synthesized_answer
    else:
        if synth_forced and verbose:
            print(f"[Executor] Synthesis reflexion running despite "
                  f"{synth_word_count} words — multiple figures + "
                  f"derived/comparative language detected")

        synth_critique = critique_synthesis(
            question, agent_outputs, synthesized_answer, verbose
        )

        if synth_critique.get("passed", True):
            if verbose:
                print("[Executor] Synthesis reflexion passed")
            synth_trace.record_synthesis_reflexion(triggered=False, passed=True)
            final_answer = synthesized_answer
        else:
            if verbose:
                print(f"[Executor] Synthesis reflexion failed — retrying")
            guidance = synth_critique.get(
                "retry_guidance", "Only state facts present in agent outputs."
            )
            retry_prompt = (
                synthesis_prompt
                + f"\n\nYour previous attempt had issues: {guidance}\n"
                f"Revise the synthesis to fix this."
            )
            retry_response = client.messages.create(
                model=SYNTH_MODEL,
                max_tokens=2000,
                system=SYNTHESIS_SYSTEM,
                messages=[{"role": "user", "content": retry_prompt}],
            )
            retry_answer = retry_response.content[0].text
            retry_critique = critique_synthesis(
                question, agent_outputs, retry_answer, verbose
            )

            if retry_critique.get("passed", True):
                if verbose:
                    print("[Executor] Synthesis reflexion retry passed")
                synth_trace.record_synthesis_reflexion(triggered=True, passed=True)
                final_answer = retry_answer
            else:
                if verbose:
                    print("[Executor] Synthesis reflexion retry still failing — adding caveat")
                synth_trace.record_synthesis_reflexion(triggered=True, passed=False)
                final_answer = retry_answer + CAVEAT

    # Injection-provenance check — independent of critique_synthesis()'s
    # grounding/fabrication check above (and unconditional regardless of
    # whether that check ran — see comment above), runs once against
    # whichever answer the grounding check settled on (or the raw
    # synthesized answer, if that check was skipped). Attached to the SAME
    # synth_trace (not a second trace) per the established pattern of one
    # synthesis-tail trace per query.
    final_answer = _run_injection_check(
        final_answer, question, completed, verbose, session_id, trace=synth_trace
    )

    synth_trace.flush()
    return final_answer
