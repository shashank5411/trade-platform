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
from query.telemetry import Trace
from query.reflexion import (
    critique_synthesis, CAVEAT, _needs_reflexion_despite_length,
    check_injection_provenance, INJECTION_CAVEAT,
    check_attribution, check_inversion, _run_grounding_gate,
    _build_retry_guidance,
)
from query.sub_agents import _strip_correction_preamble

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
    upstream_tool_calls: dict = None,
) -> tuple:
    """Run a single DAG node in a thread pool (non-blocking).

    node_id identifies this specific step in the plan (e.g. "market_2"),
    distinct from agent_type, which is the specialist that executes it
    (e.g. "market"). The same agent_type can run under multiple node_ids
    in one DAG when a question needs the same specialist twice with
    different upstream context (see planner.py's multi-hop chain rules).

    upstream_tool_calls: dict[dep_node_id -> list[tool-call record]] for
    this node's DIRECT dependencies (empty/None for a node with no
    depends_on). Threaded through to apply_reflexion() so a dependent
    node's own grounding gate can recognize a figure it was told to (and
    did) cite via the "[from prior step: ...]" bracket format as grounded
    — see GROUNDING_CHECKS_IMPLEMENTATION.md's "dependent-node attribution"
    section for why this exists and FLOW_REFERENCE.md §5 for the bug this
    fixes.
    """
    agent = get_agent(agent_type)
    if not agent:
        return node_id, f"Agent type '{agent_type}' not found in registry.", []

    try:
        loop = asyncio.get_event_loop()
        answer, tools_called = await loop.run_in_executor(
            None,
            lambda: agent.run(
                question,
                history=history,
                verbose=verbose,
                session_id=session_id,
                node_id=node_id,
                upstream_tool_calls=upstream_tool_calls,
            )
        )
        return node_id, answer, tools_called
    except Exception as e:
        return node_id, f"Node '{node_id}' (agent '{agent_type}') failed: {e}", []


# TEMPORARY — manual eyeball aid for validating check_attribution()/
# check_inversion() against real queries before trusting the telemetry
# pipeline alone (S3/Athena has a write-then-query lag; this is immediate).
# Remove once the grounding gate's behavior has been spot-checked against
# a handful of real runs — this duplicates what synth_trace.attribution_failures
# / .inversion_failures / .grounding_gate_passed already carry once flushed.
def _debug_print_grounding_gate(phase: str, gate: dict) -> None:
    print(f"\n[GroundingGate:{phase}] passed={gate['passed']} "
          f"(attribution_passed={gate['attribution_passed']}, "
          f"inversion_passed={gate['inversion_passed']})")
    if gate["attribution_failures"]:
        print(f"  attribution_failures ({len(gate['attribution_failures'])}):")
        for f in gate["attribution_failures"]:
            print(f"    - {f['raw_text']!r} (value={f['value']}) — {f['reason']}")
    if gate["inversion_failures"]:
        print(f"  inversion_failures ({len(gate['inversion_failures'])}) [non-blocking]:")
        for f in gate["inversion_failures"]:
            print(f"    - {f['raw_text']!r} (value={f['value']}, "
                  f"node={f['node_id']}, tool={f['tool']}) — {f['reason']}")


async def _resolve_synthesis(
    question:           str,
    agent_outputs:       str,
    synthesis_prompt:    str,
    synthesized_answer:  str,
    node_tool_calls:     dict,
    session_id:          str,
    verbose:             bool,
) -> tuple:
    """
    Runs the merged grounding gate (deterministic check_attribution() /
    check_inversion() + the existing LLM critique_synthesis()) against a
    synthesized answer, retries synthesis once on failure, and appends
    CAVEAT if the retry still fails. This is the exact retry-cap/caveat
    control flow that previously lived inline in execute() — extracted
    unchanged so it's unit-testable (mock client.messages.create() and
    critique_synthesis()) without needing a full DAG round to produce
    its inputs.

    Gate semantics: overall_passed = critique_passed AND attribution_passed
    (AND inversion_passed, only once INVERSION_BLOCKING is True — see
    dag_executor.py's module-level flag). The word-count skip gate
    (SYNTH_REFLEXION_MIN_WORDS) only ever governs whether the LLM
    critique_synthesis() call fires — it does NOT skip the deterministic
    grounding gate, which is cheap (no LLM call) and always runs.

    Returns (final_answer: str, synth_trace: Trace). Caller is responsible
    for running the injection check against final_answer and flushing
    synth_trace (both happen on the SAME trace object returned here, per
    the established one-trace-per-synthesis-tail pattern).
    """
    synth_word_count = len(synthesized_answer.split())
    synth_forced = _needs_reflexion_despite_length(synthesized_answer)
    skip_llm_critique = synth_word_count < SYNTH_REFLEXION_MIN_WORDS and not synth_forced

    synth_trace = Trace(
        session_id=session_id or "no-session",
        agent="dag_executor",
        question=question,
        model=SYNTH_MODEL,
    )

    # Deterministic grounding gate — no LLM call, so unlike critique_synthesis()
    # it is never word-count-gated; it always runs against whatever answer
    # is currently being evaluated.
    grounding_gate = await _run_grounding_gate(synthesized_answer, node_tool_calls)
    synth_trace.record_grounding_check(grounding_gate)
    if verbose:
        _debug_print_grounding_gate("initial", grounding_gate)

    if skip_llm_critique:
        if verbose:
            print(f"[Executor] Synthesis LLM critique skipped — under "
                  f"{SYNTH_REFLEXION_MIN_WORDS} words")
        critique_result = {"passed": True, "issues": []}
    else:
        if synth_forced and verbose:
            print(f"[Executor] Synthesis reflexion running despite "
                  f"{synth_word_count} words — multiple figures + "
                  f"derived/comparative language detected")
        critique_result = critique_synthesis(
            question, agent_outputs, synthesized_answer, verbose
        )

    overall_passed = critique_result.get("passed", True) and grounding_gate["passed"]

    if overall_passed:
        if verbose:
            print("[Executor] Synthesis reflexion passed")
        synth_trace.record_synthesis_reflexion(triggered=False, passed=True)
        return synthesized_answer, synth_trace

    if verbose:
        print("[Executor] Synthesis reflexion failed — retrying")
    guidance = _build_retry_guidance(critique_result, grounding_gate)
    retry_prompt = (
        synthesis_prompt
        + f"\n\n[INTERNAL CORRECTION NOTE — do not reference, "
        f"acknowledge, or respond to this note in your answer:\n{guidance}]\n\n"
        f"Produce the corrected synthesis directly. Do not apologize, "
        f"mention a previous attempt, or acknowledge any correction."
    )
    retry_response = client.messages.create(
        model=SYNTH_MODEL,
        max_tokens=2000,
        system=SYNTHESIS_SYSTEM,
        messages=[{"role": "user", "content": retry_prompt}],
    )
    retry_answer = retry_response.content[0].text
    retry_answer = _strip_correction_preamble(retry_answer)

    retry_critique = critique_synthesis(
        question, agent_outputs, retry_answer, verbose
    )
    retry_gate = await _run_grounding_gate(retry_answer, node_tool_calls)
    synth_trace.record_grounding_check(retry_gate)
    if verbose:
        _debug_print_grounding_gate("retry", retry_gate)

    retry_passed = retry_critique.get("passed", True) and retry_gate["passed"]

    if retry_passed:
        if verbose:
            print("[Executor] Synthesis reflexion retry passed")
        synth_trace.record_synthesis_reflexion(triggered=True, passed=True)
        return retry_answer, synth_trace

    if verbose:
        print("[Executor] Synthesis reflexion retry still failing — adding caveat")
    synth_trace.record_synthesis_reflexion(triggered=True, passed=False)
    return retry_answer + CAVEAT, synth_trace


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
) -> tuple:
    """
    Execute a DAG plan and return (final_answer, node_tool_calls, metadata).

    node_tool_calls is a dict mapping node_id -> list of tool-call records
    (same structure as Trace.tools_called, including result_full) — used by
    chart_agent.build_charts() to produce real-data charts without re-querying.

    metadata is {} on all normal paths. When the planner returned a "clarify"
    sentinel, metadata is {"awaiting_clarification": True} and node_tool_calls
    is {} (no agent was invoked).

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
            return SCOPE_DECLINE_MESSAGE, {}, {}

    if len(dag) == 1:
        node_id = list(dag.keys())[0]
        if dag[node_id].get("agent") == "clarify":
            question_for_user = dag[node_id].get("question_for_user",
                "Could you clarify your question?")
            if verbose:
                print(f"[Executor] Clarification needed, no agent "
                      f"invoked: {question_for_user}")
            return question_for_user, {}, {"awaiting_clarification": True}

    # Single node — no synthesis needed
    if len(dag) == 1:
        node_id    = list(dag.keys())[0]
        agent_type = dag[node_id].get("agent", node_id)  # fallback for old-style DAGs
        _, answer, tools_called = await _run_agent_async(
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
        return answer, {node_id: tools_called}, {}

    # Multi-agent — execute in rounds
    # Multi-agent — execute in rounds
    round_num      = 1
    node_tool_calls: dict = {}
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
            # Structured analogue of dep_answers (which is only the
            # formatted PROMPT STRING) — the underlying tool-call records
            # for each direct dependency, still accessible here since
            # node_tool_calls is populated for every prior round before
            # this round's prompts are built. Passed through so the
            # dependent node's OWN grounding gate can recognize a figure
            # it was told to (and does) cite via the MANDATORY CITATION
            # FORMAT below as grounded, instead of only ever seeing its
            # own fetched data — see GROUNDING_CHECKS_IMPLEMENTATION.md's
            # "dependent-node attribution" section.
            upstream_tool_calls = {
                dep: node_tool_calls[dep]
                for dep in node.get("depends_on", [])
                if dep in node_tool_calls
            }
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
            tasks.append(_run_agent_async(
                node_id, agent_type, enriched, history, verbose, session_id,
                upstream_tool_calls=upstream_tool_calls,
            ))

        results = await asyncio.gather(*tasks)
        for node_id, answer, tools_called in results:
            completed[node_id] = answer
            node_tool_calls[node_id] = tools_called
            remaining.discard(node_id)
            if verbose:
                print(f"\n[Executor] --- raw output: {node_id} ---\n{answer}\n"
                      f"[Executor] --- end {node_id} ---")

        round_num += 1

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
    final_answer, synth_trace = await _resolve_synthesis(
        question=question,
        agent_outputs=agent_outputs,
        synthesis_prompt=synthesis_prompt,
        synthesized_answer=synthesized_answer,
        node_tool_calls=node_tool_calls,
        session_id=session_id,
        verbose=verbose,
    )

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
    return final_answer, node_tool_calls, {}
