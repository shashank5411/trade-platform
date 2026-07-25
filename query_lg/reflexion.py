"""
reflexion.py — real grounding gate + critique + scoped batch retry loop,
plus synthesis_reflexion_node, the missing counterpart that checks the
synthesized answer as a whole, not just individual drafts.
"""

import re
import math
import json
from datetime import date
from typing import Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

from .state import (
    DraftAnswer, NodeResult, ToolCallRecord, GroundingGateResult, CritiqueResult,
)
from .models import resolve_model

CAVEAT = (
    "\n\n---\n"
    "*Data quality note: this answer could not be fully verified against "
    "fetched data. Some figures may require independent verification.*"
)

ATTRIBUTION_TOLERANCE = 0.01
INVERSION_BLOCKING = False

_FIGURE_PATTERN = re.compile(r'(\$\s?\d[\d,]*\.?\d*|\d+(?:\.\d+)?%|\b\d+\.\d+\b)')

CRITIC_SYSTEM_TEMPLATE = """You are a financial data quality critic.
Today's date is {today}. Dates on or before today are NORMAL queries —
never flag an answer as wrong just because a date is one YOU have no
training knowledge of, or because an answer correctly reports data for
a recent/current date. Check whether this answer is grounded in the
tool results provided. Flag hallucinated numbers, missing data claims,
or overgeneralization — NOT an answer's willingness to report data for
a real, non-future date. Respond ONLY with JSON:
{{"passed": true, "issues": []}} or
{{"passed": false, "issues": ["..."], "retry_guidance": "..."}}"""


def _parse_figure(raw: str) -> Optional[float]:
    cleaned = raw.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_figures(text: str) -> list[float]:
    return [v for raw in _FIGURE_PATTERN.findall(text or "") if (v := _parse_figure(raw)) is not None]


def _values_match(a: float, b: float, rel_tol: float = ATTRIBUTION_TOLERANCE) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)


def _is_unit_scaled_match(x: float, y: float, rel_tol: float = ATTRIBUTION_TOLERANCE) -> bool:
    larger, smaller = (x, y) if abs(x) >= abs(y) else (y, x)
    if abs(larger) <= 1000:
        return False
    return any(math.isclose(smaller, larger / s, rel_tol=rel_tol, abs_tol=1e-9) for s in (1_000, 1_000_000, 1_000_000_000))


def _value_grounded_in(value: float, candidates: list[float]) -> bool:
    return any(_values_match(value, c) for c in candidates) or any(_is_unit_scaled_match(value, c) for c in candidates)


def _derived_pair_values(values: list[float]) -> list[float]:
    if len(values) < 2:
        return []
    pairs = list(zip(values, values[1:]))
    if len(values) > 2:
        pairs.append((values[0], values[-1]))
    derived = []
    for a, b in pairs:
        delta = b - a
        derived += [delta, -delta]
        if a != 0:
            pct = delta / a * 100
            derived += [pct, -pct]
        if b != 0:
            pct2 = -delta / b * 100
            derived += [pct2, -pct2]
    return derived


def _flatten_tool_values(tool_calls: list[ToolCallRecord]) -> list[float]:
    values = []
    for tc in tool_calls:
        if tc.was_dedup:
            continue
        values.extend(_extract_figures(tc.result_full))
    return values


def _flatten_derived_values(tool_calls: list[ToolCallRecord]) -> list[float]:
    derived = []
    for tc in tool_calls:
        if tc.was_dedup:
            continue
        derived.extend(_derived_pair_values(_extract_figures(tc.result_full)))
    return derived


def check_attribution(
    answer_text: str,
    own_tool_calls: list[ToolCallRecord],
    extra_attribution_sources: Optional[list[ToolCallRecord]] = None,
) -> list[dict]:
    all_source_calls = own_tool_calls + (extra_attribution_sources or [])
    source_values = _flatten_tool_values(all_source_calls)
    derived_values = _flatten_derived_values(own_tool_calls)

    failures = []
    for raw in _FIGURE_PATTERN.findall(answer_text or ""):
        value = _parse_figure(raw)
        if value is None:
            continue
        if _value_grounded_in(value, source_values):
            continue
        if any(_values_match(value, dv) for dv in derived_values):
            continue
        failures.append({"kind": "attribution", "value": value, "raw_text": raw,
                          "reason": f"'{raw}' does not match any fetched tool-result value"})
    return failures


def check_inversion(answer_text: str, own_tool_calls: list[ToolCallRecord]) -> list[dict]:
    answer_values = _extract_figures(answer_text)
    failures = []
    for tc in own_tool_calls:
        if tc.was_dedup:
            continue
        for raw in _FIGURE_PATTERN.findall(tc.result_full or ""):
            value = _parse_figure(raw)
            if value is None or _value_grounded_in(value, answer_values):
                continue
            failures.append({"kind": "inversion", "value": value, "raw_text": raw, "tool": tc.name,
                              "reason": f"fetched by {tc.name} but never referenced in the answer"})
    return failures


def run_grounding_gate(
    answer_text: str,
    own_tool_calls: list[ToolCallRecord],
    extra_attribution_sources: Optional[list[ToolCallRecord]] = None,
) -> GroundingGateResult:
    attribution_failures = check_attribution(answer_text, own_tool_calls, extra_attribution_sources)
    inversion_failures = check_inversion(answer_text, own_tool_calls)
    attribution_passed = not attribution_failures
    inversion_passed = not inversion_failures
    return GroundingGateResult(
        passed=attribution_passed and (inversion_passed or not INVERSION_BLOCKING),
        attribution_passed=attribution_passed, inversion_passed=inversion_passed,
        attribution_failures=attribution_failures, inversion_failures=inversion_failures,
        inversion_blocking=INVERSION_BLOCKING,
    )


async def critique(question: str, tool_calls: list[ToolCallRecord], answer: str, model) -> CritiqueResult:
    tool_summary = "\n".join(f"Tool: {tc.name}\nResult: {tc.result_full}" for tc in tool_calls)
    content = f"Question: {question}\n\nTool results:\n{tool_summary}\n\nAnswer:\n{answer}"
    response = await model.ainvoke([SystemMessage(content=CRITIC_SYSTEM_TEMPLATE.format(today=date.today().isoformat())), HumanMessage(content=content)])
    text = response.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = {"passed": True, "issues": []}
    return CritiqueResult(**parsed)


def _build_retry_guidance(critique_result: CritiqueResult, gate: GroundingGateResult) -> str:
    sections = []
    if not critique_result.passed:
        issues = "\n".join(f"  - {i}" for i in critique_result.issues) or "  (none listed)"
        sections.append(f"JUDGE FEEDBACK:\n{issues}")
    if gate.attribution_failures:
        figs = "\n".join(f"  - {f['reason']}" for f in gate.attribution_failures)
        sections.append(f"NUMERIC VERIFICATION FAILURES (attribution):\n{figs}")
    return "\n\n".join(sections) or "Only state facts present in the tool results."


RETRY_SYSTEM_TEMPLATE = """You previously answered a question, but a
data-quality check found issues. Today's date is {today} — dates on or
before today are normal, real queries; never second-guess or hedge on a
date just because you have no personal training knowledge of it. You are
given your original tool results, your prior answer, and specifically
what failed. Revise ONLY the flagged spans — correctly re-express them
using the actual tool data provided. Do not rewrite or delete unrelated
correct content. Do not invent new figures not present in the tool
results."""


async def _retry(question: str, tool_calls: list[ToolCallRecord], prior_answer: str, guidance: str, model) -> str:
    tool_summary = "\n".join(f"Tool: {tc.name}\nResult: {tc.result_full}" for tc in tool_calls)
    content = (
        f"Question: {question}\n\nYour available tool results (unchanged):\n{tool_summary}\n\n"
        f"Your prior answer:\n{prior_answer}\n\nWhat failed:\n{guidance}\n\n"
        f"Revise ONLY the flagged spans, using the tool results above."
    )
    response = await model.ainvoke([SystemMessage(content=RETRY_SYSTEM_TEMPLATE.format(today=date.today().isoformat())), HumanMessage(content=content)])
    return response.content


async def reflexion_node(state, config: Optional[RunnableConfig] = None) -> dict:
    pending = {nid: d for nid, d in state.drafts.items() if nid not in state.node_results}
    results: dict[str, NodeResult] = {}
    model = resolve_model(config)

    for node_id, draft in pending.items():
        extra_sources = []
        for dep_id in draft.depends_on:
            if dep_id in state.node_results:
                extra_sources.extend(state.node_results[dep_id].tool_calls)

        gate = run_grounding_gate(draft.answer, draft.tool_calls, extra_sources)
        crit = await critique(state.question, draft.tool_calls, draft.answer, model=model)
        answer = draft.answer
        retried = False

        if not (gate.passed and crit.passed):
            retried = True
            guidance = _build_retry_guidance(crit, gate)
            answer = await _retry(state.question, draft.tool_calls, draft.answer, guidance, model=model)
            gate = run_grounding_gate(answer, draft.tool_calls, extra_sources)
            crit = await critique(state.question, draft.tool_calls, answer, model=model)
            if not (gate.passed and crit.passed):
                answer = answer + CAVEAT
            elif not crit.retry_guidance:
                crit = crit.model_copy(update={"retry_guidance": guidance})

        results[node_id] = NodeResult(
            node_id=node_id, agent_type=draft.agent_type, depends_on=draft.depends_on,
            answer=answer, tool_calls=draft.tool_calls,
            grounding_gate=gate, critique=crit, retried=retried,
        )

    return {"node_results": results}


SYNTHESIS_CRITIC_SYSTEM_TEMPLATE = """You are reviewing a SYNTHESIZED
answer that combines multiple specialist agents' outputs into one
response. Today's date is {today}. Check whether the synthesis is
properly grounded in those agent outputs — not whether the agent
outputs themselves are correct (already checked separately, per-agent).

Check for:
1. FABRICATED CONNECTIONS — synthesis claims a causal, explanatory, or
   comparative link (including trends, deltas, or "change over time")
   between two agents' findings that neither agent's own output stated —
   especially between structurally different measurements (different
   tickers, different instrument types, different observation dates
   treated as if they were one continuous timeline).
2. UNSUPPORTED FACTS — numbers, dates, or claims in the synthesis that
   do not appear in ANY of the agent outputs provided.
3. DROPPED COVERAGE — an agent's own STATED FACT (a real finding it
   reported) that the synthesis omits entirely without noting a gap.
   IMPORTANT — this does NOT include declining to explain a discrepancy
   or relationship between two numbers that NEITHER agent explained.
   Presenting two differing figures side by side with no invented
   explanation is CORRECT synthesis behavior, not a coverage gap — never
   flag a synthesis for failing to explain, investigate, or reconcile a
   difference the agents themselves left unexplained. Only flag this
   category when a real, stated FACT from an agent's output (a specific
   finding, not an unexplained gap between findings) is missing.
4. MISATTRIBUTED FIGURES — a figure from one agent's output presented as
   if it came from a different agent, or as if synthesis derived it
   independently.

Respond ONLY with valid JSON:
{{"passed": true, "issues": []}} or
{{"passed": false, "issues": ["..."], "retry_guidance": "..."}}"""


async def critique_synthesis(question: str, agent_outputs_text: str, answer: str, model) -> CritiqueResult:
    content = f"Original question: {question}\n\nAgent outputs synthesis was built from:\n{agent_outputs_text}\n\nSynthesized answer:\n{answer}"
    response = await model.ainvoke([
        SystemMessage(content=SYNTHESIS_CRITIC_SYSTEM_TEMPLATE.format(today=date.today().isoformat())),
        HumanMessage(content=content),
    ])
    text = response.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = {"passed": True, "issues": []}
    return CritiqueResult(**parsed)


SYNTHESIS_RETRY_SYSTEM_TEMPLATE = """You previously synthesized multiple
agents' answers into one response, but a data-quality check found
issues. Today's date is {today}. You are given the original agent
outputs, your prior synthesized answer, and specifically what failed.
Revise ONLY the flagged issues — remove any invented connection, trend,
or causation the agents themselves didn't state; add back any dropped
coverage; fix any misattribution. Do not invent new claims. Do not drop
legitimate content that wasn't flagged."""


async def _retry_synthesis(question: str, agent_outputs_text: str, prior_answer: str, guidance: str, model) -> str:
    content = (
        f"Original question: {question}\n\nAgent outputs (unchanged):\n{agent_outputs_text}\n\n"
        f"Your prior synthesized answer:\n{prior_answer}\n\nWhat failed:\n{guidance}\n\n"
        f"Revise ONLY the flagged issues, using the agent outputs above."
    )
    response = await model.ainvoke([
        SystemMessage(content=SYNTHESIS_RETRY_SYSTEM_TEMPLATE.format(today=date.today().isoformat())),
        HumanMessage(content=content),
    ])
    return response.content


async def synthesis_reflexion_node(state, config: Optional[RunnableConfig] = None) -> dict:
    if state.synthesized_answer is None:
        return {}

    model = resolve_model(config)
    all_tool_calls = [tc for r in state.node_results.values() for tc in r.tool_calls]
    agent_outputs_text = "\n\n".join(
        f"[{nid} ({r.agent_type})]\n{r.answer}" for nid, r in state.node_results.items()
    )

    gate = run_grounding_gate(state.synthesized_answer, all_tool_calls)
    crit = await critique_synthesis(state.question, agent_outputs_text, state.synthesized_answer, model=model)
    answer = state.synthesized_answer
    retried = False

    if not (gate.passed and crit.passed):
        retried = True
        guidance = _build_retry_guidance(crit, gate)
        answer = await _retry_synthesis(state.question, agent_outputs_text, state.synthesized_answer, guidance, model=model)
        gate = run_grounding_gate(answer, all_tool_calls)
        crit = await critique_synthesis(state.question, agent_outputs_text, answer, model=model)
        if not (gate.passed and crit.passed):
            answer = answer + CAVEAT
        elif not crit.retry_guidance:
            crit = crit.model_copy(update={"retry_guidance": guidance})

    return {
        "final_answer": answer,
        "synthesis_grounding_gate": gate,
        "synthesis_critique": crit,
        "synthesis_retried": retried,
    }