"""
Reflexion — self-critique pass on agent answers.

Medium strictness:
  1. Critic reviews the answer against the tool call history
  2. If issues found → agent retries once with critique as context
  3. If retry still has issues → return answer with caveat flag
  4. If clean → return answer as-is

Critic checks:
  - Are specific numbers grounded in fetched data?
  - Are there date ranges that weren't actually fetched?
  - Did any tool calls return errors that the answer ignores?
  - Are there claims about tickers/series not in the tool results?
"""

import os
import json
import anthropic

ENV = os.environ.get("ENV", "dev")

CRITIC_MODEL = (
    "claude-sonnet-4-6"
    if ENV == "prod"
    else "claude-haiku-4-5-20251001"
)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from query.config import get_client
client = get_client()

CRITIC_SYSTEM = """
You are a financial data quality critic. Your job is to review an
agent's answer and check whether it is properly grounded in the
tool call results provided.

Check for:
1. HALLUCINATED NUMBERS — specific prices, rates, percentages, dates
   that do not appear in the tool results
2. MISSING DATA — claims about time periods or tickers that were
   never fetched
3. IGNORED ERRORS — tool calls that returned errors but the answer
   presents data anyway
4. OVERGENERALIZATION — broad claims not supported by the fetched data

Respond ONLY with valid JSON:
{
  "passed": true,
  "issues": []
}
OR
{
  "passed": false,
  "issues": ["specific issue 1", "specific issue 2"],
  "retry_guidance": "what the agent should do differently"
}

Be precise — only flag genuine data grounding issues, not style or
completeness issues. If numbers match tool results, passed=true.
"""

CAVEAT = (
    "\n\n---\n"
    "*Data quality note: this answer could not be fully verified "
    "against fetched data. Some figures may require independent "
    "verification.*"
)


def critique(
    question:     str,
    tool_history: list,
    answer:       str,
    verbose:      bool = True,
) -> dict:
    """Run critic on an answer. Returns {passed, issues, retry_guidance}."""
    tool_summary = "\n".join([
        f"Tool: {t['name']}\nResult preview: {t['result_preview']}"
        for t in tool_history
    ])

    content = (
        f"Question: {question}\n\n"
        f"Tool results used:\n{tool_summary}\n\n"
        f"Agent answer:\n{answer}"
    )

    try:
        response = client.messages.create(
            model=CRITIC_MODEL,
            max_tokens=500,
            system=[{"type": "text", "text": CRITIC_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
        )
    except Exception:
        response = client.messages.create(
            model=CRITIC_MODEL,
            max_tokens=500,
            system=CRITIC_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )

    text = response.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(text)
    except Exception:
        result = {"passed": True, "issues": []}

    if verbose and not result.get("passed"):
        print(f"  [Reflexion] issues found: {result.get('issues', [])}")

    return result


def apply_reflexion(
    question:     str,
    tool_history: list,
    answer:       str,
    retry_fn,
    trace=None,
    verbose:      bool = True,
) -> str:
    """
    Full reflexion pass — critique, optional retry, caveat if needed.

    Args:
        question:     original question
        tool_history: list of tool call records from trace
        answer:       agent's first answer
        retry_fn:     callable(guidance) -> new answer (agent re-run)
        trace:        Trace object for telemetry (optional)
        verbose:      print reflexion steps
    """
    # Skip reflexion if no tool calls were made (nothing to ground-check)
    if not tool_history:
        return answer

    # Skip reflexion for short answers — critic overhead not worth it
    # for factual one-liners or brief summaries (CO-1)
    REFLEXION_MIN_WORDS = 200
    if len(answer.split()) < REFLEXION_MIN_WORDS:
        if verbose:
            print(f"  [Reflexion] skipped — answer under {REFLEXION_MIN_WORDS} words")
        return answer

    result = critique(question, tool_history, answer, verbose)

    if result.get("passed", True):
        if verbose:
            print("  [Reflexion] passed")
        return answer

    if trace:
        trace.reflexion_triggered = True

    guidance = result.get("retry_guidance",
                          "Ensure all numbers come from tool results.")
    if verbose:
        print(f"  [Reflexion] retrying with guidance: {guidance}")

    retry_answer = retry_fn(guidance)

    retry_result = critique(question, tool_history, retry_answer, verbose)

    if retry_result.get("passed", True):
        if verbose:
            print("  [Reflexion] retry passed")
        if trace:
            trace.reflexion_passed = True
        return retry_answer

    if verbose:
        print("  [Reflexion] retry still has issues — adding caveat")
    if trace:
        trace.reflexion_passed = False

    return retry_answer + CAVEAT
