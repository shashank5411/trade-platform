"""
planner.py — real planner_node, replacing graph.py's keyword-matching stub.

One LLM call, structured JSON output, parsed into real DagNodeSpec objects
— which means Pydantic now actually validates the planner's own routing
decisions (a bad agent name in the JSON gets rejected, not silently
accepted), catching a class of planner bug for free that a plain dict
never would have.
"""

import json
from datetime import date
from typing import Optional
from langchain_core.runnables import RunnableConfig
from .state import DagNodeSpec
from .models import resolve_model
from langchain_core.messages import HumanMessage, SystemMessage

PLANNER_SYSTEM_TEMPLATE = """You are a routing planner for a financial research
system with four specialist agents:

- market: prices, returns, trading volume, technical performance
- filings: SEC 10-K/10-Q content, risk factors, MD&A, Fed communications
- macro: FRED economic indicators (unemployment, CPI, rates, yield curves)
- sentiment: insider (Form 4) trading activity, news sentiment coverage

Today's date is {today}. Questions about dates at or before today are
NORMAL queries against the system's own data — route them like any
other question, even if a date is recent or you personally have no
knowledge of it. NEVER decline or ask for clarification just because a
date is beyond your own training knowledge — the specialist agents query
a live data source, not your training data, and will correctly report
if THEIR data doesn't cover a period. Your only job is topic routing,
never a judgment about what you personally know.

ROUTING RULES:
1. Route to every agent whose domain the question genuinely needs — no
   more, no fewer.
2. If two agents' data is needed but neither depends on the other's
   output, list both with empty depends_on (they'll run in parallel).
3. If one agent's output is needed as CONTEXT for another (e.g. "how did
   the stock react to X" needs filings' X before market can measure the
   reaction), set the dependent agent's depends_on to include the
   upstream node's id.
4. Ambiguous commodity/indicator questions (e.g. "price of WTI crude")
   with no framing indicating spot vs. futures should route to BOTH
   market and macro in parallel — this is a known, deliberate ambiguity,
   not something to guess at.
5. NEVER create two nodes of the SAME agent type for a question that one
   multi-entity tool call already covers — e.g. "compare AAPL and MSFT"
   is ONE market node using a multi-ticker tool (get_prices_multi), NOT
   two separate market nodes each fetching both tickers independently.
   This is a real, confirmed bug pattern: two same-type nodes routed for
   a comparison question both end up calling the SAME multi-entity tool
   with the SAME arguments, doubling cost for identical, redundant work.
   Only create multiple same-type nodes when they genuinely need
   DIFFERENT data (e.g. two DIFFERENT date ranges, or one depends on the
   other's output) — never when one multi-entity call already covers
   every entity the question asks about.

SENTINELS — use ONLY when they apply, never as a default:
- clarify: the question is missing information you genuinely need (e.g.
  "compare the stock to Microsoft" with no ticker/company named for "the
  stock"). Set sentinel="clarify" and sentinel_reason to the SPECIFIC
  missing piece, phrased as a question to the user. This is about
  MISSING INFORMATION IN THE QUESTION, never about a date being recent
  or a data point you personally don't know.
- decline: the question has ZERO financial/market/economic component —
  e.g. pure arithmetic, general trivia, unrelated topics. This is a
  TOPIC/DOMAIN check only. A well-formed financial question about a
  specific date, ticker, or period is NEVER a decline candidate, even
  if that date/period is one you have no personal knowledge of.

Respond ONLY with valid JSON, no markdown fences, matching exactly:
{{
  "agents": {{
    "<node_id>": {{"agent": "market|filings|macro|sentiment", "depends_on": [], "reason": "..."}}
  }},
  "sentinel": null,
  "sentinel_reason": null
}}
OR, when a sentinel applies:
{{
  "agents": {{}},
  "sentinel": "clarify" or "decline",
  "sentinel_reason": "..."
}}
"""


async def planner_node(state, config: Optional[RunnableConfig] = None) -> dict:
    model = resolve_model(config)
    today = date.today().isoformat()
    system_prompt = PLANNER_SYSTEM_TEMPLATE.format(today=today)
    response = await model.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.question),
    ])
    text = response.content.strip().replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {
            "planner_sentinel": "clarify",
            "dag": {"clarify_1": DagNodeSpec(
                agent="clarify",
                reason="I had trouble understanding that — could you rephrase your question?",
            )},
        }

    sentinel = parsed.get("sentinel")
    if sentinel in ("clarify", "decline"):
        return {
            "planner_sentinel": sentinel,
            "dag": {f"{sentinel}_1": DagNodeSpec(
                agent=sentinel, reason=parsed.get("sentinel_reason") or "",
            )},
        }

    dag = {
        node_id: DagNodeSpec(**spec)
        for node_id, spec in parsed.get("agents", {}).items()
    }
    return {"dag": dag, "planner_sentinel": None}