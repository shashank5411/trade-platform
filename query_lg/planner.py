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

AGENT CAPABILITY BOUNDARIES — apply BEFORE the content routing rules
below, since the routing rules assume these are already settled:
  filings has the ONLY access to Fed documents/statements (FOMC minutes,
    transcripts, speeches) and SEC 10-K/10-Q prose. macro does NOT have
    this — do not route Fed communications questions to macro.
  macro has Fed rate LEVELS/CHANGES as numeric data and FRED economic
    indicators (unemployment, CPI, yield curves, commodity spot prices
    like DCOILWTICO). macro does NOT have Fed documents or statements.
  market has price/return/volume data only (equities and commodity
    futures like CL=F, GC=F). market does NOT have news, filings, or
    indicator data.
  sentiment has the ONLY access to insider (Form 4) trading and news
    sentiment coverage. sentiment does NOT have filings or Fed document
    content.

CONTENT ROUTING RULES — apply these FIRST, before the dependency rules
above, whenever the question's phrasing matches:
6. Route to filings (NOT macro) when the question contains any of:
   "what has the Fed said", "Fed statement", "Fed communications",
   "FOMC minutes", "Powell said", "Powell speech", "Fed commentary",
   "Fed speech", "Fed governor", "Fed transcript", "Fed announcement",
   "what did the Fed say", "Fed policy stance", "Fed language".
7. Route to macro (NOT filings) when the question asks for Fed rate
   LEVELS or CHANGES as numeric data (e.g. "what was the Fed funds rate
   in 2022?").
8. Route to sentiment (NOT filings) when the question contains any of:
   "insider trades", "insider buying", "insider selling", "Form 4",
   "news sentiment", "media coverage", "analyst coverage", "were
   insiders buying", "did executives sell", "news around", "coverage
   of", "positive news", "negative news", "news about".
9. Macro + sentiment combination — when a question asks whether
   macro/economic conditions relate to or correlate with insider
   trading or news sentiment (e.g. "is unemployment affecting insider
   confidence", "does inflation correlate with insider selling", "how
   does the macro environment relate to news sentiment on X") → route
   to macro + sentiment in parallel, empty depends_on. These are
   genuinely independent signals being asked about together — neither
   agent needs the other's output first.
10. When BOTH what the Fed said AND rate/inflation data are asked about
    → filings + macro in parallel, empty depends_on.
11. When BOTH sentiment signal AND price reaction are asked about →
    sentiment + market in parallel, empty depends_on (market does NOT
    depend on sentiment — they run simultaneously). When BOTH
    insider/news signal AND SEC filing content are asked about →
    sentiment + filings in parallel, empty depends_on.

IMPLICIT DATE RESOLUTION:
When a question implies recency without specifying exact dates, do NOT
default to asking for clarification — resolve it using these default
windows instead, and pass the resolved start/end dates explicitly to
the agent in its instructions so it doesn't have to re-infer them:

  "before earnings" / "pre-earnings"      → 90-day window ending today
  "recently" / "lately" / "of late"       → last 30 days
  "this month"                            → first day of current month to today
  "this quarter"                          → first day of current quarter to today
  "this year" / "YTD"                     → Jan 1 of current year to today
  "before the announcement"               → last 30 days
  "before the merger" / "before the deal" → last 90 days
  "before the news"                       → last 30 days
  "latest" / "most recent"                → most recent available data
                                             point, no date range needed
  "current" / "right now" / "today"       → as of today's date
  "recently filed"                        → last 90 days (filings/documents)
  no time reference at all                → last 90 days for sentiment/insider,
                                             last 30 days for news,
                                             last 1 year for prices,
                                             most recent for indicators

When the question combines implicit recency with a specific event
(e.g. "before earnings", "before the Fed meeting"), prefer the event
window over the generic default rather than stacking both.

BEFORE deciding anything is "missing": if a <session_memory> block or
Conversation context is provided above the question, check it FIRST.
Follow-up phrasing like "now compare with X", "what about Y", "and
NVIDIA?" almost always means: reuse whatever tickers/entities/timeframe
were established in that prior context, substituting or adding the new
one the user just named. E.g. if the prior turn compared AAPL vs MSFT
over the last 3 months and the new question is "compare with NVIDIA
now" — that means AAPL, MSFT, AND NVDA, same 3-month window, NOT a
request missing a comparison target. Only fall through to the clarify
sentinel below if the missing piece genuinely isn't recoverable from
EITHER the question OR the provided context.

SENTINELS — use ONLY when they apply, never as a default:
- clarify: the question is missing information you genuinely need AND
  that information isn't recoverable from session_memory/conversation
  context either (e.g. "compare the stock to Microsoft" with no
  ticker/company named for "the stock", asked as the very FIRST message
  in a session with no prior context to resolve it from). Set
  sentinel="clarify" and sentinel_reason to the SPECIFIC missing piece,
  phrased as a question to the user. This is about MISSING INFORMATION
  IN THE QUESTION (and unrecoverable from context), never about a date
  being recent or a data point you personally don't know.
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

    human_content = (
        f"{state.memory_context}\n\nNew question: {state.question}"
        if state.memory_context else state.question
    )

    response = await model.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
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