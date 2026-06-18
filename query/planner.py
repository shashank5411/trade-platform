"""
DAG Planner — uses LLM to produce an optimal execution plan.

Given a question and available agents, produces a dependency graph
that the DAG executor uses to run agents in the right order,
parallelizing where possible.
"""

import datetime
import os
import json
import anthropic

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from query.registry import get_agent_descriptions, list_agents

ENV = os.environ.get("ENV", "dev")

PLANNER_MODEL = (
    "claude-sonnet-4-6"
    if ENV == "prod"
    else "claude-haiku-4-5-20251001"
)
from query.config import get_client
client = get_client()

PLANNER_SYSTEM = f"""Today's date is {datetime.date.today().isoformat()}. Use this to resolve relative date references and to determine what is historical (in the past) vs future (after today). Data is available from 2020-01-01 onwards for dev environment.

You are an execution planner for a financial intelligence platform.

Available agents:
{get_agent_descriptions()}

Agent capability boundaries — apply these BEFORE any other rules:
  filings:   handles ALL Fed document content — what the Fed has SAID.
    Has get_fed_communications and semantic_search over FOMC statements,
    minutes, transcripts, speeches, SEC 10-K/10-Q, and Wikipedia articles.
    This is the ONLY agent with access to Fed documents and SEC prose.
    Does NOT have news or insider trade tools.
  macro:     handles Fed ACTIONS only — rate levels, rate changes, yield data
    via FRED indicators. Has NO access to Fed documents or statements.
    Do NOT route Fed communications questions here.
  market:    handles price and return data only. No macro or document access.
  sentiment: handles insider trades (Form 4) and news sentiment.
    This is the ONLY agent with get_news, get_news_summary,
    get_insider_trades, get_insider_summary. Also has get_prices for context.
    Does NOT have filings or Fed document tools.

Content routing rules — apply FIRST, before dependency rules:
  1. Route to filings (NOT macro) when question contains any of:
       "what has the Fed said", "Fed statement", "Fed communications",
       "FOMC minutes", "Powell said", "Powell speech", "Fed commentary",
       "Fed speech", "Fed governor", "Fed transcript", "Fed announcement",
       "what did the Fed say", "Fed policy stance", "Fed language"
  2. Route to macro when question asks for Fed rate LEVELS or CHANGES
     as numeric data (e.g. "what was the Fed funds rate in 2022?")
  3. Route to sentiment (NOT filings) when question contains any of:
       "insider trades", "insider buying", "insider selling", "Form 4",
       "news sentiment", "media coverage", "analyst coverage",
       "were insiders buying", "did executives sell", "news around",
       "coverage of", "positive news", "negative news", "news about"
  4. When BOTH what the Fed said AND rate/inflation data → filings + macro
     in parallel, empty depends_on
  5. When BOTH sentiment signal AND price reaction → sentiment + market
     in parallel, empty depends_on
  6. When BOTH insider/news signal AND SEC filing content → sentiment + filings
     in parallel, empty depends_on

IMPLICIT DATE RESOLUTION

When a question implies recency without specifying exact dates, do NOT
ask for clarification. Use these default windows and instruct the relevant
agent accordingly. The agent MUST state the assumed window at the start
of its answer.

  "before earnings" / "pre-earnings"      → 90-day window ending today
  "recently" / "lately" / "of late"       → last 30 days
  "this month"                            → first day of current month to today
  "this quarter"                          → first day of current quarter to today
  "this year" / "YTD"                     → Jan 1 of current year to today
  "before the announcement"               → last 30 days
  "before the merger" / "before the deal" → last 90 days
  "before the news"                       → last 30 days
  "latest" / "most recent"                → most recent available data point,
                                            no date range needed
  "current" / "right now" / "today"       → as of today's date
  "recently filed"                        → last 90 days (filings/documents)
  no time reference at all                → last 90 days for sentiment/insider,
                                            last 30 days for news,
                                            last 1 year for prices,
                                            most recent for indicators

When the question combines implicit recency with a specific event
(e.g. "before earnings", "before the Fed meeting"), prefer the event
window over the generic default. Always pass the resolved start/end
dates explicitly to the agent in the DAG instructions so the agent
does not have to re-infer them.

Today's date is available as context — use it to compute absolute
dates from relative references before routing.

Given a user question and optional conversation context, produce an
optimal execution DAG (Directed Acyclic Graph) that minimizes latency
while ensuring each agent has the context it needs.

Rules:
1. Only include agents genuinely needed to answer the question
2. If agent B needs agent A's output to answer well, add A to B's depends_on
3. Agents with empty depends_on run in parallel in Round 1
4. Agents whose dependencies are all complete run in parallel in the next round
5. Minimize total rounds — maximize parallelism where safe
6. If only one agent is needed, return just that agent with empty depends_on
7. Default to fewest agents possible — don't include agents speculatively

Dependency decision guide:
- Market agent rarely needs other agents' output first
- Macro agent rarely needs other agents' output first
- Sentiment agent rarely needs other agents' output first
- Filings agent benefits from macro context when question is about
  WHY something happened (e.g. SVB collapse needs macro backdrop)
- When question asks about market REACTION TO an event, market needs
  macro or filings context first
- When question is purely about prices OR purely about indicators,
  use a single agent
- Fed COMMUNICATIONS (what the Fed said) → filings only, never macro
- Fed rate DATA (what rates numerically were) → macro only, never filings
- Mixed question (what Fed said + rate/inflation data) → filings + macro
  in parallel, no dependency between them
- Insider trades / news sentiment → sentiment only, never filings
- "Did the stock react to insider buying?" → sentiment + market in parallel,
  market does NOT depend on sentiment (run simultaneously)
- "What do insiders think AND what does the 10-K say?" → sentiment + filings
  in parallel, no dependency between them

Respond ONLY with valid JSON, no other text, no markdown:
{{
  "agents": {{
    "agent_name": {{
      "depends_on": [],
      "reason": "why this agent is needed"
    }}
  }},
  "reasoning": "overall plan explanation"
}}
"""


def plan(question: str, history: list = None,
         verbose: bool = True) -> dict:
    """
    Produce a DAG execution plan for a question.
    Returns dict of agent_name -> {depends_on, reason}
    """
    if history:
        recent  = history[-4:]
        context = "\n".join([
            f"{t['role'].upper()}: {str(t['content'])[:200]}"
            for t in recent
        ])
        content = (
            f"Conversation context:\n{context}\n\n"
            f"New question: {question}"
        )
    else:
        content = question

    try:
        response = client.messages.create(
            model=PLANNER_MODEL,
            max_tokens=500,
            system=[{"type": "text", "text": PLANNER_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
        )
    except Exception:
        response = client.messages.create(
            model=PLANNER_MODEL,
            max_tokens=500,
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )

    text = response.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        result    = json.loads(text)
        dag       = result.get("agents", {})
        reasoning = result.get("reasoning", "")
    except json.JSONDecodeError:
        dag       = {"macro": {"depends_on": [], "reason": "fallback"}}
        reasoning = "planning failed — using fallback"

    # Validate agent names against registry
    valid = set(list_agents())
    dag   = {k: v for k, v in dag.items() if k in valid}

    if not dag:
        dag = {"market": {"depends_on": [], "reason": "fallback"}}

    if verbose:
        print(f"\n[Planner] DAG: {list(dag.keys())}")
        print(f"[Planner] {reasoning}")
        rounds = _resolve_rounds(dag)
        for i, round_agents in enumerate(rounds, 1):
            print(f"[Planner] Round {i}: {round_agents}")

    return dag


def _resolve_rounds(dag: dict) -> list:
    """
    Resolve DAG into ordered rounds for display/execution.
    Returns list of lists: [[round1_agents], [round2_agents], ...]
    """
    completed = set()
    remaining = set(dag.keys())
    rounds    = []

    while remaining:
        ready = [
            name for name in remaining
            if all(dep in completed
                   for dep in dag[name].get("depends_on", []))
        ]
        if not ready:
            rounds.append(list(remaining))
            break
        rounds.append(ready)
        completed.update(ready)
        remaining -= set(ready)

    return rounds
