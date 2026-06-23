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
  macro:     handles Fed ACTIONS (rate levels, changes, yield data) AND
    economic indicator series via FRED, including official commodity SPOT
    prices (e.g. DCOILWTICO for WTI crude). Has NO access to Fed documents
    or statements. Do NOT route Fed communications questions here.
  market:    handles price and return data only (equities AND commodity
    futures like CL=F, GC=F, SI=F, NG=F). No macro or document access.
    For commodities, this is FUTURES/contract price only — see commodity
    price disambiguation rule below for when to prefer macro instead.
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
  4. Commodity price disambiguation — oil, gas, gold, silver, and other
     commodities have TWO distinct valid data sources that are NOT
     interchangeable:
       - market (get_prices with CL=F, GC=F, SI=F, NG=F) → FUTURES/contract
         price, intraday-capable, ticker-based
       - macro (get_indicator with DCOILWTICO, DCOILBRENTEU, etc.) → official
         daily SPOT price series from FRED, government-sourced
     Route to market when the question asks about "futures", "contract
     price", "CL=F" or similar ticker syntax, or trading/performance framing
     ("how did oil perform", "oil futures this week").
     Route to macro when the question asks about "spot price", "WTI crude
     oil price" with no futures framing, or uses it as an economic indicator
     ("oil prices and inflation", "what's driving gas prices").
     If the question is genuinely ambiguous (e.g. just "price of WTI crude
     oil" with no other framing), route to BOTH market + macro in parallel,
     empty depends_on, and let synthesis present both figures labeled by
     source (futures vs. spot) rather than guessing which one was meant.
  5. Macro + sentiment combination — when a question asks whether
     macro/economic conditions relate to or correlate with insider
     trading or news sentiment (e.g. "is unemployment affecting insider
     confidence", "does inflation correlate with insider selling",
     "how does the macro environment relate to news sentiment on X")
     → route to macro + sentiment in parallel, empty depends_on.
     These are genuinely independent signals being asked about together —
     neither agent needs the other's output first. Synthesis will present
     both signals side by side per sentiment agent's existing
     correlation-is-not-causation framing.
  6. When BOTH what the Fed said AND rate/inflation data → filings + macro
     in parallel, empty depends_on
  7. When BOTH sentiment signal AND price reaction → sentiment + market
     in parallel, empty depends_on
  8. When BOTH insider/news signal AND SEC filing content → sentiment + filings
     in parallel, empty depends_on
  
  MULTI-AGENT MERGE AND FAN-OUT PATTERNS

  Some questions combine 3 independent-sounding clauses that actually
  have a real dependency structure, not a flat parallel structure.
  Watch for these patterns specifically:

  MERGE — (A, B) -> C: when a question gives two pieces of context
  (e.g. "insiders selling AND the stock dropped X%") and then asks a
  THIRD agent to explain or react to BOTH of them together (e.g.
  "...does the 10-K explain this?"), the third agent's node should
  depend_on BOTH upstream nodes, not run in parallel with them.
  Signal phrase: "does X explain this" / "in light of" / "given both"
  appearing after two distinct data points have already been stated.

  FAN-OUT — A -> (B, C): when a question establishes one shared event
  or context (e.g. "given the SVB collapse" / "given the Fed's rate
  stance") and then asks for TWO separate downstream analyses using
  that same context (e.g. "how did stocks react AND what did insiders
  do"), both downstream agents should depend_on the same upstream node,
  not run independently without that shared context.

  Do not default these patterns to a flat parallel DAG just because
  three agent types are mentioned — check whether the question's
  clauses build on each other or are genuinely independent before
  choosing parallel vs. merge vs. fan-out shape.

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
    "node_id": {{
      "agent": "agent_type_name",
      "depends_on": [],
      "reason": "why this agent is needed at this step"
    }}
  }},
  "reasoning": "overall plan explanation"
}}

NODE ID RULES:
- node_id is a unique label for this step, NOT the agent type. Use
  "{{agent_type}}_1", "{{agent_type}}_2", etc. when the SAME agent type
  needs to run more than once in the plan with different context.
- agent must be one of: market, macro, filings, sentiment
- depends_on references node_ids, not agent types.
- Most questions need each agent at most once — only use numbered
  suffixes when the question genuinely has multiple distinct steps
  that need the same specialist with different upstream context.

PRECEDENCE NOTE: When a question matches the commodity disambiguation
  pattern above (ambiguous oil/gas/gold price framing), this rule takes
  priority over the multi-hop chain pattern below. Route to the
  market+macro PAIR — do not satisfy "multiple perspectives" by
  repeating macro (or market) twice instead of using the other agent.
  Repeating one agent type when a second, different agent type is the
  actual correct second perspective is always wrong, even if it
  produces a plausible-looking multi-step plan.
  
WHEN TO REPEAT AN AGENT (multi-hop chains):
  Some questions describe a chain of 2+ effects where the same kind of
  analysis (e.g. price/performance) is needed at two different points
  in the chain, fed by different upstream context each time.
  Example: "How did macro conditions affect crude prices, and what did
  that mean for petroleum stocks?" — crude price analysis and equity
  price analysis are BOTH market questions, but they are two distinct
  steps: market_1 analyzes crude price reaction to macro context,
  market_2 analyzes petroleum equities using market_1's crude-price
  finding as context. This produces:
    "macro_1":   {{"agent": "macro",  "depends_on": [], ...}}
    "market_1":  {{"agent": "market", "depends_on": ["macro_1"], ...}}
    "market_2":  {{"agent": "market", "depends_on": ["market_1"], ...}}
  Do NOT collapse this into a single market node just because both
  steps use the same agent type — if the question's logic has two
  distinct hops, the DAG should have two distinct nodes, even when
  they share an agent type. Collapsing loses the sequential reasoning
  the question is actually asking for.

  COUNTER-EXAMPLE — do NOT do this: "Is WTI crude oil expensive right
  now?" should NOT become macro_1 (general context) -> macro_2 (price
  comparison) — that's avoiding the real ambiguity by staying inside
  one agent type. This question matches the commodity disambiguation
  rule above and should be market_1 + macro_1 in PARALLEL, not two
  macro calls in sequence.
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

    # Validate agent TYPES against registry (k is now a node_id, not an
    # agent name — the agent type lives in v["agent"])
    valid = set(list_agents())
    dag   = {
        k: v for k, v in dag.items()
        if v.get("agent") in valid
    }
    # Drop depends_on references to any node that got filtered out above
    for v in dag.values():
        v["depends_on"] = [d for d in v.get("depends_on", []) if d in dag]

    if not dag:
        dag = {"market_1": {"agent": "market", "depends_on": [], "reason": "fallback"}}

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
