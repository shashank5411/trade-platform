"""
Sub-agents — specialist agents with scoped toolsets.

Four specialists:
  MarketAgent    — prices, returns, technical analysis
  MacroAgent     — indicators, economic context, Fed policy
  FilingsAgent   — SEC filings, Wikipedia, FedSpeak semantic search
  SentimentAgent — news sentiment, insider trades (Form 4), alternative data

Each specialist has:
  - A focused system prompt
  - A scoped tool subset
  - Token budget, tool deduplication, telemetry, and reflexion
"""

import datetime
import os
import json
import anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from query.tools     import get_registry, TOOLS
from query.telemetry import Trace
from query.reflexion import apply_reflexion

MODEL      = "claude-haiku-4-5-20251001"
MAX_TOKENS = 4096

ENV = os.environ.get("ENV", "dev")
from query.config import get_client
client = get_client()
registry = get_registry()

# ── Tool subsets per specialist ────────────────────────────────────────────

def get_tool(name: str) -> dict:
    """Return the tool schema dict for a given tool name."""
    for t in TOOLS:
        if t["name"] == name:
            return t
    raise KeyError(f"Tool not found in TOOLS: {name}")


MARKET_TOOLS = [t for t in TOOLS if t["name"] in {
    "get_prices",
    "get_prices_multi",
    "get_prices_by_sector",
    "get_price_on_date",
    "get_prices_on_date",
    "get_companies_in_sector",
}]

MACRO_TOOLS = [t for t in TOOLS if t["name"] in {
    "get_indicator",
    "get_indicator_multi",
    "get_indicator_on_date",
    "get_macro_snapshot",
}]

FILINGS_TOOLS = [t for t in TOOLS if t["name"] in {
    "get_fed_communications",  # FOMC statements, minutes, transcripts, speeches
    "get_documents",
    "get_prose",               # targeted section retrieval from 10-K/10-Q
    "semantic_search",
    "get_prices",              # for context — price at time of filing
    "get_macro_snapshot",      # for context — macro at time of filing
    "get_companies_in_sector", # sector discovery before filing queries
}]

SENTIMENT_TOOLS = [
    get_tool("get_news"),
    get_tool("get_news_summary"),
    get_tool("get_insider_trades"),
    get_tool("get_insider_summary"),
    # cross-tools — price context for sentiment anchoring
    get_tool("get_prices"),
    get_tool("get_price_on_date"),
    # sector discovery — find which tickers to query for insider/news data
    get_tool("get_companies_in_sector"),
]


# ── System prompts ─────────────────────────────────────────────────────────

MARKET_SYSTEM = f"""Today's date is {datetime.date.today().isoformat()}. Market price data is available from 2020-01-01 to present. Any date before today and after 2020-01-01 is valid historical data — do not reject it as future or unavailable. If a query returns empty results for a recent date, fetch the data and report what is available rather than assuming the date is invalid.

You are a market data specialist. Your job is to answer questions
about asset prices, returns, and market performance using fetched data.

Available assets:
  US equities: AAPL MSFT GOOGL AMZN JPM BAC XOM NVDA META TSLA
               BRK-B V MA UNH JNJ WMT CVX COST GS PFE SPY
  Indices:     ^GSPC ^DJI ^IXIC ^RUT ^VIX ^FTSE ^GDAXI ^N225 ^HSI
  FX:          EURUSD=X GBPUSD=X USDJPY=X DX-Y.NYB
  Commodities: GC=F CL=F SI=F NG=F

Rules:
- Always fetch data before answering — never use training knowledge for prices
- State the exact date range and granularity of data fetched
- Include start price, end price, % change, high, low in every price answer
- If a ticker returns no data, say so and try alternatives
- CRITICAL: Never reference dates, prices, or events beyond the
  last date actually present in the tool results. If data ends
  September 2024, do not describe what happened in October-December.
- Never state the specific date of a high or low price unless the
  data explicitly labels it with a date
- Never attribute price movements to specific events (earnings,
  announcements, tariffs, Fed meetings) unless that event and its
  impact are explicitly stated in the tool results
- This also applies to FINANCIAL MECHANISMS, not just events: do not
  invent or assert explanations like "the market priced in rate cuts,"
  "multiple expansion," "discount rate compression," "operating
  leverage drove this," or any other causal financial theory connecting
  price moves to macro conditions or other context — UNLESS that exact
  mechanism is explicitly stated in a tool result or in attributed prior
  analysis (using the [from prior step: ...] format). These phrases sound
  like expert analysis but are fabricated narrative if no tool result
  or prior-step content actually says them.
- What you CAN say instead: state the numbers, state that they moved in
  the same general period as context T (e.g. "petroleum equities rose
  X% during the same window the macro analysis describes as Y"), and
  explicitly flag any deeper explanation as your own unverified
  interpretation if you choose to offer one at all — e.g. "one possible
  but unverified explanation could be X; the data itself does not
  confirm this mechanism."
- Only report aggregate metrics present in the data: start price,
  end price, % change, high, low — and the exact date range covered
- If the requested date range exceeds available data, explicitly
  state what date range was actually retrieved and what is missing

THREE SOURCES OF INFORMATION — KNOW WHICH IS WHICH:
  1. Data YOU fetched via your own tool calls this turn — cite freely,
     this is your grounded evidence.
  2. Prior analysis from a DIFFERENT specialist, passed to you as context
     — this was verified by THEM, not by you. You may reference and build
     on it, but attribute it explicitly ("per the macro analysis above")
     rather than restating its specific figures as if you had fetched them
     yourself. If it seems wrong or outdated, say so rather than silently
     trusting or silently repeating it.
  3. Anything from training knowledge — forbidden for prices, indicators,
     filings content, or any other data point; only ever use tool results
     or properly attributed prior analysis.

"""

MACRO_SYSTEM = f"""Today's date is {datetime.date.today().isoformat()}. Economic indicator data is available from 2020-01-01 to present. Any date before today and after 2020-01-01 is valid historical data — do not reject it as future or unavailable. If a query returns empty results for a recent date, fetch the data and report what is available rather than assuming the date is invalid.

You are a macroeconomic specialist. Your job is to answer questions
about economic indicators, monetary policy, and macro conditions.

Available series:
  FRED:       FEDFUNDS UNRATE CPIAUCSL DGS10 DGS2 GDP M2SL UMCSENT
              T10Y2Y BAMLH0A0HYM2 DTWEXBGS DEXUSEU DEXJPUS
              DCOILWTICO (WTI crude spot, $/barrel)
  World Bank: NY.GDP.MKTP.CD NY.GDP.PCAP.CD FP.CPI.TOTL.ZG
              SP.POP.TOTL NE.TRD.GNFS.ZS
              Countries: US CN IN GB DE JP BR

Rules:
- Always fetch data — never cite indicator values from memory
- Use get_macro_snapshot for broad macro context questions
- Use get_indicator for specific series deep-dives
- Always note data vintage and staleness for indicators
- Explain what each indicator means in plain English alongside the numbers
- DCOILWTICO is the official government SPOT price for WTI crude oil —
  this IS within your toolset. If asked about "oil prices" as an economic
  indicator (not futures/contract framing), use this series and report it
  as the spot price, distinct from any futures price MarketAgent might
  also report via CL=F.
- Equity market prices (SPY, stock tickers) are outside your toolset
  — do not attempt to fetch them via indicator tools
- If a question asks about both macro indicators AND equity market
  performance, answer only the macro portion and note that equity
  performance will be provided by the MarketAgent

- If the requested date range exceeds available data, explicitly
  state what date range was actually retrieved and what is missing

THREE SOURCES OF INFORMATION — KNOW WHICH IS WHICH:
  1. Data YOU fetched via your own tool calls this turn — cite freely,
     this is your grounded evidence.
  2. Prior analysis from a DIFFERENT specialist, passed to you as context
     — this was verified by THEM, not by you. You may reference and build
     on it, but attribute it explicitly ("per the macro analysis above")
     rather than restating its specific figures as if you had fetched them
     yourself. If it seems wrong or outdated, say so rather than silently
     trusting or silently repeating it.
  3. Anything from training knowledge — forbidden for prices, indicators,
     filings content, or any other data point; only ever use tool results
     or properly attributed prior analysis.
"""

FILINGS_SYSTEM = f"""Today's date is {datetime.date.today().isoformat()}. SEC filing and document data is available from 2020-01-01 to present. Any date before today and after 2020-01-01 is valid historical data — do not reject it as future or unavailable. If a query returns empty results for a recent date, use what is available rather than assuming the date is invalid.

You are a financial documents specialist. Your job is to answer
qualitative questions about companies and economic concepts using
SEC filings and reference documents.

Available documents:
  SEC filings: AAPL MSFT GOOGL AMZN JPM BAC XOM (10-K and 10-Q)
  Wikipedia:   Inflation Recession Federal_Reserve Quantitative_easing
               2008_financial_crisis COVID-19_recession Silicon_Valley_Bank

Tools:
  get_prose       — use for targeted section retrieval when you know
                    exactly which sections are needed.
                    IMPORTANT: when a question asks about multiple sections
                    (e.g. risk factors AND MD&A), always use the
                    section_names parameter to fetch ALL needed sections
                    in a single call:
                      get_prose("AAPL", section_names=["item_1a", "item_7"])
                    Never make separate get_prose calls for sections you
                    could have batched — each extra call is a wasted iteration.
                    Common batching patterns:
                      "risk factors and MD&A"  → ["item_1a", "item_7"]
                      "business and risks"     → ["item_1", "item_1a"]
                      "full picture"           → ["item_1", "item_1a", "item_7"]
                    - NOTE for large banks (JPM, BAC): Item 7 MD&A may be a
                      cross-reference stub returning less than 500 characters.
                      If get_prose returns less than 500 chars for item_7 on
                      JPM or BAC, immediately use semantic_search with queries
                      like "net interest income", "interest rate risk",
                      "credit risk", or "NII sensitivity" instead.
  semantic_search — use for qualitative questions, concept searches,
                    finding relevant content across all documents
  get_documents   — use for fetching specific filings by entity/date

Rules:
- Always ground answers in fetched document content
- Quote or paraphrase specific text from documents
- Note the filing date and period covered for every source cited
- Use get_prose first for known section questions
  (e.g. "risk factors" → get_prose(entity, section_name="item_1a"))
- Use semantic_search for cross-company or concept searches
- Use get_documents for structured financial metrics
- Combine all three for comprehensive analysis
- Combine price and macro context when relevant to filings analysis

## Answering Guidelines
- After retrieving 2-3 tool results, synthesize and answer with what you have.
  Do not keep searching for more perfect data.
- If a question has no time scope, default to the most recent available filing.
  State your assumption explicitly at the start of your answer:
  e.g. 'Based on JPMorgan's most recent 10-K (filed February 2026)...'
- Partial information with clear sourcing is better than no answer.
  Acknowledge gaps briefly but lead with what you found.
- Never call the same tool with the same entity and section more than once.

## Reasoning Protocol

Before EVERY tool call after the first, output a scratchpad in this exact format:

<scratchpad>
known: [one sentence — what have I already retrieved?]
gap: [specific piece of information still missing to answer the question]
next_call: [tool name and exactly why it fills that gap]
seen_similar: [yes/no — have I searched for something semantically similar before?]
</scratchpad>

If seen_similar is yes, you must do ONE of:
- Explicitly state what is specifically different about this call (different
  entity, different time period, different section), then proceed
- OR skip the call entirely and synthesize from what you already have

After 5 tool calls, pause and ask: can I answer the core question with
what I have? If yes, synthesize immediately. If not, make at most 3 more
targeted calls that address specific named gaps.

## Default Scoping Rules

When a question has no time scope: use the most recent available filing.
When a question has no form type: prefer 10-K over 10-Q.
When a question has no entity: ask the user before proceeding.

Always state your scoping assumptions at the START of your answer.
Example: "Based on JPMorgan's most recent 10-K (filed February 2026)..."

## Synthesis Rules

Partial information with clear sourcing is better than no answer.
Acknowledge gaps in one sentence, then lead with what you found.
Never return an empty answer — if you hit the iteration limit,
summarize everything retrieved so far.

- If the requested date range exceeds available data, explicitly
  state what date range was actually retrieved and what is missing

THREE SOURCES OF INFORMATION — KNOW WHICH IS WHICH:
  1. Data YOU fetched via your own tool calls this turn — cite freely,
     this is your grounded evidence.
  2. Prior analysis from a DIFFERENT specialist, passed to you as context
     — this was verified by THEM, not by you. You may reference and build
     on it, but attribute it explicitly ("per the macro analysis above")
     rather than restating its specific figures as if you had fetched them
     yourself. If it seems wrong or outdated, say so rather than silently
     trusting or silently repeating it.
  3. Anything from training knowledge — forbidden for prices, indicators,
     filings content, or any other data point; only ever use tool results
     or properly attributed prior analysis.

"""


# ── Core agent loop (shared) ───────────────────────────────────────────────

def _run_agent(
    question:     str,
    system:       str,
    tools:        list,
    history:      list = None,
    verbose:      bool = True,
    agent_name:   str  = "Agent",
    session_id:   str  = None,
    token_budget: int  = 50_000,
    max_iter:     int  = 8,
    node_id:      str  = None,
) -> str:
    """Shared ReAct loop with guardrails, telemetry, and reflexion.

    node_id identifies which DAG step triggered this run (e.g. "market_2"
    when the same specialist runs more than once in one plan with
    different upstream context). Defaults to None for any caller that
    doesn't pass one (e.g. direct agent.run() calls outside the DAG
    executor, or older code paths) — Trace handles None gracefully.
    """
    messages = list(history or [])
    messages.append({"role": "user", "content": question})

    trace = Trace(
        session_id=session_id or "no-session",
        agent=agent_name,
        question=question,
        model=MODEL,
        node_id=node_id,
    )

    tool_call_seen = set()
    answer         = None

    if verbose:
        print(f"\n[{agent_name}] handling: {question[:80]}...")

    # Stable inputs — cache once per agent run; tools differ per agent class
    cached_system = [{"type": "text", "text": system,
                      "cache_control": {"type": "ephemeral"}}]
    if tools:
        cached_tools = [t.copy() for t in tools]
        cached_tools[-1] = {**cached_tools[-1],
                            "cache_control": {"type": "ephemeral"}}
    else:
        cached_tools = tools

    for iteration in range(max_iter):
        trace.record_iteration()

        # Token budget guard
        if trace.input_tokens + trace.output_tokens > token_budget:
            if verbose:
                print(f"  [{agent_name}] token budget exceeded — stopping")
            trace.hit_token_budget = True
            answer = (
                "Token budget exceeded. Here is what was found so far:\n"
                + _extract_text_from_messages(messages)
            )
            break

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=cached_system,
                tools=cached_tools,
                messages=messages,
            )
        except Exception:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system,
                tools=tools,
                messages=messages,
            )

        trace.record_tokens(
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        cache_write = 0
        cache_read  = response.usage.cache_read_input_tokens or 0
        if hasattr(response.usage, 'cache_creation') and response.usage.cache_creation:
            cache_write = getattr(
                response.usage.cache_creation, 'ephemeral_5m_input_tokens', 0
            ) or 0
        else:
            cache_write = response.usage.cache_creation_input_tokens or 0
        if cache_write > 0 or cache_read > 0:
            print(f"  [{agent_name}] cache — write: {cache_write} | read: {cache_read} tokens")
        
        messages.append({
            "role":    "assistant",
            "content": response.content,
        })

        if response.stop_reason == "end_turn":
            has_tool_use = any(
                getattr(b, "type", None) == "tool_use"
                for b in response.content
            )
            text_content = "\n".join([
                b.text for b in response.content if hasattr(b, "text")
            ])
            if not has_tool_use and "<scratchpad>" in text_content:
                messages.append({
                    "role":    "user",
                    "content": (
                        "Your scratchpad reasoning is noted. Please now "
                        "either call a tool or provide your final answer."
                    ),
                })
                continue
            answer = text_content
            break

        elif response.stop_reason == "tool_use":
            tool_results    = []
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            # Dedup checks must stay sequential — shared set mutation
            to_execute = {}
            for block in tool_use_blocks:
                call_sig  = f"{block.name}:{json.dumps(block.input, sort_keys=True)}"
                was_dedup = call_sig in tool_call_seen
                if not was_dedup:
                    tool_call_seen.add(call_sig)
                to_execute[block.id] = (block, was_dedup)

            def _run_one(block_id):
                block, was_dedup = to_execute[block_id]
                if was_dedup:
                    if verbose:
                        print(f"  [{agent_name}] dedup blocked: {block.name}")
                    return block_id, (
                        f"You already called {block.name} with these inputs. "
                        f"Use the previous result instead of calling again."
                    ), True
                if verbose:
                    print(f"  [{agent_name}] tool: {block.name}")
                result = _execute_tool(block.name, block.input, agent_name=agent_name)
                return block_id, result or "No result returned.", False

            results_map = {}
            if len(to_execute) == 0:
                # Defensive guard — tool_use_blocks existed but to_execute
                # ended up empty (e.g. all calls collapsed during dedup
                # bookkeeping). ThreadPoolExecutor(max_workers=0) raises,
                # so just skip straight to the no-results nudge below
                # rather than crashing the whole agent run.
                pass
            elif len(to_execute) == 1:
                bid, res, dedup = _run_one(list(to_execute.keys())[0])
                results_map[bid] = (res, dedup)
            else:
                with ThreadPoolExecutor(max_workers=len(to_execute)) as pool:
                    futures = {pool.submit(_run_one, bid): bid
                               for bid in to_execute}
                    for future in as_completed(futures):
                        bid, res, dedup = future.result()
                        results_map[bid] = (res, dedup)

            for block in tool_use_blocks:
                result, was_dedup = results_map[block.id]
                trace.record_tool_call(block.name, block.input, result, was_dedup)
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result,
                })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                # All calls were deduplicated — nudge the model to synthesize
                messages.append({
                    "role":    "user",
                    "content": (
                        "All tool calls were duplicates of previous calls. "
                        "You already have the data needed. Please synthesize "
                        "your final answer from the results retrieved so far."
                    ),
                })

        else:
            answer = f"Unexpected stop reason: {response.stop_reason}"
            break

    else:
        trace.hit_max_iter = True
        if verbose:
            print(f"  [{agent_name}] hit max iterations — synthesizing")
        # Force synthesis from accumulated context
        _synth_system = (
            "You are a financial research agent. You have reached your maximum "
            "number of tool calls. Based on everything retrieved in this "
            "conversation so far, provide the best answer you can to the original "
            "question. Lead with what you found. Note any gaps briefly at the end. "
            "Never return an empty answer."
        )
        _synth_messages = messages + [{
            "role": "user",
            "content": (
                f"You have hit the iteration limit. The original question was: "
                f"{question}\n\nPlease synthesize everything retrieved so far "
                f"into a complete answer. State your sources and note any gaps."
            ),
        }]
        try:
            synthesis_response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[{"type": "text", "text": _synth_system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=_synth_messages,
            )
        except Exception:
            synthesis_response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=_synth_system,
                messages=_synth_messages,
            )
        answer = synthesis_response.content[0].text
        trace.record_tokens(
            synthesis_response.usage.input_tokens,
            synthesis_response.usage.output_tokens,
        )

    # Reflexion pass
    def retry_fn(guidance: str) -> str:
        # Seed with the full conversation including all tool results
        # from the first run — agent can ground itself without re-fetching.
        # Strip the final assistant answer (last message) so the agent
        # doesn't anchor on the hallucinated response.
        prior_messages = [
            m for m in messages
            if not (
                m.get("role") == "assistant"
                and m is messages[-1]  # drop only the final assistant turn
            )
        ]
        retry_messages = prior_messages + [{
            "role":    "user",
            "content": (
                f"Your previous answer had issues: {guidance}\n\n"
                f"The tool results above contain all retrieved data. "
                f"Answer the original question using ONLY facts present "
                f"in those tool results. Do not introduce any numbers, "
                f"percentages, or quotes that do not appear verbatim in "
                f"the tool results. If a section was not retrieved, "
                f"say so explicitly rather than fabricating its content."
            ),
        }]
        for _ in range(max_iter):
            try:
                r = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=cached_system,
                    tools=cached_tools,
                    messages=retry_messages,
                )
            except Exception:
                r = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system,
                    tools=tools,
                    messages=retry_messages,
                )
            trace.record_tokens(r.usage.input_tokens, r.usage.output_tokens)
            retry_messages.append({"role": "assistant", "content": r.content})

            if r.stop_reason == "end_turn":
                text = "\n".join([b.text for b in r.content if hasattr(b, "text")])
                has_tool_use = any(
                    getattr(b, "type", None) == "tool_use" for b in r.content
                )
                if not has_tool_use and "<scratchpad>" in text:
                    retry_messages.append({
                        "role":    "user",
                        "content": (
                            "Your scratchpad reasoning is noted. Please now "
                            "either call a tool or provide your final answer."
                        ),
                    })
                    continue
                return text
            elif r.stop_reason == "tool_use":
                results = []
                for b in r.content:
                    if b.type != "tool_use":
                        continue
                    res = _execute_tool(b.name, b.input, agent_name=agent_name)
                    trace.record_tool_call(b.name, b.input, res)
                    results.append({
                        "type":        "tool_result",
                        "tool_use_id": b.id,
                        "content":     res,
                    })
                retry_messages.append({"role": "user", "content": results})
        return answer

    answer = apply_reflexion(
        question=question,
        tool_history=trace.tools_called,
        answer=answer,
        retry_fn=retry_fn,
        trace=trace,
        verbose=verbose,
    )

    import re
    answer = re.sub(
        r'<scratchpad>.*?</scratchpad>', '', answer, flags=re.DOTALL
    ).strip()

    trace.record_answer(answer)
    trace.flush()

    return answer


def _execute_tool(name: str, inputs: dict, agent_name: str = None) -> str:
    if name not in registry:
        return f"Unknown tool: {name}"
    try:
        result = registry[name](**inputs)
        return str(result) if result is not None else "No data returned."
    except Exception as e:
        return f"Tool error ({name}): {e}"


def _extract_text_from_messages(messages: list) -> str:
    """Extract last assistant text from message history."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", [])
            if isinstance(content, list):
                texts = [b.text for b in content if hasattr(b, "text")]
                if texts:
                    return "\n".join(texts)
    return "No answer generated."


# ── Specialist classes ─────────────────────────────────────────────────────

class MarketAgent:
    name         = "MarketAgent"
    TOKEN_BUDGET = 50_000
    MAX_ITER     = 8

    def run(self, question: str, history: list = None,
            verbose: bool = True, session_id: str = None,
            node_id: str = None) -> str:
        return _run_agent(question, MARKET_SYSTEM, MARKET_TOOLS,
                          history, verbose, self.name, session_id,
                          self.TOKEN_BUDGET, self.MAX_ITER,
                          node_id=node_id)


class MacroAgent:
    name         = "MacroAgent"
    TOKEN_BUDGET = 50_000
    MAX_ITER     = 8

    def run(self, question: str, history: list = None,
            verbose: bool = True, session_id: str = None,
            node_id: str = None) -> str:
        return _run_agent(question, MACRO_SYSTEM, MACRO_TOOLS,
                          history, verbose, self.name, session_id,
                          self.TOKEN_BUDGET, self.MAX_ITER,
                          node_id=node_id)


class FilingsAgent:
    name         = "FilingsAgent"
    TOKEN_BUDGET = 150_000
    MAX_ITER     = 15

    def run(self, question: str, history: list = None,
            verbose: bool = True, session_id: str = None,
            node_id: str = None) -> str:
        return _run_agent(question, FILINGS_SYSTEM, FILINGS_TOOLS,
                          history, verbose, self.name, session_id,
                          self.TOKEN_BUDGET, self.MAX_ITER,
                          node_id=node_id)

SENTIMENT_SYSTEM = f"""Today's date is {datetime.date.today().isoformat()}. News and insider trade data is available from 2020-01-01 to present. Any date before today and after 2020-01-01 is valid historical data — do not reject it as future or unavailable. If a query returns empty results for a recent date, fetch the data and report what is available rather than assuming the date is invalid.

You are SentimentAgent, a specialist in alternative data and market sentiment signals.

You have access to two data sources:
  1. News articles (Polygon) — structured per-article sentiment with reasoning
  2. SEC Form 4 insider trades — officer and director stock transactions
  3. Price tools (supporting) — to anchor sentiment signals in actual price context

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL USAGE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

News tools:
  - Always call get_news_summary FIRST to understand overall sentiment tone
  - Then call get_news for specific article details if needed
  - publisher_tier=1 or 2 for signal questions (avoid opinion tier-3 for signals)
  - Use publisher_tier=3 only when the question explicitly asks about analyst
    or opinion coverage

Insider trade tools:
  - Always call get_insider_summary FIRST for the net signal
  - Then call get_insider_trades for specific transactions if needed
  - Default date range when unspecified: last 90 days

Price tools:
  - Use get_price_on_date for point-in-time anchoring (e.g. stock price on
    earnings day, on a specific insider transaction date)
  - Use get_prices for trend context over a period
  - Only call price tools when the question explicitly asks for price context
    or when grounding a sentiment signal in price movement adds clear value

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL INTERPRETATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Insider trade interpretation — ALWAYS apply these:

  1. Type F = tax withholding on RSU/PSU vesting. These are AUTOMATIC share
     surrenders when restricted stock vests to cover tax liability. They are
     NOT sell decisions. Never count F transactions as insider selling.
     Report them separately if mentioned.

  2. Type S (open-market sales) = most executive sales are executed under
     pre-planned Rule 10b5-1 programs established months in advance. High S
     volume alone is NOT a reliable bearish signal. Report it factually
     without inferring intent.

  3. Type P (open-market purchases) = discretionary. Executives rarely buy
     open-market unless they expect appreciation. This IS a meaningful
     bullish signal.

  4. Net signal = P minus S only. F, A, D, M, X, G excluded from net.

  5. Disclosure lag: Form 4 must be filed within 2 business days of the
     transaction. Always state the transaction date, not the filing date,
     when describing when a trade occurred.

News sentiment interpretation — ALWAYS apply these:

  1. Sentiment labels (positive/negative/neutral) are from Polygon's model,
     not your own analysis. Report them as "Polygon classified X articles as
     negative" not "X articles were negative."

  2. Volume ≠ signal strength. 50 neutral articles is not bearish.
     State the sentiment breakdown clearly and let the user interpret.

  3. Publisher tier matters for signal quality. Tier-1 wire services
     (Reuters, AP, Bloomberg) carry more signal than tier-3 opinion pieces.
     Always note the tier distribution when it's relevant to signal quality.

Correlation ≠ causation — ALWAYS:
  - Never state or imply that insider activity caused a price movement
  - Never state or imply that news sentiment caused a price movement
  - Report each signal independently: "Insiders were net selling.
    Separately, the stock declined X% over the same period."
  - Use language like "coincided with", "during the same period",
    "around the time of" — never "due to", "caused by", "driven by"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEFAULT SCOPING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  - No time range specified → use last 90 days for insider trades,
    last 30 days for news
  - No ticker specified → ask for clarification before fetching
  - "Before earnings" → use 90-day window ending on the earnings date;
    if earnings date unknown, use last 90 days and note the assumption
  - Always state your assumed date range at the start of your answer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TICKERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Insider trades: AAPL, MSFT, GOOGL, AMZN, JPM, BAC, XOM
  News:           AAPL, MSFT, GOOGL, AMZN, JPM, BAC, XOM,
                  GLD (gold), USO (oil), TLT (bonds), SPY (broad market)

If asked about a ticker not in these lists, say so clearly rather than
returning empty results without explanation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANSWER FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Structure your answer as:
  1. Assumed date range (one line)
  2. Insider signal (if relevant) — net P vs S, key names/amounts, F excluded
  3. News signal (if relevant) — sentiment breakdown, tier quality, key themes
  4. Price context (if relevant) — anchored to specific dates, no causation
  5. Combined read — what the signals say independently, side by side

Keep answers concise. Do not pad with caveats beyond what the interpretation
rules require."""


class SentimentAgent:
    """
    Specialist agent for alternative data and market sentiment signals.

    Data sources:
      - SEC Form 4 insider trades (AAPL, MSFT, GOOGL, AMZN, JPM, BAC, XOM)
      - Polygon news with per-article sentiment (same tickers + GLD/USO/TLT/SPY)

    Cross-tools (supporting):
      - get_prices, get_price_on_date — anchor sentiment in price context

    Routes here for:
      - Insider buying/selling questions
      - News sentiment and coverage questions
      - Pre-earnings alternative data signals
      - Combined insider + news sentiment reads
    """

    name         = "SentimentAgent"
    TOKEN_BUDGET = 75_000
    MAX_ITER     = 10

    def run(self, question: str, history: list = None,
            verbose: bool = True, session_id: str = None,
            node_id: str = None) -> str:
        return _run_agent(question, SENTIMENT_SYSTEM, SENTIMENT_TOOLS,
                          history, verbose, self.name, session_id,
                          self.TOKEN_BUDGET, self.MAX_ITER,
                          node_id=node_id)


market_agent    = MarketAgent()
macro_agent     = MacroAgent()
filings_agent   = FilingsAgent()
sentiment_agent = SentimentAgent()
