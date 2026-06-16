"""
Sub-agents — specialist agents with scoped toolsets.

Three specialists:
  MarketAgent   — prices, returns, technical analysis
  MacroAgent    — indicators, economic context, Fed policy
  FilingsAgent  — SEC filings, Wikipedia, semantic search (Phase 5 RAG)

Each specialist has:
  - A focused system prompt
  - A scoped tool subset
  - Token budget, tool deduplication, telemetry, and reflexion
"""

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

MARKET_TOOLS = [t for t in TOOLS if t["name"] in {
    "get_prices",
    "get_prices_multi",
    "get_prices_by_sector",
    "get_price_on_date",
    "get_prices_on_date",
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
    "get_news",                # Polygon news articles with per-article sentiment
    "get_news_summary",        # aggregated sentiment overview for a ticker
    "get_prices",              # for context — price at time of filing
    "get_macro_snapshot",      # for context — macro at time of filing
}]


# ── System prompts ─────────────────────────────────────────────────────────

MARKET_SYSTEM = """
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
- Only report aggregate metrics present in the data: start price,
  end price, % change, high, low — and the exact date range covered
- If the requested date range exceeds available data, explicitly
  state what date range was actually retrieved and what is missing
"""

MACRO_SYSTEM = """
You are a macroeconomic specialist. Your job is to answer questions
about economic indicators, monetary policy, and macro conditions.

Available series:
  FRED:       FEDFUNDS UNRATE CPIAUCSL DGS10 DGS2 GDP M2SL UMCSENT
              T10Y2Y BAMLH0A0HYM2 DTWEXBGS DEXUSEU DEXJPUS
  World Bank: NY.GDP.MKTP.CD NY.GDP.PCAP.CD FP.CPI.TOTL.ZG
              SP.POP.TOTL NE.TRD.GNFS.ZS
              Countries: US CN IN GB DE JP BR

Rules:
- Always fetch data — never cite indicator values from memory
- Use get_macro_snapshot for broad macro context questions
- Use get_indicator for specific series deep-dives
- Always note data vintage and staleness for indicators
- Explain what each indicator means in plain English alongside the numbers
- Equity market prices (SPY, stock tickers) are outside your toolset
  — do not attempt to fetch them via indicator tools
- If a question asks about both macro indicators AND equity market
  performance, answer only the macro portion and note that equity
  performance will be provided by the MarketAgent
"""

FILINGS_SYSTEM = """
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
) -> str:
    """Shared ReAct loop with guardrails, telemetry, and reflexion."""
    messages = list(history or [])
    messages.append({"role": "user", "content": question})

    trace = Trace(
        session_id=session_id or "no-session",
        agent=agent_name,
        question=question,
        model=MODEL,
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
                result = _execute_tool(block.name, block.input)
                return block_id, result or "No result returned.", False

            results_map = {}
            if len(to_execute) == 1:
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
                    res = _execute_tool(b.name, b.input)
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


def _execute_tool(name: str, inputs: dict) -> str:
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
            verbose: bool = True, session_id: str = None) -> str:
        return _run_agent(question, MARKET_SYSTEM, MARKET_TOOLS,
                          history, verbose, self.name, session_id,
                          self.TOKEN_BUDGET, self.MAX_ITER)


class MacroAgent:
    name         = "MacroAgent"
    TOKEN_BUDGET = 50_000
    MAX_ITER     = 8

    def run(self, question: str, history: list = None,
            verbose: bool = True, session_id: str = None) -> str:
        return _run_agent(question, MACRO_SYSTEM, MACRO_TOOLS,
                          history, verbose, self.name, session_id,
                          self.TOKEN_BUDGET, self.MAX_ITER)


class FilingsAgent:
    name         = "FilingsAgent"
    TOKEN_BUDGET = 150_000
    MAX_ITER     = 15

    def run(self, question: str, history: list = None,
            verbose: bool = True, session_id: str = None) -> str:
        return _run_agent(question, FILINGS_SYSTEM, FILINGS_TOOLS,
                          history, verbose, self.name, session_id,
                          self.TOKEN_BUDGET, self.MAX_ITER)


market_agent  = MarketAgent()
macro_agent   = MacroAgent()
filings_agent = FilingsAgent()
