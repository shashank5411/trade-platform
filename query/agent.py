"""
ReAct agent loop — the bridge between LLM and query tools.
Implements the raw Anthropic tool use loop from first principles.
No frameworks — transparent and debuggable.
"""

import os
import sys
import json
import anthropic
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from query.tools import TOOLS, get_registry

# ── Config ─────────────────────────────────────────────────────────────────
## MODEL          = "claude-sonnet-4-6"
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS     = 4096
MAX_ITERATIONS = 10   # safety ceiling on tool call loops

SYSTEM_PROMPT = """
You are a financial analyst assistant with access to a structured 
economic intelligence platform covering:

- Stock prices (US equities, 2020-present)
- Macro economic indicators (FRED: US series, World Bank: global)  
- SEC filings (10-K annual, 10-Q quarterly for major US companies)
- Wikipedia articles on economic topics

Your methodology:
1. Always fetch data before answering quantitative questions
2. Use get_macro_snapshot first for questions about economic context
3. For company questions, fetch both prices AND documents for full picture
4. State what data you found and what period it covers
5. If data is unavailable or incomplete, say so explicitly
6. Never use your training knowledge for specific numbers — 
   always ground in fetched data

Available companies: AAPL, MSFT, GOOGL, AMZN, JPM, BAC, XOM
Available macro series: FEDFUNDS, UNRATE, CPIAUCSL, DGS10, DGS2, 
                        GDP, M2SL, UMCSENT
Available Wikipedia topics: Inflation, Recession, Federal_Reserve,
                            quantitative_easing, 2008_financial_crisis
Data range: 2020-01-01 to present (dev environment)
"""

client   = anthropic.Anthropic()
registry = get_registry()


# ── Core loop ──────────────────────────────────────────────────────────────

def run(
    question:       str,
    verbose:        bool = True,
    max_iterations: int  = MAX_ITERATIONS
) -> str:
    """
    Run the ReAct agent loop for a question.

    Args:
        question:       User's question in plain English
        verbose:        Print tool calls and results as they happen
        max_iterations: Safety ceiling on loop iterations

    Returns:
        Final answer string grounded in fetched data
    """
    messages = [{"role": "user", "content": question}]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")

    for iteration in range(max_iterations):

        # Call LLM
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        # Append LLM response to history
        messages.append({
            "role":    "assistant",
            "content": response.content
        })

        # ── End turn — LLM has final answer ───────────────────────────
        if response.stop_reason == "end_turn":
            answer = _extract_text(response)
            if verbose:
                print(f"\n{'─'*60}")
                print(f"Answer ({iteration+1} iterations):")
                print(answer)
            return answer

        # ── Tool use — LLM wants to call a function ────────────────────
        elif response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name   = block.name
                tool_inputs = block.input

                if verbose:
                    print(f"\n[Iteration {iteration+1}] "
                          f"Tool: {tool_name}")
                    print(f"  Inputs: {json.dumps(tool_inputs, indent=2)}")

                # Execute the tool
                result = _execute_tool(tool_name, tool_inputs)

                if verbose:
                    # Show first 300 chars of result
                    preview = result[:300] + "..." \
                              if len(result) > 300 else result
                    print(f"  Result preview: {preview}")

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result
                })

            # Send tool results back to LLM
            messages.append({
                "role":    "user",
                "content": tool_results
            })

        else:
            # Unexpected stop reason
            return f"Unexpected stop reason: {response.stop_reason}"

    return f"Reached max iterations ({max_iterations}) without final answer."


def _execute_tool(name: str, inputs: dict) -> str:
    """Look up and execute a tool by name. Returns string result."""
    if name not in registry:
        return f"Unknown tool: {name}"
    try:
        fn     = registry[name]
        result = fn(**inputs)
        # Ensure result is a string — LLM expects text
        return str(result) if result is not None else "No data returned."
    except Exception as e:
        return f"Tool execution error ({name}): {str(e)}"


def _extract_text(response) -> str:
    """Extract text content from LLM response."""
    texts = [
        block.text
        for block in response.content
        if hasattr(block, "text")
    ]
    return "\n".join(texts) if texts else "No text response generated."


# ── Entry point for quick testing ──────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test
    questions = [
        "What was Apple's stock price performance in 2022?",
        "What was the Fed funds rate during the 2022 rate hike cycle?",
        "What was the macro environment like in mid-2022?",
    ]

    for q in questions[:1]:  # run first question only
        answer = run(q, verbose=True)