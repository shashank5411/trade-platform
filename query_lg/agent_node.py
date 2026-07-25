"""
agent_node.py — real async ReAct loop, replacing graph.py's stubbed
agent_node.

init_chat_model (not raw anthropic SDK, not ChatAnthropic hardcoded)
per the provider-flexibility decision — swapping providers later means
changing ONE string, nothing else in this file.
"""

import json
from datetime import date
from typing import Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from .state import DraftAnswer, ToolCallRecord
from .tools import TOOLS_BY_AGENT_TYPE
from .models import resolve_model

MAX_ITER = 8
TOKEN_BUDGET = 50_000

SYSTEM_PROMPT_TEMPLATES = {
    "market": """You are a market data specialist. Today's date is {today}.
Dates on or before today are NORMAL queries against this system's live
data source — answer them like any other question, even if the date is
one you have no personal knowledge of from training. NEVER refuse or
call a date "in the future" or "speculative" based on your own training
cutoff — trust the tool results you're given as the actual current data
source, not your own memory of what dates exist. Answer using ONLY the
tools provided — never invent a price or figure. If asked about
something outside price/volume data (e.g. filings, macro indicators),
say so rather than guessing.

COMMODITY/FUTURES TICKERS — you DO have access to these via get_prices,
they are NOT out of your domain: WTI crude oil futures trade under
ticker CL=F, gold under GC=F, the US Dollar Index under DX-Y.NYB, the
VIX under ^VIX. Try get_prices with the relevant ticker BEFORE
concluding you lack commodity data — do not decline a commodity/futures
price question without first attempting the matching ticker.""",
    "filings": """You are a SEC filings specialist. Today's date is
{today}. Dates on or before today are normal queries — never refuse
based on your own training cutoff; trust the tool results as the actual
data source. Answer using ONLY the tools provided — never invent a claim
about what a filing says. If a section isn't returned by your tools, say
so rather than guessing.""",
    "macro": """You are a macroeconomic indicator specialist. Today's
date is {today}. Dates on or before today are normal queries — never
refuse based on your own training cutoff; trust the tool results as the
actual data source. Answer using ONLY the tools provided — never invent
an indicator value. Stay within FRED-style indicators; defer to other
specialists for price data or company-specific filings.""",
    "sentiment": """You are an insider-trading and news-sentiment
specialist. Today's date is {today}. Dates on or before today are normal
queries — never refuse based on your own training cutoff; trust the tool
results as the actual data source. Answer using ONLY the tools provided
— never invent a trade or headline. Distinguish routine 10b5-1 program
sales from discretionary selling when the data allows it.""",
}


async def run_agent(
    node_id: str,
    agent_type: str,
    question: str,
    config: Optional[RunnableConfig] = None,
) -> DraftAnswer:
    """The real async ReAct loop, generalized across agent types.
    Model comes from resolve_model(config) — see models.py for the
    real-vs-test resolution precedence."""
    if agent_type not in TOOLS_BY_AGENT_TYPE:
        raise ValueError(f"Unknown agent_type: {agent_type!r} — must be one of {list(TOOLS_BY_AGENT_TYPE)}")

    agent_tools = TOOLS_BY_AGENT_TYPE[agent_type]
    system_prompt = SYSTEM_PROMPT_TEMPLATES[agent_type].format(today=date.today().isoformat())

    model = resolve_model(config)
    model_with_tools = model.bind_tools(agent_tools)
    tools_by_name = {t.name: t for t in agent_tools}

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    tool_calls_made: list[ToolCallRecord] = []
    total_tokens = 0

    for _ in range(MAX_ITER):
        response: AIMessage = await model_with_tools.ainvoke(messages)
        messages.append(response)

        usage = getattr(response, "usage_metadata", None)
        if usage:
            total_tokens += usage.get("total_tokens", 0)
            if total_tokens > TOKEN_BUDGET:
                break

        if not response.tool_calls:
            return DraftAnswer(
                node_id=node_id, agent_type=agent_type, answer=response.content,
                tool_calls=tool_calls_made,
            )

        reasoning = response.content if isinstance(response.content, str) else ""

        for tc in response.tool_calls:
            tool_fn = tools_by_name[tc["name"]]
            result = await tool_fn.ainvoke(tc["args"])
            tool_calls_made.append(ToolCallRecord(
                name=tc["name"], args=tc["args"],
                result_full=str(result), result_preview=str(result)[:200],
                reasoning_before_call=reasoning or None,
            ))
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return DraftAnswer(
        node_id=node_id, agent_type=agent_type,
        answer=messages[-1].content if isinstance(messages[-1].content, str) else "[no final answer produced]",
        tool_calls=tool_calls_made,
    )