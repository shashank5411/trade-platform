"""
graph.py — the compiled graph skeleton.

Built once at import time (graph.compile() below) — reused for every
question, per the "fixed graph, not built per-query" principle. What's
dynamic is which parts get traversed and how many times, driven by state.
"""

from typing import Literal, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .state import (
    GraphState, DagNodeSpec, DraftAnswer, NodeResult,
    GroundingGateResult, CritiqueResult, ToolCallRecord,
)
from .planner import planner_node
from .agent_node import run_agent
from .reflexion import reflexion_node, synthesis_reflexion_node
from .models import resolve_model
from langchain_core.messages import SystemMessage, HumanMessage


def route_after_planner(state: GraphState) -> Literal["dispatch", "sentinel_response"]:
    return "sentinel_response" if state.planner_sentinel is not None else "dispatch"


def sentinel_response(state: GraphState) -> dict:
    if state.planner_sentinel == "decline":
        return {"final_answer": "That's outside what I can help with — I focus on financial/market questions."}

    clarify_question = next(iter(state.dag.values())).reason
    user_answer = interrupt({"question": f"Can you clarify — {clarify_question}?"})

    merged_question = f"{state.question} [clarified: {user_answer}]"
    return {"question": merged_question, "planner_sentinel": None, "dag": {}}


def route_after_sentinel(state: GraphState) -> Literal["planner", "__end__"]:
    return "planner" if state.final_answer is None else "__end__"


def dispatch_node(state: GraphState) -> dict:
    return {}


def route_after_dispatch(state: GraphState) -> list[Send] | Literal["synthesis"]:
    done = set(state.node_results.keys())
    remaining = {
        nid: spec for nid, spec in state.dag.items()
        if nid not in done and spec.agent in ("market", "filings", "macro", "sentiment")
    }
    ready = [nid for nid, spec in remaining.items() if all(d in done for d in spec.depends_on)]

    if not ready:
        return "synthesis"

    return [
        Send("agent_node", {"node_id": nid, "agent_type": state.dag[nid].agent, "question": state.question})
        for nid in ready
    ]


async def agent_node(payload: dict, config: Optional[RunnableConfig] = None) -> dict:
    node_id = payload["node_id"]
    draft = await run_agent(node_id, payload["agent_type"], payload["question"], config=config)
    return {"drafts": {node_id: draft}}


SYNTHESIS_SYSTEM = """You combine multiple specialist agents' answers
into ONE coherent response to the user's original question. Rules:

- Weave the agents' findings together naturally — don't just concatenate
  their answers back to back or repeat the same figures twice if two
  agents happened to report overlapping data.
- NEVER invent a causal or explanatory link between independent signals
  that the agents themselves didn't state. If macro and sentiment data
  are both present with no stated connection, present them as
  complementary context, not as one causing the other.
- If two agents' numbers genuinely differ for the same thing (e.g. a
  spot price vs. a futures price), frame this as two DIFFERENT, valid
  measurements — not as a discrepancy, error, or data quality issue.
  CRITICAL: this also means NEVER computing a trend, delta, or "change
  over time" BETWEEN two structurally different measurements (e.g. a
  futures contract price on one date vs. a spot price observation on a
  different date) — they are not two points on the same timeline, even
  if both are in dollars and both have dates attached. A trend/delta
  claim is only valid between two values from the SAME underlying
  series/instrument. If you're unsure whether two numbers are
  comparable this way, they aren't — just present both separately.
- Preserve every real, distinct finding — don't drop content, only
  remove genuine duplication.
- Answer the user's actual question directly, don't just summarize what
  each agent said."""


async def synthesis_node(state: GraphState, config: Optional[RunnableConfig] = None) -> dict:
    if len(state.node_results) == 1:
        only = next(iter(state.node_results.values()))
        return {"final_answer": only.answer}

    model = resolve_model(config)
    agent_outputs = "\n\n".join(
        f"[{nid} ({r.agent_type})]\n{r.answer}" for nid, r in state.node_results.items()
    )
    content = f"Original question: {state.question}\n\nAgent outputs:\n{agent_outputs}"
    response = await model.ainvoke([
        SystemMessage(content=SYNTHESIS_SYSTEM),
        HumanMessage(content=content),
    ])
    return {"synthesized_answer": response.content}


_INJECTION_REGISTER_STUB = ("ignore all prior", "you must now", "override your")

def injection_check_node(state: GraphState) -> dict:
    text = (state.final_answer or "").lower()
    tripped = any(p in text for p in _INJECTION_REGISTER_STUB)
    if not tripped:
        return {"injection_check": {"injection_suspected": False, "checked": False}}

    provenance_context = {nid: r.answer for nid, r in state.node_results.items()}

    return {
        "injection_check": {
            "injection_suspected": True, "checked": True,
            "provenance_nodes_considered": list(provenance_context.keys()),
        },
        "final_answer": state.final_answer + "\n\n---\n*Provenance note: review before acting on this.*",
    }


builder = StateGraph(GraphState)
builder.add_node("planner", planner_node)
builder.add_node("sentinel_response", sentinel_response)
builder.add_node("dispatch", dispatch_node)
builder.add_node("agent_node", agent_node)
builder.add_node("reflexion", reflexion_node)
builder.add_node("synthesis", synthesis_node)
builder.add_node("synthesis_reflexion", synthesis_reflexion_node)
builder.add_node("injection_check", injection_check_node)

builder.set_entry_point("planner")
builder.add_conditional_edges("planner", route_after_planner, {
    "dispatch": "dispatch", "sentinel_response": "sentinel_response",
})
builder.add_conditional_edges("sentinel_response", route_after_sentinel, {
    "planner": "planner", "__end__": END,
})
builder.add_conditional_edges("dispatch", route_after_dispatch, ["agent_node", "synthesis"])
builder.add_edge("agent_node", "reflexion")
builder.add_edge("reflexion", "dispatch")
builder.add_edge("synthesis", "synthesis_reflexion")
builder.add_edge("synthesis_reflexion", "injection_check")
builder.add_edge("injection_check", END)

compiled_graph = builder.compile(
    checkpointer=MemorySaver(
        serde=JsonPlusSerializer(
            allowed_msgpack_modules=[
                ("query_lg.state", "DagNodeSpec"),
                ("query_lg.state", "DraftAnswer"),
                ("query_lg.state", "NodeResult"),
                ("query_lg.state", "ToolCallRecord"),
                ("query_lg.state", "GroundingGateResult"),
                ("query_lg.state", "CritiqueResult"),
            ],
        )
    )
)