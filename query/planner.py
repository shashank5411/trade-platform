"""
DAG Planner — uses LLM to produce an optimal execution plan.

Given a question and available agents, produces a dependency graph
that the DAG executor uses to run agents in the right order,
parallelizing where possible.
"""

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

PLANNER_SYSTEM = f"""
You are an execution planner for a financial intelligence platform.

Available agents:
{get_agent_descriptions()}

Agent capability boundaries — apply these BEFORE any other rules:
  filings: handles ALL Fed document content — what the Fed has SAID.
    Has get_fed_communications (FOMC statements, minutes, press conference
    transcripts, governor speeches) and semantic_search over Fed documents.
    Use for any question about Fed communications, policy reasoning, or
    official commentary. This is the ONLY agent with access to Fed documents.
  macro:   handles Fed ACTIONS only — rate levels, rate changes, yield data
    via FRED indicators. Has NO access to Fed documents, statements, minutes,
    or speeches. Do NOT route Fed communications questions here — macro cannot
    retrieve what the Fed said, only what rates numerically were.
  market:  handles price and return data only. No macro or document access.

Content routing rules — apply FIRST, before dependency rules:
  1. Route to filings (NOT macro) when question contains any of:
       "what has the Fed said", "Fed statement", "Fed communications",
       "FOMC minutes", "Powell said", "Powell speech", "Fed commentary",
       "Fed speech", "Fed governor", "Fed transcript", "Fed announcement",
       "what did the Fed say", "Fed policy stance", "Fed language"
  2. Route to macro when question asks for Fed rate LEVELS or CHANGES
     as numeric data (e.g. "what was the Fed funds rate in 2022?")
  3. When a question asks BOTH what the Fed said AND rate/inflation data:
     route filings for the communications part AND macro for the indicator
     data — run both in parallel with empty depends_on

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
