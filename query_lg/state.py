"""
GraphState — Pydantic version.

Same conceptual design as the TypedDict version (whiteboard model, one
reducer per multi-writer field, tool calls are data not nodes, reflexion
is its own node with a draft/final split) — see langgraph_theory_refresher.md
§1-4 for the full reasoning. This file adds REAL runtime validation:
every node's return value gets checked against these models on every
graph transition, not just at type-check time.

Reflexion-as-node consequence (§4 of the refresher): an agent node
finishes BEFORE reflexion has judged it, so it writes a DraftAnswer, not
a finished NodeResult. Only the reflexion node, after judging a draft,
promotes it into a NodeResult. Three fields need reducers because more
than one parallel Send-invoked node writes to them in the same step:
`drafts`, `node_results`, `retry_counts`.
"""

from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ══════════════════════════════════════════════════════════════════════
# Leaf-level models
# ══════════════════════════════════════════════════════════════════════

class ToolCallRecord(BaseModel):
    """One tool call's result. Nested inside a DraftAnswer/NodeResult,
    never a graph node itself — tool calls are data, not nodes."""
    name: str
    args: dict
    result_full: str
    result_preview: str
    was_dedup: bool = False
    reasoning_before_call: Optional[str] = None
    # The model's stated reasoning immediately preceding THIS call —
    # captured live inside the agent node's own ReAct loop (dispatch never
    # sees individual tool calls, so this has nothing to do with dispatch).
    # Enables the trajectory-adaptation eval category natively from
    # GraphState rather than needing raw completions graded externally.


class GroundingGateResult(BaseModel):
    """Mirrors reflexion.py's _merge_gate_results() return shape.

    inversion_blocking mirrors reflexion.py's module-level
    INVERSION_BLOCKING constant, passed in explicitly (not read from a
    shared global) so this model stays self-contained. Defaults False,
    matching reflexion.py's current value."""
    passed: bool
    attribution_passed: bool
    inversion_passed: bool
    attribution_failures: list[dict] = Field(default_factory=list)
    inversion_failures: list[dict] = Field(default_factory=list)
    inversion_blocking: bool = False

    @model_validator(mode="after")
    def failures_match_pass_flags(self) -> "GroundingGateResult":
        """Two invariants, both mirroring _merge_gate_results():
        (1) each sub-check's passed flag must agree with whether it has
            recorded failures — checked for BOTH attribution AND
            inversion (the original version only checked attribution).
        (2) the TOP-LEVEL passed field must be a pure function of
            attribution_passed (always) and inversion_passed (only when
            inversion_blocking=True) — the original version never
            checked this, so passed=True with attribution_passed=False
            was silently constructible."""
        for label, sub_passed, failures in (
            ("attribution", self.attribution_passed, self.attribution_failures),
            ("inversion", self.inversion_passed, self.inversion_failures),
        ):
            if sub_passed and failures:
                raise ValueError(f"{label}_passed=True but {label}_failures is non-empty")
            if not sub_passed and not failures:
                raise ValueError(f"{label}_passed=False but {label}_failures is empty")

        expected_passed = self.attribution_passed and (
            self.inversion_passed or not self.inversion_blocking
        )
        if self.passed != expected_passed:
            raise ValueError(
                f"passed={self.passed} disagrees with sub-checks "
                f"(attribution_passed={self.attribution_passed}, "
                f"inversion_passed={self.inversion_passed}, "
                f"inversion_blocking={self.inversion_blocking} "
                f"implies passed should be {expected_passed})"
            )
        return self


class CritiqueResult(BaseModel):
    """Mirrors critique()/critique_synthesis()'s return shape."""
    passed: bool
    issues: list[str] = Field(default_factory=list)
    retry_guidance: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════
# DagNodeSpec — the planner's output shape, one entry per dag key
#
# Deliberately a WIDER agent-value set than DraftAnswer/NodeResult's
# Literal["market","filings","macro","sentiment"]: the planner can emit
# "clarify" or "decline" sentinel routes (confirmed by the eval suite —
# CLARIFY-TRIGGER-001 asserts expected_agents: [clarify]; SCOPE-OFFTOPIC-001
# exercises the "decline" path) which short-circuit straight to a canned
# message and NEVER produce a real DraftAnswer/NodeResult at all — no
# agent runs, no tool calls happen. Keeping these two literal sets
# separate is deliberate: it would be a real modeling error to let
# "clarify"/"decline" leak into NodeResult.agent_type, since a NodeResult
# represents a genuinely-executed specialist agent's output.
# ══════════════════════════════════════════════════════════════════════

class DagNodeSpec(BaseModel):
    agent: Literal["market", "filings", "macro", "sentiment", "clarify", "decline"]
    depends_on: list[str] = Field(default_factory=list)
    reason: str = ""


# ══════════════════════════════════════════════════════════════════════
# DraftAnswer — what an agent node writes, BEFORE reflexion judges it
# ══════════════════════════════════════════════════════════════════════

class DraftAnswer(BaseModel):
    node_id: str
    agent_type: Literal["market", "filings", "macro", "sentiment"]
    depends_on: list[str] = Field(default_factory=list)
    answer: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


def merge_node_results(left: dict, right: dict) -> dict:
    """REQUIRED reducer for both `drafts` and `node_results` — multiple
    Send-invoked agent nodes (or reflexion invocations) write to these in
    the same parallel step. Plain dict union is sufficient: each writer
    owns a distinct node_id key, so there's never a real key collision to
    resolve, only two disjoint dicts to combine."""
    return {**left, **right}


def merge_retry_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Same disjoint-key reasoning as merge_node_results, kept as its own
    named function (rather than reused) so a future per-key SUM behavior
    — if retries ever needed to accumulate rather than be set once per
    node — can be added here without touching merge_node_results' contract."""
    return {**left, **right}


# ══════════════════════════════════════════════════════════════════════
# NodeResult — one FINISHED entry per DAG agent node, written by the
# reflexion node after it judges a DraftAnswer (not by the agent node
# itself — see module docstring)
# ══════════════════════════════════════════════════════════════════════

class NodeResult(BaseModel):
    node_id: str
    agent_type: Literal["market", "filings", "macro", "sentiment"]
    depends_on: list[str] = Field(default_factory=list)

    answer: str                            # POST-reflexion — retry answer
                                            # (+ CAVEAT if still failing)
                                            # if a retry happened, never
                                            # the pre-retry draft
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)

    grounding_gate: GroundingGateResult
    critique: CritiqueResult
    retried: bool = False

    @model_validator(mode="after")
    def retry_implies_a_real_failure(self) -> "NodeResult":
        """A logical-consistency check no TypedDict could express: if
        retried=True, at least ONE of the two checks must have actually
        failed at some point — a retry claimed with no failing check on
        record is an inconsistent state, most likely a wiring bug (e.g.
        the reflexion node retried against the wrong draft, or recorded
        the post-retry gate/critique instead of what triggered the retry).
        NOTE: this checks the checks recorded on the FINAL answer, which
        may legitimately both be True if the retry fixed things — so this
        validator intentionally does NOT fire in that case. It only
        catches retried=True with passing checks AND no retry_guidance on
        record anywhere, which would mean nothing on this object explains
        why a retry supposedly happened."""
        if self.retried and self.grounding_gate.passed and self.critique.passed:
            if not self.critique.retry_guidance:
                raise ValueError(
                    "retried=True, both checks show passed=True on the "
                    "final answer, AND no retry_guidance is on record — "
                    "nothing here explains why a retry happened. If the "
                    "retry succeeded, retry_guidance should still show "
                    "what the FIRST attempt failed on."
                )
        return self


# ══════════════════════════════════════════════════════════════════════
# GraphState — the whole whiteboard, one instance per request
# ══════════════════════════════════════════════════════════════════════

class GraphState(BaseModel):
    # ── input — set once by the caller before graph.invoke() ──────────
    question: str
    session_id: str
    memory_context: str = ""

    # ── planner output — written once, by the planner node only ───────
    dag: dict[str, DagNodeSpec] = Field(default_factory=dict)

    # ── set by planner ONLY when it emits a clarify/decline sentinel —
    #    an EXPLICIT field rather than overloading `dag`'s shape or
    #    inferring the sentinel from dag contents. A conditional edge
    #    right after the planner node checks this FIRST, before dispatch
    #    ever runs, and short-circuits straight to final_answer (mirrors
    #    dag_executor.py's existing decline/clarify sentinel check,
    #    which today runs before any real agent is invoked) ───────────
    planner_sentinel: Optional[Literal["clarify", "decline"]] = None

    # ── written by agent nodes, BEFORE reflexion judges them ──────────
    drafts: Annotated[dict[str, DraftAnswer], merge_node_results] = Field(default_factory=dict)

    # ── written by the reflexion node, AFTER judging a draft ──────────
    node_results: Annotated[dict[str, NodeResult], merge_node_results] = Field(default_factory=dict)

    # ── retry cap enforcement — REQUIRED once reflexion is a graph
    #    cycle; a plain function's "if not passed: retry once" guarantee
    #    doesn't exist for free in a cycle, so the cap must live in state
    #    and be checked explicitly by the conditional edge ────────────
    retry_counts: Annotated[dict[str, int], merge_retry_counts] = Field(default_factory=dict)

    # ── synthesis tail — None-able because the single-agent shortcut
    #    never populates these ─────────────────────────────────────────
    synthesized_answer: Optional[str] = None
    synthesis_grounding_gate: Optional[GroundingGateResult] = None
    synthesis_critique: Optional[CritiqueResult] = None
    synthesis_retried: bool = False

    # ── request-level, run-once — not per-node ─────────────────────────
    injection_check: Optional[dict] = None

    # ── final exit ──────────────────────────────────────────────────
    final_answer: Optional[str] = None


def build_initial_state(
    question: str,
    session_id: str,
    memory_context: str = "",
) -> GraphState:
    """Construct the starting whiteboard. Called by the API/CLI entry
    point BEFORE graph.invoke() — the graph never creates a GraphState,
    only receives one and returns updates merged into it."""
    return GraphState(
        question=question,
        session_id=session_id,
        memory_context=memory_context,
    )