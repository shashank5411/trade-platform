"""
run_eval.py — Eval runner for the trade-platform DAG planner / agent system.

Loads query/evaluations/questions.yaml, runs each active question through the
real planner + executor (same code path as query/agent.py), scores by category,
and writes a single JSON result log to query/evaluations/results/.

Usage:
    python query/evaluations/run_eval.py
    python query/evaluations/run_eval.py --category routing
    python query/evaluations/run_eval.py --include-retired
    python query/evaluations/run_eval.py --id PAIR-MARKET-MACRO-001

Scoring by category:
    routing         — free, string-match: did the DAG contain exactly the
                      expected agents, and did its shape (single/parallel/
                      sequential) match expectations? No LLM call.
    tool_selection   — free, string-match: did the named agent(s) call the
                      expected tool(s) at least once? No LLM call.
    grounding        — cheap, one Haiku judge call per question: does the
                      final answer avoid forbidden phrases and, if specified,
                      contain the expected substring? Forbidden-phrase check
                      itself is free string-match; the judge call is only
                      used for the subtler "did it stay grounded" assessment.

Output: query/evaluations/results/run_<timestamp>.json
    {
      "summary": { ... },
      "results": [ {...one record per question...} ]
    }
"""

import argparse
import asyncio
import datetime
import json
import os
import sys
import threading
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from query.planner import plan, _resolve_rounds
from query.dag_executor import execute
from query.config import get_client

JUDGE_MODEL = "claude-haiku-4-5-20251001"
QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.yaml")

# Rough cost estimate only — actual pricing may differ by exact model
# version/date and doesn't account for prompt-cache discounts (your real
# cache hit rate this session was ~25-27%, which would lower true cost
# below this estimate). Update these if pricing changes; treat the
# resulting estimated_cost_usd field as directional, not exact.
HAIKU_INPUT_COST_PER_MTOK  = 1.00   # USD per 1M input tokens
HAIKU_OUTPUT_COST_PER_MTOK = 5.00   # USD per 1M output tokens
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


class Tee:
    """
    Duplicates writes to multiple streams (e.g. real stdout + a log file).
    Used to capture the full console output of an eval run — every
    [Planner]/[Executor]/[AgentName] print line — into a file alongside
    the JSON results, without touching every print() call scattered
    across planner.py/dag_executor.py/sub_agents.py.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ── Loading ──────────────────────────────────────────────────────────────────

def load_questions(include_retired: bool = False, category: str = None,
                    only_id: str = None) -> list:
    with open(QUESTIONS_PATH) as f:
        data = yaml.safe_load(f)
    qs = data["questions"]

    if not include_retired:
        qs = [q for q in qs if q.get("status") == "active"]
    if category:
        qs = [q for q in qs if q.get("category") == category]
    if only_id:
        qs = [q for q in qs if q.get("id") == only_id]

    return qs


# ── DAG shape classification ─────────────────────────────────────────────────

def classify_dag_shape(dag: dict) -> str:
    """Mirror the schema's expected_dag_shape values: single | parallel | sequential."""
    if len(dag) <= 1:
        return "single"
    has_dependency = any(v.get("depends_on") for v in dag.values())
    return "sequential" if has_dependency else "parallel"


# ── Tool-call extraction ─────────────────────────────────────────────────────
# sub_agents.py's _execute_tool now accepts an explicit agent_name kwarg
# (passed from all three call sites: _run_one's normal path, _run_one's
# dedup path doesn't call it at all, and Reflexion's retry_fn loop). This
# replaces an earlier thread-identity-guessing approach that broke whenever
# two agents were both mid-flight with overlapping inner thread pools —
# explicit is correct by construction, no inference needed.

_captured_tool_calls = {}
_capture_lock = threading.Lock()

# Token tracking — keyed by node_id (e.g. "market_1"), not agent type,
# since the same agent type can run multiple times in one question now.
_captured_tokens = {}
_token_lock = threading.Lock()


def _wrap_tool_execution():
    """
    Monkeypatch _execute_tool (for tool-call attribution) and Trace's
    __init__/record_tokens (for per-node token counts). Both reset at the
    start of each question and read back into the result record after.
    """
    from query import sub_agents as sa
    from query import telemetry as tm

    original_execute_tool = sa._execute_tool
    original_trace_init   = tm.Trace.__init__
    original_record_tokens = tm.Trace.record_tokens

    def tracking_execute_tool(name, inputs, agent_name=None):
        key = agent_name or "_unattributed"
        with _capture_lock:
            _captured_tool_calls.setdefault(key, []).append(name)
        return original_execute_tool(name, inputs, agent_name=agent_name)

    def tracking_trace_init(self, *args, **kwargs):
        original_trace_init(self, *args, **kwargs)
        # node_id may be passed positionally or as a kwarg depending on
        # caller — Trace's real signature takes it as a kwarg, so this is
        # safe, but fall back to the trace's own agent name if somehow
        # node_id wasn't set, so tokens are still attributed to something.
        key = getattr(self, "node_id", None) or getattr(self, "agent", "_unknown_node")
        with _token_lock:
            _captured_tokens.setdefault(key, {"input_tokens": 0, "output_tokens": 0})
        self._eval_token_key = key

    def tracking_record_tokens(self, input_tokens, output_tokens):
        original_record_tokens(self, input_tokens, output_tokens)
        key = getattr(self, "_eval_token_key", "_unknown_node")
        with _token_lock:
            bucket = _captured_tokens.setdefault(key, {"input_tokens": 0, "output_tokens": 0})
            bucket["input_tokens"]  += input_tokens
            bucket["output_tokens"] += output_tokens

    sa._execute_tool        = tracking_execute_tool
    tm.Trace.__init__       = tracking_trace_init
    tm.Trace.record_tokens  = tracking_record_tokens

    return (original_execute_tool, original_trace_init, original_record_tokens)


def _unwrap_tool_execution(originals):
    from query import sub_agents as sa
    from query import telemetry as tm
    original_execute_tool, original_trace_init, original_record_tokens = originals
    sa._execute_tool       = original_execute_tool
    tm.Trace.__init__      = original_trace_init
    tm.Trace.record_tokens = original_record_tokens


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_routing(question: dict, actual_agents: list, dag: dict) -> dict:
    # actual_agents may be node_ids (e.g. "market_1") rather than agent
    # types (e.g. "market") now that the planner always uses node-ID
    # naming, even for single-agent plans. Normalize via dag[node_id]["agent"]
    # before comparing against expected_agents, which is written in terms
    # of agent TYPES in questions.yaml, not node_ids.
    actual_types = [
        dag.get(node_id, {}).get("agent", node_id)
        for node_id in actual_agents
    ]

    expected = set(question.get("expected_agents", []))
    actual = set(actual_types)
    agents_match = expected == actual

    expected_shape = question.get("expected_dag_shape")
    actual_shape = classify_dag_shape(dag)
    shape_match = (expected_shape is None) or (expected_shape == actual_shape)

    passed = agents_match and shape_match
    reasons = []
    if not agents_match:
        reasons.append(f"expected agents {sorted(expected)}, got {sorted(actual)}")
    if not shape_match:
        reasons.append(f"expected shape '{expected_shape}', got '{actual_shape}'")

    return {
        "pass": passed,
        "actual_dag_shape": actual_shape,
        "actual_agent_types": sorted(actual),
        "reasons": reasons,
    }


def score_tool_selection(question: dict, called_tools_by_agent: dict) -> dict:
    expected_tools = set(question.get("expected_tools", []))
    if not expected_tools:
        return {"pass": None, "reasons": ["no expected_tools specified"]}

    # Pool tool calls across all agents that ran for this question — the
    # schema doesn't currently attribute expected_tools to a specific agent,
    # so "was this tool called by anyone in the DAG" is the check. If a
    # future question needs agent-specific tool attribution, extend the
    # schema with a per-agent expected_tools map rather than guessing here.
    all_called = set()
    for agent_calls in called_tools_by_agent.values():
        all_called.update(agent_calls)

    missing = expected_tools - all_called
    passed = len(missing) == 0

    return {
        "pass": passed,
        "called_tools_by_agent": called_tools_by_agent,
        "reasons": [] if passed else [f"missing expected tool calls: {sorted(missing)}"],
    }


def score_grounding(question: dict, answer: str) -> dict:
    forbidden = question.get("forbidden_phrases", []) or []
    negation_markers = ["not ", "no ", "didn't", "doesn't", "does not",
                         "isn't", "wasn't", "without evidence of",
                         "data does not support", "does not establish"]

    hits = []
    answer_lower = answer.lower()
    for phrase in forbidden:
        phrase_lower = phrase.lower()
        start = 0
        while True:
            idx = answer_lower.find(phrase_lower, start)
            if idx == -1:
                break
            # Check a window before the phrase for a negation marker
            window_start = max(0, idx - 50)
            window = answer_lower[window_start:idx]
            negated = any(marker in window for marker in negation_markers)
            if not negated:
                hits.append(phrase)
                break  # one unnegated hit is enough to flag this phrase
            start = idx + len(phrase_lower)

    expected_substring = question.get("expected_answer_contains")
    substring_ok = (expected_substring is None) or (expected_substring.lower() in answer_lower)

    passed = (len(hits) == 0) and substring_ok
    reasons = []
    if hits:
        reasons.append(f"forbidden phrases found (not negated): {hits}")
    if not substring_ok:
        reasons.append(f"expected substring not found: '{expected_substring}'")

    return {
        "pass": passed,
        "forbidden_phrase_hits": hits,
        "reasons": reasons,
    }


# ── Per-question execution ───────────────────────────────────────────────────

def run_one_question(question: dict, verbose: bool = True) -> dict:
    qid = question["id"]
    q_text = question["question"]
    category = question["category"]

    if verbose:
        print(f"\n{'='*60}\n[{qid}] {q_text}\n{'='*60}")

    record = {
        "run_id": None,  # filled in by caller
        "question_id": qid,
        "question": q_text,
        "category": category,
        "expected_agents": question.get("expected_agents", []),
        "actual_agents": [],
        "actual_agent_types": [],
        "expected_dag_shape": question.get("expected_dag_shape"),
        "actual_dag_shape": None,
        "routing_pass": None,
        "tool_selection_pass": None,
        "actual_tools_by_agent": {},
        "grounding_pass": None,
        "forbidden_phrase_hits": [],
        "planner_reasoning": None,
        "final_answer": None,
        "tokens_by_node": {},
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "latency_ms": None,
        "error": None,
    }

    start = time.time()
    global _captured_tool_calls, _captured_tokens
    _captured_tool_calls = {}
    _captured_tokens = {}
    originals = _wrap_tool_execution()

    try:
        dag = plan(q_text, verbose=verbose)
        record["actual_agents"] = list(dag.keys())
        record["actual_dag_shape"] = classify_dag_shape(dag)

        answer = asyncio.run(execute(q_text, dag, verbose=verbose))
        record["final_answer"] = answer

        record["actual_tools_by_agent"] = dict(_captured_tool_calls)

        record["tokens_by_node"] = dict(_captured_tokens)
        total_in  = sum(v["input_tokens"]  for v in _captured_tokens.values())
        total_out = sum(v["output_tokens"] for v in _captured_tokens.values())
        record["total_input_tokens"]  = total_in
        record["total_output_tokens"] = total_out
        record["estimated_cost_usd"] = round(
            (total_in  / 1_000_000) * HAIKU_INPUT_COST_PER_MTOK +
            (total_out / 1_000_000) * HAIKU_OUTPUT_COST_PER_MTOK,
            5,
        )

        # Routing score
        routing_result = score_routing(question, list(dag.keys()), dag)
        record["routing_pass"] = routing_result["pass"]
        record["actual_agent_types"] = routing_result["actual_agent_types"]

        # Tool-selection score (only meaningful if expected_tools was specified)
        if question.get("expected_tools"):
            tool_result = score_tool_selection(question, record["actual_tools_by_agent"])
            record["tool_selection_pass"] = tool_result["pass"]

        # Grounding score (only meaningful if category is grounding or
        # forbidden_phrases/expected_answer_contains was specified)
        if category == "grounding" or question.get("forbidden_phrases") or question.get("expected_answer_contains"):
            grounding_result = score_grounding(question, answer or "")
            record["grounding_pass"] = grounding_result["pass"]
            record["forbidden_phrase_hits"] = grounding_result["forbidden_phrase_hits"]

    except Exception as e:
        record["error"] = str(e)
        if verbose:
            print(f"  [ERROR] {e}")
    finally:
        _unwrap_tool_execution(originals)

    record["latency_ms"] = int((time.time() - start) * 1000)
    return record


# ── Run-level summary ─────────────────────────────────────────────────────────

def build_summary(run_id: str, records: list) -> dict:
    by_category = {}
    failures = []

    for r in records:
        cat = r["category"]
        by_category.setdefault(cat, {"total": 0, "passed": 0})
        by_category[cat]["total"] += 1

        # A question "passes" if every score that was actually computed for
        # it is True, and there was no execution error.
        scores = [r["routing_pass"], r["tool_selection_pass"], r["grounding_pass"]]
        computed_scores = [s for s in scores if s is not None]
        question_passed = (r["error"] is None) and all(computed_scores) if computed_scores else (r["error"] is None)

        if question_passed:
            by_category[cat]["passed"] += 1
        else:
            reason_parts = []
            if r["error"]:
                reason_parts.append(f"error: {r['error']}")
            if r["routing_pass"] is False:
                reason_parts.append(
                    f"routing mismatch (expected {r['expected_agents']}, "
                    f"got {r.get('actual_agent_types', r['actual_agents'])})"
                )
            if r["tool_selection_pass"] is False:
                reason_parts.append("tool selection mismatch")
            if r["grounding_pass"] is False:
                reason_parts.append(f"grounding failure: {r['forbidden_phrase_hits']}")
            failures.append({
                "id": r["question_id"],
                "category": cat,
                "reason": "; ".join(reason_parts) or "unknown",
            })

    pass_rates = {
        cat: f"{v['passed']}/{v['total']}"
        for cat, v in by_category.items()
    }

    total_input_tokens  = sum(r.get("total_input_tokens", 0) for r in records)
    total_output_tokens = sum(r.get("total_output_tokens", 0) for r in records)
    total_estimated_cost = round(sum(r.get("estimated_cost_usd", 0.0) for r in records), 4)

    return {
        "run_id": run_id,
        "total_questions": len(records),
        "pass_rates_by_category": pass_rates,
        "total_passed": sum(v["passed"] for v in by_category.values()),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_total_cost_usd": total_estimated_cost,
        "note": "Cost estimate uses placeholder Haiku pricing and does not "
                "account for prompt-cache discounts — treat as directional, "
                "compare against your actual billing dashboard delta.",
        "failures": failures,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["routing", "tool_selection", "grounding"])
    parser.add_argument("--include-retired", action="store_true")
    parser.add_argument("--id", help="Run only this single question ID")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    questions = load_questions(
        include_retired=args.include_retired,
        category=args.category,
        only_id=args.id,
    )

    if not questions:
        print("No matching active questions found.")
        return

    run_id  = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = os.path.join(RESULTS_DIR, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "console.log")
    json_path = os.path.join(run_dir, "results.json")

    real_stdout = sys.stdout
    log_file    = open(log_path, "w", encoding="utf-8")
    sys.stdout  = Tee(real_stdout, log_file)

    try:
        print(f"\n[Eval Run {run_id}] {len(questions)} questions to run\n")

        records = []
        for q in questions:
            record = run_one_question(q, verbose=not args.quiet)
            record["run_id"] = run_id
            records.append(record)

        summary = build_summary(run_id, records)

        with open(json_path, "w") as f:
            json.dump({"summary": summary, "results": records}, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"  EVAL RUN COMPLETE — {run_id}")
        print(f"{'='*60}")
        print(f"  Total: {summary['total_questions']}  Passed: {summary['total_passed']}")
        print(f"  By category: {summary['pass_rates_by_category']}")
        print(f"  Tokens: {summary['total_input_tokens']:,} in / "
              f"{summary['total_output_tokens']:,} out — "
              f"est. cost: ${summary['estimated_total_cost_usd']:.4f} "
              f"(rough estimate, see note in JSON)")
        if summary["failures"]:
            print(f"\n  Failures:")
            for f in summary["failures"]:
                print(f"    [{f['id']}] ({f['category']}) {f['reason']}")
        print(f"\n  Results folder: {run_dir}")
        print(f"  JSON:    {json_path}")
        print(f"  Console: {log_path}\n")
    finally:
        sys.stdout = real_stdout
        log_file.close()


if __name__ == "__main__":
    main()