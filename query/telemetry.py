"""
LLMOps telemetry — writes agent execution traces to S3 as JSON.
Queryable via Athena for performance analysis and debugging.

Schema (one JSON file per agent run):
  trace_id        STRING    UUID for this trace
  session_id      STRING
  agent           STRING    MarketAgent | MacroAgent | FilingsAgent | orchestrator
  question        STRING    first 500 chars
  iterations      INTEGER
  tools_called    ARRAY     list of {name, inputs_preview, result_preview, was_dedup}
  total_tokens    INTEGER
  input_tokens    INTEGER
  output_tokens   INTEGER
  latency_ms      INTEGER
  hit_max_iter    BOOLEAN
  hit_token_budget BOOLEAN
  reflexion_triggered BOOLEAN
  reflexion_passed    BOOLEAN
  had_dedup_hits   BOOLEAN
  attribution_warnings ARRAY  warning strings from dag_executor's
                              unattributed-figure heuristic (empty for
                              normal per-agent traces; populated only on
                              the synthetic per-round trace dag_executor
                              writes when warnings fire)
  synthesis_reflexion_triggered BOOLEAN  set only on the synthetic
                              dag_executor trace written after the
                              synthesis-level critic pass
  synthesis_reflexion_passed    BOOLEAN
  planner_parse_failure_raw_output  STRING  set only on the synthetic
                              "planner" trace written when plan()'s
                              JSON parse fails (first 2000 chars of the
                              raw LLM text, truncated or otherwise)
  planner_parse_failure_stop_reason STRING  the API stop_reason for that
                              same failed call (e.g. "max_tokens")
  injection_suspected BOOLEAN  set on the synthetic dag_executor trace
                              written after reflexion.check_injection_provenance()
                              runs — True if the post-answer provenance
                              judge suspected a directive/recommendation
                              traced to embedded tool-result content
                              rather than the user's question
  reasoning_trail ARRAY     list of {tool_called, reasoning_text} entries,
                              one per tool call, in call order — the
                              model's stated reasoning text accompanying
                              that turn's tool_use block(s). New as of the
                              `trajectory` eval category; previously this
                              text was folded into conversation history
                              and discarded, never captured anywhere.
  answer_preview  STRING    first 500 chars of final answer
  model           STRING
  env             STRING
  timestamp       STRING    ISO UTC
  year            STRING    partition key
  month           STRING    partition key
"""

import os
import json
import uuid
import boto3
from datetime import datetime, timezone

ENV     = os.environ.get("ENV", "dev")
ACCOUNT = os.environ.get("ACCOUNT", "197411402303")
BUCKET  = f"{ENV}-trade-llmops-{ACCOUNT}"
REGION  = "us-east-2"

s3 = boto3.client("s3", region_name=REGION)


class Trace:
    """
    Collects telemetry for a single agent run.
    Use as a context object — pass to sub_agents, update during run,
    flush to S3 at end.
    """
    def __init__(self, session_id: str, agent: str,
                 question: str, model: str, node_id: str = None):
        self.trace_id            = str(uuid.uuid4())
        self.session_id          = session_id or "no-session"
        self.agent               = agent
        self.node_id              = node_id
        self.question            = question[:500]
        self.model               = model
        self.env                 = ENV
        self.start_time          = datetime.now(timezone.utc)

        self.iterations          = 0
        self.tools_called        = []
        self.input_tokens        = 0
        self.output_tokens       = 0

        self.hit_max_iter        = False
        self.hit_token_budget    = False
        self.reflexion_triggered = False
        self.reflexion_passed    = True
        self.had_dedup_hits      = False
        self.answer_preview      = ""
        self.attribution_warnings = []
        self.synthesis_reflexion_triggered = False
        self.synthesis_reflexion_passed    = True
        self.planner_parse_failure_raw_output  = None
        self.planner_parse_failure_stop_reason = None
        self.injection_suspected = False
        self.reasoning_trail = []

    def record_iteration(self):
        self.iterations += 1

    def record_reasoning_turn(self, tool_called: str, reasoning_text: str):
        """Record the model's stated reasoning text alongside the tool it
        called that same turn — e.g. a <scratchpad> block for FilingsAgent,
        or plain prose for other agents. Genuinely new instrumentation as
        of this method (confirmed via reading _run_agent()'s ReAct loop —
        this text was previously folded into `messages` for conversation
        continuity and then discarded; never captured anywhere). Used by
        the `trajectory` eval category's check_trajectory_adaptation() to
        judge whether reasoning before a later tool call genuinely adapted
        to an earlier tool's actual result content."""
        self.reasoning_trail.append({
            "tool_called":     tool_called,
            "reasoning_text":  reasoning_text[:1500],
        })

    def record_tool_call(self, name: str, inputs: dict, result: str,
                         was_dedup: bool = False):
        self.tools_called.append({
            "name":           name,
            "inputs_preview": str(inputs)[:500],
            "result_preview": result[:1500],
            "was_dedup":      was_dedup,
            # Full, untruncated result — in-memory only, for Reflexion's
            # grounding check. Stripped out in flush() before the S3 write.
            "result_full":    result,
        })
        if was_dedup:
            self.had_dedup_hits = True

    def record_tokens(self, input_tokens: int, output_tokens: int):
        self.input_tokens  += input_tokens
        self.output_tokens += output_tokens

    def record_answer(self, answer: str):
        self.answer_preview = answer[:500]

    def record_attribution_warnings(self, warnings: list):
        """Attach attribution-check warnings (see dag_executor.py's
        _check_unattributed_figures) so they're queryable via Athena
        alongside the rest of the trace, not just printed to console."""
        self.attribution_warnings = warnings

    def record_synthesis_reflexion(self, triggered: bool, passed: bool):
        """Record whether synthesis-level reflexion fired and its outcome."""
        self.synthesis_reflexion_triggered = triggered
        self.synthesis_reflexion_passed = passed

    def record_planner_parse_failure(self, raw_output: str, stop_reason: str):
        """Record a planner JSON parse failure (see planner.py's plan())
        so truncation/malformed-output rates are queryable over time
        instead of relying on a console print that's invisible when
        verbose=False (e.g. during eval runs)."""
        self.planner_parse_failure_raw_output  = raw_output
        self.planner_parse_failure_stop_reason = stop_reason

    def record_injection_check(self, suspected: bool):
        """Record the outcome of reflexion.check_injection_provenance()
        — see dag_executor.py's call sites (both the single-agent
        early-return path and the multi-agent synthesis tail)."""
        self.injection_suspected = suspected

    def flush(self):
        """Write trace to S3 as JSON, partitioned by year/month."""
        now       = datetime.now(timezone.utc)
        latency   = int((now - self.start_time).total_seconds() * 1000)
        year      = now.strftime("%Y")
        month     = now.strftime("%m")
        timestamp = now.isoformat()

        tools_called_for_s3 = [
            {k: v for k, v in tc.items() if k != "result_full"}
            for tc in self.tools_called
        ]

        record = {
            "trace_id":            self.trace_id,
            "session_id":          self.session_id,
            "agent":               self.agent,
            "node_id":             self.node_id,
            "question":            self.question,
            "iterations":          self.iterations,
            "tools_called":        tools_called_for_s3,
            "total_tokens":        self.input_tokens + self.output_tokens,
            "input_tokens":        self.input_tokens,
            "output_tokens":       self.output_tokens,
            "latency_ms":          latency,
            "hit_max_iter":        self.hit_max_iter,
            "hit_token_budget":    self.hit_token_budget,
            "reflexion_triggered": self.reflexion_triggered,
            "reflexion_passed":    self.reflexion_passed,
            "had_dedup_hits":      self.had_dedup_hits,
            "answer_preview":      self.answer_preview,
            "attribution_warnings": self.attribution_warnings,
            "synthesis_reflexion_triggered": self.synthesis_reflexion_triggered,
            "synthesis_reflexion_passed":    self.synthesis_reflexion_passed,
            "planner_parse_failure_raw_output":  self.planner_parse_failure_raw_output,
            "planner_parse_failure_stop_reason": self.planner_parse_failure_stop_reason,
            "injection_suspected": self.injection_suspected,
            "reasoning_trail":     self.reasoning_trail,
            "model":               self.model,
            "env":                 self.env,
            "timestamp":           timestamp,
            "year":                year,
            "month":               month,
        }

        key = (
            f"traces/year={year}/month={month}/"
            f"{self.agent}_{self.trace_id}.json"
        )

        try:
            s3.put_object(
                Bucket=BUCKET,
                Key=key,
                Body=json.dumps(record, indent=2),
                ContentType="application/json",
            )
        except Exception as e:
            print(f"  [Telemetry] flush failed — {e}")
