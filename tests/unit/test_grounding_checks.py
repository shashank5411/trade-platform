"""
Unit tests for the deterministic attribution/inversion grounding gate.

check_attribution()/check_inversion()/_run_grounding_gate()/_build_retry_guidance()
now live in query/reflexion.py (moved from query/dag_executor.py to avoid a
circular import — apply_reflexion(), also in reflexion.py, needs to call
them directly; see GROUNDING_CHECKS_IMPLEMENTATION.md's "single-agent gate
coverage" section for why). dag_executor.py imports these same names via
`from query.reflexion import ...`, so tests that reference them as
`dag_executor.check_attribution` etc. (from before the move) continue to
work unchanged — Python resolves those as module-namespace lookups at call
time, regardless of which module originally defined the function. Newer
tests below reference `reflexion.X` directly since that's now the true
defining module.

Also covers: unit-scaling and single-pair delta/%-change matching in
check_attribution() (fixes a real false-positive that caused reflexion to
delete legitimate content — see the "unit-scaling and derived-value
matching" section below), and apply_reflexion()'s extension to run the
same grounding gate for every per-agent DAG node (single-agent DAGs and
each node of a multi-agent DAG alike).

No existing test suite covers query/ logic (the only prior test file,
tests/unit/test_trade_platform_stack.py, is a stubbed CDK infra test) —
this establishes the pattern: plain pytest functions, no classes/fixtures
beyond what's needed, matching that file's style.
"""

import asyncio
from types import SimpleNamespace

from query import dag_executor
from query import reflexion


def _tool_call(name, result_full, was_dedup=False):
    """Build a minimal Trace.tools_called-shaped record (see telemetry.py's
    record_tool_call()) — the real shape check_attribution()/check_inversion()
    consume."""
    return {
        "name": name,
        "inputs_preview": "{}",
        "result_preview": result_full[:200],
        "was_dedup": was_dedup,
        "result_full": result_full,
    }


def _fake_response(text):
    """Minimal stand-in for an anthropic.types.Message — only
    response.content[0].text is ever read by dag_executor.py's call sites."""
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


class _FakeMessages:
    def __init__(self, outer):
        self._outer = outer

    def create(self, **kwargs):
        self._outer.calls += 1
        return _fake_response(self._outer.retry_text)


class _FakeClient:
    """Records how many times messages.create() was invoked, so tests can
    assert the retry cap (exactly one retry) is respected."""
    def __init__(self, retry_text):
        self.retry_text = retry_text
        self.calls = 0
        self.messages = _FakeMessages(self)


# ══════════════════════════════════════════════════════════════════════
# check_attribution — output -> source
# ══════════════════════════════════════════════════════════════════════

def test_attribution_catches_fabricated_number():
    node_tool_calls = {
        "market_1": [_tool_call("get_prices", "AAPL closed at $184.20 on 2026-06-30.")],
    }
    answer = "AAPL closed at $184.20, up from a low of $150.00 earlier in the month."

    failures = dag_executor.check_attribution(answer, node_tool_calls)

    assert len(failures) == 1
    assert failures[0]["kind"] == "attribution"
    assert failures[0]["value"] == 150.00


def test_attribution_passes_number_within_tolerance():
    # Tool result has more decimal precision than the answer rounds to —
    # same underlying value, different formatting.
    node_tool_calls = {
        "market_1": [_tool_call("get_prices", "close: 184.2035")],
    }
    answer = "AAPL closed at $184.20."

    failures = dag_executor.check_attribution(answer, node_tool_calls)

    assert failures == []


def test_attribution_passes_when_answer_has_no_figures():
    node_tool_calls = {
        "market_1": [_tool_call("get_prices", "close: 184.20")],
    }
    failures = dag_executor.check_attribution("No numeric claims here.", node_tool_calls)
    assert failures == []


# ══════════════════════════════════════════════════════════════════════
# check_inversion — source -> output (naive scope, non-blocking — see
# GROUNDING_CHECKS_IMPLEMENTATION.md)
# ══════════════════════════════════════════════════════════════════════

def test_inversion_catches_unused_fetched_value():
    node_tool_calls = {
        "market_1": [_tool_call("get_prices", "start price 150.00, end price 184.20")],
    }
    answer = "AAPL rose to $184.20 over the period."

    failures = dag_executor.check_inversion(answer, node_tool_calls)

    assert len(failures) == 1
    assert failures[0]["kind"] == "inversion"
    assert failures[0]["value"] == 150.00
    assert failures[0]["node_id"] == "market_1"
    assert failures[0]["tool"] == "get_prices"


def test_inversion_passes_when_all_relevant_values_used():
    node_tool_calls = {
        "market_1": [_tool_call("get_prices", "start price 150.00, end price 184.20")],
    }
    answer = "AAPL rose from $150.00 to $184.20 over the period."

    failures = dag_executor.check_inversion(answer, node_tool_calls)

    assert failures == []


def test_inversion_ignores_deduped_tool_calls():
    # was_dedup=True marks _run_agent()'s own "you already called this"
    # nudge text, not real fetched data — must never be treated as a
    # fetched value.
    node_tool_calls = {
        "market_1": [_tool_call("get_prices", "start price 150.00", was_dedup=True)],
    }
    failures = dag_executor.check_inversion("No numbers mentioned that matter.", node_tool_calls)
    assert failures == []


# ══════════════════════════════════════════════════════════════════════
# Combined case — one fabricated number AND one dropped fetched value in
# the same answer (the AAPL revenue/price shape from the design discussion)
# ══════════════════════════════════════════════════════════════════════

def test_combined_fabrication_and_dropped_value_populate_both_lists():
    node_tool_calls = {
        "filings_1": [_tool_call("get_prose", "Revenue for FY2025 was $391.0 billion.")],
        "market_1":  [_tool_call("get_prices", "start price 150.00, end price 184.20")],
    }
    answer = (
        "AAPL reported revenue of $391.0 billion for FY2025, a 12.4% "
        "increase driven by strong iPhone sales, and the stock ended "
        "the period at $184.20."
    )
    # 12.4% is fabricated (not in any tool result).
    # 150.00 was fetched but never mentioned in the answer.
    # 391.0 and 184.20 are both correctly grounded.

    attribution_failures = dag_executor.check_attribution(answer, node_tool_calls)
    inversion_failures = dag_executor.check_inversion(answer, node_tool_calls)

    assert len(attribution_failures) == 1
    assert attribution_failures[0]["value"] == 12.4

    assert len(inversion_failures) == 1
    assert inversion_failures[0]["value"] == 150.00

    guidance = dag_executor._build_retry_guidance(
        {"passed": True, "issues": []},
        {"attribution_failures": attribution_failures, "inversion_failures": inversion_failures},
    )
    assert "NUMERIC VERIFICATION FAILURES (attribution)" in guidance
    assert "12.4" in guidance
    # Inversion is non-blocking today (INVERSION_BLOCKING = False) — its
    # failures must NOT leak into retry guidance yet.
    assert "inversion" not in guidance.lower()


# ══════════════════════════════════════════════════════════════════════
# _run_grounding_gate — merged gate semantics
# ══════════════════════════════════════════════════════════════════════

def test_grounding_gate_blocks_on_attribution_failure():
    node_tool_calls = {"market_1": [_tool_call("get_prices", "close 184.20")]}
    answer = "Price is $999.99."  # fabricated

    gate = asyncio.run(dag_executor._run_grounding_gate(answer, node_tool_calls))

    assert gate["attribution_passed"] is False
    assert gate["passed"] is False


def test_grounding_gate_does_not_block_on_inversion_alone():
    node_tool_calls = {
        "market_1": [_tool_call("get_prices", "start 150.00, end 184.20")],
    }
    # Every number IN the answer is grounded (184.20 matches) — attribution
    # passes. 150.00 was fetched but never mentioned — inversion fails.
    answer = "Price ended at $184.20."

    gate = asyncio.run(dag_executor._run_grounding_gate(answer, node_tool_calls))

    assert gate["attribution_passed"] is True
    assert gate["inversion_passed"] is False
    assert gate["inversion_failures"] != []
    assert gate["passed"] is True, (
        "inversion failures must not block the gate while "
        "INVERSION_BLOCKING is False"
    )


# ══════════════════════════════════════════════════════════════════════
# _resolve_synthesis — retry cap + caveat behavior fed by the merged gate
# ══════════════════════════════════════════════════════════════════════

def test_resolve_synthesis_passes_clean_answer_without_retry(monkeypatch):
    node_tool_calls = {"market_1": [_tool_call("get_prices", "close 184.20")]}
    # Padding has no numbers of its own — pushes the answer safely over
    # SYNTH_REFLEXION_MIN_WORDS (200) so the LLM critique actually runs,
    # without adding any attribution/inversion complexity of its own.
    padding = (
        "Sector conditions remained broadly stable through the quarter, "
        "with trading volumes in line with recent averages and no "
        "material change in investor positioning noted across the "
        "coverage period. "
    ) * 10
    clean_answer = "AAPL closed at $184.20 after a steady climb this quarter. " + padding

    critique_calls = {"count": 0}

    def fake_critique_synthesis(question, agent_outputs, answer, verbose):
        critique_calls["count"] += 1
        return {"passed": True, "issues": []}

    fake_client = _FakeClient(retry_text="unused — no retry should happen")
    monkeypatch.setattr(dag_executor, "critique_synthesis", fake_critique_synthesis)
    monkeypatch.setattr(dag_executor, "client", fake_client)

    final_answer, synth_trace = asyncio.run(dag_executor._resolve_synthesis(
        question="How did AAPL do?",
        agent_outputs="[MARKET ANALYSIS (market_1)]\nAAPL closed at $184.20.",
        synthesis_prompt="irrelevant for this test",
        synthesized_answer=clean_answer,
        node_tool_calls=node_tool_calls,
        session_id=None,
        verbose=False,
    ))

    assert final_answer == clean_answer
    assert fake_client.calls == 0, "no retry should fire when the gate passes"
    assert critique_calls["count"] == 1
    assert synth_trace.synthesis_reflexion_triggered is False
    assert synth_trace.synthesis_reflexion_passed is True
    assert synth_trace.grounding_gate_passed is True


def test_resolve_synthesis_retries_once_then_caveats_if_still_failing(monkeypatch):
    """Both retry attempts contain a fabricated number never present in
    any tool result — the deterministic gate fails both times. Confirms
    the existing retry cap (exactly one retry) and CAVEAT-on-exhaustion
    behavior is preserved when the FAILURE originates from the new
    deterministic gate rather than the LLM critic."""
    node_tool_calls = {"market_1": [_tool_call("get_prices", "close 184.20")]}
    initial_answer = "Price is $999.99, way outside anything fetched."

    def fake_critique_synthesis(question, agent_outputs, answer, verbose):
        # Isolate: LLM critic always says "fine" — any failure here must
        # come from the deterministic gate, not this mock.
        return {"passed": True, "issues": []}

    fake_client = _FakeClient(retry_text="Still wrong at $888.88, also unfetched.")
    monkeypatch.setattr(dag_executor, "critique_synthesis", fake_critique_synthesis)
    monkeypatch.setattr(dag_executor, "client", fake_client)

    final_answer, synth_trace = asyncio.run(dag_executor._resolve_synthesis(
        question="How did AAPL do?",
        agent_outputs="[MARKET ANALYSIS (market_1)]\nclose 184.20",
        synthesis_prompt="irrelevant for this test",
        synthesized_answer=initial_answer,
        node_tool_calls=node_tool_calls,
        session_id=None,
        verbose=False,
    ))

    assert fake_client.calls == 1, "exactly one retry attempt — cap must not be exceeded"
    assert final_answer == fake_client.retry_text + dag_executor.CAVEAT
    assert synth_trace.synthesis_reflexion_triggered is True
    assert synth_trace.synthesis_reflexion_passed is False
    assert synth_trace.grounding_gate_passed is False
    assert synth_trace.attribution_failures != []


def test_resolve_synthesis_retry_can_pass_and_drop_caveat(monkeypatch):
    """Same shape as above, but the retry answer is actually grounded —
    confirms a successful retry returns the retry text with no CAVEAT."""
    node_tool_calls = {"market_1": [_tool_call("get_prices", "close 184.20")]}
    initial_answer = "Price is $999.99, way outside anything fetched."

    def fake_critique_synthesis(question, agent_outputs, answer, verbose):
        return {"passed": True, "issues": []}

    fake_client = _FakeClient(retry_text="Corrected: price closed at $184.20.")
    monkeypatch.setattr(dag_executor, "critique_synthesis", fake_critique_synthesis)
    monkeypatch.setattr(dag_executor, "client", fake_client)

    final_answer, synth_trace = asyncio.run(dag_executor._resolve_synthesis(
        question="How did AAPL do?",
        agent_outputs="[MARKET ANALYSIS (market_1)]\nclose 184.20",
        synthesis_prompt="irrelevant for this test",
        synthesized_answer=initial_answer,
        node_tool_calls=node_tool_calls,
        session_id=None,
        verbose=False,
    ))

    assert fake_client.calls == 1
    assert final_answer == fake_client.retry_text
    assert dag_executor.CAVEAT not in final_answer
    assert synth_trace.synthesis_reflexion_triggered is True
    assert synth_trace.synthesis_reflexion_passed is True
    assert synth_trace.grounding_gate_passed is True


def test_resolve_synthesis_skips_llm_critique_for_short_clean_answer(monkeypatch):
    """The word-count skip gate (SYNTH_REFLEXION_MIN_WORDS) must still only
    govern the LLM critique_synthesis() call — the deterministic gate is
    cheap and always runs regardless. A short, figure-free answer should
    pass without ever invoking the (mocked) LLM critic."""
    node_tool_calls = {"market_1": [_tool_call("get_prices", "close 184.20")]}
    short_answer = "AAPL performance was mixed this week."  # well under 200 words, no figures

    critique_calls = {"count": 0}

    def fake_critique_synthesis(question, agent_outputs, answer, verbose):
        critique_calls["count"] += 1
        return {"passed": False, "issues": ["should never be called"]}

    fake_client = _FakeClient(retry_text="unused")
    monkeypatch.setattr(dag_executor, "critique_synthesis", fake_critique_synthesis)
    monkeypatch.setattr(dag_executor, "client", fake_client)

    final_answer, synth_trace = asyncio.run(dag_executor._resolve_synthesis(
        question="How did AAPL do?",
        agent_outputs="[MARKET ANALYSIS (market_1)]\nclose 184.20",
        synthesis_prompt="irrelevant for this test",
        synthesized_answer=short_answer,
        node_tool_calls=node_tool_calls,
        session_id=None,
        verbose=False,
    ))

    assert critique_calls["count"] == 0, "LLM critique must be skipped under the word-count gate"
    assert fake_client.calls == 0
    assert final_answer == short_answer


# ══════════════════════════════════════════════════════════════════════
# Problem 1 — unit-scaling and single-pair delta/%-change matching
#
# Real bug: "Trading was heaviest on June 26 with 261.8 million shares,
# following a sharp 6.2% dip." Both numbers are legitimate derivations
# from grounded data (raw volume 261775500.0; 6.2% computed from two
# grounded prices) but the original check_attribution() flagged both,
# and the reflexion retry "fixed" this by deleting the sentence rather
# than correctly grounding it.
# ══════════════════════════════════════════════════════════════════════

def test_attribution_matches_unit_scaled_million_value():
    node_tool_calls = {
        "market_1": [_tool_call(
            "get_prices",
            "date volume open close\n2026-06-26 261775500.0 161.20 151.17",
        )],
    }
    answer = "Trading was heaviest on June 26 with 261.8 million shares."

    failures = reflexion.check_attribution(answer, node_tool_calls)

    assert failures == []


def test_attribution_matches_single_pair_percentage_change():
    # 151.17 vs 161.20 -> (151.17-161.20)/161.20*100 = -6.222%, whose
    # positive-magnitude counterpart (+6.222) is what a "6.2% dip" phrase
    # extracts as, since _FIGURE_PATTERN never captures a minus sign.
    node_tool_calls = {
        "market_1": [_tool_call("get_prices", "open 161.20, close 151.17")],
    }
    answer = "The stock saw a sharp 6.2% dip during the session."

    failures = reflexion.check_attribution(answer, node_tool_calls)

    assert failures == []


def test_attribution_exact_real_bug_case_now_passes():
    """The exact combined case from the real run — both the unit-scaled
    volume and the derived percentage in the SAME answer."""
    node_tool_calls = {
        "market_1": [_tool_call(
            "get_prices",
            "date volume open close\n2026-06-26 261775500.0 161.20 151.17",
        )],
    }
    answer = (
        "Trading was heaviest on June 26 with 261.8 million shares, "
        "following a sharp 6.2% dip."
    )

    failures = reflexion.check_attribution(answer, node_tool_calls)

    assert failures == [], (
        "both the unit-scaled volume and the derived percentage change "
        "should now be recognized as grounded"
    )


def test_attribution_still_catches_genuinely_fabricated_number():
    """A number that is NOT derivable via scaling or a single-pair delta
    from any grounded value must still fail — confirms the fix didn't
    make the check too permissive."""
    node_tool_calls = {
        "market_1": [_tool_call(
            "get_prices",
            "date volume open close\n2026-06-26 261775500.0 161.20 151.17",
        )],
    }
    answer = "Revenue reportedly grew 45.0% year over year, a remarkable turnaround."

    failures = reflexion.check_attribution(answer, node_tool_calls)

    assert len(failures) == 1
    assert failures[0]["value"] == 45.0


def test_unit_scaled_match_guards_against_small_source_values():
    """A source value at or under 1000 must never be treated as a
    plausible scaling base — otherwise a small, coincidentally-close
    answer number (e.g. 0.005) could spuriously match an unrelated small
    fetched value (5.0) as if it had been divided by 1000."""
    assert reflexion._is_unit_scaled_match(0.005, 5.0) is False
    # But the same relationship IS accepted once the source is plausibly
    # large enough to have been expressed in scaled form.
    assert reflexion._is_unit_scaled_match(5.0, 5000.0) is True


def test_inversion_recognizes_unit_scaled_restatement_as_used():
    """A fetched value correctly restated in scaled form (e.g. '261.8
    million' for a raw 261775500.0) must be recognized as USED by
    inversion, not flagged as dropped — inversion reuses the same
    _value_grounded_in() matching as attribution."""
    node_tool_calls = {
        "market_1": [_tool_call("get_prices", "volume 261775500.0")],
    }
    answer = "Volume reached 261.8 million shares."

    failures = reflexion.check_inversion(answer, node_tool_calls)

    assert failures == []


def test_resolve_synthesis_preserves_legitimate_content_no_retry(monkeypatch):
    """Confirms the real harm is fixed: before this fix, a synthesized
    answer containing the 261.8M/6.2% pattern would fail the grounding
    gate, trigger a retry, and risk the retry deleting the sentence
    entirely rather than correctly grounding it (the observed real-run
    behavior). After the fix, the gate passes cleanly on the FIRST pass —
    zero retries fire, so final_answer is guaranteed byte-identical to
    the original synthesized_answer. This is the strongest possible
    'diff pre-retry vs post-retry' confirmation: there is no retry to
    diff against, because none fires."""
    node_tool_calls = {
        "filings_1": [_tool_call(
            "get_prices",
            "date volume open close\n2026-06-26 261775500.0 161.20 151.17",
        )],
    }
    original_sentence = (
        "Trading was heaviest on June 26 with 261.8 million shares, "
        "following a sharp 6.2% dip. "
    )
    padding = (
        "Sector conditions remained broadly stable through the quarter, "
        "with trading volumes in line with recent averages and no "
        "material change in investor positioning noted across the "
        "coverage period. "
    ) * 10
    synthesized_answer = original_sentence + padding

    def fake_critique_synthesis(question, agent_outputs, answer, verbose):
        return {"passed": True, "issues": []}

    fake_client = _FakeClient(retry_text="SHOULD NEVER BE CALLED")
    monkeypatch.setattr(dag_executor, "critique_synthesis", fake_critique_synthesis)
    monkeypatch.setattr(dag_executor, "client", fake_client)

    final_answer, synth_trace = asyncio.run(dag_executor._resolve_synthesis(
        question="What drove today's volume and price action?",
        agent_outputs="[FILINGS ANALYSIS (filings_1)]\n" + synthesized_answer,
        synthesis_prompt="irrelevant for this test",
        synthesized_answer=synthesized_answer,
        node_tool_calls=node_tool_calls,
        session_id=None,
        verbose=False,
    ))

    assert fake_client.calls == 0, "no retry should fire — the gate must pass on the first attempt"
    assert final_answer == synthesized_answer
    assert original_sentence.strip() in final_answer
    assert synth_trace.grounding_gate_passed is True


# ══════════════════════════════════════════════════════════════════════
# Problem 2 — grounding gate extended to apply_reflexion() (single-agent
# DAGs and every node of a multi-agent DAG — apply_reflexion() is called
# identically for both; see reflexion.py's apply_reflexion() docstring for
# why this can't be scoped to "single-agent DAGs only" without threading
# extra context sub_agents.py doesn't have today)
# ══════════════════════════════════════════════════════════════════════

def test_apply_reflexion_catches_fabricated_number_via_grounding_gate(monkeypatch):
    """Single-agent-style query: LLM critic says the answer is fine, but
    it contains a number fabricated with respect to this agent's own
    tool_history. Confirms the deterministic gate now catches this via
    the EXISTING per-agent retry loop (apply_reflexion + retry_fn) — the
    same mechanism single-agent DAGs actually use, per sub_agents.py's
    _run_agent() -> apply_reflexion() call chain."""
    tool_history = [_tool_call("get_prices", "close 184.20")]
    # Padded well past REFLEXION_MIN_WORDS so critique() actually runs —
    # isolates that the catch comes from the grounding gate, not the
    # word-count-forcing heuristic.
    padding = (
        "Sector conditions remained broadly stable through the quarter, "
        "with trading volumes in line with recent averages and no "
        "material change in investor positioning noted across the "
        "coverage period. "
    ) * 10
    fabricated_answer = "AAPL closed at $999.99 today, a figure not in any tool result. " + padding

    def fake_critique(question, tool_history, answer, verbose):
        return {"passed": True, "issues": []}  # isolate: LLM judge says fine

    retry_calls = {"count": 0}

    def fake_retry_fn(guidance):
        retry_calls["count"] += 1
        return "AAPL closed at $184.20 today, corrected."

    monkeypatch.setattr(reflexion, "critique", fake_critique)

    result = reflexion.apply_reflexion(
        question="How did AAPL do today?",
        tool_history=tool_history,
        answer=fabricated_answer,
        retry_fn=fake_retry_fn,
        trace=None,
        verbose=False,
        node_id="market_1",
    )

    assert retry_calls["count"] == 1, "grounding gate failure must trigger exactly one retry"
    assert result == "AAPL closed at $184.20 today, corrected."


def test_apply_reflexion_records_grounding_check_on_trace(monkeypatch):
    """Confirms the per-agent Trace object (the SAME one sub_agents.py
    flushes to S3/telemetry) picks up attribution_failures and
    grounding_gate_passed, matching the synthesis-tail trace's shape."""
    from query.telemetry import Trace

    tool_history = [_tool_call("get_prices", "close 184.20")]
    padding = ("No material change noted across the coverage period. ") * 40
    fabricated_answer = "AAPL closed at $999.99 today. " + padding

    def fake_critique(question, tool_history, answer, verbose):
        return {"passed": True, "issues": []}

    monkeypatch.setattr(reflexion, "critique", fake_critique)

    trace = Trace(session_id="test", agent="MarketAgent", question="q", model="m")

    reflexion.apply_reflexion(
        question="q",
        tool_history=tool_history,
        answer=fabricated_answer,
        retry_fn=lambda guidance: "AAPL closed at $184.20 today, corrected.",
        trace=trace,
        verbose=False,
        node_id="market_1",
    )

    assert trace.reflexion_triggered is True
    assert trace.grounding_gate_passed is True  # retry's answer is clean
    assert trace.attribution_failures == []  # reflects the RETRY's gate, not the initial failure


def test_apply_reflexion_skips_llm_critique_when_grounding_passes_and_short(monkeypatch):
    """Word-count skip gate still applies to critique() specifically — a
    short, cleanly-grounded answer should never invoke the (mocked) LLM
    critic."""
    tool_history = [_tool_call("get_prices", "close 184.20")]
    short_clean_answer = "AAPL closed at $184.20 today."  # well under 200 words

    critique_calls = {"count": 0}

    def fake_critique(question, tool_history, answer, verbose):
        critique_calls["count"] += 1
        return {"passed": False, "issues": ["should never be called"]}

    monkeypatch.setattr(reflexion, "critique", fake_critique)

    result = reflexion.apply_reflexion(
        question="How did AAPL do today?",
        tool_history=tool_history,
        answer=short_clean_answer,
        retry_fn=lambda guidance: "unused",
        trace=None,
        verbose=False,
        node_id="market_1",
    )

    assert critique_calls["count"] == 0
    assert result == short_clean_answer


def test_apply_reflexion_grounding_failure_forces_retry_despite_short_answer(monkeypatch):
    """Documents an intentional behavior change: previously, ANY answer
    under REFLEXION_MIN_WORDS returned immediately with zero checks.
    Now, a short answer with a grounding-gate failure still retries —
    the word-count gate only ever skips the LLM critique() call, never
    the free deterministic check (same reasoning as
    dag_executor.py's _resolve_synthesis())."""
    tool_history = [_tool_call("get_prices", "close 184.20")]
    short_fabricated_answer = "AAPL closed at $999.99 today."  # under 200 words, fabricated

    def fake_critique(question, tool_history, answer, verbose):
        return {"passed": True, "issues": []}

    retry_calls = {"count": 0}

    def fake_retry_fn(guidance):
        retry_calls["count"] += 1
        return "AAPL closed at $184.20 today."

    monkeypatch.setattr(reflexion, "critique", fake_critique)

    result = reflexion.apply_reflexion(
        question="How did AAPL do today?",
        tool_history=tool_history,
        answer=short_fabricated_answer,
        retry_fn=fake_retry_fn,
        trace=None,
        verbose=False,
        node_id="market_1",
    )

    assert retry_calls["count"] == 1
    assert result == "AAPL closed at $184.20 today."


def test_apply_reflexion_returns_unchanged_when_no_tool_history():
    """Regression check: the pre-existing 'nothing to ground-check' early
    return must still work unchanged with the new node_id parameter."""
    result = reflexion.apply_reflexion(
        question="What's 2+2?",
        tool_history=[],
        answer="4",
        retry_fn=lambda guidance: "unused",
        trace=None,
        verbose=False,
        node_id="market_1",
    )
    assert result == "4"


# ══════════════════════════════════════════════════════════════════════
# Dependent-node attribution — upstream_tool_calls / extra_attribution_sources
#
# Reproduces FLOW_REFERENCE.md §5's finding: a dependent node's own
# apply_reflexion() call previously only ever saw its OWN node_tool_calls,
# so a figure correctly cited via the "[from prior step: ...]" bracket
# format (mandated by dag_executor.py's round-loop prompt builder for any
# node with depends_on) would be flagged as unattributed even though the
# node did exactly what it was told to. Fixed by threading the upstream
# node(s)' own tool-call records through as attribution-only extra sources.
# ══════════════════════════════════════════════════════════════════════

def test_check_attribution_alone_still_fails_on_bracket_cited_upstream_figure():
    """Baseline: check_attribution() called directly with ONLY this node's
    own tool results (no extra sources) still fails to ground a bracket-
    cited upstream figure — confirms the underlying check itself is
    unchanged; the fix lives in what gets passed to it, not in
    check_attribution()'s own matching logic."""
    own_tool_calls = {"market_2": [_tool_call("get_prices", "AAPL close 184.20")]}
    answer = (
        "AAPL dipped following broader bank-sector jitters tied to "
        "[from prior step: a $1.8 billion bond portfolio loss] at SVB, "
        "closing at $184.20."
    )
    failures = reflexion.check_attribution(answer, own_tool_calls)
    assert any(f["value"] == 1.8 for f in failures)


def test_grounding_gate_sync_recognizes_upstream_figure_via_extra_sources():
    """The fix itself, at the _grounding_gate_sync() level: passing the
    upstream node's tool calls as extra_attribution_sources grounds the
    same bracket-cited figure that failed above."""
    own_tool_calls = {"market_2": [_tool_call("get_prices", "AAPL close 184.20")]}
    upstream_tool_calls = {
        "filings_1": [_tool_call(
            "get_fed_communications",
            "SVB collapse triggered by a bank run following a $1.8 billion bond portfolio loss",
        )],
    }
    answer = (
        "AAPL dipped following broader bank-sector jitters tied to "
        "[from prior step: a $1.8 billion bond portfolio loss] at SVB, "
        "closing at $184.20."
    )

    gate = reflexion._grounding_gate_sync(
        answer, own_tool_calls, extra_attribution_sources=upstream_tool_calls
    )

    assert gate["attribution_passed"] is True
    assert gate["attribution_failures"] == []


def test_grounding_gate_sync_upstream_sources_never_leak_into_inversion():
    """extra_attribution_sources must never reach check_inversion() — a
    dependent node isn't obligated to re-cite every upstream figure, only
    the ones relevant to its own analysis. An upstream figure the answer
    never mentions must NOT be flagged as 'fetched but unused' by THIS
    node's gate (that DAG-wide measurement already happens correctly at
    dag_executor.py's synthesis tail, which sees every node's tool calls)."""
    own_tool_calls = {"market_2": [_tool_call("get_prices", "AAPL close 184.20")]}
    upstream_tool_calls = {
        "filings_1": [_tool_call("get_fed_communications", "unrelated figure 999.99")],
    }
    answer = "AAPL closed at $184.20."  # never mentions 999.99

    gate = reflexion._grounding_gate_sync(
        answer, own_tool_calls, extra_attribution_sources=upstream_tool_calls
    )

    assert gate["inversion_failures"] == [], (
        "upstream-only figures must never surface as this node's own "
        "inversion failures"
    )


def test_apply_reflexion_grounds_bracket_cited_upstream_figure_no_retry(monkeypatch):
    """End-to-end at the apply_reflexion() level — the actual call path
    sub_agents.py uses. A dependent node's answer correctly cites an
    upstream figure via the bracket format; with upstream_tool_calls
    threaded through, this must pass on the FIRST attempt (no retry),
    where before this fix it would have failed attribution and retried."""
    own_tool_history = [_tool_call("get_prices", "AAPL close 184.20")]
    upstream_tool_calls = {
        "filings_1": [_tool_call(
            "get_fed_communications",
            "SVB collapse triggered by a bank run following a $1.8 billion bond portfolio loss",
        )],
    }
    padding = (
        "Sector conditions remained broadly stable through the quarter, "
        "with no material change in investor positioning noted across "
        "the coverage period. "
    ) * 10
    answer = (
        "AAPL dipped following broader bank-sector jitters tied to "
        "[from prior step: a $1.8 billion bond portfolio loss] at SVB, "
        "closing at $184.20. " + padding
    )

    def fake_critique(question, tool_history, answer, verbose):
        return {"passed": True, "issues": []}

    retry_calls = {"count": 0}

    def fake_retry_fn(guidance):
        retry_calls["count"] += 1
        return "SHOULD NOT BE CALLED"

    monkeypatch.setattr(reflexion, "critique", fake_critique)

    result = reflexion.apply_reflexion(
        question="How did AAPL react to the SVB collapse?",
        tool_history=own_tool_history,
        answer=answer,
        retry_fn=fake_retry_fn,
        trace=None,
        verbose=False,
        node_id="market_2",
        upstream_tool_calls=upstream_tool_calls,
    )

    assert retry_calls["count"] == 0, "no retry should fire — the upstream figure is now grounded"
    assert result == answer


def test_apply_reflexion_without_upstream_data_still_flags_the_same_figure(monkeypatch):
    """Regression proof: the SAME scenario as above, but WITHOUT passing
    upstream_tool_calls, still triggers a retry — confirming the fix
    actually depends on the new parameter being threaded through, not on
    some incidental change to check_attribution() itself."""
    own_tool_history = [_tool_call("get_prices", "AAPL close 184.20")]
    padding = (
        "Sector conditions remained broadly stable through the quarter, "
        "with no material change in investor positioning noted across "
        "the coverage period. "
    ) * 10
    answer = (
        "AAPL dipped following broader bank-sector jitters tied to "
        "[from prior step: a $1.8 billion bond portfolio loss] at SVB, "
        "closing at $184.20. " + padding
    )

    def fake_critique(question, tool_history, answer, verbose):
        return {"passed": True, "issues": []}

    retry_calls = {"count": 0}

    def fake_retry_fn(guidance):
        retry_calls["count"] += 1
        return answer  # even a "successful" retry proves a retry was needed

    monkeypatch.setattr(reflexion, "critique", fake_critique)

    reflexion.apply_reflexion(
        question="How did AAPL react to the SVB collapse?",
        tool_history=own_tool_history,
        answer=answer,
        retry_fn=fake_retry_fn,
        trace=None,
        verbose=False,
        node_id="market_2",
        # upstream_tool_calls intentionally omitted
    )

    assert retry_calls["count"] == 1


def test_apply_reflexion_no_deps_unaffected_by_fix(monkeypatch):
    """Confirms scenario 2's shape (nodes with NO upstream deps) is
    entirely unaffected: passing upstream_tool_calls=None (the default,
    and what dag_executor.py's round loop naturally produces for a node
    with empty depends_on) behaves identically to before this fix."""
    tool_history = [_tool_call("get_prices", "close 184.20")]
    clean_answer = "AAPL closed at $184.20 today."

    critique_calls = {"count": 0}

    def fake_critique(question, tool_history, answer, verbose):
        critique_calls["count"] += 1
        return {"passed": True, "issues": []}

    monkeypatch.setattr(reflexion, "critique", fake_critique)

    result = reflexion.apply_reflexion(
        question="How did AAPL do today?",
        tool_history=tool_history,
        answer=clean_answer,
        retry_fn=lambda guidance: "unused",
        trace=None,
        verbose=False,
        node_id="market_1",
        upstream_tool_calls=None,
    )

    assert result == clean_answer
