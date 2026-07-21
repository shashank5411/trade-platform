# Grounding Checks Implementation — Attribution + Inversion

Summary of the deterministic attribution/inversion grounding gate, how it
feeds the reflexion retry loop, and two follow-up fixes made after manually
running real queries against it. Companion to `ORCHESTRATION_ARCHITECTURE.md`'s
§5/§6 (guardrails inventory / reflexion mechanism), which describes the
*original prior* state this change replaces.

**Revision history in this file:**
1. Initial gate: `check_attribution()`/`check_inversion()` added to
   `query/dag_executor.py`, wired into the synthesis-tail retry loop only.
2. Two follow-ups found via manual runs (see the `[GroundingGate:...]`
   debug output added alongside the initial gate):
   - **Fixed a real false-positive class**: legitimate unit-scaled
     restatements (e.g. "261.8 million" for a raw `261775500.0`) and
     single-pair derived percentages/deltas (e.g. "6.2% dip" computed from
     two grounded prices) were being flagged as unattributed, and a
     reflexion retry had actually *deleted* the affected sentence rather
     than fix it — a real instance of the check causing harm, not just
     noise.
   - **Extended the gate to `apply_reflexion()`** (per-agent retry loop in
     `reflexion.py`, used by every `sub_agents.py` agent run), so
     single-agent DAGs — which never reach the synthesis tail — now get
     the same deterministic coverage. This required **moving the check
     functions from `dag_executor.py` to `reflexion.py`** to avoid a
     circular import; see "Where the code lives now" below.
3. **This revision** — a follow-up found while writing `FLOW_REFERENCE.md`
   (tracing scenarios 3/4 by hand surfaced it, not a live run): fixed
   **dependent-node attribution** — a node with `depends_on` couldn't see
   the upstream figures it was explicitly required to bracket-cite, so a
   correctly-formatted citation could still fail its own node's attribution
   check. See "Dependent-node attribution" below.

## What changed and why (context)

The orientation pass (before any code was touched) found that
`_check_unattributed_figures()` — the only existing "attribution-shaped"
check — was not actually a value-matching check: it flagged a dependent
node's own answer if it contained a numeric-looking span with no literal
`"[from prior step:"` bracket anywhere in the text, and never touched real
tool-result data. It was also advisory-only (`Trace.record_attribution_warnings()`
+ a console print), wired to nothing that fed the reflexion retry loop.
The thing that actually drove synthesis-retry decisions was
`critique_synthesis()` — a pure LLM judge with no deterministic numeric
verification anywhere in the path.

This implementation replaces that heuristic with real value-matching, and
merges it with the existing LLM judge into one gate, per the agreed
design:
- **Attribution is blocking** and merges with `critique_synthesis()`'s
  verdict (`passed = critique_passed AND attribution_passed`).
- **Inversion is naive-scope and non-blocking today** — computed and
  logged in full detail every time, but does not currently flip the gate.
  Promoting it to blocking is a one-line flag flip (see below), not a
  redesign.
- The old `_check_unattributed_figures()` is removed entirely — it wasn't
  equivalent to what was being built, and the new `check_attribution()`
  strictly supersedes its purpose (verifying that cited figures are
  grounded) with an actual value check instead of a citation-syntax proxy.

## Where the code lives now

The check functions moved from `query/dag_executor.py` to `query/reflexion.py`
in this revision. Why: `apply_reflexion()` (the per-agent retry loop, in
`reflexion.py`, called by every `sub_agents.py` agent run) now needs to
call `check_attribution()`/`check_inversion()` directly. `dag_executor.py`
already imports `critique_synthesis`/`CAVEAT`/etc. **from** `reflexion.py`
— if the check functions had stayed in `dag_executor.py`, giving
`reflexion.py` a reverse import would create a circular import
(`dag_executor -> reflexion -> dag_executor`), which Python's
`from X import Y` style doesn't tolerate. Moving the functions to
`reflexion.py` (which has no dependency on `dag_executor.py`) resolves
this cleanly: `apply_reflexion()` calls them as same-module functions, and
`dag_executor.py` now imports them from `reflexion.py` instead of defining
them — `from query.reflexion import (..., check_attribution,
check_inversion, _run_grounding_gate, _build_retry_guidance)`. This is a
pure relocation; nothing about the functions' external behavior changed
because of the move itself (only because of the Problem 1 fix, covered
below). Existing tests that reference `dag_executor.check_attribution`
etc. from before the move continue to pass unchanged — Python resolves
bare-name lookups in a function body against its *enclosing module's*
namespace at call time, and `dag_executor.py` still has these names bound
there via the import, regardless of which module originally defined them.

`_resolve_synthesis()` and `_debug_print_grounding_gate()` (the temporary
manual-eyeball print from the debug pass) stay in `dag_executor.py` — they're
synthesis-tail-specific orchestration, not grounding-check logic itself.

## Function signatures added/changed

```python
# In query/reflexion.py:

# Shared helpers
def _parse_figure(raw: str) -> float | None
def _extract_figures(text: str) -> list[float]
def _values_match(a: float, b: float, rel_tol: float = ATTRIBUTION_TOLERANCE) -> bool
def _is_unit_scaled_match(x: float, y: float, rel_tol: float = ATTRIBUTION_TOLERANCE) -> bool   # NEW — Problem 1
def _value_grounded_in(value: float, candidates: list) -> bool                                  # NEW — Problem 1
def _derived_pair_values(values: list) -> list[float]                                           # NEW — Problem 1
def _flatten_tool_values(node_tool_calls: dict) -> list[float]
def _flatten_derived_values(node_tool_calls: dict) -> list[float]                               # NEW — Problem 1

# The two checks — pure functions, (answer_text, node_tool_calls) -> list[failure]
def check_attribution(answer_text: str, node_tool_calls: dict) -> list[dict]   # extended, Problem 1
def check_inversion(answer_text: str, node_tool_calls: dict) -> list[dict]     # extended, Problem 1

# Merge + retry-prompt construction
def _merge_gate_results(attribution_failures: list, inversion_failures: list) -> dict
def _grounding_gate_sync(                                            # extended, this revision
    answer_text: str, node_tool_calls: dict,
    extra_attribution_sources: dict = None,                          # NEW — dependent-node fix
) -> dict
async def _run_grounding_gate(answer_text: str, node_tool_calls: dict) -> dict   # unchanged this revision
def _build_retry_guidance(critique_result: dict, grounding_gate: dict) -> str

# apply_reflexion() — extended this revision (new upstream_tool_calls param,
# threaded through to _grounding_gate_sync as extra_attribution_sources)
def apply_reflexion(
    question: str, tool_history: list, answer: str, retry_fn,
    trace=None, verbose: bool = True, node_id: str = None,
    upstream_tool_calls: dict = None,                                # NEW — dependent-node fix
) -> str

# In query/dag_executor.py:
async def _resolve_synthesis(          # unchanged this revision
    question: str, agent_outputs: str, synthesis_prompt: str,
    synthesized_answer: str, node_tool_calls: dict,
    session_id: str, verbose: bool,
) -> tuple[str, Trace]

async def _run_agent_async(            # NEW param this revision
    node_id: str, agent_type: str, question: str, history: list,
    verbose: bool, session_id: str = None,
    upstream_tool_calls: dict = None,                                # NEW — dependent-node fix
) -> tuple
```

`upstream_tool_calls`/`extra_attribution_sources` is also threaded through
all four specialist classes' `.run()` methods in `sub_agents.py`
(`MarketAgent`, `MacroAgent`, `FilingsAgent`, `SentimentAgent`) and
`_run_agent()` itself — same parameter name and default (`None`)
throughout the chain, no reshaping at any hop.

`node_tool_calls: dict[node_id -> list[record]]` is the same structure
`execute()` already builds during its DAG round loop — no new data
structure was introduced; the checks just read the `result_full` field
that was already there. `apply_reflexion()` wraps its single-agent
`tool_history: list` as `{(node_id or "agent"): tool_history}` to match
this same shape (see "Single-agent gate coverage" below).

**Removed**: `_check_unattributed_figures(node_outputs, dag)` and its call
site (the pre-synthesis per-node loop). `Trace.record_attribution_warnings()`
and the `attribution_warnings` field are replaced by
`Trace.record_grounding_check(gate_result)` and three new fields
(`attribution_failures`, `inversion_failures`, `grounding_gate_passed`) —
see `query/telemetry.py`.

## Failure object shape

Both checks return lists of small dicts, not bools/log lines, since the
retry prompt needs to explain *what* failed:

```python
# check_attribution failure
{"kind": "attribution", "value": 12.4, "raw_text": "12.4%",
 "reason": "'12.4%' does not match any fetched tool-result value within 1% tolerance"}

# check_inversion failure — has source provenance, since it's flagging a
# fetched value, not an unsupported claim
{"kind": "inversion", "value": 150.0, "raw_text": "150.00",
 "node_id": "market_1", "tool": "get_prices",
 "reason": "fetched by get_prices in node 'market_1' but never referenced in the final answer"}
```

## Tolerance logic

No existing tolerance/number-matching code was found anywhere in the repo
(confirmed via repo-wide grep for `tolerance`/`isclose`). `_values_match()`
uses stdlib `math.isclose(a, b, rel_tol=0.01, abs_tol=1e-9)` — a plain 1%
relative tolerance, with a small absolute tolerance floor so near-zero
values (e.g. a 0.0% change) don't spuriously fail relative comparison.
Both `check_attribution()` and `check_inversion()` call this same function
via `_flatten_tool_values()`/`_extract_figures()` — no duplicated matching
logic between them, per the requirement.

Figure extraction reuses the pre-existing `_FIGURE_PATTERN` regex that was
already in `dag_executor.py` (matching `$1,234.50`, `3.2%`, or bare
decimals like `184.20` — deliberately excludes bare integers like
transaction counts or years, matching this regex's pre-existing behavior
elsewhere in the file and in `reflexion.py`'s near-identical
`_MULTI_FIGURE_PATTERN`).

## The open design question (resolved)

**Inversion scope** — no signal exists in the tool-call record shape to
distinguish "values relevant to this specific question" from "everything
an agent happened to fetch" (e.g. a wider date range than asked for).
Per the agreed direction: start naive (check every fetched, non-dedup
value), keep it **non-blocking**, and log full detail so a real
false-positive rate can be measured against actual traffic before
deciding on real scoping. A "last tool call per node" heuristic was
considered and rejected — it would silently under-check in the (likely
common) case where an early call fetched the value that mattered and a
later call fetched something incidental, with no signal that it's wrong.

**Wiring**: `INVERSION_BLOCKING = False` (module constant,
`dag_executor.py`) gates both (a) whether `_run_grounding_gate()`'s
`passed` folds in `inversion_passed`, and (b) whether
`_build_retry_guidance()` includes an inversion section (that section is
already written, just commented out, immediately below the attribution
section it mirrors). Promoting inversion to blocking is flipping this one
constant — no other code changes required.

**Recommendation**: run for 1–2 weeks against real traffic (this project
already has telemetry — `attribution_failures`/`inversion_failures` are
now queryable per-trace via the existing S3/Athena telemetry pipeline, and
the eval harness in `query/evaluations/` is the natural place to add a
grounding-category eval question set) before flipping `INVERSION_BLOCKING`.
If false positives are dominated by "wider fetch than needed" cases,
scoping by DAG-node question intent (not last-call-only) would be the
next design pass — but that needs real failure data to design well, not a
guess now.

**Not touched this revision** — this stays exactly as designed originally:
`check_inversion()`'s naive scope (checks every fetched value, no
relevance filtering) and its non-blocking status (`INVERSION_BLOCKING`
still `False`). The one change that *does* touch inversion is purely a
matching-accuracy improvement shared with attribution (see below) — a
fetched value correctly restated in scaled form is now recognized as
"used" instead of being flagged as dropped, which is a bug fix to
inversion's matching logic, not a scope or blocking-status change.

## Unit-scaling and single-pair delta/%-change matching (false-positive fix)

**The bug, concretely**: a real synthesized answer stated "Trading was
heaviest on June 26 with 261.8 million shares, following a sharp 6.2%
dip." The raw fetched share volume was `261775500.0` (261.8 million is a
correct unit-scaled restatement) and 6.2% was a correctly-computed
percentage change between two grounded prices. `check_attribution()`
flagged both, since `_values_match()` only did a direct
`math.isclose()` comparison against raw fetched values — it had no notion
of unit scaling or arithmetic derivation. The reflexion retry this
triggered didn't fix the grounding — it deleted the sentence entirely,
trading away real analysis to satisfy an overly literal check. This is
confirmed end-to-end by
`test_resolve_synthesis_preserves_legitimate_content_no_retry` in the test
suite: reproducing the exact input now shows the gate passing on the
first attempt, so **no retry fires at all** — the strongest form of
"before/after" confirmation available, since there's no longer a retry to
diff against.

**Two matching extensions, both bounded exactly as scoped (no ratios,
no multi-step calculations):**

1. **Unit-scaled matching** (`_is_unit_scaled_match()`) — an answer value
   matches a source value divided by 1,000 / 1,000,000 / 1,000,000,000,
   within the existing 1% relative tolerance. Guarded so this only applies
   when the LARGER of the two magnitudes exceeds 1000 — otherwise a small,
   coincidentally-close answer number could spuriously match an unrelated
   scaled-down giant value from a different field (e.g. an answer's "3.2"
   matching some unrelated huge value's `/1e9` scaling by chance).
   Implemented as a single symmetric function (checked both directions:
   "does this answer figure equal some source value scaled down" for
   `check_attribution()`, and "does some answer figure equal this source
   value scaled down" for `check_inversion()`) rather than two directional
   near-duplicates.

2. **Single-pair delta/percentage-change matching** (`_derived_pair_values()`)
   — for every INDIVIDUAL tool call's own extracted figure list (in
   extraction order), computes deltas and percentage-changes for every
   ADJACENT pair plus the (first, last) pair — not all `O(n²)` pairs. This
   is deliberately bounded: a long price table (e.g. 52 weekly points)
   would make an all-pairs approach both expensive and too permissive
   (almost any answer number would coincidentally match *some* pairwise
   delta in a large table, defeating the point of the check).
   Adjacent-pair + first/last covers the two realistic derivation shapes
   observed — a change between consecutive observations, and a change
   over the full period (which mirrors `api.py`'s own `get_prices` summary
   stats, already reporting start/end/%-change as the standard "period
   change" framing).

   **"Same series" grouping** — the task asked me to scope this to pairs
   from the same tool/field series and to note what grouping I actually
   used if the record shape didn't support anything finer. It doesn't:
   `result_full` is plain `pandas`-formatted text with no per-field
   structure once flattened to bare numbers by regex — there's no way to
   tell "close price" from "volume" from "high" apart after extraction.
   The grouping actually used is **one individual tool call's own value
   list** (`_flatten_derived_values()` iterates `node_tool_calls`, and for
   each call independently extracts + pairs its own figures) — the
   tightest boundary the data actually supports. Two separate calls to the
   same tool (e.g. `get_prices` for two different tickers in one node) are
   correctly kept in separate groups.

   **A correctness bug caught before shipping, not after**: the first
   draft of `_derived_pair_values()` only stored the signed result of each
   subtraction/percentage (whichever direction the raw arithmetic happened
   to produce). But `_FIGURE_PATTERN` never captures a leading minus sign
   — so every value `check_attribution()` ever extracts *from the answer*
   is already non-negative by construction (a "6.2% dip" extracts as
   `+6.2`, never `-6.2`). A derived value list that only stored the
   negative-signed result of `(b-a)/a*100` for a decreasing pair would
   never match that legitimate positive-magnitude answer figure. Fixed by
   storing both signs of every delta and every percentage-change
   denominator — verified directly: `_derived_pair_values([161.20,
   151.17])` now yields `{-10.03, -6.635, -6.222, 6.222, 6.635, 10.03}`,
   and `+6.222` correctly matches an extracted "6.2%" within 1% tolerance.

**Scope explicitly not extended beyond what was asked**: no ratio matching
(e.g. P/E-style divisions), no multi-step calculations, no matching across
values from different tool calls. No other clearly-legitimate-but-flagged
pattern was observed during this pass beyond the two described — if one
turns up later, extend `check_attribution()`'s match chain
(`_value_grounded_in()` → derived-pair check) rather than growing
`_derived_pair_values()`'s pair-selection bound past adjacent+first/last,
which is where the real risk of the check becoming too permissive lives.

**`check_inversion()` also benefits from `_is_unit_scaled_match()`** (via
the same shared `_value_grounded_in()` predicate) but deliberately does
**not** get the derived-pair matching — that defense specifically excuses
arithmetic the *model* performed on grounded inputs; it has no bearing on
whether a raw fetched value was itself ever cited, which is what inversion
measures. This is a shared-helper consistency call, not a scope change to
inversion (see the "not touched this revision" note above) — the original
design already established "extract shared tolerance/number-matching logic
into a helper both checks call" as a hard requirement, and leaving
inversion on the old narrower matcher while attribution gained a better
one would have made inversion's already-noisy naive scope actively worse
by introducing NEW false positives (a legitimately-scaled value showing up
as "never referenced").

## Single-agent gate coverage (extending apply_reflexion())

**The gap**: `check_attribution()`/`check_inversion()`/`_run_grounding_gate()`
previously only ran in `dag_executor.py`'s synthesis tail, which
single-agent DAGs (`len(dag) == 1`) skip entirely — confirmed by re-reading
`execute()`: the single-node path returns after only the injection check,
never reaching `critique_synthesis()` or the grounding gate. Single-agent
queries — per project notes, likely the majority of traffic — got zero
deterministic grounding coverage, only whatever `apply_reflexion()`
already did per-agent in `sub_agents.py` (LLM-judge-based via `critique()`,
no tool-value diffing).

**Integration approach chosen: fed into the EXISTING per-agent retry loop
(`apply_reflexion()` + `retry_fn`), not a second parallel mechanism.**
`apply_reflexion()` already had its own retry-with-feedback loop —
`critique()` → one retry via the caller-supplied `retry_fn` (which itself
re-runs the agent's ReAct loop, allowing further tool calls) → re-`critique()`
→ `CAVEAT` on repeat failure. This is structurally identical to
`_resolve_synthesis()`'s pattern, just LLM-judge-driven instead of
value-diff-driven. Per the task's own steer ("prefer feeding the grounding
gate's failures into that existing loop over building a second parallel
retry mechanism"), `apply_reflexion()` now merges `critique()`'s verdict
with the (new, sync) `_grounding_gate_sync()` result exactly the way
`_resolve_synthesis()` merges `critique_synthesis()` with the async gate:
`overall_passed = critique_passed AND grounding_gate["passed"]`, one retry,
`CAVEAT` on exhaustion, guidance built via the same shared
`_build_retry_guidance()` (its shape-agnostic — `critique()` and
`critique_synthesis()` both return the identical `{passed, issues,
retry_guidance}` shape, so no special-casing was needed).

**A real design fork worth flagging explicitly, as asked**: `apply_reflexion()`
is called identically for *every* agent run — `sub_agents.py`'s `_run_agent()`
has no visibility into whether it's the sole node in a single-agent DAG or
one node of a multi-agent DAG (that's `dag_executor.py`'s context, not
`sub_agents.py`'s or `reflexion.py`'s). Reusing the existing loop
therefore **necessarily extends deterministic grounding coverage to every
DAG node, not literally only single-agent DAGs** — a multi-agent DAG's
individual nodes now also get real value-grounding on their own per-node
answers, on top of what the final synthesized answer already gets at the
tail. There is no way to scope this to "single-agent DAGs only" without
threading extra context down from `dag_executor.py` into
`sub_agents.py`/`reflexion.py` that doesn't exist today, which would mean
NOT reusing the existing loop — directly contradicting the "prefer reuse"
instruction. I judged this the correct trade: the added coverage is a
strict improvement (checks are free — no LLM cost — and Problem 1's fix
means the false-positive risk that would have made broader coverage scary
is now addressed), not a regression, but it is a real behavior change
beyond the literal "single-agent DAGs" framing and should be confirmed as
acceptable, per the request to report back on this before considering it
done.

**`node_id` threading**: `apply_reflexion()` gained a `node_id: str = None`
parameter, wired from `sub_agents.py`'s existing `_run_agent(..., node_id=node_id)`
call site (the value was already available there — `_run_agent()` already
threads it into `Trace(..., node_id=node_id)`). Used only as the dict key
`node_tool_calls = {(node_id or "agent"): tool_history}` expects — the key
itself carries no semantic meaning here since there's only ever one node's
`tool_history` in scope per call.

**Intentional behavior change to the word-count skip gate**: previously,
ANY answer under `REFLEXION_MIN_WORDS` (200) returned immediately with
*zero* checks — no `critique()` call, nothing. Now, the deterministic gate
always computes first (cheap, no LLM call — same reasoning as
`_resolve_synthesis()`), and a short answer with a *grounding* failure
still triggers a retry even though `critique()` itself is skipped under
the word-count gate. Confirmed by
`test_apply_reflexion_grounding_failure_forces_retry_despite_short_answer`.
This mirrors `_resolve_synthesis()`'s exact behavior and closes a real gap
— a short, blunt fabrication is exactly the shape a length-only gate would
otherwise miss (the same class of argument that already justified making
the injection-provenance check word-count-independent).

## Dependent-node attribution (fixes FLOW_REFERENCE.md §5)

**The bug**: `dag_executor.py`'s round loop requires any node with
`depends_on` to wrap a borrowed figure as `[from prior step: $58.20/barrel]`
(the "MANDATORY CITATION FORMAT" text, `dag_executor.py:442-464`). But
that node's own `apply_reflexion()` call built `node_tool_calls =
{(node_id or "agent"): tool_history}` — **only its own fetched tool
results** — so `check_attribution()` would extract the bracketed figure's
number, fail to find it in the node's own data (it was never supposed to
be there; it came from upstream), and flag a correctly-formatted citation
as unattributed. This wasn't caught by the original test suite because
none of those tests construct the "cites a bracket-wrapped figure from a
DIFFERENT node's tool_history" shape — that shape only exists inside
`dag_executor.execute()`'s round loop, which unit tests for the check
functions in isolation never build. It surfaced while hand-tracing
scenarios 3 and 4 for `FLOW_REFERENCE.md`.

**The fix — thread the structured upstream data through, not re-parsed
prompt text.** At the point `dag_executor.py`'s round loop builds
`dep_answers` (the formatted STRING blocks folded into the enriched
prompt), the underlying structured data is still directly accessible:
`node_tool_calls[dep]` (the same accumulator dict the round loop already
populates after every round) holds each completed dependency's own
tool-call records, keyed the same way `dep_answers`' `dep in completed`
check already relies on. So a second, parallel dict is built there:
```python
upstream_tool_calls = {
    dep: node_tool_calls[dep]
    for dep in node.get("depends_on", [])
    if dep in node_tool_calls
}
```
This was preferred over re-extracting figures from the already-formatted
`dep_answers` prompt STRING (which would have meant a second layer of
regex-parsing over text that itself came from regex-parsed tool output —
compounding precision loss for no reason when the original structured
data was one dict lookup away).

**Threaded through unchanged** — no re-shaping, no re-keying — across four
call boundaries:
```
dag_executor.py: execute()'s round loop
  -> _run_agent_async(..., upstream_tool_calls=upstream_tool_calls)   [dag_executor.py]
  -> agent.run(..., upstream_tool_calls=upstream_tool_calls)          [sub_agents.py, all 4 specialist classes]
  -> _run_agent(..., upstream_tool_calls=upstream_tool_calls)         [sub_agents.py]
  -> apply_reflexion(..., upstream_tool_calls=upstream_tool_calls)    [reflexion.py]
  -> _grounding_gate_sync(..., extra_attribution_sources=upstream_tool_calls)  [reflexion.py]
```
All new parameters default to `None`, so every call site that doesn't
have upstream data (the single-agent path in `dag_executor.py:execute()`,
and any node with empty `depends_on`) is completely unaffected — it's the
exact same `None` that flows through today.

**The attribution-only scoping decision (per the task's explicit ask)**:
`_grounding_gate_sync()` gained a third parameter,
`extra_attribution_sources: dict = None`:
```python
def _grounding_gate_sync(answer_text, node_tool_calls, extra_attribution_sources=None):
    attribution_sources = dict(node_tool_calls)
    if extra_attribution_sources:
        attribution_sources.update(extra_attribution_sources)
    attribution_failures = check_attribution(answer_text, attribution_sources)
    inversion_failures = check_inversion(answer_text, node_tool_calls)   # UNCHANGED — own data only
    return _merge_gate_results(attribution_failures, inversion_failures)
```
`check_attribution()` gets the UNION (own + upstream) as its `node_tool_calls`
argument — it doesn't care whether that dict represents "this node's own
data" or "this node's own data plus upstream," it just flattens everything
into candidate source values either way. `check_inversion()` keeps getting
ONLY the node's own `node_tool_calls`, completely unchanged. This was a
real risk called out explicitly in the task and confirmed by reasoning
through it: without this split, merging upstream data into BOTH checks
would have made every upstream figure a dependent node didn't happen to
need show up as a NEW "fetched but unused" inversion failure attributed to
that node — which is wrong, since a dependent node is only obligated to
cite the upstream figures relevant to its own analysis, not restate
everything upstream fetched. `test_grounding_gate_sync_upstream_sources_never_leak_into_inversion`
confirms this split holds.

**Only `_grounding_gate_sync()` (sync, `apply_reflexion()`'s entry point)
was extended — not `_run_grounding_gate()`** (async, `_resolve_synthesis()`'s
entry point). No change was needed there: synthesis's `node_tool_calls`
already spans the FULL DAG (every node's tool calls, not just one), so
synthesis-level attribution checking already had access to upstream data
before this fix — the gap was specific to the PER-NODE check, which is the
only one that ever saw a narrowed, single-node view.

## Judgment calls made

1. **Where the check functions live** — moved from `dag_executor.py` to
   `reflexion.py` to resolve the circular import Problem 2's "reuse the
   existing loop" requirement created. See "Where the code lives now"
   above for the full reasoning; this wasn't optional once "prefer the
   existing per-agent retry loop" was the direction, since that loop lives
   in a module `dag_executor.py` already depends on, not the reverse.

2. **The deterministic gate always runs, regardless of the word-count
   skip gate.** `SYNTH_REFLEXION_MIN_WORDS` (200 words) only ever governed
   whether the *LLM* `critique_synthesis()` call fires — that gate exists
   specifically to save LLM call cost (see `reflexion.py`'s CO-1
   reasoning), which doesn't apply to a free regex/float check. A short
   answer with a fabricated number is now still caught even when the LLM
   critique itself is skipped — confirmed by
   `test_resolve_synthesis_skips_llm_critique_for_short_clean_answer`.

3. **`_resolve_synthesis()` extraction.** The critique→retry→caveat block
   was pulled out of `execute()`'s body into its own `async` function.
   This wasn't asked for explicitly, but it's what made "reflexion retry
   loop still respects its existing cap and caveat behavior" actually
   testable without standing up a full DAG round (agent registry, mocked
   `agent.run()` calls, etc.) — the extracted function takes the
   already-synthesized answer and tool-call data as plain arguments and
   can be driven directly with a mocked `client` and mocked
   `critique_synthesis`.

4. **Concurrency**: `_run_grounding_gate()` runs both checks via
   `asyncio.gather` + `loop.run_in_executor`, matching this file's
   existing `_run_agent_async()` offloading idiom, since `execute()` is
   already async. Both checks are cheap pure functions (regex + float
   comparisons, no I/O) — the real-world latency benefit of doing this
   concurrently is close to zero, but it was implemented this way both
   because it was explicitly requested and because it matches the file's
   existing convention rather than introducing a second, inconsistent
   pattern for "fast CPU work in an async function."

5. **Retry guidance keeps judge feedback and numeric failures in labeled,
   separate sections** rather than one flattened list (per direction) —
   `_build_retry_guidance()` produces `"JUDGE FEEDBACK (...)"` and
   `"NUMERIC VERIFICATION FAILURES (attribution)"` as distinct blocks
   joined by a blank line, so the model sees two clearly different kinds
   of feedback and a future debugging pass can tell which check is
   driving retries.

6. **Telemetry**: reused the existing `Trace`/S3 pattern rather than
   inventing a new persistence mechanism. `record_grounding_check()`
   overwrites on each call (initial attempt, then again after a retry) —
   only the *final* determining attempt's result is queryable, mirroring
   `record_synthesis_reflexion()`'s existing final-state-only convention
   rather than keeping a full history.

7. **Both signs stored for every derived delta/percentage.** Caught in
   testing, not asked for explicitly: since answer-side figure extraction
   never captures a minus sign (`_FIGURE_PATTERN` has no `-` in it), a
   derived-value list that only stored the raw signed result of each
   pairwise subtraction would silently never match half of all legitimate
   derived percentages/deltas (whichever direction happened to come out
   negative). Fixed by storing both `+delta`/`-delta` and both signs of
   each percentage-change denominator — see the "Unit-scaling..." section
   above for the concrete before/after values.

8. **`_derived_pair_values()` grouped per individual tool call, not per
   node or per tool name.** The task suggested "same node_id + tool +
   similar field name" as a fallback grouping if nothing finer was
   available. Nothing finer *is* available (see above), but grouping by
   node_id+tool would still have been wrong when the same tool is called
   twice in one node for two different things (e.g. `get_prices` for two
   different tickers) — deltas would then mix figures from genuinely
   unrelated series. Grouping by individual call avoids that without
   losing anything the coarser grouping would have caught.

## Confirmation: reflexion's external behavior is unchanged

- **Retry cap**: still exactly one retry attempt on failure — confirmed by
  `test_resolve_synthesis_retries_once_then_caveats_if_still_failing`
  (asserts the fake client's `messages.create()` was called exactly once).
- **Caveat on exhaustion**: `CAVEAT` (imported verbatim from
  `reflexion.py`, untouched) is still appended when the retry still fails
  — confirmed by the same test.
- **Successful retry drops the caveat**: confirmed by
  `test_resolve_synthesis_retry_can_pass_and_drop_caveat`.
- **Word-count LLM-cost skip gate**: still applies to `critique_synthesis()`
  exactly as before — confirmed by
  `test_resolve_synthesis_skips_llm_critique_for_short_clean_answer`
  (mocked critic asserts zero calls).
- **Guardrail checks untouched**: `check_injection_provenance()`,
  `check_advice_boundary()`, `check_scope_boundary()` and the planner's
  scope/clarify sentinels were not touched by this change — only the
  grounding gate (attribution/inversion + the LLM critics feeding it) was
  in scope, per the task boundary.

**Same confirmation, extended to the new `apply_reflexion()` per-agent path
this revision added:**
- **Retry cap**: still exactly one retry attempt — confirmed by
  `test_apply_reflexion_catches_fabricated_number_via_grounding_gate`
  (asserts the fake `retry_fn` was called exactly once).
- **Caveat on exhaustion / successful-retry-drops-caveat**: not
  independently re-tested for `apply_reflexion()` this revision (the
  underlying `CAVEAT`-append branch is untouched code, already covered by
  the original per-agent reflexion behavior before the grounding gate
  existed) — the new tests focus on confirming the grounding gate's
  failures correctly *trigger* the existing retry, which is the actual
  new behavior.
- **Word-count LLM-cost skip gate**: confirmed still skips `critique()`
  specifically for a short, cleanly-grounded answer via
  `test_apply_reflexion_skips_llm_critique_when_grounding_passes_and_short`
  (zero critic calls) — and confirmed it no longer skips the *grounding*
  check itself via
  `test_apply_reflexion_grounding_failure_forces_retry_despite_short_answer`
  (an intentional behavior change, not a regression — see "Single-agent
  gate coverage" above).
- **No-tool-history early return**: confirmed unchanged with the new
  `node_id` parameter via
  `test_apply_reflexion_returns_unchanged_when_no_tool_history`.

**This revision's dependent-node fix — confirmed non-regressive across the
board:**
- `check_inversion()`'s own scope/blocking status: untouched — no edits
  made to `check_inversion()` itself, and
  `test_grounding_gate_sync_upstream_sources_never_leak_into_inversion`
  confirms upstream data never reaches it via the new parameter.
- `INVERSION_BLOCKING`: still `False`, not touched.
- Guardrail checks (`check_injection_provenance()`,
  `check_advice_boundary()`, `check_scope_boundary()`, planner
  scope/clarify sentinels): not touched.
- Nodes with no `depends_on` (scenario 2's shape): confirmed unaffected —
  `test_apply_reflexion_no_deps_unaffected_by_fix` passes
  `upstream_tool_calls=None` explicitly and shows identical behavior to
  before this revision.
- The fix's dependency on the new parameter (not some incidental change
  to `check_attribution()`) is proven directly by
  `test_apply_reflexion_without_upstream_data_still_flags_the_same_figure`
  — same exact scenario as the fix's own passing test, but with
  `upstream_tool_calls` omitted, still triggers a retry.

## Tests

`tests/unit/test_grounding_checks.py` (31 tests, all passing — 13 from the
original gate, 12 from the unit-scaling/single-agent-coverage revision, 6
from this revision's dependent-node-attribution fix). This remains the
only test file for `query/` logic in the repo; the sole prior test
(`tests/unit/test_trade_platform_stack.py`) is a stubbed CDK infra test
with its own pre-existing, unrelated failure (confirmed via `git stash`
across all three revisions now — fails identically on a clean `master`;
not something any pass touched or broke).
`tests/unit/conftest.py` sets a dummy `ANTHROPIC_API_KEY` before
collection, since `dag_executor.py`/`sub_agents.py`/`reflexion.py`/
`planner.py` all resolve an Anthropic client singleton at import time —
confirmed again this revision: `from query import agent, orchestrator,
planner, dag_executor, sub_agents, reflexion, registry, tools, memory,
api, server` imports cleanly end-to-end, no circular-import errors.

Run with: `python -m pytest tests/unit/test_grounding_checks.py -v`
