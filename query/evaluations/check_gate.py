"""
check_gate.py — CI regression gate for the eval harness.

Reads a results.json produced by run_eval.py and enforces a fixed pass-rate
floor on `stable`-tier questions only. `monitored`-tier questions (known
sampling noise — SVB routing instability, etc.) are reported but never
block the gate.

Usage:
    python query/evaluations/check_gate.py results/run_<timestamp>/results.json
    python query/evaluations/check_gate.py results/run_<timestamp>/results.json --threshold 0.60

Exit code 0 = gate passes (merge allowed).
Exit code 1 = gate fails on pass rate (block merge).
Exit code 2 = gate failed to evaluate at all (block merge) — empty stable
              set, malformed results.json, etc. Treated as a hard error,
              not a pass: an empty stable list almost always means a
              plumbing break upstream (e.g. --tier filtering wrong, --out
              writing the wrong shape, run_eval.py silently producing zero
              records), not a legitimate "nothing to check" state. A
              silently-passing gate is worse than a loudly-failing one,
              since it erodes the trust the gate exists to build.
"""
import argparse
import json
import sys

DEFAULT_THRESHOLD = 0.60


def load_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_by_tier(records: list) -> tuple[list, list]:
    """
    Split question records into (stable, monitored) by their `tier` field.
    Any record missing a `tier` field is treated as `stable` — fail-safe
    default so a question added without the new field doesn't silently
    fall out of the gate.
    """
    stable, monitored = [], []
    for r in records:
        tier = r.get("tier", "stable")
        if tier == "monitored":
            monitored.append(r)
        else:
            stable.append(r)
    return stable, monitored


def pass_rate(records: list) -> float:
    """
    Plain pass rate over a non-empty record list. Callers must check
    emptiness themselves — what an empty list MEANS differs by tier
    (see main()), so there's no single correct vacuous default here
    anymore. Raises on empty input to make that explicit.
    """
    if not records:
        raise ValueError("pass_rate() called with no records — caller must handle the empty case explicitly")
    passed = sum(1 for r in records if r.get("passed") is True)
    return passed / len(records)


def summarize(records: list, label: str) -> str:
    if not records:
        return f"{label}: no questions"
    passed = sum(1 for r in records if r.get("passed") is True)
    total = len(records)
    rate = passed / total
    failed_ids = [r.get("id", "?") for r in records if r.get("passed") is not True]
    lines = [f"{label}: {passed}/{total} ({rate:.0%})"]
    if failed_ids:
        lines.append(f"  failing: {', '.join(failed_ids)}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_path", help="Path to results.json from a run_eval.py run")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum stable-tier pass rate required to pass the gate (default {DEFAULT_THRESHOLD:.0%})",
    )
    args = parser.parse_args()

    try:
        data = load_results(args.results_path)
    except (OSError, json.JSONDecodeError) as e:
        print(f"GATE ERROR — could not read/parse '{args.results_path}': {e}")
        print("Treating as a hard failure, not a pass.")
        sys.exit(2)

    records = data.get("results", data) if isinstance(data, dict) else data
    if isinstance(records, dict):
        # tolerate {"results": [...]} or a bare top-level list
        records = records.get("results", [])

    stable, monitored = split_by_tier(records)

    # MONITORED: empty is fine and expected (e.g. nothing tagged yet, or a
    # --tier monitored run with no monitored questions defined). Vacuous
    # pass is the correct semantics here — there's genuinely nothing to
    # report, and that's not a failure of anything.
    print(summarize(monitored, "MONITORED (informational only)"))

    # STABLE: empty is a hard error, not a pass. This is the one that
    # gates merges, so silently reporting 100% on zero records is exactly
    # the failure mode that let every PR through unnoticed before this
    # was caught. If the stable set is empty, something upstream broke —
    # a --tier filter, an --out write, a run_eval.py regression — and the
    # gate should say so loudly rather than wave the PR through.
    if not stable:
        print("STABLE (gating): 0 questions found.")
        print()
        print("GATE ERROR — stable-tier question set is empty.")
        print("This almost certainly means something upstream is broken")
        print("(question filtering, --out plumbing, or run_eval.py itself),")
        print("not that there is genuinely nothing to check. Blocking merge.")
        sys.exit(2)

    print(summarize(stable, "STABLE (gating)"))
    print()

    rate = pass_rate(stable)
    print(f"Stable-tier pass rate: {rate:.1%} (threshold: {args.threshold:.0%})")

    if rate < args.threshold:
        print("GATE FAILED — blocking merge.")
        sys.exit(1)

    print("GATE PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()