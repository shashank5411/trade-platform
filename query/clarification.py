"""
query/clarification.py — Classifies whether a new user message answers
a pending clarifying question, and if so, merges it with the original
question into one complete, replanning-ready question.

Single Haiku call does both jobs (classify + merge) to avoid two round
trips. This is deliberately a narrow, single-purpose module — not a
general intent classifier.
"""

import json
from query.config import get_client
import query.memory as memory

HAIKU_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM = """You are checking whether a user's new message answers
a pending clarifying question that was just asked, or whether the
user ignored it and said something unrelated instead.

You will be given:
1. The ORIGINAL question the user first asked (which was ambiguous
   or incomplete in some way)
2. The CLARIFYING QUESTION the system asked in response
3. The user's NEW MESSAGE (their very next message after that)

If the new message is answering the clarifying question (even
tersely — a single ticker, a date, a short phrase counts as an
answer), merge it with the original question into ONE complete,
unambiguous, standalone question that fully resolves what was
missing. The merged question should read naturally, as if the user
had asked the complete thing in one message from the start.

If the new message does NOT look like an answer to the clarifying
question — e.g. it's a completely unrelated new question, a topic
change, or a request that has nothing to do with what was asked —
then this is NOT an answer.

Respond ONLY with valid JSON, no markdown, no preamble:
{"answers_pending": true, "merged_question": "..."}
OR
{"answers_pending": false, "merged_question": null}
"""


def resolve_pending_clarification(session_id: str, new_message: str) -> str:
    """
    Single entry point for handling a possibly-pending clarification
    at the start of a turn. Checks memory for pending state; if
    present, classifies whether new_message answers it and returns
    the merged question if so, otherwise the original new_message
    unchanged. Clears any pending state UNCONDITIONALLY either way —
    this is what gives the "only the immediately next turn" expiry
    rule, satisfied here in one place rather than at each caller.

    If no clarification is pending, returns new_message completely
    unchanged with no side effects (safe no-op for the common case).

    Called by BOTH server.py's /ask handler (real user turns) and
    run_eval.py's multi-turn runner (eval setup turns + the final
    scored turn) — this is the shared function that replaces two
    independent copies of the same check-merge-clear sequence.
    """
    if not session_id:
        return new_message
    pending = memory.get_pending_clarification(session_id)
    if not pending:
        return new_message
    merged = classify_and_merge(
        pending["original_question"],
        pending["question_asked"],
        new_message,
    )
    memory.clear_pending_clarification(session_id)
    return merged if merged else new_message


def classify_and_merge(
    original_question: str,
    question_asked:    str,
    new_message:       str,
) -> str | None:
    """
    Returns the merged question string if new_message answers the
    pending clarification, or None if it should be treated as an
    unrelated fresh question (pending state should be dropped
    either way — that's the caller's job, not this function's).
    """
    content = (
        f"ORIGINAL question: {original_question}\n\n"
        f"CLARIFYING QUESTION asked: {question_asked}\n\n"
        f"User's NEW MESSAGE: {new_message}"
    )
    try:
        client   = get_client()
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=300,
            system=_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        text = response.content[0].text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        if result.get("answers_pending") and result.get("merged_question"):
            return result["merged_question"]
        return None
    except Exception as e:
        print(f"[Clarification] classify_and_merge failed: {e}")
        # Fail safe — treat as NOT an answer, so the user's message
        # is never silently swallowed/misrouted if this call errors.
        return None
