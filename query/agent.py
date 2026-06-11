"""
ReAct agent — entry point with memory and orchestration.
Phase 5: named sessions, DynamoDB memory, sub-agent routing.
"""

import os
import sys
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from query.memory       import save_turn, load_turns, list_sessions, clear_session
from query.orchestrator import run as orchestrate

ENV = os.environ.get("ENV", "dev")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import anthropic
import os; get_anthropic_key = lambda: os.environ["ANTHROPIC_API_KEY"]

client = anthropic.Anthropic(api_key=get_anthropic_key())

def parse_args():
    p = argparse.ArgumentParser(
        description="Trade Platform — financial intelligence agent"
    )
    p.add_argument("--session",       default=None,
                   help="Session name for persistent memory (e.g. 'q3-analysis')")
    p.add_argument("--question",      default=None,
                   help="Single question (non-interactive mode)")
    p.add_argument("--list-sessions", action="store_true",
                   help="List all active sessions")
    p.add_argument("--clear-session", default=None,
                   help="Clear a named session")
    p.add_argument("--no-memory",     action="store_true",
                   help="Disable memory for this run")
    p.add_argument("--verbose",       action="store_true", default=True,
                   help="Show tool calls and routing")
    return p.parse_args()


def make_session_id(name: str) -> str:
    """Normalize session name to a safe DynamoDB key."""
    return name.strip().lower().replace(" ", "-")


def run_question(
    question:   str,
    session_id: str  = None,
    use_memory: bool = True,
    verbose:    bool = True,
) -> str:
    """Run a single question through the orchestrator with memory."""

    # Load history
    history = []
    if use_memory and session_id:
        history = load_turns(session_id)
        if history and verbose:
            print(f"[Memory] loaded {len(history)} prior turns "
            f"from session '{session_id}'")
            
    # Run through orchestrator
    answer = orchestrate(
        question,
        history=history,
        verbose=verbose,
        session_id=session_id,
    )

    # Save to memory
    if use_memory and session_id:
        save_turn(session_id, "user",      question)
        save_turn(session_id, "assistant", answer)

    return answer


def interactive_loop(session_id: str, use_memory: bool, verbose: bool):
    """Interactive REPL loop."""
    mem_status = f"session='{session_id}'" if session_id else "no memory"
    print(f"\nTrade Platform Agent ({mem_status})")
    print("Type 'exit' to quit, 'history' to see session turns\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break
        if question.lower() == "history":
            if session_id:
                turns = load_turns(session_id, max_turns=20)
                for t in turns:
                    print(f"\n[{t['role'].upper()}] {t['content'][:200]}")
            else:
                print("No session active.")
            continue

        answer = run_question(
            question,
            session_id=session_id,
            use_memory=use_memory,
            verbose=verbose,
        )
        print(f"\nAgent: {answer}\n")


def main():
    args = parse_args()

    # -- List sessions
    if args.list_sessions:
        sessions = list_sessions()
        if sessions:
            print("Active sessions:")
            for s in sessions:
                print(f"  {s}")
        else:
            print("No active sessions.")
        return

    # -- Clear session
    if args.clear_session:
        sid   = make_session_id(args.clear_session)
        count = clear_session(sid)
        print(f"Cleared {count} turns from session '{sid}'")
        return

    session_id = make_session_id(args.session) if args.session else None
    use_memory = not args.no_memory

    # -- Single question mode
    if args.question:
        answer = run_question(
            args.question,
            session_id=session_id,
            use_memory=use_memory,
            verbose=args.verbose,
        )
        print(f"\n{answer}")
        return

    # -- Interactive mode
    interactive_loop(session_id, use_memory, args.verbose)


if __name__ == "__main__":
    main()
