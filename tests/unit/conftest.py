import os

# query.config.get_client() is called at *import time* by several query/
# modules (dag_executor.py, sub_agents.py, reflexion.py, planner.py) to
# build a module-level Anthropic client singleton. Without a resolvable key
# (Secrets Manager unavailable in a test/CI sandbox with no AWS creds, and
# no ANTHROPIC_API_KEY set), that raises ValueError before any test module
# even imports its target. This dummy value only needs to satisfy
# anthropic.Anthropic()'s constructor (no network call happens until
# .messages.create() is actually invoked) — tests that exercise LLM calls
# mock client.messages.create() directly rather than relying on this key
# being real.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-dummy-key-not-real")
os.environ.setdefault("ENV", "dev")
