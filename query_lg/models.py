"""
models.py — shared model resolution from RunnableConfig.

WHAT GOES IN CONFIG: model selection, temperature, and (for testing) a
fake model factory to substitute for a real API call. Read via
config["configurable"].

WHAT NEVER GOES IN CONFIG: API keys / secrets. Those stay exactly where
your real system already keeps them — environment variables
(ANTHROPIC_API_KEY etc.), read automatically by init_chat_model / the
provider SDK. Config objects can end up logged or checkpointed; secrets
shouldn't travel through that path.

Every node that needs a model (planner, agent_node, reflexion) should
accept `config` as a second parameter and call resolve_model(config, ...)
rather than hardcoding a model string or constructing its own default.
"""

from typing import Optional
import logging
import os
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model

DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"

logger = logging.getLogger(__name__)


def _bootstrap_api_key() -> None:
    """Runs ONCE at import (not per-call, not per-node) — before this
    existed, query_lg relied purely on init_chat_model reading
    ANTHROPIC_API_KEY straight from the environment, with no awareness
    of Secrets Manager at all. That's fine for local dev (where you
    export it yourself) but left the deployed container with nothing
    to read on EC2, since V1's Secrets Manager fallback lives in
    query.config, not here.

    This reuses that EXACT same secret (trade-platform/{ENV}/anthropic-api-key)
    via the same resolution query.config._resolve_api_key() already
    does, so there's exactly one key managed in exactly one place — it
    just also sets it into os.environ here so init_chat_model picks it
    up with zero other code changes. If ANTHROPIC_API_KEY is already
    set (local dev, CI), this is a no-op and never touches it.
    Also bootstraps LANGSMITH_API_KEY from the same Secrets Manager
    pattern (trade-platform/{ENV}/langsmith-api-key) if not already set.
    """
    anthropic_done = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    langsmith_done = bool(os.environ.get("LANGSMITH_API_KEY", "").strip())

    if anthropic_done and langsmith_done:
        return

    env = os.environ.get("ENV", "dev")

    if not anthropic_done:
        try:
            from query.config import _resolve_api_key
            os.environ["ANTHROPIC_API_KEY"] = _resolve_api_key()
        except Exception as e:
            # Don't swallow silently: without this, the failure only surfaces
            # much later as the SDK's opaque "Could not resolve authentication
            # method" on the first model call.
            logger.error(
                "ANTHROPIC_API_KEY bootstrap failed (%s: %s) — every model "
                "call will fail. Check AWS credentials / ENV=%s, or export "
                "ANTHROPIC_API_KEY before starting.", type(e).__name__, e, env,
            )

    if not langsmith_done:
        try:
            import boto3
            sm = boto3.client("secretsmanager", region_name="us-east-2")
            response = sm.get_secret_value(
                SecretId=f"trade-platform/{env}/langsmith-api-key"
            )
            key = response.get("SecretString", "").strip()
            if key:
                os.environ["LANGSMITH_API_KEY"] = key
        except Exception as e:
            logger.warning(
                "LANGSMITH_API_KEY bootstrap failed (%s: %s) — tracing disabled.",
                type(e).__name__, e,
            )


_bootstrap_api_key()


def resolve_model(config: Optional[RunnableConfig], temperature: float = 0):
    """
    Resolve the model to use for this node's call, from config.

    Precedence:
      1. config["configurable"]["model_factory"] — a zero-arg callable
         returning a model instance. ONLY for testing — lets a full
         graph.ainvoke() run end to end with a fake model, no live API
         call, no credentials needed. Never set this in production.
      2. config["configurable"]["model_name"] — a real provider:model
         string (e.g. "anthropic:claude-sonnet-4-6", "openai:gpt-4o").
         This is the real production path — provider switching is
         changing this ONE config value at invoke time, not code.
      3. DEFAULT_MODEL — if config carries neither, per an ordinary
         direct call/test that doesn't care about model selection.

    API keys are NEVER read from config — init_chat_model resolves them
    from environment variables automatically, same as your real system.
    """
    configurable = (config or {}).get("configurable", {})

    model_factory = configurable.get("model_factory")
    if model_factory is not None:
        return model_factory()

    model_name = configurable.get("model_name", DEFAULT_MODEL)
    return init_chat_model(model_name, temperature=temperature)