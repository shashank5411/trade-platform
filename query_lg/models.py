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
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model

DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"


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