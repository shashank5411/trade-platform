"""
query/config.py — shared Anthropic client factory.

Resolves the API key in priority order:
  1. AWS Secrets Manager  (trade-platform/{env}/anthropic-api-key)
  2. ANTHROPIC_API_KEY    environment variable (local dev / CI fallback)

The client is module-level singleton — boto3 call happens once per
process, not once per request.
"""

import os
import anthropic

ENV = os.environ.get("ENV", "dev")

_client: anthropic.Anthropic | None = None


def _resolve_api_key() -> str:
    """
    Try Secrets Manager first, fall back to env var.
    Raises ValueError if neither is available.
    """
    # ── 1. Secrets Manager ───────────────────────────────────────────────────
    secret_name = f"trade-platform/{ENV}/anthropic-api-key"
    try:
        import boto3
        sm = boto3.client("secretsmanager", region_name="us-east-2")
        response = sm.get_secret_value(SecretId=secret_name)
        key = response.get("SecretString", "").strip()
        if key:
            return key
    except Exception:
        # Secrets Manager unavailable (no creds, wrong region, secret missing)
        # — fall through to env var
        pass

    # ── 2. Environment variable ──────────────────────────────────────────────
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key

    raise ValueError(
        "Anthropic API key not found. "
        f"Set Secrets Manager secret '{secret_name}' "
        "or set the ANTHROPIC_API_KEY environment variable."
    )


def get_client() -> anthropic.Anthropic:
    """Return the module-level Anthropic client, creating it on first call."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=_resolve_api_key())
    return _client