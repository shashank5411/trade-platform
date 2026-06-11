import os
from pathlib import Path
import boto3

import yaml

_ENV = os.getenv("ENVIRONMENT", "dev")
_CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "sources"

_secrets_client = boto3.client("secretsmanager", region_name="us-east-2")
_cache = {}

def load_source_config(source: str) -> dict:
    path = _CONFIGS_DIR / f"{source}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_default_start(config: dict) -> str:
    # Support both new (default_start_date) and old (default_start_period) formats
    for key in ("default_start_date", "default_start_period"):
        if key in config:
            starts = config[key]
            return starts.get(_ENV, starts.get("dev", next(iter(starts.values()))))
    raise KeyError("No default start date/period found in config")

def get_secret(secret_id: str) -> str:
    """Fetch secret from Secrets Manager, cached per process."""
    if secret_id not in _cache:
        resp = _secrets_client.get_secret_value(SecretId=secret_id)
        _cache[secret_id] = resp["SecretString"]
    return _cache[secret_id]

def get_anthropic_key() -> str:
    # Fall back to env var for local dev convenience
    return os.getenv("ANTHROPIC_API_KEY") or get_secret(
        f"trade-platform/{os.getenv('ENVIRONMENT', 'dev')}/anthropic-api-key"
    )
