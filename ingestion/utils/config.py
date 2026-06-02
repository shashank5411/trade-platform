import os
from pathlib import Path

import yaml

_ENV = os.getenv("ENVIRONMENT", "dev")
_CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "sources"


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
