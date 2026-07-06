"""
Chart Registry — single source of truth for which tools are chartable and how.

A tool is charted ONLY if it has an explicit entry here. This is intentionally
opt-in, not opt-out: adding a new dataset/tool to the platform (e.g. a future
weather or Congressional-trades tool) has ZERO effect on charting unless someone
explicitly adds a registry entry for it. There is no name-pattern matching or
"looks numeric" heuristic anywhere in the charting code — this dict is the only
place that decision is made.

Fields:
  bucket:     "timeseries" (deterministic parse, no LLM) or
              "categorical" (Haiku extraction from tool result text)
  merge_key:  tool calls sharing this key across the WHOLE query
              (all nodes, all agents) become ONE chart. Different
              merge_keys always become separate charts.
  parser:     name of the parser function in chart_agent.py that
              handles this tool's raw result text (for timeseries)
              or the Haiku extraction prompt variant to use (for
              categorical).
"""

CHART_REGISTRY = {
    "get_prices": {
        "bucket":    "timeseries",
        "merge_key": "price",
        "parser":    "parse_price_table",
    },
    "get_prices_multi": {
        "bucket":    "timeseries",
        "merge_key": "price",
        "parser":    "parse_price_table",
    },
    "get_indicator": {
        "bucket":    "timeseries",
        "merge_key": "indicator",
        "parser":    "parse_indicator_table",
    },
    "get_indicator_multi": {
        "bucket":    "timeseries",
        "merge_key": "indicator",
        "parser":    "parse_indicator_table",
    },
    "get_insider_summary": {
        "bucket":    "categorical",
        "merge_key": "insider",
        "parser":    "llm_extract_categorical",
    },
    "get_news_summary": {
        "bucket":    "categorical",
        "merge_key": "news",
        "parser":    "llm_extract_categorical",
    },
}

MAX_CHARTS_PER_ANSWER = 3


def get_chart_config(tool_name: str) -> dict | None:
    """Returns the registry entry for a tool, or None if the tool is not
    registered for charting. None means 'skip silently' — never treat this
    as an error."""
    return CHART_REGISTRY.get(tool_name)
