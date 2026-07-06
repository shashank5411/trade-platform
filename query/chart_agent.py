"""
query/chart_agent.py — Registry-based chart agent.

Replaces chart_extractor.py. Produces 0-3 Chart.js-ready specs per query by
reading the actual tool result data that agents already fetched, rather than
re-extracting from the compressed prose answer.

Public entry point: build_charts(question, answer, node_tool_calls) -> list[dict]
"""

import json
import re

from query.chart_registry import CHART_REGISTRY, MAX_CHARTS_PER_ANSWER, get_chart_config
from query.config import get_client

HAIKU_MODEL = "claude-haiku-4-5-20251001"


# ══════════════════════════════════════════════════════════════════════════
# GRANULARITY EXTRACTION (shared by both parsers)
# ══════════════════════════════════════════════════════════════════════════

def _extract_granularity_note(text: str) -> str:
    """Extract a human-readable granularity note from any tool result text.

    Handles four formats:
      - get_prices single:   "Granularity: weekly (52 periods)"
      - get_prices_multi:    "Granularity: weekly | Tickers: ..."
      - get_indicator single: "Unit: Percent | Granularity: monthly | Periods: 48"
      - get_indicator_multi:  "Granularities: FEDFUNDS=monthly, UNRATE=monthly | ..."
    """
    # Canonical granularity names
    _GRAN_NAMES = r'(annual|quarterly|monthly|weekly|daily)'

    # get_indicator_multi: "Granularities: SID=gran, ..."
    m = re.search(r'Granularities:.*?=' + _GRAN_NAMES, text, re.IGNORECASE)
    if m:
        gran = m.group(1).lower()
    else:
        # All other forms: "Granularity: gran ..."
        m = re.search(r'Granularity:\s*' + _GRAN_NAMES, text, re.IGNORECASE)
        gran = m.group(1).lower() if m else ""

    # Period count — either "(52 periods)" or "Periods: 48"
    pm = re.search(r'\((\d+)\s*periods?\)', text, re.IGNORECASE)
    if not pm:
        pm = re.search(r'Periods:\s*(\d+)', text, re.IGNORECASE)
    periods = pm.group(1) if pm else None

    if gran and periods:
        return f"{gran}, {periods} periods"
    return gran


# ══════════════════════════════════════════════════════════════════════════
# TIMESERIES PARSERS (deterministic, no LLM)
# ══════════════════════════════════════════════════════════════════════════

def _parse_price_table(text: str) -> dict:
    """Parse get_prices / get_prices_multi result text.

    Both tools produce a pandas to_string(index=False) table with a "ticker"
    column as the first column. Granularity-dependent column differences:
      - Daily:          ticker, date, open, high, low, close, adj_close, volume
      - Weekly/Monthly: ticker, date, low, high, open, close, avg_close, volume
    Both contain "close" — that column is always used as the plotted value.

    Returns {"series": [...], "granularity_note": "..."}.
    Each series: {"label": str, "unit": "$", "data": [float,...], "labels": [str,...]}.
    """
    granularity_note = _extract_granularity_note(text)

    # Locate the table section
    if "Full data:" in text:
        # get_prices_multi: table follows "Full data:\n"
        table_text = text.split("Full data:", 1)[1].strip()
    else:
        # get_prices single: table follows "Granularity: ...\n\n"
        # re.split with . not matching \n by default, so this matches the
        # granularity line + its trailing blank line correctly.
        parts = re.split(r'Granularity:.*\n\n', text, maxsplit=1)
        table_text = parts[1].strip() if len(parts) > 1 else text

    lines = table_text.splitlines()

    # Find the header row — first non-blank line whose first token is "ticker"
    header_idx = None
    headers    = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        parts = re.split(r'\s+', stripped)
        if parts and parts[0].lower() == "ticker":
            header_idx = i
            headers    = [h.lower() for h in parts]
            break

    if header_idx is None or not headers:
        return {"series": [], "granularity_note": granularity_note}

    try:
        close_idx = headers.index("close")
    except ValueError:
        try:
            close_idx = headers.index("avg_close")
        except ValueError:
            return {"series": [], "granularity_note": granularity_note}

    try:
        date_idx = headers.index("date")
    except ValueError:
        return {"series": [], "granularity_note": granularity_note}

    # Parse data rows; group by ticker (always first column, index 0)
    series_data: dict[str, list] = {}
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = re.split(r'\s+', stripped)
        if len(parts) <= max(date_idx, close_idx):
            continue
        try:
            ticker    = parts[0]
            date_val  = parts[date_idx]
            close_val = float(parts[close_idx])
            # Basic date sanity — skip header-like lines that snuck through
            if not re.match(r'^\d{4}-\d{2}-\d{2}', date_val):
                continue
            if ticker not in series_data:
                series_data[ticker] = []
            series_data[ticker].append((date_val, close_val))
        except (ValueError, IndexError):
            continue

    series = []
    for ticker, points in series_data.items():
        if not points:
            continue
        series.append({
            "label":  ticker,
            "unit":   "$",
            "data":   [p[1] for p in points],
            "labels": [p[0] for p in points],
        })

    return {"series": series, "granularity_note": granularity_note}


# Regex for a multi-indicator data row.
# Columns: indicator_id  date  value  unit...  country
# "unit" may contain spaces (e.g. "Billions of Dollars", "Current US$").
#
# The trailing "country" column is non-capturing because it is never used
# for charting (see _parse_indicator_table's is_multi branch below).
# Using a non-capturing permissive token (?:\S+)? instead of ([A-Z]{2}|None)
# fixes a data-loss bug: pd.read_csv (used by athena.py's _fetch_results)
# converts SQL NULLs to float NaN, so df.to_string() prints "NaN" for
# country-less FRED rows — not "None". The old anchored group never matched
# "NaN", silently dropping every FRED row in a mixed FRED+WorldBank result.
_INDICATOR_MULTI_ROW = re.compile(
    r'^(\S+)'                          # indicator_id
    r'\s+(\d{4}-\d{2}-\d{2})'         # date
    r'\s+([\d.eE+\-]+)'               # value (numeric, possibly scientific)
    r'\s+(.*?)'                        # unit (lazy, may have spaces)
    r'\s+(?:\S+)?'                     # country token — permissive, non-capturing
    r'\s*$'
)


def _parse_indicator_table(text: str) -> dict:
    """Parse get_indicator / get_indicator_multi result text.

    get_indicator (single):
      First line:  "Indicator Name (SERIES_ID)"
      Header line: "Unit: X | Granularity: Y | Periods: N"
      Table cols:  date, value  [or  date, value, country]

    get_indicator_multi:
      First line:  "Indicators: [...]"
      Header line: "Granularities: ... | Period: ..."
      Table cols:  indicator_id, date, value, unit, country

    Returns {"series": [...], "granularity_note": "..."}.
    Each series: {"label": str, "unit": str, "data": [...], "labels": [...]}.
    """
    granularity_note = _extract_granularity_note(text)
    lines = text.splitlines()

    # Find the table header row
    header_idx = None
    headers    = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        parts = re.split(r'\s+', stripped)
        if parts and parts[0].lower() in ("date", "indicator_id"):
            header_idx = i
            headers    = [h.lower() for h in parts]
            break

    if header_idx is None or not headers:
        return {"series": [], "granularity_note": granularity_note}

    is_multi = "indicator_id" in headers

    if is_multi:
        # Multi-mode: parse each row with the regex that handles spaces in unit
        series_data: dict[str, dict] = {}
        for line in lines[header_idx + 1:]:
            stripped = line.strip()
            if not stripped:
                continue
            m = _INDICATOR_MULTI_ROW.match(stripped)
            if not m:
                continue
            try:
                sid      = m.group(1)
                date_val = m.group(2)
                value    = float(m.group(3))
                unit     = m.group(4).strip()
                if sid not in series_data:
                    series_data[sid] = {"points": [], "unit": unit}
                series_data[sid]["points"].append((date_val, value))
                if not series_data[sid]["unit"]:
                    series_data[sid]["unit"] = unit
            except (ValueError, AttributeError):
                continue

    else:
        # Single-mode: extract series_id and unit from header text
        id_match   = re.search(r'\(([A-Za-z0-9_.^&\-]+)\)', text)
        series_id  = id_match.group(1) if id_match else "Series"
        unit_match = re.search(r'Unit:\s*(.+?)\s*\|', text)
        unit       = unit_match.group(1).strip() if unit_match else ""

        try:
            date_idx  = headers.index("date")
            value_idx = headers.index("value")
        except ValueError:
            return {"series": [], "granularity_note": granularity_note}

        points = []
        for line in lines[header_idx + 1:]:
            stripped = line.strip()
            if not stripped:
                continue
            parts = re.split(r'\s+', stripped)
            if len(parts) <= max(date_idx, value_idx):
                continue
            try:
                date_val = parts[date_idx]
                if not re.match(r'^\d{4}-\d{2}-\d{2}', date_val):
                    continue
                value = float(parts[value_idx])
                points.append((date_val, value))
            except (ValueError, IndexError):
                continue

        series_data = {series_id: {"points": points, "unit": unit}}

    # Build series list
    series = []
    for sid, data in series_data.items():
        pts = data["points"]
        if not pts:
            continue
        series.append({
            "label":  sid,
            "unit":   data["unit"],
            "data":   [p[1] for p in pts],
            "labels": [p[0] for p in pts],
        })

    return {"series": series, "granularity_note": granularity_note}


# ══════════════════════════════════════════════════════════════════════════
# TIMESERIES SPEC BUILDER
# ══════════════════════════════════════════════════════════════════════════

def _build_timeseries_spec(series_list: list, granularity_note: str) -> dict | None:
    """Merge parsed series into one Chart.js-ready spec with dual-axis support."""
    valid = [s for s in series_list if len(s.get("data", [])) >= 2]
    if not valid:
        return None

    # Dual-axis assignment: first distinct unit → "y", second → "y1",
    # third+ → whichever axis has fewer series (tie → "y").
    unit_to_axis:  dict[str, str] = {}
    axis_count:    dict[str, int] = {"y": 0, "y1": 0}

    for s in valid:
        unit = s.get("unit", "")
        if unit not in unit_to_axis:
            if not unit_to_axis:
                unit_to_axis[unit] = "y"
            elif len(unit_to_axis) == 1:
                unit_to_axis[unit] = "y1"
            else:
                # Third+ distinct unit — assign to least-loaded axis
                unit_to_axis[unit] = "y" if axis_count["y"] <= axis_count["y1"] else "y1"
        axis_count[unit_to_axis[unit]] += 1

    has_dual = "y1" in unit_to_axis.values()

    # Derive axis labels from unit assignments
    y_label  = next((u for u, a in unit_to_axis.items() if a == "y"),  "")
    y1_label = next((u for u, a in unit_to_axis.items() if a == "y1"), "")

    chart_type = "line" if len(valid) == 1 else "multiline"

    # Union of all real dates across every series, sorted chronologically.
    # ISO YYYY-MM-DD strings sort correctly as plain strings.
    all_dates = sorted(set(
        d for s in valid for d in s["labels"]
    ))
    date_index = {d: i for i, d in enumerate(all_dates)}

    datasets = []
    for s in valid:
        aligned_data = [None] * len(all_dates)
        for d, v in zip(s["labels"], s["data"]):
            aligned_data[date_index[d]] = v
        ds: dict = {"label": s["label"], "data": aligned_data}
        if has_dual:
            ds["yAxisID"] = unit_to_axis.get(s.get("unit", ""), "y")
        datasets.append(ds)

    # Chart title: list the series labels
    labels_preview = ", ".join(s["label"] for s in valid[:3])
    if len(valid) > 3:
        labels_preview += f" +{len(valid) - 3} more"

    spec = {
        "type":             chart_type,
        "title":            labels_preview,
        "x_label":          "Date",
        "y_label":          y_label or "$",
        "granularity_note": granularity_note,
        "labels":           all_dates,
        "datasets":         datasets,
    }
    if has_dual:
        spec["y1_label"] = y1_label

    return spec


# ══════════════════════════════════════════════════════════════════════════
# CATEGORICAL EXTRACTOR (Haiku, grounded in raw tool result text)
# ══════════════════════════════════════════════════════════════════════════

_CAT_PROMPT = """You extract chart data from financial tool results.

Analyze the tool result(s) below. If the data is chart-worthy return ONLY a JSON object (no markdown fences, no explanation). If not chart-worthy return exactly: null

CHART-WORTHY:
- Categorical totals or rankings (insider $ by type, net buy vs sell) -> "horizontal_bar"
- Breakdown into parts that sum to a whole (sentiment counts) -> "pie"
- Simple categorical comparison (not time-based) -> "bar"

NOT CHART-WORTHY:
- Fewer than 3 distinct data points
- Pure text with no numeric breakdown
- Time series data

JSON SCHEMA (follow exactly):
{
  "type": "horizontal_bar",
  "title": "short descriptive title",
  "x_label": "x axis label",
  "y_label": "y axis label",
  "granularity_note": "",
  "datasets": [
    {
      "label": "series name",
      "data": [1.0, 2.0, 3.0],
      "labels": ["label1", "label2", "label3"]
    }
  ]
}

RULES:
- type must be one of: bar, horizontal_bar, pie
- Only use numbers explicitly stated in the tool results. Never invent or infer.
- Data must be plain numbers (strip $, %, commas).
- For pie: one dataset, labels AND data inside it.
- Minimum 3 data points required — return null if fewer.
- If NOT chart-worthy return exactly: null

Tool results:
{tool_results}"""


def _llm_extract_categorical(texts: list) -> dict | None:
    """Haiku extraction from concatenated raw tool result text."""
    combined = "\n\n---\n\n".join(texts)
    raw = ""
    try:
        client   = get_client()
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=800,
            messages=[{
                "role":    "user",
                "content": _CAT_PROMPT.replace("{tool_results}", combined[:6000]),
            }],
        )
        raw = response.content[0].text.strip()
        print(f"[ChartAgent] Categorical raw: {repr(raw[:200])}")

        if not raw or raw.lower() == "null":
            return None

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            print("[ChartAgent] Categorical: no JSON object found")
            return None

        spec = json.loads(match.group(0))
        if not isinstance(spec, dict):
            return None
        if spec.get("type") not in ("bar", "horizontal_bar", "pie"):
            print(f"[ChartAgent] Categorical: invalid type {spec.get('type')!r}")
            return None

        datasets = spec.get("datasets")
        if not datasets or not isinstance(datasets, list):
            return None
        for ds in datasets:
            if not ds.get("data") or len(ds["data"]) < 3:
                print("[ChartAgent] Categorical rejected: fewer than 3 data points")
                return None

        # Ensure granularity_note key exists (Haiku may omit it)
        spec.setdefault("granularity_note", "")
        print(f"[ChartAgent] Categorical success: {spec.get('type')} — {spec.get('title')}")
        return spec

    except Exception as e:
        print(f"[ChartAgent] Categorical failed: {e}")
        print(f"[ChartAgent] Raw was: {repr(raw[:200])}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def build_charts(
    question:        str,
    answer:          str,
    node_tool_calls: dict,
) -> list:
    """
    Returns a list of 0-3 Chart.js-ready spec dicts built from the real
    tool result data fetched during this query (not from the prose answer).

    Each spec has:
      type, title, x_label, y_label, granularity_note, datasets
    Plus optional y1_label and per-dataset yAxisID when dual-axis applies.

    node_tool_calls: dict mapping node_id -> list of Trace.tools_called records.
    Each record has: name, inputs_preview, result_preview, was_dedup, result_full.
    """
    # 1. Flatten all tool calls, dropping dedup nudge records
    all_calls = []
    for calls in node_tool_calls.values():
        for call in calls:
            if not call.get("was_dedup", False):
                all_calls.append(call)

    # 2. Look up each call in the registry; group by merge_key
    #    — calls not in the registry are silently skipped (opt-in gate)
    groups: dict[str, dict] = {}
    for call in all_calls:
        config = get_chart_config(call.get("name", ""))
        if config is None:
            continue
        mk = config["merge_key"]
        if mk not in groups:
            groups[mk] = {"bucket": config["bucket"], "calls": []}
        groups[mk]["calls"].append(call)

    # 3. Process each group into a chart spec
    timeseries_specs  = []
    categorical_specs = []

    for mk, group in groups.items():
        bucket = group["bucket"]
        calls  = group["calls"]

        if bucket == "timeseries":
            all_series     = []
            seen_labels    = set()
            gran_note      = ""

            for call in calls:
                result_text = call.get("result_full") or call.get("result_preview", "")
                parser_name = CHART_REGISTRY[call["name"]]["parser"]

                if parser_name == "parse_price_table":
                    parsed = _parse_price_table(result_text)
                elif parser_name == "parse_indicator_table":
                    parsed = _parse_indicator_table(result_text)
                else:
                    continue

                for s in parsed.get("series", []):
                    if s["label"] not in seen_labels:
                        all_series.append(s)
                        seen_labels.add(s["label"])

                if not gran_note:
                    gran_note = parsed.get("granularity_note", "")

            spec = _build_timeseries_spec(all_series, gran_note)
            if spec:
                timeseries_specs.append(spec)

        elif bucket == "categorical":
            texts = [
                call.get("result_full") or call.get("result_preview", "")
                for call in calls
            ]
            spec = _llm_extract_categorical(texts)
            if spec:
                categorical_specs.append(spec)

    # 4. Cap at MAX_CHARTS_PER_ANSWER — drop categorical-bucket specs first
    while (
        len(timeseries_specs) + len(categorical_specs) > MAX_CHARTS_PER_ANSWER
        and categorical_specs
    ):
        categorical_specs.pop(0)

    specs = (timeseries_specs + categorical_specs)[:MAX_CHARTS_PER_ANSWER]
    print(f"[ChartAgent] {len(specs)} chart(s) built "
          f"({len(timeseries_specs)} timeseries, {len(categorical_specs)} categorical)")
    return specs
