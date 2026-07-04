"""
query/chart_extractor.py — Extracts Chart.js-ready specs from synthesis answers.
"""

import json
import re

from query.config import get_client

HAIKU_MODEL = "claude-haiku-4-5-20251001"

_PROMPT = """You extract chart data from financial analysis answers.

Analyze the question and answer. If the answer contains chart-worthy structured
data, return ONLY a JSON object (no markdown fences, no explanation, no preamble).
If not chart-worthy, return exactly: null

CHART-WORTHY:
- Time series: prices/rates/indicators across 3+ dates or periods -> "line"
- Multi-entity comparison over same dates -> "multiline"
- Categorical ranking or totals (e.g. insider $ by ticker) -> "horizontal_bar"
- Breakdown into parts summing to a whole (e.g. sentiment counts) -> "pie"
- Simple categorical comparison (not time-based) -> "bar"

NOT CHART-WORTHY:
- Fewer than 3 data points
- Pure text with no numeric data
- Single numbers without a category axis

JSON SCHEMA (follow exactly, no markdown, no preamble):
{
  "type": "line",
  "title": "short descriptive title",
  "x_label": "x axis label",
  "y_label": "y axis label",
  "datasets": [
    {
      "label": "series name",
      "data": [1.0, 2.0, 3.0],
      "labels": ["label1", "label2", "label3"]
    }
  ]
}

RULES:
- type must be one of: line, multiline, bar, horizontal_bar, pie
- Only use numbers explicitly stated in the answer. Never invent or infer.
- For multiline: all datasets share labels — put labels only on the first dataset.
- For pie: one dataset, with both labels and data inside it.
- For horizontal_bar: categories in labels, values in data, sorted descending.
- Data must be plain numbers (strip $, %, commas before putting in data array).
- If NOT chart-worthy respond with exactly the word: null

Question: {question}

Answer: {answer}"""


def extract_chart(question: str, answer: str) -> dict | None:
    """
    Call Haiku to extract a chart spec. Returns dict or None.
    Runs in threadpool — blocking call.
    """
    raw = ""
    try:
        client = get_client()
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": _PROMPT
                    .replace("{question}", question[:600])
                    .replace("{answer}", answer[:4000]),
            }],
        )
        raw = response.content[0].text.strip()
        print(f"[ChartExtractor] Raw response: {repr(raw[:300])}")

        # Null check first
        if not raw or raw.lower() == "null":
            return None

        # Find JSON object anywhere in the response
        # Handles markdown fences, preamble, or any extra text
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            print(f"[ChartExtractor] No JSON object found in response")
            return None

        json_str = match.group(0)
        spec = json.loads(json_str)

        # Validate
        if not isinstance(spec, dict):
            return None
        if spec.get("type") not in ("line", "multiline", "bar", "horizontal_bar", "pie"):
            print(f"[ChartExtractor] Invalid type: {spec.get('type')}")
            return None
        datasets = spec.get("datasets")
        if not datasets or not isinstance(datasets, list):
            return None
        for ds in datasets:
            if not ds.get("data") or len(ds["data"]) < 2:
                print(f"[ChartExtractor] Rejected: dataset has fewer than 2 points")
                return None

        print(f"[ChartExtractor] Success: {spec.get('type')} — {spec.get('title')}")
        return spec

    except Exception as e:
        print(f"[ChartExtractor] Failed: {e}")
        print(f"[ChartExtractor] Raw was: {repr(raw[:300])}")
        return None