"""
Shared ETL transformation utilities.
Used by all 5 ETL scripts to enforce canonical schema contracts.
"""

import json
import hashlib
from datetime import datetime, date, timezone
from typing import Optional


def now_utc() -> str:
    """Current UTC timestamp as ISO string for ingested_at field."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def normalize_date_to_period_start(d: date, frequency: str) -> date:
    """
    Normalize any date to the first date of its period.
    Monthly  → first of month
    Quarterly→ first of quarter (Jan/Apr/Jul/Oct)
    Annual   → Jan 1
    Daily    → unchanged
    """
    if frequency == "annual":
        return date(d.year, 1, 1)
    elif frequency == "quarterly":
        quarter_start_month = ((d.month - 1) // 3) * 3 + 1
        return date(d.year, quarter_start_month, 1)
    elif frequency == "monthly":
        return date(d.year, d.month, 1)
    else:
        return d  # daily — unchanged


def safe_float(val) -> Optional[float]:
    """
    Convert to float, returning None on null/empty/invalid.
    Never coerce None to 0.0 — nulls are meaningful (World Bank gaps).
    """
    if val is None:
        return None
    try:
        f = float(val)
        return None if (f != f) else f  # NaN check
    except (ValueError, TypeError):
        return None


def to_json_str(d: dict) -> str:
    """Serialize dict to JSON string for metadata column."""
    return json.dumps(d, default=str)


def make_doc_id(source: str, *parts) -> str:
    """
    Generate stable doc_id from source + identifying parts.
    EDGAR  → make_doc_id("EDGAR", accession_number)
    WIKI   → make_doc_id("WIKIPEDIA", title, snapshot_date)
    """
    raw = f"{source}_{'_'.join(str(p) for p in parts)}"
    # Keep it readable for short IDs, hash for long ones
    if len(raw) <= 100:
        return raw.upper().replace(" ", "_")
    return f"{source}_{hashlib.md5(raw.encode()).hexdigest()[:12]}"


def clean_text(text: str) -> str:
    """
    Strip common markup artifacts from raw text.
    Removes excessive whitespace, null bytes, non-printable chars.
    Does NOT strip content — that's source-specific ETL's job.
    """
    if not text:
        return ""
    # Null bytes
    text = text.replace("\x00", "")
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ consecutive blank lines to 2
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace
    return text.strip()


def validate_market_price_row(row: dict) -> list:
    """
    Return list of validation errors for a market_prices row.
    Empty list = valid.
    """
    errors = []
    required = ["ticker", "exchange", "date", "year",
                "currency", "country", "source", "ingested_at"]
    for f in required:
        if not row.get(f):
            errors.append(f"missing required field: {f}")
    if row.get("close") is None and row.get("adj_close") is None:
        errors.append("both close and adj_close are null")
    if row.get("volume") is not None and row["volume"] < 0:
        errors.append(f"negative volume: {row['volume']}")
    return errors


# def validate_indicator_row(row: dict) -> list:
#     """Return list of validation errors for an economic_indicators row."""
#     errors = []
#     required = ["source", "indicator_id", "indicator_name",
#                 "date", "vintage_date", "year",
#                 "unit", "frequency", "ingested_at"]
#     for f in required:
#         if not row.get(f):
#             errors.append(f"missing required field: {f}")
#     # value CAN be None (World Bank gaps) — not an error
#     return errors
def validate_indicator_row(row: dict) -> list:
    """Return list of validation errors for an economic_indicators row."""
    errors = []
    required = ["source", "indicator_id", "indicator_name",
                "date", "vintage_date", "year",
                "frequency", "ingested_at"]   # ← unit removed
    for f in required:
        if not row.get(f):
            errors.append(f"missing required field: {f}")
    return errors

def validate_document_row(row: dict) -> list:
    """Return list of validation errors for a documents row."""
    errors = []
    required = ["doc_id", "source", "year", "title",
                "entity", "doc_type", "doc_date", "ingested_at"]
    for f in required:
        if not row.get(f):
            errors.append(f"missing required field: {f}")
    if not row.get("text") or len(row.get("text", "")) < 50:
        errors.append("text missing or too short (< 50 chars) — likely empty filing")
    return errors