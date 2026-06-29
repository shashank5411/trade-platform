"""
Business logic — full tool suite for the economic intelligence platform.

Tools:
  PRICE:      get_prices, get_prices_multi, get_price_on_date, get_prices_on_date
  INDICATOR:  get_indicator, get_indicator_multi, get_indicator_on_date
  DOCUMENT:   get_documents, get_prose
  MACRO:      get_macro_snapshot (wrapper around get_indicator_on_date + SPY)
  SEARCH:     semantic_search

Partition layout change (Phase 9+):
  market_prices partitioned by ticker= / year=  (was year= / exchange=)
  - _ticker_partition() and _ticker_partitions() added for partition pruning
  - _safe_partition_value() mirrors etl_yfinance sanitization
  - All price query functions updated to include ticker= partition hint
"""

import json
import os
import re
import sys
import boto3
import pandas as pd
from datetime import date, timedelta
from typing import Optional
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from query.athena import query, AthenaError, AthenaQueryError

# ── Database names ─────────────────────────────────────────────────────────
ENV     = os.environ.get("ENV", "dev")
ACCOUNT = os.environ.get("ACCOUNT", "197411402303")

DB = {
    "fred":      f"{ENV}_trade_fred_processed",
    "worldbank": f"{ENV}_trade_worldbank_processed",
    "yfinance":  f"{ENV}_trade_yfinance_processed",
    "wikipedia": f"{ENV}_trade_wikipedia_processed",
    "sec":       f"{ENV}_trade_sec_processed",
    "sec_prose": f"{ENV}_trade_sec_prose_processed",
    "news":      f"{ENV}_trade_news_processed",
    "insiders":  f"{ENV}_trade_insiders_processed",
}

# ── Vector search ──────────────────────────────────────────────────────────
VECTOR_BUCKET  = f"{ENV}-trade-vectors-{ACCOUNT}"
VECTOR_INDEX   = "documents-index"
BEDROCK_REGION = "us-east-1"
EMBED_MODEL_ID = "cohere.embed-english-v3"
EMBED_DIM      = 1024

_bedrock   = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
_s3vectors = boto3.client("s3vectors",       region_name="us-east-2")

# ── Default macro indicators ───────────────────────────────────────────────
DEFAULT_MACRO = [
    ("FEDFUNDS", "FRED",      None),
    ("UNRATE",   "FRED",      None),
    ("CPIAUCSL", "FRED",      None),
    ("DGS10",    "FRED",      None),
    ("DGS2",     "FRED",      None),
    ("GDP",      "FRED",      None),
    ("NY.GDP.MKTP.CD", "WORLDBANK", "US"),
    ("^VIX", "yfinance_index", None),
    ("DX-Y.NYB", "yfinance_fx", None),
    ("GC=F", "yfinance_futures", None),
]

# ── get_prose limits ───────────────────────────────────────────────────────
PROSE_DEFAULT_CHARS = 8_000
PROSE_MAX_CHARS     = 20_000


# ══════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _range_days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _price_granularity(start: str, end: str) -> str:
    days = _range_days(start, end)
    if days <= 30:
        return "daily"
    elif days <= 365:
        return "weekly"
    else:
        return "monthly"


def _indicator_granularity(start: str, end: str,
                            native_freq: str = "monthly") -> str:
    days = _range_days(start, end)
    if native_freq == "annual":
        return "annual"
    elif native_freq == "quarterly":
        return "annual" if days > 730 else "quarterly"
    elif native_freq == "monthly":
        return "quarterly" if days > 730 else "monthly"
    else:  # daily
        if days > 365:
            return "monthly"
        elif days > 30:
            return "weekly"
        return "daily"


def _get_native_frequency(series_id: str, db: str) -> str:
    """
    Look up a series' actual stored frequency from the data itself, rather
    than assuming. The `frequency` column is already stored correctly on
    every row in economic_indicators (FRED's etl_fred.py infers it via
    SERIES_FREQUENCY_OVERRIDE / FREQ_HINTS at ETL time; WorldBank rows are
    always 'annual'). Reading it back here means the query layer never
    needs to duplicate that classification logic — it just asks the data
    what it actually is.

    Falls back to 'monthly' only if no rows exist yet for this series
    (e.g. not ingested yet) — same default behavior as before this fix,
    so an uningested series doesn't error, it just gets the old behavior.
    """
    sql = f"""
        SELECT frequency
        FROM economic_indicators
        WHERE indicator_id = '{series_id}'
        LIMIT 1
    """
    try:
        df = query(sql, db)
        if not df.empty:
            return df.iloc[0]["frequency"]
    except Exception:
        pass
    return "monthly"


def _year_filter(start: str, end: str) -> str:
    sy = int(start[:4])
    ey = int(end[:4])
    return f"AND CAST(year AS INTEGER) BETWEEN {sy} AND {ey}"


def _safe_partition_value(value: str) -> str:
    """
    Sanitize ticker for S3 partition path — mirrors etl_yfinance.py.
    Must stay in sync with ETL: ^GSPC→GSPC, CL=F→CL_F, BRK-B→BRK_B
    """
    return (value
            .replace("^", "")
            .replace("=", "_")
            .replace("-", "_")
            .replace(".", "_"))


def _ticker_partition(ticker: str) -> str:
    """
    Athena partition pruning hint for a single ticker.
    market_prices is partitioned ticker= / year= (Phase 9+).
    Sanitized value matches the S3 path: ^GSPC → ticker=GSPC partition.
    Prunes to one directory before the year= scan — 16x cheaper queries.

    SECURITY NOTE (added in the SQL-hardening pass): _safe_partition_value()
    above does NOT escape quote characters — it only performs the specific
    character substitutions needed to mirror etl_yfinance.py's S3 partition
    naming (^, =, -, .). A ticker containing a single quote would pass
    through it completely unescaped. _validate_ticker() below is the real
    security control, applied here BEFORE partition-sanitization.
    """
    ticker = _validate_ticker(ticker, "ticker")
    safe = _safe_partition_value(ticker)
    return f"AND ticker = '{safe}'"


def _ticker_partitions(tickers: list) -> str:
    """Athena partition pruning hint for a list of tickers (IN clause).
    Same validate-before-sanitize note as _ticker_partition() above."""
    tickers = [_validate_ticker(t, "ticker") for t in tickers]
    safe_list = "','".join(_safe_partition_value(t) for t in tickers)
    return f"AND ticker IN ('{safe_list}')"


# ══════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION (SQL hardening, pre-HTTP-surface)
# ══════════════════════════════════════════════════════════════════════════
# api.py builds Athena SQL via f-string interpolation throughout. Safe today
# only because callers are constrained by Anthropic tool schemas — but tool
# schemas in tools.py mostly use bare {"type": "string"} with no "pattern"
# regex constraint, so the schema enforces almost nothing about SHAPE (dates,
# tickers) even where it does enforce an enum for a fixed value set. These
# helpers are the real validation layer, applied at the top of every public
# tool function below, BEFORE any SQL string is built — never trust the
# schema alone as the only gate.
#
# CONFIRMED (live test against this project's real Athena endpoint, see
# 2026-06-28 session notes): boto3's start_query_execution DOES support
# genuine "?"-placeholder parameterization via ExecutionParameters, no
# prepared statement needed. Deliberately NOT used here — that would require
# threading a `params` list through athena.py's shared query()/cache-key
# functions (used by every single function in this file), and a correctly
# anchored allowlist regex is provably equivalent in security outcome to
# parameterization for every value actually used in this codebase (none of
# them need characters a reasonable allowlist would exclude). Revisit if a
# future field genuinely needs unrestricted free text.

class ToolInputError(Exception):
    """
    Raised when a tool argument fails validation, before any SQL is built.
    Caught at the top of each public tool function (mirrors how Athena
    failures are caught and converted via AthenaQueryError.agent_message())
    and converted to a clear, agent-facing tool-result string — never lets
    a malformed value reach Athena, and never lets a raw Python exception
    string leak back to the agent as the only signal of what went wrong.
    """
    def __init__(self, param: str, value, reason: str):
        self.param  = param
        self.value  = value
        self.reason = reason
        super().__init__(f"{param}={value!r}: {reason}")

    def agent_message(self) -> str:
        return (
            f"[INVALID INPUT] parameter '{self.param}' = {self.value!r}\n"
            f"Reason: {self.reason}\n"
            f"Fix the value and try again — no query was run."
        )


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_date(value: Optional[str], param: str,
                    required: bool = False) -> Optional[str]:
    """
    Strict YYYY-MM-DD validation, independent of whatever tools.py's schema
    claims (no tool schema in this codebase uses JSON Schema's "pattern"
    keyword, so nothing enforces date SHAPE before it reaches here). Once a
    value matches this anchored regex AND passes date.fromisoformat() (the
    regex alone would accept non-existent dates like 2026-13-99), it can
    only ever contain digits and hyphens — provably safe to interpolate
    directly into a SQL string literal, equivalent to parameterization for
    this exact shape.
    """
    if value is None:
        if required:
            raise ToolInputError(param, value, "required date is missing")
        return None
    if not isinstance(value, str) or not _DATE_RE.match(value):
        raise ToolInputError(param, value, "must be a date in YYYY-MM-DD format")
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ToolInputError(param, value, "not a valid calendar date")
    return value


# Covers every real example in api.py/tools.py: AAPL, BRK-B, ^GSPC, CL=F,
# DX-Y.NYB, NY.GDP.MKTP.CD, GOLDAMGBD228NLBM — letters, digits, and the
# punctuation that actually appears in real tickers/series IDs. Excludes
# quotes, semicolons, backslashes, and whitespace.
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.\-=^&]{1,48}$")


def _validate_ticker(value: str, param: str = "ticker") -> str:
    """Character-allowlist validation for ticker/series-ID-style values.
    Once a value matches this anchored regex, it cannot contain a
    quote/semicolon/backslash — safe to interpolate directly, same
    reasoning as _validate_date()."""
    if not isinstance(value, str) or not _SAFE_TOKEN_RE.match(value):
        raise ToolInputError(
            param, value,
            "must contain only letters, digits, and . _ - = ^ & characters"
        )
    return value


# `entity` (get_documents/get_prose) spans TWO genuinely different real
# shapes, confirmed via live `SELECT DISTINCT entity` against
# wikipedia_processed.documents (2026-06-28): ticker-style (AAPL, JPM —
# pure uppercase letters, already covered by _validate_ticker) AND
# Wikipedia-topic-style (lowercase, underscores, ampersand, regular
# hyphen, AND en-dash U+2013 specifically — confirmed
# "2021–2023_inflation_surge" uses U+2013, not a plain hyphen or em-dash).
# _validate_ticker's character set doesn't include en-dash, so a separate
# validator is used here rather than broadening the ticker regex for one
# unrelated shape.
_ENTITY_RE = re.compile(r"^[A-Za-z0-9_&\-–]{1,64}$")


def _validate_entity(value: str, param: str = "entity") -> str:
    if not isinstance(value, str) or not _ENTITY_RE.match(value):
        raise ToolInputError(
            param, value,
            "must contain only letters, digits, and _ & - (en-dash) characters"
        )
    return value


def _validate_limit(value, param: str = "limit", default: int = 20,
                     max_value: int = 1000) -> int:
    """limit/top_k-style values must be a real positive int within a sane
    bound — never interpolated as a raw, possibly non-numeric value into
    a LIMIT clause. Rejects bool explicitly since `isinstance(True, int)`
    is True in Python and a bool slipping through here would be a real,
    if unlikely, type-confusion bug."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolInputError(param, value, "must be an integer")
    if value < 1 or value > max_value:
        raise ToolInputError(
            param, value, f"must be between 1 and {max_value}"
        )
    return value


def _validate_number(value, param: str, min_value: float = None,
                      max_value: float = None) -> Optional[float]:
    """Numeric filter values (e.g. min_market_cap) — never interpolated as
    a raw possibly non-numeric value into a numeric comparison."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolInputError(param, value, "must be a number")
    if min_value is not None and value < min_value:
        raise ToolInputError(param, value, f"must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ToolInputError(param, value, f"must be <= {max_value}")
    return float(value)


def _validate_enum(value: Optional[str], param: str, allowed: set,
                    required: bool = False) -> Optional[str]:
    """Strict allowlist for structural/categorical values — used both for
    genuinely fixed value sets (sentiment, transaction_type) and for
    values that select a STRUCTURAL part of the SQL (database/table via a
    Python-side dict, never the user's literal string)."""
    if value is None:
        if required:
            raise ToolInputError(param, value, "required")
        return None
    if not isinstance(value, str) or value not in allowed:
        raise ToolInputError(
            param, value, f"must be one of {sorted(allowed)}"
        )
    return value


# Free-text fields with no fixed enumerable set (e.g. `industry`, which
# spans dozens of real values like "Banks—Diversified" using an em-dash,
# not a hyphen) get a broader character allowlist instead of a fixed set,
# plus defense-in-depth quote-escaping even though the allowlist already
# excludes quotes — belt-and-suspenders, costs nothing.
_FREE_TEXT_RE = re.compile(r"^[\w \t&/.,()—\-]{1,80}$")


def _validate_free_text(value: Optional[str], param: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not _FREE_TEXT_RE.match(value):
        raise ToolInputError(
            param, value,
            "contains disallowed characters (letters, digits, spaces, "
            "and & / . , ( ) - — only)"
        )
    return value


def _sql_escape(value: str) -> str:
    """Defense-in-depth single-quote escaping for free-text values that
    already passed _validate_free_text()'s character allowlist — the
    allowlist excludes quotes already, so this should never have an
    effect in practice, but doubling embedded quotes is the standard SQL
    string-literal escape and costs nothing to apply anyway."""
    return value.replace("'", "''")


# ── Structural allowlists ────────────────────────────────────────────────
# Each confirmed against real schema/documentation, not guessed — see the
# SQL-hardening session's inventory report for the specific source of each.

# Matches get_companies_in_sector's tool description, get_prices_by_sector's
# own "no results" error message, AND get_companies_in_sector's own "no
# results" error message — all three independently list the same 11 GICS
# sector names used by the `companies` table's `sector` column.
SECTOR_ALLOWLIST = {
    "Technology", "Energy", "Financials", "Health Care",
    "Consumer Discretionary", "Industrials", "Communication Services",
    "Utilities", "Real Estate", "Materials", "Consumer Staples",
}

# Matches PROJECT_OVERVIEW.md's documented market_prices schema
# (exchange STRING NYSE | NASDAQ | INDEX | FX | FUTURES | LSE | NSE) — the
# REAL 7-value set, wider than tools.py's get_prices description text
# ("NYSE, NASDAQ, LSE, NSE"), which only lists 4 as illustrative examples,
# not the complete set. Using the narrower 4-value set here would have
# rejected legitimate INDEX/FX/FUTURES exchange filters.
EXCHANGE_ALLOWLIST = {"NYSE", "NASDAQ", "INDEX", "FX", "FUTURES", "LSE", "NSE"}

# Matches get_indicator's tool schema enum exactly.
INDICATOR_SOURCE_ALLOWLIST = {"FRED", "WORLDBANK"}

# Matches get_indicator's tool description's documented World Bank country
# list (US, CN, IN, GB, DE, JP, BR) — the actual ETL-ingested set per
# PROJECT_OVERVIEW.md (5 indicators x 7 countries).
COUNTRY_ALLOWLIST = {"US", "CN", "IN", "GB", "DE", "JP", "BR"}

# Matches get_prose's tool schema enum for section_name exactly — but
# section_names (the multi-section array form) has NO enum constraint on
# its array items in tools.py today, a real schema asymmetry found during
# this audit. This allowlist closes that gap at the app level.
SECTION_NAME_ALLOWLIST = {
    "item_1", "item_1a", "item_7", "item_7a", "note_1", "note_2", "note_3",
}

# Matches get_prose's tool schema enum for form_type exactly.
FORM_TYPE_ALLOWLIST = {"10-K", "10-Q"}

# Matches get_documents' tool schema enum for doc_type exactly.
DOC_TYPE_ALLOWLIST = {"10-K", "10-Q", "wiki_article"}

# Matches get_news's tool schema enum for sentiment exactly.
SENTIMENT_ALLOWLIST = {"positive", "negative", "neutral"}

# Matches get_insider_trades' tool schema enum for transaction_type exactly.
TRANSACTION_TYPE_ALLOWLIST = {"P", "S", "A", "D", "F", "M", "X", "G", "J"}

# get_documents' `source` parameter selects which database(s) to query.
# BUG FOUND during this audit, fixed here: the tool schema's enum is
# ["EDGAR", "WIKIPEDIA"], but the old implementation did
# `DB[source.lower()]` against a dict keyed "sec"/"wikipedia" — passing
# the fully schema-valid source="EDGAR" raised an uncaught KeyError every
# time. This map is both the security allowlist AND the correct source-
# name-to-database mapping; `source`'s literal string is never used to
# build any part of the SQL or build a dict key from user input directly.
DOCUMENTS_SOURCE_DB = {
    "EDGAR":     "sec",
    "WIKIPEDIA": "wikipedia",
}


def _price_summary(df: pd.DataFrame, ticker: str) -> str:
    if df.empty:
        return f"No data for {ticker}."
    close_col = "close" if "close" in df.columns else \
                "avg_close" if "avg_close" in df.columns else \
                "week_close" if "week_close" in df.columns else \
                df.columns[-1]

    start_p = df.iloc[0][close_col]
    end_p   = df.iloc[-1][close_col]
    pct     = ((end_p - start_p) / start_p) * 100

    high_col = "high" if "high" in df.columns else \
               "week_high" if "week_high" in df.columns else close_col
    low_col  = "low" if "low" in df.columns else \
               "week_low" if "week_low" in df.columns else close_col

    return (
        f"{ticker} | "
        f"{df.iloc[0]['date'] if 'date' in df.columns else df.iloc[0].get('week_start', '')}"
        f" → "
        f"{df.iloc[-1]['date'] if 'date' in df.columns else df.iloc[-1].get('week_start', '')}\n"
        f"  Start: ${start_p:.2f}  End: ${end_p:.2f}  "
        f"Change: {pct:+.1f}%\n"
        f"  High: ${df[high_col].max():.2f}  "
        f"Low: ${df[low_col].min():.2f}"
    )


def _staleness(obs_date: str, as_of: str) -> str:
    try:
        days = (date.fromisoformat(as_of) -
                date.fromisoformat(obs_date)).days
        if days > 365:
            return f"~{days//365}yr old"
        elif days > 90:
            return f"~{days//30}mo old"
        return ""
    except ValueError:
        return ""


def _format_price_result(df: pd.DataFrame, ticker: str,
                          granularity: str) -> str:
    if df.empty:
        return f"No price data found for {ticker}."
    summary = _price_summary(df, ticker)
    rows    = len(df)
    return (
        f"{summary}\n"
        f"Granularity: {granularity} ({rows} periods)\n\n"
        f"{df.to_string(index=False)}"
    )


def _athena_error_msg(e: Exception, context: str) -> str:
    if isinstance(e, AthenaQueryError):
        return e.agent_message()
    return f"Error {context}: {e}"


# ══════════════════════════════════════════════════════════════════════════
# PRICE TOOLS
# ══════════════════════════════════════════════════════════════════════════

def get_prices(
    ticker:   str,
    start:    str,
    end:      str,
    exchange: Optional[str] = None,
    sector:   Optional[str] = None,
    industry: Optional[str] = None,
) -> str:
    try:
        ticker   = _validate_ticker(ticker.upper(), "ticker")
        start    = _validate_date(start, "start", required=True)
        end      = _validate_date(end, "end", required=True)
        exchange = _validate_enum(exchange, "exchange", EXCHANGE_ALLOWLIST)
        sector   = _validate_enum(sector, "sector", SECTOR_ALLOWLIST)
        industry = _validate_free_text(industry, "industry")
    except ToolInputError as e:
        return e.agent_message()

    granularity     = _price_granularity(start, end)
    yf              = _year_filter(start, end)
    tp              = _ticker_partition(ticker)
    ex_filter       = f"AND exchange = '{exchange}'"  if exchange  else ""
    sector_filter   = f"AND sector = '{sector}'"     if sector    else ""
    industry_filter = (f"AND industry = '{_sql_escape(industry)}'"
                        if industry else "")

    # 'ticker' is a partition column (sanitized value) — pruning happens
    # via {tp}. The original symbol lives in the 'ticker_symbol' data
    # column, recovered here via SELECT alias for display/downstream code.
    if granularity == "daily":
        sql = f"""
            SELECT ticker_symbol AS ticker, date, open, high, low,
                   close, adj_close, volume
            FROM   market_prices
            WHERE  1=1
              {tp} {yf} {ex_filter}
              {sector_filter} {industry_filter}
              AND date BETWEEN '{start}' AND '{end}'
            ORDER BY date ASC
        """
    elif granularity == "weekly":
        sql = f"""
            SELECT ticker_symbol AS ticker,
                   DATE_TRUNC('week', CAST(date AS DATE)) AS date,
                   MIN(low)            AS low,
                   MAX(high)           AS high,
                   MIN_BY(open, date)  AS open,
                   MAX_BY(close, date) AS close,
                   AVG(adj_close)      AS avg_close,
                   SUM(volume)         AS volume
            FROM   market_prices
            WHERE  1=1
              {tp} {yf} {ex_filter}
              {sector_filter} {industry_filter}
              AND date BETWEEN '{start}' AND '{end}'
            GROUP BY ticker_symbol,
                     DATE_TRUNC('week', CAST(date AS DATE))
            ORDER BY date ASC
        """
    else:  # monthly
        sql = f"""
            SELECT ticker_symbol AS ticker,
                   DATE_TRUNC('month', CAST(date AS DATE)) AS date,
                   MIN(low)            AS low,
                   MAX(high)           AS high,
                   MIN_BY(open, date)  AS open,
                   MAX_BY(close, date) AS close,
                   AVG(adj_close)      AS avg_close,
                   SUM(volume)         AS volume
            FROM   market_prices
            WHERE  1=1
              {tp} {yf} {ex_filter}
              {sector_filter} {industry_filter}
              AND date BETWEEN '{start}' AND '{end}'
            GROUP BY ticker_symbol,
                     DATE_TRUNC('month', CAST(date AS DATE))
            ORDER BY date ASC
        """

    try:
        df = query(sql, DB["yfinance"])
        return _format_price_result(df, ticker, granularity)
    except Exception as e:
        return _athena_error_msg(e, f"fetching prices for {ticker}")


def get_prices_multi(
    tickers:  list,
    start:    str,
    end:      str,
    exchange: Optional[str] = None,
    sector:   Optional[str] = None,
    industry: Optional[str] = None,
) -> str:
    try:
        tickers  = [_validate_ticker(t.upper(), "tickers") for t in tickers]
        start    = _validate_date(start, "start", required=True)
        end      = _validate_date(end, "end", required=True)
        exchange = _validate_enum(exchange, "exchange", EXCHANGE_ALLOWLIST)
        sector   = _validate_enum(sector, "sector", SECTOR_ALLOWLIST)
        industry = _validate_free_text(industry, "industry")
    except ToolInputError as e:
        return e.agent_message()

    # NOTE: ticker_list was computed here in the pre-hardening version but
    # never actually referenced anywhere in this function — dead code,
    # removed (not a security issue, just noise found during this audit).
    granularity     = _price_granularity(start, end)
    yf              = _year_filter(start, end)
    tps             = _ticker_partitions(tickers)
    ex_filter       = f"AND exchange = '{exchange}'"  if exchange  else ""
    sector_filter   = f"AND sector = '{sector}'"     if sector    else ""
    industry_filter = (f"AND industry = '{_sql_escape(industry)}'"
                        if industry else "")

    # 'ticker' is a partition column (sanitized values) — pruning happens
    # via {tps}. Original symbols live in 'ticker_symbol', recovered via
    # SELECT alias. ORDER/GROUP BY must use ticker_symbol — it's the real
    # data column; 'ticker' the partition can't be referenced post-SELECT
    # the same way in all engines, so be explicit and consistent.
    if granularity == "daily":
        sql = f"""
            SELECT ticker_symbol AS ticker, date, open, high, low,
                   close, adj_close, volume
            FROM   market_prices
            WHERE  1=1
              {tps} {yf} {ex_filter}
              {sector_filter} {industry_filter}
              AND date BETWEEN '{start}' AND '{end}'
            ORDER BY ticker_symbol ASC, date ASC
        """
    elif granularity == "weekly":
        sql = f"""
            SELECT ticker_symbol AS ticker,
                   DATE_TRUNC('week', CAST(date AS DATE)) AS date,
                   MIN(low)            AS low,
                   MAX(high)           AS high,
                   MIN_BY(open, date)  AS open,
                   MAX_BY(close, date) AS close,
                   AVG(adj_close)      AS avg_close,
                   SUM(volume)         AS volume
            FROM   market_prices
            WHERE  1=1
              {tps} {yf} {ex_filter}
              {sector_filter} {industry_filter}
              AND date BETWEEN '{start}' AND '{end}'
            GROUP BY ticker_symbol,
                     DATE_TRUNC('week', CAST(date AS DATE))
            ORDER BY ticker_symbol ASC, date ASC
        """
    else:  # monthly
        sql = f"""
            SELECT ticker_symbol AS ticker,
                   DATE_TRUNC('month', CAST(date AS DATE)) AS date,
                   MIN(low)            AS low,
                   MAX(high)           AS high,
                   MIN_BY(open, date)  AS open,
                   MAX_BY(close, date) AS close,
                   AVG(adj_close)      AS avg_close,
                   SUM(volume)         AS volume
            FROM   market_prices
            WHERE  1=1
              {tps} {yf} {ex_filter}
              {sector_filter} {industry_filter}
              AND date BETWEEN '{start}' AND '{end}'
            GROUP BY ticker_symbol,
                     DATE_TRUNC('month', CAST(date AS DATE))
            ORDER BY ticker_symbol ASC, date ASC
        """

    try:
        df = query(sql, DB["yfinance"])
        if df.empty:
            return f"No price data found for {tickers}."
        summaries = []
        for t in tickers:
            t_df = df[df["ticker"] == t]
            if not t_df.empty:
                summaries.append(_price_summary(t_df, t))
        return (
            f"Granularity: {granularity} | "
            f"Tickers: {', '.join(tickers)}\n\n"
            + "\n".join(summaries)
            + f"\n\nFull data:\n{df.to_string(index=False)}"
        )
    except Exception as e:
        return _athena_error_msg(e, f"fetching prices for {tickers}")


def get_prices_by_sector(
    sector:   str,
    start:    str,
    end:      str,
    industry: Optional[str] = None,
) -> str:
    try:
        sector   = _validate_enum(sector, "sector", SECTOR_ALLOWLIST, required=True)
        start    = _validate_date(start, "start", required=True)
        end      = _validate_date(end, "end", required=True)
        industry = _validate_free_text(industry, "industry")
    except ToolInputError as e:
        return e.agent_message()

    # Sector scan is intentional — no ticker partition filter here
    yf              = _year_filter(start, end)
    industry_filter = (f"AND industry = '{_sql_escape(industry)}'"
                        if industry else "")

    sql = f"""
        SELECT ticker_symbol AS ticker, sector, industry,
               MIN_BY(close, date)  AS start_close,
               MAX_BY(close, date)  AS end_close,
               AVG(close)           AS avg_close,
               (MAX_BY(close, date) - MIN_BY(close, date))
                 / NULLIF(MIN_BY(close, date), 0) * 100 AS pct_change
        FROM   market_prices
        WHERE  sector = '{sector}'
          {industry_filter}
          {yf}
          AND date BETWEEN '{start}' AND '{end}'
        GROUP BY ticker_symbol, sector, industry
        ORDER BY pct_change DESC
        LIMIT  20
    """

    try:
        df = query(sql, DB["yfinance"])
        if df.empty:
            return f"No price data found for sector '{sector}'."
        header = (
            f"Sector: {sector}"
            + (f" | Industry: {industry}" if industry else "")
            + f"\nPeriod: {start} → {end} | {len(df)} tickers\n"
        )
        return header + df.to_string(index=False)
    except Exception as e:
        return _athena_error_msg(e, f"fetching prices for sector '{sector}'")


def get_price_on_date(
    ticker:   str,
    date_str: str,
    exchange: Optional[str] = None
) -> str:
    try:
        ticker   = _validate_ticker(ticker.upper(), "ticker")
        date_str = _validate_date(date_str, "date_str", required=True)
        exchange = _validate_enum(exchange, "exchange", EXCHANGE_ALLOWLIST)
    except ToolInputError as e:
        return e.agent_message()

    as_of_yr  = int(date_str[:4])
    tp        = _ticker_partition(ticker)
    ex_filter = f"AND exchange = '{exchange}'" if exchange else ""

    sql = f"""
        SELECT ticker_symbol AS ticker, exchange, date, currency,
               open, high, low, close, adj_close, volume
        FROM   market_prices
        WHERE  1=1
          {tp}
          AND  CAST(year AS INTEGER) = {as_of_yr}
          AND  date <= '{date_str}'
          {ex_filter}
        ORDER BY date DESC
        LIMIT  1
    """
    try:
        df = query(sql, DB["yfinance"])
        if df.empty:
            return f"No price data found for {ticker} on or before {date_str}."
        row  = df.iloc[0]
        note = (f" (nearest trading day to {date_str})"
                if row["date"] != date_str else "")
        return (
            f"{ticker} on {row['date']}{note}\n"
            f"  Open:      ${row['open']:.2f}\n"
            f"  High:      ${row['high']:.2f}\n"
            f"  Low:       ${row['low']:.2f}\n"
            f"  Close:     ${row['close']:.2f}\n"
            f"  Adj Close: ${row['adj_close']:.2f}\n"
            f"  Volume:    {int(row['volume']):,}\n"
            f"  Currency:  {row['currency']}"
        )
    except Exception as e:
        return _athena_error_msg(e, f"fetching price for {ticker} on {date_str}")


def get_prices_on_date(
    tickers:  list,
    date_str: str,
    exchange: Optional[str] = None
) -> str:
    try:
        tickers  = [_validate_ticker(t.upper(), "tickers") for t in tickers]
        date_str = _validate_date(date_str, "date_str", required=True)
        exchange = _validate_enum(exchange, "exchange", EXCHANGE_ALLOWLIST)
    except ToolInputError as e:
        return e.agent_message()

    # NOTE: ticker_list computed but never referenced — same dead-code
    # finding as get_prices_multi, removed.
    as_of_yr    = int(date_str[:4])
    tps         = _ticker_partitions(tickers)
    ex_filter   = f"AND exchange = '{exchange}'" if exchange else ""

    sql = f"""
        SELECT ticker_symbol AS ticker, date, close, adj_close, volume, currency
        FROM (
            SELECT ticker_symbol, date, close, adj_close, volume, currency,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker_symbol
                       ORDER BY date DESC
                   ) AS rn
            FROM   market_prices
            WHERE  1=1
              {tps}
              AND  CAST(year AS INTEGER) = {as_of_yr}
              AND  date <= '{date_str}'
              {ex_filter}
        )
        WHERE rn = 1
        ORDER BY ticker_symbol ASC
    """
    try:
        df = query(sql, DB["yfinance"])
        if df.empty:
            return f"No price data for {tickers} on or before {date_str}."
        return (
            f"Prices as of {date_str} "
            f"(nearest trading day per ticker):\n\n"
            f"{df.to_string(index=False)}"
        )
    except Exception as e:
        return _athena_error_msg(e, f"fetching prices on {date_str}")


# ══════════════════════════════════════════════════════════════════════════
# INDICATOR TOOLS
# ══════════════════════════════════════════════════════════════════════════

def _indicator_db(series_id: str, source: Optional[str]) -> str:
    if source == "WORLDBANK" or (source is None and "." in series_id):
        return DB["worldbank"]
    return DB["fred"]


def _indicator_agg_sql(series_id: str, start: str, end: str,
                        granularity: str, country_filter: str,
                        db: str) -> str:
    yf = _year_filter(start, end)

    # Latest vintage per (date, country) — applied first, before any
    # date-range filtering already baked into this subquery's WHERE, and
    # before the aggregation/passthrough branches below. FRED is
    # revision-only-append (etl_fred.py never deletes old vintage files),
    # so a series can have multiple vintage rows for the same date —
    # without this, results are duplicated/double-counted.
    latest_vintage_subquery = f"""
        SELECT indicator_id, indicator_name, date, value, unit,
               frequency, country, vintage_date
        FROM (
            SELECT indicator_id, indicator_name, date, value, unit,
                   frequency, country, vintage_date,
                   ROW_NUMBER() OVER (
                       PARTITION BY indicator_id, country, date
                       ORDER BY vintage_date DESC
                   ) AS rn
            FROM economic_indicators
            WHERE indicator_id = '{series_id}'
              {yf} {country_filter}
              AND date BETWEEN '{start}' AND '{end}'
        )
        WHERE rn = 1
    """

    if granularity == "annual":
        trunc = "year"
    elif granularity == "quarterly":
        trunc = "quarter"
    elif granularity == "monthly":
        trunc = "month"
    elif granularity == "weekly":
        trunc = "week"
    else:  # daily — genuine passthrough, no aggregation
        return f"""
            SELECT indicator_id, indicator_name, date,
                   value, unit, frequency, country, vintage_date
            FROM ({latest_vintage_subquery})
            ORDER BY date ASC, country ASC
        """

    return f"""
        SELECT indicator_id,
               MIN(indicator_name) AS indicator_name,
               CAST(
                 DATE_TRUNC('{trunc}', CAST(date AS DATE))
               AS VARCHAR) AS date,
               AVG(value) AS value,
               MIN(unit)  AS unit,
               '{granularity}' AS frequency,
               country,
               MAX(vintage_date) AS vintage_date
        FROM ({latest_vintage_subquery})
        GROUP BY indicator_id, country,
                 DATE_TRUNC('{trunc}', CAST(date AS DATE))
        ORDER BY date ASC, country ASC
    """


def get_indicator(
    series_id: str,
    start:     str,
    end:       str,
    country:   Optional[str] = None,
    source:    Optional[str] = None,
    as_of:     Optional[str] = None,
) -> str:
    # NOTE: `as_of` is accepted but never referenced anywhere in this
    # function body — dead parameter, found during this audit, not a
    # security issue since an unused value can never reach SQL. Left
    # as-is rather than removing the parameter, since that's a behavior/
    # API-shape change outside this task's scope (SQL hardening only).
    try:
        series_id = _validate_ticker(series_id, "series_id")
        start     = _validate_date(start, "start", required=True)
        end       = _validate_date(end, "end", required=True)
        country   = _validate_enum(country, "country", COUNTRY_ALLOWLIST)
        source    = _validate_enum(source, "source", INDICATOR_SOURCE_ALLOWLIST)
    except ToolInputError as e:
        return e.agent_message()

    db             = _indicator_db(series_id, source)
    country_filter = f"AND country = '{country}'" if country else ""
    native_freq    = _get_native_frequency(series_id, db)
    granularity    = _indicator_granularity(start, end, native_freq)

    sql = _indicator_agg_sql(
        series_id, start, end, granularity, country_filter, db
    )

    try:
        df = query(sql, db)
        if df.empty:
            return (
                f"No data for '{series_id}' "
                f"between {start} and {end}."
            )
        header = (
            f"{df.iloc[0]['indicator_name']} ({series_id})\n"
            f"Unit: {df.iloc[0]['unit']} | "
            f"Granularity: {granularity} | "
            f"Periods: {len(df)}\n"
        )
        cols = ["date", "value", "country"] \
               if "country" in df.columns else ["date", "value"]
        return f"{header}\n{df[cols].to_string(index=False)}"
    except Exception as e:
        return _athena_error_msg(e, f"fetching indicator {series_id}")


def get_indicator_multi(
    series_ids: list,
    start:      str,
    end:        str,
    countries:  Optional[list] = None,
    source:     Optional[str]  = None,
) -> str:
    try:
        series_ids = [_validate_ticker(s, "series_ids") for s in series_ids]
        start      = _validate_date(start, "start", required=True)
        end        = _validate_date(end, "end", required=True)
        source     = _validate_enum(source, "source", INDICATOR_SOURCE_ALLOWLIST)
        if countries:
            # required=False (default) — individual list entries may
            # legitimately be None (e.g. get_macro_snapshot's DEFAULT_MACRO
            # mixes None with "US"); only non-None entries get validated.
            countries = [_validate_enum(c, "countries", COUNTRY_ALLOWLIST)
                         for c in countries]
    except ToolInputError as e:
        return e.agent_message()

    results           = []
    granularities_used = []

    for series_id in series_ids:
        db      = _indicator_db(series_id, source)
        country = countries[series_ids.index(series_id)] \
                  if countries else None
        country_filter = f"AND country = '{country}'" if country else ""
        native_freq    = _get_native_frequency(series_id, db)
        granularity    = _indicator_granularity(start, end, native_freq)
        granularities_used.append((series_id, granularity))

        sql = _indicator_agg_sql(
            series_id, start, end, granularity, country_filter, db
        )
        try:
            df = query(sql, db)
            if not df.empty:
                results.append(df)
        except Exception as e:
            results.append(pd.DataFrame([{
                "indicator_id": series_id,
                "date": "ERROR",
                "value": _athena_error_msg(e, f"fetching {series_id}"),
                "unit": "",
                "country": "",
            }]))

    if not results:
        return f"No data found for {series_ids} between {start} and {end}."

    combined = pd.concat(results).sort_values(["indicator_id", "date"])
    granularity_summary = ", ".join(
        f"{sid}={g}" for sid, g in granularities_used
    )
    return (
        f"Indicators: {series_ids}\n"
        f"Granularities: {granularity_summary} | "
        f"Period: {start} → {end}\n\n"
        f"{combined[['indicator_id','date','value','unit','country']].to_string(index=False)}"
    )


def get_indicator_on_date(
    series_ids: list,
    date_str:   str,
    countries:  Optional[list] = None,
    source:     Optional[str]  = None,
) -> str:
    try:
        series_ids = [_validate_ticker(s, "series_ids") for s in series_ids]
        date_str   = _validate_date(date_str, "date_str", required=True)
        source     = _validate_enum(source, "source", INDICATOR_SOURCE_ALLOWLIST)
        if countries:
            countries = [_validate_enum(c, "countries", COUNTRY_ALLOWLIST)
                         for c in countries]
    except ToolInputError as e:
        return e.agent_message()

    as_of_yr = int(date_str[:4])
    results  = []
    errors   = []

    for i, series_id in enumerate(series_ids):
        db      = _indicator_db(series_id, source)
        country = countries[i] if countries else None
        country_filter = f"AND country = '{country}'" if country else ""

        sql = f"""
            SELECT indicator_id, indicator_name,
                   date, value, unit, country, vintage_date
            FROM   economic_indicators
            WHERE  indicator_id = '{series_id}'
              AND  CAST(year AS INTEGER) <= {as_of_yr}
              AND  date <= '{date_str}'
              {country_filter}
            ORDER BY date DESC, vintage_date DESC
            LIMIT  1
        """
        try:
            df = query(sql, db)
            if not df.empty:
                results.append(df.iloc[0].to_dict())
        except Exception as e:
            errors.append(_athena_error_msg(e, f"fetching {series_id}"))

    if not results and errors:
        return "Errors fetching indicators:\n" + "\n".join(errors)

    lines = [f"Indicators as of {date_str}", "=" * 50]
    for r in results:
        stale     = _staleness(r["date"], date_str)
        stale_str = f" [{stale}]" if stale else ""
        lines.append(
            f"{r['indicator_name']:45s} "
            f"{r['value']:>12.2f} {r['unit']}"
            f"  (obs: {r['date']}){stale_str}"
        )
    if errors:
        lines.append("\nErrors (partial results above):")
        lines.extend(errors)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# MACRO SHORTCUT
# ══════════════════════════════════════════════════════════════════════════

def get_macro_snapshot(as_of_date: str) -> str:
    macro = get_indicator_on_date(
        [s[0] for s in DEFAULT_MACRO],
        as_of_date,
        countries=[s[2] for s in DEFAULT_MACRO],
    )
    market_signals = get_prices_on_date(
        ["SPY", "^VIX", "DX-Y.NYB", "GC=F", "CL=F"],
        as_of_date
    )
    return f"{macro}\n\nMarket signals:\n{market_signals}"


# ══════════════════════════════════════════════════════════════════════════
# DOCUMENT TOOL
# ══════════════════════════════════════════════════════════════════════════

def get_documents(
    entity:   str,
    doc_type: Optional[str] = None,
    start:    Optional[str] = None,
    end:      Optional[str] = None,
    limit:    int = 3,
    source:   Optional[str] = None
) -> str:
    try:
        if not entity:
            raise ToolInputError("entity", entity, "must be a non-empty string")
        entity   = entity.upper() if not entity[0].islower() else entity
        entity   = _validate_entity(entity, "entity")
        doc_type = _validate_enum(doc_type, "doc_type", DOC_TYPE_ALLOWLIST)
        start    = _validate_date(start, "start")
        end      = _validate_date(end, "end")
        limit    = _validate_limit(limit, "limit", default=3, max_value=50)
        source   = _validate_enum(source, "source", set(DOCUMENTS_SOURCE_DB))
    except ToolInputError as e:
        return e.agent_message()

    source_filter   = f"AND source = '{source}'"      if source   else ""
    date_filters    = ""
    year_filter     = ""

    if start and end:
        sy           = int(start[:4])
        ey           = int(end[:4])
        year_filter  = f"AND CAST(year AS INTEGER) BETWEEN {sy} AND {ey}"
        date_filters = f"AND doc_date BETWEEN '{start}' AND '{end}'"

    # BUG FIX (found during this audit, see DOCUMENTS_SOURCE_DB's comment):
    # the old `[DB[source.lower()]]` raised an uncaught KeyError for the
    # fully schema-valid source="EDGAR"/"WIKIPEDIA" — DB's keys are
    # "sec"/"wikipedia", not "edgar"/"wikipedia". DOCUMENTS_SOURCE_DB maps
    # the real schema enum values to the real DB dict keys.
    databases = (
        [DB[DOCUMENTS_SOURCE_DB[source]]] if source
        else [DB["sec"], DB["wikipedia"]]
    )

    # The `documents` table's type column is NOT consistently named across
    # databases — confirmed via live `DESCRIBE documents` (2026-06-25):
    # sec_processed has it as the `form_type` PARTITION column (etl_sec.py
    # writes the S3 path as form_type={type}/ and drops doc_type from the
    # Parquet data entirely — see its write_partition()); wikipedia_processed
    # has no such partition at all and keeps `doc_type` as a flat data
    # column instead. This is a genuine, intentional ETL-level difference
    # between the two pipelines, not an accidental crawler rename — so the
    # fix is a per-database column map, not a single corrected column name.
    TYPE_COLUMN_BY_DB = {
        DB["sec"]:       "form_type",
        DB["wikipedia"]: "doc_type",
    }

    all_results = []
    for db in databases:
        type_col        = TYPE_COLUMN_BY_DB.get(db, "doc_type")
        doc_type_filter = f"AND {type_col} = '{doc_type}'" if doc_type else ""
        sql = f"""
            SELECT doc_id, source, title, entity,
                   {type_col} as doc_type, doc_date, char_count, text
            FROM   documents
            WHERE  entity = '{entity}'
              {source_filter}
              {doc_type_filter}
              {year_filter}
              {date_filters}
            ORDER BY doc_date DESC
            LIMIT  {limit}
        """
        try:
            df = query(sql, db)
            if not df.empty:
                all_results.append(df)
        except Exception as e:
            continue

    if not all_results:
        return f"No documents found for '{entity}'."

    combined = pd.concat(all_results).sort_values(
        "doc_date", ascending=False
    ).head(limit)

    output = []
    for _, row in combined.iterrows():
        output.append(
            f"=== {row['title']} ({row['doc_date']}) ===\n"
            f"Source: {row['source']} | Type: {row['doc_type']}\n"
            f"{str(row['text'])[:2000]}"
        )
    return "\n\n".join(output)


FEDSPEAK_DOC_TYPE_ALLOWLIST = {"statement", "minutes", "transcript", "speech"}


def get_fed_communications(
    doc_type: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    entity: Optional[str] = None,
    limit: int = 5,
) -> str:
    try:
        doc_type = _validate_enum(doc_type, "doc_type", FEDSPEAK_DOC_TYPE_ALLOWLIST)
        start    = _validate_date(start, "start")
        end      = _validate_date(end, "end")
        entity   = _validate_entity(entity, "entity") if entity else None
        limit    = _validate_limit(limit, "limit", default=5, max_value=10)
    except ToolInputError as e:
        return e.agent_message()

    db      = f"{ENV}_trade_fedspeak_processed"
    table   = "documents"
    filters = ["source = 'FEDSPEAK'"]

    if doc_type:
        filters.append(f"doc_type = '{doc_type}'")
    if entity:
        filters.append(f"entity = '{entity}'")
    if start:
        filters.append(f"doc_date >= '{start}'")
    if end:
        filters.append(f"doc_date <= '{end}'")

    where = " AND ".join(filters)
    sql = f"""
        SELECT doc_id, entity, doc_type, doc_date, title,
               SUBSTR(text, 1, 8000) AS text_preview,
               char_count, url
        FROM {db}.{table}
        WHERE {where}
        ORDER BY doc_date DESC
        LIMIT {limit}
    """

    try:
        df = query(sql, db)
    except Exception as e:
        return f"Error querying FedSpeak: {e}"

    if df.empty:
        return (
            f"No Fed communications found"
            + (f" of type '{doc_type}'" if doc_type else "")
            + (f" from {start}" if start else "")
            + (f" to {end}" if end else "")
            + ". The FedSpeak pipeline may not have run yet."
        )

    results = []
    for _, row in df.iterrows():
        results.append(
            f"[{row['doc_type'].upper()}] {row['title']}\n"
            f"Date: {row['doc_date']} | Entity: {row['entity']} "
            f"| {row['char_count']:,} chars\n"
            f"URL: {row.get('url', 'N/A')}\n\n"
            f"{row['text_preview']}\n"
            f"{'─' * 60}"
        )

    return (
        f"Found {len(df)} Fed communication(s):\n\n"
        + "\n\n".join(results)
    )


# ══════════════════════════════════════════════════════════════════════════
# PROSE TOOL
# ══════════════════════════════════════════════════════════════════════════

def get_prose(
    entity:        str,
    section_name:  Optional[str]  = None,
    section_names: Optional[list] = None,
    form_type:     Optional[str]  = None,
    start:         Optional[str]  = None,
    end:           Optional[str]  = None,
    limit:         int  = 3,
    max_chars:     int  = PROSE_DEFAULT_CHARS,
) -> str:
    try:
        if not entity:
            raise ToolInputError("entity", entity, "must be a non-empty string")
        entity = _validate_entity(entity.upper(), "entity")
        section_name = _validate_enum(
            section_name, "section_name", SECTION_NAME_ALLOWLIST
        )
        if section_names:
            # BUG FOUND during this audit: tools.py's schema enum
            # constrains the single-section `section_name` field but NOT
            # `section_names`' array items — a real asymmetry. This
            # allowlist closes that gap at the app level.
            section_names = [
                _validate_enum(s, "section_names", SECTION_NAME_ALLOWLIST,
                               required=True)
                for s in section_names
            ]
        form_type = _validate_enum(form_type, "form_type", FORM_TYPE_ALLOWLIST)
        start     = _validate_date(start, "start")
        end       = _validate_date(end, "end")
        limit     = _validate_limit(limit, "limit", default=3, max_value=50)
        max_chars = _validate_limit(
            max_chars, "max_chars", default=PROSE_DEFAULT_CHARS,
            max_value=PROSE_MAX_CHARS,
        )
    except ToolInputError as e:
        return e.agent_message()

    if section_names:
        section_list   = "','".join(section_names)
        section_filter = f"AND section_name IN ('{section_list}')"
    elif section_name:
        section_filter = f"AND section_name = '{section_name}'"
    else:
        section_filter = ""

    form_filter = (f"AND form_type = '{form_type.replace('-', '')}'"
                   if form_type else "")
    year_filter = ""
    date_filter = ""

    if start and end:
        sy          = int(start[:4])
        ey          = int(end[:4])
        year_filter = f"AND CAST(year AS INTEGER) BETWEEN {sy} AND {ey}"
        date_filter = f"AND filed_date BETWEEN '{start}' AND '{end}'"

    row_limit = limit * len(section_names) if section_names else limit

    sql = f"""
        SELECT doc_id, entity, form_type, filed_date,
               section_name, section_title,
               text, char_count, extraction_method
        FROM   documents_prose
        WHERE  entity = '{entity}'
          {section_filter}
          {form_filter}
          {year_filter}
          {date_filter}
        ORDER BY filed_date DESC, section_name ASC
        LIMIT  {row_limit}
    """

    try:
        df = query(sql, DB["sec_prose"])
        if df.empty:
            return f"No prose sections found for '{entity}'."

        output = []
        for _, row in df.iterrows():
            text_preview = str(row["text"])[:max_chars]
            truncated    = len(str(row["text"])) > max_chars
            suffix       = (
                f"\n[... truncated at {max_chars} chars. "
                f"Pass max_chars=20000 to get full section ...]"
                if truncated else ""
            )
            output.append(
                f"=== {row['section_title']} "
                f"({row['form_type']} | {row['filed_date']}) ===\n"
                f"Entity: {row['entity']} | "
                f"Section: {row['section_name']} | "
                f"Extraction: {row['extraction_method']}\n\n"
                f"{text_preview}{suffix}"
            )
        return "\n\n".join(output)

    except Exception as e:
        return _athena_error_msg(e, f"fetching prose for {entity}")


# ══════════════════════════════════════════════════════════════════════════
# SEMANTIC SEARCH
# ══════════════════════════════════════════════════════════════════════════

def semantic_search(
    query:  str,
    top_k:  int = 5,
    source: Optional[str] = None,
    entity: Optional[str] = None,
) -> str:
    try:
        body = json.dumps({
            "texts":           [query[:2000]],
            "input_type":      "search_query",
            "embedding_types": ["float"],
        })
        resp   = _bedrock.invoke_model(
            modelId=EMBED_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        vector = json.loads(resp["body"].read())["embeddings"]["float"][0]
    except Exception as e:
        return f"Embedding error: {e}"

    fetch_k = top_k * 3 if (source or entity) else top_k
    try:
        result  = _s3vectors.query_vectors(
            vectorBucketName=VECTOR_BUCKET,
            indexName=VECTOR_INDEX,
            queryVector={"float32": vector},
            topK=fetch_k,
            returnMetadata=True,
            returnDistance=True,
        )
        matches = result.get("vectors", [])
    except Exception as e:
        if "empty" in str(e).lower() or \
           "ResourceNotFoundException" in str(e):
            return (
                "Vector index is not yet populated. "
                "Run etl_embed.py to embed documents first."
            )
        return f"Vector search error: {e}"

    if source:
        matches = [m for m in matches
                   if m.get("metadata", {}).get("source", "").upper()
                   == source.upper()]
    if entity:
        matches = [m for m in matches
                   if m.get("metadata", {}).get("entity", "").upper()
                   == entity.upper()]

    matches = matches[:top_k]

    if not matches:
        return f"No relevant documents found for: {query}"

    lines = [
        f"Semantic search: '{query}'",
        f"Top {len(matches)} results:\n"
    ]
    for i, match in enumerate(matches, 1):
        meta     = match.get("metadata", {})
        distance = match.get("distance", 0)
        lines.append(
            f"[{i}] {meta.get('title', 'Unknown')} "
            f"({meta.get('source', '')} | {meta.get('doc_date', '')})\n"
            f"    Entity: {meta.get('entity', '')} | "
            f"Distance: {distance:.4f} (lower=more similar)\n"
            f"    {meta.get('text', '')[:300]}\n"
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# NEWS / SENTIMENT
# ══════════════════════════════════════════════════════════════════════════

def get_news(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    sentiment: Optional[str] = None,
    publisher_tier: Optional[int] = None,
    limit: int = 10,
) -> str:
    try:
        ticker         = _validate_ticker(ticker, "ticker")
        start          = _validate_date(start, "start")
        end            = _validate_date(end, "end")
        sentiment      = _validate_enum(sentiment, "sentiment", SENTIMENT_ALLOWLIST)
        publisher_tier = _validate_limit(
            publisher_tier, "publisher_tier", default=None, max_value=3
        )
        limit          = _validate_limit(limit, "limit", default=10, max_value=100)
    except ToolInputError as e:
        return e.agent_message()

    filters = [f"primary_ticker = '{ticker}'"]

    if start:
        filters.append(f"published_at >= '{start}'")
    if end:
        filters.append(f"published_at <= '{end}T23:59:59Z'")
    if sentiment:
        filters.append(f"sentiment = '{sentiment}'")
    if publisher_tier:
        filters.append(f"publisher_tier <= {publisher_tier}")

    where = " AND ".join(filters)
    sql = f"""
        SELECT headline, description, publisher, publisher_tier,
               sentiment, sentiment_reasoning, published_at,
               article_url, keywords
        FROM news
        WHERE {where}
        ORDER BY published_at DESC
        LIMIT {limit}
    """

    try:
        df = query(sql, DB["news"])
    except Exception as e:
        return f"Error querying news: {e}"

    if df.empty:
        return (
            f"No news found for {ticker}"
            + (f" from {start}" if start else "")
            + (f" to {end}" if end else "")
            + (f" with sentiment={sentiment}" if sentiment else "")
            + ". News pipeline may not have run yet."
        )

    results = []
    for _, row in df.iterrows():
        tier_label = {1: "Tier-1 (Wire)", 2: "Tier-2", 3: "Tier-3 (Opinion)"}.get(
            int(row.get("publisher_tier", 3)), "Unknown"
        )
        results.append(
            f"[{row['sentiment'].upper()}] {row['headline']}\n"
            f"Publisher: {row['publisher']} ({tier_label}) | "
            f"Date: {row['published_at'][:10]}\n"
            f"Summary: {row['description']}\n"
            f"Sentiment reasoning: {row['sentiment_reasoning']}\n"
            f"URL: {row['article_url']}\n"
            f"{'─' * 60}"
        )

    return (
        f"News for {ticker} ({len(df)} articles):\n\n"
        + "\n\n".join(results)
    )


def get_news_summary(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    publisher_tier: Optional[int] = None,
) -> str:
    try:
        ticker         = _validate_ticker(ticker, "ticker")
        start          = _validate_date(start, "start")
        end            = _validate_date(end, "end")
        publisher_tier = _validate_limit(
            publisher_tier, "publisher_tier", default=None, max_value=3
        )
    except ToolInputError as e:
        return e.agent_message()

    filters = [f"primary_ticker = '{ticker}'"]

    if start:
        filters.append(f"published_at >= '{start}'")
    if end:
        filters.append(f"published_at <= '{end}T23:59:59Z'")
    if publisher_tier:
        filters.append(f"publisher_tier <= {publisher_tier}")

    where = " AND ".join(filters)

    sql_counts = f"""
        SELECT sentiment, COUNT(*) as cnt
        FROM news
        WHERE {where}
        GROUP BY sentiment
        ORDER BY cnt DESC
    """
    sql_publishers = f"""
        SELECT publisher, COUNT(*) as cnt
        FROM news
        WHERE {where}
        GROUP BY publisher
        ORDER BY cnt DESC
        LIMIT 5
    """

    try:
        df_counts     = query(sql_counts, DB["news"])
        df_publishers = query(sql_publishers, DB["news"])
    except Exception as e:
        return f"Error querying news summary: {e}"

    if df_counts.empty:
        return f"No news found for {ticker}. News pipeline may not have run yet."

    total = df_counts["cnt"].astype(int).sum()
    sentiment_breakdown = " | ".join(
        f"{row['sentiment']}: {row['cnt']}"
        for _, row in df_counts.iterrows()
    )
    top_publishers = ", ".join(
        f"{row['publisher']} ({row['cnt']})"
        for _, row in df_publishers.iterrows()
    )

    period = ""
    if start or end:
        period = f" ({start or 'start'} to {end or 'today'})"

    return (
        f"News summary for {ticker}{period}:\n"
        f"Total articles: {total}\n"
        f"Sentiment: {sentiment_breakdown}\n"
        f"Top publishers: {top_publishers}"
    )


# ══════════════════════════════════════════════════════════════════════════
# INSIDER TRADES
# ══════════════════════════════════════════════════════════════════════════

def get_insider_trades(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    transaction_type: Optional[str] = None,
    limit: int = 20,
) -> str:
    try:
        ticker           = _validate_ticker(ticker, "ticker")
        start            = _validate_date(start, "start")
        end              = _validate_date(end, "end")
        transaction_type = _validate_enum(
            transaction_type, "transaction_type", TRANSACTION_TYPE_ALLOWLIST
        )
        limit            = _validate_limit(limit, "limit", default=20, max_value=200)
    except ToolInputError as e:
        return e.agent_message()

    filters = [f"ticker = '{ticker}'"]

    if start:
        filters.append(f"transaction_date >= '{start}'")
    if end:
        filters.append(f"transaction_date <= '{end}'")
    if transaction_type:
        filters.append(f"transaction_type = '{transaction_type}'")

    where = " AND ".join(filters)
    sql = f"""
        SELECT filer_name, filer_role, transaction_date,
               transaction_type, shares, price_per_share,
               value_usd, ownership_type, shares_owned_after
        FROM insider_trades
        WHERE {where}
        ORDER BY transaction_date DESC
        LIMIT {limit}
    """

    try:
        df = query(sql, DB["insiders"])
    except Exception as e:
        return f"Error querying insider trades: {e}"

    if df.empty:
        return (
            f"No insider trades found for {ticker}"
            + (f" from {start}" if start else "")
            + (f" to {end}" if end else "")
            + ". Insider trades pipeline may not have run yet."
        )

    type_labels = {
        "P": "Purchase", "S": "Sale", "A": "Award",
        "D": "Disposition", "F": "Tax withholding",
        "M": "Option exercise", "G": "Gift",
    }

    results = []
    for _, row in df.iterrows():
        txn_label = type_labels.get(
            str(row["transaction_type"]), str(row["transaction_type"])
        )
        value = float(row["value_usd"] or 0)
        value_str = f"${value:,.0f}" if value > 0 else "N/A"
        results.append(
            f"[{txn_label}] {row['filer_name']} ({row['filer_role']})\n"
            f"Date: {row['transaction_date']} | "
            f"Shares: {float(row['shares'] or 0):,.0f} @ "
            f"${float(row['price_per_share'] or 0):.2f} = {value_str}\n"
            f"Ownership: {'Direct' if row['ownership_type'] == 'D' else 'Indirect'} | "
            f"Shares after: {float(row['shares_owned_after'] or 0):,.0f}\n"
            f"{'─' * 60}"
        )

    return (
        f"Insider trades for {ticker} ({len(df)} transactions):\n\n"
        + "\n\n".join(results)
    )


def get_insider_summary(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> str:
    try:
        ticker = _validate_ticker(ticker, "ticker")
        start  = _validate_date(start, "start")
        end    = _validate_date(end, "end")
    except ToolInputError as e:
        return e.agent_message()

    filters = [f"ticker = '{ticker}'"]
    if start:
        filters.append(f"transaction_date >= '{start}'")
    if end:
        filters.append(f"transaction_date <= '{end}'")

    where = " AND ".join(filters)
    sql = f"""
        SELECT
            transaction_type,
            COUNT(*) as transaction_count,
            SUM(shares) as total_shares,
            SUM(value_usd) as total_value_usd,
            COUNT(DISTINCT filer_name) as unique_insiders
        FROM insider_trades
        WHERE {where}
        GROUP BY transaction_type
        ORDER BY total_value_usd DESC
    """

    try:
        df = query(sql, DB["insiders"])
    except Exception as e:
        return f"Error querying insider summary: {e}"

    if df.empty:
        return f"No insider trades found for {ticker}. Pipeline may not have run yet."

    type_labels = {
        "P": "Open-market purchases (bullish — discretionary buy)",
        "S": "Open-market sales (may be 10b5-1 pre-planned program)",
        "F": "Tax withholding on RSU/PSU vesting (NOT a sell signal — automatic)",
        "A": "Awards / grants",
        "D": "Dispositions (transfer to trust/charity, not open-market)",
        "M": "Option exercises",
        "G": "Gifts",
        "J": "Other acquisitions/dispositions",
        "X": "Option exercises (in-the-money)",
    }

    lines = [f"Insider trading summary for {ticker}:"]
    if start or end:
        period = f"{start or 'start'} → {end or 'present'}"
        lines[0] += f" ({period})"

    total_buy_value  = 0.0
    total_sell_value = 0.0
    f_value          = 0.0

    for _, row in df.iterrows():
        txn_type = str(row["transaction_type"])
        label    = type_labels.get(txn_type, f"Type {txn_type}")
        value    = float(row["total_value_usd"] or 0)
        shares   = float(row["total_shares"] or 0)
        count    = int(row["transaction_count"])
        insiders = int(row["unique_insiders"])

        lines.append(
            f"  [{txn_type}] {label}\n"
            f"      {count} transactions by {insiders} insiders | "
            f"{shares:,.0f} shares | ${value:,.0f}"
        )

        if txn_type == "P":
            total_buy_value  += value
        elif txn_type == "S":
            total_sell_value += value
        elif txn_type == "F":
            f_value          += value

    lines.append("")
    if total_buy_value > 0 or total_sell_value > 0:
        net    = total_buy_value - total_sell_value
        signal = "NET BUYING" if net > 0 else "NET SELLING"
        lines.append(
            f"  {signal} (open-market only): ${abs(net):,.0f} net\n"
            f"    Purchases: ${total_buy_value:,.0f} | "
            f"Sales: ${total_sell_value:,.0f}"
        )
    else:
        lines.append("  No open-market purchases or sales in this period.")

    if f_value > 0:
        lines.append(
            f"\n  Note: ${f_value:,.0f} in type-F tax withholding transactions "
            f"excluded from net signal above. These are automatic share "
            f"surrenders when RSUs vest — not discretionary sell decisions."
        )

    lines.append(
        "\n  Interpretation note: Most executive open-market sales are executed "
        "under pre-planned Rule 10b5-1 programs set up months in advance. "
        "They do not necessarily reflect the insider's view of near-term "
        "stock performance. Purchases are more reliably bullish signals "
        "as they are typically discretionary."
    )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# COMPANIES REFERENCE TABLE
# ══════════════════════════════════════════════════════════════════════════

def get_companies_in_sector(
    sector: str,
    industry: Optional[str] = None,
    min_market_cap: Optional[float] = None,
    sp500_only: bool = False,
) -> str:
    try:
        sector         = _validate_enum(sector, "sector", SECTOR_ALLOWLIST,
                                         required=True)
        industry       = _validate_free_text(industry, "industry")
        min_market_cap = _validate_number(min_market_cap, "min_market_cap",
                                           min_value=0)
    except ToolInputError as e:
        return e.agent_message()

    db    = DB["yfinance"]
    table = "companies"

    filters = [f"sector = '{sector}'"]
    if industry:
        filters.append(f"industry = '{_sql_escape(industry)}'")
    if min_market_cap:
        filters.append(f"market_cap >= {min_market_cap}")
    if sp500_only:
        filters.append("sp500 = true")

    where = " AND ".join(filters)
    sql = f"""
        SELECT ticker, company_name, sector, industry,
               exchange, city, state, market_cap,
               beta, dividend_yield, pe_ratio,
               week52_high, week52_low,
               avg_volume_3m, sp500
        FROM {table}
        WHERE {where}
        ORDER BY market_cap DESC NULLS LAST
    """

    try:
        df = query(sql, db)
    except Exception as e:
        return _athena_error_msg(e, f"fetching companies in sector '{sector}'")

    if df.empty:
        return (
            f"No companies found in sector '{sector}'"
            + (f" / industry '{industry}'" if industry else "")
            + ". The companies table may not have run yet, or the sector "
            + "name may not match exactly. Try: Technology, Energy, "
            + "Financials, Health Care, Consumer Discretionary, Industrials, "
            + "Communication Services, Utilities, Real Estate, Materials, "
            + "Consumer Staples."
        )

    lines = [
        f"Companies in {sector}"
        + (f" / {industry}" if industry else "")
        + f" ({len(df)} tracked):\n"
    ]
    for _, row in df.iterrows():
        mcap = float(row.get("market_cap") or 0)
        mcap_str = (
            f"${mcap/1e12:.1f}T" if mcap >= 1e12
            else f"${mcap/1e9:.0f}B" if mcap >= 1e9
            else f"${mcap/1e6:.0f}M" if mcap >= 1e6
            else "N/A"
        )
        beta     = row.get("beta")
        beta_str = f"β{float(beta):.2f}" if beta else ""
        dy       = row.get("dividend_yield")
        dy_str   = f"yield {float(dy)*100:.1f}%" if dy else ""
        location = ""
        if row.get("city") and row.get("state"):
            location = f"{row['city']}, {row['state']}"
        elif row.get("city"):
            location = str(row["city"])

        meta = " | ".join(filter(None, [mcap_str, beta_str, dy_str, location]))
        lines.append(
            f"  {row['ticker']:6s} {str(row.get('company_name','')):<35s} "
            f"{str(row.get('industry','')):<30s} {meta}"
        )

    lines.append(
        f"\nTickers: {', '.join(df['ticker'].tolist())}"
    )
    return "\n".join(lines)