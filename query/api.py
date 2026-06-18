"""
Business logic — full tool suite for the economic intelligence platform.

Tools:
  PRICE:      get_prices, get_prices_multi, get_price_on_date, get_prices_on_date
  INDICATOR:  get_indicator, get_indicator_multi, get_indicator_on_date
  DOCUMENT:   get_documents, get_prose
  MACRO:      get_macro_snapshot (wrapper around get_indicator_on_date + SPY)
  SEARCH:     semantic_search

Phase 8 changes:
  - get_prose: added max_chars parameter (CO-2) — default 8000, agent can
    request up to 20000 for deep dives
  - get_prose: added section_names list parameter (LA-3) — fetch multiple
    sections in one Athena query instead of one call per section
  - All AthenaError catches now use AthenaQueryError.agent_message() so
    the agent receives structured, actionable error context
"""

import json
import os
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


def _year_filter(start: str, end: str) -> str:
    sy = int(start[:4])
    ey = int(end[:4])
    return f"AND CAST(year AS INTEGER) BETWEEN {sy} AND {ey}"


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
    """
    Return a structured error message for the agent.
    Uses AthenaQueryError.agent_message() for structured errors,
    falls back to generic string for unexpected exceptions.
    """
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
    ticker          = ticker.upper()
    granularity     = _price_granularity(start, end)
    yf              = _year_filter(start, end)
    ex_filter       = f"AND exchange = '{exchange}'"  if exchange  else ""
    sector_filter   = f"AND sector = '{sector}'"     if sector    else ""
    industry_filter = f"AND industry = '{industry}'" if industry  else ""

    if granularity == "daily":
        sql = f"""
            SELECT ticker, date, open, high, low,
                   close, adj_close, volume
            FROM   market_prices
            WHERE  ticker = '{ticker}'
              {yf} {ex_filter}
              {sector_filter} {industry_filter}
              AND date BETWEEN '{start}' AND '{end}'
            ORDER BY date ASC
        """
    elif granularity == "weekly":
        sql = f"""
            SELECT ticker,
                   DATE_TRUNC('week', CAST(date AS DATE)) AS date,
                   MIN(low)            AS low,
                   MAX(high)           AS high,
                   MIN_BY(open, date)  AS open,
                   MAX_BY(close, date) AS close,
                   AVG(adj_close)      AS avg_close,
                   SUM(volume)         AS volume
            FROM   market_prices
            WHERE  ticker = '{ticker}'
              {yf} {ex_filter}
              {sector_filter} {industry_filter}
              AND date BETWEEN '{start}' AND '{end}'
            GROUP BY ticker,
                     DATE_TRUNC('week', CAST(date AS DATE))
            ORDER BY date ASC
        """
    else:  # monthly
        sql = f"""
            SELECT ticker,
                   DATE_TRUNC('month', CAST(date AS DATE)) AS date,
                   MIN(low)            AS low,
                   MAX(high)           AS high,
                   MIN_BY(open, date)  AS open,
                   MAX_BY(close, date) AS close,
                   AVG(adj_close)      AS avg_close,
                   SUM(volume)         AS volume
            FROM   market_prices
            WHERE  ticker = '{ticker}'
              {yf} {ex_filter}
              {sector_filter} {industry_filter}
              AND date BETWEEN '{start}' AND '{end}'
            GROUP BY ticker,
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
    tickers         = [t.upper() for t in tickers]
    ticker_list     = "','".join(tickers)
    granularity     = _price_granularity(start, end)
    yf              = _year_filter(start, end)
    ex_filter       = f"AND exchange = '{exchange}'"  if exchange  else ""
    sector_filter   = f"AND sector = '{sector}'"     if sector    else ""
    industry_filter = f"AND industry = '{industry}'" if industry  else ""

    if granularity == "daily":
        sql = f"""
            SELECT ticker, date, open, high, low,
                   close, adj_close, volume
            FROM   market_prices
            WHERE  ticker IN ('{ticker_list}')
              {yf} {ex_filter}
              {sector_filter} {industry_filter}
              AND date BETWEEN '{start}' AND '{end}'
            ORDER BY ticker ASC, date ASC
        """
    elif granularity == "weekly":
        sql = f"""
            SELECT ticker,
                   DATE_TRUNC('week', CAST(date AS DATE)) AS date,
                   MIN(low)            AS low,
                   MAX(high)           AS high,
                   MIN_BY(open, date)  AS open,
                   MAX_BY(close, date) AS close,
                   AVG(adj_close)      AS avg_close,
                   SUM(volume)         AS volume
            FROM   market_prices
            WHERE  ticker IN ('{ticker_list}')
              {yf} {ex_filter}
              {sector_filter} {industry_filter}
              AND date BETWEEN '{start}' AND '{end}'
            GROUP BY ticker,
                     DATE_TRUNC('week', CAST(date AS DATE))
            ORDER BY ticker ASC, date ASC
        """
    else:  # monthly
        sql = f"""
            SELECT ticker,
                   DATE_TRUNC('month', CAST(date AS DATE)) AS date,
                   MIN(low)            AS low,
                   MAX(high)           AS high,
                   MIN_BY(open, date)  AS open,
                   MAX_BY(close, date) AS close,
                   AVG(adj_close)      AS avg_close,
                   SUM(volume)         AS volume
            FROM   market_prices
            WHERE  ticker IN ('{ticker_list}')
              {yf} {ex_filter}
              {sector_filter} {industry_filter}
              AND date BETWEEN '{start}' AND '{end}'
            GROUP BY ticker,
                     DATE_TRUNC('month', CAST(date AS DATE))
            ORDER BY ticker ASC, date ASC
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
    yf              = _year_filter(start, end)
    industry_filter = f"AND industry = '{industry}'" if industry else ""

    sql = f"""
        SELECT ticker, sector, industry,
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
        GROUP BY ticker, sector, industry
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
    ticker    = ticker.upper()
    as_of_yr  = int(date_str[:4])
    ex_filter = f"AND exchange = '{exchange}'" if exchange else ""

    sql = f"""
        SELECT ticker, exchange, date, currency,
               open, high, low, close, adj_close, volume
        FROM   market_prices
        WHERE  ticker = '{ticker}'
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
    tickers     = [t.upper() for t in tickers]
    ticker_list = "','".join(tickers)
    as_of_yr    = int(date_str[:4])
    ex_filter   = f"AND exchange = '{exchange}'" if exchange else ""

    sql = f"""
        SELECT ticker, date, close, adj_close, volume, currency
        FROM (
            SELECT ticker, date, close, adj_close, volume, currency,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker
                       ORDER BY date DESC
                   ) AS rn
            FROM   market_prices
            WHERE  ticker IN ('{ticker_list}')
              AND  CAST(year AS INTEGER) = {as_of_yr}
              AND  date <= '{date_str}'
              {ex_filter}
        )
        WHERE rn = 1
        ORDER BY ticker ASC
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

    if granularity == "annual":
        trunc = "year"
    elif granularity == "quarterly":
        trunc = "quarter"
    elif granularity == "monthly":
        trunc = "month"
    else:
        return f"""
            SELECT indicator_id, indicator_name, date,
                   value, unit, frequency, country, vintage_date
            FROM   economic_indicators
            WHERE  indicator_id = '{series_id}'
              {yf} {country_filter}
              AND date BETWEEN '{start}' AND '{end}'
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
        FROM   economic_indicators
        WHERE  indicator_id = '{series_id}'
          {yf} {country_filter}
          AND date BETWEEN '{start}' AND '{end}'
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
    db             = _indicator_db(series_id, source)
    country_filter = f"AND country = '{country}'" if country else ""
    native_freq    = "monthly"
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
    granularity = _indicator_granularity(start, end)
    results     = []

    for series_id in series_ids:
        db      = _indicator_db(series_id, source)
        country = countries[series_ids.index(series_id)] \
                  if countries else None
        country_filter = f"AND country = '{country}'" if country else ""

        sql = _indicator_agg_sql(
            series_id, start, end, granularity, country_filter, db
        )
        try:
            df = query(sql, db)
            if not df.empty:
                results.append(df)
        except Exception as e:
            # Include per-series errors in output so agent knows what failed
            results_note = _athena_error_msg(e, f"fetching {series_id}")
            results.append(pd.DataFrame([{
                "indicator_id": series_id,
                "date": "ERROR",
                "value": results_note,
                "unit": "",
                "country": "",
            }]))

    if not results:
        return f"No data found for {series_ids} between {start} and {end}."

    combined = pd.concat(results).sort_values(["indicator_id", "date"])
    return (
        f"Indicators: {series_ids}\n"
        f"Granularity: {granularity} | "
        f"Period: {start} → {end}\n\n"
        f"{combined[['indicator_id','date','value','unit','country']].to_string(index=False)}"
    )


def get_indicator_on_date(
    series_ids: list,
    date_str:   str,
    countries:  Optional[list] = None,
    source:     Optional[str]  = None,
) -> str:
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
    entity          = entity.upper() if not entity[0].islower() else entity
    source_filter   = f"AND source = '{source}'"      if source   else ""
    doc_type_filter = f"AND form_type = '{doc_type}'" if doc_type else ""
    date_filters    = ""
    year_filter     = ""

    if start and end:
        sy           = int(start[:4])
        ey           = int(end[:4])
        year_filter  = f"AND CAST(year AS INTEGER) BETWEEN {sy} AND {ey}"
        date_filters = f"AND doc_date BETWEEN '{start}' AND '{end}'"

    databases = (
        [DB[source.lower()]] if source
        else [DB["sec"], DB["wikipedia"]]
    )

    all_results = []
    for db in databases:
        sql = f"""
            SELECT doc_id, source, title, entity,
                   form_type as doc_type, doc_date, char_count, text
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
            # Non-fatal — try next database
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

def get_fed_communications(
    doc_type: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    entity: Optional[str] = None,
    limit: int = 5,
) -> str:
    """
    Retrieve FOMC statements, minutes, transcripts, or Fed governor speeches.
 
    Args:
        doc_type: "statement" | "minutes" | "transcript" | "speech" | None (all)
        start:    YYYY-MM-DD start date filter
        end:      YYYY-MM-DD end date filter
        entity:   "FOMC" for committee docs, or speaker name for speeches
        limit:    max number of documents to return (default 5)
    """
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
    """
    Fetch prose sections from 10-K/10-Q filings.

    section_name  — single section (backwards compatible)
    section_names — list of sections fetched in ONE Athena query (LA-3)
                    e.g. ["item_1", "item_1a", "item_7"]
                    When both are provided, section_names takes precedence.
    max_chars     — chars returned per section (CO-2)
                    default: 8000. Pass max_chars=20000 for deep dives.

    section_name options:
      item_1   — Business description
      item_1a  — Risk factors
      item_7   — MD&A
      item_7a  — Market risk
      note_1   — Accounting policies
      note_2   — Revenue segments
      note_3   — Debt details
    """
    entity    = entity.upper()
    max_chars = min(max_chars, PROSE_MAX_CHARS)  # hard cap at 20k

    # section_names list takes precedence over single section_name
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

    # When fetching multiple sections, increase the row limit proportionally
    # so we get `limit` filings worth of each section
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
    """
    Semantic search over SEC filings, Wikipedia articles, and FedSpeak documents.
    Embeds query with Cohere v3, queries S3 Vectors index, returns top-K chunks.
    source filter accepts: EDGAR, WIKIPEDIA, FEDSPEAK (or None for all).
    """
    # 1. Embed the query
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

    # 2. Query S3 Vectors
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

    # 3. Filter in Python
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

    # 4. Format results
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


# ══════════════════════════════════════════════════════════════════════════════
# NEWS / SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════

def get_news(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    sentiment: Optional[str] = None,
    publisher_tier: Optional[int] = None,
    limit: int = 10,
) -> str:
    """
    Retrieve news articles for a ticker with per-article sentiment and reasoning.
    publisher_tier: 1=wire, 2=established, 3=opinion (filter is <=tier).
    """
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
    """
    Aggregated sentiment summary for a ticker: counts by sentiment, top publishers.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# INSIDER TRADES
# ══════════════════════════════════════════════════════════════════════════════

def get_insider_trades(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    transaction_type: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    Retrieve SEC Form 4 insider trades for a ticker.
    transaction_type: P=purchase, S=sale, A=award, D=disposition, F=tax withholding.
    """
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
    """
    Aggregated insider trading signal for a ticker — net buying vs selling.
    F (tax withholding) is reported separately and excluded from the net signal.
    """
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

    # Net signal — P vs S only; F excluded because it is automatic tax withholding
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


# ══════════════════════════════════════════════════════════════════════════════
# COMPANIES REFERENCE TABLE
# ══════════════════════════════════════════════════════════════════════════════

def get_companies_in_sector(
    sector: str,
    industry: Optional[str] = None,
    min_market_cap: Optional[float] = None,
    sp500_only: bool = False,
) -> str:
    """
    Return companies in a sector/industry from the reference table.
    Used as step 1 before get_prices_multi or get_insider_summary to
    discover which tickers to query.
    """
    db    = DB["yfinance"]
    table = "companies"

    filters = [f"sector = '{sector}'"]
    if industry:
        filters.append(f"industry = '{industry}'")
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