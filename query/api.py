"""
Business logic — full tool suite for the economic intelligence platform.

Tools:
  PRICE:      get_prices, get_prices_multi, get_price_on_date, get_prices_on_date
  INDICATOR:  get_indicator, get_indicator_multi, get_indicator_on_date
  DOCUMENT:   get_documents
  MACRO:      get_macro_snapshot (wrapper around get_indicator_on_date + SPY)
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
from query.athena import query, AthenaError

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
    ("^VIX", "yfinance_index", None),  # Fear gauge
    ("DX-Y.NYB", "yfinance_fx", None), # Dollar strength
    ("GC=F", "yfinance_futures", None), # Gold
]


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

import time

def embed_chunks(chunks: list) -> list:
    embedded = []
    total    = len(chunks)

    for i, chunk in enumerate(chunks):
        vector = embed_text(chunk["text"])
        if vector is not None:
            chunk["vector"] = vector
            embedded.append(chunk)

        # Rate limiting — 2 req/sec safely under Titan default quota
        time.sleep(0.5)

        if (i + 1) % 25 == 0 or (i + 1) == total:
            pct = int((i + 1) / total * 100)
            print(f"  Embedded {i+1}/{total} chunks ({pct}%)")

    return embedded

def _indicator_granularity(start: str, end: str,
                            native_freq: str = "monthly") -> str:
    """
    Auto-select aggregation granularity for indicator data.
    Respects native frequency floor — can't go finer than source provides.
    """
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
    """Partition pruning — cast year column to integer for Athena."""
    sy = int(start[:4])
    ey = int(end[:4])
    return f"AND CAST(year AS INTEGER) BETWEEN {sy} AND {ey}"


def _price_summary(df: pd.DataFrame, ticker: str) -> str:
    """Compact summary stats for price data."""
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
        f"{ticker} | {df.iloc[0]['date'] if 'date' in df.columns else df.iloc[0].get('week_start', '')}"
        f" → {df.iloc[-1]['date'] if 'date' in df.columns else df.iloc[-1].get('week_start', '')}\n"
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


# ══════════════════════════════════════════════════════════════════════════
# PRICE TOOLS
# ══════════════════════════════════════════════════════════════════════════

def get_prices(
    ticker:   str,
    start:    str,
    end:      str,
    exchange: Optional[str] = None
) -> str:
    """Single ticker, date range, auto-granularity."""
    ticker      = ticker.upper()
    granularity = _price_granularity(start, end)
    yf          = _year_filter(start, end)
    ex_filter   = f"AND exchange = '{exchange}'" if exchange else ""

    if granularity == "daily":
        sql = f"""
            SELECT ticker, date, open, high, low,
                   close, adj_close, volume
            FROM   market_prices
            WHERE  ticker = '{ticker}'
              {yf} {ex_filter}
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
              AND date BETWEEN '{start}' AND '{end}'
            GROUP BY ticker,
                     DATE_TRUNC('month', CAST(date AS DATE))
            ORDER BY date ASC
        """

    try:
        df = query(sql, DB["yfinance"])
        return _format_price_result(df, ticker, granularity)
    except AthenaError as e:
        return f"Error fetching prices for {ticker}: {e}"


def get_prices_multi(
    tickers:  list,
    start:    str,
    end:      str,
    exchange: Optional[str] = None
) -> str:
    """Multiple tickers, date range, auto-granularity. Single Athena query."""
    tickers     = [t.upper() for t in tickers]
    ticker_list = "','".join(tickers)
    granularity = _price_granularity(start, end)
    yf          = _year_filter(start, end)
    ex_filter   = f"AND exchange = '{exchange}'" if exchange else ""

    if granularity == "daily":
        sql = f"""
            SELECT ticker, date, open, high, low,
                   close, adj_close, volume
            FROM   market_prices
            WHERE  ticker IN ('{ticker_list}')
              {yf} {ex_filter}
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
              AND date BETWEEN '{start}' AND '{end}'
            GROUP BY ticker,
                     DATE_TRUNC('month', CAST(date AS DATE))
            ORDER BY ticker ASC, date ASC
        """

    try:
        df = query(sql, DB["yfinance"])
        if df.empty:
            return f"No price data found for {tickers}."

        # Summary per ticker
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
    except AthenaError as e:
        return f"Error fetching prices for {tickers}: {e}"


def get_price_on_date(
    ticker: str,
    date_str: str,
    exchange: Optional[str] = None
) -> str:
    """Single ticker, single date. Returns nearest trading day if needed."""
    ticker    = ticker.upper()
    as_of_yr  = int(date_str[:4])
    ex_filter = f"AND exchange = '{exchange}'" if exchange else ""

    # Nearest trading day on or before requested date
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
        row = df.iloc[0]
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
    except AthenaError as e:
        return f"Error fetching price for {ticker} on {date_str}: {e}"


def get_prices_on_date(
    tickers:  list,
    date_str: str,
    exchange: Optional[str] = None
) -> str:
    """Multiple tickers, single date. Single Athena query."""
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
    except AthenaError as e:
        return f"Error fetching prices on {date_str}: {e}"


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
    """Build aggregation SQL for indicator based on granularity."""
    yf = _year_filter(start, end)

    if granularity == "annual":
        trunc = "year"
    elif granularity == "quarterly":
        trunc = "quarter"
    elif granularity == "monthly":
        trunc = "month"
    else:
        # daily or native — no aggregation
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
    """Single indicator series, date range, auto-granularity."""
    db             = _indicator_db(series_id, source)
    country_filter = f"AND country = '{country}'" if country else ""

    # Get native frequency first for granularity decision
    native_freq = "monthly"  # safe default
    granularity = _indicator_granularity(start, end, native_freq)

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

    except AthenaError as e:
        return f"Error fetching indicator {series_id}: {e}"


def get_indicator_multi(
    series_ids: list,
    start:      str,
    end:        str,
    countries:  Optional[list] = None,
    source:     Optional[str]  = None,
) -> str:
    """
    Multiple indicator series, date range, auto-granularity.
    FRED and WorldBank queried separately, merged in Python.
    """
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
        except AthenaError:
            continue

    if not results:
        return f"No data found for {series_ids} between {start} and {end}."

    combined = pd.concat(results).sort_values(
        ["indicator_id", "date"]
    )
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
    """
    Multiple indicators, single date.
    Returns nearest observation on or before date_str per series.
    Core function — get_macro_snapshot wraps this.
    """
    as_of_yr = int(date_str[:4])
    results  = []

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
        except AthenaError:
            continue

    if not results:
        return f"No indicator data found as of {date_str}."

    lines = [f"Indicators as of {date_str}", "=" * 50]
    for r in results:
        stale = _staleness(r["date"], date_str)
        stale_str = f" [{stale}]" if stale else ""
        lines.append(
            f"{r['indicator_name']:45s} "
            f"{r['value']:>12.2f} {r['unit']}"
            f"  (obs: {r['date']}){stale_str}"
        )
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

    # New — market-based macro signals
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
    """
    Fetch SEC filings or Wikipedia articles for an entity.
    Phase 5 will replace text retrieval with vector search.
    """
    entity          = entity.upper() if not entity[0].islower() else entity
    source_filter   = f"AND source = '{source}'"    if source   else ""
    doc_type_filter = f"AND form_type = '{doc_type}'" if doc_type else ""
    date_filters    = ""
    year_filter     = ""

    if start and end:
        sy          = int(start[:4])
        ey          = int(end[:4])
        year_filter = f"AND CAST(year AS INTEGER) BETWEEN {sy} AND {ey}"
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
        except AthenaError:
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


# ══════════════════════════════════════════════════════════════════════════
# PROSE TOOL
# ══════════════════════════════════════════════════════════════════════════

def get_prose(
    entity:       str,
    section_name: Optional[str] = None,
    form_type:    Optional[str] = None,
    start:        Optional[str] = None,
    end:          Optional[str] = None,
    limit:        int = 3,
) -> str:
    """
    Fetch prose sections from 10-K/10-Q filings.
    Use for qualitative questions: risk factors, MD&A,
    accounting policies, business descriptions.

    section_name options:
      item_1   — Business description
      item_1a  — Risk factors
      item_7   — MD&A
      item_7a  — Market risk
      note_1   — Accounting policies
      note_2   — Revenue segments
      note_3   — Debt details
    """
    entity         = entity.upper()
    section_filter = (f"AND section_name = '{section_name}'"
                      if section_name else "")
    form_filter    = (f"AND form_type = '{form_type.replace('-', '')}'"
                      if form_type else "")
    year_filter    = ""
    date_filter    = ""

    if start and end:
        sy          = int(start[:4])
        ey          = int(end[:4])
        year_filter = f"AND CAST(year AS INTEGER) BETWEEN {sy} AND {ey}"
        date_filter = f"AND filed_date BETWEEN '{start}' AND '{end}'"

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
        ORDER BY filed_date DESC
        LIMIT  {limit}
    """

    try:
        df = query(sql, DB["sec_prose"])
        if df.empty:
            return f"No prose sections found for '{entity}'."

        output = []
        for _, row in df.iterrows():
            output.append(
                f"=== {row['section_title']} "
                f"({row['form_type']} | {row['filed_date']}) ===\n"
                f"Entity: {row['entity']} | "
                f"Section: {row['section_name']} | "
                f"Extraction: {row['extraction_method']}\n\n"
                f"{str(row['text'])[:20000]}"
            )
        return "\n\n".join(output)

    except AthenaError as e:
        return f"Error fetching prose for {entity}: {e}"


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
    Semantic search over SEC filings and Wikipedia articles.
    Embeds query with Cohere v3, queries S3 Vectors index, returns top-K chunks.
    Filters applied in Python after retrieval (S3 Vectors filter syntax varies).
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

    # 2. Query S3 Vectors — fetch more if filtering, trim after
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
        if "empty" in str(e).lower() or "ResourceNotFoundException" in str(e):
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
        meta  = match.get("metadata", {})
        distance = match.get("distance", 0)
        lines.append(
            f"[{i}] {meta.get('title', 'Unknown')} "
            f"({meta.get('source', '')} | {meta.get('doc_date', '')})\n"
            f"    Entity: {meta.get('entity', '')} | "
            f"Distance: {distance:.4f} (lower=more similar)\n"
            f"    {meta.get('text', '')[:300]}\n"
        )

    return "\n".join(lines)
