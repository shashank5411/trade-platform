"""
tools.py — real @tool wrappers over query.api's actual functions.

Every tool here calls directly into `query.api` (the same registry your
real system's get_registry() maps to) — no reimplementation, one source
of truth. Descriptions are adapted from the real tool_definitions.py
file, since (per that file's own header) descriptions directly determine
LLM tool-selection quality — this isn't cosmetic, it's load-bearing.

ASSUMPTION, not yet confirmed: query.api's functions are synchronous.
LangChain's @tool decorator wraps a sync function so .ainvoke() still
works (runs it via a thread executor automatically) — this should be
transparent, but confirm once this runs against real credentials.

Agent-type groupings match confirmed project facts (FilingsAgent owns
get_fed_communications — FRED data is macro's domain, Fed SPEECH is
filings' domain, per prior design notes) plus the natural grouping from
each tool's own description.
"""

from typing import Optional
from langchain_core.tools import tool
from query import api


# ══════════════════════════════════════════════════════════════════════
# MARKET — price/volume, sector discovery
# ══════════════════════════════════════════════════════════════════════

@tool
def get_prices(ticker: str, start: str, end: str,
                exchange: Optional[str] = None,
                sector: Optional[str] = None,
                industry: Optional[str] = None) -> str:
    """Fetch stock price data for a SINGLE ticker over a DATE RANGE.
    Auto-selects granularity: <=30 days -> daily, 31-365 days -> weekly,
    >365 days -> monthly. Returns summary stats (start/end price, %
    change, high, low) plus aggregated price table. For MULTIPLE tickers
    use get_prices_multi. For SECTOR-LEVEL queries use
    get_prices_by_sector. For a SPECIFIC DATE use get_price_on_date."""
    return str(api.get_prices(ticker=ticker, start=start, end=end,
                               exchange=exchange, sector=sector, industry=industry))


@tool
def get_prices_multi(tickers: list[str], start: str, end: str,
                      exchange: Optional[str] = None,
                      sector: Optional[str] = None,
                      industry: Optional[str] = None) -> str:
    """Fetch stock price data for MULTIPLE tickers over a DATE RANGE in
    one query. Use for comparisons ('compare AAPL vs MSFT'). More
    efficient than repeated get_prices calls. For a SINGLE ticker use
    get_prices. For a SPECIFIC DATE use get_prices_on_date."""
    return str(api.get_prices_multi(tickers=tickers, start=start, end=end,
                                     exchange=exchange, sector=sector, industry=industry))


@tool
def get_prices_by_sector(sector: str, start: str, end: str,
                          industry: Optional[str] = None) -> str:
    """Fetch a performance summary for ALL tickers in a SECTOR over a
    DATE RANGE, ranked by performance (up to 20 tickers). Use for
    sector-level questions ('how are tech stocks doing?'). For SPECIFIC
    tickers you already know, use get_prices or get_prices_multi."""
    return str(api.get_prices_by_sector(sector=sector, start=start, end=end, industry=industry))


@tool
def get_companies_in_sector(sector: str, industry: Optional[str] = None,
                             min_market_cap: Optional[float] = None,
                             sp500_only: bool = False) -> str:
    """Look up which companies are tracked in a sector/industry. Use as
    STEP 1 before get_prices_multi when the question names a sector
    rather than specific tickers. Returns ticker list, company names,
    market cap, beta, dividend yield — the 'Tickers:' line can be passed
    directly into get_prices_multi as step 2."""
    return str(api.get_companies_in_sector(sector=sector, industry=industry,
                                            min_market_cap=min_market_cap, sp500_only=sp500_only))


@tool
def get_price_on_date(ticker: str, date_str: str, exchange: Optional[str] = None) -> str:
    """Fetch the price of a SINGLE ticker on a SPECIFIC DATE. Returns
    exact OHLCV, or nearest trading day if a weekend/holiday. For a DATE
    RANGE use get_prices. For MULTIPLE tickers on one date use
    get_prices_on_date."""
    return str(api.get_price_on_date(ticker=ticker, date_str=date_str, exchange=exchange))


@tool
def get_prices_on_date(tickers: list[str], date_str: str, exchange: Optional[str] = None) -> str:
    """Fetch prices for MULTIPLE tickers on a SPECIFIC DATE — one query,
    efficient for point-in-time comparisons. For a DATE RANGE use
    get_prices_multi."""
    return str(api.get_prices_on_date(tickers=tickers, date_str=date_str, exchange=exchange))


MARKET_TOOLS = [get_prices, get_prices_multi, get_prices_by_sector,
                 get_companies_in_sector, get_price_on_date, get_prices_on_date]


# ══════════════════════════════════════════════════════════════════════
# FILINGS — SEC documents, prose sections, semantic search, Fed comms
# ══════════════════════════════════════════════════════════════════════

@tool
def get_documents(entity: str, doc_type: Optional[str] = None,
                   start: Optional[str] = None, end: Optional[str] = None,
                   limit: int = 3, source: Optional[str] = None) -> str:
    """Fetch SEC filings (10-K/10-Q) or Wikipedia articles for
    qualitative context. Available companies: AAPL, MSFT, GOOGL, AMZN,
    JPM, BAC, XOM. Available Wikipedia topics: Inflation, Recession,
    Federal_Reserve, Quantitative_easing, 2008_financial_crisis,
    COVID-19_recession. For quantitative price data use get_prices
    tools instead."""
    return str(api.get_documents(entity=entity, doc_type=doc_type, start=start,
                                  end=end, limit=limit, source=source))


@tool
def get_prose(entity: str, section_name: Optional[str] = None,
               section_names: Optional[list[str]] = None,
               form_type: Optional[str] = None,
               start: Optional[str] = None, end: Optional[str] = None,
               limit: int = 3, max_chars: int = 8000) -> str:
    """Fetch qualitative prose sections from SEC 10-K/10-Q filings:
    item_1 (business), item_1a (risk factors), item_7 (MD&A), item_7a
    (market risk), note_1 (accounting policies), note_2 (revenue),
    note_3 (debt). Prefer section_names (a list) over repeated
    single-section calls — fetches multiple sections in ONE Athena
    query. More targeted than semantic_search when you know the
    specific section needed."""
    return str(api.get_prose(entity=entity, section_name=section_name,
                              section_names=section_names, form_type=form_type,
                              start=start, end=end, limit=limit, max_chars=max_chars))


@tool
def semantic_search(query: str, top_k: int = 5,
                     source: Optional[str] = None,
                     entity: Optional[str] = None) -> str:
    """Semantic search over SEC EDGAR filings, Wikipedia, and FedSpeak
    documents (FOMC minutes/statements/transcripts, governor speeches).
    Use for qualitative/conceptual questions: company strategy, risk
    factors, Fed policy reasoning. Complements get_documents (exact
    entity lookup) by finding relevant content by meaning. Prefer
    get_fed_communications for targeted Fed document retrieval."""
    return str(api.semantic_search(query=query, top_k=top_k, source=source, entity=entity))


@tool
def get_fed_communications(doc_type: Optional[str] = None,
                            start: Optional[str] = None, end: Optional[str] = None,
                            entity: Optional[str] = None, limit: int = 5) -> str:
    """Retrieve Federal Reserve communications: FOMC statements (rate
    decisions), minutes, press conference transcripts, governor
    speeches. Use for any 'what has the Fed said about X' question.
    entity: 'FOMC' for committee documents, or a speaker name ('Powell',
    'Waller', 'Jefferson', 'Bowman', 'Cook', 'Kugler', 'Barr') for
    speeches. Prefer over semantic_search for targeted Fed retrieval."""
    return str(api.get_fed_communications(doc_type=doc_type, start=start, end=end,
                                           entity=entity, limit=limit))


FILINGS_TOOLS = [get_documents, get_prose, semantic_search, get_fed_communications]


# ══════════════════════════════════════════════════════════════════════
# MACRO — FRED / World Bank indicators
# ══════════════════════════════════════════════════════════════════════

@tool
def get_indicator(series_id: str, start: str, end: str,
                   country: Optional[str] = None,
                   source: Optional[str] = None,
                   as_of: Optional[str] = None) -> str:
    """Fetch a SINGLE economic indicator over a DATE RANGE. FRED series:
    FEDFUNDS, UNRATE, CPIAUCSL, DGS10, DGS2, GDP, M2SL, UMCSENT, HOUST,
    INDPRO, DCOILWTICO (WTI crude SPOT price — distinct from futures,
    which live in get_prices via ticker CL=F). World Bank (dot
    notation, requires country): NY.GDP.MKTP.CD, FP.CPI.TOTL.ZG,
    SP.POP.TOTL. For MULTIPLE series use get_indicator_multi. For a
    SPECIFIC DATE use get_indicator_on_date."""
    return str(api.get_indicator(series_id=series_id, start=start, end=end,
                                  country=country, source=source, as_of=as_of))


@tool
def get_indicator_multi(series_ids: list[str], start: str, end: str,
                         countries: Optional[list[str]] = None,
                         source: Optional[str] = None) -> str:
    """Fetch MULTIPLE economic indicators over a DATE RANGE in one call
    — can mix FRED and World Bank series. More efficient than repeated
    get_indicator calls. For a SINGLE series use get_indicator."""
    return str(api.get_indicator_multi(series_ids=series_ids, start=start, end=end,
                                        countries=countries, source=source))


@tool
def get_indicator_on_date(series_ids: list[str], date_str: str,
                           countries: Optional[list[str]] = None) -> str:
    """Fetch MULTIPLE indicators as of a SPECIFIC DATE — latest
    available observation on or before the date per series, flags
    staleness. For a DATE RANGE use get_indicator/get_indicator_multi.
    For a full macro overview use get_macro_snapshot instead."""
    return str(api.get_indicator_on_date(series_ids=series_ids, date_str=date_str,
                                          countries=countries))


@tool
def get_macro_snapshot(as_of_date: str) -> str:
    """Get a comprehensive macro snapshot as of a date: Fed funds rate,
    unemployment, CPI, 10yr/2yr Treasury yields, US GDP, World Bank GDP,
    S&P 500 price — all in one call. Use FIRST for questions about the
    economic environment or macro context. Much more efficient than
    multiple get_indicator calls."""
    return str(api.get_macro_snapshot(as_of_date=as_of_date))


MACRO_TOOLS = [get_indicator, get_indicator_multi, get_indicator_on_date, get_macro_snapshot]


# ══════════════════════════════════════════════════════════════════════
# SENTIMENT — news coverage, insider (Form 4) trading activity
# ══════════════════════════════════════════════════════════════════════

@tool
def get_news(ticker: str, start: Optional[str] = None, end: Optional[str] = None,
             sentiment: Optional[str] = None, publisher_tier: Optional[int] = None,
             limit: int = 10) -> str:
    """Retrieve news articles for a ticker with per-article sentiment.
    Covers AAPL, MSFT, GOOGL, AMZN, JPM, BAC, XOM, GLD, USO, TLT, SPY.
    publisher_tier: 1=wire services, 2=financial media, 3=opinion. Use
    get_news_summary FIRST for an overview, then this for details."""
    return str(api.get_news(ticker=ticker, start=start, end=end, sentiment=sentiment,
                             publisher_tier=publisher_tier, limit=limit))


@tool
def get_news_summary(ticker: str, start: Optional[str] = None, end: Optional[str] = None,
                      publisher_tier: Optional[int] = None) -> str:
    """Get aggregated news sentiment overview for a ticker — article
    count, sentiment breakdown, top publishers. Use FIRST before
    get_news to understand overall narrative tone."""
    return str(api.get_news_summary(ticker=ticker, start=start, end=end,
                                     publisher_tier=publisher_tier))


@tool
def get_insider_trades(ticker: str, start: Optional[str] = None, end: Optional[str] = None,
                        transaction_type: Optional[str] = None, limit: int = 20) -> str:
    """Retrieve SEC Form 4 insider trades. Covers AAPL, MSFT, GOOGL,
    AMZN, JPM, BAC, XOM. Transaction types: P=open-market purchase
    (strongest bullish signal), S=open-market sale (often a pre-planned
    10b5-1 program — NOT necessarily bearish), F=tax withholding on RSU
    vesting (NOT a sell decision, never count as discretionary selling),
    A=award/grant, D=disposition to trust/charity, M/X=option exercise.
    Use get_insider_summary FIRST for the net signal, then this for
    individual transactions."""
    return str(api.get_insider_trades(ticker=ticker, start=start, end=end,
                                       transaction_type=transaction_type, limit=limit))


@tool
def get_insider_summary(ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> str:
    """Get aggregated insider trading signal for a ticker — net
    open-market buying vs. selling. Net signal = P purchases minus S
    sales ONLY; F (RSU tax withholding) is excluded from the net signal
    since it's automatic, not discretionary. Most S sales are
    pre-planned 10b5-1 programs, so high S volume alone isn't reliably
    bearish; P purchases are more reliably bullish. Use FIRST before
    get_insider_trades."""
    return str(api.get_insider_summary(ticker=ticker, start=start, end=end))


SENTIMENT_TOOLS = [get_news, get_news_summary, get_insider_trades, get_insider_summary]


TOOLS_BY_AGENT_TYPE = {
    "market": MARKET_TOOLS,
    "filings": FILINGS_TOOLS,
    "macro": MACRO_TOOLS,
    "sentiment": SENTIMENT_TOOLS,
}