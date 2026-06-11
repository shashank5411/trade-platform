"""
Anthropic tool definitions — full suite.
Descriptions are precise — they directly determine LLM tool selection quality.
"""

TOOLS = [

    # ── PRICE TOOLS ────────────────────────────────────────────────────────

    {
        "name": "get_prices",
        "description": (
            "Fetch stock price data for a SINGLE ticker over a DATE RANGE. "
            "Automatically selects granularity based on range length: "
            "≤30 days → daily bars, 31-365 days → weekly aggregation, "
            ">365 days → monthly aggregation. "
            "Returns summary stats (start/end price, % change, high, low) "
            "plus aggregated price table. "
            "Use for trend and performance questions over a period. "
            "For MULTIPLE tickers use get_prices_multi. "
            "For a SPECIFIC DATE use get_price_on_date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Single stock ticker. Examples: AAPL, MSFT, SPY. Uppercase."
                },
                "start": {
                    "type": "string",
                    "description": "Start date inclusive. Format: YYYY-MM-DD"
                },
                "end": {
                    "type": "string",
                    "description": "End date inclusive. Format: YYYY-MM-DD"
                },
                "exchange": {
                    "type": "string",
                    "description": "Optional. NYSE, NASDAQ, LSE, NSE. Omit for US stocks."
                }
            },
            "required": ["ticker", "start", "end"]
        }
    },

    {
        "name": "get_prices_multi",
        "description": (
            "Fetch stock price data for MULTIPLE TICKERS over a DATE RANGE "
            "in a single query. Same auto-granularity as get_prices. "
            "Returns per-ticker summary stats plus combined price table. "
            "Use for comparison questions: 'compare AAPL vs MSFT', "
            "'how did tech stocks perform', 'show me all my holdings'. "
            "More efficient than calling get_prices multiple times. "
            "For a SINGLE ticker use get_prices instead. "
            "For a SPECIFIC DATE use get_prices_on_date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols. Examples: ['AAPL', 'MSFT', 'GOOGL']"
                },
                "start": {
                    "type": "string",
                    "description": "Start date inclusive. Format: YYYY-MM-DD"
                },
                "end": {
                    "type": "string",
                    "description": "End date inclusive. Format: YYYY-MM-DD"
                },
                "exchange": {
                    "type": "string",
                    "description": "Optional. Filter all tickers to same exchange."
                }
            },
            "required": ["tickers", "start", "end"]
        }
    },

    {
        "name": "get_price_on_date",
        "description": (
            "Fetch the price of a SINGLE TICKER on a SPECIFIC DATE. "
            "Returns exact OHLCV for that date, or nearest trading day "
            "if the date is a weekend or holiday. "
            "Use for questions like: 'what was AAPL on March 15 2024?', "
            "'what was the closing price on earnings day?'. "
            "For a DATE RANGE use get_prices. "
            "For MULTIPLE TICKERS on one date use get_prices_on_date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Single stock ticker. Uppercase."
                },
                "date_str": {
                    "type": "string",
                    "description": "Target date. Format: YYYY-MM-DD"
                },
                "exchange": {
                    "type": "string",
                    "description": "Optional. Exchange filter."
                }
            },
            "required": ["ticker", "date_str"]
        }
    },

    {
        "name": "get_prices_on_date",
        "description": (
            "Fetch prices for MULTIPLE TICKERS on a SPECIFIC DATE. "
            "Single query — efficient for point-in-time comparisons. "
            "Returns nearest trading day per ticker. "
            "Use for: 'show me all tech stocks on Jan 1 2024', "
            "'what were AAPL and MSFT prices on the Fed announcement date?'. "
            "For a DATE RANGE use get_prices_multi."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols."
                },
                "date_str": {
                    "type": "string",
                    "description": "Target date. Format: YYYY-MM-DD"
                },
                "exchange": {
                    "type": "string",
                    "description": "Optional. Exchange filter."
                }
            },
            "required": ["tickers", "date_str"]
        }
    },

    # ── INDICATOR TOOLS ────────────────────────────────────────────────────

    {
        "name": "get_indicator",
        "description": (
            "Fetch a SINGLE economic indicator series over a DATE RANGE. "
            "Auto-selects granularity respecting native data frequency: "
            "daily series (DGS10, DGS2): ≤30d→daily, ≤365d→weekly, >365d→monthly. "
            "monthly series (UNRATE, FEDFUNDS, CPI): ≤730d→monthly, >730d→quarterly. "
            "annual series (World Bank): always annual. "
            "FRED series IDs: FEDFUNDS, UNRATE, CPIAUCSL, DGS10, DGS2, "
            "GDP, M2SL, UMCSENT, HOUST, INDPRO. "
            "World Bank IDs (use dot notation): NY.GDP.MKTP.CD, "
            "FP.CPI.TOTL.ZG, SP.POP.TOTL, NY.GDP.PCAP.CD, NE.TRD.GNFS.ZS. "
            "For MULTIPLE series use get_indicator_multi. "
            "For a SPECIFIC DATE use get_indicator_on_date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series_id": {
                    "type": "string",
                    "description": "Indicator ID. FRED: UNRATE, FEDFUNDS etc. World Bank: NY.GDP.MKTP.CD etc."
                },
                "start": {
                    "type": "string",
                    "description": "Start date. Format: YYYY-MM-DD"
                },
                "end": {
                    "type": "string",
                    "description": "End date. Format: YYYY-MM-DD"
                },
                "country": {
                    "type": "string",
                    "description": "ISO country code. Required for World Bank: US, CN, IN, GB, DE, JP, BR"
                },
                "source": {
                    "type": "string",
                    "enum": ["FRED", "WORLDBANK"],
                    "description": "Optional. Auto-detected from series_id format."
                },
                "as_of": {
                    "type": "string",
                    "description": "Optional. Point-in-time vintage for FRED. Format: YYYY-MM-DD"
                }
            },
            "required": ["series_id", "start", "end"]
        }
    },

    {
        "name": "get_indicator_multi",
        "description": (
            "Fetch MULTIPLE economic indicator series over a DATE RANGE. "
            "More efficient than calling get_indicator repeatedly. "
            "Can mix FRED and World Bank series in one call. "
            "Same auto-granularity as get_indicator. "
            "Use for: 'show me unemployment and inflation together', "
            "'compare GDP across countries', 'give me all macro series for 2022'. "
            "For a SINGLE series use get_indicator. "
            "For a SPECIFIC DATE use get_indicator_on_date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of indicator IDs. Can mix FRED and World Bank."
                },
                "start": {
                    "type": "string",
                    "description": "Start date. Format: YYYY-MM-DD"
                },
                "end": {
                    "type": "string",
                    "description": "End date. Format: YYYY-MM-DD"
                },
                "countries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional. Country per series_id (same index). ISO codes."
                },
                "source": {
                    "type": "string",
                    "enum": ["FRED", "WORLDBANK"],
                    "description": "Optional. Force source. Auto-detected if omitted."
                }
            },
            "required": ["series_ids", "start", "end"]
        }
    },

    {
        "name": "get_indicator_on_date",
        "description": (
            "Fetch MULTIPLE indicators as of a SPECIFIC DATE. "
            "Returns latest available observation on or before the date "
            "per series — handles different update frequencies gracefully. "
            "Flags staleness when observation is significantly older than requested date. "
            "Use for: 'what was unemployment on March 15 2022?', "
            "'show me all macro indicators on the day of the SVB collapse', "
            "'what were rates and inflation when AAPL hit its low?'. "
            "For a DATE RANGE use get_indicator or get_indicator_multi. "
            "For a full macro overview use get_macro_snapshot instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of indicator IDs."
                },
                "date_str": {
                    "type": "string",
                    "description": "Target date. Returns nearest available on or before. Format: YYYY-MM-DD"
                },
                "countries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional. Country per series (same index). ISO codes."
                }
            },
            "required": ["series_ids", "date_str"]
        }
    },

    # ── MACRO SHORTCUT ─────────────────────────────────────────────────────

    {
        "name": "get_macro_snapshot",
        "description": (
            "Get a comprehensive macro economic snapshot as of a specific date. "
            "Returns Fed funds rate, unemployment, CPI, 10yr/2yr Treasury yields, "
            "US GDP, World Bank GDP, and S&P 500 price — all in one call. "
            "Use this FIRST when answering questions about the economic environment, "
            "macro context, or when you need multiple macro indicators at once. "
            "Much more efficient than calling get_indicator multiple times. "
            "Returns latest available value per indicator with staleness flags. "
            "For custom indicator lists use get_indicator_on_date instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "as_of_date": {
                    "type": "string",
                    "description": "Date for snapshot. Returns latest available on or before. Format: YYYY-MM-DD"
                }
            },
            "required": ["as_of_date"]
        }
    },

    # ── DOCUMENT TOOL ──────────────────────────────────────────────────────

    {
        "name": "get_documents",
        "description": (
            "Fetch SEC filings or Wikipedia articles for qualitative context. "
            "SEC filings (10-K annual, 10-Q quarterly) contain financial narratives: "
            "revenue, net income, EPS, management commentary, risk factors, outlook. "
            "Wikipedia articles cover economic concepts and events. "
            "Use for: 'what did Apple say about supply chain in 2022?', "
            "'explain quantitative easing', 'what were MSFT risk factors in 2023?'. "
            "For quantitative price data use get_prices tools instead. "
            "Available companies: AAPL, MSFT, GOOGL, AMZN, JPM, BAC, XOM. "
            "Available Wikipedia topics: Inflation, Recession, Federal_Reserve, "
            "Quantitative_easing, 2008_financial_crisis, COVID-19_recession."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Company ticker (AAPL) or Wikipedia topic slug (inflation)."
                },
                "doc_type": {
                    "type": "string",
                    "enum": ["10-K", "10-Q", "wiki_article"],
                    "description": "Optional. Document type. Omit to search all."
                },
                "start": {
                    "type": "string",
                    "description": "Optional. Filter by filing/snapshot date. Format: YYYY-MM-DD"
                },
                "end": {
                    "type": "string",
                    "description": "Optional. Filter by filing/snapshot date. Format: YYYY-MM-DD"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max documents. Default 3. Use 1 for most specific queries.",
                    "default": 3
                },
                "source": {
                    "type": "string",
                    "enum": ["EDGAR", "WIKIPEDIA"],
                    "description": "Optional. Force specific source."
                }
            },
            "required": ["entity"]
        }
    },

    # ── PROSE TOOL ─────────────────────────────────────────────────────────

    {
        "name": "get_prose",
        "description": (
            "Fetch qualitative prose sections from SEC 10-K and 10-Q filings. "
            "Use for: risk factors (item_1a), management discussion (item_7), "
            "accounting policies (note_1), business descriptions (item_1), "
            "market risk disclosures (item_7a), revenue segments (note_2), "
            "debt details (note_3). "
            "More targeted than semantic_search when you know the specific "
            "section needed. Use semantic_search for cross-company or "
            "concept-based retrieval."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Company ticker e.g. AAPL, JPM"
                },
                "section_name": {
                    "type": "string",
                    "description": "Section to fetch",
                    "enum": ["item_1", "item_1a", "item_7",
                             "item_7a", "note_1", "note_2", "note_3"]
                },
                "form_type": {
                    "type": "string",
                    "description": "10-K or 10-Q",
                    "enum": ["10-K", "10-Q"]
                },
                "start": {
                    "type": "string",
                    "description": "Start date YYYY-MM-DD"
                },
                "end": {
                    "type": "string",
                    "description": "End date YYYY-MM-DD"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results (default 3)",
                    "default": 3
                },
            },
            "required": ["entity"]
        }
    },

    # ── SEMANTIC SEARCH ────────────────────────────────────────────────────

    {
        "name": "semantic_search",
        "description": (
            "Semantic search over SEC EDGAR filings and Wikipedia articles. "
            "Use for qualitative questions: company strategy, risk factors, "
            "business descriptions, economic concepts, historical events. "
            "Complements get_documents (which does exact entity lookup) by "
            "finding relevant content across all documents by meaning. "
            "Examples: 'Apple revenue recognition policy', "
            "'quantitative easing effects on inflation', "
            "'JPMorgan risk factors 2022'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 20)",
                    "default": 5
                },
                "source": {
                    "type": "string",
                    "description": "Filter by source: EDGAR or WIKIPEDIA (optional)",
                    "enum": ["EDGAR", "WIKIPEDIA"]
                },
                "entity": {
                    "type": "string",
                    "description": "Filter by entity/ticker e.g. AAPL, JPM (optional)"
                },
            },
            "required": ["query"]
        }
    },
]


def get_registry():
    """Maps tool names to Python implementations in api.py."""
    from query import api
    return {
        "get_prices":              api.get_prices,
        "get_prices_multi":        api.get_prices_multi,
        "get_price_on_date":       api.get_price_on_date,
        "get_prices_on_date":      api.get_prices_on_date,
        "get_indicator":           api.get_indicator,
        "get_indicator_multi":     api.get_indicator_multi,
        "get_indicator_on_date":   api.get_indicator_on_date,
        "get_macro_snapshot":      api.get_macro_snapshot,
        "get_documents":           api.get_documents,
        "get_prose":               api.get_prose,
        "semantic_search":         api.semantic_search,
    }