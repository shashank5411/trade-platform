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
            "Supports optional sector and industry filters (e.g. sector='Technology'). "
            "Use for trend and performance questions over a period. "
            "For MULTIPLE tickers use get_prices_multi. "
            "For SECTOR-LEVEL queries use get_prices_by_sector. "
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
                },
                "sector": {
                    "type": "string",
                    "description": "Optional. Filter by sector e.g. Technology, Energy, Financials."
                },
                "industry": {
                    "type": "string",
                    "description": "Optional. Filter by industry e.g. Semiconductors, Oil & Gas."
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
            "Supports optional sector and industry filters. "
            "More efficient than calling get_prices multiple times. "
            "For a SINGLE ticker use get_prices instead. "
            "For SECTOR-LEVEL discovery use get_prices_by_sector. "
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
                },
                "sector": {
                    "type": "string",
                    "description": "Optional. Filter by sector e.g. Technology, Energy, Financials."
                },
                "industry": {
                    "type": "string",
                    "description": "Optional. Filter by industry e.g. Semiconductors, Oil & Gas."
                }
            },
            "required": ["tickers", "start", "end"]
        }
    },

    {
        "name": "get_prices_by_sector",
        "description": (
            "Fetch a performance summary for ALL tickers in a SECTOR over a DATE RANGE. "
            "Returns one row per ticker with avg close, start/end price, and % change "
            "for the period — ranked by performance. Up to 20 tickers. "
            "Use for sector-level questions: 'how are tech stocks doing?', "
            "'show me energy companies this year', 'which financials performed best?', "
            "'compare semiconductor stocks over Q1'. "
            "Optionally narrow by industry within the sector. "
            "For SPECIFIC TICKERS you already know, use get_prices or get_prices_multi. "
            "For a SINGLE ticker use get_prices."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sector": {
                    "type": "string",
                    "description": (
                        "Sector name exactly as stored. Examples: Technology, Energy, "
                        "Financials, Health Care, Consumer Discretionary, Industrials, "
                        "Communication Services, Utilities, Real Estate, Materials, "
                        "Consumer Staples."
                    )
                },
                "start": {
                    "type": "string",
                    "description": "Start date inclusive. Format: YYYY-MM-DD"
                },
                "end": {
                    "type": "string",
                    "description": "End date inclusive. Format: YYYY-MM-DD"
                },
                "industry": {
                    "type": "string",
                    "description": (
                        "Optional. Narrow to a specific industry within the sector. "
                        "Examples: Semiconductors, Oil & Gas Integrated, "
                        "Internet Content & Information, Banks—Diversified."
                    )
                }
            },
            "required": ["sector", "start", "end"]
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
                    "description": (
                        "Single section to fetch. Options: item_1 (business), "
                        "item_1a (risk factors), item_7 (MD&A), item_7a (market risk), "
                        "note_1 (accounting policies), note_2 (revenue), note_3 (debt). "
                        "Use section_names instead when fetching multiple sections."
                    ),
                    "enum": ["item_1", "item_1a", "item_7", "item_7a",
                             "note_1", "note_2", "note_3"]
                },
                "section_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Fetch multiple sections in ONE call instead of calling "
                        "get_prose separately for each. Preferred over repeated "
                        "single-section calls. "
                        "e.g. [\"item_1a\", \"item_7\"] fetches risk factors AND "
                        "MD&A in a single Athena query. "
                        "Options: item_1, item_1a, item_7, item_7a, "
                        "note_1, note_2, note_3"
                    )
                },
                "form_type": {
                    "type": "string",
                    "description": "Filing type: 10-K (annual) or 10-Q (quarterly)",
                    "enum": ["10-K", "10-Q"]
                },
                "start": {
                    "type": "string",
                    "description": "Start date YYYY-MM-DD (optional)"
                },
                "end": {
                    "type": "string",
                    "description": "End date YYYY-MM-DD (optional)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of filings to return (default 3)",
                    "default": 3
                },
                "max_chars": {
                    "type": "integer",
                    "description": (
                        "Characters returned per section. Default 8000 — covers "
                        "most answers. Pass 20000 for deep dives into long sections "
                        "like MD&A or full risk factor lists."
                    ),
                    "default": 8000
                },
            },
            "required": ["entity"]
        }
    },

    # ── SEMANTIC SEARCH ────────────────────────────────────────────────────

    {
        "name": "semantic_search",
        "description": (
            "Semantic search over SEC EDGAR filings, Wikipedia articles, and "
            "FedSpeak documents (FOMC minutes, statements, press conference "
            "transcripts, Fed governor speeches). "
            "Use for qualitative questions: company strategy, risk factors, "
            "business descriptions, economic concepts, historical events, "
            "Fed policy reasoning and commentary. "
            "Complements get_documents (which does exact entity lookup) by "
            "finding relevant content across all documents by meaning. "
            "Prefer get_fed_communications for targeted Fed document retrieval; "
            "use semantic_search with source='FEDSPEAK' for broad Fed concept search. "
            "Examples: 'Apple revenue recognition policy', "
            "'quantitative easing effects on inflation', "
            "'Fed stance on inflation 2022', "
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
                    "description": "Filter by source: EDGAR (SEC filings), WIKIPEDIA, or FEDSPEAK (Fed communications) — optional",
                    "enum": ["EDGAR", "WIKIPEDIA", "FEDSPEAK"]
                },
                "entity": {
                    "type": "string",
                    "description": "Filter by entity/ticker e.g. AAPL, JPM (optional)"
                },
            },
            "required": ["query"]
        }
    },

    # ── NEWS / SENTIMENT ───────────────────────────────────────────────────

    {
        "name": "get_news",
        "description": (
            "Retrieve news articles for a stock ticker with per-article sentiment "
            "and reasoning. Use for: 'what is the news around AAPL this week', "
            "'show me negative news about JPM', 'what are analysts saying about XOM'. "
            "Covers: AAPL, MSFT, GOOGL, AMZN, JPM, BAC, XOM (equities) and "
            "GLD (gold), USO (oil), TLT (bonds), SPY (broad market). "
            "publisher_tier: 1=wire services (Reuters/AP/Bloomberg), "
            "2=financial media (MarketWatch/Benzinga/CNBC), "
            "3=opinion (Seeking Alpha/Motley Fool). "
            "Use get_news_summary first for an overview, then get_news for details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker. Examples: AAPL, JPM, GLD, SPY"
                },
                "start": {"type": "string", "description": "Start date YYYY-MM-DD"},
                "end":   {"type": "string", "description": "End date YYYY-MM-DD"},
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                    "description": "Filter by sentiment"
                },
                "publisher_tier": {
                    "type": "integer",
                    "description": "Max publisher tier: 1=wire only, 2=established+wire, 3=all"
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Max articles (default 10)"
                },
            },
            "required": ["ticker"],
        },
    },

    {
        "name": "get_news_summary",
        "description": (
            "Get aggregated news sentiment overview for a ticker. Returns total "
            "article count, sentiment breakdown (positive/negative/neutral counts), "
            "and top publishers. Use FIRST before get_news to understand overall "
            "narrative tone. Great for: 'has news been positive or negative for AAPL "
            "this month?', 'how much coverage has JPM gotten?', 'what is the media "
            "sentiment around XOM after the oil price drop?'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker"},
                "start":  {"type": "string", "description": "Start date YYYY-MM-DD"},
                "end":    {"type": "string", "description": "End date YYYY-MM-DD"},
                "publisher_tier": {
                    "type": "integer",
                    "description": "Max publisher tier (1=wire only, 2=established, 3=all)"
                },
            },
            "required": ["ticker"],
        },
    },


{
    "name": "get_fed_communications",
    "description": (
        "Retrieve Federal Reserve communications: FOMC statements (rate decisions), "
        "minutes (full meeting deliberation), press conference transcripts, and "
        "individual governor speeches. Use for questions about Fed policy, rate "
        "decisions, Powell statements, inflation outlook from the Fed, or any "
        "'what has the Fed said about X' question. Prefer over semantic_search "
        "for targeted Fed document retrieval. "
        "doc_type options: 'statement' | 'minutes' | 'transcript' | 'speech'. "
        "entity: 'FOMC' for committee documents, speaker name for speeches "
        "(e.g. 'Powell', 'Waller', 'Jefferson'). "
        "Returns up to 8,000 chars per document."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "doc_type": {
                "type": "string",
                "enum": ["statement", "minutes", "transcript", "speech"],
                "description": "Type of Fed communication to retrieve. Omit to search all types.",
            },
            "start": {
                "type": "string",
                "description": "Start date YYYY-MM-DD",
            },
            "end": {
                "type": "string",
                "description": "End date YYYY-MM-DD",
            },
            "entity": {
                "type": "string",
                "description": (
                    "'FOMC' for committee documents (statements, minutes, transcripts). "
                    "Speaker name for speeches: 'Powell', 'Waller', 'Jefferson', "
                    "'Bowman', 'Cook', 'Kugler', 'Barr'. "
                    "Omit to search all entities."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max documents to return (default 5, max 10)",
                "default": 5,
            },
        },
        "required": [],
    },
}
]


def get_registry():
    """Maps tool names to Python implementations in api.py."""
    from query import api
    return {
        "get_prices":              api.get_prices,
        "get_prices_multi":        api.get_prices_multi,
        "get_prices_by_sector":    api.get_prices_by_sector,
        "get_price_on_date":       api.get_price_on_date,
        "get_prices_on_date":      api.get_prices_on_date,
        "get_indicator":           api.get_indicator,
        "get_indicator_multi":     api.get_indicator_multi,
        "get_indicator_on_date":   api.get_indicator_on_date,
        "get_macro_snapshot":      api.get_macro_snapshot,
        "get_documents":           api.get_documents,
        "get_prose":               api.get_prose,
        "semantic_search":         api.semantic_search,
        "get_fed_communications":  api.get_fed_communications,
        "get_news":                api.get_news,
        "get_news_summary":        api.get_news_summary,
    }