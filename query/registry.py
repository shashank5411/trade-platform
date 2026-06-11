"""
Agent registry — single source of truth for all available agents.

To add a new agent:
  1. Create your specialist class in query/sub_agents.py
  2. Add an entry to AGENT_REGISTRY below
  3. That's it — planner and executor pick it up automatically
"""

from query.sub_agents import market_agent, macro_agent, filings_agent

AGENT_REGISTRY = {
    "market": {
        "agent":       market_agent,
        "description": (
            "Answers questions about asset prices, returns, and market "
            "performance. Has access to US equities (AAPL, MSFT, GOOGL, "
            "AMZN, JPM, BAC, XOM, NVDA, META, TSLA, BRK-B, V, MA, UNH, "
            "JNJ, WMT, CVX, COST, GS, PFE, SPY), global indices (S&P 500, "
            "Dow, NASDAQ, FTSE, DAX, Nikkei, Hang Seng), FX rates "
            "(EURUSD, GBPUSD, USDJPY, DXY), and commodities (Gold, Oil, "
            "Silver, Natural Gas). Use for: price history, % returns, "
            "highs/lows, cross-asset performance comparisons."
        ),
    },
    "macro": {
        "agent":       macro_agent,
        "description": (
            "Answers questions about macroeconomic conditions and indicators. "
            "Has access to FRED series (Fed funds rate, unemployment, CPI, "
            "Treasury yields, GDP, M2, consumer sentiment, credit spreads, "
            "dollar index) and World Bank data (GDP, inflation, population, "
            "trade) for US, China, India, UK, Germany, Japan, Brazil. "
            "Use for: monetary policy, inflation, economic cycles, "
            "interest rate environment, macro context for any time period."
        ),
    },
    "filings": {
        "agent":       filings_agent,
        "description": (
            "Answers qualitative questions using SEC filings and Wikipedia "
            "articles. Has access to 10-K and 10-Q filings for AAPL, MSFT, "
            "GOOGL, AMZN, JPM, BAC, XOM, and Wikipedia articles on "
            "Inflation, Recession, Federal Reserve, Quantitative Easing, "
            "2008 financial crisis, COVID-19 recession, Silicon Valley Bank. "
            "Uses semantic search for meaning-based retrieval across all "
            "documents. Use for: business strategy, risk factors, company "
            "narratives, economic concept explanations, historical event "
            "analysis."
        ),
    },
}


def get_agent(name: str):
    """Get agent instance by name. Returns None if not found."""
    entry = AGENT_REGISTRY.get(name)
    return entry["agent"] if entry else None


def get_agent_descriptions() -> str:
    """
    Format agent descriptions for the planner prompt.
    Called once at planner initialization.
    """
    lines = []
    for name, entry in AGENT_REGISTRY.items():
        lines.append(f"  {name}:\n    {entry['description']}")
    return "\n\n".join(lines)


def list_agents() -> list:
    """Return list of registered agent names."""
    return list(AGENT_REGISTRY.keys())
