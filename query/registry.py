"""
Agent registry — single source of truth for all available agents.

To add a new agent:
  1. Create your specialist class in query/sub_agents.py
  2. Add an entry to AGENT_REGISTRY below
  3. That's it — planner and executor pick it up automatically
"""

from query.sub_agents import market_agent, macro_agent, filings_agent, sentiment_agent

AGENT_REGISTRY = {
    "market": {
        "agent":       market_agent,
        "description": (
            "Answers questions about asset prices, returns, and market "
            "performance. Has access to US equities (AAPL, MSFT, GOOGL, "
            "AMZN, JPM, BAC, XOM, NVDA, META, TSLA, BRK-B, V, MA, UNH, "
            "JNJ, WMT, CVX, COST, GS, PFE, SPY), global indices (S&P 500, "
            "Dow, NASDAQ, FTSE, DAX, Nikkei, Hang Seng), FX rates "
            "(EURUSD, GBPUSD, USDJPY, DXY), and commodity FUTURES contracts "
            "(Gold GC=F, Oil CL=F, Silver SI=F, Natural Gas NG=F) via "
            "yfinance. Use for: price history, % returns, highs/lows, "
            "cross-asset performance comparisons, futures/contract pricing. "
            "NOTE: for commodities, this is futures price only — official "
            "government spot price series (e.g. WTI crude spot) live on "
            "macro agent via FRED instead. Don't assume this is the only "
            "source for oil/gold/etc — check macro agent's description too."
        ),
    },
    "macro": {
        "agent":       macro_agent,
        "description": (
            "Answers questions about macroeconomic conditions and indicators. "
            "Has access to FRED series (Fed funds rate, unemployment, CPI, "
            "Treasury yields, GDP, M2, consumer sentiment, credit spreads, "
            "dollar index, commodity SPOT prices like DCOILWTICO for WTI "
            "crude) and World Bank data (GDP, inflation, population, "
            "trade) for US, China, India, UK, Germany, Japan, Brazil. "
            "Use for: monetary policy, inflation, economic cycles, "
            "interest rate environment, macro context for any time period, "
            "official commodity spot prices as economic indicators. "
            "NOTE: for oil/gold/etc, this is the official government SPOT "
            "series — market agent has the separate futures/contract price; "
            "they are not interchangeable."
        ),
    },
    "filings": {
        "agent":       filings_agent,
        "description": (
            "Answers qualitative questions using SEC filings, Wikipedia "
            "articles, and Fed communications. "
            "IMPORTANT: this is the ONLY agent that can retrieve what the "
            "Fed has SAID — use it for any question about FOMC statements, "
            "Fed minutes, Powell speeches, Fed governor speeches, Fed policy "
            "reasoning, Fed commentary on inflation or the economy, or any "
            "'what has the Fed said about X' question. Has get_fed_communications "
            "tool for targeted Fed document retrieval. "
            "Also has access to 10-K and 10-Q filings for AAPL, MSFT, "
            "GOOGL, AMZN, JPM, BAC, XOM, and Wikipedia articles on "
            "Inflation, Recession, Federal Reserve, Quantitative Easing, "
            "2008 financial crisis, COVID-19 recession, Silicon Valley Bank. "
            "Uses semantic search for meaning-based retrieval across all "
            "documents. Use for: Fed communications, business strategy, "
            "risk factors, company narratives, economic concept explanations, "
            "historical event analysis. "
            "Does NOT have news or insider trade tools — use sentiment agent for those."
        ),
    },
    "sentiment": {
        "agent":       sentiment_agent,
        "description": (
            "Specialist in alternative data and market sentiment signals. "
            "Use for: insider buying/selling activity (SEC Form 4), "
            "news sentiment and media coverage, pre-earnings alternative signals, "
            "combined insider + news reads. "
            "Data: Form 4 insider trades for AAPL/MSFT/GOOGL/AMZN/JPM/BAC/XOM. "
            "News sentiment for those tickers plus GLD/USO/TLT/SPY. "
            "NOT for: stock prices (use market), macro indicators (use macro), "
            "SEC filings text / MD&A / risk factors (use filings)."
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
