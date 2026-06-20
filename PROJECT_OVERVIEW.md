# Trade Platform — Project Overview

**Purpose:** A global economic intelligence platform on AWS. Ingests market data, macro indicators, SEC filings, Fed communications, news sentiment, and insider trades on automated schedules; exposes them through an Athena-backed query layer driven by a multi-agent Claude-powered system with conversation memory, semantic search, and self-critique guardrails.

**Stack:** Python · AWS CDK (Python) · AWS Glue (Python Shell) · S3 · Athena · DynamoDB · Secrets Manager · S3 Vectors (Cohere Embed v3) · Anthropic SDK

**Region:** `us-east-2`  |  **Dev account:** `197411402303`  |  **Env prefix:** `dev` / `prod`

> This doc is a factual current-state reference for coding work, not a design-decision log. Architecture/reasoning discussion happens elsewhere — keep edits here limited to "what exists and where," not "what should we do."

---

## Repository Layout

```
trade-platform/
├── app.py                              # CDK entry point
├── cdk.json
├── verify_pipeline.py                  # CLI: per-source pipeline health check + reset
├── trade_platform/
│   └── trade_platform_stack.py         # All infrastructure in one CDK stack (~1200 lines)
├── ingestion/
│   ├── configs/sources/                # One YAML per data source (14 total)
│   ├── etl/                            # Raw JSON → canonical Parquet
│   │   ├── etl_yfinance.py             # market_prices (ticker=/year= partitions)
│   │   ├── etl_companies.py            # companies reference table (full-refresh)
│   │   ├── etl_fred.py                 # economic_indicators (FRED)
│   │   ├── etl_worldbank.py            # economic_indicators (WorldBank)
│   │   ├── etl_sec.py                  # documents (10-K/10-Q raw text)
│   │   ├── etl_sec_prose.py            # documents_prose (section-level extraction)
│   │   ├── etl_wikipedia.py            # documents (wiki articles)
│   │   ├── etl_fedspeak.py             # documents (source=FEDSPEAK)
│   │   ├── etl_news.py                 # news (Polygon, per-article sentiment)
│   │   ├── etl_insiders.py             # insider_trades (SEC Form 4)
│   │   └── etl_embed.py                # documents/documents_prose → S3 Vectors
│   ├── scripts/                        # Glue job entry points (ingestion)
│   │   ├── ingest_yfinance.py          # chunked: month-chunks + separate metadata file
│   │   ├── ingest_fred.py / ingest_worldbank.py
│   │   ├── ingest_sec.py               # EDGAR, paginated
│   │   ├── ingest_wikipedia.py
│   │   ├── ingest_fedspeak.py          # FOMC statements/minutes/transcripts/speeches
│   │   ├── ingest_news.py              # Polygon News API
│   │   ├── ingest_insiders.py          # SEC Form 4 XML
│   │   └── ingest_{acled,comtrade,eia,imf,unctad,wto}.py   # written, NOT wired (no ETL/CDK)
│   ├── utils/
│   │   ├── config.py                   # load_source_config(), get_default_start()
│   │   ├── watermark.py                # DynamoDB watermarks + SEC S3-tracker helpers
│   │   ├── transform.py                # safe_float, to_json_str, row validators
│   │   ├── dates.py                    # current_date_str, subtract_days, resolve_dates
│   │   └── periods.py
│   └── requirements.txt
├── query/
│   ├── agent.py                        # CLI entry point — memory + orchestrator
│   ├── orchestrator.py                 # plan() → execute(), sync wrapper around async
│   ├── planner.py                      # LLM DAG planner (Haiku dev / Sonnet prod)
│   ├── dag_executor.py                 # asyncio parallel-round executor + synthesis
│   ├── sub_agents.py                   # MarketAgent, MacroAgent, FilingsAgent, SentimentAgent
│   ├── registry.py                     # AGENT_REGISTRY — single source of truth
│   ├── memory.py                       # DynamoDB-backed session memory
│   ├── api.py                          # 17 tool implementations
│   ├── tools.py                        # Anthropic tool schemas + get_registry()
│   ├── telemetry.py                    # Trace class — S3 JSON traces for Athena
│   ├── reflexion.py                    # critique() + apply_reflexion() self-check loop
│   ├── athena.py                       # Athena client wrapper (query(), error types)
│   └── config.py                       # get_client() — Anthropic client singleton
└── scripts/
    └── bootstrap_sp500.py               # One-shot: Wikipedia → EDGAR CIK mapping
```

---

## Architecture

```
External APIs (yfinance, FRED, SEC EDGAR, World Bank, Wikipedia, Fed/FOMC, Polygon News)
        │
        ▼
Glue Python Shell Jobs   ←─ Scheduled + CONDITIONAL chain triggers
(ingestion/scripts/)
        │  raw JSON, partitioned year=  (yfinance: month-chunked + separate metadata file)
        ▼
S3 Raw Buckets           {env}-trade-{source}-raw-{account}
        │
        ├──► Glue Crawler → Raw Athena tables
        ▼
ETL Jobs (ingestion/etl/)
        │  canonical Parquet
        ▼
S3 Processed Buckets     {env}-trade-{source}-processed-{account}
        │
        ├──► Glue Crawler → Processed Athena tables (bucket-root targets — auto-discovers new tables)
        ├──► etl_embed.py ──► Bedrock Cohere Embed v3 ──► S3 Vectors: documents-index
        ▼
Athena (query/athena.py)        S3 Vectors (query/api.py)
        │                              │
        └──────────────┬───────────────┘
                       ▼
              Query API — 17 tools (query/api.py + tools.py)
                       ▼
              Orchestrator (query/orchestrator.py)  →  plan() → execute()
                       │
              ┌────────▼────────┐
              │  DAG Planner    │  Haiku (dev) / Sonnet (prod)
              │  (planner.py)   │  JSON DAG + IMPLICIT DATE RESOLUTION rules
              └────────┬────────┘
                       │
              ┌────────▼─────────────────────────────────────┐
              │  DAG Executor (dag_executor.py) — asyncio     │
              │  Parallel agents get a role-scoping hint      │
              │  injected so they don't ask for clarification │
              │  about another agent's domain                 │
              │                                                │
              │  market | macro | filings | sentiment          │
              └────────┬─────────────────────────────────────┘
                       │  per-agent answers
                       ▼
              Synthesis (Haiku/Sonnet) — SYNTHESIS_SYSTEM tells it to
              note gaps in one sentence, never block on missing data
                       ▼
              Reflexion critic pass per agent (query/reflexion.py)
              sees FULL tool results (not the 1500-char S3 preview)
                       ▼
              DynamoDB Memory (query/memory.py) — TTL 30d
                       ▼
              User: grounded natural-language answer
```

---

## Infrastructure Layer (`trade_platform_stack.py`)

### S3 Buckets

Standard loop sources (`SOURCES = ["yfinance", "fred", "worldbank", "sec", "wikipedia"]`) get raw+processed buckets via a loop. `fedspeak`, `news`, `insiders`, `sec_prose` are wired individually (own buckets/DB/crawler/jobs, not part of the `SOURCES` loop). All buckets versioned, SSE-S3, SSL-enforced, `RETAIN` on destroy.

| Bucket pattern                             | Layer     |
|--------------------------------------------|-----------|
| `{env}-trade-{source}-raw-{account}`       | Raw       |
| `{env}-trade-{source}-processed-{account}` | Processed |
| `{env}-trade-vectors-{account}`            | Vectors   |
| `{env}-trade-llmops-{account}`             | Telemetry traces |

The `companies` table has **no separate bucket** — it lives at `s3://{env}-trade-yfinance-processed-{account}/companies/data.parquet`, inside the existing yfinance processed bucket/DB.

### Glue Catalog

One database per source per layer: `{env}_trade_{source}_{layer}`, e.g. `dev_trade_fred_processed`. `companies` is a table inside `dev_trade_yfinance_processed`, not its own DB.

### Glue Crawlers

Standard 5 sources (`PROCESSED_PATHS` dict in stack) target **bucket root** for the processed layer (`""`, not a subfolder) — this lets a crawler auto-discover new tables (e.g. `companies/`) without a CDK update. Raw layer still targets `year=` prefix per source. All run `cron(0 2 * * ? *)`, schema change LOG, recrawl CRAWL_EVERYTHING.

Separately-wired crawlers (own CfnCrawler blocks, NOT affected by `PROCESSED_PATHS`): `sec_prose`, `fedspeak`, `news`, `insiders`, `llmops` — each targets its single known table prefix directly.

### Glue Jobs & Schedules

| Source     | Ingest schedule             | Notes                                          |
|------------|------------------------------|-------------------------------------------------|
| yfinance   | `0 21 ? * MON-FRI *`        | After US close. Chunked: month-chunks + 5s inter-chunk delay (Yahoo rate-limit cooldown). Timeout override 60 min. |
| fred       | `0 6 1 * ? *`                | 1st of each month                              |
| worldbank  | `0 6 1 1 ? *`                 | Jan 1st (annual)                                |
| sec        | `0 6 1 1,4,7,10 ? *`          | Quarterly. Timeout override 60 min.            |
| wikipedia  | `0 6 ? * MON *`               | Weekly Monday                                   |
| fedspeak   | manual/event-driven           | Tied to FOMC calendar, not periodic            |
| news       | weekly (Monday)               | Polygon News, 90-day dev backfill              |
| insiders   | quarterly                     | SEC Form 4, same company universe as `sec`     |
| companies  | CONDITIONAL trigger only       | Fires after `yfinance` ETL succeeds — no own schedule |

All Python Shell jobs: Glue 3.0, `max_capacity=0.0625` (1/16 DPU). `--extra-py-files`: entire `ingestion/` dir zipped as CDK S3 asset.
`ADDITIONAL_MODULES_OVERRIDE["yfinance"]` adds `python-dateutil>=2.8.0` explicitly (month-chunking uses `relativedelta`) on top of the base `yfinance, fredapi, wbdata, pyyaml` module set.

ETL jobs share `ETL_MODULES_BASE = "pandas==2.0.3,pyarrow==14.0.2,pyyaml>=6.0.0"`, with per-job overrides for `sec_prose` (+edgartools) and `embed` (+boto3).

### DynamoDB

| Table                                    | PK             | SK          | Notes                        |
|------------------------------------------|----------------|-------------|------------------------------|
| `trade-platform-{env}-watermarks`        | `source_name`  | `dataset_name` | Per-ticker/series last-ingested date |
| `trade-platform-{env}-conversations`     | `session_id`   | `timestamp` | Agent memory, TTL 30 days    |

SEC and insiders use S3 tracker files instead of DynamoDB watermarks (`tracker/sec_{ticker}_tracker.json`, `tracker/{ticker}.json`) — accession lists grow too large for a DynamoDB item.

### S3 Vectors

- Index: `documents-index` — float32, 1024 dims, cosine. CDK logical ID `VectorsIndexV2`.
- Model: `cohere.embed-english-v3` (us-east-1).
- `semantic_search` source filter accepts `EDGAR | WIKIPEDIA | FEDSPEAK`.

### LLMOps

- Bucket: `{env}-trade-llmops-{account}`, prefix `traces/year=/month=/`.
- `Trace` (telemetry.py) stores a `result_full` field on each tool-call dict **in memory only** — stripped before the S3 write in `flush()`. `result_preview` (1500 chars, unchanged) is what's actually persisted to S3/Athena.

---

## Ingestion Pipeline (`ingestion/`)

### Active Sources

| Source     | Frequency  | Dev data items                                                |
|------------|------------|-----------------------------------------------------------------|
| yfinance   | daily      | 527 tickers: S&P 500 equities + global indices + FX + commodities/futures |
| fred       | monthly/daily | ~22 series incl. FEDFUNDS, UNRATE, CPIAUCSL, DGS10, DGS2, GDP, T10Y2Y, BAMLH0A0HYM2, DTWEXBGS, DEXUSEU/JPUS/UK/INUS/CHUS, GOLDAMGBD228NLBM, DCOILWTICO |
| worldbank  | annual     | 5 indicators × 7 countries (US, CN, IN, GB, DE, JP, BR)       |
| sec        | quarterly  | 7 companies: AAPL, MSFT, GOOGL, AMZN, JPM, BAC, XOM           |
| wikipedia  | weekly     | ~10 topics (Inflation, Recession, Federal_Reserve, Quantitative_easing, 2008_financial_crisis, COVID-19_recession, Silicon_Valley_Bank, …) |
| fedspeak   | event-driven | FOMC statements/minutes/transcripts + governor speeches      |
| news       | weekly     | Polygon News, same 7 SEC companies + GLD/USO/TLT/SPY          |
| insiders   | quarterly  | SEC Form 4, same 7-company universe as `sec`                 |

**Not wired** (script + YAML config exist, no ETL/processed schema/CDK job): `acled`, `comtrade`, `eia`, `imf`, `unctad`, `wto`.

### yfinance Ingestion — Chunking Pattern (`ingest_yfinance.py`)

Fixed an OOM crash (Glue Python Shell, ~512MB ceiling) caused by (1) per-ticker metadata duplicated onto every OHLCV row, and (2) one unbounded `yf.download()` across the entire backfill range.

- `fetch_all_ticker_metadata()` — fetches `sector/market_cap/company_name/...` **once per run**, written to its own file: `{SOURCE}_metadata_{timestamp}.json` (no `_chunk` suffix).
- `_month_chunks(start, end)` — splits the backfill into calendar-month `(start, end)` tuples via `dateutil.relativedelta`. Chunk *size* stays constant regardless of backfill depth.
- Each chunk writes its own file: `{SOURCE}_{chunk_start}_{chunk_end}_{timestamp}_chunk{NNN}.json`, and updates per-ticker watermarks **immediately after that chunk** (partial-run recovery — a crash on chunk 40/78 doesn't lose chunks 1-39's watermarks).
- 5-second `time.sleep()` between chunks (not after the last one) — Yahoo rate-limits rapid back-to-back `yf.download()` calls; confirmed via testing (131/479 tickers failed in one un-delayed test run).
- `fetch_ticker_meta()` itself (the actual `yf.Ticker().info`/`fast_info` calls) is unchanged.

**Downstream**: `etl_yfinance.py`'s `read_latest_raw()` finds the latest run's timestamp from chunk filenames (`_parse_chunk_filename()`) and reads **all** chunk files sharing it, excluding the metadata file. `etl_companies.py`'s `read_latest_raw()` does the inverse — finds the single `_metadata_` file and reads only that.

### Watermark Pattern

`get_watermark(source, dataset)` / `update_watermark(...)`. Incremental: `start = oldest_watermark_across_all_tickers - max_lookback_days`. Resolved **once** before any chunking begins (`_resolve_start()` in `ingest_yfinance.py`) — chunking happens after the range is fixed, not as part of resolving it.

### EDGAR Pagination (`ingest_sec.py`)

`fetch_submissions()` merges continuation pages for companies with >1000 filings. 150ms delay between requests.

---

## ETL Layer (`ingestion/etl/`)

### market_prices (yfinance processed)

**Partition: `ticker=` / `year=`** (sanitized ticker value in the path — `^GSPC→GSPC`, `CL=F→CL_F`, `BRK-B→BRK_B`). Changed from the old `year=`/`exchange=` layout for ~16x cheaper single-ticker queries.

```
ticker_symbol  STRING    data column — ORIGINAL value (^GSPC, EURUSD=X, BRK-B)
                         NOTE: 'ticker' exists ONLY as the partition (sanitized) —
                         a Parquet data column can't share a name with a partition
                         column (HIVE_INVALID_METADATA). query/api.py SELECTs
                         'ticker_symbol AS ticker' to recover the display value.
exchange     STRING    NYSE | NASDAQ | INDEX | FX | FUTURES | LSE | NSE (data column, not partition)
date         STRING    YYYY-MM-DD
year         INTEGER   partition column — dropped from Parquet, inferred from path
country/currency/sector/industry  STRING
open/high/low/close/adj_close     DOUBLE
volume       DOUBLE    NULL for FX
source       STRING    "yfinance"
metadata     STRING    JSON blob (instrument_type, adj_close_note)
ingested_at  STRING
```

### companies (yfinance processed — full refresh, no partitions)

Single file `companies/data.parquet`, deleted and rewritten every run (`etl_companies.py`). Built from the separate metadata file (one row per ticker already — `deduplicate()` keeps latest `ingested_at` as a safety net, not strictly needed since metadata is already 1-row-per-ticker).

```
ticker, company_name, sector, industry, exchange, country, currency, city, state   STRING
market_cap, beta, dividend_yield, pe_ratio, forward_pe, week52_high, week52_low    DOUBLE
employees, avg_volume_10d, avg_volume_3m   INTEGER/BIGINT
description   STRING (truncated 500 chars)
sp500         BOOLEAN  — cross-referenced against sec_sp500.yaml
ingested_at   STRING
```

Tool: `get_companies_in_sector(sector, industry?, min_market_cap?, sp500_only?)` — sector/industry discovery step before `get_prices_multi`/`get_insider_summary`/`get_news_summary`. Returns a `Tickers: ...` line meant to be piped directly into a multi-ticker tool.

### economic_indicators (fred / worldbank processed)

```
indicator_id, indicator_name, country   STRING
date          STRING  observation date
vintage_date  STRING  FRED revision date — FRED is revision-only-APPEND, never deletes old vintages
year          INTEGER partition key
value         DOUBLE
unit          STRING
frequency     STRING  daily | weekly | monthly | quarterly | annual
ingested_at   STRING
```

**Vintage deduplication**: `_indicator_agg_sql()` in `query/api.py` wraps every query in a `ROW_NUMBER() OVER (PARTITION BY indicator_id, country, date ORDER BY vintage_date DESC) ... WHERE rn = 1` subquery BEFORE any date-range filter, GROUP BY, or passthrough — applies to both the raw passthrough branch and the annual/quarterly/monthly/weekly aggregation branches identically. Without this, a series with 2+ vintage files for overlapping dates returns duplicated/double-counted rows.

**Native frequency lookup**: `get_indicator()`/`get_indicator_multi()` call `_get_native_frequency(series_id, db)`, which reads the `frequency` column directly off the data instead of assuming `"monthly"` for every series. `get_indicator_multi()` resolves granularity **per series inside the loop** (not once, shared, before the loop) — a mixed list like `["FEDFUNDS", "DCOILWTICO"]` correctly gets `monthly` for one and `daily`/`weekly` for the other in the same call.

**Granularity branches** in `_indicator_agg_sql()`: `annual` / `quarterly` / `monthly` / `weekly` (DATE_TRUNC aggregation with AVG) / else=`daily` (raw passthrough, no aggregation). The `weekly` branch was missing until recently — `_indicator_granularity()`'s daily-native branch can return `"weekly"` for 31-365 day ranges, and before the native-frequency fix this value was unreachable in practice (the always-`"monthly"` hardcode meant only `monthly`/`quarterly` ever got produced), so the gap was invisible until then.

**`etl_fred.py`**: `SERIES_FREQUENCY_OVERRIDE` dict — checked BEFORE the fuzzy `FREQ_HINTS` label-substring matching, for series where the substring approach gives a wrong answer (e.g. `usd_eur_rate` label incorrectly fuzzy-matches the `"rate": "monthly"` hint, but `DEXUSEU` is actually daily). Covers `DGS10, DGS2, DTWEXBGS, DEXUSEU, DEXJPUS, DEXUSUK, DEXINUS, DEXCHUS, GOLDAMGBD228NLBM, DCOILWTICO, BAMLH0A0HYM2, T10Y2Y` — all daily.

### documents (sec / wikipedia / fedspeak processed)

```
doc_id, source, title, entity, doc_type, doc_date, text, char_count, year, ingested_at
source: EDGAR | WIKIPEDIA | FEDSPEAK
```

### documents_prose (sec_prose processed) — section-level 10-K/10-Q extraction
```
doc_id, entity, form_type, filed_date, section_name, section_title, text, char_count, extraction_method
```

### news (Polygon processed)
```
headline, description, publisher, publisher_tier (1=wire,2=established,3=opinion), sentiment, sentiment_reasoning, published_at, article_url, keywords, primary_ticker
```

### insider_trades (SEC Form 4 processed)
```
ticker, filer_name, filer_role, transaction_date, transaction_type (P/S/A/D/F/M/X/G/J), shares, price_per_share, value_usd, ownership_type, shares_owned_after
```

### Embedding ETL (`etl_embed.py`)

EDGAR: split on `Item X.` headers, 400 tok max/50 overlap. Wikipedia: split on `\n\n`, 350 tok max/30 overlap. Both Cohere v3, `search_document` at index time. Batches of 500 to `s3vectors.put_vectors`.

---

## Query / Agent Layer (`query/`)

### Tool API (`api.py` + `tools.py`) — 17 tools

| Category | Tools |
|---|---|
| Price | `get_prices`, `get_prices_multi`, `get_prices_by_sector`, `get_price_on_date`, `get_prices_on_date` |
| Companies | `get_companies_in_sector` |
| Indicator | `get_indicator`, `get_indicator_multi`, `get_indicator_on_date`, `get_macro_snapshot` |
| Documents | `get_documents`, `get_prose`, `semantic_search`, `get_fed_communications` |
| Sentiment | `get_news`, `get_news_summary`, `get_insider_trades`, `get_insider_summary` |

`get_fed_communications` is the only tool targeting FedSpeak directly (`doc_type`, `entity=speaker-or-FOMC`, date range). `semantic_search(source="FEDSPEAK")` covers broad concept search across the same documents.

### Self-Critique Guardrails

| Guardrail | Mechanism |
|-----------|-----------|
| Tool dedup | `call_sig = f"{name}:{json.dumps(inputs, sort_keys=True)}"` — repeat calls short-circuit |
| Token budget | Per-agent budget (Market/Macro 50k, Filings 150k, Sentiment 75k); partial answer on overflow |
| Telemetry | `Trace` (telemetry.py) → S3 `traces/year=/month=/`. `result_preview` (1500 char) is what's persisted — unchanged. |
| Reflexion | Haiku critic (`reflexion.py`) checks grounding; now reads **`result_full`** (untruncated, in-memory-only field on the same `tools_called` dict) instead of `result_preview` — fixed a bug where any tool result over 1500 chars got silently truncated before the critic ever saw it, causing false "hallucinated/missing data" verdicts on correct answers. `result_full` is stripped in `Trace.flush()` before the S3 write — persisted shape unchanged. |

### Multi-Agent Architecture

```
agent.py → orchestrator.py → planner.py (Haiku/Sonnet, JSON DAG)
                                   ↓
                          dag_executor.py (asyncio)
              ┌──────────┬──────────┬──────────────┐
              ▼          ▼          ▼              ▼
          MarketAgent MacroAgent FilingsAgent  SentimentAgent
```

| Agent | Tools | Notes |
|---|---|---|
| MarketAgent | get_prices, get_prices_multi, get_prices_by_sector, get_price_on_date, get_prices_on_date, get_companies_in_sector | |
| MacroAgent | get_indicator, get_indicator_multi, get_indicator_on_date, get_macro_snapshot | No filings/Fed-doc/sentiment tools |
| FilingsAgent | get_fed_communications, get_documents, get_prose, semantic_search, get_prices, get_macro_snapshot, get_companies_in_sector | ONLY agent with Fed document tools. Has scratchpad reasoning protocol + batching rules for `get_prose` |
| SentimentAgent | get_news, get_news_summary, get_insider_trades, get_insider_summary, get_prices, get_price_on_date, get_companies_in_sector | Strict P/S/F transaction-type interpretation rules baked into system prompt (F=tax withholding≠selling, S=often 10b5-1≠bearish, net=P−S only) |

All four share `_run_agent()` (ReAct loop, model `claude-sonnet-4-6`). Each agent's system prompt is now an f-string starting with `Today's date is {datetime.date.today().isoformat()}. ... data available from 2020-01-01 to present.` so agents don't reject genuinely-available recent dates as "future" or "unavailable."

**Planner** (`planner.py`): includes an **IMPLICIT DATE RESOLUTION** section — maps relative phrases ("recently", "before earnings", "this quarter", no time reference at all) to default windows per data type, so the planner resolves dates itself rather than asking the user. Also resolves agent routing via explicit content rules (e.g. Fed communications → filings never macro; insider/news keywords → sentiment never filings).

**Executor** (`dag_executor.py`): when a round has 2+ agents with no `depends_on`, each gets a role-scoping hint appended to its question (`ROLE_DESCRIPTIONS` dict) so it doesn't try to answer outside its domain or ask for clarification about another agent's data. Synthesis call uses a `SYNTHESIS_SYSTEM` prompt instructing it to note partial-data gaps in one sentence rather than blocking the answer.

### CLI (`agent.py`)

```
python query/agent.py --question "..." --no-memory
python query/agent.py --session "q3-analysis" --question "..."
python query/agent.py --session "q3-analysis"               # interactive REPL
python query/agent.py --list-sessions / --clear-session "..."
```

---

## verify_pipeline.py

Per-source health check + `--reset`/`--fire` CLI. `SOURCE_CONFIG` dict has an entry per source; sources with `raw_bucket: None` (derived sources with no raw layer of their own — currently `companies`, which is derived from `yfinance`'s raw data) get the raw-bucket check, watermark/tracker check, and reset's raw-bucket-clear step all guarded with `if cfg.get("raw_bucket"): ... else: print("N/A")` — only the processed-bucket/crawler/ETL-job/Athena checks run unconditionally.

---

## Key Configuration Values

| Config                            | Value                                                         |
|-----------------------------------|---------------------------------------------------------------|
| AWS region                        | `us-east-2`                                                   |
| Dev account ID                    | `197411402303`                                                |
| Embed model                       | `cohere.embed-english-v3` (us-east-1)                        |
| Vector dimensions                 | 1024                                                          |
| Glue version / Python Shell DPU   | 3.0 / 0.0625 (1/16)                                          |
| yfinance ingestion timeout        | 60 min (override; chunked run measured ~25 min for full 2023→today backfill, 527 tickers) |
| yfinance inter-chunk delay        | 5 seconds                                                     |
| Dev data start date               | 2020-01-01 (most sources) / 2023-01-01 (yfinance, insiders)  |
| Agent model (specialists)         | `claude-sonnet-4-6`                                           |
| Planner/synthesis/critic model    | `claude-haiku-4-5-20251001` (dev) / `claude-sonnet-4-6` (prod) |
| Token budget — Market/Macro       | 50,000                                                        |
| Token budget — Filings            | 150,000                                                       |
| Token budget — Sentiment          | 75,000                                                        |
| Telemetry truncation              | `result_preview` 1500 chars (S3-persisted); `result_full` untruncated (in-memory only, reflexion input) |
| Session memory TTL                | 30 days                                                       |

---

## Known Gaps (factual, not decisions-pending)

1. **6 sources unwired**: `acled`, `comtrade`, `eia`, `imf`, `unctad`, `wto` — ingestion scripts + YAML configs exist, no ETL/processed schema/CDK job.
2. **SQL built via f-string interpolation** of LLM-supplied tool args throughout `query/api.py`. Safe only because callers are constrained by Anthropic tool schemas, not raw user HTTP input — would need an allowlist/sanitizer before any HTTP-facing surface.
3. **No HTTP serving surface** — CLI only (`query/agent.py`).
4. **`sec_sp500.yaml` appears unpopulated** in dev as of last check — `companies.sp500` flag may read as all-`False`; not yet root-caused.
5. **Old wrong-vintage rows not purged** — the vintage-dedup fix (`_indicator_agg_sql`) makes queries correctly ignore stale vintages, but the stale rows themselves are still in S3/Parquet for any series ETL'd before the `etl_fred.py` frequency fix. Cleanup is a deliberate separate decision (could affect revision-history integrity if done carelessly).
6. **Yahoo rate-limiting risk scales with chunk count** — the 5s inter-chunk delay was verified to help on a 3-chunk test; a full 42-chunk backfill firing chunks this tightly could still hit Yahoo's limiter more than the 3-chunk test did. No retry-failed-tickers mechanism exists yet.
