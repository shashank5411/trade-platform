# Trade Platform — Project Overview

**Purpose:** A global economic intelligence platform on AWS. Ingests market data, macro indicators, SEC filings, and reference documents on automated schedules; exposes them through an Athena-backed query layer driven by a multi-agent Claude-powered system with conversation memory and semantic search.

**Stack:** Python · AWS CDK (Python) · AWS Glue (Python Shell) · S3 · Athena · DynamoDB · Secrets Manager · S3 Vectors (Cohere Embed v3) · Anthropic SDK

**Region:** `us-east-2`  |  **Dev account:** `197411402303`  |  **Env prefix:** `dev` / `prod`

---

## Repository Layout

```
trade-platform/
├── app.py                              # CDK entry point
├── cdk.json
├── athena_config.json                  # Athena results bucket (dev)
├── requirements.txt                    # CDK deps
├── trade_platform/
│   └── trade_platform_stack.py         # All infrastructure in one CDK stack
├── ingestion/
│   ├── configs/sources/                # One YAML per data source
│   ├── etl/                            # ETL scripts: raw JSON → canonical Parquet
│   │   ├── __init__.py
│   │   ├── etl_yfinance.py
│   │   ├── etl_fred.py
│   │   ├── etl_sec.py
│   │   ├── etl_wikipedia.py
│   │   ├── etl_worldbank.py
│   │   └── etl_embed.py                # Phase 5 — documents → S3 Vectors
│   ├── scripts/                        # Glue job entry points
│   │   ├── ingest_yfinance.py
│   │   ├── ingest_fred.py
│   │   ├── ingest_sec.py
│   │   ├── ingest_wikipedia.py
│   │   └── ingest_worldbank.py
│   ├── utils/
│   │   ├── config.py                   # load_source_config() — parses YAML
│   │   ├── watermark.py                # DynamoDB incremental-ingestion tracking
│   │   ├── transform.py                # Row validation, safe_float, to_json_str
│   │   ├── dates.py                    # Date arithmetic
│   │   └── periods.py                  # Year/month period helpers
│   └── requirements.txt
├── query/
│   ├── __init__.py                     # re-exports run_question as run
│   ├── agent.py                        # CLI entry point — memory + orchestrator
│   ├── orchestrator.py                 # Thin wrapper: plan → execute
│   ├── planner.py                      # LLM DAG planner (Haiku dev / Sonnet prod)
│   ├── dag_executor.py                 # Async parallel round executor
│   ├── sub_agents.py                   # MarketAgent, MacroAgent, FilingsAgent
│   ├── registry.py                     # Agent registry — single source of truth
│   ├── memory.py                       # DynamoDB-backed session memory
│   ├── api.py                          # 10 tool implementations + semantic_search
│   ├── tools.py                        # Anthropic tool schemas + registry
│   └── athena.py                       # Athena client wrapper
└── scripts/
    └── bootstrap_sp500.py              # One-shot: Wikipedia → EDGAR CIK mapping
```

---

## Architecture

```
External APIs
(yfinance, FRED, SEC EDGAR, World Bank, Wikipedia, …)
        │
        ▼
Glue Python Shell Jobs   ←─ Scheduled triggers (see table below)
(ingestion/scripts/)
        │  raw JSON/CSV, partitioned year=
        ▼
S3 Raw Buckets           {env}-trade-{source}-raw-{account}
        │
        ├──► Glue Crawler (2 AM UTC daily) → Raw Athena tables
        │
        ▼
ETL Jobs (ingestion/etl/)
        │  canonical Parquet, partitioned by schema
        ▼
S3 Processed Buckets     {env}-trade-{source}-processed-{account}
        │
        ├──► Glue Crawler (2 AM UTC daily) → Processed Athena tables
        │
        ├──► etl_embed.py ──► Bedrock Cohere Embed v3
        │                            │  float32 vectors, 1024 dim
        │                            ▼
        │                   S3 Vectors: documents-index (cosine)
        │
        ▼
Athena (query/athena.py)        S3 Vectors (query/api.py)
        │                              │
        └──────────────┬───────────────┘
                       ▼
              Query API (query/api.py)
              10 tools: prices, indicators, documents, semantic_search
                       │
                       ▼
              Orchestrator (query/orchestrator.py)
              plan() → execute()
                       │
              ┌────────▼────────┐
              │  DAG Planner    │  Haiku (dev) / Sonnet (prod)
              │  (planner.py)   │  produces JSON DAG with depends_on
              └────────┬────────┘
                       │
              ┌────────▼────────────────────────┐
              │  DAG Executor (dag_executor.py)  │
              │  asyncio — parallel rounds       │
              │                                  │
              │  Round 1: [macro] ──────────────►│
              │  Round 2: [market] [filings] ───►│  (parallel)
              └────────┬────────────────────────┘
                       │  per-agent answers
                       ▼
              Synthesis (Haiku/Sonnet)
                       │
                       ▼
              DynamoDB Memory (query/memory.py)
              session_id + timestamp, TTL 30d
                       │
                       ▼
              User: grounded natural-language answer
```

---

## Infrastructure Layer (`trade_platform_stack.py`)

### S3 Buckets

Two layers × five sources = **10 buckets**, plus one vectors bucket. All versioned, SSE-S3, SSL-enforced, `RETAIN` on destroy.

| Bucket pattern                             | Layer     |
|--------------------------------------------|-----------|
| `{env}-trade-{source}-raw-{account}`       | Raw       |
| `{env}-trade-{source}-processed-{account}` | Processed |
| `{env}-trade-vectors-{account}`            | Vectors   |

Sources: `yfinance`, `fred`, `worldbank`, `sec`, `wikipedia`

### Glue Catalog

One database per source per layer: `{env}_trade_{source}_{layer}`  
e.g. `dev_trade_fred_processed`

### Glue Crawlers

10 crawlers (raw + processed × 5 sources). All run on `cron(0 2 * * ? *)` (2 AM UTC).  
Schema change policy: LOG. Recrawl: CRAWL_EVERYTHING.

### Glue Jobs & Schedules

| Source     | Schedule (cron)             | Notes                       |
|------------|-----------------------------|-----------------------------|
| yfinance   | `0 21 ? * MON-FRI *`        | After US market close       |
| fred       | `0 6 1 * ? *`               | 1st of each month           |
| worldbank  | `0 6 1 1 ? *`               | Jan 1st (annual data)       |
| sec        | `0 6 1 1,4,7,10 ? *`        | Quarterly                   |
| wikipedia  | `0 6 ? * MON *`             | Weekly Monday               |

All jobs: Python Shell, Glue 3.0, `max_capacity=0.0625` (1/16 DPU), `timeout=30` min.  
`--extra-py-files`: entire `ingestion/` dir zipped as CDK S3 asset.  
`--additional-python-modules`: `yfinance`, `fredapi`, `wbdata`, `pyyaml`

### DynamoDB

| Table                                    | PK             | SK          | Notes                        |
|------------------------------------------|----------------|-------------|------------------------------|
| `trade-platform-{env}-watermarks`        | `source_name`  | `dataset_name` | Tracks last ingested date per ticker |
| `trade-platform-{env}-conversations`     | `session_id`   | `timestamp` | Agent memory, TTL 30 days    |

### S3 Vectors (Phase 5)

- Bucket: `{env}-trade-vectors-{account}`
- Index: `documents-index` — float32, **1024 dims**, cosine distance
- CDK logical ID: `VectorsIndexV2` (renamed from `VectorsIndex` when dimension changed from 1536→1024; S3 Vectors does not support in-place dimension updates, so renaming the logical ID forces CDK to delete + recreate)
- Model: `cohere.embed-english-v3` (us-east-1) — switched from Titan Embed v2 (1536d) due to account-level throttling on new accounts. Titan support ticket open.
- `input_type`: `search_document` at index time (etl_embed.py), `search_query` at retrieval time (api.py) — Cohere v3 optimizes vectors differently per use, improving retrieval quality.
- IAM: `s3vectors:PutVectors/GetVectors/QueryVectors/ListVectors/DeleteVectors` + both Cohere and Titan Embed ARNs on job role

### LLMOps (Phase 6)

- Bucket: `{env}-trade-llmops-{account}` — agent execution traces as JSON
- S3 prefix: `traces/year=/month=/` — Hive-partitioned for Athena
- Glue DB: `{env}_trade_llmops`, Crawler runs at 3 AM UTC daily
- IAM: `grant_read_write` on job role, `grant_read` on crawler role

### Secrets Manager

`trade-platform/{env}/fred-api-key` — set manually via `aws secretsmanager put-secret-value`

### IAM Roles

- **GlueCrawlerRole**: `AWSGlueServiceRole` + read all S3 buckets
- **GlueIngestionJobRole**: `AWSGlueServiceRole` + read/write all buckets + DynamoDB watermarks + conversation table + Secrets Manager FRED key + Bedrock `InvokeModel` (Cohere Embed v3 + Titan Embed v2 ARNs) + S3 Vectors full access + LLMOps bucket read/write

---

## Ingestion Pipeline (`ingestion/`)

### Source Config YAML

Each source has `ingestion/configs/sources/{source}.yaml`:

```yaml
default_start_date:
  dev:  "2020-01-01"
  prod: "2000-01-01"
frequency: daily          # daily | monthly | quarterly | annual
max_lookback_days: 30     # overlap window for late data
tickers: [...]            # or companies, indicators, series, etc.
```

### Active Sources

| Source     | Frequency  | Dev data items                                                |
|------------|------------|---------------------------------------------------------------|
| yfinance   | daily      | 63 tickers: US equities, global indices, FX pairs, commodities, futures |
| fred       | monthly    | 13 series: FEDFUNDS, UNRATE, CPIAUCSL, DGS10, DGS2, GDP, M2SL, T10Y2Y, BAMLH0A0HYM2, DTWEXBGS, DEXUSEU, DEXJPUS, UMCSENT |
| worldbank  | annual     | 5 indicators × 7 countries (US, CN, IN, GB, DE, JP, BR)       |
| sec        | quarterly  | 7 companies: AAPL, MSFT, GOOGL, AMZN, JPM, BAC, XOM           |
| wikipedia  | weekly     | 10 topics: Inflation, Recession, Federal_Reserve, Quantitative_easing, 2008_financial_crisis, COVID-19_recession, Silicon_Valley_Bank, … |

### Watermark Pattern

`get_watermark(source, dataset)` reads DynamoDB; `update_watermark(...)` writes after success.  
Incremental: each run resolves `start = oldest_watermark - max_lookback_days`.

### EDGAR Pagination (`ingest_sec.py`)

`fetch_submissions()` checks `filings.files[]` and merges continuation pages for companies with >1000 filings.  
Rate: 150 ms delay between requests (safely under 10 req/s EDGAR limit).

---

## ETL Layer (`ingestion/etl/`)

### Canonical Schemas

**market_prices** (yfinance processed, Parquet partitioned by `year=` / `exchange=`):
```
ticker       STRING    original value (^GSPC, EURUSD=X)
exchange     STRING    NYSE | NASDAQ | INDEX | FX | FUTURES | LSE | NSE
date         STRING    YYYY-MM-DD
year         INTEGER   partition key
country      STRING    US | GB | IN | …
currency     STRING    USD | GBp | INR | FX
open/high/low/close/adj_close  DOUBLE
volume       DOUBLE    NULL for FX
source       STRING    "yfinance"
metadata     STRING    JSON blob (instrument_type, adj_close_note)
ingested_at  STRING    ISO UTC timestamp
```

**economic_indicators** (fred / worldbank processed):
```
indicator_id    STRING    FEDFUNDS | NY.GDP.MKTP.CD | …
indicator_name  STRING    human-readable
date            STRING    YYYY-MM-DD (observation date)
vintage_date    STRING    YYYY-MM-DD (FRED revision date)
year            INTEGER   partition key
value           DOUBLE
unit            STRING
frequency       STRING    daily | monthly | quarterly | annual
country         STRING
ingested_at     STRING
```

**documents** (sec / wikipedia processed, partitioned by `year=`):
```
doc_id      STRING    SHA-256 of (source+entity+date+type)
source      STRING    EDGAR | WIKIPEDIA
title       STRING
entity      STRING    ticker or Wikipedia title
doc_type    STRING    10-K | 10-Q | wiki_article
doc_date    STRING    filing date or article date
text        STRING    full text content
char_count  INTEGER
year        INTEGER   partition key
ingested_at STRING
```

### Exchange Resolution

`ingest_yfinance.py` fetches `yf.Ticker(ticker).fast_info` per ticker at ingest time, stamps `exchange` + `currency` onto each raw record using `EXCHANGE_NORMALIZE` (NYQ→NYSE, NMS→NASDAQ, etc.).  
`etl_yfinance.py` reads those fields with instrument-type overrides (INDEX/FX/FUTURES) and a NASDAQ fallback for old raw files without the field.

### Embedding ETL (`etl_embed.py`)

Runs after `etl_sec.py` and `etl_wikipedia.py`. For each processed document row:

1. **Chunk** — EDGAR: split on `Item X.` section headers (max 400 tokens, 50 overlap). Wikipedia: split on `\n\n` paragraphs (max 350 tokens, 30 overlap). Both fall back to sliding window. Sizes chosen to stay under Cohere v3's 512-token limit per call.
2. **Embed** — Bedrock `invoke_model` with `cohere.embed-english-v3`, `input_type=search_document`, 1024 dims. Exponential backoff on throttle.
3. **Write** — `s3vectors.put_vectors` in batches of 500. Metadata per vector: `doc_id`, `source`, `entity`, `doc_type`, `doc_date`, `title`, `text[:500]` (retrieval preview).

---

## Query / Agent Layer (`query/`)

### Athena Client (`athena.py`)

- Submits SQL, polls for completion, reads CSV from S3
- Results bucket: `s3://dev-trade-athena-results-197411402303/`
- Database naming: `{env}_trade_{source}_processed`

### Tool API (`api.py` + `tools.py`) — 10 tools

| Tool                   | Inputs                               | Notes                                      |
|------------------------|--------------------------------------|--------------------------------------------|
| `get_prices`           | ticker, start, end, [exchange]       | Auto-granularity: ≤30d daily, ≤365d weekly, else monthly |
| `get_prices_multi`     | tickers[], start, end                | Single Athena query, summary per ticker    |
| `get_price_on_date`    | ticker, date                         | Nearest trading day ≤ date                 |
| `get_prices_on_date`   | tickers[], date                      | ROW_NUMBER() OVER PARTITION — one query    |
| `get_indicator`        | series_id, start, end, [country]     | Auto-granularity respects native frequency |
| `get_indicator_multi`  | series_ids[], start, end             | FRED and WorldBank queried separately, merged |
| `get_indicator_on_date`| series_ids[], date, [countries]      | Nearest obs ≤ date per series; staleness flags |
| `get_macro_snapshot`   | as_of_date                           | Multi-indicator snapshot: rates, inflation, yields, GDP, SPY, VIX, Gold, Oil |
| `get_documents`        | entity, [doc_type, start, end, limit]| Returns first 2000 chars of matching filings |
| `semantic_search`      | query, [top_k, source, entity]       | Embed via Cohere v3 (`search_query`) → S3 Vectors query_vectors → top-K results with scores |

### Phase 6 Guardrails (`query/telemetry.py`, `query/reflexion.py`)

Every specialist agent run now includes:

| Guardrail | Mechanism |
|-----------|-----------|
| **Tool deduplication** | `call_sig = f"{tool_name}:{json.dumps(inputs, sort_keys=True)}"` — identical calls return a short-circuit message instead of re-fetching |
| **Token budget** | 50,000 input+output tokens per agent run; stops early with partial answer if exceeded |
| **Telemetry** | `Trace` object accumulates iterations, tool calls, token counts, flags; `flush()` writes JSON to `s3://{env}-trade-llmops-{account}/traces/year=/month=/` |
| **Reflexion** | Haiku critic checks numbers are grounded in tool results; issues found → one retry with guidance injected; second failure → caveat appended to answer |

`session_id` threads from `agent.py` → `orchestrator.py` → `dag_executor.py` → each specialist's `run()` → `Trace`, so all traces for a session share the same session key.

### Multi-Agent Architecture

```
agent.py  ──►  orchestrator.py  ──►  planner.py      (Haiku/Sonnet)
                                          │  JSON DAG
                                          ▼
                                     dag_executor.py  (asyncio)
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                    MarketAgent     MacroAgent      FilingsAgent
                    (4 tools)       (4 tools)       (4 tools incl.
                                                     semantic_search)
```

**Planner** (`planner.py`): Haiku (dev) or Sonnet (prod) call that returns a JSON DAG specifying which agents are needed and their `depends_on` relationships. Includes conversation context for the last 2 exchanges so follow-up questions route correctly.

**Executor** (`dag_executor.py`): resolves rounds from the DAG, runs each round with `asyncio.gather` + `run_in_executor` (thread pool) for true parallelism. Agents in later rounds receive prior agents' answers as context. Single-agent DAGs skip synthesis entirely.

**Registry** (`registry.py`): maps agent names → instances + rich descriptions. The planner prompt is built from these descriptions. Adding a new agent = add class to `sub_agents.py` + entry in `registry.py`.

**Specialist Agents** (`sub_agents.py`):

| Agent        | Tool subset                                                  |
|--------------|--------------------------------------------------------------|
| MarketAgent  | get_prices, get_prices_multi, get_price_on_date, get_prices_on_date |
| MacroAgent   | get_indicator, get_indicator_multi, get_indicator_on_date, get_macro_snapshot |
| FilingsAgent | get_documents, semantic_search, get_prices, get_macro_snapshot |

All three share a single `_run_agent()` ReAct loop with model `claude-sonnet-4-6`.

### Memory (`memory.py`)

DynamoDB `trade-platform-{env}-conversations` table:
- `save_turn(session_id, role, content)` — writes with TTL 30 days
- `load_turns(session_id, max_turns=10)` — queries newest-first, reverses to chronological, returns `[{role, content}]` ready for the Anthropic messages array
- `list_sessions()` / `clear_session()` — for CLI management

### CLI (`agent.py`)

```
python query/agent.py --question "..." --no-memory          # single shot
python query/agent.py --session "q3-analysis" --question "..."  # with memory
python query/agent.py --session "q3-analysis"               # interactive REPL
python query/agent.py --list-sessions
python query/agent.py --clear-session "q3-analysis"
```

---

## Key Configuration Values

| Config                            | Value                                                         |
|-----------------------------------|---------------------------------------------------------------|
| AWS region                        | `us-east-2`                                                   |
| Dev account ID                    | `197411402303`                                                |
| Athena results bucket             | `s3://dev-trade-athena-results-197411402303/`                 |
| FRED secret path                  | `trade-platform/dev/fred-api-key`                             |
| Embed model                       | `cohere.embed-english-v3` (us-east-1)                        |
| Vector index name                 | `documents-index`                                             |
| Vector dimensions                 | 1024                                                          |
| Embed token limit                 | 512 tokens per chunk (~2000 chars)                            |
| EDGAR chunk size                  | 400 tokens max, 50 overlap                                    |
| Wikipedia chunk size              | 350 tokens max, 30 overlap                                    |
| Glue version                      | 3.0                                                           |
| Python Shell max_capacity         | 0.0625 DPU (1/16)                                             |
| Dev data start date               | 2020-01-01                                                    |
| Prod data start date              | 2000-01-01                                                    |
| Agent model (specialists)         | `claude-sonnet-4-6`                                           |
| Planner/synthesis model (dev)     | `claude-haiku-4-5-20251001`                                   |
| Planner/synthesis model (prod)    | `claude-sonnet-4-6`                                           |
| Reflexion/critic model (dev)      | `claude-haiku-4-5-20251001`                                   |
| Reflexion/critic model (prod)     | `claude-sonnet-4-6`                                           |
| Token budget per agent run        | 50,000 tokens                                                 |
| Session memory TTL                | 30 days                                                       |
| Max turns loaded per session      | 10 exchanges (20 DynamoDB items)                              |
| Telemetry S3 prefix               | `traces/year=/month=/`                                        |

---

## Open Architectural Decisions

### 1. Partition layout for market_prices
`etl_yfinance.py` partitions by `year=` / `exchange=`. No ticker-level pruning — Athena scans a full exchange-year file for any single-ticker query.  
**Decision needed:** Add `ticker=` as a third partition level, or keep flat and rely on Athena predicate pushdown?

### 2. etl_yfinance reads only the latest raw file
`read_latest_raw()` sorts S3 keys and takes `[-1]` — misses any backfill files.  
**Decision needed:** Process all unprocessed files (watermark-tracked) or keep single-latest pattern?

### 3. Remaining 6 sources (ACLED, Comtrade, EIA, IMF, UNCTAD, WTO)
Scripts exist but no ETL, processed schema, or CDK wiring.  
**Decision needed:** Priority order and canonical schema for each.

### 4. SQL injection at Athena boundaries
Tool functions build SQL via f-string interpolation of LLM-supplied values. Safe today because the LLM follows tool schemas, not raw user input.  
**Decision needed:** Add allowlist + regex sanitizer before exposing any HTTP endpoint.

### 5. Embedding pipeline scheduling
`etl_embed.py` runs manually or as a separate Glue job. No CDK trigger yet.  
**Decision needed:** Trigger `etl_embed` automatically after `etl_sec` / `etl_wikipedia` complete, or run on its own weekly schedule?

### 6. HTTP serving surface
CLI only today. No Lambda, API Gateway, or FastAPI wrapper.  
**Decision needed:** Streaming vs. batch response, auth model, whether to expose orchestrator or individual agents.

### 7. LLMOps Athena table
Traces land in S3; the Glue Crawler runs at 3 AM UTC. Until the first crawler run there is no Athena table.  
**Decision needed:** Run crawler on-demand after first deploy, or add a one-shot CfnTrigger with `type=ON_DEMAND` that fires at deploy time?

### 8. Reflexion scope
Reflexion currently runs on every agent answer, including short factual responses where the critic call is wasted latency.  
**Decision needed:** Add a token-count or answer-length threshold below which reflexion is skipped automatically?

### 9. Cohere Embed quota (monitoring)
Switched from Titan to Cohere v3 to avoid per-account throttling on new accounts. If Cohere rate limits are hit, exponential backoff in `embed_text()` handles up to 5 retries. `semantic_search()` returns a graceful fallback message when the vector index is empty.

### 10. Fan-out / parallel same-agent instances
Current registry supports one instance per agent type. For questions needing the same agent twice in parallel (e.g. "compare AAPL 10-K 2021 vs 2023"), a fan-out pattern would spin up parallel named instances.  
**Decision needed:** Implement when 2+ same-agent parallel calls become a common pattern.