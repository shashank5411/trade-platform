"""
Athena client — submit SQL, poll until done, return DataFrame.
All query functions in api.py use this. Never call Athena directly
from anywhere else.

Phase 8 additions:
  - LRU result cache (LA-5) — identical SQL returns cached DataFrame
  - Structured AthenaQueryError with sql, state, reason fields
    so sub-agents receive actionable error context
"""

import time
import hashlib
import boto3
import pandas as pd
from io import StringIO
from functools import lru_cache
from typing import Optional

REGION          = "us-east-2"
RESULTS_BUCKET  = "s3://dev-trade-athena-results-197411402303/"
DEFAULT_TIMEOUT = 60   # seconds
POLL_INTERVAL   = 1.5  # seconds between status checks

athena = boto3.client("athena", region_name=REGION)
s3     = boto3.client("s3",     region_name=REGION)


# ── Exceptions ─────────────────────────────────────────────────────────────

class AthenaError(Exception):
    """Legacy base — kept for backwards compatibility with api.py catches."""
    pass


class AthenaQueryError(AthenaError):
    """
    Structured query failure — carries enough context for the agent to
    understand what went wrong and decide how to recover.

    Attributes:
        sql:       The SQL that failed
        state:     FAILED or CANCELLED
        reason:    Athena's StateChangeReason string
        execution_id: Athena query execution ID for CloudWatch lookup
    """
    def __init__(self, sql: str, state: str, reason: str,
                 execution_id: str):
        self.sql          = sql
        self.state        = state
        self.reason       = reason
        self.execution_id = execution_id
        super().__init__(self._format())

    def _format(self) -> str:
        # Condense the SQL to first 200 chars so the error isn't huge
        sql_preview = self.sql.strip()[:200].replace("\n", " ")
        return (
            f"Athena query {self.state.lower()}.\n"
            f"Reason: {self.reason}\n"
            f"SQL preview: {sql_preview}...\n"
            f"Execution ID: {self.execution_id}"
        )

    def agent_message(self) -> str:
        """
        Concise message formatted for the agent's tool result.
        Tells the agent what failed and what to try instead.
        """
        reason_lower = self.reason.lower()

        if "table not found" in reason_lower or \
           "table or view" in reason_lower:
            hint = (
                "The table does not exist yet — the Glue crawler "
                "may not have run. Try a different data source or "
                "check that the ETL pipeline has completed."
            )
        elif "column" in reason_lower and "not found" in reason_lower:
            hint = (
                "A column referenced in the query does not exist. "
                "Check column names — the schema may differ from "
                "what was expected."
            )
        elif "partition" in reason_lower:
            hint = (
                "Partition pruning issue. The year range may be "
                "outside what has been ingested."
            )
        elif "no output" in reason_lower or "empty" in reason_lower:
            hint = "Query ran but returned no data for this filter."
        elif "cancelled" in self.state.lower():
            hint = "Query was cancelled — likely a timeout. Try a narrower date range."
        else:
            hint = (
                "Unexpected Athena error. Try narrowing the date "
                "range or simplifying the query."
            )

        return (
            f"[ATHENA ERROR] {self.state}: {self.reason}\n"
            f"Hint: {hint}\n"
            f"SQL: {self.sql.strip()[:300]}"
        )


class AthenaTimeout(AthenaError):
    pass


# ── LRU cache (LA-5) ───────────────────────────────────────────────────────
# Cache keyed on (sql_hash, database) — avoids re-running identical queries
# within a session. lru_cache requires hashable args so we hash the SQL string.
# maxsize=100 covers ~a full multi-turn session comfortably.

_cache: dict = {}
_CACHE_MAX = 100


def _cache_key(sql: str, database: str) -> str:
    return hashlib.md5(f"{database}::{sql}".encode()).hexdigest()


def _cache_get(sql: str, database: str) -> Optional[pd.DataFrame]:
    return _cache.get(_cache_key(sql, database))


def _cache_set(sql: str, database: str, df: pd.DataFrame) -> None:
    key = _cache_key(sql, database)
    if len(_cache) >= _CACHE_MAX:
        # Evict oldest entry (insertion-ordered dict, Python 3.7+)
        _cache.pop(next(iter(_cache)))
    _cache[key] = df


def clear_query_cache() -> None:
    """Clear the in-memory query cache. Useful between sessions."""
    _cache.clear()


# ── Core query function ────────────────────────────────────────────────────

def query(
    sql:      str,
    database: str,
    timeout:  int = DEFAULT_TIMEOUT,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Submit a SQL query to Athena, poll until complete, return DataFrame.

    Args:
        sql:       SQL string to execute
        database:  Glue database name e.g. dev_trade_fred_processed
        timeout:   Max seconds to wait before raising AthenaTimeout
        use_cache: Return cached result if available (default True)

    Returns:
        pandas DataFrame of results (empty DataFrame if no rows)

    Raises:
        AthenaQueryError: Query failed or was cancelled — structured error
                          with agent_message() for clean tool result
        AthenaTimeout:    Query did not complete within timeout seconds
    """
    # Cache hit
    if use_cache:
        cached = _cache_get(sql, database)
        if cached is not None:
            return cached.copy()  # return copy so callers can't mutate cache

    # Submit
    response = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": RESULTS_BUCKET},
    )
    execution_id = response["QueryExecutionId"]

    # Poll
    elapsed = 0
    while elapsed < timeout:
        status = athena.get_query_execution(
            QueryExecutionId=execution_id
        )["QueryExecution"]["Status"]

        state = status["State"]

        if state == "SUCCEEDED":
            df = _fetch_results(execution_id)
            if use_cache:
                _cache_set(sql, database, df)
            return df

        elif state in ("FAILED", "CANCELLED"):
            reason = status.get("StateChangeReason", "Unknown error")
            raise AthenaQueryError(
                sql=sql,
                state=state,
                reason=reason,
                execution_id=execution_id,
            )

        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    raise AthenaTimeout(
        f"Query {execution_id} did not complete in {timeout}s"
    )


def _fetch_results(execution_id: str) -> pd.DataFrame:
    """
    Fetch query results from S3 output location.
    Athena writes a CSV to S3 — read it directly.
    More efficient than paginating get_query_results for large outputs.
    """
    execution = athena.get_query_execution(
        QueryExecutionId=execution_id
    )["QueryExecution"]

    output_location = execution["ResultConfiguration"]["OutputLocation"]

    bucket = output_location.split("/")[2]
    key    = "/".join(output_location.split("/")[3:])

    obj  = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")

    if not body.strip():
        return pd.DataFrame()

    df = pd.read_csv(StringIO(body))
    return df


def query_to_str(sql: str, database: str) -> str:
    """
    Run query and return results as a formatted string.
    Used by agent.py to send results back to LLM as text.
    """
    df = query(sql, database)
    if df.empty:
        return "No results found."
    return df.to_string(index=False, max_rows=100)