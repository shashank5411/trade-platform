"""
Athena client — submit SQL, poll until done, return DataFrame.
All query functions in api.py use this. Never call Athena directly
from anywhere else.
"""

import time
import boto3
import pandas as pd
from io import StringIO
from typing import Optional

REGION          = "us-east-2"
RESULTS_BUCKET  = "s3://dev-trade-athena-results-197411402303/"
DEFAULT_TIMEOUT = 60   # seconds
POLL_INTERVAL   = 1.5  # seconds between status checks

athena = boto3.client("athena", region_name=REGION)
s3     = boto3.client("s3",     region_name=REGION)


class AthenaError(Exception):
    pass


class AthenaTimeout(Exception):
    pass


def query(
    sql:      str,
    database: str,
    timeout:  int = DEFAULT_TIMEOUT
) -> pd.DataFrame:
    """
    Submit a SQL query to Athena, poll until complete, return DataFrame.

    Args:
        sql:      SQL string to execute
        database: Glue database name e.g. dev_trade_fred_processed
        timeout:  Max seconds to wait before raising AthenaTimeout

    Returns:
        pandas DataFrame of results (empty DataFrame if no rows)

    Raises:
        AthenaError:   Query failed with an error
        AthenaTimeout: Query did not complete within timeout seconds
    """
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
            return _fetch_results(execution_id)

        elif state in ("FAILED", "CANCELLED"):
            reason = status.get("StateChangeReason", "Unknown")
            raise AthenaError(
                f"Query {execution_id} {state}: {reason}\nSQL: {sql}"
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
    # Get output location
    execution = athena.get_query_execution(
        QueryExecutionId=execution_id
    )["QueryExecution"]

    output_location = execution["ResultConfiguration"]["OutputLocation"]
    # e.g. s3://dev-trade-athena-results-197411402303/abc123.csv

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