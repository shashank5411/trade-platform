"""
Quick verification: processed-table row counts per ticker for insiders/news.
Run this BEFORE and AFTER the ETL re-run to confirm write_partition() merge
fix preserved history rather than overwriting it with only the latest run's rows.

Usage:
    python check_processed_counts.py --source insiders
    python check_processed_counts.py --source news
"""
import argparse
import time
import boto3

REGION = "us-east-2"
# Adjust these if your actual Athena database/table/output-location names differ
QUERIES = {
    "insiders": (
        "dev_trade_processed",
        "SELECT ticker, COUNT(*) as cnt FROM insider_trades GROUP BY ticker ORDER BY ticker",
    ),
    "news": (
        "dev_trade_processed",
        "SELECT primary_ticker, COUNT(*) as cnt FROM news_articles GROUP BY primary_ticker ORDER BY primary_ticker",
    ),
}

# Athena needs an S3 location to write query results to
ATHENA_OUTPUT_LOCATION = "s3://aws-athena-query-results-<ACCOUNT_ID>-us-east-2/"


def run_query(database, query):
    athena = boto3.client("athena", region_name=REGION)
    resp = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_LOCATION},
    )
    qid = resp["QueryExecutionId"]

    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)
        state = status["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(1)

    if state != "SUCCEEDED":
        reason = status["QueryExecution"]["Status"].get("StateChangeReason", "unknown")
        print(f"Query FAILED: {reason}")
        return None

    results = athena.get_query_results(QueryExecutionId=qid)
    rows = results["ResultSet"]["Rows"]
    out = []
    for row in rows[1:]:  # skip header row
        vals = [c.get("VarCharValue", "") for c in row["Data"]]
        out.append(vals)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=QUERIES.keys())
    args = parser.parse_args()

    database, query = QUERIES[args.source]
    print(f"Running against database={database}\nQuery: {query}\n")
    rows = run_query(database, query)
    if rows is None:
        return

    total = 0
    for ticker, cnt in rows:
        print(f"  {ticker:10s} {cnt}")
        total += int(cnt)
    print(f"\nTOTAL rows: {total}")


if __name__ == "__main__":
    main()