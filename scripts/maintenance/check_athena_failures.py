"""
check_athena_failures.py — pulls real StateChangeReason for recent Athena
failures, since the console's list view hides this field.

Run with your normal AWS credentials active:
    python check_athena_failures.py

Adjust MAX_RESULTS / the time window filter below if you need to look
further back than the most recent batch of executions.
"""
import boto3
from datetime import datetime, timezone

REGION = "us-east-2"
MAX_RESULTS = 50  # list_query_executions returns most-recent-first

# Narrow to the eval run's window — adjust if needed
WINDOW_START = datetime(2026, 6, 25, 2, 0, tzinfo=timezone.utc)
WINDOW_END   = datetime(2026, 6, 25, 8, 0, tzinfo=timezone.utc)

athena = boto3.client("athena", region_name=REGION)

resp = athena.list_query_executions(MaxResults=MAX_RESULTS)
qids = resp["QueryExecutionIds"]

# batch_get is more efficient than one get_query_execution call per id
details = athena.batch_get_query_execution(QueryExecutionIds=qids)["QueryExecutions"]

failures = []
for d in details:
    status = d["Status"]
    submit_time = status.get("SubmissionDateTime")
    if submit_time and not (WINDOW_START <= submit_time <= WINDOW_END):
        continue
    if status["State"] in ("FAILED", "CANCELLED"):
        failures.append({
            "id": d["QueryExecutionId"],
            "state": status["State"],
            "reason": status.get("StateChangeReason", "<no reason given>"),
            "submit_time": str(submit_time),
            "completion_time": str(status.get("CompletionDateTime")),
            "query_preview": d["Query"][:150].replace("\n", " "),
        })

if not failures:
    print(f"No FAILED/CANCELLED queries found in window "
          f"{WINDOW_START} to {WINDOW_END} among the {len(qids)} most recent executions.")
    print("Try increasing MAX_RESULTS or widening the window if your run was further back.")
else:
    failures.sort(key=lambda f: f["submit_time"])
    for f in failures:
        print("=" * 70)
        print(f"Submitted:   {f['submit_time']}")
        print(f"Completed:   {f['completion_time']}")
        print(f"State:       {f['state']}")
        print(f"Reason:      {f['reason']}")
        print(f"Query:       {f['query_preview']}...")
        print(f"Execution ID: {f['id']}")
    print("=" * 70)
    print(f"\nTotal failures in window: {len(failures)}")