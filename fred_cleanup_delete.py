import boto3

s3 = boto3.client("s3", region_name="us-east-2")
bucket = "dev-trade-fred-processed-197411402303"

with open("fred_cleanup_keys.txt") as f:
    keys = [line.strip() for line in f if line.strip()]

print(f"Deleting {len(keys)} files...")

# S3 batch delete, max 1000 per call - well under that here
response = s3.delete_objects(
    Bucket=bucket,
    Delete={"Objects": [{"Key": k} for k in keys], "Quiet": False}
)

deleted = response.get("Deleted", [])
errors = response.get("Errors", [])
print(f"Deleted: {len(deleted)}")
if errors:
    print(f"Errors: {len(errors)}")
    for e in errors:
        print(f"  {e}")
