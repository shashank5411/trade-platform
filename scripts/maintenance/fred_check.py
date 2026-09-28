import boto3
import pyarrow.parquet as pq
import io

s3 = boto3.client("s3", region_name="us-east-2")

print("OLD vintage (pre-fix, 2026-06-14):")
obj = s3.get_object(
    Bucket="dev-trade-fred-processed-<ACCOUNT_ID>",
    Key="economic_indicators/source=FRED/year=2023/indicator_id=DCOILWTICO/vintage=2026-06-14.parquet"
)
df_old = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
print("  Rows:", len(df_old))
print("  Date range:", df_old["date"].min(), "to", df_old["date"].max())
print("  Frequency field:", df_old["frequency"].unique())

print()
print("NEW vintage (post-fix, 2026-06-19):")
obj = s3.get_object(
    Bucket="dev-trade-fred-processed-<ACCOUNT_ID>",
    Key="economic_indicators/source=FRED/year=2023/indicator_id=DCOILWTICO/vintage=2026-06-19.parquet"
)
df_new = pq.read_table(io.BytesIO(obj["Body"].read())).to_pandas()
print("  Rows:", len(df_new))
print("  Date range:", df_new["date"].min(), "to", df_new["date"].max())
print("  Frequency field:", df_new["frequency"].unique())
