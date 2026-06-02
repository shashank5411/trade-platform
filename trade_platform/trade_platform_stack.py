from aws_cdk import (
    Aws,
    RemovalPolicy,
    Stack,
    aws_dynamodb as dynamodb,
    aws_glue as glue,
    aws_iam as iam,
    aws_s3 as s3,
    aws_s3_assets as s3_assets,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct

SOURCES = ["yfinance", "fred", "worldbank", "sec", "wikipedia"]
LAYERS = ["raw", "processed"]

# Add this dict near the top of the stack, alongside SOURCES/LAYERS
PROCESSED_PATHS = {
    "fred":       "economic_indicators/source=FRED/",
    "worldbank":  "economic_indicators/source=WORLDBANK/",
    "yfinance":   "market_prices/",
    "wikipedia":  "documents/source=WIKIPEDIA/",
    "sec":        "documents/source=EDGAR/",
}

RAW_PATHS = {
    "fred":       "year=",
    "worldbank":  "year=",
    "yfinance":   "year=",
    "wikipedia":  "year=",
    "sec":        "year=",
}

class TradePlatformStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, env_name: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── S3 buckets (raw + processed, one per source) ─────────────────────
        buckets: dict[str, dict[str, s3.Bucket]] = {layer: {} for layer in LAYERS}
        for layer in LAYERS:
            for source in SOURCES:
                bucket = s3.Bucket(
                    self, f"{source.capitalize()}{layer.capitalize()}Bucket",
                    bucket_name=f"{env_name}-trade-{source}-{layer}-{Aws.ACCOUNT_ID}",
                    versioned=True,
                    encryption=s3.BucketEncryption.S3_MANAGED,
                    block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
                    enforce_ssl=True,
                    removal_policy=RemovalPolicy.RETAIN,
                )
                buckets[layer][source] = bucket

        # ── Glue Data Catalog databases (raw + processed, one per source) ─────
        glue_databases: dict[str, dict[str, glue.CfnDatabase]] = {layer: {} for layer in LAYERS}
        for layer in LAYERS:
            for source in SOURCES:
                db = glue.CfnDatabase(
                    self, f"{source.capitalize()}{layer.capitalize()}GlueDb",
                    catalog_id=self.account,
                    database_input=glue.CfnDatabase.DatabaseInputProperty(
                        name=f"{env_name}_trade_{source}_{layer}",
                        description=f"[{env_name}] {layer.capitalize()} catalog for {source.upper()} ingestion",
                    ),
                )
                glue_databases[layer][source] = db

        # ── Glue IAM role shared by all crawlers ──────────────────────────────
        glue_role = iam.Role(
            self, "GlueCrawlerRole",
            assumed_by=iam.ServicePrincipal("glue.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSGlueServiceRole"
                ),
            ],
        )
        for layer_buckets in buckets.values():
            for bucket in layer_buckets.values():
                bucket.grant_read(glue_role)

        # ── Glue crawlers (raw + processed, one per source) ───────────────────
        for layer in LAYERS:
            for source in SOURCES:
                bucket = buckets[layer][source]
                crawler = glue.CfnCrawler(
                    self, f"{source.capitalize()}{layer.capitalize()}Crawler",
                    name=f"{env_name}-trade-{source}-{layer}-crawler",
                    role=glue_role.role_arn,
                    database_name=f"{env_name}_trade_{source}_{layer}",
                    targets=glue.CfnCrawler.TargetsProperty(
                        s3_targets=[
                            glue.CfnCrawler.S3TargetProperty(
                                path=f"s3://{bucket.bucket_name}/{PROCESSED_PATHS[source] if layer == 'processed' else RAW_PATHS[source]}",
                            )
                        ]
                    ),
                    description=f"[{env_name}] Crawler for {source.upper()} {layer} data",
                    schedule=glue.CfnCrawler.ScheduleProperty(
                        schedule_expression="cron(0 2 * * ? *)",
                    ),
                    schema_change_policy=glue.CfnCrawler.SchemaChangePolicyProperty(
                        update_behavior="LOG",
                        delete_behavior="LOG",
                    ),
                    recrawl_policy=glue.CfnCrawler.RecrawlPolicyProperty(
                        recrawl_behavior="CRAWL_NEW_FOLDERS_ONLY",
                    ),
                )
                crawler.add_dependency(glue_databases[layer][source])

        # ── DynamoDB watermark table ───────────────────────────────────────────
        watermarks_table = dynamodb.Table(
            self, "WatermarksTable",
            table_name=f"trade-platform-{env_name}-watermarks",
            partition_key=dynamodb.Attribute(
                name="source_name",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="dataset_name",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # ── Secrets Manager (API keys) ────────────────────────────────────────
        fred_secret = secretsmanager.Secret(
            self, "FredApiKey",
            secret_name=f"trade-platform/{env_name}/fred-api-key",
            description=f"[{env_name}] FRED API key — set via: "
                        f"aws secretsmanager put-secret-value "
                        f"--secret-id trade-platform/{env_name}/fred-api-key "
                        f"--secret-string YOUR_KEY",
        )

        # ── Glue ingestion job role ───────────────────────────────────────────
        job_role = iam.Role(
            self, "GlueIngestionJobRole",
            assumed_by=iam.ServicePrincipal("glue.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSGlueServiceRole"
                ),
            ],
        )
        for layer_buckets in buckets.values():
            for bucket in layer_buckets.values():
                bucket.grant_read_write(job_role)
        watermarks_table.grant_read_write_data(job_role)
        fred_secret.grant_read(job_role)

        # ── S3 assets — ingestion code uploaded at cdk deploy ─────────────────
        # Full ingestion dir → zipped by CDK, used as --extra-py-files so
        # 'from utils.watermark import ...' works inside Glue Python Shell
        ingestion_pkg = s3_assets.Asset(self, "IngestionPackage", path="ingestion")
        ingestion_pkg.grant_read(job_role)

        # Individual script files — Glue script_location must point to a .py file
        script_assets: dict = {}
        for source in SOURCES:
            asset = s3_assets.Asset(
                self, f"{source.capitalize()}ScriptAsset",
                path=f"ingestion/scripts/ingest_{source}.py",
            )
            asset.grant_read(job_role)
            script_assets[source] = asset

        # ── Glue Python Shell ingestion jobs ──────────────────────────────────
        ADDITIONAL_MODULES = (
            "yfinance>=0.2.0,"
            "fredapi>=0.5.0,"
            "wbdata>=1.0.0,"
            "pyyaml>=6.0.0"
        )

        JOB_SCHEDULES = {
            "yfinance":  "cron(0 21 ? * MON-FRI *)",   # weekdays after US close
            "fred":      "cron(0 6 1 * ? *)",            # 1st of each month
            "worldbank": "cron(0 6 1 1 ? *)",            # Jan 1st (annual data)
            "sec":       "cron(0 6 1 1,4,7,10 ? *)",    # quarterly
            "wikipedia": "cron(0 6 ? * MON *)",           # every Monday
        }

        for source in SOURCES:
            script = script_assets[source]
            job = glue.CfnJob(
                self, f"{source.capitalize()}IngestionJob",
                name=f"{env_name}-trade-{source}-ingestion",
                role=job_role.role_arn,
                command=glue.CfnJob.JobCommandProperty(
                    name="pythonshell",
                    python_version="3.9",
                    script_location=(
                        f"s3://{script.s3_bucket_name}/{script.s3_object_key}"
                    ),
                ),
                default_arguments={
                    "--extra-py-files": (
                        f"s3://{ingestion_pkg.s3_bucket_name}"
                        f"/{ingestion_pkg.s3_object_key}"
                    ),
                    "--additional-python-modules": ADDITIONAL_MODULES,
                    "--ENVIRONMENT":               env_name,
                    "--FRED_SECRET_NAME":          fred_secret.secret_name,
                    "--job-language":              "python",
                },
                glue_version="3.0",
                max_capacity=0.0625,
                timeout=30,
                description=f"[{env_name}] {source.upper()} data ingestion",
            )

            glue.CfnTrigger(
                self, f"{source.capitalize()}JobTrigger",
                name=f"{env_name}-trade-{source}-trigger",
                type="SCHEDULED",
                schedule=JOB_SCHEDULES[source],
                actions=[glue.CfnTrigger.ActionProperty(job_name=job.ref)],
                start_on_creation=False,
                description=f"[{env_name}] Schedule for {source.upper()} ingestion",
            )
