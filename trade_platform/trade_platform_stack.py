from aws_cdk import (
    Aws,
    RemovalPolicy,
    Stack,
    aws_dynamodb as dynamodb,
    aws_glue as glue,
    aws_iam as iam,
    aws_s3 as s3,
    aws_s3_assets as s3_assets,
    aws_s3vectors as s3vectors,    
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct

SOURCES = ["yfinance", "fred", "worldbank", "sec", "wikipedia"]
LAYERS = ["raw", "processed"]

# Add this dict near the top of the stack, alongside SOURCES/LAYERS
PROCESSED_PATHS = {
    "fred":       "",
    "worldbank":  "",
    "yfinance":   "",
    "wikipedia":  "",
    "sec":        "",
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

        # GitHub Actions OIDC Deploy Role        
        github_actions_role = iam.Role(
            self, "GitHubActionsDeployRole",
            role_name=f"GitHubActionsDeployRole-{env_name}",
            assumed_by=iam.WebIdentityPrincipal(
                f"arn:aws:iam::{Aws.ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com",
                conditions={
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                        "token.actions.githubusercontent.com:sub":
                            "repo:shashank5411/trade-platform:ref:refs/heads/master"
                    }
                }
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AdministratorAccess")
            ]
        )
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
                        recrawl_behavior="CRAWL_EVERYTHING",
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

        # ── S3 Vectors bucket + index (Phase 5 — RAG vector store) ───────────
        vectors_bucket = s3vectors.CfnVectorBucket(
            self, "VectorsBucket",
            vector_bucket_name=f"{env_name}-trade-vectors-{Aws.ACCOUNT_ID}",
            # SSE-S3 (AES256) is the default — no encryption_configuration needed
        )

        # Vector index — one index for all document embeddings
        # distance_metric: COSINE is standard for text embeddings
        # dimensions: 1536 = Titan Embed Text v2 output size
        vectors_index = s3vectors.CfnIndex(
            self, "VectorsIndexV2",        # ← new logical ID
            vector_bucket_name=f"{env_name}-trade-vectors-{Aws.ACCOUNT_ID}",
            index_name="documents-index",  # ← keep same physical name
            data_type="float32",
            dimension=1024,
            distance_metric="cosine",
        )
        vectors_index.add_dependency(vectors_bucket)

        # ── LLMOps bucket (Phase 6 — agent telemetry) ─────────────────────────
        llmops_bucket = s3.Bucket(
            self, "LLMOpsBucket",
            bucket_name=f"{env_name}-trade-llmops-{Aws.ACCOUNT_ID}",
            versioned=False,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        llmops_bucket.grant_read(glue_role)

        # Glue database for LLMOps Athena queries
        llmops_db = glue.CfnDatabase(
            self, "LLMOpsGlueDb",
            catalog_id=self.account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name=f"{env_name}_trade_llmops",
                description=f"[{env_name}] LLMOps telemetry — agent traces",
            ),
        )

        # Crawler for LLMOps telemetry
        llmops_crawler = glue.CfnCrawler(
            self, "LLMOpsCrawler",
            name=f"{env_name}-trade-llmops-crawler",
            role=glue_role.role_arn,
            database_name=f"{env_name}_trade_llmops",
            targets=glue.CfnCrawler.TargetsProperty(
                s3_targets=[
                    glue.CfnCrawler.S3TargetProperty(
                        path=f"s3://{env_name}-trade-llmops-{Aws.ACCOUNT_ID}/traces/",
                    )
                ]
            ),
            description=f"[{env_name}] Crawler for LLMOps telemetry",
            schedule=glue.CfnCrawler.ScheduleProperty(
                schedule_expression="cron(0 3 * * ? *)",
            ),
            schema_change_policy=glue.CfnCrawler.SchemaChangePolicyProperty(
                update_behavior="LOG",
                delete_behavior="LOG",
            ),
            recrawl_policy=glue.CfnCrawler.RecrawlPolicyProperty(
                recrawl_behavior="CRAWL_EVERYTHING",
            ),
        )
        llmops_crawler.add_dependency(llmops_db)

        # ── SEC prose processed bucket (Phase 7) ──────────────────────────────
        sec_prose_bucket = s3.Bucket(
            self, "SecProseBucket",
            bucket_name=f"{env_name}-trade-sec-prose-processed-{Aws.ACCOUNT_ID}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        sec_prose_bucket.grant_read(glue_role)

        # Glue database for prose
        sec_prose_db = glue.CfnDatabase(
            self, "SecProseGlueDb",
            catalog_id=self.account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name=f"{env_name}_trade_sec_prose_processed",
                description=f"[{env_name}] SEC 10-K/10-Q prose sections for RAG",
            ),
        )

        # Crawler for prose table
        sec_prose_crawler = glue.CfnCrawler(
            self, "SecProseCrawler",
            name=f"{env_name}-trade-sec-prose-processed-crawler",
            role=glue_role.role_arn,
            database_name=f"{env_name}_trade_sec_prose_processed",
            targets=glue.CfnCrawler.TargetsProperty(
                s3_targets=[
                    glue.CfnCrawler.S3TargetProperty(
                        path=f"s3://{env_name}-trade-sec-prose-processed-{Aws.ACCOUNT_ID}/documents_prose/",
                    )
                ]
            ),
            description=f"[{env_name}] Crawler for SEC prose sections",
            schedule=glue.CfnCrawler.ScheduleProperty(
                schedule_expression="cron(0 3 * * ? *)",
            ),
            schema_change_policy=glue.CfnCrawler.SchemaChangePolicyProperty(
                update_behavior="LOG",
                delete_behavior="LOG",
            ),
            recrawl_policy=glue.CfnCrawler.RecrawlPolicyProperty(
                recrawl_behavior="CRAWL_EVERYTHING",
            ),
        )
        sec_prose_crawler.add_dependency(sec_prose_db)

        # ── DynamoDB conversation memory table (Phase 5 — agent memory) ───────
        conversation_table = dynamodb.Table(
            self, "ConversationTable",
            table_name=f"trade-platform-{env_name}-conversations",
            partition_key=dynamodb.Attribute(
                name="session_id",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl",
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

        anthropic_secret = secretsmanager.Secret.from_secret_name_v2(
            self, "AnthropicApiKey",
                f"trade-platform/{env_name}/anthropic-api-key",
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
        anthropic_secret.grant_read(job_role)
        llmops_bucket.grant_read_write(job_role)
        sec_prose_bucket.grant_read_write(job_role)

        # ── Bedrock + S3 Vectors permissions for job_role (Phase 5) ──────────
        job_role.add_to_policy(iam.PolicyStatement(
            sid="BedrockEmbeddings",
            effect=iam.Effect.ALLOW,
            actions=["bedrock:InvokeModel"],
            resources=[
                "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0",
                "arn:aws:bedrock:us-east-1::foundation-model/cohere.embed-english-v3",
                "arn:aws:bedrock:us-east-1::foundation-model/cohere.embed-multilingual-v3",
            ],
        ))

        job_role.add_to_policy(iam.PolicyStatement(
            sid="BedrockHaikuFallback",
            effect=iam.Effect.ALLOW,
            actions=["bedrock:InvokeModel"],
            resources=[
                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-haiku-4-5",
                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-haiku-4-5-20251001",
            ],
        ))

        job_role.add_to_policy(iam.PolicyStatement(
            sid="S3VectorsAccess",
            effect=iam.Effect.ALLOW,
            actions=[
                "s3vectors:PutVectors",
                "s3vectors:GetVectors",
                "s3vectors:DeleteVectors",
                "s3vectors:QueryVectors",
                "s3vectors:ListVectors",
            ],
            resources=[
                f"arn:aws:s3vectors:{self.region}:{self.account}:bucket/{env_name}-trade-vectors-{Aws.ACCOUNT_ID}",
                f"arn:aws:s3vectors:{self.region}:{self.account}:bucket/{env_name}-trade-vectors-{Aws.ACCOUNT_ID}/index/documents-index",
            ],
        ))

        conversation_table.grant_read_write_data(job_role)

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

        # ── ETL script assets ─────────────────────────────────────────────────
        ETL_SOURCES = ["yfinance", "fred", "worldbank", "wikipedia"]  # standard ETL jobs
        etl_script_assets: dict = {}
        for source in ETL_SOURCES:
            asset = s3_assets.Asset(
                self, f"{source.capitalize()}EtlScriptAsset",
                path=f"ingestion/etl/etl_{source}.py",
            )
            asset.grant_read(job_role)
            etl_script_assets[source] = asset

        # SEC has 3 chained ETL scripts — handled separately
        etl_sec_asset = s3_assets.Asset(
            self, "EtlSecScriptAsset",
            path="ingestion/etl/etl_sec.py",
        )
        etl_sec_asset.grant_read(job_role)

        etl_sec_prose_asset = s3_assets.Asset(
            self, "EtlSecProseScriptAsset",
            path="ingestion/etl/etl_sec_prose.py",
        )
        etl_sec_prose_asset.grant_read(job_role)

        etl_embed_asset = s3_assets.Asset(
            self, "EtlEmbedScriptAsset",
            path="ingestion/etl/etl_embed.py",
        )
        etl_embed_asset.grant_read(job_role)

    # ── Glue Python Shell ingestion jobs ──────────────────────────────────
    # Base modules for all jobs
        ADDITIONAL_MODULES = (
            "yfinance>=0.2.0,"
            "fredapi>=0.5.0,"
            "wbdata==0.3.0,"
            "pyyaml>=6.0.0"
        )
 
        ADDITIONAL_MODULES_OVERRIDE = {
            "sec": (
                "yfinance>=0.2.0,"
                "fredapi>=0.5.0,"
                "pyyaml>=6.0.0,"
                "pyarrow==14.0.2,"
                "edgartools>=3.0.0"
            ),
        }

        TIMEOUT_OVERRIDE = {
           "yfinance": 60,  # 479 S&P 500 tickers — yf.Ticker().info per-ticker
                     # metadata calls add up beyond the default 30 min
        }

        JOB_SCHEDULES = {
            "yfinance":  "cron(0 21 ? * MON-FRI *)",   # weekdays after US close
            "fred":      "cron(0 6 1 * ? *)",            # 1st of each month
            "worldbank": "cron(0 6 1 1 ? *)",            # Jan 1st (annual data)
            "sec":       "cron(0 6 1 1,4,7,10 ? *)",    # quarterly
            "wikipedia": "cron(0 6 ? * MON *)",           # every Monday
        }

        for source in [s for s in SOURCES if s != "sec"]:
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
                    "--additional-python-modules": ADDITIONAL_MODULES_OVERRIDE.get(
                        source, ADDITIONAL_MODULES
                    ),
                    "--ENVIRONMENT":               env_name,
                    "--FRED_SECRET_NAME":          fred_secret.secret_name,
                    "--job-language":              "python",
                },
                glue_version="3.0",
                max_capacity=0.0625,
                timeout=TIMEOUT_OVERRIDE.get(source, 30),
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

        sec_script = script_assets["sec"]
        sec_ingestion_job = glue.CfnJob(
            self, "SecIngestionJob",
            name=f"{env_name}-trade-sec-ingestion",
            role=job_role.role_arn,
            command=glue.CfnJob.JobCommandProperty(
                name="pythonshell",
                python_version="3.9",
                script_location=(
                    f"s3://{sec_script.s3_bucket_name}/{sec_script.s3_object_key}"
                ),
            ),
            default_arguments={
                "--extra-py-files": (
                    f"s3://{ingestion_pkg.s3_bucket_name}"
                    f"/{ingestion_pkg.s3_object_key}"
                ),
                "--additional-python-modules": ADDITIONAL_MODULES_OVERRIDE["sec"],
                "--ENVIRONMENT":               env_name,
                "--FRED_SECRET_NAME":          fred_secret.secret_name,
                "--job-language":              "python",
            },
            glue_version="3.0",
            max_capacity=0.0625,
            timeout=60,
            description=f"[{env_name}] SEC EDGAR ingestion — submissions + facts",
        )

        glue.CfnTrigger(
            self, "SecIngestionTrigger",
            name=f"{env_name}-trade-sec-scheduled-trigger",
            type="SCHEDULED",
            schedule=JOB_SCHEDULES["sec"],
            actions=[glue.CfnTrigger.ActionProperty(job_name=sec_ingestion_job.ref)],
            start_on_creation=False,
            description=f"[{env_name}] Schedule for SEC ingestion",
        )

        # ── ETL jobs + conditional trigger chains ─────────────────────────────

        ETL_MODULES_BASE = (
            "pandas==2.0.3,"
            "pyarrow==14.0.2,"
            "pyyaml>=6.0.0"
        )

        ETL_MODULES_OVERRIDE = {
            "sec_prose": (
                "pandas==2.0.3,"
                "pyarrow==14.0.2,"
                "pyyaml>=6.0.0,"
                "boto3>=1.28.0,"
                "edgartools>=3.0.0"
            ),
            "embed": (
                "pandas==2.0.3,"
                "pyarrow==14.0.2,"
                "pyyaml>=6.0.0,"
                "boto3>=1.28.0"
            ),
        }

        # ── Standard ETL jobs (yfinance, fred, worldbank, wikipedia) ──────────
        for source in ETL_SOURCES:
            etl_job = glue.CfnJob(
                self, f"{source.capitalize()}EtlJob",
                name=f"{env_name}-trade-{source}-etl",
                role=job_role.role_arn,
                command=glue.CfnJob.JobCommandProperty(
                    name="pythonshell",
                    python_version="3.9",
                    script_location=(
                        f"s3://{etl_script_assets[source].s3_bucket_name}"
                        f"/{etl_script_assets[source].s3_object_key}"
                    ),
                ),
                default_arguments={
                    "--extra-py-files": (
                        f"s3://{ingestion_pkg.s3_bucket_name}"
                        f"/{ingestion_pkg.s3_object_key}"
                    ),
                    "--additional-python-modules": ETL_MODULES_BASE,
                    "--ENVIRONMENT": env_name,
                    "--job-language": "python",
                },
                glue_version="3.0",
                max_capacity=0.0625,
                timeout=90,
                description=f"[{env_name}] {source.upper()} raw → Parquet ETL",
            )

            glue.CfnTrigger(
                self, f"{source.capitalize()}EtlTrigger",
                name=f"{env_name}-trade-{source}-etl-trigger",
                type="CONDITIONAL",
                start_on_creation=True,
                actions=[glue.CfnTrigger.ActionProperty(job_name=etl_job.ref)],
                predicate=glue.CfnTrigger.PredicateProperty(
                    logical="AND",
                    conditions=[
                        glue.CfnTrigger.ConditionProperty(
                            logical_operator="EQUALS",
                            job_name=f"{env_name}-trade-{source}-ingestion",
                            state="SUCCEEDED",
                        )
                    ],
                ),
                description=(
                    f"[{env_name}] Fire etl_{source} after ingest_{source} succeeds"
                ),
            )

        # ── etl_companies: yfinance metadata → companies reference table ──────
        etl_companies_asset = s3_assets.Asset(
            self, "EtlCompaniesScriptAsset",
            path="ingestion/etl/etl_companies.py",
        )
        etl_companies_asset.grant_read(job_role)

        etl_companies_job = glue.CfnJob(
            self, "EtlCompaniesJob",
            name=f"{env_name}-trade-companies-etl",
            role=job_role.role_arn,
            command=glue.CfnJob.JobCommandProperty(
                name="pythonshell",
                python_version="3.9",
                script_location=(
                    f"s3://{etl_companies_asset.s3_bucket_name}"
                    f"/{etl_companies_asset.s3_object_key}"
                ),
            ),
            default_arguments={
                "--extra-py-files": (
                    f"s3://{ingestion_pkg.s3_bucket_name}"
                    f"/{ingestion_pkg.s3_object_key}"
                ),
                "--additional-python-modules": ETL_MODULES_BASE,
                "--ENVIRONMENT": env_name,
                "--job-language": "python",
            },
            glue_version="3.0",
            max_capacity=0.0625,
            timeout=90,
            description=f"[{env_name}] yfinance raw → companies reference table",
        )

        glue.CfnTrigger(
            self, "EtlCompaniesTrigger",
            name=f"{env_name}-trade-companies-etl-trigger",
            type="CONDITIONAL",
            start_on_creation=True,
            actions=[glue.CfnTrigger.ActionProperty(
                job_name=etl_companies_job.ref
            )],
            predicate=glue.CfnTrigger.PredicateProperty(
                logical="AND",
                conditions=[
                    glue.CfnTrigger.ConditionProperty(
                        logical_operator="EQUALS",
                        job_name=f"{env_name}-trade-yfinance-etl",
                        state="SUCCEEDED",
                    )
                ],
            ),
            description=f"[{env_name}] Fire etl_companies after etl_yfinance succeeds",
        )

        # ── SEC ETL chain ─────────────────────────────────────────────────────
        etl_sec_job = glue.CfnJob(
            self, "EtlSecJob",
            name=f"{env_name}-trade-sec-etl",
            role=job_role.role_arn,
            command=glue.CfnJob.JobCommandProperty(
                name="pythonshell",
                python_version="3.9",
                script_location=(
                    f"s3://{etl_sec_asset.s3_bucket_name}"
                    f"/{etl_sec_asset.s3_object_key}"
                ),
            ),
            default_arguments={
                "--extra-py-files": (
                    f"s3://{ingestion_pkg.s3_bucket_name}"
                    f"/{ingestion_pkg.s3_object_key}"
                ),
                "--additional-python-modules": ETL_MODULES_BASE,
                "--ENVIRONMENT": env_name,
                "--job-language": "python",
            },
            glue_version="3.0",
            max_capacity=0.0625,
            timeout=30,
            description=f"[{env_name}] SEC XBRL → Parquet ETL",
        )

        etl_sec_prose_job = glue.CfnJob(
            self, "EtlSecProseJob",
            name=f"{env_name}-trade-sec-prose-etl",
            role=job_role.role_arn,
            command=glue.CfnJob.JobCommandProperty(
                name="pythonshell",
                python_version="3.9",
                script_location=(
                    f"s3://{etl_sec_prose_asset.s3_bucket_name}"
                    f"/{etl_sec_prose_asset.s3_object_key}"
                ),
            ),
            default_arguments={
                "--extra-py-files": (
                    f"s3://{ingestion_pkg.s3_bucket_name}"
                    f"/{ingestion_pkg.s3_object_key}"
                ),
                "--additional-python-modules": ETL_MODULES_OVERRIDE["sec_prose"],
                "--ENVIRONMENT": env_name,
                "--job-language": "python",
            },
            glue_version="3.0",
            max_capacity=0.0625,
            timeout=60,
            description=f"[{env_name}] SEC 10-K/10-Q prose sections → Parquet",
        )

        etl_embed_job = glue.CfnJob(
            self, "EtlEmbedJob",
            name=f"{env_name}-trade-sec-embed-etl",
            role=job_role.role_arn,
            command=glue.CfnJob.JobCommandProperty(
                name="pythonshell",
                python_version="3.9",
                script_location=(
                    f"s3://{etl_embed_asset.s3_bucket_name}"
                    f"/{etl_embed_asset.s3_object_key}"
                ),
            ),
            default_arguments={
                "--extra-py-files": (
                    f"s3://{ingestion_pkg.s3_bucket_name}"
                    f"/{ingestion_pkg.s3_object_key}"
                ),
                "--additional-python-modules": ETL_MODULES_OVERRIDE["embed"],
                "--ENVIRONMENT": env_name,
                "--job-language": "python",
            },
            glue_version="3.0",
            max_capacity=0.0625,
            timeout=60,
            description=f"[{env_name}] Cohere embed → S3 Vectors",
        )

        # Step 1: ingest_sec → etl_sec
        glue.CfnTrigger(
            self, "EtlSecTrigger",
            name=f"{env_name}-trade-sec-etl-trigger",
            type="CONDITIONAL",
            start_on_creation=True,
            actions=[glue.CfnTrigger.ActionProperty(job_name=etl_sec_job.ref)],
            predicate=glue.CfnTrigger.PredicateProperty(
                logical="AND",
                conditions=[
                    glue.CfnTrigger.ConditionProperty(
                        logical_operator="EQUALS",
                        job_name=f"{env_name}-trade-sec-ingestion",
                        state="SUCCEEDED",
                    )
                ],
            ),
            description=f"[{env_name}] Fire etl_sec after ingest_sec succeeds",
        )

        # Step 2: etl_sec → etl_sec_prose
        glue.CfnTrigger(
            self, "EtlSecProseTrigger",
            name=f"{env_name}-trade-sec-prose-etl-trigger",
            type="CONDITIONAL",
            start_on_creation=True,
            actions=[glue.CfnTrigger.ActionProperty(job_name=etl_sec_prose_job.ref)],
            predicate=glue.CfnTrigger.PredicateProperty(
                logical="AND",
                conditions=[
                    glue.CfnTrigger.ConditionProperty(
                        logical_operator="EQUALS",
                        job_name=etl_sec_job.ref,
                        state="SUCCEEDED",
                    )
                ],
            ),
            description=f"[{env_name}] Fire etl_sec_prose after etl_sec succeeds",
        )

        # Step 3: etl_sec_prose → etl_embed
        glue.CfnTrigger(
            self, "EtlEmbedTrigger",
            name=f"{env_name}-trade-sec-embed-trigger",
            type="CONDITIONAL",
            start_on_creation=True,
            actions=[glue.CfnTrigger.ActionProperty(job_name=etl_embed_job.ref)],
            predicate=glue.CfnTrigger.PredicateProperty(
                logical="AND",
                conditions=[
                    glue.CfnTrigger.ConditionProperty(
                        logical_operator="EQUALS",
                        job_name=etl_sec_prose_job.ref,
                        state="SUCCEEDED",
                    )
                ],
            ),
            description=f"[{env_name}] Fire etl_embed after etl_sec_prose succeeds",
        )

        # ── ON_DEMAND manual triggers — correct entry points for manual runs ──────────
        # CONDITIONAL triggers only fire when upstream job is itself trigger-initiated.
        # Fire these to kick off the full chain manually.
        # When ready to automate: replace with SCHEDULED triggers on same job.
        # Usage: aws glue start-trigger --name {env}-trade-{source}-manual-trigger

        for source in SOURCES:
            glue.CfnTrigger(
                self, f"{source.capitalize()}ManualTrigger",
                name=f"{env_name}-trade-{source}-manual-trigger",
                type="ON_DEMAND",
                actions=[glue.CfnTrigger.ActionProperty(
                    job_name=f"{env_name}-trade-{source}-ingestion"
                )],
                description=(
                    f"[{env_name}] Manual entry point for {source.upper()} "
                    f"ingestion — fires conditional ETL chain"
                ),
            )

        # ── FedSpeak buckets, DB, crawler, jobs, triggers ─────────────────────

        fedspeak_raw_bucket = s3.Bucket(
            self, "FedSpeakRawBucket",
            bucket_name=f"{env_name}-trade-fedspeak-raw-{Aws.ACCOUNT_ID}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        fedspeak_raw_bucket.grant_read_write(job_role)
        fedspeak_raw_bucket.grant_read(glue_role)

        fedspeak_processed_bucket = s3.Bucket(
            self, "FedSpeakProcessedBucket",
            bucket_name=f"{env_name}-trade-fedspeak-processed-{Aws.ACCOUNT_ID}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        fedspeak_processed_bucket.grant_read_write(job_role)
        fedspeak_processed_bucket.grant_read(glue_role)

        fedspeak_db = glue.CfnDatabase(
            self, "FedSpeakGlueDb",
            catalog_id=self.account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name=f"{env_name}_trade_fedspeak_processed",
                description=f"[{env_name}] FOMC minutes, statements, transcripts, speeches",
            ),
        )

        fedspeak_crawler = glue.CfnCrawler(
            self, "FedSpeakCrawler",
            name=f"{env_name}-trade-fedspeak-processed-crawler",
            role=glue_role.role_arn,
            database_name=f"{env_name}_trade_fedspeak_processed",
            targets=glue.CfnCrawler.TargetsProperty(
                s3_targets=[
                    glue.CfnCrawler.S3TargetProperty(
                        path=f"s3://{env_name}-trade-fedspeak-processed-{Aws.ACCOUNT_ID}/documents/",
                    )
                ]
            ),
            description=f"[{env_name}] Crawler for FedSpeak documents",
            schedule=glue.CfnCrawler.ScheduleProperty(
                schedule_expression="cron(0 3 * * ? *)",
            ),
            schema_change_policy=glue.CfnCrawler.SchemaChangePolicyProperty(
                update_behavior="LOG",
                delete_behavior="LOG",
            ),
            recrawl_policy=glue.CfnCrawler.RecrawlPolicyProperty(
                recrawl_behavior="CRAWL_EVERYTHING",
            ),
        )
        fedspeak_crawler.add_dependency(fedspeak_db)

        ingest_fedspeak_asset = s3_assets.Asset(
            self, "FedSpeakScriptAsset",
            path="ingestion/scripts/ingest_fedspeak.py",
        )
        ingest_fedspeak_asset.grant_read(job_role)

        etl_fedspeak_asset = s3_assets.Asset(
            self, "FedSpeakEtlScriptAsset",
            path="ingestion/etl/etl_fedspeak.py",
        )
        etl_fedspeak_asset.grant_read(job_role)

        # pypdf needed for PDF text extraction (Fed minutes, transcripts)
        FEDSPEAK_MODULES = (
            "pandas==2.0.3,"
            "pyarrow==14.0.2,"
            "pyyaml>=6.0.0,"
            "pypdf>=3.0.0"
        )

        fedspeak_ingest_job = glue.CfnJob(
            self, "FedSpeakIngestionJob",
            name=f"{env_name}-trade-fedspeak-ingestion",
            role=job_role.role_arn,
            command=glue.CfnJob.JobCommandProperty(
                name="pythonshell",
                python_version="3.9",
                script_location=(
                    f"s3://{ingest_fedspeak_asset.s3_bucket_name}"
                    f"/{ingest_fedspeak_asset.s3_object_key}"
                ),
            ),
            default_arguments={
                "--extra-py-files": (
                    f"s3://{ingestion_pkg.s3_bucket_name}"
                    f"/{ingestion_pkg.s3_object_key}"
                ),
                "--additional-python-modules": FEDSPEAK_MODULES,
                "--ENVIRONMENT": env_name,
                "--job-language": "python",
            },
            glue_version="3.0",
            max_capacity=0.0625,
            timeout=30,
            description=f"[{env_name}] FedSpeak ingestion — FOMC calendar + speeches RSS",
        )

        fedspeak_etl_job = glue.CfnJob(
            self, "FedSpeakEtlJob",
            name=f"{env_name}-trade-fedspeak-etl",
            role=job_role.role_arn,
            command=glue.CfnJob.JobCommandProperty(
                name="pythonshell",
                python_version="3.9",
                script_location=(
                    f"s3://{etl_fedspeak_asset.s3_bucket_name}"
                    f"/{etl_fedspeak_asset.s3_object_key}"
                ),
            ),
            default_arguments={
                "--extra-py-files": (
                    f"s3://{ingestion_pkg.s3_bucket_name}"
                    f"/{ingestion_pkg.s3_object_key}"
                ),
                "--additional-python-modules": ETL_MODULES_BASE,
                "--ENVIRONMENT": env_name,
                "--job-language": "python",
            },
            glue_version="3.0",
            max_capacity=0.0625,
            timeout=30,
            description=f"[{env_name}] FedSpeak raw JSON → Parquet ETL",
        )

        glue.CfnTrigger(
            self, "FedSpeakEtlTrigger",
            name=f"{env_name}-trade-fedspeak-etl-trigger",
            type="CONDITIONAL",
            start_on_creation=True,
            actions=[glue.CfnTrigger.ActionProperty(job_name=fedspeak_etl_job.ref)],
            predicate=glue.CfnTrigger.PredicateProperty(
                logical="AND",
                conditions=[
                    glue.CfnTrigger.ConditionProperty(
                        logical_operator="EQUALS",
                        job_name=fedspeak_ingest_job.ref,
                        state="SUCCEEDED",
                    )
                ],
            ),
            description=f"[{env_name}] Fire etl_fedspeak after ingest_fedspeak succeeds",
        )

        glue.CfnTrigger(
            self, "FedSpeakManualTrigger",
            name=f"{env_name}-trade-fedspeak-manual-trigger",
            type="ON_DEMAND",
            actions=[glue.CfnTrigger.ActionProperty(
                job_name=fedspeak_ingest_job.ref
            )],
            description=(
                f"[{env_name}] Manual entry point for FedSpeak ingestion "
                f"— fires conditional ETL chain"
            ),
        )

        # ── Polygon News CDK additions ────────────────────────────────────────

        polygon_secret = secretsmanager.Secret.from_secret_name_v2(
            self, "PolygonApiKey",
            f"trade-platform/{env_name}/polygon-api-key",
        )
        polygon_secret.grant_read(job_role)

        news_raw_bucket = s3.Bucket(
            self, "NewsRawBucket",
            bucket_name=f"{env_name}-trade-news-raw-{Aws.ACCOUNT_ID}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        news_raw_bucket.grant_read_write(job_role)
        news_raw_bucket.grant_read(glue_role)

        news_processed_bucket = s3.Bucket(
            self, "NewsProcessedBucket",
            bucket_name=f"{env_name}-trade-news-processed-{Aws.ACCOUNT_ID}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        news_processed_bucket.grant_read_write(job_role)
        news_processed_bucket.grant_read(glue_role)

        news_db = glue.CfnDatabase(
            self, "NewsGlueDb",
            catalog_id=self.account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name=f"{env_name}_trade_news_processed",
                description=f"[{env_name}] Polygon news articles with sentiment",
            ),
        )

        news_crawler = glue.CfnCrawler(
            self, "NewsCrawler",
            name=f"{env_name}-trade-news-processed-crawler",
            role=glue_role.role_arn,
            database_name=f"{env_name}_trade_news_processed",
            targets=glue.CfnCrawler.TargetsProperty(
                s3_targets=[
                    glue.CfnCrawler.S3TargetProperty(
                        path=f"s3://{env_name}-trade-news-processed-{Aws.ACCOUNT_ID}/news/",
                    )
                ]
            ),
            description=f"[{env_name}] Crawler for Polygon news articles",
            schedule=glue.CfnCrawler.ScheduleProperty(
                schedule_expression="cron(0 3 * * ? *)",
            ),
            schema_change_policy=glue.CfnCrawler.SchemaChangePolicyProperty(
                update_behavior="LOG",
                delete_behavior="LOG",
            ),
            recrawl_policy=glue.CfnCrawler.RecrawlPolicyProperty(
                recrawl_behavior="CRAWL_EVERYTHING",
            ),
        )
        news_crawler.add_dependency(news_db)

        ingest_news_asset = s3_assets.Asset(
            self, "NewsScriptAsset",
            path="ingestion/scripts/ingest_news.py",
        )
        ingest_news_asset.grant_read(job_role)

        etl_news_asset = s3_assets.Asset(
            self, "NewsEtlScriptAsset",
            path="ingestion/etl/etl_news.py",
        )
        etl_news_asset.grant_read(job_role)

        news_ingest_job = glue.CfnJob(
            self, "NewsIngestionJob",
            name=f"{env_name}-trade-news-ingestion",
            role=job_role.role_arn,
            command=glue.CfnJob.JobCommandProperty(
                name="pythonshell",
                python_version="3.9",
                script_location=(
                    f"s3://{ingest_news_asset.s3_bucket_name}"
                    f"/{ingest_news_asset.s3_object_key}"
                ),
            ),
            default_arguments={
                "--extra-py-files": (
                    f"s3://{ingestion_pkg.s3_bucket_name}"
                    f"/{ingestion_pkg.s3_object_key}"
                ),
                "--additional-python-modules": ETL_MODULES_BASE,
                "--ENVIRONMENT": env_name,
                "--job-language": "python",
            },
            glue_version="3.0",
            max_capacity=0.0625,
            timeout=30,
            description=f"[{env_name}] Polygon/Massive news ingestion — weekly",
        )

        news_etl_job = glue.CfnJob(
            self, "NewsEtlJob",
            name=f"{env_name}-trade-news-etl",
            role=job_role.role_arn,
            command=glue.CfnJob.JobCommandProperty(
                name="pythonshell",
                python_version="3.9",
                script_location=(
                    f"s3://{etl_news_asset.s3_bucket_name}"
                    f"/{etl_news_asset.s3_object_key}"
                ),
            ),
            default_arguments={
                "--extra-py-files": (
                    f"s3://{ingestion_pkg.s3_bucket_name}"
                    f"/{ingestion_pkg.s3_object_key}"
                ),
                "--additional-python-modules": ETL_MODULES_BASE,
                "--ENVIRONMENT": env_name,
                "--job-language": "python",
            },
            glue_version="3.0",
            max_capacity=0.0625,
            timeout=30,
            description=f"[{env_name}] Polygon news raw JSON → Parquet ETL",
        )

        glue.CfnTrigger(
            self, "NewsEtlTrigger",
            name=f"{env_name}-trade-news-etl-trigger",
            type="CONDITIONAL",
            start_on_creation=True,
            actions=[glue.CfnTrigger.ActionProperty(job_name=news_etl_job.ref)],
            predicate=glue.CfnTrigger.PredicateProperty(
                logical="AND",
                conditions=[
                    glue.CfnTrigger.ConditionProperty(
                        logical_operator="EQUALS",
                        job_name=news_ingest_job.ref,
                        state="SUCCEEDED",
                    )
                ],
            ),
            description=f"[{env_name}] Fire etl_news after ingest_news succeeds",
        )

        glue.CfnTrigger(
            self, "NewsScheduledTrigger",
            name=f"{env_name}-trade-news-scheduled-trigger",
            type="SCHEDULED",
            schedule="cron(0 6 ? * MON *)",
            actions=[glue.CfnTrigger.ActionProperty(job_name=news_ingest_job.ref)],
            start_on_creation=False,
            description=f"[{env_name}] Weekly Monday news ingestion",
        )

        glue.CfnTrigger(
            self, "NewsManualTrigger",
            name=f"{env_name}-trade-news-manual-trigger",
            type="ON_DEMAND",
            actions=[glue.CfnTrigger.ActionProperty(
                job_name=news_ingest_job.ref
            )],
            description=f"[{env_name}] Manual entry point for news ingestion",
        )

        # ── SEC Form 4 Insider Trades CDK additions ───────────────────────────

        insiders_raw_bucket = s3.Bucket(
            self, "InsidersRawBucket",
            bucket_name=f"{env_name}-trade-insiders-raw-{Aws.ACCOUNT_ID}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        insiders_raw_bucket.grant_read_write(job_role)
        insiders_raw_bucket.grant_read(glue_role)

        insiders_processed_bucket = s3.Bucket(
            self, "InsidersProcessedBucket",
            bucket_name=f"{env_name}-trade-insiders-processed-{Aws.ACCOUNT_ID}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        insiders_processed_bucket.grant_read_write(job_role)
        insiders_processed_bucket.grant_read(glue_role)

        insiders_db = glue.CfnDatabase(
            self, "InsidersGlueDb",
            catalog_id=self.account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name=f"{env_name}_trade_insiders_processed",
                description=f"[{env_name}] SEC Form 4 insider trades",
            ),
        )

        insiders_crawler = glue.CfnCrawler(
            self, "InsidersCrawler",
            name=f"{env_name}-trade-insiders-processed-crawler",
            role=glue_role.role_arn,
            database_name=f"{env_name}_trade_insiders_processed",
            targets=glue.CfnCrawler.TargetsProperty(
                s3_targets=[
                    glue.CfnCrawler.S3TargetProperty(
                        path=f"s3://{env_name}-trade-insiders-processed-{Aws.ACCOUNT_ID}/insider_trades/",
                    )
                ]
            ),
            description=f"[{env_name}] Crawler for insider trades",
            schedule=glue.CfnCrawler.ScheduleProperty(
                schedule_expression="cron(0 3 * * ? *)",
            ),
            schema_change_policy=glue.CfnCrawler.SchemaChangePolicyProperty(
                update_behavior="LOG",
                delete_behavior="LOG",
            ),
            recrawl_policy=glue.CfnCrawler.RecrawlPolicyProperty(
                recrawl_behavior="CRAWL_EVERYTHING",
            ),
        )
        insiders_crawler.add_dependency(insiders_db)

        ingest_insiders_asset = s3_assets.Asset(
            self, "InsidersScriptAsset",
            path="ingestion/scripts/ingest_insiders.py",
        )
        ingest_insiders_asset.grant_read(job_role)

        etl_insiders_asset = s3_assets.Asset(
            self, "InsidersEtlScriptAsset",
            path="ingestion/etl/etl_insiders.py",
        )
        etl_insiders_asset.grant_read(job_role)

        insiders_ingest_job = glue.CfnJob(
            self, "InsidersIngestionJob",
            name=f"{env_name}-trade-insiders-ingestion",
            role=job_role.role_arn,
            command=glue.CfnJob.JobCommandProperty(
                name="pythonshell",
                python_version="3.9",
                script_location=(
                    f"s3://{ingest_insiders_asset.s3_bucket_name}"
                    f"/{ingest_insiders_asset.s3_object_key}"
                ),
            ),
            default_arguments={
                "--extra-py-files": (
                    f"s3://{ingestion_pkg.s3_bucket_name}"
                    f"/{ingestion_pkg.s3_object_key}"
                ),
                "--additional-python-modules": ETL_MODULES_BASE,
                "--ENVIRONMENT": env_name,
                "--job-language": "python",
            },
            glue_version="3.0",
            max_capacity=0.0625,
            timeout=90,
            description=f"[{env_name}] SEC Form 4 insider trades ingestion",
        )

        insiders_etl_job = glue.CfnJob(
            self, "InsidersEtlJob",
            name=f"{env_name}-trade-insiders-etl",
            role=job_role.role_arn,
            command=glue.CfnJob.JobCommandProperty(
                name="pythonshell",
                python_version="3.9",
                script_location=(
                    f"s3://{etl_insiders_asset.s3_bucket_name}"
                    f"/{etl_insiders_asset.s3_object_key}"
                ),
            ),
            default_arguments={
                "--extra-py-files": (
                    f"s3://{ingestion_pkg.s3_bucket_name}"
                    f"/{ingestion_pkg.s3_object_key}"
                ),
                "--additional-python-modules": ETL_MODULES_BASE,
                "--ENVIRONMENT": env_name,
                "--job-language": "python",
            },
            glue_version="3.0",
            max_capacity=0.0625,
            timeout=30,
            description=f"[{env_name}] Insider trades raw → Parquet ETL",
        )

        glue.CfnTrigger(
            self, "InsidersEtlTrigger",
            name=f"{env_name}-trade-insiders-etl-trigger",
            type="CONDITIONAL",
            start_on_creation=True,
            actions=[glue.CfnTrigger.ActionProperty(
                job_name=insiders_etl_job.ref
            )],
            predicate=glue.CfnTrigger.PredicateProperty(
                logical="AND",
                conditions=[
                    glue.CfnTrigger.ConditionProperty(
                        logical_operator="EQUALS",
                        job_name=insiders_ingest_job.ref,
                        state="SUCCEEDED",
                    )
                ],
            ),
            description=f"[{env_name}] Fire etl_insiders after ingest_insiders succeeds",
        )

        glue.CfnTrigger(
            self, "InsidersScheduledTrigger",
            name=f"{env_name}-trade-insiders-scheduled-trigger",
            type="SCHEDULED",
            schedule="cron(0 8 1 1,4,7,10 ? *)",
            actions=[glue.CfnTrigger.ActionProperty(
                job_name=insiders_ingest_job.ref
            )],
            start_on_creation=False,
            description=f"[{env_name}] Quarterly insider trades ingestion",
        )

        glue.CfnTrigger(
            self, "InsidersManualTrigger",
            name=f"{env_name}-trade-insiders-manual-trigger",
            type="ON_DEMAND",
            actions=[glue.CfnTrigger.ActionProperty(
                job_name=insiders_ingest_job.ref
            )],
            description=f"[{env_name}] Manual entry point for insider trades ingestion",
        )
