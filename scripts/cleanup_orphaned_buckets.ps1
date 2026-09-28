# Run after: cdk deploy TradePlatformStack-dev (project pivot)
# Deletes all orphaned buckets from previous deployments.
# worldbank raw/processed are intentionally excluded (still in use).

$old_buckets = @(
    # Auto-named buckets from initial deploys (failed + replaced)
    "tradeplatformstack-dev-acledrawbucket2b5254f5-ffihjxwk44n3",
    "tradeplatformstack-dev-comtraderawbucket66664709-mmrqfbby2ryg",
    "tradeplatformstack-dev-eiarawbucketf049106a-j0de9vxv5gqq",
    "tradeplatformstack-dev-imfrawbucketf4cb891b-nqynckwbeifs",
    "tradeplatformstack-dev-unctadrawbucket7144f81b-mlwovmseucsr",
    "tradeplatformstack-dev-worldbankrawbucket3f8fa072-ldx5klzeeruf",
    "tradeplatformstack-dev-wtorawbucket696208ec-b2crvvwkuiat",
    "tradeplatformstack-dev-acledrawbucket2b5254f5-ol9ktxddfqqe",
    "tradeplatformstack-dev-comtraderawbucket66664709-i4p2g1qldxgl",
    "tradeplatformstack-dev-eiarawbucketf049106a-qxrufndz1ozy",
    "tradeplatformstack-dev-imfrawbucketf4cb891b-pmmnsh5mlw59",
    "tradeplatformstack-dev-unctadrawbucket7144f81b-wnko9xdjbcbi",
    "tradeplatformstack-dev-worldbankrawbucket3f8fa072-gbpi5cbbpc6n",
    "tradeplatformstack-dev-wtorawbucket696208ec-tlfrrf6s0rg7",

    # Explicit-named buckets from old shipping sources (orphaned after pivot deploy)
    "dev-trade-comtrade-raw-<ACCOUNT_ID>",
    "dev-trade-comtrade-processed-<ACCOUNT_ID>",
    "dev-trade-unctad-raw-<ACCOUNT_ID>",
    "dev-trade-unctad-processed-<ACCOUNT_ID>",
    "dev-trade-eia-raw-<ACCOUNT_ID>",
    "dev-trade-eia-processed-<ACCOUNT_ID>",
    "dev-trade-acled-raw-<ACCOUNT_ID>",
    "dev-trade-acled-processed-<ACCOUNT_ID>",
    "dev-trade-imf-raw-<ACCOUNT_ID>",
    "dev-trade-imf-processed-<ACCOUNT_ID>",
    "dev-trade-wto-raw-<ACCOUNT_ID>",
    "dev-trade-wto-processed-<ACCOUNT_ID>"
)

foreach ($bucket in $old_buckets) {
    Write-Host "Deleting $bucket ..."
    aws s3 rb "s3://$bucket" --force
}

Write-Host "Done."
