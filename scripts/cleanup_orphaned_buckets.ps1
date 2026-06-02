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
    "dev-trade-comtrade-raw-197411402303",
    "dev-trade-comtrade-processed-197411402303",
    "dev-trade-unctad-raw-197411402303",
    "dev-trade-unctad-processed-197411402303",
    "dev-trade-eia-raw-197411402303",
    "dev-trade-eia-processed-197411402303",
    "dev-trade-acled-raw-197411402303",
    "dev-trade-acled-processed-197411402303",
    "dev-trade-imf-raw-197411402303",
    "dev-trade-imf-processed-197411402303",
    "dev-trade-wto-raw-197411402303",
    "dev-trade-wto-processed-197411402303"
)

foreach ($bucket in $old_buckets) {
    Write-Host "Deleting $bucket ..."
    aws s3 rb "s3://$bucket" --force
}

Write-Host "Done."
