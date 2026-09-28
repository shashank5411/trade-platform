import boto3
import json

s3v = boto3.client('s3vectors', region_name='us-east-2')
bucket = 'dev-trade-vectors-<ACCOUNT_ID>'
index  = 'documents-index'

# List all vector keys and delete in batches of 500
deleted = 0
next_token = None

while True:
    kwargs = dict(vectorBucketName=bucket, indexName=index, maxResults=500, returnMetadata=False)
    if next_token:
        kwargs['nextToken'] = next_token
    resp   = s3v.list_vectors(**kwargs)
    keys   = [v['key'] for v in resp.get('vectors', [])]
    if keys:
        s3v.delete_vectors(vectorBucketName=bucket, indexName=index, keys=keys)
        deleted += len(keys)
        print(f'Deleted {deleted} vectors...')
    next_token = resp.get('nextToken')
    if not next_token:
        break

print(f'Done. Total deleted: {deleted}')
