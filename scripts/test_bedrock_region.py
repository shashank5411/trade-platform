import boto3, json, time

# Test Cohere Embed v4
print("Testing Cohere Embed v4...")
b = boto3.client('bedrock-runtime', region_name='us-east-1')
body = json.dumps({
    "texts": ["test financial embedding for SEC filing"],
    "input_type": "search_document",
    "embedding_types": ["float"]
})
for attempt in range(3):
    try:
        r = b.invoke_model(
            modelId='cohere.embed-v4',
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        result = json.loads(r['body'].read())
        emb = result['embeddings']['float'][0]
        print('Cohere v4 — Success! Length: ' + str(len(emb)))
        break
    except Exception as e:
        print(f'  Attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:100]}')
        time.sleep(5)

# Also test Cohere Embed v3 as fallback
print("\nTesting Cohere Embed v3...")
body_v3 = json.dumps({
    "texts": ["test financial embedding"],
    "input_type": "search_document"
})
for attempt in range(3):
    try:
        r = b.invoke_model(
            modelId='cohere.embed-english-v3',
            body=body_v3,
            contentType='application/json',
            accept='application/json'
        )
        result = json.loads(r['body'].read())
        emb = result['embeddings'][0]
        print('Cohere v3 — Success! Length: ' + str(len(emb)))
        break
    except Exception as e:
        print(f'  Attempt {attempt+1} failed: {type(e).__name__}: {str(e)[:100]}')
        time.sleep(5)