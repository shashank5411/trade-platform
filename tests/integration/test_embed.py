# INTEGRATION/LIVE: reads real documents from S3 (PROSE_BUCKET) via
# ingestion/etl/etl_embed.py. Requires AWS credentials and must be run
# from the repo root: python tests/integration/test_embed.py
import sys
sys.path.insert(0, 'ingestion')
from etl.etl_embed import *

prose_df = read_processed_docs(PROSE_BUCKET, 'documents_prose/year=2025/entity=AAPL/')

all_chunks = []
doc_metadata = {}
for _, row in prose_df.iterrows():
    prose_doc_id = str(row['doc_id'])
    doc_metadata[prose_doc_id] = {
        'source':   'EDGAR',
        'entity':   'AAPL',
        'doc_type': row.get('section_name', ''),
        'doc_date': str(row.get('filed_date', '')),
        'title':    row.get('section_title', ''),
    }
    text = str(row.get('text', ''))
    if text and len(text) >= 50:
        chunks = chunk_edgar(text, prose_doc_id)
        all_chunks.extend(chunks)

print('Total chunks:', len(all_chunks))
print('Sample chunk doc_id:', all_chunks[0]['doc_id'])
print('Sample chunk_id:', all_chunks[0]['chunk_id'])
print('Metadata lookup:', doc_metadata.get(all_chunks[0]['doc_id'], {}))
