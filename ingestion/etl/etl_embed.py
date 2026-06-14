"""
ETL: Embedding pipeline — documents processed Parquet → S3 Vectors.

Reads canonical document rows from:
  SEC EDGAR processed bucket       (documents/)
  Wikipedia processed bucket       (documents/)
  SEC prose processed bucket       (documents_prose/) — Phase 7

Chunks text per source strategy:
  EDGAR:     section-level (Item X. headers), max 400 tokens, 50 overlap
  WIKIPEDIA: paragraph-level (\\n\\n splits), max 350 tokens, 30 overlap

Embeds each chunk with Bedrock Cohere Embed English v3 (1024 dims).
Writes vectors to S3 Vectors index: documents-index.

Run after etl_sec.py, etl_wikipedia.py, and etl_sec_prose.py.
"""
import sys
import os
import zipfile

# Glue places --extra-py-files zip in glue-python-libs-* but does not extract it
# Extract it manually so internal packages like utils/ are importable
_libs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _entry in os.listdir('/tmp/'):
    if _entry.startswith('glue-python-libs-'):
        _libs_dir = os.path.join('/tmp/', _entry)
        for _f in os.listdir(_libs_dir):
            if _f.endswith('.zip'):
                with zipfile.ZipFile(os.path.join(_libs_dir, _f)) as _z:
                    _z.extractall(_libs_dir)
        sys.path.insert(0, _libs_dir)
        break

import os
import sys
import json
import re
import boto3
import pandas as pd
from io import BytesIO
from datetime import datetime, timezone
import time

from botocore.config import Config

sys.path.insert(0, _libs_dir)
sys.path.insert(0, "/tmp/ingestion")

def _arg(key, default=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--{key}", default=os.environ.get(key.upper(), default))
    args, _ = parser.parse_known_args()
    return getattr(args, key)

ENV           = _arg("env",     "dev")
ACCOUNT       = _arg("account", "197411402303")
REGION        = _arg("region",  "us-east-2")

# Optional filters — scope the run to a single entity and/or year
# Usage: --entity AAPL --year 2025
# Omit to process everything
FILTER_ENTITY = _arg("entity", None)
FILTER_YEAR   = _arg("year",   None)

# S3 processed buckets
SEC_PROC_BUCKET  = _arg("sec_processed_bucket",
    f"{ENV}-trade-sec-processed-{ACCOUNT}")
WIKI_PROC_BUCKET = _arg("wiki_processed_bucket",
    f"{ENV}-trade-wikipedia-processed-{ACCOUNT}")
PROSE_BUCKET     = _arg("prose_bucket",
    f"{ENV}-trade-sec-prose-processed-{ACCOUNT}")

# S3 Vectors
VECTOR_BUCKET = _arg("vector_bucket",
    f"{ENV}-trade-vectors-{ACCOUNT}")
VECTOR_INDEX  = "documents-index"

# Bedrock
EMBED_MODEL_ID   = "cohere.embed-english-v3"
EMBED_DIM        = 1024
EMBED_BATCH_SIZE = 10
MAX_TOKENS_PER_CHUNK = {
    "EDGAR":     400,
    "WIKIPEDIA": 350,
}
OVERLAP_TOKENS = {
    "EDGAR":     50,
    "WIKIPEDIA": 30,
}

s3        = boto3.client("s3",              region_name=REGION)
bedrock   = boto3.client("bedrock-runtime", region_name="us-east-1")
s3vectors = boto3.client("s3vectors",       region_name=REGION)


# ── Tokenizer ─────────────────────────────────────────────────────────────

def approx_tokens(text: str) -> int:
    """~4 chars per token — good enough for chunking decisions."""
    return max(1, len(text) // 4)


# ── Chunkers ───────────────────────────────────────────────────────────────

def chunk_sliding(text: str, chunk_id_base: str, source: str,
                  root_doc_id: str = None) -> list:
    """
    Sliding window fallback — used when section/paragraph exceeds max.
    root_doc_id: the original document doc_id, used for metadata lookup.
                 Falls back to chunk_id_base if not provided.
    """
    max_tok   = MAX_TOKENS_PER_CHUNK[source]
    overlap   = OVERLAP_TOKENS[source]
    win_chars = max_tok * 4
    ovl_chars = overlap * 4
    step      = max(1, win_chars - ovl_chars)
    root      = root_doc_id or chunk_id_base
    chunks    = []
    pos       = 0
    idx       = 0
    while pos < len(text):
        chunk_text = text[pos:pos + win_chars]
        if chunk_text.strip():
            chunks.append({
                "chunk_id": f"{chunk_id_base}_w{idx}",
                "doc_id":   root,           # always the root doc_id
                "text":     chunk_text,
                "section":  idx,
            })
        pos += step
        idx += 1
    return chunks


def chunk_edgar(text: str, doc_id: str) -> list:
    """
    Section-level chunking for EDGAR filings.
    Splits on Item X. / Item XX. headers first.
    Falls back to token-window chunking if section exceeds max tokens.
    """
    max_tok  = MAX_TOKENS_PER_CHUNK["EDGAR"]
    overlap  = OVERLAP_TOKENS["EDGAR"]
    pattern  = re.compile(r'(?=Item\s+\d+[A-Z]?\.\s)', re.IGNORECASE)
    sections = pattern.split(text)
    sections = [s.strip() for s in sections if s.strip()]

    chunks = []
    for sec_idx, section in enumerate(sections):
        if approx_tokens(section) <= max_tok:
            chunks.append({
                "chunk_id": f"{doc_id}_s{sec_idx}",
                "doc_id":   doc_id,         # ← explicit doc_id
                "text":     section,
                "section":  sec_idx,
            })
        else:
            win_chars = max_tok * 4
            ovl_chars = overlap * 4
            step      = max(1, win_chars - ovl_chars)
            pos       = 0
            sub_idx   = 0
            while pos < len(section):
                chunk_text = section[pos:pos + win_chars]
                if chunk_text.strip():
                    chunks.append({
                        "chunk_id": f"{doc_id}_s{sec_idx}_{sub_idx}",
                        "doc_id":   doc_id, # ← explicit doc_id
                        "text":     chunk_text,
                        "section":  sec_idx,
                    })
                pos     += step
                sub_idx += 1

    if not chunks:
        chunks = chunk_sliding(text, doc_id, "EDGAR", root_doc_id=doc_id)

    return chunks


def chunk_wikipedia(text: str, doc_id: str) -> list:
    """
    Paragraph-level chunking for Wikipedia articles.
    Splits on double newlines. Merges short paragraphs.
    Falls back to sliding window if paragraph exceeds max tokens.
    """
    max_tok    = MAX_TOKENS_PER_CHUNK["WIKIPEDIA"]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks    = []
    chunk_idx = 0
    buffer    = ""

    for para in paragraphs:
        candidate = (buffer + "\n\n" + para).strip() if buffer else para
        if approx_tokens(candidate) <= max_tok:
            buffer = candidate
        else:
            if buffer:
                chunks.append({
                    "chunk_id": f"{doc_id}_p{chunk_idx}",
                    "doc_id":   doc_id,     # ← explicit doc_id
                    "text":     buffer,
                    "section":  chunk_idx,
                })
                chunk_idx += 1
            if approx_tokens(para) > max_tok:
                # pass root_doc_id so nested sliding chunks resolve correctly
                sub = chunk_sliding(
                    para,
                    f"{doc_id}_p{chunk_idx}",
                    "WIKIPEDIA",
                    root_doc_id=doc_id,     # ← root doc_id, not _p{N}
                )
                chunks.extend(sub)
                chunk_idx += len(sub)
                buffer = ""
            else:
                buffer = para

    if buffer:
        chunks.append({
            "chunk_id": f"{doc_id}_p{chunk_idx}",
            "doc_id":   doc_id,             # ← explicit doc_id
            "text":     buffer,
            "section":  chunk_idx,
        })

    return chunks


def chunk_document(row: pd.Series) -> list:
    """Route to correct chunker based on source."""
    doc_id = row["doc_id"]
    text   = str(row.get("text", ""))
    source = str(row.get("source", "")).upper()

    if not text or len(text) < 50:
        return []

    if source == "EDGAR":
        return chunk_edgar(text, doc_id)
    else:
        return chunk_wikipedia(text, doc_id)


# ── S3 helpers ─────────────────────────────────────────────────────────────

def read_processed_docs(bucket: str, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    """Read all Parquet files under prefix from processed bucket."""
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend([o["Key"] for o in page.get("Contents", [])
                     if o["Key"].endswith(".parquet")])

    if not keys:
        print(f"  No Parquet files found under s3://{bucket}/{prefix}")
        return pd.DataFrame(), []

    dfs = []
    for key in keys:
        obj    = s3.get_object(Bucket=bucket, Key=key)
        sub_df = pd.read_parquet(BytesIO(obj["Body"].read()))
        sub_df["_source_key"] = key
        dfs.append(sub_df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Read {len(df)} rows from s3://{bucket}/{prefix} "
          f"({len(keys)} files)")
    return df, keys


# ── Bedrock embedding ──────────────────────────────────────────────────────

def embed_text(text: str) -> list:
    """
    Embed a single text chunk with Cohere Embed English v3.
    input_type: search_document for indexing, search_query for querying.
    """
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            body = json.dumps({
                "texts":           [text[:2000]],
                "input_type":      "search_document",
                "embedding_types": ["float"],
            })
            resp = bedrock.invoke_model(
                modelId=EMBED_MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(resp["body"].read())
            return result["embeddings"]["float"][0]
        except Exception as e:
            if "ThrottlingException" in str(e):
                wait = 5 * (2 ** attempt)
                print(f"  Throttled — waiting {wait}s "
                      f"(attempt {attempt+1}/{max_attempts})")
                time.sleep(wait)
            else:
                print(f"  WARN: embed failed — {e}")
                return None
    print(f"  WARN: giving up after {max_attempts} attempts")
    return None


def embed_chunks(chunks: list) -> list:
    """
    Embed all chunks in batches of EMBED_BATCH_SIZE.
    Returns list of chunks with 'vector' field added.
    Falls back to one-at-a-time with backoff on ThrottlingException.
    """
    embedded = []
    total    = len(chunks)

    for batch_start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[batch_start:batch_start + EMBED_BATCH_SIZE]

        try:
            body = json.dumps({
                "texts":           [c["text"][:2000] for c in batch],
                "input_type":      "search_document",
                "embedding_types": ["float"],
            })
            resp    = bedrock.invoke_model(
                modelId=EMBED_MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            vectors = json.loads(resp["body"].read())["embeddings"]["float"]
            for chunk, vector in zip(batch, vectors):
                chunk["vector"] = vector
                embedded.append(chunk)

        except Exception as e:
            if "ThrottlingException" in str(e):
                print(f"  Throttled on batch — falling back to one-at-a-time")
                for chunk in batch:
                    vector = embed_text(chunk["text"])
                    if vector is not None:
                        chunk["vector"] = vector
                        embedded.append(chunk)
            else:
                print(f"  WARN: batch embed failed — {e}")

        time.sleep(0.5)

        processed = min(batch_start + EMBED_BATCH_SIZE, total)
        if processed % 10 == 0 or processed == total:
            pct = int(processed / total * 100)
            print(f"  Embedded {processed}/{total} chunks ({pct}%)")

    return embedded


# ── S3 Vectors write ───────────────────────────────────────────────────────

def write_vectors(chunks: list, doc_metadata: dict) -> int:
    """
    Write embedded chunks to S3 Vectors index.
    Uses chunk['doc_id'] directly — no fragile string parsing.
    Batches of 500 (S3 Vectors PutVectors limit).
    """
    if not chunks:
        return 0

    BATCH_SIZE    = 500
    total_written = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch   = chunks[i:i + BATCH_SIZE]
        vectors = []
        for chunk in batch:
            meta = doc_metadata.get(chunk["doc_id"], {})  # ← direct lookup
            vectors.append({
                "key":  chunk["chunk_id"],
                "data": {"float32": chunk["vector"]},
                "metadata": {
                    "doc_id":   chunk["doc_id"],
                    "source":   meta.get("source",   ""),
                    "entity":   meta.get("entity",   ""),
                    "doc_type": meta.get("doc_type", ""),
                    "doc_date": meta.get("doc_date", ""),
                    "title":    meta.get("title",    ""),
                    "text":     chunk["text"][:500],
                },
            })

        # Deduplicate by key within batch — prevents duplicate key errors
        seen_keys = set()
        unique_batch = []
        for chunk in vectors:
            if chunk["key"] not in seen_keys:
                seen_keys.add(chunk["key"])
                unique_batch.append(chunk)
        vectors = unique_batch
        if not vectors:
            continue

        try:
            s3vectors.put_vectors(
                vectorBucketName=VECTOR_BUCKET,
                indexName=VECTOR_INDEX,
                vectors=vectors,
            )
        except Exception as e:
            if "duplicate keys" in str(e).lower() or "ValidationException" in str(e):
                print(f"  WARN: duplicate keys in batch — skipping batch of {len(vectors)}")
                continue
            raise
        total_written += len(vectors)
        print(f"  Wrote {total_written}/{len(chunks)} vectors to S3 Vectors")

    return total_written


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    print(f"ETL Embed — documents → S3 Vectors | env={ENV}")
    print(f"  Vector bucket: {VECTOR_BUCKET}")
    print(f"  Vector index:  {VECTOR_INDEX}")
    print(f"  Embed model:   {EMBED_MODEL_ID}")
    if FILTER_ENTITY or FILTER_YEAR:
        print(f"  Filter: entity={FILTER_ENTITY or 'all'} "
              f"year={FILTER_YEAR or 'all'}")

    all_chunks    = []
    doc_metadata  = {}
    total_docs    = 0
    total_skipped = 0

    # ── Read SEC EDGAR documents ───────────────────────────────────────────
    # Skip if entity filter is set — EDGAR processed docs don't partition by entity
    if not FILTER_ENTITY:
        print("\nReading SEC EDGAR documents...")
        sec_df, _ = read_processed_docs(SEC_PROC_BUCKET, "documents/source=EDGAR/")
        if not sec_df.empty:
            for _, row in sec_df.iterrows():
                doc_metadata[row["doc_id"]] = {
                    "source":   "EDGAR",
                    "entity":   row.get("entity", ""),
                    "doc_type": row.get("doc_type", ""),
                    "doc_date": str(row.get("doc_date", "")),
                    "title":    row.get("title", ""),
                }
                chunks = chunk_document(row)
                if chunks:
                    all_chunks.extend(chunks)
                    total_docs += 1
                else:
                    total_skipped += 1

    # ── Read Wikipedia documents ───────────────────────────────────────────
    # Skip if entity filter is set — Wikipedia doesn't partition by entity
    if not FILTER_ENTITY:
        print("\nReading Wikipedia documents...")
        wiki_df, _ = read_processed_docs(WIKI_PROC_BUCKET, "documents/source=WIKIPEDIA/")
        if not wiki_df.empty:
            for _, row in wiki_df.iterrows():
                doc_metadata[row["doc_id"]] = {
                    "source":   "WIKIPEDIA",
                    "entity":   row.get("entity", ""),
                    "doc_type": "wiki_article",
                    "doc_date": str(row.get("doc_date", "")),
                    "title":    row.get("title", ""),
                }
                chunks = chunk_document(row)
                if chunks:
                    all_chunks.extend(chunks)
                    total_docs += 1
                else:
                    total_skipped += 1

    # ── Read SEC prose sections ────────────────────────────────────────────
    print("\nReading SEC prose sections...")
    # Build prefix based on filters — prose partitions by year and entity
    prose_prefix = "documents_prose/"
    if FILTER_YEAR and FILTER_ENTITY:
        prose_prefix = (f"documents_prose/year={FILTER_YEAR}/"
                        f"entity={FILTER_ENTITY}/")
    elif FILTER_YEAR:
        prose_prefix = f"documents_prose/year={FILTER_YEAR}/"
    elif FILTER_ENTITY:
        # No year filter — must scan all years for this entity
        # S3 doesn't support non-leading prefix scans so we list all and filter
        prose_prefix = "documents_prose/"

    prose_df, prose_keys = read_processed_docs(PROSE_BUCKET, prose_prefix)

    # Build entity lookup from Hive partition path — entity is not a column
    key_entity_map = {}
    for key in prose_keys:
        match = re.search(r'entity=([^/]+)/', key)
        if match:
            key_entity_map[key] = match.group(1)

    if not prose_df.empty:
        for _, row in prose_df.iterrows():
            prose_doc_id = str(row["doc_id"])
            entity_match = re.search(r'entity=([^/]+)/', row["_source_key"])
            entity = entity_match.group(1) if entity_match else FILTER_ENTITY or ""
            doc_metadata[prose_doc_id] = {
                "source":   "EDGAR",
                "entity":   entity,
                "doc_type": row.get("section_name", ""),
                "doc_date": str(row.get("filed_date", "")),
                "title":    row.get("section_title", ""),
            }
            text = str(row.get("text", ""))
            if text and len(text) >= 50:
                chunks = chunk_edgar(text, prose_doc_id)
                if chunks:
                    all_chunks.extend(chunks)
                    total_docs += 1
                else:
                    total_skipped += 1
            else:
                total_skipped += 1

    print(f"\nTotal: {total_docs} docs → {len(all_chunks)} chunks "
          f"({total_skipped} docs skipped — too short)")

    if not all_chunks:
        print("No chunks to embed — exiting.")
        return

    # ── Embed ──────────────────────────────────────────────────────────────
    print(f"\nEmbedding {len(all_chunks)} chunks with Cohere Embed v3...")
    embedded_chunks = embed_chunks(all_chunks)
    print(f"Successfully embedded: {len(embedded_chunks)}/{len(all_chunks)}")

    # ── Write to S3 Vectors ────────────────────────────────────────────────
    print(f"\nWriting {len(embedded_chunks)} vectors to S3 Vectors...")
    total_written = write_vectors(embedded_chunks, doc_metadata)

    print(f"\n{'─'*50}")
    print(f"Done. {total_docs} docs, {len(all_chunks)} chunks, "
          f"{total_written} vectors written.")


if __name__ == "__main__":
    main()