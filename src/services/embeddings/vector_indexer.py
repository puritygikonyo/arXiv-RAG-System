"""
Vector Indexer — generates embeddings and stores them in OpenSearch.

THIS IS THE BRIDGE between:
  - The chunking service (produces Chunk objects)
  - The Jina AI client (produces vectors)
  - OpenSearch (stores vectors for similarity search)

WHAT HAPPENS HERE:

  Input:  [Chunk("attention mechanisms..."), Chunk("we propose..."), ...]
               ↓
  Step 1: Extract text from each chunk
               ↓
  Step 2: Send to Jina AI in batches → get back vectors
               ↓
  Step 3: Pair each chunk with its vector
               ↓
  Step 4: Bulk index into OpenSearch chunks index
               ↓
  Output: Chunks are searchable by both keyword AND semantic similarity

TWO INDEXES:
  arxiv_papers  → one document per paper (BM25 search on full paper)
  arxiv_chunks  → one document per chunk (vector search on sections)

  Phase 6 hybrid search queries BOTH and combines results.
"""

from datetime import UTC, datetime

from src.config import get_settings
from src.logger import get_logger
from src.services.embeddings.chunker import Chunk, chunk_paper
from src.services.embeddings.jina import embed_texts_batched
from src.services.search.client import get_opensearch_client

logger = get_logger(__name__)
settings = get_settings()

# Mapping for the chunks index
# Simpler than the papers index — mainly stores the vector + metadata
CHUNKS_INDEX_MAPPING: dict = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.knn": True,
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "arxiv_id": {"type": "keyword"},
            "text": {"type": "text"},
            "chunk_index": {"type": "integer"},
            "total_chunks": {"type": "integer"},
            "title": {"type": "text"},
            "primary_category": {"type": "keyword"},
            "ingested_at": {
                "type": "date",
                "format": "strict_date_optional_time||epoch_millis",
            },
            # The actual vector — this is what enables semantic search
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,
            },
        }
    },
}


def ensure_chunks_index_exists() -> None:
    """Create the chunks index if it doesn't exist."""
    client = get_opensearch_client()
    index_name = settings.opensearch_chunks_index

    if client.indices.exists(index=index_name):
        logger.info("chunks_index_exists", index=index_name)
        return

    logger.info("chunks_index_creating", index=index_name)
    client.indices.create(index=index_name, body=CHUNKS_INDEX_MAPPING)
    logger.info("chunks_index_created", index=index_name)


def index_paper_with_embeddings(paper: dict) -> dict:
    """
    Full pipeline for a single paper:
      1. Chunk the paper text
      2. Generate embeddings for all chunks
      3. Index chunks into OpenSearch
      4. Update the paper's own embedding in the papers index

    Args:
        paper: dict with arxiv_id, title, abstract, etc.

    Returns:
        dict with counts: {"chunks_indexed": N, "errors": N}
    """
    arxiv_id = paper["arxiv_id"]
    logger.info("indexing_paper_with_embeddings", arxiv_id=arxiv_id)

    # Step 1: chunk the paper
    chunks = chunk_paper(paper)
    if not chunks:
        logger.warning("no_chunks_produced", arxiv_id=arxiv_id)
        return {"chunks_indexed": 0, "errors": 0}

    # Step 2: generate embeddings for all chunk texts
    chunk_texts = [chunk.text for chunk in chunks]
    try:
        vectors = embed_texts_batched(chunk_texts)
    except Exception as e:
        logger.error("embedding_failed", arxiv_id=arxiv_id, error=str(e))
        return {"chunks_indexed": 0, "errors": len(chunks)}

    # Step 3: index chunks with their embeddings
    result = _bulk_index_chunks(chunks, vectors)

    # Step 4: update the paper's own embedding in the papers index
    # Use the first chunk's embedding as the paper-level embedding
    # (represents the title + beginning of abstract)
    if vectors:
        _update_paper_embedding(arxiv_id, vectors[0])

    logger.info(
        "paper_embedding_complete",
        arxiv_id=arxiv_id,
        chunks=len(chunks),
        indexed=result["chunks_indexed"],
    )

    return result


def _bulk_index_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> dict:
    """
    Bulk index chunks with their embeddings into OpenSearch.

    Args:
        chunks: list of Chunk objects
        vectors: list of embedding vectors (same order as chunks)
    """
    client = get_opensearch_client()

    if len(chunks) != len(vectors):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(vectors)} vectors"
        )

    bulk_body = []
    now = datetime.now(UTC).isoformat()

    for chunk, vector in zip(chunks, vectors):
        # Action line
        action = {
            "index": {
                "_index": settings.opensearch_chunks_index,
                "_id": chunk.chunk_id,
            }
        }

        # Document with embedding
        document = {
            "chunk_id": chunk.chunk_id,
            "arxiv_id": chunk.arxiv_id,
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "title": chunk.title,
            "primary_category": chunk.primary_category,
            "embedding": vector,         # ← the 1024-dimensional vector
            "ingested_at": now,
        }

        bulk_body.append(action)
        bulk_body.append(document)

    response = client.bulk(body=bulk_body, refresh=True)

    indexed = sum(
        1 for item in response["items"]
        if item["index"]["status"] in (200, 201)
    )
    errors = len(chunks) - indexed

    logger.info(
        "chunks_bulk_indexed",
        total=len(chunks),
        indexed=indexed,
        errors=errors,
    )

    return {"chunks_indexed": indexed, "errors": errors}


def _update_paper_embedding(arxiv_id: str, vector: list[float]) -> None:
    """
    Update the embedding field on the paper document in the papers index.

    This allows hybrid search to also do vector search at the paper level,
    not just at the chunk level.
    """
    client = get_opensearch_client()

    try:
        client.update(
            index=settings.opensearch_papers_index,
            id=arxiv_id,
            body={"doc": {"embedding": vector}},
        )
        logger.debug("paper_embedding_updated", arxiv_id=arxiv_id)
    except Exception as e:
        # Non-fatal — BM25 search still works without this
        logger.warning(
            "paper_embedding_update_failed",
            arxiv_id=arxiv_id,
            error=str(e),
        )


def bulk_index_papers_with_embeddings(papers: list[dict]) -> dict:
    """
    Process multiple papers through the full embedding pipeline.

    This is called by the Airflow DAG in Phase 8 to process
    all newly ingested papers in one batch.

    Args:
        papers: list of paper dicts

    Returns:
        aggregate counts across all papers
    """
    total_chunks = 0
    total_errors = 0

    for paper in papers:
        result = index_paper_with_embeddings(paper)
        total_chunks += result["chunks_indexed"]
        total_errors += result["errors"]

    logger.info(
        "bulk_embedding_complete",
        papers=len(papers),
        total_chunks=total_chunks,
        total_errors=total_errors,
    )

    return {
        "papers_processed": len(papers),
        "chunks_indexed": total_chunks,
        "errors": total_errors,
    }
