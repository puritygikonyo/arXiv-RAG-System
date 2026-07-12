"""
Chunking Service — splits paper text into smaller pieces.

WHY DO WE CHUNK?

  Embedding models have a token limit (Jina v3 handles ~8192 tokens,
  but quality degrades with very long texts).

  More importantly, when a user searches, we want to find the EXACT
  section of a paper that answers their question — not just that the
  paper is generally relevant.

  Example:
    Paper: 8000 words covering 5 different topics
    Query: "gradient vanishing problem in RNNs"

    Without chunking:
      → Whole paper gets one embedding → might miss the specific section

    With chunking:
      → Paper split into 10 chunks → chunk 3 covers RNNs specifically
      → Chunk 3's embedding is very close to the query
      → We return the exact relevant section

CHUNKING STRATEGY — Sliding Window:

  We use a sliding window with overlap so context isn't lost at chunk
  boundaries. Think of it like this:

  Text: [-------- chunk 1 --------]
                          [-------- chunk 2 --------]
                                          [-------- chunk 3 --------]
                   ↑ overlap ↑    ↑ overlap ↑

  chunk_size    = 512 tokens (configured in .env)
  chunk_overlap = 50 tokens  (configured in .env)

  The overlap means the end of chunk 1 is repeated at the start of
  chunk 2. This prevents a sentence from being cut in half with
  neither chunk having full context.

WHAT WE CHUNK:
  For arXiv papers we chunk the abstract (title + abstract combined).
  Full paper text would require PDF parsing — that's a future enhancement.
"""

from dataclasses import dataclass

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class Chunk:
    """
    A single chunk of text from a paper.

    Stores both the text content AND metadata about where it came from
    so we can reconstruct the full paper context when returning results.
    """
    chunk_id: str           # unique ID: "{arxiv_id}_chunk_{index}"
    arxiv_id: str           # which paper this came from
    text: str               # the actual text content
    chunk_index: int        # position in the paper (0, 1, 2, ...)
    total_chunks: int       # total chunks this paper was split into
    title: str              # paper title (for display in results)
    primary_category: str   # paper category (for filtering)


def chunk_paper(paper: dict) -> list[Chunk]:
    """
    Split a paper into overlapping chunks for embedding.

    Args:
        paper: dict with keys: arxiv_id, title, abstract,
               primary_category, (optional) full_text

    Returns:
        list of Chunk objects ready for embedding and indexing

    For arXiv papers we combine title + abstract as the text to chunk.
    If a full_text field is provided (future PDF extraction), we use that.
    """
    arxiv_id = paper["arxiv_id"]
    title = paper.get("title", "")

    # Use full_text if available (Phase 6+ enhancement), otherwise title + abstract
    text_to_chunk = paper.get("full_text") or f"{title}\n\n{paper.get('abstract', '')}"

    # Clean up whitespace
    text_to_chunk = " ".join(text_to_chunk.split())

    if not text_to_chunk.strip():
        logger.warning("empty_text_for_chunking", arxiv_id=arxiv_id)
        return []

    # Split into word-level tokens (simple but effective for English text)
    # In production you'd use a proper tokeniser like tiktoken
    words = text_to_chunk.split()

    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap

    # Generate chunks using sliding window
    chunks: list[Chunk] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append(Chunk(
            chunk_id=f"{arxiv_id}_chunk_{len(chunks)}",
            arxiv_id=arxiv_id,
            text=chunk_text,
            chunk_index=len(chunks),
            total_chunks=0,  # will update after we know total
            title=title,
            primary_category=paper.get("primary_category", ""),
        ))

        # Move window forward by (chunk_size - overlap)
        # This creates the overlapping effect
        step = chunk_size - chunk_overlap
        start += step

        # Stop if we've covered all the text
        if end == len(words):
            break

    # Now we know the total, update each chunk
    total = len(chunks)
    for chunk in chunks:
        chunk.total_chunks = total

    logger.debug(
        "paper_chunked",
        arxiv_id=arxiv_id,
        word_count=len(words),
        num_chunks=total,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )

    return chunks


def chunk_papers_batch(papers: list[dict]) -> list[Chunk]:
    """
    Chunk multiple papers at once.

    Args:
        papers: list of paper dicts

    Returns:
        flat list of all chunks from all papers
    """
    all_chunks: list[Chunk] = []

    for paper in papers:
        chunks = chunk_paper(paper)
        all_chunks.extend(chunks)

    logger.info(
        "batch_chunking_complete",
        papers=len(papers),
        total_chunks=len(all_chunks),
        avg_chunks_per_paper=round(len(all_chunks) / len(papers), 1) if papers else 0,
    )

    return all_chunks
