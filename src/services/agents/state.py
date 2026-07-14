"""
AgentState: the shared object passed between every node in the LangGraph
workflow. Each node reads the fields it needs and returns a dict of the
fields it wants to update — LangGraph merges that dict into the state.
"""

from typing import TypedDict, Literal


class ChunkResult(TypedDict):
    """A single retrieved chunk, carried through grading and generation."""
    chunk_id: str
    paper_id: str
    paper_title: str
    text: str
    score: float          # retrieval score (from hybrid search / RRF)
    relevance: float       # grader-assigned score, 0.0-1.0 (set later)


class AgentState(TypedDict):
    # --- input ---
    query: str                          # the user's original question

    # --- guardrail node output ---
    is_on_topic: bool
    off_topic_reason: str

    # --- retrieval node output ---
    search_query: str                   # query actually sent to hybrid search
                                         # (== query on first pass, rewritten after)
    chunks: list[ChunkResult]

    # --- grader node output ---
    graded_chunks: list[ChunkResult]    # chunks with relevance scores filled in
    avg_relevance: float

    # --- control flow ---
    retrieval_attempts: int             # guards against infinite rewrite loops
    max_retrieval_attempts: int

    # --- rewriter node output ---
    rewritten_query: str

    # --- generator node output ---
    answer: str
    citations: list[str]                # paper_ids cited in the answer

    # --- final status, used by the router / API layer ---
    status: Literal["pending", "off_topic", "no_relevant_docs", "answered"]