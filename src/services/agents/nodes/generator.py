"""
Generator node: the final node on the success path. Takes the graded
chunks (filtered to relevant ones) and the user's ORIGINAL question
(not the possibly-rewritten search query) and produces an answer with
inline citations back to paper_ids.

This node streams by default when called through the LangGraph astream
API from the FastAPI endpoint (Step 7) — here we just define the LLM
call; streaming is handled at the graph/endpoint level, not inside the
node itself.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.config import get_settings
from src.logger import get_logger
from src.services.agents.state import AgentState, ChunkResult

logger = get_logger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a research assistant that answers questions using \
excerpts from academic papers. You will be given a question and a set of \
text chunks, each labeled with the paper it came from.

Rules:
- Answer ONLY using information in the provided chunks. If the chunks don't \
fully answer the question, say what's missing rather than guessing.
- Cite claims inline using the paper title in parentheses, e.g. "...as shown \
in (Attention Is All You Need)."
- Be concise and direct. Do not pad the answer with generic filler.
- Do not fabricate paper titles or findings not present in the chunks.
"""

# only chunks at or above this relevance make it into the prompt, even if
# the overall average cleared the threshold that got us here
MIN_CHUNK_RELEVANCE = 0.5


def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        streaming=settings.llm_streaming,
        timeout=30,
    )


def _build_context(chunks: list[ChunkResult]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f'[Paper: "{c["paper_title"]}"]\n{c["text"]}')
    return "\n\n---\n\n".join(blocks)


async def generator_node(state: AgentState) -> dict:
    """Generate the final answer from the graded, relevant chunks."""
    query = state["query"]   # the user's original question, not search_query
    graded_chunks = state.get("graded_chunks", [])

    relevant_chunks = [
        c for c in graded_chunks if c["relevance"] >= MIN_CHUNK_RELEVANCE
    ]

    if not relevant_chunks:
        logger.warning("generator_no_relevant_chunks", query=query[:100])
        return {
            "answer": (
                "I couldn't find enough relevant material in the paper "
                "database to answer that confidently. Try rephrasing the "
                "question or asking about a more specific topic covered "
                "in the indexed papers."
            ),
            "citations": [],
            "status": "no_relevant_docs",
        }

    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Question: {query}\n\nChunks:\n\n{_build_context(relevant_chunks)}"
        ),
    ]

    response = await llm.ainvoke(messages)
    answer = response.content

    citations = sorted({c["paper_id"] for c in relevant_chunks})

    logger.info(
        "generator_complete",
        query=query[:100],
        chunks_used=len(relevant_chunks),
        citation_count=len(citations),
    )

    return {
        "answer": answer,
        "citations": citations,
        "status": "answered",
    }