"""
Grader node: scores each retrieved chunk for relevance to the user's
question. Sends all chunks in a single Groq call (rather than one call
per chunk) so the model can judge them relative to each other and so we
don't burn N round trips for N chunks.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langfuse.decorators import observe, langfuse_context

from src.config import get_settings
from src.logger import get_logger
from src.services.agents.state import AgentState, ChunkResult
from src.services.rate_limit import groq_semaphore


logger = get_logger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a relevance grader for a research paper search \
system. You'll be given a user's question and a list of text chunks \
retrieved from academic papers. For EACH chunk, decide how relevant it is \
to answering the question.

Respond with ONLY a JSON array, no other text, one entry per chunk in the \
same order given:
[{"chunk_id": "...", "relevance": 0.0}, {"chunk_id": "...", "relevance": 0.0}]

relevance is a float from 0.0 (completely unrelated) to 1.0 (directly \
answers the question). Be strict — a chunk that's merely on the same \
general topic but doesn't address the specific question should score low \
(0.2-0.4), not high.
"""


def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,
        max_tokens=settings.llm_max_tokens,
        timeout=30,
    )


def _build_user_message(query: str, chunks: list[ChunkResult]) -> str:
    lines = [f"Question: {query}", "", "Chunks:"]
    for c in chunks:
        # truncate chunk text so we don't blow the context window on
        # long chunks when there are several of them
        snippet = c["text"][:600]
        lines.append(f'- chunk_id="{c["chunk_id"]}" from "{c["paper_title"]}": {snippet}')
    return "\n".join(lines)


@observe(name="grader_node")
async def grader_node(state: AgentState) -> dict:
    """Score each chunk in state['chunks'] for relevance to the query."""
    query = state["query"]
    chunks = state.get("chunks", [])

    if not chunks:
        logger.info("grader_no_chunks", query=query)
        langfuse_context.update_current_observation(
            output={"graded_chunks": 0, "avg_relevance": 0.0},
        )
        return {"graded_chunks": [], "avg_relevance": 0.0}

    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_user_message(query, chunks)),
    ]

    try:
        async with groq_semaphore:
            response = await llm.ainvoke(messages)
            scores = json.loads(response.content)
            score_by_id = {s["chunk_id"]: float(s["relevance"]) for s in scores}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        # Fail safe here means the OPPOSITE of the guardrail: if grading
        # breaks, treat everything as low relevance (0.3) rather than
        # blindly trusting ungraded chunks and generating from noise.
        logger.warning("grader_parse_failed", error=str(exc), query=query)
        score_by_id = {c["chunk_id"]: 0.3 for c in chunks}

    graded_chunks: list[ChunkResult] = [
        {**c, "relevance": score_by_id.get(c["chunk_id"], 0.3)}
        for c in chunks
    ]

    avg_relevance = sum(c["relevance"] for c in graded_chunks) / len(graded_chunks)

    logger.info(
        "grader_complete",
        query=query[:100],
        chunk_count=len(graded_chunks),
        avg_relevance=round(avg_relevance, 3),
    )

    langfuse_context.update_current_observation(
        output={
            "chunk_count": len(graded_chunks),
            "avg_relevance": round(avg_relevance, 3),
        },
    )
    # Numeric score attached to this observation — this is what makes
    # avg_relevance filterable/sortable in the Langfuse dashboard as a
    # first-class metric, rather than something buried in output JSON.
    langfuse_context.score_current_observation(
        name="avg_relevance",
        value=avg_relevance,
    )

    return {
        "graded_chunks": graded_chunks,
        "avg_relevance": avg_relevance,
    }