"""
Rewriter node: fires when the grader scores retrieval too low. Asks Groq
to reformulate the search query — broader terms, synonyms, different
phrasing — then hands it back to the retriever for another pass.

This node does NOT touch state["query"] (the user's original question,
used for the final answer's phrasing later) — it only updates
state["search_query"], which is what the retriever actually searches
with. Keeping these separate means the final answer still speaks to what
the user actually asked, even if we had to search around it internally.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langfuse.decorators import observe, langfuse_context

from src.config import get_settings
from src.logger import get_logger
from src.services.agents.state import AgentState
from src.services.rate_limit import groq_semaphore


logger = get_logger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a search query rewriter for an academic paper \
search system. The user's question didn't retrieve relevant results on \
the last attempt. Rewrite the search query to find better matches — try \
broader terminology, synonyms, or a different angle on the same question. \
Do NOT change what's being asked, only how it's phrased for search.

Respond with ONLY a JSON object, no other text:
{"rewritten_query": "..."}
"""


def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.3,   # a little variety helps avoid rewriting to the same query
        max_tokens=150,
        timeout=30,
    )


@observe(name="rewriter_node")
async def rewriter_node(state: AgentState) -> dict:
    """Rewrite state['search_query'] based on the failed retrieval attempt."""
    original_query = state["query"]
    previous_search_query = state.get("search_query", original_query)
    graded_chunks = state.get("graded_chunks", [])

    context_lines = [
        f"Original question: {original_query}",
        f"Previous search query tried: {previous_search_query}",
    ]
    if graded_chunks:
        context_lines.append("Chunks retrieved but scored low relevance:")
        for c in graded_chunks[:3]:
            context_lines.append(f'- "{c["paper_title"]}": {c["text"][:150]}...')
    else:
        context_lines.append("No chunks were retrieved at all.")

    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="\n".join(context_lines)),
    ]

    try:
        async with groq_semaphore:
            response = await llm.ainvoke(messages)
            parsed = json.loads(response.content)
            rewritten_query = str(parsed["rewritten_query"]).strip()
            if not rewritten_query:
                raise ValueError("empty rewritten_query")
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        # Fall back to the original question verbatim rather than blocking
        # the loop — worst case the retriever just retries the same search.
        logger.warning("rewriter_parse_failed", error=str(exc), query=original_query)
        rewritten_query = original_query

    logger.info(
        "rewriter_complete",
        original_query=original_query[:100],
        rewritten_query=rewritten_query[:100],
    )

    langfuse_context.update_current_observation(
        output={"rewritten_query": rewritten_query},
    )

    return {
        "rewritten_query": rewritten_query,
        "search_query": rewritten_query,   # this is what the retriever reads next
    }