"""
Guardrail node: the first stop in the graph. Uses a cheap, fast Groq call
to decide whether the user's question is answerable from the arXiv corpus
(research papers, ML/CS/physics/etc topics) before we spend a retrieval
and generation pass on it.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.config import get_settings
from src.logger import get_logger
from src.services.agents.state import AgentState

settings = get_settings()
logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a topic classifier for an arXiv research paper \
search assistant. Your only job is to decide if a user question is \
answerable using academic papers from arXiv (covers: computer science, \
physics, math, statistics, quantitative biology, economics, and related \
research fields).

Respond with ONLY a JSON object, no other text:
{"on_topic": true or false, "reason": "one short sentence"}

Examples of on_topic=true: questions about algorithms, model architectures, \
research findings, comparisons between papers, math/physics concepts, \
"what does paper X say about Y".

Examples of on_topic=false: general chit-chat, requests unrelated to \
research/academia, requests to write code unrelated to a paper, personal \
advice, current events, anything not answerable from academic papers.
"""


def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,
        max_tokens=100,
        timeout=30,
    )


async def guardrail_node(state: AgentState) -> dict:
    """Classify the query as on-topic or off-topic."""
    query = state["query"]
    llm = _get_llm()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {query}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        parsed = json.loads(response.content)
        is_on_topic = bool(parsed.get("on_topic", True))
        reason = str(parsed.get("reason", ""))
    except (json.JSONDecodeError, KeyError, AttributeError) as exc:
        # Fail open: if the classifier call breaks, let the query through
        # rather than blocking a legitimate user. Log it so you notice.
        logger.warning("guardrail_parse_failed", error=str(exc), query=query)
        is_on_topic = True
        reason = "guardrail parse failure, defaulted to on-topic"

    logger.info("guardrail_result", query=query, on_topic=is_on_topic, reason=reason)

    return {
        "is_on_topic": is_on_topic,
        "off_topic_reason": reason,
        "status": "pending" if is_on_topic else "off_topic",
        "search_query": query,          # seed search_query for the retrieval node
        "retrieval_attempts": 0,
        "max_retrieval_attempts": settings.agent_max_retrieval_attempts,
    }