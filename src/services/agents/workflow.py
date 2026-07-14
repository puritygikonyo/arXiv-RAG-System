"""
LangGraph workflow: wires the five nodes into the full agentic RAG graph.

    START
      |
      v
  guardrail ----(off-topic)----> reject ----> END
      |
   (on-topic)
      v
  retriever
      |
      v
   grader
      |
      +--(avg_relevance >= threshold)-------------------> generator --> END
      |
      +--(avg_relevance < threshold, attempts < max)----> rewriter --> retriever (loop)
      |
      +--(avg_relevance < threshold, attempts exhausted)-> generator --> END
                                                            (generator itself handles
                                                             the "nothing relevant"
                                                             case — see generator.py)
"""

from langgraph.graph import END, START, StateGraph

from src.config import get_settings
from src.logger import get_logger
from src.services.agents.nodes.generator import generator_node
from src.services.agents.nodes.grader import grader_node
from src.services.agents.nodes.guardrail import guardrail_node
from src.services.agents.nodes.retriever import retriever_node
from src.services.agents.nodes.rewriter import rewriter_node
from src.services.agents.state import AgentState

logger = get_logger(__name__)
settings = get_settings()


async def reject_node(state: AgentState) -> dict:
    """Terminal node for off-topic questions. No LLM call needed here —
    the guardrail already did the classification and gave us a reason."""
    reason = state.get("off_topic_reason", "")
    logger.info("agent_rejected_query", query=state["query"][:100], reason=reason)
    return {
        "answer": (
            "That doesn't look like something I can answer from the arXiv "
            "paper database. I can help with questions about research "
            "papers, algorithms, models, and academic topics in CS, "
            "physics, math, and related fields."
        ),
        "citations": [],
        "status": "off_topic",
    }


def _route_after_guardrail(state: AgentState) -> str:
    return "retriever" if state["is_on_topic"] else "reject"


def _route_after_grader(state: AgentState) -> str:
    threshold = settings.agent_relevance_threshold
    attempts = state.get("retrieval_attempts", 0)
    max_attempts = state.get(
        "max_retrieval_attempts", settings.agent_max_retrieval_attempts
    )

    if state.get("avg_relevance", 0.0) >= threshold:
        return "generator"
    if attempts < max_attempts:
        return "rewriter"
    # retries exhausted, still low relevance — let the generator produce
    # the "couldn't find enough material" message itself rather than
    # looping forever
    return "generator"


def build_agent_graph():
    """Construct and compile the LangGraph agentic RAG workflow."""
    graph = StateGraph(AgentState)

    graph.add_node("guardrail", guardrail_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("grader", grader_node)
    graph.add_node("rewriter", rewriter_node)
    graph.add_node("generator", generator_node)
    graph.add_node("reject", reject_node)

    graph.add_edge(START, "guardrail")

    graph.add_conditional_edges(
        "guardrail",
        _route_after_guardrail,
        {"retriever": "retriever", "reject": "reject"},
    )

    graph.add_edge("retriever", "grader")

    graph.add_conditional_edges(
        "grader",
        _route_after_grader,
        {"generator": "generator", "rewriter": "rewriter"},
    )

    graph.add_edge("rewriter", "retriever")   # loop back for another pass

    graph.add_edge("generator", END)
    graph.add_edge("reject", END)

    return graph.compile()


# Compiled once at import time and reused across requests — compiling is
# not free, and the graph structure never changes at runtime.
agent_graph = build_agent_graph()