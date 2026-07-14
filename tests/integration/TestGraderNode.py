"""
Throwaway script to test retriever + grader chained together.
Delete once the graph is wired into the real app.
"""

import asyncio

from src.services.agents.nodes.grader import grader_node
from src.services.agents.nodes.retriever import retriever_node
from src.services.search.client import init_opensearch


async def main() -> None:
    await init_opensearch()

    query = "How does the transformer architecture handle long sequences?"

    retrieval_result = await retriever_node(
        {"search_query": query, "retrieval_attempts": 0}
    )
    print(f"chunks found: {len(retrieval_result['chunks'])}")

    grading_result = await grader_node(
        {"query": query, "chunks": retrieval_result["chunks"]}
    )

    print(f"avg_relevance: {grading_result['avg_relevance']:.3f}")
    for c in grading_result["graded_chunks"]:
        print(f"  - relevance={c['relevance']:.2f}  {c['paper_title']}")


if __name__ == "__main__":
    asyncio.run(main())