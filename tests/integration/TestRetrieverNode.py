"""
Throwaway script to test the retriever node standalone, outside FastAPI's
lifespan. Delete this once the graph is wired into the real app — normally
init_opensearch() is called once at startup in main.py's lifespan, not
manually like this.
"""

import asyncio

from src.services.agents.nodes.retriever import retriever_node
from src.services.search.client import init_opensearch


async def main() -> None:
    await init_opensearch()

    result = await retriever_node(
        {"search_query": "gradient vanishing problem", "retrieval_attempts": 0}
    )

    print(f"chunks found: {len(result['chunks'])}")
    if result["chunks"]:
        for c in result["chunks"]:
            print(f"  - [{c['score']}] {c['paper_title']} :: {c['text'][:80]}...")
    else:
        print("  none — check that arxiv_chunks index has documents")


if __name__ == "__main__":
    asyncio.run(main())