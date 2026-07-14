"""
End-to-end test of the full Phase 7 agent graph — guardrail through
generator. Delete once you're comfortable and/or have converted this
into a real pytest test.
"""

import asyncio

from src.services.agents.workflow import agent_graph
from src.services.search.client import init_opensearch


async def ask(question: str) -> None:
    print(f"\n{'=' * 70}\nQ: {question}\n{'=' * 70}")

    result = await agent_graph.ainvoke({"query": question})

    print(f"status: {result['status']}")
    print(f"retrieval_attempts: {result.get('retrieval_attempts', 0)}")
    print(f"avg_relevance: {result.get('avg_relevance', 0):.3f}")
    print(f"citations: {result.get('citations', [])}")
    print(f"\nanswer:\n{result['answer']}")


async def main() -> None:
    await init_opensearch()

    # on-topic, should retrieve, grade well, and generate an answer
    await ask("How does the transformer architecture handle long sequences?")

    # off-topic, should be rejected by the guardrail immediately
    await ask("What's a good pizza topping?")


if __name__ == "__main__":
    asyncio.run(main())