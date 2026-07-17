
"""
Client-side concurrency throttling for third-party API calls.

Load testing (Step 6) showed that 10 concurrent users produced enough
simultaneous Jina and Groq calls to trip their free-tier rate limits
(429s from Jina, automatic retry-with-backoff from Groq's client).

These semaphores don't raise your actual throughput ceiling — that's
set by the provider, not by us. What they do is cap how many requests
our own app fires at each provider simultaneously, so extra requests
queue politely on our side instead of all firing at once and getting
rejected. A queued request is slower but succeeds; a rejected request
fails and (for Jina) surfaces as a hard error in the pipeline.

Limits are conservative starting points — 3 concurrent calls per
provider — chosen to be comfortably under typical free-tier ceilings
without serializing everything down to 1-at-a-time. Tune based on
what each provider's dashboard reports as your actual rate limit.
"""

import asyncio

# Caps concurrent Jina embedding calls (retriever.py, semantic_cache.py)
jina_semaphore = asyncio.Semaphore(3)

# Caps concurrent Groq LLM calls (guardrail, grader, rewriter, generator)
groq_semaphore = asyncio.Semaphore(3)