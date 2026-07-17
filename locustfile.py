"""
Phase 8 load test — validates cache hit/miss latency under concurrent load.

Simulates a realistic traffic mix: most requests repeat a small set of
"popular" questions (should mostly hit the semantic cache), a smaller
share ask brand-new questions (guaranteed cache misses, full pipeline).
This mirrors real usage better than either "all repeats" (unrealistically
cache-friendly) or "all unique" (unrealistically cache-hostile).

Run:
    uv run locust -f locustfile.py --host http://localhost:8000

Then open http://localhost:8089, set:
    Number of users: 10
    Ramp up: 2 (users/sec)
Click Start.

Targets (revised from the original <50ms/<3s spec after Step 4 profiling
showed real network calls to Jina + Upstash make sub-second cache hits
unrealistic on the free tier):
    Cache hit  (popular question, repeated): under ~3s p95
    Cache miss (new question, full pipeline): under ~30s p95
"""

import random

from locust import HttpUser, task, between

# A small, fixed pool of questions — repeatedly asking these should
# produce cache hits after the first request seeds each one.
POPULAR_QUESTIONS = [
    "What is attention in transformers?",
    "What is a transformer model?",
    "How does BERT work?",
    "What is the vanishing gradient problem?",
]

# Randomized per-request suffix guarantees a cache MISS every time —
# no two of these will ever match an existing cache entry above the
# 0.80 similarity threshold, since they're each functionally distinct
# questions, not paraphrases of each other.
NOVEL_QUESTION_TEMPLATES = [
    "Explain the significance of paper {n} in NLP research",
    "What does study number {n} conclude about model scaling?",
    "Summarize the key contribution of research topic {n}",
]


class AskUser(HttpUser):
    # Wait 1-3 seconds between requests per simulated user — approximates
    # a real person reading an answer before asking the next question,
    # rather than hammering the endpoint with zero pause.
    wait_time = between(1, 3)

    @task(7)
    def ask_popular_question(self):
        """
        70% of requests: ask one of a small fixed set of questions.
        First time each one is asked, it's a cache MISS (runs the full
        graph and seeds the cache). Every subsequent ask of the same
        question should be a cache HIT.
        """
        question = random.choice(POPULAR_QUESTIONS)
        self._send_ask(question, label="/ask [popular]")

    @task(3)
    def ask_novel_question(self):
        """
        30% of requests: a guaranteed-unique question. Always a cache
        MISS — this is what exercises the full 15-30s pipeline under
        concurrent load, which is the more expensive path to validate.
        """
        template = random.choice(NOVEL_QUESTION_TEMPLATES)
        question = template.format(n=random.randint(1, 100_000))
        self._send_ask(question, label="/ask [novel]")

    def _send_ask(self, question: str, label: str):
        with self.client.post(
            "/api/v1/ask",
            json={"question": question},
            name=label,          # groups stats by label instead of raw URL
            catch_response=True,
            timeout=60,           # generous ceiling — a cache-miss full
                                  # pipeline run can legitimately take 20-30s
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return

            body = response.text
            if '"node": "error"' in body:
                response.failure("SSE stream contained an error event")
            elif '"node": "done"' not in body:
                response.failure("SSE stream never reached done event")
            else:
                response.success()
                