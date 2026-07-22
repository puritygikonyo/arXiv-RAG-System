# Render deployment — FastAPI app only.
# OpenSearch now runs externally on Aiven's managed free tier, so this
# container no longer bundles OpenSearch — just the Python app itself.

FROM python:3.14-slim

# Install uv (Astral's installer script)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy dependency files first so Docker can cache this layer
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --all-extras --no-dev

# Now copy the rest of the application
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY src/certs/ ./src/certs/

EXPOSE 8000

# Render assigns PORT dynamically at runtime — shell form so ${PORT}
# actually expands, unlike exec form which passes it literally
CMD uv run uvicorn src.main:app --host 0.0.0.0 --port "${PORT:-8000}" 

