# Phase 10 — HuggingFace Spaces deployment
#
# One container running two processes: OpenSearch (search engine) and
# your FastAPI app. Bundled together because there's no free hosted
# OpenSearch option at any usable scale — see phase 10 discussion.
#
# Base image: official OpenSearch image, which already includes the
# bundled JDK OpenSearch needs. Python is installed on top via uv,
# which downloads its own standalone Python build rather than relying
# on the base image's package manager — avoids Python-version
# mismatches with your local dev environment.

FROM opensearchproject/opensearch:2.19.0

# The OpenSearch image runs as a non-root 'opensearch' user by default.
# Switch to root briefly to install Python tooling and app dependencies.
USER root

# Install uv (Astral's installer script — no system package manager needed)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install the exact Python version your app was developed against
RUN uv python install 3.14

WORKDIR /app

# Copy dependency files first so Docker can cache this layer — rebuilds
# are much faster when only application code changes, not dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --all-extras --no-dev

# Now copy the rest of the application
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# HuggingFace Spaces expects the container to listen on port 7860
EXPOSE 8000

# OpenSearch config for a single-node, resource-constrained container:
# - discovery.type=single-node: don't try to form a multi-node cluster
# - DISABLE_SECURITY_PLUGIN: skip TLS/auth setup, matches your local
#   dev config (opensearch_use_ssl=False) — fine for a demo, not for
#   anything handling real sensitive data
# - Xms/Xmx capped at 512m: HF's free tier has limited RAM shared
#   between OpenSearch and your Python app in the same container
ENV discovery.type=single-node
ENV DISABLE_SECURITY_PLUGIN=true
ENV OPENSEARCH_JAVA_OPTS="-Xms512m -Xmx512m"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]