#!/bin/bash
# Starts OpenSearch in the background, waits until it's actually ready
# to accept requests, then starts the FastAPI app in the foreground.
#
# Render sets the PORT environment variable dynamically at runtime —
# unlike HF Spaces' fixed 7860, Render can assign a different port,
# so the app must read whatever value it's given rather than hardcode one.
set -e

echo "Starting OpenSearch..."
/usr/share/opensearch/opensearch-docker-entrypoint.sh &

echo "Waiting for OpenSearch to become healthy..."
until curl -s http://localhost:9200/_cluster/health > /dev/null; do
  sleep 2
  echo "  ...still waiting for OpenSearch"
done
echo "OpenSearch is up."

echo "Starting FastAPI app..."
cd /app
exec uv run uvicorn src.main:app --host 0.0.0.0 --port "${PORT:-8000}"