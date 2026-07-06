# =============================================================================
# arXiv RAG System — Makefile
# =============================================================================
# Run `make help` to see all available commands.
# All commands use `uv` for Python — install from https://docs.astral.sh/uv/
# =============================================================================

.DEFAULT_GOAL := help
.PHONY: help start stop restart status logs health setup format lint type-check test test-cov test-unit test-integration clean migrate seed

# Detect OS for open command
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
  OPEN_CMD = xdg-open
else
  OPEN_CMD = open
endif

# Colours for pretty output
CYAN  := \033[36m
GREEN := \033[32m
RESET := \033[0m

# =============================================================================
# HELP
# =============================================================================
help: ## Show this help message
	@echo ""
	@echo "  arXiv RAG System — available commands"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  $(CYAN)%-22s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# =============================================================================
# DOCKER SERVICES (OpenSearch only)
# =============================================================================
start: ## Start all Docker services (OpenSearch + Dashboards)
	@echo "$(GREEN)Starting services...$(RESET)"
	docker compose up -d
	@echo "$(GREEN)Services started. Run 'make health' to verify.$(RESET)"

stop: ## Stop all Docker services
	@echo "$(CYAN)Stopping services...$(RESET)"
	docker compose down

restart: ## Restart all Docker services
	docker compose down
	docker compose up -d

status: ## Show running Docker service status
	docker compose ps

logs: ## Tail logs from all services (Ctrl+C to exit)
	docker compose logs -f

logs-opensearch: ## Tail OpenSearch logs only
	docker compose logs -f opensearch

# =============================================================================
# HEALTH CHECKS
# =============================================================================
health: ## Check health of all services
	@echo ""
	@echo "  Checking services..."
	@echo ""
	@curl -s -o /dev/null -w "  OpenSearch:          %{http_code}\n" http://localhost:9200/_cluster/health || echo "  OpenSearch:          ✗ not reachable"
	@curl -s -o /dev/null -w "  OpenSearch Dashboards: %{http_code}\n" http://localhost:5601/api/status || echo "  Dashboards:          ✗ not reachable"
	@curl -s -o /dev/null -w "  FastAPI:             %{http_code}\n" http://localhost:8000/api/v1/health || echo "  FastAPI:             ✗ not running (start with 'make serve')"
	@echo ""

# =============================================================================
# PYTHON / APPLICATION
# =============================================================================
setup: ## Install all Python dependencies with uv
	@echo "$(GREEN)Installing dependencies...$(RESET)"
	uv sync --all-extras

setup-dev: ## Install dev dependencies only
	uv sync --extra dev

serve: ## Run FastAPI development server (with hot reload)
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod: ## Run FastAPI in production mode (no reload)
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

open-docs: ## Open FastAPI interactive docs in browser
	$(OPEN_CMD) http://localhost:8000/docs

open-dashboards: ## Open OpenSearch Dashboards in browser
	$(OPEN_CMD) http://localhost:5601

# =============================================================================
# DATABASE (Alembic migrations)
# =============================================================================
migrate: ## Run database migrations (alembic upgrade head)
	uv run alembic upgrade head

migrate-down: ## Rollback last migration
	uv run alembic downgrade -1

migrate-create: ## Create a new migration (usage: make migrate-create name="add_users_table")
	uv run alembic revision --autogenerate -m "$(name)"

migrate-history: ## Show migration history
	uv run alembic history --verbose

seed: ## Seed database with sample papers for development
	uv run python scripts/seed_db.py

# =============================================================================
# CODE QUALITY
# =============================================================================
format: ## Format code with ruff
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

lint: ## Lint code with ruff (no auto-fix)
	uv run ruff check src tests scripts

type-check: ## Run mypy type checking
	uv run mypy src

check: format lint type-check ## Run all code quality checks (format + lint + types)

# =============================================================================
# TESTING
# =============================================================================
test: ## Run all tests
	uv run pytest

test-unit: ## Run unit tests only (no external services needed)
	uv run pytest -m unit

test-integration: ## Run integration tests (requires running services)
	uv run pytest -m integration

test-cov: ## Run tests with coverage report
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(RESET)"

test-watch: ## Run tests in watch mode (re-runs on file change)
	uv run pytest --tb=short -q -f

# =============================================================================
# CLEAN UP
# =============================================================================
clean: ## Remove all generated files (cache, coverage, __pycache__)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleaned.$(RESET)"

clean-docker: ## Remove Docker volumes (WARNING: deletes all OpenSearch data)
	docker compose down --volumes
	@echo "$(GREEN)Docker volumes removed.$(RESET)"

clean-all: clean clean-docker ## Remove everything (code cache + Docker volumes)
