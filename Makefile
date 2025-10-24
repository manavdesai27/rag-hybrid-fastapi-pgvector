# Simple helpers to manage the RAG stack locally
COMPOSE=infra/docker-compose.yml

.PHONY: up ingest ingest-json evals health logs down clean env

env:
	@if [ ! -f .env ]; then cp .env.example .env && echo "Copied .env.example to .env"; else echo ".env already exists"; fi
	@echo "Ensure OPENAI_API_KEY is set in .env"

up:
	docker compose -f $(COMPOSE) up --build -d

health:
	@echo "Waiting for API health at http://localhost:8000/health ..."
	@for i in $$(seq 1 60); do \
		if curl -sSf http://localhost:8000/health >/dev/null; then \
			echo "API is healthy"; exit 0; \
		fi; \
		echo "Waiting... ($$i)"; sleep 2; \
	done; \
	echo "API did not become healthy in time"; exit 1

ingest:
	docker compose -f $(COMPOSE) run --rm api python -m app.ingestion.ingest_stripe

ingest-json:
	@if [ -z "$(URL)" ]; then echo "Usage: make ingest-json URL=https://example.com/openapi.json"; exit 2; fi
	docker compose -f $(COMPOSE) run --rm api python -m app.ingestion.ingest_json_url --url "$(URL)"

evals:
	docker compose -f $(COMPOSE) run --rm api python -m app.evals.run_ragas

logs:
	docker compose -f $(COMPOSE) logs -f api

down:
	docker compose -f $(COMPOSE) down -v

clean: down
	@echo "Cleaned containers and volumes"
