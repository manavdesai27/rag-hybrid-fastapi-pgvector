# Production-Grade RAG: Hybrid Search + Citations (Stripe Docs)

A reliable, production-minded Retrieval-Augmented Generation (RAG) system:
- Hybrid retrieval (keyword BM25 + dense embeddings)
- Grounded answers with explicit citations/snippets
- Redis caching, FastAPI backend, Streamlit UI
- Postgres + pgvector vector store
- Optional reranking and observability hooks
- Clean, interview-ready artifacts and scripts

This repo is intentionally simple to run and easy to explain in interviews.

## Quickstart

Prerequisites:
- Docker and Docker Compose installed
- An OpenAI API key

1) Configure environment
- Copy env template and set your API key:
  cp .env.example .env
  # edit .env and set OPENAI_API_KEY

2) Start the stack
- Run all services (Postgres+pgvector, Redis, API, UI):
  podman-compose -f infra/docker-compose.yml up --build -d

3) Ingest docs
- Index a small curated subset of Stripe API docs (configurable via .env):
  podman-compose -f infra/docker-compose.yml run --rm api python -m app.ingestion.ingest_stripe

4) Use the app
- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

## What you can say in interviews (plain language)

- “It searches the docs first, then answers using only those passages, with citations. If it can’t find the answer, it says it doesn’t know.”
- “Hybrid search blends keyword match (great for IDs/acronyms) with dense semantic search (great for paraphrases).”
- “We added caching to reduce latency and cost, and we log traces/metrics to monitor regression later.”

## Architecture

- Ingestion:
  - Fetch target pages from Stripe docs, clean HTML → text
  - Chunk by headings/paragraphs with overlapping windows
  - Store chunks + embeddings in Postgres (pgvector)
- Retrieval:
  - Hybrid: BM25 (keyword) + dense vector search
  - Optional cross-encoder reranker to refine the top candidates
- Generation:
  - LLM answers only from retrieved evidence, adds citations/snippets
  - Refusal policy when confidence is low or evidence is weak
- Caching/Observability:
  - Redis cache for repeated queries
  - Langfuse/OpenTelemetry wiring points (optional)

## Services

- api (FastAPI): /ask endpoint
- ui (Streamlit): interactive demo
- db (Postgres + pgvector)
- redis (for caching)

## Endpoints (FastAPI)

- POST /ask
  - Body: {"question": "..."}
  - Returns: {"answer": "...", "citations": [{"source": "...", "url": "...", "snippet": "..."}], "latency_ms": ...}

## Configuration (.env)

- OPENAI_API_KEY=sk-...
- OPENAI_MODEL=gpt-4o-mini (or gpt-4o, gpt-4o-mini, gpt-4.1-mini)
- OPENAI_EMBEDDING_MODEL=text-embedding-3-small
- DATABASE_URL=postgresql+psycopg2://rag_user:rag_pass@db:5432/rag_db
- REDIS_URL=redis://redis:6379/0
- DOCS_SEED_URLS=https://docs.stripe.com/api,https://docs.stripe.com/payments
- MAX_DOCS=30            # limit for quick ingest
- CHUNK_SIZE=800         # characters
- CHUNK_OVERLAP=150
- TOP_K=8                # retrieved passages
- RERANKER_ENABLED=false # default off to avoid model download
- CACHE_TTL_SECONDS=600

## Local development (without Docker)

- Python 3.11+
- pip install -r requirements.txt
- Run Postgres + pgvector and Redis locally (update .env DATABASE_URL/REDIS_URL accordingly)
- Uvicorn:
  uvicorn app.main:app --reload --port 8000
- Streamlit:
  streamlit run ui/app.py

## Roadmap (optional stretch)

- Evals: RAGAS harness + CI gate (fail PR if faithfulness drops)
- Hallucination detection: simple uncertainty classifier on retrieved coverage
- Query router: classify navigational vs. factual vs. keyword-heavy and adapt retrieval
- Advanced rerankers: ColBERT/ANCE (OSS path), or hosted rerank APIs

## License

MIT
