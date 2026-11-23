# Raggy: Context-Aware RAG Pipeline - Implementation Plan

## Goal

Build a containerized FastAPI RAG backend that answers domain-specific queries
with grounded citations, using OpenAI embeddings, Qdrant vector search,
LangChain, HyDE, Step-Back Prompting, Pydantic structured outputs, and a
re-ranking layer.

This file is the build contract. We implement it phase by phase, committing
and pushing after each phase.

## Resume mapping (what each line needs)

1. "Containerized FastAPI RAG backend using OpenAI embeddings and Qdrant
   vector search to answer domain-specific queries with grounded citations."
   Needs: FastAPI, Docker, OpenAI embeddings, Qdrant, ingest API, query API,
   citation formatting.
2. "Implemented HyDE and Step-Back Prompting alongside Pydantic structured
   outputs for intelligent query routing across complex search prompts."
   Needs: query router, Pydantic response models, HyDE expansion,
   Step-Back question generation.
3. "Added a re-ranking layer to filter retrieval noise and prioritize
   top-relevance context chunks before LLM synthesis."
   Needs: re-ranker module, score filtering, top-k selection before answer
   generation.

## Tech stack (pinned in Phase 1)

- Python 3.11+ with FastAPI and Uvicorn
- Pydantic v2 + pydantic-settings for config and structured outputs
- LangChain for text splitting and prompt templates
- OpenAI API for embeddings (`text-embedding-3-small`) and chat
  (`gpt-4o-mini` or similar)
- Qdrant for vector search (local Docker instance, no cloud needed)
- Docker + Docker Compose for containerized dev and demo
- Pytest + HTTPX for API tests

## Phases (10 total, each is one commit)

### Phase 1: Project scaffold + health API + Docker (shipped)

- Folder layout: `app/`, `tests/`
- FastAPI app with `GET /` and `GET /health`
- Central config via `pydantic-settings` (env-driven, `.env.example`)
- `Dockerfile` for the API
- `.gitignore`, `requirements.txt`, `README.md`
- First test: health endpoint returns ok
- Learning focus: FastAPI app factory, Pydantic settings, Docker basics

### Phase 2: Document models + text chunking service (shipped)

- Pydantic schemas: `DocumentIn`, `Chunk`
- `ChunkingService` using LangChain `RecursiveCharacterTextSplitter`
- Unit tests for chunker (overlap, empty text, long text)
- Learning focus: why chunking matters, chunk size vs overlap tradeoff

### Phase 3: Embeddings + Qdrant vector store wiring (shipped)

- `EmbeddingService` (OpenAI embeddings, batched, retry on rate limit)
- `VectorStore` wrapper around Qdrant client (create collection, upsert,
  search)
- Docker Compose: API + Qdrant services together
- Tests with fakes/mocks (no real API key needed for CI)
- Learning focus: embeddings, cosine similarity, vector DB collections

### Phase 4: Document ingest API (in progress)

- `POST /documents/ingest`: text in, chunks + embeddings + Qdrant upsert
- `GET /documents/count` or collection stats endpoint
- Request validation, error handling, idempotent doc ids
- Learning focus: FastAPI dependency injection, service layer pattern

### Phase 5: Basic RAG query with grounded citations

- `RetrieverService`: embed query, top-k Qdrant search
- `AnswerService`: stuff context into prompt, call chat model, return
  answer plus `[1]`, `[2]` style citations mapped to chunk ids
- `POST /query`: request/response schemas, returns answer + sources
- Learning focus: RAG loop, grounded generation, citation formatting

### Phase 6: Pydantic structured outputs + intelligent query router

- Structured `RAGAnswer` model (answer, citations, confidence)
- `QueryRouter`: classify query as FACTUAL / COMPLEX / COMPARATIVE, route
  to plain retrieval vs advanced pipeline
- Router uses cheap model call with structured output parsing
- Tests for router decisions on sample queries
- Learning focus: structured outputs, routing pattern, cost control

### Phase 7: HyDE (Hypothetical Document Embeddings)

- `HyDEService`: generate hypothetical answer, embed it, retrieve with it
- Feature flag + router integration (use HyDE for vague/short queries)
- Before/after retrieval comparison in tests
- Learning focus: why HyDE helps vocabulary mismatch

### Phase 8: Step-Back Prompting

- `StepBackService`: generate broader step-back question, retrieve for
  both original + step-back, merge results
- Router integration (use for complex/multi-hop queries)
- Tests for step-back question generation and merged retrieval
- Learning focus: abstraction level mismatch in retrieval

### Phase 9: Re-ranking layer

- `RerankerService`: score (query, chunk) pairs, filter low scores,
  keep top-n before LLM synthesis
- Plug in as final filter in query pipeline (works with plain, HyDE,
  and Step-Back paths)
- Config knobs: `top_k_retrieve`, `top_n_rerank`, `min_score`
- Tests: noisy chunks get filtered, relevant chunks survive
- Learning focus: precision vs recall, two-stage retrieval

### Phase 10: Full-stack polish + docs + demo data

- `docker compose up` runs API + Qdrant end to end
- Seed script with sample domain docs (e.g. company FAQ)
- Full test suite green, README with curl examples
- `ENGINEERING_NOTES.md`: every Python, RAG, and engineering decision
  explained for learning (accumulated across all phases)
- Learning focus: Docker Compose networking, demo readiness

## Engineering rules for all phases

- One phase = one commit = one push. No mixing phases in a commit.
- Every phase adds or updates tests. Nothing merges red.
- Services depend on interfaces/protocols so tests can use fakes
  (no real OpenAI/Qdrant needed for unit tests).
- Config always comes from environment via `pydantic-settings`.
- Each phase updates `README.md` (what works now) and appends to
  `ENGINEERING_NOTES.md` (what we learned and why).

## Definition of done (final)

- `docker compose up` gives a working API + Qdrant.
- `POST /documents/ingest` stores docs; `POST /query` returns a cited answer.
- Router, HyDE, Step-Back, and re-ranker are all live behind config flags.
- `pytest` passes. README has copy-paste curl demos.
- `ENGINEERING_NOTES.md` explains all Python backend, RAG, and
  engineering decisions end to end.
