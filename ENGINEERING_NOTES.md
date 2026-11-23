# Raggy Engineering Notes

Running log of every Python backend, RAG pipeline, and engineering decision.
Each chunk appends a section here. At the end this becomes the full
end-to-end explanation.

---

## Chunk 1: Scaffold + health API + Docker

### What we built

- `app/main.py`: FastAPI app with `GET /` and `GET /health`.
- `app/config.py`: `Settings` class via `pydantic-settings`, loaded from
  environment or `.env`.
- `tests/test_health.py`: two tests proving the app boots and health is ok.
- `Dockerfile`: slim Python image that installs deps and runs Uvicorn.
- `.env.example`, `.gitignore`, `requirements.txt`, `README.md`.

### Python backend decisions

1. **FastAPI over Flask/Django.** FastAPI gives async support, automatic
   OpenAPI docs at `/docs`, and native Pydantic validation. For an API-only
   RAG backend there is no need for Django templates or ORM; Flask would
   need extra plugins for what FastAPI includes.
2. **One `Settings` object as the single source of truth.** Every tunable
   (model names, Qdrant URL, later chunk sizes and top-k) lives in
   `app/config.py` and loads from the environment. Reason: twelve-factor
   config. Same image runs in dev, Docker, and CI with different env files,
   no code edits. `extra="ignore"` keeps stray env vars from crashing boot.
3. **`get_settings()` factory instead of a bare global.** `main.py` calls it
   once now, but later chunks will inject it as a FastAPI dependency so
   tests can override config (e.g. point Qdrant at a fake) without touching
   real services.
4. **Thin routes, logic later in services.** `main.py` only defines
   endpoints. Chunk 4 onward puts ingest/retrieve/answer logic in
   `app/services/`, keeping routes small and testable. This is the service
   layer pattern: routes handle HTTP, services handle business logic.
5. **Health endpoint from day one.** `/health` is what Docker
   `HEALTHCHECK`, Compose `depends_on`, and future load balancers poll.
   Adding it in Chunk 1 means every later chunk is deploy-checkable.

### RAG pipeline decisions (forward-looking)

- Reserved `openai_embed_model = text-embedding-3-small`: cheap, fast, and
  good enough for domain FAQ retrieval. Easy to swap later via env.
- Reserved `qdrant_collection = raggy_docs`: one collection per domain keeps
  vectors isolated; multi-tenancy would add one collection per tenant.
- No retrieval code yet on purpose: Chunk 1 proves the deployable skeleton
  first, so later RAG bugs are never confused with packaging bugs.

### Engineering decisions

- **Docker first, not last.** The `Dockerfile` exists in Chunk 1 so every
  chunk is verified containerizable. Python version pinned to 3.11-slim:
  small image, widely supported wheels for scientific packages Qdrant and
  LangChain will pull in later.
- **Tests before features.** `test_health.py` uses FastAPI `TestClient`
  (no live server needed). This sets the rule: each chunk ships tests.
- **`.env` never committed.** Only `.env.example` is tracked, so the real
  OpenAI key stays out of git history.
- **System design note (scaling):** the API is stateless; all state will
  live in Qdrant. That means later we can run N API replicas behind a load
  balancer with no session stickiness, and health checks tell the balancer
  which replicas are alive.

### How to verify this chunk

- `pytest -q` passes (2 tests).
- `uvicorn app.main:app` then `GET /health` returns status ok.
- `docker build -t raggy .` succeeds.
