# Raggy Engineering Notes

Running log of every Python backend, RAG pipeline, and engineering decision.
Each phase appends a section here. At the end this becomes the full
end-to-end explanation.

---

## Phase 1: Scaffold + health API + Docker

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
   once now, but later phases will inject it as a FastAPI dependency so
   tests can override config (e.g. point Qdrant at a fake) without touching
   real services.
4. **Thin routes, logic later in services.** `main.py` only defines
   endpoints. Phase 4 onward puts ingest/retrieve/answer logic in
   `app/services/`, keeping routes small and testable. This is the service
   layer pattern: routes handle HTTP, services handle business logic.
5. **Health endpoint from day one.** `/health` is what Docker
   `HEALTHCHECK`, Compose `depends_on`, and future load balancers poll.
   Adding it in Phase 1 means every later phase is deploy-checkable.

### RAG pipeline decisions (forward-looking)

- Reserved `openai_embed_model = text-embedding-3-small`: cheap, fast, and
  good enough for domain FAQ retrieval. Easy to swap later via env.
- Reserved `qdrant_collection = raggy_docs`: one collection per domain keeps
  vectors isolated; multi-tenancy would add one collection per tenant.
- No retrieval code yet on purpose: Phase 1 proves the deployable skeleton
  first, so later RAG bugs are never confused with packaging bugs.

### Engineering decisions

- **Docker first, not last.** The `Dockerfile` exists in Phase 1 so every
  phase is verified containerizable. Python version pinned to 3.11-slim:
  small image, widely supported wheels for scientific packages Qdrant and
  LangChain will pull in later.
- **Tests before features.** `test_health.py` uses FastAPI `TestClient`
  (no live server needed). This sets the rule: each phase ships tests.
- **`.env` never committed.** Only `.env.example` is tracked, so the real
  OpenAI key stays out of git history.
- **System design note (scaling):** the API is stateless; all state will
  live in Qdrant. That means later we can run N API replicas behind a load
  balancer with no session stickiness, and health checks tell the balancer
  which replicas are alive.

### How to verify this phase

- `pytest -q` passes (2 tests).
- `uvicorn app.main:app` then `GET /health` returns status ok.
- `docker build -t raggy .` succeeds.

---

## Phase 2: Document models + text chunking service

### What we built

- `app/schemas.py`: `DocumentIn` (ingest input) and `Chunk` (one passage).
- `app/services/chunking.py`: `ChunkingService` wrapping LangChain's
  `RecursiveCharacterTextSplitter`.
- `app/config.py`: `chunk_size` (500) and `chunk_overlap` (50) knobs.
- `tests/test_chunking.py`: 7 tests (splitting, overlap, metadata,
  generated ids, invalid sizes, empty input).
- Also in this phase: repo-wide rename of build steps from "Chunk N" to
  "Phase N" and a professional README rewrite, so the public repo reads
  like a real project. The word "chunk" now appears only as RAG domain
  vocabulary (text chunks), which is the correct term there.

### Python backend decisions

1. **Pydantic models as the API boundary.** `DocumentIn.text` has
   `min_length=1`, so empty submissions are rejected with a 422 before any
   logic runs. Validation lives in the schema, not in `if` statements
   scattered through services.
2. **Service tolerates what the schema rejects.** `split_text("")`
   returns `[]` instead of raising. Reason: the service is reusable logic
   (seed scripts, backfills) where blank input is possible; the schema is
   the strict HTTP contract. Strict at the boundary, lenient inside.
3. **Optional `doc_id` with server-side generation.** Callers that track
   ids pass them (enables idempotent re-ingest in Phase 4); callers that
   do not get a `uuid4` hex. One model serves both without a second
   endpoint.
4. **Deterministic chunk ids (`{doc_id}:{position}`).** Same document
   re-ingested produces the same ids, so the Qdrant upsert in Phase 4
   overwrites instead of duplicating. Random ids would leak duplicates on
   every retry.
5. **Constructor validation (`0 <= overlap < size`).** Overlap equal to
   size would loop or produce garbage in the splitter; failing fast in
   `__init__` turns a silent corruption bug into a loud error.
6. **`langchain-text-splitters` instead of full `langchain`.** The full
   package pulls a heavy dependency tree; the splitters package is the
   same LangChain code with a minimal footprint. Fewer deps, faster
   Docker builds, smaller images.

### RAG pipeline decisions

- **Recursive splitting over fixed windows.** The splitter tries
  paragraph, then line, then sentence, then word boundaries, so cuts land
  on natural breaks instead of mid-word. Fixed-size slicing is simpler
  but severs sentences and hurts embedding quality.
- **Character length, not token counting.** `length_function=len` keeps
  splitting fast with no tokenizer dependency. Token-accurate splitting
  (tiktoken) matters when filling exact context windows; retrieval
  passages only need roughly-right sizes.
- **500 chars with 50 overlap as defaults.** Small enough that each
  passage is topically focused (better cosine matches), large enough to
  hold a full fact. Overlap preserves sentences that straddle a cut so
  no fact is split across two half-contexts. Both are env-tunable.
- **Metadata propagation.** Title and custom metadata are copied onto
  every chunk now, so Phase 4 can store them as Qdrant payloads for
  filtering ("only search the pricing docs") and citation display later.

### Engineering decisions

- **Config knobs before the feature needs them.** `chunk_size` and
  `chunk_overlap` land in `Settings` in the same phase as the splitter,
  so tuning never requires a code edit.
- **Overlap test asserts behavior, not exact strings.** The test checks
  neighbors share words and no word is lost, rather than hardcoding
  splitter output. It survives library version bumps that tweak cut
  positions.
- **System design note (scaling):** chunking is CPU-bound and
  embarrassingly parallel (each document is independent). Later, ingest
  can fan out across workers or replicas with no shared state; the only
  coordination point is the Qdrant upsert.

### How to verify this phase

- `pytest -q` passes (9 tests: 2 health + 7 chunking).
