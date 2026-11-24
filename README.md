<div align="center">
  <h1>Raggy</h1>
  <p><strong>Context-Aware RAG Backend with Grounded Citations</strong></p>
  <p>FastAPI + OpenAI + Qdrant, with HyDE, Step-Back Prompting, and re-ranking.</p>
  <p>
    <img src="https://img.shields.io/badge/Status-In%20Progress-orange.svg" alt="Status" />
    <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/Qdrant-DC244C?style=flat&logo=qdrant&logoColor=white" alt="Qdrant" />
    <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white" alt="Docker" />
    <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white" alt="LangChain" />
  </p>
</div>

---

## Overview

Raggy is a containerized retrieval-augmented generation (RAG) backend for
answering domain-specific questions. Documents are split into passages,
embedded with OpenAI models, and stored in Qdrant for dense vector search.
At query time the pipeline retrieves candidate passages, re-ranks them for
relevance, and synthesizes an answer where every factual claim points back
to a cited source passage.

Advanced query handling (HyDE hypothetical-document expansion and Step-Back
abstraction) kicks in through an intelligent router, so simple factual
questions stay cheap while complex prompts get the full pipeline.

---

## Architecture Highlights

### 1. Two-Stage Retrieval (Dense Search + Re-rank)
Qdrant returns a wide net of top-k candidates by cosine similarity; a
re-ranking layer then scores each `(query, passage)` pair, drops low-score
noise, and forwards only the top-n passages to the LLM. Recall first,
precision second.

### 2. Grounded Answers with Citations
The answer model must cite the passages it used (`[1]`, `[2]`, ...), and
the API returns the cited passage text alongside the answer so clients can
render and verify sources.

### 3. Intelligent Query Routing
A lightweight classifier (Pydantic structured outputs) labels each query as
factual, vague, or complex, and routes it to plain retrieval, HyDE
expansion, or Step-Back retrieval. Simple questions skip the expensive
paths entirely.

### 4. HyDE for Vocabulary Mismatch
For short or vague queries, a hypothetical answer is generated first and
its embedding (which looks like document language, not question language)
drives retrieval.

### 5. Step-Back Prompting for Complex Queries
For multi-hop questions, a broader step-back question is generated,
retrieved in parallel with the original, and both result sets are merged
before re-ranking.

### 6. Stateless API Design
All state lives in Qdrant; the API holds no sessions. Any number of
replicas can serve behind a load balancer, with `/health` as the
readiness signal.

---

## Technology Stack

* **API**: Python 3.11, FastAPI, Uvicorn, Pydantic v2
* **RAG**: LangChain text splitters, OpenAI embeddings (`text-embedding-3-small`) and chat (`gpt-4o-mini`)
* **Vector Search**: Qdrant (self-hosted via Docker)
* **Packaging**: Docker, Docker Compose
* **Testing**: Pytest, HTTPX

---

## Project Structure

```text
raggy/
├── app/
│   ├── main.py          # FastAPI entrypoint and routes
│   ├── config.py        # Environment-driven settings (pydantic-settings)
│   ├── schemas.py       # Pydantic document and search models
│   ├── dependencies.py  # FastAPI service wiring
│   ├── routers/
│   │   ├── documents.py # Ingest + stats endpoints
│   │   └── query.py     # Cited Q&A endpoint
│   └── services/
│       ├── chunking.py  # LangChain-based text splitting service
│       ├── embeddings.py # OpenAI embedding service with batching
│       ├── vector_store.py # Qdrant storage wrapper
│       ├── ingest.py    # Split-embed-store orchestration
│       ├── retriever.py # Query embedding + top-k search
│       ├── chat.py      # OpenAI chat + structured outputs
│       ├── router.py    # Query classification and routing
│       ├── hyde.py      # Hypothetical-answer expansion
│       ├── stepback.py  # Broad-question expansion + merge
│       ├── rerank.py    # LLM relevance filter before synthesis
│       └── answer.py    # Cited answer synthesis
├── tests/
│   ├── fakes.py         # Shared fakes (embeddings, chat, memory Qdrant)
│   ├── test_health.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_documents.py
│   ├── test_query.py
│   ├── test_router.py
│   ├── test_hyde.py
│   ├── test_stepback.py
│   ├── test_rerank.py
│   └── test_seed.py
├── scripts/
│   └── seed.py          # Ingest demo FAQ docs into a running API
├── Dockerfile
├── .dockerignore
├── docker-compose.yml   # API + Qdrant stack
├── requirements.txt     # Runtime deps (lean production image)
├── requirements-dev.txt # Runtime + test deps for local dev
└── .env.example         # Copy to .env and fill in
```

---

## Getting Started

### Prerequisites
* Python 3.11+
* Docker + Docker Compose (for Qdrant and the containerized run)

### Installation & Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/xyron24/raggy.git
   cd raggy
   ```

2. **Create a virtualenv and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-dev.txt
   ```

3. **Environment setup:**
   ```bash
   cp .env.example .env
   ```
   `OPENAI_API_KEY` is required for real embeddings; tests use a fake
   client and in-memory Qdrant, so they need no key.

4. **Run the API:**
   ```bash
   uvicorn app.main:app --reload
   ```
   Open http://localhost:8000/health (interactive docs at `/docs`).

### Run with Docker

```bash
docker build -t raggy .
docker run -p 8000:8000 raggy
```

### Run the full stack (API + Qdrant)

```bash
docker compose up --build
```

The API reads `QDRANT_URL=http://qdrant:6333` inside Compose; Qdrant data
persists in a named volume across restarts.

### Seed demo data

With the stack running and `OPENAI_API_KEY` set in `.env`:

```bash
python scripts/seed.py
```

This ingests five Acme Cloud FAQ documents, prints collection stats, and
shows a sample query to try. `RAGGY_URL` overrides the target API.

### Run tests

```bash
pytest -q
```

---

## API Reference

| Method | Path                | Description                              |
| ------ | ------------------- | ---------------------------------------- |
| GET    | `/`                 | Service name and environment             |
| GET    | `/health`           | Liveness probe, returns `{"status": "ok"}` |
| POST   | `/documents/ingest` | Split, embed, and store a document       |
| GET    | `/documents/stats`  | Point count and collection info          |
| POST   | `/query`            | Answer with cited sources, route, confidence |

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "faq1", "title": "FAQ", "text": "Refunds are issued within 30 days."}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?"}'
```

More endpoints are documented in the table above as they ship.
