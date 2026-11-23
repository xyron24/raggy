# Raggy - Context-Aware RAG Pipeline

Containerized FastAPI RAG backend using OpenAI embeddings and Qdrant vector
search, with HyDE, Step-Back Prompting, Pydantic structured outputs, and a
re-ranking layer.

Build plan: [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)
Learning log: [ENGINEERING_NOTES.md](./ENGINEERING_NOTES.md) (updated every chunk)

## Current status: Chunk 1 done

- `GET /` names the service
- `GET /health` returns `{"status": "ok", ...}`
- Config via environment (see `.env.example`)
- Docker image builds and runs

## Run locally

```bash
cd raggy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000/health

## Run with Docker

```bash
cd raggy
docker build -t raggy .
docker run -p 8000:8000 raggy
```

## Run tests

```bash
cd raggy
pytest -q
```

## Project layout

```text
raggy/
  app/
    main.py      # FastAPI entrypoint, routes
    config.py    # pydantic-settings config
  tests/
    test_health.py
  Dockerfile
  requirements.txt
  .env.example
  IMPLEMENTATION_PLAN.md
  ENGINEERING_NOTES.md
```
