"""Seed data tests: every demo doc must satisfy the ingest schema."""

import importlib.util
from pathlib import Path

from app.schemas import DocumentIn


def load_seed_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "seed.py"
    spec = importlib.util.spec_from_file_location("seed", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_docs_are_valid_documents():
    seed = load_seed_module()
    assert len(seed.DOCS) >= 3
    seen: set[str] = set()
    for raw in seed.DOCS:
        doc = DocumentIn(**raw)
        assert doc.doc_id and doc.doc_id not in seen
        seen.add(doc.doc_id)
        assert doc.title
        assert len(doc.text) > 50


def test_seed_sample_query_is_nonblank():
    seed = load_seed_module()
    assert seed.SAMPLE_QUERY.strip()
