"""Health and root endpoint tests: the API boots and reports ok."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_names_service():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "raggy"
    assert "environment" in body


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
