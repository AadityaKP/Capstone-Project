import os

from fastapi.testclient import TestClient

from backend.database import initialize_database
from backend.main import app


def test_health_and_scenario_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    initialize_database()

    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["database"] == "sqlite"

        created = client.post(
            "/api/scenarios",
            json={"name": "API test", "config": {"initial_mrr": 75000}},
        )
        assert created.status_code == 201
        assert created.json()["config"]["initial_mrr"] == 75000

        scenarios = client.get("/api/scenarios")
        assert scenarios.status_code == 200
        assert scenarios.json()[0]["name"] == "API test"


def test_rejects_unknown_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    initialize_database()

    with TestClient(app) as client:
        response = client.post(
            "/api/runs",
            json={"policy": "not-a-policy", "episodes": 1},
        )

    assert response.status_code == 422
