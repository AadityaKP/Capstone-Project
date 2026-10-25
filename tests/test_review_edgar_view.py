"""The EDGAR dataset view's two read-only endpoints.

They are additive: review_service is imported inside the handlers, so a missing
or broken EDGAR file must surface as a 503 on these routes only and never touch
the founder product's routes (/api/health here stands in for them).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import review_service
from backend.main import app

DEMO = ["DOCN", "MDB", "PCTY", "ZETA", "ZM"]

needs_panel = pytest.mark.skipif(
    not (review_service.EDGAR_CSV.is_file() and review_service.EDGAR_FACTS_CSV.is_file()),
    reason="data/edgar_ratios.csv and data/edgar_facts.csv are git-ignored local exports",
)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@needs_panel
def test_panel_is_the_five_demo_companies(client):
    body = client.get("/api/review/panel", params={"limit": 200}).json()
    assert body["tickers"] == DEMO
    assert {r["ticker"] for r in body["rows"]} <= set(DEMO)
    assert body["total"] >= len(body["rows"]) > 0


@needs_panel
def test_panel_filter_sort_and_paging(client):
    asc = client.get("/api/review/panel", params={"ticker": "PCTY", "order": "asc", "limit": 5}).json()
    desc = client.get("/api/review/panel", params={"ticker": "PCTY", "order": "desc", "limit": 5}).json()
    assert {r["ticker"] for r in asc["rows"]} == {"PCTY"}
    assert asc["rows"][0]["fiscal_period"] < desc["rows"][0]["fiscal_period"]
    page2 = client.get("/api/review/panel", params={"ticker": "PCTY", "order": "asc", "limit": 5, "offset": 5}).json()
    assert page2["rows"][0]["fiscal_period"] > asc["rows"][-1]["fiscal_period"]


@needs_panel
def test_meta_header_counts_the_subset(client):
    ds = client.get("/api/review/meta").json()["dataset"]
    assert ds["demo_subset"]["n_companies"] == 5
    assert ds["demo_subset"]["tickers"] == DEMO
    assert ds["n_companies"] >= 5


def test_panel_rejects_bad_query(client):
    assert client.get("/api/review/panel", params={"order": "sideways"}).status_code == 422
    assert client.get("/api/review/panel", params={"limit": 0}).status_code == 422


def test_missing_data_is_a_503_that_leaves_the_product_untouched(client, monkeypatch):
    monkeypatch.setattr(review_service, "_panel_cache", None)
    monkeypatch.setattr(review_service, "EDGAR_FACTS_CSV", review_service.ROOT / "data" / "does_not_exist.csv")
    monkeypatch.setattr(review_service, "EDGAR_CSV", review_service.ROOT / "data" / "does_not_exist.csv")
    assert client.get("/api/review/panel").status_code == 503
    assert client.get("/api/review/meta").status_code == 503
    assert client.get("/api/health").status_code == 200
