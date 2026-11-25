"""With PostgreSQL storage, ordinary reads are answered by SQL and never load the whole results frame."""
import pytest

pytest.importorskip("sqlalchemy")

from churn_api.storage.pg_stores import PostgresDatasetStore   # noqa: E402
from conftest import SAMPLES, make_app, upload                  # noqa: E402


@pytest.fixture
def pg_app(tmp_path, pg_url, bundle, monkeypatch):
    app = make_app(tmp_path, bundle, DATABASE_URL=pg_url)
    r = upload(app.test_client(), SAMPLES / "churn_sample_dirty.csv", "dirty")
    assert r.status_code == 201, r.get_json()

    def boom(*args, **kwargs):
        raise AssertionError("the whole results frame was loaded")

    monkeypatch.setattr(PostgresDatasetStore, "results", boom)       # any whole-frame read now fails the test
    return app, r.get_json()["dataset"]["id"]


def test_customer_list_and_detail_are_answered_by_sql(pg_app):
    app, did = pg_app
    client = app.test_client()
    listing = client.get(f"/v1/datasets/{did}/customers?tier=High&sort_by=Age&order=asc&limit=5&offset=1")
    assert listing.status_code == 200 and listing.get_json()["total"] > 0
    first = client.get(f"/v1/datasets/{did}/customers?limit=1").get_json()["rows"][0]
    key = first.get("CustomerId") or first["row_id"]
    detail = client.get(f"/v1/datasets/{did}/customers/{key}")
    assert detail.status_code == 200 and "explanation" in detail.get_json()


def test_assistant_tools_are_answered_by_sql(pg_app):
    app, _ = pg_app
    client = app.test_client()
    def call(name, **arguments):
        return client.post("/v1/chat/tools/call", json={"name": name, "arguments": arguments})
    segments = call("get_segment_breakdown", by="Geography")
    assert segments.status_code == 200 and len(segments.get_json()["result"]["segments"]) == 3
    top = call("list_top_risk_customers", limit=3, tier="High")
    assert top.status_code == 200 and len(top.get_json()["result"]["customers"]) == 3
    cid = top.get_json()["result"]["customers"][0]["CustomerId"]
    assert call("get_customer", customer_id=str(cid)).status_code == 200


def test_hostile_input_is_just_text(pg_app):
    app, did = pg_app
    client = app.test_client()
    for query in ("sort_by=churn_score;DROP TABLE dataset_results", "search=%27%3B%20DROP%20TABLE%20datasets%3B--",
                  "tier=High%27%20OR%20%271%27%3D%271", "geography=%25", "search=%25", "search=_"):
        r = client.get(f"/v1/datasets/{did}/customers?{query}")
        assert r.status_code == 200, query
    assert client.get(f"/v1/datasets/{did}/customers?limit=1").get_json()["total"] > 0     # the table is still there


def test_unknown_dataset_is_a_404_not_a_500(pg_app):
    app, _ = pg_app
    client = app.test_client()
    assert client.get("/v1/datasets/" + "0" * 32 + "/customers").status_code == 404
    assert client.get("/v1/datasets/" + "0" * 32 + "/customers/1").status_code == 404