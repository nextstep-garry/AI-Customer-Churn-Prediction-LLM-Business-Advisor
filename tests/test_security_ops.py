import pytest

from conftest import ROOT, SAMPLES, VALID, make_app, upload


def test_health_is_open_and_v1_requires_the_key_when_configured(tmp_path, bundle):
    c = make_app(tmp_path, bundle, API_KEY="s3cret").test_client()
    assert c.get("/health").status_code == 200 and c.get("/ready").status_code == 200
    assert c.post("/v1/predict", json=VALID).status_code == 401
    assert c.post("/v1/predict", json=VALID, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/v1/predict", json=VALID, headers={"X-API-Key": "s3cret"}).status_code == 200
    assert c.get("/v1/model").json["error"]["code"] == "unauthorized"


def test_rate_limit_returns_429_with_retry_after(tmp_path, bundle):
    c = make_app(tmp_path, bundle, RATE_LIMIT_PER_MIN=3).test_client()
    codes = [c.get("/v1/model").status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    r = c.get("/v1/model")
    assert int(r.headers["Retry-After"]) >= 1 and r.json["error"]["code"] == "rate_limited"
    assert c.get("/health").status_code == 200  # ops routes are never limited


def test_request_id_is_echoed_or_replaced(client):
    assert client.get("/health", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"] == "abc-123"
    bad = client.get("/health", headers={"X-Request-ID": "x y<z>"}).headers["X-Request-ID"]
    assert bad != "x y<z>" and len(bad) == 16
    assert client.get("/nope").json["request_id"]


@pytest.mark.parametrize("method,path,status,code", [("get", "/nope", 404, "not_found"),
                                                     ("get", "/v1/predict", 405, "method_not_allowed")])
def test_errors_are_json_not_html(client, method, path, status, code):
    r = getattr(client, method)(path)
    assert r.status_code == status and r.is_json and r.json["error"]["code"] == code


def test_unexpected_errors_do_not_leak_internals(app, client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret-internal-detail /srv/path")
    monkeypatch.setattr(app.extensions["churn"].pipeline, "predict_records", boom)
    r = client.post("/v1/predict", json=VALID)
    assert r.status_code == 500 and r.json["error"]["code"] == "internal_error"
    assert "secret" not in r.get_data(as_text=True) and r.json["request_id"]


def test_api_stays_up_when_the_model_cannot_load(tmp_path):
    empty = tmp_path / "no_model"
    empty.mkdir()
    c = make_app(tmp_path / "d", None, MODEL_DIR=empty).test_client()
    assert c.get("/health").status_code == 200
    ready = c.get("/ready")
    assert ready.status_code == 503 and "model_manifest.json" in ready.json["detail"]
    assert c.post("/v1/predict", json=VALID).json["error"]["code"] == "model_unavailable"
    assert upload(c, SAMPLES / "churn_sample_unlabeled.csv").status_code == 503


def test_predictions_are_logged_to_history_and_can_be_cleared(client):
    client.post("/v1/predict", json=VALID)
    client.post("/v1/predict", json={**VALID, "Age": 60})
    hist = client.get("/v1/history").json["predictions"]
    assert [h["input"]["Age"] for h in hist] == [60, 42]          # newest first
    assert {"score", "risk_level", "model_version", "request_id"} <= set(hist[0])
    client.post("/v1/predict", json={**VALID, "Age": 500})          # rejected input is not logged
    assert len(client.get("/v1/history").json["predictions"]) == 2
    assert client.delete("/v1/history").json["deleted"] == 2 and client.get("/v1/history").json["predictions"] == []


def test_model_and_schema_endpoints(client):
    m = client.get("/v1/model").json
    assert m["score_type"] == "raw" and m["threshold"] == 0.5 and len(m["feature_names"]) == 13
    assert m["installed"]["scikit-learn"] == m["trained_with"]["scikit-learn"]
    fields = {f["name"]: f for f in client.get("/v1/schema").json["fields"]}
    assert fields["Age"]["min"] == 18 and fields["Geography"]["choices"] == ["France", "Germany", "Spain"]
    assert fields["EstimatedSalary"]["exclusive_min"] is True
