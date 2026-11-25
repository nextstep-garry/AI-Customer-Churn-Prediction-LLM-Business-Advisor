import json

import pandas as pd

from conftest import SAMPLES, VALID, make_app, upload


def test_dirty_upload_runs_the_whole_pipeline(client):
    r = upload(client, SAMPLES / "churn_sample_dirty.csv")
    assert r.status_code == 201
    rows = r.json["dataset"]["rows"]
    # Injected into the synthetic file by sample_data/generate_sample_data.py
    assert rows == {"uploaded": 2015, "scored": 1968, "rejected": 32, "repaired": 0, "duplicates_removed": 15}
    assert [s["stage"] for s in r.json["stages"]][:2] == ["read", "dedupe"]
    assert r.json["stages"][-1]["stage"] == "summarise"
    assert r.json["dataset"]["has_target"] is True


def test_summary_is_complete_and_strict_json(client, dirty_dataset):
    raw = client.get(f"/v1/datasets/{dirty_dataset}/summary").data.decode()
    s = json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(AssertionError(f"non-JSON constant {c}")))
    assert s["kpis"]["customers_scored"] == 1968 and s["kpis"]["rows_rejected"] == 32
    assert sum(t["customers"] for t in s["risk_tiers"]) == 1968
    assert set(s["segments"]) >= {"Geography", "AgeBand", "NumOfProducts"}
    assert 0.5 < s["evaluation"]["roc_auc"] <= 1 and "note" in s["evaluation"]
    assert s["drivers"] and s["quality"]["drift"] and s["quality"]["rejected_examples"]
    assert sum(s["score_histogram"]["counts"]) == 1968


def test_repair_policy_scores_everything(client):
    r = upload(client, SAMPLES / "churn_sample_dirty.csv", policy="repair")
    assert r.json["dataset"]["rows"]["scored"] == 2000 and r.json["dataset"]["rows"]["repaired"] == 32


def test_unlabeled_upload_has_no_evaluation(client):
    r = upload(client, SAMPLES / "churn_sample_unlabeled.csv")
    ds = r.json["dataset"]
    assert ds["has_target"] is False and ds["rows"]["scored"] == 500
    assert client.get(f"/v1/datasets/{ds['id']}/summary").json["evaluation"] is None


def test_upload_error_handling(client, tmp_path, bundle):
    assert client.post("/v1/datasets", data={}, content_type="multipart/form-data").json["error"]["code"] == "missing_file"
    assert upload(client, b"a,b\n1,2\n", filename="x.xlsx").status_code == 415
    assert upload(client, b"", filename="x.csv").json["error"]["code"] == "invalid_dataset"
    assert upload(client, b"a,b\n1,2\n").json["error"]["code"] == "invalid_dataset"          # missing columns
    assert upload(client, SAMPLES / "churn_sample_unlabeled.csv", policy="nope").status_code == 400
    all_bad = pd.DataFrame([{**VALID, "Age": 500}] * 3).to_csv(index=False).encode()
    r = upload(client, all_bad)
    assert r.status_code == 400 and "None of the 3 rows passed validation" in r.json["error"]["message"]
    small = make_app(tmp_path / "small", bundle, MAX_ROWS=10).test_client()
    assert "more than 10 rows" in upload(small, SAMPLES / "churn_sample_unlabeled.csv").json["error"]["message"]
    tiny = make_app(tmp_path / "tiny", bundle, MAX_UPLOAD_MB=0).test_client()
    assert upload(tiny, SAMPLES / "churn_sample_unlabeled.csv").status_code == 413


def test_semicolon_and_latin1_files_are_read(client):
    df = pd.DataFrame([{**VALID, "Surname": "Müller"}] * 3)
    r = upload(client, df.to_csv(index=False, sep=";").encode("latin-1"))
    assert r.status_code == 201 and r.json["dataset"]["rows"]["scored"] == 1  # 3 identical rows -> 1 after dedupe


def test_customers_endpoint_filters_sorts_and_pages(client, dirty_dataset):
    base = f"/v1/datasets/{dirty_dataset}/customers"
    high = client.get(base + "?tier=High&limit=500").json
    assert high["total"] == 351 and all(r["risk_level"] == "High" for r in high["rows"])
    scores = [r["churn_score"] for r in high["rows"]]
    assert scores == sorted(scores, reverse=True)
    page2 = client.get(base + "?tier=High&limit=10&offset=10").json["rows"]
    assert [r["row_id"] for r in page2] == [r["row_id"] for r in high["rows"][10:20]]
    de = client.get(base + "?geography=germany&limit=500").json
    assert de["total"] > 0 and {r["Geography"] for r in de["rows"]} == {"Germany"}
    assert client.get(base + "?limit=999999").json["rows"].__len__() <= 500


def test_customer_lookup_by_id_and_row_id(client, dirty_dataset):
    top = client.get(f"/v1/datasets/{dirty_dataset}/customers?limit=1").json["rows"][0]
    for key in (top["CustomerId"], top["row_id"]):
        r = client.get(f"/v1/datasets/{dirty_dataset}/customers/{key}")
        assert r.status_code == 200 and r.json["customer"]["churn_score"] == top["churn_score"]
        assert len(r.json["explanation"]["drivers"]) == 8
    assert client.get(f"/v1/datasets/{dirty_dataset}/customers/999999999").status_code == 404


def test_customer_explanation_matches_stored_score(client, bundle, dirty_dataset):
    """Re-scoring a stored row for the explanation must reproduce the stored score exactly."""
    import numpy as np
    top = client.get(f"/v1/datasets/{dirty_dataset}/customers?limit=1").json["rows"][0]
    margin = client.get(f"/v1/datasets/{dirty_dataset}/customers/{top['row_id']}").json["explanation"]["margin"]
    assert 1 / (1 + np.exp(-margin)) == np.float64(top["churn_score"]).round(4) or abs(1 / (1 + np.exp(-margin)) - top["churn_score"]) < 1e-5


def test_download_is_csv_and_neutralises_formula_injection(client):
    evil = pd.DataFrame([{**VALID, "Surname": "=HYPERLINK(\"http://evil\")"}, {**VALID, "Age": 50, "Surname": "-2+3"}])
    ds = upload(client, evil.to_csv(index=False).encode(), name="../../etc/passwd").json["dataset"]["id"]
    r = client.get(f"/v1/datasets/{ds}/download?include_surname=true")
    assert r.status_code == 200 and r.mimetype == "text/csv"
    assert "etc_passwd_predictions.csv" in r.headers["Content-Disposition"] and "/" not in r.headers["Content-Disposition"].split("filename=")[1]
    out = pd.read_csv(pd.io.common.BytesIO(r.data))
    assert out["Surname"].str.startswith("'").all()
    assert client.get(f"/v1/datasets/{ds}/download?kind=bogus").status_code == 400


def test_ids_cannot_escape_the_data_directory(client):
    for bad in ("..", "..%2F..%2Fetc", "not-a-hex-id", "0" * 31):
        assert client.get(f"/v1/datasets/{bad}").status_code == 404


def test_rescore_and_delete(client, dirty_dataset):
    before = client.get(f"/v1/datasets/{dirty_dataset}").json
    r = client.post(f"/v1/datasets/{dirty_dataset}/rescore")
    assert r.status_code == 200 and r.json["dataset"]["rows"] == before["rows"]
    assert r.json["dataset"]["updated_at"] >= before["updated_at"]
    assert client.delete(f"/v1/datasets/{dirty_dataset}").status_code == 204
    assert client.get(f"/v1/datasets/{dirty_dataset}/summary").status_code == 404
    assert client.get("/v1/datasets").json["datasets"] == []


def test_datasets_are_listed_newest_first(client):
    a = upload(client, SAMPLES / "churn_sample_unlabeled.csv", "first").json["dataset"]["id"]
    b = upload(client, SAMPLES / "churn_sample_unlabeled.csv", "second").json["dataset"]["id"]
    assert [d["id"] for d in client.get("/v1/datasets").json["datasets"]] == [b, a]
