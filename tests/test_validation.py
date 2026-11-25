"""The 18 invalid payloads the audit found returning HTTP 200 (report 3.2) must now be rejected."""
import pytest

from conftest import VALID

INVALID_FIELD_CASES = [
    ("CreditScore", -500), ("CreditScore", 99999), ("CreditScore", None),
    ("Age", -5), ("Age", 0), ("Age", 400),
    ("Tenure", -10), ("Tenure", 500),
    ("Balance", -1e9), ("Balance", 1e15),
    ("NumOfProducts", 0), ("NumOfProducts", 17), ("NumOfProducts", 2.7),
    ("HasCrCard", 7), ("HasCrCard", "banana"),
    ("IsActiveMember", -3), ("IsActiveMember", None),
    ("EstimatedSalary", 1e12), ("EstimatedSalary", 0), ("EstimatedSalary", -1),
    ("Geography", "Atlantis"), ("Gender", 123), ("Gender", "Robot"),
]


@pytest.mark.parametrize("field,value", INVALID_FIELD_CASES)
def test_invalid_value_is_rejected_with_field_level_error(client, field, value):
    r = client.post("/v1/predict", json={**VALID, field: value})
    assert r.status_code == 400
    assert r.json["error"]["code"] == "validation_error"
    assert [d["field"] for d in r.json["error"]["details"]] == [field]
    assert "request_id" in r.json


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_json_nan_and_infinity_literals_are_rejected(client, literal):
    body = ('{"CreditScore": 619, "Geography": "France", "Gender": "Female", "Age": %s, "Tenure": 2, '
            '"Balance": %s, "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 1000}'
            % (literal, literal))
    r = client.post("/v1/predict", data=body, content_type="application/json")
    assert r.status_code == 400 and r.json["error"]["code"] == "validation_error"


def test_missing_field_and_unknown_field_are_rejected(client):
    partial = {k: v for k, v in VALID.items() if k != "Tenure"}
    r = client.post("/v1/predict", json=partial)
    assert r.status_code == 400 and r.json["error"]["details"][0]["field"] == "Tenure"
    r = client.post("/v1/predict", json={**VALID, "Colour": "blue"})
    assert r.status_code == 400 and r.json["error"]["details"][0]["field"] == "Colour"


@pytest.mark.parametrize("body,ctype", [("not json", "application/json"), ("[1,2]", "application/json"),
                                        ("", "application/json")])
def test_non_object_bodies_are_400(client, body, ctype):
    assert client.post("/v1/predict", data=body, content_type=ctype).status_code == 400


def test_boundaries_are_accepted(client):
    for patch in ({"CreditScore": 350}, {"CreditScore": 850}, {"Age": 18}, {"Age": 92}, {"Tenure": 0},
                  {"Tenure": 10}, {"NumOfProducts": 4}, {"Balance": 0}, {"EstimatedSalary": 0.01}):
        assert client.post("/v1/predict", json={**VALID, **patch}).status_code == 200, patch


def test_categories_are_case_insensitive(client):
    r = client.post("/v1/predict", json={**VALID, "Geography": " germany ", "Gender": "MALE"})
    assert r.status_code == 200


def test_out_of_training_range_gives_warning_not_error(client):
    r = client.post("/v1/predict", json={**VALID, "Balance": 900_000})
    assert r.status_code == 200
    assert any("Balance" in w for w in r.json["warnings"])
    assert client.post("/v1/predict", json=VALID).json["warnings"] == []


def test_batch_endpoint_validates_every_record_and_caps_size(client, app):
    ok = client.post("/v1/predict/batch", json={"records": [VALID, {**VALID, "Age": 60}]})
    assert ok.status_code == 200 and ok.json["count"] == 2
    bad = client.post("/v1/predict/batch", json={"records": [VALID, {**VALID, "Age": 5}]})
    assert bad.status_code == 400 and bad.json["error"]["details"][0]["field"] == "records.1.Age"
    cap = app.extensions["churn"].cfg.MAX_BATCH_RECORDS
    big = client.post("/v1/predict/batch", json={"records": [VALID] * (cap + 1)})
    assert big.status_code == 400 and big.json["error"]["code"] == "too_many_records"
