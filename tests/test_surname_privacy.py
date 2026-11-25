"""Surnames are stored but hidden unless a request asks for them (and the operator allows it)."""
import pandas as pd
import pytest

from churn_api.config import Config
from churn_api.services import privacy
from conftest import SAMPLES, make_app, upload


def a_surname(app, dataset_id) -> str:
    return str(app.extensions["churn"].store.results(dataset_id)["Surname"].dropna().iloc[0])


def header(response) -> list[str]:
    return response.get_data(as_text=True).splitlines()[0].split(",")


def test_list_and_detail_hide_surnames_by_default(client, dirty_dataset):
    rows = client.get(f"/v1/datasets/{dirty_dataset}/customers?limit=50").get_json()["rows"]
    assert rows and all("Surname" not in r for r in rows)
    detail = client.get(f"/v1/datasets/{dirty_dataset}/customers/{rows[0]['CustomerId']}").get_json()
    assert "Surname" not in detail["customer"] and "explanation" in detail


def test_a_request_can_ask_for_surnames(client, dirty_dataset):
    rows = client.get(f"/v1/datasets/{dirty_dataset}/customers?limit=5&include_surname=true").get_json()["rows"]
    assert rows and all("Surname" in r for r in rows)
    detail = client.get(f"/v1/datasets/{dirty_dataset}/customers/{rows[0]['CustomerId']}?include_surname=true").get_json()
    assert detail["customer"]["Surname"] == rows[0]["Surname"]


@pytest.mark.parametrize("kind", ["predictions", "rejected"])
def test_downloads_hide_surnames_unless_asked(client, dirty_dataset, kind):
    hidden = client.get(f"/v1/datasets/{dirty_dataset}/download?kind={kind}")
    assert hidden.status_code == 200 and "Surname" not in header(hidden)
    shown = client.get(f"/v1/datasets/{dirty_dataset}/download?kind={kind}&include_surname=true")
    assert "Surname" in header(shown)


def test_assistant_tools_never_return_surnames(client, dirty_dataset):
    def call(name, **arguments):
        return client.post("/v1/chat/tools/call", json={"name": name, "arguments": arguments}).get_json()["result"]
    top = call("list_top_risk_customers", limit=5)
    assert top["customers"] and all("Surname" not in c for c in top["customers"])
    assert "Surname" not in call("get_customer", customer_id=str(top["customers"][0]["CustomerId"]))["customer"]


def test_the_stored_summary_does_not_leak_surnames(client, dirty_dataset):
    text = client.get(f"/v1/datasets/{dirty_dataset}/summary").get_data(as_text=True)
    names = {str(s) for s in pd.read_csv(SAMPLES / "churn_sample_dirty.csv")["Surname"].dropna() if len(str(s)) >= 5}
    assert sorted(n for n in names if f'"{n}"' in text) == []


def test_a_hidden_surname_cannot_be_discovered_through_search(app, client, dirty_dataset):
    name = a_surname(app, dirty_dataset)
    url = f"/v1/datasets/{dirty_dataset}/customers"
    assert client.get(url, query_string={"search": name}).get_json()["total"] == 0
    shown = client.get(url, query_string={"search": name, "include_surname": "true"}).get_json()
    assert shown["total"] >= 1 and all(name.lower() in r["Surname"].lower() for r in shown["rows"])


def test_searching_by_customer_id_still_works(client, dirty_dataset):
    url = f"/v1/datasets/{dirty_dataset}/customers"
    key = str(client.get(url + "?limit=1").get_json()["rows"][0]["CustomerId"])
    assert client.get(url, query_string={"search": key}).get_json()["total"] >= 1


def test_the_operator_can_forbid_surname_access(tmp_path, bundle):
    app = make_app(tmp_path, bundle, ALLOW_SURNAME_ACCESS=False)
    client = app.test_client()
    ds = upload(client, SAMPLES / "churn_sample_dirty.csv", "dirty").get_json()["dataset"]["id"]
    assert client.get(f"/v1/datasets/{ds}/customers?limit=1").status_code == 200
    for path in (f"/v1/datasets/{ds}/customers?include_surname=true",
                 f"/v1/datasets/{ds}/customers/1?include_surname=true",
                 f"/v1/datasets/{ds}/download?include_surname=true"):
        r = client.get(path)
        assert r.status_code == 403 and r.get_json()["error"]["code"] == "surname_not_allowed"


def test_strip_helpers_return_copies_without_surnames():
    df = pd.DataFrame({"row_id": [1], "Surname": ["Hill"], "Age": [42]})
    assert list(privacy.strip_frame(df).columns) == ["row_id", "Age"] and "Surname" in df.columns
    assert "Surname" not in privacy.strip_row(df.iloc[0]).index
    assert list(privacy.strip_frame(df[["Age"]]).columns) == ["Age"]            # nothing to remove


def test_the_config_switch(monkeypatch):
    monkeypatch.delenv("ALLOW_SURNAME_ACCESS", raising=False)
    assert Config.from_env().ALLOW_SURNAME_ACCESS is True
    monkeypatch.setenv("ALLOW_SURNAME_ACCESS", "false")
    assert Config.from_env().ALLOW_SURNAME_ACCESS is False
    


def test_strip_json_removes_the_key_at_any_depth():
    data = {"a": [{"Surname": "Hill", "Age": 42, "more": {"Surname": "X", "ok": 1}}], "Surname": "Y", "n": 3}
    assert privacy.strip_json(data) == {"a": [{"Age": 42, "more": {"ok": 1}}], "n": 3}
    assert "Surname" in data and data["a"][0]["Surname"] == "Hill"           # the original is untouched