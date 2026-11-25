"""The chatbot's toolbox works today, independent of any LLM."""
import pytest

from conftest import SAMPLES, VALID, upload


def call(client, name, **arguments):
    return client.post("/v1/chat/tools/call", json={"name": name, "arguments": arguments})


def test_catalogue_is_valid_json_schema(client):
    tools = client.get("/v1/chat/tools").json["tools"]
    assert {t["name"] for t in tools} == {"list_datasets", "get_portfolio_summary", "get_segment_breakdown",
                                          "list_top_risk_customers", "get_customer", "predict_what_if"}
    for t in tools:
        assert t["description"] and t["input_schema"]["type"] == "object"


def test_tools_default_to_the_latest_dataset_and_agree_with_the_dashboard(client, dirty_dataset):
    summary = client.get(f"/v1/datasets/{dirty_dataset}/summary").json
    via_tool = call(client, "get_portfolio_summary").json["result"]          # no dataset_id given
    assert via_tool["kpis"] == summary["kpis"] and via_tool["risk_tiers"] == summary["risk_tiers"]
    seg = call(client, "get_segment_breakdown", by="Geography").json["result"]["segments"]
    expected = summary["segments"]["Geography"]
    assert [r["segment"] for r in seg] == [r["segment"] for r in expected]       # the same groups in the same order
    for got, want in zip(seg, expected):
        assert got.keys() == want.keys()
        for key, value in got.items():
            if isinstance(value, float):
                assert value == pytest.approx(want[key], rel=1e-9)                # same numbers, up to float rounding
            else:
                assert value == want[key]


def test_top_risk_and_customer_tools(client, dirty_dataset):
    top = call(client, "list_top_risk_customers", limit=3, tier="High", geography="Germany").json["result"]
    assert len(top["customers"]) == 3 and all(c["Geography"] == "Germany" for c in top["customers"])
    cid = top["customers"][0]["CustomerId"]
    cust = call(client, "get_customer", customer_id=str(cid)).json["result"]
    assert cust["customer"]["churn_score"] == top["customers"][0]["churn_score"] and cust["explanation"]["drivers"]


def test_what_if_scores_and_explains(client):
    r = call(client, "predict_what_if", customer=VALID).json["result"]
    assert r["churn_score"] == 0.5517 and r["explanation"]["drivers"]
    worse = call(client, "predict_what_if", customer={**VALID, "NumOfProducts": 3}).json["result"]
    assert worse["churn_score"] != r["churn_score"]


@pytest.mark.parametrize("name,args,code,status", [
    ("nope", {}, "unknown_tool", 404),
    ("get_segment_breakdown", {"by": "ShoeSize"}, "tool_error", 400),
    ("predict_what_if", {"customer": {**VALID, "Age": 5}}, "tool_error", 400),
])
def test_bad_calls_return_clear_errors(client, dirty_dataset, name, args, code, status):
    r = call(client, name, **args)
    assert r.status_code == status and r.json["error"]["code"] == code


def test_tools_say_so_when_no_dataset_exists(client):
    r = call(client, "get_portfolio_summary")
    assert r.status_code == 400 and "No dataset" in r.json["error"]["message"]
    assert call(client, "list_datasets").json["result"]["datasets"] == []
    assert client.post("/v1/chat", json={}).status_code == 400      # chat needs a message (see test_chat.py)
