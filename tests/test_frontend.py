"""Drive the real Streamlit app (entry point + navigation) against a real HTTP backend."""
import threading

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest            # noqa: E402
from werkzeug.serving import make_server            # noqa: E402

from conftest import ROOT, SAMPLES, make_app, use_backend  # noqa: E402

FE = ROOT / "frontend"
PAGES = ["views/portfolio.py", "views/single.py", "views/upload.py", "views/assistant.py", "views/model_info.py"]


@pytest.fixture(scope="module")
def backend(tmp_path_factory, bundle):
    app = make_app(tmp_path_factory.mktemp("fe_data"), bundle)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    use_backend(url)
    yield url
    server.shutdown()
    use_backend("http://127.0.0.1:9")


@pytest.fixture
def api(backend):
    import requests
    for d in requests.get(f"{backend}/v1/datasets").json()["datasets"]:
        requests.delete(f"{backend}/v1/datasets/{d['id']}")
    requests.delete(f"{backend}/v1/history")

    class Api:
        url = backend
        def upload(self, path, policy="reject", name="demo"):
            r = requests.post(f"{backend}/v1/datasets", files={"file": (path.name, path.read_bytes())},
                              data={"name": name, "policy": policy})
            r.raise_for_status()
            return r.json()["dataset"]
        get = staticmethod(lambda p, **kw: requests.get(backend + p, **kw).json())
    return Api()


def app_at(page=None) -> AppTest:
    at = AppTest.from_file(str(FE / "streamlit_app.py"), default_timeout=90).run()
    if page:
        at = at.switch_page(page).run()
    return at


def problems(at):
    return [str(e.value)[:300] for e in at.exception] + [e.value[:200] for e in at.error]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_no_data(api, page):
    at = app_at(page)
    assert problems(at) == []


def test_empty_dashboard_points_to_upload(api):
    at = app_at()
    assert any("No dataset yet" in i.value for i in at.info)
    assert any("Backend connected" in c.value for c in at.sidebar.caption)


def test_dashboard_shows_the_scored_dataset(api):
    ds = api.upload(SAMPLES / "churn_sample_dirty.csv")
    at = app_at("views/portfolio.py")
    assert problems(at) == []
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Customers scored"] == "1,968"
    assert metrics["Rejected / repaired rows"] == "32 / 0"
    assert [t.label for t in at.tabs] == ["Overview", "Segments", "Customers", "Drivers", "Model performance", "Data quality"]
    assert len(at.get("plotly_chart")) >= 6      # tiers, histogram, segment, drivers, confusion, roc
    assert any("Customers scored" in m.label for m in at.metric) and ds["rows"]["scored"] == 1968


def test_dashboard_segment_and_customer_filters_work(api):
    api.upload(SAMPLES / "churn_sample_dirty.csv")
    at = app_at("views/portfolio.py")
    seg = next(s for s in at.selectbox if s.label == "Break down by")
    at = seg.select("AgeBand").run()
    assert problems(at) == []
    tier = next(s for s in at.selectbox if s.label == "Risk tier")
    at = tier.select("High").run()
    assert problems(at) == []
    high_total = api.get(f"/v1/datasets/{api.get('/v1/datasets')['datasets'][0]['id']}/customers?tier=High&limit=1")["total"]
    assert any(f"**{high_total:,}** customers match" in m.value for m in at.markdown)


def test_unlabelled_dataset_explains_missing_performance_view(api):
    api.upload(SAMPLES / "churn_sample_unlabeled.csv")
    at = app_at("views/portfolio.py")
    assert problems(at) == [] and any("no usable `Exited` column" in i.value for i in at.info)


def test_single_prediction_end_to_end(api):
    at = app_at("views/single.py")
    next(b for b in at.button if "Predict" in b.label).click().run()
    assert problems(at) == []
    card = next(m.value for m in at.markdown if 'class="risk-card"' in m.value and "<h2>" in m.value)
    assert "Medium" in card and "Likely to churn" in card and "0.55" in card       # README sample: 0.5517, Medium
    assert any("not a probability" in c.value for c in at.caption)                  # raw score is not sold as a probability
    assert len(at.get("plotly_chart")) == 2                                         # gauge + "why this score"
    assert len(api.get("/v1/history")["predictions"]) == 1


def test_single_prediction_uses_form_values(api):
    at = app_at("views/single.py")
    next(s for s in at.slider if s.label == "Age").set_value(70)
    next(s for s in at.selectbox if s.label == "Country").select("Germany")
    next(b for b in at.button if "Predict" in b.label).click().run()
    assert problems(at) == []
    sent = api.get("/v1/history")["predictions"][0]["input"]
    assert sent["Age"] == 70 and sent["Geography"] == "Germany"


def test_assistant_chat_and_tool_explorer(api, monkeypatch):
    monkeypatch.setattr("churn_api.api.chat._rag_ask",
                        lambda q: {"answer": "XGBoost won on the test set.", "sources": ["model_evaluation.md"]})
    api.upload(SAMPLES / "churn_sample_unlabeled.csv")
    at = app_at("views/assistant.py")
    at = at.chat_input[0].set_value("Why was XGBoost selected?").run()
    assert problems(at) == []
    assert any("XGBoost won" in m.markdown[0].value for m in at.chat_message if m.name == "assistant")
    assert any("Answered from the project documents" in c.value and "Documents: model_evaluation.md" in c.value
               for c in at.caption)
    next(s for s in at.selectbox if s.label == "Tool").select("get_portfolio_summary")
    at = next(b for b in at.button if b.label == "Run tool").click().run()
    assert problems(at) == [] and len(at.json) >= 1


def test_upload_page_rescore_and_delete(api):
    ds = api.upload(SAMPLES / "churn_sample_unlabeled.csv", name="to-manage")
    at = app_at("views/upload.py")
    at = next(b for b in at.button if b.label.startswith("Re-score")).click().run()
    assert problems(at) == [] and any("Re-scored" in s.value for s in at.success)
    delete = next(b for b in at.button if b.label == "Delete")
    assert delete.disabled                                  # needs the confirmation tick first
    next(c for c in at.checkbox if c.label == "Confirm delete").check().run()
    next(b for b in at.button if b.label == "Delete").click().run()
    assert api.get("/v1/datasets")["datasets"] == []


def test_backend_down_gives_a_friendly_message_not_a_traceback(api, monkeypatch):
    import streamlit as st
    from churn_ui.client import ApiClient, ApiError

    def down(self, *a, **k):
        raise ApiError("Cannot reach the backend at http://x. Is it running?", code="unreachable")
    monkeypatch.setattr(ApiClient, "_request", down)
    st.cache_data.clear()
    at = app_at("views/single.py")
    assert not at.exception and any("Backend unavailable" in e.value for e in at.error)
    st.cache_data.clear()


def test_assistant_data_question_shows_dataset_route_and_data_lookup(api, monkeypatch):
    def llm(system, user):
        if system.startswith("You choose which data tool"):
            return '{"tool": "get_segment_breakdown", "arguments": {"by": "Geography"}}'
        return "Germany has the highest flag rate, about $5 per customer."
    monkeypatch.setattr("churn_api.api.chat._llm", llm)
    api.upload(SAMPLES / "churn_sample_unlabeled.csv")
    at = app_at("views/assistant.py")
    assert any("Data questions use" in c.value for c in at.caption)
    at = at.chat_input[0].set_value("Compare churn risk by country in my uploaded file").run()
    assert problems(at) == []
    assert any("Germany has the highest" in m.markdown[0].value and "\\$5" in m.markdown[0].value   # $ is escaped
               for m in at.chat_message if m.name == "assistant")
    assert any("Answered from your uploaded data" in c.value and "Data lookup: get_segment_breakdown" in c.value
               for c in at.caption)


def test_assistant_without_a_dataset_says_so(api):
    at = app_at("views/assistant.py")
    assert problems(at) == []
    assert any("No dataset uploaded yet" in c.value for c in at.caption)
    
    


def test_customer_table_hides_surnames_until_the_box_is_ticked(api):
    api.upload(SAMPLES / "churn_sample_dirty.csv")
    at = app_at("views/portfolio.py")
    assert problems(at) == []

    def customer_table(at):
        return next(d.value for d in at.dataframe if {"row_id", "churn_score"} <= set(d.value.columns))

    assert "Surname" not in customer_table(at).columns                      # hidden by default
    box = next(c for c in at.checkbox if c.label == "Show surnames")
    at = box.check().run()
    assert problems(at) == []
    assert "Surname" in customer_table(at).columns                          # the box brings them back