"""POST /v1/chat: RAG answers. The RAG pipeline itself is mocked (it needs torch, chromadb and an LLM)."""
import pytest

from churn_api.api import chat as chat_mod


def post(client, content="Why was XGBoost selected?"):
    return client.post("/v1/chat", json={"messages": [{"role": "user", "content": content}]})


def test_returns_answer_and_sources(client, monkeypatch):
    seen = []
    monkeypatch.setattr(chat_mod, "_rag_ask", lambda q: seen.append(q) or {"answer": "Best AUC.", "sources": ["a.md"]})
    r = client.post("/v1/chat", json={"messages": [
        {"role": "user", "content": "first"}, {"role": "assistant", "content": "x"},
        {"role": "user", "content": "  latest question "}]})
    assert r.status_code == 200 and r.json == {"reply": "Best AUC.", "sources": ["a.md"], "route": "knowledge"}
    assert seen == ["latest question"]                      # only the last user message goes to RAG


@pytest.mark.parametrize("body", [{}, {"messages": "hi"}, {"messages": []},
                                  {"messages": [{"role": "assistant", "content": "hi"}]},
                                  {"messages": [{"role": "user", "content": "   "}]}])
def test_bad_requests_are_400(client, body):
    assert client.post("/v1/chat", json=body).status_code == 400


def test_too_long_question_is_400(client):
    assert post(client, "x" * 1001).status_code == 400


def test_rag_failure_hides_internal_error(client, monkeypatch):
    monkeypatch.setattr(chat_mod, "_rag_ask", lambda q: {
        "answer": "The language model could not answer right now.",
        "error": "Could not reach ollama at localhost:11434", "sources": []})
    r = post(client)
    assert r.status_code == 503 and r.json["error"]["code"] == "chat_unavailable"
    assert "11434" not in r.get_data(as_text=True)


def test_works_without_the_churn_model_loaded(tmp_path, monkeypatch):
    from conftest import make_app
    monkeypatch.setattr(chat_mod, "_rag_ask", lambda q: {"answer": "ok", "sources": []})
    app = make_app(tmp_path, MODEL_DIR=tmp_path / "missing")      # model fails to load; API stays up
    assert post(app.test_client()).status_code == 200

def test_data_question_end_to_end_over_http(client, dirty_dataset, monkeypatch):
    def llm(system, user):
        if system.startswith("You choose which data tool"):
            return '{"tool": "get_segment_breakdown", "arguments": {"by": "Geography"}}'
        return "Germany is highest."
    monkeypatch.setattr(chat_mod, "_llm", llm)
    monkeypatch.setattr(chat_mod, "_rag_ask", lambda q: (_ for _ in ()).throw(AssertionError("RAG must not run")))
    r = post(client, "Compare churn risk by country across my uploaded file")
    assert r.status_code == 200
    assert r.json == {"reply": "Germany is highest.", "sources": ["data tool: get_segment_breakdown"], "route": "data"}


def test_dataset_id_in_the_request_reaches_the_tool(client, dirty_dataset, monkeypatch):
    monkeypatch.setattr(chat_mod, "_llm", lambda s, u: '{"tool": "get_portfolio_summary"}'
                        if s.startswith("You choose") else "ok")
    body = {"messages": [{"role": "user", "content": "Summarise my uploaded file"}], "dataset_id": "nope"}
    r = client.post("/v1/chat", json=body)
    assert r.status_code == 200 and "not found" in r.json["reply"]


def test_dataset_id_must_be_a_string(client):
    body = {"messages": [{"role": "user", "content": "hi"}], "dataset_id": 5}
    assert client.post("/v1/chat", json=body).status_code == 400


def test_model_not_loaded_still_answers_knowledge_questions(tmp_path, monkeypatch):
    from conftest import make_app
    monkeypatch.setattr(chat_mod, "_rag_ask", lambda q: {"answer": "ok", "sources": ["a.md"]})
    app = make_app(tmp_path, MODEL_DIR=tmp_path / "missing")
    assert post(app.test_client(), "Why was XGBoost selected?").status_code == 200