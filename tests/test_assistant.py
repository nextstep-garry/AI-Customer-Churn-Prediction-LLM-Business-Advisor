"""Step 4: the orchestrator picks the right source and degrades honestly. Real tools, fake LLM and fake RAG."""
import json

import pytest

from churn_api.api.deps import ctx
from churn_api.services.assistant import DATA_HINT, ChatUnavailable, answer_question
from churn_api.services.tools import build_tools

K_Q = "Why was XGBoost selected?"                                   # -> knowledge
D_Q = "Compare churn risk by country across my uploaded file"        # -> data
B_Q = "Which country has the highest churn?"                         # -> both

PLAN_GEO = json.dumps({"tool": "get_segment_breakdown", "arguments": {"by": "Geography"}})
RAG_OK = {"answer": "XGBoost had the best test AUC.", "sources": ["model_evaluation.md"]}
RAG_REFUSAL = {"answer": "This information is not available in the project knowledge base.", "sources": []}
RAG_ERROR = {"answer": "The language model could not answer right now.", "sources": [], "error": "ollama down"}


class Spy:
    """Counts calls so tests can prove which sources were (not) used."""
    def __init__(self, rag=RAG_OK, plan=PLAN_GEO, writer="Germany has the highest flag rate (43.8%).", llm_down=False):
        self.rag_calls, self.llm_calls = [], []
        self._rag, self._plan, self._writer, self._llm_down = rag, plan, writer, llm_down

    def rag_ask(self, q):
        self.rag_calls.append(q)
        if isinstance(self._rag, Exception):
            raise self._rag
        return self._rag

    def llm(self, system, user):
        self.llm_calls.append(system[:20])
        if self._llm_down:
            raise RuntimeError("ollama down")
        return self._plan if system.startswith("You choose which data tool") else self._writer


@pytest.fixture
def tools(app):
    with app.app_context():
        yield build_tools(ctx())


def ask(q, spy, tools, dataset_id=None):
    return answer_question(q, dataset_id, get_tools=lambda: tools, rag_ask=spy.rag_ask, llm=spy.llm)


# ---------------------------------------------------------------------------------------------- knowledge
def test_knowledge_question_uses_only_rag(tools):
    spy = Spy()
    a = answer_question(K_Q, None, get_tools=lambda: pytest.fail("tools must not be built"),
                        rag_ask=spy.rag_ask, llm=spy.llm)
    assert (a.reply, a.sources, a.route) == (RAG_OK["answer"], ["model_evaluation.md"], "knowledge")
    assert spy.llm_calls == []


def test_knowledge_question_with_rag_failure_is_unavailable(tools):
    with pytest.raises(ChatUnavailable, match="language model"):
        ask(K_Q, Spy(rag=RAG_ERROR), tools)


def test_knowledge_question_lets_rag_exceptions_propagate(tools):
    with pytest.raises(RuntimeError):
        ask(K_Q, Spy(rag=RuntimeError("chroma exploded")), tools)


# ---------------------------------------------------------------------------------------------- data
def test_data_question_runs_planner_tool_and_writer(tools, dirty_dataset):
    spy = Spy()
    a = ask(D_Q, spy, tools)
    assert a.route == "data" and a.tool == "get_segment_breakdown"
    assert a.reply == "Germany has the highest flag rate (43.8%)." and a.sources == ["data tool: get_segment_breakdown"]
    assert spy.rag_calls == [] and len(spy.llm_calls) == 2           # planner + writer, no RAG


def test_data_question_with_no_dataset_explains_what_is_missing(tools):
    a = ask(D_Q, Spy(), tools)
    assert a.route == "data" and a.sources == [] and "No dataset has been uploaded" in a.reply


def test_dataset_id_from_the_app_is_used(tools, dirty_dataset):
    assert ask(D_Q, Spy(), tools, dataset_id=dirty_dataset).sources
    assert "not found" in ask(D_Q, Spy(), tools, dataset_id="nope").reply


def test_unusable_plan_falls_back_to_documents(tools, dirty_dataset):
    spy = Spy(plan="I have no idea")
    a = ask(D_Q, spy, tools)
    assert a.route == "knowledge" and a.sources == ["model_evaluation.md"] and spy.rag_calls == [D_Q]


def test_writer_failure_after_a_good_tool_result_falls_back_to_documents(tools, dirty_dataset):
    spy = Spy()
    calls = {"n": 0}
    base = spy.llm
    def flaky(system, user):                                         # planner works, writer's LLM call dies
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("timeout")
        return base(system, user)
    a = answer_question(D_Q, None, get_tools=lambda: tools, rag_ask=spy.rag_ask, llm=flaky)
    assert a.route == "knowledge" and a.sources == ["model_evaluation.md"]


def test_data_failure_and_documents_refusal_gives_a_helpful_hint(tools, dirty_dataset):
    a = ask(D_Q, Spy(plan="no idea", rag=RAG_REFUSAL), tools)
    assert a.reply == DATA_HINT and a.sources == []


def test_everything_down_is_unavailable_not_a_fake_answer(tools, dirty_dataset):
    with pytest.raises(ChatUnavailable):
        ask(D_Q, Spy(plan="no idea", rag=RAG_ERROR, llm_down=True), tools)


def test_model_not_loaded_makes_the_data_path_fail_cleanly():
    def no_model():
        raise RuntimeError("model not loaded")
    spy = Spy()
    a = answer_question(D_Q, None, get_tools=no_model, rag_ask=spy.rag_ask, llm=spy.llm)
    assert a.route == "knowledge" and spy.llm_calls == []            # fell back to documents without calling the LLM


# ---------------------------------------------------------------------------------------------- both
def test_both_combines_labelled_answers_and_all_sources(tools, dirty_dataset):
    a = ask(B_Q, Spy(), tools)
    assert a.route == "both"
    assert a.reply.startswith("**From your uploaded data**") and "**From the project documents**" in a.reply
    assert a.sources == ["data tool: get_segment_breakdown", "model_evaluation.md"]


def test_both_with_a_document_refusal_returns_only_the_data_answer(tools, dirty_dataset):
    a = ask(B_Q, Spy(rag=RAG_REFUSAL), tools)
    assert a.reply == "Germany has the highest flag rate (43.8%)." and a.sources == ["data tool: get_segment_breakdown"]


def test_both_with_a_crashing_rag_still_returns_the_data_answer(tools, dirty_dataset):
    a = ask(B_Q, Spy(rag=RuntimeError("chroma exploded")), tools)
    assert a.route == "both" and a.sources == ["data tool: get_segment_breakdown"]


def test_both_without_a_dataset_answers_from_documents_and_says_so(tools):
    a = ask(B_Q, Spy(), tools)
    assert a.route == "knowledge" and a.reply.startswith(RAG_OK["answer"])
    assert "No dataset has been uploaded" in a.reply and a.sources == ["model_evaluation.md"]


def test_both_with_nothing_available_is_unavailable(tools, dirty_dataset):
    with pytest.raises(ChatUnavailable):
        ask(B_Q, Spy(plan="no idea", rag=RAG_ERROR), tools)