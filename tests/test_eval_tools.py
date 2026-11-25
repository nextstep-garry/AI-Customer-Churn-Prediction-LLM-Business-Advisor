"""The evaluation harness itself: a perfect fake LLM must score 100%, and a sloppy one must be caught."""
import json

import pytest

import eval_tools as ev
from churn_api.api.deps import ctx
from churn_api.services.tools import build_tools


# ---------------------------------------------------------------------------------------------- number checks
def test_numbers_in_handles_commas_percentages_and_ids():
    assert ev.numbers_in("1,968 customers, 43.8% flagged, id 15863884") == {1968.0, 43.8, 15863884.0}
    assert ev.numbers_in("") == set()


def test_ungrounded_finds_invented_numbers_only():
    data = '{"customers":1968,"flag_rate":"43.8%"}'
    assert ev.ungrounded("1,968 customers; 43.8% flagged", data, "q") == []
    assert ev.ungrounded("There are 2,500 customers", data, "q") == [2500.0]
    assert ev.ungrounded("The top 5 are listed", data, "Show the top 5") == []        # number from the question
    assert ev.ungrounded("2 of them", data, "q") == []                                 # trivial 0-10 are allowed


def test_contains_fact_compares_numbers_as_numbers_and_text_case_insensitively():
    assert ev.contains_fact("There are 1,968 customers", "1968")
    assert not ev.contains_fact("There are 1,969 customers", "1968")
    assert ev.contains_fact("GERMANY is highest", "Germany")


def test_refusal_pattern():
    assert ev.REFUSAL.search("The email address is not available in the data.")
    assert not ev.REFUSAL.search("The email address is jo@example.com.")


# ---------------------------------------------------------------------------------------------- harness
class Oracle:
    """A perfect LLM: plans exactly the ideal tool call and writes an answer from the true facts."""
    def __init__(self, questions, sloppy=False):
        self.by_text = {q.text: q for q in questions}
        self.sloppy = sloppy

    def __call__(self, system, user):
        text = user.split("Question: ", 1)[1].split("\n", 1)[0]
        q = self.by_text[text]
        if system.startswith("You choose which data tool"):
            return json.dumps(q.plan)
        if q.refusal:
            return "That information is not available in the data."
        extra = " About 87654.321 of them churn." if self.sloppy else ""
        return "Result: " + ", ".join(q.must) + "." + extra


RAG_OK = lambda q: {"answer": "XGBoost had the best test AUC.", "sources": ["model_evaluation.md"]}   # noqa: E731
RAG_NO = lambda q: {"answer": "This information is not available in the project knowledge base.", "sources": []}  # noqa: E731


@pytest.fixture
def setup(app, dirty_dataset):
    with app.app_context():
        tools = build_tools(ctx())
        yield app, ev.build_questions(tools)


def test_a_perfect_llm_passes_every_question(setup):
    app, questions = setup
    rows = ev.run_eval(app, Oracle(questions), RAG_OK, with_rag=True)
    failed = [(r.key, r.problem, r.route, r.tool) for r in rows if not r.passed]
    assert failed == [] and len(rows) == len(questions)


def test_data_only_mode_skips_document_questions(setup):
    app, questions = setup
    rows = ev.run_eval(app, Oracle(questions), RAG_NO, with_rag=False)
    assert {r.key for r in rows}.isdisjoint({"both_country", "docs_xgb"})
    assert all(r.passed for r in rows)                    # what_if correctly ends with no data tool


def test_invented_numbers_are_caught(setup):
    app, questions = setup
    rows = ev.run_eval(app, Oracle(questions, sloppy=True), RAG_NO, with_rag=False)
    portfolio = next(r for r in rows if r.key == "portfolio")
    assert not portfolio.grounded_ok and not portfolio.passed and "87654.321" in portfolio.problem


def test_wrong_tool_and_missing_facts_are_caught(setup):
    app, questions = setup
    class Wrong(Oracle):
        def __call__(self, system, user):
            if system.startswith("You choose which data tool"):
                return '{"tool": "get_portfolio_summary", "arguments": {}}'      # always the same tool
            return "Something vague."
    rows = ev.run_eval(app, Wrong(questions), RAG_NO, with_rag=False)
    country = next(r for r in rows if r.key == "country")
    assert not country.tool_ok and not country.facts_ok and not country.passed


def test_a_broken_planner_fails_what_if_check_and_is_reported(setup):
    app, questions = setup
    class Broken(Oracle):
        def __call__(self, system, user):
            if system.startswith("You choose which data tool"):
                return "no idea"
            return "x"
    rows = ev.run_eval(app, Broken(questions), RAG_NO, with_rag=False)
    assert not next(r for r in rows if r.key == "country").passed
    assert next(r for r in rows if r.key == "what_if").passed        # no tool is the correct outcome there


def test_report_summarises(setup):
    app, questions = setup
    rows = ev.run_eval(app, Oracle(questions), RAG_NO, with_rag=False)
    text = ev.report(rows)
    assert "PASS" in text and f"{len(rows)}/{len(rows)} passed" in text