"""Step 3: run a validated plan, compact the tool output, and have the LLM word the answer. Fake LLM, real tools."""
import pytest

from churn_api.api.deps import ctx
from churn_api.services.planner import Plan
from churn_api.services.tool_answer import (MAX_DATA_CHARS, ToolResult, build_answer_prompt, compact, data_text,
                                            run_plan, write_answer)
from churn_api.services.tools import build_tools


@pytest.fixture
def tools(app):
    with app.app_context():
        yield build_tools(ctx())


def test_run_plan_executes_the_tool_and_labels_the_source(tools, dirty_dataset):
    r = run_plan(Plan(tool="get_segment_breakdown", arguments={"by": "Geography"}), tools)
    assert r.ok and r.source == "data tool: get_segment_breakdown"
    assert {s["segment"] for s in r.data["segments"]} == {"France", "Germany", "Spain"}


def test_dataset_id_comes_from_the_app_not_the_llm(tools, dirty_dataset):
    ok = run_plan(Plan(tool="get_portfolio_summary"), tools, dataset_id=dirty_dataset)
    assert ok.ok
    bad = run_plan(Plan(tool="get_portfolio_summary"), tools, dataset_id="nope")
    assert not bad.ok and "not found" in bad.error


def test_no_dataset_uploaded_gives_a_clear_message(tools):
    r = run_plan(Plan(tool="get_portfolio_summary"), tools)
    assert not r.ok and "No dataset has been uploaded" in r.error


def test_unknown_customer_is_a_clean_error(tools, dirty_dataset):
    r = run_plan(Plan(tool="get_customer", arguments={"customer_id": "99999999"}), tools)
    assert not r.ok and "not found" in r.error


def test_plan_without_a_tool_is_not_executed(tools):
    assert not run_plan(Plan(), tools).ok


def test_unexpected_crash_is_contained_and_not_leaked(tools, dirty_dataset, monkeypatch):
    def boom(args):
        raise RuntimeError("secret stack detail /srv/app/x.py")
    monkeypatch.setattr(tools["get_portfolio_summary"], "handler", boom)
    r = run_plan(Plan(tool="get_portfolio_summary"), tools)
    assert not r.ok and r.error == "The data tool failed unexpectedly." and "secret" not in r.error


# ---------------------------------------------------------------------------------------- compaction
def test_compact_removes_personal_data_and_noise_and_formats_numbers():
    out = compact({"Surname": "Hill", "row_id": 7, "CustomerId": 15863884, "flag_rate": 0.31013916,
                   "share": 0.5, "avg_score": 0.4328753349, "will_churn": True, "note": "x" * 500,
                   "text": "y" * 500})
    assert "Surname" not in out and "row_id" not in out and "note" not in out
    assert out["CustomerId"] == 15863884 and out["will_churn"] is True
    assert out["flag_rate"] == "31.0%" and out["share"] == "50.0%"       # ready-made percentages
    assert out["avg_score"] == 0.4329                                    # scores stay numbers, rounded
    assert out["text"].endswith("...") and len(out["text"]) < 170


def test_compact_truncates_long_lists_with_a_note():
    out = compact({"customers": list(range(25))})["customers"]
    assert out[:10] == list(range(10)) and out[10] == {"_note": "15 more items not shown"}


def test_real_portfolio_summary_fits_the_llm_budget(tools, dirty_dataset):
    r = run_plan(Plan(tool="get_portfolio_summary"), tools)
    text = data_text(r.data)
    assert len(text) <= MAX_DATA_CHARS and "truncated" not in text
    assert "customers_scored" in text and "%" in text                    # the facts survive

def test_real_customer_rows_never_expose_surnames(app, tools, dirty_dataset):
    r = run_plan(Plan(tool="list_top_risk_customers", arguments={"limit": 50}), tools)
    assert all("Surname" not in c for c in r.data["customers"])                   # removed at the source
    surnames = set(app.extensions["churn"].store.results(dirty_dataset)["Surname"].dropna())
    text = data_text(r.data)
    assert len(text) <= MAX_DATA_CHARS
    assert not any(f'"{s}"' in text for s in surnames) and "Surname" not in text

# ---------------------------------------------------------------------------------------- answer writer
def test_prompt_contains_question_tool_scope_and_data():
    r = ToolResult("get_segment_breakdown", {"by": "Geography"}, data={"by": "Geography", "segments": []})
    system, user = build_answer_prompt("  Which country is worst? ", r)
    assert "Question: Which country is worst?" in user and "get_segment_breakdown (by=Geography)" in user
    assert '"by":"Geography"' in user and user.endswith("Answer:")
    assert "not a probability" in system and "human reviewer" in system


def test_write_answer_uses_the_llm_and_strips(tools, dirty_dataset):
    r = run_plan(Plan(tool="get_portfolio_summary"), tools)
    seen = []
    text = write_answer("How many are high risk?", r, lambda s, u: seen.append(u) or "  351 customers.  ")
    assert text == "351 customers." and "customers_scored" in seen[0]


def test_write_answer_refuses_failed_results_and_propagates_llm_errors():
    with pytest.raises(ValueError):
        write_answer("q", ToolResult(error="No dataset has been uploaded yet."), lambda s, u: "x")
    ok = ToolResult("get_portfolio_summary", {}, data={"a": 1})
    def down(s, u):
        raise RuntimeError("ollama down")
    with pytest.raises(RuntimeError):
        write_answer("q", ok, down)