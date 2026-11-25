"""Step 2 planner: the LLM proposes a tool call as JSON; we parse leniently and validate strictly. Uses a fake LLM."""
import json

import pytest

from churn_api.services.planner import EXAMPLES, build_planner_prompt, extract_json, plan
from churn_api.services.tools import build_tools

TOOLS = build_tools(None)          # building the registry needs no app context; only running a tool does


def fake(reply):
    return lambda system, user: reply


def run(question, reply):
    return plan(question, TOOLS, fake(reply))


def test_clean_reply_gives_a_valid_plan():
    p = run("Compare risk by country", '{"tool": "get_segment_breakdown", "arguments": {"by": "Geography"}}')
    assert p.ok and p.tool == "get_segment_breakdown" and p.arguments == {"by": "Geography"}


@pytest.mark.parametrize("reply", [
    '```json\n{"tool": "get_portfolio_summary", "arguments": {}}\n```',
    'Sure! Here you go: {"tool": "get_portfolio_summary", "arguments": {}} Hope that helps.',
    '{"tool": "get_portfolio_summary"}',                                     # arguments omitted
])
def test_messy_but_recoverable_replies(reply):
    assert run("How many are high risk?", reply).ok


def test_enum_values_are_case_corrected_and_hidden_or_unknown_args_dropped():
    p = run("top german customers",
            '{"tool": "list_top_risk_customers", "arguments": {"geography": "germany", "tier": "high",'
            ' "limit": "5", "dataset_id": "guess123", "colour": "red"}}')
    assert p.ok and p.arguments == {"geography": "Germany", "tier": "High", "limit": 5}


def test_customer_id_is_accepted_only_if_it_is_in_the_question():
    reply = '{"tool": "get_customer", "arguments": {"customer_id": "15634602"}}'
    assert run("Why is customer 15634602 at risk?", reply).ok
    bad = run("Why is that customer at risk?", reply)
    assert not bad.ok and "does not appear" in bad.error


@pytest.mark.parametrize("reply,fragment", [
    ("I am not sure", "no_json_in_reply"),
    ('{"tool": "delete_everything", "arguments": {}}', "unknown_tool"),
    ('{"tool": "predict_what_if", "arguments": {"customer": {}}}', "unknown_tool"),     # not allowed for the planner
    ('{"tool": "get_segment_breakdown", "arguments": {}}', "Missing required"),
    ('{"tool": "get_segment_breakdown", "arguments": {"by": "ShoeSize"}}', "not a valid value"),
    ('{"tool": "list_top_risk_customers", "arguments": {"limit": "lots"}}', "whole number"),
    ('{"tool": "get_portfolio_summary", "arguments": "all of it"}', "arguments_not_an_object"),
])
def test_unsafe_replies_are_rejected_with_a_reason(reply, fragment):
    p = run("whatever 15634602", reply)
    assert not p.ok and fragment in p.error and p.raw == reply


def test_null_tool_means_no_tool_needed_not_an_error():
    p = run("What is the weather?", '{"tool": null, "arguments": {}}')
    assert p.tool is None and p.error is None and not p.ok


def test_llm_failure_is_reported_not_raised():
    def boom(system, user):
        raise RuntimeError("ollama down")
    p = plan("anything", TOOLS, boom)
    assert p.error == "planner_llm_failed" and p.tool is None


def test_prompt_lists_every_allowed_tool_and_hides_dataset_id():
    system, user = build_planner_prompt("  Which country is worst?  ", TOOLS)
    for name in TOOLS:
        assert (name in system) == (name != "predict_what_if")
    assert "dataset_id" not in system.split("Examples:")[0]
    assert "by*: Geography|Gender" in system                    # required marker + enum shown
    assert user == "Question: Which country is worst?\nJSON:"


@pytest.mark.parametrize("question,answer", EXAMPLES)
def test_every_prompt_example_is_itself_valid(question, answer):
    """If a tool is renamed or its schema changes, the few-shot examples must be updated too."""
    p = run(question, json.dumps(answer))
    assert p.error is None and p.tool == answer["tool"]


def test_extract_json_ignores_non_objects():
    assert extract_json("[1,2,3]") is None
    assert extract_json('noise {bad json} {"a": 1}') == {"a": 1}\n

# ---------------------------------------------------------------------------------------------- rule-planned questions
from churn_api.services.planner import known_intent                                          # noqa: E402


@pytest.mark.parametrize("question,args", [
    ("tell me top churner", {}),
    ("who are the top churners", {}),
    ("Show the worst churners in Germany", {"geography": "Germany"}),
    ("Who is likely to leave?", {"tier": "High"}),
    ("who is at risk in spain", {"tier": "High", "geography": "Spain"}),
])
def test_top_churner_questions_are_planned_by_rule_without_the_llm(question, args):
    def llm_must_not_run(system, user):
        raise AssertionError("the language model must not be called for a rule-planned question")
    p = plan(question, TOOLS, llm_must_not_run)
    assert p.ok and p.tool == "list_top_risk_customers" and p.arguments == args and p.raw == "(planned by rule)"


def test_other_questions_still_go_to_the_llm():
    calls = []
    def llm(system, user):
        calls.append(user)
        return '{"tool": "get_portfolio_summary"}'
    assert plan("How many are high risk?", TOOLS, llm).ok
    assert plan("What are the top churn drivers?", TOOLS, llm).tool == "get_portfolio_summary"    # not a customer list
    assert len(calls) == 2


@pytest.mark.parametrize("question", ["tell me top churner", "who are the top churners", "Who is likely to leave?",
                                      "Show the worst churners in Germany"])
def test_router_and_planner_agree_on_these_phrases(question):
    from churn_api.services.router import DATA, route_question
    assert known_intent(question) is not None and route_question(question).route == DATA
