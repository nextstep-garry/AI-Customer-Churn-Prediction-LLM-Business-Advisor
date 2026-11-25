"""Step 1 router: sends each question to documents (RAG), live data (tools) or both. No LLM involved."""
import pytest

from churn_api.services.router import BOTH, DATA, KNOWLEDGE, route_question

CASES = [
    # --- documents only: these are real questions from the RAG evaluation set ---
    ("What missing value strategy was used?", KNOWLEDGE),
    ("Why was XGBoost selected?", KNOWLEDGE),
    ("What does SP-03 say about eligibility?", KNOWLEDGE),
    ("What happens if API_KEY is not set?", KNOWLEDGE),
    ("How many customers were scored?", KNOWLEDGE),          # a fact in the project docs, not the live upload
    ("What are the risk tiers?", KNOWLEDGE),                 # a definition
    ("What is the cost of a retention campaign?", KNOWLEDGE),
    ("hello", KNOWLEDGE),                                    # unsure -> the pre-tools behaviour
    ("", KNOWLEDGE),
    # --- live data only ---
    ("tell me top churner", DATA),
    ("who are the top churners", DATA),
    ("Who is likely to leave?", DATA),
    ("What are the top churn drivers?", KNOWLEDGE),          # a document question: must stay on the documents
    ("tell me about project", KNOWLEDGE),
    ("Why is customer 15634602 likely to leave?", DATA),
    ("Show the top 10 customers in Germany", DATA),
    ("What if this customer held two products instead of one?", DATA),
    ("How many customers in my uploaded file are high risk?", DATA),
    ("Which customers are highest risk?", DATA),
    ("Summarise the latest upload", DATA),
    # --- both ---
    ("Which country has the highest churn?", BOTH),
    ("How many high-risk customers are there?", BOTH),
    ("Can I contact customer 15634602 under SP-03?", BOTH),  # a customer AND a policy
]


@pytest.mark.parametrize("question,expected", CASES)
def test_route(question, expected):
    assert route_question(question).route == expected


def test_reasons_are_reported_for_logging():
    r = route_question("Why is customer 15634602 likely to leave?")
    assert r.route == DATA and r.reasons == ["data:customer_id"]
    assert route_question("hello").reasons == ["default"]


def test_case_and_whitespace_do_not_matter():
    assert route_question("  WHY WAS XGBOOST SELECTED?  ").route == KNOWLEDGE
    assert route_question(None).route == KNOWLEDGE