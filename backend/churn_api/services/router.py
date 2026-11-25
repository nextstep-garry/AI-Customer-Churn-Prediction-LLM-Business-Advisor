"""Step 1 of the tool-using assistant: decide WHERE a question should be answered from.

    KNOWLEDGE  documents only (RAG): model, data prep, app, sample policies and playbooks
    DATA       live numbers only (tools): the uploaded dataset, one customer, a what-if
    BOTH       needs the documents AND the live numbers (e.g. a segment finding + the policy on acting on it)

Deliberately rule-based: no LLM call, so it is instant, free and predictable, and the reason for every
decision is returned for logging. When unsure it answers KNOWLEDGE, which is exactly what the chat did
before tools existed, so this can never make an answer worse than today's.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

KNOWLEDGE, DATA, BOTH = "knowledge", "data", "both"

# --- strong signals: the person is clearly talking about THEIR uploaded data -----------------------------
_STRONG_DATA = {
    "customer_id": r"\b\d{7,9}\b",                                    # ids look like 15634602
    "my_data": r"\b(uploaded|my (data(set)?|file|csv|customers?|portfolio)|our (customers?|portfolio|data(set)?)"
               r"|(this|latest|current|last) (upload|dataset|file|batch))\b",
    "what_if": r"\bwhat[ -]?if\b|\bhypothetical\b|\bsimulate\b|\bif (he|she|they|the customer) (had|were|was|has)\b",
    "top_n": r"\btop \d+\b",
    "top_churners": r"\b(top|worst|riskiest|most likely)\b.{0,25}\b(churners?|leavers?)\b",   # tell me top churner
    "who_churns": r"\bwho\b.{0,30}\b(churn|churns|leave|leaves|leaving|at risk|likely)\b",     # who is likely to leave
    "list_customers": r"\b(list|show|give me|name|find)\b.{0,40}\bcustomers\b"
                      r"|\b(riskiest|highest[- ]risk|most at[- ]risk)\s+customers\b"
                      r"|\bwhich customers\b.{0,40}\b(risk|churn|leave|leaving)\b",
}

# --- knowledge signals: documents, rules, methods, the app itself ------------------------------------------
_KNOWLEDGE = (
    r"\b(policy|policies|sp-\d+|kb-\d+|playbook|play [a-f]|strateg(y|ies)|retention (play|action)s?|eligib\w*"
    r"|model card|manifest|calibrat\w*|xgboost|random forest|logistic|shap|smote|hyper-?parameters?|tuning"
    r"|pre-?process\w*|pipeline|feature (selection|engineering)|leakage|imbalance|cross[- ]valid\w*"
    r"|api|docker|endpoint|env(ironment)?|config\w*|readme|changelog|threshold|methodolog\w*"
    r"|why (was|were|did we)|how (was|were|did we)|roc|auc|precision|recall|f1)\b"
)

# --- weak signals: aggregate questions about segments that either source could answer ------------------------
_SEGMENT = (r"\b(germany|german|france|french|spain|spanish|geography|country|countries|gender|male|female"
            r"|age ?bands?|age groups?|products?|segments?|tiers?|high[- ]risk|medium[- ]risk|low[- ]risk"
            r"|churners?|inactive|active members?)\b")
_ASKING = r"\b(which|how many|what (percent|percentage|share|proportion)|highest|lowest|most|least|average|compare|top|worst)\b"


@dataclass
class Route:
    route: str
    reasons: list[str] = field(default_factory=list)


def route_question(question: str) -> Route:
    q = (question or "").lower()
    strong = [name for name, pat in _STRONG_DATA.items() if re.search(pat, q)]
    knowledge = re.search(_KNOWLEDGE, q) is not None

    if strong and knowledge:
        return Route(BOTH, [f"data:{n}" for n in strong] + ["knowledge:keywords"])
    if strong:
        return Route(DATA, [f"data:{n}" for n in strong])
    if knowledge:
        return Route(KNOWLEDGE, ["knowledge:keywords"])
    if re.search(_SEGMENT, q) and re.search(_ASKING, q):
        return Route(BOTH, ["weak:segment_question"])
    return Route(KNOWLEDGE, ["default"])