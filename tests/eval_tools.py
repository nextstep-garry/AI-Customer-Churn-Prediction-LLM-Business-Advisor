"""Evaluate the assistant's DATA path with a real LLM (the RAG path has its own 64-question evaluation).

    python tests/eval_tools.py                 # data questions only (needs Ollama / your LLM running)
    python tests/eval_tools.py --with-rag      # also the document and mixed questions (loads the embedder too)

For every question it checks:
    route      the router sent it where we expect (knowledge / data / both)
    tool       the planner chose an acceptable data tool (or correctly chose none)
    facts      the answer contains facts computed from the tools themselves (never hard-coded numbers)
    grounded   EVERY number in the answer appears in the data the writer was given (catches invented numbers)
    refusal    for questions the data cannot answer, the answer says so

The checking code is plain functions, so tests/test_eval_tools.py proves the harness itself works with a fake LLM.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "tests")]

from churn_api.api.deps import ctx                                            # noqa: E402
from churn_api.services.assistant import ChatUnavailable, answer_question     # noqa: E402
from churn_api.services.tools import build_tools                              # noqa: E402

REFUSAL = re.compile(r"not available|not (included|provided|contained|present|in the data)|no (email|such)|"
                     r"does not (contain|include)|cannot (find|provide)|can't (find|provide)", re.I)
WRITER_MARK = "You answer questions about an uploaded"          # start of the writer's system prompt


# ------------------------------------------------------------------------------------------ number checking
def numbers_in(text: str) -> set[float]:
    """All numbers in `text` as floats. '1,968' -> 1968.0, '43.8%' -> 43.8"""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text or "")
    return {round(float(n), 6) for n in re.findall(r"\d+(?:\.\d+)?", text)}


def ungrounded(answer: str, data_text: str, question: str) -> list[float]:
    """Numbers in the answer that are neither in the data, nor in the question, nor a trivial 0-10."""
    allowed = numbers_in(data_text) | numbers_in(question) | {float(i) for i in range(11)}
    return sorted(numbers_in(answer) - allowed)


def contains_fact(answer: str, fact: str) -> bool:
    if re.fullmatch(r"[\d,]+(?:\.\d+)?", fact):                  # numeric fact: compare as a number
        return numbers_in(fact) <= numbers_in(answer)
    return fact.lower() in answer.lower()


# ------------------------------------------------------------------------------------------ question set
@dataclass
class Q:
    key: str
    text: str
    route: str                                    # expected route
    plan: dict                                    # the ideal planner JSON (also used by the test oracle)
    must: list[str] = field(default_factory=list)  # facts the answer must contain
    refusal: bool = False                         # the data cannot answer this: the answer must say so
    needs_rag: bool = False
    also_ok: tuple = ()                           # other routes that are an acceptable outcome

    @property
    def tools(self) -> tuple:
        t = self.plan.get("tool")
        return (t,) if t else ()


def build_questions(tools: dict) -> list[Q]:
    """Questions plus ground truth, both computed from the tools on the loaded dataset."""
    h = lambda name, **a: tools[name].handler(a)                  # noqa: E731
    summary = h("get_portfolio_summary")
    top = h("list_top_risk_customers", limit=1)["customers"][0]
    top_de = h("list_top_risk_customers", limit=1, geography="Germany")["customers"][0]
    worst = lambda by: max(h("get_segment_breakdown", by=by)["segments"], key=lambda s: s["flag_rate"])["segment"]  # noqa: E731
    cid = str(top["CustomerId"])
    return [
        Q("portfolio", "How many customers are in my uploaded file and how many are high risk?", "data",
          {"tool": "get_portfolio_summary", "arguments": {}},
          [str(summary["kpis"]["customers_scored"]), str(summary["kpis"]["high_risk"])]),
        Q("country", "Compare churn risk by country in my uploaded file", "data",
          {"tool": "get_segment_breakdown", "arguments": {"by": "Geography"}}, [str(worst("Geography"))]),
        Q("ageband", "Which age band has the highest churn risk in my uploaded file?", "data",
          {"tool": "get_segment_breakdown", "arguments": {"by": "AgeBand"}}, [str(worst("AgeBand"))]),
        Q("products", "Break down my uploaded customers by number of products", "data",
          {"tool": "get_segment_breakdown", "arguments": {"by": "NumOfProducts"}}, []),
        Q("top_germany", "Show the top 5 highest-risk customers in Germany", "data",
          {"tool": "list_top_risk_customers", "arguments": {"geography": "Germany", "limit": 5}},
          [str(top_de["CustomerId"])]),
        Q("customer", f"Why is customer {cid} likely to leave?", "data",
          {"tool": "get_customer", "arguments": {"customer_id": cid}}, [cid]),
        Q("email", f"What is the email address of customer {cid}?", "data",
          {"tool": "get_customer", "arguments": {"customer_id": cid}}, [], refusal=True),
        Q("datasets", "Which datasets have I uploaded?", "data",
          {"tool": "list_datasets", "arguments": {}}, []),
        Q("what_if", f"What if customer {cid} had 2 products instead of 1?", "data",
          {"tool": None, "arguments": {}}, also_ok=("knowledge",)),         # known limit: no invented customer data;
                                                                           # falling back to the documents is fine
        Q("both_country", "Which country has the highest churn?", "both",
          {"tool": "get_segment_breakdown", "arguments": {"by": "Geography"}}, [str(worst("Geography"))], needs_rag=True),
        Q("docs_xgb", "Why was XGBoost selected?", "knowledge", {"tool": None, "arguments": {}}, needs_rag=True),
    ]


# ------------------------------------------------------------------------------------------ running
class Recorder:
    """Wraps the LLM so the harness can see the exact DATA the writer was given."""
    def __init__(self, llm):
        self.llm, self.data_text = llm, ""

    def __call__(self, system: str, user: str) -> str:
        if system.startswith(WRITER_MARK):
            self.data_text = user.split("DATA:\n", 1)[-1].rsplit("\n\nAnswer:", 1)[0]
        return self.llm(system, user)


@dataclass
class Row:
    key: str
    route_ok: bool = False
    tool_ok: bool = False
    facts_ok: bool = True
    grounded_ok: bool = True
    refusal_ok: bool = True
    seconds: float = 0.0
    route: str = ""
    tool: str = ""
    answer: str = ""
    problem: str = ""

    @property
    def passed(self) -> bool:
        return self.route_ok and self.tool_ok and self.facts_ok and self.grounded_ok and self.refusal_ok and not self.problem


def evaluate(q: Q, tools: dict, llm, rag_ask) -> Row:
    rec = Recorder(llm)
    started = time.perf_counter()
    row = Row(q.key)
    try:
        ans = answer_question(q.text, None, get_tools=lambda: tools, rag_ask=rag_ask, llm=rec)
    except ChatUnavailable as exc:
        row.problem = f"unavailable: {exc}"
        return row
    row.seconds = round(time.perf_counter() - started, 1)
    row.route, row.tool, row.answer = ans.route, ans.tool or "", ans.reply
    row.route_ok = ans.route == q.route or ans.route in q.also_ok
    row.tool_ok = (ans.tool in q.tools) if q.tools else (ans.tool is None)
    row.facts_ok = all(contains_fact(ans.reply, f) for f in q.must)
    row.grounded_ok = not (ans.tool and rec.data_text and ungrounded(ans.reply, rec.data_text, q.text))
    row.refusal_ok = bool(REFUSAL.search(ans.reply)) if q.refusal else True
    if not row.grounded_ok:
        row.problem = f"numbers not in the data: {ungrounded(ans.reply, rec.data_text, q.text)}"
    return row


def run_eval(app, llm, rag_ask, with_rag: bool) -> list[Row]:
    with app.app_context():
        tools = build_tools(ctx())
        return [evaluate(q, tools, llm, rag_ask) for q in build_questions(tools) if with_rag or not q.needs_rag]


def report(rows: list[Row]) -> str:
    lines = [f"{'question':<14}{'pass':<6}{'route':<11}{'tool':<26}{'sec':>6}  notes"]
    for r in rows:
        notes = [n for n, ok in (("wrong route", r.route_ok), ("wrong tool", r.tool_ok), ("missing facts", r.facts_ok),
                                 ("no refusal", r.refusal_ok)) if not ok] + ([r.problem] if r.problem else [])
        lines.append(f"{r.key:<14}{'PASS' if r.passed else 'FAIL':<6}{r.route:<11}{r.tool:<26}{r.seconds:>6}  {'; '.join(notes)}")
    done = sum(r.passed for r in rows)
    lines.append(f"\n{done}/{len(rows)} passed   |   grounded: {sum(r.grounded_ok for r in rows)}/{len(rows)}   |   "
                 f"avg {sum(r.seconds for r in rows) / max(len(rows), 1):.1f}s per question")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-rag", action="store_true", help="also run the document and mixed questions")
    ap.add_argument("--out", default="tool_eval_results.csv")
    args = ap.parse_args()

    from conftest import SAMPLES, make_app, upload                              # reuse the test helpers
    from churn_api.api import chat as chat_mod
    app = make_app(Path(tempfile.mkdtemp()))
    assert upload(app.test_client(), SAMPLES / "churn_sample_dirty.csv", "eval").status_code == 201
    no_rag = lambda q: {"answer": "This information is not available in the project knowledge base.", "sources": []}  # noqa: E731
    rows = run_eval(app, chat_mod._llm, chat_mod._rag_ask if args.with_rag else no_rag, args.with_rag)

    print(report(rows))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__))
        w.writeheader()
        w.writerows({k: getattr(r, k) for k in Row.__dataclass_fields__} for r in rows)
    print(f"\nFull answers written to {args.out}")
    return 0 if all(r.passed for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())