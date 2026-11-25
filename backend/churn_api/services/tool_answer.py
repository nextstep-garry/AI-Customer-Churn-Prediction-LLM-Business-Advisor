"""Step 3 of the tool-using assistant: run a validated plan, then have the LLM word the result.

    run_plan()      executes the tool chosen in Step 2 (never raises; errors come back inside ToolResult)
    compact()       shrinks the tool output so it fits a small LLM's context, removes personal data, and turns
                    rates into ready-made percentages so the LLM never has to do arithmetic
    write_answer()  asks the LLM to answer the question using ONLY that data

The LLM is injected as `llm(system_prompt, user_prompt) -> str`, like in planner.py.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

from .tools import ToolError

log = logging.getLogger(__name__)

PII_KEYS = frozenset({"Surname"})                                      # never sent to the LLM or shown
NOISE_KEYS = frozenset({"row_id", "row_status", "data_issues", "base_value", "margin", "method", "note", "basis"})
MAX_ITEMS = 10                                                         # list items kept per list
MAX_STRING = 160
MAX_DATA_CHARS = 4500                                                  # ~1,200 tokens: leaves room in a 4,096-token window


@dataclass
class ToolResult:
    tool: str | None = None
    arguments: dict = field(default_factory=dict)
    data: dict | None = None
    error: str | None = None                                           # a message that is safe to show the user

    @property
    def ok(self) -> bool:
        return self.error is None and self.data is not None

    @property
    def source(self) -> str:
        return f"data tool: {self.tool}"


# ------------------------------------------------------------------------------------------ executor
def run_plan(plan, tools: dict, dataset_id: str | None = None) -> ToolResult:
    """Execute `plan` (from planner.plan). `dataset_id` comes from the app, never from the LLM."""
    if not plan.ok:
        return ToolResult(error="No data tool was selected for this question.")
    args = dict(plan.arguments)
    if dataset_id:
        args["dataset_id"] = dataset_id
    try:
        data = tools[plan.tool].handler(args)
    except ToolError as exc:                                           # e.g. no dataset uploaded, customer not found
        return ToolResult(plan.tool, plan.arguments, error=str(exc))
    except ValueError as exc:                                          # analytics-level validation
        return ToolResult(plan.tool, plan.arguments, error=str(exc))
    except Exception:                                                  # a bug must not crash the chat
        log.exception("tool_failed", extra={"tool": plan.tool})
        return ToolResult(plan.tool, plan.arguments, error="The data tool failed unexpectedly.")
    return ToolResult(plan.tool, plan.arguments, data=data)


# ------------------------------------------------------------------------------------------ compaction
def compact(value, key: str | None = None):
    """Recursively shrink tool output for the LLM (see module docstring)."""
    if isinstance(value, dict):
        return {k: compact(v, k) for k, v in value.items() if k not in PII_KEYS and k not in NOISE_KEYS}
    if isinstance(value, list):
        items = [compact(v) for v in value[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            items.append({"_note": f"{len(value) - MAX_ITEMS} more items not shown"})
        return items
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if key and key.endswith(("rate", "share")) and 0 <= value <= 1:
            return f"{value * 100:.1f}%"                              # 0.3101 -> "31.0%"
        return round(value, 4)
    if isinstance(value, str) and len(value) > MAX_STRING:
        return value[:MAX_STRING] + "..."
    return value


def data_text(data: dict) -> str:
    text = json.dumps(compact(data), separators=(",", ":"), default=str)
    if len(text) > MAX_DATA_CHARS:
        text = text[:MAX_DATA_CHARS] + '..."(data truncated)'
    return text


# ------------------------------------------------------------------------------------------ answer writer
SYSTEM_PROMPT = """You answer questions about an uploaded bank-customer churn dataset.
You are given DATA: the exact output of an analytics tool. Answer using ONLY that data.

Rules:
- Copy numbers exactly as they appear in DATA. Do not calculate new numbers, estimate, or round further.
- If DATA does not contain what the question asks for (for example an email address), say that this information is not available in the data.
- Values written like "31.0%" are already percentages. A churn_score or avg_score is a raw risk score used for ranking, not a probability: never say "X% chance".
- Risk drivers show what pushes the model's score up or down. They are not proven causes of leaving.
- Do not recommend contacting a customer or making an offer. A human reviewer decides actions.
- Be concise: 2 to 5 sentences, or a short list when the question asks for several customers or segments.
- Start directly with the answer. Do not mention DATA, JSON or tools."""


def build_answer_prompt(question: str, result: ToolResult) -> tuple[str, str]:
    scope = ", ".join(f"{k}={v}" for k, v in result.arguments.items()) or "no filters"
    user = (f"Question: {question.strip()}\n\n"
            f"Tool: {result.tool} ({scope})\n"
            f"DATA:\n{data_text(result.data)}\n\n"
            "Answer:")
    return SYSTEM_PROMPT, user


def write_answer(question: str, result: ToolResult, llm: Callable[[str, str], str]) -> str:
    """Return the LLM's answer. Propagates the LLM's exceptions: the caller decides how to degrade."""
    if not result.ok:
        raise ValueError("write_answer needs a successful ToolResult.")
    system, user = build_answer_prompt(question, result)
    return llm(system, user).strip()