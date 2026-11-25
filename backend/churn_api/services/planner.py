"""Step 2 of the tool-using assistant: turn a data question into ONE validated tool call.

The LLM is asked to reply with a single JSON object, {"tool": "<name>", "arguments": {...}}. We never trust it:
the reply is parsed leniently (code fences, extra prose), then checked against the tool's own JSON Schema, and
anything we cannot verify is rejected. Nothing is executed here -- that is Step 3.

The LLM is injected as a plain function `llm(system_prompt, user_prompt) -> str`, so this module has no
dependency on the RAG package and is tested with a fake LLM.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)

# predict_what_if needs ALL ten raw customer fields. A small LLM would invent them, and invented inputs
# produce confident, wrong scores, so the planner is not allowed to choose it (handled in a later step).
EXCLUDED_TOOLS = frozenset({"predict_what_if"})
HIDDEN_ARGS = frozenset({"dataset_id"})          # the app supplies this; the LLM must not guess it


@dataclass
class Plan:
    tool: str | None = None                      # None = "no tool needed" (when error is also None)
    arguments: dict = field(default_factory=dict)
    error: str | None = None                     # set when the LLM reply could not be turned into a safe call
    raw: str = ""                                # the LLM's reply, kept for logging

    @property
    def ok(self) -> bool:
        return self.tool is not None and self.error is None


# ------------------------------------------------------------------------------------------ prompt
EXAMPLES = [
    ("Which customers in Germany are the highest risk?",
     {"tool": "list_top_risk_customers", "arguments": {"geography": "Germany", "tier": "High"}}),
    ("Compare churn risk by age band",
     {"tool": "get_segment_breakdown", "arguments": {"by": "AgeBand"}}),
    ("Why is customer 15634602 likely to leave?",
     {"tool": "get_customer", "arguments": {"customer_id": "15634602"}}),
    ("How many customers are high risk?",
     {"tool": "get_portfolio_summary", "arguments": {}}),
    ("What is the weather today?",
     {"tool": None, "arguments": {}}),
]


def _arg_text(name: str, spec: dict, required: bool) -> str:
    if "enum" in spec:
        kind = "|".join(str(v) for v in spec["enum"])
    else:
        kind = spec.get("type", "string")
    return f"{name}{'*' if required else ''}: {kind}"


def _tool_lines(tools: dict) -> str:
    lines = []
    for name, tool in tools.items():
        if name in EXCLUDED_TOOLS:
            continue
        schema = tool.input_schema
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        args = ", ".join(_arg_text(n, s, n in required) for n, s in props.items() if n not in HIDDEN_ARGS)
        lines.append(f"- {name}({args}) - {tool.description}")
    return "\n".join(lines)


def build_planner_prompt(question: str, tools: dict) -> tuple[str, str]:
    system = (
        "You choose which data tool answers a question about an uploaded bank-customer churn dataset.\n"
        "Reply with ONE JSON object and nothing else: {\"tool\": \"<name>\", \"arguments\": {...}}\n"
        "Rules:\n"
        "- Use only the tools listed below. An argument marked * is required.\n"
        "- For arguments with a list of values (a|b|c), use exactly one of those values.\n"
        "- Copy a customer id exactly as written in the question. Never invent one.\n"
        "- Leave out arguments the question does not mention.\n"
        "- If no tool can answer the question, reply {\"tool\": null, \"arguments\": {}}.\n\n"
        f"Tools:\n{_tool_lines(tools)}\n\n"
        "Examples:\n" + "\n".join(f"Question: {q}\nJSON: {json.dumps(a)}" for q, a in EXAMPLES)
    )
    return system, f"Question: {question.strip()}\nJSON:"


# ------------------------------------------------------------------------------------------ parsing
def extract_json(text: str) -> dict | None:
    """First JSON object found in `text`, tolerating ```json fences and chatter around it."""
    text = re.sub(r"```(?:json)?", "", text or "")
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _clean_args(tool, raw_args: dict, question: str) -> tuple[dict, str | None]:
    """Validate/normalise arguments against the tool's JSON Schema. Returns (args, error)."""
    schema = tool.input_schema
    props = schema.get("properties", {})
    out: dict = {}
    for name, value in raw_args.items():
        spec = props.get(name)
        if spec is None or name in HIDDEN_ARGS or value in (None, ""):
            continue                                                   # unknown / hidden / empty -> drop
        if "enum" in spec:                                             # accept "germany" for "Germany"
            match = next((e for e in spec["enum"] if str(e).lower() == str(value).strip().lower()), None)
            if match is None:
                return {}, f"'{value}' is not a valid value for '{name}' (allowed: {spec['enum']})."
            out[name] = match
        elif spec.get("type") == "integer":
            try:
                out[name] = int(value)
            except (TypeError, ValueError):
                return {}, f"'{name}' must be a whole number."
        elif spec.get("type") == "boolean":
            if isinstance(value, bool):
                out[name] = value
            elif str(value).lower() in ("true", "false"):
                out[name] = str(value).lower() == "true"
            else:
                return {}, f"'{name}' must be true or false."
        elif spec.get("type") == "string":
            out[name] = str(value).strip()
        else:
            return {}, f"'{name}' has an unsupported type."
    if "customer_id" in out and out["customer_id"] not in question:    # guard against an invented id
        return {}, f"customer_id '{out['customer_id']}' does not appear in the question."
    missing = [r for r in schema.get("required", []) if r not in out]
    if missing:
        return {}, f"Missing required argument(s): {missing}."
    return out, None


# ------------------------------------------------------------------------------------------ main entry
# Phrases whose meaning is certain: the person wants the highest-risk CUSTOMERS. A 3B model sometimes answers these
# with a segment table instead, so they are planned by rule (instant, and no language-model call).
# Kept in step with router.py; tests/test_planner.py checks that both agree.
_TOP_CHURNERS = r"\b(top|worst|riskiest|most likely)\b.{0,25}\b(churners?|leavers?)\b"
_WHO_CHURNS = r"\bwho\b.{0,30}\b(churn|churns|leave|leaves|leaving|at risk|likely)\b"
_PLACES = {"germany": "Germany", "german": "Germany", "france": "France", "french": "France",
           "spain": "Spain", "spanish": "Spain"}


def known_intent(question: str):
    """(tool, arguments) for a few unambiguous question shapes, otherwise None."""
    q = (question or "").lower()
    if re.search(_TOP_CHURNERS, q):
        args: dict = {}
    elif re.search(_WHO_CHURNS, q):
        args = {"tier": "High"}
    else:
        return None
    for word, country in _PLACES.items():
        if re.search(r"\b" + word + r"\b", q):
            args["geography"] = country
    return "list_top_risk_customers", args


def plan(question: str, tools: dict, llm: Callable[[str, str], str]) -> Plan:
    intent = known_intent(question)
    if intent and intent[0] in tools:                                  # certain meaning: no language-model call needed
        clean, err = _clean_args(tools[intent[0]], intent[1], question)
        if err is None:
            return Plan(tool=intent[0], arguments=clean, raw="(planned by rule)")
    system, user = build_planner_prompt(question, tools)
    try:
        raw = llm(system, user)
    except Exception as exc:                                           # LLM down, timeout, bad key...
        log.warning("planner_llm_failed", extra={"error": str(exc)})
        return Plan(error="planner_llm_failed")

    obj = extract_json(raw)
    if obj is None:
        return Plan(error="no_json_in_reply", raw=raw)
    name = obj.get("tool")
    if name is None:
        return Plan(raw=raw)                                           # valid: no tool fits
    if name in EXCLUDED_TOOLS or name not in tools:
        return Plan(error=f"unknown_tool:{name}", raw=raw)
    args = obj.get("arguments")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return Plan(error="arguments_not_an_object", raw=raw)
    clean, err = _clean_args(tools[name], args, question)
    if err:
        return Plan(error=err, raw=raw)
    return Plan(tool=name, arguments=clean, raw=raw)