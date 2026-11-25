"""Tool registry: the structured capabilities a chatbot (or any agent) may call.

The Phase-4 assistant needs no new analytics code: an LLM picks a tool from this list, we run the
handler, and it phrases the result. The same functions back the dashboards, so chat answers and
charts can never disagree. Schemas are JSON Schema, i.e. directly usable as `tools=[...]` for
Anthropic / OpenAI style function calling (`input_schema` / `parameters`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import ValidationError

from ..errors import validation_details
from ..schemas import CustomerIn
from ..storage.datasets import DatasetNotFound
from ..utils import frame_to_records
from . import analytics, privacy, queries


class ToolError(ValueError):
    def __init__(self, message: str, details=None):
        super().__init__(message)
        self.details = details


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], dict]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


_DS = {"type": "string", "description": "Dataset id. Omit to use the most recently uploaded dataset."}


def build_tools(c) -> dict[str, Tool]:
    """`c` is the app context (api.deps.Ctx): store + pipeline."""

    def resolve(args: dict) -> str:
        ds = args.get("dataset_id") or c.store.latest_id()
        if not ds:
            raise ToolError("No dataset has been uploaded yet.")
        try:
            c.store.meta(ds)
        except DatasetNotFound:
            raise ToolError(f"Dataset '{ds}' not found.") from None
        return ds

    def list_datasets(args):
        return {"datasets": c.store.list()}

    def portfolio_summary(args):
        s = c.store.summary(resolve(args))
        return {"kpis": s["kpis"], "risk_tiers": s["risk_tiers"], "model": s["model"],
                "top_drivers": s["drivers"][:5], "evaluation": s["evaluation"], "drift": s["quality"]["drift"]}

    def segment_breakdown(args):
        by = args.get("by")
        if by not in analytics.SEGMENT_FIELDS:
            raise ToolError(f"'by' must be one of {analytics.SEGMENT_FIELDS}.")
        return {"by": by, "segments": queries.segment_breakdown(c.store, resolve(args), by)}

    def top_risk(args):
        limit = max(1, min(int(args.get("limit", 10)), 50))
        total, rows = queries.query_customers(
            c.store, resolve(args), tier=args.get("tier"), geography=args.get("geography"),
            gender=args.get("gender"), limit=limit, search_surname=False)
        return {"total_matching": total, "customers": frame_to_records(privacy.strip_frame(rows))}

    def get_customer(args):
        ds = resolve(args)
        row = queries.find_customer(c.store, ds, args.get("customer_id", ""))
        if row is None:
            raise ToolError(f"Customer '{args.get('customer_id')}' not found in dataset {ds}.")
        out = {"customer": frame_to_records(privacy.strip_row(row).to_frame().T)[0]}
        if args.get("explain", True):
            out["explanation"] = c.pipeline.explain_customer(row)
        return out

    def what_if(args):
        try:
            rec = CustomerIn.model_validate(args.get("customer", {})).model_dump()
        except ValidationError as exc:
            raise ToolError("Invalid customer.", validation_details(exc)) from None
        return c.pipeline.predict_records([rec], explain=True)[0]

    S = "object"
    tools = [
        Tool("list_datasets", "List uploaded datasets (id, name, row counts, model version).",
             {"type": S, "properties": {}}, list_datasets),
        Tool("get_portfolio_summary",
             "KPIs for a scored dataset: customers scored, predicted churners, risk-tier counts, top churn "
             "drivers, model evaluation (if labels were uploaded) and data drift vs training.",
             {"type": S, "properties": {"dataset_id": _DS}}, portfolio_summary),
        Tool("get_segment_breakdown",
             "Flag rate, average score and high-risk counts per segment (e.g. by Geography or AgeBand).",
             {"type": S, "properties": {"dataset_id": _DS, "by": {"type": "string", "enum": analytics.SEGMENT_FIELDS}},
              "required": ["by"]}, segment_breakdown),
        Tool("list_top_risk_customers", "Highest-risk customers, optionally filtered by tier / country / gender.",
             {"type": S, "properties": {"dataset_id": _DS, "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                                        "tier": {"type": "string", "enum": ["Low", "Medium", "High"]},
                                        "geography": {"type": "string", "enum": ["France", "Germany", "Spain"]},
                                        "gender": {"type": "string", "enum": ["Female", "Male"]}}}, top_risk),
        Tool("get_customer", "One customer's record, score, risk tier and the factors driving the score.",
             {"type": S, "properties": {"dataset_id": _DS, "customer_id": {"type": "string"},
                                        "explain": {"type": "boolean"}}, "required": ["customer_id"]}, get_customer),
        Tool("predict_what_if", "Score a hypothetical customer (all 10 raw fields) and explain the result.",
             {"type": S, "properties": {"customer": {"type": S, "description": "The 10 raw customer fields; see GET /v1/schema."}},
              "required": ["customer"]}, what_if),
    ]
    return {t.name: t for t in tools}