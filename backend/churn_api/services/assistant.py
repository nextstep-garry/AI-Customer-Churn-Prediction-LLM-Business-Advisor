"""Step 4 of the tool-using assistant: one entry point that answers a chat question from the right source.

    KNOWLEDGE  documents only (RAG), exactly as before tools existed
    DATA       router -> planner -> tool -> writer; if that path fails technically, fall back to the documents
    BOTH       the data answer and the document answer, labelled and combined

Every dependency is injected (get_tools, rag_ask, llm), so this module is tested without Flask, Ollama or ChromaDB.
It never invents an answer to hide a failure: it either answers from a real source, explains what is missing,
or raises ChatUnavailable when nothing could run at all.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from .planner import plan as make_plan
from .router import BOTH, DATA, KNOWLEDGE, route_question
from .tool_answer import run_plan, write_answer

log = logging.getLogger(__name__)

DATA_HINT = ("I could not match that to a data lookup. Try asking about a segment (for example "
             "\"Compare risk by country\"), the highest-risk customers, a customer ID, or the portfolio summary.")
DEFAULT_UNAVAILABLE = "The assistant could not answer right now."


class ChatUnavailable(Exception):
    """Nothing could answer (e.g. the LLM is down). The message is safe to show to the user."""


@dataclass
class ChatAnswer:
    reply: str
    sources: list = field(default_factory=list)
    route: str = KNOWLEDGE
    tool: str | None = None


@dataclass
class _Data:                                   # outcome of the tool path
    status: str                                # "answered" | "message" (safe text to show) | "failed"
    text: str = ""
    source: str | None = None
    tool: str | None = None


@dataclass
class _Rag:                                    # outcome of the document path
    text: str = ""
    sources: list = field(default_factory=list)
    error: str | None = None

    @property
    def answered(self) -> bool:                # the RAG pipeline gives a refusal no sources
        return self.error is None and bool(self.sources)


def _ask_rag(question: str, rag_ask: Callable[[str], dict], strict: bool) -> _Rag:
    try:
        res = rag_ask(question)
    except Exception as exc:
        if strict:
            raise                              # the caller maps it to an HTTP error
        log.warning("rag_failed_in_hybrid_route", extra={"detail": str(exc)})
        return _Rag(error=str(exc))
    if res.get("error"):
        log.error("rag_failed", extra={"detail": res["error"]})
        return _Rag(text=res.get("answer") or DEFAULT_UNAVAILABLE, error=res["error"])
    return _Rag(text=res["answer"], sources=res.get("sources", []))


def _data_path(question: str, dataset_id: str | None, get_tools: Callable[[], dict],
               llm: Callable[[str, str], str]) -> _Data:
    started = time.perf_counter()
    try:
        tools = get_tools()
    except Exception as exc:                   # e.g. the churn model is not loaded
        log.warning("tools_unavailable", extra={"detail": str(exc)})
        return _Data("failed")

    plan = make_plan(question, tools, llm)
    if not plan.ok:
        log.info("planner_no_tool", extra={"detail": plan.error or "model said no tool fits"})
        return _Data("failed")

    result = run_plan(plan, tools, dataset_id)
    if not result.ok:                          # a ToolError message, safe to show (no dataset, customer not found...)
        return _Data("message", result.error, tool=result.tool)

    try:
        text = write_answer(question, result, llm)
    except Exception as exc:
        log.warning("writer_failed", extra={"detail": str(exc)})
        return _Data("failed")
    if not text:
        return _Data("failed")
    log.info("data_path_ok", extra={"tool": result.tool, "ms": round((time.perf_counter() - started) * 1000)})
    return _Data("answered", text, result.source, result.tool)


def answer_question(question: str, dataset_id: str | None, *, get_tools: Callable[[], dict],
                    rag_ask: Callable[[str], dict], llm: Callable[[str, str], str]) -> ChatAnswer:
    routed = route_question(question)
    log.info("chat_routed", extra={"route": routed.route, "why": routed.reasons})

    # ---- documents only: unchanged behaviour ---------------------------------------------------------
    if routed.route == KNOWLEDGE:
        rag = _ask_rag(question, rag_ask, strict=True)
        if rag.error:
            raise ChatUnavailable(rag.text)
        return ChatAnswer(rag.text, rag.sources, KNOWLEDGE)

    data = _data_path(question, dataset_id, get_tools, llm)

    # ---- live data ---------------------------------------------------------------------------------------
    if routed.route == DATA:
        if data.status == "answered":
            return ChatAnswer(data.text, [data.source], DATA, data.tool)
        if data.status == "message":
            return ChatAnswer(data.text, [], DATA, data.tool)
        rag = _ask_rag(question, rag_ask, strict=False)          # tool path failed: maybe the documents know
        if rag.answered:
            return ChatAnswer(rag.text, rag.sources, KNOWLEDGE)
        if rag.error:
            raise ChatUnavailable(rag.text or DEFAULT_UNAVAILABLE)
        return ChatAnswer(DATA_HINT, [], DATA)

    # ---- both ----------------------------------------------------------------------------------------------
    rag = _ask_rag(question, rag_ask, strict=False)
    if data.status == "answered" and rag.answered:
        reply = f"**From your uploaded data**\n{data.text}\n\n**From the project documents**\n{rag.text}"
        return ChatAnswer(reply, [data.source, *rag.sources], BOTH, data.tool)
    if data.status == "answered":
        return ChatAnswer(data.text, [data.source], BOTH, data.tool)
    if rag.answered:
        note = f"\n\n_Note: {data.text}_" if data.status == "message" else ""
        return ChatAnswer(rag.text + note, rag.sources, KNOWLEDGE)
    if data.status == "message":
        return ChatAnswer(data.text, [], BOTH, data.tool)
    if rag.error:
        raise ChatUnavailable(rag.text or DEFAULT_UNAVAILABLE)
    return ChatAnswer(DATA_HINT, [], BOTH)