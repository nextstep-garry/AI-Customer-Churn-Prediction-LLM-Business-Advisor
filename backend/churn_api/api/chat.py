"""Chatbot contract (Phase 4).

Working now:  GET  /v1/chat/tools        the tool catalogue (JSON Schema)
              POST /v1/chat/tools/call   execute one tool -- exactly what the LLM loop will do
Working now:  POST /v1/chat              RAG answer about the project, from the ../rag knowledge base

The RAG code lives in <repo>/rag/ and is imported lazily on the first chat request, so the API still
starts (and /ready stays green) if its heavy dependencies are not installed.
"""
import logging
import sys
from pathlib import Path

from flask import jsonify, request

from ..errors import ApiError
from ..services.assistant import ChatUnavailable, answer_question
from ..services.tools import ToolError, build_tools

from . import v1
from .deps import ctx, require_model


@v1.get("/chat/tools")
def list_tools():
    return jsonify({"tools": [t.spec() for t in build_tools(ctx()).values()]})


@v1.post("/chat/tools/call")
def call_tool():
    c = require_model()
    body = request.get_json(silent=True) or {}
    tools = build_tools(c)
    tool = tools.get(body.get("name", ""))
    if tool is None:
        raise ApiError(404, "unknown_tool", f"Unknown tool. Available: {sorted(tools)}")
    try:
        return jsonify({"tool": tool.name, "result": tool.handler(body.get("arguments") or {})})
    except ToolError as exc:
        raise ApiError(400, "tool_error", str(exc), exc.details) from None


log = logging.getLogger(__name__)
RAG_DIR = Path(__file__).resolve().parents[3] / "rag"   # churn_prediction_app/rag
MAX_QUESTION_CHARS = 1000


# def _rag_ask(question: str) -> dict:
#     """Run the RAG pipeline. Imported lazily: loading torch/chromadb is slow and optional."""
#     if str(RAG_DIR) not in sys.path:
#         sys.path.insert(0, str(RAG_DIR))          # the rag modules use flat imports (import config, ...)
#     try:
#         from rag_pipeline import ask
#     except ImportError as exc:
#         log.error("rag_import_failed", extra={"error": str(exc)})
#         raise ApiError(503, "chat_unavailable",
#                        "The assistant's dependencies are not installed. Run: pip install -r rag/requirements.txt") from None
#     return ask(question)

def _ensure_rag_on_path() -> None:
    if str(RAG_DIR) not in sys.path:
        sys.path.insert(0, str(RAG_DIR))          # the rag modules use flat imports (import config, ...)


def _rag_ask(question: str) -> dict:
    """Run the RAG pipeline. Imported lazily: loading torch/chromadb is slow and optional."""
    _ensure_rag_on_path()
    try:
        from rag_pipeline import ask
    except ImportError as exc:
        log.error("rag_import_failed", extra={"error": str(exc)})
        raise ApiError(503, "chat_unavailable",
                       "The assistant's dependencies are not installed. Run: pip install -r rag/requirements.txt") from None
    return ask(question)


def _llm(system_prompt: str, user_prompt: str) -> str:
    """Send a prompt to the configured LLM (same provider settings as RAG). Used by the tool planner and writer."""
    _ensure_rag_on_path()
    from generate import generate_text
    return generate_text(system_prompt, user_prompt)


def _tools() -> dict:
    return build_tools(require_model())          # raises 503 if the churn model is not loaded

@v1.post("/chat")
def chat():
    body = request.get_json(silent=True) or {}
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise ApiError(400, "bad_request", "'messages' must be a list of {role, content}.")
    question = next((str(m.get("content", "")).strip() for m in reversed(messages)
                     if isinstance(m, dict) and m.get("role") == "user"), "")
    if not question:
        raise ApiError(400, "bad_request", "No user message found in 'messages'.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ApiError(400, "bad_request", f"Question is too long (max {MAX_QUESTION_CHARS} characters).")

    # result = _rag_ask(question)
    # if result.get("error"):                       # internal detail goes to the log, not to the client
    #     log.error("rag_failed", extra={"error": result["error"]})
    #     raise ApiError(503, "chat_unavailable", result.get("answer") or "The assistant could not answer right now.")
    # return jsonify({"reply": result["answer"], "sources": result.get("sources", [])})
    dataset_id = body.get("dataset_id")
    if dataset_id is not None and not isinstance(dataset_id, str):
        raise ApiError(400, "bad_request", "'dataset_id' must be a string.")

    try:
        ans = answer_question(question, dataset_id or None, get_tools=_tools, rag_ask=_rag_ask, llm=_llm)
    except ChatUnavailable as exc:                # internal detail was already logged; this text is user-safe
        raise ApiError(503, "chat_unavailable", str(exc)) from None
    return jsonify({"reply": ans.reply, "sources": ans.sources, "route": ans.route})