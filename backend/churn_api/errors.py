"""One JSON error envelope for every failure: {"error": {code, message, details}, "request_id"}.

Internal exception text is never returned to clients (audit finding 3.5); it goes to the log.
"""
from __future__ import annotations

import logging

from flask import Flask, g, jsonify
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException

log = logging.getLogger(__name__)

_CODES = {400: "bad_request", 401: "unauthorized", 404: "not_found", 405: "method_not_allowed",
          413: "payload_too_large", 415: "unsupported_media_type", 429: "rate_limited"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details=None, headers: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details
        self.headers = headers or {}


def error_response(status: int, code: str, message: str, details=None, headers: dict | None = None):
    body = {"error": {"code": code, "message": message}, "request_id": g.get("request_id")}
    if details:
        body["error"]["details"] = details
    return jsonify(body), status, (headers or {})


def validation_details(exc: ValidationError) -> list[dict]:
    """Field-level errors, without echoing the offending input back."""
    out = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "body"
        out.append({"field": loc, "message": err["msg"], "type": err["type"]})
    return out


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(ApiError)
    def _api_error(exc: ApiError):
        return error_response(exc.status, exc.code, exc.message, exc.details, exc.headers)

    @app.errorhandler(ValidationError)
    def _validation(exc: ValidationError):
        return error_response(400, "validation_error", "Request failed validation.", validation_details(exc))

    @app.errorhandler(HTTPException)
    def _http(exc: HTTPException):
        code = _CODES.get(exc.code, "http_error")
        msg = {413: "Upload is larger than the server allows."}.get(exc.code, exc.description or exc.name)
        return error_response(exc.code or 500, code, msg)

    try:  # PostgreSQL is optional: the handler exists only when SQLAlchemy is installed
        from sqlalchemy.exc import OperationalError
    except ImportError:
        pass
    else:
        @app.errorhandler(OperationalError)
        def _database_down(exc: OperationalError):
            log.error("database_error", extra={"error": str(getattr(exc, "orig", exc)).splitlines()[0][:200]})
            return error_response(503, "database_unavailable", "The database is unavailable. Try again shortly.")

    @app.errorhandler(Exception)
    def _unexpected(exc: Exception):
        log.exception("unhandled_exception")
        return error_response(500, "internal_error", "Internal server error. Quote the request_id when reporting it.")