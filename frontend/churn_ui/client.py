"""Thin HTTP client for the Flask backend. All network access in the UI goes through here."""
from __future__ import annotations

from typing import Any

import requests

from . import config


class ApiError(Exception):
    """Any failed call. `details` holds field-level validation errors when the backend sent them."""

    def __init__(self, message: str, *, status: int = 0, code: str = "error", details=None, request_id=None):
        super().__init__(message)
        self.message, self.status, self.code = message, status, code
        self.details, self.request_id = details or [], request_id


class ApiClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: float | None = None, upload_timeout: float | None = None):
        # Settings are read at construction (not import) time, so tests and embedders can repoint the client.
        self.base_url = (base_url or config.BACKEND_URL).rstrip("/")
        self.timeout = timeout or config.REQUEST_TIMEOUT
        self.upload_timeout = upload_timeout or config.UPLOAD_TIMEOUT
        self.session = requests.Session()
        if api_key or config.API_KEY:
            self.session.headers["X-API-Key"] = api_key or config.API_KEY

    # ------------------------------------------------------------------ plumbing
    def _request(self, method: str, path: str, *, timeout: float | None = None, raw: bool = False, **kwargs) -> Any:
        try:
            resp = self.session.request(method, self.base_url + path, timeout=timeout or self.timeout, **kwargs)
        except requests.ConnectionError:
            raise ApiError(f"Cannot reach the backend at {self.base_url}. Is it running?", code="unreachable") from None
        except requests.Timeout:
            raise ApiError("The backend took too long to respond.", code="timeout") from None
        if resp.status_code >= 400:
            try:
                body = resp.json()
                err = body.get("error", {})
                raise ApiError(err.get("message", resp.reason), status=resp.status_code, code=err.get("code", "error"),
                               details=err.get("details"), request_id=body.get("request_id"))
            except ValueError:
                raise ApiError(f"Backend returned HTTP {resp.status_code}.", status=resp.status_code) from None
        if raw:
            return resp.content
        return resp.json() if resp.content else None

    # ------------------------------------------------------------------ meta
    def ready(self) -> dict:
        """Readiness. Returns {'status': 'ready'|'unavailable', ...}; never raises for a 503."""
        try:
            return self._request("GET", "/ready")
        except ApiError as exc:
            if exc.status == 503:
                return {"status": "unavailable", "detail": exc.message}
            raise

    def model(self) -> dict:
        return self._request("GET", "/v1/model")

    def schema(self) -> dict:
        return self._request("GET", "/v1/schema")

    # ------------------------------------------------------------------ single customer
    def predict(self, record: dict, explain: bool = True) -> dict:
        return self._request("POST", "/v1/predict", params={"explain": str(explain).lower()}, json=record)

    def history(self, limit: int = 10) -> list[dict]:
        return self._request("GET", "/v1/history", params={"limit": limit})["predictions"]

    def clear_history(self) -> int:
        return self._request("DELETE", "/v1/history")["deleted"]

    # ------------------------------------------------------------------ datasets
    def upload(self, filename: str, data: bytes, name: str, policy: str) -> dict:
        return self._request("POST", "/v1/datasets", timeout=self.upload_timeout,
                             files={"file": (filename, data, "text/csv")}, data={"name": name, "policy": policy})

    def datasets(self) -> list[dict]:
        return self._request("GET", "/v1/datasets")["datasets"]

    def summary(self, dataset_id: str) -> dict:
        return self._request("GET", f"/v1/datasets/{dataset_id}/summary")

    def customers(self, dataset_id: str, **params) -> dict:
        return self._request("GET", f"/v1/datasets/{dataset_id}/customers",
                             params={k: v for k, v in params.items() if v not in (None, "", "All")})

    # def customer(self, dataset_id: str, key) -> dict:
    #     return self._request("GET", f"/v1/datasets/{dataset_id}/customers/{key}")
    def customer(self, dataset_id: str, key, include_surname: bool = False) -> dict:
        return self._request("GET", f"/v1/datasets/{dataset_id}/customers/{key}",
                             params={"include_surname": "true"} if include_surname else None)

    def download(self, dataset_id: str, kind: str = "predictions") -> bytes:
        return self._request("GET", f"/v1/datasets/{dataset_id}/download", params={"kind": kind}, raw=True)

    def rescore(self, dataset_id: str) -> dict:
        return self._request("POST", f"/v1/datasets/{dataset_id}/rescore", timeout=self.upload_timeout)

    def delete_dataset(self, dataset_id: str) -> None:
        self._request("DELETE", f"/v1/datasets/{dataset_id}")

    # ------------------------------------------------------------------ assistant (Phase 4)
    def chat_tools(self) -> list[dict]:
        return self._request("GET", "/v1/chat/tools")["tools"]

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self._request("POST", "/v1/chat/tools/call", json={"name": name, "arguments": arguments})["result"]

    def chat(self, messages: list[dict], dataset_id: str | None) -> dict:
        # RAG answers can take 10-30 s on a local LLM (more on first load), so use the long timeout
        return self._request("POST", "/v1/chat", timeout=self.upload_timeout,
                             json={"messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                                   "dataset_id": dataset_id})
