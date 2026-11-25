"""Churn prediction API (Flask). Application factory: `create_app()`."""
from __future__ import annotations

import logging
import re
import time
import uuid

from flask import Flask, g, request

from .api import ops, v1
from .api.deps import Ctx
from .config import Config
from .errors import error_response, register_error_handlers
from .logging_config import configure_logging
from .ml.bundle import ModelBundle
from .ml.exceptions import DatasetError
from .ml.pipeline import ChurnPipeline
from .security import init_security
from .storage.datasets import DatasetStore
from .storage.history import HistoryStore
from .utils import ChurnJSONProvider

log = logging.getLogger(__name__)
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def create_app(config: Config | None = None, *, bundle: ModelBundle | None = None) -> Flask:
    cfg = config or Config.from_env()
    configure_logging(cfg.LOG_LEVEL, cfg.LOG_JSON)

    app = Flask(__name__)
    app.json = ChurnJSONProvider(app)
    app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_UPLOAD_MB * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False

    engine = None
    if cfg.DATABASE_URL:  # PostgreSQL: the data lives in the database; a database that is down must not take the API down
        from .storage import db  # imported only when used, so the file-store setup needs no SQLAlchemy
        from .storage.pg_stores import PostgresDatasetStore, PostgresHistoryStore
        engine = db.make_engine(cfg.DATABASE_URL)
        store, history = PostgresDatasetStore(engine), PostgresHistoryStore(engine)
    else:
        store, history = DatasetStore(cfg.DATA_DIR), HistoryStore(cfg.DATA_DIR)
    ctx = Ctx(cfg=cfg, store=store, history=history, db=engine)
    if engine is not None:
        try:
            db.init_schema(engine)
        except Exception as exc:  # noqa: BLE001
            ctx.db_error = db.describe_error(exc, cfg.DATABASE_URL)
            log.error("database_unavailable", extra={"error": ctx.db_error})

    try:  # a model that fails to load must not take the whole API down: /ready reports it
        ctx.bundle = bundle or ModelBundle.load(cfg.MODEL_DIR, strict_versions=cfg.STRICT_VERSION_CHECK)
        ctx.pipeline = ChurnPipeline(ctx.bundle)
    except Exception as exc:  # noqa: BLE001
        ctx.startup_error = str(exc)
        log.error("model_load_failed", extra={"error": str(exc)})
    app.extensions["churn"] = ctx
    if not cfg.API_KEY:
        log.warning("api_key_not_set: /v1 endpoints are open. Set API_KEY before exposing this service.")

    @app.before_request
    def _request_context():
        rid = request.headers.get("X-Request-ID", "")
        g.request_id = rid if _SAFE_ID.match(rid) else uuid.uuid4().hex[:16]
        g.t0 = time.perf_counter()

    init_security(app, cfg.API_KEY, cfg.RATE_LIMIT_PER_MIN)

    @app.before_request
    def _database_gate():
        """While the database is down (or was at startup) answer 503, and retry the setup so it recovers by itself."""
        if ctx.db_error is not None and request.path not in ("/health", "/ready"):
            if ctx.ensure_database():
                return error_response(503, "database_unavailable", "The database is unavailable. Try again shortly.")

    @app.after_request
    def _log_request(resp):
        resp.headers["X-Request-ID"] = g.get("request_id", "")
        ms = round((time.perf_counter() - g.get("t0", time.perf_counter())) * 1000, 1)
        level = logging.DEBUG if request.path in ("/health", "/ready") else logging.INFO
        log.log(level, "request", extra={"method": request.method, "path": request.path,
                                         "status": resp.status_code, "ms": ms})
        return resp

    register_error_handlers(app)

    @app.errorhandler(DatasetError)
    def _dataset_error(exc):
        return error_response(400, "invalid_dataset", str(exc))

    app.register_blueprint(ops)
    app.register_blueprint(v1)
    return app