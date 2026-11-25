from __future__ import annotations

from dataclasses import dataclass

from flask import current_app

from ..config import Config
from ..errors import ApiError
from ..ml.bundle import ModelBundle
from ..ml.pipeline import ChurnPipeline
from ..storage.datasets import DatasetStore
from ..storage.history import HistoryStore


@dataclass
class Ctx:
    cfg: Config
    store: DatasetStore
    history: HistoryStore
    bundle: ModelBundle | None = None
    pipeline: ChurnPipeline | None = None
    startup_error: str | None = None
    db: object | None = None             # SQLAlchemy engine when DATABASE_URL is set
    db_error: str | None = None          # why the database was not usable at startup (password-free)

    def ensure_database(self) -> str | None:
        """None when the database is fine or not configured; otherwise a short reason.
        A failed startup is retried here, so a database that comes up after the API heals without a restart."""
        if not self.cfg.DATABASE_URL:
            return None
        from ..storage import db
        try:
            if self.db_error is not None:
                db.init_schema(self.db)
                self.db_error = None
            db.ping(self.db)
        except Exception as exc:  # noqa: BLE001
            self.db_error = db.describe_error(exc, self.cfg.DATABASE_URL)
        return self.db_error


def ctx() -> Ctx:
    return current_app.extensions["churn"]


def require_model() -> Ctx:
    c = ctx()
    if c.pipeline is None:
        raise ApiError(503, "model_unavailable", "The model is not loaded. See /ready and the server log.")
    return c