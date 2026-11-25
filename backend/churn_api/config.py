"""Environment-driven configuration. Nothing is hard-coded; see .env.example."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # churn_prediction_app/ (this file: backend/churn_api/config.py)


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw in (None, "") else int(raw)


@dataclass(frozen=True)
class Config:
    HOST: str = "127.0.0.1"
    PORT: int = 5000
    DEBUG: bool = False                 # the Werkzeug debugger is opt-in (FLASK_DEBUG=1)
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True
    MODEL_DIR: Path = REPO_ROOT / "models"
    DATA_DIR: Path = REPO_ROOT / "data"  # uploads, results and prediction history live here
    API_KEY: str = ""                    # empty = open (dev only); set it for anything shared
    DATABASE_URL: str = ""               # postgresql+psycopg://user:pass@host:5432/db ; empty = file store under DATA_DIR
    ALLOW_SURNAME_ACCESS: bool = True    # False = the API never returns surnames, even when a request asks for them
    MAX_UPLOAD_MB: int = 50
    MAX_ROWS: int = 200_000
    MAX_BATCH_RECORDS: int = 5_000
    RATE_LIMIT_PER_MIN: int = 240        # per API key (or client IP); 0 disables
    STRICT_VERSION_CHECK: bool = True    # refuse to start if scikit-learn/xgboost differ from training

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        cfg = cls(
            HOST=os.getenv("HOST", cls.HOST),
            PORT=_int("PORT", cls.PORT),
            DEBUG=_bool("FLASK_DEBUG", cls.DEBUG),
            LOG_LEVEL=os.getenv("LOG_LEVEL", cls.LOG_LEVEL).upper(),
            LOG_JSON=_bool("LOG_JSON", cls.LOG_JSON),
            MODEL_DIR=Path(os.getenv("MODEL_DIR", str(cls.MODEL_DIR))),
            DATA_DIR=Path(os.getenv("DATA_DIR", str(cls.DATA_DIR))),
            API_KEY=os.getenv("API_KEY", cls.API_KEY),
            DATABASE_URL=os.getenv("DATABASE_URL", cls.DATABASE_URL).strip(),
            ALLOW_SURNAME_ACCESS=_bool("ALLOW_SURNAME_ACCESS", cls.ALLOW_SURNAME_ACCESS),
            MAX_UPLOAD_MB=_int("MAX_UPLOAD_MB", cls.MAX_UPLOAD_MB),
            MAX_ROWS=_int("MAX_ROWS", cls.MAX_ROWS),
            MAX_BATCH_RECORDS=_int("MAX_BATCH_RECORDS", cls.MAX_BATCH_RECORDS),
            RATE_LIMIT_PER_MIN=_int("RATE_LIMIT_PER_MIN", cls.RATE_LIMIT_PER_MIN),
            STRICT_VERSION_CHECK=not _bool("ALLOW_VERSION_MISMATCH", False),
        )
        return replace(cfg, **overrides) if overrides else cfg