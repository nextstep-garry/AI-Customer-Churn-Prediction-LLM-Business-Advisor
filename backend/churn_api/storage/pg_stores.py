"""PostgreSQL versions of DatasetStore and HistoryStore.

Same methods and same return shapes as the file-based stores, so nothing above them changes: the API, the analytics and
the assistant's tools all keep calling store.results(), store.summary(), history.recent() ...  Pandas is still the working
format for the ML pipeline; this module only moves where its output is kept.

Every write that touches several tables is one transaction, so a crash never leaves a half-saved dataset. Reads are keyed on
the dataset's updated_at, so several workers always see a re-score straight away.
"""
from __future__ import annotations

import io
import math
import re
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sqlalchemy as sa

from . import schema as s
from .datasets import DatasetNotFound
from .pg_queries import PgQueries

_ID = re.compile(r"^[0-9a-f]{32}$")
RESULT_COLUMNS = [c.name for c in s.dataset_results.columns if c.name != "dataset_id"]
OPTIONAL_COLUMNS = ("CustomerId", "Surname", "raw_score", "Exited")               # absent from the frame when the upload lacked them
REJECTED_COLUMNS = ["row_id", "CustomerId", "Surname", "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
                    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "data_issues"]   # header when none
_META_COLUMNS = [c for c in s.datasets.columns if c.name != "summary"]   # never pull the big summary just to list datasets


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime, timespec: str = "microseconds") -> str:
    return value.astimezone(timezone.utc).isoformat(timespec=timespec)


def _clean_json(value):
    """Make a value valid JSON for PostgreSQL: numpy -> Python, NaN/inf -> null."""
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _plain(value):
    """One value as PostgreSQL wants it: missing (NaN, NA, NaT, None) -> None, numpy scalars -> Python."""
    if value is None or pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def _id_text(value) -> str | None:
    """An identifier as text: 15634602 and 15634602.0 both become '15634602'; anything else is kept as written."""
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def _records(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    """DataFrame -> list of dicts holding plain Python values. (df.where(mask, None) is NOT used: pandas turns the
    None back into NaN, which PostgreSQL rejects in a number column.)"""
    sub = df[[c for c in columns if c in df.columns]].astype(object)
    return [{k: _plain(v) for k, v in rec.items()} for rec in sub.to_dict("records")]


def _meta(row) -> dict:
    return {
        "id": row.id, "name": row.name, "original_filename": row.original_filename,
        "created_at": _iso(row.created_at), "updated_at": _iso(row.updated_at),
        "status": row.status, "policy": row.policy, "has_target": row.has_target,
        "model": {"name": row.model_name, "version": row.model_version, "score_type": row.score_type},
        "rows": {"uploaded": row.rows_uploaded, "scored": row.rows_scored, "rejected": row.rows_rejected,
                 "repaired": row.rows_repaired, "duplicates_removed": row.duplicates_removed},
    }


def _run_columns(run, model: dict) -> dict:
    st = run.stats
    return dict(status="completed", policy=st["policy"], has_target=bool(st["target_present"]),
                model_name=model["name"], model_version=model["version"], score_type=model["score_type"],
                rows_uploaded=st["rows_uploaded"], rows_scored=st["rows_valid"], rows_rejected=st["rows_rejected"],
                rows_repaired=st["rows_repaired"], duplicates_removed=st["duplicates_removed"],
                summary=_clean_json(run.summary))


class PostgresDatasetStore(PgQueries):
    def __init__(self, engine: sa.Engine, cache_size: int = 4):
        self.engine = engine
        self._cache: OrderedDict = OrderedDict()
        self._cache_size, self._lock = cache_size, threading.Lock()

    @staticmethod
    def _check(dataset_id: str) -> str:
        if not _ID.match(str(dataset_id)):
            raise DatasetNotFound(dataset_id)
        return dataset_id

    # ------------------------------------------------------------------ write
    @staticmethod
    def _write_run(conn, dataset_id: str, run) -> None:
        """Replace this dataset's result and rejected rows (the caller owns the transaction)."""
        extra = [c for c in run.results.columns if c not in RESULT_COLUMNS]
        if extra:
            raise RuntimeError(f"The pipeline produced columns the database schema does not have: {extra}. "
                               "Add them to storage/schema.py and bump SCHEMA_VERSION.")
        conn.execute(sa.delete(s.dataset_results).where(s.dataset_results.c.dataset_id == dataset_id))
        conn.execute(sa.delete(s.dataset_rejected).where(s.dataset_rejected.c.dataset_id == dataset_id))
        rows = _records(run.results, RESULT_COLUMNS)
        for r in rows:
            r["dataset_id"] = dataset_id
            r["CustomerId"] = _id_text(r.get("CustomerId"))
        if rows:
            conn.execute(sa.insert(s.dataset_results), rows)
        rejected = []
        for i, rec in enumerate(_records(run.rejected, list(run.rejected.columns))):
            rejected.append({"dataset_id": dataset_id, "row_id": rec.get("row_id", i),
                             "data_issues": rec.get("data_issues") or "", "record": _clean_json(rec)})
        if rejected:
            conn.execute(sa.insert(s.dataset_rejected), rejected)

    def save(self, *, name: str, original_filename: str, raw_bytes: bytes, run, model: dict) -> dict:
        dataset_id, now = uuid.uuid4().hex, _utcnow()
        with self.engine.begin() as conn:
            conn.execute(sa.insert(s.datasets).values(id=dataset_id, name=name, original_filename=original_filename,
                                                      created_at=now, updated_at=now, **_run_columns(run, model)))
            conn.execute(sa.insert(s.dataset_raw).values(dataset_id=dataset_id, csv=raw_bytes))
            self._write_run(conn, dataset_id, run)
        return self.meta(dataset_id)

    def replace_results(self, dataset_id: str, run, model: dict) -> dict:
        self._check(dataset_id)
        with self.engine.begin() as conn:
            found = conn.execute(sa.select(s.datasets.c.id).where(s.datasets.c.id == dataset_id).with_for_update()).first()
            if found is None:
                raise DatasetNotFound(dataset_id)
            self._write_run(conn, dataset_id, run)
            conn.execute(sa.update(s.datasets).where(s.datasets.c.id == dataset_id)
                         .values(updated_at=_utcnow(), **_run_columns(run, model)))
        return self.meta(dataset_id)

    def delete(self, dataset_id: str) -> None:
        self._check(dataset_id)
        with self.engine.begin() as conn:
            gone = conn.execute(sa.delete(s.datasets).where(s.datasets.c.id == dataset_id)).rowcount   # children cascade
        if not gone:
            raise DatasetNotFound(dataset_id)
        with self._lock:
            for k in [k for k in self._cache if k[0] == dataset_id]:
                del self._cache[k]

    # ------------------------------------------------------------------ read
    def meta(self, dataset_id: str) -> dict:
        self._check(dataset_id)
        with self.engine.connect() as conn:
            row = conn.execute(sa.select(*_META_COLUMNS).where(s.datasets.c.id == dataset_id)).first()
        if row is None:
            raise DatasetNotFound(dataset_id)
        return _meta(row)

    def list(self) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(sa.select(*_META_COLUMNS)
                                .order_by(s.datasets.c.created_at.desc(), s.datasets.c.id.desc())).all()
        return [_meta(r) for r in rows]

    def latest_id(self) -> str | None:
        with self.engine.connect() as conn:
            return conn.execute(sa.select(s.datasets.c.id)
                                .order_by(s.datasets.c.created_at.desc(), s.datasets.c.id.desc()).limit(1)).scalar()

    def summary(self, dataset_id: str) -> dict:
        self._check(dataset_id)
        with self.engine.connect() as conn:
            found = conn.execute(sa.select(s.datasets.c.summary).where(s.datasets.c.id == dataset_id)).first()
        if found is None:
            raise DatasetNotFound(dataset_id)
        return found[0]

    def raw_bytes(self, dataset_id: str) -> bytes:
        self._check(dataset_id)
        with self.engine.connect() as conn:
            found = conn.execute(sa.select(s.dataset_raw.c.csv).where(s.dataset_raw.c.dataset_id == dataset_id)).first()
        if found is None:
            raise DatasetNotFound(dataset_id)
        return bytes(found[0])

    def results(self, dataset_id: str) -> pd.DataFrame:
        """Scored rows. Treat as read-only: the frame is shared through a small LRU cache."""
        self._check(dataset_id)
        with self.engine.connect() as conn:
            head = conn.execute(sa.select(s.datasets.c.updated_at, s.datasets.c.has_target)
                                .where(s.datasets.c.id == dataset_id)).first()
            if head is None:
                raise DatasetNotFound(dataset_id)
            key = (dataset_id, head.updated_at)
            with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]
            cols = [s.dataset_results.c[n] for n in RESULT_COLUMNS]
            rows = conn.execute(sa.select(*cols).where(s.dataset_results.c.dataset_id == dataset_id)
                                .order_by(s.dataset_results.c.row_id)).mappings().all()
        df = pd.DataFrame([dict(r) for r in rows], columns=RESULT_COLUMNS)
        ids = df["CustomerId"]
        numeric = pd.to_numeric(ids, errors="coerce")
        if len(df) and numeric.notna().sum() == ids.notna().sum():     # every id is numeric: as read_csv would give
            df["CustomerId"] = numeric
        for name in OPTIONAL_COLUMNS:                                  # a column the upload never had is absent, as in the CSV
            if name in df.columns and df[name].isna().all() and not (name == "Exited" and head.has_target):
                df = df.drop(columns=name)
        with self._lock:
            self._cache[key] = df
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return df

    def rejected(self, dataset_id: str) -> pd.DataFrame:
        self._check(dataset_id)
        with self.engine.connect() as conn:
            if conn.execute(sa.select(s.datasets.c.id).where(s.datasets.c.id == dataset_id)).first() is None:
                raise DatasetNotFound(dataset_id)
            rows = conn.execute(sa.select(s.dataset_rejected.c.record).where(s.dataset_rejected.c.dataset_id == dataset_id)
                                .order_by(s.dataset_rejected.c.row_id)).all()
        if not rows:
            return pd.DataFrame(columns=REJECTED_COLUMNS)
        # The CSV store handed back whatever read_csv infers (a column of digits is numeric). Do the same, so both
        # stores give identical frames; rejected rows are few, so the round trip costs nothing.
        return pd.read_csv(io.StringIO(pd.DataFrame([r[0] for r in rows]).to_csv(index=False)))


class PostgresHistoryStore:
    """Audit trail of single-customer predictions."""

    def __init__(self, engine: sa.Engine):
        self.engine = engine

    def add(self, *, request_id: str | None, model_name: str, model_version: str, record: dict, result: dict) -> None:
        with self.engine.begin() as conn:
            conn.execute(sa.insert(s.predictions).values(
                request_id=request_id, model_name=model_name, model_version=model_version, input=_clean_json(record),
                score=float(result["churn_score"]), score_type=result["score_type"], risk_level=result["risk_level"],
                will_churn=bool(result["will_churn"])))

    def recent(self, limit: int = 20) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(sa.select(s.predictions).order_by(s.predictions.c.id.desc()).limit(limit)).mappings().all()
        return [{"id": r["id"], "ts": _iso(r["ts"], "seconds"), "request_id": r["request_id"],
                 "model_name": r["model_name"], "model_version": r["model_version"], "score": r["score"],
                 "score_type": r["score_type"], "risk_level": r["risk_level"], "will_churn": bool(r["will_churn"]),
                 "input": r["input"]} for r in rows]

    def clear(self) -> int:
        with self.engine.begin() as conn:
            return conn.execute(sa.delete(s.predictions)).rowcount