"""One-time import of the file store (data/) into PostgreSQL.

    cd backend
    python -m churn_api.storage.import_files --dry-run      # show what would happen, change nothing
    python -m churn_api.storage.import_files                # do it

The target is DATABASE_URL; the source is --data-dir (default: DATA_DIR, else ./data). Each dataset keeps its id, name and
timestamps. After copying, the copy is compared with the original and removed again if anything differs. Safe to repeat:
datasets already in PostgreSQL are skipped. Nothing in the data folder is ever modified or deleted.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sqlalchemy as sa

from . import db
from . import schema as s
from .datasets import DatasetStore
from .history import HistoryStore
from .pg_stores import PostgresDatasetStore, _clean_json


@dataclass
class Report:
    imported: list = field(default_factory=list)          # dataset ids (in a dry run: the ones that would be imported)
    skipped: list = field(default_factory=list)           # already in PostgreSQL
    failed: list = field(default_factory=list)            # (dataset id, message)
    history_imported: int = 0
    history_skipped: int = 0

    @property
    def ok(self) -> bool:
        return not self.failed


def _parse_time(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _exists(engine, dataset_id: str) -> bool:
    with engine.connect() as conn:
        return conn.execute(sa.select(s.datasets.c.id).where(s.datasets.c.id == dataset_id)).first() is not None


def _remove(engine, dataset_id: str) -> None:
    with engine.begin() as conn:
        conn.execute(sa.delete(s.datasets).where(s.datasets.c.id == dataset_id))        # children cascade


def import_dataset(engine, files: DatasetStore, dataset_id: str) -> None:
    """Copy one dataset in a single transaction, keeping its id and timestamps."""
    meta = files.meta(dataset_id)
    results, rejected = files.results(dataset_id), files.rejected(dataset_id)
    summary, raw = _clean_json(files.summary(dataset_id)), files.raw_bytes(dataset_id)
    model, rows = meta["model"], meta["rows"]
    with engine.begin() as conn:
        conn.execute(sa.insert(s.datasets).values(
            id=dataset_id, name=meta["name"], original_filename=meta["original_filename"],
            status=meta.get("status", "completed"), policy=meta["policy"], has_target=bool(meta["has_target"]),
            model_name=model["name"], model_version=model["version"], score_type=model["score_type"],
            rows_uploaded=rows["uploaded"], rows_scored=rows["scored"], rows_rejected=rows["rejected"],
            rows_repaired=rows["repaired"], duplicates_removed=rows["duplicates_removed"], summary=summary,
            created_at=_parse_time(meta["created_at"]), updated_at=_parse_time(meta["updated_at"])))
        conn.execute(sa.insert(s.dataset_raw).values(dataset_id=dataset_id, csv=raw))
        PostgresDatasetStore._write_run(conn, dataset_id, SimpleNamespace(results=results, rejected=rejected))


def verify_dataset(pg: PostgresDatasetStore, files: DatasetStore, dataset_id: str) -> list[str]:
    """Compare the copy with the original. Returns a list of problems; empty means identical."""
    problems: list[str] = []
    f_meta, p_meta = files.meta(dataset_id), pg.meta(dataset_id)
    for key in ("name", "original_filename", "policy", "has_target", "rows", "model"):
        if f_meta.get(key) != p_meta.get(key):
            problems.append(f"metadata '{key}' differs")
    for key in ("created_at", "updated_at"):
        if _parse_time(f_meta[key]) != _parse_time(p_meta[key]):
            problems.append(f"metadata '{key}' differs")
    f, p = files.results(dataset_id), pg.results(dataset_id)
    if list(f.columns) != list(p.columns):
        problems.append("result columns differ")
    elif len(f) != len(p):
        problems.append(f"result rows differ ({len(f)} vs {len(p)})")
    else:
        if f["row_id"].tolist() != p["row_id"].tolist():
            problems.append("row ids differ")
        if not np.allclose(f["churn_score"].to_numpy(dtype=float), p["churn_score"].to_numpy(dtype=float),
                           rtol=1e-12, atol=0.0):
            problems.append("scores differ")
    if len(files.rejected(dataset_id)) != len(pg.rejected(dataset_id)):
        problems.append("rejected rows differ")
    if files.raw_bytes(dataset_id) != pg.raw_bytes(dataset_id):
        problems.append("the original CSV differs")
    return problems


def _history_exists(conn, row: dict, ts: datetime) -> bool:
    found = conn.execute(sa.select(s.predictions.c.id).where(
        s.predictions.c.ts == ts, s.predictions.c.score == float(row["score"]),
        sa.func.coalesce(s.predictions.c.request_id, "") == (row.get("request_id") or "")).limit(1)).first()
    return found is not None


def import_history(engine, history: HistoryStore, *, dry_run: bool = False) -> tuple[int, int]:
    """Copy the prediction history, oldest first so the newest-first order carries over. Returns (imported, skipped)."""
    rows = sorted(history.recent(1_000_000), key=lambda r: r["id"])
    imported = skipped = 0
    with engine.begin() as conn:
        for r in rows:
            ts = _parse_time(r["ts"])
            if _history_exists(conn, r, ts):
                skipped += 1
                continue
            imported += 1
            if not dry_run:
                conn.execute(sa.insert(s.predictions).values(
                    ts=ts, request_id=r.get("request_id"), model_name=r.get("model_name"),
                    model_version=r.get("model_version"), input=_clean_json(r["input"]), score=float(r["score"]),
                    score_type=r.get("score_type"), risk_level=r.get("risk_level"), will_churn=bool(r["will_churn"])))
    return imported, skipped


def run_import(engine, files: DatasetStore, history: HistoryStore, *, dry_run: bool = False, verify: bool = True,
               log=lambda message: None) -> Report:
    report, pg = Report(), PostgresDatasetStore(engine)
    for meta in files.list():
        ds = meta["id"]
        label = f"{ds}  {meta.get('name', '')}"
        try:
            if _exists(engine, ds):
                report.skipped.append(ds)
                log(f"already in PostgreSQL  {label}")
                continue
            if not dry_run:
                import_dataset(engine, files, ds)
                problems = verify_dataset(pg, files, ds) if verify else []
                if problems:
                    _remove(engine, ds)                         # never leave a copy that does not match
                    raise ValueError("copy does not match the original: " + "; ".join(problems))
            report.imported.append(ds)
            log(f"{'would import' if dry_run else 'imported':<22}{label} ({meta['rows']['scored']} customers)")
        except Exception as exc:                                # noqa: BLE001  one bad dataset must not stop the rest
            report.failed.append((ds, f"{type(exc).__name__}: {exc}"))
            log(f"FAILED                {label}: {type(exc).__name__}: {exc}")
    report.history_imported, report.history_skipped = import_history(engine, history, dry_run=dry_run)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m churn_api.storage.import_files",
                                 description="Copy the file store (data/) into PostgreSQL (DATABASE_URL).")
    ap.add_argument("--data-dir", help="folder holding datasets/ and history.sqlite3 (default: DATA_DIR, else ./data)")
    ap.add_argument("--dry-run", action="store_true", help="show what would be imported; change nothing")
    ap.add_argument("--no-verify", action="store_true", help="skip the comparison after each dataset")
    args = ap.parse_args(argv)

    from ..config import Config
    cfg = Config.from_env()
    if not cfg.DATABASE_URL:
        print("DATABASE_URL is not set: nothing to import into. Set it to the PostgreSQL database first.")
        return 2
    data_dir = Path(args.data_dir) if args.data_dir else cfg.DATA_DIR
    if not data_dir.is_dir():
        print(f"The data folder '{data_dir}' does not exist.")
        return 2

    engine = db.make_engine(cfg.DATABASE_URL)
    try:
        try:
            db.init_schema(engine)
        except Exception as exc:                                # noqa: BLE001
            print(f"Cannot use the database: {db.describe_error(exc, cfg.DATABASE_URL)}")
            return 2
        print(f"{'DRY RUN: ' if args.dry_run else ''}importing from {data_dir}")
        report = run_import(engine, DatasetStore(data_dir), HistoryStore(data_dir), dry_run=args.dry_run,
                            verify=not args.no_verify, log=print)
    finally:
        engine.dispose()

    verb = "would be imported" if args.dry_run else "imported"
    print(f"\nDatasets: {len(report.imported)} {verb}, {len(report.skipped)} already there, {len(report.failed)} failed")
    print(f"Predictions: {report.history_imported} {verb}, {report.history_skipped} already there")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())