"""File-based registry of uploaded datasets and their pipeline results.

    data/datasets/<id>/  meta.json  raw.csv  results.csv  rejected.csv  summary.json

The id is server-generated (never derived from the user's filename), which rules out path
traversal. Writes are atomic (temp file + rename) so a crash never leaves a half-written dataset.
Keeping raw.csv is what makes re-scoring possible when the model is replaced.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_ID = re.compile(r"^[0-9a-f]{32}$")


class DatasetNotFound(LookupError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class DatasetStore:
    def __init__(self, data_dir: Path, cache_size: int = 4):
        self.root = Path(data_dir) / "datasets"
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: OrderedDict = OrderedDict()
        self._cache_size, self._lock = cache_size, threading.Lock()

    # ------------------------------------------------------------------ paths
    def _dir(self, dataset_id: str) -> Path:
        if not _ID.match(str(dataset_id)):
            raise DatasetNotFound(dataset_id)
        d = self.root / dataset_id
        if not (d / "meta.json").exists():
            raise DatasetNotFound(dataset_id)
        return d

    # ------------------------------------------------------------------ write
    def _write_run(self, d: Path, run, model: dict) -> None:
        results_csv = run.results.to_csv(index=False).encode()
        _atomic_write(d / "results.csv", results_csv)
        _atomic_write(d / "rejected.csv", run.rejected.to_csv(index=False).encode())
        _atomic_write(d / "summary.json", json.dumps(run.summary).encode())

    def _meta_from_run(self, run, model: dict) -> dict:
        s = run.stats
        return {
            "status": "completed", "policy": s["policy"], "has_target": s["target_present"],
            "model": {"name": model["name"], "version": model["version"], "score_type": model["score_type"]},
            "rows": {"uploaded": s["rows_uploaded"], "scored": s["rows_valid"], "rejected": s["rows_rejected"],
                     "repaired": s["rows_repaired"], "duplicates_removed": s["duplicates_removed"]},
        }

    def save(self, *, name: str, original_filename: str, raw_bytes: bytes, run, model: dict) -> dict:
        dataset_id = uuid.uuid4().hex
        final = self.root / dataset_id
        tmp = self.root / f".tmp-{dataset_id}"
        tmp.mkdir()
        try:
            (tmp / "raw.csv").write_bytes(raw_bytes)
            meta = {"id": dataset_id, "name": name, "original_filename": original_filename,
                    "created_at": _now(), "updated_at": _now(), **self._meta_from_run(run, model)}
            self._write_run(tmp, run, model)
            (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
            os.replace(tmp, final)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return meta

    def replace_results(self, dataset_id: str, run, model: dict) -> dict:
        d = self._dir(dataset_id)
        meta = json.loads((d / "meta.json").read_text())
        self._write_run(d, run, model)
        meta.update(self._meta_from_run(run, model))
        meta["updated_at"] = _now()
        _atomic_write(d / "meta.json", json.dumps(meta, indent=2).encode())
        return meta

    def delete(self, dataset_id: str) -> None:
        shutil.rmtree(self._dir(dataset_id))
        with self._lock:
            for k in [k for k in self._cache if k[0] == dataset_id]:
                del self._cache[k]

    # ------------------------------------------------------------------ read
    def meta(self, dataset_id: str) -> dict:
        return json.loads((self._dir(dataset_id) / "meta.json").read_text())

    def list(self) -> list[dict]:
        metas = []
        for d in self.root.iterdir():
            if d.is_dir() and _ID.match(d.name) and (d / "meta.json").exists():
                metas.append(json.loads((d / "meta.json").read_text()))
        return sorted(metas, key=lambda m: m["created_at"], reverse=True)

    def latest_id(self) -> str | None:
        items = self.list()
        return items[0]["id"] if items else None

    def summary(self, dataset_id: str) -> dict:
        return json.loads((self._dir(dataset_id) / "summary.json").read_text())

    def raw_bytes(self, dataset_id: str) -> bytes:
        return (self._dir(dataset_id) / "raw.csv").read_bytes()

    def results(self, dataset_id: str) -> pd.DataFrame:
        """Scored rows. Treat as read-only: the frame is shared through a small LRU cache."""
        path = self._dir(dataset_id) / "results.csv"
        key = (dataset_id, path.stat().st_mtime_ns)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        df = pd.read_csv(path)
        if "data_issues" in df.columns:
            df["data_issues"] = df["data_issues"].fillna("")
        with self._lock:
            self._cache[key] = df
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return df

    def rejected(self, dataset_id: str) -> pd.DataFrame:
        return pd.read_csv(self._dir(dataset_id) / "rejected.csv")
