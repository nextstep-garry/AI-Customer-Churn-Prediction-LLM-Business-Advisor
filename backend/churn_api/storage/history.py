"""Audit trail of single-customer predictions (SQLite, stdlib only)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


class HistoryStore:
    def __init__(self, data_dir: Path):
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(data_dir) / "history.sqlite3"
        with closing(self._conn()) as c, c:
            c.execute("""CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, request_id TEXT, model_name TEXT,
                model_version TEXT, input_json TEXT NOT NULL, score REAL NOT NULL, score_type TEXT,
                risk_level TEXT, will_churn INTEGER NOT NULL)""")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, *, request_id: str | None, model_name: str, model_version: str, record: dict, result: dict) -> None:
        with closing(self._conn()) as c, c:
            c.execute("INSERT INTO predictions (ts, request_id, model_name, model_version, input_json, score,"
                      " score_type, risk_level, will_churn) VALUES (?,?,?,?,?,?,?,?,?)",
                      (datetime.now(timezone.utc).isoformat(timespec="seconds"), request_id, model_name,
                       model_version, json.dumps(record), result["churn_score"], result["score_type"],
                       result["risk_level"], int(result["will_churn"])))

    def recent(self, limit: int = 20) -> list[dict]:
        with closing(self._conn()) as c:
            rows = c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{**{k: r[k] for k in r.keys() if k != "input_json"}, "input": json.loads(r["input_json"]),
                 "will_churn": bool(r["will_churn"])} for r in rows]

    def clear(self) -> int:
        with closing(self._conn()) as c, c:
            return c.execute("DELETE FROM predictions").rowcount
