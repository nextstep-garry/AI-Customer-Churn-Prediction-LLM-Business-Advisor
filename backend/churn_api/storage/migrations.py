"""Schema migrations: one SQL file per version, in sql_migrations/, applied in order.

To change the database layout:
  1. change the table in schema.py (NEW databases are created from it) and bump SCHEMA_VERSION;
  2. add ONE file to sql_migrations/ named NNNN_what_it_does.sql, where NNNN is the new version (0004, 0005, ...). It holds
     plain SQL that makes the same change to an EXISTING database. Never edit a file that has been applied anywhere:
     history is frozen, add a new file instead;
  3. run the tests: they fail if a version has no file, if there is a gap, or if a file is empty or badly named.
All pending files run in ONE transaction together with the version bump (PostgreSQL DDL is transactional), so a failure
changes nothing. Statements that cannot run inside a transaction (CREATE INDEX CONCURRENTLY, VACUUM) must be run by hand.
Commit the files together with the code change: git history is the audit trail.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent / "sql_migrations"
_FILE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def label(self) -> str:
        return f"{self.version:04d}_{self.name}"


def load(directory: Path | str | None = None) -> list[Migration]:
    """Read every migration file in `directory` (default: sql_migrations/), sorted by version."""
    folder = Path(directory) if directory else DIRECTORY
    if not folder.is_dir():
        raise MigrationError(f"The migrations folder '{folder}' is missing.")
    found: list[Migration] = []
    for path in sorted(folder.glob("*.sql")):
        match = _FILE.match(path.name)
        if match is None:
            raise MigrationError(f"'{path.name}' is not a valid migration file name. Use NNNN_description.sql "
                                 "(four digits, an underscore, then lowercase letters, digits and underscores).")
        sql = path.read_text(encoding="utf-8-sig")
        if not sql.strip():
            raise MigrationError(f"Migration file '{path.name}' is empty.")
        found.append(Migration(int(match.group(1)), match.group(2), sql))
    found.sort(key=lambda m: m.version)
    versions = [m.version for m in found]
    duplicated = sorted({v for v in versions if versions.count(v) > 1})
    if duplicated:
        raise MigrationError(f"Two migration files have the same version number: {duplicated}.")
    return found


def validate(migrations: list[Migration], first: int, last: int) -> None:
    """The files must be exactly the versions first..last: nothing missing, nothing extra, no gaps."""
    have = [m.version for m in migrations]
    want = list(range(first, last + 1))
    if have != want:
        missing, extra = sorted(set(want) - set(have)), sorted(set(have) - set(want))
        raise MigrationError("The migration files do not match the schema version. "
                             f"Expected versions {first} to {last}; missing: {missing or 'none'}; "
                             f"unexpected: {extra or 'none'}.")


def run(conn, migration: Migration) -> None:
    """Execute one migration's SQL on `conn`, inside the caller's transaction (any error propagates and rolls it back).
    The SQL goes straight to the PostgreSQL driver without parameter handling, so it may hold several statements."""
    raw = conn.connection.driver_connection
    try:
        with raw.cursor() as cursor:
            cursor.execute(migration.sql)
    except Exception as exc:                                              # noqa: BLE001
        reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        raise MigrationError(f"Migration {migration.label} failed: {reason}") from exc