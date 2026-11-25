# """PostgreSQL connection and schema bootstrap.

# The engine is created lazily (no connection until first use), so a database that is not up yet cannot stop the API from
# starting; /ready reports it and retries. Nothing here ever logs or returns the connection string's password.
# """
# from __future__ import annotations

# import logging

# import sqlalchemy as sa

# from .schema import SCHEMA_VERSION, metadata, schema_version

# log = logging.getLogger(__name__)
# _LOCK_ID = 727_001          # arbitrary app-wide number for the advisory lock below


# class SchemaMismatch(RuntimeError):
#     pass

# def make_engine(url: str) -> sa.Engine:
#     return sa.create_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True, pool_recycle=1800,
#                             connect_args={"connect_timeout": 5})
# #def make_engine(url: str) -> sa.Engine:
#     #return sa.create_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True, pool_recycle=1800)


# def init_schema(engine: sa.Engine) -> None:
#     """Create missing tables and record the layout version. Safe to call repeatedly and from several workers at once."""
#     with engine.begin() as conn:
#         conn.execute(sa.text("SELECT pg_advisory_xact_lock(:id)"), {"id": _LOCK_ID})   # workers starting together queue here
#         metadata.create_all(conn)
#         found = conn.execute(sa.select(sa.func.max(schema_version.c.version))).scalar()
#         if found is None:
#             conn.execute(sa.insert(schema_version).values(version=SCHEMA_VERSION))
#         elif found != SCHEMA_VERSION:
#             raise SchemaMismatch(f"The database has schema version {found}, this code expects {SCHEMA_VERSION}. "
#                                  "Run the migration, or point DATABASE_URL at a fresh database.")


# def ping(engine: sa.Engine) -> None:
#     with engine.connect() as conn:
#         conn.execute(sa.text("SELECT 1"))


# def describe_error(exc: Exception, url: str = "") -> str:
#     """A one-line, password-free description that is safe to log and to show in /ready."""
#     text = str(getattr(exc, "orig", None) or exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
#     try:
#         password = sa.make_url(url).password
#     except Exception:           # noqa: BLE001
#         password = None
#     return text.replace(password, "***") if password else text


"""PostgreSQL connection, schema bootstrap and migrations.

The engine is created lazily (no connection until first use), so a database that is not up yet cannot stop the API from
starting; /ready reports it and retries. Nothing here ever logs or returns the connection string's password.

How the layout is kept current (to add a change, see migrations.py):
  - a brand-new database is created directly at SCHEMA_VERSION;
  - an older database is upgraded by running the SQL files above its version (sql_migrations/), in ONE transaction under
    a lock, so several workers starting together are safe and a failed migration changes nothing;
  - a newer database, or one older than MIN_SUPPORTED_VERSION, is refused with a clear message.
"""
from __future__ import annotations

import logging

import sqlalchemy as sa

from . import migrations as mig
from .schema import SCHEMA_VERSION, metadata, schema_version

log = logging.getLogger(__name__)
_LOCK_ID = 727_001          # arbitrary app-wide number for the advisory lock below
MIN_SUPPORTED_VERSION = 2   # version 1 was an early development layout and cannot be upgraded


class SchemaMismatch(RuntimeError):
    pass


def make_engine(url: str) -> sa.Engine:
    return sa.create_engine(url, pool_size=5, max_overflow=5, pool_pre_ping=True, pool_recycle=1800,
                            connect_args={"connect_timeout": 5})


def check_supported(found: int, *, target: int = SCHEMA_VERSION, minimum: int = MIN_SUPPORTED_VERSION) -> None:
    """Raise SchemaMismatch unless a database at version `found` can be used or upgraded by this code."""
    if found > target:
        raise SchemaMismatch(f"The database has schema version {found}, which is newer than this code ({target}). "
                             "Update the application, or point DATABASE_URL at another database.")
    if found < minimum:
        raise SchemaMismatch(f"The database has schema version {found}, created by an early development version that "
                             "cannot be upgraded. Create a fresh database, then re-import the data with "
                             "python -m churn_api.storage.import_files.")


def current_version(engine: sa.Engine) -> int | None:
    """The database's schema version, or None when it has no schema yet. Changes nothing."""
    if not sa.inspect(engine).has_table("schema_version"):
        return None
    with engine.connect() as conn:
        return conn.execute(sa.select(sa.func.max(schema_version.c.version))).scalar()


def pending_versions(engine: sa.Engine) -> list[int]:
    found = current_version(engine)
    return [] if found is None else list(range(found + 1, SCHEMA_VERSION + 1))


def init_schema(engine: sa.Engine, *, migrations=None, target: int | None = None, minimum: int | None = None) -> list[int]:
    """Create or upgrade the schema. Safe to call repeatedly and from several workers at once.
    Returns the migration versions that were applied (empty for a new or already current database).
    `migrations`, `target` and `minimum` default to the shipped files and constants; tests override them."""
    target = SCHEMA_VERSION if target is None else target
    minimum = MIN_SUPPORTED_VERSION if minimum is None else minimum
    files = mig.load() if migrations is None else list(migrations)
    mig.validate(files, minimum + 1, target)                  # the files must cover exactly the versions above `minimum`
    by_version = {m.version: m for m in files}
    with engine.begin() as conn:
        conn.execute(sa.text("SELECT pg_advisory_xact_lock(:id)"), {"id": _LOCK_ID})   # workers starting together queue here
        metadata.create_all(conn)
        found = conn.execute(sa.select(sa.func.max(schema_version.c.version))).scalar()
        if found is None:                                      # a brand-new database: created at the latest version
            conn.execute(sa.insert(schema_version).values(version=target))
            return []
        check_supported(found, target=target, minimum=minimum)
        applied: list[int] = []
        for version in range(found + 1, target + 1):
            mig.run(conn, by_version[version])
            conn.execute(sa.insert(schema_version).values(version=version))
            applied.append(version)
    if applied:
        log.info("schema_migrated", extra={"from_version": found, "to_version": target})
    return applied


def ping(engine: sa.Engine) -> None:
    with engine.connect() as conn:
        conn.execute(sa.text("SELECT 1"))


def describe_error(exc: Exception, url: str = "") -> str:
    """A one-line, password-free description that is safe to log and to show in /ready."""
    text = str(getattr(exc, "orig", None) or exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    try:
        password = sa.make_url(url).password
    except Exception:           # noqa: BLE001
        password = None
    return text.replace(password, "***") if password else text