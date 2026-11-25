"""Show or apply schema migrations by hand.

    cd backend
    python -m churn_api.storage.migrate            # show the status, change nothing
    python -m churn_api.storage.migrate --apply    # upgrade the database

The target is DATABASE_URL. The app also upgrades automatically when it starts; this tool is for checking first.
"""
from __future__ import annotations

import argparse
import sys

from . import db
from . import migrations as mig
from .schema import SCHEMA_VERSION


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m churn_api.storage.migrate",
                                 description="Show or apply schema migrations (DATABASE_URL).")
    ap.add_argument("--apply", action="store_true", help="upgrade the database (default: only show the status)")
    args = ap.parse_args(argv)

    from ..config import Config
    cfg = Config.from_env()
    if not cfg.DATABASE_URL:
        print("DATABASE_URL is not set: there is no database to check.")
        return 2
    try:
        files = {m.version: m for m in mig.load()}
    except mig.MigrationError as exc:
        print(exc)
        return 1

    engine = db.make_engine(cfg.DATABASE_URL)
    try:
        try:
            found = db.current_version(engine)
        except Exception as exc:                                       # noqa: BLE001
            print(f"Cannot use the database: {db.describe_error(exc, cfg.DATABASE_URL)}")
            return 2
        print(f"Database: {engine.url.render_as_string(hide_password=True)}")
        if found is None:
            print(f"The database is empty: it is created at version {SCHEMA_VERSION} on first use.")
            if not args.apply:
                return 0
        else:
            print(f"Schema version in the database: {found} (this code expects {SCHEMA_VERSION})")
            try:
                db.check_supported(found)
            except db.SchemaMismatch as exc:
                print(exc)
                return 1
            pending = db.pending_versions(engine)
            if not pending:
                print("The database schema is up to date.")
                return 0
            print("Pending migrations: " + ", ".join(map(str, pending)))
            for version in pending:
                print(f"  {files[version].label}" if version in files else f"  {version:04d} (file missing)")
            if not args.apply:
                print("Run again with --apply to upgrade.")
                return 0
        try:
            applied = db.init_schema(engine)
        except (db.SchemaMismatch, mig.MigrationError) as exc:
            print(exc)
            return 1
        print("Applied migrations: " + ", ".join(map(str, applied)) if applied else "The database is ready.")
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())