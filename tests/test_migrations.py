"""Schema migrations: SQL files, ordered upgrades, refusals, rollback, concurrency, command line (REAL PostgreSQL)."""
import re
import threading

import pytest

sa = pytest.importorskip("sqlalchemy")

from churn_api.storage import db, migrate, migrations                      # noqa: E402
from churn_api.storage import schema as s                                  # noqa: E402
from churn_api.storage.migrations import Migration, MigrationError          # noqa: E402

NEW_INDEXES = {"ix_results_geography_lower", "ix_results_gender_lower", "ix_results_tier_lower"}
FIRST = db.MIN_SUPPORTED_VERSION + 1
PENDING_FROM_MIN = list(range(FIRST, s.SCHEMA_VERSION + 1))                 # what an oldest-supported database must apply


@pytest.fixture
def engine(pg_url):
    e = db.make_engine(pg_url)
    yield e
    e.dispose()


def write(folder, name, sql="SELECT 1;"):
    (folder / name).write_text(sql, encoding="utf-8")


def index_definitions(engine) -> dict[str, str]:
    with engine.connect() as c:
        rows = c.execute(sa.text("select indexname, indexdef from pg_indexes where schemaname = current_schema() "
                                 "and tablename = 'dataset_results'")).all()
    return {name: re.sub(r"\s+", " ", definition) for name, definition in rows if name in NEW_INDEXES}


def versions(engine) -> list[int]:
    with engine.connect() as c:
        return sorted(c.execute(sa.select(s.schema_version.c.version)).scalars().all())


def set_version(engine, version: int) -> None:
    with engine.begin() as c:
        c.execute(sa.text("DELETE FROM schema_version"))
        c.execute(sa.insert(s.schema_version).values(version=version))


def make_old(engine, version: int = db.MIN_SUPPORTED_VERSION) -> None:
    """Turn a freshly created database into one that looks as if an older release had created it."""
    with engine.begin() as c:
        for name in NEW_INDEXES:
            c.execute(sa.text(f"DROP INDEX IF EXISTS {name}"))
    set_version(engine, version)


def add_dataset(engine, name="keep me") -> None:
    with engine.begin() as c:
        c.execute(sa.insert(s.datasets).values(
            id="a" * 32, name=name, original_filename="x.csv", status="completed", policy="reject", has_target=False,
            model_name="m", model_version="1", score_type="ranking", rows_uploaded=0, rows_scored=0, rows_rejected=0,
            rows_repaired=0, duplicates_removed=0, summary={}))


# ---------------------------------------------------------------------------------------------- the files
def test_the_shipped_migration_files_match_the_schema_version():
    """Bumping SCHEMA_VERSION without adding the SQL file must fail here, not in production."""
    files = migrations.load()
    migrations.validate(files, FIRST, s.SCHEMA_VERSION)                      # nothing missing, nothing extra, no gaps
    assert all(m.sql.strip() for m in files)


def test_files_are_ordered_by_version_number(tmp_path):
    write(tmp_path, "0010_later.sql")
    write(tmp_path, "0002_first.sql")                                        # 10 must sort after 2 numerically
    loaded = migrations.load(tmp_path)
    assert [m.version for m in loaded] == [2, 10] and loaded[0].label == "0002_first"


@pytest.mark.parametrize("name", ["3_short.sql", "0003-dash.sql", "0003_Upper.sql", "0003_.sql", "0003.sql"])
def test_badly_named_files_are_refused(tmp_path, name):
    write(tmp_path, name)
    with pytest.raises(MigrationError, match="not a valid migration file name"):
        migrations.load(tmp_path)


def test_empty_and_duplicate_files_are_refused(tmp_path):
    write(tmp_path, "0003_empty.sql", "  \n")
    with pytest.raises(MigrationError, match="is empty"):
        migrations.load(tmp_path)
    (tmp_path / "0003_empty.sql").unlink()
    write(tmp_path, "0003_a.sql")
    write(tmp_path, "0003_b.sql")
    with pytest.raises(MigrationError, match="same version number"):
        migrations.load(tmp_path)


def test_gaps_and_unexpected_versions_are_refused():
    two = [Migration(3, "a", "SELECT 1"), Migration(5, "b", "SELECT 1")]
    with pytest.raises(MigrationError, match=r"missing: \[4\]"):
        migrations.validate(two, 3, 5)
    three = [Migration(3, "a", "SELECT 1"), Migration(4, "b", "SELECT 1"), Migration(7, "c", "SELECT 1")]
    with pytest.raises(MigrationError, match=r"unexpected: \[7\]"):
        migrations.validate(three, 3, 4)


def test_a_missing_folder_is_reported(tmp_path):
    with pytest.raises(MigrationError, match="missing"):
        migrations.load(tmp_path / "nope")


# ---------------------------------------------------------------------------------------------- the database
def test_a_fresh_database_is_created_at_the_latest_version(engine):
    assert db.current_version(engine) is None                                # nothing there yet
    assert db.init_schema(engine) == []                                      # created, nothing to migrate
    assert versions(engine) == [s.SCHEMA_VERSION] and db.pending_versions(engine) == []
    assert set(index_definitions(engine)) == NEW_INDEXES


def test_an_older_database_is_upgraded_and_keeps_its_data(engine):
    db.init_schema(engine)
    add_dataset(engine)
    make_old(engine)
    assert db.pending_versions(engine) == PENDING_FROM_MIN and index_definitions(engine) == {}
    assert db.init_schema(engine) == PENDING_FROM_MIN
    assert db.pending_versions(engine) == [] and set(index_definitions(engine)) == NEW_INDEXES
    assert versions(engine) == [db.MIN_SUPPORTED_VERSION, *PENDING_FROM_MIN]
    with engine.connect() as c:
        assert c.execute(sa.select(s.datasets.c.name)).scalar() == "keep me"


def test_the_sql_migration_builds_the_same_indexes_as_a_new_database(engine):
    """schema.py (new databases) and the SQL file (existing databases) must never drift apart."""
    db.init_schema(engine)
    fresh = index_definitions(engine)
    assert set(fresh) == NEW_INDEXES
    make_old(engine)
    assert index_definitions(engine) == {}
    db.init_schema(engine)
    assert index_definitions(engine) == fresh


def test_upgrading_twice_changes_nothing(engine):
    db.init_schema(engine)
    make_old(engine)
    db.init_schema(engine)
    before = versions(engine)
    assert db.init_schema(engine) == [] and versions(engine) == before


@pytest.mark.parametrize("version,fragment", [(99, "newer than this code"), (1, "cannot be upgraded")])
def test_unsupported_versions_are_refused(engine, version, fragment):
    db.init_schema(engine)
    set_version(engine, version)
    with pytest.raises(db.SchemaMismatch, match=f"schema version {version}") as err:
        db.init_schema(engine)
    assert fragment in str(err.value)


def test_files_run_in_version_order_and_may_hold_several_statements(engine):
    db.init_schema(engine)
    set_version(engine, 2)
    steps = [Migration(3, "create_probe", "CREATE TABLE probe (id serial PRIMARY KEY, n int);"),
             Migration(4, "two_rows", "-- a comment; containing a semicolon\nINSERT INTO probe (n) VALUES (4);\n"
                                      "INSERT INTO probe (n) VALUES (40);"),
             Migration(5, "last_row", "INSERT INTO probe (n) VALUES (5);")]
    assert db.init_schema(engine, migrations=steps, target=5, minimum=2) == [3, 4, 5]
    with engine.connect() as c:
        assert c.execute(sa.text("SELECT n FROM probe ORDER BY id")).scalars().all() == [4, 40, 5]
    assert versions(engine) == [2, 3, 4, 5]


def test_a_failing_migration_changes_nothing(engine):
    db.init_schema(engine)
    set_version(engine, 2)
    broken = [Migration(3, "fine_on_its_own", "CREATE TABLE probe (id int);"),
              Migration(4, "boom", "CREATE TABLE probe_two (id int); SELECT 1/0;")]
    with pytest.raises(MigrationError, match="0004_boom"):
        db.init_schema(engine, migrations=broken, target=4, minimum=2)
    assert versions(engine) == [2]                                           # the version did not move
    tables = set(sa.inspect(engine).get_table_names())
    assert "probe" not in tables and "probe_two" not in tables               # nothing half-done is left behind


def test_an_incomplete_set_of_files_is_refused_before_anything_runs(engine):
    db.init_schema(engine)
    set_version(engine, 2)
    with pytest.raises(MigrationError, match=r"missing: \[4\]"):
        db.init_schema(engine, migrations=[Migration(3, "only_one", "CREATE TABLE probe (id int);")], target=4, minimum=2)
    assert versions(engine) == [2] and "probe" not in set(sa.inspect(engine).get_table_names())


def test_workers_upgrading_at_the_same_time_apply_it_once(engine, pg_url):
    db.init_schema(engine)
    make_old(engine)
    outcomes, errors = [], []

    def start():
        e = db.make_engine(pg_url)
        try:
            outcomes.append(db.init_schema(e))
        except Exception as exc:                                             # noqa: BLE001
            errors.append(exc)
        finally:
            e.dispose()

    threads = [threading.Thread(target=start) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert sorted(v for applied in outcomes for v in applied) == PENDING_FROM_MIN      # applied by exactly one worker
    assert versions(engine) == [db.MIN_SUPPORTED_VERSION, *PENDING_FROM_MIN]


# ---------------------------------------------------------------------------------------------- command line
def test_status_then_apply_from_the_command_line(engine, pg_url, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", pg_url)
    db.init_schema(engine)
    make_old(engine)
    wanted = ", ".join(map(str, PENDING_FROM_MIN))

    assert migrate.main([]) == 0
    out = capsys.readouterr().out
    assert f"Pending migrations: {wanted}" in out and "--apply" in out
    assert "0003_case_insensitive_filter_indexes" in out                     # the file is named, not just a number
    password = sa.make_url(pg_url).password
    assert not password or password not in out                               # the password is never printed
    assert versions(engine) == [db.MIN_SUPPORTED_VERSION]                    # status changes nothing

    assert migrate.main(["--apply"]) == 0
    assert f"Applied migrations: {wanted}" in capsys.readouterr().out
    assert versions(engine) == [db.MIN_SUPPORTED_VERSION, *PENDING_FROM_MIN]

    assert migrate.main([]) == 0
    assert "up to date" in capsys.readouterr().out


def test_the_command_line_needs_a_database(monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert migrate.main([]) == 2 and "DATABASE_URL is not set" in capsys.readouterr().out


def test_the_command_line_refuses_a_newer_database(engine, pg_url, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", pg_url)
    db.init_schema(engine)
    set_version(engine, 99)
    assert migrate.main(["--apply"]) == 1
    assert "newer than this code" in capsys.readouterr().out and versions(engine) == [99]