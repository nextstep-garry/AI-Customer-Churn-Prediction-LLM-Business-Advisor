"""PostgreSQL foundation: schema, constraints, safe startup, readiness. Runs against a REAL PostgreSQL (see conftest)."""
import threading

import pytest

sa = pytest.importorskip("sqlalchemy")

from churn_api.config import Config                                    # noqa: E402
from churn_api.storage import db                                       # noqa: E402
from churn_api.storage import schema as s                              # noqa: E402
from conftest import make_app                                          # noqa: E402

TABLES = {"datasets", "dataset_raw", "dataset_results", "dataset_rejected", "predictions", "schema_version"}


@pytest.fixture
def engine(pg_url):
    e = db.make_engine(pg_url)
    db.init_schema(e)
    yield e
    e.dispose()


def add_dataset(conn, dataset_id="a" * 32, **extra):
    conn.execute(sa.insert(s.datasets).values(
        id=dataset_id, name="demo", original_filename="demo.csv", status="completed", policy="reject",
        has_target=True, model_name="m", model_version="1.0.0", score_type="ranking", rows_uploaded=3,
        rows_scored=2, rows_rejected=1, rows_repaired=0, duplicates_removed=0, summary={"kpis": {"n": 2}}, **extra))


def result_row(dataset_id, row_id, **extra):
    return dict(dataset_id=dataset_id, row_id=row_id, CustomerId=15634602 + row_id, Surname="Hill", CreditScore=619,
                Geography="France", Gender="Female", Age=42, Tenure=2, Balance=0.0, NumOfProducts=1, HasCrCard=1,
                IsActiveMember=1, EstimatedSalary=101348.88, churn_score=0.55, will_churn=True, risk_level="Medium",
                outside_training_range=False, row_status="ok", data_issues="", Exited=1, **extra)


# ---------------------------------------------------------------------------------------------- schema bootstrap
def test_init_schema_creates_every_table_and_records_the_version(engine):
    assert set(sa.inspect(engine).get_table_names()) == TABLES
    with engine.connect() as c:
        assert c.execute(sa.select(s.schema_version.c.version)).scalars().all() == [s.SCHEMA_VERSION]


def test_init_schema_is_idempotent(engine):
    db.init_schema(engine)
    db.init_schema(engine)
    with engine.connect() as c:
        assert c.execute(sa.select(sa.func.count()).select_from(s.schema_version)).scalar() == 1


def test_workers_starting_at_the_same_time_do_not_collide(pg_url):
    errors = []
    def start():
        e = db.make_engine(pg_url)
        try:
            db.init_schema(e)
        except Exception as exc:                                       # noqa: BLE001
            errors.append(exc)
        finally:
            e.dispose()
    threads = [threading.Thread(target=start) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []


def test_a_database_with_another_schema_version_is_refused(engine):
    with engine.begin() as c:
        c.execute(sa.update(s.schema_version).values(version=99))
    with pytest.raises(db.SchemaMismatch, match="schema version 99"):
        db.init_schema(engine)


def test_expected_indexes_exist(engine):
    names = {i["name"] for i in sa.inspect(engine).get_indexes("dataset_results")}
    assert {"ix_results_score", "ix_results_tier", "ix_results_geography", "ix_results_customer"} <= names


# ---------------------------------------------------------------------------------------------- data integrity
def test_deleting_a_dataset_deletes_everything_that_belongs_to_it(engine):
    with engine.begin() as c:
        add_dataset(c, "a" * 32)
        add_dataset(c, "b" * 32)
        c.execute(sa.insert(s.dataset_raw).values(dataset_id="a" * 32, csv=b"x"))
        c.execute(sa.insert(s.dataset_results), [result_row("a" * 32, 1), result_row("a" * 32, 2), result_row("b" * 32, 1)])
        c.execute(sa.insert(s.dataset_rejected).values(dataset_id="a" * 32, row_id=3, data_issues="Age", record={"Age": "abc"}))
    with engine.begin() as c:
        c.execute(sa.delete(s.datasets).where(s.datasets.c.id == "a" * 32))
    with engine.connect() as c:
        count = lambda t: c.execute(sa.select(sa.func.count()).select_from(t)).scalar()       # noqa: E731
        assert (count(s.dataset_raw), count(s.dataset_rejected), count(s.dataset_results)) == (0, 0, 1)   # b's row survives


def test_a_result_row_needs_a_real_dataset_and_a_unique_row_id(engine):
    with engine.begin() as c:
        add_dataset(c)
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as c:                  # no such dataset
        c.execute(sa.insert(s.dataset_results).values(result_row("f" * 32, 1)))
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as c:                  # same row twice
        c.execute(sa.insert(s.dataset_results), [result_row("a" * 32, 1), result_row("a" * 32, 1)])


def test_required_columns_are_enforced(engine):
    with engine.begin() as c:
        add_dataset(c)
    bad = result_row("a" * 32, 1)
    bad["churn_score"] = None
    with pytest.raises(sa.exc.IntegrityError), engine.begin() as c:
        c.execute(sa.insert(s.dataset_results).values(bad))


def test_values_survive_a_round_trip(engine):
    raw = bytes(range(256)) + "é;ü\r\n".encode("latin-1")                            # not valid UTF-8 on purpose
    with engine.begin() as c:
        add_dataset(c)
        c.execute(sa.insert(s.dataset_raw).values(dataset_id="a" * 32, csv=raw))
        c.execute(sa.insert(s.dataset_results).values(result_row("a" * 32, 1)))
        c.execute(sa.insert(s.dataset_rejected).values(dataset_id="a" * 32, row_id=9, data_issues="Age: not a number",
                                                       record={"Age": "abc", "CreditScore": None}))
    with engine.connect() as c:
        assert c.execute(sa.select(s.dataset_raw.c.csv)).scalar() == raw
        assert c.execute(sa.select(s.datasets.c.summary)).scalar() == {"kpis": {"n": 2}}
        row = c.execute(sa.select(s.dataset_results)).mappings().one()
        assert row["CustomerId"] == "15634603" and row["Geography"] == "France" and row["will_churn"] is True
        assert c.execute(sa.select(s.dataset_rejected.c.record)).scalar() == {"Age": "abc", "CreditScore": None}


def test_prediction_history_table_accepts_a_record(engine):
    with engine.begin() as c:
        c.execute(sa.insert(s.predictions).values(request_id="r1", model_name="m", model_version="1", input={"Age": 42},
                                                  score=0.5, score_type="ranking", risk_level="Medium", will_churn=True))
    with engine.connect() as c:
        row = c.execute(sa.select(s.predictions)).mappings().one()
        assert row["id"] == 1 and row["input"] == {"Age": 42} and row["ts"] is not None


# ---------------------------------------------------------------------------------------------- the application
def test_without_database_url_nothing_changes(tmp_path):
    body = make_app(tmp_path, DATABASE_URL="").test_client().get("/ready").get_json()      # "" = explicitly no database
    assert body["status"] == "ready" and "database" not in body


def test_ready_reports_a_healthy_database(tmp_path, pg_url, bundle):
    app = make_app(tmp_path, bundle, DATABASE_URL=pg_url)
    r = app.test_client().get("/ready")
    assert r.status_code == 200 and r.get_json()["database"] == "ok"
    assert TABLES <= set(sa.inspect(app.extensions["churn"].db).get_table_names())


def test_a_database_that_is_down_does_not_stop_the_api_and_never_leaks_the_password(tmp_path, bundle):
    url = "postgresql+psycopg://churn:s3cret-pw@127.0.0.1:1/churn"                    # nothing listens on port 1
    app = make_app(tmp_path, bundle, DATABASE_URL=url)
    client = app.test_client()
    assert client.get("/health").status_code == 200                                  # the API itself is up
    r = client.get("/ready")
    assert r.status_code == 503 and r.get_json()["detail"].startswith("database:")
    assert "s3cret-pw" not in r.get_data(as_text=True)


def test_the_api_heals_when_the_database_appears_later(tmp_path, pg_admin_url, bundle):
    name = "late_" + __import__("uuid").uuid4().hex[:8]
    url = sa.make_url(pg_admin_url).set(database=name).render_as_string(hide_password=False)
    app = make_app(tmp_path, bundle, DATABASE_URL=url)                               # the database does not exist yet
    client = app.test_client()
    assert client.get("/ready").status_code == 503
    admin = sa.create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as c:
            c.execute(sa.text(f'CREATE DATABASE "{name}"'))
        assert client.get("/ready").status_code == 200                               # no restart needed
        assert TABLES <= set(sa.inspect(app.extensions["churn"].db).get_table_names())
    finally:
        app.extensions["churn"].db.dispose()
        with admin.connect() as c:
            c.execute(sa.text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def test_describe_error_hides_the_password():
    url = "postgresql+psycopg://u:hunter2@h/db"
    assert "hunter2" not in db.describe_error(RuntimeError("login failed for password hunter2"), url)


def test_config_reads_database_url_from_the_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "  postgresql+psycopg://u:p@h/db  ")
    assert Config.from_env().DATABASE_URL == "postgresql+psycopg://u:p@h/db"
    monkeypatch.delenv("DATABASE_URL")
    assert Config.from_env().DATABASE_URL == ""