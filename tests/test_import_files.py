"""Importing an existing file store (data/) into PostgreSQL."""
import pandas as pd
import pytest

pytest.importorskip("sqlalchemy")

from churn_api.storage import db                                            # noqa: E402
from churn_api.storage import import_files as imp                           # noqa: E402
from churn_api.storage.datasets import DatasetStore                         # noqa: E402
from churn_api.storage.history import HistoryStore                          # noqa: E402
from churn_api.storage.pg_stores import PostgresDatasetStore, PostgresHistoryStore   # noqa: E402
from conftest import SAMPLES, make_app, upload                              # noqa: E402

RESULT = {"churn_score": 0.5517, "score_type": "ranking", "risk_level": "Medium", "will_churn": True}


def same_frame(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Same columns and values; numeric/boolean dtypes must match, text may be `str` or `object`."""
    assert list(a.columns) == list(b.columns)
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col].dtype) or pd.api.types.is_bool_dtype(a[col].dtype):
            assert a[col].dtype == b[col].dtype, (col, a[col].dtype, b[col].dtype)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


@pytest.fixture
def old(tmp_path, bundle):
    """A file-store data folder with two datasets and two predictions, as the app wrote it before PostgreSQL."""
    app = make_app(tmp_path / "old", bundle, DATABASE_URL="")
    ids = []
    for source, name in ((SAMPLES / "churn_sample_dirty.csv", "dirty"), (SAMPLES / "churn_sample_unlabeled.csv", "unlabelled")):
        r = upload(app.test_client(), source, name)
        assert r.status_code == 201, r.get_json()
        ids.append(r.get_json()["dataset"]["id"])
    history = app.extensions["churn"].history
    history.add(request_id="r1", model_name="m", model_version="1.0.0", record={"Age": 42}, result=RESULT)
    history.add(request_id="r2", model_name="m", model_version="1.0.0", record={"Age": 60},
                result=dict(RESULT, will_churn=False))
    return tmp_path / "old", ids


@pytest.fixture
def target(pg_url):
    engine = db.make_engine(pg_url)
    db.init_schema(engine)
    yield engine
    engine.dispose()


def test_import_copies_everything_and_verifies_it(old, target):
    path, ids = old
    files, history = DatasetStore(path), HistoryStore(path)
    report = imp.run_import(target, files, history)
    assert report.ok and sorted(report.imported) == sorted(ids) and report.skipped == []
    pg = PostgresDatasetStore(target)
    assert {m["id"] for m in pg.list()} == set(ids)                          # the original ids are kept
    for ds in ids:
        same_frame(files.results(ds), pg.results(ds))
        assert files.raw_bytes(ds) == pg.raw_bytes(ds)
        assert files.meta(ds)["name"] == pg.meta(ds)["name"]
        assert imp._parse_time(files.meta(ds)["created_at"]) == imp._parse_time(pg.meta(ds)["created_at"])


def test_history_is_copied_in_order_with_its_original_times(old, target):
    path, _ = old
    imp.run_import(target, DatasetStore(path), HistoryStore(path))
    source, copy = HistoryStore(path).recent(10), PostgresHistoryStore(target).recent(10)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "id"} for r in rows]       # noqa: E731
    assert strip(copy) == strip(source) and [r["request_id"] for r in copy] == ["r2", "r1"]


def test_running_twice_changes_nothing(old, target):
    path, ids = old
    imp.run_import(target, DatasetStore(path), HistoryStore(path))
    second = imp.run_import(target, DatasetStore(path), HistoryStore(path))
    assert second.imported == [] and sorted(second.skipped) == sorted(ids)
    assert (second.history_imported, second.history_skipped) == (0, 2)
    assert len(PostgresDatasetStore(target).list()) == 2 and len(PostgresHistoryStore(target).recent(10)) == 2


def test_dry_run_writes_nothing(old, target):
    path, ids = old
    report = imp.run_import(target, DatasetStore(path), HistoryStore(path), dry_run=True)
    assert sorted(report.imported) == sorted(ids) and report.history_imported == 2          # "would import"
    assert PostgresDatasetStore(target).list() == [] and PostgresHistoryStore(target).recent(10) == []


def test_a_broken_dataset_is_reported_and_does_not_stop_the_others(old, target):
    path, ids = old
    (path / "datasets" / ids[0] / "results.csv").unlink()
    report = imp.run_import(target, DatasetStore(path), HistoryStore(path))
    assert [d for d, _ in report.failed] == [ids[0]] and report.imported == [ids[1]] and not report.ok
    assert [m["id"] for m in PostgresDatasetStore(target).list()] == [ids[1]]


def test_a_copy_that_fails_verification_is_removed_again(old, target, monkeypatch):
    monkeypatch.setattr(imp, "verify_dataset", lambda *a, **k: ["rows differ"])
    path, _ = old
    report = imp.run_import(target, DatasetStore(path), HistoryStore(path))
    assert len(report.failed) == 2 and "does not match" in report.failed[0][1]
    assert PostgresDatasetStore(target).list() == []                         # nothing half-trusted is left behind


def test_command_line_imports_then_reports_already_there(old, pg_url, monkeypatch, capsys):
    path, _ = old
    monkeypatch.setenv("DATABASE_URL", pg_url)
    assert imp.main(["--data-dir", str(path)]) == 0
    assert "imported" in capsys.readouterr().out
    assert imp.main(["--data-dir", str(path)]) == 0
    assert "already in PostgreSQL" in capsys.readouterr().out


def test_command_line_needs_a_database_and_a_data_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert imp.main(["--data-dir", str(tmp_path)]) == 2
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:1/x")
    assert imp.main(["--data-dir", str(tmp_path / "does-not-exist")]) == 2