"""The PostgreSQL stores behave exactly like the file stores. Each test puts the same upload through both."""
import io
import time

import pandas as pd
import pytest

sa = pytest.importorskip("sqlalchemy")

from churn_api.storage import schema as s                                   # noqa: E402
from churn_api.storage.datasets import DatasetNotFound                      # noqa: E402
from churn_api.storage.pg_stores import _clean_json                         # noqa: E402
from conftest import ROOT, SAMPLES, make_app, upload                        # noqa: E402

SOURCES = {"dirty": SAMPLES / "churn_sample_dirty.csv",                      # labelled, with rejected rows
           "unlabelled": SAMPLES / "churn_sample_unlabeled.csv",             # no Exited column
           "real": ROOT / "training" / "data" / "churn_raw.csv"}             # 10,000 messy rows, IDs like "15589358X"


@pytest.fixture
def twin(tmp_path, pg_url, bundle):
    return make_app(tmp_path / "files", bundle, DATABASE_URL=""), make_app(tmp_path / "pg", bundle, DATABASE_URL=pg_url)


def put(app, source, name="x", policy="reject") -> str:
    r = upload(app.test_client(), source, name, policy)
    assert r.status_code == 201, r.get_json()
    return r.get_json()["dataset"]["id"]


def same_frame(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Same columns, same values, and the SAME dtype for every numeric/boolean column. A text column may come back as
    pandas' `str` or as `object` (the CSV store gives object for an empty column): the values are what matters there."""
    assert list(a.columns) == list(b.columns)
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col].dtype) or pd.api.types.is_bool_dtype(a[col].dtype):
            assert a[col].dtype == b[col].dtype, (col, a[col].dtype, b[col].dtype)
        else:
            assert not pd.api.types.is_numeric_dtype(b[col].dtype), (col, a[col].dtype, b[col].dtype)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def store(app):
    return app.extensions["churn"].store


def engine(app):
    return app.extensions["churn"].db


# ---------------------------------------------------------------------------------------------- same data back
@pytest.mark.parametrize("source", SOURCES)
def test_results_rejected_summary_and_raw_match_the_file_store(twin, source):
    files, pg = twin
    fid, pid = put(files, SOURCES[source]), put(pg, SOURCES[source])
    same_frame(store(files).results(fid), store(pg).results(pid))
    same_frame(store(files).rejected(fid), store(pg).rejected(pid))
    without_time = lambda d: {k: v for k, v in d.items() if k != "generated_at"}      # the only field that differs: the time   # noqa: E731
    assert without_time(_clean_json(store(files).summary(fid))) == without_time(store(pg).summary(pid))
    assert store(files).raw_bytes(fid) == store(pg).raw_bytes(pid) == SOURCES[source].read_bytes()


@pytest.mark.parametrize("source", SOURCES)
def test_metadata_has_the_same_shape_and_values(twin, source):
    files, pg = twin
    f, p = store(files).meta(put(files, SOURCES[source])), store(pg).meta(put(pg, SOURCES[source]))
    assert set(f) == set(p)
    ignore = {"id", "created_at", "updated_at"}
    assert {k: v for k, v in f.items() if k not in ignore} == {k: v for k, v in p.items() if k not in ignore}
    assert len(p["created_at"]) == len(f["created_at"])                      # same ISO format, microseconds and offset


def test_list_and_latest_id_are_newest_first(twin):
    _, pg = twin
    ids = [put(pg, SOURCES["unlabelled"], name=f"n{i}") for i in range(3)]
    assert [m["name"] for m in store(pg).list()] == ["n2", "n1", "n0"]
    assert store(pg).latest_id() == ids[-1]
    assert all("summary" not in m for m in store(pg).list())                 # listing never drags the big summary along


def test_latest_id_is_none_when_empty(twin):
    assert store(twin[1]).latest_id() is None and store(twin[1]).list() == []


# ---------------------------------------------------------------------------------------------- re-score, cache, delete
def test_rescore_replaces_results_keeps_identity_and_refreshes_the_cache(twin):
    _, pg = twin
    did = put(pg, SOURCES["dirty"])
    before_meta, before = store(pg).meta(did), store(pg).results(did)
    assert store(pg).results(did) is before                                  # served from the cache
    assert pg.test_client().post(f"/v1/datasets/{did}/rescore").status_code == 200
    after_meta, after = store(pg).meta(did), store(pg).results(did)
    assert after_meta["id"] == did and after_meta["created_at"] == before_meta["created_at"]
    assert after_meta["updated_at"] > before_meta["updated_at"]
    assert after is not before and len(after) == len(before)                # new frame, rows replaced not duplicated
    with engine(pg).connect() as c:
        assert c.execute(sa.select(sa.func.count()).select_from(s.dataset_results)).scalar() == len(before)


def test_a_failed_rescore_changes_nothing(twin):
    _, pg = twin
    did = put(pg, SOURCES["dirty"])
    good, meta = store(pg).results(did).copy(), store(pg).meta(did)
    run = pg.extensions["churn"].pipeline.run_dataset(pd.read_csv(io.BytesIO(SOURCES["dirty"].read_bytes())), "reject")
    run.results["Age"] = run.results["Age"].astype(object)
    run.results.loc[run.results.index[-1], "Age"] = "not a number"           # fails half way through the inserts
    with pytest.raises(sa.exc.DBAPIError):
        store(pg).replace_results(did, run, {"name": "m", "version": "9", "score_type": "ranking"})
    same_frame(store(pg).results(did), good)                                # the old rows are all still there
    assert store(pg).meta(did) == meta                                      # and the metadata was not touched


def test_delete_removes_every_row_everywhere_and_evicts_the_cache(twin):
    _, pg = twin
    keep, gone = put(pg, SOURCES["dirty"], "keep"), put(pg, SOURCES["dirty"], "gone")
    store(pg).results(gone)
    store(pg).delete(gone)
    with engine(pg).connect() as c:
        for table, column in ((s.dataset_results, "dataset_id"), (s.dataset_rejected, "dataset_id"),
                              (s.dataset_raw, "dataset_id")):
            left = c.execute(sa.select(table.c[column]).where(table.c[column] == gone)).all()
            assert left == []
    assert len(store(pg).results(keep)) > 0                                  # the other dataset is untouched
    with pytest.raises(DatasetNotFound):
        store(pg).results(gone)


@pytest.mark.parametrize("method", ["meta", "summary", "raw_bytes", "results", "rejected", "delete"])
@pytest.mark.parametrize("bad_id", ["0" * 32, "../etc/passwd", "short", ""])
def test_unknown_or_malformed_ids_are_not_found(twin, method, bad_id):
    with pytest.raises(DatasetNotFound):
        getattr(store(twin[1]), method)(bad_id)


def test_replace_results_on_an_unknown_dataset_is_not_found(twin):
    with pytest.raises(DatasetNotFound):
        store(twin[1]).replace_results("0" * 32, None, {})


# ---------------------------------------------------------------------------------------------- history
def test_prediction_history_matches_the_sqlite_store(twin):
    files, pg = twin
    record = {"Age": 42, "Geography": "France"}
    result = {"churn_score": 0.5517, "score_type": "ranking", "risk_level": "Medium", "will_churn": True}
    for app in (files, pg):
        h = app.extensions["churn"].history
        h.add(request_id="r1", model_name="m", model_version="1.0.0", record=record, result=result)
        h.add(request_id="r2", model_name="m", model_version="1.0.0", record=record, result=dict(result, will_churn=False))
    f, p = (a.extensions["churn"].history.recent(10) for a in (files, pg))
    assert [set(x) for x in f] == [set(x) for x in p]
    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("ts", "id")} for r in rows]   # noqa: E731
    assert strip(f) == strip(p) and [r["request_id"] for r in p] == ["r2", "r1"]           # newest first
    assert len(p[0]["ts"]) == len(f[0]["ts"])                                                  # same timestamp format
    assert (files.extensions["churn"].history.clear(), pg.extensions["churn"].history.clear()) == (2, 2)
    assert pg.extensions["churn"].history.recent() == []


def test_history_through_the_api(twin):
    _, pg = twin
    body = {"CreditScore": 619, "Geography": "France", "Gender": "Female", "Age": 42, "Tenure": 2, "Balance": 0,
            "NumOfProducts": 1, "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 101348.88}
    assert pg.test_client().post("/v1/predict", json=body).status_code == 200
    saved = pg.test_client().get("/v1/history").get_json()["predictions"]
    assert len(saved) == 1 and saved[0]["input"]["Age"] == 42


# ---------------------------------------------------------------------------------------------- the database goes away
def test_a_down_database_gives_503_not_500_and_recovers_by_itself(tmp_path, pg_admin_url, bundle):
    name = "late_" + __import__("uuid").uuid4().hex[:8]
    url = sa.make_url(pg_admin_url).set(database=name).render_as_string(hide_password=False)
    app = make_app(tmp_path, bundle, DATABASE_URL=url)                       # the database does not exist yet
    client = app.test_client()
    down = client.get("/v1/datasets")
    assert down.status_code == 503 and down.get_json()["error"]["code"] == "database_unavailable"
    assert client.get("/health").status_code == 200                          # liveness still answers
    admin = sa.create_engine(pg_admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as c:
            c.execute(sa.text(f'CREATE DATABASE "{name}"'))
        assert client.get("/v1/datasets").get_json()["datasets"] == []      # no restart needed
        assert upload(client, SOURCES["unlabelled"]).status_code == 201
    finally:
        engine(app).dispose()
        with admin.connect() as c:
            c.execute(sa.text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def test_a_database_error_in_the_middle_of_a_request_is_a_503(twin, monkeypatch):
    _, pg = twin
    def boom(*a, **k):
        raise sa.exc.OperationalError("SELECT 1", {}, Exception("server closed the connection unexpectedly"))
    monkeypatch.setattr(store(pg), "list", boom)
    r = pg.test_client().get("/v1/datasets")
    assert r.status_code == 503 and r.get_json()["error"]["code"] == "database_unavailable"
    assert "server closed" not in r.get_data(as_text=True)                  # internal text stays in the log


# ---------------------------------------------------------------------------------------------- speed
def test_saving_ten_thousand_rows_is_fast_enough(twin):
    _, pg = twin
    started = time.perf_counter()
    did = put(pg, SOURCES["real"])
    seconds = time.perf_counter() - started
    assert store(pg).meta(did)["rows"]["scored"] == 9773
    assert seconds < 15, f"upload of 10,000 rows took {seconds:.1f}s"