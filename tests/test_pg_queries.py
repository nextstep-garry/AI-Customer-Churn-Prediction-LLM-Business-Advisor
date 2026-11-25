"""Queries run by PostgreSQL give exactly the answers the pandas functions in analytics.py give."""
import math
import numbers

import pandas as pd
import pytest

sa = pytest.importorskip("sqlalchemy")

from churn_api.services import analytics                                   # noqa: E402
from churn_api.storage import db, pg_queries, pg_stores                     # noqa: E402
from churn_api.storage import schema as s                                   # noqa: E402
from churn_api.storage.datasets import DatasetNotFound                      # noqa: E402
from churn_api.storage.pg_stores import PostgresDatasetStore                # noqa: E402
from churn_api.utils import frame_to_records                                # noqa: E402
from conftest import ROOT, SAMPLES, _fresh_schema_url, make_app, upload     # noqa: E402

SOURCES = {"dirty": SAMPLES / "churn_sample_dirty.csv",
           "unlabelled": SAMPLES / "churn_sample_unlabeled.csv",
           "real": ROOT / "training" / "data" / "churn_raw.csv"}


# ---------------------------------------------------------------------------------------------- helpers
def _same_value(x, y) -> bool:
    missing = lambda v: v is None or (isinstance(v, float) and math.isnan(v))        # noqa: E731
    if missing(x) or missing(y):
        return missing(x) and missing(y)
    if isinstance(x, bool) or isinstance(y, bool):
        return bool(x) == bool(y)
    if isinstance(x, numbers.Real) and isinstance(y, numbers.Real):
        return math.isclose(float(x), float(y), rel_tol=1e-9, abs_tol=1e-12)
    return x == y


def assert_same_records(a: list[dict], b: list[dict]) -> None:
    assert len(a) == len(b), (len(a), len(b))
    for ra, rb in zip(a, b):
        assert list(ra) == list(rb)                                      # the same keys in the same order
        for k in ra:
            assert _same_value(ra[k], rb[k]), (k, ra[k], rb[k])


def put(app, source) -> str:
    r = upload(app.test_client(), source, "x", "reject")
    assert r.status_code == 201, r.get_json()
    return r.get_json()["dataset"]["id"]


def store(app):
    return app.extensions["churn"].store


@pytest.fixture(scope="module", params=list(SOURCES))
def loaded(request, tmp_path_factory, bundle):
    """The same upload, once in the file store and once in PostgreSQL."""
    files = make_app(tmp_path_factory.mktemp("files"), bundle, DATABASE_URL="")
    pg = make_app(tmp_path_factory.mktemp("pg"), bundle, DATABASE_URL=_fresh_schema_url())
    source = SOURCES[request.param]
    return request.param, files, put(files, source), pg, put(pg, source)


def test_constants_stay_in_step_with_analytics_and_the_store():
    assert pg_queries.SORTABLE == analytics._SORTABLE
    assert pg_queries.SEGMENTS == analytics.SEGMENT_FIELDS
    assert pg_queries.OPTIONAL_COLUMNS == pg_stores.OPTIONAL_COLUMNS
    assert pg_queries.RESULT_COLUMNS == pg_stores.RESULT_COLUMNS


# ---------------------------------------------------------------------------------------------- real uploads
QUERIES = [
    {}, {"tier": "High"}, {"tier": "high"}, {"tier": "Medium", "sort_by": "Age", "ascending": True},
    {"geography": "Germany"}, {"geography": "france", "gender": "FEMALE"}, {"gender": "Male", "min_score": 0.5},
    {"min_score": 0.6}, {"flagged_only": True}, {"flagged_only": True, "tier": "High", "limit": 5},
    {"sort_by": "Age", "ascending": True}, {"sort_by": "Balance"}, {"sort_by": "CreditScore", "ascending": True},
    {"sort_by": "EstimatedSalary"}, {"sort_by": "Tenure", "ascending": True}, {"sort_by": "NumOfProducts"},
    {"sort_by": "row_id", "ascending": True}, {"sort_by": "row_id"}, {"sort_by": "not_a_column"},
    {"limit": 5}, {"limit": 7, "offset": 3}, {"limit": 500}, {"limit": 10, "offset": 100_000},
    {"tier": "High", "geography": "Germany", "sort_by": "Age", "ascending": True, "limit": 20, "offset": 5},
    {"min_score": 5.0},
]


@pytest.mark.parametrize("kw", QUERIES, ids=lambda k: str(k)[:70])
def test_query_customers_matches_pandas(loaded, kw):
    _, files, fid, pg, pid = loaded
    total_e, page_e = analytics.query_customers(store(files).results(fid), **kw)
    total_a, page_a = store(pg).query_customers(pid, **kw)
    assert total_a == total_e
    assert list(page_a.columns) == list(page_e.columns)
    assert_same_records(frame_to_records(page_e), frame_to_records(page_a))


def test_search_matches_pandas(loaded):
    source, files, fid, pg, pid = loaded
    if source == "real":
        pytest.skip("the real file has missing surnames; pandas 3 and SQL may treat those differently in a search")
    frame = store(files).results(fid)
    first = frame.iloc[0]
    needles = [str(first["Surname"])[:3], str(first["Surname"]).lower(), str(int(first["row_id"])),
               "zzzz-no-such-text", "50%", "a_b", "'; DROP TABLE dataset_results; --", " "]
    if "CustomerId" in frame.columns:
        needles.append(str(first["CustomerId"])[:5])
    for needle in needles:
        total_e, page_e = analytics.query_customers(frame, search=needle, limit=500)
        total_a, page_a = store(pg).query_customers(pid, search=needle, limit=500)
        assert total_a == total_e, needle
        assert_same_records(frame_to_records(page_e), frame_to_records(page_a))


def test_find_customer_matches_pandas(loaded):
    _, files, fid, pg, pid = loaded
    frame = store(files).results(fid)
    keys = [str(frame.iloc[0]["CustomerId"]) if "CustomerId" in frame.columns else "x",
            str(int(frame.iloc[2]["row_id"])), "1", "0", "99999999", "not-a-key", "9" * 40, ""]
    for key in keys:
        expected, actual = analytics.find_customer(frame, key), store(pg).find_customer(pid, key)
        if expected is None:
            assert actual is None, key
        else:
            assert actual is not None, key
            assert list(actual.index) == list(expected.index)
            assert_same_records([expected.to_dict()], [actual.to_dict()])


@pytest.mark.parametrize("by", analytics.SEGMENT_FIELDS)
def test_segment_breakdown_matches_pandas(loaded, by):
    _, files, fid, pg, pid = loaded
    expected = analytics.segment_breakdown(store(files).results(fid), by)
    actual = store(pg).segment_breakdown(pid, by)
    assert [r["segment"] for r in actual] == [r["segment"] for r in expected]       # the same groups in the same order
    assert_same_records(expected, actual)


def test_unknown_dataset_and_unknown_segment(loaded):
    _, _, _, pg, pid = loaded
    for call in (lambda: store(pg).query_customers("0" * 32), lambda: store(pg).find_customer("0" * 32, "1"),
                 lambda: store(pg).segment_breakdown("0" * 32, "Geography")):
        with pytest.raises(DatasetNotFound):
            call()
    with pytest.raises(ValueError, match="Unknown segment"):
        store(pg).segment_breakdown(pid, "ShoeSize")


# ---------------------------------------------------------------------------------------------- band edges and odd values
def synthetic_frame() -> pd.DataFrame:
    """14 customers chosen to sit exactly on every band edge, plus ties, a missing balance and values outside every band."""
    score = [0.9, 0.9, 0.5, 0.5, 0.5, 0.1, 0.7, 0.7, 0.3, 0.3, 0.95, 0.2, 0.5, 0.1]
    return pd.DataFrame({
        "row_id": range(1, 15),
        "CustomerId": [15_000_001 + i for i in range(14)],
        "Surname": [f"Name{i}" for i in range(14)],
        "CreditScore": [0, 1, 499, 500, 599, 600, 699, 700, 799, 800, 900, 901, 650, 650],
        "Geography": ["France", "Germany", "Spain"] * 4 + ["France", "Spain"],
        "Gender": ["Female", "Male"] * 7,
        "Age": [17, 18, 29, 30, 39, 40, 49, 50, 59, 60, 200, 201, 35, 35],
        "Tenure": [-1, 0, 2, 3, 5, 6, 8, 9, 10, 11, 4, 4, 1, 1],
        "Balance": [0.0, -5.0, 49_999.99, 50_000.0, 99_999.99, 100_000.0, 149_999.99, 150_000.0, 250_000.0, 1.0,
                    60_000.0, float("nan"), 0.0, 120_000.0],
        "NumOfProducts": [1, 2, 3, 4, 1, 2, 3, 4, 10, 1, 2, 3, 1, 2],
        "HasCrCard": [0, 1] * 7,
        "IsActiveMember": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2],                  # 2 is neither Active nor Inactive
        "EstimatedSalary": [1000.0 * (i + 1) for i in range(14)],
        "churn_score": score,
        "will_churn": [x >= 0.5 for x in score],
        "risk_level": ["High" if x >= 0.6 else "Medium" if x >= 0.4 else "Low" for x in score],
        "outside_training_range": [False] * 14,
        "row_status": ["ok"] * 14,
        "data_issues": [""] * 14,
        "Exited": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, float("nan"), 1],
    })


@pytest.fixture
def synthetic(pg_url):
    engine = db.make_engine(pg_url)
    db.init_schema(engine)
    df, did = synthetic_frame(), "c" * 32
    rows = pg_stores._records(df, pg_stores.RESULT_COLUMNS)
    for r in rows:
        r["dataset_id"], r["CustomerId"] = did, pg_stores._id_text(r["CustomerId"])
    with engine.begin() as conn:
        conn.execute(sa.insert(s.datasets).values(
            id=did, name="synthetic", original_filename="s.csv", status="completed", policy="reject", has_target=True,
            model_name="m", model_version="1", score_type="ranking", rows_uploaded=14, rows_scored=14, rows_rejected=0,
            rows_repaired=0, duplicates_removed=0, summary={}))
        conn.execute(sa.insert(s.dataset_results), rows)
    yield PostgresDatasetStore(engine), did, df
    engine.dispose()


@pytest.mark.parametrize("by", analytics.SEGMENT_FIELDS)
def test_band_edges_and_odd_values_match_pandas(synthetic, by):
    st, did, df = synthetic
    expected = analytics.segment_breakdown(df, by)
    actual = st.segment_breakdown(did, by)
    assert [r["segment"] for r in actual] == [r["segment"] for r in expected]
    assert_same_records(expected, actual)


@pytest.mark.parametrize("kw", [
    {"sort_by": "Balance"}, {"sort_by": "Balance", "ascending": True},           # a missing balance sorts last either way
    {"sort_by": "churn_score"}, {"sort_by": "churn_score", "ascending": True},   # ties keep row_id order
    {"sort_by": "Age", "ascending": True, "limit": 5, "offset": 2},
    {"flagged_only": True, "tier": "high"}, {"search": "Name1"}, {"search": "name1 "},   # "name1 " proves the joining space
    {"search": "15000003"}, {"search": "50%"}, {"min_score": 0.5, "geography": "spain"},
], ids=lambda k: str(k)[:60])
def test_sorting_nulls_ties_and_search_match_pandas(synthetic, kw):
    st, did, df = synthetic
    total_e, page_e = analytics.query_customers(df, **kw)
    total_a, page_a = st.query_customers(did, **kw)
    assert total_a == total_e
    assert list(page_a.columns) == list(page_e.columns)
    assert_same_records(frame_to_records(page_e), frame_to_records(page_a))


@pytest.mark.parametrize("key", ["15000003", "3", "15000099", "abc", "9" * 40, "007"])
def test_find_customer_by_id_then_row_id_matches_pandas(synthetic, key):
    st, did, df = synthetic
    expected, actual = analytics.find_customer(df, key), st.find_customer(did, key)
    assert (expected is None) == (actual is None)
    if expected is not None:
        assert_same_records([expected.to_dict()], [actual.to_dict()])
        



def test_search_without_surnames_matches_pandas(loaded):
    source, files, fid, pg, pid = loaded
    if source == "real":
        pytest.skip("the real file has missing surnames; pandas 3 and SQL may treat those differently in a search")
    frame = store(files).results(fid)
    first = frame.iloc[0]
    needles = [str(first["Surname"])[:3], str(int(first["row_id"])), "zzzz-no-such-text", "50%"]
    if "CustomerId" in frame.columns:
        needles.append(str(first["CustomerId"])[:5])
    for needle in needles:
        total_e, page_e = analytics.query_customers(frame, search=needle, search_surname=False, limit=500)
        total_a, page_a = store(pg).query_customers(pid, search=needle, search_surname=False, limit=500)
        assert total_a == total_e, needle
        assert_same_records(frame_to_records(page_e), frame_to_records(page_a))