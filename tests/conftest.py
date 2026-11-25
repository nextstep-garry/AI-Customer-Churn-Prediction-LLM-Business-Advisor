import atexit
import io
import json
import os
import tempfile
import uuid
from pathlib import Path

import pytest

from churn_api import create_app
from churn_api.config import Config
from churn_api.ml.bundle import ModelBundle

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
SAMPLES = ROOT / "sample_data"
os.environ.pop("DATABASE_URL", None) 

VALID = dict(CreditScore=619, Geography="France", Gender="Female", Age=42, Tenure=2, Balance=0.0,
             NumOfProducts=1, HasCrCard=1, IsActiveMember=1, EstimatedSalary=101348.88)


@pytest.fixture(scope="session")
def bundle():
    return ModelBundle.load(ROOT / "models")


def make_app(tmp_path, bundle=None, **overrides):
    """A test app. With CHURN_TEST_BACKEND=postgres every app gets its own private PostgreSQL schema, so the whole
    suite can be run against PostgreSQL unchanged (see the PostgreSQL section at the end of this file)."""
    opts = dict(DATA_DIR=tmp_path, API_KEY="", RATE_LIMIT_PER_MIN=0, LOG_LEVEL="CRITICAL", MODEL_DIR=ROOT / "models",
                DATABASE_URL="")
    if os.getenv("CHURN_TEST_BACKEND") == "postgres" and "DATABASE_URL" not in overrides:
        overrides["DATABASE_URL"] = _fresh_schema_url()
    opts.update(overrides)
    app = create_app(Config.from_env(**opts), bundle=bundle)
    engine = app.extensions["churn"].db
    if engine is not None:
        _PG.setdefault("engines", []).append(engine)           # closed after the test, see _release_database_connections
    return app


@pytest.fixture
def app(tmp_path, bundle):
    return make_app(tmp_path, bundle)


@pytest.fixture
def client(app):
    return app.test_client()


def upload(client, source, name="test", policy="reject", filename="data.csv"):
    """POST a CSV (path or bytes) to /v1/datasets."""
    data = source.read_bytes() if isinstance(source, Path) else source
    return client.post("/v1/datasets", data={"file": (io.BytesIO(data), filename), "name": name, "policy": policy},
                       content_type="multipart/form-data")


@pytest.fixture
def dirty_dataset(client):
    r = upload(client, SAMPLES / "churn_sample_dirty.csv", "dirty")
    assert r.status_code == 201, r.json
    return r.json["dataset"]["id"]


def use_backend(url, monkeypatch=None):
    """Point the Streamlit UI at `url`. Clears Streamlit caches so nothing from another backend leaks in."""
    import sys
    if str(ROOT / "frontend") not in sys.path:
        sys.path.insert(0, str(ROOT / "frontend"))
    import streamlit as st
    from churn_ui import config
    if monkeypatch is not None:
        monkeypatch.setattr(config, "BACKEND_URL", url)
    else:
        config.BACKEND_URL = url
    st.cache_resource.clear()
    st.cache_data.clear()


# ----------------------------------------------------------------------------------------------- PostgreSQL for tests
# One PostgreSQL database, one private SCHEMA per test app. Creating a schema is near-instant; creating a database is
# slow (about 17 s each on a Windows machine), and the whole suite can be run on PostgreSQL (CHURN_TEST_BACKEND=postgres).
_PG: dict = {}


def _psycopg_url(url: str) -> str:
    return url.replace("postgresql://", "postgresql+psycopg://", 1) if url.startswith("postgresql://") else url


def _admin_url() -> str:
    """TEST_DATABASE_URL if set, otherwise a throw-away server started with `pgserver` (Linux/macOS). Once per run."""
    if "url" not in _PG:
        explicit = os.getenv("TEST_DATABASE_URL")
        if explicit:
            _PG["url"] = _psycopg_url(explicit)
        else:
            pgserver = pytest.importorskip("pgserver", reason="no PostgreSQL: set TEST_DATABASE_URL or `pip install pgserver`")
            server = pgserver.get_server(tempfile.mkdtemp(prefix="pgtest_"))
            atexit.register(server.cleanup)
            _PG["url"] = _psycopg_url(server.get_uri())
        atexit.register(_drop_schemas)
    return _PG["url"]


def _admin_engine():
    import sqlalchemy as sa
    if "admin" not in _PG:
        _PG["admin"] = sa.create_engine(_admin_url(), isolation_level="AUTOCOMMIT")
    return _PG["admin"]


def _fresh_schema_url() -> str:
    """A URL whose connections see only a brand-new, empty schema (dropped when the test run ends)."""
    import sqlalchemy as sa
    name = "s_" + uuid.uuid4().hex[:12]
    with _admin_engine().connect() as c:
        c.execute(sa.text(f'CREATE SCHEMA "{name}"'))
    _PG.setdefault("schemas", []).append(name)
    url = sa.make_url(_admin_url()).update_query_dict({"options": f"-csearch_path={name}"})
    return url.render_as_string(hide_password=False)


def _drop_schemas() -> None:
    import sqlalchemy as sa
    for engine in _PG.pop("engines", []):
        engine.dispose()
    try:
        with _admin_engine().connect() as c:
            for name in _PG.get("schemas", []):
                c.execute(sa.text(f'DROP SCHEMA IF EXISTS "{name}" CASCADE'))      # each in its own transaction
    except Exception:                                                  # noqa: BLE001  (the server may already be gone)
        pass
    if "admin" in _PG:
        _PG["admin"].dispose()


@pytest.fixture(scope="session")
def pg_admin_url():
    """A PostgreSQL server that may create databases. Only the few tests that need a real database use this."""
    return _admin_url()


@pytest.fixture
def pg_url():
    """The URL of a brand-new, empty, private schema in the test database. Fast: no database is created."""
    return _fresh_schema_url()


@pytest.fixture(autouse=True)
def _release_database_connections():
    """Close every test app's database connections after each test. Without this, hundreds of apps keep connections
    open and PostgreSQL (default limit 100) runs out. A disposed engine reconnects on demand, so shared apps still work."""
    yield
    for engine in _PG.pop("engines", []):
        engine.dispose()