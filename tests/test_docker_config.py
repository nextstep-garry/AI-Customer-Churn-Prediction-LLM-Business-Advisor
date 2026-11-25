"""docker-compose.yml, the backend image and .dockerignore are wired for PostgreSQL. Reads the files only: no Docker needed."""
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def env_of(service: dict) -> dict:
    raw = service.get("environment") or {}
    if isinstance(raw, list):
        return dict(item.split("=", 1) for item in raw)
    return {k: ("" if v is None else str(v)) for k, v in raw.items()}


def test_the_stack_has_the_three_services(compose):
    assert {"postgres", "backend", "frontend"} <= set(compose["services"])


def test_postgres_starts_by_default_and_reports_health(compose):
    pg = compose["services"]["postgres"]
    assert "profiles" not in pg and pg["image"].startswith("postgres:")
    assert pg["healthcheck"]["test"] and pg["restart"] == "unless-stopped"


def test_the_database_password_is_required_not_defaulted(compose):
    assert env_of(compose["services"]["postgres"])["POSTGRES_PASSWORD"].startswith("${POSTGRES_PASSWORD:?")


def test_database_data_lives_in_a_named_volume(compose):
    assert "pg_data:/var/lib/postgresql/data" in compose["services"]["postgres"]["volumes"]


def test_the_database_is_not_published_to_the_network(compose):
    ports = compose["services"]["postgres"].get("ports", [])
    assert all(str(p).startswith("127.0.0.1:") for p in ports)           # none at all is the default


def test_the_backend_connects_to_the_postgres_service(compose):
    url = env_of(compose["services"]["backend"])["DATABASE_URL"]
    assert url.startswith("postgresql+psycopg://") and "@postgres:5432/" in url and "${POSTGRES_PASSWORD}" in url


def test_start_order_waits_for_healthy_services(compose):
    assert compose["services"]["backend"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert compose["services"]["frontend"]["depends_on"]["backend"]["condition"] == "service_healthy"


def test_no_secret_is_written_into_the_compose_file(compose):
    for name, service in compose["services"].items():
        for key, value in env_of(service).items():
            if re.search(r"PASSWORD|SECRET|TOKEN|KEY", key):
                assert value == "" or value.startswith("${"), f"{name}: {key} must come from the environment"


def test_every_named_volume_is_declared(compose):
    for name, service in compose["services"].items():
        for entry in service.get("volumes", []):
            source = str(entry).split(":")[0]
            if not source.startswith((".", "/", "~")) and "/" not in source and "\\" not in source:
                assert source in compose["volumes"], f"{name} uses undeclared volume '{source}'"


def test_env_example_documents_the_password_without_a_value():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    match = re.search(r"^POSTGRES_PASSWORD=(.*)$", text, re.M)
    assert match and match.group(1).strip() == ""                       # documented, but never a real value


def test_backend_image_contains_the_code_the_database_needs():
    dockerfile = (ROOT / "docker" / "backend.Dockerfile").read_text(encoding="utf-8")
    for needed in ("COPY backend backend", "COPY rag rag", "rag/requirements.txt"):
        assert needed in dockerfile, needed


def test_dockerignore_keeps_secrets_and_junk_out_but_keeps_the_migrations():
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "venv" in ignore and ".env" in ignore                         # no virtual environments, no secrets
    assert "sql" not in ignore.lower()                                   # nothing that could drop the migration files
    assert list((ROOT / "backend" / "churn_api" / "storage" / "sql_migrations").glob("*.sql"))