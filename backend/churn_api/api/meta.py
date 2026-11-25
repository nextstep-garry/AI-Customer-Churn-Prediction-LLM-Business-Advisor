from flask import jsonify

from ..ml import spec
from ..services.analytics import SEGMENT_FIELDS
from . import ops, v1
from .deps import ctx, require_model


@ops.get("/health")
def health():
    """Liveness: the process is up. Says nothing about the model."""
    return jsonify({"status": "ok"})


@ops.get("/ready")
def ready():
    """Readiness: the model loaded and, when DATABASE_URL is set, the database answers."""
    c = ctx()
    if c.pipeline is None:
        return jsonify({"status": "unavailable", "detail": c.startup_error}), 503
    body = {"status": "ready", "model": {"name": c.bundle.model_name, "version": c.bundle.model_version,
                                         "score_type": c.bundle.score_type}}
    if c.cfg.DATABASE_URL:
        problem = c.ensure_database()
        if problem:
            return jsonify({"status": "unavailable", "detail": f"database: {problem}"}), 503
        body["database"] = "ok"
    return jsonify(body)
    # """Readiness: the model loaded and the data directory is usable."""
    # c = ctx()
    # if c.pipeline is None:
    #     return jsonify({"status": "unavailable", "detail": c.startup_error}), 503
    # return jsonify({"status": "ready", "model": {"name": c.bundle.model_name, "version": c.bundle.model_version,
    #                                              "score_type": c.bundle.score_type}})


@v1.get("/model")
def model_info():
    return jsonify(require_model().bundle.info())


@v1.get("/schema")
def schema():
    """The input contract. The Streamlit form builds its widgets from this, so ranges live in one place."""
    return jsonify({"fields": spec.field_specs(), "optional_dataset_columns": [*spec.ID_COLUMNS, spec.TARGET_COLUMN],
                    "segments": SEGMENT_FIELDS})
