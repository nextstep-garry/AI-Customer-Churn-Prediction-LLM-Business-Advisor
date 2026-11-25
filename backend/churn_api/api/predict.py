import logging

from flask import g, jsonify, request

from ..config import Config
from ..errors import ApiError
from ..schemas import BatchIn, CustomerIn
from . import v1
from .deps import require_model

log = logging.getLogger(__name__)


def _json_body():
    body = request.get_json(silent=True)
    if body is None:
        raise ApiError(400, "invalid_json", "Request body must be valid JSON (Content-Type: application/json).")
    return body


@v1.post("/predict")
def predict_one():
    """Score one customer. `?explain=true` adds the top factors behind the score (TreeSHAP)."""
    c = require_model()
    record = CustomerIn.model_validate(_json_body()).model_dump()
    explain = request.args.get("explain", "false").lower() in {"1", "true", "yes"}
    result = c.pipeline.predict_records([record], explain=explain)[0]
    try:
        c.history.add(request_id=g.get("request_id"), model_name=c.bundle.model_name,
                      model_version=c.bundle.model_version, record=record, result=result)
    except Exception:  # noqa: BLE001 -- an audit-log failure must not fail the prediction
        log.exception("history_write_failed")
    log.info("prediction", extra={"risk_level": result["risk_level"], "score": result["churn_score"]})
    return jsonify({**result, "model_name": c.bundle.model_name, "model_version": c.bundle.model_version,
                    "request_id": g.get("request_id")})


@v1.post("/predict/batch")
def predict_batch():
    """Score many JSON records at once: {"records": [ {...}, ... ]}. For files, use POST /v1/datasets."""
    c = require_model()
    max_n = c.cfg.MAX_BATCH_RECORDS
    body = _json_body()
    if isinstance(body, dict) and isinstance(body.get("records"), list) and len(body["records"]) > max_n:
        raise ApiError(400, "too_many_records", f"At most {max_n} records per request; upload a file instead.")
    records = [r.model_dump() for r in BatchIn.model_validate(body).records]
    results = c.pipeline.predict_records(records)
    return jsonify({"count": len(results), "model_name": c.bundle.model_name,
                    "model_version": c.bundle.model_version, "predictions": results})
