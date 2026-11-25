"""Dataset endpoints: upload -> run the whole pipeline -> browse results."""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import pandas as pd
from flask import Response, jsonify, request

from ..errors import ApiError
from ..ml.exceptions import DatasetError
from ..services import privacy, queries
from ..storage.datasets import DatasetNotFound
from ..utils import frame_to_records
from . import v1
from .deps import ctx, require_model

log = logging.getLogger(__name__)


def read_csv_bytes(data: bytes, max_rows: int) -> pd.DataFrame:
    """Parse an uploaded CSV, tolerating UTF-8 BOM / Latin-1 and semicolon-separated exports."""
    if not data.strip():
        raise DatasetError("The uploaded file is empty.")
    last_exc = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=encoding, nrows=max_rows + 1, low_memory=False)
            if df.shape[1] == 1 and ";" in str(df.columns[0]):
                df = pd.read_csv(io.BytesIO(data), encoding=encoding, sep=";", nrows=max_rows + 1, low_memory=False)
            break
        except UnicodeDecodeError as exc:
            last_exc = exc
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DatasetError(f"Could not parse the file as CSV: {exc}") from None
    else:
        raise DatasetError(f"Could not decode the file: {last_exc}")
    if len(df) > max_rows:
        raise DatasetError(f"The file has more than {max_rows:,} rows (the configured limit).")
    return df


def _get_or_404(fn, dataset_id):
    try:
        return fn(dataset_id)
    except DatasetNotFound:
        raise ApiError(404, "dataset_not_found", f"No dataset with id '{dataset_id}'.") from None


def _surname_requested(cfg) -> bool:
    """True when the request asks for surnames (?include_surname=true) and the operator allows it."""
    asked = request.args.get("include_surname", "false").lower() in {"1", "true", "yes"}
    if asked and not cfg.ALLOW_SURNAME_ACCESS:
        raise ApiError(403, "surname_not_allowed", "Surname access is disabled on this server (ALLOW_SURNAME_ACCESS=false).")
    return asked


def _csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralise spreadsheet-formula injection (=, +, -, @) in text cells before download."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            s = df[col].astype("string")
            risky = s.str.match(r"^[=+\-@\t\r]").fillna(False).astype(bool)
            df[col] = s.where(~risky, "'" + s).astype(object)
    return df


@v1.post("/datasets")
def upload_dataset():
    """multipart/form-data: file=<csv>, name=<optional>, policy=reject|repair"""
    c = require_model()
    file = request.files.get("file")
    if file is None or not file.filename:
        raise ApiError(400, "missing_file", "Attach a CSV file as multipart field 'file'.")
    if not file.filename.lower().endswith((".csv", ".txt")):
        raise ApiError(415, "unsupported_media_type", "Only .csv files are supported.")
    policy = request.form.get("policy", "reject")
    if policy not in {"reject", "repair"}:
        raise ApiError(400, "invalid_policy", "policy must be 'reject' or 'repair'.")
    name = (request.form.get("name") or Path(file.filename).stem).strip()[:80] or "dataset"

    raw_bytes = file.read()
    df = read_csv_bytes(raw_bytes, c.cfg.MAX_ROWS)
    run = c.pipeline.run_dataset(df, policy)
    meta = c.store.save(name=name, original_filename=Path(file.filename).name[:120], raw_bytes=raw_bytes,
                        run=run, model=c.bundle.info())
    log.info("dataset_saved", extra={"dataset_id": meta["id"], "rows": meta["rows"]})
    return jsonify({"dataset": meta, "stages": run.stages}), 201


@v1.get("/datasets")
def list_datasets():
    return jsonify({"datasets": ctx().store.list()})


@v1.get("/datasets/<dataset_id>")
def get_dataset(dataset_id):
    return jsonify(_get_or_404(ctx().store.meta, dataset_id))


# @v1.get("/datasets/<dataset_id>/summary")
# def dataset_summary(dataset_id):
#     c = ctx()
#     return jsonify({"dataset": _get_or_404(c.store.meta, dataset_id), **_get_or_404(c.store.summary, dataset_id)})

@v1.get("/datasets/<dataset_id>/summary")
def dataset_summary(dataset_id):
    c = ctx()
    include = _surname_requested(c.cfg)
    summary = _get_or_404(c.store.summary, dataset_id)
    return jsonify({"dataset": _get_or_404(c.store.meta, dataset_id),
                    **(summary if include else privacy.strip_json(summary))})

@v1.get("/datasets/<dataset_id>/customers")
def dataset_customers(dataset_id):
    c = ctx()
    include = _surname_requested(c.cfg)
    a = request.args
    total, page = _get_or_404(lambda ds: queries.query_customers(
        c.store, ds, tier=a.get("tier"), geography=a.get("geography"), gender=a.get("gender"),
        min_score=a.get("min_score", type=float), flagged_only=a.get("flagged_only", "false").lower() == "true",
        search=a.get("search"), search_surname=include, sort_by=a.get("sort_by", "churn_score"),
        ascending=a.get("order", "desc").lower() == "asc",
        limit=max(1, min(a.get("limit", default=50, type=int), 500)),
        offset=max(0, a.get("offset", default=0, type=int))), dataset_id)
    return jsonify({"total": total, "rows": frame_to_records(page if include else privacy.strip_frame(page))})


@v1.get("/datasets/<dataset_id>/customers/<key>")
def dataset_customer(dataset_id, key):
    c = require_model()
    include = _surname_requested(c.cfg)
    row = _get_or_404(lambda ds: queries.find_customer(c.store, ds, key), dataset_id)
    if row is None:
        raise ApiError(404, "customer_not_found", f"No customer '{key}' in this dataset (CustomerId or row_id).")
    shown = row if include else privacy.strip_row(row)
    out = {"customer": frame_to_records(shown.to_frame().T)[0]}
    if request.args.get("explain", "true").lower() == "true":
        out["explanation"] = c.pipeline.explain_customer(row)
    return jsonify(out)


@v1.get("/datasets/<dataset_id>/download")
def download(dataset_id):
    c = ctx()
    kind = request.args.get("kind", "predictions")
    if kind not in {"predictions", "rejected"}:
        raise ApiError(400, "invalid_kind", "kind must be 'predictions' or 'rejected'.")
    include = _surname_requested(c.cfg)
    meta = _get_or_404(c.store.meta, dataset_id)
    df = _get_or_404(c.store.results if kind == "predictions" else c.store.rejected, dataset_id)
    if not include:
        df = privacy.strip_frame(df)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", meta["name"]).strip("_") or "dataset"
    return Response(_csv_safe(df).to_csv(index=False), mimetype="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{slug}_{kind}.csv"'})


@v1.post("/datasets/<dataset_id>/rescore")
def rescore(dataset_id):
    """Re-run the pipeline on the stored upload with the CURRENT model (use after replacing the model files)."""
    c = require_model()
    meta = _get_or_404(c.store.meta, dataset_id)
    df = read_csv_bytes(_get_or_404(c.store.raw_bytes, dataset_id), c.cfg.MAX_ROWS)
    run = c.pipeline.run_dataset(df, meta["policy"])
    return jsonify({"dataset": c.store.replace_results(dataset_id, run, c.bundle.info()), "stages": run.stages})


@v1.delete("/datasets/<dataset_id>")
def delete_dataset(dataset_id):
    _get_or_404(ctx().store.delete, dataset_id)
    return "", 204