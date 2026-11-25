# What changed, mapped to the audit report

Status legend: **Done** = implemented and covered by tests · **Partial** = some of it, gap stated · **Not done** = out of scope here.
Nothing below was tested on Docker, on Windows, or on Python 3.13 (see "Not verified").

## The 26 audit requirements

| # | Requirement | Status | What was done / what is left |
|---|---|---|---|
| 1 | Architecture | Done | Package with `ml/ api/ services/ storage/`, app factory, blueprints, one shared feature function. The 291 dead lines are gone; venvs are gone (`.gitignore`). |
| 2 | Frontend | Done | Five-page Streamlit app; `BACKEND_URL` from env; every backend call guarded; no raw tracebacks. Input form is built from `GET /v1/schema`. |
| 3 | Input validation | Done | Pydantic schema (ranges from training data, `extra=forbid`, NaN/Inf/null rejected). All 18 audit payloads now return 400 with field-level errors (`tests/test_validation.py`). |
| 4 | REST API | Done | `/v1` routes, JSON errors for 404/405/413/429/500. OpenAPI/Swagger is **not** generated (see #21). |
| 5 | Preprocessing | Done | Vectorised, and proven identical to your original per-row code on 304 golden rows (max diff 7e-15). |
| 6 | Feature engineering | Unchanged | Same formulas. Winsorising discrete columns (NumOfProducts capped at 3.5) is documented as a limitation, not changed. |
| 7 | Feature selection | Not done | Analysis still lives only in the notebook. |
| 8 | ML model | Unchanged | No retraining. `export_artifacts.py` produces the deployable files from your notebook output; `train.py` (full retraining) is **not** written. |
| 9 | Evaluation | Partial | Metrics in the manifest and Model page; evaluation (AUC, PR, confusion, ROC) computed on any labelled upload; `calibration_report.json`. No standalone `evaluate.py` with plots. |
| 10 | Probability and risk | Done | Risk tiers everywhere. Default model: relabelled "risk score" (not a probability) with interim tiers read off the audit's calibration table. `models/calibrated`: Platt calibrator, tiers 15%/40%, reproduces the audit's Brier 0.1627 to 0.1009, ECE 0.238 to 0.027, tiers 1,224/413/362 with observed churn 6.5/18.9/68.8%. |
| 11 | SHAP | Done | Per-customer and dataset-level. Uses XGBoost's built-in TreeSHAP on the early-stopping trees, not the `shap` package (same algorithm, no numba dependency). Additivity is tested. |
| 12 | Single prediction | Done | Score, tier, flag, warnings for out-of-training-range values, drivers, history. |
| 13 | Batch prediction | Done | `/v1/predict/batch` and file upload. 10,000 rows in about 0.4 s (audit measured about 98 s with the per-row loop). |
| 14 | Results and download | Done | Predictions and rejected-rows CSVs; spreadsheet-formula injection neutralised. |
| 15 | Error handling | Done | One envelope, request ids, no internal text leaked, API stays up if the model fails to load (`/ready` reports it). |
| 16 | Logging | Done | JSON logs with request id, per-request latency, dataset and prediction events. |
| 17 | Versioning | Partial | `model_manifest.json` with SHA-256 of every artifact, library versions, threshold; load fails on hash or scikit-learn/xgboost mismatch. Requirements re-pinned (scikit-learn 1.6.1). No MLflow/registry. |
| 18 | History | Done | SQLite audit trail (stdlib `sqlite3`, not SQLAlchemy). |
| 19 | Security | Partial | API key (constant-time), rate limit (per process only), upload size and row caps, debug off by default, path-traversal-proof ids, CSV-injection safe. No user accounts, TLS or encryption at rest. |
| 20 | Testing | Done | 221 tests (backend, real-data, calibration, and the Streamlit pages against a live server). |
| 21 | API documentation | Partial | README table plus self-describing `/v1/schema` and `/v1/chat/tools`. No OpenAPI file. |
| 22 | Project documentation | Done | README, this file, `models/README.md`, `.env.example`. |
| 23 | Deployment | Partial | Dockerfiles, compose, liveness (`/health`) and readiness (`/ready`) written. **Not built or run** (no Docker in the build sandbox). |
| 24 | Monitoring and drift | Partial | Per-upload drift against the training scaler's mean/std and out-of-range counts. No PSI/KS on logged inputs, no Prometheus. |
| 25 | CI/CD | Partial | GitHub Actions runs the tests, plus an advisory `pip-audit`. No ruff, no Docker build. |
| 26 | Scalability | Partial | Vectorised batch path. Measured: a 200,000-row / 15 MB upload runs the whole pipeline in about 4 s (Flask test client, one process). Uploads are synchronous; rate limiter and dataset cache are per process; not load-tested with concurrent users. |

## Things found while doing it (not in the audit)

* The real raw file's `f`/`m` gender values and `unknown`/`???`/`nan` geography tokens would have been rejected by a naive cleaner; the notebook treats them as aliases/missing, and so does the cleaner now.
* "Training ranges" copied from the report were rounded (`Balance` max 250,898 vs real 250,898.09) and flagged two training rows as out of range; they are now computed from the data.
* Six `CustomerId` values in the real file are junk (`15589358X`, `NaN`): ids are passed through untouched, so they neither break the upload nor get "fixed".
* Scores were `float32` end to end, so a dashboard average and the same figure via the chat tool differed by 2e-10; scores are `float64` now.

## Assistant (added after the audit)

| What | Status | Notes |
|---|---|---|
| RAG answers about the project | Done | `rag/`: heading-aware chunks, bge-small embeddings, ChromaDB, similarity gate 0.50, swappable LLM (Ollama, Hugging Face, OpenAI, Gemini). 55 of 64 test questions passed, all 9 unanswerable ones refused. |
| Router | Done | Rule-based, no LLM call. Defaults to documents when unsure. |
| Tool planner and writer | Done | JSON tool call validated against each tool's schema; answers use only the tool's data; surnames removed; rates pre-formatted. |
| Evaluation of the data path | Done | `tests/eval_tools.py` checks route, tool, facts and that every number in an answer is in the data. Not yet run against a real LLM in this environment. |
| What-if questions in chat | Not done | The planner may not use `predict_what_if`; use the Tool explorer. |
| Conversation memory | Not done | Only the last message is used, so follow-ups such as "and in Spain?" lose context. |
| Hybrid search / re-ranking | Not done | Vector search only; 3 feature-selection questions missed in the RAG evaluation. |

## Decisions for you to review

1. **Default model.** `models/` (raw score, v1.0.0) is what you trained. `models/calibrated` gives real probabilities with audit-validated tiers and flags the same customers. I recommend switching (`MODEL_DIR=models/calibrated`) but did not, because it changes the number your README quotes (0.5517 becomes 0.3142).
2. **Invalid rows** default to *reject* (2.3% of your raw file, 227 rows). *Repair* keeps them; on your data it matches the notebook except Age (median vs the notebook's hard-coded 39).
3. **Age 18-92.** Matches the training population (min 18); the notebook itself accepted 1-92.
4. **Flag threshold** keeps your 0.5 decision boundary (22% on the calibrated scale). Validation-optimal is about 0.28 calibrated (about 0.53-0.54 raw); the gain is small, so I did not move it.
5. **`API_KEY` is empty by default**, so the API is open; the backend logs a warning until you set it.

## Not done

Retraining pipeline (`train.py`), MLflow, feature-selection changes, dropping winsorisation (the audit found it neutral), OpenAPI, ruff, Prometheus/locust, PSI/KS drift on logged predictions, async jobs for very large uploads.

## Not verified

Docker images; Windows and Python 3.13 (your original environment; the pinned versions are the same ones your venv used, but I ran 3.12 on Linux); a real browser (Streamlit pages were driven headlessly with `AppTest`, so layout and styling have not been eyeballed).
