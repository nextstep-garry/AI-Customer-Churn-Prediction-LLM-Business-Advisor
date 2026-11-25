# Churn Predictor

Flask API + Streamlit app that scores bank customers for churn risk, one at a time or a whole file at once,
and explains why. Rebuilt from the original prototype following the audit report (see [CHANGES.md](CHANGES.md)).

| Page | What it does |
|---|---|
| **Portfolio dashboard** | Pick an uploaded dataset: KPIs, risk tiers, score distribution, segment breakdowns, customer browser with per-customer explanations, global drivers, model performance (if the file has `Exited`), data quality and drift |
| **Single customer** | Type in one customer, get the score, risk tier and the factors behind it |
| **Upload dataset** | Upload a CSV; the backend cleans, validates, scores and summarises it (each pipeline step is shown) |
| **Assistant** | Chat that answers questions about the project from the RAG knowledge base in `rag/`, with source files shown. Its data tools work separately (see below) |
| **Model** | Version, metrics, limitations, library-version check |

## Run it

Tested on Python 3.12 with the exact pinned versions in the requirements files (the same versions your original
`backend/venv` used). Use two terminals.

```bash
# Terminal 1: backend                       # Windows: venv\Scripts\activate
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python wsgi.py                              # dev server on http://127.0.0.1:5000
# production-style:  waitress-serve --listen=0.0.0.0:5000 wsgi:app     (Windows OK)
#                    gunicorn -w 2 -b 0.0.0.0:5000 wsgi:app            (Linux)

# Terminal 2: frontend
cd frontend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py              # http://localhost:8501
```

Then open **Upload dataset** and try `sample_data/churn_sample_dirty.csv` (synthetic, deliberately messy) or your own
`training/data/churn_raw.csv`. Docker: `docker compose up --build` (written but not built here; see CHANGES.md).
Settings are environment variables, listed in [.env.example](.env.example).

## How a dataset flows through

```
CSV upload --> POST /v1/datasets
   read (BOM/Latin-1/semicolon tolerant) -> match headers loosely -> drop exact duplicates
   -> fix typos/abbreviations (franc, spainn, germny, f/m) -> validate every value against ml/spec.py
   -> reject invalid rows (default) or repair them with median/mode -> features (one vectorised function)
   -> XGBoost -> [calibrator] -> threshold + risk tier -> SHAP drivers, drift, segments, evaluation
   -> stored under data/datasets/<id>/  (raw file kept so it can be re-scored later)
```

10,000 rows take about 0.4 s. "Reject" vs "repair": *reject* sets bad rows aside and lists them; *repair* replaces
bad cells with the file's mode/median (how the notebook treated its training data) and flags each repaired row.
On your real `churn_raw.csv`, repair reproduces the notebook's cleaned data exactly, except Age where the notebook
hard-coded 39 and this uses the file's median (33 rows).

## Replacing the model (the "add the pkl model later" workflow)

Everything model-specific lives in one folder described by `model_manifest.json`; the code never names a file.

1. Export your notebook's `churn_model.pkl` and `preprocessing.pkl` (cells in `training/notebook_saving_cells.py`).
2. Edit `training/model_card.json` (version, metrics, ranges, tiers), then:
   `python training/export_artifacts.py --source <churn_model.pkl> --preprocessing <preprocessing.pkl> --out models`
   This writes the manifest with a SHA-256 for every file and the library versions they were built with.
3. Restart the backend (it refuses to start on a hash or scikit-learn/xgboost version mismatch), then press
   **Re-score** on the Upload page for datasets scored earlier. Or point `MODEL_DIR` at another folder.

The new model must use the same 10 raw input fields and the same preprocessing keys
(`scaler, winsor_bounds, numeric_cols, gender_encoder, geography_encoder`). New input features mean editing
`ml/spec.py` and `ml/features.py`; the golden-fixture tests will tell you exactly what changed.

**Calibrated variant.** The shipped default (`models/`, v1.0.0) is your model as trained: its score is a *ranking
score*, not a probability (it runs about 2x the real churn rate), and the UI says so. `models/calibrated/` (v1.1.0)
adds a Platt calibrator fit on the notebook's validation split, so 0.30 means about 30%. It flags exactly the same
customers. Switch with `MODEL_DIR=models/calibrated`; regenerate with `python training/fit_calibrator.py`, which
refuses to run unless it reproduces the notebook's split (so it cannot be fit on training rows).

## API

Versioned under `/v1`; every error is `{"error": {code, message, details?}, "request_id"}`. Send `X-API-Key` if `API_KEY` is set.

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /ready` | liveness; readiness (model loaded) |
| `GET /v1/model`, `GET /v1/schema` | manifest, metrics, tiers; the input contract (the form is built from it) |
| `POST /v1/predict?explain=true` | one customer: score, tier, flag, warnings, top SHAP drivers |
| `POST /v1/predict/batch` | `{"records": [...]}`, up to `MAX_BATCH_RECORDS` |
| `POST /v1/datasets` | multipart `file`, `name`, `policy=reject\|repair` |
| `GET /v1/datasets`, `GET /v1/datasets/<id>`, `.../summary` | list, metadata, full dashboard JSON |
| `GET /v1/datasets/<id>/customers` | filter (`tier, geography, gender, search, min_score`), sort, page |
| `GET /v1/datasets/<id>/customers/<key>` | one customer (by CustomerId or row_id) with explanation |
| `GET /v1/datasets/<id>/download?kind=predictions\|rejected` | CSV (spreadsheet-formula safe) |
| `POST /v1/datasets/<id>/rescore`, `DELETE /v1/datasets/<id>` | re-run with the current model; delete |
| `GET/DELETE /v1/history` | prediction audit trail (SQLite) |
| `GET /v1/chat/tools`, `POST /v1/chat/tools/call` | the data toolbox (JSON Schema), and run one tool directly |
| `POST /v1/chat` | `{"messages": [...], "dataset_id"?}` returns `{"reply", "sources", "route"}`; routes each question to documents, live data or both |

```bash
curl -X POST localhost:5000/v1/predict -H 'Content-Type: application/json' -d '{"CreditScore":619,"Geography":"France",
 "Gender":"Female","Age":42,"Tenure":2,"Balance":0,"NumOfProducts":1,"HasCrCard":1,"IsActiveMember":1,"EstimatedSalary":101348.88}'
```

## Assistant

`POST /v1/chat` answers from the right source, decided per question by a rule-based router (`services/router.py`):

| Route | Example | Answered by |
|---|---|---|
| `knowledge` | "Why was XGBoost selected?" | the RAG pipeline in `rag/` (embed, ChromaDB, LLM) |
| `data` | "Compare churn risk by country in my uploaded file" | planner, tool, writer (below) |
| `both` | "Which country has the highest churn?" | both, shown in two labelled sections |

A data question takes three steps (`services/planner.py`, `services/tool_answer.py`, `services/assistant.py`):

1. **Plan:** the LLM replies with one JSON tool call. The reply is validated against the tool's JSON Schema; unknown tools,
   bad values and invented customer ids are rejected.
2. **Run:** the tool executes on the dataset the Assistant page shows (`dataset_id`); the LLM never chooses it.
3. **Write:** the LLM words the result from the tool's data only. Rates arrive as ready-made percentages, surnames are
   removed before anything reaches the LLM, and the answer cites `data tool: <name>`.

If the data path fails, the assistant falls back to the documents, then to a hint, and returns a 503 only when nothing
can run. It never invents an answer to hide a failure. `predict_what_if` is deliberately not available to the planner
(a small LLM would invent the ten customer fields); use the Tool explorer for it. Only the last message is used, so
follow-ups such as "and in Spain?" lose their context.

The six tools (`services/tools.py`: portfolio summary, segment breakdown, top-risk customers, customer lookup with
explanation, what-if scoring, list datasets) call the same functions as the dashboard, so chat answers and charts cannot
disagree. `GET /v1/chat/tools` lists them as JSON Schema and `POST /v1/chat/tools/call` runs one; the Assistant page has
a Tool explorer for trying them directly.

Setup: `pip install -r rag/requirements.txt`, set `LLM_PROVIDER`/`LLM_MODEL` (see `.env.example`; for Ollama run
`ollama pull qwen2.5:3b` and keep it running). In Docker, Ollama runs on the host: `docker-compose.yml` points
`OLLAMA_URL` at `host.docker.internal`. After editing `rag/documents` or `rag/knowledge_base`, run
`cd rag && python ingest.py`. Timing on a local 3B model: about 10-30 s for a document answer, 15-40 s for a data answer,
up to a minute for a mixed one; the first request also loads the embedder.

Check answer quality with your real LLM (a data answer is "grounded" only if every number in it is in the data):

```bash
python tests/eval_tools.py               # data questions
python tests/eval_tools.py --with-rag    # plus document and mixed questions
```

## Tests

```bash
pip install -r requirements-dev.txt && python -m pytest -q       # 221 tests, about 45 s
```

Highlights: the audit's 18 invalid payloads are rejected; predictions match the **original** implementation to 1e-16
on 304 golden rows (`tests/fixtures/`); SHAP contributions add up to the served score; the real `churn_raw.csv` is
cross-checked against the notebook's own cleaning; the calibrator reproduces the audit's Brier/ECE/tier numbers;
the Streamlit pages are driven end to end against a live backend; the assistant's router, planner, tool answers and
fallbacks are tested with a fake LLM, and `tests/eval_tools.py` is itself tested to catch invented numbers.

## Security and data notes

Set `API_KEY` before exposing the API; the rate limiter is per process (limit at a reverse proxy when running several
workers); terminate TLS at a proxy. Uploaded customer data is stored unencrypted under `data/` (git-ignored) until you
delete it. There are no user accounts. Downloaded CSVs neutralise cells starting with `= + - @`.

## Layout

```
backend/churn_api/  ml/ (spec, bundle, features, cleaning, pipeline, explain, drift)
                    services/ (analytics, tools, router, planner, tool_answer, assistant)
                    api/ (blueprints)  storage/ (datasets, history)  config, errors, security, logging_config
frontend/           streamlit_app.py  views/ (5 pages)  churn_ui/ (client, charts, theme)
models/             deployable artifacts + manifest;  models/calibrated/ (opt-in)
rag/                the assistant's document pipeline: ingest, retrieve, prompt, generate, rag_pipeline,
                    documents/, knowledge_base/, chroma_db/ (the vector index)
training/           export_artifacts.py, fit_calibrator.py, notebook_cleaning.py, model_card.json, data/churn_raw.csv
tests/ (incl. eval_tools.py)  sample_data/  docker/  .github/workflows/
```
