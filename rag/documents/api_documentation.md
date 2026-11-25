---
doc_id: PK-12
title: Backend API Documentation
status: FACTUAL - FROM THE CHURN APP README AND CHANGES (knowledge_base/project_docs) AND THE NOTEBOOK
synthetic: false
source: knowledge_base/project_docs/README.md, CHANGES.md, models_README.md, model_card.json; Copy_of_Churn_Modelling_Full_Pipeline.ipynb Sections 26-27
created: 2026-10-08
version: 0.2
---

# Backend API Documentation

## 1. What the backend is
The churn app has a Flask backend API and a Streamlit frontend. The backend runs on http://127.0.0.1:5000 (started with `python wsgi.py`) and the frontend on http://localhost:8501 (started with `streamlit run streamlit_app.py`). The churn notebook itself contains no API code: it only saves the model. Full details are in the app README (project_docs/README.md).

## 2. Endpoints
All endpoints are versioned under `/v1`.
- `POST /v1/predict?explain=true` scores one customer and returns the score, risk tier, flag, warnings and the top SHAP drivers.
- `POST /v1/predict/batch` scores many customers: `{"records": [...]}`, up to MAX_BATCH_RECORDS.
- `GET /v1/model` returns the model manifest, metrics and tiers. `GET /v1/schema` returns the input contract.
- `GET /health` is a liveness check and `GET /ready` checks that the model is loaded.
- `POST /v1/datasets` uploads a CSV (policy reject or repair). `GET /v1/datasets` lists datasets, and `GET /v1/datasets/<id>/summary` returns the dashboard data.
- `GET /v1/history` and `DELETE /v1/history` show and clear the prediction audit trail (SQLite).
- `GET /v1/chat/tools`, `POST /v1/chat/tools/call` and `POST /v1/chat` are the chatbot toolbox. `POST /v1/chat` returns 501 for now.
The RAG project assistant (a question-answering endpoint about the project) is planned and is kept separate from the prediction endpoints: it never calls the model, and `/v1/predict` never calls the RAG pipeline.

## 3. Inputs the prediction endpoint needs
A prediction needs the 10 raw customer fields: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember and EstimatedSalary. Example request body from the README: {"CreditScore":619,"Geography":"France","Gender":"Female","Age":42,"Tenure":2,"Balance":0,"NumOfProducts":1,"HasCrCard":1,"IsActiveMember":1,"EstimatedSalary":101348.88}.

## 4. Input validation and valid values
The backend validates every request with a Pydantic schema: ranges come from the training data, extra fields are forbidden, and NaN, Inf and null are rejected. An invalid payload returns HTTP 400 with field-level errors. Every error has the form {"error": {code, message, details}, "request_id"}. Valid values:
| Field | Valid values |
|---|---|
| CreditScore | 350 to 850 |
| Geography | France, Germany or Spain |
| Gender | female or male |
| Age | 18 to 92 (the training population starts at 18; the notebook cleaning accepted 1 to 92) |
| Tenure | 0 to 10 |
| Balance | 0 to 250,898.09 (training maximum) |
| NumOfProducts | 1 to 4 |
| HasCrCard | 0 or 1 |
| IsActiveMember | 0 or 1 |
| EstimatedSalary | 11.58 to 199,992.48 (training range) |

## 5. Order of processing at prediction time
The model expects the same transformations as training, in the same order:
1. Compute Zero_Balance_Flag and Balance_Salary_Ratio (Balance divided by EstimatedSalary) from the raw, unscaled values.
2. Encode Gender (female = 0, male = 1) and Geography as Geography_germany and Geography_spain (France is the baseline, both 0).
3. Apply the saved winsorisation bounds to the six numeric columns (CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary).
4. Apply the saved StandardScaler to those same six columns.
5. Arrange the 13 features in the saved feature order and call the XGBoost model.
6. Compare the model output with the threshold (0.50) to get the flag.

## 6. Output: a risk score, not a probability
The default model (XGBoost version 1.0.0, in `models/`) returns a raw ranking score, not a probability: it runs about 2 times the real churn rate, and the UI says so. Risk tiers use the raw score: Low below 0.50, Medium 0.50 to 0.60, High 0.60 and above. The optional calibrated variant (`models/calibrated`, version 1.1.0) adds a Platt calibrator, so 0.30 means about 30%, with tiers at 15% and 40%. It flags exactly the same customers. Switch with `MODEL_DIR=models/calibrated`.

## 7. Model artifacts
The API loads exactly the files listed in model_manifest.json: `xgb_model.json` (XGBoost in its native format) and `preprocessing.pkl` (fitted encoders, winsorisation bounds, scaler), plus an optional `calibrator.pkl`. It refuses to start if a SHA-256 hash or the scikit-learn or xgboost version does not match. To replace the model, re-export from the notebook with `training/export_artifacts.py`, restart the API, then use Re-score on the Upload page.

## 8. Security notes
`API_KEY` is empty by default, so the API is open until you set it (the backend logs a warning); send the key in the `X-API-Key` header. The rate limiter is per process. Uploaded customer data is stored unencrypted under `data/` until deleted. There are no user accounts, and TLS should be terminated at a reverse proxy.
