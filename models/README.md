# models/

Deployable artifacts. The API loads exactly what `model_manifest.json` lists and refuses to start if a
SHA-256 or a library version does not match.

| File | What |
|---|---|
| `model_manifest.json` | version, threshold, feature order, library versions, artifact hashes, metrics, training ranges, risk tiers |
| `xgb_model.json` | XGBoost in its native, version-portable format |
| `preprocessing.pkl` | fitted encoders, winsorisation bounds, scaler (scikit-learn 1.6.1) |
| `calibrator.pkl` | optional Platt calibrator (only in calibrated model folders) |
| `source/` | the notebook's original `churn_model.pkl` / `model_metadata.pkl`; inputs to `training/export_artifacts.py`, never loaded by the API |

Replace the model: re-export from the notebook with `training/export_artifacts.py`, restart the API, then use
*Re-score* on the Upload page for datasets scored earlier. Point `MODEL_DIR` at another folder to switch models.
