"""Turn the notebook's saved objects into the deployable, versioned artifact set.

    python training/export_artifacts.py \
        --source models/source/churn_model.pkl \
        --preprocessing models/preprocessing.pkl \
        --out models

Writes into --out:
    xgb_model.json         XGBoost in its native, version-portable format
    model_manifest.json    version, threshold, feature order, library versions, SHA-256 of every
                           artifact, metrics, training ranges and risk tiers. The API refuses to
                           start if a hash or library version does not match.

This is the single place that produces what the backend loads (audit finding: "no script produces
the loaded files"). To ship a new model: re-export from the notebook, run this, restart the API.
Curated facts (metrics, ranges, tiers) live in training/model_card.json; computed facts are derived here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pickled_sklearn_version(obj) -> str | None:
    """scikit-learn stamps its version into every estimator's pickled state."""
    try:
        return obj.__getstate__().get("_sklearn_version")
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="notebook output: churn_model.pkl")
    ap.add_argument("--preprocessing", type=Path, required=True, help="notebook output: preprocessing.pkl")
    ap.add_argument("--out", type=Path, required=True, help="deployable artifact directory")
    ap.add_argument("--model-card", type=Path, default=HERE / "model_card.json")
    ap.add_argument("--calibrator", type=Path, default=None, help="optional calibrator.pkl (see fit_calibrator.py)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="decision threshold on the SERVED score scale (default: the notebook's raw-score threshold)")
    args = ap.parse_args()

    card = json.loads(args.model_card.read_text())
    src = joblib.load(args.source)
    prep = joblib.load(args.preprocessing)
    model = src["model"]

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "xgb_model.json"
    model.save_model(model_path)

    prep_path = args.out / "preprocessing.pkl"
    if args.preprocessing.resolve() != prep_path.resolve():
        prep_path.write_bytes(args.preprocessing.read_bytes())

    artifacts = {
        "model": {"file": model_path.name, "sha256": sha256(model_path)},
        "preprocessing": {"file": prep_path.name, "sha256": sha256(prep_path)},
        "calibrator": None,
    }
    if args.calibrator:
        cal_path = args.out / "calibrator.pkl"
        if args.calibrator.resolve() != cal_path.resolve():
            cal_path.write_bytes(args.calibrator.read_bytes())
        artifacts["calibrator"] = {"file": cal_path.name, "sha256": sha256(cal_path)}

    manifest = {
        "schema_version": 1,
        "model_name": src["model_name"],
        "model_version": card["model_version"],
        "description": card.get("description", ""),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_with": {
            "scikit-learn": pickled_sklearn_version(prep["scaler"]),
            "xgboost": version("xgboost"),
            "joblib": version("joblib"),
        },
        "threshold": float(args.threshold if args.threshold is not None else src["threshold"]),
        "feature_names": list(src["feature_names"]),
        "artifacts": artifacts,
        "metrics": card["metrics"],
        "training_ranges": card["training_ranges"],
        "risk_tiers": card["risk_tiers"],
        "known_limitations": card.get("known_limitations", []),
    }
    out_path = args.out / "model_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {model_path.name}, {out_path.name}  (model {manifest['model_name']} v{manifest['model_version']})")
    print(f"  trained_with: {manifest['trained_with']}")


if __name__ == "__main__":
    main()
