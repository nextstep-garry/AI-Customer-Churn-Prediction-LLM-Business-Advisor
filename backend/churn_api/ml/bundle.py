"""ModelBundle: the pluggable model object.

Everything the API needs to score a customer -- XGBoost model, fitted preprocessing objects,
optional calibrator, threshold, feature order, risk tiers -- is loaded from one directory and
described by one file, `model_manifest.json`. Swapping in a new model means replacing the files
in that directory (or pointing MODEL_DIR elsewhere); no code changes.

Loading is strict on purpose (audit findings 3.6 / D4):
  * every artifact's SHA-256 must match the manifest;
  * scikit-learn / xgboost must match the versions the artifacts were built with
    (override with ALLOW_VERSION_MISMATCH=1 for local experiments).
"""
from __future__ import annotations

import hashlib
import json
import logging
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .exceptions import ModelLoadError

log = logging.getLogger(__name__)
_CHECKED_LIBS = ("scikit-learn", "xgboost")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _major_minor(v: str) -> tuple[str, str]:
    parts = v.split(".")
    return parts[0], parts[1] if len(parts) > 1 else "0"


class ModelBundle:
    def __init__(self, *, model, prep: dict, manifest: dict, calibrator=None, model_dir: Path | None = None):
        self.model = model
        self.prep = prep
        self.manifest = manifest
        self.calibrator = calibrator
        self.model_dir = model_dir

        self.model_name: str = manifest["model_name"]
        self.model_version: str = manifest["model_version"]
        self.feature_names: list[str] = list(manifest["feature_names"])
        self.threshold: float = float(manifest["threshold"])
        tiers = manifest["risk_tiers"]
        self.risk_labels: list[str] = list(tiers["labels"])
        self.risk_cutoffs: list[float] = [float(c) for c in tiers["cutoffs"]]
        self.training_ranges: dict = {k: v for k, v in manifest.get("training_ranges", {}).items() if k != "source"}
        self.score_type: str = "calibrated" if calibrator is not None else "raw"

        if len(self.risk_cutoffs) != len(self.risk_labels) - 1 or self.risk_cutoffs != sorted(self.risk_cutoffs):
            raise ModelLoadError("risk_tiers: need ascending cutoffs, one fewer than labels.")
        if tiers.get("basis") != self.score_type:
            raise ModelLoadError(
                f"risk_tiers.basis is '{tiers.get('basis')}' but the loaded score is '{self.score_type}'. "
                "Tier cut-offs must be defined on the same scale as the served score "
                "(add/remove calibrator.pkl and re-run training/export_artifacts.py)."
            )

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_dir: Path, *, strict_versions: bool = True, verify_hashes: bool = True) -> "ModelBundle":
        model_dir = Path(model_dir)
        manifest_path = model_dir / "model_manifest.json"
        if not manifest_path.exists():
            raise ModelLoadError(
                f"{manifest_path} not found. Run: python training/export_artifacts.py "
                "--source <churn_model.pkl> --preprocessing <preprocessing.pkl> --out models")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 1:
            raise ModelLoadError(f"Unsupported manifest schema_version: {manifest.get('schema_version')}")

        paths: dict[str, Path] = {}
        for key, art in manifest["artifacts"].items():
            if art is None:
                continue
            p = model_dir / art["file"]
            if not p.exists():
                raise ModelLoadError(f"Artifact '{key}' listed in the manifest is missing: {p}")
            if verify_hashes and sha256_file(p) != art["sha256"]:
                raise ModelLoadError(
                    f"Artifact '{key}' ({p.name}) does not match the SHA-256 in model_manifest.json. "
                    "It changed after export; re-run training/export_artifacts.py or restore the original file.")
            paths[key] = p

        problems = []
        for lib in _CHECKED_LIBS:
            wanted, have = manifest.get("trained_with", {}).get(lib), version(lib)
            if wanted and _major_minor(wanted) != _major_minor(have):
                problems.append(f"{lib}: artifacts built with {wanted}, installed {have}")
        if problems:
            msg = ("Library version mismatch -- " + "; ".join(problems) +
                   ". Install the pinned versions from backend/requirements.txt, or set ALLOW_VERSION_MISMATCH=1.")
            if strict_versions:
                raise ModelLoadError(msg)
            log.warning(msg)

        from xgboost import XGBClassifier  # imported lazily: keeps `import churn_api` cheap

        model = XGBClassifier()
        model.load_model(paths["model"])
        with warnings.catch_warnings():
            warnings.simplefilter("error" if strict_versions else "default")  # InconsistentVersionWarning -> error
            try:
                prep = joblib.load(paths["preprocessing"])
                calibrator = joblib.load(paths["calibrator"]) if "calibrator" in paths else None
            except Warning as exc:  # noqa: BLE001
                raise ModelLoadError(f"Artifact loaded with a warning treated as an error: {exc}") from exc

        missing = [k for k in ("scaler", "winsor_bounds", "numeric_cols", "gender_encoder", "geography_encoder") if k not in prep]
        if missing:
            raise ModelLoadError(f"preprocessing.pkl is missing: {missing}")
        bundle = cls(model=model, prep=prep, manifest=manifest, calibrator=calibrator, model_dir=model_dir)
        log.info("model_loaded", extra={"model": bundle.model_name, "version": bundle.model_version,
                                        "score_type": bundle.score_type})
        return bundle

    # ------------------------------------------------------------------ scoring
    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw XGBoost probability of class 1 (uses the early-stopping best iteration)."""
        return self.model.predict_proba(X[self.feature_names])[:, 1].astype("float64")

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return raw
        p = np.clip(raw, 1e-6, 1 - 1e-6)
        return self.calibrator.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]

    def risk_levels(self, score: np.ndarray) -> np.ndarray:
        idx = np.digitize(score, self.risk_cutoffs)  # cutoff belongs to the higher tier
        return np.asarray(self.risk_labels, dtype=object)[idx]

    # ------------------------------------------------------------------ description
    def info(self) -> dict:
        m = self.manifest
        return {
            "name": self.model_name, "version": self.model_version, "description": m.get("description", ""),
            "created_at": m.get("created_at"), "score_type": self.score_type,
            "threshold": self.threshold,
            "risk_tiers": {"labels": self.risk_labels, "cutoffs": self.risk_cutoffs,
                           "basis": m["risk_tiers"].get("basis"), "note": m["risk_tiers"].get("note", "")},
            "feature_names": self.feature_names,
            "metrics": m.get("metrics", {}), "training_ranges": self.training_ranges,
            "known_limitations": m.get("known_limitations", []),
            "trained_with": m.get("trained_with", {}),
            "installed": {lib: version(lib) for lib in _CHECKED_LIBS},
            "artifact_hashes": {k: v["sha256"][:12] for k, v in m["artifacts"].items() if v},
        }
