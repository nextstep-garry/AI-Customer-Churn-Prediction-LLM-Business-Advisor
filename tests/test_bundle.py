"""Model loading is strict: tampering, version drift and mis-scaled tiers must fail loudly."""
import json
import shutil

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from conftest import ROOT
from churn_api.ml.bundle import ModelBundle, sha256_file
from churn_api.ml.exceptions import ModelLoadError


@pytest.fixture
def model_copy(tmp_path):
    dst = tmp_path / "models"
    dst.mkdir()
    for f in ("xgb_model.json", "preprocessing.pkl", "model_manifest.json"):
        shutil.copy(ROOT / "models" / f, dst / f)
    return dst


def edit_manifest(d, fn):
    p = d / "model_manifest.json"
    m = json.loads(p.read_text())
    fn(m)
    p.write_text(json.dumps(m))


def test_pristine_copy_loads(model_copy):
    b = ModelBundle.load(model_copy)
    assert b.score_type == "raw" and b.model_version == "1.0.0"


def test_tampered_artifact_is_refused(model_copy):
    with (model_copy / "preprocessing.pkl").open("ab") as fh:
        fh.write(b"x")
    with pytest.raises(ModelLoadError, match="SHA-256"):
        ModelBundle.load(model_copy)


def test_missing_artifact_and_missing_manifest(model_copy, tmp_path):
    (model_copy / "xgb_model.json").unlink()
    with pytest.raises(ModelLoadError, match="missing"):
        ModelBundle.load(model_copy)
    with pytest.raises(ModelLoadError, match="model_manifest.json"):
        ModelBundle.load(tmp_path)


def test_library_version_mismatch_is_fatal_unless_allowed(model_copy):
    edit_manifest(model_copy, lambda m: m["trained_with"].update({"scikit-learn": "1.4.0"}))
    with pytest.raises(ModelLoadError, match="scikit-learn: artifacts built with 1.4.0"):
        ModelBundle.load(model_copy)
    assert ModelBundle.load(model_copy, strict_versions=False).model_name


def test_tier_basis_must_match_the_served_score(model_copy):
    edit_manifest(model_copy, lambda m: m["risk_tiers"].update({"basis": "calibrated", "cutoffs": [0.15, 0.4]}))
    with pytest.raises(ModelLoadError, match="basis"):
        ModelBundle.load(model_copy)


def test_bad_tier_cutoffs_are_refused(model_copy):
    edit_manifest(model_copy, lambda m: m["risk_tiers"].update({"cutoffs": [0.6, 0.5]}))
    with pytest.raises(ModelLoadError, match="ascending"):
        ModelBundle.load(model_copy)


def test_calibrator_switches_score_type_and_tiers(model_copy, bundle):
    """Plug-in path for a fitted Platt calibrator: monotone, tiers on the calibrated scale."""
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.01, 0.99, 800)
    y = (rng.random(800) < np.clip(raw ** 2, 0, 1)).astype(int)
    z = np.log(raw / (1 - raw)).reshape(-1, 1)
    joblib.dump(LogisticRegression().fit(z, y), model_copy / "calibrator.pkl")

    def patch(m):
        m["artifacts"]["calibrator"] = {"file": "calibrator.pkl", "sha256": sha256_file(model_copy / "calibrator.pkl")}
        m["risk_tiers"].update({"basis": "calibrated", "cutoffs": [0.15, 0.40]})
        m["threshold"] = 0.3
    edit_manifest(model_copy, patch)
    b = ModelBundle.load(model_copy)
    assert b.score_type == "calibrated"
    grid = np.linspace(0.02, 0.98, 50)
    assert (np.diff(b.calibrate(grid)) > 0).all()
    assert b.risk_levels(np.array([0.10, 0.15, 0.39, 0.40, 0.9])).tolist() == ["Low", "Medium", "Medium", "High", "High"]
