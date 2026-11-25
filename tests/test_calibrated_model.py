"""The calibrated model folder (models/calibrated) and the script that produced it."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES, ROOT, VALID, make_app
from churn_api.ml.bundle import ModelBundle
from churn_api.ml.features import build_feature_frame

CAL_DIR = ROOT / "models" / "calibrated"
pytestmark = pytest.mark.skipif(not CAL_DIR.exists(), reason="models/calibrated not generated")


@pytest.fixture(scope="module")
def cal():
    return ModelBundle.load(CAL_DIR)


def test_calibrated_bundle_metadata(cal):
    assert cal.score_type == "calibrated" and cal.model_version == "1.1.0"
    assert cal.risk_cutoffs == [0.15, 0.40] and 0.2 < cal.threshold < 0.25


def test_calibration_preserves_the_decision_boundary_exactly(bundle, cal):
    """Same customers flagged before and after: only the meaning of the number changes."""
    g = pd.read_csv(FIXTURES / "golden_predictions.csv")
    X = build_feature_frame(g, bundle)
    raw = bundle.predict_raw(X)
    served = cal.calibrate(cal.predict_raw(X))
    assert ((raw >= bundle.threshold) == (served >= cal.threshold)).all()
    order = np.argsort(raw)
    assert (np.diff(served[order]) >= 0).all()                     # monotone: ranking (AUC) unchanged


def test_report_reproduces_the_audit_numbers():
    r = json.loads((CAL_DIR / "calibration_report.json").read_text())
    assert list(r["split"].values()) == [6996, 1000, 1999]
    t = r["test"]
    assert t["roc_auc_raw"] == pytest.approx(t["roc_auc_calibrated"])
    assert round(t["brier_raw"], 4) == 0.1627 and round(t["brier_calibrated"], 4) == 0.1009    # audit 8.3b
    assert t["ece_calibrated"] < 0.03 < 0.2 < t["ece_raw"]
    assert [(x["customers"], round(x["observed_churn"], 3)) for x in r["test_tiers"]] == [(1224, 0.065), (413, 0.189), (362, 0.688)]


def test_api_serves_probabilities_when_pointed_at_the_calibrated_model(tmp_path, cal):
    c = make_app(tmp_path, cal, MODEL_DIR=CAL_DIR).test_client()
    r = c.post("/v1/predict", json=VALID).json
    assert r["score_type"] == "calibrated" and r["model_version"] == "1.1.0"
    assert 0.15 <= r["churn_score"] < 0.40 and r["risk_level"] == "Medium"   # raw 0.5517 is roughly a 25% probability
    assert c.get("/v1/model").json["risk_tiers"]["basis"] == "calibrated"


def test_dashboard_labels_change_with_the_score_type(tmp_path_factory, cal, monkeypatch):
    """Frontend adapts: probability wording and percentages, no 'not a probability' caveat."""
    import threading
    from werkzeug.serving import make_server
    pytest.importorskip("streamlit")
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from conftest import use_backend

    server = make_server("127.0.0.1", 0, make_app(tmp_path_factory.mktemp("cal"), cal, MODEL_DIR=CAL_DIR), threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        use_backend(f"http://127.0.0.1:{server.server_port}", monkeypatch)
        from churn_ui.theme import get_client
        get_client().upload("u.csv", (ROOT / "sample_data" / "churn_sample_unlabeled.csv").read_bytes(), "cal", "reject")
        at = AppTest.from_file(str(ROOT / "frontend" / "streamlit_app.py"), default_timeout=90).run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert any(m.label == "Average churn probability" for m in at.metric)
        assert not any("not a probability" in c.value for c in at.caption)
        at = at.switch_page("views/single.py").run()
        next(b for b in at.button if "Predict" in b.label).click().run()
        card = next(m.value for m in at.markdown if "<h2>" in m.value)
        assert "Churn probability" in card and "%" in card
    finally:
        server.shutdown()
        st.cache_resource.clear()
        st.cache_data.clear()


def test_fit_script_refuses_data_it_cannot_reproduce(tmp_path):
    """Leak guard: if the split can't be reproduced, no calibrator is written."""
    raw = pd.read_csv(ROOT / "training" / "data" / "churn_raw.csv")
    bad = tmp_path / "tampered.csv"
    raw.iloc[:9000].to_csv(bad, index=False)
    out = tmp_path / "out"
    p = subprocess.run([sys.executable, str(ROOT / "training" / "fit_calibrator.py"), "--raw", str(bad), "--out", str(out)],
                       capture_output=True, text=True)
    assert p.returncode != 0 and "REFUSING" in (p.stdout + p.stderr)
    assert not (out / "calibrator.pkl").exists()
