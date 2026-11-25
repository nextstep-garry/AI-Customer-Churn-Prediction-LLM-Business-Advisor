import numpy as np
import pandas as pd

from conftest import FIXTURES
from churn_api.ml.explain import contributions
from churn_api.ml.features import build_feature_frame

golden = pd.read_csv(FIXTURES / "golden_predictions.csv")


def test_shap_contributions_add_up_to_the_served_margin(bundle):
    """Report 8.3h: contributions must be computed on the same trees predict_proba uses (best iteration)."""
    X = build_feature_frame(golden, bundle)
    contribs, base = contributions(bundle, X)
    p = bundle.predict_raw(X)
    np.testing.assert_allclose(contribs.sum(axis=1) + base, np.log(p / (1 - p)), atol=1e-4)


def test_explain_endpoint_returns_ordered_drivers(client):
    from conftest import VALID
    ex = client.post("/v1/predict?explain=true", json=VALID).json["explanation"]
    mags = [abs(d["contribution"]) for d in ex["drivers"]]
    assert mags == sorted(mags, reverse=True) and len(ex["drivers"]) == 8
    assert {"feature", "label", "contribution", "direction"} <= set(ex["drivers"][0])
    assert "explanation" not in client.post("/v1/predict", json=VALID).json
