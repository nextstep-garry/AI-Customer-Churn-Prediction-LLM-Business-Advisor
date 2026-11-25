"""The refactor must not change a single prediction.

Golden fixtures were produced by running the ORIGINAL per-row inference_pipeline.py (before refactor)
on 304 rows, including boundary rows and the README sample.
"""
import numpy as np
import pandas as pd

from conftest import FIXTURES, VALID
from churn_api.ml.features import build_feature_frame
from churn_api.ml.pipeline import ChurnPipeline

golden = pd.read_csv(FIXTURES / "golden_predictions.csv")
golden_features = pd.read_csv(FIXTURES / "golden_features.csv")


def test_feature_frame_matches_original_implementation(bundle):
    X = build_feature_frame(golden, bundle)
    assert list(X.columns) == bundle.feature_names
    np.testing.assert_allclose(X.to_numpy(), golden_features[bundle.feature_names].to_numpy(), atol=1e-9)


def test_probabilities_match_original_implementation(bundle):
    p = bundle.predict_raw(build_feature_frame(golden, bundle))
    np.testing.assert_allclose(p, golden["expected_probability"].to_numpy(), atol=1e-9)


def test_batch_and_single_paths_agree(bundle):
    pipe = ChurnPipeline(bundle)
    records = golden.drop(columns="expected_probability").head(25).to_dict("records")
    batch = [r["churn_score"] for r in pipe.predict_records(records)]
    single = [pipe.predict_records([r])[0]["churn_score"] for r in records]
    assert batch == single


def test_readme_sample_is_0_5517_through_the_api(client):
    r = client.post("/v1/predict", json=VALID)
    assert r.status_code == 200 and r.json["churn_score"] == 0.5517
    assert r.json["score_type"] == "raw" and r.json["risk_level"] == "Medium"


def test_scoring_does_not_depend_on_row_order_or_batch_composition(bundle):
    pipe = ChurnPipeline(bundle)
    recs = golden.drop(columns="expected_probability").head(20).to_dict("records")
    fwd = [r["churn_score"] for r in pipe.predict_records(recs)]
    rev = [r["churn_score"] for r in pipe.predict_records(recs[::-1])][::-1]
    assert fwd == rev
