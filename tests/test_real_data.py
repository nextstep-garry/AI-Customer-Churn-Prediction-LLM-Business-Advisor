"""Cross-check against the REAL raw file and the notebook's own cleaning (skipped if the CSV is absent)."""
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, upload
from churn_api.ml.cleaning import clean_dataset

RAW = ROOT / "training" / "data" / "churn_raw.csv"
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="training/data/churn_raw.csv not present")
sys.path.insert(0, str(ROOT / "training"))


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(RAW)


def test_notebook_cleaning_reproduces_the_audit_numbers(raw):
    from notebook_cleaning import notebook_clean
    nb = notebook_clean(raw)
    assert len(nb) == 9995 and round(nb["Exited"].mean(), 3) == 0.204   # audit report section 8.1
    assert nb["Age"].min() == 18                                        # so the API's 18-92 bound matches training


def test_repair_policy_reproduces_the_notebook_except_the_hardcoded_age_fill(raw):
    from notebook_cleaning import notebook_clean
    nb = notebook_clean(raw)
    mine = clean_dataset(raw, "repair").valid.set_index("row_id")
    nb_ids = nb.index.map(lambda i: raw.index.get_loc(i) + 1)
    m, n = mine.loc[nb_ids], nb.set_axis(nb_ids)
    for col in ["CreditScore", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]:
        assert (m[col].astype(float) == n[col].astype(float)).all(), col
    for col in ["Geography", "Gender"]:
        assert (m[col].str.lower() == n[col].str.lower()).all(), col
    # The notebook filled bad ages with a hard-coded 39; the API uses the file's median. Nothing else may differ.
    differs = m["Age"].astype(float) != n["Age"].astype(float)
    assert differs.sum() == 33 and (n.loc[differs, "Age"] == 39).all()


def test_real_file_through_the_api(client, raw):
    r = upload(client, RAW.read_bytes(), "real", policy="reject")
    rows = r.json["dataset"]["rows"]
    assert rows == {"uploaded": 10015, "scored": 9773, "rejected": 227, "repaired": 0, "duplicates_removed": 15}
    ds = upload(client, RAW.read_bytes(), "real-repair", policy="repair").json["dataset"]["id"]
    s = client.get(f"/v1/datasets/{ds}/summary").json
    assert s["kpis"]["customers_scored"] == 10000 and s["kpis"]["outside_training_range"] == 0
    assert [d["feature"] for d in s["drivers"][:3]] == ["NumOfProducts", "Age", "IsActiveMember"]   # audit 8.3h ranking
    assert all(d["status"] == "negligible" for d in s["quality"]["drift"])                          # it IS the training data
