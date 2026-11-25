import numpy as np
import pandas as pd
import pytest

from conftest import VALID
from churn_api.ml.cleaning import clean_dataset
from churn_api.ml.exceptions import DatasetError


def frame(n=5, **patch):
    """n distinct-but-otherwise-identical customers (distinct ids, so they are not duplicates)."""
    df = pd.DataFrame([VALID] * n)
    df["CustomerId"] = range(1000, 1000 + n)
    for k, v in patch.items():
        df[k] = v
    return df


def test_headers_are_matched_loosely():
    df = frame().rename(columns={"CreditScore": "credit_score", "Geography": "GEOGRAPHY", "IsActiveMember": "Is Active Member"})
    assert len(clean_dataset(df).valid) == 5


def test_missing_required_columns_raise():
    with pytest.raises(DatasetError, match="Tenure"):
        clean_dataset(frame().drop(columns=["Tenure"]))


def test_typos_are_corrected_but_unknown_countries_are_not():
    df = frame(4)
    df["Geography"] = ["franc", "spainn", "germnny", "usa"]
    r = clean_dataset(df)
    assert r.stats["typos_corrected"] == 3
    assert sorted(r.valid["Geography"]) == ["France", "Germany", "Spain"]
    assert r.stats["rows_rejected"] == 1 and "Geography" in r.rejected["data_issues"].iloc[0]


@pytest.mark.parametrize("field,bad", [("Balance", -99999.99), ("Balance", 999999999.99), ("EstimatedSalary", 99999999.99),
                                       ("EstimatedSalary", 0), ("NumOfProducts", 80), ("NumOfProducts", -1),
                                       ("HasCrCard", 2), ("CreditScore", 999), ("Age", np.nan), ("Age", "abc"),
                                       ("Tenure", 2.5), ("Age", np.inf)])
def test_invalid_values_reject_only_that_row(field, bad):
    df = frame(3)
    df[field] = df[field].astype(object)
    df.loc[1, field] = bad
    r = clean_dataset(df)
    assert r.stats["rows_rejected"] == 1 and len(r.valid) == 2
    assert r.rejected["row_id"].tolist() == [2] and field in r.rejected["data_issues"].iloc[0]


def test_repair_policy_uses_mode_and_median_and_flags_rows():
    df = pd.concat([frame(6), frame(1, NumOfProducts=99)], ignore_index=True)
    df.loc[0, "Balance"] = -1.0
    df.loc[1:5, "Balance"] = [100, 200, 300, 400, 500]
    r = clean_dataset(df, "repair")
    assert r.stats["rows_repaired"] == 2 and r.stats["rows_rejected"] == 0
    fixed = r.valid.set_index("row_id")
    assert fixed.loc[7, "NumOfProducts"] == 1                       # mode
    assert fixed.loc[1, "Balance"] == pytest.approx(350.0, abs=200)  # median of valid balances (incl. 0.0 from row 7)
    assert set(r.valid["row_status"]) == {"ok", "repaired"}
    assert r.valid.loc[r.valid.row_status == "repaired", "data_issues"].str.startswith("repaired:").all()


def test_duplicates_ignore_the_row_number_column():
    df = pd.concat([frame(2), frame(2)], ignore_index=True)
    df.insert(0, "RowNumber", range(1, 5))
    r = clean_dataset(df)
    assert r.stats["duplicates_removed"] == 2 and len(r.valid) == 2


def test_target_column_is_kept_only_when_binary():
    df = frame(4)
    df["Exited"] = [0, 1, 2, np.nan]
    y = clean_dataset(df).valid["Exited"]
    assert y.iloc[:2].tolist() == [0, 1] and y.iloc[2:].isna().all()


def test_empty_frame_and_bad_policy():
    with pytest.raises(DatasetError):
        clean_dataset(frame(0))
    with pytest.raises(DatasetError):
        clean_dataset(frame(), "yolo")


def test_gender_abbreviations_are_accepted_and_unknown_tokens_are_missing():
    """'f'/'m' occur in the real raw file; 'u', 'other', 'nan' and blanks do not identify a gender."""
    df = frame(7)
    df["Gender"] = ["f", "M", "Female", "  male ", "u", "other", None]
    r = clean_dataset(df)
    assert r.valid.set_index("row_id")["Gender"].to_dict() == {1: "Female", 2: "Male", 3: "Female", 4: "Male"}
    assert set(r.rejected["row_id"]) == {5, 6, 7}


def test_geography_missing_tokens_and_typos_from_the_real_file():
    df = frame(7)
    df["Geography"] = ["FRANCE", "germny", "spainn", "unknown", "???", "nan", None]
    r = clean_dataset(df)
    assert r.valid.set_index("row_id")["Geography"].to_dict() == {1: "France", 2: "Germany", 3: "Spain"}
    assert set(r.rejected["row_id"]) == {4, 5, 6, 7}
