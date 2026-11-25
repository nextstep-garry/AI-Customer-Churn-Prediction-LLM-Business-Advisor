"""The notebook's raw-data cleaning (Sections 5-7), reproduced line for line as plain pandas.

Purpose: an independent reference for the training population. It is used to
  * reproduce the notebook's exact train/validation/test split (needed to fit a calibrator without leakage), and
  * cross-check the API's dataset cleaner (churn_api.ml.cleaning) against what training actually did.
Do not import this from the backend; the API has its own vectorised, spec-driven cleaner.
"""
import numpy as np
import pandas as pd

NUMERIC_COLS = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
BINARY_COLS = ["HasCrCard", "IsActiveMember"]


def notebook_clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop_duplicates().copy()                                           # Section 5
    for c in NUMERIC_COLS + BINARY_COLS:                                        # 6.1
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.loc[(df["Age"] <= 0) | (df["Age"] > 92), "Age"] = np.nan                 # 6.2 (fill value is hard-coded 39)
    df["Age"] = df["Age"].fillna(39)

    df.loc[~df["Tenure"].between(0, 10), "Tenure"] = np.nan                     # 6.3
    df["Tenure"] = df["Tenure"].fillna(df["Tenure"].median())

    df["NumOfProducts"] = df["NumOfProducts"].replace([-1, 0, 80, 999], np.nan)  # 6.4
    df.loc[~df["NumOfProducts"].between(1, 4), "NumOfProducts"] = np.nan
    df["NumOfProducts"] = df["NumOfProducts"].fillna(df["NumOfProducts"].mode()[0])

    df.loc[(df["CreditScore"] < 350) | (df["CreditScore"] > 850), "CreditScore"] = np.nan   # 6.5
    df["CreditScore"] = df["CreditScore"].fillna(df["CreditScore"].median())

    df.loc[df["Balance"].isin([-99999.99, 999999999.99]), "Balance"] = np.nan   # 6.6
    df["Balance"] = df["Balance"].fillna(df["Balance"].median())

    df.loc[(df["EstimatedSalary"] <= 0) | (df["EstimatedSalary"] == 99999999.99), "EstimatedSalary"] = np.nan  # 6.7
    df["EstimatedSalary"] = df["EstimatedSalary"].fillna(df["EstimatedSalary"].median())

    for c in BINARY_COLS:                                                       # 6.8 / 6.9
        df.loc[df[c] == 2, c] = np.nan
        df[c] = df[c].fillna(df[c].mode()[0])

    geo = df["Geography"].astype("string").str.strip().str.lower()              # 6.10
    geo = geo.replace(["nan", "na", "<na>", "unknown", "???"], pd.NA)
    geo = geo.replace({"franc": "france", "france": "france", "spainn": "spain", "germnny": "germany"})
    geo = geo.mask(geo == "usa", pd.NA).replace({"germny": "germany"})
    df["Geography"] = geo.fillna(geo.mode()[0])

    g = df["Gender"].astype("string").str.strip().str.lower()                   # 6.11
    g = g.replace({"m": "male", "f": "female"}).replace(["nan", "na", "u", "other"], pd.NA)
    df["Gender"] = g.fillna(g.mode()[0])

    df = df.dropna(subset=["Exited"])                                           # 7.3
    df["Exited"] = df["Exited"].astype(int)
    return df
