"""Raw customer rows -> model-ready features. One vectorised implementation for both the
single-customer and the batch path (audit Phase 2: "one build_feature_frame shared by both").

Pipeline (identical to training; verified bit-exact against the original per-row code):
    engineered features -> Gender label-encode -> Geography one-hot -> winsorise -> scale -> order

Only *fitted* objects from preprocessing.pkl are used; nothing is ever re-fit here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .spec import RAW_FIELDS


def build_feature_frame(df: pd.DataFrame, bundle) -> pd.DataFrame:
    """`df` must already be validated/cleaned (see cleaning.py / schemas.py)."""
    prep = bundle.prep
    numeric_cols = list(prep["numeric_cols"])

    X = df[RAW_FIELDS].copy().reset_index(drop=True)
    X["Gender"] = X["Gender"].astype(str).str.strip().str.lower()
    X["Geography"] = X["Geography"].astype(str).str.strip().str.lower()
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="raise").astype(float)
    X["Balance"] = X["Balance"].astype(float)

    # 1. engineered features (formulas, not learned objects)
    X["Zero_Balance_Flag"] = (X["Balance"] == 0).astype(int)
    ratio = X["Balance"] / X["EstimatedSalary"].replace(0, np.nan)
    X["Balance_Salary_Ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 2. categorical encoding with the fitted encoders
    X["Gender"] = prep["gender_encoder"].transform(X["Gender"])
    geo_enc = prep["geography_encoder"]
    geo = pd.DataFrame(geo_enc.transform(X[["Geography"]]),
                       columns=geo_enc.get_feature_names_out(["Geography"]), index=X.index)
    X = pd.concat([X.drop(columns=["Geography"]), geo], axis=1)

    # 3. winsorise with training-set bounds, then scale with the training-set scaler
    for col, (lo, hi) in prep["winsor_bounds"].items():
        if col in X.columns:
            X[col] = X[col].clip(lower=lo, upper=hi)
    X[numeric_cols] = prep["scaler"].transform(X[numeric_cols])

    # 4. exact training column order
    missing = [f for f in bundle.feature_names if f not in X.columns]
    if missing:
        raise ValueError(f"Required model features are missing: {missing}")
    X = X[bundle.feature_names]
    if not np.isfinite(X.to_numpy(dtype=float)).all():  # defence in depth: XGBoost would silently route NaN
        raise ValueError("Non-finite values reached the model; input validation should have rejected them.")
    return X
