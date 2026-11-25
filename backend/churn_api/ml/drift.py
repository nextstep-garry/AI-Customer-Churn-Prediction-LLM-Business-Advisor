"""Data-drift signal for an uploaded dataset, relative to the training data.

Uses the mean/std stored in the fitted StandardScaler (fit on the winsorised training split),
so no extra reference file is needed. Shift is expressed in training standard deviations.
Rule of thumb: < 0.1 negligible, 0.1-0.25 small, 0.25-0.5 moderate, > 0.5 large.
"""
from __future__ import annotations

import pandas as pd


def mean_shift(clean: pd.DataFrame, prep: dict) -> list[dict]:
    scaler, bounds = prep["scaler"], prep["winsor_bounds"]
    out = []
    for i, col in enumerate(prep["numeric_cols"]):
        vals = clean[col].astype(float)
        if col in bounds:
            vals = vals.clip(*bounds[col])
        train_mean, train_std = float(scaler.mean_[i]), float(scaler.scale_[i])
        shift = (float(vals.mean()) - train_mean) / train_std if train_std else 0.0
        a = abs(shift)
        out.append({"feature": col, "training_mean": train_mean, "upload_mean": float(vals.mean()),
                    "shift_std": shift,
                    "status": "large" if a > 0.5 else "moderate" if a > 0.25 else "small" if a > 0.1 else "negligible"})
    return out
