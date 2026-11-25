"""Per-customer and dataset-level explanations via TreeSHAP.

Uses XGBoost's built-in TreeSHAP (`pred_contribs`) restricted to the early-stopping best
iteration, so the contributions add up to exactly the margin behind the served score.
(Calling the booster without `iteration_range` would use all 93 trees instead of the 63
that predict_proba uses -- the hazard called out in audit finding 8.3h.)  This is the same
algorithm as shap.TreeExplainer, without the heavy `shap`/numba dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


def contributions(bundle, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (contribs [n, n_features], base_value [n]) in log-odds (margin) space."""
    X = X[bundle.feature_names]
    booster = bundle.model.get_booster()
    dm = xgb.DMatrix(X, feature_names=list(X.columns))
    best = getattr(bundle.model, "best_iteration", None)
    kwargs = {"iteration_range": (0, int(best) + 1)} if best is not None else {}
    c = booster.predict(dm, pred_contribs=True, **kwargs)
    return c[:, :-1], c[:, -1]


def _label(feature: str, raw: dict) -> str:
    g = raw.get("Geography")
    table = {
        "Gender": lambda: f"Gender: {raw['Gender']}",
        "HasCrCard": lambda: f"Credit card: {'Yes' if int(raw['HasCrCard']) else 'No'}",
        "IsActiveMember": lambda: f"Active member: {'Yes' if int(raw['IsActiveMember']) else 'No'}",
        "Zero_Balance_Flag": lambda: f"Zero balance: {'Yes' if float(raw['Balance']) == 0 else 'No'}",
        "Balance_Salary_Ratio": lambda: f"Balance / salary ratio: {float(raw['Balance']) / float(raw['EstimatedSalary']):.2f}",
        "Geography_germany": lambda: f"In Germany: {'Yes' if g == 'Germany' else 'No'}",
        "Geography_spain": lambda: f"In Spain: {'Yes' if g == 'Spain' else 'No'}",
        "CreditScore": lambda: f"Credit score: {int(raw['CreditScore'])}",
        "Age": lambda: f"Age: {int(raw['Age'])}",
        "Tenure": lambda: f"Tenure: {int(raw['Tenure'])} yrs",
        "Balance": lambda: f"Balance: {float(raw['Balance']):,.0f}",
        "NumOfProducts": lambda: f"Products: {int(raw['NumOfProducts'])}",
        "EstimatedSalary": lambda: f"Salary: {float(raw['EstimatedSalary']):,.0f}",
    }
    return table.get(feature, lambda: feature)()


def explain_row(bundle, X_row: pd.DataFrame, raw: dict, top_k: int = 8) -> dict:
    contribs, base = contributions(bundle, X_row)
    c = contribs[0]
    order = np.argsort(-np.abs(c))[:top_k]
    return {
        "method": "TreeSHAP (log-odds of the raw model score)",
        "base_value": float(base[0]),
        "margin": float(c.sum() + base[0]),
        "drivers": [
            {"feature": bundle.feature_names[i], "label": _label(bundle.feature_names[i], raw),
             "contribution": float(c[i]), "direction": "increases risk" if c[i] > 0 else "decreases risk"}
            for i in order
        ],
    }


FEATURE_LABELS = {
    "CreditScore": "Credit score", "Gender": "Gender", "Age": "Age", "Tenure": "Tenure",
    "Balance": "Account balance", "NumOfProducts": "Number of products", "HasCrCard": "Has credit card",
    "IsActiveMember": "Active member", "EstimatedSalary": "Estimated salary",
    "Zero_Balance_Flag": "Zero balance", "Balance_Salary_Ratio": "Balance / salary ratio",
    "Geography_germany": "Country: Germany", "Geography_spain": "Country: Spain",
}


def global_drivers(bundle, X: pd.DataFrame, sample: int = 3000, seed: int = 0) -> list[dict]:
    """Mean |SHAP| per feature over (a sample of) the scored dataset."""
    if len(X) > sample:
        X = X.sample(sample, random_state=seed)
    contribs, _ = contributions(bundle, X)
    imp = np.abs(contribs).mean(axis=0)
    order = np.argsort(-imp)
    return [{"feature": bundle.feature_names[i],
             "label": FEATURE_LABELS.get(bundle.feature_names[i], bundle.feature_names[i]),
             "importance": float(imp[i])} for i in order]
